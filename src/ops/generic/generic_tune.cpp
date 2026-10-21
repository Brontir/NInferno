// The dense tuner of the generic projection path (generic_gemm.h: tune_dense, tuned_dense_route).
//
// A problem without a route-table band runs the shape-class heuristic at decode widths, whose
// schedules were measured on a few cards: on another card the best register-MMA tile moves (the
// RTX 4090: Q6_K at K >= 14336 19-27 % faster on four token tiles than on the heuristic's one,
// Q8_0 up to 30 % on another warp split), and no rule per card keeps up with a zoo of them.
// The tuner times, on this GPU and on the problem's own weight, every admitted workspace-free A16
// schedule (register MMA, staged, NVFP4 tiled) at every decode width 1..16 the heuristic leaves on
// those families, and every R8 schedule where the heuristic takes R8 (decode and verify widths),
// each call after an L2 flush, the median of its repetitions. Where the policy allows A8, ggml's
// matrix kernel (Family::GgmlMmq, verify widths 5..16) competes with the A16 schedules: it wins on
// one card and format and loses on another (generic_gemm.cpp, kMmqMinTokens); so does R8 at the
// verify widths 2..4 of a format that does not take it there by rule (r8_verify_tunable). A
// schedule replaces the heuristic's only where it is kMargin faster, so noise keeps the heuristic.
//
// Prefill widths (T >= 17) are timed per band of widths (kPrefillBands: one representative width
// each) among routes that give a token the bytes of the heuristic's: the W4A8 schedules for a W4A8
// call (every W4A8 tile sums a token's K tiles in order, then its four blocks: the same bytes, and
// the same activation workspace), and under the prefill profile the routes that sum as the
// profile's one route does (same_summation: the WarpsK = 1 register-MMA and staged schedules). So a
// card no route table measured takes measured prefill tiles too, and a token's bytes still do not
// depend on its chunk's width (the heuristic's tiles were measured on the RTX 5090, 5080, 4090 and
// 3090; a W4A8 format new to the path, Q3_K or Q2_K, was never measured on any).
//
// It runs where the engine first computes the problem outside a CUDA graph capture (the warm-up's
// eager prefill, before any decode graph is captured), so the graphs take the tuned routes. The
// choice persists per GPU class and problem in a JSON file: a restart takes the same routes, and
// so the same bytes, without timing. What it leaves alone, by construction:
// - T >= 17 under another route class: the tiled, W4A4 and NVFP4 prefill routes, and the speed
//   profile's A16 bands (which split K by width);
// - W4A4 calls, and A16 decode calls that take a workspace (sized by the heuristic). An R8 call
// only
//   moves to another R8 schedule: the same workspace, and the same bytes (a lane strides K by whole
//   blocks, one butterfly ends each row), so R8 verify keeps giving the bytes of R8 decode that
//   greedy speculation relies on. This is what replaces per-board R8 entries (r8_decode_table.h)
//   and chip rules on a card they never measured;
// - A16 decode never moves to R8 or back (another activation precision);
// - widths a loaded route table covers (measured and published: it stays);
// - the invariant route profile and a live route probe.
// It times only when asked, NINFERNO_DENSE_TUNE=on (=retune also ignores the file): by default
// the engine starts with no timing and reads the measurements already made, the user's own file
// and the release's routes/tuner/dense_tune.json (the same entries, per GPU class and schedule
// set), over the GPU class's published route table (routes/), else the heuristic. The engine
// enables it for a loaded model (enable_dense_tune); tests and benches keep the heuristic.
// NINFERNO_DENSE_TUNE_CACHE names the file.

#include "core/arena.h"
#include "core/sys.h"
#include "core/cache_file.h"
#include "core/device.h"
#include "core/device_class.h"
#include "core/per_device.h"
#include "core/tune_timer.h"
#include "core/units.h"
#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/route_table.h"
#include "ops/common/native_nvfp4.h"
#include "ops/common/route_probe.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_table.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace ninferno::ops::detail::generic {
namespace {

using Json = nlohmann::json;

constexpr double kMargin = 0.97; // a schedule must take at most this share of the heuristic's time
constexpr int kWarm      = 1;
constexpr int kReps      = 5;

// Prefill bands: widths up to kPrefillBandTokens[b] (and past the last) take band b's route, timed
// at that width against the heuristic's route there, and only where the heuristic picks that same
// route (a width whose heuristic tile differs was never timed). 17 is a 16-token draft's verify;
// 32, 64, 127 and 128 straddle the heuristics' switch widths (the register MMA's prefill column
// at 32 and 64, W4A8's 128x64 from 65 and its producer-warp and 128x128 tiles from 128).
inline constexpr std::array<std::int32_t, 9> kPrefillBandTokens = {17,  32,  64,   127, 128,
                                                                   256, 512, 1024, 2048};
inline constexpr std::size_t kPrefillBands                      = kPrefillBandTokens.size();

std::size_t prefill_band(std::int32_t tokens) {
    for (std::size_t b = 0; b < kPrefillBands; ++b) {
        if (tokens <= kPrefillBandTokens[b]) return b;
    }
    return kPrefillBands - 1;
}

struct Tuned {
    std::array<std::optional<Route>, kDenseTuneMaxTokens + 1> routes{};
    std::array<std::optional<Route>, kPrefillBands> prefill{};
    std::array<std::optional<Route>, kPrefillBands>
        prefill_base{}; // what each band was timed against
};

// The problem as the dispatch looks it up on the calling thread's device (device, format, policy,
// epilogue, n, k): the policy and not its compute class, which names the GGUF formats a16 under
// every policy although AllowA8 takes R8 at decode and verify widths. A route timed on one GPU is
// that GPU's (core/per_device.h): a process driving two keeps one result per device. The schedule
// sets are the process's own, so only the cache file's key names them (with the GPU class).
using Key = std::tuple<int, int, int, int, std::int32_t, std::int32_t>;

Key key_of(const Problem& p) {
    return {current_device_ordinal(),
            static_cast<int>(p.qtype),
            static_cast<int>(p.policy),
            static_cast<int>(p.epilogue),
            p.n,
            p.k};
}

std::mutex g_mutex;
std::map<Key, Tuned> g_tuned; // route per T (nullopt: the heuristic's)
std::set<Key> g_attempted;    // problems seen (tuned, read from the file, or given up)
std::set<Key> g_bands_pending; // decode routes read, the bands to time at the next eager call

// Processes that may try a problem's skipped bands again: a failure that repeats (a route that
// throws on this shape, no memory beside the KV pool) leaves the heuristic's routes there.
constexpr int kPrefillAttempts = 3;
std::vector<std::string> g_report;
Json g_cache;
bool g_cache_loaded = false;

// The cache file's key: everything a route's time depends on: the GPU class, the format, the
// policy, the epilogue, the geometry and the compiled schedule sets (a build with other schedules
// times again).
std::string problem_key(const Problem& p) {
    const char* policy = p.policy == LinearPolicy::AllowA4   ? "a4"
                         : p.policy == LinearPolicy::AllowA8 ? "a8"
                                                             : "a16";
    std::ostringstream k;
    k << device_class().id() << '|' << qtype_name(p.qtype) << '|' << policy << '|'
      << epilogue_name(p.epilogue) << "|n" << p.n << "|k" << p.k << "|s" << rmma_schedule_count()
      << '.' << staged_schedule_count() << '.' << nvfp4_tiled_schedule_count() << '.'
      << r8_schedule_count() << '.' << w4a8_schedule_count();
    // Entries from before ggml's matrix kernel competed never timed it.
    if (ggml_mmq_format(p.qtype) && p.policy != LinearPolicy::A16Only) k << "|mmq";
    // Nor R8 at verify widths for a format that does not take it by rule.
    if (r8_verify_compiled(p.qtype) && p.policy != LinearPolicy::A16Only) k << "|r8v";
    // Entries from before ggml's vector kernel took T <= 8 by rule timed those widths.
    if (ggml_vec_format(p.qtype) && p.policy != LinearPolicy::A16Only) k << "|vec";
    // Nor, without tensor mma.sync, the vector kernel's 9..16 tokens in two parts.
    if (ggml_vec_format(p.qtype) && p.policy != LinearPolicy::A16Only &&
        !device_class().features.mma_tensor) {
        k << "16";
    }
    // Without tensor mma.sync the one-tile register MMA sums lane by lane (mma_step_local): entries
    // from before timed the shuffled emulation.
    if (!device_class().features.mma_tensor) k << "|ll4";
    // Nor, on a chip without tensor cores (mma_slow), the vector kernel and mmq rules it has.
    if (device_class().features.mma_slow) k << "|slow";
    // Where BF16 MMA is emulated the 5..16-token band is ggml's matrix kernel by heuristic, which
    // entries from before timed nothing against.
    if (ggml_mmq_format(p.qtype) && p.policy != LinearPolicy::A16Only &&
        !device_class().features.mma_bf16_native) {
        k << "|mmqb";
    }
    // NVFP4 decodes on R8 without the native kernels (r8_supports): an entry from before that timed
    // the register MMA at T = 1, and its bands none.
    if (p.qtype == QType::NVFP4 && p.policy != LinearPolicy::A16Only && r8_supports(p.qtype)) {
        k << "|r8";
    }
    return k.str();
}

std::filesystem::path cache_path() {
    if (const auto path = sys::env("NINFERNO_DENSE_TUNE_CACHE"); !path.empty()) { return path; }
    const auto dir = sys::cache_dir();
    return dir.empty() ? dir : dir / "dense_tune.json";
}

// /3: the prefill bands; /4: the heuristic's new prefill choices (bands timed against the old
// ones would no longer match their prefill_base and drop untimed).
constexpr const char* kSchema = "ninferno-dense-tune/4";

void load_cache_locked() {
    if (g_cache_loaded) { return; }
    g_cache_loaded = true;
    g_cache        = Json{{"schema", kSchema}, {"problems", Json::object()}};
    // An unreadable cache is a cache miss: the problem is timed again and the file rewritten.
    if (std::optional<Json> j = read_cache_file(cache_path(), kSchema, "problems")) {
        g_cache = std::move(*j);
    }
    // The release's measurements (routes/tuner/dense_tune.json, keyed like this file: GPU class,
    // format, policy, geometry, schedule sets): the entries the user's own file lacks.
    if (const std::string dir = route_table_dir(); !dir.empty()) {
        if (std::optional<Json> shipped =
                read_cache_file(std::filesystem::path(dir) / "tuner" / "dense_tune.json", kSchema,
                                "problems")) {
            auto& problems = g_cache["problems"];
            for (auto it = (*shipped)["problems"].begin(); it != (*shipped)["problems"].end();
                 ++it) {
                if (!problems.contains(it.key())) problems[it.key()] = it.value();
            }
        }
    }
}

std::string save_cache_locked() {
    const auto path = cache_path();
    if (path.empty()) { return "no cache directory (HOME unset)"; }
    try {
        save_cache_file(path, g_cache, kSchema, "problems");
        return {};
    } catch (const std::exception& error) {
        return std::string("cache not written: ") + error.what();
    }
}

// The heuristic's route at T is one the tuner may replace: an A16 family that needs no workspace,
// or R8 (another R8 schedule needs the workspace this one was sized for: the pre-quantized
// activation at verify widths, none at T = 1).
bool replaceable(Route r, std::int32_t k, std::int32_t tokens) {
    if (r.family == Family::R8Mma) return true;
    // The vector kernel's rule: R8, ggml's matrix kernel and the A16 schedules time against it.
    if (r.family == Family::GgmlVec) return true;
    // ggml's matrix kernel where BF16 MMA is emulated (Turing, T 5..16): the A16 schedules and R8
    // time against it, its workspace sized for every call that may take it.
    if (r.family == Family::GgmlMmq) return true;
    return (r.family == Family::RegisterMma || r.family == Family::StagedMma ||
            r.family == Family::Nvfp4TiledMma) &&
           route_workspace_bytes(r, k, tokens) == 0;
}

// The schedules that may replace `base`: another R8 schedule for R8 (the same workspace; a lane
// strides K by whole blocks and one butterfly ends each row whatever the schedule, so a token's
// bytes stay those of every R8 schedule, and speculative verify keeps reproducing decode), the
// workspace-free A16 families otherwise.
std::vector<Route> candidates(Route base, const Weight& weight, LinearPolicy policy) {
    std::vector<Route> routes;
    if (base.family == Family::R8Mma) {
        for (int i = 0; i < r8_schedule_count(); ++i) {
            routes.push_back({Family::R8Mma, static_cast<std::uint8_t>(i)});
        }
        std::erase_if(routes, [&](Route r) { return !route_admits(r, weight, policy); });
        return routes;
    }
    for (int i = 0; i < rmma_schedule_count(); ++i) {
        if (rmma_schedule_launchable(i)) routes.push_back({Family::RegisterMma, static_cast<std::uint8_t>(i)});
    }
    for (int i = 0; i < staged_schedule_count(); ++i) {
        routes.push_back({Family::StagedMma, static_cast<std::uint8_t>(i)});
    }
    for (int i = 0; i < nvfp4_tiled_schedule_count(); ++i) {
        routes.push_back({Family::Nvfp4TiledMma, static_cast<std::uint8_t>(i)});
    }
    std::erase_if(routes, [&](Route r) { return !route_admits(r, weight, policy); });
    return routes;
}

// The prefill-band candidates of `base` (the heuristic's route at a prefill width): the W4A8
// schedules for W4A8, the register-order routes (same_summation) for one of those; none otherwise.
std::vector<Route> prefill_candidates(Route base, const Weight& weight, LinearPolicy policy) {
    std::vector<Route> routes;
    if (base.family == Family::W4a8Mma) {
        const bool producers = w4a8_schedule_info(base.schedule).producers > 0;
        for (int i = 0; i < w4a8_schedule_count(); ++i) {
            if (w4a8_schedule_info(i).producers > 0 && !producers) continue;
            routes.push_back({Family::W4a8Mma, static_cast<std::uint8_t>(i)});
        }
    } else {
        for (int i = 0; i < rmma_schedule_count(); ++i) {
            if (rmma_schedule_launchable(i)) {
                routes.push_back({Family::RegisterMma, static_cast<std::uint8_t>(i)});
            }
        }
        for (int i = 0; i < staged_schedule_count(); ++i) {
            routes.push_back({Family::StagedMma, static_cast<std::uint8_t>(i)});
        }
        std::erase_if(routes, [&](Route r) { return !same_summation(base, r); });
    }
    std::erase_if(routes, [&](Route r) { return !route_admits(r, weight, policy); });
    return routes;
}

bool on_device(const void* p) {
    cudaPointerAttributes a{};
    if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return a.type == cudaMemoryTypeDevice || a.type == cudaMemoryTypeManaged;
}

struct Bench {
    Call call;
    Tensor x, out;
    WorkspaceArena* workspace = nullptr;
    TuneTimer* timer          = nullptr;

    double median_us(Route route, std::int32_t tokens) {
        Tensor xt(x.data, DType::BF16, {x.ne[0], tokens});
        Tensor ot(out.data, DType::BF16, {out.ne[0], tokens});
        Call c   = call;
        c.x      = &xt;
        c.out[0] = &ot;
        return timer->median_us([&](cudaStream_t s) {
            workspace->reset();
            run_route(route, c, workspace, s);
        });
    }
};

void time_prefill_bands(const Problem& problem, const Call& model_call, cudaStream_t stream,
                        Tuned& tuned, Json& entry, std::ostringstream& summary, int& improved);
void time_prefill_bands_guarded(const Problem& problem, const Call& model_call, cudaStream_t stream,
                                Tuned& tuned, Json& entry, std::ostringstream& summary,
                                int& improved);

// Times the problem on `weight`; returns the per-T routes and the cache entry.
Tuned time_problem(const Problem& problem, const Call& model_call, cudaStream_t stream,
                   Json& entry, std::ostringstream& summary, int& improved) {
    const Weight& weight = *model_call.weight;
    const std::int32_t k = weight.k;
    const std::int32_t t_max = kDenseTuneMaxTokens;
    DeviceBuffer x(static_cast<std::size_t>(k) * t_max * 2);
    DeviceBuffer out(static_cast<std::size_t>(model_call.rows) * t_max * 2);
    {
        std::mt19937 rng(20260930);
        std::normal_distribution<float> dist(0.0F, 1.0F);
        std::vector<__nv_bfloat16> host(static_cast<std::size_t>(k) * t_max);
        for (auto& v : host) { v = __float2bfloat16(dist(rng)); }
        x.copy_from_host(host.data(), x.bytes);
    }
    out.fill(0);
    // R8 at verify widths reads the activation pre-quantized into the workspace (the W4A8 planes).
    // ggml's matrix kernel: its activation, FP32 plane (SwiGLU: gate and up rows) and fixup.
    const int plane_rows =
        model_call.epilogue == Epilogue::SwiGlu ? 2 * model_call.rows : model_call.rows;
    DeviceArena workspace(
        std::max<std::size_t>({route_workspace_bytes(Route{Family::W4a8Mma, 0}, k, kR8VerifyTokens),
                               ggml_mmq_workspace_bytes(k, plane_rows, t_max) + 256,
                               ggml_vec_workspace_bytes(k, t_max) + 256, 256}));
    Bench bench{};
    bench.workspace            = &workspace;
    bench.call.weight          = &weight;
    bench.call.row_begin       = model_call.row_begin;
    bench.call.rows            = model_call.rows;
    bench.call.up_offset       = model_call.up_offset;
    bench.call.gelu_gate       = model_call.gelu_gate;
    bench.call.relu2           = model_call.relu2;
    bench.call.epilogue        = model_call.epilogue;
    bench.call.segment_rows[0] = model_call.rows;
    bench.call.segments        = 1;
    bench.x                    = Tensor(x.p, DType::BF16, {k, t_max});
    bench.out                  = Tensor(out.p, DType::BF16, {model_call.rows, t_max});
    TuneTimer timer(stream, kWarm, kReps);
    bench.timer = &timer;

    const auto a16 = candidates(Route{Family::RegisterMma, 0}, weight, problem.policy);
    const auto r8  = candidates(Route{Family::R8Mma, 0}, weight, problem.policy);
    Tuned tuned;
    entry = Json{{"routes", Json::object()}, {"us", Json::object()}};
    for (std::int32_t t = 1; t <= t_max; ++t) {
        const auto name  = std::to_string(t);
        const Route base = heuristic_route(problem, weight, t, false);
        entry["routes"][name] = nullptr;
        if (replaceable(base, k, t) && !table_route(problem, t, false)) {
            const double base_us = bench.median_us(base, t);
            double best_us       = base_us;
            Route best           = base;
            auto routes          = base.family == Family::R8Mma ? r8 : a16;
            if (base.family != Family::R8Mma &&
                ggml_mmq_call(weight, problem.policy, model_call.rows, t)) {
                routes.push_back(Route{Family::GgmlMmq, 0});
            }
            // R8 at a width the format does not take it at by rule: a token's bytes are then R8
            // decode's, which greedy speculation reproduces.
            if (base.family != Family::R8Mma && r8_tunable(weight, problem.policy, t)) {
                routes.insert(routes.end(), r8.begin(), r8.end());
            }
            for (const Route r : routes) {
                if (r == base) continue;
                const double us = bench.median_us(r, t);
                if (us < best_us) {
                    best_us = us;
                    best    = r;
                }
            }
            entry["us"][name] = {base_us, best_us};
            if (best_us <= kMargin * base_us) {
                tuned.routes[static_cast<std::size_t>(t)] = best;
                entry["routes"][name]                      = route_name(best);
                ++improved;
                summary << " T" << t << " " << format_us(base_us) << "->" << format_us(best_us)
                        << " us (" << route_name(best) << ")";
            }
        }
    }
    time_prefill_bands_guarded(problem, model_call, stream, tuned, entry, summary, improved);
    return tuned;
}

// The bands on their own: a failure there keeps the decode widths' timing, and the entry says so
// ("prefill_skipped"), so the next process times the bands again.
void time_prefill_bands_guarded(const Problem& problem, const Call& model_call, cudaStream_t stream,
                                Tuned& tuned, Json& entry, std::ostringstream& summary,
                                int& improved) {
    const auto& count  = entry.contains("prefill_attempts") ? entry["prefill_attempts"] : Json();
    const int attempts = count.is_number_integer() ? count.get<int>() : 0;
    // A failure part way keeps none of the bands, in the routes or the report.
    const int improved_before = improved;
    const auto summary_before = summary.str();
    try {
        time_prefill_bands(problem, model_call, stream, tuned, entry, summary, improved);
        if (entry.contains("prefill_skipped")) entry["prefill_attempts"] = attempts + 1;
    } catch (const std::exception& error) {
        (void)cudaGetLastError();
        tuned.prefill      = {};
        tuned.prefill_base = {};
        improved           = improved_before;
        summary.str(summary_before);
        summary.seekp(0, std::ios_base::end);
        for (const std::int32_t t : kPrefillBandTokens)
            entry["prefill"][std::to_string(t)] = nullptr;
        entry["prefill_base"]     = Json::object();
        entry["prefill_skipped"]  = error.what();
        entry["prefill_attempts"] = attempts + 1;
        summary << " prefill bands not timed (" << error.what() << ")";
    }
}

// The prefill bands (kPrefillBandTokens), on buffers of their own: the widest band's activation
// and output, and the W4A8 workspace at that width. Without the memory for them the bands keep the
// heuristic's routes and the decode widths their timing.
void time_prefill_bands(const Problem& problem, const Call& model_call, cudaStream_t stream,
                        Tuned& tuned, Json& entry, std::ostringstream& summary, int& improved) {
    const Weight& weight  = *model_call.weight;
    const std::int32_t k  = weight.k;
    entry["prefill"]      = Json::object();
    entry["prefill_us"]   = Json::object();
    entry["prefill_base"] = Json::object();
    entry.erase("prefill_skipped");
    entry.erase("prefill_attempts");
    tuned.prefill      = {};
    tuned.prefill_base = {};
    for (const std::int32_t t : kPrefillBandTokens) entry["prefill"][std::to_string(t)] = nullptr;
    // NVFP4 prefill takes its own invariant routes before any band, but for W4A8: the A8 prefill
    // of a build without the native NVFP4 kernels.
    const bool nvfp4 = weight.qtype == QType::NVFP4;
    if (nvfp4 && kNativeNvfp4) return;
    // The widest band whose buffers stay within 64 MiB each and whose call within 250 GFLOP (an
    // output head of 152K rows: T <= 128); wider bands keep the heuristic's route.
    const auto fits = [&](std::int32_t t) {
        constexpr double kBytes = 64.0 * kMiB;
        return 2.0 * model_call.rows * t <= kBytes && 2.0 * k * t <= kBytes &&
               2.0 * model_call.rows * k * t <= 2.5e11;
    };
    std::int32_t t_max = 0;
    for (const std::int32_t t : kPrefillBandTokens) {
        if (fits(t)) t_max = t;
    }
    if (t_max == 0) return;
    // The bands to time: candidates beyond the heuristic's route, and no route table entry (a
    // table that measured an A16 width keeps it; select_route never takes one for W4A8).
    std::array<std::vector<Route>, kPrefillBands> band_routes{};
    bool any = false, w4a8 = false;
    for (std::size_t b = 0; b < kPrefillBands && kPrefillBandTokens[b] <= t_max; ++b) {
        const std::int32_t t  = kPrefillBandTokens[b];
        const Route base      = heuristic_route(problem, weight, t, true);
        tuned.prefill_base[b] = base;
        // What the band is timed against: a later heuristic that picks another route there
        // makes the band stale (tune_dense times the bands again).
        entry["prefill_base"][std::to_string(t)] = route_name(base);
        if (base.family != Family::W4a8Mma && table_route(problem, t, true)) continue;
        if (nvfp4 && base.family != Family::W4a8Mma) continue;
        // Without tensor mma.sync (Volta) the tiled kernel takes the width before any band
        // (volta_tiled_route): a band timed there is never dispatched.
        if (volta_tiled_route(weight, problem.policy, t)) continue;
        band_routes[b] = prefill_candidates(base, weight, problem.policy);
        if (band_routes[b].size() < 2) continue;
        any  = true;
        w4a8 = w4a8 || base.family == Family::W4a8Mma;
    }
    if (!any) return;
    std::optional<DeviceBuffer> x, out;
    std::optional<DeviceArena> workspace;
    try {
        x.emplace(static_cast<std::size_t>(k) * t_max * 2);
        out.emplace(static_cast<std::size_t>(model_call.rows) * t_max * 2);
        workspace.emplace(std::max<std::size_t>(
            w4a8 ? route_workspace_bytes(Route{Family::W4a8Mma, 0}, k, t_max) : 0, 256));
    } catch (const std::exception&) {
        (void)cudaGetLastError();
        tuned.prefill_base       = {};
        entry["prefill_base"]    = Json::object();
        entry["prefill_skipped"] = "no memory for T " + std::to_string(t_max);
        summary << " prefill bands not timed (no memory for T " << t_max << ")";
        return;
    }
    {
        std::mt19937 rng(20261005);
        std::normal_distribution<float> dist(0.0F, 1.0F);
        std::vector<__nv_bfloat16> host(static_cast<std::size_t>(k) * t_max);
        for (auto& v : host) { v = __float2bfloat16(dist(rng)); }
        x->copy_from_host(host.data(), x->bytes);
    }
    out->fill(0);
    Bench bench{};
    bench.workspace            = &*workspace;
    bench.call.weight          = &weight;
    bench.call.row_begin       = model_call.row_begin;
    bench.call.rows            = model_call.rows;
    bench.call.up_offset       = model_call.up_offset;
    bench.call.gelu_gate       = model_call.gelu_gate;
    bench.call.relu2           = model_call.relu2;
    bench.call.epilogue        = model_call.epilogue;
    bench.call.segment_rows[0] = model_call.rows;
    bench.call.segments        = 1;
    bench.x                    = Tensor(x->p, DType::BF16, {k, t_max});
    bench.out                  = Tensor(out->p, DType::BF16, {model_call.rows, t_max});
    TuneTimer timer(stream, kWarm, kReps);
    bench.timer = &timer;

    for (std::size_t b = 0; b < kPrefillBands && kPrefillBandTokens[b] <= t_max; ++b) {
        const std::int32_t t = kPrefillBandTokens[b];
        const auto name      = std::to_string(t);
        // The prefill profile's route: an A16 problem's one route at every prefill width, a W4A8
        // problem's tile for this width.
        const Route base   = *tuned.prefill_base[b];
        const auto& routes = band_routes[b];
        if (routes.size() < 2) continue;
        const double base_us = bench.median_us(base, t);
        double best_us       = base_us;
        Route best           = base;
        for (const Route r : routes) {
            if (r == base) continue;
            const double us = bench.median_us(r, t);
            if (us < best_us) {
                best_us = us;
                best    = r;
            }
        }
        entry["prefill_us"][name] = {base_us, best_us};
        if (best_us <= kMargin * base_us) {
            tuned.prefill[b]       = best;
            entry["prefill"][name] = route_name(best);
            ++improved;
            summary << " T<=" << t << " " << format_us(base_us) << "->" << format_us(best_us)
                    << " us (" << route_name(best) << ")";
        }
    }
}

} // namespace

namespace {
std::atomic<bool> g_enabled{false};
} // namespace

namespace {
// Inside a capture nothing can be timed (or allocated): the problem is tried again at its next
// call, which the eager prefill makes before a decode graph is captured again.
bool eager(cudaStream_t stream) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return capture == cudaStreamCaptureStatusNone;
}

// A cached entry's prefill bands into `tuned`; false when they are to be timed again: skipped
// (no memory, a failure), missing, or timed against a route the heuristic no longer picks. A band
// this process does not admit (a weight's alignment) keeps the heuristic's route.
bool read_prefill_bands(const Problem& problem, const Weight& weight, const Json& entry,
                        Tuned& out) {
    Tuned tuned;
    if (!entry.contains("prefill") || !entry["prefill"].is_object() ||
        !entry.contains("prefill_base") || !entry["prefill_base"].is_object()) {
        return false;
    }
    if (entry.contains("prefill_skipped")) {
        // Tried enough: the heuristic's routes at every band.
        const auto& n = entry.contains("prefill_attempts") ? entry["prefill_attempts"] : Json();
        return n.is_number_integer() && n.get<int>() >= kPrefillAttempts;
    }
    const Json& bases = entry["prefill_base"];
    for (std::size_t b = 0; b < kPrefillBands; ++b) {
        const std::int32_t t = kPrefillBandTokens[b];
        const auto name      = std::to_string(t);
        if (!entry["prefill"].contains(name)) return false;
        const auto& v = entry["prefill"][name];
        if (!v.is_null() && !v.is_string()) return false;
        const bool timed = bases.contains(name);
        if (timed && !bases[name].is_string()) return false;
        if (!timed) {
            if (!v.is_null()) return false; // a band needs the base it was timed against
            continue;                       // past the timed widths, or nothing to time there
        }
        const Route base = heuristic_route(problem, weight, t, true);
        if (bases[name].get<std::string>() != route_name(base)) return false;
        tuned.prefill_base[b] = base;
        if (v.is_null()) continue;
        const auto r = parse_route(v.get<std::string>());
        if (!r) return false;
        const auto candidates = prefill_candidates(base, weight, problem.policy);
        if (std::find(candidates.begin(), candidates.end(), *r) != candidates.end()) {
            tuned.prefill[b] = *r;
        }
    }
    out.prefill      = tuned.prefill;
    out.prefill_base = tuned.prefill_base;
    return true;
}
} // namespace

void tune_dense(const Problem& problem, const Call& call, cudaStream_t stream) {
    if (!g_enabled.load(std::memory_order_relaxed) || invariant_routes_active() ||
        route_probe_state.quantized_act >= 0 || route_probe_state.tokens_as > 0) {
        return;
    }
    const TuneMode mode = tune_mode("NINFERNO_DENSE_TUNE");
    // Off (the default): the cached measurements are read, nothing is timed.
    const bool read_only = mode == TuneMode::Off;
    const Key id = key_of(problem);
    std::lock_guard lock(g_mutex);
    if (g_attempted.count(id) != 0) return;
    if (g_bands_pending.count(id) != 0 && !eager(stream)) return;
    // An offloaded weight's time is the transfer's; a u4 weight in another column order gathers
    // its input through the call's workspace first.
    if (call.weight->k_order != nullptr || !on_device(call.weight->qdata)) {
        g_attempted.insert(id);
        return;
    }
    const std::string key = problem_key(problem);

    load_cache_locked();
    auto& problems = g_cache["problems"];
    if (mode != TuneMode::Retune && problems.contains(key)) {
        Tuned tuned;
        bool complete     = true;
        const auto& entry = problems[key];
        for (std::int32_t t = 1; t <= kDenseTuneMaxTokens && complete; ++t) {
            const auto name = std::to_string(t);
            if (!entry.contains("routes") || !entry["routes"].contains(name)) {
                complete = false;
                break;
            }
            const auto& v = entry["routes"][name];
            if (v.is_null()) continue;
            if (!v.is_string()) {
                complete = false;
                break;
            }
            const auto r = parse_route(v.get<std::string>());
            if (!r || !route_admits(*r, *call.weight, problem.policy)) {
                complete = false;
            } else {
                tuned.routes[static_cast<std::size_t>(t)] = *r;
            }
        }
        const bool bands = complete && read_prefill_bands(problem, *call.weight, entry, tuned);
        if (bands) {
            g_tuned[id] = tuned;
            g_attempted.insert(id);
            return;
        }
        if (complete) {
            // The decode widths stand; the bands were skipped or timed against another heuristic.
            g_tuned[id] = tuned;
            if (read_only) {
                g_attempted.insert(id);
                return;
            }
            if (!eager(stream)) {
                g_bands_pending.insert(id);
                return;
            }
            g_bands_pending.erase(id);
            g_attempted.insert(id);
            std::ostringstream summary;
            summary << "dense tune " << key << " (prefill bands):";
            int improved = 0;
            Json fresh   = entry;
            time_prefill_bands_guarded(problem, call, stream, tuned, fresh, summary, improved);
            g_tuned[id]      = tuned;
            problems[key]    = fresh;
            const auto saved = save_cache_locked();
            if (improved == 0) summary << " the heuristic routes stay at every band";
            if (!saved.empty()) summary << "; " << saved;
            g_report.push_back(summary.str());
            return;
        }
    }
    if (read_only) {
        g_attempted.insert(id);
        return;
    }
    if (!eager(stream)) return;
    g_attempted.insert(id);

    std::ostringstream summary;
    summary << "dense tune " << key << ":";
    int improved = 0;
    Json entry;
    try {
        g_tuned[id] = time_problem(problem, call, stream, entry, summary, improved);
    } catch (const std::exception& error) {
        (void)cudaGetLastError();
        g_report.push_back("dense tune " + key + ": failed (" + error.what() + "); heuristic routes");
        return;
    }
    problems[key]    = entry;
    const auto saved = save_cache_locked();
    if (improved == 0) summary << " the heuristic routes stay at every T";
    if (!saved.empty()) summary << "; " << saved;
    g_report.push_back(summary.str());
}

std::optional<Route> tuned_dense_route(const Problem& problem, std::int32_t tokens) {
    if (tokens < 1 || tokens > kDenseTuneMaxTokens) return std::nullopt;
    std::lock_guard lock(g_mutex);
    if (g_tuned.empty()) return std::nullopt;
    const auto it = g_tuned.find(key_of(problem));
    if (it == g_tuned.end()) return std::nullopt;
    return it->second.routes[static_cast<std::size_t>(tokens)];
}

std::optional<Route> tuned_prefill_route(const Problem& problem, std::int32_t tokens,
                                         Route heuristic) {
    if (tokens <= kDenseTuneMaxTokens || !g_enabled.load(std::memory_order_relaxed)) {
        return std::nullopt;
    }
    std::lock_guard lock(g_mutex);
    if (g_tuned.empty()) return std::nullopt;
    const auto it = g_tuned.find(key_of(problem));
    if (it == g_tuned.end()) return std::nullopt;
    const std::size_t b = prefill_band(tokens);
    const auto& base    = it->second.prefill_base[b];
    if (!base || !(*base == heuristic)) return std::nullopt;
    return it->second.prefill[b];
}

std::vector<Route> prefill_band_candidates(Route base, const Weight& weight, LinearPolicy policy) {
    return prefill_candidates(base, weight, policy);
}

std::vector<std::string> dense_tune_report() {
    std::lock_guard lock(g_mutex);
    return g_report;
}

void enable_dense_tune(bool on) {
    // A misspelt mode is refused at load, not at the first projection the tuner sees.
    if (on) { (void)tune_mode("NINFERNO_DENSE_TUNE"); }
    g_enabled.store(on, std::memory_order_relaxed);
}

void clear_dense_tune() {
    std::lock_guard lock(g_mutex);
    g_tuned.clear();
    g_attempted.clear();
    g_bands_pending.clear();
    g_report.clear();
    g_cache_loaded = false;
}

} // namespace ninferno::ops::detail::generic
