// The route tuner (route_tune_tool.h): offline autotuner for the generic projection path. It writes a
// per-GPU-class route table (ninferno-route-table/1; include/ninferno/ops/route_table.h) so that a
// new model gets tuned routes as data, with no C++. It runs as `ninferno-serve route-tune` (a
// release, through the dispatcher's engine for the GPU) and as the bench's ninferno_route_tune.
//
//   ninferno-serve route-tune --shapes FILE [--out routes/<key>.json] [--key chip|variant|arch]
//                     [--samples 15] [--prefill-chunk 1024] [--prefill-only 1]
//                     [--points T,T,...] [--rotate 0|1] [--routes-out FILE.csv]
//                     [--selected-out FILE.csv]
//
// FILE holds one problem per line, "<format> <compute> <epilogue> <n> <k>", for example
// "nvfp4 a4 store 3584 1024" (format bf16|fp8|nvfp4|q4|q5|q6|q8, compute a16|a8|a4, epilogue
// store|add|swiglu; n is the op's output rows, SwiGLU's gate rows; a4 is taken only on nvfp4
// and a8 only on fp8, the compute classes a table entry can have). The engine writes such a file
// for a real model when run with NINFERNO_ROUTE_RECORD=FILE: every problem that reached the generic
// path, once. Duplicate lines are ignored.
//
// For every problem and every token count of the sweep, every admissible compiled route runs on a
// format-correct random weight with the L2 flushed before each sample; the median decides. Bands
// are the sweep points' winners; the prefill route is the token-invariant route with the least
// total time over the prefill widths (T = chunk/2 .. 8*chunk). The table is keyed by the current
// device's chip ("gb202", the default), its SM variant ("gb202:170sm", --key variant: only where
// boards of one chip tune apart) or its architecture ("sm120", --key arch). --prefill-only 1 skips
// the band sweep and writes only prefill routes: decode keeps whatever it takes today. Prefill
// bands (T = 32 .. 8*chunk, doubling) narrow the prefill route by width with routes that sum in its
// order (generic::same_summation), where one beats it by more than 2 %. A GGUF block format keys
// a16 under every policy, while the default policy (AllowA8) also runs ggml's matrix kernel at
// verify widths: the band sweep times it there as the dense tuner does (generic_tune.cpp), and the
// dispatch takes such a band only where the policy admits it. --points replaces the
// band sweep's token counts; --routes-out writes every timed (problem, T, route) as CSV, with the
// cost model's price of the route (route_cost_seconds: the model the selector ranks by at T >= 17
// outside the prefill profile, applied here at every T) and a register-MMA kernel's registers and
// local memory (spills or stack arrays) beside the timing.
// --selected-out writes, per problem and band point, the route the production path runs there
// (select_route resolved for the call), which is timed too: tools/bench/regret_collect.py.
// A decode or verify sample (T <= 16) is one call on each of a set of weight copies spanning 2x L2
// and 128 MiB, as decode runs its layers back to back; --rotate 0 times one call after the L2
// flush there too, as at the wider widths.
#include "core/arena.h"
#include "core/device.h"
#include "core/device_class.h"
#include "core/tune_timer.h"
#include "core/weight_view.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/random_weight.h"
#include "ops/generic/route_table.h"
#include "ops/generic/route_tune_tool.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {

using namespace ninferno;
namespace generic = ninferno::ops::detail::generic;
using bench::random_weight;
using bench::RandomWeight;

struct Problem {
    QType qtype;
    generic::ComputeClass compute;
    generic::Epilogue epilogue;
    std::int32_t n;
    std::int32_t k;
};

// Every format the route table names (qtype_name), so a recording of any artifact tunes.
QType parse_format(const std::string& s) {
    for (int v = 0; v <= 0xff; ++v) {
        const auto q = static_cast<QType>(v);
        if (s == generic::qtype_name(q)) return q;
    }
    throw std::invalid_argument("unknown format " + s);
}

generic::Epilogue parse_epilogue(const std::string& s) {
    if (s == "store") return generic::Epilogue::Store;
    if (s == "add") return generic::Epilogue::AddResidual;
    if (s == "swiglu") return generic::Epilogue::SwiGlu;
    throw std::invalid_argument("unknown epilogue " + s);
}

std::vector<std::int32_t> decode_points() { return {1, 2, 3, 4, 5, 6, 7, 8, 12, 16}; }
std::vector<std::int32_t> wide_points() { return {24, 32, 48, 64, 96, 128, 192, 256, 512}; }

} // namespace

namespace ninferno::tools {

int route_tune_main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> opt;
        for (int i = 1; i + 1 < argc; i += 2) opt[argv[i]] = argv[i + 1];
        if (!opt.count("--shapes")) {
            std::cerr << "usage: ninferno-serve route-tune --shapes FILE [--out FILE] "
                         "[--key chip|variant|arch] [--samples N] [--prefill-chunk C] "
                         "[--prefill-only 1] [--points T,...] [--routes-out FILE.csv] "
                         "[--selected-out FILE.csv] [--rotate 0|1]\n";
            return 2;
        }
        const int samples       = opt.count("--samples") ? std::stoi(opt["--samples"]) : 15;
        const int chunk         = opt.count("--prefill-chunk") ? std::stoi(opt["--prefill-chunk"]) : 1024;
        const bool prefill_only = opt.count("--prefill-only") && opt["--prefill-only"] == "1";
        const bool rotate       = !(opt.count("--rotate") && opt["--rotate"] == "0");
        const auto& device      = device_class();
        const std::string key_kind = opt.count("--key") ? opt["--key"] : "chip";
        if (key_kind != "chip" && key_kind != "variant" && key_kind != "arch") {
            throw std::invalid_argument("--key must be chip, variant or arch");
        }
        const std::string key = key_kind == "chip"      ? device.chip
                                : key_kind == "variant" ? device.id()
                                                        : device.arch();
        // ':' cannot appear in a Windows file name: "gb202:170sm" is written as gb202-170sm.json.
        std::string file_key = key;
        std::replace(file_key.begin(), file_key.end(), ':', '-');
        const std::string out_path = opt.count("--out") ? opt["--out"] : file_key + ".json";
        std::vector<std::int32_t> custom_points;
        if (opt.count("--points")) {
            std::stringstream list(opt["--points"]);
            for (std::string item; std::getline(list, item, ',');) {
                custom_points.push_back(std::stoi(item));
            }
        }
        std::ofstream routes_csv;
        if (opt.count("--routes-out")) {
            routes_csv.open(opt["--routes-out"]);
            if (!routes_csv) throw std::runtime_error("cannot open " + opt["--routes-out"]);
            routes_csv << "format,compute,epilogue,n,k,T,route,median_us,model_us,registers,"
                          "local_bytes\n";
        }
        // The route the production path runs at each band point (#301): select_route's choice
        // with the tables this device loads, resolved for the contiguous call and this workspace,
        // and timed like the candidates, so a regret collector compares it to them.
        std::ofstream selected_csv;
        if (opt.count("--selected-out")) {
            selected_csv.open(opt["--selected-out"]);
            if (!selected_csv) throw std::runtime_error("cannot open " + opt["--selected-out"]);
            selected_csv << "format,compute,epilogue,n,k,T,asked,selected\n";
        }

        std::vector<Problem> problems;
        {
            std::ifstream in(opt["--shapes"]);
            if (!in) throw std::runtime_error("cannot open " + opt["--shapes"]);
            std::set<std::tuple<int, int, int, int, int>> seen;
            std::string line;
            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream fields(line);
                std::string fmt, comp, epi;
                std::int32_t n = 0, k = 0;
                if (!(fields >> fmt >> comp >> epi >> n >> k)) continue;
                // The table keys a problem by the compute class the format has (route_table.cpp
                // refuses the whole file over any other): A4 only on nvfp4, A8 only on fp8.
                if ((comp == "a4" && fmt != "nvfp4") || (comp == "a8" && fmt != "fp8")) {
                    std::cerr << "skipped '" << line << "': " << fmt << " has no " << comp
                              << " table entry\n";
                    continue;
                }
                const Problem p{parse_format(fmt),
                                comp == "a4"   ? generic::ComputeClass::A4
                                : comp == "a8" ? generic::ComputeClass::A8
                                               : generic::ComputeClass::A16,
                                parse_epilogue(epi), n, k};
                if (seen.insert({static_cast<int>(p.qtype), static_cast<int>(p.compute),
                                 static_cast<int>(p.epilogue), n, k})
                        .second) {
                    problems.push_back(p);
                }
            }
        }
        std::cerr << "device " << device.name << " class " << device.id() << " (table key "
                  << key << "), "
                  << problems.size() << " problems\n";

        std::vector<std::int32_t> prefill_points;
        for (int t = chunk / 2; t <= 8 * chunk; t *= 2) prefill_points.push_back(t);
        // Widths of the prefill bands: 32 .. 8 * chunk, doubling.
        std::vector<std::int32_t> band_points;
        for (int t = 32; t <= 8 * chunk; t *= 2) band_points.push_back(t);
        std::int32_t t_max = *std::max_element(prefill_points.begin(), prefill_points.end());
        for (const auto t : custom_points) t_max = std::max(t_max, t);

        TuneTimer timer(nullptr, 1, samples);
        // Whether the run times a decode or verify width (T <= 16): the band sweep's points, or
        // the prefill widths from chunk / 2.
        const bool times_small_t =
            chunk / 2 <= 16 ||
            (!prefill_only && (custom_points.empty() ||
                               std::any_of(custom_points.begin(), custom_points.end(),
                                           [](std::int32_t t) { return t <= 16; })));
        std::vector<generic::RouteEntry> entries;

        for (const Problem& p : problems) {
            const bool swiglu     = p.epilogue == generic::Epilogue::SwiGlu;
            const std::int32_t rows = swiglu ? 2 * p.n : p.n;
            RandomWeight w        = random_weight(p.qtype, rows, p.k, 7U + p.n + p.k);
            // A decode or verify sample is one call on each of `copies` weights in turn, between
            // one event pair, as consecutive decode layers read theirs: up to 64 copies spanning
            // 2x L2 and 128 MiB, so a sample holds many calls (a single call of a few microseconds
            // is timed only to the timer's tick, 1.024 us on the consumer boards and GV100) and its
            // reads stay cold where the timer has no flush buffer; the activations stay in L2.
            // Only where the run times such a width.
            std::vector<RandomWeight> spare;
            if (rotate && times_small_t) {
                const std::size_t bytes = std::max<std::size_t>(w.parent.geometry.bytes, 1);
                const std::size_t span  = std::max<std::size_t>(2 * device.l2_bytes, 128U << 20);
                const std::size_t copies =
                    std::clamp<std::size_t>((span + bytes - 1) / bytes, 1, 64);
                for (std::size_t c = 1; c < copies; ++c) {
                    spare.push_back(random_weight(p.qtype, rows, p.k, 7U + p.n + p.k + 977U * c));
                }
            }
            const auto policy = p.compute == generic::ComputeClass::A4 ? ops::LinearPolicy::AllowA4
                                : p.compute == generic::ComputeClass::A8
                                    ? ops::LinearPolicy::AllowA8
                                    : ops::LinearPolicy::A16Only;
            DeviceBuffer x(static_cast<std::size_t>(p.k) * t_max * 2);
            {
                std::vector<std::uint16_t> host(static_cast<std::size_t>(p.k) * t_max);
                std::mt19937 rng(11);
                std::normal_distribution<float> nd(0.0F, 1.0F);
                for (auto& v : host) {
                    const float f = nd(rng);
                    std::uint32_t bits;
                    std::memcpy(&bits, &f, 4);
                    v = static_cast<std::uint16_t>(bits >> 16);
                }
                x.copy_from_host(host.data(), host.size() * 2);
            }
            DeviceBuffer y(static_cast<std::size_t>(p.n) * t_max * 2);
            y.fill(0);
            DeviceArena workspace(std::max<std::size_t>(
                {generic::route_workspace_bytes({generic::Family::W4a4Mma, 0}, p.k, t_max),
                 generic::route_workspace_bytes({generic::Family::W4a8Mma, 0}, p.k, t_max),
                 generic::route_workspace_bytes({generic::Family::R8Mma, 0}, p.k,
                                                generic::kR8MaxTokens),
                 generic::ggml_mmq_workspace_bytes(p.k, rows, 16) + 256,
                 std::size_t{256}}));

            std::vector<generic::Route> routes;
            for (int i = 0; i < generic::rmma_schedule_count(); ++i) {
                routes.push_back({generic::Family::RegisterMma, static_cast<std::uint8_t>(i)});
            }
            for (int i = 0; i < generic::w4a4_schedule_count(); ++i) {
                routes.push_back({generic::Family::W4a4Mma, static_cast<std::uint8_t>(i)});
            }
            for (int i = 0; i < generic::w4a8_schedule_count(); ++i) {
                routes.push_back({generic::Family::W4a8Mma, static_cast<std::uint8_t>(i)});
            }
            for (int i = 0; i < generic::r8_schedule_count(); ++i) {
                routes.push_back({generic::Family::R8Mma, static_cast<std::uint8_t>(i)});
            }
            for (int i = 0; i < generic::staged_schedule_count(); ++i) {
                routes.push_back({generic::Family::StagedMma, static_cast<std::uint8_t>(i)});
            }
            for (int i = 0; i < generic::tiled_schedule_count(); ++i) {
                routes.push_back({generic::Family::TiledMma, static_cast<std::uint8_t>(i)});
            }
            for (int i = 0; i < generic::nvfp4_tiled_schedule_count(); ++i) {
                routes.push_back({generic::Family::Nvfp4TiledMma, static_cast<std::uint8_t>(i)});
            }
            std::erase_if(routes, [&](generic::Route r) {
                return !generic::route_admits(r, w.weight, policy);
            });
            // ggml's matrix kernel where the default policy runs it (the table keys the call a16).
            const bool mmq = p.compute == generic::ComputeClass::A16 &&
                             generic::ggml_mmq_format(p.qtype);

            const auto measure = [&](generic::Route r, std::int32_t t) {
                Tensor tx(x.p, DType::BF16, {p.k, t});
                Tensor ty(y.p, DType::BF16, {p.n, t});
                generic::Call call;
                call.x               = &tx;
                call.weight          = &w.weight;
                call.rows            = p.n;
                call.up_offset       = swiglu ? p.n : 0;
                call.epilogue        = p.epilogue;
                call.out[0]          = &ty;
                call.segment_rows[0] = p.n;
                // One warm sample, then the median of `samples`, each after the tuners' read flush
                // (core/tune_timer.h): a memset flush left the L2 dirty and the timed call paid its
                // write-back (RTX 4070 Ti, Q6_K 4096x12288 T=4, rmma.m1.n1.w1x8: 169 us with the
                // memset flush, 125 with this one, ninferno_linear_bench 125). A decode or verify
                // call (T <= 16) runs over the copies: a flushed single call of a few microseconds
                // is timed only to the event tick and counts its launch, which consecutive layers
                // overlap (RTX 5090, Q4_K 4096x4096 T=1: 12.3 us flushed, 8.2 over copies; 13 of 40
                // small-T winners changed; pass-to-pass spread 2.2 -> 0.6 % at p90, GV100 20 ->
                // 0.6 %). A wider call is long against both and keeps the one flushed call.
                const bool over_copies = t <= 16;
                // The GPU waits while the host enqueues the sample (20 us a call), so a sample of
                // short calls times the kernels and not the host's launch rate. Launch overlap
                // between the calls is the stream's front end only: the tuner sets no PDL joins.
                timer.set_lead_us(over_copies && !spare.empty() ? 20.0 * (1 + spare.size()) : 0.0);
                const double us =
                    timer.median_us([&](cudaStream_t s) {
                        call.weight = &w.weight;
                        generic::run_route(r, call, &workspace, s);
                        if (!over_copies) return;
                        for (const RandomWeight& c : spare) {
                            call.weight = &c.weight;
                            generic::run_route(r, call, &workspace, s);
                        }
                    }) /
                    static_cast<double>(over_copies ? 1 + spare.size() : 1);
                if (routes_csv) {
                    routes_csv << generic::qtype_name(p.qtype) << ','
                               << generic::compute_class_name(p.compute) << ','
                               << generic::epilogue_name(p.epilogue) << ',' << p.n << ',' << p.k
                               << ',' << t << ',' << generic::route_name(r) << ',' << us << ',';
                    // The cost model's time for the route (at every T, though the selector ranks
                    // by it only at T >= 17 outside the prefill profile, and not for NVFP4), and
                    // the register-MMA kernel's compiled registers and local memory (spills or
                    // stack arrays): what a timing far from its price is read against. Empty
                    // where the model prices no such route or the route is no register MMA.
                    const generic::Problem gp{p.qtype, policy, p.n, p.k, p.epilogue};
                    if (const auto s = generic::route_cost_seconds(gp, w.weight, r, t)) {
                        routes_csv << *s * 1e6;
                    }
                    routes_csv << ',';
                    const auto res = r.family == generic::Family::RegisterMma
                                         ? generic::rmma_resources(r.schedule, w.weight,
                                                                   p.epilogue, t)
                                         : std::nullopt;
                    if (res) {
                        routes_csv << res->registers << ',' << res->local_bytes;
                    } else {
                        routes_csv << ',';
                    }
                    routes_csv << '\n';
                }
                return us; // microseconds
            };
            // One measurement per (route, T): the band sweep, the prefill total and the prefill
            // bands share them.
            std::map<std::pair<int, std::int32_t>, double> timed;
            const auto time_route = [&](generic::Route r, std::int32_t t) {
                const std::pair<int, std::int32_t> key{static_cast<int>(r.family) * 256 + r.schedule, t};
                const auto it = timed.find(key);
                if (it != timed.end()) return it->second;
                return timed[key] = measure(r, t);
            };

            generic::RouteEntry entry{p.qtype, p.compute, p.epilogue, p.n, p.k, {}, std::nullopt, {}, false};
            std::vector<std::int32_t> points;
            if (!prefill_only) points = custom_points;
            if (!prefill_only && points.empty()) {
                points = decode_points();
                for (const auto t : wide_points()) points.push_back(t);
            }
            std::vector<std::pair<std::int32_t, generic::Route>> winners;
            for (const auto t : points) {
                double best = 1e30;
                generic::Route best_route{};
                for (const auto r : routes) {
                    // A4 is taken only from 17 tokens (decode and verify run A16); the staged
                    // family and the NVFP4 tiles serve prefill widths.
                    if ((r.family == generic::Family::W4a4Mma ||
                         r.family == generic::Family::StagedMma ||
                         r.family == generic::Family::Nvfp4TiledMma ||
                         r.family == generic::Family::TiledMma) &&
                        t < 17) {
                        continue;
                    }
                    // R8 serves decode and verify widths only.
                    if (r.family == generic::Family::R8Mma && t > generic::kR8MaxTokens) continue;
                    const double us = time_route(r, t);
                    if (us < best) {
                        best       = us;
                        best_route = r;
                    }
                }
                if (mmq && generic::ggml_mmq_call(w.weight, ops::LinearPolicy::AllowA8, p.n, t)) {
                    const generic::Route r{generic::Family::GgmlMmq, 0};
                    const double us = time_route(r, t);
                    if (us < best) {
                        best       = us;
                        best_route = r;
                    }
                }
                winners.push_back({t, best_route});
                if (selected_csv) {
                    const generic::Problem gp{p.qtype, policy, p.n, p.k, p.epilogue};
                    const generic::Route asked =
                        generic::select_route(gp, w.weight, t, ops::prefill_routes_active());
                    const generic::CallLayout layout{.x_aligned16    = true,
                                                     .ldx            = p.k,
                                                     .tokens         = t,
                                                     .rows           = p.n,
                                                     .segments       = 1,
                                                     .epilogue       = p.epilogue,
                                                     .workspace_free = workspace.capacity()};
                    const generic::Route ran =
                        generic::resolve_route(asked, w.weight, layout).executed;
                    time_route(ran, t);
                    selected_csv << generic::qtype_name(p.qtype) << ','
                                 << generic::compute_class_name(p.compute) << ','
                                 << generic::epilogue_name(p.epilogue) << ',' << p.n << ','
                                 << p.k << ',' << t << ',' << generic::route_name(asked) << ','
                                 << generic::route_name(ran) << '\n';
                }
                std::printf("%s %s %s n=%d k=%d T=%d best=%s %.2f us\n",
                            generic::qtype_name(p.qtype), generic::compute_class_name(p.compute),
                            generic::epilogue_name(p.epilogue), p.n, p.k, t,
                            generic::route_name(best_route).c_str(), best);
            }
            for (std::size_t i = 0; i < winners.size(); ++i) {
                const std::int32_t lo = winners[i].first;
                const std::int32_t hi = i + 1 < winners.size() ? winners[i + 1].first - 1 : 1 << 24;
                if (!entry.bands.empty() && entry.bands.back().route == winners[i].second) {
                    entry.bands.back().t_max = hi;
                } else {
                    entry.bands.push_back({lo, hi, winners[i].second});
                }
            }
            double best_total = 1e30;
            for (const auto r : routes) {
                if (!generic::route_is_token_invariant(r)) continue;
                double total = 0;
                for (const auto t : prefill_points) total += time_route(r, t);
                std::printf("  prefill candidate %s %.2f us\n", generic::route_name(r).c_str(), total);
                if (total < best_total) {
                    best_total    = total;
                    entry.prefill = r;
                }
            }
            if (entry.prefill) {
                std::printf("%s %s %s n=%d k=%d prefill=%s %.2f us over T=%d..%d\n",
                            generic::qtype_name(p.qtype), generic::compute_class_name(p.compute),
                            generic::epilogue_name(p.epilogue), p.n, p.k,
                            generic::route_name(*entry.prefill).c_str(), best_total,
                            prefill_points.front(), prefill_points.back());
                // Prefill bands: per width, the fastest route that sums as the prefill route does
                // (the same bytes), where it beats the prefill route by more than 2 %. Band i spans
                // [point i, point i+1 - 1] (the first from 17, the last open); between bands the
                // prefill route answers.
                std::vector<std::pair<std::int32_t, generic::Route>> width_winners;
                for (const auto t : band_points) {
                    const double base   = time_route(*entry.prefill, t);
                    double best         = base;
                    generic::Route pick = *entry.prefill;
                    for (const auto r : routes) {
                        if (r == *entry.prefill || !generic::same_summation(r, *entry.prefill)) continue;
                        const double us = time_route(r, t);
                        if (us < best && us < base * 0.98) {
                            best = us;
                            pick = r;
                        }
                    }
                    std::printf("  prefill T=%d best=%s %.2f us (prefill route %.2f us)\n", t,
                                generic::route_name(pick).c_str(), best, base);
                    width_winners.push_back({t, pick});
                }
                for (std::size_t i = 0; i < width_winners.size(); ++i) {
                    const std::int32_t lo = i == 0 ? 17 : width_winners[i].first;
                    const std::int32_t hi =
                        i + 1 < width_winners.size() ? width_winners[i + 1].first - 1 : 1 << 24;
                    const generic::Route r = width_winners[i].second;
                    if (!entry.prefill_bands.empty() && entry.prefill_bands.back().route == r &&
                        entry.prefill_bands.back().t_max == lo - 1) {
                        entry.prefill_bands.back().t_max = hi;
                    } else if (!(r == *entry.prefill)) {
                        entry.prefill_bands.push_back({lo, hi, r});
                    }
                }
            }
            entries.push_back(std::move(entry));
            std::fflush(stdout);
        }
        std::ofstream out(out_path);
        out << generic::write_route_table(key, device.name, entries);
        std::cerr << "wrote " << out_path << " (" << entries.size() << " entries)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "route-tune: " << e.what() << '\n';
        return 1;
    }
}

} // namespace ninferno::tools
