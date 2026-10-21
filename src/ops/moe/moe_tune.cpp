// The load-time MoE tuner (ninferno/ops/moe.h: tune_moe).
//
// A layer without a route-table band runs the heuristic plan at decode widths: the streaming K split
// of every stage, whose schedules were measured on a few shapes. On a shape they were not measured
// on the best plan can be far off it (NemotronH's IQ3_XXS layer on the RTX 5080: 15-22 % at T = 1..12).
// The tuner times the plans on the layer's own weights, on this GPU, at T = 1..max_tokens:
// every gate/up schedule under the default down, then every down schedule under the best gate/up,
// each call after an L2 flush, the median of its repetitions. A plan replaces the default only
// where it is kMargin faster, so noise keeps the default. The choice persists per GPU class and
// layer in a JSON file: a restart takes the same plans, and so the same bytes, without timing.

#include "core/arena.h"
#include "core/sha256.h"
#include "core/sys.h"
#include "core/cache_file.h"
#include "core/device.h"
#include "core/device_class.h"
#include "core/tune_timer.h"
#include "ninferno/ops/moe.h"
#include "ops/moe/moe_internal.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>

#ifndef _WIN32
#include <unistd.h>
#endif
#include <string>
#include <vector>

namespace ninferno::ops {
namespace detail::moe {
namespace {

using Json = nlohmann::json;

constexpr double kMargin = 0.97; // a plan must take at most this share of the default's time
constexpr int kWarm      = 2;
constexpr int kReps      = 9;

std::mutex g_mutex;
// Layer key -> the tuned plan per T (index T; nullopt: the default stays).
std::map<std::string, std::array<std::optional<MoePlan>, kTuneMaxTokens + 1>> g_plans;
Json g_cache;
bool g_cache_loaded = false;

std::int32_t hidden_of(const MoeWeights& w) { return w.router.k; }

// Everything a plan's time depends on: the GPU class, both banks' format and layout, the geometry,
// the activation (a non-gated expert runs another stage), the expert policy and which banks have
// an R8 kernel (R8 candidates: a build that adds one for a format times again) and the compiled
// schedule sets (a build with other schedules times again).
std::string layer_key(const MoeWeights& w, const MoeConfig& c) {
    const bool gated = c.activation != MoeActivation::Relu2;
    const std::int32_t experts = w.router.n;
    const std::int32_t inter   = w.gate_up.n / ((gated ? 2 : 1) * experts);
    const auto bank = [](const Weight& b) {
        return std::to_string(static_cast<int>(b.qtype)) +
               (b.layout == QuantLayout::GgmlFrag ? "f" : "");
    };
    std::ostringstream k;
    k << device_class().id() << '|' << bank(w.gate_up) << '/' << bank(w.down) << "|E" << experts
      << "|k" << c.top_k << "|H" << hidden_of(w) << "|F" << inter << "|act"
      << static_cast<int>(c.activation) << (allows_a8(w.expert_policy) ? "|a8" : "|a16") << "|s"
      << schedule_count() << '.' << r8_schedule_count();
    if (allows_a8(w.expert_policy)) {
        k << (r8_expert_supports(w.gate_up) ? "g" : "") << (r8_expert_supports(w.down) ? "d" : "");
    }
    return k.str();
}

std::filesystem::path cache_path() {
    if (const auto path = sys::env("NINFERNO_MOE_TUNE_CACHE"); !path.empty()) { return path; }
    const auto dir = sys::cache_dir();
    return dir.empty() ? dir : dir / "moe_tune.json";
}

constexpr const char* kSchema = "ninferno-moe-tune/2";

void load_cache_locked() {
    if (g_cache_loaded) { return; }
    g_cache_loaded = true;
    g_cache         = Json{{"schema", kSchema}, {"layers", Json::object()}};
    // An unreadable cache is a cache miss: the layer is timed again and the file rewritten.
    if (std::optional<Json> j = read_cache_file(cache_path(), kSchema, "layers")) {
        g_cache = std::move(*j);
    }
}

std::string save_cache_locked() {
    const auto path = cache_path();
    if (path.empty()) { return "no cache directory (HOME unset)"; }
    try {
        save_cache_file(path, g_cache, kSchema, "layers");
        return {};
    } catch (const std::exception& error) {
        return std::string("cache not written: ") + error.what();
    }
}

// The plan a (gate/up, down) stage pair names, built as a route table's is (parse_plan): the
// router and job width follow from the stages, so a stored name gives back the timed plan.
std::optional<MoePlan> candidate(const MoePlan& base, int gu_schedule, int gu_r8, int dn_schedule,
                                 int dn_r8, std::int32_t hidden) {
    MoePlan p       = base;
    p.gate_up_schedule = gu_r8 >= 0 ? 0 : gu_schedule;
    p.gate_up_r8       = gu_r8;
    p.down_schedule    = dn_r8 >= 0 ? 0 : dn_schedule;
    p.down_r8          = dn_r8;
    p.a16              = false;
    return parse_plan(plan_name(p), hidden);
}

struct Bench {
    const MoeWeights& w;
    const MoeConfig& c;
    cudaStream_t stream;
    Tensor x, out;
    WorkspaceArena& workspace;
    TuneTimer& timer;

    float median_us(const MoePlan& p, std::int32_t tokens) {
        Tensor xt(x.data, DType::BF16, {x.ne[0], tokens});
        Tensor ot(out.data, DType::BF16, {out.ne[0], tokens});
        return timer.median_us([&](cudaStream_t s) {
            workspace.reset();
            moe_with_plan(xt, w, c, MoeEpilogue::Store, ot, workspace, s, p);
        });
    }
};

} // namespace

std::optional<MoePlan> tuned_plan(const MoeWeights& w, const MoeConfig& c, std::int32_t tokens) {
    if (tokens < 1 || tokens > kTuneMaxTokens) { return std::nullopt; }
    std::lock_guard lock(g_mutex);
    if (g_plans.empty()) { return std::nullopt; }
    const auto it = g_plans.find(layer_key(w, c));
    if (it == g_plans.end()) { return std::nullopt; }
    return it->second[static_cast<std::size_t>(tokens)];
}

void clear_tuned_plans() {
    std::lock_guard lock(g_mutex);
    g_plans.clear();
    g_cache_loaded = false;
}

} // namespace detail::moe

std::map<std::string, std::string> moe_tune_sources() {
    std::map<std::string, std::string> out;
    TuneMode mode = TuneMode::Off;
    try {
        mode = tune_mode("NINFERNO_MOE_TUNE");
    } catch (const std::invalid_argument&) {}
    if (mode != TuneMode::Off) {
        const auto path      = detail::moe::cache_path();
        auto user            = path.empty() ? std::nullopt : named_file_sha256(path);
        out["moe_tune.user"] = user ? std::move(*user) : std::string("absent");
    }
    return out;
}

MoeTuneReport tune_moe(const MoeWeights& weights, const MoeConfig& config, std::int32_t max_tokens,
                       cudaStream_t stream) {
    using namespace detail::moe;
    MoeTuneReport report;
    // Off unless asked: the engine starts with no timing, on the route table's MoE bands.
    const TuneMode mode = tune_mode("NINFERNO_MOE_TUNE");
    if (mode == TuneMode::Off) {
        report.summary = "moe tune: off (NINFERNO_MOE_TUNE=on times the layer)";
        return report;
    }
    max_tokens = std::clamp<std::int32_t>(max_tokens, 1, kTuneMaxTokens);
    const auto on_device = [](const void* p) {
        cudaPointerAttributes a{};
        if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        return a.type == cudaMemoryTypeDevice || a.type == cudaMemoryTypeManaged;
    };
    if (!on_device(weights.gate_up.qdata) || !on_device(weights.down.qdata)) {
        report.summary = "moe tune: routed experts not in device memory (offload): default plans";
        return report;
    }
    (void)select_plan(weights, config, 1, false); // validates the layer (throws as moe() would)
    const std::string key    = layer_key(weights, config);
    const std::int32_t hidden = hidden_of(weights);

    std::lock_guard lock(g_mutex);
    if (g_plans.count(key) != 0) {
        report.cached  = true;
        report.summary = "moe tune " + key + ": already tuned";
        return report;
    }
    load_cache_locked();
    auto& layers = g_cache["layers"];
    // A cached layer: its plans, if every name still parses and every T asked is there.
    if (mode != TuneMode::Retune && layers.contains(key)) {
        std::array<std::optional<MoePlan>, kTuneMaxTokens + 1> plans{};
        bool complete = true;
        const auto& entry = layers[key];
        for (std::int32_t t = 1; t <= max_tokens && complete; ++t) {
            const auto name = std::to_string(t);
            if (!entry.contains("plans") || !entry["plans"].contains(name)) {
                complete = false;
                break;
            }
            const auto& v = entry["plans"][name];
            if (v.is_null()) { continue; }
            const auto p = parse_plan(v.get<std::string>(), hidden);
            if (!p) { complete = false; } else { plans[static_cast<std::size_t>(t)] = *p; }
        }
        if (complete) {
            g_plans[key]   = plans;
            report.ran     = true;
            report.cached  = true;
            report.summary = "moe tune " + key + ": from " + cache_path().string();
            return report;
        }
    }

    // Time. Every buffer is the tuner's own and freed on return; the flush is skipped where the
    // device has no room for it (the times then read a warm L2, the ranking mostly holds).
    DeviceBuffer x(static_cast<std::size_t>(hidden) * max_tokens * 2);
    DeviceBuffer out(x.bytes);
    {
        std::mt19937 rng(20260930);
        std::normal_distribution<float> dist(0.0F, 1.0F);
        std::vector<__nv_bfloat16> host(static_cast<std::size_t>(hidden) * max_tokens);
        for (auto& v : host) { v = __float2bfloat16(dist(rng)); }
        x.copy_from_host(host.data(), x.bytes);
    }
    DeviceArena workspace(
        std::max<std::size_t>(moe_workspace_bytes(weights, config, 1, max_tokens), 256));
    TuneTimer timer(stream, kWarm, kReps);
    Bench bench{weights,
                config,
                stream,
                Tensor(x.p, DType::BF16, {hidden, max_tokens}),
                Tensor(out.p, DType::BF16, {hidden, max_tokens}),
                workspace,
                timer};

    std::array<std::optional<MoePlan>, kTuneMaxTokens + 1> plans{};
    Json entry{{"plans", Json::object()}, {"us", Json::object()}};
    std::ostringstream summary;
    summary << "moe tune " << key << ":";
    int improved = 0;
    try {
        for (std::int32_t t = 1; t <= max_tokens; ++t) {
            const auto name = std::to_string(t);
            entry["plans"][name] = nullptr;
            // A route-table band is measured and published: it stays.
            if (table_covers(weights, config, t)) { continue; }
            const MoePlan base = untuned_plan(weights, config, t, false);
            const float base_us = bench.median_us(base, t);
            struct Stage {
                int schedule = 0, r8 = -1;
            };
            const auto stages = [&](const Weight& bank) {
                std::vector<Stage> out;
                for (int s = 0; s < schedule_count(); ++s) { out.push_back({s, -1}); }
                if (r8_admits(weights, bank, t)) {
                    for (int s = 0; s < r8_schedule_count(); ++s) { out.push_back({0, s}); }
                }
                return out;
            };
            const Stage base_gu{base.gate_up_schedule, base.gate_up_r8};
            const Stage base_dn{base.down_schedule, base.down_r8};
            MoePlan best  = base;
            float best_us = base_us;
            Stage gu_best = base_gu;
            for (const auto gu : stages(weights.gate_up)) {
                const auto p = candidate(base, gu.schedule, gu.r8, base_dn.schedule, base_dn.r8, hidden);
                if (!p || !plan_compiled(weights, *p)) { continue; }
                const float us = bench.median_us(*p, t);
                if (us < best_us) {
                    best_us = us;
                    best    = *p;
                    gu_best = gu;
                }
            }
            for (const auto dn : stages(weights.down)) {
                const auto p = candidate(base, gu_best.schedule, gu_best.r8, dn.schedule, dn.r8, hidden);
                if (!p || !plan_compiled(weights, *p)) { continue; }
                const float us = bench.median_us(*p, t);
                if (us < best_us) {
                    best_us = us;
                    best    = *p;
                }
            }
            entry["us"][name] = {base_us, best_us};
            if (best_us <= kMargin * base_us) {
                plans[static_cast<std::size_t>(t)] = best;
                entry["plans"][name]               = plan_name(best);
                ++improved;
                summary << " T" << t << " " << format_us(base_us) << "->" << format_us(best_us)
                        << " us (" << plan_name(best) << ")";
            }
        }
    } catch (const std::exception& error) {
        (void)cudaGetLastError();
        report.summary = "moe tune " + key + ": failed (" + error.what() + "); default plans";
        return report;
    }
    g_plans[key] = plans;
    layers[key]  = entry;
    const auto saved = save_cache_locked();
    if (improved == 0) { summary << " the default plans stay at every T"; }
    if (!saved.empty()) { summary << "; " << saved; }
    report.ran      = true;
    report.improved = improved;
    report.summary  = summary.str();
    return report;
}

} // namespace ninferno::ops
