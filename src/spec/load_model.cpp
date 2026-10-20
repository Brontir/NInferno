#include "spec/load_model.h"

#include <algorithm>
#include <stdexcept>

namespace ninferno::spec {

std::optional<RoundCost> round_cost_of(const GpuKey& gpu, double bytes_per_weight,
                                       bool pass_per_draft) {
    struct Arch {
        int cc;
        double flops_per_clock_sm; // dense FP16 tensor flops with FP32 accumulation, per SM and clock
        double eta;
        std::int32_t knee;
        double beta;
    };
    static constexpr Arch kArchs[] = {
        {80, 2048, 0.60, 24, 0.0045},
        {86, 512, 1.125, 32, 0.0155},
        {89, 512, 0.99, 8, 0.0069},
        {120, 512, 1.13, 12, 0.0122},
    };
    if (gpu.sms <= 0 || gpu.sm_clock_khz <= 0 || gpu.dram_gbps <= 0 || bytes_per_weight <= 0) {
        return std::nullopt;
    }
    for (const Arch& a : kArchs) {
        if (a.cc != gpu.compute_capability) { continue; }
        const double flops = a.eta * a.flops_per_clock_sm * gpu.sms * gpu.sm_clock_khz * 1e3;
        return RoundCost{.alpha = 2.0 * gpu.dram_gbps * 1e9 / (flops * bytes_per_weight),
                         .beta  = a.beta,
                         .knee  = a.knee,
                         .delta = pass_per_draft ? 0.070 : 0.060,
                         .pass_per_draft = pass_per_draft};
    }
    return std::nullopt;
}

double round_time(const RoundCost& cost, std::int32_t rows, std::int32_t depth) {
    if (rows < 1 || depth < 0) { throw std::invalid_argument("round_time: rows >= 1, depth >= 0"); }
    const double w      = static_cast<double>(rows) * (depth + 1);
    const double passes = depth == 0 ? 0.0 : (cost.pass_per_draft ? depth : 1.0);
    return 1.0 + cost.alpha * (w - 1.0) + cost.beta * std::max(0.0, w - cost.knee) +
           passes * cost.delta;
}

double tokens_per_round(double a, double d, std::int32_t depth) {
    double tokens = 1.0, accepted = 1.0;
    for (std::int32_t i = 0; i < depth; ++i) {
        double p = a;
        for (std::int32_t j = 0; j < i; ++j) { p *= d; }
        accepted *= p;
        tokens += accepted;
    }
    return tokens;
}

double first_draft_acceptance(double tokens, std::int32_t depth, double d) {
    if (depth < 1 || tokens <= 1.0) { return 0.0; }
    double lo = 0.0, hi = 0.999;
    if (tokens_per_round(hi, d, depth) <= tokens) { return hi; }
    for (int i = 0; i < 60; ++i) {
        const double mid = 0.5 * (lo + hi);
        (tokens_per_round(mid, d, depth) < tokens ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
}

std::vector<LoadBound> modeled_load(const RoundCost& cost, double a, double d,
                                    std::span<const std::int32_t> depths,
                                    std::int32_t max_rows) {
    std::vector<std::int32_t> candidates(depths.begin(), depths.end());
    candidates.push_back(0);
    std::sort(candidates.begin(), candidates.end());
    const auto best_at = [&](std::int32_t rows, std::int32_t ceiling) {
        std::int32_t best = 0;
        double speed      = 1.0 / round_time(cost, rows, 0);
        for (const std::int32_t k : candidates) {
            if (k == 0 || k > ceiling) { continue; }
            const double s = tokens_per_round(a, d, k) / round_time(cost, rows, k);
            if (s > speed) { speed = s; best = k; }
        }
        return best;
    };
    const std::int32_t deepest = candidates.back();
    std::int32_t previous      = best_at(1, deepest);
    std::vector<LoadBound> out;
    if (previous != deepest) { out.push_back(LoadBound{.lanes = 1, .depth = previous}); }
    for (std::int32_t rows = 2; rows <= max_rows; ++rows) {
        const std::int32_t depth = best_at(rows, previous);
        if (depth != previous) {
            out.push_back(LoadBound{.lanes = rows, .depth = depth});
            previous = depth;
        }
        if (depth == 0) { break; }
    }
    return out;
}

} // namespace ninferno::spec
