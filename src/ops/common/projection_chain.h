#pragma once

// Whole-chain pricing of the bounded fusion alternatives of a projection chain (#310): pure
// functions, no device and no CUDA, so a CPU reproduces the choice
// (docs/maintainer/deterministic-decisions.md). An alternative is a list of steps (kernel
// launches), each priced as it runs alone, its own reads and writes included: a projection by the
// generic cost model (generic::route_cost_seconds) or the shape's table, a pointwise kernel by the
// bytes it moves over the device's DRAM bandwidth (pointwise_seconds). A tensor a fusion keeps in
// registers is a tensor the unfused chain's producer writes and its consumer reads, so the chain's
// price carries the intermediate bytes through those steps; no constant is added for it. The
// launch count is no price here (no launch constant is fitted); it breaks a tie.
//
// Selection: the lowest total seconds among the priced alternatives; a tie goes to fewer
// launches, then to the earlier one in the caller's canonical order. An alternative with an
// unpriced step is not selectable; none priced selects nothing (the caller keeps its rule).

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::ops::detail::chain {

struct Step {
    std::optional<double> seconds; // nullopt: the cost model does not price this step
    int launches = 1;
};

struct Alternative {
    std::string name;
    std::vector<Step> steps;
};

// The chain's seconds and launches; nullopt when any step is unpriced or the chain is empty.
struct Price {
    double seconds = 0.0;
    int launches   = 0;
};

[[nodiscard]] inline std::optional<Price> price(const Alternative& alternative) {
    if (alternative.steps.empty()) { return std::nullopt; }
    Price total;
    for (const Step& step : alternative.steps) {
        if (!step.seconds || !(*step.seconds >= 0.0) || step.launches < 0) { return std::nullopt; }
        total.seconds += *step.seconds;
        total.launches += step.launches;
    }
    return total;
}

// A pointwise kernel's time: the bytes it reads and writes at the device's DRAM bandwidth (bytes
// per second, from the card's facts). nullopt without a bandwidth.
[[nodiscard]] inline std::optional<double>
pointwise_seconds(double bytes_read, double bytes_written, double dram_bytes_per_second) {
    if (!(dram_bytes_per_second > 0.0) || bytes_read < 0.0 || bytes_written < 0.0) {
        return std::nullopt;
    }
    return (bytes_read + bytes_written) / dram_bytes_per_second;
}

// The index of the selected alternative, nullopt when none is priced.
[[nodiscard]] inline std::optional<std::size_t>
select(const std::vector<Alternative>& alternatives) {
    std::optional<std::size_t> best;
    Price best_price;
    for (std::size_t i = 0; i < alternatives.size(); ++i) {
        const auto p = price(alternatives[i]);
        if (!p) { continue; }
        // Strictly better only: an equal alternative later in the order never displaces an
        // earlier one.
        const bool better = !best || p->seconds < best_price.seconds ||
                            (p->seconds == best_price.seconds && p->launches < best_price.launches);
        if (better) {
            best       = i;
            best_price = *p;
        }
    }
    return best;
}

// The alternatives of a gate/up projection pair that already exist, in canonical order:
//   0 "fused": one kernel writes both outputs (linear_pair's fused Q8 kernel); `fused` is that
//     kernel's time, `fused_launches` how many kernels it runs.
//   1 "separate": the two linear() calls.
struct PairPrices {
    std::optional<double> fused;
    int fused_launches = 1;
    std::optional<double> first;  // linear() for the first weight
    std::optional<double> second; // linear() for the second
};

[[nodiscard]] inline std::vector<Alternative> pair_alternatives(const PairPrices& p) {
    return {Alternative{"fused", {Step{p.fused, p.fused_launches}}},
            Alternative{"separate", {Step{p.first, 1}, Step{p.second, 1}}}};
}

// The alternatives of a SwiGLU gate/up projection (rows = 2 * inter, `tokens` columns), canonical:
//   0 "fused_epilogue": one projection whose epilogue forms silu(gate) * up (linear_swiglu); it
//     writes inter x tokens only.
//   1 "projection_then_pointwise": the plain projection writes the 2 x inter x tokens gate/up
//     matrix, a pointwise kernel reads it and writes inter x tokens.
// `fused` is the linear_swiglu kernel's time, `projection` the plain gate/up projection's;
// `activation_bytes` is the element size of the intermediate (2 for BF16).
struct SwigluPrices {
    std::optional<double> fused;
    std::optional<double> projection;
    std::int32_t inter           = 0;
    std::int32_t tokens          = 0;
    double activation_bytes      = 2.0;
    double dram_bytes_per_second = 0.0;
};

[[nodiscard]] inline std::vector<Alternative> swiglu_alternatives(const SwigluPrices& p) {
    const double out_bytes = static_cast<double>(p.inter) * p.tokens * p.activation_bytes;
    // The plain projection's time covers its own write of the 2 x inter x tokens matrix; the
    // pointwise kernel reads it back and writes the product.
    const auto pointwise = pointwise_seconds(2.0 * out_bytes, out_bytes, p.dram_bytes_per_second);
    return {Alternative{"fused_epilogue", {Step{p.fused, 1}}},
            Alternative{"projection_then_pointwise", {Step{p.projection, 1}, Step{pointwise, 1}}}};
}

} // namespace ninferno::ops::detail::chain
