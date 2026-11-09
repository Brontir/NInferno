#pragma once

// Whole-chain pricing of the bounded fusion alternatives of a projection chain (#310): pure
// functions, no device and no CUDA, so a CPU reproduces the choice
// (docs/maintainer/deterministic-decisions.md). An alternative is a list of steps (kernel
// launches), each priced as it runs alone: a projection by the generic cost model
// (generic::route_cost_seconds) or the shape table, a pointwise kernel by the bytes it moves.
// route_cost_seconds prices a projection's weights (and, for the heavy register-MMA path, its L2
// reads) but not its output write, so projection_step_seconds adds that write explicitly (output
// bytes over the DRAM bandwidth), the same way for every alternative. A tensor a fusion keeps in
// registers is one the unfused chain's producer writes and its consumer reads: the larger write is
// in the producer's projection step, the read in the pointwise step, which runs at the L2's
// bandwidth when the tensor fits in the L2 and at the DRAM's otherwise. The memory numbers are
// numbers the caller passes (the facts' DRAM bandwidth, the device's L2 size, and the L2 bandwidth
// the generic cost model uses: a fitted rate measured on sm_86 and sm_120 only,
// route_cost_constants.h, so a caller passes it only where that model prices its L2 term and 0
// elsewhere, which reads at the DRAM's); no constant is added here. The tensor counts as held in
// the L2 when twice its bytes fit, the generic model's own rule (its weights stream through the
// L2 beside it). The launch count is no price (no launch
// constant is fitted); it breaks a tie.
//
// Selection: the lowest total seconds among the priced alternatives; totals within a relative 1e-9
// of each other are a tie (the repo's kWorthTie band, product/choice/choice_constants.h), and a tie
// goes to fewer launches, then to the earlier alternative in the caller's canonical order. An
// alternative with an unpriced step (or a non-finite one, or a step of no launch) is not
// selectable; none priced selects nothing (the caller keeps its rule).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::ops::detail::chain {

// Seconds within this relative band are a tie.
inline constexpr double kTie = 1e-9;

struct Step {
    std::optional<double> seconds; // nullopt: the cost model does not price this step
    int launches = 1;
};

struct Alternative {
    std::string name;
    std::vector<Step> steps;
};

// The chain's seconds and launches; nullopt when any step is unpriced, non-finite or negative, a
// step has no launch, or the chain is empty.
struct Price {
    double seconds = 0.0;
    int launches   = 0;
};

[[nodiscard]] inline std::optional<Price> price(const Alternative& alternative) {
    if (alternative.steps.empty()) { return std::nullopt; }
    Price total;
    for (const Step& step : alternative.steps) {
        if (!step.seconds || !std::isfinite(*step.seconds) || !(*step.seconds >= 0.0) ||
            step.launches <= 0) {
            return std::nullopt;
        }
        total.seconds += *step.seconds;
        total.launches += step.launches;
    }
    return total;
}

// The device's memory facts, as the caller reads them (0: unknown).
struct Memory {
    double dram_bytes_per_second = 0.0;
    double l2_bytes              = 0.0;
    double l2_bytes_per_second   = 0.0; // the whole device's, as the generic cost model has it
};

// A projection step: the cost model's seconds plus its output write to the DRAM, which that model
// does not price. nullopt when either is unknown.
[[nodiscard]] inline std::optional<double>
projection_step_seconds(std::optional<double> route_seconds, double output_bytes,
                        const Memory& memory) {
    if (!route_seconds || !(memory.dram_bytes_per_second > 0.0) || !(output_bytes >= 0.0)) {
        return std::nullopt;
    }
    return *route_seconds + output_bytes / memory.dram_bytes_per_second;
}

// A pointwise kernel's time: `bytes_read` of the producer's output (`resident_bytes` of it, written
// just before) at the L2's bandwidth when that tensor fits in the L2 and the L2 bandwidth is
// known, else at the DRAM's; `bytes_written` at the DRAM's. nullopt without a DRAM bandwidth.
[[nodiscard]] inline std::optional<double> pointwise_seconds(double bytes_read,
                                                             double bytes_written,
                                                             double resident_bytes,
                                                             const Memory& memory) {
    if (!(memory.dram_bytes_per_second > 0.0) || !(bytes_read >= 0.0) || !(bytes_written >= 0.0) ||
        !(resident_bytes >= 0.0)) {
        return std::nullopt;
    }
    const bool in_l2 = memory.l2_bytes > 0.0 && memory.l2_bytes_per_second > 0.0 &&
                       2.0 * resident_bytes <= memory.l2_bytes;
    const double read_bw = in_l2 ? memory.l2_bytes_per_second : memory.dram_bytes_per_second;
    return bytes_read / read_bw + bytes_written / memory.dram_bytes_per_second;
}

// The index of the selected alternative, nullopt when none is priced.
[[nodiscard]] inline std::optional<std::size_t>
select(const std::vector<Alternative>& alternatives) {
    std::optional<std::size_t> best;
    Price best_price;
    for (std::size_t i = 0; i < alternatives.size(); ++i) {
        const auto p = price(alternatives[i]);
        if (!p) { continue; }
        bool better = !best;
        if (best) {
            const double tol = kTie * std::max(p->seconds, best_price.seconds);
            // Strictly better only: a tie that fewer launches do not break leaves the earlier one.
            better = p->seconds < best_price.seconds - tol ||
                     (std::abs(p->seconds - best_price.seconds) <= tol &&
                      p->launches < best_price.launches);
        }
        if (better) {
            best       = i;
            best_price = *p;
        }
    }
    return best;
}

// The alternatives of a gate/up projection pair that already exist, in canonical order:
//   0 "fused": one kernel writes both outputs (linear_pair's fused Q8 kernel); `fused` is that
//     kernel's cost-model time, `fused_launches` how many kernels it runs.
//   1 "separate": the two linear() calls.
// The route seconds are the cost model's (no output write); each alternative writes the same two
// outputs, `output_bytes` each, added by one helper.
struct PairPrices {
    std::optional<double> fused;
    int fused_launches = 1;
    std::optional<double> first;  // linear() for the first weight
    std::optional<double> second; // linear() for the second
    double output_bytes = 0.0;    // one weight's output: rows x tokens x element size
    Memory memory;
};

[[nodiscard]] inline std::vector<Alternative> pair_alternatives(const PairPrices& p) {
    const auto one = [&](const std::optional<double>& s) {
        return projection_step_seconds(s, p.output_bytes, p.memory);
    };
    return {Alternative{"fused",
                        {Step{projection_step_seconds(p.fused, 2.0 * p.output_bytes, p.memory),
                              p.fused_launches}}},
            Alternative{"separate", {Step{one(p.first), 1}, Step{one(p.second), 1}}}};
}

// The alternatives of a SwiGLU gate/up projection (rows = 2 * inter, `tokens` columns), canonical:
//   0 "fused_epilogue": one projection whose epilogue forms silu(gate) * up (linear_swiglu); it
//     writes inter x tokens only.
//   1 "projection_then_pointwise": the plain projection writes the 2 x inter x tokens gate/up
//     matrix, a pointwise kernel reads it and writes inter x tokens.
// `fused` is the linear_swiglu kernel's cost-model time, `projection` the plain gate/up
// projection's; `activation_bytes` is the element size of the intermediate (2 for BF16).
struct SwigluPrices {
    std::optional<double> fused;
    std::optional<double> projection;
    std::int32_t inter      = 0;
    std::int32_t tokens     = 0;
    double activation_bytes = 2.0;
    Memory memory;
};

[[nodiscard]] inline std::vector<Alternative> swiglu_alternatives(const SwigluPrices& p) {
    const double out_bytes = static_cast<double>(p.inter) * p.tokens * p.activation_bytes;
    // The plain projection writes the 2 x inter x tokens matrix (in its step); the pointwise
    // kernel reads it back, from the L2 when it fits, and writes the product.
    const auto pointwise = pointwise_seconds(2.0 * out_bytes, out_bytes, 2.0 * out_bytes, p.memory);
    return {Alternative{"fused_epilogue",
                        {Step{projection_step_seconds(p.fused, out_bytes, p.memory), 1}}},
            Alternative{"projection_then_pointwise",
                        {Step{projection_step_seconds(p.projection, 2.0 * out_bytes, p.memory), 1},
                         Step{pointwise, 1}}}};
}

} // namespace ninferno::ops::detail::chain
