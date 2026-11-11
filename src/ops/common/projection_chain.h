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
// L2 beside it).
//
// Launches: a launch has a measured cost per generation, but this header holds none. Two
// constants exist elsewhere, and they differ. kLaunchTime (src/spec/round_cost.h) is a true
// per-launch t0 inside graph replay (1.5e-6 s, Ada only): that is the number to pass as
// `launch_seconds`. kKernelCost (src/product/choice/choice_constants.h) is not a pure launch cost:
// it is a kernel's residual beyond its bytes per generation, ramp and epilogue included; do not
// pass it as a launch price. select() adds `launch_seconds` to every launch of a chain; 0 (the
// default) prices no launch, and the launch count then only breaks a tie. Beware of counting a
// launch twice: if the steps' seconds already include the launch (a route_cost_seconds that
// carries a launch term, a kernel residual), a nonzero `launch_seconds` double-counts it. Roadmap
// #617 step B unifies the constants; they are not moved here.
//
// Selection, as #600 picks within the resolution: pass 1 takes the anchor, the lowest total
// seconds (steps plus launches x launch_seconds) among the priced alternatives, the earliest on
// equal totals. Pass 2 keeps the anchor unless an alternative whose ln(total / anchor) is within
// `resolution` runs strictly fewer launches; among those, the fewest, then the earliest in the
// caller's canonical order. An alternative whose total is not finite is unpriced. The
// resolution is the speed estimate's (#301, DECISIONS 75), which the caller reads where it prices
// (product::choice::Choice::resolution: the reference card's generation's measured decode error,
// worth_band(card), in ln time units; this device-free header does not include it). The pick lies
// within the band of the minimum and does not depend on the order of the alternatives except
// through the two tie-breaks. A resolution of 0 compares exactly. An alternative with an unpriced
// step (or a non-finite one, or a step of no launch) is not selectable; none priced selects
// nothing (the caller keeps its rule).

#include <algorithm>
#include <cmath>
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
        if (!std::isfinite(total.seconds)) { return std::nullopt; } // the sum overflowed
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

// Whether two totals are indistinguishable at `resolution` (the ln of their ratio at most). Equal
// totals always are; a zero total is within the band of nothing but another zero.
[[nodiscard]] inline bool within_resolution(double a, double b, double resolution) {
    if (a == b) { return true; }
    if (!(resolution > 0.0) || !(a > 0.0) || !(b > 0.0)) { return false; }
    return std::abs(std::log(a / b)) <= resolution;
}

// The index of the selected alternative, nullopt when none is priced. `resolution` is in ln time
// units (0, negative or non-finite: exact comparison); `launch_seconds` is added per launch (0,
// negative or non-finite: launches are no price). Where a caller gets them: the header comment.
// FUSION-SITE: projection_chain
[[nodiscard]] inline std::optional<std::size_t> select(const std::vector<Alternative>& alternatives,
                                                       double resolution     = 0.0,
                                                       double launch_seconds = 0.0) {
    if (!std::isfinite(resolution) || !(resolution > 0.0)) { resolution = 0.0; }
    if (!std::isfinite(launch_seconds) || !(launch_seconds > 0.0)) { launch_seconds = 0.0; }
    std::vector<std::optional<Price>> priced(alternatives.size());
    std::vector<double> totals(alternatives.size(), 0.0);
    // Pass 1: the anchor, the lowest total, the earliest on equal totals.
    std::optional<std::size_t> anchor;
    for (std::size_t i = 0; i < alternatives.size(); ++i) {
        priced[i] = price(alternatives[i]);
        if (!priced[i]) { continue; }
        totals[i] = priced[i]->seconds + priced[i]->launches * launch_seconds;
        if (!std::isfinite(totals[i])) {
            priced[i].reset(); // the launch term overflowed
            continue;
        }
        if (!anchor || totals[i] < totals[*anchor]) { anchor = i; }
    }
    if (!anchor) { return std::nullopt; }
    // Pass 2, #600's rule: the anchor stays unless an alternative within the resolution of it
    // runs strictly fewer launches; among those, the fewest, the earliest on equal launches. Not
    // order dependent.
    std::size_t best = *anchor;
    for (std::size_t i = 0; i < alternatives.size(); ++i) {
        if (!priced[i] || !within_resolution(totals[i], totals[*anchor], resolution)) { continue; }
        const int launches = priced[i]->launches;
        const int held     = priced[best]->launches;
        if (launches < held) { best = i; } // ascending: the earliest of the fewest
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
