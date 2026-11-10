#pragma once

// Chip facts both speed models read: the configuration choice's step model
// (product/choice/choice_constants.h) and the speculation round cost (spec/round_cost.h). One
// definition each, so the two models cannot drift apart. Fitted calibrations do not live here (the
// MoE routed share: spec/moe_routed_share.h).

#include <string_view>

namespace ninferno {

// dp4a's int8 rate a SM-clock: a generation without an int8 MMA (Volta) runs its int8 route at it.
// Derived from the CUDA C++ Programming Guide's arithmetic throughput table, compute capability
// 7.0: 64 32-bit integer multiply-adds a SM-clock, each dp4a 4 int8 multiply-adds (2 operations
// each).
inline constexpr double kDp4aOpsPerSmClock = 512;

// The cost of a kernel launch node inside graph replay: the one place a launch is priced (#617
// step B). Readers: the speculation round cost (spec/round_cost.h kLaunchTime) and, through its
// `launch_seconds` argument, the chain pricing (ops/common/projection_chain.h). Other launch-like
// numbers are different facts and keep their own names, all pinned in tests/test_launch_node.cpp:
//   - route_cost::kQuantLaunchSeconds (1.7e-6, Fitted): the A8 activation quantizer kernel, its
//     launch plus its scale and sum planes, inside the route model; not a bare node.
//   - choice kKernelCost: a kernel's residual beyond its bytes (ramp and epilogue included), not a
//     launch.
//   - core/pdl.cuh: a captured node with the programmatic attribute takes 1.3-1.5 us against
//     0.84-0.96 us without it (RTX 3090); a comment, read by no model.
// A generation without a row is unknown (0): launches then only break a tie.
enum class LaunchNodeBasis { Counted, Fitted };

struct LaunchNode {
    std::string_view generation;
    double seconds;
    LaunchNodeBasis basis;
    // Holdout: none recorded (the record type of PR #635 is not on master). The row was fitted
    // from the Ada census (RTX 4060 Ti, 4070 Ti, 4090) as the per-launch t0 inside graph replay.
    std::string_view holdout;
};

inline constexpr LaunchNode kLaunchNode[] = {
    {"ada", 1.5e-6, LaunchNodeBasis::Fitted, "none recorded"},
};

[[nodiscard]] constexpr double launch_node_seconds(std::string_view generation) noexcept {
    for (const auto& n : kLaunchNode) {
        if (n.generation == generation) { return n.seconds; }
    }
    return 0.0;
}

} // namespace ninferno
