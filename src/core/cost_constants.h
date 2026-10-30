#pragma once

// Chip facts both speed models read: the configuration choice's step model
// (product/choice/choice_constants.h) and the speculation round cost (spec/round_cost.h). One
// definition each, so the two models cannot drift apart. Fitted calibrations do not live here (the
// MoE routed share: spec/moe_routed_share.h).

namespace ninferno {

// dp4a's int8 rate a SM-clock: a generation without an int8 MMA (Volta) runs its int8 route at it.
// Derived from the CUDA C++ Programming Guide's arithmetic throughput table, compute capability
// 7.0: 64 32-bit integer multiply-adds a SM-clock, each dp4a 4 int8 multiply-adds (2 operations
// each).
inline constexpr double kDp4aOpsPerSmClock = 512;

} // namespace ninferno
