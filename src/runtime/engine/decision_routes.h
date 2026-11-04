#pragma once

// The decision manifest's route lines (#304): for every distinct problem the generic projection
// path ran in this process (ops::detail::generic::seen_problems), the route the selector takes at
// each decode and verify width (1..16) and at the prefill chunk. A bounded, canonical set: one line
// per problem, the widths as runs of one route, no per-token trace.
//
// A line holds everything its replay needs besides the board's class and the route tables: the
// format, policy, epilogue, N, K, the stored rows and layout, and the chunk. replay_routes asks the
// selector again from the lines alone, so a manifest can be checked on a CPU against a board's
// class (and the tables of a release): the same lines, or the lines that moved.
//
//   "<format> <policy> <epilogue> <N>x<K> of <stored rows> layout <layout>"
//       = "1-3 <route>, 4-16 <route>; prefill <chunk> <route>"   ("none": the path does not read
//         such a weight)

#include "core/device_class.h"
#include "ops/generic/generic_gemm.h"
#include "runtime/engine/decision_fingerprint.h"

#include <cstdint>
#include <span>

namespace ninferno::runtime {

inline constexpr std::int32_t kDecisionRouteWidths = 16; // decode and verify widths 1..16

// The route lines of `problems` on `device` (route_query: the board's class, the build and the
// route tables the default lookup reads for it, or the device's own where it is the current one).
[[nodiscard]] DecisionSection
routes_of(const DeviceClass& device, std::span<const ops::detail::generic::SeenProblem> problems,
          std::int32_t prefill_chunk);

// The lines `recorded` asked again on `device`: each key parsed back into its problem and chunk.
// A key that does not parse is std::invalid_argument.
[[nodiscard]] DecisionSection replay_routes(const DeviceClass& device,
                                            const DecisionSection& recorded);

} // namespace ninferno::runtime
