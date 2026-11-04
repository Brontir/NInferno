#pragma once

// The decision manifest's route lines (#304): for every distinct problem the generic projection
// path ran during this start (ops::detail::generic::seen_problems: the loading thread's own,
// cleared when an Engine loads), the route the selector selects (select_route, under the start's
// route profile) at each decode and verify width (1..16) and at the prefill chunk. A bounded,
// canonical set: one line per problem, the widths as runs of one route, no per-token trace, and a
// count line.
//
// What a line does not say: a call's own layout can still take it off the selected route
// (resolve_route: alignment, workspace, segments; workspace_free_route), a weight's column order
// or stacked divisors are not described, and a problem first run after startup (a path the
// warm-up does not take: vision projections, a later drafter) is not in the section. A start that
// runs no warm-up (every size pinned) records none and says so in the count line.
//
// A line holds everything its replay needs besides the board's class and the route tables: the
// format, policy, epilogue, N, K, the stored rows and layout, and the chunk. replay_routes asks the
// selector again from the lines alone, so a manifest can be checked on a CPU against a board's
// class (and the tables of a release): the same lines, or the lines that moved.
//
//   "<format> <policy> <epilogue> <N>x<K> of <stored rows> <layout name> <speed|invariant>"
//       = "1-3 <route>, 4-16 <route>[; one lane <runs>][; several lanes <runs>]; prefill <chunk>
//          <route>"   ("none": the path does not read such a weight). The runs by width alone,
//         then those of a verify round of one sequence and of several (ScopedRoundLanes) where
//         they differ.

#include "core/device_class.h"
#include "ops/generic/generic_gemm.h"
#include "runtime/engine/decision_fingerprint.h"

#include <cstdint>
#include <span>

namespace ninferno::runtime {

inline constexpr std::int32_t kDecisionRouteWidths = 16; // decode and verify widths 1..16
inline constexpr const char* kDecisionRouteCount   = "problems"; // the count line's key
inline constexpr int kDecisionRouteSeveralLanes    = 4;          // "several lanes": a C4 round

// The route lines of `problems` on `device` (route_query: the board's class, the build and the
// route tables the default lookup reads for it, or the device's own where it is the current one),
// under the invariant route profile where `invariant` (an exact start).
[[nodiscard]] DecisionSection
routes_of(const DeviceClass& device, std::span<const ops::detail::generic::SeenProblem> problems,
          std::int32_t prefill_chunk, bool invariant);

// The lines `recorded` asked again on `device`: each key parsed back into its problem and chunk.
// A key that does not parse is std::invalid_argument.
[[nodiscard]] DecisionSection replay_routes(const DeviceClass& device,
                                            const DecisionSection& recorded);

} // namespace ninferno::runtime
