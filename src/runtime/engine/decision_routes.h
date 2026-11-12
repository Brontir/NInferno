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
// format, policy, epilogue, N, K, the stored rows and layout, and the chunk; and the start's
// --route-policy (#614 F1), which is the manifest's options.maintainer.route_policy
// (route_policy_of) and is passed to the replay. replay_routes asks the selector again from the
// lines and that policy alone, so a manifest can be checked on a CPU against a board's class (and
// the tables of a release): the same lines, or the lines that moved. Every function here asks the
// selector under the policy it is given (ops::ScopedRoutePolicy: process-wide for the call, so
// not beside a live Engine of another policy).
//
//   "<format> <policy> <epilogue> <N>x<K> of <stored rows> <layout name> <speed|invariant>"
//       = "1-3 <route>, 4-16 <route>[; one lane <runs>][; several lanes <runs>]; prefill <chunk>
//          <route>"   ("none": the path does not read such a weight). The runs by width alone,
//         then those of a verify round of one sequence and of several (ScopedRoundLanes) where
//         they differ.
//
// The route_evidence section (#614, A3) has the same keys and the same shape of line, each run
// naming the evidence beside the route: "<route> authority=<name> evidence=<kind>[ donor=<board>
// d=<distance>]" (route_authorities.h evidence_text). Its routes are those of the routes section
// (both ask route_answer_query, which is deterministic; each section makes its own pass). A
// lanes clause appears where that round's runs, evidence included, differ from the width-alone
// runs, so an evidence line can carry one where its routes line has none. It is in the manifest
// text, not in the fingerprint and not
// in the manifest's hash (decision_fingerprint.h), so the id of every existing manifest holds.

#include "core/device_class.h"
#include "ninferno/option_enums.h"
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
[[nodiscard]] DecisionSection routes_of(const DeviceClass& device,
                                        std::span<const ops::detail::generic::SeenProblem> problems,
                                        std::int32_t prefill_chunk, bool invariant,
                                        RoutePolicy policy = RoutePolicy::Legacy);

// The route_evidence lines of `problems`, as routes_of's with the evidence of each answer.
[[nodiscard]] DecisionSection route_evidence_of(
    const DeviceClass& device, std::span<const ops::detail::generic::SeenProblem> problems,
    std::int32_t prefill_chunk, bool invariant, RoutePolicy policy = RoutePolicy::Legacy);

// The route policy a manifest's options section records (maintainer.route_policy); legacy where
// the key is absent (a manifest from before F1). A value that is no policy is
// std::invalid_argument.
[[nodiscard]] RoutePolicy route_policy_of(const DecisionSection& options);

// The lines `recorded` asked again on `device` under `policy` (route_policy_of the manifest's
// options): each key parsed back into its problem and chunk. A key that does not parse is
// std::invalid_argument.
[[nodiscard]] DecisionSection replay_routes(const DeviceClass& device,
                                            const DecisionSection& recorded,
                                            RoutePolicy policy = RoutePolicy::Legacy);
// The same for a recorded route_evidence section.
[[nodiscard]] DecisionSection replay_route_evidence(const DeviceClass& device,
                                                    const DecisionSection& recorded,
                                                    RoutePolicy policy = RoutePolicy::Legacy);

} // namespace ninferno::runtime
