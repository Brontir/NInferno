#pragma once

// The generic selector as a pure query: the route a call of the projection path takes on a board
// given by its class (core/device_class.h) rather than the current device. The answer is
// select_route's own (the same rules, tables and build), so a cost model outside the kernels reads
// the selector instead of restating it. What a route costs at a width is the kernels' own model to
// give (the route time model that builds on this query), not a classification here: the kernel a
// route launches can depend on the call (Volta's HMMA verify, ggml MMQ's column tile).
//
// Nothing is read but `device`, the build and the route tables on disk, with one exception: at 17
// tokens or more a gated register-MMA price (rmma_terms_measured) reads the registers and spill of
// the HOST GPU's compiled kernels (rmma_resources), whatever board the query substitutes. The query
// substitutes the class for the call (ScopedDeviceClass), and the selector's per-device state keeps
// a substitute's answers apart by its content (core/per_device.h, ops/route_table.cpp; routes timed
// on the device by the dense tuner are not a foreign substitute's). A weight is described by its
// format and shape alone, in the layout the converter stores it in (default_weight_layout); its
// bytes are never read.

#include "core/device_class.h"
#include "core/weight.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_authorities.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::ops::detail::generic {

// The route a call of `tokens` tokens of an `n` x `k` projection of `qtype` under `policy` and
// `epilogue` takes on `device` (SwiGLU: `n` output rows, the weight 2 n rows); nullopt where the
// generic path does not read such a weight (generic::supports).
// `parent_rows`: the rows of the stored matrix the call reads `n` of (a fused QKV or stacked
// projection's slice; 0: the call's own): the weight is described whole, as the engine passes a
// slice (native_weight of its complete parent, the call's rows in Problem::n). `policy` is the
// call's as the engine passes it: an NVFP4 weight without a calibrated activation divisor runs
// AllowA8, not AllowA4 (ops/weight_input.cpp projection_slice). The route tables read are the
// default lookup's for `device`; where `device` is the current device's own class, that device's
// (an explicit --route-table included). select_route's exceptions propagate. `layout`: the stored
// layout where it is not the converter's default for the format (a recorded call's, SeenProblem).
[[nodiscard]] std::optional<Route> route_query(const DeviceClass& device, QType qtype,
                                               LinearPolicy policy, Epilogue epilogue,
                                               std::int32_t n, std::int32_t k, std::int32_t tokens,
                                               bool prefill = false, std::int32_t parent_rows = 0,
                                               std::optional<QuantLayout> layout = std::nullopt);

// route_query's answer with the evidence it rests on (#614, A3): the route and the authority that
// answered, from the one selection (select_answer) route_query returns the route of. nullopt where
// route_query gives none.
[[nodiscard]] std::optional<RouteAnswer>
route_answer_query(const DeviceClass& device, QType qtype, LinearPolicy policy, Epilogue epilogue,
                   std::int32_t n, std::int32_t k, std::int32_t tokens, bool prefill = false,
                   std::int32_t parent_rows = 0, std::optional<QuantLayout> layout = std::nullopt);

// The bytes a weight value of such a call takes (bytes_per_weight) on the weight route_query
// describes for it: what the decision manifest's replay of a recorded call prices (#704). A row
// view's live price (Weight::priced_bytes) is this, to the bit. Throws std::invalid_argument for a
// non-positive N or K, a negative `parent_rows` or a shape the layout cannot hold.
[[nodiscard]] double described_bytes_per_weight(QType qtype, LinearPolicy policy, Epilogue epilogue,
                                                std::int32_t n, std::int32_t k,
                                                std::int32_t parent_rows          = 0,
                                                std::optional<QuantLayout> layout = std::nullopt);

// The route such a call selects and the one it runs on (#298): route_query's choice as run() hands
// it on (workspace_free_route: no workspace, no quantized family that needs one), resolved as
// run_route resolves it for a call of `layout` (its alignment, stride, segments and the workspace
// free to it; its tokens, rows and epilogue are the query's own). The default layout is the
// engine's contiguous, aligned call with `workspace_free` bytes free (nullopt: no workspace).
// nullopt where route_query gives none.
[[nodiscard]] std::optional<ResolvedRoute>
resolved_route_query(const DeviceClass& device, QType qtype, LinearPolicy policy, Epilogue epilogue,
                     std::int32_t n, std::int32_t k, std::int32_t tokens,
                     std::optional<std::size_t> workspace_free, bool prefill = false,
                     std::int32_t parent_rows = 0, bool x_aligned16 = true, int ldx = 0,
                     int segments = 1);

// What a route is, for a call (#298): one read-only description assembled from the facts the
// launch path and the cost model already own; nothing here is stored or fitted. Every field is the
// answer of the function the launch (or the selector) asks: `resolved` of resolve_route, the
// workspace of route_workspace_bytes (ggml's matrix kernel: mmq_workspace_bytes, the slice
// run_route launches; a register MMA reading the group-sums plane: rmma_group_sums_bytes), the tile
// and traffic of route_tile and route_traffic (declared geometry and a prediction of requested
// bytes, not counters), `prepared` of prepared_rule, the price of route_cost_seconds, and the
// kernel of route_resources (the compiled registers, shared memory, threads and grid, read from the
// HOST GPU's binary: only when asked for). Not device-free otherwise either: at 17 tokens or more a
// gated register-MMA price (select_route, route_cost_seconds) reads the host GPU's compiled rmma
// facts, whatever `device` substitutes. The description is of the call route_query describes: the
// engine's contiguous, aligned activation, the weight in the converter's layout (see above), not a
// recorded call's own layout.
struct RouteDescription {
    ResolvedRoute resolved;
    std::size_t workspace_bytes = 0; // what `resolved.executed` takes from the call's workspace
    std::optional<RouteTile> tile;
    std::optional<RouteTraffic> traffic;
    std::optional<PreparedRule> prepared;
    // cost_route's time where it prices the route, over the call's share of the weight: the
    // selector's own number for a slice of a fused parent too (bytes_per_weight, #640).
    std::optional<double> cost_seconds;
    std::optional<KernelResources> kernel;
    // The authority and evidence of the route the call asked for (`resolved.asked`), from the
    // walk that gave that route (describe_call): select_answer's, or, for a call without workspace
    // that takes the A16 heuristic's route (workspace_free_route), that heuristic's. Empty for
    // describe_route, whose route is the caller's, not a selection. resolve_route can still demote
    // the asked route afterwards (`resolved.reason`); the evidence is of the asked route.
    std::optional<Evidence> evidence;
};

// The description of a call that runs `asked` (an explicit route, as the tuner and the pin tests
// give one): nullopt where the generic path does not read such a weight or route_admits refuses the
// route for it. `workspace_free`: as resolved_route_query's. `with_kernel` reads the compiled
// kernel's resources on the host GPU (CUDA required; a failed read leaves `kernel` empty).
[[nodiscard]] std::optional<RouteDescription>
describe_route(const DeviceClass& device, Route asked, QType qtype, LinearPolicy policy,
               Epilogue epilogue, std::int32_t n, std::int32_t k, std::int32_t tokens,
               std::optional<std::size_t> workspace_free, std::int32_t parent_rows = 0,
               bool with_kernel = false);

// The description of the call the selector routes: the route run() hands on
// (resolved_route_query's choice), described as describe_route does. nullopt where route_query
// gives none.
[[nodiscard]] std::optional<RouteDescription>
describe_call(const DeviceClass& device, QType qtype, LinearPolicy policy, Epilogue epilogue,
              std::int32_t n, std::int32_t k, std::int32_t tokens,
              std::optional<std::size_t> workspace_free, bool prefill = false,
              std::int32_t parent_rows = 0, bool with_kernel = false);

// One line of key=value fields: "asked=<route> runs=<route> [demoted=<why>] [r8_prequantized]
// workspace=<bytes> [tile=<rows>x<tokens>[ ksplit=<n>]] [weight_bytes=<b> activation_bytes=<b>
// output_bytes=<b>] [prepared=<rep>] [cost_us=<t>] [regs=<r> local=<b> smem=<static>+<dynamic>
// threads=<t> grid=<x>x<y>x<z>] [authority=<name> evidence=<kind> [donor=<board> d=<distance>]]".
// The last group (evidence_text) is present where `evidence` is: a described call, not an explicit
// route; the fields before it are unchanged.
[[nodiscard]] std::string describe_text(const RouteDescription& description);

// The header of the route tuner's --selected-out CSV: the columns of every version before
// (pci_device_id last among them, #684), then "authority,evidence" (evidence_columns) and
// "ranker" (ranker_answer's route, #681) at the end, which tools/bench/regret_collect.py and
// route_table_prune.py read by name.
inline constexpr const char* kSelectedCsvHeader =
    "format,compute,epilogue,n,k,T,asked,selected,select_ns,run,device,pci_device_id,"
    "authority,evidence,ranker";

// The selector's cost model as a pure query (price_routes, generic_cost.cpp): every route it
// offers for such a call on `device`, with its time in seconds (the A16 scale applied), in the
// order it offers them. Empty where the facts have no roofline, the generic path does not read
// such a weight, or the width is not priced. It prices any width it is given: the selector reads
// these prices (cost_route) only from 17 tokens and never for NVFP4, and picks by rule elsewhere,
// so outside that domain they are the formula's figures, not the selector's choice (the CPU
// replica says so). What the CPU replica (`ninferno-serve price`) and the kernel-roofline test
// compare the roofline's terms against.
struct PricedRouteQuery {
    Route route;
    double seconds = 0;
};

[[nodiscard]] std::vector<PricedRouteQuery>
priced_routes_query(const DeviceClass& device, QType qtype, LinearPolicy policy, Epilogue epilogue,
                    std::int32_t n, std::int32_t k, std::int32_t tokens);

// Whether the selector admits `route` for such a call on `device`: route_admits on the described
// weight (every refusal of the family's rule: not compiled, format, codec tile, resource,
// instruction, alignment, extent), as describe_route decides before it describes. The one rule
// the price tool and its test read; false where the generic path does not read such a weight.
[[nodiscard]] bool route_admitted_query(const DeviceClass& device, Route route, QType qtype,
                                        LinearPolicy policy, Epilogue epilogue, std::int32_t n,
                                        std::int32_t k);

// The payload bytes of the weight such a call is described with (the converter's layout of the
// stored matrix: 2 n rows under SwiGLU, `parent_rows` where the call reads a slice): the bytes per
// value the cost model prices by are these over n x k.
[[nodiscard]] std::uint64_t described_weight_bytes(QType qtype, Epilogue epilogue, std::int32_t n,
                                                   std::int32_t k, std::int32_t parent_rows = 0);

} // namespace ninferno::ops::detail::generic
