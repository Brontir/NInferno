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

#include <cstdint>
#include <optional>
#include <string>

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
// run_route launches), the tile and traffic of route_tile and route_traffic (declared geometry
// and a prediction of requested bytes, not counters), `prepared` of prepared_rule, the price of
// route_cost_seconds, and the kernel of route_resources (the compiled registers, shared memory,
// threads and grid, read from the HOST GPU's binary: only when asked for). Not device-free
// otherwise either: at 17 tokens or more a gated register-MMA price (select_route,
// route_cost_seconds) reads the host GPU's compiled rmma facts, whatever `device` substitutes.
// The description is of the call route_query describes: the engine's contiguous, aligned
// activation, the weight in the converter's layout (see above), not a recorded call's own layout.
struct RouteDescription {
    ResolvedRoute resolved;
    std::size_t workspace_bytes = 0; // what `resolved.executed` takes from the call's workspace
    std::optional<RouteTile> tile;
    std::optional<RouteTraffic> traffic;
    std::optional<PreparedRule> prepared;
    std::optional<double> cost_seconds; // cost_route's time where it prices the route
    std::optional<KernelResources> kernel;
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
// threads=<t> grid=<x>x<y>x<z>]".
[[nodiscard]] std::string describe_text(const RouteDescription& description);

} // namespace ninferno::ops::detail::generic
