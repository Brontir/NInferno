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

// The selector's cost model as a pure query (price_routes, generic_cost.cpp): every route it
// offers for such a call on `device`, with its time in seconds (the A16 scale applied), in the
// order it offers them. Empty where the facts have no roofline, the generic path does not read
// such a weight, or the width is not priced. What the CPU replica (`ninferno-serve price`) and the
// kernel-roofline test compare the roofline's terms against.
struct PricedRouteQuery {
    Route route;
    double seconds = 0;
};

[[nodiscard]] std::vector<PricedRouteQuery>
priced_routes_query(const DeviceClass& device, QType qtype, LinearPolicy policy, Epilogue epilogue,
                    std::int32_t n, std::int32_t k, std::int32_t tokens);

// The payload bytes of the weight such a call is described with (the converter's layout of the
// stored matrix: 2 n rows under SwiGLU, `parent_rows` where the call reads a slice): the bytes per
// value the cost model prices by are these over n x k.
[[nodiscard]] std::uint64_t described_weight_bytes(QType qtype, Epilogue epilogue, std::int32_t n,
                                                   std::int32_t k, std::int32_t parent_rows = 0);

} // namespace ninferno::ops::detail::generic
