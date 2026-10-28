#pragma once

// The generic selector as a pure query: the route a call of the projection path takes on a board
// given by its class (core/device_class.h) rather than the current device, and what that route
// computes, as its kernels issue it. The answer is select_route's own (the same rules, tables and
// build), so a cost model outside the kernels reads the selector instead of restating it
// (spec/round_cost.h prices a speculation round's verify by it).
//
// Nothing is read but `device`, the build and the route tables on disk: the query substitutes the
// class for the call (ScopedDeviceClass), and the caches the selector keeps per device keep a
// substitute's answers apart by its content (core/per_device.h, ops/route_table.cpp). A weight is
// described by its format and shape alone, in the layout the converter stores it in
// (default_weight_layout); its bytes are never read.

#include "core/device_class.h"
#include "core/weight.h"
#include "ops/generic/generic_gemm.h"

#include <cstdint>
#include <optional>
#include <string>

namespace ninferno::ops::detail::generic {

// The arithmetic a route's kernels issue: the dp4a vector path (every token its own products),
// the BF16 MMA (A16: the weight dequantized to BF16), the int8 MMA (A8: int8 activations), the
// block-scaled FP4 MMA (A4), or scalar FMAs.
enum class RouteArithmetic : std::uint8_t { Dp4a, Bf16Mma, Int8Mma, Fp4Mma, Fma };

[[nodiscard]] const char* route_arithmetic_name(RouteArithmetic a);

struct RouteShape {
    Route route;
    RouteArithmetic arithmetic = RouteArithmetic::Fma;
    // The token columns one pass of the route's tile covers: a call of T tokens computes
    // ceil(T / token_tile) x token_tile columns (1: every token on its own).
    std::int32_t token_tile = 1;
};

// The route a call of `tokens` tokens of an `n` x `k` projection of `qtype` under `policy` and
// `epilogue` takes on `device` (SwiGLU: `n` output rows, the weight 2 n rows), and its shape;
// nullopt where the generic path does not read such a weight (generic::supports).
[[nodiscard]] std::optional<RouteShape> route_shape(const DeviceClass& device, QType qtype,
                                                    LinearPolicy policy, Epilogue epilogue,
                                                    std::int32_t n, std::int32_t k,
                                                    std::int32_t tokens, bool prefill = false);

// What a route computes (route_shape's arithmetic and token tile) from its family and schedule.
[[nodiscard]] RouteShape shape_of_route(Route route);

} // namespace ninferno::ops::detail::generic
