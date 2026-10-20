#pragma once

// ninferno::ops - per-GPU-class width overrides for the shape-owned Linear tables ("route_as").
//
// The shape tables (src/ops/linear/<format>/shapes/*) were tuned on one GPU class: each picks a
// compiled route per token count T. On another chip the crossovers between those routes move (a
// card with half the SMs wants the wider tile of the next band earlier), but the routes
// themselves stay right. A route_as band says "for a call of t_min..t_max tokens, take the route
// this shape's table takes at T'": no new kernels and no C++ per card, only data.
//
// Data. The bands live in the GPU class's route table (include/ninferno/ops/route_table.h,
// routes/<class key>.json), read by the one loader that also serves the generic path's "bands":
//
//   {"format": "nvfp4", "compute": "a4", "epilogue": "store", "n": 2816, "k": 4096,
//    "route_as": [[256, 384, 512], [512, 768, 1024]]}
//
// "compute" names the table of the shape that is asked: "a16" (BF16 activations), "a4" (NVFP4
// W4A4) or "a8" (FP8 W8A8). Which of them runs for a call is still the shape's own decision (its
// uses_a4 / uses_a8 threshold): that is a numerics choice, never made by this file. Only the
// pure Linear tables ("epilogue": "store") read route_as. The FP8 shapes' A8 side runs one
// schedule for every T today, so an "a8" band is accepted and keyed but steers nothing yet.
//
// Numerics contract. A route_as override must compute the same bits as the table's own choice at
// the call's T; it only changes the schedule. ninferno_linear_route_sweep_bench --random-weight
// compares every candidate's output with the dispatcher's, and tools/route_as_table.py emits only
// candidates whose output is identical at every swept T of the band. The loader refuses a band
// whose T' is below its own t_max, so a call is never asked about fewer tokens than it has and a
// prefill width (T >= 17) is only re-asked at another prefill width of the same table: the chunk
// invariance of the prefill route profile holds with the table loaded.
//
// A live route probe (route_probe.h) measures the shape table's own routes: the class overrides
// stand aside while one is active, so the sweep bench can measure a card with its table loaded.

#include "core/weight.h"
#include "ops/common/route_compute.h"
#include "ops/common/route_probe.h"

#include <cstdint>
#include <iterator>
#include <type_traits>

namespace ninferno::ops::detail {

using ShapeCompute = RouteCompute;

struct ShapeRouteKey {
    QType qtype;
    ShapeCompute compute;
    std::int32_t n;
    std::int32_t k;
};

// The width `key`'s table is asked about for a call of `tokens` tokens: the loaded class table's
// route_as band holding `tokens` (never below `tokens`), then the route probe.
[[nodiscard]] std::int32_t route_tokens(const ShapeRouteKey& key, std::int32_t tokens);

// The entry of a shape table (entries, or pointers to them, with n and k) registered for n x k;
// nullptr when the table has none.
template <class Table>
[[nodiscard]] constexpr auto find_shape(const Table& table, std::int32_t n,
                                        std::int32_t k) noexcept {
    using Element = std::remove_cvref_t<decltype(*std::begin(table))>;
    if constexpr (std::is_pointer_v<Element>) {
        for (const Element shape : table) {
            if (shape->n == n && shape->k == k) { return shape; }
        }
        return Element{nullptr};
    } else {
        for (const Element& shape : table) {
            if (shape.n == n && shape.k == k) { return &shape; }
        }
        return static_cast<const Element*>(nullptr);
    }
}

} // namespace ninferno::ops::detail
