#pragma once

// ninferno::ops - the activation class of a routed projection, shared by every consumer of the
// per-GPU-class route table (include/ninferno/ops/route_table.h): the generic projection path's
// bands (ops/generic/route_table.h) and the shape-owned tables' route_as overrides
// (ops/common/shape_routes.h). It is the "compute" key of a table entry.

#include <cstdint>

namespace ninferno::ops::detail {

// a16: BF16 activations; a4: NVFP4 W4A4 (NVFP4 weights only); a8: FP8 W8A8 (FP8 weights only).
enum class RouteCompute : std::uint8_t { A16, A4, A8 };

[[nodiscard]] constexpr const char* route_compute_name(RouteCompute c) noexcept {
    return c == RouteCompute::A4 ? "a4" : c == RouteCompute::A8 ? "a8" : "a16";
}

} // namespace ninferno::ops::detail
