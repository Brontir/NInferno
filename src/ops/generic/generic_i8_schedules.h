#pragma once

// The I8 tiles' table and efficiencies (generic_i8.cu), lifted out of generic_gemm.cpp so the host
// part of the selector (generic_routes.cpp) and the picker that reads them share one copy (#620).

#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_cost_constants.h"

#include <cstddef>
#include <iterator>

namespace ninferno::ops::detail::generic {

// The I8 tiles (generic_i8.cu): 256x128 (8 warps), 128x128 and 128x64 (4 and 2 warps), four
// stages each. All give the same bytes.
inline constexpr I8ScheduleInfo kI8Schedules[] = {
    {256, 128, 256, 4, 4 * (256 * 64 + 128 * 32 + 256 * 8) + 2 * 128 * 16},
    {128, 128, 128, 4, 4 * (128 * 64 + 128 * 32 + 128 * 8) + 2 * 128 * 16},
    {128, 64, 64, 4, 4 * (128 * 64 + 64 * 32 + 128 * 8) + 2 * 64 * 16},
};

// Each I8 tile's fraction of the int8 peak at a full grid (stage-1 probe, RTX 3090, 4090 and
// 5070 Ti, 27B shapes): 256x128 0.53, 128x128 0.46, 128x64 0.42.
inline constexpr auto kI8Eff = route_cost::kI8Efficiency;
static_assert(kI8Eff.values.size() == std::size(kI8Schedules));

} // namespace ninferno::ops::detail::generic
