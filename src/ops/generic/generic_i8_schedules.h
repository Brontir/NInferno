#pragma once

// The I8 tiles' table and efficiencies (generic_i8.cu), lifted out of generic_gemm.cpp so the host
// part of the selector (generic_routes.cpp) and the picker that reads them share one copy (#620).

#include "ops/common/arch_limits.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_cost_constants.h"

#include <cstddef>
#include <iterator>

namespace ninferno::ops::detail::generic {

// The I8 tiles (generic_i8.cu): 256x128 (8 warps), 128x128 and 128x64 (4 and 2 warps). `stages`
// and `smem_bytes` are the 4-bit tile's; a Q5 / Q6 tile adds its high plane to every stage and may
// run fewer stages: i8_stages and i8_smem_bytes below are what a format's tile compiles with.
inline constexpr I8ScheduleInfo kI8Schedules[] = {
    {256, 128, 256, 4, 4 * (256 * 64 + 128 * 32 + 256 * 8) + 2 * 128 * 16},
    {128, 128, 128, 4, 4 * (128 * 64 + 128 * 32 + 128 * 8) + 2 * 128 * 16},
    {128, 64, 64, 4, 4 * (128 * 64 + 64 * 32 + 128 * 8) + 2 * 64 * 16},
};

// The shared memory of an I8 tile of `tokens` x `rows` with `bits`-bit codes (generic_i8.cu
// I8Tile, which static_asserts it equals this): per stage the int8 activation (64 bytes a token),
// the low nibbles (32 a row), the Q5 / Q6 high plane (8 / 16 a row), the activation scale and sum
// (4 + 4 a token); and the ring of fp16 weight scales (2 chunks of 8 groups a row). One copy for
// the kernel, the price and the fit check (#761).
[[nodiscard]] constexpr int i8_high_bytes(int bits) noexcept {
    return bits == 4 ? 0 : bits == 5 ? 8 : 16;
}

[[nodiscard]] constexpr std::size_t i8_stage_bytes(int tokens, int rows, int bits) noexcept {
    return static_cast<std::size_t>(tokens * 64 + rows * 32 + rows * i8_high_bytes(bits) +
                                    tokens * 4 + tokens * 4);
}

[[nodiscard]] constexpr std::size_t i8_ring_bytes(int rows) noexcept {
    return static_cast<std::size_t>(2 * rows * 8 * 2);
}

[[nodiscard]] constexpr std::size_t i8_smem_bytes(int tokens, int rows, int bits,
                                                  int stages) noexcept {
    return static_cast<std::size_t>(stages) * i8_stage_bytes(tokens, rows, bits) +
           i8_ring_bytes(rows);
}

// The stages a tile compiles with: four, or three where four do not fit one block's shared memory
// (Q6's high plane at 256x128: 100 KiB against 99).
[[nodiscard]] constexpr int i8_stages(int tokens, int rows, int bits) noexcept {
    return i8_smem_bytes(tokens, rows, bits, 4) <= kMaxSharedBytesPerBlock ? 4 : 3;
}

// The 4-bit rows of the table are these formulas' (the table stays the 4-bit tile's for the
// readers that name a schedule without a format: route names, tiles, the picker's waves).
static_assert(kI8Schedules[0].smem_bytes == i8_smem_bytes(256, 128, 4, 4));
static_assert(kI8Schedules[1].smem_bytes == i8_smem_bytes(128, 128, 4, 4));
static_assert(kI8Schedules[2].smem_bytes == i8_smem_bytes(128, 64, 4, 4));

// Each I8 tile's fraction of the int8 peak at a full grid (stage-1 probe, RTX 3090, 4090 and
// 5070 Ti, 27B shapes): 256x128 0.53, 128x128 0.46, 128x64 0.42.
inline constexpr auto kI8Eff = route_cost::kI8Efficiency;
static_assert(kI8Eff.values.size() == std::size(kI8Schedules));

} // namespace ninferno::ops::detail::generic
