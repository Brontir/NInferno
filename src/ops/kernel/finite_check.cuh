#pragma once

// Implements: include/ninferno/ops/finite_check.h
// Match: contiguous 16-bit (BF16/FP16) or 32-bit (FP32) values [H,C,R].
// Algorithm assumptions: one CTA scans one column of one row; a non-finite value has an
// all-ones exponent, so the test is a mask compare on the raw bits for every float format.

#include <cstdint>

namespace ninferno::ops {

inline constexpr int kFiniteCheckBlock = 256;

template <typename Bits>
__global__ void finite_check_kernel(const Bits* __restrict__ values, std::int32_t hidden,
                                    std::int32_t columns, Bits exponent_mask,
                                    const std::int32_t* __restrict__ valid_columns,
                                    std::uint32_t* status) {
    const int column = static_cast<int>(blockIdx.x);
    const int row    = static_cast<int>(blockIdx.y);
    if (valid_columns != nullptr && column >= valid_columns[row]) { return; }
    const Bits* base = values + (static_cast<std::int64_t>(row) * columns + column) * hidden;
    int bad          = 0;
    for (int h = static_cast<int>(threadIdx.x); h < hidden; h += blockDim.x) {
        bad |= (base[h] & exponent_mask) == exponent_mask;
    }
    // A plain store: every writer stores the same 1, so no atomic crosses PCIe.
    if (__syncthreads_or(bad) && threadIdx.x == 0) { status[row] = 1U; }
}

} // namespace ninferno::ops
