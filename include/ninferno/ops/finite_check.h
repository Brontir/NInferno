#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninferno::ops {

/**
 * Flags rows holding a NaN or an infinity:
 *
 *   status[r] = 1 if some values[h,c,r] with c < valid_columns[r] is not finite,
 *
 * and status[r] is left unchanged otherwise. `values` is contiguous BF16, FP16 or FP32
 * [H,C,R] (ne[0..2]; view an [H,R] tensor as [H,1,R]). `valid_columns` is null (every
 * column counts) or device I32 [R]. `status` is R device-visible words, typically a mapped
 * host alias; the caller zeroes them. The Op only reads `values` and writes the words it flags.
 */
void finite_check(const Tensor& values, const std::int32_t* valid_columns, std::uint32_t* status,
                  cudaStream_t stream);

} // namespace ninferno::ops
