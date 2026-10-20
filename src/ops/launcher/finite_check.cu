// Implements: include/ninferno/ops/finite_check.h
// Match: validated contiguous BF16, FP16 or FP32 values.
// Algorithm assumptions: grid (C,R); the exponent mask selects the format.
#include "ops/launcher/finite_check.h"

#include "ops/kernel/finite_check.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>

namespace ninferno::ops::detail {

void finite_check_launch(const Tensor& values, std::int32_t hidden, std::int32_t columns,
                         std::int32_t rows, const std::int32_t* valid_columns,
                         std::uint32_t* status, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned int>(columns), static_cast<unsigned int>(rows));
    if (values.dtype == DType::FP32) {
        finite_check_kernel<std::uint32_t><<<grid, kFiniteCheckBlock, 0, stream>>>(
            static_cast<const std::uint32_t*>(values.data), hidden, columns, 0x7F800000U,
            valid_columns, status);
    } else {
        const auto mask =
            static_cast<std::uint16_t>(values.dtype == DType::BF16 ? 0x7F80U : 0x7C00U);
        finite_check_kernel<std::uint16_t>
            <<<grid, kFiniteCheckBlock, 0, stream>>>(static_cast<const std::uint16_t*>(values.data),
                                                     hidden, columns, mask, valid_columns, status);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninferno::ops::detail
