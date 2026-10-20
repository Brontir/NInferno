#pragma once
#include "core/device.h"
#include "core/device_class.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninferno::ops::detail {

// NINFERNO-SPECIAL(arch:sm70): Tesla V100 takes the row-split split4 kernel (one row a CTA, four
// warps along K, upstream ninfer's kernel as ninfer-all kept it; ninfer-v100-sm70's route) for the
// K = 5120 input projections at T = 2..10, where the unified direct SIMT tile runs 25 % slower at
// the MTP4 verify width: V100 27B, T = 5, GDN value+z (12288 rows) 134.8 against 169.0 us,
// attention (7168 rows) 68.6 against 85.9.
inline bool q5_rowsplit_split4_without_tensor_mma() noexcept {
    try {
        return !device_class().features.mma_tensor; // the current device's (Volta: none)
    } catch (...) { return false; }
}

// The split-output projection [first; second] = W x through the row-split split4 kernel at exactly
// Cols columns, K = Hidden. False (nothing launched) unless the layout is the one the kernel
// addresses: a contiguous activation, an unpadded K and a tail output packed at its row count.
template <int Cols, int Hidden, int SplitRow>
bool launch_q5_rowsplit_split4_split_output(const Tensor& x, const Weight& weight, Tensor& first,
                                            Tensor& second, cudaStream_t stream) {
    static_assert(Hidden % 1024 == 0, "split4 streams 1024-value slabs");
    const std::int32_t tail_rows = static_cast<std::int32_t>(second.ne[0]);
    if (x.ne[1] != Cols || x.ne[0] != Hidden ||
        x.nb[1] != static_cast<std::size_t>(Hidden) * sizeof(__nv_bfloat16) ||
        weight.padded_shape[1] != Hidden || first.ne[0] != SplitRow ||
        second.nb[1] != static_cast<std::size_t>(tail_rows) * sizeof(__nv_bfloat16)) {
        return false;
    }
    const std::int32_t rows = SplitRow + tail_rows;
    q5_rowsplit_gemm_simt_split4_kernel<Cols, Hidden / 1024, Hidden, true, SplitRow>
        <<<dim3(static_cast<unsigned>(rows), 1u, 1u), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(first.data), static_cast<__nv_bfloat16*>(second.data), rows,
            static_cast<std::int32_t>(first.nb[1] / sizeof(__nv_bfloat16)),
            static_cast<std::int32_t>(weight.padded_shape[1]));
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// The projection W x of `rows` rows through the row-split split4 kernel at exactly Cols columns,
// K = Hidden, each row's Cols FP32 sums handed to `epilogue` (Q5Split4StoreEpilogue's call) by one
// thread: for a consumer that needs a whole row unrounded, as the GDN convolution does. False
// (nothing launched) unless the activation is contiguous and K unpadded.
template <int Cols, int Hidden, class Epilogue>
bool launch_q5_rowsplit_split4_epilogue(const Tensor& x, const Weight& weight, std::int32_t rows,
                                        const Epilogue& epilogue, cudaStream_t stream) {
    static_assert(Hidden % 1024 == 0, "split4 streams 1024-value slabs");
    if (x.ne[1] != Cols || x.ne[0] != Hidden ||
        x.nb[1] != static_cast<std::size_t>(Hidden) * sizeof(__nv_bfloat16) ||
        weight.padded_shape[1] != Hidden || rows <= 0) {
        return false;
    }
    q5_rowsplit_gemm_simt_split4_kernel<Cols, Hidden / 1024, Hidden, false, 0, Epilogue>
        <<<dim3(static_cast<unsigned>(rows), 1u, 1u), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales), nullptr, nullptr, rows, 0,
            static_cast<std::int32_t>(weight.padded_shape[1]), epilogue);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

} // namespace ninferno::ops::detail
