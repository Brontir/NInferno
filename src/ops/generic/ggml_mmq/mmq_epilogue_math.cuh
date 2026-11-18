#pragma once

// The ggml matrix route's epilogue arithmetic (#617 I), one definition for the separate epilogue
// kernel (generic_mmq_epilogue.cu) and the tile kernel's own store (ggml_mmq_internal.cuh's
// NINFERNO_MMQ_STORE): the same operations in the same order, so a fused call writes the bytes the
// separate kernel would.
#include <cuda_bf16.h>

#include <cstddef>

namespace ninferno::ops::mmq_epilogue {

// GELU's tanh form through the logistic, as the generic route computes it (generic_gelu_tanh): the
// sum 0.5 * (1 + tanh(u)) cancels to 0 for a gate below about -5.
__device__ __forceinline__ float gelu_tanh(float g) {
    constexpr float kTwoSqrt2Pi = 1.59576912160573071176F; // 2 * sqrt(2 / pi)
    return g / (1.0F + expf(-kTwoSqrt2Pi * (g + 0.044715F * g * g * g)));
}

// The value an output element takes before the residual add and the BF16 rounding.
__device__ __forceinline__ float value(bool gated, bool gelu, bool relu2, float v, float up) {
    if (gated) return (gelu ? gelu_tanh(v) : v / (1.0F + __expf(-v))) * up;
    if (relu2) {
        v = fmaxf(v, 0.0F);
        return v * v;
    }
    return v;
}

// The tile kernel's store of one sum: element (j, i) of the tile whose FP32 slot is dst[j * stride
// + i]. Without an epilogue (kind 0) the slot is written as is; else the element (row, token) of
// the BF16 output, the gated product reading the up half the up product left in the plane. The
// tile's origin in the plane is the one division here, the same for every element of the tile (the
// compiler hoists it out of the write-back's loops); a division per element made the tile kernel
// 21 % slower on the RTX 3060 (#617 I A/B).
template <class Planes>
__device__ __forceinline__ void store(const Planes& p, float* dst, int j, int i, int stride,
                                      float v) {
    if (p.epi_kind == 0) {
        dst[j * stride + i] = v;
        return;
    }
    const int at   = static_cast<int>(dst - p.epi_plane);
    const int col0 = at / p.epi_stride;
    const int col  = col0 + j;
    const int row  = at - col0 * p.epi_stride + i;
    auto* out =
        static_cast<__nv_bfloat16*>(p.epi_out) + static_cast<std::size_t>(col) * p.epi_ld + row;
    const bool gated = p.epi_kind == 3;
    const float up =
        gated ? p.epi_plane[static_cast<std::size_t>(col) * p.epi_stride + p.epi_up + row] : 0.0F;
    float r = value(gated, p.epi_gelu != 0, p.epi_relu2 != 0, v, up);
    if (p.epi_kind == 2) r += __bfloat162float(*out);
    *out = __float2bfloat16(r);
}

} // namespace ninferno::ops::mmq_epilogue
