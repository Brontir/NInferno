#pragma once

// The FP8 row codec of Gemma4KvLayer (include/ninferno/ops/gemma4_attention.h): one row (one
// position of one KV head, D values) is E4M3FN codes times one FP32 scale, scale =
// absmax(row) / 448 (0 for a zero row) and code = E4M3(x / scale), saturating. Every writer of a
// Gemma 4 / dense KV row (gemma4_kv_append, gemma4_qkv_prepare, dense_qkv_prepare) and the
// attention kernels' widening go through these helpers, so the stored and the read values agree.

#include "ops/common/fp8_e4m3.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninferno::ops::detail {

inline constexpr float kGemma4Fp8Max = 448.0f;

// The row scale and its reciprocal from the row's absolute maximum.
__device__ __forceinline__ void gemma4_fp8_row_scale(float absmax, float& scale, float& inverse) {
    scale   = absmax > 0.0f ? absmax / kGemma4Fp8Max : 0.0f;
    inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
}

__device__ __forceinline__ unsigned gemma4_fp8x2(float a, float b, float inverse) {
    return __nv_cvt_float2_to_fp8x2(make_float2(a * inverse, b * inverse), __NV_SATFINITE,
                                    __NV_E4M3);
}

// E contiguous values of a row held by one lane (E = D / 32) as E codes at dst.
template <int E>
__device__ __forceinline__ void gemma4_fp8_store(std::uint8_t* dst, const float (&x)[E],
                                                 float inverse) {
    static_assert(E == 2 || E == 4 || E == 8 || E == 16);
    unsigned w[(E + 3) / 4];
#pragma unroll
    for (int i = 0; i < (E + 3) / 4; ++i) { w[i] = 0; }
#pragma unroll
    for (int j = 0; j < E; j += 2) {
        w[j / 4] |= gemma4_fp8x2(x[j], x[j + 1], inverse) << (8 * (j % 4));
    }
    if constexpr (E == 16) {
        *reinterpret_cast<uint4*>(dst) = make_uint4(w[0], w[1], w[2], w[3]);
    } else if constexpr (E == 8) {
        *reinterpret_cast<uint2*>(dst) = make_uint2(w[0], w[1]);
    } else if constexpr (E == 4) {
        *reinterpret_cast<unsigned*>(dst) = w[0];
    } else {
        *reinterpret_cast<unsigned short*>(dst) = static_cast<unsigned short>(w[0]);
    }
}

// Quantize one row held by a full warp (E values per lane) into codes at row_codes and its scale
// at *row_scale (written by lane 0).
template <int E>
__device__ __forceinline__ void gemma4_fp8_store_row(std::uint8_t* row_codes, float* row_scale,
                                                     const float (&x)[E], int lane) {
    float absmax = 0.0f;
#pragma unroll
    for (int j = 0; j < E; ++j) { absmax = fmaxf(absmax, fabsf(x[j])); }
    absmax = warp_max(absmax);
    float scale, inverse;
    gemma4_fp8_row_scale(absmax, scale, inverse);
    gemma4_fp8_store<E>(row_codes + lane * E, x, inverse);
    if (lane == 0) { *row_scale = scale; }
}

// Four codes (one 32-bit word) times the row scale, as two BF16 pairs.
__device__ __forceinline__ uint2 gemma4_fp8x4_to_bf16(unsigned w, float scale) {
    const __half2 lo = e4m3x2_to_half2(w & 0xffffu);
    const __half2 hi = e4m3x2_to_half2(w >> 16);
    const float2 a = __half22float2(lo);
    const float2 b = __half22float2(hi);
    const __nv_bfloat162 p = __floats2bfloat162_rn(a.x * scale, a.y * scale);
    const __nv_bfloat162 q = __floats2bfloat162_rn(b.x * scale, b.y * scale);
    return make_uint2(*reinterpret_cast<const unsigned*>(&p), *reinterpret_cast<const unsigned*>(&q));
}

// Four E4M3 codes of two key pairs, (lo.x, lo.y) for the low two and (hi.x, hi.y) for the high two,
// widened as gemma4_fp8x4_to_bf16 widens a row (scale * code, rounded to BF16) with each key's
// scale.
__device__ __forceinline__ uint2 gemma4_fp8x4_to_bf16_keys(unsigned w, float2 lo, float2 hi) {
    const float2 a = __half22float2(e4m3x2_to_half2(w & 0xffffu));
    const float2 b = __half22float2(e4m3x2_to_half2(w >> 16));
    const __nv_bfloat162 p = __floats2bfloat162_rn(a.x * lo.x, a.y * lo.y);
    const __nv_bfloat162 q = __floats2bfloat162_rn(b.x * hi.x, b.y * hi.y);
    return make_uint2(*reinterpret_cast<const unsigned*>(&p),
                      *reinterpret_cast<const unsigned*>(&q));
}

} // namespace ninferno::ops::detail
