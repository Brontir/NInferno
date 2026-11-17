#pragma once

// The row codec of Gemma4KvLayer (include/ninferno/ops/gemma4_attention.h): one row (one position
// of one KV head, D values) is one byte a value times one FP32 scale. Every writer of a Gemma 4 /
// dense KV row (gemma4_kv_append, gemma4_qkv_prepare, dense_qkv_prepare) and the attention
// kernels' widening go through these helpers, so the stored and the read values agree.
//   - FP8 (sm_89 and newer): E4M3FN codes, scale = absmax(row) / 448 (0 for a zero row), code =
//     E4M3(x / scale), saturating; the conversion is in hardware.
//   - INT8 (sm_75 to sm_86, kGemma4RowInt8, #619 steps 9-10): cuda_fp8.h widens E4M3 in software
//     there, so the same layout holds int8 codes instead: scale = absmax(row) / 127, code =
//     round(x / scale) (|code| <= 127), read back as code x scale (one I2F and one FMUL a value).
// One tree builds one architecture (NINFERNO_BUILD_ARCH), so the writer and the reader of a build
// use one codec, and the host allocates its planes in the codec's dtype (gemma4_row_codec_dtype).

#include "ops/common/fp8_e4m3.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/gemma4/row_codec.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninferno::ops::detail {

using ops::kGemma4RowInt8; // row_codec.h: the build's codec (Volta keeps E4M3)

inline constexpr float kGemma4RowMax = ops::kGemma4RowCodeMax;

// The row scale and its reciprocal from the row's absolute maximum.
__device__ __forceinline__ void gemma4_fp8_row_scale(float absmax, float& scale, float& inverse) {
    scale   = absmax > 0.0f ? absmax / kGemma4RowMax : 0.0f;
    inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
}

// One int8 code of x / scale (x * inverse), rounded to nearest and held to +-127.
__device__ __forceinline__ unsigned gemma4_int8_code(float x, float inverse) {
    const int q = max(-127, min(127, __float2int_rn(x * inverse)));
    return static_cast<unsigned>(q) & 0xffu;
}

// Two codes (the build's codec) of a and b, in the low 16 bits.
__device__ __forceinline__ unsigned gemma4_fp8x2(float a, float b, float inverse) {
    if constexpr (kGemma4RowInt8) {
        return gemma4_int8_code(a, inverse) | (gemma4_int8_code(b, inverse) << 8);
    } else {
        return __nv_cvt_float2_to_fp8x2(make_float2(a * inverse, b * inverse), __NV_SATFINITE,
                                        __NV_E4M3);
    }
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

// The four codes of one 32-bit word as floats, unscaled (the build's codec).
__device__ __forceinline__ void gemma4_codes4(unsigned w, float2& a, float2& b) {
    if constexpr (kGemma4RowInt8) {
        a = make_float2(static_cast<float>(static_cast<signed char>(w & 0xffu)),
                        static_cast<float>(static_cast<signed char>((w >> 8) & 0xffu)));
        b = make_float2(static_cast<float>(static_cast<signed char>((w >> 16) & 0xffu)),
                        static_cast<float>(static_cast<signed char>(w >> 24)));
    } else {
        a = __half22float2(e4m3x2_to_half2(w & 0xffffu));
        b = __half22float2(e4m3x2_to_half2(w >> 16));
    }
}

// Four codes (one 32-bit word) times the row scale, as two BF16 pairs.
__device__ __forceinline__ uint2 gemma4_fp8x4_to_bf16(unsigned w, float scale) {
    float2 a, b;
    gemma4_codes4(w, a, b);
    const __nv_bfloat162 p = __floats2bfloat162_rn(a.x * scale, a.y * scale);
    const __nv_bfloat162 q = __floats2bfloat162_rn(b.x * scale, b.y * scale);
    return make_uint2(*reinterpret_cast<const unsigned*>(&p), *reinterpret_cast<const unsigned*>(&q));
}

// Four codes of two key pairs, (lo.x, lo.y) for the low two and (hi.x, hi.y) for the high two,
// widened as gemma4_fp8x4_to_bf16 widens a row (scale * code, rounded to BF16) with each key's
// scale.
__device__ __forceinline__ uint2 gemma4_fp8x4_to_bf16_keys(unsigned w, float2 lo, float2 hi) {
    float2 a, b;
    gemma4_codes4(w, a, b);
    const __nv_bfloat162 p = __floats2bfloat162_rn(a.x * lo.x, a.y * lo.y);
    const __nv_bfloat162 q = __floats2bfloat162_rn(b.x * hi.x, b.y * hi.y);
    return make_uint2(*reinterpret_cast<const unsigned*>(&p),
                      *reinterpret_cast<const unsigned*>(&q));
}

} // namespace ninferno::ops::detail
