#pragma once

#include "ops/common/math.cuh"
#include "ops/common/e2m1.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninferno::ops::detail {

__device__ __forceinline__ float2 decode_nvfp4_e2m1x2(std::uint8_t storage) {
    return __half22float2(e2m1x2_to_half2(storage));
}

__device__ __forceinline__ float decode_nvfp4_e4m3(std::uint8_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = static_cast<std::uint16_t>(storage) | (static_cast<std::uint16_t>(storage) << 8);
    return static_cast<float2>(value).x;
}

// One e4m3 block scale in the K16M128x4 layout. A 512-byte tile holds 128 rows by four groups,
// with the rows interleaved so that four rows 32 apart share one 16-byte line. `ScaleTilesPerRow`
// is the matrix's column count over 64; a caller that owns rows of a larger plane than a
// registered linear shape supplies it directly.
template <int ScaleTilesPerRow>
__device__ __forceinline__ std::int64_t nvfp4_scale_byte(int row, int group) {
    const int m_tile     = row / 128;
    const int row_inner  = row - m_tile * 128;
    const int scale_tile = group / 4;
    const int scale_lane = group & 3;
    return (static_cast<std::int64_t>(m_tile) * ScaleTilesPerRow + scale_tile) * 512 +
           (row_inner & 31) * 16 + (row_inner >> 5) * 4 + scale_lane;
}

__device__ __forceinline__ unsigned nvfp4_scaled_pair_bf16(std::uint8_t code, std::uint8_t scale) {
#if (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) &&    \
    defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000
    // PTX 9.2 exposes native pair widening on the family-specific targets from sm_100. E2M1 times
    // a finite E4M3 scale needs at most six significand bits, so the BF16 multiplication is exact.
    unsigned values, multiplier, result;
    asm("{\n"
        ".reg .b8 packed;\n"
        "mov.b32 {packed, _, _, _}, %1;\n"
        "cvt.rn.bf16x2.e2m1x2 %0, packed;\n"
        "}\n"
        : "=r"(values)
        : "r"(static_cast<unsigned>(code)));
    const auto scale_pair = static_cast<std::uint16_t>(scale | (static_cast<unsigned>(scale) << 8));
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(multiplier) : "h"(scale_pair));
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(result) : "r"(values), "r"(multiplier));
    return result;
#else
    // CUDA 13.1 and the other targets keep the exact FP32 expansion.
    const float2 values    = decode_nvfp4_e2m1x2(code);
    const float multiplier = decode_nvfp4_e4m3(scale);

    union {
        __nv_bfloat162 pair;
        unsigned bits;
    } result;

    result.pair = __floats2bfloat162_rn(values.x * multiplier, values.y * multiplier);
    return result.bits;
#endif
}

__device__ __forceinline__ int nvfp4_a16_shared_col_64(int row, int col) {
    return col ^ ((row & 7) << 3);
}

__device__ __forceinline__ std::int64_t nvfp4_scale_byte_offset(int row, int group, int k) {
    return static_cast<std::int64_t>((row / 128) * (k / 64) + group / 4) * 512 + (row & 31) * 16 +
           ((row & 127) >> 5) * 4 + (group & 3);
}

struct alignas(8) Nvfp4QuantizedK16 {
    std::uint32_t codes_lo;
    std::uint32_t codes_hi;
    std::uint8_t scale;
};

static_assert(alignof(Nvfp4QuantizedK16) == 8);

// The E2M1 encode instruction (cvt.rn.satfinite.e2m1x2.f32) exists from sm_100; on earlier
// device passes the pack falls back to the reference rounding below, the same nearest-value
// semantics the host converter implements (convert/numerics.cpp round_e2m1): the eight code
// magnitudes {0,.5,1,1.5,2,3,4,6}, nearest wins, a tie takes the even code.
// cvt with .e2m1x2 / FP8-to-half: Blackwell's architecture- and family-specific targets
// (sm_100a.., sm_120a, sm_120f: __CUDA_ARCH_FAMILY_SPECIFIC__), not a plain target's PTX.
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FAMILY_SPECIFIC__) &&                            \
    __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000
#    define NINFERNO_HAS_E2M1_CVT 1
#else
#define NINFERNO_HAS_E2M1_CVT 0
#endif

__device__ __forceinline__ std::uint8_t round_e2m1_device(float value) {
    const float magnitude          = fabsf(value);
    constexpr float kMagnitudes[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    std::uint8_t best              = 0;
    float best_distance            = magnitude;
#pragma unroll
    for (std::uint8_t i = 1; i < 8; ++i) {
        const float distance = fabsf(magnitude - kMagnitudes[i]);
        if (distance < best_distance || (distance == best_distance && (i & 1) == 0)) {
            best_distance = distance;
            best          = i;
        }
    }
    return static_cast<std::uint8_t>((value < 0.0F ? 0x8U : 0x0U) | best);
}
__device__ __forceinline__ void
pack_nvfp4_e2m1x16(const float2 (&values)[8], std::uint32_t& codes_lo, std::uint32_t& codes_hi) {
#if NINFERNO_HAS_E2M1_CVT
    asm volatile("{\n"
                 ".reg .b8 b0;\n"
                 ".reg .b8 b1;\n"
                 ".reg .b8 b2;\n"
                 ".reg .b8 b3;\n"
                 ".reg .b8 b4;\n"
                 ".reg .b8 b5;\n"
                 ".reg .b8 b6;\n"
                 ".reg .b8 b7;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16;\n"
                 "mov.b32 %0, {b0,b1,b2,b3};\n"
                 "mov.b32 %1, {b4,b5,b6,b7};\n"
                 "}\n"
                 : "=r"(codes_lo), "=r"(codes_hi)
                 : "f"(values[0].x), "f"(values[0].y), "f"(values[1].x), "f"(values[1].y),
                   "f"(values[2].x), "f"(values[2].y), "f"(values[3].x), "f"(values[3].y),
                   "f"(values[4].x), "f"(values[4].y), "f"(values[5].x), "f"(values[5].y),
                   "f"(values[6].x), "f"(values[6].y), "f"(values[7].x), "f"(values[7].y));
#else
    codes_lo = 0;
    codes_hi = 0;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        // Byte g of each word is one pair, x in the low nibble: the cvt path's layout.
        codes_lo |= static_cast<std::uint32_t>(round_e2m1_device(values[g].x)) << (8 * g);
        codes_lo |= static_cast<std::uint32_t>(round_e2m1_device(values[g].y)) << (8 * g + 4);
        codes_hi |= static_cast<std::uint32_t>(round_e2m1_device(values[g + 4].x)) << (8 * g);
        codes_hi |= static_cast<std::uint32_t>(round_e2m1_device(values[g + 4].y)) << (8 * g + 4);
    }
#endif
}

// Half a group, packed by one lane. A fused epilogue splits the sixteen values of a group across
// two neighbouring lanes so that every thread of the block takes part.
__device__ __forceinline__ std::uint32_t pack_nvfp4_e2m1x8(const float2 (&values)[4]) {
    std::uint32_t codes = 0;
#if NINFERNO_HAS_E2M1_CVT
    asm volatile("{\n"
                 ".reg .b8 b0;\n"
                 ".reg .b8 b1;\n"
                 ".reg .b8 b2;\n"
                 ".reg .b8 b3;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
                 "mov.b32 %0, {b0,b1,b2,b3};\n"
                 "}\n"
                 : "=r"(codes)
                 : "f"(values[0].x), "f"(values[0].y), "f"(values[1].x), "f"(values[1].y),
                   "f"(values[2].x), "f"(values[2].y), "f"(values[3].x), "f"(values[3].y));
#else
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        codes |= static_cast<std::uint32_t>(round_e2m1_device(values[g].x)) << (8 * g);
        codes |= static_cast<std::uint32_t>(round_e2m1_device(values[g].y)) << (8 * g + 4);
    }
#endif
    return codes;
}

// Sixteen bf16 values already in registers, for a caller that holds them there rather than writing
// them out for a separate pass to read back. Note that this divides where the sparse-MoE fused
// epilogue multiplies by a reciprocal, so the two can differ by one code at an exact tie.
// Nothing in the engine compares the two planes: each route is checked end to end against an
// oracle that quantises neither.
__device__ __forceinline__ Nvfp4QuantizedK16
quantize_nvfp4_k16_bits(const std::uint32_t (&represented)[8], float input_scale_divisor) {
    float2 values[8];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = bf16x2_bits_to_float2(represented[pair]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }

    Nvfp4QuantizedK16 result{};
    const float scale_unencoded = __fdiv_rn(input_scale_divisor * max_abs, 6.0F);
    result.scale                = __nv_cvt_float_to_fp8(scale_unencoded, __NV_SATFINITE, __NV_E4M3);
    if (result.scale == 0) { return result; }

    const float decoded_scale = decode_nvfp4_e4m3(result.scale);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair].x = __fdiv_rn(values[pair].x * input_scale_divisor, decoded_scale);
        values[pair].y = __fdiv_rn(values[pair].y * input_scale_divisor, decoded_scale);
    }
    pack_nvfp4_e2m1x16(values, result.codes_lo, result.codes_hi);
    return result;
}

__device__ __forceinline__ Nvfp4QuantizedK16 quantize_nvfp4_k16(const __nv_bfloat16* source,
                                                                float input_scale_divisor) {
    const uint4 packed0                = load_vec<uint4>(source);
    const uint4 packed1                = load_vec<uint4>(source + 8);
    const std::uint32_t represented[8] = {
        packed0.x, packed0.y, packed0.z, packed0.w, packed1.x, packed1.y, packed1.z, packed1.w,
    };
    return quantize_nvfp4_k16_bits(represented, input_scale_divisor);
}

} // namespace ninferno::ops::detail
