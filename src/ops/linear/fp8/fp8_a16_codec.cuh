#pragma once

// Exact E4M3-to-BF16 operand widening shared by the small-T and GEMM A16 routes.

#include "ops/common/fp8_e4m3.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninferno::ops::detail {

// Places the sign, exponent and mantissa of both E4M3 codes in BF16 fields: code bits 0..6 land
// on BF16 bits 4..10 and the sign on bit 15, so the BF16 value is the code's value times 2^-120,
// subnormal codes included (they fall on BF16 subnormals). E4M3FN NaN (0x7f) has no BF16 image
// here; converted weights never hold it.
__device__ __forceinline__ unsigned fp8_e4m3x2_bf16_fields(unsigned packed) {
    const unsigned spread = __byte_perm(packed, 0U, 0x1404U);
    return ((spread >> 4) & 0x07f007f0U) | (spread & 0x80008000U);
}

__device__ __forceinline__ unsigned fp8_e4m3x2_to_bf16x2_bits(unsigned packed) {
#if (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) &&    \
    defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000
    // PTX 9.2 exposes the native pair widening, on the architecture- and family-specific targets
    // from sm_100 (ptxas refuses it for sm_89 and a plain target's PTX).
    unsigned result;
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(result) : "h"(static_cast<std::uint16_t>(packed)));
    return result;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // No pair widening before sm_100 (sm_89 has it only to FP16, Ampere not at all): the fields
    // times 2^120 in one BF16 multiply, exact because every product is a normal BF16 value.
    const unsigned fields        = fp8_e4m3x2_bf16_fields(packed);
    const __nv_bfloat162 product = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&fields),
                                           __bfloat162bfloat162(__ushort_as_bfloat16(0x7b80U)));
    return *reinterpret_cast<const unsigned*>(&product);
#else
    // Before sm_80 (no BF16 arithmetic) the exact FP16 bit form: cuda_fp8.h widens in software
    // there.
    const __half2 half_pair = e4m3x2_to_half2_bits(packed);

    union {
        __nv_bfloat162 pair;
        unsigned bits;
    } result;

    // Through FP32 explicitly: every E4M3 value is exact in FP16, FP32 and BF16.
    result.pair = __floats2bfloat162_rn(__low2float(half_pair), __high2float(half_pair));
    return result.bits;
#endif
}

// The pair as FP32: the exact BF16 pair is the upper half of each float.
__device__ __forceinline__ float2 fp8_e4m3x2_to_float2(unsigned packed) {
    const unsigned bits = fp8_e4m3x2_to_bf16x2_bits(packed);
    return make_float2(__uint_as_float(bits << 16), __uint_as_float(bits & 0xffff0000U));
}

__device__ __forceinline__ int fp8_a16_shared_col_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

} // namespace ninferno::ops::detail
