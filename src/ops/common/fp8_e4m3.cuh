#pragma once

// E4M3 codes widened to FP16 for the decoders that read FP8 rows (KV caches, weights). From
// sm_89 the hardware converts a code pair in one cvt; below it cuda_fp8.h widens each code in
// software, with a normalizing loop for subnormals. There the code's exponent and mantissa bits
// go in place into a half's and the half is scaled by 2^8 (the exponent bias difference, 15 - 7):
// the same value for every finite code, subnormals and signed zeros included. The NaN code
// (S.1111.111), which no saturating writer stores, widens to +-480 instead of NaN.

#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace ninferno::ops::detail {

// The two codes in the low 16 bits of `pair`: the low byte to .x, the high byte to .y. The bit
// form, which every architecture runs (the test compares it with CUDA's conversion).
__device__ __forceinline__ __half2 e4m3x2_to_half2_bits(unsigned pair) {
    const unsigned spread = __byte_perm(pair, 0u, 0x1404u); // high code << 24 | low code << 8
    const unsigned bits   = (spread & 0x80008000u) | ((spread >> 1) & 0x3f803f80u);
    return __hmul2(*reinterpret_cast<const __half2*>(&bits), __float2half2_rn(256.0f));
}

// The conversion this architecture takes: the hardware's from sm_89, else the bit form.
__device__ __forceinline__ __half2 e4m3x2_to_half2(unsigned pair) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    return __half2(
        __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair & 0xffffu), __NV_E4M3));
#else
    return e4m3x2_to_half2_bits(pair);
#endif
}

__device__ __forceinline__ float e4m3_to_float(unsigned code) {
    return __low2float(e4m3x2_to_half2(code & 0xffu));
}

} // namespace ninferno::ops::detail
