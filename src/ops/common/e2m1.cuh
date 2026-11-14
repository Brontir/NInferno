#pragma once

// E2M1 (FP4) code pairs to FP16 pairs, for every FP4 decoder in the tree: the NVFP4 and MXFP4
// weight decoders and the NVFP4 KV-cache codec. The sm_100, sm_110 and sm_120 families convert a
// pair with one cvt.rn.f16x2.e2m1x2. Elsewhere CUDA's own conversion decodes each code through
// FP6 in software, which left the FP4 KV caches at about half of INT8's decode speed on Ada.

#include "core/arch_family.cuh"

#include <cuda_fp16.h>
#include <cuda_fp4.h>

#include <cstdint>

namespace ninferno::ops::detail {

// The bit form, exact on every GPU. A code's three magnitude bits placed at FP16 bits 9..11 read
// as E2M1(code) * 2^-14 for all eight magnitudes (code 1, E2M1 0.5, becomes the FP16 subnormal
// 2^-15), the sign bit moves to bit 15, and one multiply by 2^14 restores the value. The low
// nibble is the low half, as in the native conversion. Callable on every architecture so a test
// can compare it with CUDA's conversion for all 256 bytes.
__device__ __forceinline__ __half2 e2m1x2_to_half2_bits(std::uint32_t byte) {
    const std::uint32_t magnitude = ((byte & 0x7U) << 9) | ((byte & 0x70U) << 21);
    const std::uint32_t sign      = ((byte & 0x8U) << 12) | ((byte & 0x80U) << 24);
    const std::uint32_t bits      = magnitude | sign;
    const __half2 scaled          = *reinterpret_cast<const __half2*>(&bits);
    return __hmul2(scaled, __half2half2(__ushort_as_half(0x7400U))); // 2^14
}

// Nibbles P and P + 4 of a 32-bit code word (P = 0..3) as two FP16 values, nibble P in the low
// half, each E2M1(code) * 2^-14: the bit form above without its multiply. The two nibbles sit 16
// bits apart, so one shift places both magnitudes (bits 9..11 and 25..27) and another both signs
// (15 and 31). A consumer folds the 2^14 into a later FP32 scale: E2M1(code) * 2^-14 times an E4M3
// cache scale (a multiple of 2^-9, at most 448) is a multiple of 2^-24 of at most 6 significant
// bits, so the FP16 product is exact here as it is at full scale (kv_cache_nvfp4_dequant_f16x8).
template <int P>
__device__ __forceinline__ std::uint32_t e2m1_nibble_pair_bits(std::uint32_t word) {
    static_assert(P >= 0 && P < 4);
    std::uint32_t placed = word >> 3;
    if constexpr (P < 3) placed = word << (9 - 4 * P);
    const std::uint32_t sign = (word << (12 - 4 * P)) & 0x80008000U;
    return (placed & 0x0e000e00U) | sign;
}

// The low byte of `byte` as two FP16 values, low nibble first.
__device__ __forceinline__ __half2 e2m1x2_to_half2(std::uint32_t byte) {
#if NINFERNO_ARCH_FAMILY_SPECIFIC >= 1000 && NINFERNO_ARCH_FAMILY_SPECIFIC < 1300
    // NINFERNO-SPECIAL(arch:sm100,sm110,sm120): the families with the hardware conversion.
    return __half2(
        __nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(byte & 0xffU), __NV_E2M1));
#else
    return e2m1x2_to_half2_bits(byte);
#endif
}

} // namespace ninferno::ops::detail
