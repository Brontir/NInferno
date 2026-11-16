#pragma once

// The register decoders of the NVFP4 KV slab kernel (slab_mma.cuh, #849).
//
// A lane holds 32-bit code words: eight nibbles, the values d0 .. d0 + 7 of one key (byte b holds
// d0 + 2b in its low nibble, d0 + 2b + 1 in its high one). Pair p (0..3) of a word is one FP16x2
// register of an MMA fragment. The families with the hardware E2M1 conversion (sm_100, sm_110,
// sm_120) take byte p, nibbles 2p and 2p + 1, in one cvt. The others take nibbles p and p + 4 in
// the bit form without its multiply (e2m1_nibble_pair_bits: two shifts, two masks), so a value
// comes out at 2^-14 of itself and the kernel's FP32 scales restore it (kNvfp4SlabFold). Times its
// group's E4M3 scale, each FP16 value is exact either way, so the MMA products and their FP32 sums
// are those of the values kv_cache_nvfp4_dequant_f16x16 gives, times a power of two.

#include "ops/common/e2m1.cuh"
#include "ops/common/fp8_e4m3.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace ninferno::ops::detail {

// NINFERNO-SPECIAL(arch:sm100,sm110,sm120): the families with the hardware E2M1 conversion pair a
// word's nibbles by bytes; the others by the bit form's nibbles p and p + 4 (e2m1.cuh's gate).
#if NINFERNO_ARCH_FAMILY_SPECIFIC >= 1000 && NINFERNO_ARCH_FAMILY_SPECIFIC < 1300
#    define NINFERNO_NVFP4_SLAB_HARDWARE_E2M1 1
#else
#    define NINFERNO_NVFP4_SLAB_HARDWARE_E2M1 0
#endif

// The FP32 factor that restores the decoded values (device code only: the host pass has no
// architecture).
__device__ __forceinline__ constexpr float nvfp4_slab_fold() {
    return NINFERNO_NVFP4_SLAB_HARDWARE_E2M1 ? 1.0F : 16384.0F;
}

// The nibble of a word (0..7: value d0 + nibble) that half `half` (0 low, 1 high) of pair `pair`
// holds.
__device__ __forceinline__ constexpr int nvfp4_slab_nibble(int pair, int half) {
    return NINFERNO_NVFP4_SLAB_HARDWARE_E2M1 ? 2 * pair + half : pair + 4 * half;
}

// Pair P of a code word as FP16 bits, at nvfp4_slab_fold() times less than the value.
template <int P>
__device__ __forceinline__ std::uint32_t nvfp4_slab_pair(std::uint32_t word) {
    static_assert(P >= 0 && P < 4);
#if NINFERNO_NVFP4_SLAB_HARDWARE_E2M1
    const __half2 value = e2m1x2_to_half2(word >> (8 * P));
    return *reinterpret_cast<const std::uint32_t*>(&value);
#else
    return e2m1_nibble_pair_bits<P>(word);
#endif
}

// The byte_perm selector nvfp4_slab_scale2 takes for byte `byte` (0..3) of a scale word.
__device__ __forceinline__ std::uint32_t nvfp4_slab_scale_selector(std::uint32_t byte) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    return byte * 0x11U; // the byte in bytes 0 and 1: one cvt widens the pair
#else
    return byte * 0x1010U + 0x0404U; // the byte in bytes 1 and 3, zeros below each
#endif
}

// One E4M3 scale byte of `word` (picked by nvfp4_slab_scale_selector) as the FP16 pair (s, s),
// exact for every nonnegative finite code (the cache's scales: the quantizer stores at least
// 2^-9, and an unloaded key's zero). From sm_89 the hardware cvt; below it the code's exponent and
// mantissa bits in place into a half's, times 2^8 (fp8_e4m3.cuh's bit form, without the sign).
__device__ __forceinline__ __half2 nvfp4_slab_scale2(std::uint32_t word, std::uint32_t selector) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    return e4m3x2_to_half2(__byte_perm(word, 0U, selector));
#else
    const std::uint32_t bits = (__byte_perm(word, 0U, selector) >> 1) & 0x3f803f80U;
    return __hmul2(*reinterpret_cast<const __half2*>(&bits), __float2half2_rn(256.0F));
#endif
}

// A code pair times its scale pair: one FP16x2 MMA operand register.
__device__ __forceinline__ std::uint32_t nvfp4_slab_scaled(std::uint32_t pair, __half2 scale) {
    const __half2 value = __hmul2(*reinterpret_cast<const __half2*>(&pair), scale);
    return *reinterpret_cast<const std::uint32_t*>(&value);
}

} // namespace ninferno::ops::detail
