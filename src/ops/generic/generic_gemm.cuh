#pragma once

// Generic projection kernels: any [N,K] weight in any registered format, any T.
//
// out[r,t] = epilogue(sum_k W[r,k] * x[k,t]) with BF16 activations (A16 semantics).
//
// One kernel family serves every format. The weight is the MMA's A operand (16 output rows per
// m-tile) and the activation its B operand (8 tokens per n-tile); no shared memory holds operands.
// Each lane reads its own contiguous run of kRun values of a row per K step and dequantizes it in
// registers; it reads the activation run with the same K assignment, so a K step is 4 * kRun values
// and the four lanes of a quad cover it. The assignment only reorders the terms inside one K step.
// The order in which K steps accumulate is fixed (0, 1, ...), and with WarpsK == 1 one warp owns
// the whole K of its rows. Every WarpsK == 1 schedule therefore sums each output element in exactly
// the same order whatever its tile is, and a token's bytes do not depend on how many tokens share
// the call or where it sits in it.
//
// Groupwise formats (Q4/Q5/Q6 G64, Q8 G32, U4 G32/G64/G128 with zero points) feed the MMA unscaled codes
// (the integers, and code - zero, are exact in BF16) and apply the group scale to each K step's
// FP32 partial, one K step being one group. NVFP4
// codes times their E4M3 block scale are exact in BF16 and are dequantized directly. FP8 codes are
// exact in BF16. The FP8 row multiplier and the NVFP4 weight divisor are applied in the epilogue;
// an FP8 128x128 block multiplier scales each K step's FP32 partial (a K step lies in one block).
// ggml K-quants have several scale groups in one K step (half a 256-value superblock): the
// decoders are "spread", MMA j of a step taking the step's values 16j .. 16j + 15, so every MMA
// lies in one group; each group's FP32 partial takes its own scale, and Q4_K/Q5_K/Q2_K subtract
// their min term, dmin*m times the group's activation sum (an MMA of the activations with ones).

#include "core/pdl.cuh"
#include "ops/common/e2m1.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/ggml_iq_tables.cuh"
#include "ops/linear/fp8/fp8_a16_codec.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace ninferno::ops::detail::generic {

// Cluster > 1 splits K over the CTAs of a (1, 1, Cluster) thread-block cluster (grid z): the CTA of
// rank r sums K steps [steps*r/C, steps*(r+1)/C), and rank 0 adds the ranks' partials from
// distributed shared memory in rank order. It is for problems too small to give every SM a CTA
// (decode of a few dozen rows); the order depends on nothing but the schedule.
template <int WarpsM, int WarpsK, int MTiles, int NTiles, int Cluster = 1>
struct RmmaSchedule {
    static_assert(WarpsM > 0 && WarpsK > 0 && WarpsM * WarpsK <= 16);
    static_assert(MTiles > 0 && MTiles <= 4 && NTiles > 0 && NTiles <= 8);
    static_assert(Cluster == 1 || Cluster == 2 || Cluster == 4 || Cluster == 8);
    static constexpr int kWarpsM     = WarpsM;
    static constexpr int kWarpsK     = WarpsK;
    static constexpr int kMTiles     = MTiles;
    static constexpr int kNTiles     = NTiles;
    static constexpr int kCluster    = Cluster;
    static constexpr int kThreads    = WarpsM * WarpsK * 32;
    static constexpr int kRowsPerCta = WarpsM * MTiles * 16;
    static constexpr int kTokensPerCta = NTiles * 8;
    // A one-tile warp (the decode shapes) is latency-bound: two CTAs per SM, so its registers are
    // held to what that allows (one for the sixteen-warp CTA, which two would hold to 64).
    static constexpr int kMinBlocks = MTiles * NTiles == 1 && kThreads <= 256 ? 2 : 1;
};

// A decoder that reads a whole 32-value block per step (kRun == 32: every ggml block, U4Zp,
// MXFP4) keeps a 2 x 8 tile's codes and scales in registers: under SwiGLU (two row sets) it
// outgrows 255 and spills (ptxas: 456-1024 B of stack per thread). Neither the heuristic nor a
// route table takes that tile for these formats, so it is not compiled for them
// (rmma_tile_compiled says so; route_admits and the MoE plan check refuse it): every compiled
// GGUF register-MMA kernel keeps its stack within 512 B.
template <class Decoder, int MTiles, int NTiles>
inline constexpr bool rmma_compiled = !(Decoder::kRun == 32 && MTiles * NTiles >= 16);

// ---------------------------------------------------------------------------------------------
// Weight decoders. A decoder reads the run of kRun consecutive K values one lane owns in one K step
// of one row, and turns values 4j..4j+3 of the run into the two BF16x2 registers an m16n8k16 MMA
// reads for the row: `lo` holds values 4j, 4j+1 and `hi` values 4j+2, 4j+3.
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ unsigned bf16x2_bits(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&v);
}

// Two signed integers into one BF16x2 register (exact for |q| <= 256).
__device__ __forceinline__ unsigned int_pair_bf16(int lo, int hi) {
    return bf16x2_bits(static_cast<float>(lo), static_cast<float>(hi));
}

// Four bit fields of x (1, 2 or 3 bits wide, packed), each into the low bits of its own byte.
__device__ __forceinline__ unsigned spread1x4(unsigned x) {
    return (x | (x << 7) | (x << 14) | (x << 21)) & 0x01010101U;
}

__device__ __forceinline__ unsigned spread2x4(unsigned x) {
    return (x | (x << 6) | (x << 12) | (x << 18)) & 0x03030303U;
}

__device__ __forceinline__ unsigned spread3x4(unsigned x) {
    return (x & 0x7U) | ((x & 0x38U) << 5) | ((x & 0x1c0U) << 10) | ((x & 0xe00U) << 15);
}

// Four nibbles (bits 0 .. 15 of n), each into its own byte.
__device__ __forceinline__ unsigned spread4x4(unsigned n) {
    return (n & 0xfU) | ((n & 0xf0U) << 4) | ((n & 0xf00U) << 8) | ((n & 0xf000U) << 12);
}

__device__ __forceinline__ unsigned uint4_word(const uint4& v, int j) {
    return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w;
}

// Bytes 0 and 2 of w (codes < 128; bytes 1 and 3 are ignored) as a BF16 pair of code - bias, where
// `offset` is the BF16 pair of 128 + bias (exact: bias < 128).
__device__ __forceinline__ unsigned codes02_bf16(unsigned w, unsigned offset) {
    const unsigned v        = (w & 0x00ff00ffU) | 0x43004300U;
    const __nv_bfloat162 r = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&v),
                                     *reinterpret_cast<const __nv_bfloat162*>(&offset));
    return *reinterpret_cast<const unsigned*>(&r);
}

// Four byte codes of w (each < 128) as BF16 pairs of code - Bias: lo holds bytes 0, 1 and hi bytes
// 2, 3. Byte 0x43 above a code c makes the BF16 value 128 + c exactly (exponent 2^7, mantissa c),
// and one BF16x2 subtraction of 128 + Bias leaves the small integer c - Bias, also exact. Two byte
// permutes and two subtractions replace four integer-to-float conversions and two packs.
//
// F16: the same codes as FP16 pairs, for the MMA that takes FP16 operands (kF16Operands): byte
// 0x64 above a code c makes the FP16 value 1024 + c exactly (exponent 2^10, mantissa c < 256), and
// one FP16x2 subtraction of 1024 + Bias leaves c - Bias.
template <int Bias, bool F16 = false>
__device__ __forceinline__ void byte_codes(unsigned w, unsigned& lo, unsigned& hi) {
    static_assert(Bias >= 0 && Bias < 128, "128 + Bias must be exact in BF16");
    if constexpr (F16) {
        const __half2 offset = __float2half2_rn(1024.0F + static_cast<float>(Bias));
        unsigned a           = __byte_perm(w, 0x64U, 0x4140U);
        unsigned b           = __byte_perm(w, 0x64U, 0x4342U);
        const __half2 va     = __hsub2(*reinterpret_cast<const __half2*>(&a), offset);
        const __half2 vb     = __hsub2(*reinterpret_cast<const __half2*>(&b), offset);
        lo                   = *reinterpret_cast<const unsigned*>(&va);
        hi                   = *reinterpret_cast<const unsigned*>(&vb);
    } else {
        const __nv_bfloat162 offset = __float2bfloat162_rn(128.0F + static_cast<float>(Bias));
        unsigned a                  = __byte_perm(w, 0x43U, 0x4140U);
        unsigned b                  = __byte_perm(w, 0x43U, 0x4342U);
        const __nv_bfloat162 va     = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&a), offset);
        const __nv_bfloat162 vb     = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&b), offset);
        lo                          = *reinterpret_cast<const unsigned*>(&va);
        hi                          = *reinterpret_cast<const unsigned*>(&vb);
    }
}

// NINFERNO-SPECIAL(arch:sm75): Turing's MMA takes FP16, not BF16 (ops/common/mma.cuh rounds every
// BF16 operand register to FP16 and composes each k16 from two k8). A decoder whose weight codes
// come from byte_codes alone (small integers, exact in either element) declares kF16 =
// kF16Operands: in the sm_75 device pass its codes are built as FP16 and mma_step converts the
// activations once per step instead of every operand of every MMA (A16 there measured 3.2x slower
// than R8: Llama 3.1 8B Q4_K_M 27.3 against 85.9 tok/s). Every other build keeps BF16; the
// sm_70 build too, whose MMA is FP32 FMAs that read BF16 as cheaply as FP16 (ops/common/mma.cuh).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 800
inline constexpr bool kF16Operands = true;
#else
inline constexpr bool kF16Operands = false;
#endif

__device__ __forceinline__ unsigned ld_stream_u32(const std::uint8_t* p) {
    unsigned v;
    asm volatile("ld.global.cs.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ uint2 ld_stream_v2(const std::uint8_t* p) {
    uint2 v;
    asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

__device__ __forceinline__ uint4 ld_stream_v4(const std::uint8_t* p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ std::int64_t nvfp4_scale_offset(const GemmArgs& a, int row, int group) {
    const int row_inner = row & 127;
    return (static_cast<std::int64_t>(row >> 7) * a.scale_tiles + (group >> 2)) * 512 +
           (row_inner & 31) * 16 + (row_inner >> 5) * 4 + (group & 3);
}

// E2M1 codes times two are the integers 0, 1, 2, 3, 4, 6, 8, 12 (bit 3: the sign), exact in BF16.
// The magnitudes' BF16 low and high bytes, entries 0..7 over two words each.
inline constexpr unsigned kE2m1x2Low0  = 0x40008000U; // 0x00, 0x80, 0x00, 0x40
inline constexpr unsigned kE2m1x2Low1  = 0x4000c080U; // 0x80, 0xc0, 0x00, 0x40
inline constexpr unsigned kE2m1x2High0 = 0x40403f00U; // 0x00, 0x3f, 0x40, 0x40
inline constexpr unsigned kE2m1x2High1 = 0x41414040U; // 0x40, 0x40, 0x41, 0x41

// Four E2M1 codes (nibble n of c: code n) as the BF16 pairs of their doubles: lo holds codes 0, 1
// and hi codes 2, 3.
__device__ __forceinline__ void e2m1x2_bf16(unsigned c, unsigned& lo, unsigned& hi) {
    const unsigned m = c & 0x7777U;
    const unsigned l = __byte_perm(kE2m1x2Low0, kE2m1x2Low1, m);
    const unsigned h = __byte_perm(kE2m1x2High0, kE2m1x2High1, m);
    lo               = __byte_perm(l, h, 0x5140U) | ((c & 0x0008U) << 12) | ((c & 0x0080U) << 24);
    hi               = __byte_perm(l, h, 0x7362U) | ((c & 0x0800U) << 4) | ((c & 0x8000U) << 16);
}

struct Nvfp4Decoder {
    static constexpr int kRun        = 16; // one 16-value scale group: eight code bytes
    static constexpr bool kStepScale = false;
    struct Run {
        uint2 codes;
        unsigned scale;
    };
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        Run r;
        r.codes = ld_stream_v2(a.codes + static_cast<std::int64_t>(row) * (a.k / 2) + k0 / 2);
        r.scale = a.scales[nvfp4_scale_offset(a, row, k0 / 16)];
        return r;
    }
    __device__ static __forceinline__ Run zero() { return {make_uint2(0, 0), 0}; }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ unsigned dq(unsigned byte, __half2 scale) {
        const __half2 product  = __hmul2(e2m1x2_to_half2(byte), scale);
        const __nv_bfloat162 w = __float22bfloat162_rn(__half22float2(product));
        return *reinterpret_cast<const unsigned*>(&w);
    }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const unsigned word = j < 2 ? r.codes.x : r.codes.y;
        const int shift     = (j & 1) * 16;
// cvt with .e2m1x2 / FP8-to-half: Blackwell's architecture- and family-specific targets
// (sm_100a.., sm_120a, sm_120f: __CUDA_ARCH_FAMILY_SPECIFIC__), not a plain target's PTX.
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FAMILY_SPECIFIC__) &&                            \
    __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000
        const __half_raw h =
            __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(r.scale & 0xffU), __NV_E4M3);
        const __half s(h);
        const __half2 scale(s, s);
        lo = dq(word >> shift, scale);
        hi = dq(word >> (shift + 8), scale);
#else
        // No FP4 converter below sm_100 (the library emulates it in a long integer sequence): the
        // doubled codes by byte permutes times half the block scale. Both factors and their
        // product (at most 8 significant bits) are exact in BF16, so these are the converter
        // path's bits.
        __nv_fp8_e4m3 block_scale;
        block_scale.__x = static_cast<__nv_fp8_storage_t>(r.scale & 0xffU);
        const __nv_bfloat162 half_scale =
            __float2bfloat162_rn(0.5F * static_cast<float>(block_scale));
        unsigned doubled_lo, doubled_hi;
        e2m1x2_bf16(word >> shift, doubled_lo, doubled_hi);
        const __nv_bfloat162 product_lo =
            __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&doubled_lo), half_scale);
        const __nv_bfloat162 product_hi =
            __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&doubled_hi), half_scale);
        lo = *reinterpret_cast<const unsigned*>(&product_lo);
        hi = *reinterpret_cast<const unsigned*>(&product_hi);
#endif
    }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs& a, int row) {
        if (a.divisors != nullptr) { return 1.0F / a.divisors[row / a.divisor_rows]; }
        return a.inverse_divisor;
    }
};

// E8M0 exponent word -> its value 2^(e-127) as binary32 (e = 0 is the subnormal 2^-127; e = 255,
// NaN, is refused by every producer).
__device__ __forceinline__ float e8m0_value(unsigned e) {
    return e != 0 ? __uint_as_float(e << 23) : __uint_as_float(0x00400000U);
}

// MXFP4: a run is one whole 32-value exponent group, sixteen code bytes. E2M1 times a power of
// two is exact in BF16, so the weight is dequantized directly and needs no epilogue scale.
struct Mxfp4Decoder {
    static constexpr int kRun        = 32;
    static constexpr bool kStepScale = false;
    struct Run {
        uint4 codes;
        float scale;
    };
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        Run r;
        r.codes = ld_stream_v4(a.codes + static_cast<std::int64_t>(row) * (a.k / 2) + k0 / 2);
        r.scale = e8m0_value(a.scales[static_cast<std::int64_t>(row) * (a.k / 32) + k0 / 32]);
        return r;
    }
    __device__ static __forceinline__ Run zero() { return {make_uint4(0, 0, 0, 0), 0.0F}; }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ unsigned dq(unsigned byte, float scale) {
        float2 v = __half22float2(e2m1x2_to_half2(byte));
        v.x *= scale;
        v.y *= scale;
        const __nv_bfloat162 w = __float22bfloat162_rn(v);
        return *reinterpret_cast<const unsigned*>(&w);
    }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const unsigned word = (j >> 1) == 0   ? r.codes.x
                              : (j >> 1) == 1 ? r.codes.y
                              : (j >> 1) == 2 ? r.codes.z
                                              : r.codes.w;
        const int shift     = (j & 1) * 16;
        lo                  = dq(word >> shift, r.scale);
        hi                  = dq(word >> (shift + 8), r.scale);
    }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

struct Fp8Decoder {
    static constexpr int kRun        = 16;
    static constexpr bool kStepScale = false;
    struct Run {
        uint4 codes;
    };
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        return {ld_stream_v4(a.codes + static_cast<std::int64_t>(row) * a.k + k0)};
    }
    __device__ static __forceinline__ Run zero() { return {make_uint4(0, 0, 0, 0)}; }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ unsigned widen(unsigned packed) {
        return fp8_e4m3x2_to_bf16x2_bits(packed & 0xffffU);
    }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const unsigned w = j == 0 ? r.codes.x : j == 1 ? r.codes.y : j == 2 ? r.codes.z : r.codes.w;
        lo               = widen(w);
        hi               = widen(w >> 16);
    }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs& a, int row) {
        const std::uint16_t bits = reinterpret_cast<const std::uint16_t*>(a.scales)[row];
        return __bfloat162float(__ushort_as_bfloat16(bits));
    }
};

// FP8 with a binary32 row multiplier: per-channel or per-tensor scales kept as published.
struct Fp8RowF32Decoder : Fp8Decoder {
    __device__ static __forceinline__ float row_scale(const GemmArgs& a, int row) {
        return reinterpret_cast<const float*>(a.scales)[row];
    }
};

// FP8 with one binary32 multiplier per 128x128 block. A K step is 64 values, inside one block
// column, so the multiplier of the lane's row and block scales the step's FP32 partial.
struct Fp8B128Decoder {
    static constexpr int kRun        = 16;
    static constexpr bool kStepScale = true;
    struct Run {
        uint4 codes;
        float scale;
    };
    __device__ static __forceinline__ float block_scale(const GemmArgs& a, int row, int k) {
        return reinterpret_cast<const float*>(
            a.scales)[static_cast<std::int64_t>(row >> 7) * a.scale_tiles + (k >> 7)];
    }
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        return {ld_stream_v4(a.codes + static_cast<std::int64_t>(row) * a.k + k0),
                block_scale(a, row, k0)};
    }
    __device__ static __forceinline__ Run zero() { return {make_uint4(0, 0, 0, 0), 0.0F}; }
    // A lane past K: no codes, but the block's multiplier (every lane scales the quad's sum).
    __device__ static __forceinline__ Run masked(const GemmArgs& a, int row, int step_k0) {
        return {make_uint4(0, 0, 0, 0), block_scale(a, row, step_k0)};
    }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const unsigned w = j == 0 ? r.codes.x : j == 1 ? r.codes.y : j == 2 ? r.codes.z : r.codes.w;
        lo               = Fp8Decoder::widen(w);
        hi               = Fp8Decoder::widen(w >> 16);
    }
    __device__ static __forceinline__ float step_scale(const Run& r) { return r.scale; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

struct Bf16Decoder {
    static constexpr int kRun        = 8;
    static constexpr bool kStepScale = false;
    struct Run {
        uint4 words;
    };
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        return {ld_stream_v4(a.codes + (static_cast<std::int64_t>(row) * a.k + k0) * 2)};
    }
    __device__ static __forceinline__ Run zero() { return {make_uint4(0, 0, 0, 0)}; }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        lo = j == 0 ? r.words.x : r.words.z;
        hi = j == 0 ? r.words.y : r.words.w;
    }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// Groupwise RowSplit G64: base codes, optional lane-major high-bit plane, one FP16 scale per group.
// A K step of 64 values is exactly one group.
template <int Bits>
struct GroupwiseG64Decoder {
    static_assert(Bits == 4 || Bits == 5 || Bits == 6);
    static constexpr bool kF16       = kF16Operands;
    static constexpr int kBias       = 1 << (Bits - 1);
    static constexpr int kRun        = 16;
    static constexpr bool kStepScale = true;
    struct Run {
        uint2 codes;
        unsigned high;
        unsigned scale;
    };
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int group        = k0 >> 6;
        const int lane_offset  = k0 & 63; // 16 * quad
        const std::int64_t gid = static_cast<std::int64_t>(row) * a.groups_per_row + group;
        Run r;
        r.codes = *reinterpret_cast<const uint2*>(a.codes + gid * 32 + lane_offset / 2);
        if constexpr (Bits == 5) {
            r.high = *reinterpret_cast<const unsigned short*>(a.high + gid * 8 + lane_offset / 8);
        } else if constexpr (Bits == 6) {
            r.high = *reinterpret_cast<const unsigned*>(a.high + gid * 16 + lane_offset / 4);
        } else {
            r.high = 0;
        }
        r.scale = reinterpret_cast<const unsigned short*>(a.scales)[gid];
        return r;
    }
    __device__ static __forceinline__ Run zero() { return {make_uint2(0, 0), 0, 0}; }
    // A lane past K in a row's last, partial group: no codes, but the group scale, because the
    // MMA sums the quad's four runs into every lane's partial and each lane scales its partial.
    __device__ static __forceinline__ Run masked(const GemmArgs& a, int row, int step_k0) {
        const std::int64_t gid = static_cast<std::int64_t>(row) * a.groups_per_row + (step_k0 >> 6);
        return {make_uint2(0, 0), 0, reinterpret_cast<const unsigned short*>(a.scales)[gid]};
    }
    __device__ static __forceinline__ int code(const Run& r, int v) {
        const unsigned word = v < 8 ? r.codes.x : r.codes.y;
        const unsigned nib  = (word >> ((v & 7) * 4)) & 0xfU;
        if constexpr (Bits == 4) {
            return static_cast<int>(nib ^ 8U) - 8;
        } else if constexpr (Bits == 5) {
            const unsigned u = nib | (((r.high >> v) & 1U) << 4);
            return static_cast<int>(u ^ 16U) - 16;
        } else {
            const unsigned u = nib | (((r.high >> (2 * v)) & 3U) << 4);
            return static_cast<int>(u ^ 32U) - 32;
        }
    }

    // Values 4j .. 4j + 3 as four code bytes (nibble, high bits above it), the sign bit flipped:
    // c ^ 2^(Bits-1) is the code biased by 2^(Bits-1), which byte_codes takes off exactly, so the
    // pairs equal code()'s values bit for bit at a quarter of the instructions.
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const unsigned word = j < 2 ? r.codes.x : r.codes.y;
        unsigned q          = spread4x4((word >> (16 * (j & 1))) & 0xffffU);
        if constexpr (Bits == 5) {
            q |= spread1x4((r.high >> (4 * j)) & 0xfU) << 4;
        } else if constexpr (Bits == 6) {
            q |= spread2x4((r.high >> (8 * j)) & 0xffU) << 4;
        }
        byte_codes<kBias, kF16>(q ^ (0x01010101U * kBias), lo, hi);
    }
    __device__ static __forceinline__ float step_scale(const Run& r) {
        return __half2float(__ushort_as_half(static_cast<unsigned short>(r.scale)));
    }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// Q3 G128 / G64: a run is 16 three-bit codes, 48 bits starting on an even byte (code j of a group
// in bits 3j..3j+2 of its code bytes); a K step of 64 values stays inside one group.
template <int Group>
struct GroupwiseQ3Decoder {
    static_assert(Group == 64 || Group == 128);
    static constexpr bool kF16       = kF16Operands;
    static constexpr int kRun        = 16;
    static constexpr bool kStepScale = true;
    static constexpr int kGroupBytes = Group * 3 / 8;

    struct Run {
        uint2 codes; // x: bits 0..31 of the run, y: bits 32..47
        unsigned scale;
    };

    __device__ static __forceinline__ std::int64_t gid(const GemmArgs& a, int row, int k0) {
        return static_cast<std::int64_t>(row) * a.groups_per_row + k0 / Group;
    }

    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const std::int64_t g = gid(a, row, k0);
        const auto* p        = reinterpret_cast<const unsigned short*>(a.codes + g * kGroupBytes +
                                                                       (k0 % Group) * 3 / 8);
        Run r;
        r.codes = make_uint2(static_cast<unsigned>(p[0]) | (static_cast<unsigned>(p[1]) << 16),
                             static_cast<unsigned>(p[2]));
        r.scale = reinterpret_cast<const unsigned short*>(a.scales)[g];
        return r;
    }

    __device__ static __forceinline__ Run zero() { return {make_uint2(0, 0), 0}; }

    __device__ static __forceinline__ Run masked(const GemmArgs& a, int row, int step_k0) {
        return {make_uint2(0, 0),
                reinterpret_cast<const unsigned short*>(a.scales)[gid(a, row, step_k0)]};
    }

    __device__ static __forceinline__ int code(const Run& r, int v) {
        const unsigned long long bits =
            (static_cast<unsigned long long>(r.codes.y) << 32) | r.codes.x;
        const unsigned u = static_cast<unsigned>(bits >> (3 * v)) & 7U;
        return static_cast<int>(u ^ 4U) - 4;
    }

    // Values 4j .. 4j + 3: bits 12j .. 12j + 11 of the run as four code bytes, signs flipped
    // (code()'s values bit for bit, through byte_codes).
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const unsigned long long bits =
            (static_cast<unsigned long long>(r.codes.y) << 32) | r.codes.x;
        const unsigned x = static_cast<unsigned>(bits >> (12 * j)) & 0xfffU;
        byte_codes<4, kF16>(spread3x4(x) ^ 0x04040404U, lo, hi);
    }

    __device__ static __forceinline__ float step_scale(const Run& r) {
        return __half2float(__ushort_as_half(static_cast<unsigned short>(r.scale)));
    }

    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// Q8 G32: a K step of 32 values is exactly one group.
struct Q8Decoder {
    static constexpr int kRun        = 8;
    static constexpr bool kStepScale = true;
    struct Run {
        uint2 codes;
        unsigned scale;
    };
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int group        = k0 >> 5;
        const int lane_offset  = k0 & 31;
        const std::int64_t gid = static_cast<std::int64_t>(row) * a.groups_per_row + group;
        Run r;
        r.codes = *reinterpret_cast<const uint2*>(a.codes + gid * 32 + lane_offset);
        r.scale = reinterpret_cast<const unsigned short*>(a.scales)[gid];
        return r;
    }
    __device__ static __forceinline__ Run zero() { return {make_uint2(0, 0), 0}; }
    __device__ static __forceinline__ Run masked(const GemmArgs& a, int row, int step_k0) {
        const std::int64_t gid = static_cast<std::int64_t>(row) * a.groups_per_row + (step_k0 >> 5);
        return {make_uint2(0, 0), reinterpret_cast<const unsigned short*>(a.scales)[gid]};
    }
    __device__ static __forceinline__ int code(const Run& r, int v) {
        const unsigned word = v < 4 ? r.codes.x : r.codes.y;
        return static_cast<int>(static_cast<signed char>((word >> ((v & 3) * 8)) & 0xffU));
    }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        lo = int_pair_bf16(code(r, 4 * j), code(r, 4 * j + 1));
        hi = int_pair_bf16(code(r, 4 * j + 2), code(r, 4 * j + 3));
    }
    __device__ static __forceinline__ float step_scale(const Run& r) {
        return __half2float(__ushort_as_half(static_cast<unsigned short>(r.scale)));
    }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// U4 G128 with zero points: a lane's run is 32 codes (sixteen bytes) of one group, so a K step of
// 128 values is exactly one group; (code - zero) is exact in BF16 and the group scale multiplies
// the step's FP32 partial. Word w of the run holds values 8w .. 8w + 7, value v in nibble v % 8.
struct U4ZpDecoder {
    static constexpr int kRun        = 32;
    static constexpr bool kStepScale = true;
    struct Run {
        uint4 codes;
        unsigned offset; // BF16 pair of 128 + zero (the format keeps zero < 128)
        unsigned scale;
    };
    __device__ static __forceinline__ unsigned offset_of(unsigned zero) {
        const __nv_bfloat162 v = __float2bfloat162_rn(128.0F + static_cast<float>(zero));
        return *reinterpret_cast<const unsigned*>(&v);
    }
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const std::int64_t gid = static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 7);
        const auto* scales = reinterpret_cast<const unsigned short*>(a.scales);
        return {ld_stream_v4(a.codes + static_cast<std::int64_t>(row) * (a.k / 2) + k0 / 2),
                offset_of(a.high[gid]), scales[gid]};
    }
    __device__ static __forceinline__ Run zero() {
        return {make_uint4(0, 0, 0, 0), offset_of(0), 0};
    }
    // A zero-code run (masked) must contribute nothing: its zero point is 0.
    __device__ static __forceinline__ Run masked(const GemmArgs& a, int row, int step_k0) {
        const std::int64_t gid = static_cast<std::int64_t>(row) * a.groups_per_row + (step_k0 >> 7);
        return {make_uint4(0, 0, 0, 0), offset_of(0),
                reinterpret_cast<const unsigned short*>(a.scales)[gid]};
    }
    // Values 4j .. 4j + 3 are nibbles 4*(j % 2) .. + 3 of word j / 2: the even nibbles of a word
    // as bytes, the odd ones as bytes, and a byte permute pairs them.
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const unsigned w    = uint4_word(r.codes, j >> 1);
        const unsigned even = w & 0x0f0f0f0fU;        // nibbles 0, 2, 4, 6
        const unsigned odd  = (w >> 4) & 0x0f0f0f0fU; // nibbles 1, 3, 5, 7
        const unsigned b    = 2U * static_cast<unsigned>(j & 1);
        lo = codes02_bf16(__byte_perm(even, odd, b | ((b + 4U) << 8)), r.offset);
        hi = codes02_bf16(__byte_perm(even, odd, (b + 1U) | ((b + 5U) << 8)), r.offset);
    }
    __device__ static __forceinline__ float step_scale(const Run& r) {
        return __half2float(__ushort_as_half(static_cast<unsigned short>(r.scale)));
    }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// U4 G32 / G64 with zero points (same planes as G128): a lane's run is G/4 codes of one group, so
// a K step of G values is exactly one group, as for Q8 G32 and Q4 G64; the codes of a run are
// U4ZpDecoder's first G/32 words.
template <int G>
struct U4ZpSmallDecoder {
    static_assert(G == 32 || G == 64);
    static constexpr int kRun        = G / 4;
    static constexpr bool kStepScale = true;
    struct Run {
        uint2 codes; // G32: .x only
        unsigned offset;
        unsigned scale;
    };
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const std::int64_t gid = static_cast<std::int64_t>(row) * a.groups_per_row + k0 / G;
        const auto* p = a.codes + static_cast<std::int64_t>(row) * (a.k / 2) + k0 / 2;
        uint2 codes;
        if constexpr (G == 32) {
            codes = make_uint2(ld_stream_u32(p), 0);
        } else {
            codes = ld_stream_v2(p);
        }
        return {codes, U4ZpDecoder::offset_of(a.high[gid]),
                reinterpret_cast<const unsigned short*>(a.scales)[gid]};
    }
    __device__ static __forceinline__ Run zero() {
        return {make_uint2(0, 0), U4ZpDecoder::offset_of(0), 0};
    }
    __device__ static __forceinline__ Run masked(const GemmArgs& a, int row, int step_k0) {
        const std::int64_t gid = static_cast<std::int64_t>(row) * a.groups_per_row + step_k0 / G;
        return {make_uint2(0, 0), U4ZpDecoder::offset_of(0),
                reinterpret_cast<const unsigned short*>(a.scales)[gid]};
    }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const unsigned w    = (j >> 1) == 0 ? r.codes.x : r.codes.y;
        const unsigned even = w & 0x0f0f0f0fU;
        const unsigned odd  = (w >> 4) & 0x0f0f0f0fU;
        const unsigned b    = 2U * static_cast<unsigned>(j & 1);
        lo = codes02_bf16(__byte_perm(even, odd, b | ((b + 4U) << 8)), r.offset);
        hi = codes02_bf16(__byte_perm(even, odd, (b + 1U) | ((b + 5U) << 8)), r.offset);
    }
    __device__ static __forceinline__ float step_scale(const Run& r) {
        return __half2float(__ushort_as_half(static_cast<unsigned short>(r.scale)));
    }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

using U4ZpG32Decoder = U4ZpSmallDecoder<32>;
using U4ZpG64Decoder = U4ZpSmallDecoder<64>;

// U4 G32 / G64 with zero points, K in whole 128-value steps: a spread decoder over the same
// planes. A step holds 128 / G groups; a quad-run decoder (kQuadRuns) gives quad q of group g the
// G/4 consecutive values g*G + q*G/4 .. + G/4 - 1, spread over the group's G/16 MMAs four at a
// time (MMA i of the step: group i / (G/16), values 4*(i % (G/16)) .. + 3 of the lane's span), so
// a lane's codes of a group are G/8 contiguous bytes, every MMA lies in one group and its FP32
// partial takes that group's scale. pair() cuts codes as U4ZpDecoder does, with the group's zero.
template <int G>
struct U4ZpQuadDecoder {
    static_assert(G == 32 || G == 64);
    static constexpr int kRun        = 32;
    static constexpr bool kStepScale = true;
    static constexpr int kSplits     = 128 / G;
    static constexpr bool kQuadRuns  = true;
    struct Run {
        uint4 codes;    // word w: the lane's code bytes 4w .. 4w + 3 of the step, in group order
        unsigned zeros; // the step's zero points, one byte per group
        uint2 scales;   // the step's binary16 scales, two per word
    };
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int base = k0 & ~127;
        const int q    = (k0 >> 5) & 3;
        const std::int64_t gid = static_cast<std::int64_t>(row) * a.groups_per_row + base / G;
        const std::uint8_t* p  = a.codes + static_cast<std::int64_t>(row) * (a.k / 2) + base / 2;
        Run r;
        if constexpr (G == 32) {
            r.codes  = make_uint4(ld_stream_u32(p + 4 * q), ld_stream_u32(p + 16 + 4 * q),
                                  ld_stream_u32(p + 32 + 4 * q), ld_stream_u32(p + 48 + 4 * q));
            r.zeros  = *reinterpret_cast<const unsigned*>(a.high + gid);
            r.scales = *reinterpret_cast<const uint2*>(a.scales + 2 * gid);
        } else {
            const uint2 lo = ld_stream_v2(p + 8 * q);
            const uint2 hi = ld_stream_v2(p + 32 + 8 * q);
            r.codes  = make_uint4(lo.x, lo.y, hi.x, hi.y);
            r.zeros  = *reinterpret_cast<const unsigned short*>(a.high + gid);
            r.scales = make_uint2(*reinterpret_cast<const unsigned*>(a.scales + 2 * gid), 0);
        }
        return r;
    }
    __device__ static __forceinline__ Run zero() {
        return {make_uint4(0, 0, 0, 0), 0, make_uint2(0, 0)};
    }
    // K is whole steps (the dispatch's condition), so no lane of a step is past K.
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    // BF16 128 + zero, twice: exponent 7 with the zero point (< 128) as mantissa.
    __device__ static __forceinline__ unsigned offset(const Run& r, int g) {
        const unsigned h = 0x4300U | ((r.zeros >> (8 * g)) & 0xffU);
        return h | (h << 16);
    }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const unsigned w    = uint4_word(r.codes, j >> 1);
        const unsigned even = w & 0x0f0f0f0fU;
        const unsigned odd  = (w >> 4) & 0x0f0f0f0fU;
        const unsigned b    = 2U * static_cast<unsigned>(j & 1);
        const unsigned off  = offset(r, j / (G / 16));
        lo = codes02_bf16(__byte_perm(even, odd, b | ((b + 4U) << 8)), off);
        hi = codes02_bf16(__byte_perm(even, odd, (b + 1U) | ((b + 5U) << 8)), off);
    }
    __device__ static __forceinline__ float sub_scale(const Run& r, int g) {
        const unsigned w = (g >> 1) == 0 ? r.scales.x : r.scales.y;
        return __half2float(__ushort_as_half(static_cast<unsigned short>(w >> (16 * (g & 1)))));
    }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

using U4ZpG32QuadDecoder = U4ZpQuadDecoder<32>;
using U4ZpG64QuadDecoder = U4ZpQuadDecoder<64>;

// ---------------------------------------------------------------------------------------------
// ggml K-quants, 256-value superblocks along K, kept as published. These decoders are spread: a K
// step is 128 values (half h of a superblock), and MMA j of it covers the step's values
// 16j .. 16j + 15 (quad q: 16j + 4q .. + 3), so every MMA lies in one scale group (Q4_K/Q5_K:
// 32 values, MMAs 2g and 2g + 1; Q6_K, Q3_K, Q2_K: 16 values, MMA g) and its FP32 partial takes
// that group's scale. A run keeps the published words the lane needs; pair() cuts MMA j's four
// codes out of them (one per byte, byte_codes), sub_scale() and sub_min() a group's factors.
// Word u of a 32-byte span is its bytes 16u + 4q .. 16u + 4q + 3: the lane's codes of the MMAs
// with j % 2 == u.
// ---------------------------------------------------------------------------------------------

// ggml get_scale_min_k4: the 6-bit scale and min of sub-block j (0..7) of a Q4_K/Q5_K head
// (d, dmin, then the 12 packed scale bytes). The scalar fallback's.
__device__ __forceinline__ unsigned kq_byte(const uint4& head, int i) {
    const unsigned w = i < 4 ? head.y : i < 8 ? head.z : head.w;
    return (w >> ((i & 3) * 8)) & 0xffU;
}

__device__ __forceinline__ void kq_scale_min(const uint4& head, int j, float& sc, float& m) {
    unsigned s, mm;
    if (j < 4) {
        s  = kq_byte(head, j) & 63U;
        mm = kq_byte(head, j + 4) & 63U;
    } else {
        s  = (kq_byte(head, j + 4) & 0xfU) | ((kq_byte(head, j - 4) >> 6) << 4);
        mm = (kq_byte(head, j + 4) >> 4) | ((kq_byte(head, j) >> 6) << 4);
    }
    sc = static_cast<float>(s);
    m  = static_cast<float>(mm);
}

__device__ __forceinline__ float half_bits(unsigned bits) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits & 0xffffU)));
}

// Byte i of w, as a float.
__device__ __forceinline__ float byte_of(unsigned w, int i) {
    return static_cast<float>((w >> (8 * i)) & 0xffU);
}

// Q4_K (Bits 4, 144-byte blocks: d, dmin, 12 scale bytes, 128 nibble bytes) and Q5_K (Bits 5,
// 176-byte blocks: the same head, 32 high-bit bytes, 128 nibble bytes). Sub-block b (32 values)
// is the low (even b) or high (odd b) nibbles of bytes 32*(b/2) .. + 31, plus bit b of high-bit
// bytes 0 .. 31 (Q5_K); w = d*sc*q - dmin*m with ggml's packed 6-bit scales and mins.
// load() cuts the step's scales, mins and high bits out as it loads (master's form); load_raw()
// only loads and unpack() cuts them when the step computes (UnpackOf, rmma_kernel's latency
// variant): in load() each row's fetch waits for its own head before the next row's loads issue.
template <int Bits>
struct GgmlQ45KDecoder {
    static constexpr bool kF16 = kF16Operands;
    static_assert(Bits == 4 || Bits == 5);
    static constexpr int kRun         = 32;
    static constexpr bool kStepScale  = true;
    static constexpr int kSplits      = 4;
    static constexpr bool kMinTerm    = true;
    static constexpr bool kStageActs  = true;
    static constexpr int kBlockBytes  = Bits == 4 ? 144 : 176;
    static constexpr int kNibbleStart = Bits == 4 ? 16 : 48;
    struct Run {
        uint4 nib;   // word 2c + u: word u of the step's nibble bytes 32c .. 32c + 31
        uint2 hbits; // Q5_K: word u of the high-bit bytes, shifted so bit g is group g
        // The step's four 6-bit scales and mins (sub-blocks 4h .. 4h + 3), bytes. load_raw():
        // scale bytes 0..3, 4..7 and 8..11 (head.y, z, w) until unpack(); hi is unused after it.
        unsigned sc, mn, hi;
        unsigned dd; // binary16 d (low half) and dmin (high half)
    };
    // Group g of step h is sub-block 4h + g: nibble g % 2 of nibble span g / 2.
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int p  = k0 & 255;
        const int h  = p >> 7;       // half of the superblock
        const int qd = (p >> 5) & 3; // this lane's quad
        const std::uint8_t* b =
            a.codes + (static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 8)) * kBlockBytes;
        const uint4 head = *reinterpret_cast<const uint4*>(b);
        const auto* c    = reinterpret_cast<const unsigned*>(b + kNibbleStart + 64 * h + 4 * qd);
        Run r;
        r.nib   = make_uint4(c[0], c[4], c[8], c[12]);
        r.hbits = make_uint2(0, 0);
        if constexpr (Bits == 5) {
            const auto* hb = reinterpret_cast<const unsigned*>(b + 16 + 4 * qd);
            r.hbits        = make_uint2(hb[0] >> (4 * h), hb[4] >> (4 * h));
        }
        // Scale bytes 0..11 are head.y, head.z, head.w. Sub-blocks 0..3: six bits of bytes j and
        // j + 4; 4..7: the nibbles of byte j + 4 under the top two bits of bytes j - 4 and j.
        r.sc = h == 0 ? head.y & 0x3f3f3f3fU
                      : (head.w & 0x0f0f0f0fU) | ((head.y >> 2) & 0x30303030U);
        r.mn = h == 0 ? head.z & 0x3f3f3f3fU
                      : ((head.w >> 4) & 0x0f0f0f0fU) | ((head.z >> 2) & 0x30303030U);
        r.hi = 0;
        r.dd = head.x;
        return r;
    }
    __device__ static __forceinline__ Run load_raw(const GemmArgs& a, int row, int k0) {
        const int p  = k0 & 255;
        const int h  = p >> 7;
        const int qd = (p >> 5) & 3;
        const std::uint8_t* b =
            a.codes + (static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 8)) * kBlockBytes;
        const uint4 head = *reinterpret_cast<const uint4*>(b);
        const auto* c    = reinterpret_cast<const unsigned*>(b + kNibbleStart + 64 * h + 4 * qd);
        Run r;
        r.nib   = make_uint4(c[0], c[4], c[8], c[12]);
        r.hbits = make_uint2(0, 0);
        if constexpr (Bits == 5) {
            const auto* hb = reinterpret_cast<const unsigned*>(b + 16 + 4 * qd);
            r.hbits        = make_uint2(hb[0], hb[4]);
        }
        r.sc = head.y;
        r.mn = head.z;
        r.hi = head.w;
        r.dd = head.x;
        return r;
    }
    // A load_raw() run of step base step_k0 (half h of its superblock) into load()'s form.
    __device__ static __forceinline__ void unpack(Run& r, int step_k0) {
        const int h      = (step_k0 >> 7) & 1;
        const unsigned y = r.sc, z = r.mn, w = r.hi;
        r.sc = h == 0 ? y & 0x3f3f3f3fU : (w & 0x0f0f0f0fU) | ((y >> 2) & 0x30303030U);
        r.mn = h == 0 ? z & 0x3f3f3f3fU : ((w >> 4) & 0x0f0f0f0fU) | ((z >> 2) & 0x30303030U);
        if constexpr (Bits == 5) r.hbits = make_uint2(r.hbits.x >> (4 * h), r.hbits.y >> (4 * h));
    }
    __device__ static __forceinline__ Run zero() {
        return {make_uint4(0, 0, 0, 0), make_uint2(0, 0), 0, 0, 0, 0};
    }
    // K is whole superblocks, so no lane of a step is past K.
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const int g = j >> 1;
        unsigned q  = (uint4_word(r.nib, 2 * (g >> 1) + (j & 1)) >> (4 * (g & 1))) & 0x0f0f0f0fU;
        if constexpr (Bits == 5) {
            q |= ((((j & 1) ? r.hbits.y : r.hbits.x) >> g) & 0x01010101U) << 4;
        }
        byte_codes<0, kF16>(q, lo, hi);
    }
    __device__ static __forceinline__ float sub_scale(const Run& r, int g) {
        return half_bits(r.dd) * byte_of(r.sc, g);
    }
    __device__ static __forceinline__ float sub_min(const Run& r, int g) {
        return half_bits(r.dd >> 16) * byte_of(r.mn, g);
    }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// Q6_K planes: per superblock 128 ql bytes and 64 qh bytes (codes), 16 int8 scales (high plane),
// binary16 d (scale plane). Value v of half h: quarter qq = v / 32, l = v % 32: low (qq < 2) or
// high nibble of ql[64h + 32*(qq & 1) + l], bits 2*qq of qh[32h + l]; scale sc[8h + 2*qq + l/16];
// w = d*sc*(q - 32). MMA j is quarter j / 2, word j % 2; its scale is sc[8h + j].
struct GgmlQ6KDecoder {
    static constexpr bool kF16       = kF16Operands;
    static constexpr int kRun        = 32;
    static constexpr bool kStepScale = true;
    static constexpr int kSplits     = 8;
    static constexpr bool kStageActs = true;
    struct Run {
        uint4 ql;   // word 2a + u: word u of ql bytes 64h + 32a .. + 31
        uint2 qh;   // word u of qh bytes 32h .. 32h + 31
        uint2 sc;   // int8 scales 8h .. 8h + 7
        unsigned d; // binary16 d
    };
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int p            = k0 & 255;
        const int h            = p >> 7;
        const int qd           = (p >> 5) & 3;
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 8);
        const std::uint8_t* cb = a.codes + blk * 192;
        const auto* ql         = reinterpret_cast<const unsigned*>(cb + 64 * h + 4 * qd);
        const auto* qh         = reinterpret_cast<const unsigned*>(cb + 128 + 32 * h + 4 * qd);
        Run r;
        r.ql = make_uint4(ql[0], ql[4], ql[8], ql[12]);
        r.qh = make_uint2(qh[0], qh[4]);
        r.sc = *reinterpret_cast<const uint2*>(a.high + blk * 16 + 8 * h);
        r.d  = reinterpret_cast<const unsigned short*>(a.scales)[blk];
        return r;
    }
    // Codes 32 (qh bits 10 everywhere): value 0.
    __device__ static __forceinline__ Run zero() {
        return {make_uint4(0, 0, 0, 0), make_uint2(0xaaaaaaaaU, 0xaaaaaaaaU), make_uint2(0, 0), 0};
    }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const int qq     = j >> 1;
        const unsigned h = (j & 1) ? r.qh.y : r.qh.x;
        const unsigned q =
            ((uint4_word(r.ql, 2 * (qq & 1) + (j & 1)) >> (4 * (qq >> 1))) & 0x0f0f0f0fU) |
            (((h >> (2 * qq)) & 0x03030303U) << 4);
        byte_codes<32, kF16>(q, lo, hi);
    }
    __device__ static __forceinline__ float sub_scale(const Run& r, int g) {
        const unsigned w = g < 4 ? r.sc.x : r.sc.y;
        return half_bits(r.d) *
               static_cast<float>(static_cast<signed char>((w >> (8 * (g & 3))) & 0xffU));
    }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// A quad's 32-byte span in two steps. quad_span_load: lane q (bit e = q & 1, pair c = q >> 1)
// loads words 4e + 2c, 4e + 2c + 1 of the span, so one load instruction covers the span's sector
// exactly once. quad_span_words (a decoder's finish(), the whole warp): words q and q + 4 of the
// span, one word swapped with lane q ^ 1.
__device__ __forceinline__ uint2 quad_span_load(const std::uint8_t* span, int q) {
    return *reinterpret_cast<const uint2*>(span + 16 * (q & 1) + 8 * (q >> 1));
}
__device__ __forceinline__ uint2 quad_span_words(uint2 v) {
    unsigned lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    const bool odd    = (lane & 1U) != 0; // the lane's quad position q, as q & 1
    const unsigned in = __shfl_xor_sync(0xffffffffU, odd ? v.x : v.y, 1);
    return odd ? make_uint2(in, v.y) : make_uint2(v.x, in);
}

// Q6_K with wide loads (rmma_kernel's wide variant; the launch picks it, rmma_wide_pays). The
// plain decoder's four-byte code loads at a 16-byte stride take every sector of the step's ql and
// qh bytes twice (ncu, RTX 4090, Gemma 4 12B ffn_down at T = 6: L1 hit 48 %, a third of the
// stalls lg_throttle); here three eight-byte loads per lane read each sector once and finish()
// swaps one word a span inside the quad, at the step's MMAs (mma_step, after the step's
// activation loads issue: the swap waits for the weights). The finished run is the plain one.
struct GgmlQ6KWideDecoder : GgmlQ6KDecoder {
    static constexpr bool kFinish = true;
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int p            = k0 & 255;
        const int h            = p >> 7;
        const int qd           = (p >> 5) & 3;
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 8);
        const std::uint8_t* cb = a.codes + blk * 192;
        const uint2 ql0        = quad_span_load(cb + 64 * h, qd);
        const uint2 ql1        = quad_span_load(cb + 64 * h + 32, qd);
        Run r;
        r.ql = make_uint4(ql0.x, ql0.y, ql1.x, ql1.y);
        r.qh = quad_span_load(cb + 128 + 32 * h, qd);
        r.sc = *reinterpret_cast<const uint2*>(a.high + blk * 16 + 8 * h);
        r.d  = reinterpret_cast<const unsigned short*>(a.scales)[blk];
        return r;
    }
    // zero() has every word alike, so finish() keeps it.
    __device__ static __forceinline__ Run finish(Run r) {
        const uint2 ql0 = quad_span_words(make_uint2(r.ql.x, r.ql.y));
        const uint2 ql1 = quad_span_words(make_uint2(r.ql.z, r.ql.w));
        r.ql            = make_uint4(ql0.x, ql0.y, ql1.x, ql1.y);
        r.qh            = quad_span_words(r.qh);
        return r;
    }
};

// Q3_K planes: per superblock 32 hmask bytes then 64 qs bytes (codes), 12 packed 6-bit scale
// bytes (high plane), binary16 d (scale plane). Value v of half h: quarter jj = v / 32,
// l = v % 32: bits 2*jj of qs[32h + l] minus 4 unless bit 4h + jj of hmask[l] is set; scale
// s = 8h + 2*jj + l/16 (low 4 bits from byte s%8 >> 4*(s/8), high 2 from byte 8 + s%4 >> 2*(s/4));
// w = d*(sc - 32)*q, q in [-4, 3]. MMA j is quarter j / 2, word j % 2; its scale is s = 8h + j.
// As with Q4_K, load_raw() only loads and unpack() cuts out the half's scales and hmask bits.
struct GgmlQ3KDecoder {
    static constexpr bool kF16       = kF16Operands;
    static constexpr int kRun        = 32;
    static constexpr bool kStepScale = true;
    static constexpr int kSplits     = 8;
    static constexpr bool kStageActs = true;
    struct Run {
        uint2 qs;    // word u of qs bytes 32h .. 32h + 31
        uint2 hm;    // word u of the hmask bytes, shifted so bit jj is quarter jj of half h
        uint2 sc;    // 6-bit scales 8h .. 8h + 7 (value sc - 32); load_raw(): scale words 0, 1
        unsigned s2; // load_raw(): scale word 2 (the scales' top bits) until unpack()
        unsigned d;  // binary16 d
    };
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int p            = k0 & 255;
        const int h            = p >> 7;
        const int qd           = (p >> 5) & 3;
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 8);
        const std::uint8_t* cb = a.codes + blk * 96;
        const auto* qs         = reinterpret_cast<const unsigned*>(cb + 32 + 32 * h + 4 * qd);
        const auto* hm         = reinterpret_cast<const unsigned*>(cb + 4 * qd);
        const auto* sw         = reinterpret_cast<const unsigned*>(a.high + blk * 12);
        const unsigned s0 = sw[0], s1 = sw[1], s2 = sw[2];
        Run r;
        r.qs = make_uint2(qs[0], qs[4]);
        r.hm = make_uint2(hm[0] >> (4 * h), hm[4] >> (4 * h));
        r.sc = make_uint2(((s0 >> (4 * h)) & 0x0f0f0f0fU) | (((s2 >> (4 * h)) & 0x03030303U) << 4),
                          ((s1 >> (4 * h)) & 0x0f0f0f0fU) |
                              (((s2 >> (4 * h + 2)) & 0x03030303U) << 4));
        r.s2 = 0;
        r.d  = reinterpret_cast<const unsigned short*>(a.scales)[blk];
        return r;
    }
    __device__ static __forceinline__ Run load_raw(const GemmArgs& a, int row, int k0) {
        const int p            = k0 & 255;
        const int h            = p >> 7;
        const int qd           = (p >> 5) & 3;
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 8);
        const std::uint8_t* cb = a.codes + blk * 96;
        const auto* qs         = reinterpret_cast<const unsigned*>(cb + 32 + 32 * h + 4 * qd);
        const auto* hm         = reinterpret_cast<const unsigned*>(cb + 4 * qd);
        const auto* sw         = reinterpret_cast<const unsigned*>(a.high + blk * 12);
        Run r;
        r.qs = make_uint2(qs[0], qs[4]);
        r.hm = make_uint2(hm[0], hm[4]);
        r.sc = make_uint2(sw[0], sw[1]);
        r.s2 = sw[2];
        r.d  = reinterpret_cast<const unsigned short*>(a.scales)[blk];
        return r;
    }
    __device__ static __forceinline__ void unpack(Run& r, int step_k0) {
        const int h       = (step_k0 >> 7) & 1;
        const unsigned s0 = r.sc.x, s1 = r.sc.y, s2 = r.s2;
        r.hm = make_uint2(r.hm.x >> (4 * h), r.hm.y >> (4 * h));
        r.sc = make_uint2(((s0 >> (4 * h)) & 0x0f0f0f0fU) | (((s2 >> (4 * h)) & 0x03030303U) << 4),
                          ((s1 >> (4 * h)) & 0x0f0f0f0fU) |
                              (((s2 >> (4 * h + 2)) & 0x03030303U) << 4));
    }
    // Codes q + 4 = 4 (hmask bits set): value 0. A zero run is never unpacked into a stored
    // value: it stands for rows past the call's (rmma_kernel stores none of them).
    __device__ static __forceinline__ Run zero() {
        return {make_uint2(0, 0), make_uint2(0xffffffffU, 0xffffffffU),
                make_uint2(0x20202020U, 0x20202020U), 0, 0};
    }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const int jj     = j >> 1;
        const unsigned q = ((((j & 1) ? r.qs.y : r.qs.x) >> (2 * jj)) & 0x03030303U) |
                           (((((j & 1) ? r.hm.y : r.hm.x) >> jj) & 0x01010101U) << 2);
        byte_codes<4, kF16>(q, lo, hi);
    }
    __device__ static __forceinline__ float sub_scale(const Run& r, int g) {
        return half_bits(r.d) * (byte_of(g < 4 ? r.sc.x : r.sc.y, g & 3) - 32.0F);
    }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// Q2_K planes: per superblock 64 qs bytes (codes), 16 scale bytes (high plane: low nibble sc,
// high nibble m), binary16 d and dmin (scale plane). Value v of half h: jj = v / 32, l = v % 32:
// bits 2*jj of qs[32h + l]; scale byte 8h + 2*jj + l/16; w = d*sc*q - dmin*m. MMA j is quarter
// j / 2, word j % 2; its scale byte is 8h + j.
struct GgmlQ2KDecoder {
    static constexpr bool kF16       = kF16Operands;
    static constexpr int kRun        = 32;
    static constexpr bool kStepScale = true;
    static constexpr int kSplits     = 8;
    static constexpr bool kMinTerm   = true;
    struct Run {
        uint2 qs;    // word u of qs bytes 32h .. 32h + 31
        uint2 sc;    // scale bytes 8h .. 8h + 7
        unsigned dd; // binary16 d (low half) and dmin (high half)
    };
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int p            = k0 & 255;
        const int h            = p >> 7;
        const int qd           = (p >> 5) & 3;
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 8);
        const auto* qs = reinterpret_cast<const unsigned*>(a.codes + blk * 64 + 32 * h + 4 * qd);
        Run r;
        r.qs = make_uint2(qs[0], qs[4]);
        r.sc = *reinterpret_cast<const uint2*>(a.high + blk * 16 + 8 * h);
        r.dd = reinterpret_cast<const unsigned*>(a.scales)[blk];
        return r;
    }
    __device__ static __forceinline__ Run zero() { return {make_uint2(0, 0), make_uint2(0, 0), 0}; }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        byte_codes<0, kF16>((((j & 1) ? r.qs.y : r.qs.x) >> (2 * (j >> 1))) & 0x03030303U, lo, hi);
    }
    __device__ static __forceinline__ float sub_scale(const Run& r, int g) {
        const unsigned w = g < 4 ? r.sc.x : r.sc.y;
        return half_bits(r.dd) * static_cast<float>((w >> (8 * (g & 3))) & 0xfU);
    }
    __device__ static __forceinline__ float sub_min(const Run& r, int g) {
        const unsigned w = g < 4 ? r.sc.x : r.sc.y;
        return half_bits(r.dd >> 16) * static_cast<float>((w >> (8 * (g & 3) + 4)) & 0xfU);
    }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

using GgmlQ4KDecoder = GgmlQ45KDecoder<4>;
using GgmlQ5KDecoder = GgmlQ45KDecoder<5>;

// ---------------------------------------------------------------------------------------------
// ggml's 32-value block formats and i-quants. Every value they hold is an integer exact in BF16
// (a code, a codebook entry, a grid magnitude with its sign) times a per-block (or per-32-value
// sub-block) scale, so they run as spread decoders with one scale group per 32 values.
// ---------------------------------------------------------------------------------------------

// Four 4-bit codes (bytes of q, each < 16) looked up in the kvalues_iq4nl codebook as BF16 pairs
// of exact integers: lo holds codes 0, 1 and hi codes 2, 3. The codebook's BF16 words sit as
// their low and high bytes in eight constant words: byte permutes pick entries 0..7 and 8..15 of
// each, and a third permute takes the half each code's bit 3 names.
__device__ __forceinline__ void iq4nl_bf16(unsigned q, unsigned& lo, unsigned& hi) {
    using namespace ggml_iq;
    const unsigned t   = q | (q >> 4);
    const unsigned c   = __byte_perm(t, 0U, 0x4420U); // nibble n: code n
    const unsigned s   = c & 0x7777U;
    const unsigned sel = ((c & 0x8888U) >> 1) | 0x3210U;
    const unsigned l   = __byte_perm(__byte_perm(kIq4nlLow0, kIq4nlLow1, s),
                                     __byte_perm(kIq4nlLow2, kIq4nlLow3, s), sel);
    const unsigned h   = __byte_perm(__byte_perm(kIq4nlHigh0, kIq4nlHigh1, s),
                                     __byte_perm(kIq4nlHigh2, kIq4nlHigh3, s), sel);
    lo = __byte_perm(l, h, 0x5140U);
    hi = __byte_perm(l, h, 0x7362U);
}

// Four sign bits (nib bit i: value i negative) applied to the BF16 pairs of a spread MMA.
__device__ __forceinline__ void apply_signs(unsigned nib, unsigned& lo, unsigned& hi) {
    const unsigned e = ((nib * 0x00204081U) & 0x01010101U) << 7; // byte i: 0x80 where set
    lo ^= __byte_perm(e, 0U, 0x1404U);
    hi ^= __byte_perm(e, 0U, 0x3424U);
}

// Byte `b` of each of four words, packed into one word (word i's byte in byte i).
__device__ __forceinline__ unsigned pick_bytes(unsigned w0, unsigned w1, unsigned w2, unsigned w3,
                                               unsigned b) {
    const unsigned sel = b | ((b + 4U) << 4);
    return __byte_perm(__byte_perm(w0, w1, sel), __byte_perm(w2, w3, sel), 0x5410U);
}

// Q4_0, Q4_1, Q5_0, Q5_1 and IQ4_NL planes: per 32-value block its 16 qs bytes (codes; value j
// is the low nibble of qs[j], value j + 16 the high nibble), Q5's qh word (high plane: bit j is
// value j's fifth bit) and binary16 d (Q4_1 / Q5_1: d, m; scale plane). A K step (128 values) is
// four blocks, each one scale group: MMA j is block j / 2, nibble j % 2. Q4_0 w = d*(q - 8),
// Q5_0 w = d*(q - 16), IQ4_NL w = d*kvalues_iq4nl[q]; Q4_1 / Q5_1 w = d*q + m, the min term.
enum class Ggml32 { Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL };
template <Ggml32 F>
struct Ggml32Decoder {
    static constexpr bool kF16       = kF16Operands && F != Ggml32::IQ4_NL;
    static constexpr bool kFive      = F == Ggml32::Q5_0 || F == Ggml32::Q5_1;
    static constexpr bool kWithMin   = F == Ggml32::Q4_1 || F == Ggml32::Q5_1;
    static constexpr int kRun        = 32;
    static constexpr bool kStepScale = true;
    static constexpr int kSplits     = 4;
    static constexpr bool kMinTerm   = kWithMin;
    struct Run {
        uint4 qs; // word b: qs bytes 4qd .. 4qd + 3 of the step's block b
        uint4 qh; // Q5: block b's qh word shifted right by 4qd
        uint4 dm; // _1: word b is block b's d (low half) and m; else .x, .y hold the four d
    };
    // K need only be whole blocks: a step past the row's end reads its blocks as zero and
    // mma_step skips their groups (kGroupTail).
    static constexpr bool kGroupTail = true;
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int qd    = (k0 >> 5) & 3;
        const int first = (k0 >> 5) & ~3;
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + first;
        if ((a.groups_per_row & 3) != 0) return load_blocks(a, blk, qd, a.groups_per_row - first);
        const auto* c = reinterpret_cast<const unsigned*>(a.codes + blk * 16 + 4 * qd);
        Run r;
        r.qs = make_uint4(c[0], c[4], c[8], c[12]);
        r.qh = make_uint4(0, 0, 0, 0);
        if constexpr (kFive) {
            const uint4 h = *reinterpret_cast<const uint4*>(a.high + blk * 4);
            r.qh = make_uint4(h.x >> (4 * qd), h.y >> (4 * qd), h.z >> (4 * qd), h.w >> (4 * qd));
        }
        if constexpr (kWithMin) {
            r.dm = *reinterpret_cast<const uint4*>(a.scales + blk * 4);
        } else {
            const uint2 d = *reinterpret_cast<const uint2*>(a.scales + blk * 2);
            r.dm          = make_uint4(d.x, d.y, 0, 0);
        }
        return r;
    }
    // d = 0 (and m = 0): value 0.
    __device__ static __forceinline__ Run zero() {
        return {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    }
    // K not whole 128-value steps: a row's first block is not 4-aligned, so the planes are read
    // block by block; the `valid` blocks left in the row are read, the rest are zero.
    __device__ static __forceinline__ Run load_blocks(const GemmArgs& a, std::int64_t blk, int qd,
                                                      int valid) {
        unsigned qs[4] = {0, 0, 0, 0}, qh[4] = {0, 0, 0, 0}, dm[4] = {0, 0, 0, 0};
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            if (b < valid) {
                qs[b] = *reinterpret_cast<const unsigned*>(a.codes + (blk + b) * 16 + 4 * qd);
                if constexpr (kFive) {
                    qh[b] = *reinterpret_cast<const unsigned*>(a.high + (blk + b) * 4) >> (4 * qd);
                }
                if constexpr (kWithMin) {
                    dm[b] = *reinterpret_cast<const unsigned*>(a.scales + (blk + b) * 4);
                } else {
                    dm[b] = *reinterpret_cast<const unsigned short*>(a.scales + (blk + b) * 2);
                }
            }
        }
        Run r;
        r.qs = make_uint4(qs[0], qs[1], qs[2], qs[3]);
        r.qh = make_uint4(qh[0], qh[1], qh[2], qh[3]);
        r.dm = kWithMin ? make_uint4(dm[0], dm[1], dm[2], dm[3])
                        : make_uint4(dm[0] | (dm[1] << 16), dm[2] | (dm[3] << 16), 0, 0);
        return r;
    }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const int g = j >> 1;
        unsigned q  = (uint4_word(r.qs, g) >> (4 * (j & 1))) & 0x0f0f0f0fU;
        if constexpr (kFive) {
            const unsigned b = (uint4_word(r.qh, g) >> (16 * (j & 1))) & 0xfU;
            q |= ((b * 0x00204081U) & 0x01010101U) << 4;
        }
        if constexpr (F == Ggml32::IQ4_NL) {
            iq4nl_bf16(q, lo, hi);
        } else {
            byte_codes<F == Ggml32::Q4_0 ? 8 : F == Ggml32::Q5_0 ? 16 : 0, kF16>(q, lo, hi);
        }
    }
    __device__ static __forceinline__ float sub_scale(const Run& r, int g) {
        if constexpr (kWithMin) {
            return half_bits(uint4_word(r.dm, g));
        } else {
            return half_bits((g < 2 ? r.dm.x : r.dm.y) >> (16 * (g & 1)));
        }
    }
    // The min term subtracts sub_min times the group's activation sum: -m adds m.
    __device__ static __forceinline__ float sub_min(const Run& r, int g) {
        return -half_bits(uint4_word(r.dm, g) >> 16);
    }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

using GgmlQ40Decoder   = Ggml32Decoder<Ggml32::Q4_0>;
using GgmlQ41Decoder   = Ggml32Decoder<Ggml32::Q4_1>;
using GgmlQ50Decoder   = Ggml32Decoder<Ggml32::Q5_0>;
using GgmlQ51Decoder   = Ggml32Decoder<Ggml32::Q5_1>;
using GgmlIq4NlDecoder = Ggml32Decoder<Ggml32::IQ4_NL>;

// MXFP4 without an FP4 converter (the Ada build: sm_89 emulates cvt e2m1x2 in a long integer
// sequence, which bounds the decode at a quarter of the memory bandwidth). A spread quad-run
// decoder: a step is four 32-value exponent groups, quad q holding group g's values 8q .. 8q + 7
// (code bytes 4q .. 4q + 3, value 2i in byte i's low nibble) and MMA 2g + h their half h. The
// doubled codes are looked up by byte permutes, and a group's FP32 partial takes 2^(e - 128). K
// need only be whole groups: a step past the row's end reads its groups as zero and mma_step
// stops before them (kGroupTail).
struct Mxfp4SpreadDecoder {
    static constexpr int kRun        = 32;
    static constexpr bool kStepScale = true;
    static constexpr int kSplits     = 4;
    static constexpr bool kQuadRuns  = true;
    static constexpr bool kGroupTail = true;
    struct Run {
        uint4 codes;   // word g: the lane's code bytes 4q .. 4q + 3 of the step's group g
        unsigned exps; // byte g: group g's E8M0 exponent
    };
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int q      = (k0 >> 5) & 3;
        const int first  = (k0 >> 5) & ~3;
        const int groups = a.k >> 5;
        const int valid  = groups - first;
        const std::uint8_t* p = a.codes + static_cast<std::int64_t>(row) * (a.k / 2) + first * 16;
        const std::uint8_t* e = a.scales + static_cast<std::int64_t>(row) * groups + first;
        Run r;
        if (valid >= 4) {
            r.codes = make_uint4(ld_stream_u32(p + 4 * q), ld_stream_u32(p + 16 + 4 * q),
                                 ld_stream_u32(p + 32 + 4 * q), ld_stream_u32(p + 48 + 4 * q));
            if ((groups & 3) == 0) {
                r.exps = *reinterpret_cast<const unsigned*>(e);
                return r;
            }
        } else {
            unsigned w[4] = {0, 0, 0, 0};
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                if (b < valid) w[b] = ld_stream_u32(p + 16 * b + 4 * q);
            }
            r.codes = make_uint4(w[0], w[1], w[2], w[3]);
        }
        r.exps = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            if (b < valid) r.exps |= static_cast<unsigned>(e[b]) << (8 * b);
        }
        return r;
    }
    __device__ static __forceinline__ Run zero() { return {make_uint4(0, 0, 0, 0), 0}; }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const unsigned w = uint4_word(r.codes, j >> 1);
        e2m1x2_bf16((j & 1) ? (w >> 16) : (w & 0xffffU), lo, hi);
    }
    __device__ static __forceinline__ float sub_scale(const Run& r, int g) {
        return 0.5F * e8m0_value((r.exps >> (8 * g)) & 0xffU);
    }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// The MXFP4 register-MMA decoder of this build: the FP4 converter's where there is one.
#ifdef NINFERNO_WITHOUT_NVFP4
using Mxfp4RmmaDecoder = Mxfp4SpreadDecoder;
#else
using Mxfp4RmmaDecoder = Mxfp4Decoder;
#endif

// IQ4_XS planes: per superblock its 128 qs bytes (codes; sub-block ib of 32 values is qs bytes
// 16ib .. 16ib + 15, low nibbles first), scales_l (4 bytes, high plane) and d, scales_h (scale
// plane). Sub-block ib: ls = nibble ib of scales_l | bits 2ib of scales_h << 4; w =
// d*(ls - 32)*kvalues_iq4nl[q]. A K step is half h of a superblock: sub-blocks 4h .. 4h + 3,
// one scale group each; MMA j is sub-block 4h + j / 2, nibble j % 2.
struct GgmlIq4XsDecoder {
    static constexpr int kRun        = 32;
    static constexpr bool kStepScale = true;
    static constexpr int kSplits     = 4;
    struct Run {
        uint4 qs;     // word g: qs bytes 16(4h + g) + 4qd .. + 3
        unsigned sl;  // scales_l >> 16h: nibble g is sub-block 4h + g's low four bits
        unsigned dsh; // d (low half), scales_h >> 8h (high half: bits 2g are sub-block 4h + g's)
    };
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int p            = k0 & 255;
        const int h            = p >> 7;
        const int qd           = (p >> 5) & 3;
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 8);
        const auto* c = reinterpret_cast<const unsigned*>(a.codes + blk * 128 + 64 * h + 4 * qd);
        const unsigned w = reinterpret_cast<const unsigned*>(a.scales)[blk];
        Run r;
        r.qs  = make_uint4(c[0], c[4], c[8], c[12]);
        r.sl  = reinterpret_cast<const unsigned*>(a.high)[blk] >> (16 * h);
        r.dsh = (w & 0xffffU) | (((w >> 16) >> (8 * h)) << 16);
        return r;
    }
    __device__ static __forceinline__ Run zero() { return {make_uint4(0, 0, 0, 0), 0, 0}; }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        iq4nl_bf16((uint4_word(r.qs, j >> 1) >> (4 * (j & 1))) & 0x0f0f0f0fU, lo, hi);
    }
    __device__ static __forceinline__ float sub_scale(const Run& r, int g) {
        const int ls = static_cast<int>(((r.sl >> (4 * g)) & 0xfU) |
                                        (((r.dsh >> (16 + 2 * g)) & 3U) << 4));
        return half_bits(r.dsh) * static_cast<float>(ls - 32);
    }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// IQ3_XXS planes: per superblock its 64 qs bytes (codes: grid index of values 4i .. 4i + 3),
// scales_and_signs (32 bytes, high plane: per 32-value sub-block one word, four 7-bit indices
// into ksigns_iq2xs for its 8-value groups and a 4-bit ls on top) and d (scale plane).
// w = d*(0.5 + ls)*0.5*iq3xxs_grid[index]*sign; ksigns_iq2xs[i] is i with its parity in bit 7.
// The grid is staged in shared memory. A K step is half h; MMA j takes grid entry
// 32h + 4j + qd (one index byte), in sub-block 4h + j / 2.
struct GgmlIq3XxsDecoder {
    static constexpr bool kF16       = kF16Operands;
    static constexpr int kRun        = 32;
    static constexpr bool kStepScale = true;
    static constexpr int kSplits     = 4;
    static constexpr int kTableWords = 256;
    struct Run {
        uint2 idx;    // byte j: MMA j's grid index
        uint4 ss;     // word g: sub-block 4h + g's scales_and_signs >> 7*(qd / 2)
        unsigned dsc; // d (low half), the four sub-blocks' ls (nibbles of the high half)
    };
    __device__ static __forceinline__ unsigned* table() {
        __shared__ unsigned grid[kTableWords];
        return grid;
    }
    __device__ static __forceinline__ void stage() {
        for (int i = static_cast<int>(threadIdx.x); i < kTableWords;
             i += static_cast<int>(blockDim.x))
            table()[i] = ggml_iq::kIq3xxsGrid[i];
    }
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int p            = k0 & 255;
        const int h            = p >> 7;
        const unsigned qd      = static_cast<unsigned>(p >> 5) & 3U;
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 8);
        const auto* q          = reinterpret_cast<const uint4*>(a.codes + blk * 64 + 32 * h);
        const uint4 w0 = q[0], w1 = q[1];
        const uint4 s4 = *reinterpret_cast<const uint4*>(a.high + blk * 32 + 16 * h);
        const unsigned shift = 7U * (qd >> 1);
        const unsigned ls    = (s4.x >> 28) | ((s4.y >> 28) << 4) | ((s4.z >> 28) << 8) |
                            ((s4.w >> 28) << 12);
        Run r;
        r.idx = make_uint2(pick_bytes(w0.x, w0.y, w0.z, w0.w, qd),
                           pick_bytes(w1.x, w1.y, w1.z, w1.w, qd));
        r.ss  = make_uint4(s4.x >> shift, s4.y >> shift, s4.z >> shift, s4.w >> shift);
        r.dsc = static_cast<unsigned>(reinterpret_cast<const unsigned short*>(a.scales)[blk]) |
                (ls << 16);
        return r;
    }
    __device__ static __forceinline__ Run zero() {
        return {make_uint2(0, 0), make_uint4(0, 0, 0, 0), 0};
    }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const unsigned idx = ((j < 4 ? r.idx.x : r.idx.y) >> (8 * (j & 3))) & 0xffU;
        byte_codes<0, kF16>(table()[idx], lo, hi);
        const unsigned t  = (uint4_word(r.ss, j >> 1) >> (14 * (j & 1))) & 127U;
        const unsigned s8 = t | ((static_cast<unsigned>(__popc(t)) & 1U) << 7);
        apply_signs((s8 >> (4U * (threadIdx.x & 1U))) & 0xfU, lo, hi);
    }
    __device__ static __forceinline__ float sub_scale(const Run& r, int g) {
        const auto ls = static_cast<float>((r.dsc >> (16 + 4 * g)) & 0xfU);
        return half_bits(r.dsc) * (0.5F + ls) * 0.5F;
    }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// IQ3_S planes: per superblock its 104 code bytes (qs[64]: low eight bits of the grid index of
// values 4i .. 4i + 3; qh[8]: bit i is that index's ninth bit; signs[32]: bit v of the bytes is
// value v's sign), scales (4 bytes, high plane: a 4-bit ls per 32-value sub-block) and d (scale
// plane). w = d*(1 + 2*ls)*iq3s_grid[index]*sign; the grid is staged in shared memory. A K step
// is half h; MMA j takes grid entry 32h + 4j + qd, in sub-block 4h + j / 2.
struct GgmlIq3SDecoder {
    static constexpr bool kF16       = kF16Operands;
    static constexpr int kRun        = 32;
    static constexpr bool kStepScale = true;
    static constexpr int kSplits     = 4;
    static constexpr int kTableWords = 512;
    struct Run {
        uint2 idx;    // byte j: the low eight bits of MMA j's grid index
        unsigned qh;  // qh bytes 4h .. 4h + 3 >> qd: bit 4j is MMA j's ninth index bit
        uint4 sg;     // sign bytes 16h .. + 15 >> 4qd: bits 16(j % 2) of word j / 2 are MMA j's
        unsigned dsc; // d (low half), scales >> 16h (high half: nibble g is sub-block 4h + g's ls)
    };
    __device__ static __forceinline__ unsigned* table() {
        __shared__ unsigned grid[kTableWords];
        return grid;
    }
    __device__ static __forceinline__ void stage() {
        for (int i = static_cast<int>(threadIdx.x); i < kTableWords;
             i += static_cast<int>(blockDim.x))
            table()[i] = ggml_iq::kIq3sGrid[i];
    }
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int p            = k0 & 255;
        const int h            = p >> 7;
        const unsigned qd      = static_cast<unsigned>(p >> 5) & 3U;
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 8);
        const std::uint8_t* cb = a.codes + blk * 104;
        const auto* q          = reinterpret_cast<const uint2*>(cb + 32 * h);
        const uint2 q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
        const auto* sg = reinterpret_cast<const uint2*>(cb + 72 + 16 * h);
        const uint2 s0 = sg[0], s1 = sg[1];
        const unsigned sh = 4U * qd;
        Run r;
        r.idx = make_uint2(pick_bytes(q0.x, q0.y, q1.x, q1.y, qd),
                           pick_bytes(q2.x, q2.y, q3.x, q3.y, qd));
        r.qh  = *reinterpret_cast<const unsigned*>(cb + 64 + 4 * h) >> qd;
        r.sg  = make_uint4(s0.x >> sh, s0.y >> sh, s1.x >> sh, s1.y >> sh);
        r.dsc = static_cast<unsigned>(reinterpret_cast<const unsigned short*>(a.scales)[blk]) |
                ((reinterpret_cast<const unsigned*>(a.high)[blk] >> (16 * h)) << 16);
        return r;
    }
    __device__ static __forceinline__ Run zero() {
        return {make_uint2(0, 0), 0, make_uint4(0, 0, 0, 0), 0};
    }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const unsigned idx = (((j < 4 ? r.idx.x : r.idx.y) >> (8 * (j & 3))) & 0xffU) |
                             (((r.qh >> (4 * j)) & 1U) << 8);
        byte_codes<0, kF16>(table()[idx], lo, hi);
        apply_signs((uint4_word(r.sg, j >> 1) >> (16 * (j & 1))) & 0xfU, lo, hi);
    }
    __device__ static __forceinline__ float sub_scale(const Run& r, int g) {
        return half_bits(r.dsc) *
               static_cast<float>(1 + 2 * static_cast<int>((r.dsc >> (16 + 4 * g)) & 0xfU));
    }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// Four 2-bit codes (nibbles of sel, each < 4) looked up as BF16 words in a four-entry table held
// as the low and high bytes of two words (byte c: entry c): lo holds codes 0, 1 and hi 2, 3.
__device__ __forceinline__ void lut4_bf16(unsigned sel, unsigned low, unsigned high, unsigned& lo,
                                          unsigned& hi) {
    const unsigned l = __byte_perm(low, 0U, sel);
    const unsigned h = __byte_perm(high, 0U, sel);
    lo = __byte_perm(l, h, 0x5140U);
    hi = __byte_perm(l, h, 0x7362U);
}

// IQ1_S / IQ1_M operands: grid value + delta (grid -1, 0, 1 as codes 0, 1, 2; delta +-1/8), all
// exact in BF16: delta +1/8 gives -0.875, 0.125, 1.125; delta -1/8 gives -1.125, -0.125, 0.875.
constexpr unsigned kIq1PlusLow   = 0x00900060U;
constexpr unsigned kIq1PlusHigh  = 0x003f3ebfU;
constexpr unsigned kIq1MinusLow  = 0x00600090U;
constexpr unsigned kIq1MinusHigh = 0x003fbebfU;

// Four 8-bit codes (bytes of w, each < 3) as base-3 digits: digit n of every byte, the way ggml's
// TQ1_0 reads it (q = byte * 3^n mod 256, digit = q*3 >> 8), two bytes per 16-bit lane.
__device__ __forceinline__ unsigned trits(unsigned w, unsigned pow3) {
    const unsigned even = ((w & 0x00ff00ffU) * pow3) & 0x00ff00ffU;
    const unsigned odd  = (((w >> 8) & 0x00ff00ffU) * pow3) & 0x00ff00ffU;
    return (((even * 3U) >> 8) & 0x00030003U) | ((((odd * 3U) >> 8) & 0x00030003U) << 8);
}

// Bytes b and b + 2 of words x and y, packed (x's in bytes 0, 1, y's in bytes 2, 3).
__device__ __forceinline__ unsigned pick_pairs(unsigned x, unsigned y, unsigned b) {
    return __byte_perm(x, y, b | ((b + 2U) << 4) | ((b + 4U) << 8) | ((b + 6U) << 12));
}

// Half-words b of words x and y, packed (x's low, y's high).
__device__ __forceinline__ unsigned pick_halves(unsigned x, unsigned y, unsigned b) {
    const unsigned s = 2U * b;
    return __byte_perm(x, y, s | ((s + 1U) << 4) | ((s + 4U) << 8) | ((s + 5U) << 12));
}

// The IQ2 decoders' common step: MMA j's lane takes word (lane % 2) of grid entry `index` (four
// magnitudes, bytes), with the sign bits of its four values (low nibble of `signs`).
template <class D>
__device__ __forceinline__ void iq2_pair(unsigned index, unsigned signs, unsigned& lo,
                                         unsigned& hi) {
    byte_codes<0, D::kF16>(D::table()[2U * index + (threadIdx.x & 1U)], lo, hi);
    apply_signs(signs & 0xfU, lo, hi);
}

// IQ2_XXS planes: per superblock its 64 qs bytes (codes; per 32-value sub-block two words: the
// grid indices of its four 8-value groups, one byte each, then four 7-bit indices into
// ksigns_iq2xs and a 4-bit ls on top) and d (scale plane). w = d*(0.5 + ls)*0.25*grid*sign. The
// grid (eight magnitudes per entry, two words) is staged in shared memory. A K step is half h;
// MMA j's lane qd takes group 2(j % 2) + qd / 2 of sub-block 4h + j / 2, word qd % 2 of it.
struct GgmlIq2XxsDecoder {
    static constexpr bool kF16       = kF16Operands;
    static constexpr int kRun        = 32;
    static constexpr bool kStepScale = true;
    static constexpr int kSplits     = 4;
    static constexpr int kTableWords = 512;
    struct Run {
        uint2 idx;    // byte j: MMA j's grid index
        uint4 ss;     // word g: sub-block 4h + g's sign word >> 7*(qd / 2)
        unsigned dsc; // d (low half), the four sub-blocks' ls (nibbles of the high half)
    };
    __device__ static __forceinline__ unsigned* table() {
        __shared__ unsigned grid[kTableWords];
        return grid;
    }
    __device__ static __forceinline__ void stage() {
        for (int i = static_cast<int>(threadIdx.x); i < kTableWords;
             i += static_cast<int>(blockDim.x))
            table()[i] = ggml_iq::kIq2xxsGrid[i];
    }
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int p            = k0 & 255;
        const int h            = p >> 7;
        const unsigned b       = static_cast<unsigned>(p >> 6) & 1U; // qd / 2
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 8);
        const auto* q          = reinterpret_cast<const uint4*>(a.codes + blk * 64 + 32 * h);
        const uint4 w0 = q[0], w1 = q[1];
        const unsigned shift = 7U * b;
        Run r;
        r.idx = make_uint2(pick_pairs(w0.x, w0.z, b), pick_pairs(w1.x, w1.z, b));
        r.ss  = make_uint4(w0.y >> shift, w0.w >> shift, w1.y >> shift, w1.w >> shift);
        r.dsc = static_cast<unsigned>(reinterpret_cast<const unsigned short*>(a.scales)[blk]) |
                (((w0.y >> 28) | ((w0.w >> 28) << 4) | ((w1.y >> 28) << 8) | ((w1.w >> 28) << 12))
                 << 16);
        return r;
    }
    __device__ static __forceinline__ Run zero() {
        return {make_uint2(0, 0), make_uint4(0, 0, 0, 0), 0};
    }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const unsigned idx = ((j < 4 ? r.idx.x : r.idx.y) >> (8 * (j & 3))) & 0xffU;
        const unsigned t   = (uint4_word(r.ss, j >> 1) >> (14 * (j & 1))) & 127U;
        const unsigned s8  = t | ((static_cast<unsigned>(__popc(t)) & 1U) << 7);
        iq2_pair<GgmlIq2XxsDecoder>(idx, s8 >> (4U * (threadIdx.x & 1U)), lo, hi);
    }
    __device__ static __forceinline__ float sub_scale(const Run& r, int g) {
        const auto ls = static_cast<float>((r.dsc >> (16 + 4 * g)) & 0xfU);
        return half_bits(r.dsc) * (0.5F + ls) * 0.25F;
    }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// IQ2_XS planes: per superblock its 64 qs bytes (codes; 32 16-bit entries, one per 8-value group:
// a 9-bit grid index and a 7-bit ksigns_iq2xs index on top), scales (8 bytes, high plane: per
// 32-value sub-block two 4-bit ls, one per 16 values) and d (scale plane). w = d*(0.5 + ls)*0.25*
// grid*sign; the grid is staged in shared memory. A K step is half h; MMA j's lane qd takes entry
// 16h + 2j + qd / 2, its 16 values one scale group.
struct GgmlIq2XsDecoder {
    static constexpr bool kF16       = kF16Operands;
    static constexpr int kRun        = 32;
    static constexpr bool kStepScale = true;
    static constexpr int kSplits     = 8;
    static constexpr int kTableWords = 1024;
    struct Run {
        uint4 q;     // half-word j % 2 of word j / 2: MMA j's entry
        unsigned sc; // scale bytes 4h .. 4h + 3: nibble j is MMA j's ls
        unsigned d;
    };
    __device__ static __forceinline__ unsigned* table() {
        __shared__ unsigned grid[kTableWords];
        return grid;
    }
    __device__ static __forceinline__ void stage() {
        for (int i = static_cast<int>(threadIdx.x); i < kTableWords;
             i += static_cast<int>(blockDim.x))
            table()[i] = ggml_iq::kIq2xsGrid[i];
    }
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int p            = k0 & 255;
        const int h            = p >> 7;
        const unsigned b       = static_cast<unsigned>(p >> 6) & 1U;
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 8);
        const auto* q          = reinterpret_cast<const uint4*>(a.codes + blk * 64 + 32 * h);
        const uint4 w0 = q[0], w1 = q[1];
        Run r;
        r.q  = make_uint4(pick_halves(w0.x, w0.y, b), pick_halves(w0.z, w0.w, b),
                          pick_halves(w1.x, w1.y, b), pick_halves(w1.z, w1.w, b));
        r.sc = *reinterpret_cast<const unsigned*>(a.high + blk * 8 + 4 * h);
        r.d  = reinterpret_cast<const unsigned short*>(a.scales)[blk];
        return r;
    }
    __device__ static __forceinline__ Run zero() { return {make_uint4(0, 0, 0, 0), 0, 0}; }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const unsigned e  = (uint4_word(r.q, j >> 1) >> (16 * (j & 1))) & 0xffffU;
        const unsigned t  = e >> 9;
        const unsigned s8 = t | ((static_cast<unsigned>(__popc(t)) & 1U) << 7);
        iq2_pair<GgmlIq2XsDecoder>(e & 511U, s8 >> (4U * (threadIdx.x & 1U)), lo, hi);
    }
    __device__ static __forceinline__ float sub_scale(const Run& r, int g) {
        const auto ls = static_cast<float>((r.sc >> (4 * g)) & 0xfU);
        return half_bits(r.d) * (0.5F + ls) * 0.25F;
    }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// IQ2_S planes: per superblock its 72 code bytes (qs[32]: low eight bits of the grid index of
// 8-value group i; signs[32]: sign bits of group i; qh[8]: bits 2l, 2l + 1 of byte ib are the
// ninth and tenth index bits of group 4ib + l), scales (8 bytes, high plane, as IQ2_XS) and d
// (scale plane). w = d*(0.5 + ls)*0.25*grid*sign; the grid is staged in shared memory. A K step
// is half h; MMA j's lane qd takes group 16h + 2j + qd / 2.
struct GgmlIq2SDecoder {
    static constexpr bool kF16       = kF16Operands;
    static constexpr int kRun        = 32;
    static constexpr bool kStepScale = true;
    static constexpr int kSplits     = 8;
    static constexpr int kTableWords = 2048;
    struct Run {
        uint2 idx;   // byte j: the low eight bits of MMA j's grid index
        uint2 sg;    // byte j: MMA j's sign bits
        unsigned qh; // qh bytes 4h .. 4h + 3 >> 2(qd / 2): bits 4j, 4j + 1 are MMA j's index bits 8, 9
        unsigned sc; // scale bytes 4h .. 4h + 3: nibble j is MMA j's ls
        unsigned d;
    };
    __device__ static __forceinline__ unsigned* table() {
        __shared__ unsigned grid[kTableWords];
        return grid;
    }
    __device__ static __forceinline__ void stage() {
        for (int i = static_cast<int>(threadIdx.x); i < kTableWords;
             i += static_cast<int>(blockDim.x))
            table()[i] = ggml_iq::kIq2sGrid[i];
    }
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int p            = k0 & 255;
        const int h            = p >> 7;
        const unsigned b       = static_cast<unsigned>(p >> 6) & 1U;
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 8);
        const std::uint8_t* cb = a.codes + blk * 72;
        const auto* q          = reinterpret_cast<const uint2*>(cb + 16 * h);
        const auto* s          = reinterpret_cast<const uint2*>(cb + 32 + 16 * h);
        const uint2 q0 = q[0], q1 = q[1], s0 = s[0], s1 = s[1];
        Run r;
        r.idx = make_uint2(pick_pairs(q0.x, q0.y, b), pick_pairs(q1.x, q1.y, b));
        r.sg  = make_uint2(pick_pairs(s0.x, s0.y, b), pick_pairs(s1.x, s1.y, b));
        r.qh  = *reinterpret_cast<const unsigned*>(cb + 64 + 4 * h) >> (2U * b);
        r.sc  = *reinterpret_cast<const unsigned*>(a.high + blk * 8 + 4 * h);
        r.d   = reinterpret_cast<const unsigned short*>(a.scales)[blk];
        return r;
    }
    __device__ static __forceinline__ Run zero() {
        return {make_uint2(0, 0), make_uint2(0, 0), 0, 0, 0};
    }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const unsigned idx = (((j < 4 ? r.idx.x : r.idx.y) >> (8 * (j & 3))) & 0xffU) |
                             (((r.qh >> (4 * j)) & 3U) << 8);
        const unsigned sg = ((j < 4 ? r.sg.x : r.sg.y) >> (8 * (j & 3))) & 0xffU;
        iq2_pair<GgmlIq2SDecoder>(idx, sg >> (4U * (threadIdx.x & 1U)), lo, hi);
    }
    __device__ static __forceinline__ float sub_scale(const Run& r, int g) {
        const auto ls = static_cast<float>((r.sc >> (4 * g)) & 0xfU);
        return half_bits(r.d) * (0.5F + ls) * 0.25F;
    }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// IQ1_S planes: per superblock its 32 qs bytes (codes: low eight bits of the grid index of
// 8-value group i), qh (16 bytes, high plane: per 32-value sub-block a 16-bit word, bits 3l ..
// 3l + 2 the high index bits of its group l, bits 12..14 ls, bit 15 the delta's sign) and d
// (scale plane). w = d*(2*ls + 1)*(grid + delta), grid -1, 0, 1 and delta +-1/8: the operand
// grid + delta is exact in BF16. The grid (a nibble per value) is staged in shared memory. A K
// step is half h; MMA j's lane qd takes group 2(j % 2) + qd / 2 of sub-block 4h + j / 2.
struct GgmlIq1SDecoder {
    static constexpr int kRun        = 32;
    static constexpr bool kStepScale = true;
    static constexpr int kSplits     = 4;
    static constexpr int kTableWords = 2048;
    struct Run {
        uint2 idx; // byte j: the low eight bits of MMA j's grid index
        uint2 qh;  // half-word g: sub-block 4h + g's qh word
        unsigned d;
    };
    __device__ static __forceinline__ unsigned* table() {
        __shared__ unsigned grid[kTableWords];
        return grid;
    }
    __device__ static __forceinline__ void stage() {
        for (int i = static_cast<int>(threadIdx.x); i < kTableWords;
             i += static_cast<int>(blockDim.x))
            table()[i] = ggml_iq::kIq1sGrid[i];
    }
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int p            = k0 & 255;
        const int h            = p >> 7;
        const unsigned b       = static_cast<unsigned>(p >> 6) & 1U;
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 8);
        const uint4 w = *reinterpret_cast<const uint4*>(a.codes + blk * 32 + 16 * h);
        Run r;
        r.idx = make_uint2(pick_pairs(w.x, w.y, b), pick_pairs(w.z, w.w, b));
        r.qh  = *reinterpret_cast<const uint2*>(a.high + blk * 16 + 8 * h);
        r.d   = reinterpret_cast<const unsigned short*>(a.scales)[blk];
        return r;
    }
    __device__ static __forceinline__ Run zero() { return {make_uint2(0, 0), make_uint2(0, 0), 0}; }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ unsigned qh_word(const Run& r, int g) {
        return ((g < 2 ? r.qh.x : r.qh.y) >> (16 * (g & 1))) & 0xffffU;
    }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const unsigned w   = qh_word(r, j >> 1);
        const unsigned l   = 2U * static_cast<unsigned>(j & 1) + ((threadIdx.x >> 1) & 1U);
        const unsigned idx = (((j < 4 ? r.idx.x : r.idx.y) >> (8 * (j & 3))) & 0xffU) |
                             (((w >> (3U * l)) & 7U) << 8);
        const unsigned sel = table()[idx] >> (16U * (threadIdx.x & 1U));
        const bool minus   = (w & 0x8000U) != 0;
        lut4_bf16(sel & 0x3333U, minus ? kIq1MinusLow : kIq1PlusLow,
                  minus ? kIq1MinusHigh : kIq1PlusHigh, lo, hi);
    }
    __device__ static __forceinline__ float sub_scale(const Run& r, int g) {
        return half_bits(r.d) *
               static_cast<float>(2 * static_cast<int>((qh_word(r, g) >> 12) & 7U) + 1);
    }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// IQ1_M planes: per superblock its 32 qs bytes (codes, as IQ1_S), qh (16 bytes, high plane: byte
// 2ib + l / 2, nibble l % 2 holds group 4ib + l's three high index bits and, in bit 3, its delta's
// sign) and scales (8 bytes, scale plane: four 16-bit words, 3-bit ls per 16 values in bits
// 0..11, the superblock's binary16 d in their top nibbles). w = d*(2*ls + 1)*(grid + delta). A
// K step is half h; MMA j's lane qd takes group 2(j % 2) + qd / 2 of sub-block 4h + j / 2 (qh
// byte 8h + j, nibble qd / 2), its 16 values one scale group.
struct GgmlIq1MDecoder {
    static constexpr int kRun        = 32;
    static constexpr bool kStepScale = true;
    static constexpr int kSplits     = 8;
    static constexpr int kTableWords = 2048;
    struct Run {
        uint2 idx;   // byte j: the low eight bits of MMA j's grid index
        uint2 qh;    // qh bytes 8h .. 8h + 7 >> 4(qd / 2): byte j's low nibble is MMA j's
        unsigned sc; // scale words 2h, 2h + 1: bits 16(j / 4) + 3(j % 4) are MMA j's ls
        unsigned d;  // binary16 d
    };
    __device__ static __forceinline__ unsigned* table() {
        __shared__ unsigned grid[kTableWords];
        return grid;
    }
    __device__ static __forceinline__ void stage() {
        for (int i = static_cast<int>(threadIdx.x); i < kTableWords;
             i += static_cast<int>(blockDim.x))
            table()[i] = ggml_iq::kIq1sGrid[i];
    }
    __device__ static __forceinline__ unsigned d_bits(uint2 sw) {
        return ((sw.x >> 12) & 0xfU) | ((sw.x >> 24) & 0xf0U) | ((sw.y >> 4) & 0xf00U) |
               ((sw.y >> 16) & 0xf000U);
    }
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int p            = k0 & 255;
        const int h            = p >> 7;
        const unsigned b       = static_cast<unsigned>(p >> 6) & 1U;
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 8);
        const uint4 w  = *reinterpret_cast<const uint4*>(a.codes + blk * 32 + 16 * h);
        const uint2 qh = *reinterpret_cast<const uint2*>(a.high + blk * 16 + 8 * h);
        const uint2 sw = *reinterpret_cast<const uint2*>(a.scales + blk * 8);
        Run r;
        r.idx = make_uint2(pick_pairs(w.x, w.y, b), pick_pairs(w.z, w.w, b));
        r.qh  = make_uint2(qh.x >> (4U * b), qh.y >> (4U * b));
        r.sc  = h ? sw.y : sw.x;
        r.d   = d_bits(sw);
        return r;
    }
    __device__ static __forceinline__ Run zero() {
        return {make_uint2(0, 0), make_uint2(0, 0), 0, 0};
    }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const unsigned n   = ((j < 4 ? r.qh.x : r.qh.y) >> (8 * (j & 3))) & 0xfU;
        const unsigned idx = (((j < 4 ? r.idx.x : r.idx.y) >> (8 * (j & 3))) & 0xffU) |
                             ((n & 7U) << 8);
        const unsigned sel = table()[idx] >> (16U * (threadIdx.x & 1U));
        const bool minus   = (n & 8U) != 0;
        lut4_bf16(sel & 0x3333U, minus ? kIq1MinusLow : kIq1PlusLow,
                  minus ? kIq1MinusHigh : kIq1PlusHigh, lo, hi);
    }
    __device__ static __forceinline__ float sub_scale(const Run& r, int g) {
        const unsigned ls = (r.sc >> (16 * (g >> 2) + 3 * (g & 3))) & 7U;
        return half_bits(r.d) * static_cast<float>(2 * static_cast<int>(ls) + 1);
    }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// TQ2_0 planes: per superblock its 64 qs bytes (codes; value p of half h: bits 2(p / 32) of byte
// 32h + p % 32, as Q2_K) and d (scale plane). w = d*(q - 1). One scale per step; two groups keep
// the spread value order. MMA j is quarter j / 2, word j % 2.
struct GgmlTq20Decoder {
    static constexpr bool kF16       = kF16Operands;
    static constexpr int kRun        = 32;
    static constexpr bool kStepScale = true;
    static constexpr int kSplits     = 2;
    struct Run {
        uint2 qs; // word u: qs bytes 32h + 16u + 4qd .. + 3
        unsigned d;
    };
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int p            = k0 & 255;
        const int h            = p >> 7;
        const int qd           = (p >> 5) & 3;
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 8);
        const auto* qs = reinterpret_cast<const unsigned*>(a.codes + blk * 64 + 32 * h + 4 * qd);
        Run r;
        r.qs = make_uint2(qs[0], qs[4]);
        r.d  = reinterpret_cast<const unsigned short*>(a.scales)[blk];
        return r;
    }
    __device__ static __forceinline__ Run zero() { return {make_uint2(0, 0), 0}; }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        byte_codes<1, kF16>((((j & 1) ? r.qs.y : r.qs.x) >> (2 * (j >> 1))) & 0x03030303U, lo, hi);
    }
    __device__ static __forceinline__ float sub_scale(const Run& r, int) { return half_bits(r.d); }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// TQ1_0 planes: per superblock its 52 code bytes (qs[48] then qh[4], five or four base-3 digits
// per byte) and d (scale plane). Values 0..159: digit v / 32 of qs[v % 32]; 160..239: digit
// (v - 160) / 16 of qs[32 + (v - 160) % 16]; 240..255: digit (v - 240) / 4 of qh[v % 4].
// w = d*(digit - 1). A lane's four values always share one digit of four consecutive bytes: half
// 0's MMA j takes digit j / 2 of qs[16(j % 2) + 4qd ..]; half 1's MMAs 0, 1 digit 4 of the same,
// MMAs 2..6 digit j - 2 of qs[32 + 4qd ..], MMA 7 digit qd of qh.
struct GgmlTq10Decoder {
    static constexpr bool kF16       = kF16Operands;
    static constexpr int kRun        = 32;
    static constexpr bool kStepScale = true;
    static constexpr int kSplits     = 2;
    struct Run {
        uint4 w;    // qs[4qd ..], qs[16 + 4qd ..], qs[32 + 4qd ..], qh
        unsigned d; // binary16 d (low half); the half h in bit 16, qd in bits 17, 18
    };
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int p            = k0 & 255;
        const unsigned h       = static_cast<unsigned>(p >> 7);
        const unsigned qd      = static_cast<unsigned>(p >> 5) & 3U;
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 8);
        const auto* c          = reinterpret_cast<const unsigned*>(a.codes + blk * 52);
        Run r;
        r.w = make_uint4(c[qd], c[4 + qd], c[8 + qd], c[12]);
        r.d = static_cast<unsigned>(reinterpret_cast<const unsigned short*>(a.scales)[blk]) |
              (h << 16) | (qd << 17);
        return r;
    }
    __device__ static __forceinline__ Run zero() { return {make_uint4(0, 0, 0, 0), 0}; }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        constexpr unsigned kPow3[5] = {1, 3, 9, 27, 81};
        unsigned codes;
        if (((r.d >> 16) & 1U) == 0) {
            codes = trits((j & 1) ? r.w.y : r.w.x, kPow3[j >> 1]);
        } else if (j < 2) {
            codes = trits((j & 1) ? r.w.y : r.w.x, kPow3[4]);
        } else if (j < 7) {
            codes = trits(r.w.z, kPow3[j - 2 < 5 ? j - 2 : 0]);
        } else {
            const unsigned qd = (r.d >> 17) & 3U;
            codes             = trits(r.w.w, qd == 0 ? 1U : qd == 1 ? 3U : qd == 2 ? 9U : 27U);
        }
        byte_codes<1, kF16>(codes, lo, hi);
    }
    __device__ static __forceinline__ float sub_scale(const Run& r, int) { return half_bits(r.d); }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// Q1_0 planes: per 128-value block its 16 code bytes (bit v % 8 of byte v / 8: value v's sign)
// and d (scale plane). w = +d where the bit is set, else -d. A K step is one block; two scale
// groups (both d) keep the spread value order. MMA j's lane qd takes nibble qd % 2 of byte
// 2j + qd / 2.
struct GgmlQ10Decoder {
    static constexpr int kRun        = 32;
    static constexpr bool kStepScale = true;
    static constexpr int kSplits     = 2;
    struct Run {
        uint2 bits; // byte j (>> 4(qd % 2)): MMA j's four bits in its low nibble
        unsigned d;
    };
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const unsigned qd      = static_cast<unsigned>(k0 >> 5) & 3U;
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 7);
        const uint4 w          = *reinterpret_cast<const uint4*>(a.codes + blk * 16);
        const unsigned b = qd >> 1, s = 4U * (qd & 1U);
        Run r;
        r.bits = make_uint2(pick_pairs(w.x, w.y, b) >> s, pick_pairs(w.z, w.w, b) >> s);
        r.d    = reinterpret_cast<const unsigned short*>(a.scales)[blk];
        return r;
    }
    __device__ static __forceinline__ Run zero() { return {make_uint2(0, 0), 0}; }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const unsigned nib = ((j < 4 ? r.bits.x : r.bits.y) >> (8 * (j & 3))) & 0xfU;
        lo = hi = 0x3f803f80U; // BF16 1.0, twice
        apply_signs(~nib & 0xfU, lo, hi);
    }
    __device__ static __forceinline__ float sub_scale(const Run& r, int) { return half_bits(r.d); }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// Q2_0 planes: per 64-value block its 16 code bytes (bits 2(v % 4) of byte v / 4) and d (scale
// plane). w = d*(q - 1). A K step is two blocks, one scale group each; MMA j's lane qd takes
// byte 4(j % 4) + qd of block j / 4.
struct GgmlQ20Decoder {
    static constexpr bool kF16       = kF16Operands;
    static constexpr int kRun        = 32;
    static constexpr bool kStepScale = true;
    static constexpr int kSplits     = 2;
    struct Run {
        uint2 q;    // byte j: MMA j's four codes
        unsigned d; // the two blocks' binary16 d
    };
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const unsigned qd = static_cast<unsigned>(k0 >> 5) & 3U;
        const std::int64_t blk =
            static_cast<std::int64_t>(row) * a.groups_per_row + ((k0 >> 6) & ~1);
        const auto* c  = reinterpret_cast<const uint4*>(a.codes + blk * 16);
        const uint4 w0 = c[0], w1 = c[1];
        Run r;
        r.q = make_uint2(pick_bytes(w0.x, w0.y, w0.z, w0.w, qd),
                         pick_bytes(w1.x, w1.y, w1.z, w1.w, qd));
        r.d = *reinterpret_cast<const unsigned*>(a.scales + blk * 2);
        return r;
    }
    __device__ static __forceinline__ Run zero() { return {make_uint2(0, 0), 0}; }
    __device__ static __forceinline__ Run masked(const GemmArgs&, int, int) { return zero(); }
    __device__ static __forceinline__ void pair(const Run& r, int j, unsigned& lo, unsigned& hi) {
        const unsigned q = ((j < 4 ? r.q.x : r.q.y) >> (8 * (j & 3))) & 0xffU;
        byte_codes<1, kF16>((q | (q << 6) | (q << 12) | (q << 18)) & 0x03030303U, lo, hi);
    }
    __device__ static __forceinline__ float sub_scale(const Run& r, int g) {
        return half_bits(r.d >> (16 * g));
    }
    __device__ static __forceinline__ float step_scale(const Run&) { return 1.0F; }
    __device__ static __forceinline__ float row_scale(const GemmArgs&, int) { return 1.0F; }
};

// ---------------------------------------------------------------------------------------------
// ggml_frag_v1 (core/ggml_frag.h): routed-expert banks whose code bytes are stored in the order
// the decoders above read them. Each fragment-order decoder is its published decoder with load()
// reading the permuted code plane: every Run field equals the published load's for the same
// (row, k0), so the MMA gets identical registers and the output is byte-identical. The grouped
// MoE kernel is their only reader.
// ---------------------------------------------------------------------------------------------

// Q4_0: in each whole 64-byte span of four blocks (a K step), 16-byte chunk qd holds word qd of
// the four blocks. A row's last partial span is natural and read block by block.
struct GgmlQ40FragDecoder : GgmlQ40Decoder {
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int qd           = (k0 >> 5) & 3;
        const int first        = (k0 >> 5) & ~3;
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + first;
        const int valid        = a.groups_per_row - first;
        if (valid < 4) return load_blocks(a, blk, qd, valid);
        Run r;
        r.qs = *reinterpret_cast<const uint4*>(a.codes + blk * 16 + 16 * qd);
        r.qh = make_uint4(0, 0, 0, 0);
        if ((a.groups_per_row & 3) == 0) {
            const uint2 d = *reinterpret_cast<const uint2*>(a.scales + blk * 2);
            r.dm          = make_uint4(d.x, d.y, 0, 0);
        } else {
            // The span's scales are only 2-byte aligned when rows are not whole spans.
            const auto* s     = reinterpret_cast<const unsigned short*>(a.scales) + blk;
            const unsigned lo = static_cast<unsigned>(s[0]) | (static_cast<unsigned>(s[1]) << 16);
            const unsigned hi = static_cast<unsigned>(s[2]) | (static_cast<unsigned>(s[3]) << 16);
            r.dm              = make_uint4(lo, hi, 0, 0);
        }
        return r;
    }
};

// Q4_K (144-byte blocks): each 64-byte half of the 128 nibble bytes 4 x 4-transposed by words;
// the 16-byte head unchanged.
struct GgmlQ4KFragDecoder : GgmlQ4KDecoder {
    // The base's raw load reads the untransposed nibbles: deleted, so UnpackOf is false here and
    // no latency variant decodes this layout with it.
    __device__ static Run load_raw(const GemmArgs&, int, int) = delete;
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int p  = k0 & 255;
        const int h  = p >> 7;
        const int qd = (p >> 5) & 3;
        const std::uint8_t* b =
            a.codes + (static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 8)) * 144;
        const uint4 head = *reinterpret_cast<const uint4*>(b);
        Run r;
        r.nib   = *reinterpret_cast<const uint4*>(b + 16 + 64 * h + 16 * qd);
        r.hbits = make_uint2(0, 0);
        r.sc    = h == 0 ? head.y & 0x3f3f3f3fU
                         : (head.w & 0x0f0f0f0fU) | ((head.y >> 2) & 0x30303030U);
        r.mn    = h == 0 ? head.z & 0x3f3f3f3fU
                         : ((head.w >> 4) & 0x0f0f0f0fU) | ((head.z >> 2) & 0x30303030U);
        r.hi    = 0;
        r.dd    = head.x;
        return r;
    }
};

// IQ3_XXS: in each 32-byte half of qs, lane qd's eight index bytes (published bytes 4m + qd) are
// contiguous at 8qd (an 8 x 4 byte transpose). Sign and d planes unchanged.
struct GgmlIq3XxsFragDecoder : GgmlIq3XxsDecoder {
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int p            = k0 & 255;
        const int h            = p >> 7;
        const unsigned qd      = static_cast<unsigned>(p >> 5) & 3U;
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 8);
        const uint4 s4 = *reinterpret_cast<const uint4*>(a.high + blk * 32 + 16 * h);
        const unsigned shift = 7U * (qd >> 1);
        const unsigned ls    = (s4.x >> 28) | ((s4.y >> 28) << 4) | ((s4.z >> 28) << 8) |
                            ((s4.w >> 28) << 12);
        Run r;
        r.idx = *reinterpret_cast<const uint2*>(a.codes + blk * 64 + 32 * h + 8 * qd);
        r.ss  = make_uint4(s4.x >> shift, s4.y >> shift, s4.z >> shift, s4.w >> shift);
        r.dsc = static_cast<unsigned>(reinterpret_cast<const unsigned short*>(a.scales)[blk]) |
                (ls << 16);
        return r;
    }
};

// IQ2_S (72 code bytes per superblock): bytes 0..63 are, per half h, [qs even][signs even][qs odd]
// [signs odd] (8 bytes each) at 32h, so lane pair b reads its index and sign bytes as two 8-byte
// words (a 72-byte superblock is only 8-byte aligned). qh (bytes 64..71) unchanged.
struct GgmlIq2SFragDecoder : GgmlIq2SDecoder {
    __device__ static __forceinline__ Run load(const GemmArgs& a, int row, int k0) {
        const int p            = k0 & 255;
        const int h            = p >> 7;
        const unsigned b       = static_cast<unsigned>(p >> 6) & 1U;
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k0 >> 8);
        const std::uint8_t* cb = a.codes + blk * 72;
        const auto* v          = reinterpret_cast<const uint2*>(cb + 32 * h + 16 * b);
        Run r;
        r.idx = v[0];
        r.sg  = v[1];
        r.qh  = *reinterpret_cast<const unsigned*>(cb + 64 + 4 * h) >> (2U * b);
        r.sc  = *reinterpret_cast<const unsigned*>(a.high + blk * 8 + 4 * h);
        r.d   = reinterpret_cast<const unsigned short*>(a.scales)[blk];
        return r;
    }
};

// ---------------------------------------------------------------------------------------------
// One K step: the activations and the MMAs of every tile, shared by the generic and the MoE
// kernels. A decoder's traits: kSplits scale groups per step (default 1), kMinTerm.
// ---------------------------------------------------------------------------------------------
template <class D, class = void>
struct SplitsOf : std::integral_constant<int, 1> {};
template <class D>
struct SplitsOf<D, std::void_t<decltype(D::kSplits)>> : std::integral_constant<int, D::kSplits> {};
template <class D, class = void>
struct MinTermOf : std::false_type {};
template <class D>
struct MinTermOf<D, std::void_t<decltype(D::kMinTerm)>> : std::bool_constant<D::kMinTerm> {};

// A decoder with a codebook (kTableWords words) stages it in shared memory with stage() before
// its kernel's first step; the kernel follows with a barrier.
template <class D, class = void>
struct TableOf : std::false_type {};
template <class D>
struct TableOf<D, std::void_t<decltype(D::kTableWords)>> : std::true_type {};

// A spread decoder (split steps or a min term) maps MMA j of a step to the step's values
// 16j .. 16j + 15; any other decoder gives quad q the run q*R .. q*R + R - 1, four per MMA.
template <class D>
inline constexpr bool SpreadOf = SplitsOf<D>::value > 1 || MinTermOf<D>::value;
// A group-tail decoder (kGroupTail) takes K of whole scale groups rather than whole steps: the
// kernels decide a step's tail for the whole step (every lane holds part of each group) and
// mma_step stops at the first group past K.
template <class D, class = void>
struct GroupTailOf : std::false_type {};
template <class D>
struct GroupTailOf<D, std::void_t<decltype(D::kGroupTail)>> : std::bool_constant<D::kGroupTail> {};
// A decoder with load_raw() (the K-quant heads) can leave part of a run as loaded and finish it
// with unpack(run, step_k0) when the step computes: rmma_kernel's latency variant, where no
// instruction of a fetch waits for its loads and the next step's loads stay in flight while
// this step computes. load() gives the finished run (every other kernel, the MoE ones too).
template <class D, class = void>
struct UnpackOf : std::false_type {};
template <class D>
struct UnpackOf<D, std::void_t<decltype(D::unpack(std::declval<typename D::Run&>(), 0)),
                               decltype(D::load_raw(std::declval<const GemmArgs&>(), 0, 0))>>
    : std::true_type {};

// A decoder whose SwiGLU decode tile stages its activations ahead (kStageActs; rmma_kernel's
// kActStage).
template <class D, class = void>
struct StageActsOf : std::false_type {};
template <class D>
struct StageActsOf<D, std::void_t<decltype(D::kStageActs)>> : std::bool_constant<D::kStageActs> {};

// A decoder with finish() (kFinish: the wide Q6_K) loads its runs in a raw form that finish()
// completes with the whole warp once the loads have landed: mma_step finishes the step's runs in
// place after issuing the step's activation loads (finished in the fetch, the swap waited for the
// loads there; ahead of the activation loads, it held them back by the weights' latency; on a
// copy, both forms live, the 128-register tiles spilled).
template <class D, class = void>
struct FinishOf : std::false_type {};
template <class D>
struct FinishOf<D, std::void_t<decltype(D::kFinish)>> : std::bool_constant<D::kFinish> {};

// The wide-load form of a decoder (rmma_kernel's wide variant; void: none).
template <class D>
struct WideOf {
    using type = void;
};
template <>
struct WideOf<GgmlQ6KDecoder> {
    using type = GgmlQ6KWideDecoder;
};

template <class D, class = void>
struct QuadRunsOf : std::false_type {};
template <class D>
struct QuadRunsOf<D, std::void_t<decltype(D::kQuadRuns)>> : std::bool_constant<D::kQuadRuns> {};

template <class D, class = void>
struct F16Of : std::false_type {};

template <class D>
struct F16Of<D, std::void_t<decltype(D::kF16)>> : std::bool_constant<D::kF16> {};

// The MMA of a decoder's element: BF16, or FP16 operands (F16Of) with the activations converted.
template <bool F16>
__device__ __forceinline__ void mma_operands(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    if constexpr (F16) {
        mma_f16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    } else {
        mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    }
}

// A step's activations as the decoder's MMA reads them: FP16 for an F16Of decoder (Turing), each
// register converted once for every tile and set of the step.
template <class Decoder, int NT, int M>
__device__ __forceinline__ void act_operands(unsigned (&act)[NT][2 * M]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 800
    if constexpr (F16Of<Decoder>::value) {
#    pragma unroll
        for (int n = 0; n < NT; ++n)
#    pragma unroll
            for (int i = 0; i < 2 * M; ++i) act[n][i] = bf16x2_to_f16x2(act[n][i]);
    }
#else
    (void)act;
#endif
}

// A lane's activations of M consecutive MMAs from K value k0 for tokens col[n] (null: zeros):
// act[n][2i], act[n][2i + 1] are MMA i's B fragment, in the decoder's value order. A spread
// decoder's MMA i covers k0 + 16i .. k0 + 16i + 15; any other decoder loads a whole step.
template <class Decoder, int NT, int M>
__device__ __forceinline__ void load_act(unsigned (&act)[NT][2 * M],
                                         const __nv_bfloat16* const (&col)[NT], int k0, int quad) {
    constexpr int R = Decoder::kRun;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        if constexpr (SpreadOf<Decoder>) {
#pragma unroll
            for (int i = 0; i < M; ++i) {
                // A quad-run decoder: quad q owns the group's values 4Mq .. 4Mq + 4M - 1.
                const int at = QuadRunsOf<Decoder>::value ? 4 * M * quad + 4 * i : 16 * i + 4 * quad;
                uint2 w = make_uint2(0, 0);
                if (col[n] != nullptr) w = load_vec<uint2>(col[n] + k0 + at);
                act[n][2 * i]     = w.x;
                act[n][2 * i + 1] = w.y;
            }
        } else {
            static_assert(M == R / 4, "a run decoder loads its whole step");
#pragma unroll
            for (int v = 0; v < R / 8; ++v) {
                uint4 w = make_uint4(0, 0, 0, 0);
                if (col[n] != nullptr) w = load_vec<uint4>(col[n] + k0 + quad * R + v * 8);
                act[n][v * 4 + 0] = w.x;
                act[n][v * 4 + 1] = w.y;
                act[n][v * 4 + 2] = w.z;
                act[n][v * 4 + 3] = w.w;
            }
        }
    }
}

// One K step (base step_k0) of the MT m16 tiles of each row set (runs[s][m][0], [1]: rows g and
// g + 8) for tokens col[n], shared by the generic and the MoE kernels. A spread decoder runs group
// by group, loading only the group's activations: the group's MMAs into an FP32 partial per tile,
// scaled by the tile rows' group scales, minus (min-term decoders) the rows' group mins times the
// group's activation sums, an MMA with ones shared by every tile. A decoder without splits has
// one scale group per step (kStepScale: the step's partial is scaled; otherwise the MMA
// accumulates straight into acc). Every accumulator takes its terms in the same order whatever
// MT and the set count are.
//
// Staged (rmma_kernel's act_stage): a spread decoder's activations of the step were copied ahead
// into shared memory, chunk c (MMA c of the step, 8 bytes) of this lane at stage[32 * c].
// Ahead (rmma_kernel's act_ring): they were loaded into registers with the step's weights, MMA c's
// B fragment at ahead[2c], ahead[2c + 1] (load_act of the whole step: the same values).
template <class Decoder, int NT, int J, int Sets, int MT, bool Staged = false, bool Ahead = false>
__device__ __forceinline__ void mma_step(float (&acc)[Sets][MT][NT][4],
                                         const typename Decoder::Run (&runs)[Sets][MT][2],
                                         const __nv_bfloat16* const (&col)[NT], int step_k0,
                                         int quad, int k, const uint2* stage = nullptr,
                                         const unsigned* ahead = nullptr) {
    static_assert(!Staged || (SpreadOf<Decoder> && NT == 1), "staged: one tile, spread");
    static_assert(!Ahead || (SpreadOf<Decoder> && NT == 1 && !Staged), "ahead: one tile, spread");
    if constexpr (SpreadOf<Decoder>) {
        constexpr int S = SplitsOf<Decoder>::value;
        constexpr int G = J / S; // MMAs per scale group
        static_assert(J % S == 0, "a spread step's MMAs split evenly into scale groups");
        using Run              = typename Decoder::Run;
        constexpr bool kFinish = FinishOf<Decoder>::value;
        static_assert(!kFinish || (!Staged && !GroupTailOf<Decoder>::value),
                      "a finish decoder loads its activations here, for whole steps");
        // kFinish: every group's activations, issued before the finish waits for the weights
        // (issued after it, RTX 4090 Q6_K add 3840x15360 w1x4 at T = 6: 58.9 -> 66.5 us). The
        // runs are the kernel's ring (rmma_kernel's compute passes it as loaded), finished in place.
        unsigned early[kFinish ? S : 1][NT][2 * G];
        if constexpr (kFinish) {
#pragma unroll
            for (int g = 0; g < S; ++g)
                load_act<Decoder, NT, G>(early[g], col, step_k0 + 16 * G * g, quad);
            auto& ring = const_cast<Run(&)[Sets][MT][2]>(runs);
#pragma unroll
            for (int s = 0; s < Sets; ++s)
#pragma unroll
                for (int m = 0; m < MT; ++m)
#pragma unroll
                    for (int h = 0; h < 2; ++h) ring[s][m][h] = Decoder::finish(ring[s][m][h]);
        }
        const Run(&ready)[Sets][MT][2] = runs;
#pragma unroll
        for (int g = 0; g < S; ++g) {
            if constexpr (GroupTailOf<Decoder>::value) {
                if (step_k0 + 16 * G * g >= k) break; // the row's end: warp-uniform
            }
            unsigned act[NT][2 * G];
            if constexpr (kFinish) {
#pragma unroll
                for (int n = 0; n < NT; ++n)
#pragma unroll
                    for (int i = 0; i < 2 * G; ++i) act[n][i] = early[g][n][i];
            } else if constexpr (Staged) {
#pragma unroll
                for (int i = 0; i < G; ++i) {
                    const uint2 w     = stage[32 * (g * G + i)];
                    act[0][2 * i]     = w.x;
                    act[0][2 * i + 1] = w.y;
                }
            } else if constexpr (Ahead) {
#pragma unroll
                for (int i = 0; i < 2 * G; ++i) act[0][i] = ahead[2 * G * g + i];
            } else {
                load_act<Decoder, NT, G>(act, col, step_k0 + 16 * G * g, quad);
            }
            act_operands<Decoder, NT, G>(act);
            float sums[NT][4];
            if constexpr (MinTermOf<Decoder>::value) {
                // 1.0, twice: BF16, or FP16 for an F16Of decoder.
                constexpr unsigned kOnes = F16Of<Decoder>::value ? 0x3c003c00U : 0x3f803f80U;
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    sums[n][0] = sums[n][1] = sums[n][2] = sums[n][3] = 0.0F;
#pragma unroll
                    for (int i = 0; i < G; ++i)
                        mma_operands<F16Of<Decoder>::value>(sums[n][0], sums[n][1], sums[n][2],
                                                            sums[n][3], kOnes, kOnes, kOnes, kOnes,
                                                            act[n][2 * i], act[n][2 * i + 1]);
                }
            }
#pragma unroll
            for (int s = 0; s < Sets; ++s) {
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    const auto& r0 = ready[s][m][0];
                    const auto& r1 = ready[s][m][1];
                    float part[NT][4];
#pragma unroll
                    for (int n = 0; n < NT; ++n)
                        part[n][0] = part[n][1] = part[n][2] = part[n][3] = 0.0F;
#pragma unroll
                    for (int i = 0; i < G; ++i) {
                        unsigned a0, a1, a2, a3;
                        Decoder::pair(r0, g * G + i, a0, a2);
                        Decoder::pair(r1, g * G + i, a1, a3);
#pragma unroll
                        for (int n = 0; n < NT; ++n)
                            mma_operands<F16Of<Decoder>::value>(part[n][0], part[n][1], part[n][2],
                                                                part[n][3], a0, a1, a2, a3,
                                                                act[n][2 * i], act[n][2 * i + 1]);
                    }
                    const float s0 = Decoder::sub_scale(r0, g);
                    const float s1 = Decoder::sub_scale(r1, g);
                    auto& c        = acc[s][m];
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        c[n][0] = fmaf(part[n][0], s0, c[n][0]);
                        c[n][1] = fmaf(part[n][1], s0, c[n][1]);
                        c[n][2] = fmaf(part[n][2], s1, c[n][2]);
                        c[n][3] = fmaf(part[n][3], s1, c[n][3]);
                    }
                    if constexpr (MinTermOf<Decoder>::value) {
                        const float m0 = Decoder::sub_min(r0, g);
                        const float m1 = Decoder::sub_min(r1, g);
#pragma unroll
                        for (int n = 0; n < NT; ++n) {
                            c[n][0] = fmaf(-sums[n][0], m0, c[n][0]);
                            c[n][1] = fmaf(-sums[n][1], m0, c[n][1]);
                            c[n][2] = fmaf(-sums[n][2], m1, c[n][2]);
                            c[n][3] = fmaf(-sums[n][3], m1, c[n][3]);
                        }
                    }
                }
            }
        }
    } else {
        static_assert(!FinishOf<Decoder>::value, "a finish decoder is a spread one");
        unsigned act[NT][2 * J];
        load_act<Decoder, NT, J>(act, col, step_k0, quad);
        act_operands<Decoder, NT, J>(act);
#pragma unroll
        for (int s = 0; s < Sets; ++s) {
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                const auto& r0 = runs[s][m][0];
                const auto& r1 = runs[s][m][1];
                auto& c        = acc[s][m];
                if constexpr (Decoder::kStepScale) {
                    float part[NT][4];
#pragma unroll
                    for (int n = 0; n < NT; ++n)
                        part[n][0] = part[n][1] = part[n][2] = part[n][3] = 0.0F;
#pragma unroll
                    for (int j = 0; j < J; ++j) {
                        unsigned a0, a1, a2, a3;
                        Decoder::pair(r0, j, a0, a2);
                        Decoder::pair(r1, j, a1, a3);
#pragma unroll
                        for (int n = 0; n < NT; ++n)
                            mma_operands<F16Of<Decoder>::value>(part[n][0], part[n][1], part[n][2],
                                                                part[n][3], a0, a1, a2, a3,
                                                                act[n][2 * j], act[n][2 * j + 1]);
                    }
                    const float s0 = Decoder::step_scale(r0);
                    const float s1 = Decoder::step_scale(r1);
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        c[n][0] = fmaf(part[n][0], s0, c[n][0]);
                        c[n][1] = fmaf(part[n][1], s0, c[n][1]);
                        c[n][2] = fmaf(part[n][2], s1, c[n][2]);
                        c[n][3] = fmaf(part[n][3], s1, c[n][3]);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < J; ++j) {
                        unsigned a0, a1, a2, a3;
                        Decoder::pair(r0, j, a0, a2);
                        Decoder::pair(r1, j, a1, a3);
#pragma unroll
                        for (int n = 0; n < NT; ++n)
                            mma_operands<F16Of<Decoder>::value>(c[n][0], c[n][1], c[n][2], c[n][3],
                                                                a0, a1, a2, a3, act[n][2 * j],
                                                                act[n][2 * j + 1]);
                    }
                }
            }
        }
    }
}

#if NINFERNO_FMA_MMA
// NINFERNO-SPECIAL(arch:sm70): without tensor mma.sync every MMA of mma_step is FP32 FMAs over 32
// warp shuffles of the operands for all eight columns of the tile (ops/common/mma.cuh), whatever
// the call's tokens. A one-tile call (NT = 1, T <= 8) takes the products lane by lane instead:
// lane (g, t) holds rows g and g + 8 at its K positions (a0 / a2, a1 / a3), the same positions as
// the B registers lane (n, t) loads for column n, so it reads token n's activations there itself
// and sums each row and token over its own positions; the four lanes of a quad add their sums
// once, after the last step (rmma_kernel). Scales and min terms are linear in those sums, applied
// per lane as mma_step applies them per tile. No shuffles, and no products for absent tokens.
// acc[s][m][h][n]: row g + 8h, token n of the tile; col[n]: token n's activations (null: none).
// runs: unpacked (UnpackOf) by the kernel.
template <class Decoder, int J, int Sets, int MT>
__device__ __forceinline__ void mma_step_local(float (&acc)[Sets][MT][2][8],
                                               const typename Decoder::Run (&loaded)[Sets][MT][2],
                                               const __nv_bfloat16* const (&col)[8], int tokens,
                                               int step_k0, int quad, int k) {
    static_assert(!F16Of<Decoder>::value, "the sm_70 build keeps BF16 codes");
    // A finish decoder's runs finished (no launch takes one on sm_70: rmma_wide_pays).
    using Run = typename Decoder::Run;
    Run done[FinishOf<Decoder>::value ? Sets : 1][FinishOf<Decoder>::value ? MT : 1][2];
    const Run(&runs)[Sets][MT][2] = [&]() -> const Run(&)[Sets][MT][2] {
        if constexpr (FinishOf<Decoder>::value) {
            for (int s = 0; s < Sets; ++s)
                for (int m = 0; m < MT; ++m)
                    for (int h = 0; h < 2; ++h) done[s][m][h] = Decoder::finish(loaded[s][m][h]);
            return done;
        } else {
            return loaded;
        }
    }();
    // Two products of a code register and an activation register, added to `sum`.
    const auto dot2 = [](float sum, unsigned a, unsigned b) {
        const float2 w = bf16x2_pair(a);
        const float2 v = bf16x2_pair(b);
        return fmaf(w.y, v.y, fmaf(w.x, v.x, sum));
    };
    const auto act = [&](int n, int offset) -> unsigned {
        return col[n] != nullptr ? load_vec<unsigned>(col[n] + offset) : 0U;
    };
    if constexpr (SpreadOf<Decoder>) {
        constexpr int S = SplitsOf<Decoder>::value;
        constexpr int G = J / S;
        static_assert(J % S == 0, "a spread step's MMAs split evenly into scale groups");
#pragma unroll
        for (int g = 0; g < S; ++g) {
            if constexpr (GroupTailOf<Decoder>::value) {
                if (step_k0 + 16 * G * g >= k) break; // the row's end: warp-uniform
            }
            const int base = step_k0 + 16 * G * g;
            // MMA i's B registers: values at, at + 1 (b0) and at + 2, at + 3 (b1) (load_act).
            int at[G];
#pragma unroll
            for (int i = 0; i < G; ++i)
                at[i] = base + (QuadRunsOf<Decoder>::value ? 4 * G * quad + 4 * i : 16 * i + 4 * quad);
            float sums[8];
            if constexpr (MinTermOf<Decoder>::value) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    sums[n] = 0.0F;
                    if (n >= tokens) continue;
#pragma unroll
                    for (int i = 0; i < G; ++i) {
                        sums[n] = dot2(sums[n], 0x3f803f80U, act(n, at[i]));
                        sums[n] = dot2(sums[n], 0x3f803f80U, act(n, at[i] + 2));
                    }
                }
            }
#pragma unroll
            for (int s = 0; s < Sets; ++s) {
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    const auto& r0 = runs[s][m][0];
                    const auto& r1 = runs[s][m][1];
                    unsigned a[G][4];
#pragma unroll
                    for (int i = 0; i < G; ++i) {
                        Decoder::pair(r0, g * G + i, a[i][0], a[i][2]);
                        Decoder::pair(r1, g * G + i, a[i][1], a[i][3]);
                    }
                    const float s0 = Decoder::sub_scale(r0, g);
                    const float s1 = Decoder::sub_scale(r1, g);
                    float m0 = 0.0F, m1 = 0.0F;
                    if constexpr (MinTermOf<Decoder>::value) {
                        m0 = Decoder::sub_min(r0, g);
                        m1 = Decoder::sub_min(r1, g);
                    }
                    auto& c = acc[s][m];
#pragma unroll
                    for (int n = 0; n < 8; ++n) {
                        if (n >= tokens) continue;
                        float p0 = 0.0F, p1 = 0.0F;
#pragma unroll
                        for (int i = 0; i < G; ++i) {
                            const unsigned b0 = act(n, at[i]);
                            const unsigned b1 = act(n, at[i] + 2);
                            p0 = dot2(dot2(p0, a[i][0], b0), a[i][2], b1);
                            p1 = dot2(dot2(p1, a[i][1], b0), a[i][3], b1);
                        }
                        c[0][n] = fmaf(p0, s0, c[0][n]);
                        c[1][n] = fmaf(p1, s1, c[1][n]);
                        if constexpr (MinTermOf<Decoder>::value) {
                            c[0][n] = fmaf(-sums[n], m0, c[0][n]);
                            c[1][n] = fmaf(-sums[n], m1, c[1][n]);
                        }
                    }
                }
            }
        }
    } else {
        // load_act's whole step: register r holds values quad * R + 2r and + 1.
        const int base = step_k0 + quad * Decoder::kRun;
#pragma unroll
        for (int s = 0; s < Sets; ++s) {
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                const auto& r0 = runs[s][m][0];
                const auto& r1 = runs[s][m][1];
                unsigned a[J][4];
#pragma unroll
                for (int j = 0; j < J; ++j) {
                    Decoder::pair(r0, j, a[j][0], a[j][2]);
                    Decoder::pair(r1, j, a[j][1], a[j][3]);
                }
                float s0 = 1.0F, s1 = 1.0F;
                if constexpr (Decoder::kStepScale) {
                    s0 = Decoder::step_scale(r0);
                    s1 = Decoder::step_scale(r1);
                }
                auto& c = acc[s][m];
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    if (n >= tokens) continue;
                    float p0 = 0.0F, p1 = 0.0F;
#pragma unroll
                    for (int j = 0; j < J; ++j) {
                        const unsigned b0 = act(n, base + 4 * j);
                        const unsigned b1 = act(n, base + 4 * j + 2);
                        p0 = dot2(dot2(p0, a[j][0], b0), a[j][2], b1);
                        p1 = dot2(dot2(p1, a[j][1], b0), a[j][3], b1);
                    }
                    if constexpr (Decoder::kStepScale) {
                        c[0][n] = fmaf(p0, s0, c[0][n]);
                        c[1][n] = fmaf(p1, s1, c[1][n]);
                    } else {
                        c[0][n] += p0;
                        c[1][n] += p1;
                    }
                }
            }
        }
    }
}
#endif

// Scalar element decode, for the fallback kernel (K that the register MMA cannot tile).
__device__ __forceinline__ float decode_element(const GemmArgs& a, WeightCodec codec, int row,
                                                int k) {
    switch (codec) {
    case WeightCodec::Bf16: {
        const auto* w = reinterpret_cast<const std::uint16_t*>(a.codes);
        return __bfloat162float(__ushort_as_bfloat16(w[static_cast<std::int64_t>(row) * a.k + k]));
    }
    case WeightCodec::Fp8: {
        __nv_fp8_e4m3 v;
        v.__x = a.codes[static_cast<std::int64_t>(row) * a.k + k];
        return static_cast<float>(v) * Fp8Decoder::row_scale(a, row);
    }
    case WeightCodec::Fp8RowF32: {
        __nv_fp8_e4m3 v;
        v.__x = a.codes[static_cast<std::int64_t>(row) * a.k + k];
        return static_cast<float>(v) * Fp8RowF32Decoder::row_scale(a, row);
    }
    case WeightCodec::Fp8B128: {
        __nv_fp8_e4m3 v;
        v.__x = a.codes[static_cast<std::int64_t>(row) * a.k + k];
        return static_cast<float>(v) * Fp8B128Decoder::block_scale(a, row, k);
    }
    case WeightCodec::Nvfp4: {
        const float2 values = __half22float2(
            e2m1x2_to_half2(a.codes[static_cast<std::int64_t>(row) * (a.k / 2) + k / 2]));
        __nv_fp8_e4m3 s;
        s.__x = a.scales[nvfp4_scale_offset(a, row, k / 16)];
        return ((k & 1) ? values.y : values.x) * static_cast<float>(s) *
               Nvfp4Decoder::row_scale(a, row);
    }
    case WeightCodec::Mxfp4: {
        const float2 values = __half22float2(
            e2m1x2_to_half2(a.codes[static_cast<std::int64_t>(row) * (a.k / 2) + k / 2]));
        return ((k & 1) ? values.y : values.x) *
               e8m0_value(a.scales[static_cast<std::int64_t>(row) * (a.k / 32) + k / 32]);
    }
    case WeightCodec::Q4:
    case WeightCodec::Q5:
    case WeightCodec::Q6: {
        const int group        = k >> 6;
        const int v            = k & 63;
        const std::int64_t gid = static_cast<std::int64_t>(row) * a.groups_per_row + group;
        const unsigned b       = a.codes[gid * 32 + v / 2];
        unsigned u             = (v & 1) ? (b >> 4) : (b & 0xfU);
        int q                  = 0;
        if (codec == WeightCodec::Q4) {
            q = static_cast<int>(u ^ 8U) - 8;
        } else if (codec == WeightCodec::Q5) {
            u |= ((static_cast<unsigned>(a.high[gid * 8 + v / 8]) >> (v & 7)) & 1U) << 4;
            q = static_cast<int>(u ^ 16U) - 16;
        } else {
            const int bit = 2 * v;
            u |= ((static_cast<unsigned>(a.high[gid * 16 + bit / 8]) >> (bit & 7)) & 3U) << 4;
            q = static_cast<int>(u ^ 32U) - 32;
        }
        const auto scale = reinterpret_cast<const unsigned short*>(a.scales)[gid];
        return static_cast<float>(q) * __half2float(__ushort_as_half(scale));
    }
    case WeightCodec::Q3G128:
    case WeightCodec::Q3G64: {
        const int g            = codec == WeightCodec::Q3G128 ? 128 : 64;
        const std::int64_t gid = static_cast<std::int64_t>(row) * a.groups_per_row + k / g;
        const std::uint8_t* c  = a.codes + gid * (g * 3 / 8);
        const int bit          = 3 * (k % g);
        unsigned w             = c[bit >> 3];
        if ((bit & 7) > 5) w |= static_cast<unsigned>(c[(bit >> 3) + 1]) << 8;
        const int q      = static_cast<int>(((w >> (bit & 7)) & 7U) ^ 4U) - 4;
        const auto scale = reinterpret_cast<const unsigned short*>(a.scales)[gid];
        return static_cast<float>(q) * __half2float(__ushort_as_half(scale));
    }
    case WeightCodec::U4Zp:
    case WeightCodec::U4ZpG32:
    case WeightCodec::U4ZpG64: {
        const int g = codec == WeightCodec::U4ZpG32 ? 32 : codec == WeightCodec::U4ZpG64 ? 64 : 128;
        const std::int64_t gid = static_cast<std::int64_t>(row) * a.groups_per_row + k / g;
        const unsigned b = a.codes[static_cast<std::int64_t>(row) * (a.k / 2) + k / 2];
        const int code   = static_cast<int>((k & 1) ? (b >> 4) : (b & 0xfU));
        const auto scale = reinterpret_cast<const unsigned short*>(a.scales)[gid];
        return static_cast<float>(code - static_cast<int>(a.high[gid])) *
               __half2float(__ushort_as_half(scale));
    }
    case WeightCodec::GgmlQ4K:
    case WeightCodec::GgmlQ5K: {
        const bool q5          = codec == WeightCodec::GgmlQ5K;
        const std::uint8_t* b  = a.codes + (static_cast<std::int64_t>(row) * a.groups_per_row +
                                           (k >> 8)) * (q5 ? 176 : 144);
        const uint4 head       = *reinterpret_cast<const uint4*>(b);
        const int p            = k & 255;
        const int sb           = p >> 5;
        const int l            = p & 31;
        float sc, m;
        kq_scale_min(head, sb, sc, m);
        unsigned q = (b[(q5 ? 48 : 16) + (sb >> 1) * 32 + l] >> ((sb & 1) * 4)) & 0xfU;
        if (q5) q |= ((static_cast<unsigned>(b[16 + l]) >> sb) & 1U) << 4;
        return half_bits(head.x) * sc * static_cast<float>(q) - half_bits(head.x >> 16) * m;
    }
    case WeightCodec::GgmlQ6K: {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k >> 8);
        const std::uint8_t* c  = a.codes + blk * 192;
        const int p            = k & 255;
        const int half         = p >> 7;
        const int qq           = (p & 127) >> 5;
        const int l            = p & 31;
        const unsigned ql      = c[64 * half + l + 32 * (qq & 1)];
        const unsigned qh      = c[128 + 32 * half + l];
        const int q =
            static_cast<int>(((qq >= 2 ? ql >> 4 : ql) & 0xfU) | (((qh >> (2 * qq)) & 3U) << 4));
        const int sc = static_cast<signed char>(a.high[blk * 16 + 8 * half + 2 * qq + l / 16]);
        const float d = half_bits(reinterpret_cast<const unsigned short*>(a.scales)[blk]);
        return d * static_cast<float>(sc) * static_cast<float>(q - 32);
    }
    case WeightCodec::GgmlQ3K: {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k >> 8);
        const std::uint8_t* c  = a.codes + blk * 96;
        const std::uint8_t* sb = a.high + blk * 12;
        const int p            = k & 255;
        const int half         = p >> 7;
        const int j            = (p & 127) >> 5;
        const int l            = p & 31;
        const int s            = 8 * half + 2 * j + l / 16;
        const int low          = (c[32 + 32 * half + l] >> (2 * j)) & 3;
        const int q            = ((c[l] >> (4 * half + j)) & 1) ? low : low - 4;
        const int sc = static_cast<int>(((sb[s & 7] >> (4 * (s >> 3))) & 0xfU) |
                                        (((sb[8 + (s & 3)] >> (2 * (s >> 2))) & 3U) << 4));
        const float d = half_bits(reinterpret_cast<const unsigned short*>(a.scales)[blk]);
        return d * static_cast<float>(sc - 32) * static_cast<float>(q);
    }
    case WeightCodec::GgmlQ2K: {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k >> 8);
        const int p            = k & 255;
        const int half         = p >> 7;
        const int j            = (p & 127) >> 5;
        const int l            = p & 31;
        const int q    = (a.codes[blk * 64 + 32 * half + l] >> (2 * j)) & 3;
        const int sc   = a.high[blk * 16 + 8 * half + 2 * j + l / 16];
        const auto* dd = reinterpret_cast<const unsigned short*>(a.scales) + 2 * blk;
        return half_bits(dd[0]) * static_cast<float>(sc & 0xf) * static_cast<float>(q) -
               half_bits(dd[1]) * static_cast<float>(sc >> 4);
    }
    // ggml 32-value blocks and i-quants, in ggml's dequantize_row_* operation order (explicitly
    // rounded: no contraction), so a value is bit-exact with ggml's (and gguf-py's) binary32.
    case WeightCodec::GgmlQ40:
    case WeightCodec::GgmlQ41:
    case WeightCodec::GgmlQ50:
    case WeightCodec::GgmlQ51:
    case WeightCodec::GgmlIq4Nl: {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k >> 5);
        const int v            = k & 31;
        const unsigned b       = a.codes[blk * 16 + (v & 15)];
        int q                  = static_cast<int>(v < 16 ? (b & 0xfU) : (b >> 4));
        const bool with_min    = codec == WeightCodec::GgmlQ41 || codec == WeightCodec::GgmlQ51;
        const auto* sw         = reinterpret_cast<const unsigned short*>(a.scales);
        const float d          = half_bits(sw[with_min ? 2 * blk : blk]);
        if (codec == WeightCodec::GgmlQ50 || codec == WeightCodec::GgmlQ51) {
            const unsigned qh = reinterpret_cast<const unsigned*>(a.high)[blk];
            q |= static_cast<int>(((qh >> v) & 1U) << 4);
        }
        if (codec == WeightCodec::GgmlIq4Nl) {
            return __fmul_rn(d, static_cast<float>(ggml_iq::kIq4nlValues[q]));
        }
        if (with_min) {
            return __fadd_rn(__fmul_rn(static_cast<float>(q), d), half_bits(sw[2 * blk + 1]));
        }
        return __fmul_rn(static_cast<float>(q - (codec == WeightCodec::GgmlQ40 ? 8 : 16)), d);
    }
    case WeightCodec::GgmlIq4Xs: {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k >> 8);
        const int p            = k & 255;
        const int ib           = p >> 5;
        const int l            = p & 31;
        const unsigned b       = a.codes[blk * 128 + 16 * ib + (l & 15)];
        const unsigned q       = l < 16 ? (b & 0xfU) : (b >> 4);
        const unsigned sl      = a.high[blk * 4 + ib / 2];
        const unsigned dsh     = reinterpret_cast<const unsigned*>(a.scales)[blk];
        const int ls = static_cast<int>(((sl >> (4 * (ib & 1))) & 0xfU) |
                                        (((dsh >> (16 + 2 * ib)) & 3U) << 4));
        const float dl = __fmul_rn(half_bits(dsh), static_cast<float>(ls - 32));
        return __fmul_rn(dl, static_cast<float>(ggml_iq::kIq4nlValues[q]));
    }
    case WeightCodec::GgmlIq3Xxs: {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k >> 8);
        const int p            = k & 255;
        const unsigned aux     = reinterpret_cast<const unsigned*>(a.high + blk * 32)[p >> 5];
        const unsigned index   = a.codes[blk * 64 + (p >> 2)];
        const unsigned t       = (aux >> (7 * ((p & 31) >> 3))) & 127U;
        const unsigned signs   = t | ((static_cast<unsigned>(__popc(t)) & 1U) << 7);
        const float d  = half_bits(reinterpret_cast<const unsigned short*>(a.scales)[blk]);
        const float db = __fmul_rn(__fmul_rn(d, __fadd_rn(0.5F, static_cast<float>(aux >> 28))),
                                   0.5F);
        const float g =
            static_cast<float>((ggml_iq::kIq3xxsGrid[index] >> (8 * (p & 3))) & 0xffU);
        return __fmul_rn(__fmul_rn(db, g), ((signs >> (p & 7)) & 1U) ? -1.0F : 1.0F);
    }
    case WeightCodec::GgmlIq3S: {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k >> 8);
        const std::uint8_t* cb = a.codes + blk * 104;
        const int p            = k & 255;
        const int gi           = p >> 2;
        const unsigned index   = cb[gi] | (((static_cast<unsigned>(cb[64 + (gi >> 3)]) >> (gi & 7)) & 1U) << 8);
        const unsigned sc      = (a.high[blk * 4 + (p >> 6)] >> (4 * ((p >> 5) & 1))) & 0xfU;
        const float d  = half_bits(reinterpret_cast<const unsigned short*>(a.scales)[blk]);
        const float db = __fmul_rn(d, static_cast<float>(1 + 2 * static_cast<int>(sc)));
        const float g  = static_cast<float>((ggml_iq::kIq3sGrid[index] >> (8 * (p & 3))) & 0xffU);
        return __fmul_rn(__fmul_rn(db, g),
                         ((cb[72 + (p >> 3)] >> (p & 7)) & 1U) ? -1.0F : 1.0F);
    }
    case WeightCodec::GgmlIq2Xxs:
    case WeightCodec::GgmlIq2Xs:
    case WeightCodec::GgmlIq2S: {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k >> 8);
        const int p            = k & 255;
        const int ib           = p >> 5;
        const int l            = (p & 31) >> 3;
        const std::uint8_t* cb = a.codes + blk * (codec == WeightCodec::GgmlIq2S ? 72 : 64);
        unsigned index = 0, signs = 0, ls = 0;
        if (codec == WeightCodec::GgmlIq2Xxs) {
            const auto* aux = reinterpret_cast<const unsigned*>(cb + 8 * ib);
            index           = (aux[0] >> (8 * l)) & 0xffU;
            const unsigned t = (aux[1] >> (7 * l)) & 127U;
            signs            = t | ((static_cast<unsigned>(__popc(t)) & 1U) << 7);
            ls               = aux[1] >> 28;
        } else {
            const unsigned sc = a.high[blk * 8 + ib];
            ls                = (sc >> (4 * (l >> 1))) & 0xfU;
            if (codec == WeightCodec::GgmlIq2Xs) {
                const unsigned e = reinterpret_cast<const unsigned short*>(cb)[p >> 3];
                const unsigned t = e >> 9;
                index            = e & 511U;
                signs            = t | ((static_cast<unsigned>(__popc(t)) & 1U) << 7);
            } else {
                index = cb[p >> 3] | (((static_cast<unsigned>(cb[64 + ib]) >> (2 * l)) & 3U) << 8);
                signs = cb[32 + (p >> 3)];
            }
        }
        const unsigned* grid = codec == WeightCodec::GgmlIq2Xxs ? ggml_iq::kIq2xxsGrid
                               : codec == WeightCodec::GgmlIq2Xs ? ggml_iq::kIq2xsGrid
                                                                 : ggml_iq::kIq2sGrid;
        const float d  = half_bits(reinterpret_cast<const unsigned short*>(a.scales)[blk]);
        const float db = __fmul_rn(__fmul_rn(d, __fadd_rn(0.5F, static_cast<float>(ls))), 0.25F);
        const float g =
            static_cast<float>((grid[2 * index + ((p & 7) >> 2)] >> (8 * (p & 3))) & 0xffU);
        return __fmul_rn(__fmul_rn(db, g), ((signs >> (p & 7)) & 1U) ? -1.0F : 1.0F);
    }
    case WeightCodec::GgmlIq1S:
    case WeightCodec::GgmlIq1M: {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k >> 8);
        const int p            = k & 255;
        const int ib           = p >> 5;
        const int l            = (p & 31) >> 3;
        unsigned index         = a.codes[blk * 32 + 4 * ib + l];
        bool minus             = false;
        float dl               = 0.0F;
        if (codec == WeightCodec::GgmlIq1S) {
            const unsigned h = reinterpret_cast<const unsigned short*>(a.high + blk * 16)[ib];
            index |= ((h >> (3 * l)) & 7U) << 8;
            minus = (h & 0x8000U) != 0;
            dl    = __fmul_rn(half_bits(reinterpret_cast<const unsigned short*>(a.scales)[blk]),
                              static_cast<float>(2 * static_cast<int>((h >> 12) & 7U) + 1));
        } else {
            const unsigned n = (static_cast<unsigned>(a.high[blk * 16 + 2 * ib + (l >> 1)]) >>
                                (4 * (l & 1))) & 0xfU;
            index |= (n & 7U) << 8;
            minus = (n & 8U) != 0;
            const uint2 sw = *reinterpret_cast<const uint2*>(a.scales + blk * 8);
            const unsigned sc =
                ((ib < 4 ? sw.x : sw.y) >> (16 * ((ib >> 1) & 1))) & 0xffffU;
            const unsigned ls = (sc >> (6 * (ib & 1) + 3 * (l >> 1))) & 7U;
            dl = __fmul_rn(half_bits(GgmlIq1MDecoder::d_bits(sw)),
                           static_cast<float>(2 * static_cast<int>(ls) + 1));
        }
        const int g = static_cast<int>((ggml_iq::kIq1sGrid[index] >> (4 * (p & 7))) & 0xfU) - 1;
        return __fmul_rn(dl, __fadd_rn(static_cast<float>(g), minus ? -0.125F : 0.125F));
    }
    case WeightCodec::GgmlTq10: {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k >> 8);
        const int v            = k & 255;
        const std::uint8_t* c  = a.codes + blk * 52;
        unsigned byte = 0, n = 0;
        if (v < 160) {
            byte = c[v & 31];
            n    = static_cast<unsigned>(v >> 5);
        } else if (v < 240) {
            byte = c[32 + ((v - 160) & 15)];
            n    = static_cast<unsigned>((v - 160) >> 4);
        } else {
            byte = c[48 + (v & 3)];
            n    = static_cast<unsigned>((v - 240) >> 2);
        }
        const unsigned pow3 = n == 0 ? 1U : n == 1 ? 3U : n == 2 ? 9U : n == 3 ? 27U : 81U;
        const unsigned q    = (byte * pow3) & 0xffU;
        const int xi        = static_cast<int>((q * 3U) >> 8);
        return __fmul_rn(static_cast<float>(xi - 1),
                         half_bits(reinterpret_cast<const unsigned short*>(a.scales)[blk]));
    }
    case WeightCodec::GgmlTq20: {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k >> 8);
        const int v            = k & 255;
        const int q = (a.codes[blk * 64 + 32 * (v >> 7) + (v & 31)] >> (2 * ((v & 127) >> 5))) & 3;
        return __fmul_rn(static_cast<float>(q - 1),
                         half_bits(reinterpret_cast<const unsigned short*>(a.scales)[blk]));
    }
    case WeightCodec::GgmlQ10: {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k >> 7);
        const int v            = k & 127;
        const float d = half_bits(reinterpret_cast<const unsigned short*>(a.scales)[blk]);
        return ((a.codes[blk * 16 + (v >> 3)] >> (v & 7)) & 1) ? d : -d;
    }
    case WeightCodec::GgmlQ20: {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (k >> 6);
        const int v            = k & 63;
        const int q            = (a.codes[blk * 16 + (v >> 2)] >> (2 * (v & 3))) & 3;
        return __fmul_rn(static_cast<float>(q - 1),
                         half_bits(reinterpret_cast<const unsigned short*>(a.scales)[blk]));
    }
    case WeightCodec::Q8: {
        const int group        = k >> 5;
        const std::int64_t gid = static_cast<std::int64_t>(row) * a.groups_per_row + group;
        const auto code        = static_cast<signed char>(a.codes[gid * 32 + (k & 31)]);
        const auto scale       = reinterpret_cast<const unsigned short*>(a.scales)[gid];
        return static_cast<float>(code) * __half2float(__ushort_as_half(scale));
    }
    }
    return 0.0F;
}

// ---------------------------------------------------------------------------------------------
// Epilogues.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float generic_silu(float v) { return v / (1.0F + expf(-v)); }

// GELU's tanh form through the logistic, 0.5 * (1 + tanh(u)) = 1 / (1 + exp(-2u)): no
// cancellation where tanh(u) nears -1. The sum form gave exactly 0 for a gate below about -5 (Q5_K
// GeGLU [666, 2560]: 0 against 1.01e-12), where this one keeps the tail.
__device__ __forceinline__ float generic_gelu_tanh(float v) {
    constexpr float kTwoSqrt2Pi = 1.59576912160573071176F; // 2 * sqrt(2 / pi)
    return v / (1.0F + expf(-kTwoSqrt2Pi * (v + 0.044715F * v * v * v)));
}

// The SwiGlu epilogue's gate activation: SiLU, or GELU under GemmArgs::gelu_gate (GeGLU).
__device__ __forceinline__ float generic_gate(const GemmArgs& a, float v) {
    return a.gelu_gate != 0 ? generic_gelu_tanh(v) : generic_silu(v);
}

template <Epilogue E>
__device__ __forceinline__ void store_output(const GemmArgs& a, int row, int token, float value) {
    int s = 0;
#pragma unroll
    for (int i = 0; i < kMaxSegments - 1; ++i) {
        s += (i + 1 < a.segments && row >= a.segment_end[i]) ? 1 : 0;
    }
    const int begin = s == 0 ? 0 : a.segment_end[s - 1];
    auto* p         = static_cast<__nv_bfloat16*>(a.out[s]) +
              static_cast<std::int64_t>(token) * a.ld[s] + (row - begin);
    // An explicit round-to-nearest add, never contracted with a scaling product in front of it:
    // whether nvcc fuses the two into an FMA differs between the full-tile and the predicated
    // instantiations of one kernel, and a token's bytes must not depend on its call's width.
    if constexpr (E == Epilogue::AddResidual) { value = __fadd_rn(value, __bfloat162float(*p)); }
    if constexpr (E == Epilogue::Store) {
        if (a.relu2 != 0) {
            const float r = fmaxf(value, 0.0F);
            value         = r * r;
        }
    }
    *p = __float2bfloat16_rn(value);
}

// Local output row of tile row `tile_row` in row block `block_row` for the shared-memory tile
// kernels (W4A4, tiled A16): with SwiGLU, gate and up rows interleave in groups of eight, so a
// block of BlockN tile rows makes BlockN / 2 outputs.
template <bool SwiGlu, int BlockN>
__device__ __forceinline__ int w4a4_local_row(int block_row, int tile_row) {
    if constexpr (SwiGlu) {
        const int g = tile_row >> 3;
        return block_row * (BlockN / 2) + (g >> 1) * 8 + (tile_row & 7);
    } else {
        return block_row * BlockN + tile_row;
    }
}

// A run as load() gives it, or (Raw) as load_raw() does.
template <class Decoder, bool Raw>
__device__ __forceinline__ typename Decoder::Run load_run(const GemmArgs& a, int row, int k0) {
    if constexpr (Raw) {
        return Decoder::load_raw(a, row, k0);
    } else {
        return Decoder::load(a, row, k0);
    }
}

// ---------------------------------------------------------------------------------------------
// The register MMA kernel. Grid: x = row blocks of kRowsPerCta local rows, y = token tiles.
// ---------------------------------------------------------------------------------------------
//
// Latency (the K-quant decoders of a one-tile schedule, UnpackOf or StageActsOf; the launch picks
// it, generic_gemm_launch.cuh): load_raw() and unpack() when the step computes, and under SwiGLU
// the activations staged ahead with cp.async. Every other instantiation is the plain kernel.
// The CTAs per SM the kernel is built for: the schedule's, and for a finish decoder (the wide
// Q6_K) on one m16 tile of one row set at least as many as 128 registers a thread allow, the
// plain kernels' size there (ptxas for sm_89 gave the wide w1x4 159 registers against 124: three
// CTAs per SM against four).
template <class Decoder, class Schedule, Epilogue E>
inline constexpr int kRmmaMinBlocks =
    FinishOf<Decoder>::value && E != Epilogue::SwiGlu && Schedule::kMTiles == 1 &&
            65536 / (Schedule::kThreads * 128) > Schedule::kMinBlocks
        ? 65536 / (Schedule::kThreads * 128)
        : Schedule::kMinBlocks;

template <class Decoder, class Schedule, Epilogue E, bool Latency = false>
__global__ __launch_bounds__(Schedule::kThreads, (kRmmaMinBlocks<Decoder, Schedule, E>)) void
    rmma_kernel(
    GemmArgs a) {
    constexpr int R        = Decoder::kRun;
    constexpr int KStep    = 4 * R;
    constexpr int J        = R / 4; // MMAs per K step
    constexpr int MT       = Schedule::kMTiles;
    constexpr int NT       = Schedule::kNTiles;
    constexpr int WK       = Schedule::kWarpsK;
    constexpr bool kSwiGlu = E == Epilogue::SwiGlu;
    constexpr int kSets    = kSwiGlu ? 2 : 1; // gate rows, then the matching up rows
    using Run              = typename Decoder::Run;

    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int warp_m = warp / WK;
    const int warp_k = warp - warp_m * WK;
    const int quad   = lane & 3;
    const int group  = lane >> 2;
    const int local0 = static_cast<int>(blockIdx.x) * Schedule::kRowsPerCta + warp_m * MT * 16;
    const int token0 = static_cast<int>(blockIdx.y) * (NT * 8);
    constexpr int C  = Schedule::kCluster;
    // This CTA's K steps: all of them, or its cluster rank's share (blockIdx.z is the rank).
    const int all_steps = (a.k + KStep - 1) / KStep;
    const int s_begin   = C == 1 ? 0 : (all_steps * static_cast<int>(blockIdx.z)) / C;
    const int steps     = C == 1 ? all_steps : (all_steps * (static_cast<int>(blockIdx.z) + 1)) / C;
    const auto* x    = static_cast<const __nv_bfloat16*>(a.x);

    // Call::release_next: the next call (a joined one) may launch once every CTA has started.
    // Under GemmArgs::pdl_prefetch that waits until after this CTA's own wait below: a joined call
    // reads x without waiting, and x is the previous kernel's.
    if constexpr (E == Epilogue::Store) {
        if (a.release_next != 0 && a.pdl_prefetch == 0 && threadIdx.x == 0) {
            pdl::trigger_dependents();
        }
    }

    if constexpr (TableOf<Decoder>::value) {
        Decoder::stage();
        __syncthreads();
    }

    float acc[kSets][MT][NT][4];
#pragma unroll
    for (int s = 0; s < kSets; ++s)
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int n = 0; n < NT; ++n)
                acc[s][m][n][0] = acc[s][m][n][1] = acc[s][m][n][2] = acc[s][m][n][3] = 0.0F;
#if NINFERNO_FMA_MMA
    // A one-tile schedule sums lane by lane (mma_step_local) for a tile of up to four tokens, the
    // quads' sums added at the end; wider tiles keep the emulated MMA (V100, Gemma 4 31B's Q5_K
    // head: 9.45 -> 6.25 ms at T = 1, 9.87 -> 12.1 ms lane by lane at T = 8).
    constexpr bool kLocal = NT == 1;
    float local_acc[kLocal ? kSets : 1][kLocal ? MT : 1][2][8];
    if constexpr (kLocal) {
#    pragma unroll
        for (int s = 0; s < kSets; ++s)
#    pragma unroll
            for (int m = 0; m < MT; ++m)
#    pragma unroll
                for (int n = 0; n < 8; ++n) local_acc[s][m][0][n] = local_acc[s][m][1][n] = 0.0F;
    }
    const int local_tokens = min(8, a.tokens - token0);
    const bool local       = kLocal && local_tokens <= 4 && a.lane_local != 0;
#endif

    Run ring[2][kSets][MT][2];

    // A K-quant SwiGLU step of one token tile (the verify widths, T <= 8) also copies its
    // activations ahead, with cp.async into shared memory, each lane the 8-byte chunks its own
    // MMAs read (mma_step, Staged), so no lane reads another's and the wait is the lane's own.
    // Loaded in the step itself they were the second long-scoreboard stall after the weights (ncu,
    // RTX 3060, Q4_K SwiGLU 12288x4096 T = 8: a quarter of the stall samples on the MMAs reading
    // them); staged, that call took 263 -> 188 us on w1x8 and Q5_K's 303 -> 232. Store and
    // add, one row set and half the weight per activation, lost 5-20 % with it (the shared memory
    // comes out of L1, which the quad's weight words hit): they load in the step.
    constexpr bool kRaw      = Latency && UnpackOf<Decoder>::value;
    constexpr bool kActStage = Latency && StageActsOf<Decoder>::value && kSwiGlu && NT == 1 &&
                               Schedule::kWarpsM * WK <= 8 && !GroupTailOf<Decoder>::value &&
                               !QuadRunsOf<Decoder>::value;
#if NINFERNO_FMA_MMA || !NINFERNO_HAS_CP_ASYNC
    constexpr bool kStaged = false; // no cp.async (sm_75 copies synchronously, memory.cuh)
#else
    constexpr bool kStaged = kActStage;
#endif
#ifndef NINFERNO_RMMA_AHEAD_EXP
#    define NINFERNO_RMMA_AHEAD_EXP 0
#endif
#if NINFERNO_FMA_MMA
    constexpr bool kAhead = false;
#else
    constexpr bool kAhead = Latency && !kStaged && SpreadOf<Decoder> && NT == 1 &&
                            !FinishOf<Decoder>::value && !GroupTailOf<Decoder>::value &&
                            !QuadRunsOf<Decoder>::value &&
                            ((kSwiGlu && (NINFERNO_RMMA_AHEAD_EXP & 1)) ||
                             (!kSwiGlu && (NINFERNO_RMMA_AHEAD_EXP & 2)));
#endif
    unsigned act_ring[kAhead ? 2 : 1][1][kAhead ? 2 * J : 1];
    __shared__ uint2 act_stage[kStaged ? 2 : 1][kStaged ? Schedule::kWarpsM * WK : 1]
                              [kStaged ? J : 1][32];
    const auto stage_act = [&](auto slot_tag, int step) {
        if constexpr (kStaged) {
            constexpr int slot = decltype(slot_tag)::value;
            const int token    = token0 + group;
            const int base     = step * KStep;
            const bool live    = token < a.tokens && base < a.k;
            const __nv_bfloat16* src = x + (live ? static_cast<std::int64_t>(token) * a.ldx : 0);
#pragma unroll
            for (int c = 0; c < J; ++c) {
                cp_async_zfill<8>(&act_stage[slot][warp][c][lane], src + base + 16 * c + 4 * quad,
                                  live ? 8 : 0);
            }
            cp_commit();
        } else if constexpr (kAhead) {
            constexpr int slot = decltype(slot_tag)::value;
            const int token    = token0 + group;
            const __nv_bfloat16* const c[1] = {
                token < a.tokens && step * KStep + quad * R < a.k
                    ? x + static_cast<std::int64_t>(token) * a.ldx
                    : nullptr};
            load_act<Decoder, 1, J>(act_ring[slot], c, step * KStep, quad);
        }
    };

    // `stage`: the step's activations copied too (the first step's wait for x under pdl_prefetch).
    const auto fetch = [&](auto slot_tag, int step, bool stage = true) {
        constexpr int slot = decltype(slot_tag)::value;
        const int k0       = step * KStep + quad * R;
        const bool k_ok    = (GroupTailOf<Decoder>::value ? step * KStep : k0) < a.k;
        if (stage) stage_act(slot_tag, step);
#pragma unroll
        for (int s = 0; s < kSets; ++s)
#pragma unroll
            for (int m = 0; m < MT; ++m)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int l   = local0 + m * 16 + h * 8 + group;
                    const int row = a.row_begin + l + (s == 1 ? a.up_offset : 0);
                    ring[slot][s][m][h] = l >= a.rows ? Decoder::zero()
                                          : k_ok      ? load_run<Decoder, kRaw>(a, row, k0)
                                                      : Decoder::masked(a, row, step * KStep);
                }
    };

    // `ahead`: the next step's copy was issued after this one's (one younger group to leave).
    const auto compute = [&](auto slot_tag, int step, bool ahead) {
        constexpr int slot = decltype(slot_tag)::value;
        const int k0       = (GroupTailOf<Decoder>::value ? 0 : quad * R) + step * KStep;
        // The step's runs unpacked. SwiGLU unpacks the ring in place: an unpacked copy kept both
        // forms live and spilled its 128-register decode tile (Q5_K w1x8: 80 local loads and
        // stores, 303 -> 435 us at T = 8, RTX 3060). One row set unpacks a copy, which ptxas
        // schedules better there (Q5_K store 8192x4096 w1x4 at T = 8: 81.9 us against 92.9 in
        // place; Q4_K even).
        constexpr bool kCopy = kRaw && !kSwiGlu;
        Run unpacked[kCopy ? kSets : 1][kCopy ? MT : 1][2];
        if constexpr (kRaw) {
#pragma unroll
            for (int s = 0; s < kSets; ++s)
#pragma unroll
                for (int m = 0; m < MT; ++m)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        if constexpr (kCopy) {
                            unpacked[s][m][h] = ring[slot][s][m][h];
                            Decoder::unpack(unpacked[s][m][h], step * KStep);
                        } else {
                            Decoder::unpack(ring[slot][s][m][h], step * KStep);
                        }
                    }
        }
        const Run(&runs)[kSets][MT][2] = [&]() -> const Run(&)[kSets][MT][2] {
            if constexpr (kCopy) {
                return unpacked;
            } else {
                return ring[slot];
            }
        }();
#if NINFERNO_FMA_MMA
        if constexpr (kLocal) {
            if (local) {
            const __nv_bfloat16* tcol[8];
#    pragma unroll
            for (int n = 0; n < 8; ++n) {
                tcol[n] = n < local_tokens && k0 < a.k
                              ? x + static_cast<std::int64_t>(token0 + n) * a.ldx
                              : nullptr;
            }
            mma_step_local<Decoder, J, kSets, MT>(local_acc, runs, tcol, local_tokens,
                                                  step * KStep, quad, a.k);
            return;
            }
        }
#endif
        const __nv_bfloat16* col[NT];
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int token = token0 + n * 8 + group;
            col[n] = token < a.tokens && k0 < a.k ? x + static_cast<std::int64_t>(token) * a.ldx
                                                  : nullptr;
        }
        if constexpr (kStaged) {
            if (ahead) {
                cp_wait<1>();
            } else {
                cp_wait<0>();
            }
            mma_step<Decoder, NT, J, kSets, MT, true>(acc, runs, col, step * KStep, quad, a.k,
                                                      &act_stage[slot][warp][0][lane]);
        } else if constexpr (kAhead) {
            (void)ahead;
            mma_step<Decoder, NT, J, kSets, MT, false, true>(acc, runs, col, step * KStep, quad,
                                                             a.k, nullptr, act_ring[slot][0]);
        } else {
            (void)ahead;
            mma_step<Decoder, NT, J, kSets, MT>(acc, runs, col, step * KStep, quad, a.k);
        }
    };

    using Slot0 = std::integral_constant<int, 0>;
    using Slot1 = std::integral_constant<int, 1>;
    if (s_begin + warp_k < steps) fetch(Slot0{}, s_begin + warp_k, false);
    // GemmArgs::pdl_prefetch: the first step's weights are in flight; x and the output are the
    // previous kernel's to finish. Then the next launch may start: it waits for this one before
    // it reads what this one writes, and a joined call (which reads x without waiting) starts
    // only once x is complete.
    if (a.pdl_prefetch != 0) {
        pdl::wait_for_dependencies();
        if (threadIdx.x == 0) pdl::trigger_dependents();
    }
    // The first step's staged activations: x copied only after that wait.
    if (s_begin + warp_k < steps) stage_act(Slot0{}, s_begin + warp_k);
#pragma unroll 1
    for (int step = s_begin + warp_k; step < steps; step += 2 * WK) {
        if (step + WK < steps) fetch(Slot1{}, step + WK);
        compute(Slot0{}, step, step + WK < steps);
        if (step + WK >= steps) break;
        if (step + 2 * WK < steps) fetch(Slot0{}, step + 2 * WK);
        compute(Slot1{}, step + WK, step + 2 * WK < steps);
    }
#if NINFERNO_FMA_MMA
    if constexpr (kLocal) {
        if (local) {
        // The quad's four K positions summed; lane t keeps tokens 2t and 2t + 1 (the C fragment).
#    pragma unroll
        for (int s = 0; s < kSets; ++s)
#    pragma unroll
            for (int m = 0; m < MT; ++m)
#    pragma unroll
                for (int h = 0; h < 2; ++h)
#    pragma unroll
                    for (int n = 0; n < 8; ++n) {
                        float v = local_acc[s][m][h][n];
                        v += __shfl_xor_sync(0xffffffffU, v, 1);
                        v += __shfl_xor_sync(0xffffffffU, v, 2);
                        if (n == 2 * quad) acc[s][m][0][2 * h] = v;
                        if (n == 2 * quad + 1) acc[s][m][0][2 * h + 1] = v;
                    }
        }
    }
#endif

    // Warps with warp_k > 0 hand their partials to warp_k 0; under a cluster they stay (every thread
    // takes part in the cluster barriers) but do no further work.
    bool live = true;
    if constexpr (WK > 1) {
        __shared__ float partial[WK - 1][Schedule::kWarpsM][kSets][MT][NT][4][32];
        if (warp_k > 0) {
#pragma unroll
            for (int s = 0; s < kSets; ++s)
#pragma unroll
                for (int m = 0; m < MT; ++m)
#pragma unroll
                    for (int n = 0; n < NT; ++n)
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                            partial[warp_k - 1][warp_m][s][m][n][e][lane] = acc[s][m][n][e];
        }
        __syncthreads();
        if (warp_k != 0) {
            if constexpr (C == 1) return;
            live = false;
        }
        if (live) {
#pragma unroll
            for (int w = 0; w < WK - 1; ++w)
#pragma unroll
                for (int s = 0; s < kSets; ++s)
#pragma unroll
                    for (int m = 0; m < MT; ++m)
#pragma unroll
                        for (int n = 0; n < NT; ++n)
#pragma unroll
                            for (int e = 0; e < 4; ++e)
                                acc[s][m][n][e] += partial[w][warp_m][s][m][n][e][lane];
        }
    }
    if constexpr (C > 1) {
// cooperative_groups::cluster_group does not exist below sm_90 and nvcc parses the discarded
// if constexpr branch too (the port's standing rule): the preprocessor carves it out on Ada,
// where no C > 1 schedule is instantiated anyway (generic_gemm_launch.cuh).
#ifndef NINFERNO_WITHOUT_NVFP4
        namespace cg = cooperative_groups;
        const cg::cluster_group cluster = cg::this_cluster();
        __shared__ float rank_partial[Schedule::kWarpsM][kSets][MT][NT][4][32];
        if (live) {
#pragma unroll
            for (int s = 0; s < kSets; ++s)
#pragma unroll
                for (int m = 0; m < MT; ++m)
#pragma unroll
                    for (int n = 0; n < NT; ++n)
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                            rank_partial[warp_m][s][m][n][e][lane] = acc[s][m][n][e];
        }
        cluster.sync();
        const bool rank0 = cluster.block_rank() == 0;
        if (rank0 && live) {
#pragma unroll 1
            for (int r = 1; r < C; ++r) {
                const auto& other = *cluster.map_shared_rank(&rank_partial, r);
#pragma unroll
                for (int s = 0; s < kSets; ++s)
#pragma unroll
                    for (int m = 0; m < MT; ++m)
#pragma unroll
                        for (int n = 0; n < NT; ++n)
#pragma unroll
                            for (int e = 0; e < 4; ++e)
                                acc[s][m][n][e] += other[warp_m][s][m][n][e][lane];
            }
        }
        // No CTA leaves (and frees its shared memory) before rank 0 has read it.
        cluster.sync();
        if (!rank0 || !live) return;
#endif
    }

    // C fragment: rows group and group + 8 of each m16 tile, tokens 2*quad and 2*quad + 1.
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int l = local0 + m * 16 + group + (e >= 2 ? 8 : 0);
            if (l >= a.rows) continue;
            const float scale0 = Decoder::row_scale(a, a.row_begin + l);
            float scale1       = 0.0F;
            if constexpr (kSwiGlu) scale1 = Decoder::row_scale(a, a.row_begin + l + a.up_offset);
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const int token = token0 + n * 8 + 2 * quad + (e & 1);
                if (token >= a.tokens) continue;
                float value = acc[0][m][n][e] * scale0;
                if constexpr (kSwiGlu) {
                    value = generic_gate(a, value) * (acc[kSets - 1][m][n][e] * scale1);
                }
                store_output<E>(a, l, token, value);
            }
        }
    }
    // A joined launch (Call::join_previous) ran alongside the kernel before it; the grid completes
    // only after that one has, so the stream's next kernel follows both.
    if constexpr (E == Epilogue::Store) {
        if (a.join_previous != 0) pdl::wait_for_dependencies();
    }
}

// Scalar fallback: one warp per (row, token), lanes over K. Correct for every K, slow.
template <Epilogue E>
__global__ __launch_bounds__(256) void scalar_kernel(GemmArgs a, WeightCodec codec) {
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const std::int64_t warp =
        static_cast<std::int64_t>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5);
    const std::int64_t total = static_cast<std::int64_t>(a.rows) * a.tokens;
    if (warp >= total) return;
    const int l            = static_cast<int>(warp % a.rows);
    const int token        = static_cast<int>(warp / a.rows);
    const auto* x          = static_cast<const __nv_bfloat16*>(a.x) + static_cast<std::int64_t>(token) * a.ldx;
    constexpr int kSets    = E == Epilogue::SwiGlu ? 2 : 1;
    float result[2]        = {0.0F, 0.0F};
#pragma unroll
    for (int s = 0; s < kSets; ++s) {
        const int row = a.row_begin + l + (s == 1 ? a.up_offset : 0);
        float sum     = 0.0F;
        for (int k = lane; k < a.k; k += 32) {
            sum = fmaf(decode_element(a, codec, row, k), __bfloat162float(x[k]), sum);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffU, sum, o);
        result[s] = sum;
    }
    if (lane != 0) return;
    float value = result[0];
    if constexpr (E == Epilogue::SwiGlu) value = generic_gate(a, value) * result[1];
    store_output<E>(a, l, token, value);
}

} // namespace ninferno::ops::detail::generic
