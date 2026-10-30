#pragma once

#include "ops/common/memory.cuh"

#include <cuda_fp16.h>

namespace ninferno::ops {

// An int32 below 2^22 in magnitude (an int8 dot product of up to 256 terms) as a float, exactly
// what static_cast<float> gives, without the conversion instruction. The W4A8 tile's epilogue
// converted four MMA results per m16n8k32 s8 (a sm_86 build of the superblock codecs: 13878
// I2FP.F32.S32 for 3456 IMMA). A loop of int32 -> float conversions each feeding an FFMA
// (a micro-benchmark) runs 1.25x faster this way on an RTX 3090 (1.53 against 1.91 ms) and the
// same on an RTX 5090. The accumulator starts at kFloatBiasInt, the bits of 1.5 * 2^23, instead of
// 0: the MMA (or dp4a) adds the dot product into its low mantissa bits, and biased_int_to_float
// subtracts 1.5 * 2^23 again, exactly.
inline constexpr int kFloatBiasInt = 0x4B400000;

__device__ __forceinline__ float biased_int_to_float(int biased) {
    return __int_as_float(biased) - 12582912.0F;
}

// Where the biased accumulators pay. Measured against I2FP (prefill A/B, same binary otherwise):
// RTX 4090, 16K tokens: Muse 30B -1 %, Qwen3.8 27B -1.5 %, Gemma 4 31B -8 %; RTX 5080
// (the IADD form) 27B UD-IQ4_XS -2.4 %; RTX 5080 (7K): Qwen3.6 35B-A3B -3.5 %,
// Qwen3.8 -2 %, Gemma 4 31B -1.4 %. The RTX 2080 Ti gains: Qwen3.5 9B Q4_K_M prefill, A B B A in
// one tree, 8K tokens +3.3 %, 2.4K +4.6 %, C = 4 decode even to +6 %. The micro-benchmark's
// x1.25 on the RTX 3090 (an I2FP feeding an FFMA) did not carry over to the kernels on Ada and
// Blackwell, and Ampere has no kernel A/B yet. NINFERNO-SPECIAL(arch:sm75) — list an
// architecture here only with a kernel A/B on it.
// NINFERNO-SPECIAL(arch:sm70): Volta, below sm_80 too, takes it without a kernel A/B of its own.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
inline constexpr bool kBiasedIntAccumulators = true;
#else
inline constexpr bool kBiasedIntAccumulators = false;
#endif

// The value an int8 MMA or dp4a accumulator starts at, and its sum as a float: kFloatBiasInt and
// biased_int_to_float where kBiasedIntAccumulators, else 0 and the plain conversion (I2FP).
__device__ __forceinline__ constexpr int int_accumulator_zero() {
    return kBiasedIntAccumulators ? kFloatBiasInt : 0;
}

__device__ __forceinline__ float int_accumulator_to_float(int c) {
    if constexpr (kBiasedIntAccumulators) {
        return biased_int_to_float(c);
    } else {
        return static_cast<float>(c);
    }
}

// Before sm_75 (Volta) there is no ldmatrix and none of the MMA shapes this header issues (Volta's
// one shape, m8n8k4, splits the warp into quad-pairs with fragments unlike every later shape). The
// sm_70 build keeps the sm_80 fragments and computes them without tensor cores: an ldmatrix is a
// shuffle of the row addresses and a shared load per lane, the f16 / bf16 / tf32 products FP32
// FMAs over warp shuffles of the operands, the s8 products dp4a. Every lane of the warp takes
// part, as in the instruction each replaces; the sums are the MMA's or exacter (FP32 operands and
// accumulation throughout, BF16 at full range).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
#    define NINFERNO_FMA_MMA 1
#else
#    define NINFERNO_FMA_MMA 0
#endif

#if NINFERNO_FMA_MMA
__device__ __forceinline__ unsigned shared_load_b32(unsigned addr) {
    unsigned v;
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(addr));
    return v;
}

__device__ __forceinline__ unsigned shared_load_b16(unsigned addr) {
    unsigned short v;
    asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(v) : "r"(addr));
    return v;
}

// Matrix m of an ldmatrix: lanes 8m..8m+7 hold its row addresses; lane (g, t) = (lane / 4,
// lane % 4) receives row g's b16 elements 2t and 2t + 1.
__device__ __forceinline__ unsigned ldmatrix_fma(unsigned addr, int m) {
    const int lane = threadIdx.x & 31;
    return shared_load_b32(__shfl_sync(0xffffffffU, addr, m * 8 + (lane >> 2)) + (lane & 3) * 4);
}

// The transposed form: lane (g, t) receives element g of rows 2t (low half) and 2t + 1.
__device__ __forceinline__ unsigned ldmatrix_t_fma(unsigned addr, int m) {
    const int lane     = threadIdx.x & 31;
    const int row      = m * 8 + 2 * (lane & 3);
    const unsigned col = (lane >> 2) * 2;
    const unsigned r0  = __shfl_sync(0xffffffffU, addr, row);
    const unsigned r1  = __shfl_sync(0xffffffffU, addr, row + 1);
    return shared_load_b16(r0 + col) | (shared_load_b16(r1 + col) << 16);
}
#endif

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
#if NINFERNO_FMA_MMA
    r0 = ldmatrix_fma(addr, 0);
    r1 = ldmatrix_fma(addr, 1);
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
#endif
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
#if NINFERNO_FMA_MMA
    r0 = ldmatrix_fma(addr, 0);
    r1 = ldmatrix_fma(addr, 1);
    r2 = ldmatrix_fma(addr, 2);
    r3 = ldmatrix_fma(addr, 3);
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
#endif
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
#if NINFERNO_FMA_MMA
    r0 = ldmatrix_t_fma(addr, 0);
    r1 = ldmatrix_t_fma(addr, 1);
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
#endif
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
#if NINFERNO_FMA_MMA
    r0 = ldmatrix_t_fma(addr, 0);
    r1 = ldmatrix_t_fma(addr, 1);
    r2 = ldmatrix_t_fma(addr, 2);
    r3 = ldmatrix_t_fma(addr, 3);
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
#endif
}

#if NINFERNO_FMA_MMA
// The m16n8k16 product of 16-bit operands in FP32 FMAs. Lane (g, t) holds A[g][2t..2t+1] (a0),
// A[g + 8][2t..2t+1] (a1), A[g][2t+8..2t+9] (a2), A[g + 8][2t+8..2t+9] (a3), B[2t..2t+1][g] (b0),
// B[2t+8..2t+9][g] (b1), and C[g][2t + j] / C[g + 8][2t + j]; `pair` turns a register into its two
// values (the lower index in the low half).
template <class Pair>
__device__ __forceinline__ void mma_k16_fma(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                            unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                            unsigned b1, Pair pair) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#    pragma unroll
    for (int kp = 0; kp < 8; ++kp) { // k = 2 kp + h, held by lanes of column group kp % 4
        const int src_a  = g * 4 + (kp & 3);
        const float2 lo  = pair(__shfl_sync(0xffffffffU, kp < 4 ? a0 : a2, src_a));
        const float2 hi  = pair(__shfl_sync(0xffffffffU, kp < 4 ? a1 : a3, src_a));
        const float2 bj0 = pair(__shfl_sync(0xffffffffU, kp < 4 ? b0 : b1, (2 * t) * 4 + (kp & 3)));
        const float2 bj1 =
            pair(__shfl_sync(0xffffffffU, kp < 4 ? b0 : b1, (2 * t + 1) * 4 + (kp & 3)));
        c0 = fmaf(lo.y, bj0.y, fmaf(lo.x, bj0.x, c0));
        c1 = fmaf(lo.y, bj1.y, fmaf(lo.x, bj1.x, c1));
        c2 = fmaf(hi.y, bj0.y, fmaf(hi.x, bj0.x, c2));
        c3 = fmaf(hi.y, bj1.y, fmaf(hi.x, bj1.x, c3));
    }
}

__device__ __forceinline__ float2 f16x2_pair(unsigned v) {
    return __half22float2(*reinterpret_cast<const __half2*>(&v));
}

__device__ __forceinline__ float2 bf16x2_pair(unsigned v) {
    return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000U));
}

// The emulated products are called, not inlined: ~100 instructions per MMA unrolled into every
// tile multiplied the sm_70 build's code (ptxas past 3 GB per generic GEMM translation unit). The
// operands travel in registers (float4 / uint4 / uint2 by value); the call costs a few moves.
static __device__ __noinline__ float4 mma_f16_fma_call(float4 c, uint4 a, uint2 b) {
    mma_k16_fma(c.x, c.y, c.z, c.w, a.x, a.y, a.z, a.w, b.x, b.y, f16x2_pair);
    return c;
}

static __device__ __noinline__ float4 mma_bf16_fma_call(float4 c, uint4 a, uint2 b) {
    mma_k16_fma(c.x, c.y, c.z, c.w, a.x, a.y, a.z, a.w, b.x, b.y, bf16x2_pair);
    return c;
}

// The s8 m16n8k32 (kChunks 8) or m16n8k16 (kChunks 4) product in dp4a: lane (g, t) holds
// A[g][4t..4t+3] (a0), A[g + 8][4t..4t+3] (a1), their k + 16 quads in a2 / a3, B[4t..4t+3][g]
// (b0) and B[4t+16..4t+19][g] (b1); C as in the f16 form. Integer and exact, as the MMA.
template <int kChunks>
__device__ __forceinline__ void mma_s8_dp4a(int& c0, int& c1, int& c2, int& c3, unsigned a0,
                                            unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                            unsigned b1) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#    pragma unroll
    for (int q = 0; q < kChunks; ++q) {
        const int src_a = g * 4 + (q & 3);
        const int ag    = static_cast<int>(__shfl_sync(0xffffffffU, q < 4 ? a0 : a2, src_a));
        const int ag8   = static_cast<int>(__shfl_sync(0xffffffffU, q < 4 ? a1 : a3, src_a));
        const int bn0 =
            static_cast<int>(__shfl_sync(0xffffffffU, q < 4 ? b0 : b1, (2 * t) * 4 + (q & 3)));
        const int bn1 =
            static_cast<int>(__shfl_sync(0xffffffffU, q < 4 ? b0 : b1, (2 * t + 1) * 4 + (q & 3)));
        c0 = __dp4a(ag, bn0, c0);
        c1 = __dp4a(ag, bn1, c1);
        c2 = __dp4a(ag8, bn0, c2);
        c3 = __dp4a(ag8, bn1, c3);
    }
}

template <int kChunks>
static __device__ __noinline__ int4 mma_s8_dp4a_call(int4 c, uint4 a, uint2 b) {
    mma_s8_dp4a<kChunks>(c.x, c.y, c.z, c.w, a.x, a.y, a.z, a.w, b.x, b.y);
    return c;
}

__device__ __forceinline__ void mma_fma_store(float& c0, float& c1, float& c2, float& c3,
                                              float4 r) {
    c0 = r.x, c1 = r.y, c2 = r.z, c3 = r.w;
}
#endif

// NINFERNO-SPECIAL(arch:sm75): Turing's tensor cores take f16 at m16n8k8 and s8 at m8n8k16 only.
// Its build composes the sm_80 shapes from those, fragment for fragment: a m16n8k16 A fragment's
// {a0, a1} are the k 0-7 half of a m16n8k8 fragment and {a2, a3} the k 8-15 half (B: b0, b1), so
// two m16n8k8 give the same sums; a s8 m16n8k32 is four m8n8k16 (rows g / g + 8 by k halves),
// integer and exact. BF16 has no Turing MMA: its operands are rounded to f16 (8 mantissa bits fit
// f16's 10; magnitudes saturate at 65504, below 2^-14 they turn subnormal), the form llama.cpp
// computes on Turing. Accumulation stays FP32.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 800
#    define NINFERNO_SM75_MMA 1
#else
#    define NINFERNO_SM75_MMA 0
#endif

// NINFERNO-SPECIAL(arch:sm70): Volta's tensor cores take one shape, mma.m8n8k4 with f16 operands.
// The warp runs four independent products, one per quad-pair (lanes 4q .. 4q + 3 and
// 16 + 4q .. 16 + 4q + 3): each lane holds one row of its quad-pair's A (row lane % 4, + 4 for the
// high lanes; a0 = k 0, 1 and a1 = k 2, 3) and one column of its B (the same index; b0 = k 0, 1,
// b1 = k 2, 3); the FP32 accumulator c[i] is row (lane & 1) + (i & 2) (+ 4 for the high lanes),
// column (i & 4) + (lane & 2) + (i & 1) (PTX ISA, mma.m8n8k4 fragments). Kernels with their own
// Volta fragment layout issue it (generic_tiled.cuh); the sm_80 fragments above stay on FMAs.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
#    define NINFERNO_HMMA884 1
#else
#    define NINFERNO_HMMA884 0
#endif

#if NINFERNO_HMMA884
__device__ __forceinline__ void mma_m8n8k4_f16(float (&c)[8], unsigned a0, unsigned a1, unsigned b0,
                                               unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]),
                   "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// The same product with B row-major: each lane holds row lane % 4 of its quad-pair's B, columns
// 0..3 (low lanes) or 4..7 (high lanes); b0 = columns 0, 1 and b1 = 2, 3 (PTX ISA, mma.m8n8k4).
__device__ __forceinline__ void mma_m8n8k4_f16_row_b(float (&c)[8], unsigned a0, unsigned a1,
                                                     unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]),
                   "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
#endif

#if NINFERNO_SM75_MMA
__device__ __forceinline__ void mma_f16_k8(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                           unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ void mma_s8_m8(int& c0, int& c1, unsigned a, unsigned b) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a), "r"(b));
}
#endif

// A finite bf16 value saturated to the f16 range; NaN and infinities pass as they are, as the
// bf16 MMA would carry them (fmaxf would turn a NaN into the lower bound).
__device__ __forceinline__ float bf16_value_in_f16_range(float v) {
    constexpr float kMax = 65504.0F;
    return isfinite(v) ? fminf(fmaxf(v, -kMax), kMax) : v;
}

// A bf16x2 register as f16x2, finite values saturated to the f16 range.
__device__ __forceinline__ unsigned bf16x2_to_f16x2(unsigned v) {
    const float lo  = bf16_value_in_f16_range(__uint_as_float(v << 16));
    const float hi  = bf16_value_in_f16_range(__uint_as_float(v & 0xffff0000U));
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
#if NINFERNO_FMA_MMA
    mma_fma_store(c0, c1, c2, c3,
                  mma_f16_fma_call(make_float4(c0, c1, c2, c3), make_uint4(a0, a1, a2, a3),
                                   make_uint2(b0, b1)));
#elif NINFERNO_SM75_MMA
    mma_f16_k8(c0, c1, c2, c3, a0, a1, b0);
    mma_f16_k8(c0, c1, c2, c3, a2, a3, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
#if NINFERNO_FMA_MMA
    mma_fma_store(c0, c1, c2, c3,
                  mma_bf16_fma_call(make_float4(c0, c1, c2, c3), make_uint4(a0, a1, a2, a3),
                                    make_uint2(b0, b1)));
#elif NINFERNO_SM75_MMA
    mma_f16(c0, c1, c2, c3, bf16x2_to_f16x2(a0), bf16x2_to_f16x2(a1), bf16x2_to_f16x2(a2),
            bf16x2_to_f16x2(a3), bf16x2_to_f16x2(b0), bf16x2_to_f16x2(b1));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// mma_bf16 for operands that leave the f16 range. On every build but Turing it is mma_bf16. On
// Turing the f16 conversion flushes magnitudes below 2^-24 to zero and saturates above 65504;
// operands scaled by accumulated decays (a per-channel decay prefix can reach 2^-100) vanish.
// NINFERNO-SPECIAL(arch:sm75): Turing computes the m16n8k16 product in FP32 FMAs over the warp's
// fragments instead. Lane (g, t) = (lane / 4, lane % 4) holds A[g][2t..2t+1] (a0),
// A[g + 8][2t..2t+1] (a1), A[g][2t+8..2t+9] (a2), A[g + 8][2t+8..2t+9] (a3), B[2t..2t+1][g] (b0),
// B[2t+8..2t+9][g] (b1), and C[g][2t + j] / C[g + 8][2t + j]. The low half of a bf16x2 register is
// the lower index. Every lane of the warp takes part, as in the MMA it replaces.
__device__ __forceinline__ void mma_bf16_full_range(float& c0, float& c1, float& c2, float& c3,
                                                    unsigned a0, unsigned a1, unsigned a2,
                                                    unsigned a3, unsigned b0, unsigned b1) {
#if NINFERNO_FMA_MMA
    mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#elif NINFERNO_SM75_MMA
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const auto half = [](unsigned v, int h) {
        return __uint_as_float(h == 0 ? v << 16 : v & 0xffff0000U);
    };
#    pragma unroll
    for (int kp = 0; kp < 8; ++kp) { // k = 2 kp + h, held by lanes of column group kp % 4
        const int src_a    = g * 4 + (kp & 3);
        const unsigned lo  = __shfl_sync(0xffffffffU, kp < 4 ? a0 : a2, src_a);
        const unsigned hi  = __shfl_sync(0xffffffffU, kp < 4 ? a1 : a3, src_a);
        const unsigned bj0 = __shfl_sync(0xffffffffU, kp < 4 ? b0 : b1, (2 * t) * 4 + (kp & 3));
        const unsigned bj1 = __shfl_sync(0xffffffffU, kp < 4 ? b0 : b1, (2 * t + 1) * 4 + (kp & 3));
#    pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float a_lo = half(lo, h), a_hi = half(hi, h);
            c0 = fmaf(a_lo, half(bj0, h), c0);
            c1 = fmaf(a_lo, half(bj1, h), c1);
            c2 = fmaf(a_hi, half(bj0, h), c2);
            c3 = fmaf(a_hi, half(bj1, h), c3);
        }
    }
#else
    mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#endif
}

// An A fragment a warp multiplies by many B fragments (attention's Q and P) converted once:
// mma_a_bf16 is a bf16x2 register as this build's BF16 MMA reads it, and mma_bf16_prepared takes A
// so prepared and B as bf16x2. The sums are those of mma_bf16 on the unconverted A.
// NINFERNO-SPECIAL(arch:sm75): only Turing converts (to f16, as mma_bf16 does on every call; A is
// four of the six registers, so a Q fragment reused over a whole key tile converted per MMA cost
// most of the conversion); every other build passes the register through.
__device__ __forceinline__ unsigned mma_a_bf16(unsigned v) {
#if NINFERNO_SM75_MMA
    return bf16x2_to_f16x2(v);
#else
    return v;
#endif
}

__device__ __forceinline__ void mma_bf16_prepared(float& c0, float& c1, float& c2, float& c3,
                                                  unsigned a0, unsigned a1, unsigned a2,
                                                  unsigned a3, unsigned b0, unsigned b1) {
#if NINFERNO_SM75_MMA
    mma_f16(c0, c1, c2, c3, a0, a1, a2, a3, bf16x2_to_f16x2(b0), bf16x2_to_f16x2(b1));
#else
    mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#endif
}

// Same shape as mma_f16 but with an FP16 accumulator: two result registers
// instead of four, and twice the arithmetic throughput of the f32-accumulate
// form on sm_120a (502.9 vs 253.4 TFLOP/s measured on an RTX 5090). The narrow
// accumulator carries about ten bits of mantissa, so callers must fold it into
// a wider running sum instead of carrying it across a long contraction.
__device__ __forceinline__ void mma_f16_acc16(unsigned& c0, unsigned& c1, unsigned a0, unsigned a1,
                                              unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if NINFERNO_FMA_MMA
    // c0 = C[g][2t..2t+1] and c1 = C[g + 8][2t..2t+1] as f16x2, summed in FP32 and rounded once.
    const float2 top0 = f16x2_pair(c0), bottom0 = f16x2_pair(c1);
    const float4 r   = mma_f16_fma_call(make_float4(top0.x, top0.y, bottom0.x, bottom0.y),
                                        make_uint4(a0, a1, a2, a3), make_uint2(b0, b1));
    const float2 top = make_float2(r.x, r.y), bottom = make_float2(r.z, r.w);
    const __half2 t2 = __floats2half2_rn(top.x, top.y);
    const __half2 b2 = __floats2half2_rn(bottom.x, bottom.y);
    c0               = *reinterpret_cast<const unsigned*>(&t2);
    c1               = *reinterpret_cast<const unsigned*>(&b2);
#elif NINFERNO_SM75_MMA
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(a1), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a2), "r"(a3), "r"(b1));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if NINFERNO_FMA_MMA
    const int4 r = mma_s8_dp4a_call<8>(make_int4(c0, c1, c2, c3), make_uint4(a0, a1, a2, a3),
                                       make_uint2(b0, b1));
    c0 = r.x, c1 = r.y, c2 = r.z, c3 = r.w;
#elif NINFERNO_SM75_MMA
    mma_s8_m8(c0, c1, a0, b0);
    mma_s8_m8(c0, c1, a2, b1);
    mma_s8_m8(c2, c3, a1, b0);
    mma_s8_m8(c2, c3, a3, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_s8_k16(int& c0, int& c1, int& c2, int& c3, unsigned a0,
                                           unsigned a1, unsigned b0) {
#if NINFERNO_FMA_MMA
    const int4 r = mma_s8_dp4a_call<4>(make_int4(c0, c1, c2, c3), make_uint4(a0, a1, 0U, 0U),
                                       make_uint2(b0, 0U));
    c0 = r.x, c1 = r.y, c2 = r.z, c3 = r.w;
#elif NINFERNO_SM75_MMA
    mma_s8_m8(c0, c1, a0, b0);
    mma_s8_m8(c2, c3, a1, b0);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
#endif
}

// NVFP4 block-scaled MMA is the one instruction family in this header Blackwell added: sm_89
// (Ada) lacks kind::mxf4nvf4 and kind::f8f6f4. On a non-120a device pass the NVFP4 wrapper is
// not defined at all — nvcc evaluates a function-body static_assert in an uninstantiated
// __forceinline__ device function, so an #else with an assert fires from a TU that merely
// includes the header. TUs that call it ship only in 120a builds (the nvfp4/ source filter);
// an Ada TU referencing it therefore fails at compile with "identifier not found", which names
// the site.
//
// The block-scaled kinds exist only on the architecture- and family-specific targets of consumer
// Blackwell (sm_120a, sm_121a, sm_120f), which nvcc marks with __CUDA_ARCH_FAMILY_SPECIFIC__ (1200
// for sm_120a and sm_120f, 1210 for sm_121a); a plain sm_120 or a later generation's plain
// target compiles the portable forms instead.
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FAMILY_SPECIFIC__) &&                            \
    __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1200 && __CUDA_ARCH_FAMILY_SPECIFIC__ < 1300
#    define NINFERNO_SM120_BLOCK_SCALED_MMA 1
#else
#    define NINFERNO_SM120_BLOCK_SCALED_MMA 0
#endif
#if !defined(__CUDA_ARCH__) || NINFERNO_SM120_BLOCK_SCALED_MMA
#    define NINFERNO_HAS_NVFP4_MMA 1
#else
#define NINFERNO_HAS_NVFP4_MMA 0
#endif

// FP8 E4M3 x E4M3 -> FP32. NINFERNO-SPECIAL(arch:sm120): on consumer Blackwell the block-scaled
// form avoids the plain form's half-rate FP32 accumulation; UE8M0 0x7f is exactly 1, so it computes
// the raw E4M3 dot product and callers keep their row scales in FP32 after the MMA. From sm_89 the
// un-prefixed m16n8k32 spelling is the same product (probe-verified on an RTX 4090, CUDA 13.1).
// Before sm_89 there is no E4M3 MMA and the product is emulated below. The FP8 KV storages score
// through it on every card; the FP8-activation linear routes still require
// DeviceFeatures::mma_fp8 (fp8_a8_allowed), where the A16 routes are faster than the emulation.
__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
#if NINFERNO_SM120_BLOCK_SCALED_MMA
    constexpr unsigned kUnitScale8          = 0x7Fu;
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                 "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                 "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(kUnitScale8),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(kUnitScale8), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    // Below sm_89 the same product runs as two FP16 m16n8k16 MMAs (mma_f16 carries the sm_75 and
    // FMA forms). An E4M3 code is exact in FP16: its sign to bit 15 and its seven exponent and
    // mantissa bits one place down give the code's value times 2^-8, normals and subnormals alike,
    // so every product is exact and 2^-16 too small, and the sum is FP32 as on sm_89. Slots
    // (2t, 2t+1) of an FP16 fragment take the real k (4t, 4t+1) of an E4M3 register and slots
    // (2t+8, 2t+9) take (4t+2, 4t+3); A and B are permuted alike, so the dot product is the same.
    const auto low = [](unsigned x) {
        const unsigned h = __byte_perm(x, 0u, 0x1404);
        return (h & 0x80008000u) | ((h >> 1) & 0x3f803f80u);
    };
    const auto high = [](unsigned x) {
        const unsigned h = __byte_perm(x, 0u, 0x3424);
        return (h & 0x80008000u) | ((h >> 1) & 0x3f803f80u);
    };
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    mma_f16(d0, d1, d2, d3, low(a0), low(a1), high(a0), high(a1), low(b0), high(b0));
    mma_f16(d0, d1, d2, d3, low(a2), low(a3), high(a2), high(a3), low(b1), high(b1));
    c0 = fmaf(d0, 65536.0f, c0);
    c1 = fmaf(d1, 65536.0f, c1);
    c2 = fmaf(d2, 65536.0f, c2);
    c3 = fmaf(d3, 65536.0f, c3);
#endif
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
#if NINFERNO_SM75_MMA || NINFERNO_FMA_MMA
    // No TF32 MMA before sm_80: the m16n8k8 product in FP32 FMAs over the warp's fragments.
    // Lane (g, t) = (lane / 4, lane % 4) holds A[g][t] (a0), A[g + 8][t] (a1), A[g][t + 4] (a2),
    // A[g + 8][t + 4] (a3), B[t][g] (b0), B[t + 4][g] (b1), and C[g][2t + j] / C[g + 8][2t + j].
    // The operands keep all FP32 bits (TF32 would drop 13 of them): at least as exact.
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#    pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int src_a  = g * 4 + (k & 3);
        const float a_lo = __uint_as_float(__shfl_sync(0xffffffffU, k < 4 ? a0 : a2, src_a));
        const float a_hi = __uint_as_float(__shfl_sync(0xffffffffU, k < 4 ? a1 : a3, src_a));
#    pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int n = 2 * t + j;
            const float b =
                __uint_as_float(__shfl_sync(0xffffffffU, k < 4 ? b0 : b1, n * 4 + (k & 3)));
            if (j == 0) {
                c0 = fmaf(a_lo, b, c0);
                c2 = fmaf(a_hi, b, c2);
            } else {
                c1 = fmaf(a_lo, b, c1);
                c3 = fmaf(a_hi, b, c3);
            }
        }
    }
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

#if NINFERNO_HAS_NVFP4_MMA
__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}

// E4M3 (A, row) x E2M1 (B, col; each code in bits 2..5 of its byte) -> FP32 with one UE8M0 scale
// per row of A and per column of B over the k32 step: D += (A * 2^(sfa - 127)) (B * 2^(sfb - 127)).
// Byte `Byte` of sfa / sfb holds the step's scales (so one register carries four steps); lanes
// with lane % 4 == 0 give row g of A and column g of B, lane % 4 == 1 row g + 8 of A (thread-id 0;
// probe-verified on an RTX 5090, CUDA 13.2).
template <int Byte>
__device__ __forceinline__ void
mma_mxf8_e4m3_e2m1(float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1,
                   unsigned a2, unsigned a3, unsigned b0, unsigned b1, unsigned sfa, unsigned sfb) {
    static_assert(Byte >= 0 && Byte < 4);
    constexpr unsigned short kByte     = Byte;
    constexpr unsigned short kThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                 "m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                 "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"(kByte),
                   "h"(kThreadId), "r"(sfb), "h"(kByte), "h"(kThreadId));
}
#endif

} // namespace ninferno::ops
