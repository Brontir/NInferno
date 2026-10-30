#pragma once

// Generic W4A8: 2- to 8-bit weights of any [N,K] (K % 128 == 0) against activations
// quantized to int8 with one binary32 scale per 32 values (and the quantized block's sum, for the
// formats with a min term), on int8 tensor cores (mma m16n8k32 s8, int32 accumulators).
//
// Every weight format here is an integer code times a scale per 32 values (plus a per-32 min;
// Q6_K, Q3_K, IQ2_XS and IQ2_S: a scale per 16; Q2_K: a scale and a min per 16):
// Q4_0 d*(q - 8), Q4_1 d*q + m, Q5_0 d*(q - 16), Q5_1 d*q + m, IQ4_NL / IQ4_XS d*kvalues[q], Q4_K
// (d*sc)*q - dmin*m, Q5_K the same with five bits and u4_g128_zp s*(q - z) (the 128-value group's
// scale on each of its 32-value blocks), Q6_K (d*sc)*(q - 32), Q3_K (d*(sc - 32))*q with q in
// -4 .. 3, Q2_K (d*sc)*q - dmin*m with q in 0 .. 3 and a scale and min per 16, q8_g32 (GGUF
// Q8_0) d*q, and the
// codebook formats IQ3_XXS, IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S: d*f(ls)*grid*sign with grid magnitudes
// below 64, so the signed grid value is the int8 code. So one 32-value block of a row and one of
// a token make one int32 dot product per MMA, which becomes FP32 times the weight's and the
// activation's scales: sum_blocks s_w*s_x*dot(q_w, q_x) (+ m_w*s_x*sum(q_x)), which is sum w*x_q.
// The CTA decodes its weight rows to int8 in shared memory once per 128-value K tile (the decode
// cost is shared by every token of the tile), and the MMA loop reads both operands with ldmatrix. K
// tiles accumulate in order and no schedule splits K, so a token's bytes do not depend on the
// call's T.

#include "core/device.h"
#include "ops/common/arch_limits.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/generic/generic_gemm.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninferno::ops::detail::generic {

template <int BlockM, int BlockN, int WarpsM, int WarpsN>
struct W4a8Schedule {
    static_assert(BlockM % (16 * WarpsM) == 0 && BlockN % (8 * WarpsN) == 0);
    static constexpr int kBlockM  = BlockM; // tokens
    static constexpr int kBlockN  = BlockN; // weight rows
    static constexpr int kWarpsM  = WarpsM;
    static constexpr int kWarpsN  = WarpsN;
    static constexpr int kThreads = WarpsM * WarpsN * 32;
    static constexpr int kWarpM   = BlockM / WarpsM;
    static constexpr int kWarpN   = BlockN / WarpsN;
    static constexpr int kMmaM    = kWarpM / 16;
    static constexpr int kMmaN    = kWarpN / 8;
    static constexpr int kBlocks  = 4; // 32-value blocks per 128-value K tile
    // One buffer: int8 codes (128 per row) and per 32-value block two floats (scale and min / sum).
    static constexpr int kBufferBytes = (BlockM + BlockN) * (128 + 32);
    static constexpr int kSmemBytes   = 2 * kBufferBytes;
};

// The activation quantization of one 32-value block, shared by the W4A8 quantizer and the register
// int8 decode kernel so both make the same codes: rint(x*127/amax) clamped to +-127, scale
// amax/127 (0 for an all-zero block) and the codes' integer sum. `words` receives the codes four
// per word, in value order.
__device__ __forceinline__ void w4a8_quantize_block(const float (&v)[32], unsigned (&words)[8],
                                                    float& scale, int& qsum) {
    float amax = 0.0F;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        amax = fmaxf(amax, fabsf(v[i]));
    }
    const float inv = amax > 0.0F ? 127.0F / amax : 0.0F;
    qsum            = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        unsigned w = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = max(-127, min(127, __float2int_rn(v[4 * i + j] * inv)));
            w |= (static_cast<unsigned>(q) & 0xffU) << (8 * j);
            qsum += q;
        }
        words[i] = w;
    }
    scale = amax / 127.0F;
}

// The 32 BF16 values at `src` (16-byte aligned) as floats.
__device__ __forceinline__ void w4a8_load_block(const __nv_bfloat16* src, float (&v)[32]) {
    const auto* p = reinterpret_cast<const uint4*>(src);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint4 w       = p[i];
        const unsigned u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[8 * i + 2 * j]     = __uint_as_float(u[j] << 16);
            v[8 * i + 2 * j + 1] = __uint_as_float(u[j] & 0xffff0000U);
        }
    }
}

// Four magnitude bytes (each 1..127) as int8 bytes, negated where bits 0..3 of `signs` are set:
// -m is ~m + 1, at most 255 for m >= 1, so one word add carries no byte into the next (every IQ
// grid's magnitudes are at least 1). __vsub4 is four instructions from sm_75 on; this is two.
__device__ __forceinline__ unsigned signed_bytes(unsigned magnitudes, unsigned signs) {
    const unsigned ones = ((signs & 0xfU) * 0x00204081U) & 0x01010101U;
    return (magnitudes ^ (ones * 0xffU)) + ones;
}

// ggml's ksigns_iq2xs[t]: the seven sign bits t with their parity as the eighth.
__device__ __forceinline__ unsigned ksigns_byte(unsigned t) {
    return t | ((static_cast<unsigned>(__popc(t)) & 1U) << 7);
}


// 16-byte chunk c of a 128-byte shared row r sits at chunk c ^ (r % 8): eight rows of one
// ldmatrix column land in eight distinct bank groups.
__device__ __forceinline__ int w4a8_chunk(int row, int chunk) { return chunk ^ (row & 7); }

// Four codes (bytes of q, each < 16) looked up in kvalues_iq4nl as int8 bytes.
__device__ __forceinline__ unsigned iq4nl_s8(unsigned q) {
    constexpr unsigned kW0 = 0xbfad9881U; // -127, -104, -83, -65
    constexpr unsigned kW1 = 0xf6eadDcfU; // -49, -35, -22, -10
    constexpr unsigned kW2 = 0x26190d01U; // 1, 13, 25, 38
    constexpr unsigned kW3 = 0x71594535U; // 53, 69, 89, 113
    const unsigned t   = q | (q >> 4);
    const unsigned c   = __byte_perm(t, 0U, 0x4420U); // nibble n: code n
    const unsigned s   = c & 0x7777U;
    const unsigned sel = ((c & 0x8888U) >> 1) | 0x3210U;
    return __byte_perm(__byte_perm(kW0, kW1, s), __byte_perm(kW2, kW3, s), sel);
}

// The eight int8 words of a 32-value block from its ggml nibble words (value j: low nibble of
// byte j for j < 16, high nibble of byte j - 16), minus `bias` per byte.
__device__ __forceinline__ void ggml_nibbles_s8(const uint4& q, unsigned bias, uint4& lo,
                                                uint4& hi) {
    lo = make_uint4(__vsub4(q.x & 0x0f0f0f0fU, bias), __vsub4(q.y & 0x0f0f0f0fU, bias),
                    __vsub4(q.z & 0x0f0f0f0fU, bias), __vsub4(q.w & 0x0f0f0f0fU, bias));
    hi = make_uint4(__vsub4((q.x >> 4) & 0x0f0f0f0fU, bias),
                    __vsub4((q.y >> 4) & 0x0f0f0f0fU, bias),
                    __vsub4((q.z >> 4) & 0x0f0f0f0fU, bias),
                    __vsub4((q.w >> 4) & 0x0f0f0f0fU, bias));
}

// Four fifth bits (bits 0..3 of b) as 0x10 in bytes 0..3.
__device__ __forceinline__ unsigned fifth_bits(unsigned b) {
    return ((b * 0x00204081U) & 0x01010101U) << 4;
}

// Four two-bit fields (bits 2i..2i+1 of b, i < 4) at bits 4..5 of bytes 0..3.
__device__ __forceinline__ unsigned high_pairs(unsigned b) {
    b                = b & 0xffU;
    const unsigned n = (b | (b << 12)) & 0x000f000fU; // fields 0, 1 at bits 0..3; 2, 3 at 16..19
    return ((n | (n << 6)) & 0x03030303U) << 4;
}

__device__ __forceinline__ uint4 or_fifth(uint4 v, unsigned qh) {
    return make_uint4(v.x | fifth_bits(qh & 0xfU), v.y | fifth_bits((qh >> 4) & 0xfU),
                      v.z | fifth_bits((qh >> 8) & 0xfU), v.w | fifth_bits((qh >> 12) & 0xfU));
}

// ---------------------------------------------------------------------------------------------
// Weight codecs: load() takes what 32-value block kb of a row needs (a few registers), decode()
// makes its 32 int8 codes (lo: values 0..15, hi: 16..31), scale and min.
//
// A codec whose neighbouring blocks share their bytes loads kSpan blocks at once: load(kb) takes
// blocks kb + kStride*s (s < kSpan) and decode(raw, s, ...) makes block s. A codec with kHalves
// has a scale per 16 values: decode's `min` is the scale of values 16..31 (it has no min term),
// and the kernel runs the block as two k16 MMAs; with kHalfMin decode adds the two halves' mins.
// ---------------------------------------------------------------------------------------------

template <class C>
__host__ __device__ constexpr int w4a8_span() {
    if constexpr (requires { C::kSpan; }) {
        return C::kSpan;
    } else {
        return 1;
    }
}
template <class C>
__host__ __device__ constexpr int w4a8_stride() {
    if constexpr (requires { C::kStride; }) {
        return C::kStride;
    } else {
        return 1;
    }
}
template <class C>
__host__ __device__ constexpr bool w4a8_halves() {
    if constexpr (requires { C::kHalves; }) {
        return C::kHalves;
    } else {
        return false;
    }
}

// A codec whose buffer holds E2M1 codes in byte containers with power-of-two block scales, for the
// block-scaled MMA (w4a8_mma_tile_mx) against MXFP8 activations (launch_moe_mx8_quantize).
template <class C>
__host__ __device__ constexpr bool w4a8_block_scaled() {
    if constexpr (requires { C::kBlockScaled; }) {
        return C::kBlockScaled;
    } else {
        return false;
    }
}

// A kHalves codec with a min per 16 values (Q2_K): decode also makes the mins of values 0..15 and
// 16..31, which the kernel takes against the token's code sums over the same 16 values (a k16 MMA
// or dp4a against ones), and the buffer holds them in a plane of their own (b_hmin).
template <class C>
__host__ __device__ constexpr bool w4a8_half_min() {
    if constexpr (requires { C::kHalfMin; }) {
        return C::kHalfMin;
    } else {
        return false;
    }
}

template <Ggml32 F>
struct W4a8Ggml32 {
    static constexpr bool kFive    = F == Ggml32::Q5_0 || F == Ggml32::Q5_1;
    static constexpr bool kWithMin = F == Ggml32::Q4_1 || F == Ggml32::Q5_1;
    static constexpr bool kMin     = kWithMin;
    struct Raw {
        uint4 qs;
        unsigned qh;
        unsigned dm;
    };
    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + kb;
        Raw r;
        r.qs = *reinterpret_cast<const uint4*>(a.codes + blk * 16);
        r.qh = kFive ? reinterpret_cast<const unsigned*>(a.high)[blk] : 0U;
        r.dm = kWithMin ? reinterpret_cast<const unsigned*>(a.scales)[blk]
                        : reinterpret_cast<const unsigned short*>(a.scales)[blk];
        return r;
    }
    __device__ static __forceinline__ void decode(const Raw& r, uint4& lo, uint4& hi, float& scale,
                                                  float& min) {
        if constexpr (F == Ggml32::IQ4_NL) {
            lo = make_uint4(iq4nl_s8(r.qs.x & 0x0f0f0f0fU), iq4nl_s8(r.qs.y & 0x0f0f0f0fU),
                            iq4nl_s8(r.qs.z & 0x0f0f0f0fU), iq4nl_s8(r.qs.w & 0x0f0f0f0fU));
            hi = make_uint4(iq4nl_s8((r.qs.x >> 4) & 0x0f0f0f0fU),
                            iq4nl_s8((r.qs.y >> 4) & 0x0f0f0f0fU),
                            iq4nl_s8((r.qs.z >> 4) & 0x0f0f0f0fU),
                            iq4nl_s8((r.qs.w >> 4) & 0x0f0f0f0fU));
        } else {
            ggml_nibbles_s8(r.qs, 0U, lo, hi);
            if constexpr (kFive) {
                lo = or_fifth(lo, r.qh);
                hi = or_fifth(hi, r.qh >> 16);
            }
            if constexpr (!kWithMin) {
                const unsigned bias = F == Ggml32::Q4_0 ? 0x08080808U : 0x10101010U;
                lo = make_uint4(__vsub4(lo.x, bias), __vsub4(lo.y, bias), __vsub4(lo.z, bias),
                                __vsub4(lo.w, bias));
                hi = make_uint4(__vsub4(hi.x, bias), __vsub4(hi.y, bias), __vsub4(hi.z, bias),
                                __vsub4(hi.w, bias));
            }
        }
        scale = half_bits(r.dm);
        min   = kWithMin ? half_bits(r.dm >> 16) : 0.0F;
    }
};

// Four MXFP4 codes (bytes of q, each < 16) as int8 bytes of twice their E2M1 value: 0, 1, 2, 3,
// 4, 6, 8, 12 and the same negated (-0 is 0). Doubling makes every E2M1 value an integer, so the
// int8 code is exact and the factor 1/2 moves into the block scale.
__device__ __forceinline__ unsigned e2m1x2_s8(unsigned q) {
    constexpr unsigned kW0 = 0x03020100U; // 0, 1, 2, 3
    constexpr unsigned kW1 = 0x0c080604U; // 4, 6, 8, 12
    constexpr unsigned kW2 = 0xfdfeff00U; // 0, -1, -2, -3
    constexpr unsigned kW3 = 0xf4f8fafcU; // -4, -6, -8, -12
    const unsigned t   = q | (q >> 4);
    const unsigned c   = __byte_perm(t, 0U, 0x4420U); // nibble n: code n
    const unsigned s   = c & 0x7777U;
    const unsigned sel = ((c & 0x8888U) >> 1) | 0x3210U;
    return __byte_perm(__byte_perm(kW0, kW1, s), __byte_perm(kW2, kW3, s), sel);
}

// MXFP4 (OCP microscaling): per 32-value block 16 code bytes, value 2i in byte i's low nibble
// and 2i + 1 in its high nibble, and one E8M0 exponent byte; w = 2^(e - 127) * e2m1(q). The
// doubled code is the int8 value and the block scale 2^(e - 128). Codes are interleaved, not in
// ggml's split halves, so the low- and high-nibble lookups are zipped back into value order.
struct W4a8Mxfp4 {
    static constexpr bool kMin = false;
    struct Raw {
        uint4 qs;
        unsigned e;
    };
    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * (a.k / 32) + kb;
        Raw r;
        r.qs = *reinterpret_cast<const uint4*>(a.codes + blk * 16);
        r.e  = a.scales[blk];
        return r;
    }
    // Eight values from one code word: low-nibble values 0, 2, 4, 6 and high-nibble 1, 3, 5, 7.
    __device__ static __forceinline__ void zip(unsigned w, unsigned& first, unsigned& second) {
        const unsigned even = e2m1x2_s8(w & 0x0f0f0f0fU);
        const unsigned odd  = e2m1x2_s8((w >> 4) & 0x0f0f0f0fU);
        first               = __byte_perm(even, odd, 0x5140U);
        second              = __byte_perm(even, odd, 0x7362U);
    }
    __device__ static __forceinline__ void decode(const Raw& r, uint4& lo, uint4& hi, float& scale,
                                                  float& min) {
        zip(r.qs.x, lo.x, lo.y);
        zip(r.qs.y, lo.z, lo.w);
        zip(r.qs.z, hi.x, hi.y);
        zip(r.qs.w, hi.z, hi.w);
        scale = 0.5F * e8m0_value(r.e);
        min   = 0.0F;
    }
};

// MXFP4 for the block-scaled MMA: each code shifted into bits 2..5 of its own byte
// (kind::mxf8f6f4's E2M1 container), in value order, and the block scale 2^(e - 127) as a float
// whose exponent bits are e: w4a8_mma_tile_mx hands them back to the MMA as its UE8M0 operand (e =
// 0 -> 2^-127, whose exponent field is 0 as well), so the weights enter the MMA exactly as stored.
struct W4a8Mxfp4Mx {
    static constexpr bool kMin         = false;
    static constexpr bool kBlockScaled = true;
    using Raw                          = W4a8Mxfp4::Raw;

    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        return W4a8Mxfp4::load(a, row, kb);
    }

    __device__ static __forceinline__ void zip(unsigned w, unsigned& first, unsigned& second) {
        const unsigned even = (w & 0x0f0f0f0fU) << 2;
        const unsigned odd  = (w >> 2) & 0x3c3c3c3cU;
        first               = __byte_perm(even, odd, 0x5140U);
        second              = __byte_perm(even, odd, 0x7362U);
    }

    __device__ static __forceinline__ void decode(const Raw& r, uint4& lo, uint4& hi, float& scale,
                                                  float& min) {
        zip(r.qs.x, lo.x, lo.y);
        zip(r.qs.y, lo.z, lo.w);
        zip(r.qs.z, hi.x, hi.y);
        zip(r.qs.w, hi.z, hi.w);
        scale = e8m0_value(r.e);
        min   = 0.0F;
    }
};

// NVFP4: the MXFP4 code layout (value 2i in byte i's low nibble, 2i + 1 in its high nibble) with an
// E4M3 scale per 16 values in the block-scale tile order (nvfp4_scale_offset) and the weight's
// row divisor (Nvfp4Decoder::row_scale). The doubled code is the int8 value; values 0..15 take
// 0.5 * e4m3(s0) * row_scale, 16..31 the same of s1 (a kHalves codec: two k16 MMAs per block).
// Groups 2kb and 2kb + 1 share a four-group run of the tile, so one 16-bit load holds both.
struct W4a8Nvfp4 {
    static constexpr bool kMin    = false;
    static constexpr bool kHalves = true;

    struct Raw {
        uint4 qs;
        unsigned s;
        float row;
    };

    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * (a.k / 32) + kb;
        Raw r;
        r.qs = *reinterpret_cast<const uint4*>(a.codes + blk * 16);
        r.s =
            *reinterpret_cast<const unsigned short*>(a.scales + nvfp4_scale_offset(a, row, 2 * kb));
        r.row = Nvfp4Decoder::row_scale(a, row);
        return r;
    }

    __device__ static __forceinline__ float e4m3(unsigned byte) {
        const __half_raw h =
            __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(byte & 0xffU), __NV_E4M3);
        return __half2float(__half(h));
    }

    __device__ static __forceinline__ void decode(const Raw& r, uint4& lo, uint4& hi, float& scale,
                                                  float& min) {
        W4a8Mxfp4::zip(r.qs.x, lo.x, lo.y);
        W4a8Mxfp4::zip(r.qs.y, lo.z, lo.w);
        W4a8Mxfp4::zip(r.qs.z, hi.x, hi.y);
        W4a8Mxfp4::zip(r.qs.w, hi.z, hi.w);
        scale = 0.5F * e4m3(r.s) * r.row;
        min   = 0.5F * e4m3(r.s >> 8) * r.row;
    }
};

// IQ4_XS: sub-block ib of a superblock is qs bytes 16ib .. 16ib + 15 (ggml nibble order), scale
// d*(ls - 32).
struct W4a8Iq4Xs {
    static constexpr bool kMin = false;
    struct Raw {
        uint4 qs;
        unsigned dsh; // d (low half), scales_h (high half)
        unsigned sl;  // scales_l word
        unsigned ib;
    };
    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (kb >> 3);
        const int ib           = kb & 7;
        Raw r;
        r.qs  = *reinterpret_cast<const uint4*>(a.codes + blk * 128 + 16 * ib);
        r.dsh = reinterpret_cast<const unsigned*>(a.scales)[blk];
        r.sl  = reinterpret_cast<const unsigned*>(a.high)[blk];
        r.ib  = static_cast<unsigned>(ib);
        return r;
    }
    __device__ static __forceinline__ void decode(const Raw& r, uint4& lo, uint4& hi, float& scale,
                                                  float& min) {
        lo = make_uint4(iq4nl_s8(r.qs.x & 0x0f0f0f0fU), iq4nl_s8(r.qs.y & 0x0f0f0f0fU),
                        iq4nl_s8(r.qs.z & 0x0f0f0f0fU), iq4nl_s8(r.qs.w & 0x0f0f0f0fU));
        hi = make_uint4(iq4nl_s8((r.qs.x >> 4) & 0x0f0f0f0fU), iq4nl_s8((r.qs.y >> 4) & 0x0f0f0f0fU),
                        iq4nl_s8((r.qs.z >> 4) & 0x0f0f0f0fU), iq4nl_s8((r.qs.w >> 4) & 0x0f0f0f0fU));
        const int ls = static_cast<int>(((r.sl >> (4 * r.ib)) & 0xfU) |
                                        (((r.dsh >> (16 + 2 * r.ib)) & 3U) << 4));
        scale = half_bits(r.dsh) * static_cast<float>(ls - 32);
        min   = 0.0F;
    }
};

// Q4_K / Q5_K (ggml_blocks_v1: whole superblocks): sub-block j is low (j even) or high (j odd)
// nibbles of qs bytes 32(j / 2) .. + 31 (Q5_K: fifth bit j of qh[l]); scale d*sc_j, min
// -dmin*m_j.
template <int Bits>
struct W4a8Q45K {
    static constexpr bool kMin  = true;
    static constexpr int kSpan  = 2; // blocks 2p, 2p + 1: low and high nibbles of qs 32p .. + 31
    static constexpr int kBytes = Bits == 5 ? 176 : 144;
    struct Raw {
        uint4 head;
        uint4 qa, qb; // qs bytes 32p .. + 15, + 16 .. + 31
        uint4 ha, hb; // Q5_K: qh bytes 0 .. 15, 16 .. 31
        unsigned j;   // 2p
    };
    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (kb >> 3);
        const std::uint8_t* b  = a.codes + blk * kBytes;
        const int j            = kb & 7;
        const auto* q = reinterpret_cast<const uint4*>(b + (Bits == 5 ? 48 : 16) + 16 * j);
        Raw r;
        r.head = *reinterpret_cast<const uint4*>(b);
        r.qa   = q[0];
        r.qb   = q[1];
        if constexpr (Bits == 5) {
            r.ha = *reinterpret_cast<const uint4*>(b + 16);
            r.hb = *reinterpret_cast<const uint4*>(b + 32);
        } else {
            r.ha = r.hb = make_uint4(0, 0, 0, 0);
        }
        r.j = static_cast<unsigned>(j);
        return r;
    }
    __device__ static __forceinline__ uint4 nib(const uint4& q, unsigned shift) {
        return make_uint4((q.x >> shift) & 0x0f0f0f0fU, (q.y >> shift) & 0x0f0f0f0fU,
                          (q.z >> shift) & 0x0f0f0f0fU, (q.w >> shift) & 0x0f0f0f0fU);
    }
    // Bit j of each qh byte as 0x10.
    __device__ static __forceinline__ uint4 bit(const uint4& h, unsigned j) {
        const unsigned m = 0x01010101U;
        return make_uint4(((h.x >> j) & m) << 4, ((h.y >> j) & m) << 4, ((h.z >> j) & m) << 4,
                          ((h.w >> j) & m) << 4);
    }
    __device__ static __forceinline__ void decode(const Raw& r, int s, uint4& lo, uint4& hi,
                                                  float& scale, float& min) {
        const unsigned j = r.j + static_cast<unsigned>(s);
        lo               = nib(r.qa, 4U * static_cast<unsigned>(s));
        hi               = nib(r.qb, 4U * static_cast<unsigned>(s));
        if constexpr (Bits == 5) {
            const uint4 bl = bit(r.ha, j), bh = bit(r.hb, j);
            lo = make_uint4(lo.x | bl.x, lo.y | bl.y, lo.z | bl.z, lo.w | bl.w);
            hi = make_uint4(hi.x | bh.x, hi.y | bh.y, hi.z | bh.z, hi.w | bh.w);
        }
        float sc, m;
        kq_scale_min(r.head, static_cast<int>(j), sc, m);
        scale = half_bits(r.head.x) * sc;
        min   = -(half_bits(r.head.x >> 16) * m);
    }
};

// Q6_K planes (GgmlQ6KDecoder): per superblock 128 ql bytes then 64 qh bytes (codes), 16 int8
// scales (high plane), binary16 d (scale plane). Block j = 4h + u of a superblock: low (u < 2) or
// high nibble of ql[64h + 32(u & 1) + l], bits 2u of qh[32h + l], minus 32 (int8 -32 .. 31);
// values 0..15 take d*sc[8h + 2u], 16..31 d*sc[8h + 2u + 1]. A load holds blocks u and u + 2,
// which share their ql bytes.
struct W4a8Q6K {
    static constexpr bool kMin    = false;
    static constexpr bool kHalves = true;
    static constexpr int kSpan    = 2;
    static constexpr int kStride  = 2;
    struct Raw {
        uint4 la, lb; // ql bytes 64h + 32u .. + 15, + 16 .. + 31
        uint4 ha, hb; // qh bytes 32h .. + 15, + 16 .. + 31
        uint2 sc;     // scales 8h .. 8h + 7
        unsigned d;
        unsigned u;
    };
    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (kb >> 3);
        const int h            = (kb >> 2) & 1;
        const int u            = kb & 3; // 0 or 1
        const std::uint8_t* cb = a.codes + blk * 192;
        const auto* ql         = reinterpret_cast<const uint4*>(cb + 64 * h + 32 * u);
        const auto* qh         = reinterpret_cast<const uint4*>(cb + 128 + 32 * h);
        Raw r;
        r.la = ql[0];
        r.lb = ql[1];
        r.ha = qh[0];
        r.hb = qh[1];
        r.sc = *reinterpret_cast<const uint2*>(a.high + blk * 16 + 8 * h);
        r.d  = reinterpret_cast<const unsigned short*>(a.scales)[blk];
        r.u  = static_cast<unsigned>(u);
        return r;
    }
    __device__ static __forceinline__ unsigned word(unsigned l, unsigned h, unsigned shift,
                                                    unsigned hshift) {
        return __vsub4(((l >> shift) & 0x0f0f0f0fU) | (((h >> hshift) & 0x03030303U) << 4),
                       0x20202020U);
    }
    __device__ static __forceinline__ uint4 codes(const uint4& l, const uint4& h, unsigned shift,
                                                  unsigned hshift) {
        return make_uint4(word(l.x, h.x, shift, hshift), word(l.y, h.y, shift, hshift),
                          word(l.z, h.z, shift, hshift), word(l.w, h.w, shift, hshift));
    }
    __device__ static __forceinline__ float sub_scale(const Raw& r, unsigned i) {
        const unsigned w = i < 4 ? r.sc.x : r.sc.y;
        return half_bits(r.d) * static_cast<float>(static_cast<signed char>((w >> (8 * (i & 3))) & 0xffU));
    }
    __device__ static __forceinline__ void decode(const Raw& r, int s, uint4& lo, uint4& hi,
                                                  float& scale, float& scale_hi) {
        const unsigned u = r.u + 2U * static_cast<unsigned>(s);
        const unsigned shift = 4U * static_cast<unsigned>(s), hshift = 2U * u;
        lo       = codes(r.la, r.ha, shift, hshift);
        hi       = codes(r.lb, r.hb, shift, hshift);
        scale    = sub_scale(r, 2U * u);
        scale_hi = sub_scale(r, 2U * u + 1U);
    }
};

// Q3_K planes (GgmlQ3KDecoder): per superblock 32 hmask bytes then 64 qs bytes (codes), 12 packed
// 6-bit scale bytes (high plane), binary16 d (scale plane). Block j = 4h + jj of a superblock:
// bits 2jj of qs[32h + l], plus 4 where bit 4h + jj of hmask[l] is set, minus 4 (int8 -4 .. 3);
// values 0..15 take d*(sc[8h + 2jj] - 32), 16..31 d*(sc[8h + 2jj + 1] - 32). A load holds the four
// blocks of half h, which share their qs and hmask bytes.
struct W4a8Q3K {
    static constexpr bool kMin    = false;
    static constexpr bool kHalves = true;
    static constexpr int kSpan    = 4;
    struct Raw {
        uint4 qa, qb; // qs bytes 32h .. + 15, + 16 .. + 31
        uint4 ma, mb; // hmask bytes 0 .. 15, 16 .. 31, each word shifted right by 4h
        uint2 sc;     // 6-bit scales 8h .. 8h + 7 (value sc - 32), bytes
        unsigned d;
    };
    __device__ static __forceinline__ uint4 shifted(const uint4& m, unsigned sh) {
        return make_uint4(m.x >> sh, m.y >> sh, m.z >> sh, m.w >> sh);
    }
    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (kb >> 3);
        const unsigned sh      = 4U * (static_cast<unsigned>(kb >> 2) & 1U);
        const std::uint8_t* cb = a.codes + blk * 96;
        const auto* qs         = reinterpret_cast<const uint4*>(cb + 32 + 8 * sh);
        const auto* hm         = reinterpret_cast<const uint4*>(cb);
        const auto* sw         = reinterpret_cast<const unsigned*>(a.high + blk * 12);
        const unsigned s0 = sw[0], s1 = sw[1], s2 = sw[2];
        Raw r;
        r.qa = qs[0];
        r.qb = qs[1];
        r.ma = shifted(hm[0], sh);
        r.mb = shifted(hm[1], sh);
        r.sc = make_uint2(((s0 >> sh) & 0x0f0f0f0fU) | (((s2 >> sh) & 0x03030303U) << 4),
                          ((s1 >> sh) & 0x0f0f0f0fU) | (((s2 >> (sh + 2)) & 0x03030303U) << 4));
        r.d  = reinterpret_cast<const unsigned short*>(a.scales)[blk];
        return r;
    }
    __device__ static __forceinline__ unsigned word(unsigned q, unsigned m, unsigned jj) {
        return __vsub4(((q >> (2U * jj)) & 0x03030303U) | (((m >> jj) & 0x01010101U) << 2),
                       0x04040404U);
    }
    __device__ static __forceinline__ uint4 codes(const uint4& q, const uint4& m, unsigned jj) {
        return make_uint4(word(q.x, m.x, jj), word(q.y, m.y, jj), word(q.z, m.z, jj),
                          word(q.w, m.w, jj));
    }
    __device__ static __forceinline__ float sub_scale(const Raw& r, unsigned g) {
        const unsigned w = g < 4 ? r.sc.x : r.sc.y;
        const int sc     = static_cast<int>((w >> (8 * (g & 3))) & 0xffU);
        return half_bits(r.d) * static_cast<float>(sc - 32);
    }
    __device__ static __forceinline__ void decode(const Raw& r, int s, uint4& lo, uint4& hi,
                                                  float& scale, float& scale_hi) {
        const unsigned jj = static_cast<unsigned>(s);
        lo       = codes(r.qa, r.ma, jj);
        hi       = codes(r.qb, r.mb, jj);
        scale    = sub_scale(r, 2U * jj);
        scale_hi = sub_scale(r, 2U * jj + 1U);
    }
};

// Q2_K (GgmlQ2KDecoder planes: 64 qs bytes per superblock, 16 scale bytes with sc in the low and m
// in the high nibble, binary16 d and dmin as one word): value l of 32-value block 4h + jj is bits
// 2jj of qs[32h + l] and takes scale byte 8h + 2jj + l / 16; w = (d*sc)*q - dmin*m, q in 0 .. 3.
// A load takes the four blocks of half superblock h.
struct W4a8Q2K {
    static constexpr bool kMin     = false;
    static constexpr bool kHalves  = true;
    static constexpr bool kHalfMin = true;
    static constexpr int kSpan     = 4;

    struct Raw {
        uint4 qa, qb; // qs bytes 32h .. + 15, + 16 .. + 31
        uint2 sc;     // scale bytes 8h .. 8h + 7
        unsigned dd;  // binary16 d (low half) and dmin (high half)
    };

    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (kb >> 3);
        const int h            = (kb >> 2) & 1;
        const auto* qs         = reinterpret_cast<const uint4*>(a.codes + blk * 64 + 32 * h);
        Raw r;
        r.qa = qs[0];
        r.qb = qs[1];
        r.sc = *reinterpret_cast<const uint2*>(a.high + blk * 16 + 8 * h);
        r.dd = reinterpret_cast<const unsigned*>(a.scales)[blk];
        return r;
    }

    __device__ static __forceinline__ uint4 codes(const uint4& q, unsigned jj) {
        const unsigned sh = 2U * jj;
        return make_uint4((q.x >> sh) & 0x03030303U, (q.y >> sh) & 0x03030303U,
                          (q.z >> sh) & 0x03030303U, (q.w >> sh) & 0x03030303U);
    }

    __device__ static __forceinline__ void decode(const Raw& r, int s, uint4& lo, uint4& hi,
                                                  float& scale, float& scale_hi, float& min_lo,
                                                  float& min_hi) {
        const unsigned jj = static_cast<unsigned>(s);
        lo                = codes(r.qa, jj);
        hi                = codes(r.qb, jj);
        const unsigned w  = (jj < 2 ? r.sc.x : r.sc.y) >> (16U * (jj & 1U)); // bytes 2jj, 2jj + 1
        const float d     = half_bits(r.dd);
        const float dmin  = half_bits(r.dd >> 16);
        scale             = d * static_cast<float>(w & 0xfU);
        scale_hi          = d * static_cast<float>((w >> 8) & 0xfU);
        min_lo            = -(dmin * static_cast<float>((w >> 4) & 0xfU));
        min_hi            = -(dmin * static_cast<float>((w >> 12) & 0xfU));
    }
};

// u4_g{32,64,128}_zp: natural nibble order (value 2i low nibble of byte i), scale and zero per G
// values (the planes of U4ZpDecoder / U4ZpSmallDecoder). Block kb (32 values) is in group
// kb / (G / 32).
template <int G>
struct W4a8U4ZpG {
    static_assert(G == 32 || G == 64 || G == 128);
    static constexpr bool kMin = false;
    struct Raw {
        uint4 codes;
        unsigned zero;
        unsigned scale;
    };
    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t gid = static_cast<std::int64_t>(row) * a.groups_per_row + kb / (G / 32);
        Raw r;
        r.codes = *reinterpret_cast<const uint4*>(a.codes + static_cast<std::int64_t>(row) * (a.k / 2) +
                                                  kb * 16);
        r.zero  = a.high[gid];
        r.scale = reinterpret_cast<const unsigned short*>(a.scales)[gid];
        return r;
    }
    __device__ static __forceinline__ void pairs(unsigned w, unsigned z, unsigned& x, unsigned& y) {
        const unsigned even = w & 0x0f0f0f0fU, odd = (w >> 4) & 0x0f0f0f0fU;
        x = __vsub4(__byte_perm(even, odd, 0x5140U), z);
        y = __vsub4(__byte_perm(even, odd, 0x7362U), z);
    }
    __device__ static __forceinline__ void decode(const Raw& r, uint4& lo, uint4& hi, float& scale,
                                                  float& min) {
        const unsigned z = r.zero * 0x01010101U;
        pairs(r.codes.x, z, lo.x, lo.y);
        pairs(r.codes.y, z, lo.z, lo.w);
        pairs(r.codes.z, z, hi.x, hi.y);
        pairs(r.codes.w, z, hi.z, hi.w);
        scale = half_bits(r.scale);
        min   = 0.0F;
    }
};

using W4a8U4Zp    = W4a8U4ZpG<128>;
using W4a8U4ZpG32 = W4a8U4ZpG<32>;
using W4a8U4ZpG64 = W4a8U4ZpG<64>;

// q4_g64 / q5_g64 / q6_g64 (RowSplit, the published ninferno artifacts' groupwise formats): a group
// of 64 values is 32 code bytes (value 2i the low nibble of byte i), for 5 and 6 bits a high plane
// of 1 or 2 bits per value (value v at bit Bits-4 times v of the group's 8 or 16 bytes), and one
// binary16 scale; codes are two's complement. Block kb is half kb & 1 of group kb / 2.
template <int Bits>
struct W4a8G64 {
    static_assert(Bits == 4 || Bits == 5 || Bits == 6);
    static constexpr bool kMin = false;
    struct Raw {
        uint4 codes;
        uint2 high; // Q5: x; Q6: x, y
        unsigned scale;
    };
    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t gid = static_cast<std::int64_t>(row) * a.groups_per_row + (kb >> 1);
        const int half         = kb & 1;
        Raw r;
        r.codes = *reinterpret_cast<const uint4*>(a.codes + gid * 32 + half * 16);
        if constexpr (Bits == 5) {
            r.high = make_uint2(*reinterpret_cast<const unsigned*>(a.high + gid * 8 + half * 4), 0U);
        } else if constexpr (Bits == 6) {
            r.high = *reinterpret_cast<const uint2*>(a.high + gid * 16 + half * 8);
        } else {
            r.high = make_uint2(0U, 0U);
        }
        r.scale = reinterpret_cast<const unsigned short*>(a.scales)[gid];
        return r;
    }
    // Values 4q .. 4q + 3 of the block as signed bytes: their nibbles from code word w (8 values),
    // their high bits from the plane.
    __device__ static __forceinline__ unsigned quad(const Raw& r, int q) {
        const unsigned w    = q < 2 ? r.codes.x : q < 4 ? r.codes.y : q < 6 ? r.codes.z : r.codes.w;
        const unsigned even = w & 0x0f0f0f0fU, odd = (w >> 4) & 0x0f0f0f0fU;
        unsigned u          = (q & 1) == 0 ? __byte_perm(even, odd, 0x5140U)
                                           : __byte_perm(even, odd, 0x7362U);
        if constexpr (Bits == 5) {
            u |= fifth_bits((r.high.x >> (4 * q)) & 0xfU);
            return __vsub4(u ^ 0x10101010U, 0x10101010U);
        } else if constexpr (Bits == 6) {
            u |= high_pairs((q < 4 ? r.high.x : r.high.y) >> (8 * (q & 3)));
            return __vsub4(u ^ 0x20202020U, 0x20202020U);
        } else {
            return __vsub4(u ^ 0x08080808U, 0x08080808U);
        }
    }
    __device__ static __forceinline__ void decode(const Raw& r, uint4& lo, uint4& hi, float& scale,
                                                  float& min) {
        lo    = make_uint4(quad(r, 0), quad(r, 1), quad(r, 2), quad(r, 3));
        hi    = make_uint4(quad(r, 4), quad(r, 5), quad(r, 6), quad(r, 7));
        scale = half_bits(r.scale);
        min   = 0.0F;
    }
};

// q3_g128 / q3_g64: three-bit two's-complement codes, code j of a group in bits 3j..3j+2 of its
// code bytes, so block kb (32 values) is 12 bytes at (kb % (Group / 32)) * 12 of its group, 4-byte
// aligned; one binary16 scale per group.
template <int Group>
struct W4a8Q3 {
    static_assert(Group == 64 || Group == 128);
    static constexpr bool kMin       = false;
    static constexpr int kGroupBytes = Group * 3 / 8;

    struct Raw {
        unsigned w0, w1, w2;
        unsigned scale;
    };

    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t gid =
            static_cast<std::int64_t>(row) * a.groups_per_row + (kb * 32) / Group;
        const auto* p = reinterpret_cast<const unsigned*>(a.codes + gid * kGroupBytes +
                                                          (kb % (Group / 32)) * 12);
        return {p[0], p[1], p[2], reinterpret_cast<const unsigned short*>(a.scales)[gid]};
    }

    // Four codes from bit `shift` of `bits` as signed bytes ((u ^ 4) - 4 per byte).
    __device__ static __forceinline__ unsigned quad(unsigned long long bits, int shift) {
        unsigned u = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u |= (static_cast<unsigned>(bits >> (shift + 3 * i)) & 7U) << (8 * i);
        }
        return __vsub4(u ^ 0x04040404U, 0x04040404U);
    }

    __device__ static __forceinline__ void decode(const Raw& r, uint4& lo, uint4& hi, float& scale,
                                                  float& min) {
        // Bits 0..63 and 32..95 of the block: values 0..19 from the first, 20..31 the second.
        const unsigned long long a = (static_cast<unsigned long long>(r.w1) << 32) | r.w0;
        const unsigned long long b = (static_cast<unsigned long long>(r.w2) << 32) | r.w1;
        lo                         = make_uint4(quad(a, 0), quad(a, 12), quad(a, 24), quad(a, 36));
        hi                         = make_uint4(quad(a, 48), quad(b, 28), quad(b, 40), quad(b, 52));
        scale                      = half_bits(r.scale);
        min   = 0.0F;
    }
};

// q8_g32 (GGUF Q8_0, row-split planes): block kb of a row is 32 int8 codes, scale binary16 d.
struct W4a8Q8 {
    static constexpr bool kMin = false;
    struct Raw {
        uint4 lo, hi;
        unsigned d;
    };
    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t gid = static_cast<std::int64_t>(row) * a.groups_per_row + kb;
        const auto* q          = reinterpret_cast<const uint4*>(a.codes + gid * 32);
        return {q[0], q[1], reinterpret_cast<const unsigned short*>(a.scales)[gid]};
    }
    __device__ static __forceinline__ void decode(const Raw& r, uint4& lo, uint4& hi, float& scale,
                                                  float& min) {
        lo    = r.lo;
        hi    = r.hi;
        scale = half_bits(r.d);
        min   = 0.0F;
    }
};

// Where a codebook decoder reads its grid words: the global table through the read-only cache, or
// a copy the kernel staged in shared memory (r8_kernel). A warp's lookups are random words of a 1..8
// KB table: from global memory each takes one L1 wavefront per distinct 128-byte line (up to 32),
// from shared memory one per bank conflict (RTX 2080 Ti, IQ1 / IQ2 R8 decode at 170..230 GB/s of
// 616 with the global reads).
// pair(table, e) reads words 2e and 2e + 1 (an 8-value IQ2 entry): one 64-bit shared load, whose
// half-warp phases spread over all 32 banks; two 32-bit loads of the even and odd words used only
// 16 banks each (about 5-way conflicts on random entries, RTX 2080 Ti IQ2 R8 decode).
struct GlobalGrid {
    __device__ __forceinline__ unsigned operator()(const std::uint32_t* table, unsigned i) const {
        return __ldg(table + i);
    }
    __device__ __forceinline__ uint2 pair(const std::uint32_t* table, unsigned e) const {
        return make_uint2(__ldg(table + 2 * e), __ldg(table + 2 * e + 1));
    }
};
struct SharedGrid {
    const std::uint32_t* words; // 8-byte aligned (r8_kernel)
    __device__ __forceinline__ unsigned operator()(const std::uint32_t*, unsigned i) const {
        return words[i];
    }
    __device__ __forceinline__ uint2 pair(const std::uint32_t*, unsigned e) const {
        return reinterpret_cast<const uint2*>(words)[e];
    }
};

// IQ3_XXS (GgmlIq3XxsDecoder planes): sub-block ib of a superblock is grid indices qs[8ib .. 8ib +
// 7] (four values each) and word ib of the high plane: four 7-bit sign indices (ksigns, one per
// 8-value group) with ls on top; scale d*(0.5 + ls)*0.5.
struct W4a8Iq3Xxs {
    static constexpr bool kMin = false;
    struct Raw {
        uint2 qs;
        unsigned aux;
        unsigned d;
    };
    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (kb >> 3);
        const int ib           = kb & 7;
        return {*reinterpret_cast<const uint2*>(a.codes + blk * 64 + 8 * ib),
                reinterpret_cast<const unsigned*>(a.high + blk * 32)[ib],
                reinterpret_cast<const unsigned short*>(a.scales)[blk]};
    }
    static constexpr int kGridWords = 256;
    __device__ static __forceinline__ const std::uint32_t* grid() { return ggml_iq::kIq3xxsGrid; }
    template <class G = GlobalGrid>
    __device__ static __forceinline__ void decode(const Raw& r, uint4& lo, uint4& hi, float& scale,
                                                  float& min, G grid_src = {}) {
        unsigned w[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned idx = l < 2 ? r.qs.x : r.qs.y;
            const unsigned s   = ksigns_byte((r.aux >> (7 * l)) & 127U);
            const unsigned i0  = (idx >> (16 * (l & 1))) & 0xffU;
            const unsigned i1  = (idx >> (16 * (l & 1) + 8)) & 0xffU;
            w[2 * l]           = signed_bytes(grid_src(ggml_iq::kIq3xxsGrid, i0), s);
            w[2 * l + 1]       = signed_bytes(grid_src(ggml_iq::kIq3xxsGrid, i1), s >> 4);
        }
        lo    = make_uint4(w[0], w[1], w[2], w[3]);
        hi    = make_uint4(w[4], w[5], w[6], w[7]);
        scale = half_bits(r.d) * (0.5F + static_cast<float>(r.aux >> 28)) * 0.5F;
        min   = 0.0F;
    }
};

// IQ3_XXS for the producer warps of w4a8_ws_kernel: two neighbouring sub-blocks per load (16 index
// bytes and two sign words), the grid and ksigns' byte masks from a table the producers stage in
// shared memory (lut[0..255] grid words, lut[256 + 2*s7], lut[257 + 2*s7] the masks of ksigns[s7]'s
// low and high nibble). The same bytes as W4a8Iq3Xxs::decode.
struct W4a8Iq3XxsPaired {
    static constexpr bool kMin     = false;
    static constexpr int kSpan     = 2;
    static constexpr int kLutWords = 256 + 2 * 128;
    struct Raw {
        uint4 qs;
        uint2 aux;
        unsigned d;
    };
    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (kb >> 3);
        const int ib           = kb & 7; // even
        return {*reinterpret_cast<const uint4*>(a.codes + blk * 64 + 8 * ib),
                *reinterpret_cast<const uint2*>(a.high + blk * 32 + 4 * ib),
                reinterpret_cast<const unsigned short*>(a.scales)[blk]};
    }
    __device__ static __forceinline__ unsigned byte_masks(unsigned signs) {
        return (((signs & 0xfU) * 0x00204081U) & 0x01010101U) * 0xffU;
    }
    __device__ static __forceinline__ void init_lut(std::uint32_t* lut, int tid, int threads) {
        for (int i = tid; i < 256; i += threads) lut[i] = ggml_iq::kIq3xxsGrid[i];
        for (int i = tid; i < 128; i += threads) {
            const unsigned s     = ksigns_byte(static_cast<unsigned>(i));
            lut[256 + 2 * i]     = byte_masks(s);
            lut[256 + 2 * i + 1] = byte_masks(s >> 4);
        }
    }
    __device__ static __forceinline__ void decode(const Raw& r, int s, const std::uint32_t* lut,
                                                  uint4& lo, uint4& hi, float& scale,
                                                  float& min) {
        const unsigned q0 = s == 0 ? r.qs.x : r.qs.z, q1 = s == 0 ? r.qs.y : r.qs.w;
        const unsigned aux = s == 0 ? r.aux.x : r.aux.y;
        unsigned w[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned idx = l < 2 ? q0 : q1;
            const unsigned i0  = (idx >> (16 * (l & 1))) & 0xffU;
            const unsigned i1  = (idx >> (16 * (l & 1) + 8)) & 0xffU;
            const unsigned s7 = (aux >> (7 * l)) & 127U;
            const uint2 m     = *reinterpret_cast<const uint2*>(lut + 256 + 2 * s7);
            w[2 * l]      = __vsub4(lut[i0] ^ m.x, m.x);
            w[2 * l + 1]  = __vsub4(lut[i1] ^ m.y, m.y);
        }
        lo    = make_uint4(w[0], w[1], w[2], w[3]);
        hi    = make_uint4(w[4], w[5], w[6], w[7]);
        scale = half_bits(r.d) * (0.5F + static_cast<float>(aux >> 28)) * 0.5F;
        min   = 0.0F;
    }
};

// The codec the producer warps of w4a8_ws_kernel decode a format with.
template <class C>
struct W4a8WsCodec {
    using type = C;
};
template <>
struct W4a8WsCodec<W4a8Iq3Xxs> {
    using type = W4a8Iq3XxsPaired;
};

template <class C>
__host__ __device__ constexpr int w4a8_lut_words() {
    if constexpr (requires { C::kLutWords; }) {
        return C::kLutWords;
    } else {
        return 0;
    }
}

// IQ3_S (GgmlIq3SDecoder planes: 104 code bytes per superblock, qs[64], qh[8], signs[32]; one
// scale word in the high plane): sub-block ib is indices qs[8ib .. 8ib + 7] with ninth bits qh[ib]
// (bit m for index m), sign bytes 4ib .. 4ib + 3 (one per 8-value group) and ls, nibble ib of the
// scale word; scale d*(1 + 2*ls).
struct W4a8Iq3S {
    static constexpr bool kMin = false;
    struct Raw {
        uint2 qs;
        unsigned sg;
        unsigned qh;  // qh[ib]
        unsigned dls; // d (low half), ls (bits 16..19)
    };
    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (kb >> 3);
        const int ib           = kb & 7;
        const std::uint8_t* cb = a.codes + blk * 104;
        const unsigned ls = (reinterpret_cast<const unsigned*>(a.high)[blk] >> (4 * ib)) & 0xfU;
        return {*reinterpret_cast<const uint2*>(cb + 8 * ib),
                *reinterpret_cast<const unsigned*>(cb + 72 + 4 * ib), cb[64 + ib],
                static_cast<unsigned>(reinterpret_cast<const unsigned short*>(a.scales)[blk]) |
                    (ls << 16)};
    }
    static constexpr int kGridWords = 512;
    __device__ static __forceinline__ const std::uint32_t* grid() { return ggml_iq::kIq3sGrid; }
    template <class G = GlobalGrid>
    __device__ static __forceinline__ void decode(const Raw& r, uint4& lo, uint4& hi, float& scale,
                                                  float& min, G grid_src = {}) {
        unsigned w[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned idx = l < 2 ? r.qs.x : r.qs.y;
            const unsigned s   = (r.sg >> (8 * l)) & 0xffU;
            const unsigned i0 =
                ((idx >> (16 * (l & 1))) & 0xffU) | (((r.qh >> (2 * l)) & 1U) << 8);
            const unsigned i1 =
                ((idx >> (16 * (l & 1) + 8)) & 0xffU) | (((r.qh >> (2 * l + 1)) & 1U) << 8);
            w[2 * l]     = signed_bytes(grid_src(ggml_iq::kIq3sGrid, i0), s);
            w[2 * l + 1] = signed_bytes(grid_src(ggml_iq::kIq3sGrid, i1), s >> 4);
        }
        lo    = make_uint4(w[0], w[1], w[2], w[3]);
        hi    = make_uint4(w[4], w[5], w[6], w[7]);
        scale = half_bits(r.dls) * static_cast<float>(1 + 2 * static_cast<int>(r.dls >> 16));
        min   = 0.0F;
    }
};

// IQ3_S for the producer warps of w4a8_ws_kernel: two neighbouring sub-blocks per load, the grid
// and the sign bytes' masks from a table the producers stage in shared memory (lut[0..511] grid
// words, lut[512 + 2*s], lut[513 + 2*s] the byte masks of sign byte s's low and high nibble). The
// same bytes as W4a8Iq3S::decode (signed_bytes(g, s) == (g ^ m) - m bytewise).
struct W4a8Iq3SPaired {
    static constexpr bool kMin     = false;
    static constexpr int kSpan     = 2;
    static constexpr int kLutWords = 512 + 2 * 256;
    struct Raw {
        uint2 qs0, qs1;
        uint2 sg;
        unsigned qh;  // qh[ib], qh[ib + 1] in bytes 0, 1
        unsigned dls; // d (low half), ls of ib (bits 16..19) and of ib + 1 (bits 20..23)
    };
    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (kb >> 3);
        const int ib           = kb & 7; // even
        const std::uint8_t* cb = a.codes + blk * 104;
        const unsigned ls = (reinterpret_cast<const unsigned*>(a.high)[blk] >> (4 * ib)) & 0xffU;
        return {*reinterpret_cast<const uint2*>(cb + 8 * ib),
                *reinterpret_cast<const uint2*>(cb + 8 * ib + 8),
                *reinterpret_cast<const uint2*>(cb + 72 + 4 * ib),
                *reinterpret_cast<const unsigned short*>(cb + 64 + ib),
                static_cast<unsigned>(reinterpret_cast<const unsigned short*>(a.scales)[blk]) |
                    (ls << 16)};
    }
    __device__ static __forceinline__ unsigned byte_masks(unsigned signs) {
        return (((signs & 0xfU) * 0x00204081U) & 0x01010101U) * 0xffU;
    }
    __device__ static __forceinline__ void init_lut(std::uint32_t* lut, int tid, int threads) {
        for (int i = tid; i < 512; i += threads) lut[i] = ggml_iq::kIq3sGrid[i];
        for (int i = tid; i < 256; i += threads) {
            lut[512 + 2 * i]     = byte_masks(static_cast<unsigned>(i));
            lut[512 + 2 * i + 1] = byte_masks(static_cast<unsigned>(i) >> 4);
        }
    }
    __device__ static __forceinline__ void decode(const Raw& r, int s, const std::uint32_t* lut,
                                                  uint4& lo, uint4& hi, float& scale,
                                                  float& min) {
        const uint2 qs     = s == 0 ? r.qs0 : r.qs1;
        const unsigned sg  = s == 0 ? r.sg.x : r.sg.y;
        const unsigned qh  = (r.qh >> (8 * s)) & 0xffU;
        const unsigned ls  = (r.dls >> (16 + 4 * s)) & 0xfU;
        unsigned w[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned idx = l < 2 ? qs.x : qs.y;
            const unsigned sb  = (sg >> (8 * l)) & 0xffU;
            const unsigned i0 =
                ((idx >> (16 * (l & 1))) & 0xffU) | (((qh >> (2 * l)) & 1U) << 8);
            const unsigned i1 =
                ((idx >> (16 * (l & 1) + 8)) & 0xffU) | (((qh >> (2 * l + 1)) & 1U) << 8);
            const uint2 m = *reinterpret_cast<const uint2*>(lut + 512 + 2 * sb);
            w[2 * l]      = __vsub4(lut[i0] ^ m.x, m.x);
            w[2 * l + 1]  = __vsub4(lut[i1] ^ m.y, m.y);
        }
        lo    = make_uint4(w[0], w[1], w[2], w[3]);
        hi    = make_uint4(w[4], w[5], w[6], w[7]);
        scale = half_bits(r.dls) * static_cast<float>(1 + 2 * static_cast<int>(ls));
        min   = 0.0F;
    }
};
template <>
struct W4a8WsCodec<W4a8Iq3S> {
    using type = W4a8Iq3SPaired;
};

// IQ1_S / IQ1_M: a grid value g in {-1, 0, 1} (code g + 1, a nibble per value in kIq1sGrid) plus
// a delta of +-1/8 makes 8g +- 1 in int8 exactly, the scale taking the 1/8.
// The decoders read kIq1sGridHalves, whose byte k holds the codes of values k (low nibble) and
// k + 4 (high nibble): one mask and shift give four codes times 8 per output word.
// Four codes times 8 (bytes 0, 8, 16) to 8(code - 1) +- 1: each byte offset by 128 so that the
// subtraction of 7 or 9 borrows nowhere, then the offset flipped back (sm_75 runs __vadd4 as
// several instructions).
__device__ __forceinline__ unsigned iq1_values(unsigned codes8, unsigned k) {
    return ((codes8 | 0x80808080U) - k) ^ 0x80808080U;
}
constexpr unsigned kIq1Plus  = 0x07070707U; // 8(code - 1) + 1 = 8 code - 7
constexpr unsigned kIq1Minus = 0x09090909U; // 8(code - 1) - 1 = 8 code - 9

// IQ1_S (GgmlIq1SDecoder planes): sub-block ib is codes bytes 4ib .. 4ib + 3 (the low eight bits of
// its four groups' grid indices) and qh word ib (bits 3l .. 3l + 2 group l's high bits, 12..14 ls,
// 15 the delta's sign); w = d*(2ls + 1)*(g + delta).
struct W4a8Iq1S {
    static constexpr bool kMin = false;
    struct Raw {
        unsigned qs;
        unsigned qh;
        unsigned d;
    };
    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (kb >> 3);
        const int ib           = kb & 7;
        return {*reinterpret_cast<const unsigned*>(a.codes + blk * 32 + 4 * ib),
                reinterpret_cast<const unsigned short*>(a.high)[blk * 8 + ib],
                reinterpret_cast<const unsigned short*>(a.scales)[blk]};
    }
    static constexpr int kGridWords = 2048;
    __device__ static __forceinline__ const std::uint32_t* grid() {
        return ggml_iq::kIq1sGridHalves;
    }
    template <class G = GlobalGrid>
    __device__ static __forceinline__ void decode(const Raw& r, uint4& lo, uint4& hi, float& scale,
                                                  float& min, G grid_src = {}) {
        const unsigned k = (r.qh & 0x8000U) != 0 ? kIq1Minus : kIq1Plus;
        unsigned w[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned idx = ((r.qs >> (8 * l)) & 0xffU) | (((r.qh >> (3 * l)) & 7U) << 8);
            const unsigned g   = grid_src(ggml_iq::kIq1sGridHalves, idx);
            w[2 * l]           = iq1_values((g << 3) & 0x78787878U, k);
            w[2 * l + 1]       = iq1_values((g >> 1) & 0x78787878U, k);
        }
        lo    = make_uint4(w[0], w[1], w[2], w[3]);
        hi    = make_uint4(w[4], w[5], w[6], w[7]);
        scale = half_bits(r.d) * static_cast<float>(2 * static_cast<int>((r.qh >> 12) & 7U) + 1) *
                0.125F;
        min   = 0.0F;
    }
};

// IQ1_M (GgmlIq1MDecoder planes): sub-block ib is codes bytes 4ib .. 4ib + 3, qh bytes 2ib and
// 2ib + 1 (nibble l: group l's three high index bits, bit 3 its delta's sign) and the scale words:
// ls of values 0..15 and 16..31 in bits 6(ib % 2) and 6(ib % 2) + 3 of word ib / 2, the binary16 d
// in the four words' top nibbles. A scale per 16 values.
struct W4a8Iq1M {
    static constexpr bool kMin    = false;
    static constexpr bool kHalves = true;
    struct Raw {
        unsigned qs;
        unsigned qh;
        uint2 sw;
        int ib;
    };
    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (kb >> 3);
        const int ib           = kb & 7;
        return {*reinterpret_cast<const unsigned*>(a.codes + blk * 32 + 4 * ib),
                reinterpret_cast<const unsigned short*>(a.high)[blk * 8 + ib],
                *reinterpret_cast<const uint2*>(a.scales + blk * 8), ib};
    }
    static constexpr int kGridWords = 2048;
    __device__ static __forceinline__ const std::uint32_t* grid() {
        return ggml_iq::kIq1sGridHalves;
    }
    template <class G = GlobalGrid>
    __device__ static __forceinline__ void decode(const Raw& r, uint4& lo, uint4& hi, float& scale,
                                                  float& scale_hi, G grid_src = {}) {
        unsigned w[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned n    = (r.qh >> (4 * l)) & 0xfU;
            const unsigned idx  = ((r.qs >> (8 * l)) & 0xffU) | ((n & 7U) << 8);
            const unsigned g    = grid_src(ggml_iq::kIq1sGridHalves, idx);
            const unsigned k    = (n & 8U) != 0 ? kIq1Minus : kIq1Plus;
            w[2 * l]            = iq1_values((g << 3) & 0x78787878U, k);
            w[2 * l + 1]        = iq1_values((g >> 1) & 0x78787878U, k);
        }
        lo                  = make_uint4(w[0], w[1], w[2], w[3]);
        hi                  = make_uint4(w[4], w[5], w[6], w[7]);
        const unsigned word = (r.ib & 4) != 0 ? r.sw.y : r.sw.x; // sc[ib / 2]
        const unsigned sc   = (word >> (16 * ((r.ib >> 1) & 1))) & 0xffffU;
        const int shift     = 6 * (r.ib & 1);
        const float d       = half_bits(GgmlIq1MDecoder::d_bits(r.sw)) * 0.125F;
        scale    = d * static_cast<float>(2 * static_cast<int>((sc >> shift) & 7U) + 1);
        scale_hi = d * static_cast<float>(2 * static_cast<int>((sc >> (shift + 3)) & 7U) + 1);
    }
};

// IQ2_XXS (GgmlIq2XxsDecoder planes: 64 code bytes per superblock): sub-block ib is two words,
// four grid indices (one byte per 8-value group) then four 7-bit sign indices with ls on top;
// scale d*(0.5 + ls)*0.25. A grid entry is eight magnitudes (two words).
struct W4a8Iq2Xxs {
    static constexpr bool kMin = false;
    struct Raw {
        uint2 q;
        unsigned d;
    };
    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (kb >> 3);
        return {*reinterpret_cast<const uint2*>(a.codes + blk * 64 + 8 * (kb & 7)),
                reinterpret_cast<const unsigned short*>(a.scales)[blk]};
    }
    static constexpr int kGridWords = 512;
    __device__ static __forceinline__ const std::uint32_t* grid() { return ggml_iq::kIq2xxsGrid; }
    template <class G = GlobalGrid>
    __device__ static __forceinline__ void decode(const Raw& r, uint4& lo, uint4& hi, float& scale,
                                                  float& min, G grid_src = {}) {
        unsigned w[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned e = (r.q.x >> (8 * l)) & 0xffU;
            const unsigned s = ksigns_byte((r.q.y >> (7 * l)) & 127U);
            const uint2 g    = grid_src.pair(ggml_iq::kIq2xxsGrid, e);
            w[2 * l]         = signed_bytes(g.x, s);
            w[2 * l + 1]     = signed_bytes(g.y, s >> 4);
        }
        lo    = make_uint4(w[0], w[1], w[2], w[3]);
        hi    = make_uint4(w[4], w[5], w[6], w[7]);
        scale = half_bits(r.d) * (0.5F + static_cast<float>(r.q.y >> 28)) * 0.25F;
        min   = 0.0F;
    }
};

// IQ2_XS (GgmlIq2XsDecoder planes): sub-block ib is four 16-bit entries (9-bit grid index, 7-bit
// sign index on top) and scale byte ib of the high plane, one ls per 16 values: d*(0.5 + ls)*0.25
// for values 0..15 (low nibble) and 16..31 (high nibble).
struct W4a8Iq2Xs {
    static constexpr bool kMin    = false;
    static constexpr bool kHalves = true;
    struct Raw {
        uint2 q;
        unsigned sc;
        unsigned d;
    };
    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (kb >> 3);
        const int ib           = kb & 7;
        return {*reinterpret_cast<const uint2*>(a.codes + blk * 64 + 8 * ib), a.high[blk * 8 + ib],
                reinterpret_cast<const unsigned short*>(a.scales)[blk]};
    }
    static constexpr int kGridWords = 1024;
    __device__ static __forceinline__ const std::uint32_t* grid() { return ggml_iq::kIq2xsGrid; }
    template <class G = GlobalGrid>
    __device__ static __forceinline__ void decode(const Raw& r, uint4& lo, uint4& hi, float& scale,
                                                  float& scale_hi, G grid_src = {}) {
        unsigned w[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned e   = ((l < 2 ? r.q.x : r.q.y) >> (16 * (l & 1))) & 0xffffU;
            const unsigned idx = e & 511U;
            const unsigned s   = ksigns_byte(e >> 9);
            const uint2 g      = grid_src.pair(ggml_iq::kIq2xsGrid, idx);
            w[2 * l]           = signed_bytes(g.x, s);
            w[2 * l + 1]       = signed_bytes(g.y, s >> 4);
        }
        lo       = make_uint4(w[0], w[1], w[2], w[3]);
        hi       = make_uint4(w[4], w[5], w[6], w[7]);
        scale    = half_bits(r.d) * (0.5F + static_cast<float>(r.sc & 0xfU)) * 0.25F;
        scale_hi = half_bits(r.d) * (0.5F + static_cast<float>((r.sc >> 4) & 0xfU)) * 0.25F;
    }
};

// IQ2_S (GgmlIq2SDecoder planes: 72 code bytes per superblock, qs[32], signs[32], qh[8]; scales as
// IQ2_XS): 8-value group 4ib + l takes index qs[4ib + l] with bits 2l, 2l + 1 of qh[ib] as its
// ninth and tenth bits, and sign byte signs[4ib + l].
struct W4a8Iq2S {
    static constexpr bool kMin    = false;
    static constexpr bool kHalves = true;
    struct Raw {
        unsigned qs, sg, qh, sc, d;
    };
    __device__ static __forceinline__ Raw load(const GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (kb >> 3);
        const int ib           = kb & 7;
        const std::uint8_t* cb = a.codes + blk * 72;
        return {*reinterpret_cast<const unsigned*>(cb + 4 * ib),
                *reinterpret_cast<const unsigned*>(cb + 32 + 4 * ib), cb[64 + ib],
                a.high[blk * 8 + ib], reinterpret_cast<const unsigned short*>(a.scales)[blk]};
    }
    static constexpr int kGridWords = 2048;
    __device__ static __forceinline__ const std::uint32_t* grid() { return ggml_iq::kIq2sGrid; }
    template <class G = GlobalGrid>
    __device__ static __forceinline__ void decode(const Raw& r, uint4& lo, uint4& hi, float& scale,
                                                  float& scale_hi, G grid_src = {}) {
        unsigned w[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned idx = ((r.qs >> (8 * l)) & 0xffU) | (((r.qh >> (2 * l)) & 3U) << 8);
            const unsigned s   = (r.sg >> (8 * l)) & 0xffU;
            const uint2 g      = grid_src.pair(ggml_iq::kIq2sGrid, idx);
            w[2 * l]           = signed_bytes(g.x, s);
            w[2 * l + 1]       = signed_bytes(g.y, s >> 4);
        }
        lo       = make_uint4(w[0], w[1], w[2], w[3]);
        hi       = make_uint4(w[4], w[5], w[6], w[7]);
        scale    = half_bits(r.d) * (0.5F + static_cast<float>(r.sc & 0xfU)) * 0.25F;
        scale_hi = half_bits(r.d) * (0.5F + static_cast<float>((r.sc >> 4) & 0xfU)) * 0.25F;
    }
};

// ---------------------------------------------------------------------------------------------
// The kernels. Two schedules of one arithmetic: w4a8_kernel, where every warp loads, decodes and
// multiplies (two shared-memory buffers, two barriers per K tile), and w4a8_ws_kernel, where
// producer warps copy the activation tile and decode the weight tile into a ring of stages while
// the MMA warps only multiply (one named-barrier pair per stage). Both run the MMAs and the FP32
// epilogue of every accumulator in the same order (K tiles in order, then the four 32-value
// blocks), so every schedule of either kernel writes the same bytes.
// ---------------------------------------------------------------------------------------------

// HalfMin (w4a8_half_min): one more plane of two floats per block and weight row.
template <class S, bool HalfMin = false>
struct W4a8Buffer {
    static constexpr int kBytes = S::kBufferBytes + (HalfMin ? S::kBlockN * 32 : 0);
    static constexpr int kSmem  = 2 * kBytes;
    std::uint8_t* a_codes; // [BlockM][128]
    std::uint8_t* b_codes; // [BlockN][128]
    float* a_scale;        // [BlockM][4]
    float* a_sum;          // [BlockM][4]
    float* b_scale;        // [BlockN][4]
    float* b_min;          // [BlockN][4]: the min, or a kHalves codec's scale of values 16..31
    float* b_hmin;         // [BlockN][4][2]: a kHalfMin codec's mins of values 0..15, 16..31
    __device__ static __forceinline__ W4a8Buffer at(unsigned char* smem, int index) {
        unsigned char* p = smem + index * kBytes;
        W4a8Buffer b;
        b.a_codes = p;
        b.b_codes = p + S::kBlockM * 128;
        auto* f   = reinterpret_cast<float*>(p + (S::kBlockM + S::kBlockN) * 128);
        b.a_scale = f;
        b.a_sum   = f + S::kBlockM * 4;
        b.b_scale = f + S::kBlockM * 8;
        b.b_min   = f + S::kBlockM * 8 + S::kBlockN * 4;
        b.b_hmin  = HalfMin ? f + (S::kBlockM + S::kBlockN) * 8 : nullptr;
        return b;
    }
};

// The classic kernel's buffer for codec C, and its shared bytes.
template <class C, class S>
using W4a8BufferOf = W4a8Buffer<S, w4a8_half_min<C>()>;

// A stage of the producer/consumer ring: the per-token sums only for a codec with a min term, the
// second weight plane only for a codec that reads one.
template <class C, class S, int Stages>
struct W4a8Stage {
    static_assert(!w4a8_half_min<C>(), "the producer ring has no half-min plane");
    static constexpr bool kSum    = C::kMin;
    static constexpr bool kAux    = C::kMin || w4a8_halves<C>();
    static constexpr int kFloats  = S::kBlockM * (kSum ? 8 : 4) + S::kBlockN * (kAux ? 8 : 4);
    static constexpr int kBytes   = (S::kBlockM + S::kBlockN) * 128 + kFloats * 4;
    static constexpr int kSmem    = Stages * kBytes;
    std::uint8_t* a_codes;
    std::uint8_t* b_codes;
    float* a_scale;
    float* a_sum;
    float* b_scale;
    float* b_min;
    __device__ static __forceinline__ W4a8Stage at(unsigned char* smem, int stage) {
        unsigned char* p = smem + stage * kBytes;
        W4a8Stage b;
        b.a_codes = p;
        b.b_codes = p + S::kBlockM * 128;
        auto* f   = reinterpret_cast<float*>(p + (S::kBlockM + S::kBlockN) * 128);
        b.a_scale = f;
        b.a_sum   = kSum ? f + S::kBlockM * 4 : nullptr; // absent planes: no pointer into the next
        b.b_scale = f + S::kBlockM * (kSum ? 8 : 4);
        b.b_min   = kAux ? b.b_scale + S::kBlockN * 4 : nullptr;
        return b;
    }
};

// The weight tile of one K tile: `Threads` threads load the raw words of (row, block) tasks into
// registers, then decode them into a buffer's int8 codes and scales.
template <class C, class S, int Threads, bool SwiGlu>
struct W4a8Weights {
    static constexpr int kSpan   = w4a8_span<C>();
    static constexpr int kStride = w4a8_stride<C>();
    static constexpr bool kAux   = C::kMin || w4a8_halves<C>(); // b_min holds something
    static_assert(4 % (kSpan * kStride) == 0);
    static constexpr int kLoads = 4 / kSpan;           // loads per weight row per K tile
    static constexpr int kTasks = S::kBlockN * kLoads; // (weight row, load) per K tile
    static constexpr int kPer   = (kTasks + Threads - 1) / Threads;
    typename C::Raw raw[kPer];
    bool ok[kPer];

    // The first block of load q of a row's tile: q < kStride side by side, then kSpan * kStride
    // apart.
    __device__ static __forceinline__ int first_block(int q) {
        return q % kStride + (q / kStride) * kSpan * kStride;
    }
    __device__ __forceinline__ void load(const GemmArgs& a, int block_row, int tid, int k_tile) {
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int task  = tid + i * Threads;
            const int n     = task / kLoads;
            const int local = w4a4_local_row<SwiGlu, S::kBlockN>(block_row, n);
            const int kb    = k_tile * 4 + first_block(task % kLoads);
            // The last tile's blocks past K stay zero (a span codec's K is whole tiles).
            ok[i] = task < kTasks && local < a.rows && kb * 32 < a.k;
            if (ok[i]) {
                const int wrow = a.row_begin + local + (SwiGlu && ((n >> 3) & 1) ? a.up_offset : 0);
                raw[i]         = C::load(a, wrow, kb);
            }
        }
    }
    template <class B>
    __device__ __forceinline__ void store(const B& buf, int tid,
                                          const std::uint32_t* lut = nullptr) const {
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int task = tid + i * Threads;
            if (task >= kTasks) continue;
            const int n = task / kLoads, j0 = first_block(task % kLoads);
            auto* row   = buf.b_codes + n * 128;
#pragma unroll
            for (int sp = 0; sp < kSpan; ++sp) {
                const int j = j0 + sp * kStride;
                uint4 lo = make_uint4(0, 0, 0, 0), hi = lo;
                float scale = 0.0F, min = 0.0F, min_lo = 0.0F, min_hi = 0.0F;
                if (ok[i]) {
                    if constexpr (w4a8_half_min<C>()) {
                        C::decode(raw[i], sp, lo, hi, scale, min, min_lo, min_hi);
                    } else if constexpr (kSpan == 1) {
                        C::decode(raw[i], lo, hi, scale, min);
                    } else if constexpr (w4a8_lut_words<C>() > 0) {
                        C::decode(raw[i], sp, lut, lo, hi, scale, min);
                    } else {
                        C::decode(raw[i], sp, lo, hi, scale, min);
                    }
                }
                *reinterpret_cast<uint4*>(row + 16 * w4a8_chunk(n, 2 * j))     = lo;
                *reinterpret_cast<uint4*>(row + 16 * w4a8_chunk(n, 2 * j + 1)) = hi;
                buf.b_scale[n * 4 + j] = scale;
                if constexpr (kAux) buf.b_min[n * 4 + j] = min;
                if constexpr (w4a8_half_min<C>()) {
                    *reinterpret_cast<float2*>(buf.b_hmin + n * 8 + 2 * j) =
                        make_float2(min_lo, min_hi);
                }
            }
        }
    }
};

// The activation planes' row length: K rounded up to whole 128-value K tiles (the quantizer writes
// the tail as zero codes and scales, launch_w4a8), so a 32-value-block format of any K % 32 == 0
// (Gemma 4 26B's shared MLP down, K 2112) runs whole tiles.
__device__ __forceinline__ int w4a8_act_ld(int k) { return (k + 127) / 128 * 128; }

// The activation tile of one K tile by `Threads` threads: a fixed number of 16-byte copies per
// thread, and the per-token sums only for a codec with a min term.
template <class C, class S, int Threads, class B>
__device__ __forceinline__ void w4a8_load_activation(const GemmArgs& a, const W4a8Activation& act,
                                                     int token_begin, int tid, int k_tile,
                                                     const B& buf) {
    static_assert((S::kBlockM * 8) % Threads == 0);
#pragma unroll
    for (int i = 0; i < S::kBlockM * 8 / Threads; ++i) {
        const int task  = tid + i * Threads;
        const int r     = task >> 3, c = task & 7;
        const int token = token_begin + r;
        const bool ok   = token < a.tokens;
        const auto* src = act.codes + static_cast<std::int64_t>(ok ? token : 0) * w4a8_act_ld(a.k) +
                          k_tile * 128 + c * 16;
        cp_async_zfill<16, Cache::cg>(buf.a_codes + r * 128 + 16 * w4a8_chunk(r, c), src,
                                      ok ? 16 : 0);
    }
    constexpr int kPlanes = C::kMin ? 2 : 1; // scales, then sums
    const int kb_row      = w4a8_act_ld(a.k) / 32;
    for (int task = tid; task < S::kBlockM * kPlanes; task += Threads) {
        const int r = task / kPlanes, which = task % kPlanes;
        const int token = token_begin + r;
        const bool ok   = token < a.tokens;
        const float* src = (which ? act.sums : act.scales) +
                           static_cast<std::int64_t>(ok ? token : 0) * kb_row + k_tile * 4;
        cp_async_zfill<16, Cache::cg>((which ? buf.a_sum : buf.a_scale) + r * 4, src, ok ? 16 : 0);
    }
}

// One K tile of a warp's MMAs from buffer `cur`: four 32-value steps, each ldmatrix of both
// operands, the MMAs and the FP32 epilogue sum_blocks s_w*s_x*dot (+ m_w*sum_x; a kHalfMin codec:
// per 16 values, the token's code sums from a k16 MMA against ones). loaded(kk) runs
// once step kk's operands are in registers, stepped(kk) after its MMAs. Scales4 reads the four
// blocks' scales of a row as one float4 before the steps, else one float per step.
template <class C, class S, bool Scales4, class B, class Loaded, class Stepped>
__device__ __forceinline__ void w4a8_mma_tile(float (&acc)[S::kMmaM][S::kMmaN][4], const B& cur,
                                              int warp_m, int warp_n, int lane, Loaded&& loaded,
                                              Stepped&& stepped) {
    constexpr bool kHalves  = w4a8_halves<C>();
    constexpr bool kHalfMin = w4a8_half_min<C>();
    constexpr bool kAux     = C::kMin || kHalves;
    static_assert(!kHalfMin || kHalves);
    const int g            = lane >> 2;
    const int t            = lane & 3;
    // The four blocks' scales of a token or weight row are one float4 in shared memory.
    const auto scales4 = [](const float* plane, int row) {
        return *reinterpret_cast<const float4*>(plane + row * 4);
    };
    const auto lane_of = [](const float4& v, int kk) {
        return kk == 0 ? v.x : kk == 1 ? v.y : kk == 2 ? v.z : v.w;
    };
    float4 as4[S::kMmaM][2], ws4[S::kMmaN][2];
    if constexpr (Scales4) {
#pragma unroll
        for (int m = 0; m < S::kMmaM; ++m) {
            const int base_row = warp_m * S::kWarpM + m * 16;
            as4[m][0]          = scales4(cur.a_scale, base_row + g);
            as4[m][1]          = scales4(cur.a_scale, base_row + g + 8);
        }
#pragma unroll
        for (int n = 0; n < S::kMmaN; ++n) {
            const int base_row = warp_n * S::kWarpN + n * 8;
            ws4[n][0]          = scales4(cur.b_scale, base_row + 2 * t);
            ws4[n][1]          = scales4(cur.b_scale, base_row + 2 * t + 1);
        }
    }
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        unsigned af[S::kMmaM][4];
        unsigned bf[S::kMmaN][2];
        float as[S::kMmaM][2], ws[S::kMmaN][2];
        float asum[S::kMmaM][2], wmin[S::kMmaN][2];
        // kHalfMin: s_x times the code sum of values 0..15 of tokens g, g + 8, then of 16..31;
        // the weight rows' mins of the same halves.
        float hsum[S::kMmaM][4], hmin[S::kMmaN][2][2];
#pragma unroll
        for (int m = 0; m < S::kMmaM; ++m) {
            const int base_row = warp_m * S::kWarpM + m * 16;
            const int row      = base_row + (lane & 7) + ((lane >> 3) & 1) * 8;
            ldmatrix_x4(af[m][0], af[m][1], af[m][2], af[m][3],
                        smem_addr(cur.a_codes + row * 128 +
                                  16 * w4a8_chunk(row, 2 * kk + (lane >> 4))));
            if constexpr (Scales4) {
                as[m][0] = lane_of(as4[m][0], kk);
                as[m][1] = lane_of(as4[m][1], kk);
            } else {
                as[m][0] = cur.a_scale[(base_row + g) * 4 + kk];
                as[m][1] = cur.a_scale[(base_row + g + 8) * 4 + kk];
            }
            if constexpr (C::kMin) {
                asum[m][0] = cur.a_sum[(base_row + g) * 4 + kk];
                asum[m][1] = cur.a_sum[(base_row + g + 8) * 4 + kk];
            }
            if constexpr (kHalfMin) {
                // Every column of B is ones: c0 / c2 are tokens g / g + 8's sums.
                int c0 = int_accumulator_zero(), c1 = int_accumulator_zero(),
                    c2 = int_accumulator_zero(), c3 = int_accumulator_zero();
                int d0 = int_accumulator_zero(), d1 = int_accumulator_zero(),
                    d2 = int_accumulator_zero(), d3 = int_accumulator_zero();
                mma_s8_k16(c0, c1, c2, c3, af[m][0], af[m][1], 0x01010101U);
                mma_s8_k16(d0, d1, d2, d3, af[m][2], af[m][3], 0x01010101U);
                hsum[m][0] = as[m][0] * int_accumulator_to_float(c0);
                hsum[m][1] = as[m][1] * int_accumulator_to_float(c2);
                hsum[m][2] = as[m][0] * int_accumulator_to_float(d0);
                hsum[m][3] = as[m][1] * int_accumulator_to_float(d2);
            }
        }
#pragma unroll
        for (int n = 0; n < S::kMmaN; ++n) {
            const int base_row = warp_n * S::kWarpN + n * 8;
            const int row      = base_row + (lane & 7);
            ldmatrix_x2(bf[n][0], bf[n][1],
                        smem_addr(cur.b_codes + row * 128 +
                                  16 * w4a8_chunk(row, 2 * kk + ((lane >> 3) & 1))));
            if constexpr (Scales4) {
                ws[n][0] = lane_of(ws4[n][0], kk);
                ws[n][1] = lane_of(ws4[n][1], kk);
            } else {
                ws[n][0] = cur.b_scale[(base_row + 2 * t) * 4 + kk];
                ws[n][1] = cur.b_scale[(base_row + 2 * t + 1) * 4 + kk];
            }
            if constexpr (kAux) {
                wmin[n][0] = cur.b_min[(base_row + 2 * t) * 4 + kk];
                wmin[n][1] = cur.b_min[(base_row + 2 * t + 1) * 4 + kk];
            }
            if constexpr (kHalfMin) {
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const float2 v = *reinterpret_cast<const float2*>(
                        cur.b_hmin + (base_row + 2 * t + c) * 8 + 2 * kk);
                    hmin[n][c][0] = v.x;
                    hmin[n][c][1] = v.y;
                }
            }
        }
        loaded(kk);
#pragma unroll
        for (int m = 0; m < S::kMmaM; ++m)
#pragma unroll
            for (int n = 0; n < S::kMmaN; ++n) {
                float* o = acc[m][n];
                if constexpr (kHalves) {
                    // Values 0..15 (A words 0, 1; B word 0), then 16..31 (A 2, 3; B 1).
                    int c0 = int_accumulator_zero(), c1 = int_accumulator_zero(),
                        c2 = int_accumulator_zero(), c3 = int_accumulator_zero();
                    int d0 = int_accumulator_zero(), d1 = int_accumulator_zero(),
                        d2 = int_accumulator_zero(), d3 = int_accumulator_zero();
                    mma_s8_k16(c0, c1, c2, c3, af[m][0], af[m][1], bf[n][0]);
                    mma_s8_k16(d0, d1, d2, d3, af[m][2], af[m][3], bf[n][1]);
                    o[0] = fmaf(int_accumulator_to_float(c0), as[m][0] * ws[n][0], o[0]);
                    o[1] = fmaf(int_accumulator_to_float(c1), as[m][0] * ws[n][1], o[1]);
                    o[2] = fmaf(int_accumulator_to_float(c2), as[m][1] * ws[n][0], o[2]);
                    o[3] = fmaf(int_accumulator_to_float(c3), as[m][1] * ws[n][1], o[3]);
                    o[0] = fmaf(int_accumulator_to_float(d0), as[m][0] * wmin[n][0], o[0]);
                    o[1] = fmaf(int_accumulator_to_float(d1), as[m][0] * wmin[n][1], o[1]);
                    o[2] = fmaf(int_accumulator_to_float(d2), as[m][1] * wmin[n][0], o[2]);
                    o[3] = fmaf(int_accumulator_to_float(d3), as[m][1] * wmin[n][1], o[3]);
                    if constexpr (kHalfMin) {
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            o[0] = fmaf(hsum[m][2 * h], hmin[n][0][h], o[0]);
                            o[1] = fmaf(hsum[m][2 * h], hmin[n][1][h], o[1]);
                            o[2] = fmaf(hsum[m][2 * h + 1], hmin[n][0][h], o[2]);
                            o[3] = fmaf(hsum[m][2 * h + 1], hmin[n][1][h], o[3]);
                        }
                    }
                } else {
                    int c0 = int_accumulator_zero(), c1 = int_accumulator_zero(),
                        c2 = int_accumulator_zero(), c3 = int_accumulator_zero();
                    mma_s8(c0, c1, c2, c3, af[m][0], af[m][1], af[m][2], af[m][3], bf[n][0],
                           bf[n][1]);
                    o[0] = fmaf(int_accumulator_to_float(c0), as[m][0] * ws[n][0], o[0]);
                    o[1] = fmaf(int_accumulator_to_float(c1), as[m][0] * ws[n][1], o[1]);
                    o[2] = fmaf(int_accumulator_to_float(c2), as[m][1] * ws[n][0], o[2]);
                    o[3] = fmaf(int_accumulator_to_float(c3), as[m][1] * ws[n][1], o[3]);
                    if constexpr (C::kMin) {
                        o[0] = fmaf(asum[m][0], wmin[n][0], o[0]);
                        o[1] = fmaf(asum[m][0], wmin[n][1], o[1]);
                        o[2] = fmaf(asum[m][1], wmin[n][0], o[2]);
                        o[3] = fmaf(asum[m][1], wmin[n][1], o[3]);
                    }
                }
            }
        stepped(kk);
    }
}

#if NINFERNO_HAS_NVFP4_MMA
// The four 32-value blocks' power-of-two scales of a row as UE8M0 bytes of one word (block j in
// byte j): the exponent field of each float (the scales are positive).
__device__ __forceinline__ unsigned w4a8_ue8m0x4(const float* plane, int row) {
    const uint4 v = *reinterpret_cast<const uint4*>(plane + row * 4);
    return __byte_perm(__byte_perm(v.x >> 23, v.y >> 23, 0x0040U),
                       __byte_perm(v.z >> 23, v.w >> 23, 0x0040U), 0x5410U);
}

// w4a8_mma_tile for a block-scaled codec (w4a8_block_scaled) against MXFP8 activations: the same
// ldmatrix fragments (E4M3 and E2M1 containers are byte operands like int8), one block-scaled MMA
// per (m, n, step) accumulating in FP32 with the scales applied by the tensor core, so no integer
// epilogue. Each lane's scale words: A row g (lane % 4 == 0) or g + 8 (lane % 4 == 1), B column g.
template <class S, class B, class Loaded, class Stepped>
__device__ __forceinline__ void w4a8_mma_tile_mx(float (&acc)[S::kMmaM][S::kMmaN][4], const B& cur,
                                                 int warp_m, int warp_n, int lane, Loaded&& loaded,
                                                 Stepped&& stepped) {
    const int g = lane >> 2;
    const int t = lane & 3;
    unsigned sfa[S::kMmaM], sfb[S::kMmaN];
#    pragma unroll
    for (int m = 0; m < S::kMmaM; ++m) {
        sfa[m] = w4a8_ue8m0x4(cur.a_scale, warp_m * S::kWarpM + m * 16 + g + (t & 1) * 8);
    }
#    pragma unroll
    for (int n = 0; n < S::kMmaN; ++n) {
        sfb[n] = w4a8_ue8m0x4(cur.b_scale, warp_n * S::kWarpN + n * 8 + g);
    }
    const auto step = [&]<int kk>(std::integral_constant<int, kk>) {
        unsigned af[S::kMmaM][4];
        unsigned bf[S::kMmaN][2];
#    pragma unroll
        for (int m = 0; m < S::kMmaM; ++m) {
            const int row = warp_m * S::kWarpM + m * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
            ldmatrix_x4(
                af[m][0], af[m][1], af[m][2], af[m][3],
                smem_addr(cur.a_codes + row * 128 + 16 * w4a8_chunk(row, 2 * kk + (lane >> 4))));
        }
#    pragma unroll
        for (int n = 0; n < S::kMmaN; ++n) {
            const int row = warp_n * S::kWarpN + n * 8 + (lane & 7);
            ldmatrix_x2(bf[n][0], bf[n][1],
                        smem_addr(cur.b_codes + row * 128 +
                                  16 * w4a8_chunk(row, 2 * kk + ((lane >> 3) & 1))));
        }
        loaded(kk);
#    pragma unroll
        for (int m = 0; m < S::kMmaM; ++m)
#    pragma unroll
            for (int n = 0; n < S::kMmaN; ++n) {
                float* o = acc[m][n];
                mma_mxf8_e4m3_e2m1<kk>(o[0], o[1], o[2], o[3], af[m][0], af[m][1], af[m][2],
                                       af[m][3], bf[n][0], bf[n][1], sfa[m], sfb[n]);
            }
        stepped(kk);
    };
    step(std::integral_constant<int, 0>{});
    step(std::integral_constant<int, 1>{});
    step(std::integral_constant<int, 2>{});
    step(std::integral_constant<int, 3>{});
}
#endif

// C fragment: tokens g, g + 8 of each m tile; weight tile rows 2t, 2t + 1 of each n tile.
template <class S, Epilogue E>
__device__ __forceinline__ void w4a8_store_tile(const GemmArgs& a,
                                                const float (&acc)[S::kMmaM][S::kMmaN][4],
                                                int block_row, int token_begin, int warp_m,
                                                int warp_n, int lane) {
    constexpr bool kSwiGlu = E == Epilogue::SwiGlu;
    static_assert(!kSwiGlu || (S::kMmaN % 2) == 0);
    const int g = lane >> 2;
    const int t = lane & 3;
#pragma unroll
    for (int m = 0; m < S::kMmaM; ++m) {
        const int t0 = token_begin + warp_m * S::kWarpM + m * 16 + g;
#pragma unroll
        for (int n = 0; n < S::kMmaN; n += (kSwiGlu ? 2 : 1)) {
            const int tile_row = warp_n * S::kWarpN + n * 8 + 2 * t;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int local = w4a4_local_row<kSwiGlu, S::kBlockN>(block_row, tile_row + c);
                if (local >= a.rows) continue;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int token = t0 + h * 8;
                    if (token >= a.tokens) continue;
                    float value = acc[m][n][h * 2 + c];
                    if constexpr (kSwiGlu)
                        value = generic_gate(a, value) * acc[m][n + 1][h * 2 + c];
                    store_output<E>(a, local, token, value);
                }
            }
        }
    }
}

// The classic kernel's two schedule choices (bits of Flavor): 1 reads the scales as float4 (else
// one float per step), 2 decodes the next tile's weights between the second and third MMA steps
// (else after the fourth). Neither changes an operation of the arithmetic. Measured per tile
// against the base kernel (one float per step, decode after the fourth step) with the
// W4A8 micro-bench, IQ3_XXS/IQ3_S/Q4_K/Q6_K (+ IQ4_XS, Q4_0, Q8) at Qwen3.6-27B and Muse 30B
// shapes, at the widths each tile serves (the base against itself moves 0.99-1.02):
// - 32x128 (T <= 32): the mid-tile decode loses (RTX 5080 1.03-1.07, e.g. Q6_K 1.06); float4
//   scales alone (geomean per format): RTX 5080 0.88-1.01 (IQ3_XXS 0.88), RTX 4090 0.95-1.01.
// - 64x64 (T 33..127): RTX 5080 both 0.83-1.00 (Q4_K 0.88, Q6_K 0.94); RTX 4090 float4 scales alone
//   0.92-0.98 (both: Q6_K 0.95, Q8 1.01).
// - 128x64 and the others: both (RTX 5080 0.91-0.94, RTX 4090 0.89-0.95 at 128x64).
// The sm_89 build is the one without NVFP4 (CMakeLists: one architecture per build tree).
inline constexpr int kW4a8Scales4 = 1, kW4a8DecodeMid = 2;
template <class S>
constexpr int w4a8_classic_flavor() {
#ifdef NINFERNO_WITHOUT_NVFP4
    constexpr bool kAda = true;
#else
    constexpr bool kAda = false;
#endif
    if constexpr (S::kBlockM == 32) {
        return kW4a8Scales4;
    } else if constexpr (S::kBlockM == 64 && S::kBlockN == 64) {
        return kAda ? kW4a8Scales4 : kW4a8Scales4 | kW4a8DecodeMid;
    } else if constexpr (S::kBlockM == 128 && S::kBlockN == 128) {
        // RTX 4090: the float4 scales of a 64x32 warp tile hold 64 registers the epilogue needs.
        // Mid-tile decode alone against both (token-major grid, Q4_K at Muse 30B shapes, T =
        // 1024 / 512): SwiGLU 19968x6656 0.97 / 0.98, Store 6656x19968 0.87 / 0.85, 4096x6656 0.86
        // / 0.87 (Q6_K, which the picker keeps off this tile on sm_89, 1.02-1.03).
        return kAda ? kW4a8DecodeMid : kW4a8Scales4 | kW4a8DecodeMid;
    } else {
        return kW4a8Scales4 | kW4a8DecodeMid;
    }
}

// The CTA's weight row block and token tile (w4a8_grid).
__device__ __forceinline__ int w4a8_block_row(const W4a8Activation& act) {
    return static_cast<int>(act.token_major ? blockIdx.y : blockIdx.x);
}
__device__ __forceinline__ int w4a8_token_tile(const W4a8Activation& act) {
    return static_cast<int>(act.token_major ? blockIdx.x : blockIdx.y);
}

// The grid of a W4A8 launch: row blocks along x and token tiles along y, or with token_major (row
// blocks within the y limit) the other way round, so a wave's CTAs share their weight rows through
// L2. The CTA order changes no operation of a CTA: the same bytes.
template <class S, Epilogue E>
dim3 w4a8_grid(const GemmArgs& args, W4a8Activation& act) {
    constexpr int kRowsPerBlock = E == Epilogue::SwiGlu ? S::kBlockN / 2 : S::kBlockN;
    const unsigned row_blocks   = (args.rows + kRowsPerBlock - 1) / kRowsPerBlock;
    const unsigned token_tiles  = (args.tokens + S::kBlockM - 1) / S::kBlockM;
    if (row_blocks > kMaxGridYZ) act.token_major = 0;
    if (act.token_major) return dim3(token_tiles, row_blocks);
    if (token_tiles > kMaxGridYZ)
        throw std::invalid_argument("generic W4A8: token extent exceeds the grid");
    return dim3(row_blocks, token_tiles);
}

template <class C, class S, Epilogue E, int Flavor = w4a8_classic_flavor<S>()>
__global__ __launch_bounds__(S::kThreads) void w4a8_kernel(GemmArgs a, W4a8Activation act) {
    constexpr bool kSwiGlu = E == Epilogue::SwiGlu;
    using Buf              = W4a8BufferOf<C, S>;
    extern __shared__ __align__(16) unsigned char w4a8_shared[];
    const int tid         = static_cast<int>(threadIdx.x);
    const int block_row   = w4a8_block_row(act);
    const int token_begin = w4a8_token_tile(act) * S::kBlockM;
    const int k_tiles     = w4a8_act_ld(a.k) / 128;

    W4a8Weights<C, S, S::kThreads, kSwiGlu> w;
    w.load(a, block_row, tid, 0);
    w.store(Buf::at(w4a8_shared, 0), tid);
    w4a8_load_activation<C, S, S::kThreads>(a, act, token_begin, tid, 0, Buf::at(w4a8_shared, 0));
    cp_commit();

    float acc[S::kMmaM][S::kMmaN][4] = {};
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int warp_m = warp / S::kWarpsN;
    const int warp_n = warp - warp_m * S::kWarpsN;

#pragma unroll 1
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const auto cur  = Buf::at(w4a8_shared, k_tile & 1);
        const auto next = Buf::at(w4a8_shared, (k_tile + 1) & 1);
        const bool more = k_tile + 1 < k_tiles;
        if (more) {
            w.load(a, block_row, tid, k_tile + 1);
            w4a8_load_activation<C, S, S::kThreads>(a, act, token_begin, tid, k_tile + 1, next);
        }
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        // The next tile's weights decode between the second and third MMA steps, where the
        // scheduler has MMAs to overlap them with, or after the last (buffer `next` was last read
        // before this tile's barrier).
        constexpr int kDecodeStep = (Flavor & kW4a8DecodeMid) ? 1 : 3;
        w4a8_mma_tile<C, S, (Flavor & kW4a8Scales4) != 0>(acc, cur, warp_m, warp_n, lane,
                                                           [](int) {}, [&](int kk) {
                                                               if (kk == kDecodeStep && more) {
                                                                   w.store(next, tid);
                                                               }
                                                           });
        __syncthreads();
    }
    w4a8_store_tile<S, E>(a, acc, block_row, token_begin, warp_m, warp_n, lane);
}

// Named barriers of the producer/consumer ring (0 is __syncthreads): stage s is full at 1 + s,
// empty at 1 + Stages + s.
__device__ __forceinline__ void w4a8_bar_sync(int id, int threads) {
    asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(threads) : "memory");
}
__device__ __forceinline__ void w4a8_bar_arrive(int id, int threads) {
    asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(threads) : "memory");
}

// Warps 0 .. WarpsM*WarpsN - 1 multiply (S's warp grid); Producers more warps fill the ring: for
// K tile k, stage k % Stages takes the activation copies and the decoded weights, the producers
// wait for their copies and mark the stage full; the MMA warps mark it empty once they have read
// it. The producers load tile k + 1's raw weight words while tile k's copies land.
template <class C, class S, Epilogue E, int Stages, int Producers>
__global__ __launch_bounds__(S::kThreads + 32 * Producers) void w4a8_ws_kernel(GemmArgs a,
                                                                             W4a8Activation act) {
    constexpr bool kSwiGlu = E == Epilogue::SwiGlu;
    constexpr int kPT      = 32 * Producers;
    constexpr int kAll     = S::kThreads + kPT;
    using Cw               = typename W4a8WsCodec<C>::type; // how the producers decode C
    using Stage            = W4a8Stage<Cw, S, Stages>;
    constexpr int kLutBar  = 15; // the producers' own barrier, after the ring's 1 + 2 * Stages
    static_assert(Stages >= 2 && 2 * Stages + 1 <= kLutBar);
    extern __shared__ __align__(16) unsigned char w4a8_shared[];
    const int tid         = static_cast<int>(threadIdx.x);
    const int block_row   = w4a8_block_row(act);
    const int token_begin = w4a8_token_tile(act) * S::kBlockM;
    const int k_tiles     = w4a8_act_ld(a.k) / 128;

    if (tid >= S::kThreads) {
        const int ptid = tid - S::kThreads;
        // The codec's table sits after the ring.
        auto* lut = reinterpret_cast<std::uint32_t*>(w4a8_shared + Stage::kSmem);
        if constexpr (w4a8_lut_words<Cw>() > 0) {
            Cw::init_lut(lut, ptid, kPT);
            w4a8_bar_sync(kLutBar, kPT);
        }
        W4a8Weights<Cw, S, kPT, kSwiGlu> w;
        w.load(a, block_row, ptid, 0);
        int s = 0;
#pragma unroll 1
        for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
            if (k_tile >= Stages) w4a8_bar_sync(1 + Stages + s, kAll);
            const auto buf = Stage::at(w4a8_shared, s);
            w4a8_load_activation<Cw, S, kPT>(a, act, token_begin, ptid, k_tile, buf);
            cp_commit();
            w.store(buf, ptid, lut);
            if (k_tile + 1 < k_tiles) w.load(a, block_row, ptid, k_tile + 1);
            cp_wait<0>();
            w4a8_bar_arrive(1 + s, kAll);
            if (++s == Stages) s = 0;
        }
        return;
    }

    float acc[S::kMmaM][S::kMmaN][4] = {};
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int warp_m = warp / S::kWarpsN;
    const int warp_n = warp - warp_m * S::kWarpsN;
    int s            = 0;
#pragma unroll 1
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        w4a8_bar_sync(1 + s, kAll);
        // The stage is free again once the last step's operands are in registers; the producers
        // only wait for it when a tile k_tile + Stages exists.
        const bool refill = k_tile + Stages < k_tiles;
        w4a8_mma_tile<Cw, S, true>(acc, Stage::at(w4a8_shared, s), warp_m, warp_n, lane,
                            [&](int kk) {
                                if (kk == 3 && refill) w4a8_bar_arrive(1 + Stages + s, kAll);
                            },
                            [](int) {});
        if (++s == Stages) s = 0;
    }
    w4a8_store_tile<S, E>(a, acc, block_row, token_begin, warp_m, warp_n, lane);
}


// ---------------------------------------------------------------------------------------------
// Launch. Keep the schedule list in the order of kW4a8Schedules in generic_gemm.cpp. The last
// field is the producer warps: 0 for w4a8_kernel, else w4a8_ws_kernel with that many producers
// and a ring of three stages in the 99 KiB of dynamic shared memory a block may have. The
// producer-warp tile is compiled only for the codecs the picker gives it (w4a8_ws_compiled; route
// admission refuses the others: w4a8_ws_supports, generic_gemm.cpp).
// ---------------------------------------------------------------------------------------------

#define NINFERNO_GENERIC_W4A8_SCHEDULES(X)                                                         \
    X(0, 32, 128, 2, 4, 0)                                                                         \
    X(1, 64, 128, 2, 4, 0)                                                                         \
    X(2, 128, 128, 2, 4, 0)                                                                        \
    X(3, 128, 64, 4, 2, 0)                                                                         \
    X(4, 64, 64, 2, 2, 0)                                                                          \
    X(5, 128, 64, 4, 2, 4)

inline constexpr int kW4a8MaxDynamicSmem = 99 * 1024;

// The codecs w4a8_ws_kernel is compiled for: IQ3_XXS, IQ2_S, Q4_K, Q6_K, Q4_0, IQ4_XS, IQ2_XS
// and IQ2_XXS (keep in step
// with w4a8_ws_supports, generic_gemm.cpp).
template <class C>
constexpr bool w4a8_ws_compiled() {
    return std::is_same_v<C, W4a8Iq3Xxs> || std::is_same_v<C, W4a8Iq2S> ||
           std::is_same_v<C, W4a8Q45K<4>> || std::is_same_v<C, W4a8Q6K> ||
           std::is_same_v<C, W4a8Ggml32<Ggml32::Q4_0>> || std::is_same_v<C, W4a8Iq4Xs> ||
           std::is_same_v<C, W4a8Iq2Xs> || std::is_same_v<C, W4a8Iq2Xxs> ||
           std::is_same_v<C, W4a8Iq3S>;
}

template <class C, class S, Epilogue E, int Flavor = w4a8_classic_flavor<S>()>
void w4a8_launch_schedule(const GemmArgs& args, W4a8Activation act, cudaStream_t stream) {
    constexpr int kSmem = W4a8BufferOf<C, S>::kSmem;
    if constexpr (kSmem > kMaxSharedBytesPerBlock) {
        // route_admits never names it (w4a8_fits); a direct call is refused.
        throw std::invalid_argument("generic W4A8: schedule exceeds this build's shared memory");
    } else {
        const dim3 grid = w4a8_grid<S, E>(args, act);
        CUDA_CHECK(cudaFuncSetAttribute(w4a8_kernel<C, S, E, Flavor>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
        w4a8_kernel<C, S, E, Flavor><<<grid, S::kThreads, kSmem, stream>>>(args, act);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <class C, class S, Epilogue E, int Producers>
void w4a8_ws_launch_schedule(const GemmArgs& args, W4a8Activation act, cudaStream_t stream) {
    using Cw              = typename W4a8WsCodec<C>::type;
    constexpr int kLut    = 4 * w4a8_lut_words<Cw>();
    constexpr int kStages = 3;
    constexpr int kSmem   = W4a8Stage<Cw, S, kStages>::kSmem + kLut;
    static_assert(kSmem <= kW4a8MaxDynamicSmem, "the three-stage ring must fit");
    if constexpr (kSmem > kMaxSharedBytesPerBlock) {
        // The ring is sized for 99 KiB; route_admits never names it on a smaller build (w4a8_fits).
        throw std::invalid_argument(
            "generic W4A8: the producer ring exceeds this build's shared memory");
    } else {
        const dim3 grid = w4a8_grid<S, E>(args, act);
        const auto kernel = w4a8_ws_kernel<C, S, E, kStages, Producers>;
        CUDA_CHECK(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
        kernel<<<grid, S::kThreads + 32 * Producers, kSmem, stream>>>(args, act);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <class C, Epilogue E>
void w4a8_launch_epilogue(int schedule, const GemmArgs& args, W4a8Activation act,
                          cudaStream_t stream) {
    switch (schedule) {
#define NINFERNO_GENERIC_CASE(I, BM, BN, WM, WN, P)                                                \
    case I:                                                                                        \
        if constexpr (P == 0) {                                                                    \
            w4a8_launch_schedule<C, W4a8Schedule<BM, BN, WM, WN>, E>(args, act, stream);           \
            return;                                                                                \
        } else if constexpr (w4a8_ws_compiled<C>()) {                                              \
            w4a8_ws_launch_schedule<C, W4a8Schedule<BM, BN, WM, WN>, E, P>(args, act, stream);     \
            return;                                                                                \
        } else {                                                                                   \
            throw std::invalid_argument("generic W4A8: no producer-warp tile for this format");    \
        }
        NINFERNO_GENERIC_W4A8_SCHEDULES(NINFERNO_GENERIC_CASE)
#undef NINFERNO_GENERIC_CASE
    default:
        throw std::invalid_argument("generic W4A8: unknown schedule");
    }
}


template <class C>
void w4a8_launch_codec(int schedule, Epilogue epilogue, const GemmArgs& args, W4a8Activation act,
                       cudaStream_t stream) {
    switch (epilogue) {
    case Epilogue::Store:
        w4a8_launch_epilogue<C, Epilogue::Store>(schedule, args, act, stream);
        return;
    case Epilogue::AddResidual:
        w4a8_launch_epilogue<C, Epilogue::AddResidual>(schedule, args, act, stream);
        return;
    case Epilogue::SwiGlu:
        w4a8_launch_epilogue<C, Epilogue::SwiGlu>(schedule, args, act, stream);
        return;
    }
    throw std::invalid_argument("generic W4A8: unknown epilogue");
}

// One translation unit per codec group (compile time); each returns false for a codec it lacks.
bool launch_w4a8_ggml32(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                        W4a8Activation act, cudaStream_t stream);
bool launch_w4a8_superblock(int schedule, WeightCodec codec, Epilogue epilogue,
                            const GemmArgs& args, W4a8Activation act, cudaStream_t stream);
bool launch_w4a8_codebook(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                          W4a8Activation act, cudaStream_t stream);

} // namespace ninferno::ops::detail::generic
