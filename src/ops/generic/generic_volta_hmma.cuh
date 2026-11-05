// The m8n8k4 HMMA fragments of Volta's weight formats (generic_volta_hmma.cu explains the layout):
// the dense decode / verify kernel there and the grouped MoE kernel (moe_volta_hmma.cu) read the
// same lane loads, scale groups and A fragments.
#pragma once

#include "ops/common/mma.cuh"
#include "ops/generic/generic_gemm.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace ninferno::ops::detail::generic::volta {

inline constexpr int kWarps    = 4; // a CTA: four warps of eight rows
inline constexpr int kWarpRows = 8; // one weight row per fragment row
inline constexpr int kCtaRows  = kWarps * kWarpRows;
// The staged activation, per quad-pair: 16 groups of four values for each of NTok tokens, then 16
// halves of padding so the quad-pairs' fragment loads fall on different banks.
template <int NTok>
inline constexpr int kQuadStride = 16 * NTok * 4 + 16;

// Token t of group s sits at slot t ^ swizzle(s): a half-warp's staging stores (one token, sixteen
// groups) and its fragment loads (one group, eight tokens) each fall on distinct banks.
template <int NTok>
__device__ __forceinline__ int token_slot(int s, int t) {
    return t ^ (NTok == 8 ? (s >> 1) & 7 : s & 15);
}

__device__ __forceinline__ unsigned h2_bits(__half2 h) { return *reinterpret_cast<unsigned*>(&h); }

__device__ __forceinline__ __half2 bits_h2(unsigned u) { return *reinterpret_cast<__half2*>(&u); }

// Bits 0..3 and 16..19 of w with the f16 exponent of 1024 above them, (w & 0x000f000f) |
// 0x64006400, in one LOP3: with both constants immediate the compiler emitted two (a SASS LOP3
// takes one immediate; gate/up SwiGLU 256 -> 192 LOP3, 2272 -> 2208 instructions).
__device__ __forceinline__ unsigned nibbles_h2(unsigned w) {
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(w), "n"(0x000f000f), "r"(0x64006400U));
    return r;
}

// The f16 pair of biased codes (1024 + code, as nibbles_h2 makes them) less `bias_bits`: exact.
__device__ __forceinline__ unsigned biased_h2(unsigned bits, unsigned bias_bits) {
    return h2_bits(__hsub2(bits_h2(bits), bits_h2(bias_bits)));
}

__device__ __forceinline__ float half_bits_f32(unsigned bits) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}

// Each format: load(a, row, sb, q) the lane's bytes of superblock sb; clear(); kGroups scale groups
// a lane's 64 values split into (16 / kGroups MMAs each); scale(q, g, ds, dm) of group g
// (w = ds code - dm); fragment(m, q) of MMA m (0..15) as two f16 pairs; dest_run(src), where the
// staged activation puts source run src (32 values) of the superblock: quad-pair D / 2, half D % 2.

// Q4_K / Q5_K: 144 / 176-byte blocks (d, dmin, twelve scale bytes, Q5_K's 32 high-bit bytes, 128
// nibble bytes). Quad-pair q: nibble bytes 32q .. 32q + 31 (low nibbles sub-block 2q, high 2q + 1).
template <int Bits>
struct HK45 {
    static constexpr int kBlockBytes = Bits == 4 ? 144 : 176;
    static constexpr int kGroups     = 2;
    static constexpr bool kMin       = true;
    // The operand is 1024 + code: the min takes 1024 d times the group's activation sum off.
    static constexpr float kBias = 1024.0F;
    uint4 head;
    uint4 qs[2];
    uint4 qh[2];

    __device__ __forceinline__ void load(const GemmArgs& a, int row, int sb, int q) {
        const std::uint8_t* b =
            a.codes + (static_cast<std::int64_t>(row) * a.groups_per_row + sb) * kBlockBytes;
        head          = *reinterpret_cast<const uint4*>(b);
        const auto* p = reinterpret_cast<const uint4*>(b + (Bits == 4 ? 16 : 48) + 32 * q);
        qs[0]         = p[0];
        qs[1]         = p[1];
        if constexpr (Bits == 5) {
            qh[0] = reinterpret_cast<const uint4*>(b + 16)[0];
            qh[1] = reinterpret_cast<const uint4*>(b + 16)[1];
        }
    }

    __device__ __forceinline__ void clear() {
        head  = make_uint4(0, 0, 0, 0);
        qs[0] = qs[1] = make_uint4(0, 0, 0, 0);
        qh[0] = qh[1] = make_uint4(0, 0, 0, 0);
    }

    // Sub-block 2q + g's scale and min (ggml get_scale_min_k4) applied to d and dmin.
    __device__ __forceinline__ void scale(int q, int g, float& ds, float& dm) const {
        const int j       = 2 * q + g;
        const unsigned s0 = head.y, s4 = head.z, s8 = head.w;
        unsigned sc, mn;
        if (j < 4) {
            sc = (s0 >> (8 * j)) & 63U;
            mn = (s4 >> (8 * j)) & 63U;
        } else {
            const int jj = 8 * (j - 4);
            sc           = ((s8 >> jj) & 0x0fU) | (((s0 >> jj) >> 6 & 3U) << 4);
            mn           = ((s8 >> jj) >> 4 & 0x0fU) | (((s4 >> jj) >> 6 & 3U) << 4);
        }
        ds = half_bits_f32(head.x) * static_cast<float>(sc);
        dm = half_bits_f32(head.x >> 16) * static_cast<float>(mn);
    }

    // MMA m: run m / 8 (0: low nibbles, sub-block 2q; 1: high, 2q + 1), code word m % 8.
    __device__ __forceinline__ void fragment(int m, int q, unsigned& a0, unsigned& a1) const {
        const int run = m >> 3, wd = m & 7;
        const unsigned words[8] = {qs[0].x, qs[0].y, qs[0].z, qs[0].w,
                                   qs[1].x, qs[1].y, qs[1].z, qs[1].w};
        const unsigned w        = words[wd] >> (4 * run);
        unsigned lo = nibbles_h2(w), hi = nibbles_h2(w >> 8);
        if constexpr (Bits == 5) {
            const unsigned hwords[8] = {qh[0].x, qh[0].y, qh[0].z, qh[0].w,
                                        qh[1].x, qh[1].y, qh[1].z, qh[1].w};
            const unsigned hb        = hwords[wd] >> (2 * q + run);
            lo |= (hb << 4) & 0x00100010U;
            hi |= (hb >> 4) & 0x00100010U;
        }
        a0 = lo;
        a1 = hi;
    }

    __device__ static constexpr int dest_run(int src) { return src; }
};

// Q4_K in fragment order (ggml_frag_v1, core/ggml_frag.cpp): the 16 head bytes as published, each
// 64-byte half of the codes a 4 x 4 transpose of its 4-byte words (fragment word (w, i) holds the
// published row i's word w). Quad-pair q's published rows 2 (q % 2) and 2 (q % 2) + 1 of half q / 2
// come from components 2 (q % 2) and + 1 of the half's four 16-byte words: the same codes in the
// same order as HK45<4>, so both layouts of a bank give the same sums.
struct HK4Frag : HK45<4> {
    __device__ __forceinline__ void load(const GemmArgs& a, int row, int sb, int q) {
        const std::uint8_t* b =
            a.codes + (static_cast<std::int64_t>(row) * a.groups_per_row + sb) * kBlockBytes;
        head          = *reinterpret_cast<const uint4*>(b);
        const auto* p = reinterpret_cast<const uint4*>(b + 16 + 64 * (q >> 1));
        const uint4 w0 = p[0], w1 = p[1], w2 = p[2], w3 = p[3];
        if (q & 1) {
            qs[0] = make_uint4(w0.z, w1.z, w2.z, w3.z);
            qs[1] = make_uint4(w0.w, w1.w, w2.w, w3.w);
        } else {
            qs[0] = make_uint4(w0.x, w1.x, w2.x, w3.x);
            qs[1] = make_uint4(w0.y, w1.y, w2.y, w3.y);
        }
    }
};

// ggml's 32-value blocks (Q4_0, Q4_1, Q5_0, Q5_1) as planes: 16 code bytes a block (byte j: value j
// low nibble, j + 16 high), Q5's 32 fifth bits a word, a binary16 d (Q4_0, Q5_0) or d and m (Q4_1,
// Q5_1) a block. Quad-pair q: blocks 8 sb + 2q and + 1 (K a multiple of 64). The operand is 1024 +
// code (the f16 bit trick, exact): the zero point and the 1024 come off the A fragment (kBias), and a
// min's m adds against the run's activation sum (w = d q + m: the sum's -dm is -(-m)).
template <int Bits, bool Min>
struct HG32 {
    static constexpr int kGroups = 2;
    static constexpr bool kMin   = Min; // Q4_1 / Q5_1 take the run sums
    static constexpr float kBias = 1024.0F + (Min ? 0.0F : (Bits == 4 ? 8.0F : 16.0F));
    uint4 c[2];
    unsigned qh[2];
    unsigned dm[2];

    __device__ __forceinline__ void load(const GemmArgs& a, int row, int sb, int q) {
        if (8 * sb + 2 * q >= a.k / 32) {
            clear();
            return;
        }
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + 8 * sb + 2 * q;
        const auto* p          = reinterpret_cast<const uint4*>(a.codes + blk * 16);
        c[0]                   = p[0];
        c[1]                   = p[1];
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            qh[b] = Bits == 5 ? reinterpret_cast<const unsigned*>(a.high)[blk + b] : 0U;
            dm[b] = Min ? reinterpret_cast<const unsigned*>(a.scales)[blk + b]
                        : reinterpret_cast<const unsigned short*>(a.scales)[blk + b];
        }
    }

    __device__ __forceinline__ void clear() {
        c[0] = c[1] = make_uint4(0, 0, 0, 0);
        qh[0] = qh[1] = dm[0] = dm[1] = 0;
    }

    __device__ __forceinline__ void scale(int, int g, float& ds, float& d_min) const {
        ds    = half_bits_f32(dm[g]);
        d_min = Min ? -half_bits_f32(dm[g] >> 16) : 0.0F;
    }

    // MMA m: block m / 8; within it, MMA j = m % 8 takes word j % 4's low (j < 4) or high nibbles:
    // values 16 (j / 4) + 4 (j % 4) .. + 3, as (v0, v2) and (v1, v3).
    __device__ __forceinline__ void fragment(int m, int, unsigned& a0, unsigned& a1) const {
        const int b = m >> 3, j = m & 7, nib = j >> 2;
        const unsigned words[4] = {c[b].x, c[b].y, c[b].z, c[b].w};
        const unsigned w        = words[j & 3] >> (4 * nib);
        unsigned lo = nibbles_h2(w), hi = nibbles_h2(w >> 8);
        if constexpr (Bits == 5) {
            const unsigned hb = qh[b] >> (16 * nib + 4 * (j & 3));
            lo |= ((hb & 1U) << 4) | ((hb & 4U) << 18);
            hi |= ((hb & 2U) << 3) | ((hb & 8U) << 17);
        }
        a0 = lo;
        a1 = hi;
    }

    __device__ static constexpr int dest_run(int src) { return src; }
};

// Q6_K planes: 128 ql then 64 qh bytes a superblock (codes), 16 int8 scales (high), binary16 d
// (scales). Quad-pair q: half h = q / 2, ql bytes 64 h + 32 (q % 2) .. + 31 (low nibbles quarter
// q % 2, high nibbles quarter q % 2 + 2), qh bytes 32 h .. + 31 (bit pairs 2 (q % 2) and
// 2 (q % 2) + 4). Scale groups of 16 values: group g = quarter q % 2 + 2 (g / 2), half g % 2.
struct HK6 {
    static constexpr int kGroups = 4;
    static constexpr bool kMin   = false;
    uint4 ql[2];
    uint4 qh[2];
    uint4 sc;
    unsigned d;

    __device__ __forceinline__ void load(const GemmArgs& a, int row, int sb, int q) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + sb;
        const int h = q >> 1, qq = q & 1;
        const std::uint8_t* b = a.codes + blk * 192;
        const auto* l         = reinterpret_cast<const uint4*>(b + 64 * h + 32 * qq);
        const auto* e         = reinterpret_cast<const uint4*>(b + 128 + 32 * h);
        ql[0]                 = l[0];
        ql[1]                 = l[1];
        qh[0]                 = e[0];
        qh[1]                 = e[1];
        sc                    = *reinterpret_cast<const uint4*>(a.high + blk * 16);
        d                     = reinterpret_cast<const unsigned short*>(a.scales)[blk];
    }

    __device__ __forceinline__ void clear() {
        ql[0] = ql[1] = qh[0] = qh[1] = sc = make_uint4(0, 0, 0, 0);
        d                                  = 0;
    }

    __device__ __forceinline__ void scale(int q, int g, float& ds, float& dm) const {
        const unsigned words[4] = {sc.x, sc.y, sc.z, sc.w};
        const int i             = 8 * (q >> 1) + 2 * ((q & 1) + 2 * (g >> 1)) + (g & 1);
        const int v = static_cast<signed char>((words[i >> 2] >> (8 * (i & 3))) & 0xffU);
        ds          = half_bits_f32(d) * static_cast<float>(v);
        dm          = 0.0F;
    }

    __device__ __forceinline__ void fragment(int m, int q, unsigned& a0, unsigned& a1) const {
        const int run = m >> 3, wd = m & 7;
        const unsigned lw[8] = {ql[0].x, ql[0].y, ql[0].z, ql[0].w,
                                ql[1].x, ql[1].y, ql[1].z, ql[1].w};
        const unsigned hw[8] = {qh[0].x, qh[0].y, qh[0].z, qh[0].w,
                                qh[1].x, qh[1].y, qh[1].z, qh[1].w};
        const unsigned w     = lw[wd] >> (4 * run);
        const unsigned hb    = hw[wd] >> (2 * (q & 1) + 4 * run);
        // Low four bits from ql, the two above from qh, under the exponent of 1024, then 1024 + 32
        // off (the zero point). Two LOP3s a pair (622 -> 595 loop instructions) measured 4 us
        // slower on the 128-CTA Q6_K add (GV100), so the plain expression stays.
        const unsigned lo =
            ((w & 0x000f000fU) | ((hb << 4) & 0x00300030U)) | 0x64006400U;
        const unsigned hi =
            (((w >> 8) & 0x000f000fU) | ((hb >> 4) & 0x00300030U)) | 0x64006400U;
        a0 = biased_h2(lo, 0x64206420U);
        a1 = biased_h2(hi, 0x64206420U);
    }

    // Source run 4 h + quarter goes to quad-pair 2 h + quarter % 2, half quarter / 2 of it.
    __device__ static constexpr int dest_run(int src) {
        return 4 * (src >> 2) + 2 * (src & 1) + ((src >> 1) & 1);
    }
};

// Q8 G32: int8 codes in row order (codes), a binary16 scale a group of 32 (scales). Quad-pair q:
// groups 8 sb + 2q and 8 sb + 2q + 1.
struct HQ8 {
    static constexpr int kGroups = 2;
    static constexpr bool kMin   = false;
    uint4 c[4];
    unsigned d[2];

    __device__ __forceinline__ void load(const GemmArgs& a, int row, int sb, int q) {
        // A row of K % 256 != 0 (K a multiple of 64: a quad-pair's two groups are both in it or
        // both past it) ends inside its last superblock: the quad-pairs past it hold zeros. The
        // bound is K's groups, not the stored row's (padded to 128 values, its padding never read).
        if (8 * sb + 2 * q >= a.k / 32) {
            clear();
            return;
        }
        const std::int64_t g = static_cast<std::int64_t>(row) * a.groups_per_row + 8 * sb + 2 * q;
        const auto* p        = reinterpret_cast<const uint4*>(a.codes + g * 32);
        c[0]                 = p[0];
        c[1]                 = p[1];
        c[2]                 = p[2];
        c[3]                 = p[3];
        const auto* s        = reinterpret_cast<const unsigned short*>(a.scales);
        d[0]                 = s[g];
        d[1]                 = s[g + 1];
    }

    __device__ __forceinline__ void clear() {
        c[0] = c[1] = c[2] = c[3] = make_uint4(0, 0, 0, 0);
        d[0] = d[1] = 0;
    }

    __device__ __forceinline__ void scale(int, int g, float& ds, float& dm) const {
        ds = half_bits_f32(d[g]);
        dm = 0.0F;
    }

    __device__ __forceinline__ void fragment(int m, int, unsigned& a0, unsigned& a1) const {
        const unsigned words[16] = {c[0].x, c[0].y, c[0].z, c[0].w, c[1].x, c[1].y, c[1].z, c[1].w,
                                    c[2].x, c[2].y, c[2].z, c[2].w, c[3].x, c[3].y, c[3].z, c[3].w};
        const unsigned w         = words[m] ^ 0x80808080U; // code + 128
        // (byte 0, byte 2) and (byte 1, byte 3) under the 0x64 exponent byte, in one PRMT each
        // (the exponent bytes from the second operand); 1024 + 128 back.
        a0 = biased_h2(__byte_perm(w, 0x64646464U, 0x4240U), 0x64806480U);
        a1 = biased_h2(__byte_perm(w, 0x64646464U, 0x4341U), 0x64806480U);
    }

    __device__ static constexpr int dest_run(int src) { return src; }
};

// BF16 rows (the MoE router): K contiguous. Quad-pair q: values 64q .. 64q + 63 of the superblock
// (K a multiple of 64), converted to f16 pairs in the fragment; one scale group of 1.
struct HB16 {
    static constexpr int kGroups = 1;
    static constexpr bool kMin   = false;
    uint4 v[8];

    __device__ __forceinline__ void load(const GemmArgs& a, int row, int sb, int q) {
        const int k0 = 256 * sb + 64 * q;
        if (k0 >= a.k) {
            clear();
            return;
        }
        const auto* p = reinterpret_cast<const uint4*>(a.codes + (static_cast<std::int64_t>(row) * a.k + k0) * 2);
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = p[i];
    }

    __device__ __forceinline__ void clear() {
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = make_uint4(0, 0, 0, 0);
    }

    __device__ __forceinline__ void scale(int, int, float& ds, float& dm) const {
        ds = 1.0F;
        dm = 0.0F;
    }

    // MMA m: values 4m .. 4m + 3, words 2m (v0, v1) and 2m + 1 (v2, v3) as (v0, v2), (v1, v3).
    __device__ __forceinline__ void fragment(int m, int, unsigned& a0, unsigned& a1) const {
        const uint4 u      = v[m >> 1];
        const unsigned w0  = (m & 1) ? u.z : u.x;
        const unsigned w1  = (m & 1) ? u.w : u.y;
        a0 = h2_bits(__floats2half2_rn(__uint_as_float(w0 << 16), __uint_as_float(w1 << 16)));
        a1 = h2_bits(__floats2half2_rn(__uint_as_float(w0 & 0xffff0000U),
                                       __uint_as_float(w1 & 0xffff0000U)));
    }

    __device__ static constexpr int dest_run(int src) { return src; }
};

// MXFP4: 32-value blocks of 16 code bytes (byte j: value 2j low nibble, 2j + 1 high) with an
// E8M0 scale byte each. Quad-pair q: blocks 8 sb + 2q and 8 sb + 2q + 1, contiguous (32 bytes). The
// operand is twice the E2M1 value, an f16 integer (0, 1, 2, 3, 4, 6, 8, 12 and their negatives):
// one PRMT a pair picks the f16 high bytes from a table (the low bytes are zero), the sign bit
// follows; the group's scale is half its E8M0 value.
struct HX4 {
    static constexpr int kGroups = 2;
    static constexpr bool kMin   = false;
    uint4 c[2];
    unsigned e[2];

    __device__ __forceinline__ void load(const GemmArgs& a, int row, int sb, int q) {
        // K a multiple of 64: a quad-pair's two blocks are both in the row or both past it.
        if (8 * sb + 2 * q >= a.k / 32) {
            clear();
            return;
        }
        const std::int64_t blk = static_cast<std::int64_t>(row) * (a.k / 32) + 8 * sb + 2 * q;
        const auto* p          = reinterpret_cast<const uint4*>(a.codes + blk * 16);
        c[0]                   = p[0];
        c[1]                   = p[1];
        e[0]                   = a.scales[blk];
        e[1]                   = a.scales[blk + 1];
    }

    __device__ __forceinline__ void clear() {
        c[0] = c[1] = make_uint4(0, 0, 0, 0);
        e[0] = e[1] = 0;
    }

    __device__ __forceinline__ void scale(int, int g, float& ds, float& dm) const {
        ds = 0.5F * e8m0_value(e[g]);
        dm = 0.0F;
    }

    // MMA m: block m / 8, code word (m / 2) % 4, its half m % 2: values v0..v3 as (v0, v2) and
    // (v1, v3), the activation staged with the middle two of every four swapped.
    __device__ __forceinline__ void fragment(int m, int, unsigned& a0, unsigned& a1) const {
        const unsigned words[8] = {c[0].x, c[0].y, c[0].z, c[0].w, c[1].x, c[1].y, c[1].z, c[1].w};
        const unsigned w        = words[m >> 1] >> (16 * (m & 1));
        constexpr unsigned kLo  = 0x42403C00U; // f16 high bytes of 0, 1, 2, 3
        constexpr unsigned kHi  = 0x4A484644U; // of 4, 6, 8, 12
        a0 = __byte_perm(kLo, kHi, (w << 4) & 0x7070U) | ((w << 12) & 0x8000U) |
             ((w << 20) & 0x80000000U);
        a1 = __byte_perm(kLo, kHi, w & 0x7070U) | ((w << 8) & 0x8000U) | ((w << 16) & 0x80000000U);
    }

    __device__ static constexpr int dest_run(int src) { return src; }
};

template <class F, int NTok>
__device__ __forceinline__ void stage_store(__half* act, float* sums, const uint2 (&v)[NTok / 2]) {
#pragma unroll
    for (int i = 0; i < NTok / 2; ++i) {
        const int task = static_cast<int>(threadIdx.x) + i * kWarps * 32;
        const int t    = task >> 6;
        const int g    = task & 63;
        const int run  = F::dest_run(g >> 3);
        const int dst  = run * 8 + (g & 7); // destination group, 0..63
        const int q = dst >> 4, s = dst & 15;
        // bf16 (v0, v1), (v2, v3) -> f16 (v0, v2), (v1, v3)
        const unsigned lo = bf16x2_to_f16x2(__byte_perm(v[i].x, v[i].y, 0x5410U)); // v0, v2
        const unsigned hi = bf16x2_to_f16x2(__byte_perm(v[i].x, v[i].y, 0x7632U)); // v1, v3
        *reinterpret_cast<uint2*>(act + q * kQuadStride<NTok> +
                                  (s * NTok + token_slot<NTok>(s, t)) * 4) = make_uint2(lo, hi);
        if constexpr (F::kMin) {
            // The eight groups of a run sit on eight neighbouring lanes (task & 7 = g & 7).
            const float2 l = __half22float2(bits_h2(lo));
            const float2 h = __half22float2(bits_h2(hi));
            float sum      = (l.x + h.x) + (l.y + h.y);
            sum += __shfl_xor_sync(0xffffffffU, sum, 1);
            sum += __shfl_xor_sync(0xffffffffU, sum, 2);
            sum += __shfl_xor_sync(0xffffffffU, sum, 4);
            if ((g & 7) == 0) sums[t * 8 + run] = sum;
        }
    }
}

} // namespace ninferno::ops::detail::generic::volta
