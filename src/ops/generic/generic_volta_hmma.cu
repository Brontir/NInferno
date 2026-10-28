// Volta's speculative-verify widths (T = 2..16) on its tensor cores. sm_70 has no mma.sync of the
// shapes the register-MMA tiles issue (m16n8k16 BF16/F16, m16n8k32 s8): there those tiles run on
// FP32 FMAs over warp shuffles (ops/common/mma.cuh) and ggml's vector kernel re-reads the
// activation for every column on dp4a, so both grow with T and neither streams the weight at the
// DRAM rate (GV100, Qwen3.5-9B Q4_K_M, dflash:7 round: Q6_K LM head 163 GB/s, target verify
// 176-312 GB/s, of 870). Volta's one tensor-core shape is mma.m8n8k4 with f16 operands
// (HMMA.884): this kernel turns each weight code into f16 once, straight from the lane's own loads
// into an A fragment, and multiplies it against eight (or sixteen) token columns.
//
// The m8n8k4 fragments (PTX ISA): a warp runs four independent products, one per quad-pair q
// (lanes 4q..4q+3 and 16+4q..16+4q+3). Lane f of a quad-pair (f = lane % 4, + 4 for the high
// lanes) holds row f of A (k 0..3) and column f of B (k 0..3); accumulator i of a lane is row
// (lane & 1) + (i & 2) (+ 4 for the high lanes), column (i & 4) + (lane & 2) + (i & 1). Which four
// K positions the four k slots stand for is free, as long as A and B agree: so each lane takes the
// codes of its own bytes in the order they come out of them, and the CTA stages the activation in
// that order once (bf16 to f16, as the T >= 17 tiled route does).
//
// Layout: a warp holds eight weight rows (fragment row f = weight row f) and splits each 256-value
// superblock over its quad-pairs, 64 values each (the format's own runs of 32: Q4_K / Q5_K
// sub-blocks 2q and 2q+1, Q6_K quarters q % 2 and q % 2 + 2 of half q / 2, Q8 groups 2q and
// 2q+1). One code word (four bytes) gives two half2 per run: (byte 0, byte 2) and (byte 1, byte 3),
// so the staged activation swaps the middle two of every four values.
//
// Arithmetic: the A operand is the code itself (less its zero point; Q4_K / Q5_K: plus 1024), an
// exact f16 integer, so the tensor cores sum exact products of codes and f16 activations in FP32
// over one scale group (16 or 32 values); the group's scale (and a K-quant's min against the
// group's activation sum) is applied in FP32 after it. Q4_K / Q5_K leave the 1024 the f16 bit
// trick puts under each code in the operand and take 1024 d times the group's activation sum off
// with the min, which already reads that sum (HK45::kBias): one FFMA a group instead of an HSUB2 a
// code pair, at FP32 rounding of the 1024-scaled sum. No weight value is ever rounded to f16,
// whatever its scale. The
// groups add in K order, the quad-pairs' sums meet at the end in a fixed order
// ((q0 + q1) + (q2 + q3)): a token's bytes depend on its own column only, never on the call's T or
// its other tokens, and repeat run to run.
#include "core/device.h"
#include "core/device_class.h"
#include "ops/common/mma.cuh"
#include "ops/generic/generic_gemm.cuh"
#include "ops/generic/generic_gemm_internal.h"

#include <cuda_fp16.h>

#include <stdexcept>

namespace ninferno::ops::detail::generic {
namespace {

constexpr int kWarps    = 4; // a CTA: four warps of eight rows
constexpr int kWarpRows = 8; // one weight row per fragment row
constexpr int kCtaRows  = kWarps * kWarpRows;
// The staged activation, per quad-pair: 16 groups of four values for each of NTok tokens, then 16
// halves of padding so the quad-pairs' fragment loads fall on different banks.
template <int NTok>
constexpr int kQuadStride = 16 * NTok * 4 + 16;

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

// (a & imm) | c in one LOP3, c a register (one SASS LOP3 takes a single immediate).
template <unsigned Imm>
__device__ __forceinline__ unsigned and_or(unsigned a, unsigned c) {
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "n"(Imm), "r"(c));
    return r;
}

// The f16 pair of biased codes (1024 + code, from nibbles_h2 / and_or) less `bias_bits`: exact.
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
        // off (the zero point). Two LOP3s a pair through and_or (622 -> 595 loop instructions) lost
        // 4 us on the 128-CTA Q6_K add: one CTA an SM issues each warp's chain back to back, and
        // and_or chains the two masks where the compiler's four instructions run side by side.
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

// Stage superblock `sb` of the activation as f16 in fragment order: run src of token t goes to
// quad-pair D / 2, half D % 2 of it (D = dest_run(src)), the middle two of every four swapped.
// For a format with a min term, also each staged run's sum over its 32 values (the f16 values the
// MMA multiplies, summed in FP32 in a fixed order) for token t: sums[t * 8 + D].
template <class F, int NTok>
__device__ __forceinline__ void stage_load(const GemmArgs& a, int sb, uint2 (&v)[NTok / 2]) {
    const auto* x = static_cast<const __nv_bfloat16*>(a.x);
#pragma unroll
    for (int i = 0; i < NTok / 2; ++i) {
        const int task = static_cast<int>(threadIdx.x) + i * kWarps * 32; // NTok * 64 tasks
        const int t    = task >> 6;
        const int g    = task & 63; // four-value group of the superblock
        v[i]           = t < a.tokens ? *reinterpret_cast<const uint2*>(
                                  x + static_cast<std::int64_t>(t) * a.ldx + 256 * sb + 4 * g)
                                      : make_uint2(0, 0);
    }
}

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

template <class F, Epilogue E, int NG, int Ahead>
__global__ __launch_bounds__(kWarps * 32) void volta_hmma_kernel(GemmArgs a) {
#if NINFERNO_HMMA884
    constexpr int kSets   = E == Epilogue::SwiGlu ? 2 : 1;
    constexpr int kNTok   = 8 * NG;
    constexpr int kGroups = F::kGroups;
    constexpr int kSteps  = 16 / kGroups; // MMAs a scale group
    __shared__ __align__(16) __half act[2][4 * kQuadStride<kNTok>];
    __shared__ float sums[2][F::kMin ? kNTok * 8 : 1];

    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int q     = (lane >> 2) & 3;
    const int f     = (lane & 3) + ((lane >> 4) << 2);
    const int l     = static_cast<int>(blockIdx.x) * kCtaRows + warp * kWarpRows + f; // A row
    const bool live = l < a.rows;
    const int sbs   = a.k >> 8;
    // The lanes holding the A rows of this lane's accumulators: rows (lane & 1) + 4 (lane / 16)
    // and that + 2, in this quad-pair.
    const int src_a = 4 * q + (lane & 1) + ((lane >> 4) << 4);

    float acc[kSets][NG][8];
#    pragma unroll
    for (int s = 0; s < kSets; ++s)
#    pragma unroll
        for (int g = 0; g < NG; ++g)
#    pragma unroll
            for (int i = 0; i < 8; ++i) acc[s][g][i] = 0.0F;

    // The weight stream Ahead superblocks ahead: cur, then nx1 (and nx2) in flight. Little's law:
    // an SM needs ~5.7 KB in flight for the measured 660 GB/s over ~1000 cycles of latency at the
    // 1.44 GHz it runs under load (GV100), and a CTA keeps 32 rows of a superblock in flight (Q4_K
    // 4.6 KB, SwiGLU 9.2; the other formats' superblocks are larger). With two CTAs or more an SM
    // one superblock ahead covers it; the second one held 23 more registers a lane (gate/up
    // 160 -> 137) and added issue work to a loop that is issue-bound. A grid of fewer CTAs than
    // two an SM keeps the second (launch_format).
    F cur[kSets], nx1[kSets], nx2[kSets];
#    pragma unroll
    for (int s = 0; s < kSets; ++s) {
        const int row = a.row_begin + l + (s == 1 ? a.up_offset : 0);
        cur[s].clear();
        nx1[s].clear();
        nx2[s].clear();
        if (live) {
            cur[s].load(a, row, 0, q);
            if (Ahead == 2 && sbs > 1) nx1[s].load(a, row, 1, q);
        }
    }
    uint2 xv[kNTok / 2];
    stage_load<F, kNTok>(a, 0, xv);
    stage_store<F, kNTok>(act[0], sums[0], xv);
    __syncthreads();

#    pragma unroll 1
    for (int sb = 0; sb < sbs; ++sb) {
        const bool more = sb + 1 < sbs;
#    pragma unroll
        for (int s = 0; s < kSets; ++s) {
            const int row = a.row_begin + l + (s == 1 ? a.up_offset : 0);
            if (Ahead == 1 && more && live) nx1[s].load(a, row, sb + 1, q);
            if (Ahead == 2 && sb + 2 < sbs && live) nx2[s].load(a, row, sb + 2, q);
        }
        if (more) stage_load<F, kNTok>(a, sb + 1, xv);

        const __half* at = act[sb & 1] + q * kQuadStride<kNTok>;
#    pragma unroll
        for (int gr = 0; gr < kGroups; ++gr) {
            float part[kSets][NG][8];
#    pragma unroll
            for (int s = 0; s < kSets; ++s)
#    pragma unroll
                for (int g = 0; g < NG; ++g)
#    pragma unroll
                    for (int i = 0; i < 8; ++i) part[s][g][i] = 0.0F;
#    pragma unroll
            for (int st = 0; st < kSteps; ++st) {
                const int m = gr * kSteps + st;
                uint2 bf[NG];
#    pragma unroll
                for (int g = 0; g < NG; ++g) {
                    bf[g] = *reinterpret_cast<const uint2*>(
                        at + (m * kNTok + token_slot<kNTok>(m, 8 * g + f)) * 4);
                }
#    pragma unroll
                for (int s = 0; s < kSets; ++s) {
                    unsigned a0, a1;
                    cur[s].fragment(m, q, a0, a1);
#    pragma unroll
                    for (int g = 0; g < NG; ++g)
                        mma_m8n8k4_f16(part[s][g], a0, a1, bf[g].x, bf[g].y);
                }
            }
            // The group's scale (and min) of the accumulators' rows, from the lanes holding them.
#    pragma unroll
            for (int s = 0; s < kSets; ++s) {
                float ds, dm;
                cur[s].scale(q, gr, ds, dm);
                if constexpr (requires { F::kBias; }) dm = fmaf(F::kBias, ds, dm);
                const float ds0 = __shfl_sync(0xffffffffU, ds, src_a);
                const float ds2 = __shfl_sync(0xffffffffU, ds, src_a + 2);
                float dm0 = 0.0F, dm2 = 0.0F;
                if constexpr (F::kMin) {
                    dm0 = __shfl_sync(0xffffffffU, dm, src_a);
                    dm2 = __shfl_sync(0xffffffffU, dm, src_a + 2);
                }
#    pragma unroll
                for (int g = 0; g < NG; ++g)
#    pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const float d = (i & 2) ? ds2 : ds0;
                        acc[s][g][i]  = fmaf(d, part[s][g][i], acc[s][g][i]);
                        if constexpr (F::kMin) {
                            const int token = 8 * g + (i & 4) + (lane & 2) + (i & 1);
                            const float xs  = sums[sb & 1][token * 8 + 2 * q + gr];
                            acc[s][g][i]    = fmaf(-((i & 2) ? dm2 : dm0), xs, acc[s][g][i]);
                        }
                    }
            }
        }
        if (more) stage_store<F, kNTok>(act[(sb + 1) & 1], sums[(sb + 1) & 1], xv);
        __syncthreads();
#    pragma unroll
        for (int s = 0; s < kSets; ++s) {
            cur[s] = nx1[s];
            if (Ahead == 2) nx1[s] = nx2[s];
        }
    }

    // The quad-pairs' partial sums: ((q0 + q1) + (q2 + q3)) in every lane of the four.
#    pragma unroll
    for (int s = 0; s < kSets; ++s)
#    pragma unroll
        for (int g = 0; g < NG; ++g)
#    pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[s][g][i] += __shfl_xor_sync(0xffffffffU, acc[s][g][i], 4);
                acc[s][g][i] += __shfl_xor_sync(0xffffffffU, acc[s][g][i], 8);
            }
    // Quad-pair q stores accumulators 2q and 2q + 1.
    const int row0 = static_cast<int>(blockIdx.x) * kCtaRows + warp * kWarpRows;
#    pragma unroll
    for (int i = 0; i < 8; ++i) {
        if ((i >> 1) != q) continue;
        const int row = row0 + (lane & 1) + (i & 2) + ((lane >> 4) << 2);
        if (row >= a.rows) continue;
#    pragma unroll
        for (int g = 0; g < NG; ++g) {
            const int token = 8 * g + (i & 4) + (lane & 2) + (i & 1);
            if (token >= a.tokens) continue;
            float value = acc[0][g][i];
            if constexpr (E == Epilogue::SwiGlu)
                value = generic_gate(a, value) * acc[kSets - 1][g][i];
            store_output<E>(a, row, token, value);
        }
    }
#else
    __trap();
#endif
}

template <class F, Epilogue E>
void launch_format(const GemmArgs& args, cudaStream_t stream) {
    const unsigned blocks = static_cast<unsigned>((args.rows + kCtaRows - 1) / kCtaRows);
    // Two CTAs an SM or more: one superblock ahead (volta_hmma_kernel). GV100 dflash:7 round:
    // gate/up (384 CTAs) 156.5 -> 152.6 us; a 4096-row add (128 CTAs) 71.1 -> 73.2 one ahead.
    const bool one_ahead = blocks >= 2U * static_cast<unsigned>(device_class().sms);
    if (args.tokens <= 8) {
        if (one_ahead) {
            volta_hmma_kernel<F, E, 1, 1><<<blocks, kWarps * 32, 0, stream>>>(args);
        } else {
            volta_hmma_kernel<F, E, 1, 2><<<blocks, kWarps * 32, 0, stream>>>(args);
        }
    } else if (one_ahead) {
        volta_hmma_kernel<F, E, 2, 1><<<blocks, kWarps * 32, 0, stream>>>(args);
    } else {
        volta_hmma_kernel<F, E, 2, 2><<<blocks, kWarps * 32, 0, stream>>>(args);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class F>
void launch_epilogue(Epilogue epilogue, const GemmArgs& args, cudaStream_t stream) {
    switch (epilogue) {
    case Epilogue::Store:
        launch_format<F, Epilogue::Store>(args, stream);
        return;
    case Epilogue::AddResidual:
        launch_format<F, Epilogue::AddResidual>(args, stream);
        return;
    case Epilogue::SwiGlu:
        launch_format<F, Epilogue::SwiGlu>(args, stream);
        return;
    }
}

} // namespace

bool volta_hmma_codec(WeightCodec codec) {
    return codec == WeightCodec::GgmlQ4K || codec == WeightCodec::GgmlQ5K ||
           codec == WeightCodec::GgmlQ6K || codec == WeightCodec::Q8;
}

bool volta_hmma_takes(WeightCodec codec, const GemmArgs& args) {
    // join_previous asks for a programmatic dependent launch, which sm_70 does not have: stream
    // order already runs the call after the one before it (a segmented projection's later parts).
    if (args.lane_local == 0 || args.tokens < kVoltaHmmaMinTokens ||
        args.tokens > kVoltaHmmaMaxTokens) {
        return false;
    }
    // Whole superblocks; the staging reads four bf16 values at a time.
    return volta_hmma_codec(codec) && (args.k % 256) == 0 && (args.ldx % 4) == 0 &&
           (reinterpret_cast<std::uintptr_t>(args.x) % 8) == 0;
}

void launch_volta_hmma(WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                       cudaStream_t stream) {
    switch (codec) {
    case WeightCodec::GgmlQ4K:
        launch_epilogue<HK45<4>>(epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ5K:
        launch_epilogue<HK45<5>>(epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ6K:
        launch_epilogue<HK6>(epilogue, args, stream);
        return;
    case WeightCodec::Q8:
        launch_epilogue<HQ8>(epilogue, args, stream);
        return;
    default:
        throw std::invalid_argument("volta_hmma: unsupported codec");
    }
}

} // namespace ninferno::ops::detail::generic
