// Volta's speculative-verify widths (T = 2..16) on its tensor cores. sm_70 has no mma.sync of the
// shapes the register-MMA tiles issue (m16n8k16 BF16/F16, m16n8k32 s8): there those tiles run on
// FP32 FMAs over warp shuffles (ops/common/mma.cuh) and ggml's vector kernel re-reads the
// activation for every column on dp4a, so both grow with T and neither streams the weight at the
// DRAM rate (GV100, Qwen3.5-9B Q4_K_M, dflash:7 round: Q6_K LM head 163 GB/s, target verify
// 176-312 GB/s, of 870). Volta's one tensor-core shape is mma.m8n8k4 with f16 operands
// (HMMA.884): this kernel decodes each weight value once, straight from the lane's own loads into
// an f16 A fragment, and multiplies it against eight (or sixteen) token columns.
//
// The m8n8k4 fragments (PTX ISA): a warp runs four independent products, one per quad-pair q
// (lanes 4q..4q+3 and 16+4q..16+4q+3). Lane f of a quad-pair (f = lane % 4, + 4 for the high
// lanes) holds row f of A (k 0..3) and column f of B (k 0..3). Which four K positions the four k
// slots stand for is free, as long as A and B agree: so each lane decodes the values of its own
// code bytes in the order they come out of the bytes, and the CTA stages the activation in that
// order once (bf16 to f16, as the T >= 17 tiled route does).
//
// Layout: a warp holds eight weight rows (fragment row f = weight row f) and splits each 256-value
// superblock over its quad-pairs, 64 values each (the format's own runs of 32: Q4_K / Q5_K
// sub-blocks 2q and 2q+1, Q6_K quarters q % 2 and q % 2 + 2 of half q / 2, Q8 groups 2q and
// 2q+1). One code word (four bytes) gives two half2 per run: (byte 0, byte 2) and (byte 1, byte 3),
// so the staged activation swaps the middle two of every four values. The quad-pairs' partial
// sums meet at the end in a fixed order ((q0 + q1) + (q2 + q3)): a token's bytes depend on its own
// column only, never on the call's T or its other tokens, and repeat run to run.
//
// Weight values as f16: Q4_K / Q5_K w = d sc q - dmin m, Q6_K w = d sc (q - 32), Q8 w = d q, the
// code exact in f16 (the 0x6400 exponent trick less 1024), then one f16 FMA (or product) with the
// group's scale and min rounded to f16. A group whose values could leave the f16 range (scales no
// real model has) takes the FP32 product saturated to +-65504, as Volta's CUTLASS prefill does.
#include "core/device.h"
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

__device__ __forceinline__ unsigned h2_bits(__half2 h) { return *reinterpret_cast<unsigned*>(&h); }

__device__ __forceinline__ __half2 bits_h2(unsigned u) { return *reinterpret_cast<__half2*>(&u); }

// Two codes in the low bits of each half (exact integers below 1024) as f16, less `bias`
// (1024 + the format's zero point): exact.
__device__ __forceinline__ __half2 codes_h2(unsigned bits, unsigned bias_bits) {
    return __hsub2(bits_h2(bits | 0x64006400U), bits_h2(bias_bits));
}

// One run's scale and min (w = ds q - dm) as f16 pairs, with whether the f16 path holds the run:
// |ds| max|q| + |dm| inside the f16 range by a factor of two.
struct RunScale {
    __half2 ds, ndm;
    float fds, fdm;
    bool fast;
};

__device__ __forceinline__ RunScale run_scale(float ds, float dm, float max_code) {
    RunScale s;
    s.fds  = ds;
    s.fdm  = dm;
    s.fast = fabsf(ds) * max_code + fabsf(dm) < 32768.0F;
    s.ds   = __float2half2_rn(s.fast ? ds : 0.0F);
    s.ndm  = __float2half2_rn(s.fast ? -dm : 0.0F);
    return s;
}

// w = ds q - dm for two exact codes q (f16): one f16 FMA, or the saturated FP32 product.
__device__ __forceinline__ unsigned weight_h2(__half2 q, const RunScale& s) {
    if (s.fast) return h2_bits(__hfma2(q, s.ds, s.ndm));
    const float2 f = __half22float2(q);
    const float lo = bf16_value_in_f16_range(fmaf(f.x, s.fds, -s.fdm));
    const float hi = bf16_value_in_f16_range(fmaf(f.y, s.fds, -s.fdm));
    return h2_bits(__floats2half2_rn(lo, hi));
}

__device__ __forceinline__ float half_bits_f32(unsigned bits) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}

// Q4_K / Q5_K: 144 / 176-byte blocks (d, dmin, twelve scale bytes, Q5_K's 32 high-bit bytes, 128
// nibble bytes). Quad-pair q: nibble bytes 32q .. 32q + 31 (low nibbles sub-block 2q, high 2q + 1).
template <int Bits>
struct HK45 {
    static constexpr int kBlockBytes = Bits == 4 ? 144 : 176;
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

    // Sub-block j's scale and min (ggml get_scale_min_k4) applied to d and dmin.
    __device__ __forceinline__ RunScale scale(int j) const {
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
        return run_scale(half_bits_f32(head.x) * static_cast<float>(sc),
                         half_bits_f32(head.x >> 16) * static_cast<float>(mn),
                         Bits == 4 ? 15.0F : 31.0F);
    }

    // The A fragment of mma m (0..15): run m / 8 (0: low nibbles, sub-block 2q; 1: high, 2q + 1),
    // code word m % 8.
    __device__ __forceinline__ void fragment(int m, int q, const RunScale (&s)[2], unsigned& a0,
                                             unsigned& a1) const {
        const int run = m >> 3, wd = m & 7;
        const unsigned words[8] = {qs[0].x, qs[0].y, qs[0].z, qs[0].w,
                                   qs[1].x, qs[1].y, qs[1].z, qs[1].w};
        const unsigned w        = words[wd] >> (4 * run);
        unsigned lo = w & 0x000f000fU, hi = (w >> 8) & 0x000f000fU;
        if constexpr (Bits == 5) {
            const unsigned hwords[8] = {qh[0].x, qh[0].y, qh[0].z, qh[0].w,
                                        qh[1].x, qh[1].y, qh[1].z, qh[1].w};
            const unsigned hb        = hwords[wd] >> (2 * q + run);
            lo |= (hb << 4) & 0x00100010U;
            hi |= (hb >> 4) & 0x00100010U;
        }
        a0 = weight_h2(codes_h2(lo, 0x64006400U), s[run]);
        a1 = weight_h2(codes_h2(hi, 0x64006400U), s[run]);
    }

    __device__ __forceinline__ void scales(int q, RunScale (&s)[2]) const {
        s[0] = scale(2 * q);
        s[1] = scale(2 * q + 1);
    }

    // The source run (32 values) of a staged run: the identity.
    __device__ static constexpr int dest_run(int src) { return src; }
};

// Q6_K planes: 128 ql then 64 qh bytes a superblock (codes), 16 int8 scales (high), binary16 d
// (scales). Quad-pair q: half h = q / 2, ql bytes 64 h + 32 (q % 2) .. + 31 (low nibbles quarter
// q % 2, high nibbles quarter q % 2 + 2), qh bytes 32 h .. + 31 (bit pairs 2 (q % 2) and
// 2 (q % 2) + 4).
struct HK6 {
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

    // Scales of this lane's runs: run r (quarter q % 2 + 2 r), the 16-value halves of it.
    __device__ __forceinline__ void scales(int q, RunScale (&s)[4]) const {
        const int h = q >> 1, qq = q & 1;
        const unsigned words[4] = {sc.x, sc.y, sc.z, sc.w};
        const float fd          = half_bits_f32(d);
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int i = 8 * h + 2 * (qq + 2 * r) + half;
                const int v = static_cast<signed char>((words[i >> 2] >> (8 * (i & 3))) & 0xffU);
                s[2 * r + half] = run_scale(fd * static_cast<float>(v), 0.0F, 32.0F);
            }
    }

    __device__ __forceinline__ void fragment(int m, int q, const RunScale (&s)[4], unsigned& a0,
                                             unsigned& a1) const {
        const int run = m >> 3, wd = m & 7;
        const unsigned lw[8] = {ql[0].x, ql[0].y, ql[0].z, ql[0].w,
                                ql[1].x, ql[1].y, ql[1].z, ql[1].w};
        const unsigned hw[8] = {qh[0].x, qh[0].y, qh[0].z, qh[0].w,
                                qh[1].x, qh[1].y, qh[1].z, qh[1].w};
        const unsigned w     = lw[wd] >> (4 * run);
        const unsigned hb    = hw[wd] >> (2 * (q & 1) + 4 * run);
        const unsigned lo    = (w & 0x000f000fU) | ((hb << 4) & 0x00300030U);
        const unsigned hi    = ((w >> 8) & 0x000f000fU) | ((hb >> 4) & 0x00300030U);
        // 1024 + 32: the code less its zero point 32, exact.
        const RunScale& r = s[2 * run + (wd >> 2)];
        a0                = weight_h2(codes_h2(lo, 0x64206420U), r);
        a1                = weight_h2(codes_h2(hi, 0x64206420U), r);
    }

    // Source run 4 h + quarter goes to quad-pair 2 h + quarter % 2, run quarter / 2 of it.
    __device__ static constexpr int dest_run(int src) {
        return 4 * (src >> 2) + 2 * (src & 1) + ((src >> 1) & 1);
    }
};

// Q8 G32: int8 codes in row order (codes), a binary16 scale a group of 32 (scales). Quad-pair q:
// groups 8 sb + 2q and 8 sb + 2q + 1.
struct HQ8 {
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

    __device__ __forceinline__ void scales(int, RunScale (&s)[2]) const {
        s[0] = run_scale(half_bits_f32(d[0]), 0.0F, 128.0F);
        s[1] = run_scale(half_bits_f32(d[1]), 0.0F, 128.0F);
    }

    __device__ __forceinline__ void fragment(int m, int, const RunScale (&s)[2], unsigned& a0,
                                             unsigned& a1) const {
        const unsigned words[16] = {c[0].x, c[0].y, c[0].z, c[0].w, c[1].x, c[1].y, c[1].z, c[1].w,
                                    c[2].x, c[2].y, c[2].z, c[2].w, c[3].x, c[3].y, c[3].z, c[3].w};
        const unsigned w         = words[m] ^ 0x80808080U; // code + 128
        // (byte 0, byte 2) and (byte 1, byte 3) under the 0x64 exponent byte; 1024 + 128 back.
        a0 = weight_h2(__hsub2(bits_h2(__byte_perm(w, 0x64646464U, 0x4240U)), bits_h2(0x64806480U)),
                       s[m >> 3]);
        a1 = weight_h2(__hsub2(bits_h2(__byte_perm(w, 0x64646464U, 0x4341U)), bits_h2(0x64806480U)),
                       s[m >> 3]);
    }

    __device__ static constexpr int dest_run(int src) { return src; }
};

template <class F>
struct ScaleCount {
    static constexpr int value = 2;
};

template <>
struct ScaleCount<HK6> {
    static constexpr int value = 4;
};

// Stage superblock `sb` of the activation as f16 in fragment order: run src of token t goes to
// quad-pair D / 2, run D % 2 of it (D = dest_run(src)), the middle two of every four swapped.
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
__device__ __forceinline__ void stage_store(__half* act, const uint2 (&v)[NTok / 2]) {
#pragma unroll
    for (int i = 0; i < NTok / 2; ++i) {
        const int task = static_cast<int>(threadIdx.x) + i * kWarps * 32;
        const int t    = task >> 6;
        const int g    = task & 63;
        const int dst  = F::dest_run(g >> 3) * 8 + (g & 7); // destination group, 0..63
        const int q = dst >> 4, s = dst & 15;
        // bf16 (v0, v1), (v2, v3) -> f16 (v0, v2), (v1, v3)
        const unsigned lo = __byte_perm(v[i].x, v[i].y, 0x5410U); // v0, v2
        const unsigned hi = __byte_perm(v[i].x, v[i].y, 0x7632U); // v1, v3
        *reinterpret_cast<uint2*>(act + q * kQuadStride<NTok> + (s * NTok + t) * 4) =
            make_uint2(bf16x2_to_f16x2(lo), bf16x2_to_f16x2(hi));
    }
}

template <class F, Epilogue E, int NG>
__global__ __launch_bounds__(kWarps * 32) void volta_hmma_kernel(GemmArgs a) {
#if NINFERNO_HMMA884
    constexpr int kSets = E == Epilogue::SwiGlu ? 2 : 1;
    constexpr int kNTok = 8 * NG;
    constexpr int kSc   = ScaleCount<F>::value;
    __shared__ __align__(16) __half act[2][4 * kQuadStride<kNTok>];

    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int q     = (lane >> 2) & 3;
    const int f     = (lane & 3) + ((lane >> 4) << 2);
    const int l     = static_cast<int>(blockIdx.x) * kCtaRows + warp * kWarpRows + f; // A row
    const bool live = l < a.rows;
    const int sbs   = a.k >> 8;

    float acc[kSets][NG][8];
#    pragma unroll
    for (int s = 0; s < kSets; ++s)
#    pragma unroll
        for (int g = 0; g < NG; ++g)
#    pragma unroll
            for (int i = 0; i < 8; ++i) acc[s][g][i] = 0.0F;

    F cur[kSets];
#    pragma unroll
    for (int s = 0; s < kSets; ++s) {
        if (live) {
            cur[s].load(a, a.row_begin + l + (s == 1 ? a.up_offset : 0), 0, q);
        } else {
            cur[s].clear();
        }
    }
    uint2 xv[kNTok / 2];
    stage_load<F, kNTok>(a, 0, xv);
    stage_store<F, kNTok>(act[0], xv);
    __syncthreads();

#    pragma unroll 1
    for (int sb = 0; sb < sbs; ++sb) {
        const bool more = sb + 1 < sbs;
        F nxt[kSets];
#    pragma unroll
        for (int s = 0; s < kSets; ++s) {
            nxt[s] = cur[s];
            if (more && live)
                nxt[s].load(a, a.row_begin + l + (s == 1 ? a.up_offset : 0), sb + 1, q);
        }
        if (more) stage_load<F, kNTok>(a, sb + 1, xv);

        const __half* at = act[sb & 1] + q * kQuadStride<kNTok>;
        RunScale sc[kSets][kSc];
#    pragma unroll
        for (int s = 0; s < kSets; ++s) cur[s].scales(q, sc[s]);
#    pragma unroll
        for (int m = 0; m < 16; ++m) {
            uint2 bf[NG];
#    pragma unroll
            for (int g = 0; g < NG; ++g) {
                bf[g] = *reinterpret_cast<const uint2*>(at + (m * kNTok + 8 * g + f) * 4);
            }
#    pragma unroll
            for (int s = 0; s < kSets; ++s) {
                unsigned a0, a1;
                cur[s].fragment(m, q, sc[s], a0, a1);
#    pragma unroll
                for (int g = 0; g < NG; ++g) mma_m8n8k4_f16(acc[s][g], a0, a1, bf[g].x, bf[g].y);
            }
        }
        if (more) stage_store<F, kNTok>(act[(sb + 1) & 1], xv);
        __syncthreads();
#    pragma unroll
        for (int s = 0; s < kSets; ++s) cur[s] = nxt[s];
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
    // Accumulator i: row (lane & 1) + (i & 2) + 4 (lane / 16), column (i & 4) + (lane & 2) +
    // (i & 1). Quad-pair q stores accumulators 2q and 2q + 1.
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
    if (args.tokens <= 8) {
        volta_hmma_kernel<F, E, 1><<<blocks, kWarps * 32, 0, stream>>>(args);
    } else {
        volta_hmma_kernel<F, E, 2><<<blocks, kWarps * 32, 0, stream>>>(args);
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
    if (args.lane_local == 0 || args.join_previous != 0 || args.tokens < kVoltaHmmaMinTokens ||
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
