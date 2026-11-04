#pragma once

// The GDN input RMSNorm and a/b control projection as one SIMT kernel: one CTA per head and
// token tile computes both control dots and the token's complete sum of squares, applies the RMS
// scale after the dots, and writes the gates; the CTAs of a tile also write disjoint pair-aligned
// slices of h, so no separate norm kernel runs. A token's bytes depend on the geometry and
// `Threads` only, never on `Tile` or on the other tokens of the call.
//
// kD/kH fix the geometry at compile time (the registered 27B form); 0 reads it from d/heads.
// The a/b rows are BF16 (as, bs null), or with kQ8 Q8_G32_FP16 row-split rows: int8 codes at a
// row stride of d and one FP16 scale per 32 values at as/bs (a GGUF model's Q8_0 control), each
// value code * scale in FP32 before the same dots.
#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninferno::ops::detail {

// Eight consecutive values of row `row` from `base` (a multiple of 8) as FP32.
template <bool kQ8>
__device__ __forceinline__ void gdn_control_row8(const void* w, const unsigned short* scales,
                                                 int row, int D, int base, float (&out)[8]) {
    if constexpr (kQ8) {
        const uint2 c = *reinterpret_cast<const uint2*>(static_cast<const signed char*>(w) +
                                                        std::int64_t(row) * D + base);
        const float scale =
            __half2float(__ushort_as_half(scales[std::int64_t(row) * (D / 32) + base / 32]));
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned word = i < 4 ? c.x : c.y;
            out[i] = float(static_cast<signed char>((word >> ((i & 3) * 8)) & 0xffU)) * scale;
        }
    } else {
        const int4 v =
            load_vec<int4>(static_cast<const __nv_bfloat16*>(w) + std::int64_t(row) * D + base);
        const auto* p = reinterpret_cast<const unsigned*>(&v);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = bf16x2_bits_to_float2(p[i]);
            out[2 * i]     = f.x;
            out[2 * i + 1] = f.y;
        }
    }
}

template <int Tile, int Threads, int kD = 0, int kH = 0, bool kQ8 = false>
__global__ __launch_bounds__(Threads) void gdn_norm_gating_simt_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* nw, const void* aw, const void* bw,
    const unsigned short* as, const unsigned short* bs, const float* alog, const float* bias,
    __nv_bfloat16* h, float* g, float* beta, int tokens, float eps, int d, int heads) {
    // A programmatic dependent (the GDN input projection, which reads h) may launch now: it waits
    // for this kernel before it reads, and streams its weights meanwhile.
    if (threadIdx.x == 0) pdl::trigger_dependents();
    constexpr int Warps = Threads / 32;
    const int D = kD != 0 ? kD : d, H = kH != 0 ? kH : heads;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, head = blockIdx.x,
              first = blockIdx.y * Tile;
    float aa[Tile]{}, bb[Tile]{}, ss[Tile]{};
    for (int base = tid * 8; base < D; base += Threads * 8) {
        float af[8], bf[8];
        gdn_control_row8<kQ8>(aw, as, head, D, base, af);
        gdn_control_row8<kQ8>(bw, bs, head, D, base, bf);
        const int4 nv  = load_vec<int4>(nw + base);
        const auto* np = reinterpret_cast<const unsigned*>(&nv);
#pragma unroll
        for (int t = 0; t < Tile; ++t) {
            if (first + t >= tokens) continue;
            const int4 xv  = load_vec<int4>(x + std::int64_t(first + t) * D + base);
            const auto* xp = reinterpret_cast<const unsigned*>(&xv);
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float2 a{af[2 * p], af[2 * p + 1]}, b{bf[2 * p], bf[2 * p + 1]};
                const float2 n = bf16x2_bits_to_float2(np[p]), v = bf16x2_bits_to_float2(xp[p]);
                const float z0 = v.x * (1.0f + n.x), z1 = v.y * (1.0f + n.y);
                aa[t] = fmaf(a.x, z0, aa[t]);
                aa[t] = fmaf(a.y, z1, aa[t]);
                bb[t] = fmaf(b.x, z0, bb[t]);
                bb[t] = fmaf(b.y, z1, bb[t]);
                ss[t] = fmaf(v.x, v.x, ss[t]);
                ss[t] = fmaf(v.y, v.y, ss[t]);
            }
        }
    }
    __shared__ float partial[3][Tile][Warps], inverse[Tile];
#pragma unroll
    for (int t = 0; t < Tile; ++t) {
        aa[t] = warp_reduce_sum(aa[t]);
        bb[t] = warp_reduce_sum(bb[t]);
        ss[t] = warp_reduce_sum(ss[t]);
        if (lane == 0) {
            partial[0][t][warp] = aa[t];
            partial[1][t][warp] = bb[t];
            partial[2][t][warp] = ss[t];
        }
    }
    __syncthreads();
    if (tid < Tile) {
        float a = 0, b = 0, s = 0;
#pragma unroll
        for (int w = 0; w < Warps; ++w) {
            a += partial[0][tid][w];
            b += partial[1][tid][w];
            s += partial[2][tid][w];
        }
        const float inv = rsqrtf(s / D + eps);
        inverse[tid]    = inv;
        if (first + tid < tokens) {
            const auto i = std::int64_t(first + tid) * H + head;
            g[i]         = -expf(alog[head]) * softplus(a * inv + bias[head]);
            beta[i]      = sigmoid(b * inv);
        }
    }
    __syncthreads();
    // Disjoint, pair-aligned h slices across the heads avoid a separate norm kernel.
    const int pairs_per_head = (D / 2 + H - 1) / H;
    for (int k = tid; k < pairs_per_head; k += Threads) {
        const int pair = head * pairs_per_head + k;
        if (pair >= D / 2) break;
        const float2 n = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(nw)[pair]);
#pragma unroll
        for (int t = 0; t < Tile; ++t)
            if (first + t < tokens) {
                const auto i   = std::int64_t(first + t) * (D / 2) + pair;
                const float2 v = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(x)[i]);
                reinterpret_cast<__nv_bfloat162*>(h)[i] = __floats2bfloat162_rn(
                    v.x * inverse[t] * (1 + n.x), v.y * inverse[t] * (1 + n.y));
            }
    }
}

} // namespace ninferno::ops::detail
