// Volta's grouped expert stages (T >= 17: prefill, wide verify) on its tensor cores (MXFP4, Q8, the
// K-quants Q4_K / Q5_K / Q6_K and ggml's Q4_1 / Q5_0 / Q5_1; Q4_0, whose fragment-order twin
// has no decoder here, keeps the other routes). Without
// mma.sync of the shapes the W4A8 and register-MMA tiles issue, sm_70 runs those on dp4a and FP32
// FMAs over warp shuffles (ops/common/mma.cuh): on the V100, gpt-oss-20b prefill at 26K tokens
// spent two thirds of its GPU time in the grouped W4A8 MXFP4 stages, 996 tok/s against
// llama.cpp's 2888 (which decodes the experts to f16 and multiplies on cuBLAS / CUTLASS sm70).
// Here each CTA takes 32 weight rows of one job's expert and 32 of its slots, and multiplies them
// on mma.m8n8k4 as generic_volta_hmma.cu does for a dense call: the weight codes become exact f16
// integers in the lane's own A fragments (generic_volta_hmma.cuh, a fragment per format; Q4_K also
// in its fragment-order layout), the
// slots' activation is staged in shared memory as f16 once a superblock, and each scale group's
// scale is applied in the lane's A fragments (f16) before its MMAs. No weight is decoded to memory: a row tile reads its expert's codes once for
// every 32 slots of the job.
#include "core/device.h"
#include "ops/common/mma.cuh"
#include "ops/generic/generic_volta_hmma.cuh"
#include "ops/moe/moe_gemm.cuh"
#include "ops/moe/moe_internal.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace ninferno::ops::detail::moe {
namespace {

using generic::volta::HK45;
using generic::volta::HK4Frag;
using generic::volta::HB16;
using generic::volta::HG32;
using generic::volta::HK6;
using generic::volta::HQ8;
using generic::volta::HX4;
using generic::volta::kCtaRows;
using generic::volta::kQuadStride;
using generic::volta::kWarpRows;
using generic::volta::kWarps;
using generic::volta::stage_store;
using generic::volta::token_slot;

// Slots a CTA multiplies: four token groups of eight (a lane's accumulators: 8 a group and set).
constexpr int kNG   = 4;
constexpr int kNTok = 8 * kNG;
static_assert(kVoltaHmmaSlots == kNTok);

// The activation of superblock `sb` for the CTA's slots: slot t reads row rows[t] of x (gate/up:
// its token; down: the slot itself), zeros past the job's count and past K.
template <int NTok>
__device__ __forceinline__ void stage_load_rows(const GemmArgs& a, const int* rows, int count,
                                                int sb, uint2 (&v)[NTok / 2]) {
    const auto* x = static_cast<const __nv_bfloat16*>(a.x);
    const int k   = a.w.k;
#pragma unroll
    for (int i = 0; i < NTok / 2; ++i) {
        const int task = static_cast<int>(threadIdx.x) + i * kWarps * 32;
        const int t    = task >> 6;
        const int g    = task & 63;
        v[i]           = t < count && 256 * sb + 4 * g < k
                             ? *reinterpret_cast<const uint2*>(
                                   x + static_cast<std::int64_t>(rows[t]) * a.ldx + 256 * sb + 4 * g)
                             : make_uint2(0, 0);
    }
}

template <class F, Stage St>
__global__ __launch_bounds__(kWarps * 32) void moe_volta_hmma_kernel(GemmArgs a) {
#if NINFERNO_HMMA884
    constexpr bool kGated  = St == Stage::GateUp;
    constexpr bool kRouter = St == Stage::Router;
    constexpr bool kFirst  = St == Stage::GateUp || St == Stage::Up;
    constexpr int kSets   = kGated ? 2 : 1;
    constexpr int kGroups = F::kGroups;
    constexpr int kSteps  = 16 / kGroups; // MMAs a scale group
    __shared__ __align__(16) __half act[2][4 * kQuadStride<kNTok>];
    __shared__ float sums[2][F::kMin ? kNTok * 8 : 1];
    __shared__ int rows_x[kNTok];

    // The router: one dense call, 32 tokens a CTA column; the stages: one job's 32 slots.
    int col0 = 0, count = 0, row_base = 0;
    if constexpr (kRouter) {
        col0  = static_cast<int>(blockIdx.y) * kNTok;
        count = min(kNTok, a.tokens - col0);
    } else {
        const int job = static_cast<int>(blockIdx.y) / a.sub_tiles;
        const int sub = static_cast<int>(blockIdx.y) - job * a.sub_tiles;
        if (job >= *a.job_total) return;
        col0     = a.job_begin[job] + sub * kNTok;
        count    = min(kNTok, a.job_begin[job] + a.job_count[job] - col0);
        row_base = a.job_expert[job] * a.expert_rows;
    }
    if (count <= 0) return;
    if (threadIdx.x < kNTok) {
        const int t = static_cast<int>(threadIdx.x);
        rows_x[t]   = t < count ? (kFirst ? a.packed_token[col0 + t] : col0 + t) : 0;
    }

    const generic::GemmArgs& w = a.w;
    const int lane             = static_cast<int>(threadIdx.x) & 31;
    const int warp             = static_cast<int>(threadIdx.x) >> 5;
    const int q                = (lane >> 2) & 3;
    const int f                = (lane & 3) + ((lane >> 4) << 2);
    const int l                = static_cast<int>(blockIdx.x) * kCtaRows + warp * kWarpRows + f;
    const bool live            = l < a.rows;
    const int sbs              = (w.k + 255) >> 8;
    const int src_a            = 4 * q + (lane & 1) + ((lane >> 4) << 4);

    float acc[kSets][kNG][8];
#    pragma unroll
    for (int s = 0; s < kSets; ++s)
#    pragma unroll
        for (int g = 0; g < kNG; ++g)
#    pragma unroll
            for (int i = 0; i < 8; ++i) acc[s][g][i] = 0.0F;

    // One superblock ahead (the CTAs of a prefill call are many: generic_volta_hmma.cu
    // launch_format's two-an-SM case).
    F cur[kSets], nx1[kSets];
    int wrow[kSets];
#    pragma unroll
    for (int s = 0; s < kSets; ++s) {
        wrow[s] = row_base + l * a.row_step + (s == 1 ? a.up_offset : 0);
        cur[s].clear();
        nx1[s].clear();
        if (live) cur[s].load(w, wrow[s], 0, q);
    }
    __syncthreads(); // rows_x
    uint2 xv[kNTok / 2];
    stage_load_rows<kNTok>(a, rows_x, count, 0, xv);
    stage_store<F, kNTok>(act[0], sums[0], xv);
    __syncthreads();

#    pragma unroll 1
    for (int sb = 0; sb < sbs; ++sb) {
        const bool more = sb + 1 < sbs;
#    pragma unroll
        for (int s = 0; s < kSets; ++s) {
            if (more && live) nx1[s].load(w, wrow[s], sb + 1, q);
        }
        if (more) stage_load_rows<kNTok>(a, rows_x, count, sb + 1, xv);

        const __half* at = act[sb & 1] + q * kQuadStride<kNTok>;
#    pragma unroll
        for (int gr = 0; gr < kGroups; ++gr) {
            // The lane's own row's group scale goes into its A fragments (f16, as llama.cpp's
            // decode to f16 rounds it): the MMAs accumulate into acc directly, with no per-group
            // partial sums (gate/up held 64 more registers for them and spilled at 255).
            __half2 dsh[kSets];
            float dmr[kSets][2];
#    pragma unroll
            for (int s = 0; s < kSets; ++s) {
                float ds, dm;
                cur[s].scale(q, gr, ds, dm);
                dsh[s] = __float2half2_rn(ds);
                if constexpr (F::kMin) {
                    dmr[s][0] = __shfl_sync(0xffffffffU, dm, src_a);
                    dmr[s][1] = __shfl_sync(0xffffffffU, dm, src_a + 2);
                }
            }
#    pragma unroll
            for (int st = 0; st < kSteps; ++st) {
                const int m = gr * kSteps + st;
                uint2 bf[kNG];
#    pragma unroll
                for (int g = 0; g < kNG; ++g) {
                    bf[g] = *reinterpret_cast<const uint2*>(
                        at + (m * kNTok + token_slot<kNTok>(m, 8 * g + f)) * 4);
                }
#    pragma unroll
                for (int s = 0; s < kSets; ++s) {
                    unsigned a0, a1;
                    cur[s].fragment(m, q, a0, a1);
                    if constexpr (requires { F::kBias; }) {
                        const __half2 bias = __float2half2_rn(F::kBias);
                        a0 = generic::volta::h2_bits(__hsub2(generic::volta::bits_h2(a0), bias));
                        a1 = generic::volta::h2_bits(__hsub2(generic::volta::bits_h2(a1), bias));
                    }
                    a0 = generic::volta::h2_bits(__hmul2(generic::volta::bits_h2(a0), dsh[s]));
                    a1 = generic::volta::h2_bits(__hmul2(generic::volta::bits_h2(a1), dsh[s]));
#    pragma unroll
                    for (int g = 0; g < kNG; ++g)
                        mma_m8n8k4_f16(acc[s][g], a0, a1, bf[g].x, bf[g].y);
                }
            }
            if constexpr (F::kMin) {
#    pragma unroll
                for (int s = 0; s < kSets; ++s)
#    pragma unroll
                    for (int g = 0; g < kNG; ++g)
#    pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int slot = 8 * g + (i & 4) + (lane & 2) + (i & 1);
                            const float xs = sums[sb & 1][slot * 8 + 2 * q + gr];
                            acc[s][g][i]   = fmaf(-dmr[s][(i & 2) >> 1], xs, acc[s][g][i]);
                        }
            }
        }
        if (more) stage_store<F, kNTok>(act[(sb + 1) & 1], sums[(sb + 1) & 1], xv);
        __syncthreads();
#    pragma unroll
        for (int s = 0; s < kSets; ++s) cur[s] = nx1[s];
    }

    // The quad-pairs' partial sums: ((q0 + q1) + (q2 + q3)) in every lane of the four.
#    pragma unroll
    for (int s = 0; s < kSets; ++s)
#    pragma unroll
        for (int g = 0; g < kNG; ++g)
#    pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[s][g][i] += __shfl_xor_sync(0xffffffffU, acc[s][g][i], 4);
                acc[s][g][i] += __shfl_xor_sync(0xffffffffU, acc[s][g][i], 8);
            }
    // Quad-pair q stores accumulators 2q and 2q + 1, with the expert row's bias and activation.
    const int row0 = static_cast<int>(blockIdx.x) * kCtaRows + warp * kWarpRows;
    auto* out      = static_cast<__nv_bfloat16*>(a.out_bf16);
#    pragma unroll
    for (int i = 0; i < 8; ++i) {
        if ((i >> 1) != q) continue;
        const int lr = row0 + (lane & 1) + (i & 2) + ((lane >> 4) << 2);
        if (lr >= a.rows) continue;
        const std::int64_t r0 = static_cast<std::int64_t>(row_base) +
                                static_cast<std::int64_t>(lr) * a.row_step;
        if constexpr (kRouter) {
            const float bias = moe_bias(a, lr);
#    pragma unroll
            for (int g = 0; g < kNG; ++g) {
                const int slot = 8 * g + (i & 4) + (lane & 2) + (i & 1);
                if (slot < count) {
                    a.out_f32[static_cast<std::int64_t>(col0 + slot) * a.rows + lr] =
                        acc[0][g][i] + bias;
                }
            }
            continue;
        }
        const float bias0 = moe_bias(a, r0);
        const float bias1 = kGated ? moe_bias(a, r0 + a.up_offset) : 0.0F;
#    pragma unroll
        for (int g = 0; g < kNG; ++g) {
            const int slot = 8 * g + (i & 4) + (lane & 2) + (i & 1);
            if (slot >= count) continue;
            const float v0 = acc[0][g][i] + bias0;
            float value    = v0;
            if constexpr (kGated) {
                value = moe_activation(a, v0, acc[kSets - 1][g][i] + bias1);
            } else if constexpr (St == Stage::Up) {
                value = moe_activation(a, v0, v0);
            }
            out[static_cast<std::int64_t>(col0 + slot) * a.ldo + lr] = __float2bfloat16_rn(value);
        }
    }
#else
    __trap();
#endif
}

template <class F>
void launch_format(Stage stage, const GemmArgs& args, dim3 grid, cudaStream_t stream) {
    switch (stage) {
    case Stage::GateUp:
        moe_volta_hmma_kernel<F, Stage::GateUp><<<grid, kWarps * 32, 0, stream>>>(args);
        break;
    case Stage::Up:
        moe_volta_hmma_kernel<F, Stage::Up><<<grid, kWarps * 32, 0, stream>>>(args);
        break;
    case Stage::Down:
        moe_volta_hmma_kernel<F, Stage::Down><<<grid, kWarps * 32, 0, stream>>>(args);
        break;
    case Stage::Router:
        moe_volta_hmma_kernel<F, Stage::Router><<<grid, kWarps * 32, 0, stream>>>(args);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool moe_volta_hmma_codec(generic::WeightCodec codec) {
    using generic::WeightCodec;
    return codec == WeightCodec::GgmlQ4K || codec == WeightCodec::GgmlQ5K ||
           codec == WeightCodec::GgmlQ6K || codec == WeightCodec::Q8 ||
           codec == WeightCodec::Mxfp4 || codec == WeightCodec::GgmlQ41 || codec == WeightCodec::GgmlQ50 ||
           codec == WeightCodec::GgmlQ51;
}

int moe_volta_hmma_k_multiple(generic::WeightCodec codec) {
    using generic::WeightCodec;
    switch (codec) {
    case WeightCodec::GgmlQ4K:
    case WeightCodec::GgmlQ5K:
    case WeightCodec::GgmlQ6K: return 256;
    default: return 64; // the 32-value block formats: whole pairs of blocks
    }
}

void launch_moe_volta_hmma(generic::WeightCodec codec, bool frag, Stage stage,
                           const GemmArgs& args, dim3 grid, cudaStream_t stream) {
    using generic::WeightCodec;
    if (frag && codec != WeightCodec::GgmlQ4K) {
        throw std::invalid_argument("moe volta hmma: no fragment-order decoder for this format");
    }
    if (stage == Stage::Router) {
        if (codec != WeightCodec::Bf16) {
            throw std::invalid_argument("moe volta hmma: the router stage reads BF16 rows");
        }
        launch_format<HB16>(stage, args, grid, stream);
        return;
    }
    switch (codec) {
    case WeightCodec::GgmlQ4K:
        if (frag) {
            launch_format<HK4Frag>(stage, args, grid, stream);
        } else {
            launch_format<HK45<4>>(stage, args, grid, stream);
        }
        return;
    case WeightCodec::GgmlQ5K: launch_format<HK45<5>>(stage, args, grid, stream); return;
    case WeightCodec::GgmlQ6K: launch_format<HK6>(stage, args, grid, stream); return;
    case WeightCodec::Q8: launch_format<HQ8>(stage, args, grid, stream); return;
    case WeightCodec::Mxfp4: launch_format<HX4>(stage, args, grid, stream); return;
    case WeightCodec::GgmlQ41: launch_format<HG32<4, true>>(stage, args, grid, stream); return;
    case WeightCodec::GgmlQ50: launch_format<HG32<5, false>>(stage, args, grid, stream); return;
    case WeightCodec::GgmlQ51: launch_format<HG32<5, true>>(stage, args, grid, stream); return;
    default: throw std::invalid_argument("moe volta hmma: unsupported expert format");
    }
}

} // namespace ninferno::ops::detail::moe
