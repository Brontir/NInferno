#pragma once

// The grouped-GEMM kernel of the generic MoE Op (see moe_internal.h for the pipeline).
//
// It is the generic projection path's register MMA (ops/generic/generic_gemm.cuh): the weight is
// the MMA's A operand, read straight from global memory and dequantized in registers by the same
// per-format decoders; the activation is the B operand, eight columns per n-tile. What differs is
// the addressing and the epilogue:
//   Router  one dense call: rows are the E experts, columns the T tokens; FP32 logits (+ bias).
//   GateUp  blockIdx.y is (job, sub-tile): the job's expert selects its 2F rows of the stacked
//           plane, and each column is a slot whose token is gathered through packed_token; the
//           epilogue applies the activation to the gate and up rows (+ biases) and stores BF16.
//   Up      GateUp for a non-gated expert (Relu2): one up row per output, one set of sums.
//   Down    the same jobs; the slot's intermediate row is the activation; + bias, BF16 out.
// As in the generic path, every WarpsK == 1 schedule sums each output element over K in one fixed
// order whatever its tile, job or call width is, so the prefill profile's routes are token-
// invariant.
//
// A router has few rows (E experts: a few 16-row tiles), so its call runs on a few SMs and each
// warp walks its K steps alone; with the two-slot ring every other step waits for its load. The
// Router stage instead loads kRouterBatch of a warp's steps at once and then computes them: the
// same steps in the same order into the same accumulators (the bytes do not change), with one
// load latency per batch instead of one per two steps.

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/generic/generic_gemm.cuh"
#include "ops/moe/moe_internal.h"
#include "ops/moe/moe_route.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninferno::ops::detail::moe {

__device__ __forceinline__ float moe_bias(const GemmArgs& a, std::int64_t index) {
    if (a.bias == nullptr) return 0.0F;
    if (a.bias_bf16) return __bfloat162float(static_cast<const __nv_bfloat16*>(a.bias)[index]);
    return static_cast<const float*>(a.bias)[index];
}

__device__ __forceinline__ float moe_sigmoid(float v) { return 1.0F / (1.0F + expf(-v)); }

__device__ __forceinline__ float moe_activation(const GemmArgs& a, float g, float u) {
    switch (a.activation) {
    case 1: { // GeGluTanh
        const float inner = 0.7978845608028654F * (g + 0.044715F * g * g * g);
        return 0.5F * g * (1.0F + tanhf(inner)) * u;
    }
    case 2: { // ClampedSwiGlu (gpt-oss)
        const float gc = fminf(g, a.limit);
        const float uc = fminf(fmaxf(u, -a.limit), a.limit);
        return (uc + 1.0F) * gc * moe_sigmoid(a.alpha * gc);
    }
    case 3: { // Relu2 (non-gated, Stage::Up: g is the up sum too)
        const float r = fmaxf(u, 0.0F);
        return r * r;
    }
    default: // SwiGlu
        return g * moe_sigmoid(g) * u;
    }
}

inline constexpr int kRouterBatch = 16;

// As the generic path's rmma_kernel, a one-tile expert stage (the decode shapes: latency-bound)
// keeps two CTAs per SM: without the bound a 256-thread codebook CTA took 208 registers (sm_120a)
// and ran one CTA per SM. The router stage (kRouterBatch loads in flight per warp) spills under
// that bound (BF16, 1 x 8 warps: 144 bytes of stack on sm_89) and keeps one.
template <class Decoder, class Schedule, Stage S>
__global__ __launch_bounds__(Schedule::kThreads,
                             S == Stage::Router ? 1 : Schedule::kMinBlocks) void moe_gemm_kernel(
    GemmArgs a) {
    constexpr int R      = Decoder::kRun;
    constexpr int KStep  = 4 * R;
    constexpr int J      = R / 4;
    constexpr int MT     = Schedule::kMTiles;
    constexpr int NT     = Schedule::kNTiles;
    constexpr int WK     = Schedule::kWarpsK;
    constexpr int kSets  = S == Stage::GateUp ? 2 : 1;
    // The expert's first projection: slots gather their tokens, the epilogue applies the activation.
    constexpr bool kFirst = S == Stage::GateUp || S == Stage::Up;
    constexpr int kWidth = NT * 8;
    using Run            = typename Decoder::Run;

    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int warp_m = warp / WK;
    const int warp_k = warp - warp_m * WK;
    const int quad   = lane & 3;
    const int group  = lane >> 2;
    const int local0 = static_cast<int>(blockIdx.x) * Schedule::kRowsPerCta + warp_m * MT * 16;
    // Under GemmArgs::pdl the producer (the previous stage, the router's normed input) may still
    // run: every global load comes after the wait. The trigger lets the next stage's CTAs take
    // the SMs this grid leaves, to wait there.
    pdl::wait_for_dependencies();
    if (threadIdx.x == 0) pdl::trigger_dependents();
    if constexpr (S == Stage::Router) {
        if (a.xq != nullptr) {
            const std::int64_t tasks = static_cast<std::int64_t>(a.tokens) * (a.xq_ld / 32);
            const std::int64_t cta = static_cast<std::int64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
            const std::int64_t stride =
                static_cast<std::int64_t>(gridDim.x) * gridDim.y * Schedule::kThreads;
            for (std::int64_t task = cta * Schedule::kThreads + threadIdx.x; task < tasks;
                 task += stride) {
                quantize_x_task(static_cast<const __nv_bfloat16*>(a.xq), a.xq_ldx, a.xq_k, a.xq_ld,
                                task, a.xq_codes, a.xq_scales, a.xq_sums);
            }
        }
    }

    // Columns of this CTA: [col0, col0 + count) of the call (Router) or of the slot order.
    int expert = 0;
    int col0   = 0;
    int count  = 0;
    if constexpr (S == Stage::Router) {
        col0  = static_cast<int>(blockIdx.y) * kWidth;
        count = min(kWidth, a.tokens - col0);
    } else {
        const int job = static_cast<int>(blockIdx.y) / a.sub_tiles;
        const int sub = static_cast<int>(blockIdx.y) - job * a.sub_tiles;
        if (job >= *a.job_total) return;
        expert = a.job_expert[job];
        col0   = a.job_begin[job] + sub * kWidth;
        count  = min(kWidth, a.job_begin[job] + a.job_count[job] - col0);
    }
    if (count <= 0) return;
    // Every return above is the whole CTA's, so the barrier after staging is reached by all.
    if constexpr (generic::TableOf<Decoder>::value) {
        Decoder::stage();
        __syncthreads();
    }
    const std::int64_t row_base = static_cast<std::int64_t>(expert) * a.expert_rows;

    // This lane's activation column per n-tile.
    const __nv_bfloat16* xcol[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        const int c = n * 8 + group;
        int src     = 0;
        if (c < count) {
            src = col0 + c;
            if constexpr (kFirst) src = a.packed_token[src];
        }
        xcol[n] = c < count ? static_cast<const __nv_bfloat16*>(a.x) +
                                  static_cast<std::int64_t>(src) * a.ldx
                            : nullptr;
    }

    const int steps = (a.w.k + KStep - 1) / KStep;
    float acc[kSets][MT][NT][4];
#pragma unroll
    for (int s = 0; s < kSets; ++s)
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int n = 0; n < NT; ++n)
                acc[s][m][n][0] = acc[s][m][n][1] = acc[s][m][n][2] = acc[s][m][n][3] = 0.0F;

    const auto weight_row = [&](int l, int s) -> int {
        return static_cast<int>(row_base + static_cast<std::int64_t>(l) * a.row_step +
                                (s == 1 ? a.up_offset : 0));
    };

    using Runs       = Run[kSets][MT][2];
    const auto load_step = [&](Runs& dst, int step) {
        const int k0    = step * KStep + quad * R;
        const bool k_ok = (generic::GroupTailOf<Decoder>::value ? step * KStep : k0) < a.w.k;
#pragma unroll
        for (int s = 0; s < kSets; ++s)
#pragma unroll
            for (int m = 0; m < MT; ++m)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int l  = local0 + m * 16 + h * 8 + group;
                    dst[s][m][h] = l >= a.rows ? Decoder::zero()
                                   : k_ok      ? Decoder::load(a.w, weight_row(l, s), k0)
                                               : Decoder::masked(a.w, weight_row(l, s),
                                                                 step * KStep);
                }
    };

    const auto compute_step = [&](const Runs& src, int step) {
        const int k0 = (generic::GroupTailOf<Decoder>::value ? 0 : quad * R) + step * KStep;
        const __nv_bfloat16* col[NT];
#pragma unroll
        for (int n = 0; n < NT; ++n) col[n] = k0 < a.w.k ? xcol[n] : nullptr;
        generic::mma_step<Decoder, NT, J, kSets, MT>(acc, src, col, step * KStep, quad, a.w.k);
    };

    if constexpr (S == Stage::Router) {
        // A warp's steps warp_k, warp_k + WK, ... in order, kRouterBatch loads in flight.
        Run batch[kRouterBatch][kSets][MT][2];
#pragma unroll 1
        for (int base = warp_k; base < steps; base += kRouterBatch * WK) {
#pragma unroll
            for (int d = 0; d < kRouterBatch; ++d) {
                if (base + d * WK < steps) load_step(batch[d], base + d * WK);
            }
#pragma unroll
            for (int d = 0; d < kRouterBatch; ++d) {
                if (base + d * WK < steps) compute_step(batch[d], base + d * WK);
            }
        }
    } else {
        Run ring[2][kSets][MT][2];
        if (warp_k < steps) load_step(ring[0], warp_k);
#pragma unroll 1
        for (int step = warp_k; step < steps; step += 2 * WK) {
            if (step + WK < steps) load_step(ring[1], step + WK);
            compute_step(ring[0], step);
            if (step + WK >= steps) break;
            if (step + 2 * WK < steps) load_step(ring[0], step + 2 * WK);
            compute_step(ring[1], step + WK);
        }
    }

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
        if (warp_k != 0) return;
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

    // C fragment: rows group and group + 8 of each m16 tile, columns 2*quad and 2*quad + 1.
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int l = local0 + m * 16 + group + (e >= 2 ? 8 : 0);
            if (l >= a.rows || warp_k != 0) continue;
            const int row0     = weight_row(l, 0);
            const float scale0 = Decoder::row_scale(a.w, row0);
            float scale1       = 0.0F;
            int row1           = 0;
            if constexpr (S == Stage::GateUp) {
                row1   = weight_row(l, 1);
                scale1 = Decoder::row_scale(a.w, row1);
            }
            const float bias0 = moe_bias(a, S == Stage::Router ? l : row0);
            const float bias1 = S == Stage::GateUp ? moe_bias(a, row1) : 0.0F;
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const int c = n * 8 + 2 * quad + (e & 1);
                if (c >= count) continue;
                const float v0 = acc[0][m][n][e] * scale0 + bias0;
                if constexpr (S == Stage::Router) {
                    a.out_f32[static_cast<std::int64_t>(col0 + c) * a.rows + l] = v0;
                } else {
                    float value = v0;
                    if constexpr (S == Stage::GateUp) {
                        value = moe_activation(a, v0, acc[kSets - 1][m][n][e] * scale1 + bias1);
                    } else if constexpr (S == Stage::Up) {
                        value = moe_activation(a, v0, v0);
                    }
                    static_cast<__nv_bfloat16*>(a.out_bf16)[static_cast<std::int64_t>(col0 + c) *
                                                                 a.ldo +
                                                             l] = __float2bfloat16_rn(value);
                }
            }
        }
    }
}

template <class Decoder, class Schedule, Stage S>
void launch_moe_schedule(const GemmArgs& args, dim3 grid, cudaStream_t stream) {
    CUDA_CHECK(pdl::launch({grid, dim3(Schedule::kThreads), 0, stream}, args.pdl != 0, false,
                           moe_gemm_kernel<Decoder, Schedule, S>, args));
}

template <class Decoder, Stage S>
void launch_moe_stage(int schedule, const GemmArgs& args, dim3 grid, cudaStream_t stream) {
    switch (schedule) {
#define NINFERNO_MOE_CASE(I, WM, WK, MT, NT)                                                       \
    case I:                                                                                        \
        if constexpr (generic::rmma_compiled<Decoder, MT, NT>) {                                   \
            launch_moe_schedule<Decoder, generic::RmmaSchedule<WM, WK, MT, NT>, S>(args, grid,     \
                                                                                   stream);        \
            return;                                                                                \
        }                                                                                          \
        break;
        NINFERNO_MOE_SCHEDULES(NINFERNO_MOE_CASE)
#undef NINFERNO_MOE_CASE
    default:
        throw std::invalid_argument("moe: unknown schedule");
    }
    throw std::invalid_argument("moe: schedule not compiled for this format");
}

template <class Decoder>
void launch_moe(Stage stage, int schedule, const GemmArgs& args, dim3 grid, cudaStream_t stream) {
    switch (stage) {
    case Stage::Router:
        launch_moe_stage<Decoder, Stage::Router>(schedule, args, grid, stream);
        return;
    case Stage::GateUp:
        launch_moe_stage<Decoder, Stage::GateUp>(schedule, args, grid, stream);
        return;
    case Stage::Down:
        launch_moe_stage<Decoder, Stage::Down>(schedule, args, grid, stream);
        return;
    case Stage::Up:
        launch_moe_stage<Decoder, Stage::Up>(schedule, args, grid, stream);
        return;
    }
}

} // namespace ninferno::ops::detail::moe
