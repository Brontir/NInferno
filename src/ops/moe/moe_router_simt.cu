// The generic MoE's router at decode widths (moe.cpp router_simt_takes): a BF16 router of E rows
// read by one warp per row, x staged in shared memory, FP32 dot products. The register-MMA router
// takes 16 rows per CTA: 16 CTAs for 256 experts (8 for 128) on 170 SMs, each streaming 64 KB
// alone, 4.1 us per layer in the Qwen3.6-35B decode census. Here E / 4 CTAs of four warps share
// the read. A row's sum runs lane by lane over K in steps of 256 values and closes with one
// butterfly: the order depends on K alone, so a token's logits are the same at every T (decode
// and speculative verify agree), not those of the MMA router; the invariant profile keeps that one.
//
// Route (moe.cpp route_in_router): the CTA that finishes last (a ticket) then selects every token's
// experts and plans the slots, select_plan_warp_kernel's arithmetic on the logits the grid just
// wrote: one launch and one dependent round trip fewer per MoE layer.
#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/moe/moe_internal.h"
#include "ops/moe/moe_route.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail::moe {
namespace {

// Steps of 256 values (32 lanes x 8) whose loads a warp has in flight at once: a whole row of
// up to 3072 values.
constexpr int kSimtSteps = 12;

// The route ticket: zero at load, the last CTA sets it back (one router at a time on a device: a
// MoE layer's calls are in stream order).
__device__ int g_route_ticket;

template <int T, bool Route>
__global__ __launch_bounds__(kRouterSimtWarps *
                             32) void router_simt_kernel(GemmArgs a, SelectArgs s, PlanArgs p) {
    extern __shared__ __align__(16) unsigned char router_smem[];
    const int lane           = static_cast<int>(threadIdx.x) & 31;
    const int warp           = static_cast<int>(threadIdx.x) >> 5;
    const int k              = a.w.k;
    const int row            = static_cast<int>(blockIdx.x) * kRouterSimtWarps + warp;
    const bool live          = row < a.rows;
    const std::uint8_t* wrow = a.w.codes + static_cast<std::int64_t>(live ? row : 0) * k * 2;
    // The weights do not come from the previous kernel: the first steps' loads go out before the
    // programmatic wait, x and every store after it.
    uint4 wv[kSimtSteps];
#pragma unroll
    for (int st = 0; st < kSimtSteps; ++st) {
        const int kk = st * 256 + lane * 8;
        wv[st]       = live && kk < k ? __ldcs(reinterpret_cast<const uint4*>(wrow + kk * 2))
                                      : make_uint4(0, 0, 0, 0);
    }
    pdl::wait_for_dependencies();
    if (threadIdx.x == 0) pdl::trigger_dependents();
    if (a.xq != nullptr) {
        const std::int64_t tasks  = static_cast<std::int64_t>(a.tokens) * (a.xq_ld / 32);
        const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * kRouterSimtWarps * 32;
        for (std::int64_t task =
                 static_cast<std::int64_t>(blockIdx.x) * kRouterSimtWarps * 32 + threadIdx.x;
             task < tasks; task += stride) {
            quantize_x_task(static_cast<const __nv_bfloat16*>(a.xq), a.xq_ldx, a.xq_k, a.xq_ld,
                            task, a.xq_codes, a.xq_scales, a.xq_sums);
        }
    }
    auto* xs      = reinterpret_cast<__nv_bfloat16*>(router_smem);
    const auto* x = static_cast<const __nv_bfloat16*>(a.x);
    for (int i = static_cast<int>(threadIdx.x) * 8; i < T * k; i += kRouterSimtWarps * 32 * 8) {
        const int t = i / k;
        *reinterpret_cast<uint4*>(xs + i) =
            *reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(t) * a.ldx + (i - t * k));
    }
    __syncthreads();
    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.0F;
    for (int base = 0; base < k; base += kSimtSteps * 256) {
        if (base > 0) {
#pragma unroll
            for (int st = 0; st < kSimtSteps; ++st) {
                const int kk = base + st * 256 + lane * 8;
                wv[st] = live && kk < k ? __ldcs(reinterpret_cast<const uint4*>(wrow + kk * 2))
                                        : make_uint4(0, 0, 0, 0);
            }
        }
#pragma unroll
        for (int st = 0; st < kSimtSteps; ++st) {
            const int kk = base + st * 256 + lane * 8;
            if (kk >= k) break;
            const auto* wp = reinterpret_cast<const __nv_bfloat162*>(&wv[st]);
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const uint4 xv = *reinterpret_cast<const uint4*>(xs + t * k + kk);
                const auto* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 wf = __bfloat1622float2(wp[j]);
                    const float2 xf = __bfloat1622float2(xp[j]);
                    acc[t]          = fmaf(wf.x, xf.x, acc[t]);
                    acc[t]          = fmaf(wf.y, xf.y, acc[t]);
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        float v = acc[t];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffU, v, o);
        if (lane == 0 && live) {
            float bias = 0.0F;
            if (a.bias != nullptr) {
                bias = a.bias_bf16
                           ? __bfloat162float(static_cast<const __nv_bfloat16*>(a.bias)[row])
                           : static_cast<const float*>(a.bias)[row];
            }
            a.out_f32[static_cast<std::int64_t>(t) * a.rows + row] = v + bias;
        }
    }
    if constexpr (Route) {
        // Every CTA's logits out before its ticket; the last one reads them all. x's staging is
        // done (the barrier below), so its shared memory holds the ids and the job offsets.
        __threadfence();
        bool last = false;
        if (threadIdx.x == 0) {
            last = atomicAdd(&g_route_ticket, 1) == static_cast<int>(gridDim.x) - 1;
            if (last) g_route_ticket = 0;
        }
        if (!__syncthreads_or(last)) return;
        __threadfence();
        int* ids     = reinterpret_cast<int*>(router_smem);
        int* jobs_at = ids + 32;
        if (warp < s.tokens) select_token(s, warp, lane, ids);
        __syncthreads();
        if (warp == 0) plan_warp(p, lane, ids, jobs_at);
    }
}

template <int T, bool Route>
void launch_route(const GemmArgs& args, const SelectArgs& select, const PlanArgs& plan,
                  cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>((args.rows + kRouterSimtWarps - 1) / kRouterSimtWarps));
    // The route's ids and job offsets reuse x's staging: 64 ints.
    const std::size_t smem =
        std::max<std::size_t>(router_simt_smem_bytes(T, args.w.k), Route ? 64 * sizeof(int) : 0);
    const auto kernel = router_simt_kernel<T, Route>;
    // Urgent: a decode layer's shared expert runs beside it on the branch stream (moe.cpp
    // shared_branch), and its pending CTAs otherwise go first.
    CUDA_CHECK(pdl::launch({grid, dim3(kRouterSimtWarps * 32), smem, stream}, args.pdl != 0, true,
                           kernel, args, select, plan));
}

template <int T>
void launch_t(const GemmArgs& args, const SelectArgs* select, const PlanArgs* plan,
              cudaStream_t stream) {
    if (select != nullptr) {
        launch_route<T, true>(args, *select, *plan, stream);
    } else {
        launch_route<T, false>(args, SelectArgs{}, PlanArgs{}, stream);
    }
}

} // namespace

void launch_router_simt(const GemmArgs& args, cudaStream_t stream, const SelectArgs* select,
                        const PlanArgs* plan) {
    if (select != nullptr &&
        (plan == nullptr || plan->assignments > 32 || select->tokens > kRouterSimtWarps)) {
        throw std::invalid_argument("moe: the router routes at most 32 assignments");
    }
    switch (args.tokens) {
    case 1:
        launch_t<1>(args, select, plan, stream);
        return;
    case 2:
        launch_t<2>(args, select, plan, stream);
        return;
    case 3:
        launch_t<3>(args, select, plan, stream);
        return;
    case 4:
        launch_t<4>(args, select, plan, stream);
        return;
    default:
        throw std::invalid_argument("moe: the SIMT router takes 1..4 tokens");
    }
}

} // namespace ninferno::ops::detail::moe
