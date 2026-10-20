// Routing kernels of the generic MoE Op: top-k selection, the expert-major slot plan with its job
// list, and the weighted combine. Every one is deterministic: the selection breaks exact ties
// toward the lower expert id, the plan orders an expert's slots by assignment (token-major, then
// rank), and the combine adds a token's experts in rank order.
#include "ops/moe/moe_route.cuh"

#include "core/per_device.h"
#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/arch_limits.h"
#include "ops/moe/moe_internal.h"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninferno::ops::detail::moe {
namespace {

constexpr int kSelectWarps    = 8;
// A combine CTA covers 64 runs of eight features of one token: a token of 2048..2816 features
// spreads over four to six CTAs instead of one.
constexpr int kCombineThreads = 64;
// Experts whose runs a thread has in flight before it sums them.
constexpr int kCombineBatch = 8;

__global__ __launch_bounds__(kSelectWarps * 32) void select_kernel(SelectArgs a) {
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int token = static_cast<int>(blockIdx.x) * kSelectWarps + (static_cast<int>(threadIdx.x) >> 5);
    if (token >= a.tokens) return;
    select_token(a, token, lane);
}

__global__ __launch_bounds__(kPlanWarps * 32) void plan_kernel(PlanArgs a) {
    extern __shared__ int plan_smem[];
    plan_body<kPlanWarps>(a, plan_smem);
}

// A call of at most kPlanWarps tokens selects and plans in one CTA: warp t selects token t (as
// select_kernel's warp does), and after the barrier the plan reads the ids the CTA wrote. The
// same arithmetic as the two launches, one launch fewer.
__global__ __launch_bounds__(kPlanWarps * 32) void select_plan_kernel(SelectArgs s, PlanArgs p) {
    extern __shared__ int plan_smem[];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (warp < s.tokens) select_token(s, warp, lane);
    __syncthreads();
    plan_body<kPlanWarps>(p, plan_smem);
}

// The same with plan_warp: one warp per token (at least one), no per-expert counters. The ids
// reach the plan through shared memory, not by a global store and reload.
__global__ __launch_bounds__(kPlanWarps * 32) void select_plan_warp_kernel(SelectArgs s,
                                                                           PlanArgs p) {
    __shared__ int ids[32];
    __shared__ int jobs_at[32];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (warp < s.tokens) select_token(s, warp, lane, ids);
    __syncthreads();
    if (warp == 0) plan_warp(p, lane, ids, jobs_at);
}

// Grid (token, run of 64 * 8 features); each thread owns one run of eight features. It issues
// the loads of up to kCombineBatch experts' runs before it sums them, in k order: the sums of the
// loop that loaded one run at a time, the same bytes (one CTA per token with a load per k waited
// on L2 eight times; Qwen3.6-35B 1.86 -> 1.40 us, Nemotron 2.48 -> 1.36 us, Gemma 4 26B 2.31 ->
// 1.22 us in a graph, RTX 5090).
__global__ __launch_bounds__(kCombineThreads) void combine_kernel(CombineArgs a) {
    // CombineArgs::pdl: y comes from the down stage.
    pdl::wait_for_dependencies();
    if (threadIdx.x == 0) pdl::trigger_dependents();
    const int t = static_cast<int>(blockIdx.x);
    const int h =
        (static_cast<int>(blockIdx.y) * kCombineThreads + static_cast<int>(threadIdx.x)) * 8;
    if (h >= a.hidden) return;
    const auto* y    = static_cast<const __nv_bfloat16*>(a.y);
    auto* out        = static_cast<__nv_bfloat16*>(a.out) + static_cast<std::int64_t>(t) * a.hidden;
    const int* slots = a.slot_of + static_cast<std::int64_t>(t) * a.top_k;
    const float* weights = a.weights + static_cast<std::int64_t>(t) * a.top_k;
    uint4 shared{};
    uint4 residual{};
    if (a.shared != nullptr) {
        shared = *reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(a.shared) +
                                                 static_cast<std::int64_t>(t) * a.hidden + h);
    }
    if (a.add_residual) residual = *reinterpret_cast<const uint4*>(out + h);
    float gate = 1.0F;
    if (a.shared_gate != nullptr) {
        const float l = __bfloat162float(static_cast<const __nv_bfloat16*>(a.shared_gate)[t]);
        gate          = 1.0F / (1.0F + expf(-l));
    }
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k0 = 0; k0 < a.top_k; k0 += kCombineBatch) {
        uint4 raw[kCombineBatch];
        float w[kCombineBatch];
#pragma unroll
        for (int i = 0; i < kCombineBatch; ++i) {
            if (k0 + i < a.top_k) {
                w[i]   = weights[k0 + i];
                raw[i] = *reinterpret_cast<const uint4*>(
                    y + static_cast<std::int64_t>(slots[k0 + i]) * a.hidden + h);
            }
        }
#pragma unroll
        for (int i = 0; i < kCombineBatch; ++i) {
            if (k0 + i < a.top_k) {
                const auto* v = reinterpret_cast<const __nv_bfloat162*>(&raw[i]);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __bfloat1622float2(v[j]);
                    acc[2 * j]     = fmaf(w[i], f.x, acc[2 * j]);
                    acc[2 * j + 1] = fmaf(w[i], f.y, acc[2 * j + 1]);
                }
            }
        }
    }
    if (a.shared != nullptr) {
        const auto* v = reinterpret_cast<const __nv_bfloat162*>(&shared);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(v[j]);
            acc[2 * j]     = fmaf(gate, f.x, acc[2 * j]);
            acc[2 * j + 1] = fmaf(gate, f.y, acc[2 * j + 1]);
        }
    }
    if (a.add_residual) {
        const auto* r = reinterpret_cast<const __nv_bfloat162*>(&residual);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(r[j]);
            acc[2 * j] += f.x;
            acc[2 * j + 1] += f.y;
        }
    }
    uint4 packed;
    auto* o = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
    *reinterpret_cast<uint4*>(out + h) = packed;
}

} // namespace

void launch_select(const SelectArgs& args, cudaStream_t stream) {
    const unsigned blocks = static_cast<unsigned>((args.tokens + kSelectWarps - 1) / kSelectWarps);
    select_kernel<<<blocks, kSelectWarps * 32, 0, stream>>>(args);
    CUDA_CHECK(cudaGetLastError());
}

namespace {

// Plan counters: (kPlanWarps + 2) ints per expert. NINFERNO-SPECIAL(arch:sm75): kMaxExperts of them
// take 72 KiB, over Turing's 64 KiB; the kernels ask for what the build holds, and a call for what
// its own experts need (910 on Turing, more than any MoE served).
constexpr std::size_t kPlanSharedCeiling =
    std::min(sizeof(int) * (kPlanWarps + 2) * kMaxExperts, kMaxSharedBytesPerBlock);

std::size_t plan_shared_bytes(int experts) {
    const std::size_t bytes = sizeof(int) * static_cast<std::size_t>(kPlanWarps + 2) * experts;
    if (bytes > kPlanSharedCeiling) {
        throw std::invalid_argument("moe: " + std::to_string(experts) +
                                    " experts exceed this build's plan shared memory");
    }
    return bytes;
}

} // namespace

void launch_plan(const PlanArgs& args, cudaStream_t stream) {
    const std::size_t smem = plan_shared_bytes(args.experts);
    static ::ninferno::PerDevice<bool> attribute;
    ::ninferno::allow_dynamic_smem(attribute, plan_kernel, static_cast<int>(kPlanSharedCeiling));
    plan_kernel<<<1, kPlanWarps * 32, smem, stream>>>(args);
    CUDA_CHECK(cudaGetLastError());
}

void launch_select_plan(const SelectArgs& select, const PlanArgs& plan, cudaStream_t stream) {
    if (select.tokens > kPlanWarps) {
        throw std::invalid_argument("moe: a fused selection and plan covers at most 16 tokens");
    }
    // Urgent, as the router: one CTA on the path that sets the layer's length, which beside the
    // shared expert's grids on the branch stream waited for an SM (RTX 5090, Nemotron 3.5 C=1
    // decode: 11 us beside them).
    if (plan.assignments <= 32) {
        const unsigned threads = static_cast<unsigned>(std::max(select.tokens, 1) * 32);
        CUDA_CHECK(pdl::launch({dim3(1), dim3(threads), 0, stream}, false, true,
                               select_plan_warp_kernel, select, plan));
        return;
    }
    const std::size_t smem = plan_shared_bytes(plan.experts);
    static ::ninferno::PerDevice<bool> attribute;
    ::ninferno::allow_dynamic_smem(attribute, select_plan_kernel, static_cast<int>(kPlanSharedCeiling));
    CUDA_CHECK(pdl::launch({dim3(1), dim3(kPlanWarps * 32), smem, stream}, false, true,
                           select_plan_kernel, select, plan));
}

namespace {
__device__ int g_combine_tickets[kCombineTickets]; // zero-initialized with the module
} // namespace

// The array's address on the current device (every device has its own copy of the module).
int* combine_tickets() {
    constexpr int kDevices = 64;
    static std::array<std::atomic<int*>, kDevices> cache{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kDevices) {
        throw std::runtime_error("moe: device ordinal past the combine ticket cache");
    }
    int* tickets = cache[device].load(std::memory_order_acquire);
    if (tickets == nullptr) {
        void* p = nullptr;
        CUDA_CHECK(cudaGetSymbolAddress(&p, g_combine_tickets));
        tickets = static_cast<int*>(p);
        cache[device].store(tickets, std::memory_order_release);
    }
    return tickets;
}

void launch_combine(const CombineArgs& args, cudaStream_t stream) {
    const dim3 grid(
        static_cast<unsigned>(args.tokens),
        static_cast<unsigned>((args.hidden / 8 + kCombineThreads - 1) / kCombineThreads));
    if (args.pdl != 0) {
        CUDA_CHECK(
            pdl::launch_dependent({grid, dim3(kCombineThreads), 0, stream}, combine_kernel, args));
        return;
    }
    combine_kernel<<<grid, kCombineThreads, 0, stream>>>(args);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninferno::ops::detail::moe
