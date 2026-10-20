#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/sparse_moe/decode/sparse_moe_decode.h"
#include "ops/sparse_moe/sparse_moe_route.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

constexpr int kHidden           = 2048;
constexpr int kRouterRows       = 257;
constexpr int kTopK             = 8;
constexpr int kRouterPartitions = 4;
constexpr int kRouterWarps      = 4;
// The widest window whose selection S1's last block can run instead of S2 (one warp a token).
constexpr int kS1SelectMax = 8;

// A token's router score of one row: its four partitions, summed in partition order. Read
// through L2, since in S1's last block other blocks of the same grid wrote them.
__device__ __forceinline__ float token_router_score(const float* partial_scores, int token,
                                                    int row) {
    float sum = 0.0f;
#pragma unroll
    for (int partition = 0; partition < kRouterPartitions; ++partition) {
        sum += __ldcg(partial_scores +
                      (static_cast<std::int64_t>(token) * kRouterRows + row) * kRouterPartitions +
                      partition);
    }
    return sum;
}

// Ticket for the last-arriving S1 block, as D1's: atomicInc wraps at gridDim.x-1, so the counter
// returns to zero on its own.
__device__ unsigned int g_sparse_moe_small_t_route_ticket = 0;

template <int Tokens, bool Select>
__global__ void
sparse_moe_small_t_s1_kernel(const __nv_bfloat16* __restrict__ x,
                             const __nv_bfloat16* __restrict__ router,
                             float* __restrict__ partial_scores, int* __restrict__ token_ids,
                             float* __restrict__ token_alpha, float* __restrict__ shared_scale) {
    static_assert(Tokens >= 1 && Tokens <= kSparseMoeSmallTMax);
    static_assert(!Select || Tokens <= kS1SelectMax);
    __shared__ float partial[kRouterWarps][Tokens];
    if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    const int row       = static_cast<int>(blockIdx.x) / kRouterPartitions;
    const int partition = static_cast<int>(blockIdx.x) - row * kRouterPartitions;
    const int warp      = static_cast<int>(threadIdx.x) >> 5;
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int k = partition * (kHidden / kRouterPartitions) + static_cast<int>(threadIdx.x) * 4;

    const uint2 weight = load_vec<uint2>(router + static_cast<std::int64_t>(row) * kHidden + k);
    const float2 w0    = bf16x2_bits_to_float2(weight.x);
    const float2 w1    = bf16x2_bits_to_float2(weight.y);

    float accumulator[Tokens];
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const uint2 input  = load_vec<uint2>(x + static_cast<std::int64_t>(token) * kHidden + k);
        const float2 x0    = bf16x2_bits_to_float2(input.x);
        const float2 x1    = bf16x2_bits_to_float2(input.y);
        float sum          = 0.0f;
        sum                = fmaf(w0.x, x0.x, sum);
        sum                = fmaf(w0.y, x0.y, sum);
        sum                = fmaf(w1.x, x1.x, sum);
        accumulator[token] = fmaf(w1.y, x1.y, sum);
    }

#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const float sum = warp_reduce_sum(accumulator[token]);
        if (lane == 0) { partial[warp][token] = sum; }
    }
    __syncthreads();

    if (warp == 0) {
        for (int token = lane; token < Tokens; token += 32) {
            float sum = 0.0f;
#pragma unroll
            for (int source_warp = 0; source_warp < kRouterWarps; ++source_warp) {
                sum += partial[source_warp][token];
            }
            partial_scores[(static_cast<std::int64_t>(token) * kRouterRows + row) *
                               kRouterPartitions +
                           partition] = sum;
        }
    }

    if constexpr (Select) {
        // S2's selection used to be a launch of its own, a graph node that costs more than its
        // work where no programmatic launch hides it. The block that arrives last runs it instead,
        // one warp a token: the same scores summed in the same order and the same routine, so the
        // routing is bit-identical.
        __shared__ float scores[kRouterWarps][kRouterRows];
        __shared__ float selected_logits[kRouterWarps][kTopK];
        __shared__ bool is_last_block;
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            const unsigned int ticket =
                atomicInc(&g_sparse_moe_small_t_route_ticket, gridDim.x - 1u);
            is_last_block = ticket == gridDim.x - 1u;
        }
        __syncthreads();
        if (!is_last_block) { return; }
        for (int token = warp; token < Tokens; token += kRouterWarps) {
            for (int r = lane; r < kRouterRows; r += 32) {
                scores[warp][r] = token_router_score(partial_scores, token, r);
            }
            __syncwarp();
            sparse_moe_select_top8_warp(scores[warp], token_ids + token * kTopK,
                                        token_alpha + token * kTopK, shared_scale + token,
                                        selected_logits[warp]);
            __syncwarp();
        }
    }
}

// S2 runs one CTA per token: the router scores of a single token take 1 060 B of shared
// memory, so the block no longer has to be wide enough to hold the whole decode window.
constexpr int kS2Threads = 128;

__global__ void sparse_moe_small_t_s2_kernel(const float* __restrict__ partial_scores,
                                             int* __restrict__ token_ids,
                                             float* __restrict__ token_alpha,
                                             float* __restrict__ shared_scale) {
    __shared__ float scores[kRouterRows];
    __shared__ float selected_logits[kTopK];
    const int tid   = static_cast<int>(threadIdx.x);
    const int token = static_cast<int>(blockIdx.x);
    if (tid == 0) { pdl::trigger_dependents(); }
    pdl::wait_for_dependencies();
    for (int row = tid; row < kRouterRows; row += kS2Threads) {
        scores[row] = token_router_score(partial_scores, token, row);
    }
    __syncthreads();

    if (tid < 32) {
        sparse_moe_select_top8_warp(scores, token_ids + token * kTopK, token_alpha + token * kTopK,
                                    shared_scale + token, selected_logits);
    }
}

template <int Tokens, bool Select>
void launch_s1(const __nv_bfloat16* x, const __nv_bfloat16* router, float* partial_scores,
               const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream) {
    sparse_moe_small_t_s1_kernel<Tokens, Select>
        <<<kRouterRows * kRouterPartitions, kRouterWarps * 32, 0, stream>>>(
            x, router, partial_scores, static_cast<int*>(workspace.token_ids.data),
            static_cast<float*>(workspace.token_alpha.data),
            static_cast<float*>(workspace.shared_scale.data));
    CUDA_CHECK(cudaGetLastError());
}

void launch_s2(const float* partial_scores, int* token_ids, float* token_alpha, float* shared_scale,
               std::int32_t tokens, cudaStream_t stream) {
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(static_cast<unsigned int>(tokens)), dim3(kS2Threads), 0, stream},
        sparse_moe_small_t_s2_kernel, partial_scores, token_ids, token_alpha, shared_scale));
}

void launch_s3_tiled(const Tensor& x, const SparseMoeWeights& weights,
                     const SparseMoeSmallTPlan& plan, const SparseMoeSmallTWorkspace& workspace,
                     cudaStream_t stream) {
    sparse_moe_decode_launch_d3_small_t(
        x, weights, static_cast<const int*>(workspace.token_ids.data),
        static_cast<float*>(workspace.scratch.data), plan.tokens, plan.d3_schedule, stream);
}

void launch_s4_tiled(const SparseMoeWeights& weights, Tensor& destination,
                     const SparseMoeSmallTPlan& plan, const SparseMoeSmallTWorkspace& workspace,
                     cudaStream_t stream) {
    sparse_moe_decode_launch_d4_small_t(
        weights, destination, static_cast<const int*>(workspace.token_ids.data),
        static_cast<const float*>(workspace.token_alpha.data),
        static_cast<const float*>(workspace.shared_scale.data),
        static_cast<const float*>(workspace.scratch.data), plan.tokens, plan.d4_schedule, stream);
}

template <class Launch>
void dispatch_tokens(std::int32_t tokens, Launch&& launch) {
    switch (tokens) {
    case 1:
        launch.template operator()<1>();
        return;
    case 2:
        launch.template operator()<2>();
        return;
    case 3:
        launch.template operator()<3>();
        return;
    case 4:
        launch.template operator()<4>();
        return;
    case 5:
        launch.template operator()<5>();
        return;
    case 6:
        launch.template operator()<6>();
        return;
    case 7:
        launch.template operator()<7>();
        return;
    case 8:
        launch.template operator()<8>();
        return;
    case 9:
        launch.template operator()<9>();
        return;
    case 10:
        launch.template operator()<10>();
        return;
    case 11:
        launch.template operator()<11>();
        return;
    case 12:
        launch.template operator()<12>();
        return;
    case 13:
        launch.template operator()<13>();
        return;
    case 14:
        launch.template operator()<14>();
        return;
    case 15:
        launch.template operator()<15>();
        return;
    case 16:
        launch.template operator()<16>();
        return;
    case 17:
        launch.template operator()<17>();
        return;
    case 18:
        launch.template operator()<18>();
        return;
    case 19:
        launch.template operator()<19>();
        return;
    case 20:
        launch.template operator()<20>();
        return;
    case 21:
        launch.template operator()<21>();
        return;
    case 22:
        launch.template operator()<22>();
        return;
    case 23:
        launch.template operator()<23>();
        return;
    case 24:
        launch.template operator()<24>();
        return;
    case 25:
        launch.template operator()<25>();
        return;
    case 26:
        launch.template operator()<26>();
        return;
    case 27:
        launch.template operator()<27>();
        return;
    case 28:
        launch.template operator()<28>();
        return;
    case 29:
        launch.template operator()<29>();
        return;
    case 30:
        launch.template operator()<30>();
        return;
    case 31:
        launch.template operator()<31>();
        return;
    case 32:
        launch.template operator()<32>();
        return;
    case 33:
        launch.template operator()<33>();
        return;
    case 34:
        launch.template operator()<34>();
        return;
    case 35:
        launch.template operator()<35>();
        return;
    case 36:
        launch.template operator()<36>();
        return;
    case 37:
        launch.template operator()<37>();
        return;
    case 38:
        launch.template operator()<38>();
        return;
    case 39:
        launch.template operator()<39>();
        return;
    case 40:
        launch.template operator()<40>();
        return;
    case 41:
        launch.template operator()<41>();
        return;
    case 42:
        launch.template operator()<42>();
        return;
    case 43:
        launch.template operator()<43>();
        return;
    case 44:
        launch.template operator()<44>();
        return;
    case 45:
        launch.template operator()<45>();
        return;
    case 46:
        launch.template operator()<46>();
        return;
    default:
        throw std::invalid_argument("sparse_moe small-T: unsupported token count");
    }
}

} // namespace

void sparse_moe_small_t_launch(const Tensor& x, const SparseMoeWeights& weights,
                               Tensor& destination, const SparseMoeSmallTPlan& plan,
                               const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream) {
    // S1's last block selects for a window of at most kS1SelectMax tokens: one launch fewer on
    // every card, with or without programmatic launch (greedy A/B, the same text in every run).
    const bool select = plan.tokens <= kS1SelectMax;
    dispatch_tokens(plan.tokens, [&]<int Tokens>() {
        const auto* input  = static_cast<const __nv_bfloat16*>(x.data);
        const auto* router = static_cast<const __nv_bfloat16*>(weights.router_shared_gate.qdata);
        auto* partial      = static_cast<float*>(workspace.scratch.data);
        if constexpr (Tokens <= kS1SelectMax) {
            if (select) {
                launch_s1<Tokens, true>(input, router, partial, workspace, stream);
                return;
            }
        }
        launch_s1<Tokens, false>(input, router, partial, workspace, stream);
    });
    if (!select) {
        launch_s2(static_cast<const float*>(workspace.scratch.data),
                  static_cast<int*>(workspace.token_ids.data),
                  static_cast<float*>(workspace.token_alpha.data),
                  static_cast<float*>(workspace.shared_scale.data), plan.tokens, stream);
    }
    launch_s3_tiled(x, weights, plan, workspace, stream);
    launch_s4_tiled(weights, destination, plan, workspace, stream);
}

} // namespace ninferno::ops::detail
