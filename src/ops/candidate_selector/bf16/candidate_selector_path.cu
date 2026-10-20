#include "ops/candidate_selector/bf16/candidate_selector_path_kernels.h"
#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/generic/generic_gemm.cuh"
#include "ops/kernel/sampling_device.cuh"
#include <cuda_bf16.h>
#include <stdexcept>
#include <type_traits>

namespace ninferno::ops::detail {
namespace {
constexpr int kCandidates = 16, kRank = 256;

struct DeviceArgs {
    const std::int32_t* ids;
    const float* unary;
    const __nv_bfloat16* hidden;
    const std::int32_t* anchors;
    const __nv_bfloat16* predecessor;
    const __nv_bfloat16* successor;
    // Generic codebooks (a format other than BF16): read element by element, FP32 dequantized.
    generic::GemmArgs predecessor_args, successor_args;
    generic::WeightCodec predecessor_codec, successor_codec;
    const std::int32_t* positions;
    const SamplingConfig* configs;
    std::int32_t* drafts;
    float* q;
    int steps;
};

// BF16 codebooks stage successor rows as stored; generic ones as their FP32 values.
template <bool Generic> struct alignas(16) SelectorShared {
    using Value = std::conditional_t<Generic, float, __nv_bfloat16>;
    Value successors[kCandidates * kRank];
    float product[kRank];
    float edge[kCandidates];
    int predecessor, base_position;
    float temperature;
    unsigned long long seed;
};

// The probabilities written here are the same FP32 values consumed by the draw.
__device__ int draw_rank(float edge, float temperature, unsigned long long seed, int position,
                         float* q) {
    const int lane      = threadIdx.x & 31;
    const float maximum = warp_max(edge);
    if (temperature <= 0.0F) {
        const unsigned winners =
            __ballot_sync(kFullWarpMask, lane < kCandidates && edge == maximum);
        const int selected = winners == 0 ? 0 : __ffs(winners) - 1;
        if (lane < kCandidates) q[lane] = lane == selected ? 1.0F : 0.0F;
        return selected;
    }
    const float weight      = lane < kCandidates ? __expf((edge - maximum) / temperature) : 0.0F;
    const float probability = weight / warp_sum(weight);
    if (lane < kCandidates) q[lane] = probability;
    float uniform =
        lane == 0 ? sampling_uniform(seed, position, kSamplePurposeDFlash2Proposal, 0U) : 0.0F;
    uniform          = __shfl_sync(kFullWarpMask, uniform, 0);
    float cumulative = probability;
#pragma unroll
    for (int offset = 1; offset < kCandidates; offset *= 2) {
        const float previous = __shfl_up_sync(kFullWarpMask, cumulative, offset);
        if (lane >= offset) cumulative += previous;
    }
    const unsigned hits = __ballot_sync(kFullWarpMask, lane < kCandidates && uniform < cumulative);
    if (hits) return __ffs(hits) - 1;
    // Rounding left the uniform past the last cumulative sum: take the last rank that has mass,
    // so the draft's q stays the probability it was drawn with (never 0).
    const unsigned massive = __ballot_sync(kFullWarpMask, lane < kCandidates && probability > 0.0F);
    return massive ? 31 - __clz(massive) : 0;
}

__device__ __forceinline__ float as_float(float value) { return value; }
__device__ __forceinline__ float as_float(__nv_bfloat16 value) { return __bfloat162float(value); }

template <bool Generic>
__device__ void score_row(const DeviceArgs& a, int column, SelectorShared<Generic>& shared) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int token = lane == 0 ? a.ids[column * kCandidates + warp] : 0;
    token     = __shfl_sync(kFullWarpMask, token, 0);
    if constexpr (Generic) {
        for (int r = lane; r < kRank; r += 32)
            shared.successors[warp * kRank + r] =
                generic::decode_element(a.successor_args, a.successor_codec, token, r);
    } else {
        cp_async<16, Cache::cg>(&shared.successors[warp * kRank + lane * 8],
                                a.successor + static_cast<std::int64_t>(token) * kRank + lane * 8);
        cp_commit();
    }
    // Publish the preceding draw before reading its token. Successor prefetch is independent.
    __syncthreads();
    if (tid < kRank) {
        float predecessor = 0.0F;
        if constexpr (Generic) {
            predecessor = generic::decode_element(a.predecessor_args, a.predecessor_codec,
                                                  shared.predecessor, tid);
        } else {
            predecessor = __bfloat162float(
                a.predecessor[static_cast<std::int64_t>(shared.predecessor) * kRank + tid]);
        }
        shared.product[tid] =
            predecessor *
            __bfloat162float(a.hidden[static_cast<std::int64_t>(column) * kRank + tid]);
    }
    if constexpr (!Generic) { cp_wait<0>(); }
    __syncthreads();
    {
        const int c = warp;
        float sum   = 0;
#pragma unroll
        for (int r = lane; r < kRank; r += 32)
            sum = fmaf(shared.product[r], as_float(shared.successors[c * kRank + r]), sum);
        sum = warp_reduce_sum(sum);
        if (lane == 0) shared.edge[c] = a.unary[column * kCandidates + c] + sum;
    }
    __syncthreads();
}

template <bool Generic>
__global__ __launch_bounds__(512, 1) void selector_walk_kernel(DeviceArgs a) {
    __shared__ SelectorShared<Generic> shared;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, batch = blockIdx.x;
    if (tid == 0) {
        shared.predecessor   = a.anchors[batch];
        shared.base_position = a.positions[batch];
        shared.temperature   = a.configs[batch].temperature;
        shared.seed          = a.configs[batch].seed;
    }
#pragma unroll 1
    for (int step = 0; step < a.steps; ++step) {
        const int column = batch * a.steps + step;
        score_row<Generic>(a, column, shared);
        if (warp == 0) {
            const float edge   = lane < kCandidates ? shared.edge[lane] : -CUDART_INF_F;
            const int selected = draw_rank(edge, shared.temperature, shared.seed,
                                           shared.base_position + step, a.q + column * kCandidates);
            if (lane == 0) {
                shared.predecessor = a.ids[column * kCandidates + selected];
                a.drafts[column]   = shared.predecessor;
            }
        }
    }
}

template <bool Generic>
__global__ __launch_bounds__(512, 2) void selector_lattice_kernel(DeviceArgs a, float* edges) {
    const int column = blockIdx.x, p = blockIdx.y;
    const int step = column % a.steps, batch = column / a.steps;
    if (step == 0 && p != 0) return;
    __shared__ SelectorShared<Generic> shared;
    if (threadIdx.x == 0)
        shared.predecessor = step == 0 ? a.anchors[batch] : a.ids[(column - 1) * kCandidates + p];
    score_row<Generic>(a, column, shared);
    if (threadIdx.x < kCandidates)
        edges[(static_cast<std::int64_t>(column) * kCandidates + p) * kCandidates + threadIdx.x] =
            shared.edge[threadIdx.x];
}

__global__ __launch_bounds__(32) void selector_lattice_walk_kernel(DeviceArgs a,
                                                                   const float* edges) {
    const int lane = threadIdx.x, batch = blockIdx.x;
    const auto seed         = a.configs[batch].seed;
    const float temperature = a.configs[batch].temperature;
    const int position      = a.positions[batch];
    int predecessor_rank    = 0;
#pragma unroll 1
    for (int step = 0; step < a.steps; ++step) {
        const int column = batch * a.steps + step;
        const float edge =
            lane < kCandidates
                ? edges[(static_cast<std::int64_t>(column) * kCandidates + predecessor_rank) *
                            kCandidates +
                        lane]
                : -CUDART_INF_F;
        int selected =
            draw_rank(edge, temperature, seed, position + step, a.q + column * kCandidates);
        predecessor_rank = selected;
        if (lane == 0) a.drafts[column] = a.ids[column * kCandidates + selected];
    }
}

} // namespace

void candidate_selector_path_launch(SelectorRoute route, const Tensor& candidate_ids,
                                    const Tensor& unary_scores, const Tensor& projected_hidden,
                                    const Tensor& anchors,
                                    const SelectorCodebook& predecessor_codebook,
                                    const SelectorCodebook& successor_codebook,
                                    const Tensor& base_positions,
                                    const SamplingConfig* configs, Tensor& drafts,
                                    Tensor& proposal_q, const SelectorWorkspace& workspace,
                                    cudaStream_t stream) {
    if (predecessor_codebook.generic != successor_codebook.generic) {
        throw std::invalid_argument("candidate_selector_path: codebooks of one kind expected");
    }
    DeviceArgs args{};
    args.ids               = static_cast<const std::int32_t*>(candidate_ids.data);
    args.unary             = static_cast<const float*>(unary_scores.data);
    args.hidden            = static_cast<const __nv_bfloat16*>(projected_hidden.data);
    args.anchors           = static_cast<const std::int32_t*>(anchors.data);
    args.predecessor       = static_cast<const __nv_bfloat16*>(predecessor_codebook.bf16);
    args.successor         = static_cast<const __nv_bfloat16*>(successor_codebook.bf16);
    args.predecessor_args  = predecessor_codebook.args;
    args.successor_args    = successor_codebook.args;
    args.predecessor_codec = predecessor_codebook.codec;
    args.successor_codec   = successor_codebook.codec;
    args.positions         = static_cast<const std::int32_t*>(base_positions.data);
    args.configs           = configs;
    args.drafts            = static_cast<std::int32_t*>(drafts.data);
    args.q                 = static_cast<float*>(proposal_q.data);
    args.steps             = candidate_ids.ne[1];
    const bool generic_books = predecessor_codebook.generic;
    if (route == SelectorRoute::Direct) {
        if (generic_books)
            selector_walk_kernel<true><<<candidate_ids.ne[2], 512, 0, stream>>>(args);
        else
            selector_walk_kernel<false><<<candidate_ids.ne[2], 512, 0, stream>>>(args);
    } else {
        auto* edges = static_cast<float*>(workspace.edges.data);
        const dim3 grid(args.steps * candidate_ids.ne[2], kCandidates);
        if (generic_books)
            selector_lattice_kernel<true><<<grid, 512, 0, stream>>>(args, edges);
        else
            selector_lattice_kernel<false><<<grid, 512, 0, stream>>>(args, edges);
        CUDA_CHECK(cudaGetLastError());
        selector_lattice_walk_kernel<<<candidate_ids.ne[2], 32, 0, stream>>>(args, edges);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninferno::ops::detail
