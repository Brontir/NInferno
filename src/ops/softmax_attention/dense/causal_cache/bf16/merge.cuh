#pragma once

#include "ops/softmax_attention/dense/causal_cache/bf16/tile_io.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/split_policy.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/softmax.cuh"
#include "ops/softmax_attention/common/causal_epilogue.cuh"
#include "ops/softmax_attention/common/causal_merge_columns.cuh"

namespace ninferno::ops::detail {

// Merge one query/head's split statistics once per CTA, from this thread's split's stored maximum
// and sum (loaded beside the positions; thread `tid` holds split `tid`). Published scalars are
// separate from the reduction/weight storage, so later writes cannot race another warp's scalar
// read.
template <class Schedule>
__device__ __forceinline__ float bf16_kv_merge_statistics(float stored_m, float stored_l,
                                                          int splits, float* weights,
                                                          float* warp_sums, float* scalars) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float m      = tid < splits ? stored_m : -CUDART_INF_F;
    const float warp_m = warp_max(m);
    if (lane == 0) warp_sums[warp] = warp_m;
    __syncthreads();
    if (warp == 0) {
        const float maximum = warp_max(tid < Schedule::kWarps ? warp_sums[tid] : -CUDART_INF_F);
        if (tid == 0) scalars[0] = maximum;
    }
    __syncthreads();
    const float maximum     = scalars[0];
    const float l           = tid < splits ? stored_l : 0.0f;
    const float weight      = l > 0.0f && maximum > -CUDART_INF_F ? exp2_approx(m - maximum) : 0.0f;
    const float denominator = block_reduce_sum<Schedule::kThreads>(l * weight, warp_sums);
    if (tid == 0) scalars[1] = denominator;
    __syncthreads();
    const float total = scalars[1];
    if (tid < splits) weights[tid] = total > 0.0f ? weight : 0.0f;
    __syncthreads();
    return total;
}

// One CTA merges DChunk output columns of one query head and token; the independent loads go out
// together, as causal_natural_merge_kernel's do (common/causal_merge.cuh).
template <typename Geometry, class Schedule, bool MultiBatch, bool Masked, bool Aligned = false>
__launch_bounds__(Schedule::kThreads) __global__
    void bf16_kv_merge_kernel(const float* partial_acc, const float* partial_m,
                              const float* partial_l, const std::int32_t* positions,
                              const std::int32_t* valid_columns, std::int32_t tokens,
                              std::int32_t batch_size, Bf16KvPartition partition,
                              __nv_bfloat16* out, const __nv_bfloat16* gate) {
    const int split_count = partition.capacity;
    constexpr int DChunk  = Schedule::kDChunk;
    static_assert(DChunk <= Geometry::kHeadDim);
    using Columns = CausalMergeColumns<Geometry, DChunk, Schedule::kThreads>;

    const int q_head      = static_cast<int>(blockIdx.x);
    const int d_start     = static_cast<int>(blockIdx.y) * DChunk;
    const int flat_column = static_cast<int>(blockIdx.z);
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid = threadIdx.x;
    if (q_head >= Geometry::QHeads || token >= tokens) { return; }
    if constexpr (MultiBatch) {
        if (batch >= batch_size) { return; }
    }

    if constexpr (MultiBatch) {
        positions += batch * tokens;
        const std::int64_t partial_acc_row = static_cast<std::int64_t>(batch) * Geometry::kHeadDim *
                                             Geometry::QHeads * tokens * split_count;
        const std::int64_t partial_stat_row =
            static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_acc += partial_acc_row;
        partial_m += partial_stat_row;
        partial_l += partial_stat_row;
    }
    // Every split the partials hold (the capacity), before the live count is known.
    const auto stat_index = causal_stat_index<Geometry>(q_head, token, tid, tokens);
    const float stored_m  = tid < split_count ? partial_m[stat_index] : -CUDART_INF_F;
    const float stored_l  = tid < split_count ? partial_l[stat_index] : 0.0f;
    int output_column     = token;
    if constexpr (MultiBatch) { output_column += batch * tokens; }
    const int d      = d_start + tid;
    const bool owner = tid < DChunk && d < Geometry::kHeadDim;
    __nv_bfloat16 gate_value{};
    if (gate && owner) gate_value = gate[causal_q_index<Geometry>(q_head, d, output_column)];
    const int live_columns = Masked ? valid_columns[batch] : tokens;
    if constexpr (Masked) {
        const int absolute_column = token;
        if (absolute_column >= live_columns) {
            if (owner) {
                const auto index = causal_q_index<Geometry>(q_head, d, output_column);
                out[index]       = causal_gated_output(0.0f, gate ? gate + index : nullptr);
            }
            return;
        }
    }

    // Aligned: each column merges the splits its own position runs (Bf16KvPartition::aligned_row),
    // as a decode of that column alone does. Otherwise the call's window sets the partition.
    const int window = Aligned ? positions[token] + 1 : positions[live_columns - 1] + 1;
    const int active_split_count =
        Aligned ? Bf16KvPartition::aligned_live(window).splits : partition.live(window).splits;
    Columns columns(partial_acc, q_head, d_start, token, tokens, active_split_count);

    __shared__ float weights[Schedule::kThreads], warp_sums[Schedule::kWarps], scalars[2];
    __shared__ __align__(16) float stage[Columns::kStageFloats];
    const float head_l = bf16_kv_merge_statistics<Schedule>(stored_m, stored_l, active_split_count,
                                                            weights, warp_sums, scalars);
    const float numerator = columns.numerator(weights, stage);
    if (!owner) return;

    const float value = (head_l > 0.0f) ? numerator / head_l : 0.0f;
    const auto index  = causal_q_index<Geometry>(q_head, d, output_column);
    out[index]        = causal_gated_output(value, gate != nullptr, gate_value);
}


} // namespace ninferno::ops::detail
