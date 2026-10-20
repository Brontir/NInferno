#pragma once

#include "ops/softmax_attention/common/context_query.cuh"

namespace ninferno::ops {

struct SlidingWindowAttentionPolicy {
    using Geo                        = ContextQueryD128;
    static constexpr bool PageMapped = false;

    const std::int32_t* positions;
    int padded_context;
    int window_mask;
    // Int8 ring only: the lane's FP16 G64 scales, [2] x padded x heads (core/cyclic_kv_cache.h).
    const __half* k_scale = nullptr;
    const __half* v_scale = nullptr;

    __device__ __forceinline__ int context_count(int value) const { return min(value, window_mask); }

    __device__ __forceinline__ int query_position(int token) const { return positions[token]; }

    __device__ __forceinline__ bool allow_context(int query, int key) const {
        return key >= query - (window_mask);
    }

    __device__ __forceinline__ std::int64_t context_tile(int kv_head, int, int) const {
        return static_cast<std::int64_t>(kContextQueryHeadDim) * padded_context * kv_head;
    }

    __device__ __forceinline__ std::int64_t context_index(std::int64_t tile, int d, int position,
                                                          int) const {
        const int slot = position & (window_mask);
        return tile + d + static_cast<std::int64_t>(kContextQueryHeadDim) * slot;
    }

    __device__ __forceinline__ std::int64_t scale_index(int kv_head, int key0, int, int row,
                                                        int d) const {
        const int slot = (key0 + row) & (window_mask);
        return (d >> 6) + 2 * (static_cast<std::int64_t>(padded_context) * kv_head +
                               static_cast<std::int64_t>(slot));
    }

    template <int Tokens, int KeyBlock>
    __device__ __forceinline__ void prime(int, int, int, int) {}
};

// KT/VT: BF16/FP16 ring, or int8 codes (Int8Group64; the scales follow each lane's codes).
template <int Tokens, int WarpsPerCta, int KeyBlock, bool DirectOutput, typename KT = __nv_bfloat16,
          typename VT = __half>
__launch_bounds__(WarpsPerCta * 32, 2) __global__
    void sliding_window_attention_split_partial_kernel(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
        const __nv_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ positions,
        const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ lanes,
        const KT* __restrict__ context_k, const VT* __restrict__ context_v, int padded_context,
        int window_mask, int max_context, int split_capacity, float scale,
        float* __restrict__ partial_acc, float* __restrict__ partial_m,
        float* __restrict__ partial_l, __nv_bfloat16* __restrict__ out) {
    constexpr bool Int8 = std::is_same_v<KT, std::int8_t>;
    static_assert(Int8 == std::is_same_v<VT, std::int8_t>);
    const int batch = static_cast<int>(blockIdx.z);
    const std::int64_t rows = static_cast<std::int64_t>(padded_context) * kContextQueryKVHeads;
    const std::int64_t lane_elements =
        rows *
        (Int8 ? kContextQueryHeadDim + 2 * (kContextQueryHeadDim / 64) : kContextQueryHeadDim);
    context_k += lane_elements * lanes[batch];
    context_v += lane_elements * lanes[batch];
    const std::int32_t* batch_positions = positions + static_cast<std::int64_t>(Tokens) * batch;
    SlidingWindowAttentionPolicy policy{
        .positions      = batch_positions,
        .padded_context = padded_context,
        .window_mask = window_mask,
    };
    if constexpr (Int8) {
        policy.k_scale = reinterpret_cast<const __half*>(context_k + rows * kContextQueryHeadDim);
        policy.v_scale = reinterpret_cast<const __half*>(context_v + rows * kContextQueryHeadDim);
    }
    context_query_split_partial_body<SlidingWindowAttentionPolicy, Tokens, WarpsPerCta,
                                     KeyBlock, DirectOutput>(
        q, query_k, query_v, valid_columns, context_k, context_v, policy,
        DirectOutput && valid_columns[batch] == 0 ? 0 : batch_positions[0], max_context,
        split_capacity, scale, partial_acc, partial_m, partial_l, out);
}

template <int Tokens, int KeyBlock, int WarpsPerBlock>
__launch_bounds__(WarpsPerBlock * 32, 2) __global__
    void sliding_window_attention_reduce_kernel(const float* __restrict__ partial_acc,
                                                const float* __restrict__ partial_m,
                                                const float* __restrict__ partial_l,
                                                const std::int32_t* __restrict__ positions,
                                                const std::int32_t* __restrict__ valid_columns,
                                                int window_mask, int max_context, int split_capacity,
                                                __nv_bfloat16* __restrict__ out) {
    static_assert(WarpsPerBlock >= 1 && WarpsPerBlock <= 8);
    constexpr int MaxSplits = 32;
    constexpr unsigned Mask = 0xffffffffu;
    __shared__ float weights[WarpsPerBlock][MaxSplits];

    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int batch      = static_cast<int>(blockIdx.z);
    const int output_row = static_cast<int>(blockIdx.x) * WarpsPerBlock + warp;
    const int token      = output_row / kContextQueryQHeads;
    const int q_head     = output_row - token * kContextQueryQHeads;
    if (warp >= WarpsPerBlock || token >= Tokens) return;

    constexpr std::int64_t QueryElements =
        static_cast<std::int64_t>(kContextQueryHeadDim) * kContextQueryQHeads * Tokens;
    constexpr std::int64_t StatElements = static_cast<std::int64_t>(kContextQueryQHeads) * Tokens;
    partial_acc += QueryElements * split_capacity * batch;
    partial_m += StatElements * split_capacity * batch;
    partial_l += StatElements * split_capacity * batch;
    positions += static_cast<std::int64_t>(Tokens) * batch;
    out += QueryElements * batch;

    const int length = positions[0];
    if (length < 0 || length > max_context || token >= valid_columns[batch]) {
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int d                                  = lane + item * 32;
            out[context_query_q_index(q_head, d, token)] = __float2bfloat16(0.0f);
        }
        return;
    }
    const int context_count = min(length, window_mask);
    const int context_tiles = (context_count + KeyBlock - 1) / KeyBlock;
    const int active_splits = context_tiles > 0 ? min(context_tiles, split_capacity) : 1;

    float local_m = -CUDART_INF_F;
    for (int split = lane; split < active_splits; split += 32) {
        local_m = fmaxf(local_m, partial_m[context_query_stat_index<Tokens>(q_head, token, split)]);
    }
    const float global_m = warp_max<32>(local_m, Mask);

    float local_l = 0.0f;
    for (int split = lane; split < active_splits; split += 32) {
        const auto stat      = context_query_stat_index<Tokens>(q_head, token, split);
        const float weight   = expf(partial_m[stat] - global_m);
        weights[warp][split] = weight;
        local_l += partial_l[stat] * weight;
    }
    const float global_l = warp_sum<32>(local_l, Mask);
    __syncwarp(Mask);

#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d     = lane + item * 32;
        float numerator = 0.0f;
        for (int split = 0; split < active_splits; ++split) {
            numerator += partial_acc[context_query_partial_index<Tokens>(q_head, d, token, split)] *
                         weights[warp][split];
        }
        const float value = global_l > 0.0f ? numerator / global_l : 0.0f;
        out[context_query_q_index(q_head, d, token)] = __float2bfloat16(value);
    }
}

} // namespace ninferno::ops
