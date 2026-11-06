#pragma once

// FOLD (arXiv 2610.20392): a drafter's full-attention layer reads its context as a local window of
// exact positions and a far memory at one shared position, normalized apart and merged by their
// masses. The window: the last W context entries and the query block, the ordinary scores. The far
// memory: the entries before them, each key turned from its own position j to the shared one
// t - W - B (t the block's first position, B its width), so a block row r reads every far entry at
// distance W + B + r; their scores sharpened by alpha, their mass the l_alpha norm of the
// exponentiated scores. The keys stay as the cache stores them (rotated at their own positions):
// the turn happens on the staged tile.

#include "ops/common/dflash_rope.cuh"
#include "ops/softmax_attention/dense/context/kernel.cuh"

namespace ninferno::ops {

// The window: the last `window` context entries and the query block.
template <class Geometry>
struct FoldWindowPolicy : ContextAttentionPolicy<Geometry> {
    int window = 0;

    __device__ __forceinline__ int context_count(int value) const { return min(value, window); }
};

// The far memory: the context entries before the window, their keys turned to the shared
// position; no query keys.
template <class Geometry>
struct FoldFarPolicy : ContextAttentionPolicy<Geometry> {
    using Geo                           = Geometry;
    static constexpr bool kQueryKeys    = false;
    static constexpr bool kRerotateKeys = true;

    int far_end = 0; // the first window entry: entries [0, far_end) are far
    int shared  = 0; // the shared position t - W - B

    __device__ __forceinline__ bool allow_context(int, int key) const { return key < far_end; }

    // Turns each live far key of the staged tile (rows key0 ..) by shared - j: a key rotated at j
    // becomes one rotated at the shared position. The angle of row `row` and pair i is
    // (shared - key0) f_i - row f_i: the first in FP64 (dflash_rope.cuh, a large position), the
    // second in FP32 (row < 64), combined by the angle difference. True where it turned a row.
    template <int KeyBlock, int Threads>
    __device__ __forceinline__ bool rerotate(__nv_bfloat16* k_s, int key0, int valid_keys,
                                             int tid) const {
        static_assert(Geo::D == 128, "FOLD turns D128 keys (the DFlash RoPE table)");
        if (key0 >= far_end) { return false; }
        constexpr int Half = Geo::D / 2;
        const int rows     = min(valid_keys, far_end - key0);
        const double base  = static_cast<double>(shared - key0);
        for (int item = tid; item < rows * Half; item += Threads) {
            const int row  = item / Half;
            const int pair = item - row * Half;
            float s0, c0;
            rope_angle_sincos(base * kDflashRopeInvFrequency[pair], &s0, &c0);
            float s1, c1;
            sincosf(static_cast<float>(row * kDflashRopeInvFrequency[pair]), &s1, &c1);
            const float c = c0 * c1 + s0 * s1; // cos(a - b)
            const float s = s0 * c1 - c0 * s1; // sin(a - b)
            __nv_bfloat16& x = k_s[row * Geo::D + context_query_swz(row, pair)];
            __nv_bfloat16& y = k_s[row * Geo::D + context_query_swz(row, pair + Half)];
            const float xf   = __bfloat162float(x);
            const float yf   = __bfloat162float(y);
            x                = __float2bfloat16(xf * c - yf * s);
            y                = __float2bfloat16(yf * c + xf * s);
        }
        return true;
    }
};

template <class Geo, int Tokens, int WarpsPerCta, int KeyBlock, bool Far, typename KT = __nv_bfloat16,
          typename VT = __half>
__launch_bounds__(WarpsPerCta * 32, 2) __global__ void context_fold_partial_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
    const __nv_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ context_length,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ table_rows,
    const KT* __restrict__ context_k, const VT* __restrict__ context_v,
    const __half* __restrict__ context_k_scale, const __half* __restrict__ context_v_scale,
    const std::int32_t* __restrict__ block_tables, int physical_pages, int logical_pages,
    int max_context, int split_capacity, float scale, int window,
    __nv_bfloat16* __restrict__ partial_acc, float* __restrict__ partial_m,
    float* __restrict__ partial_l) {
    const int batch  = static_cast<int>(blockIdx.z);
    const int length = context_length[batch];
    const auto base  = [&](auto& policy) {
        policy.block_table    = block_tables + static_cast<std::int64_t>(logical_pages) * table_rows[batch];
        policy.physical_pages = physical_pages;
        policy.logical_pages  = logical_pages;
        policy.k_scale        = context_k_scale;
        policy.v_scale        = context_v_scale;
    };
    if constexpr (Far) {
        FoldFarPolicy<Geo> policy{};
        base(policy);
        policy.far_end = max(length - window, 0);
        policy.shared  = length - window - valid_columns[batch];
        context_query_split_partial_body<FoldFarPolicy<Geo>, Tokens, WarpsPerCta, KeyBlock, false>(
            q, query_k, query_v, valid_columns, context_k, context_v, policy, length, max_context,
            split_capacity, scale, partial_acc, partial_m, partial_l,
            static_cast<__nv_bfloat16*>(nullptr));
    } else {
        FoldWindowPolicy<Geo> policy{};
        base(policy);
        policy.window = window;
        context_query_split_partial_body<FoldWindowPolicy<Geo>, Tokens, WarpsPerCta, KeyBlock, false>(
            q, query_k, query_v, valid_columns, context_k, context_v, policy, length, max_context,
            split_capacity, scale, partial_acc, partial_m, partial_l,
            static_cast<__nv_bfloat16*>(nullptr));
    }
}

// One branch's splits of a (head, token): their log-sum-exp (natural log, in the branch's own
// score units) and the normalized output component `d`, from the partials as the split body wrote
// them (an inactive split: m = -inf).
template <int Tokens, class Geo>
__device__ __forceinline__ void fold_branch(const __nv_bfloat16* acc, const float* m, const float* l,
                                            int splits, int q_head, int token, int d, float& lse,
                                            float& value) {
    float global_m = -CUDART_INF_F;
    for (int split = 0; split < splits; ++split) {
        global_m = fmaxf(global_m, m[context_query_stat_index<Tokens, Geo>(q_head, token, split)]);
    }
    if (global_m == -CUDART_INF_F) {
        lse   = -CUDART_INF_F;
        value = 0.0f;
        return;
    }
    float sum = 0.0f, numerator = 0.0f;
    for (int split = 0; split < splits; ++split) {
        const auto stat = context_query_stat_index<Tokens, Geo>(q_head, token, split);
        const float w   = expf(m[stat] - global_m);
        sum += l[stat] * w;
        numerator +=
            __bfloat162float(acc[context_query_partial_index<Tokens, Geo>(q_head, d, token, split)]) * w;
    }
    lse   = sum > 0.0f ? global_m + logf(sum) : -CUDART_INF_F;
    value = sum > 0.0f ? numerator / sum : 0.0f;
}

// The merge: o = (Z_N o_N + Z_F o_F) / (Z_N + Z_F), log Z_N the window's log-sum-exp, log Z_F the
// far branch's divided by alpha (its scores were alpha times the ordinary ones).
template <class Geo, int Tokens>
__launch_bounds__(128) __global__ void context_fold_reduce_kernel(
    const __nv_bfloat16* __restrict__ window_acc, const float* __restrict__ window_m,
    const float* __restrict__ window_l, int window_splits, const __nv_bfloat16* __restrict__ far_acc,
    const float* __restrict__ far_m, const float* __restrict__ far_l, int far_splits,
    const std::int32_t* __restrict__ context_length, const std::int32_t* __restrict__ valid_columns,
    int window, int window_key_block, int far_key_block, float alpha, __nv_bfloat16* __restrict__ out) {
    const int q_head = static_cast<int>(blockIdx.x);
    const int token  = static_cast<int>(blockIdx.y);
    const int batch  = static_cast<int>(blockIdx.z);
    const int d      = static_cast<int>(threadIdx.x);
    constexpr std::int64_t QueryElements = static_cast<std::int64_t>(Geo::D) * Geo::Hq * Tokens;
    constexpr std::int64_t StatElements  = static_cast<std::int64_t>(Geo::Hq) * Tokens;
    window_acc += QueryElements * window_splits * batch;
    window_m += StatElements * window_splits * batch;
    window_l += StatElements * window_splits * batch;
    far_acc += QueryElements * far_splits * batch;
    far_m += StatElements * far_splits * batch;
    far_l += StatElements * far_splits * batch;
    out += QueryElements * batch;
    if (q_head >= Geo::Hq || token >= Tokens || d >= Geo::D) { return; }
    const int length = context_length[batch];
    if (token >= valid_columns[batch]) {
        out[context_query_q_index<Geo>(q_head, d, token)] = __float2bfloat16(0.0f);
        return;
    }
    const auto active = [](int count, int key_block, int capacity) {
        const int tiles = (count + key_block - 1) / key_block;
        return tiles > 0 ? min(tiles, capacity) : 1;
    };
    float window_lse, window_value, far_lse, far_value;
    fold_branch<Tokens, Geo>(window_acc, window_m, window_l,
                             active(min(length, window), window_key_block, window_splits), q_head,
                             token, d, window_lse, window_value);
    fold_branch<Tokens, Geo>(far_acc, far_m, far_l, active(length, far_key_block, far_splits), q_head,
                             token, d, far_lse, far_value);
    far_lse = far_lse == -CUDART_INF_F ? far_lse : far_lse / alpha;
    const float top = fmaxf(window_lse, far_lse);
    float value     = 0.0f;
    if (top > -CUDART_INF_F) {
        const float zw = window_lse == -CUDART_INF_F ? 0.0f : expf(window_lse - top);
        const float zf = far_lse == -CUDART_INF_F ? 0.0f : expf(far_lse - top);
        value          = (zw * window_value + zf * far_value) / (zw + zf);
    }
    out[context_query_q_index<Geo>(q_head, d, token)] = __float2bfloat16(value);
}

} // namespace ninferno::ops
