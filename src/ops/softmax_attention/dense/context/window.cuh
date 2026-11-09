#pragma once

// A window over a paged context: the last `window` context entries and the query block, the
// ordinary scores (context_window_attention); FOLD's window branch (fold.cuh) is the same read
// with each staged key turned to its own position.

#include "ops/softmax_attention/dense/context/kernel.cuh"

namespace ninferno::ops {

// The window: the last `window` context entries (keys as cached) and the query block.
template <class Geometry>
struct ContextWindowPolicy : ContextAttentionPolicy<Geometry> {
    using Geo     = Geometry;
    int window    = 0;
    int key_block = 32; // the kernel's tile height
    int first     = 0;  // the first window entry: max(length - window, 0)

    // The window's tiles start at a multiple of the tile height: a tile that straddled a page
    // boundary would read its second page's rows from the physical page after the first (the
    // pages of a context are not adjacent), so the leading tile takes the entries below `first`
    // too and allow_context masks them.
    __device__ __forceinline__ int context_count(int value) const {
        const int begin = max(value - window, 0);
        return value - (begin / key_block) * key_block;
    }

    __device__ __forceinline__ bool allow_context(int, int key) const { return key >= first; }
};

// context_window_attention: the window alone over keys cached rotated at their own positions, no
// far memory. The split bodies are the window branch's; the reduce is the plain one over the
// window's tile count.
template <class Geo, int Tokens, int WarpsPerCta, int KeyBlock, typename KT = __nv_bfloat16,
          typename VT = __half>
__launch_bounds__(WarpsPerCta * 32, 2) __global__ void context_window_partial_kernel(
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
    ContextWindowPolicy<Geo> policy{};
    policy.block_table =
        block_tables + static_cast<std::int64_t>(logical_pages) * table_rows[batch];
    policy.physical_pages = physical_pages;
    policy.logical_pages  = logical_pages;
    policy.k_scale        = context_k_scale;
    policy.v_scale        = context_v_scale;
    policy.window         = window;
    policy.key_block      = KeyBlock;
    policy.first          = max(length - window, 0);
    context_query_split_partial_body<ContextWindowPolicy<Geo>, Tokens, WarpsPerCta, KeyBlock,
                                     false>(q, query_k, query_v, valid_columns, context_k,
                                            context_v, policy, length, max_context, split_capacity,
                                            scale, partial_acc, partial_m, partial_l,
                                            static_cast<__nv_bfloat16*>(nullptr));
}

template <class Geo, int Tokens, int KeyBlock>
__launch_bounds__(128, 2) __global__
    void context_window_reduce_kernel(const __nv_bfloat16* __restrict__ partial_acc,
                                      const float* __restrict__ partial_m,
                                      const float* __restrict__ partial_l,
                                      const std::int32_t* __restrict__ context_length,
                                      const std::int32_t* __restrict__ valid_columns,
                                      int max_context, int split_capacity, int window,
                                      __nv_bfloat16* __restrict__ out) {
    const int length = context_length[static_cast<int>(blockIdx.z)];
    // ContextWindowPolicy::context_count: the window's tiles start at a multiple of the tile
    // height.
    const int count = length - (max(length - window, 0) / KeyBlock) * KeyBlock;
    context_query_reduce_body<Tokens, KeyBlock, Geo>(partial_acc, partial_m, partial_l, length,
                                                     count, valid_columns, max_context,
                                                     split_capacity, out);
}

} // namespace ninferno::ops
