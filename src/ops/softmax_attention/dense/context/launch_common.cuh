#pragma once

// The scaffold the context attention launches share (launch.cu, launch_fold.cu,
// launch_window.cu): the token count and the tile height become template arguments, and a split
// partial kernel launches over BF16 or int8 pages with the shared memory it needs.

#include "ops/softmax_attention/dense/context/launch.h"

#include "core/device.h"
#include "core/per_device.h"
#include "ops/softmax_attention/dense/context/kernel.cuh"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninferno::ops::detail {

template <class Geo, class Launch, int... Offsets>
void dispatch_tokens_over(std::int32_t tokens, const char* op, Launch&& launch,
                          std::integer_sequence<int, Offsets...>) {
    const bool found =
        ((tokens == Offsets + 1
              ? (launch.template operator()<Offsets + 1, kContextQueryWarps<Geo, Offsets + 1>>(),
                 true)
              : false) ||
         ...);
    if (!found) { throw std::invalid_argument(std::string(op) + ": unsupported T"); }
}

// Calls launch.template operator()<Tokens, Warps>() for T = 1..16.
template <class Geo, class Launch>
void dispatch_tokens(std::int32_t tokens, const char* op, Launch&& launch) {
    dispatch_tokens_over<Geo>(tokens, op, launch, std::make_integer_sequence<int, 16>{});
}

// Calls launch.template operator()<KeyBlock>() for the plan tile height: 32 for a CTA of at most
// 64 query rows, 64 for more than 8 tokens.
template <int Tokens, int Warps, class Launch>
void select_key_block(const ContextAttentionPlan& plan, const char* op, Launch&& launch) {
    constexpr int Rows = Warps * 16;
    if (plan.key_block == 32) {
        if constexpr (Rows <= 64) {
            launch.template operator()<32>();
            return;
        }
    } else if (plan.key_block == 64) {
        if constexpr (Tokens > 8) {
            launch.template operator()<64>();
            return;
        }
    }
    throw std::invalid_argument(std::string(op) + ": inconsistent plan");
}

// A plan whose partials the caller reads back (split) and that belongs to this token case.
inline void require_split_plan(const ContextAttentionPlan& plan, int warps, const char* op) {
    if (plan.warps != warps || plan.route != ContextAttentionRoute::SplitKv ||
        plan.split_capacity < 1) {
        throw std::invalid_argument(std::string(op) + ": inconsistent plan");
    }
}

// The inputs of one context attention call.
struct ContextCall {
    const Tensor& q;
    const Tensor& query_k;
    const Tensor& query_v;
    const Tensor& context_lengths;
    const Tensor& valid_columns;
    const Tensor& table_rows;
    const PagedKVBatchLayerView& context;
    float scale;
};

template <auto Kernel, class KT, class VT>
void launch_window_partial(const ContextCall& c, const ContextAttentionPlan& plan, int kv_heads,
                           std::size_t smem_bytes, std::int32_t window, Tensor& partial_acc,
                           Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    if (smem_bytes > 48u * 1024u) {
        // One opt-in per kernel (per device): the kernel is the template argument.
        static ::ninferno::PerDevice<bool> opted;
        ::ninferno::allow_dynamic_smem(opted, Kernel, static_cast<int>(smem_bytes));
    }
    const dim3 grid(kv_heads, plan.split_capacity, c.q.ne[3]);
    Kernel<<<grid, plan.warps * 32, smem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(c.q.data),
        static_cast<const __nv_bfloat16*>(c.query_k.data),
        static_cast<const __nv_bfloat16*>(c.query_v.data),
        static_cast<const std::int32_t*>(c.context_lengths.data),
        static_cast<const std::int32_t*>(c.valid_columns.data),
        static_cast<const std::int32_t*>(c.table_rows.data),
        static_cast<const KT*>(c.context.k_pages.data),
        static_cast<const VT*>(c.context.v_pages.data),
        static_cast<const __half*>(c.context.k_scale_pages.data),
        static_cast<const __half*>(c.context.v_scale_pages.data),
        static_cast<const std::int32_t*>(c.context.block_tables.data), c.context.k_pages.ne[2],
        c.context.block_tables.ne[0], c.context.block_tables.ne[0] * kPagedKVPageSize,
        plan.split_capacity, c.scale, window, static_cast<__nv_bfloat16*>(partial_acc.data),
        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

// A window or FOLD split partial kernel over the call pages: Kernel over BF16 K and FP16 V,
// Int8Kernel over int8 pages (their raw staging rides after the operand tiles).
template <class Geo, int KeyBlock, auto Kernel, auto Int8Kernel>
void launch_window_partial_by_storage(const ContextCall& c, const ContextAttentionPlan& plan,
                                      std::int32_t window, Tensor& partial_acc, Tensor& partial_m,
                                      Tensor& partial_l, cudaStream_t stream) {
    constexpr std::size_t SmemBytes = 2u * KeyBlock * Geo::D * sizeof(__nv_bfloat16);
    if (c.context.storage == KvCacheStorage::Int8Group64) {
        launch_window_partial<Int8Kernel, std::int8_t, std::int8_t>(
            c, plan, Geo::Hkv, SmemBytes + kContextQueryInt8StageBytes<Geo::D, KeyBlock>, window,
            partial_acc, partial_m, partial_l, stream);
    } else {
        launch_window_partial<Kernel, __nv_bfloat16, __half>(c, plan, Geo::Hkv, SmemBytes, window,
                                                             partial_acc, partial_m, partial_l,
                                                             stream);
    }
}

} // namespace ninferno::ops::detail
