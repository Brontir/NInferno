#include "ops/softmax_attention/dense/context/launch.h"
#include "core/per_device.h"

#include "core/device.h"
#include "ops/softmax_attention/dense/context/fold.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

template <int Tokens, class Launch>
void fold_token_case(Launch&& launch) {
    launch.template operator()<Tokens, kContextQueryWarps<ContextQueryD128, Tokens>>();
}

template <class Launch>
void fold_tokens(std::int32_t tokens, Launch&& launch) {
    switch (tokens) {
#define NINFERNO_FOLD_TOKEN_CASE(TOKENS)                                                           \
    case TOKENS:                                                                                   \
        fold_token_case<TOKENS>(launch);                                                           \
        return
        NINFERNO_FOLD_TOKEN_CASE(1);
        NINFERNO_FOLD_TOKEN_CASE(2);
        NINFERNO_FOLD_TOKEN_CASE(3);
        NINFERNO_FOLD_TOKEN_CASE(4);
        NINFERNO_FOLD_TOKEN_CASE(5);
        NINFERNO_FOLD_TOKEN_CASE(6);
        NINFERNO_FOLD_TOKEN_CASE(7);
        NINFERNO_FOLD_TOKEN_CASE(8);
        NINFERNO_FOLD_TOKEN_CASE(9);
        NINFERNO_FOLD_TOKEN_CASE(10);
        NINFERNO_FOLD_TOKEN_CASE(11);
        NINFERNO_FOLD_TOKEN_CASE(12);
        NINFERNO_FOLD_TOKEN_CASE(13);
        NINFERNO_FOLD_TOKEN_CASE(14);
        NINFERNO_FOLD_TOKEN_CASE(15);
        NINFERNO_FOLD_TOKEN_CASE(16);
#undef NINFERNO_FOLD_TOKEN_CASE
    default:
        throw std::invalid_argument("context_fold_attention: unsupported T");
    }
}

// One branch's partials over its plan (always split: the merge reads both branches' partials).
template <bool Far>
void fold_branch_launch(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                        const Tensor& context_lengths, const Tensor& valid_columns,
                        const Tensor& table_rows, float scale, std::int32_t window,
                        const PagedKVBatchLayerView& context, const ContextAttentionPlan& plan,
                        Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                        cudaStream_t stream) {
    using Geo = ContextQueryD128;
    fold_tokens(q.ne[2], [&]<int Tokens, int Warps>() {
        if (plan.warps != Warps || plan.route != ContextAttentionRoute::SplitKv ||
            plan.split_capacity < 1) {
            throw std::invalid_argument("context_fold_attention: inconsistent plan");
        }
        const auto launch = [&]<int KeyBlock>() {
            constexpr std::size_t SmemBytes = 2u * KeyBlock * Geo::D * sizeof(__nv_bfloat16);
            const dim3 grid(Geo::Hkv, plan.split_capacity, q.ne[3]);
            const auto run = [&]<typename KT, typename VT>(auto kernel, std::size_t bytes) {
                if (bytes > 48u * 1024u) {
                    static ::ninferno::PerDevice<bool> opted;
                    ::ninferno::allow_dynamic_smem(opted, kernel, static_cast<int>(bytes));
                }
                kernel<<<grid, Warps * 32, bytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(query_k.data),
                    static_cast<const __nv_bfloat16*>(query_v.data),
                    static_cast<const std::int32_t*>(context_lengths.data),
                    static_cast<const std::int32_t*>(valid_columns.data),
                    static_cast<const std::int32_t*>(table_rows.data),
                    static_cast<const KT*>(context.k_pages.data),
                    static_cast<const VT*>(context.v_pages.data),
                    static_cast<const __half*>(context.k_scale_pages.data),
                    static_cast<const __half*>(context.v_scale_pages.data),
                    static_cast<const std::int32_t*>(context.block_tables.data),
                    context.k_pages.ne[2], context.block_tables.ne[0],
                    context.block_tables.ne[0] * kPagedKVPageSize, plan.split_capacity, scale,
                    window, static_cast<__nv_bfloat16*>(partial_acc.data),
                    static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
                CUDA_CHECK(cudaGetLastError());
            };
            if (context.storage == KvCacheStorage::Int8Group64) {
                run.template operator()<std::int8_t, std::int8_t>(
                    &context_fold_partial_kernel<Geo, Tokens, Warps, KeyBlock, Far, std::int8_t,
                                                 std::int8_t>,
                    SmemBytes + kContextQueryInt8StageBytes<Geo::D, KeyBlock>);
            } else {
                run.template operator()<__nv_bfloat16, __half>(
                    &context_fold_partial_kernel<Geo, Tokens, Warps, KeyBlock, Far>, SmemBytes);
            }
        };
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
        throw std::invalid_argument("context_fold_attention: inconsistent plan");
    });
}

} // namespace

ContextFoldPlan context_fold_resolve_plan(ContextAttentionProfile profile, std::int32_t tokens,
                                          ContextAttentionExecutionEnvelope envelope,
                                          std::int32_t window) {
    if (profile != ContextAttentionProfile::D128Q32K8 || window < 1) {
        throw std::invalid_argument("context_fold_attention plan: D128 drafters with a window");
    }
    // Both branches split (a direct plan writes no partials): an envelope of at least one entry.
    const std::uint32_t w = static_cast<std::uint32_t>(window);
    const ContextAttentionExecutionEnvelope window_envelope{
        .min_context = std::min(envelope.min_context, w),
        .max_context = std::max(1u, std::min(envelope.max_context, w)),
    };
    const ContextAttentionExecutionEnvelope far_envelope{
        .min_context = envelope.min_context,
        .max_context = std::max(1u, envelope.max_context),
    };
    return {context_attention_resolve_plan(profile, tokens, window_envelope),
            context_attention_resolve_plan(profile, tokens, far_envelope)};
}

void context_fold_launch(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                         const Tensor& context_lengths, const Tensor& valid_columns,
                         const Tensor& table_rows, float scale, ContextFold fold,
                         const PagedKVBatchLayerView& context, const ContextFoldPlan& plan,
                         FoldPartials& window, FoldPartials& far, Tensor& q_far, Tensor& out,
                         cudaStream_t stream) {
    fold_branch_launch<false>(q, query_k, query_v, context_lengths, valid_columns, table_rows, scale,
                              fold.window, context, plan.window, window.acc, window.m, window.l,
                              stream);
    fold_tokens(q.ne[2], [&]<int Tokens, int>() {
        using Geo = ContextQueryD128;
        context_fold_query_kernel<Geo, Tokens><<<dim3(Tokens, q.ne[3]), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::int32_t*>(context_lengths.data),
            static_cast<const std::int32_t*>(valid_columns.data), fold.window,
            static_cast<__nv_bfloat16*>(q_far.data));
        CUDA_CHECK(cudaGetLastError());
    });
    fold_branch_launch<true>(q_far, query_k, query_v, context_lengths, valid_columns, table_rows,
                             scale * fold.alpha, fold.window, context, plan.far, far.acc, far.m,
                             far.l, stream);
    fold_tokens(q.ne[2], [&]<int Tokens, int>() {
        using Geo = ContextQueryD128;
        const dim3 grid(Geo::Hq, Tokens, q.ne[3]);
        context_fold_reduce_kernel<Geo, Tokens><<<grid, Geo::D, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(window.acc.data),
            static_cast<const float*>(window.m.data), static_cast<const float*>(window.l.data),
            plan.window.split_capacity, static_cast<const __nv_bfloat16*>(far.acc.data),
            static_cast<const float*>(far.m.data), static_cast<const float*>(far.l.data),
            plan.far.split_capacity, static_cast<const std::int32_t*>(context_lengths.data),
            static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<int>(context.block_tables.ne[0] * kPagedKVPageSize), fold.window,
            plan.window.key_block, plan.far.key_block, fold.alpha,
            static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace ninferno::ops::detail
