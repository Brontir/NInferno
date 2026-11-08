#include "ops/softmax_attention/dense/context/launch_common.cuh"
#include "ops/softmax_attention/dense/context/fold.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

constexpr const char* kOp = "context_fold_attention";

// One branch partials over its plan (always split: the merge reads both branches partials).
template <bool Far>
void fold_branch_launch(const ContextCall& c, std::int32_t window, const ContextAttentionPlan& plan,
                        Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                        cudaStream_t stream) {
    using Geo = ContextQueryD128;
    dispatch_tokens<Geo>(c.q.ne[2], kOp, [&]<int Tokens, int Warps>() {
        require_split_plan(plan, Warps, kOp);
        select_key_block<Tokens, Warps>(plan, kOp, [&]<int KeyBlock>() {
            launch_window_partial_by_storage<
                Geo, KeyBlock, &context_fold_partial_kernel<Geo, Tokens, Warps, KeyBlock, Far>,
                &context_fold_partial_kernel<Geo, Tokens, Warps, KeyBlock, Far, std::int8_t,
                                             std::int8_t>>(c, plan, window, partial_acc, partial_m,
                                                           partial_l, stream);
        });
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
    const ContextCall call{q,          query_k, query_v, context_lengths, valid_columns,
                           table_rows, context, scale};
    fold_branch_launch<false>(call, fold.window, plan.window, window.acc, window.m, window.l,
                              stream);
    dispatch_tokens<ContextQueryD128>(q.ne[2], kOp, [&]<int Tokens, int>() {
        using Geo = ContextQueryD128;
        context_fold_query_kernel<Geo, Tokens><<<dim3(Tokens, q.ne[3]), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::int32_t*>(context_lengths.data),
            static_cast<const std::int32_t*>(valid_columns.data), fold.window,
            static_cast<__nv_bfloat16*>(q_far.data));
        CUDA_CHECK(cudaGetLastError());
    });
    const ContextCall far_call{q_far,         query_k,    query_v, context_lengths,
                               valid_columns, table_rows, context, scale * fold.alpha};
    fold_branch_launch<true>(far_call, fold.window, plan.far, far.acc, far.m, far.l, stream);
    dispatch_tokens<ContextQueryD128>(q.ne[2], kOp, [&]<int Tokens, int>() {
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
