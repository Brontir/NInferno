#include "ops/softmax_attention/dense/context/launch_common.cuh"
#include "ops/softmax_attention/dense/context/window.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

constexpr const char* kOp = "context_window_attention";

template <class Geo>
void window_launch_profile(const ContextCall& c, std::int32_t window,
                           const ContextAttentionPlan& plan, Tensor& partial_acc,
                           Tensor& partial_m, Tensor& partial_l, Tensor& out,
                           cudaStream_t stream) {
    dispatch_tokens<Geo>(c.q.ne[2], kOp, [&]<int Tokens, int Warps>() {
        require_split_plan(plan, Warps, kOp);
        select_key_block<Tokens, Warps>(plan, kOp, [&]<int KeyBlock>() {
            launch_window_partial_by_storage<
                Geo, KeyBlock, &context_window_partial_kernel<Geo, Tokens, Warps, KeyBlock>,
                &context_window_partial_kernel<Geo, Tokens, Warps, KeyBlock, std::int8_t,
                                               std::int8_t>>(c, plan, window, partial_acc,
                                                             partial_m, partial_l, stream);
            const dim3 reduce_grid(Geo::Hq, Tokens, c.q.ne[3]);
            context_window_reduce_kernel<Geo, Tokens, KeyBlock><<<reduce_grid, 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(c.context_lengths.data),
                static_cast<const std::int32_t*>(c.valid_columns.data),
                static_cast<int>(c.context.block_tables.ne[0] * kPagedKVPageSize),
                plan.split_capacity, window, static_cast<__nv_bfloat16*>(out.data));
            CUDA_CHECK(cudaGetLastError());
        });
    });
}

} // namespace

ContextAttentionPlan context_window_resolve_plan(ContextAttentionProfile profile,
                                                 std::int32_t tokens,
                                                 ContextAttentionExecutionEnvelope envelope,
                                                 std::int32_t window) {
    if (window < 1) { throw std::invalid_argument("context_window_attention plan: window >= 1"); }
    // The plan always splits (the reduce reads the partials): an envelope of at least one entry.
    const std::uint32_t w = static_cast<std::uint32_t>(window);
    return context_attention_resolve_plan(
        profile, tokens,
        {.min_context = std::min(envelope.min_context, w),
         .max_context = std::max(1u, std::min(envelope.max_context, w))});
}

void context_window_launch(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                           const Tensor& context_lengths, const Tensor& valid_columns,
                           const Tensor& table_rows, float scale, std::int32_t window,
                           const PagedKVBatchLayerView& context, const ContextAttentionPlan& plan,
                           Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out,
                           cudaStream_t stream) {
    const ContextCall call{q,       query_k,       query_v, context_lengths,
                           valid_columns, table_rows, context, scale};
    switch (plan.profile) {
    case ContextAttentionProfile::D128Q32K8:
        window_launch_profile<ContextQueryD128>(call, window, plan, partial_acc, partial_m,
                                                partial_l, out, stream);
        return;
    case ContextAttentionProfile::D64Q64K8:
        window_launch_profile<ContextQueryD64>(call, window, plan, partial_acc, partial_m,
                                               partial_l, out, stream);
        return;
    }
    throw std::invalid_argument("context_window_attention: inconsistent plan");
}

} // namespace ninferno::ops::detail
