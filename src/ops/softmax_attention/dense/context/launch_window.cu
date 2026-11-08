#include "ops/softmax_attention/dense/context/launch.h"
#include "core/per_device.h"

#include "core/device.h"
#include "ops/softmax_attention/dense/context/fold.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninferno::ops::detail {
namespace {

template <class Geo, class Launch, int... Offsets>
void window_dispatch(std::int32_t tokens, Launch&& launch, std::integer_sequence<int, Offsets...>) {
    const bool found =
        ((tokens == Offsets + 1
              ? (launch.template operator()<Offsets + 1, kContextQueryWarps<Geo, Offsets + 1>>(),
                 true)
              : false) ||
         ...);
    if (!found) { throw std::invalid_argument("context_window_attention: unsupported T"); }
}

template <class Geo>
void window_launch_profile(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                           const Tensor& context_lengths, const Tensor& valid_columns,
                           const Tensor& table_rows, float scale, std::int32_t window,
                           const PagedKVBatchLayerView& context, const ContextAttentionPlan& plan,
                           Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out,
                           cudaStream_t stream) {
    window_dispatch<Geo>(
        q.ne[2],
        [&]<int Tokens, int Warps>() {
            if (plan.warps != Warps || plan.route != ContextAttentionRoute::SplitKv ||
                plan.split_capacity < 1) {
                throw std::invalid_argument("context_window_attention: inconsistent plan");
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
                        &context_window_partial_kernel<Geo, Tokens, Warps, KeyBlock, std::int8_t,
                                                       std::int8_t>,
                        SmemBytes + kContextQueryInt8StageBytes<Geo::D, KeyBlock>);
                } else {
                    run.template operator()<__nv_bfloat16, __half>(
                        &context_window_partial_kernel<Geo, Tokens, Warps, KeyBlock>, SmemBytes);
                }
                const dim3 reduce_grid(Geo::Hq, Tokens, q.ne[3]);
                context_window_reduce_kernel<Geo, Tokens, KeyBlock>
                    <<<reduce_grid, 128, 0, stream>>>(
                        static_cast<const __nv_bfloat16*>(partial_acc.data),
                        static_cast<const float*>(partial_m.data),
                        static_cast<const float*>(partial_l.data),
                        static_cast<const std::int32_t*>(context_lengths.data),
                        static_cast<const std::int32_t*>(valid_columns.data),
                        static_cast<int>(context.block_tables.ne[0] * kPagedKVPageSize),
                        plan.split_capacity, window, static_cast<__nv_bfloat16*>(out.data));
                CUDA_CHECK(cudaGetLastError());
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
            throw std::invalid_argument("context_window_attention: inconsistent plan");
        },
        std::make_integer_sequence<int, 16>{});
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
    switch (plan.profile) {
    case ContextAttentionProfile::D128Q32K8:
        window_launch_profile<ContextQueryD128>(q, query_k, query_v, context_lengths, valid_columns,
                                                table_rows, scale, window, context, plan,
                                                partial_acc, partial_m, partial_l, out, stream);
        return;
    case ContextAttentionProfile::D64Q64K8:
        window_launch_profile<ContextQueryD64>(q, query_k, query_v, context_lengths, valid_columns,
                                               table_rows, scale, window, context, plan,
                                               partial_acc, partial_m, partial_l, out, stream);
        return;
    }
    throw std::invalid_argument("context_window_attention: inconsistent plan");
}

} // namespace ninferno::ops::detail
