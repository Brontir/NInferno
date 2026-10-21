#include "ops/softmax_attention/dense/causal_cache/bf16/launch.h"
#include "ops/softmax_attention/common/causal_column_chunks.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/template_launch.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/plan.h"
#include "ops/kv_cache/append/launch.h"

namespace ninferno::ops::detail {
namespace {

template <class G, Bf16KvInstance Instance, bool Writable, class Input>
void grouped(const CausalAttentionOperands& p, Bf16KvCacheView<Writable> cache, Input input,
             const Bf16KvCausalPlan& plan, CausalPartialView partial, cudaStream_t stream) {
    using S           = typename Bf16KvInstanceTraits<Instance>::Schedule;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_bf16_kv_grouped_mma<G, S, MultiBatch, Masked>(p, cache, input, plan.partition,
                                                             partial, stream);
        launch_bf16_kv_merge<G, Bf16KvMergeSchedule<256>, MultiBatch, Masked>(
            p, cache, plan.partition, partial, stream);
    };
    if (p.batch == 1) {
        if (cache.valid_columns)
            invoke.template operator()<false, true>();
        else
            invoke.template operator()<false, false>();
    } else {
        if (cache.valid_columns)
            invoke.template operator()<true, true>();
        else
            invoke.template operator()<true, false>();
    }
}

template <class G, bool Writable, class Input>
void grouped_instance(const CausalAttentionOperands& p, Bf16KvCacheView<Writable> cache,
                      Input input, const Bf16KvCausalPlan& plan, CausalPartialView partial,
                      cudaStream_t stream) {
    switch (plan.instance) {
    case Bf16KvInstance::GroupedDecode:
        return grouped<G, Bf16KvInstance::GroupedDecode>(p, cache, input, plan, partial, stream);
    case Bf16KvInstance::Grouped32:
        return grouped<G, Bf16KvInstance::Grouped32>(p, cache, input, plan, partial, stream);
    case Bf16KvInstance::Grouped64:
        return grouped<G, Bf16KvInstance::Grouped64>(p, cache, input, plan, partial, stream);
    default:
        throw std::logic_error("BF16 grouped launch received a tiled plan");
    }
}

void tiled(const CausalAttentionOperands& p, Bf16KvReadView cache, const Bf16KvCausalPlan& plan,
           cudaStream_t stream) {
    const auto invoke = [&]<class G>() {
        if (plan.instance == Bf16KvInstance::Tiled128)
            launch_bf16_kv_tiled_mma<
                G, typename Bf16KvInstanceTraits<Bf16KvInstance::Tiled128>::Schedule>(p, cache,
                                                                                      stream);
        else
            launch_bf16_kv_tiled_mma<
                G, typename Bf16KvInstanceTraits<Bf16KvInstance::Tiled64>::Schedule>(p, cache,
                                                                                     stream);
    };
    with_causal_geometry(p.query_heads, p.kv_heads, invoke);
}

template <class Input>
void execute_grouped(const Tensor& q, Input input, const Tensor& positions, float scale,
                     PagedKVBatchLayerView cache, const Tensor* valid, const Tensor* rows,
                     const Bf16KvCausalPlan& plan, WorkspaceArena& workspace, Tensor& out,
                     cudaStream_t stream, const __nv_bfloat16* gate = nullptr) {
    auto scope      = workspace.scope();
    auto storage    = allocate_causal_partials(workspace, plan.query_heads, plan.launch_columns,
                                               plan.partition.capacity, plan.batch);
    auto p          = make_causal_operands(q, positions, out, scale, plan.envelope.max_visible_keys,
                                           cache.num_kv_heads);
    p.gate          = gate;
    const auto view = bf16_kv_cache_view<Input::writes_cache>(cache, valid, rows);
    const auto partial = storage.view();
    // A chunked call reads its keys from the cache (bf16_kv_append_attention appends them first).
    if constexpr (Input::writes_cache) {
        if (plan.launch_columns < plan.width)
            throw std::logic_error("BF16 chunked attention appends its keys before the chunks");
    }
    for_each_causal_column_chunk(
        p, view.valid_columns, plan.launch_columns, workspace, stream,
        [&](const CausalAttentionOperands& part, const std::int32_t* valid_columns) {
            auto chunk_view          = view;
            chunk_view.valid_columns = valid_columns;
            with_causal_geometry(p.query_heads, p.kv_heads, [&]<class G>() {
                grouped_instance<G>(part, chunk_view, input, plan, partial, stream);
            });
        });
}
} // namespace

bool bf16_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid_columns,
                              const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, DeviceExecutionView execution, const Tensor* gate) {
    const cudaStream_t stream = execution.stream;
    const auto plan = make_bf16_kv_causal_plan(q.ne[1], cache.num_kv_heads, q.ne[2], q.ne[3],
                                               envelope, execution.multiprocessor_count);
    if (!plan.grouped()) {
        kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
        tiled(make_causal_operands(q, positions, out, scale, envelope.max_visible_keys,
                                   cache.num_kv_heads),
              bf16_kv_cache_view<false>(cache, &valid_columns, &table_rows), plan, stream);
        return false;
    }
    if (plan.launch_columns < plan.width) {
        kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
        execute_grouped(q, CausalCachedInput{}, positions, scale, cache, &valid_columns,
                        &table_rows, plan, workspace, out, stream,
                        gate ? static_cast<const __nv_bfloat16*>(gate->data) : nullptr);
        return true;
    }
    execute_grouped(q,
                    CausalAppendInput{static_cast<const __nv_bfloat16*>(k.data),
                                      static_cast<const __nv_bfloat16*>(v.data)},
                    positions, scale, cache, &valid_columns, &table_rows, plan, workspace, out,
                    stream, gate ? static_cast<const __nv_bfloat16*>(gate->data) : nullptr);
    return true;
}

void bf16_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                              const PagedKVLayerView& cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const auto plan = make_bf16_kv_causal_plan(q.ne[1], cache.num_kv_heads, q.ne[2], 1, envelope,
                                               execution.multiprocessor_count);
    const auto view = single_row_paged_kv_batch_view(cache);
    if (!plan.grouped()) {
        tiled(make_causal_operands(q, positions, out, scale, envelope.max_visible_keys,
                                   cache.num_kv_heads),
              bf16_kv_cache_view<false>(view), plan, stream);
    } else {
        execute_grouped(q, CausalCachedInput{}, positions, scale, view, nullptr, nullptr, plan,
                        workspace, out, stream);
    }
}
} // namespace ninferno::ops::detail
