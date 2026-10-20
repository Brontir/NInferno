#include "ops/softmax_attention/dense/causal_cache/int8/launch.h"
#include "core/per_device.h"
#include "ops/softmax_attention/common/causal_column_chunks.h"
#include "ops/softmax_attention/dense/causal_cache/int8/instances.h"
#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/template_launch.cuh"
#include "ops/softmax_attention/dense/causal_cache/int8/key_major_kt.cuh"
#include "ops/softmax_attention/dense/causal_cache/int8/volta_splitd.h"
#include "ops/kv_cache/append/launch.h"
#include "ninferno/ops/prefill_routes.h"
#include "core/device_class.h"
#include "core/measured_boards.h"

#include <cstdlib>

namespace ninferno::ops::detail {
namespace {

// The twelve-warp schedule for three row tiles, by the nearest measured board. DFlash2 decode at a
// 24K window, six passes: A100 127.1 -> 131.8 tok/s, RTX 4090 154.1 -> 156.9, A10 69.2 -> 70.1;
// short windows within noise. The RTX 3090 loses at depth (128.3 -> 125.5), as in ninfer-all's own
// profile, and keeps six warps.
constexpr measured::Gate kWideGroupedPays[] = {{measured::kA100Sxm40, true},
                                               {measured::kRtx4090, true},
                                               {measured::kA10, true},
                                               {measured::kRtx3090, false}};

bool wide_grouped_takes() {
    return !invariant_routes_active() && measured::measured_gate(kWideGroupedPays);
}

template <class G, int Tokens, class Input, bool Writable>
void grouped(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache, Input input,
             CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    using Instance    = Int8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        if constexpr (Instance::kHasWide) {
            if (wide_grouped_takes()) {
                launch_int8_kv_grouped_mma<G, typename Instance::WideSchedule, MultiBatch, Masked>(
                    p, cache, input, partition, partial, stream);
                return launch_causal_natural_merge<G, typename Instance::Merge, MultiBatch, Masked,
                                                   false>(p, cache.valid_columns, partition,
                                                          partial, stream);
            }
        }
        launch_int8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked>(
            p, cache, input, partition, partial, stream);
        launch_causal_natural_merge<G, typename Instance::Merge, MultiBatch, Masked, false>(
            p, cache.valid_columns, partition, partial, stream);
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

template <class G, class Input, bool Writable>
void grouped_instance(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache,
                      Input input, CausalKvPartition partition, CausalPartialView partial,
                      cudaStream_t stream) {
    switch (p.width) {
#define NINFERNO_INT8_GROUPED(T)                                                                   \
    case T:                                                                                        \
        return grouped<G, T>(p, cache, input, partition, partial, stream)
        NINFERNO_INT8_GROUPED(1);
        NINFERNO_INT8_GROUPED(2);
        NINFERNO_INT8_GROUPED(3);
        NINFERNO_INT8_GROUPED(4);
        NINFERNO_INT8_GROUPED(5);
        NINFERNO_INT8_GROUPED(6);
        NINFERNO_INT8_GROUPED(7);
        NINFERNO_INT8_GROUPED(8);
#undef NINFERNO_INT8_GROUPED
    }
    throw std::logic_error("INT8 grouped plan exceeds the selected token tile");
}

template <class Input>
void execute_grouped(const Tensor& q, const Tensor& positions, float scale,
                     PagedKVBatchLayerView cache, const Tensor* valid, const Tensor* rows,
                     Input input, const Int8KvCausalPlan& plan, WorkspaceArena& workspace,
                     Tensor& out, cudaStream_t stream, const __nv_bfloat16* gate = nullptr) {
    const auto view =
        make_quantized_causal_cache_view<Int8KvCacheView<Input::writes_cache>>(cache, valid, rows);
    auto scope         = workspace.scope();
    const auto partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                                  plan.partition.capacity, plan.batch);
    auto p = make_causal_operands(q, positions, out, scale, plan.envelope.max_visible_keys,
                                  cache.num_kv_heads);
    p.gate = gate;
    with_causal_geometry(plan.query_heads, plan.kv_heads, [&]<class G>() {
        grouped_instance<G>(p, view, input, plan.partition, partial.view(), stream);
    });
}

template <class G, int Tokens>
void parallel_grouped(const CausalAttentionOperands& p, Int8KvReadView cache,
                      CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    using Instance    = Int8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_int8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked, false,
                                   CausalCachedInput, true>(p, cache, {}, partition, partial,
                                                            stream);
        launch_causal_natural_merge<G, typename Instance::Merge, MultiBatch, Masked, false>(
            p, cache.valid_columns, partition, partial, stream);
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

void execute_parallel(const CausalAttentionOperands& p, Int8KvReadView cache,
                      const Int8KvCausalPlan& plan, WorkspaceArena& workspace,
                      cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto partial = allocate_causal_partials(workspace, plan.query_heads, plan.launch_columns,
                                                  plan.partition.capacity, plan.batch);
    for_each_causal_column_chunk(
        p, cache.valid_columns, plan.launch_columns, workspace, stream,
        [&](const CausalAttentionOperands& part, const std::int32_t* valid) {
            auto view          = cache;
            view.valid_columns = valid;
            with_causal_geometry(plan.query_heads, plan.kv_heads, [&]<class G>() {
                parallel_grouped<G, Int8KvCausalPlan::kTokenTile>(part, view, plan.partition,
                                                                  partial.view(), stream);
            });
        });
}

// sm80+: the register-resident prompt kernel (int8/prompt_fast.cuh) takes the tiled family. It
// needs 98 KiB of shared memory and m16n8k32 integer MMA.

void tiled(const CausalAttentionOperands& p, Int8KvReadView cache, int multiprocessors,
           cudaStream_t stream) {
    with_causal_geometry(p.query_heads, p.kv_heads, [&]<class G>() {
#if NINFERNO_BUILD_ARCH >= 80
        if constexpr (G::kHeadDim == kInt8PromptFastD) {
            return launch_int8_kv_prompt_fast<G>(p, cache, multiprocessors, stream);
        }
#endif
        launch_int8_kv_tiled_mma<G, Int8KvTiledInstance>(p, cache, stream);
    });
}

// Volta (kInt8KvKeyMajorHmma): the grouped and parallel-grouped widths on mma.m8n8k4, with the
// grouped family's partitions and partials and the same natural merge.
template <class G, int Tokens, bool ParallelQueries>
void key_major(const CausalAttentionOperands& p, Int8KvReadView cache, CausalKvPartition partition,
               CausalPartialView partial, cudaStream_t stream) {
    using Shape       = Int8KvKeyMajorShape<G, Tokens>;
    using Merge       = typename Int8KvGroupedInstance<G, Tokens>::Merge;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        validate_quantized_causal_operands<G>(p, cache);
        // Up to 32 rows: the transposed kernel (key_major_kt.cuh), 36 % less time per call at 24K.
        if constexpr (!ParallelQueries && kInt8KvKeyMajorKt<G, Tokens>) {
            const dim3 grid(G::KVHeads, partition.capacity, p.batch);
            int8_kv_key_major_kt_kernel<G, Tokens, MultiBatch, Masked><<<grid, 128, 0, stream>>>(
                p.q, p.positions, cache.keys, cache.values, cache.key_scales, cache.value_scales,
                cache.tables, cache.valid_columns, cache.table_rows, cache.table_stride, p.width,
                p.visible_capacity, partition, p.scale, partial.acc, partial.maximum, partial.sum);
            CUDA_CHECK(cudaGetLastError());
            launch_causal_natural_merge<G, Merge, MultiBatch, Masked, false>(
                p, cache.valid_columns, partition, partial, stream);
            return;
        }
        static constexpr auto kernel =
            int8_kv_key_major_hmma_kernel<G, Tokens, MultiBatch, Masked, ParallelQueries>;
        static ::ninferno::PerDevice<bool> status;
        ::ninferno::allow_dynamic_smem(status, kernel, Shape::kSharedBytes);
        const dim3 grid(G::KVHeads * (ParallelQueries ? div_up(p.width, Tokens) : 1),
                        partition.capacity, p.batch);
        kernel<<<grid, Shape::kThreads, Shape::kSharedBytes, stream>>>(
            p.q, p.positions, cache.keys, cache.values, cache.key_scales, cache.value_scales,
            cache.tables, cache.valid_columns, cache.table_rows, cache.table_stride, p.width,
            p.visible_capacity, partition, p.scale, partial.acc, partial.maximum, partial.sum);
        CUDA_CHECK(cudaGetLastError());
        launch_causal_natural_merge<G, Merge, MultiBatch, Masked, false>(
            p, cache.valid_columns, partition, partial, stream);
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

void execute_key_major(const CausalAttentionOperands& p, Int8KvReadView cache,
                       const Int8KvCausalPlan& plan, WorkspaceArena& workspace,
                       cudaStream_t stream) {
    auto scope = workspace.scope();
    // The parallel tiles run in chunks of the plan's launch columns, as execute_parallel does: the
    // workspace query sizes their partials for one chunk.
    if (plan.family == Int8KvFamily::ParallelGrouped) {
        const auto partial =
            allocate_causal_partials(workspace, plan.query_heads, plan.launch_columns,
                                     plan.partition.capacity, plan.batch);
        for_each_causal_column_chunk(
            p, cache.valid_columns, plan.launch_columns, workspace, stream,
            [&](const CausalAttentionOperands& part, const std::int32_t* valid) {
                auto view          = cache;
                view.valid_columns = valid;
                with_causal_geometry(plan.query_heads, plan.kv_heads, [&]<class G>() {
                    key_major<G, Int8KvCausalPlan::kTokenTile, true>(
                        part, view, plan.partition, partial.view(), stream);
                });
            });
        return;
    }
    const auto partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                                  plan.partition.capacity, plan.batch);
    with_causal_geometry(plan.query_heads, plan.kv_heads, [&]<class G>() {
        switch (p.width) {
#define NINFERNO_INT8_KEY_MAJOR(T)                                                                 \
    case T:                                                                                        \
        return key_major<G, T, false>(p, cache, plan.partition, partial.view(), stream)
            NINFERNO_INT8_KEY_MAJOR(1);
            NINFERNO_INT8_KEY_MAJOR(2);
            NINFERNO_INT8_KEY_MAJOR(3);
            NINFERNO_INT8_KEY_MAJOR(4);
            NINFERNO_INT8_KEY_MAJOR(5);
            NINFERNO_INT8_KEY_MAJOR(6);
            NINFERNO_INT8_KEY_MAJOR(7);
            NINFERNO_INT8_KEY_MAJOR(8);
#undef NINFERNO_INT8_KEY_MAJOR
        }
        throw std::logic_error("INT8 key-major plan exceeds the selected token tile");
    });
}

// Volta's key-major kernel, outside the invariant route profile: a column's bytes there follow the
// call's width (softmax_attention shape invariance on the V100), so that profile keeps the
// parallel token tiles every width runs on elsewhere.
bool key_major_takes() { return kInt8KvKeyMajorHmma && !invariant_routes_active(); }

} // namespace

bool int8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid, const Tensor& rows,
                              float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, DeviceExecutionView execution, const Tensor* gate) {
    const cudaStream_t stream = execution.stream;
    const auto* gate_data = gate ? static_cast<const __nv_bfloat16*>(gate->data) : nullptr;
    const auto plan       = make_int8_kv_causal_plan(q.ne[1], cache.num_kv_heads, q.ne[2], q.ne[3],
                                                     envelope, paged_kv_row_keys(cache.block_tables), execution.multiprocessor_count);
    if (plan.family != Int8KvFamily::Grouped || key_major_takes()) {
        kv_cache_append_batch_launch(k, v, positions, valid, rows, cache, stream);
        auto p = make_causal_operands(q, positions, out, scale, envelope.max_visible_keys,
                                      cache.num_kv_heads);
        const auto view =
            make_quantized_causal_cache_view<Int8KvCacheView<false>>(cache, &valid, &rows);
        if (plan.family == Int8KvFamily::Tiled) {
            // Volta: FP16 flash attention on mma.m8n8k4 (the tiled kernel's MMA runs on FMAs).
            if (int8_volta_splitd_takes(q.ne[1], cache.num_kv_heads, q.ne[2], q.ne[3], envelope) &&
                workspace.capacity() - workspace.used() >=
                    int8_volta_splitd_workspace_bytes(q.ne[1], cache.num_kv_heads, q.ne[2],
                                                      envelope.max_visible_keys)) {
                int8_volta_splitd(q, rows, scale, cache, envelope, workspace, out, stream);
            } else {
                tiled(p, view, execution.multiprocessor_count, stream);
            }
            return false;
        }
        p.gate = gate_data;
        if (key_major_takes())
            execute_key_major(p, view, plan, workspace, stream);
        else
            execute_parallel(p, view, plan, workspace, stream);
        return true;
    }
    execute_grouped(q, positions, scale, cache, &valid, &rows,
                    CausalAppendInput{static_cast<const __nv_bfloat16*>(k.data),
                                      static_cast<const __nv_bfloat16*>(v.data)},
                    plan, workspace, out, stream, gate_data);
    return true;
}

void int8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                              const PagedKVLayerView& cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const auto plan = make_int8_kv_causal_plan(q.ne[1], cache.num_kv_heads, q.ne[2], 1, envelope,
                                               paged_kv_row_keys(cache.block_table), execution.multiprocessor_count);
    const auto view = single_row_paged_kv_batch_view(cache);
    if (plan.family == Int8KvFamily::Tiled)
        tiled(make_causal_operands(q, positions, out, scale, envelope.max_visible_keys,
                                   cache.num_kv_heads),
              make_quantized_causal_cache_view<Int8KvCacheView<false>>(view),
              execution.multiprocessor_count, stream);
    else if (key_major_takes())
        execute_key_major(make_causal_operands(q, positions, out, scale, envelope.max_visible_keys,
                                               cache.num_kv_heads),
                          make_quantized_causal_cache_view<Int8KvCacheView<false>>(view), plan,
                          workspace, stream);
    else if (plan.family == Int8KvFamily::ParallelGrouped)
        execute_parallel(make_causal_operands(q, positions, out, scale, envelope.max_visible_keys,
                                              cache.num_kv_heads),
                         make_quantized_causal_cache_view<Int8KvCacheView<false>>(view), plan,
                         workspace, stream);
    else
        execute_grouped(q, positions, scale, view, nullptr, nullptr, CausalCachedInput{}, plan,
                        workspace, out, stream);
}

} // namespace ninferno::ops::detail
