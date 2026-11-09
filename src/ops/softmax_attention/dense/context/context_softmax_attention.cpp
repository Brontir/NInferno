#include "ninferno/ops/softmax_attention.h"

#include "core/layout.h"
#include "ops/common/tensor_checks.h"
#include "ops/softmax_attention/dense/context/launch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninferno::ops {
namespace {

detail::ContextAttentionProfile require_profile(AttentionHeadGeometry geometry, const char* op) {
    if (!valid_attention_head_geometry(geometry)) {
        throw std::invalid_argument(std::string(op) + ": unsupported head geometry");
    }
    try {
        return detail::context_attention_profile(geometry);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument(std::string(op) + ": unsupported head geometry");
    }
}

std::uint32_t validate_context(const PagedKVBatchLayerView& context, AttentionHeadGeometry geometry,
                               const char* op) {
    const std::int32_t head_dim = geometry.head_dim;
    const std::int32_t kv_heads = geometry.kv_heads;
    const bool int8             = context.storage == KvCacheStorage::Int8Group64 && head_dim == 128;
    if ((context.storage != KvCacheStorage::BFloat16 && !int8) ||
        context.num_kv_heads != kv_heads || context.head_dim != head_dim) {
        throw std::invalid_argument(std::string(op) + ": invalid context geometry or dtype");
    }
    const std::int32_t physical_pages = context.k_pages.ne[2];
    if (physical_pages <= 0 || context.v_pages.ne[2] != physical_pages ||
        context.block_tables.ne[0] <= 0 || context.block_tables.ne[1] <= 0) {
        throw std::invalid_argument(std::string(op) + ": invalid context capacity");
    }
    if (int8 ? (context.k_pages.dtype != DType::I8 || context.v_pages.dtype != DType::I8)
             : (context.k_pages.dtype != DType::BF16 || context.v_pages.dtype != DType::FP16)) {
        throw std::invalid_argument(std::string(op) + ": context K/V dtype does not match storage");
    }
    require_shape(context.k_pages, head_dim, kPagedKVPageSize, physical_pages, kv_heads, op,
                  "context k pages");
    require_shape(context.v_pages, head_dim, kPagedKVPageSize, physical_pages, kv_heads, op,
                  "context v pages");
    require_contiguous_nonnull(context.k_pages, op, "context k pages");
    require_contiguous_nonnull(context.v_pages, op, "context v pages");
    require_shape(context.block_tables, context.block_tables.ne[0], context.block_tables.ne[1], 1,
                  1, op, "context block tables");
    if (context.block_tables.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": context block tables must be I32");
    }
    require_contiguous_nonnull(context.block_tables, op, "context block tables");
    if (int8) {
        for (const Tensor* s : {&context.k_scale_pages, &context.v_scale_pages}) {
            if (s->dtype != DType::FP16) {
                throw std::invalid_argument(std::string(op) + ": int8 context scales must be FP16");
            }
            require_shape(*s, head_dim / 64, kPagedKVPageSize, physical_pages, kv_heads, op,
                          "context scale pages");
            require_contiguous_nonnull(*s, op, "context scale pages");
        }
    } else if (context.k_scale_pages.data != nullptr || context.v_scale_pages.data != nullptr) {
        throw std::invalid_argument(std::string(op) + ": BF16 context must not have scales");
    }
    const std::uint64_t logical_capacity =
        static_cast<std::uint64_t>(context.block_tables.ne[0]) * kPagedKVPageSize;
    if (logical_capacity > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(std::string(op) + ": logical context capacity exceeds int32");
    }
    return static_cast<std::uint32_t>(logical_capacity);
}

struct PartialWorkspace {
    Tensor acc;
    Tensor m;
    Tensor l;
};

template <class Allocator>
PartialWorkspace allocate_workspace(Allocator& workspace, AttentionHeadGeometry geometry,
                                    std::int32_t tokens, std::int32_t splits,
                                    std::int32_t batch_size) {
    const std::int32_t d = geometry.head_dim, hq = geometry.query_heads;
    return {
        workspace.alloc(DType::BF16, {d, hq, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {hq, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {hq, tokens, splits * batch_size}),
    };
}

detail::ContextAttentionProfile validate_call(const Tensor& q, const Tensor& query_k,
                                              const Tensor& query_v, const Tensor& context_lengths,
                                              const Tensor& valid_columns, const Tensor& table_rows,
                                              AttentionHeadGeometry geometry, float scale,
                                              const PagedKVBatchLayerView& context,
                                              ContextAttentionExecutionEnvelope envelope,
                                              const Tensor& out) {
    constexpr const char* op    = "context_softmax_attention";
    const auto profile          = require_profile(geometry, op);
    const std::int32_t head_dim = geometry.head_dim;
    const std::int32_t q_heads  = geometry.query_heads;
    const std::int32_t kv_heads = geometry.kv_heads;
    if (q.dtype != DType::BF16 || query_k.dtype != DType::BF16 || query_v.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument("context_softmax_attention: q/k/v/out must be BF16");
    }
    if (context_lengths.dtype != DType::I32 || valid_columns.dtype != DType::I32 ||
        table_rows.dtype != DType::I32) {
        throw std::invalid_argument(
            "context_softmax_attention: lengths/valid_columns/table_rows must be I32");
    }
    const std::int32_t tokens = q.ne[2];
    const std::int32_t batch  = q.ne[3];
    if (tokens < 1 || tokens > 16) {
        throw std::invalid_argument("context_softmax_attention: optimized domain is T=1..16");
    }
    if (batch < 1 || batch > 8) {
        throw std::invalid_argument("context_softmax_attention: B must be 1..8");
    }
    require_shape(q, head_dim, q_heads, tokens, batch, op, "q");
    require_shape(query_k, head_dim, kv_heads, tokens, batch, op, "query k");
    require_shape(query_v, head_dim, kv_heads, tokens, batch, op, "query v");
    require_shape(context_lengths, batch, 1, 1, 1, op, "context lengths");
    require_shape(valid_columns, batch, 1, 1, 1, op, "valid columns");
    require_shape(table_rows, batch, 1, 1, 1, op, "table rows");
    require_shape(out, head_dim, q_heads, tokens, batch, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(query_k, op, "query k");
    require_contiguous_nonnull(query_v, op, "query v");
    require_contiguous_nonnull(context_lengths, op, "context lengths");
    require_contiguous_nonnull(valid_columns, op, "valid columns");
    require_contiguous_nonnull(table_rows, op, "table rows");
    require_contiguous_nonnull(out, op, "out");
    const std::uint32_t logical_capacity = validate_context(context, geometry, op);
    if (envelope.min_context > envelope.max_context || envelope.max_context > logical_capacity) {
        throw std::invalid_argument("context_softmax_attention: invalid execution envelope");
    }
    const float expected_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    if (!std::isfinite(scale) || std::abs(scale - expected_scale) > 1e-7f) {
        throw std::invalid_argument("context_softmax_attention: scale must be 1/sqrt(head_dim)");
    }

    return profile;
}

} // namespace

std::size_t context_softmax_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, ContextAttentionExecutionEnvelope envelope,
    std::int32_t min_tokens, std::int32_t max_tokens, std::int32_t batch_size) {
    const auto profile = require_profile(geometry, "context_softmax_attention workspace");
    if (min_tokens < 1 || max_tokens < min_tokens || max_tokens > 16 || batch_size < 1 ||
        batch_size > 8 || envelope.min_context > envelope.max_context ||
        envelope.max_context >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(
            "context_softmax_attention workspace: invalid envelope or token interval");
    }
    const auto endpoint_capacity = [&](std::int32_t tokens) {
        const auto plan = detail::context_attention_resolve_plan(profile, tokens, envelope);
        WorkspaceLayoutBuilder layout;
        (void)allocate_workspace(layout, geometry, tokens, plan.split_capacity, batch_size);
        return layout.peak_bytes(1);
    };

    std::size_t maximum = 0;
    if (min_tokens <= 8) { maximum = endpoint_capacity(std::min(max_tokens, 8)); }
    if (max_tokens >= 9) { maximum = std::max(maximum, endpoint_capacity(max_tokens)); }
    return maximum;
}

void context_softmax_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                               const Tensor& context_lengths, const Tensor& valid_columns,
                               const Tensor& table_rows, AttentionHeadGeometry geometry,
                               float scale, const PagedKVBatchLayerView& context,
                               ContextAttentionExecutionEnvelope envelope,
                               WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    const auto profile = validate_call(q, query_k, query_v, context_lengths, valid_columns,
                                       table_rows, geometry, scale, context, envelope, out);
    const std::int32_t tokens = q.ne[2];
    const std::int32_t batch  = q.ne[3];

    auto scope               = workspace.scope();
    const auto plan = detail::context_attention_resolve_plan(profile, tokens, envelope);
    PartialWorkspace partial =
        allocate_workspace(workspace, geometry, tokens, plan.split_capacity, batch);
    detail::context_attention_launch(q, query_k, query_v, context_lengths, valid_columns,
                                     table_rows, scale, context, plan, partial.acc, partial.m,
                                     partial.l, out, stream);
}

std::size_t context_window_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, ContextAttentionExecutionEnvelope envelope, std::int32_t window,
    std::int32_t min_tokens, std::int32_t max_tokens, std::int32_t batch_size) {
    const auto profile = require_profile(geometry, "context_window_attention workspace");
    if (min_tokens < 1 || max_tokens < min_tokens || max_tokens > 16 || batch_size < 1 ||
        batch_size > 8 || window < 1 || envelope.min_context > envelope.max_context ||
        envelope.max_context >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(
            "context_window_attention workspace: invalid envelope, window or token interval");
    }
    const auto endpoint_capacity = [&](std::int32_t tokens) {
        const auto plan = detail::context_window_resolve_plan(profile, tokens, envelope, window);
        WorkspaceLayoutBuilder layout;
        (void)allocate_workspace(layout, geometry, tokens, plan.split_capacity, batch_size);
        return layout.peak_bytes(1);
    };
    std::size_t maximum = 0;
    if (min_tokens <= 8) { maximum = endpoint_capacity(std::min(max_tokens, 8)); }
    if (max_tokens >= 9) { maximum = std::max(maximum, endpoint_capacity(max_tokens)); }
    return maximum;
}

void context_window_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                              const Tensor& context_lengths, const Tensor& valid_columns,
                              const Tensor& table_rows, AttentionHeadGeometry geometry, float scale,
                              std::int32_t window, const PagedKVBatchLayerView& context,
                              ContextAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, cudaStream_t stream) {
    const auto profile = validate_call(q, query_k, query_v, context_lengths, valid_columns,
                                       table_rows, geometry, scale, context, envelope, out);
    if (window < 1) {
        throw std::invalid_argument("context_window_attention: window must be positive");
    }
    const std::int32_t tokens = q.ne[2];
    const std::int32_t batch  = q.ne[3];
    auto scope                = workspace.scope();
    const auto plan = detail::context_window_resolve_plan(profile, tokens, envelope, window);
    PartialWorkspace partial =
        allocate_workspace(workspace, geometry, tokens, plan.split_capacity, batch);
    detail::context_window_launch(q, query_k, query_v, context_lengths, valid_columns, table_rows,
                                  scale, window, context, plan, partial.acc, partial.m, partial.l,
                                  out, stream);
}

std::size_t context_fold_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, ContextAttentionExecutionEnvelope envelope, std::int32_t window,
    std::int32_t min_tokens, std::int32_t max_tokens, std::int32_t batch_size) {
    const auto profile = require_profile(geometry, "context_fold_attention workspace");
    if (min_tokens < 1 || max_tokens < min_tokens || max_tokens > 16 || batch_size < 1 ||
        batch_size > 8 || window < 1 || envelope.min_context > envelope.max_context ||
        envelope.max_context >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(
            "context_fold_attention workspace: invalid envelope, window or token interval");
    }
    const auto endpoint_capacity = [&](std::int32_t tokens) {
        const auto plan = detail::context_fold_resolve_plan(profile, tokens, envelope, window);
        WorkspaceLayoutBuilder layout;
        (void)allocate_workspace(layout, geometry, tokens, plan.window.split_capacity, batch_size);
        (void)allocate_workspace(layout, geometry, tokens, plan.far.split_capacity, batch_size);
        (void)layout.alloc(DType::BF16, {geometry.head_dim, geometry.query_heads, tokens, batch_size});
        return layout.peak_bytes(1);
    };
    std::size_t maximum = 0;
    if (min_tokens <= 8) { maximum = endpoint_capacity(std::min(max_tokens, 8)); }
    if (max_tokens >= 9) { maximum = std::max(maximum, endpoint_capacity(max_tokens)); }
    return maximum;
}

void context_fold_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                            const Tensor& context_lengths, const Tensor& valid_columns,
                            const Tensor& table_rows, AttentionHeadGeometry geometry, float scale,
                            ContextFold fold, const PagedKVBatchLayerView& context,
                            ContextAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& out, cudaStream_t stream) {
    const auto profile = validate_call(q, query_k, query_v, context_lengths, valid_columns,
                                       table_rows, geometry, scale, context, envelope, out);
    if (fold.window < 1 || !(fold.alpha >= 1.0F) || !std::isfinite(fold.alpha)) {
        throw std::invalid_argument("context_fold_attention: window must be positive, alpha >= 1");
    }
    const std::int32_t tokens = q.ne[2];
    const std::int32_t batch  = q.ne[3];
    auto scope                = workspace.scope();
    const auto plan = detail::context_fold_resolve_plan(profile, tokens, envelope, fold.window);
    const PartialWorkspace w =
        allocate_workspace(workspace, geometry, tokens, plan.window.split_capacity, batch);
    const PartialWorkspace f =
        allocate_workspace(workspace, geometry, tokens, plan.far.split_capacity, batch);
    Tensor q_far = workspace.alloc(DType::BF16, {geometry.head_dim, geometry.query_heads, tokens, batch});
    detail::FoldPartials window{w.acc, w.m, w.l}, far{f.acc, f.m, f.l};
    detail::context_fold_launch(q, query_k, query_v, context_lengths, valid_columns, table_rows,
                                scale, fold, context, plan, window, far, q_far, out, stream);
}

} // namespace ninferno::ops
