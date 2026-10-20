#pragma once

#include "ninferno/ops/softmax_attention.h"

namespace ninferno::ops::detail {

enum class ContextAttentionRoute {
    Direct,
    SplitKv,
};

// The registered head geometries: [D, Hq, Hkv] = [128, 32, 8] (group 4) and [64, 64, 8] (group 8).
enum class ContextAttentionProfile {
    D128Q32K8,
    D64Q64K8,
};

struct ContextAttentionPlan {
    ContextAttentionProfile profile = ContextAttentionProfile::D128Q32K8;
    ContextAttentionRoute route     = ContextAttentionRoute::SplitKv;
    std::int32_t tokens             = 0;
    std::int32_t warps              = 0;
    std::int32_t key_block          = 0;
    std::int32_t split_capacity     = 0;
};

// The registered profile of `geometry`; throws for any other.
[[nodiscard]] ContextAttentionProfile context_attention_profile(AttentionHeadGeometry geometry);

[[nodiscard]] ContextAttentionPlan
context_attention_resolve_plan(ContextAttentionProfile profile, std::int32_t tokens,
                               ContextAttentionExecutionEnvelope envelope);

void context_attention_launch(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                              const Tensor& context_lengths, const Tensor& valid_columns,
                              const Tensor& table_rows, float scale,
                              const PagedKVBatchLayerView& context,
                              const ContextAttentionPlan& plan, Tensor& partial_acc,
                              Tensor& partial_m, Tensor& partial_l, Tensor& out,
                              cudaStream_t stream);

} // namespace ninferno::ops::detail
