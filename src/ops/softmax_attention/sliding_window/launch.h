#pragma once

#include "ninferno/ops/sliding_window_attention.h"

namespace ninferno::ops::detail {

inline constexpr std::int32_t kSlidingWindowMaxSplits = 32;

enum class SlidingWindowAttentionRoute {
    Direct,
    SplitKv,
};

// The key tile: 32 rows, or 64 for D64 above 8 query columns, whose CTA holds Tokens x 8 query rows
// (up to 128) and stages them over the two operand tiles (context_query.cuh: Br <= 2 x KeyBlock).
[[nodiscard]] constexpr int sliding_window_key_block(int head_dim, int tokens) {
    return head_dim == 64 && tokens > 8 ? 64 : 32;
}

// The route (direct or split) depends on the window, the query width, the batch and the envelope,
// not on the head geometry: the profiles of one window share their graph topology classes. The
// key tile and so the split count follow the geometry.
struct SlidingWindowAttentionPlan {
    SlidingWindowAttentionRoute route;
    std::int32_t tokens;
    std::int32_t key_block;
    std::int32_t reduce_warps;
    std::int32_t split_capacity;
    std::int32_t max_context;
    std::int32_t window;
};

[[nodiscard]] SlidingWindowAttentionPlan
sliding_window_attention_resolve_plan(std::uint32_t window, std::int32_t tokens, std::int32_t batch,
                                      SlidingWindowAttentionExecutionEnvelope envelope,
                                      AttentionHeadGeometry geometry = {128, 32, 8});
[[nodiscard]] const char* sliding_window_attention_route_name(SlidingWindowAttentionRoute route);

void sliding_window_attention_launch(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                                     const Tensor& positions, const Tensor& valid_columns,
                                     const Tensor& lanes, AttentionHeadGeometry geometry,
                                     float scale, const CyclicKVCacheLayerView& context,
                                     const SlidingWindowAttentionPlan& plan, Tensor& partial_acc,
                                     Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                     cudaStream_t stream);

} // namespace ninferno::ops::detail
