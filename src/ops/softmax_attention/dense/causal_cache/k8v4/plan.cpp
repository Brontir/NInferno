#include "ops/softmax_attention/dense/causal_cache/k8v4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/operands.h"
#include "ops/softmax_attention/dense/causal_cache/plan_rules.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/softmax_attention/common/mxfp8_tiled_plan.h"
#include <algorithm>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {
constexpr int kGroupedPrefillMaxWidth = 80;
} // namespace

K8V4KvCausalPlan make_k8v4_kv_causal_plan(int heads, int kv_heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope, int row_keys,
 int multiprocessor_count) {
    require_causal_plan_inputs("K8V4", heads, kv_heads, width, batch, envelope);
    if (multiprocessor_count <= 0)
        throw std::invalid_argument("K8V4 attention: invalid SM count");
    constexpr int grouped_limit = K8V4KvCausalPlan::kTokenTile;
    const bool invariant        = invariant_routes_active();
    const int launch_columns    = causal_launch_columns(width, invariant);
    // Under the invariant route profile (prefill_routes.h) every width takes the parallel token
    // tiles, a call wider than one launch in chunks of it: one kernel instance, so a column's
    // arithmetic does not follow the call's width or the rows beside it; splits aligned to keys.
    if (invariant)
        return {K8V4KvFamily::ParallelGrouped,
                heads,
                kv_heads,
                width,
                batch,
                grouped_limit,
                envelope,
                CausalKvPartition::aligned_to_row(row_keys, envelope.max_visible_keys),
                launch_columns};
    const auto family = width <= grouped_limit             ? K8V4KvFamily::Grouped
                        : width <= kGroupedPrefillMaxWidth ? K8V4KvFamily::ParallelGrouped
                                                           : K8V4KvFamily::Tiled;
    if (family == K8V4KvFamily::Tiled)
        return {family,        heads,
                kv_heads,      width,
                batch,         0,
                envelope,      mxfp8_tiled_partition(heads, width, envelope.max_visible_keys, multiprocessor_count),
                launch_columns};
    const int tiles =
        family == K8V4KvFamily::ParallelGrouped ? (width + grouped_limit - 1) / grouped_limit : 1;
    const int query_tile        = family == K8V4KvFamily::ParallelGrouped && width <= 16
                                      ? (width + 1) / 2
                                      : std::min(width, grouped_limit);
    const int independent_tiles = batch * kv_heads * tiles;
    // Decode keeps two CTAs resident per SM; the speculative widths keep one.
    const bool two_ctas_per_sm = width == 1;
    const auto partition =
        causal_wave_partition(independent_tiles, two_ctas_per_sm, envelope.max_visible_keys, multiprocessor_count);
    return {family, heads, kv_heads, width, batch, query_tile, envelope, partition, launch_columns};
}

std::size_t k8v4_kv_workspace_bytes(int heads, int kv_heads, int batch, int min_width,
                                    int max_width, CausalAttentionExecutionEnvelope envelope, int multiprocessor_count) {
    std::size_t maximum  = 0;
    const bool invariant = invariant_routes_active();
    const auto widths =
        causal_grouped_workspace_widths(min_width, max_width, kGroupedPrefillMaxWidth, invariant);
    for (int width = widths.first; width <= widths.last; ++width) {
        const auto plan = make_k8v4_kv_causal_plan(heads, kv_heads, width, batch, envelope, 0, multiprocessor_count);
        if (plan.family == K8V4KvFamily::Tiled) continue;
        const int splits = plan.partition.capacity;
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, splits, batch);
        if (widths.chunks) allocate_causal_chunk_metadata(layout);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    if (invariant) return maximum;
    return std::max(maximum, mxfp8_tiled_workspace_bytes(
                                 heads, std::max(min_width, kGroupedPrefillMaxWidth + 1), max_width,
                                 envelope.max_visible_keys, multiprocessor_count));
}

} // namespace ninferno::ops::detail
