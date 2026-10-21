#include "ops/softmax_attention/dense/causal_cache/nvfp4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/operands.h"
#include "ops/softmax_attention/dense/causal_cache/plan_rules.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/common/native_nvfp4.h"
#include <algorithm>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {
constexpr int kGroupedPrefillMaxWidth = 192;
// The warp-specialized tiled kernel (setmaxnreg) is built with the native NVFP4 family only;
// elsewhere the parallel grouped tiles serve every width past the grouped token tile.
constexpr bool kTiledKernel = kNativeNvfp4;
} // namespace

Nvfp4KvCausalPlan make_nvfp4_kv_causal_plan(int heads, int kv_heads, int width, int batch,
                                            CausalAttentionExecutionEnvelope envelope,
                                            int multiprocessor_count) {
    require_causal_plan_inputs("NVFP4", heads, kv_heads, width, batch, envelope);
    if (multiprocessor_count <= 0)
        throw std::invalid_argument("NVFP4 attention: invalid SM count");
    constexpr int grouped_limit = Nvfp4KvCausalPlan::kTokenTile;
    const bool invariant        = invariant_routes_active();
    const int launch_columns    = causal_launch_columns(width, invariant);
    // Under the invariant route profile (prefill_routes.h) every width takes the parallel token
    // tiles, a call wider than one launch in chunks of it: one kernel instance, so a column's
    // arithmetic does not follow the call's width or the rows beside it; splits aligned to keys.
    if (invariant)
        return {Nvfp4KvFamily::ParallelGrouped,
                heads,
                kv_heads,
                width,
                batch,
                grouped_limit,
                envelope,
                CausalKvPartition::aligned_to_positions(envelope.max_visible_keys),
                launch_columns};
    const auto family           = width <= grouped_limit ? Nvfp4KvFamily::Grouped
                                  : width <= kGroupedPrefillMaxWidth || !kTiledKernel
                                      ? Nvfp4KvFamily::ParallelGrouped
                                      : Nvfp4KvFamily::Tiled;
    const int tiles =
        family == Nvfp4KvFamily::ParallelGrouped ? (width + grouped_limit - 1) / grouped_limit : 1;
    const int independent_tiles = batch * kv_heads * tiles;
    const int query_tile        = family == Nvfp4KvFamily::ParallelGrouped && width <= 16
                                      ? (width + 1) / 2
                                      : std::min(width, grouped_limit);
    const int row_tiles         = (query_tile * (heads / kv_heads) + 15) / 16;
    const bool two_ctas_per_sm  = row_tiles <= 2;
    const auto partition =
        causal_wave_partition(independent_tiles, two_ctas_per_sm, envelope.max_visible_keys, multiprocessor_count);
    return {family, heads, kv_heads, width, batch, query_tile, envelope, partition, launch_columns};
}

std::size_t nvfp4_kv_workspace_bytes(int heads, int kv_heads, int batch, int min_width,
                                     int max_width, CausalAttentionExecutionEnvelope envelope, int multiprocessor_count) {
    std::size_t maximum = 0;
    const auto widths   = causal_grouped_workspace_widths(
        min_width, max_width, kTiledKernel ? kGroupedPrefillMaxWidth : max_width,
        invariant_routes_active());
    for (int width = widths.first; width <= widths.last; ++width) {
        const auto plan = make_nvfp4_kv_causal_plan(heads, kv_heads, width, batch, envelope, multiprocessor_count);
        if (plan.family == Nvfp4KvFamily::Tiled) continue;
        const int splits = plan.partition.capacity;
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, splits, batch);
        if (widths.chunks) allocate_causal_chunk_metadata(layout);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return maximum;
}

} // namespace ninferno::ops::detail
