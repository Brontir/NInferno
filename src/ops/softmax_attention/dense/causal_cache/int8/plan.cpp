#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/operands.h"
#include "ops/softmax_attention/dense/causal_cache/int8/volta_splitd.h"
#include "ops/softmax_attention/dense/causal_cache/plan_rules.h"
#include "ninferno/ops/prefill_routes.h"
#include <algorithm>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {
constexpr int kGroupedPrefillMaxWidth = 256;
} // namespace

Int8KvCausalPlan make_int8_kv_causal_plan(int heads, int kv_heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope,
                                          int multiprocessor_count) {
    require_causal_plan_inputs("INT8", heads, kv_heads, width, batch, envelope);
    if (multiprocessor_count <= 0) throw std::invalid_argument("INT8 attention: invalid SM count");
    constexpr int grouped_limit = Int8KvCausalPlan::kTokenTile;
    const bool invariant        = invariant_routes_active();
    const int launch_columns    = causal_launch_columns(width, invariant);
    // Under the invariant route profile (prefill_routes.h) every width takes the parallel token
    // tiles, a call wider than one launch in chunks of it: one kernel instance, so a column's
    // arithmetic does not follow the call's width or the rows beside it; splits aligned to keys.
    if (invariant)
        return {Int8KvFamily::ParallelGrouped,
                heads,
                kv_heads,
                width,
                batch,
                envelope,
                CausalKvPartition::aligned_to_positions(envelope.max_visible_keys),
                launch_columns};
    const auto family = width <= grouped_limit             ? Int8KvFamily::Grouped
                        : width <= kGroupedPrefillMaxWidth ? Int8KvFamily::ParallelGrouped
                                                           : Int8KvFamily::Tiled;
    const int tiles =
        family == Int8KvFamily::ParallelGrouped ? (width + grouped_limit - 1) / grouped_limit : 1;
    const int independent_tiles = batch * kv_heads * tiles;
    // NINFERNO-SPECIAL(geometry:24/4,16/2): the wave rules were measured on four and two KV
    // heads; the other geometries take the rules of their KV head count.
    const bool two_ctas_per_sm = kv_heads == 4 || width <= 4;
    const auto partition       = causal_wave_partition(independent_tiles, two_ctas_per_sm,
                                                       envelope.max_visible_keys, multiprocessor_count);
    return {family, heads, kv_heads, width, batch, envelope, partition, launch_columns};
}

std::size_t int8_kv_workspace_bytes(int heads, int kv_heads, int batch, int min_width,
                                    int max_width, CausalAttentionExecutionEnvelope envelope,
                                    int multiprocessor_count) {
    std::size_t maximum  = 0;
    const bool invariant = invariant_routes_active();
    const auto widths =
        causal_grouped_workspace_widths(min_width, max_width, kGroupedPrefillMaxWidth, invariant);
    for (int width = widths.first; width <= widths.last; ++width) {
        const auto plan = make_int8_kv_causal_plan(heads, kv_heads, width, batch, envelope,
                                                   multiprocessor_count);
        if (plan.family == Int8KvFamily::Tiled) continue;
        const int splits = plan.partition.capacity;
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, splits, batch);
        if (widths.chunks) allocate_causal_chunk_metadata(layout);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    // A Volta prefill call wider than the grouped tiles takes split-D flash attention. A run's
    // envelope is exact (its visible keys), a plan's spans 1..capacity: size the widest exact run,
    // or the call finds no room and keeps the tiled kernel.
    const CausalAttentionExecutionEnvelope exact{envelope.max_visible_keys,
                                                 envelope.max_visible_keys};
    if (int8_volta_splitd_takes(heads, kv_heads, max_width, batch, exact)) {
        maximum = std::max(maximum, int8_volta_splitd_workspace_bytes(heads, kv_heads, max_width,
                                                                      envelope.max_visible_keys));
    }
    return maximum;
}

} // namespace ninferno::ops::detail
