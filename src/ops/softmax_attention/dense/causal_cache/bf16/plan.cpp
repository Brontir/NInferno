#include "ops/softmax_attention/dense/causal_cache/bf16/plan.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/operands.h"
#include "ops/softmax_attention/dense/causal_cache/plan_rules.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/softmax_attention/common/causal_partition.h"
#include "core/device_class.h"
#include <algorithm>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {
constexpr int kGroupedPrefillMaxWidth = 256;
} // namespace

Bf16KvCausalPlan make_bf16_kv_causal_plan(int heads, int kv_heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope,
                                          int multiprocessor_count) {
    require_causal_plan_inputs("BF16", heads, kv_heads, width, batch, envelope);
    if (multiprocessor_count <= 0) throw std::invalid_argument("BF16 attention: invalid SM count");
    const int group     = heads / kv_heads;
    const int visible   = envelope.max_visible_keys;
    const bool invariant = invariant_routes_active();
    if (width > kGroupedPrefillMaxWidth && !invariant) {
        // NINFERNO-SPECIAL(geometry:16/2): 128-row tiles measured only for Qwen3.5 group 8.
        const auto instance = heads == 16 && kv_heads == 2 && width >= 1024 && width % 128 == 0 &&
                                      visible - width >= 8192
                                  ? Bf16KvInstance::Tiled128
                                  : Bf16KvInstance::Tiled64;
        return {instance, {}, heads, kv_heads, width, batch, envelope, width};
    }
    const int rows                  = width * group;
    // Under the invariant route profile one instance serves every width (a column's arithmetic
    // does not follow the call's width; wider calls run in chunks, causal_column_chunks.h) and
    // the splits are aligned to the keys.
    const auto instance             = invariant    ? Bf16KvInstance::Grouped32
                                      : width == 1 ? Bf16KvInstance::GroupedDecode
                                      : rows <= 32 ? Bf16KvInstance::Grouped32
                                                   : Bf16KvInstance::Grouped64;
    const auto description          = bf16_kv_instance_description(instance);
    const int tiles                 = div_up(rows, description.query_rows);
    const int independent_tiles     = batch * kv_heads * tiles;
    const int sms                   = multiprocessor_count;
    const bool many_memory_tiles    = description.query_rows == 32 && independent_tiles >= 16;
    const bool multiple_query_tiles = tiles > 1;
    // Small-query partitions may use up to six waves; wider prefill targets two. Upstream tuned
    // 85 CTAs per tile on the RTX 5090 (170 SMs); half the device's SMs keeps the waves.
    const int long_ctas = width <= 128 / group && (many_memory_tiles || multiple_query_tiles)
                              ? std::clamp(sms / 2 * independent_tiles, 2 * sms, 6 * sms)
                              : 2 * sms;
    Bf16KvPartition partition{1, causal_partition_target(2 * sms, independent_tiles),
                              causal_partition_target(long_ctas, independent_tiles),
                              description.key_rows};
    // The envelope bounds the largest live row. Other batch rows may be shorter, and a masked row
    // (fewer valid columns than the width) ends its window up to width - 1 keys before the
    // envelope's minimum. The rounded partition is not monotonic in the window, so the capacity
    // covers every window a call can reach (on 128 SMs, W 11 at 2056 keys splits 22 ways, 2046
    // keys 32).
    const int low = batch == 1
                        ? std::max(1, static_cast<int>(envelope.min_visible_keys) - (width - 1))
                        : 1;
    partition.capacity = std::max(
        partition.interval_capacity(low, std::min(visible, 32767), partition.normal_target),
        partition.interval_capacity(std::max(low, 32768), visible, partition.long_target));
    if (invariant) {
        // Splits aligned to each token's own position, as CausalKvPartition::aligned_to_positions.
        partition.aligned  = true;
        partition.capacity = CausalKvPartition::aligned_capacity(visible);
    }
    return {instance, partition, heads,    kv_heads,
            width,    batch,     envelope, causal_launch_columns(width, invariant)};
}

std::size_t bf16_kv_workspace_bytes(int heads, int kv_heads, int batch, int min_width,
                                    int max_width, CausalAttentionExecutionEnvelope envelope,
                                    int multiprocessor_count) {
    std::size_t maximum = 0;
    const auto widths   = causal_grouped_workspace_widths(
        min_width, max_width, kGroupedPrefillMaxWidth, invariant_routes_active());
    for (int width = widths.first; width <= widths.last; ++width) {
        const auto plan = make_bf16_kv_causal_plan(heads, kv_heads, width, batch, envelope,
                                                   multiprocessor_count);
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, plan.partition.capacity, batch);
        if (widths.chunks) allocate_causal_chunk_metadata(layout);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return maximum;
}

} // namespace ninferno::ops::detail
