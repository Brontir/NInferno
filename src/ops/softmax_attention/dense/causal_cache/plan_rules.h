#pragma once

#include "core/device_class.h"
#include "ops/softmax_attention/common/causal_column_chunks.h"
#include "ops/softmax_attention/common/causal_geometry.h"
#include "ops/softmax_attention/common/causal_partition.h"
#include "ninferno/ops/softmax_attention.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninferno::ops::detail {

// Every causal KV plan admits the same inputs: an admitted geometry, 1-8 rows (16 tokens each
// past one row) and a non-empty envelope within the maximum visible keys.
inline void require_causal_plan_inputs(const char* dtype, int heads, int kv_heads, int width,
                                       int batch, CausalAttentionExecutionEnvelope envelope) {
    if (!causal_geometry_supported(heads, kv_heads) || width < 1 || batch < 1 || batch > 8 ||
        (batch > 1 && width > 16) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys)
        throw std::invalid_argument(std::string(dtype) + " attention: invalid plan inputs");
}

static_assert(CausalKvPartition::kAlignedSplits <= CausalKvPartition::kMaxSplits);
static_assert(CausalKvPartition::kAlignedMaxRowKeys == kCausalAttentionMaximumVisibleKeys);
// The launches take key_shift 6..14.
static_assert(CausalKvPartition::aligned_key_shift(CausalKvPartition::kAlignedMaxRowKeys) <= 14);
static_assert(CausalKvPartition::aligned_key_shift(1 << 18) == 12); // rows up to 256K: unchanged

// The widths whose grouped partials a workspace for calls of [min_width, max_width] columns holds:
// under the invariant route profile those of its launches (causal_launch_columns), with the chunk
// metadata when a call is wider; otherwise the grouped widths up to `grouped_max`.
struct CausalGroupedWorkspaceWidths {
    int first;
    int last;
    bool chunks;
};

inline CausalGroupedWorkspaceWidths
causal_grouped_workspace_widths(int min_width, int max_width, int grouped_max, bool invariant) {
    if (!invariant) return {min_width, std::min(max_width, grouped_max), false};
    const int last = causal_launch_columns(max_width, true);
    return {std::min(min_width, last), last, max_width > last};
}

// The split partition of a quantized grouped plan: splits for one wave of the device's SMs over
// the independent tiles, or two waves when the kernel keeps two CTAs resident per SM or when
// rounding to whole tiles would leave over 10 % of the SMs idle.
inline CausalKvPartition causal_wave_partition(int independent_tiles, bool two_ctas_per_sm,
                                               int max_visible_keys, int multiprocessor_count) {
    const int sms       = multiprocessor_count;
    const int wave_ctas = (sms / independent_tiles) * independent_tiles;
    const int budget    = two_ctas_per_sm || wave_ctas < sms * 9 / 10 ? 2 * sms : sms;
    CausalKvPartition partition{
        1, std::clamp(budget / independent_tiles, 1, CausalKvPartition::kMaxSplits)};
    partition.capacity = partition.active(max_visible_keys);
    return partition;
}

} // namespace ninferno::ops::detail
