#pragma once

#include "core/device_class.h"
#include "ops/softmax_attention/common/causal_column_chunks.h"
#include "ops/softmax_attention/common/causal_geometry.h"
#include "ops/softmax_attention/common/causal_partition.h"
#include "ninferno/ops/softmax_attention.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace ninferno::ops::detail {

// The causal KV plans' shared execution contract: at most kCausalMaxBatch rows a call, and past
// one row at most kCausalMaxBatchedWidth columns each (the speculative verify widths).
inline constexpr int kCausalMaxBatch        = 8;
inline constexpr int kCausalMaxBatchedWidth = 16;
static_assert(kCausalMaxBatch == 8 && kCausalMaxBatchedWidth == 16,
              "causal_domain_refusal's messages name these bounds");

// What kind of restriction refuses a causal call (#312): Semantic, no call at all (no row or
// column, an empty or inverted envelope of visible keys); Contract, the plans' shared execution
// contract (the row and width bounds above, the visible keys within
// kCausalAttentionMaximumVisibleKeys, the partition's capacity); Coverage, what is compiled (a
// head pair not instantiated: causal_geometry.h).
enum class CausalRestriction : std::uint8_t { Semantic, Contract, Coverage };
[[nodiscard]] constexpr const char* causal_restriction_name(CausalRestriction kind) noexcept {
    switch (kind) {
    case CausalRestriction::Semantic: return "not a call";
    case CausalRestriction::Contract: return "the execution contract";
    case CausalRestriction::Coverage: return "not compiled";
    }
    return "";
}
struct CausalRefusal {
    CausalRestriction kind;
    const char* why;
};

// The one domain every causal path admits (#312): the plans (one width), the workspace query and
// the batched launch (a width interval) ask it, so they cannot disagree. nullopt: admitted.
[[nodiscard]] constexpr std::optional<CausalRefusal>
causal_domain_refusal(int heads, int kv_heads, int batch, int min_width, int max_width,
                      CausalAttentionExecutionEnvelope envelope) noexcept {
    using K = CausalRestriction;
    if (batch < 1 || min_width < 1 || max_width < min_width) {
        return CausalRefusal{K::Semantic, "a call needs a row and a column (and widths in order)"};
    }
    if (envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys) {
        return CausalRefusal{K::Semantic, "an empty or inverted envelope of visible keys"};
    }
    if (batch > kCausalMaxBatch) { return CausalRefusal{K::Contract, "more than 8 rows a call"}; }
    if (batch > 1 && max_width > kCausalMaxBatchedWidth) {
        return CausalRefusal{K::Contract, "more than 16 columns a row in a call of several rows"};
    }
    if (envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys) {
        return CausalRefusal{K::Contract, "more visible keys than the partition holds"};
    }
    if (!causal_geometry_supported(heads, kv_heads)) {
        return CausalRefusal{K::Coverage, "a head pair no kernel is instantiated for"};
    }
    return std::nullopt;
}

// "<op>: <why> (<kind>)", the refusal said before any launch.
[[nodiscard]] inline std::string causal_refusal_message(const char* op, CausalRefusal refusal) {
    return std::string(op) + ": " + refusal.why + " (" + causal_restriction_name(refusal.kind) +
           ")";
}

// Every causal KV plan admits the same inputs (causal_domain_refusal at one width).
inline void require_causal_plan_inputs(const char* dtype, int heads, int kv_heads, int width,
                                       int batch, CausalAttentionExecutionEnvelope envelope) {
    if (const auto refusal =
            causal_domain_refusal(heads, kv_heads, batch, width, width, envelope)) {
        throw std::invalid_argument(
            causal_refusal_message((std::string(dtype) + " attention").c_str(), *refusal));
    }
}

static_assert(CausalKvPartition::kAlignedSplits <= CausalKvPartition::kMaxSplits);
static_assert(CausalKvPartition::kAlignedMaxVisibleKeys == kCausalAttentionMaximumVisibleKeys);
// The aligned shifts run 9..14: 512 keys up to 32768 visible, 16384 at the longest call.
static_assert(CausalKvPartition::aligned_key_shift(1) == 9);
static_assert(CausalKvPartition::aligned_key_shift(32768) == 9);
static_assert(CausalKvPartition::aligned_key_shift(32769) == 10);
// The smallest split's fixed traffic (the grouped CTA's 32 query rows in BF16, its FP32 partial
// out and back into the merge: 10 bytes a dimension) is at most a sixth of the BF16 keys and
// values it streams (4 bytes a dimension).
static_assert(6 * 32 * 10 <= (1 << CausalKvPartition::kAlignedMinKeyShift) * 4);
static_assert(CausalKvPartition::aligned_key_shift(1 << 18) == 12);
static_assert(
    CausalKvPartition::aligned_key_shift(CausalKvPartition::kAlignedMaxVisibleKeys) == 14);
// A launch's partials hold at most kAlignedSplits per column, whatever the visible keys.
static_assert(CausalKvPartition::aligned_capacity(kCausalAttentionMaximumVisibleKeys) ==
              CausalKvPartition::kAlignedSplits);

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
