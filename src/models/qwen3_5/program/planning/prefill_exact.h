#pragma once

#include "models/qwen3_5/program/planning/neutral_frontier.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>

namespace ninferno::models::qwen3_5::runtime_support {

// The shortest frontier worth a truncated restore: below it the prefill saved is smaller than the
// planning work of one more candidate.
inline constexpr std::uint32_t kTruncatedReuseMinimumTokens = 256;

// A sequence's prefill-exact frontier is the end of the leading KV region written by cold prefill
// calls. Rows a decode (T = 1) route wrote differ in bytes from those a prefill computes, so a
// prefix reuse that must equal a cold prefill may keep only rows below it. These are its
// transitions; SequenceState::prefill_exact_frontier documents the field.

// A restore at `base` keeps the source's rows below it, exact where the source's were. When `base`
// lies inside the exact region the result equals `base`, and the prefill calls which follow extend
// it (prefill_exact_after_call).
[[nodiscard]] constexpr std::uint32_t restored_prefill_exact(std::uint32_t source_exact,
                                                             std::uint32_t base) noexcept {
    return std::min(source_exact, base);
}

// A prefill call which takes the Text KV from `valid_before` to `valid_after` extends the exact
// region only when it starts at its end. Decode and forced-token calls never call this.
[[nodiscard]] constexpr std::uint32_t prefill_exact_after_call(std::uint32_t exact,
                                                               std::uint32_t valid_before,
                                                               std::uint32_t valid_after) noexcept {
    return exact == valid_before ? valid_after : exact;
}

// The frontier at which a request can reuse a resident sequence's KV truncated to the common
// prefix of the two ledgers (Program::supports_truncated_reuse models only), or none.
//
//   common_prefix            tokens the prompt and the source's ledger share
//   prompt_tokens/prompt_static_frontiers   the request's prompt and its rewrite execution
//   frontiers source_exact/source_static_frontiers    the source's prefill-exact frontier, and the
//   rewrite
//                                           execution frontiers its resident identity holds
//
// The result is neutral for the request's prefill (so the suffix it recomputes equals a cold
// prefill's) and for the source's own prefill (so the rows it keeps equal a cold prefill's), lies
// inside the source's exact region, and leaves a suffix to prefill: a truncated reuse has no tail
// hidden to sample from.
[[nodiscard]] constexpr std::optional<std::uint32_t> truncated_reuse_frontier(
    std::uint32_t common_prefix, std::uint32_t prompt_tokens,
    std::span<const std::uint32_t> prompt_static_frontiers, std::uint32_t source_exact,
    std::span<const std::uint32_t> source_static_frontiers, std::uint32_t minimum_tokens) noexcept {
    if (prompt_tokens < 2) { return std::nullopt; }
    const std::uint32_t limit = std::min({common_prefix, source_exact, prompt_tokens - 1U});
    const std::optional<std::uint32_t> source_side =
        neutral_prefill_frontier(limit, source_exact, source_static_frontiers);
    if (!source_side) { return std::nullopt; }
    const std::optional<std::uint32_t> frontier =
        neutral_prefill_frontier(*source_side, prompt_tokens, prompt_static_frontiers);
    if (!frontier || *frontier < minimum_tokens || *frontier >= prompt_tokens ||
        !is_neutral_prefill_frontier(*frontier, source_exact, source_static_frontiers)) {
        return std::nullopt;
    }
    return frontier;
}

} // namespace ninferno::models::qwen3_5::runtime_support
