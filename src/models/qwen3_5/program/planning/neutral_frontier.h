#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninferno::models::qwen3_5::runtime_support {

// A Qwen prefill is byte-exact only against the same call decomposition. Two kinds of call
// boundary leave every token's bytes as they are:
//   - a static execution frontier of the prompt (its rewrite execution frontiers): every
//     execution of the prompt, cache on or off, splits there;
//   - a point on the 64-token grid (a multiple of the Gated DeltaNet scan's 16-token chunk), inside
//     a static segment that itself starts on the grid, at least kNeutralMinTokens before the
//     segment's end: the calls on both sides stay on the grid and at least 17 tokens long, so they
//     take the prefill routes and cut the scan into the same absolute chunks (see
//     execution::prefill_call_end).
// A capture or restore frontier anywhere else changes the bytes of the tokens after it.
inline constexpr std::uint32_t kNeutralGridTokens = 64;
inline constexpr std::uint32_t kNeutralMinTokens  = 17;

// The largest frontier at or below `frontier` where a prefill call may end without changing any
// byte of the prompt's execution, or none. `static_frontiers` are the prompt's rewrite execution
// frontiers, ordered and unique; the prompt end is always neutral.
[[nodiscard]] constexpr std::optional<std::uint32_t>
neutral_prefill_frontier(std::uint32_t frontier, std::uint32_t prompt_tokens,
                         std::span<const std::uint32_t> static_frontiers) noexcept {
    if (frontier == 0 || frontier > prompt_tokens) { return std::nullopt; }
    if (frontier == prompt_tokens) { return frontier; }
    const auto next = std::lower_bound(static_frontiers.begin(), static_frontiers.end(), frontier);
    if (next != static_frontiers.end() && *next == frontier) { return frontier; }
    const std::uint32_t segment_begin = next == static_frontiers.begin() ? 0U : *(next - 1);
    const std::uint32_t segment_end =
        next == static_frontiers.end() ? prompt_tokens : std::min(*next, prompt_tokens);
    if (segment_begin % kNeutralGridTokens == 0 && segment_end >= kNeutralMinTokens) {
        const std::uint32_t limit   = std::min(frontier, segment_end - kNeutralMinTokens);
        const std::uint32_t aligned = limit / kNeutralGridTokens * kNeutralGridTokens;
        if (aligned > segment_begin) { return aligned; }
    }
    if (segment_begin != 0) { return segment_begin; }
    return std::nullopt;
}

// True when a restore at `frontier` executes the prompt exactly as a cold prefill does.
[[nodiscard]] constexpr bool is_neutral_prefill_frontier(
    std::uint32_t frontier, std::uint32_t prompt_tokens,
    std::span<const std::uint32_t> static_frontiers) noexcept {
    const std::optional<std::uint32_t> neutral =
        neutral_prefill_frontier(frontier, prompt_tokens, static_frontiers);
    return neutral && *neutral == frontier;
}

// A plan's cache opportunities with each frontier moved down to its neutral point. An opportunity
// without one is dropped; two of one kind that land on the same frontier merge, keeping the
// earlier input order and both evidences. Everything that reads the plan's opportunities (its
// captures, its shortlist keys, the resource manager's shared candidates) then names one frontier.
template <class Opportunity>
[[nodiscard]] std::vector<Opportunity>
neutral_cache_opportunities(std::span<const Opportunity> opportunities,
                            std::uint32_t prompt_tokens,
                            std::span<const std::uint32_t> static_frontiers) {
    std::vector<Opportunity> neutral;
    neutral.reserve(opportunities.size());
    for (Opportunity opportunity : opportunities) {
        const std::optional<std::uint32_t> frontier =
            neutral_prefill_frontier(opportunity.frontier, prompt_tokens, static_frontiers);
        if (!frontier) { continue; }
        opportunity.frontier = *frontier;
        const auto duplicate =
            std::find_if(neutral.begin(), neutral.end(), [&](const Opportunity& existing) {
                return existing.kind == opportunity.kind &&
                       existing.frontier == opportunity.frontier;
            });
        if (duplicate == neutral.end()) {
            neutral.push_back(opportunity);
        } else {
            duplicate->input_order = std::min(duplicate->input_order, opportunity.input_order);
            duplicate->evidence |= opportunity.evidence;
        }
    }
    return neutral;
}

} // namespace ninferno::models::qwen3_5::runtime_support
