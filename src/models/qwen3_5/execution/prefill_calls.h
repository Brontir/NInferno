#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

// How a Qwen3.5 prefill cuts a segment into calls; no device or model dependencies, so the
// planner and the tests use the same rule as the execution.
namespace ninferno::models::qwen3_5::execution {

// Calls of at most 16 tokens take the decode/verify routes (A16 projections, small-T attention,
// the small-T MoE), which round differently from the prefill routes. A prefill never leaves such a
// call behind a longer one: the call before it stops early instead, so every token of a segment of
// at least kPrefillMinTokens is computed by the prefill routes whatever the chunk size.
inline constexpr std::uint32_t kPrefillMinTokens = 17;

// The early stop lands on an absolute grid of the Gated DeltaNet scan's chunks. The chunked scan
// cuts a call into 16-token chunks counted from the call's first token (the last one padded) and
// folds each into the state as one block, so a call boundary that divides a chunk changes the bytes
// after it; a boundary on this grid, a multiple of the 16-token chunk, is the only kind the chunk
// size cannot move a token across. A BF16 or FP16 state is rounded at every chunk end, so the call
// before such a boundary stores the state one call carries past it. Under --activations exact the
// scan runs token by token and any boundary is neutral for it
// (ops/linear_attention/gated_delta_net/gated_delta_net.cpp).
inline constexpr std::uint32_t kPrefillGridTokens = 64;

// Absolute end of the next prefill call over [cursor, end) when at most `chunk` fit in one call.
// A call that would leave 1..16 tokens before `end` ends instead on the last grid point that
// leaves at least kPrefillMinTokens, when that keeps it at least kPrefillMinTokens long itself.
[[nodiscard]] constexpr std::uint32_t prefill_call_end(std::uint32_t chunk, std::uint32_t cursor,
                                                       std::uint32_t end) noexcept {
    const std::uint32_t left  = end - cursor;
    const std::uint32_t count = left < chunk ? left : chunk;
    const std::uint32_t rest  = left - count;
    if (rest != 0 && rest < kPrefillMinTokens && end >= kPrefillMinTokens) {
        const std::uint32_t early =
            (end - kPrefillMinTokens) / kPrefillGridTokens * kPrefillGridTokens;
        if (early > cursor && early - cursor >= kPrefillMinTokens) { return early; }
    }
    return cursor + count;
}

// Tokens of that call.
[[nodiscard]] constexpr std::uint32_t prefill_call_tokens(std::uint32_t chunk, std::uint32_t cursor,
                                                          std::uint32_t end) noexcept {
    return prefill_call_end(chunk, cursor, end) - cursor;
}

// The widest call on the decode and verify routes, for the families without a scan grid (the
// dense family, Gemma 4, Nemotron-H and their CLI sessions).
inline constexpr std::uint32_t kNarrowCallTokens = kPrefillMinTokens - 1;

// Tokens of the next call over `left` remaining tokens when at most `chunk` fit in one call: a
// tail that would be left narrow is avoided by stopping short, so both calls run the prefill
// routes.
[[nodiscard]] constexpr std::uint32_t ungridded_call_tokens(std::uint32_t chunk,
                                                            std::uint32_t left) noexcept {
    const std::uint32_t count = left < chunk ? left : chunk;
    const std::uint32_t rest  = left - count;
    if (rest != 0 && rest <= kNarrowCallTokens && chunk > 2 * kNarrowCallTokens + 1) {
        return left - kNarrowCallTokens - 1;
    }
    return count;
}

// The same for the CLI sessions (ninferno-dense, ninferno-gemma4): the next call of a session's
// `chunk` over `remaining` prompt tokens.
[[nodiscard]] constexpr std::int32_t session_call_tokens(std::size_t remaining,
                                                         std::int32_t chunk) noexcept {
    const auto left =
        static_cast<std::uint32_t>(remaining < std::numeric_limits<std::uint32_t>::max()
                                       ? remaining
                                       : std::numeric_limits<std::uint32_t>::max());
    return static_cast<std::int32_t>(
        ungridded_call_tokens(static_cast<std::uint32_t>(chunk), left));
}

} // namespace ninferno::models::qwen3_5::execution
