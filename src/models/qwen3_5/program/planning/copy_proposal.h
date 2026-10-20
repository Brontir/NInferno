#pragma once

// Copy rounds' proposal (SpeculativeOptions::ngram_draft_tokens), host-only so CPU tests check it.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninferno::models::qwen3_5::detail {

// A copy proposal (SpeculativeOptions::ngram_draft_tokens): the tokens that followed the most
// recent earlier occurrence of the longest suffix of `history` that recurs, at least `min_match`
// and at most kLongestCopyMatch tokens long; at most `limit` of them, written to `out`. One pass
// from the end: a candidate must end in the history's last token.
inline constexpr std::size_t kLongestCopyMatch = 32;

inline std::size_t copy_proposal(std::span<const std::int32_t> history, std::size_t min_match,
                                 std::size_t limit, std::int32_t* out) {
    const std::size_t size = history.size();
    if (limit == 0 || min_match == 0 || size <= min_match) { return 0; }
    const std::int32_t* h   = history.data();
    const std::int32_t last = h[size - 1];
    std::size_t best        = 0;
    std::size_t best_end    = 0;
    // `end`: the last token of a candidate occurrence, before the suffix itself.
    for (std::size_t end = size - 1; end-- > min_match - 1;) {
        if (h[end] != last) { continue; }
        std::size_t n = 1;
        while (n < kLongestCopyMatch && n <= end && h[end - n] == h[size - 1 - n]) { ++n; }
        if (n >= min_match && n > best) {
            best     = n;
            best_end = end;
            if (n == kLongestCopyMatch) { break; }
        }
    }
    if (best == 0) { return 0; }
    const std::size_t from = best_end + 1;
    const std::size_t take = std::min(limit, size - from);
    std::copy_n(h + from, take, out);
    return take;
}

} // namespace ninferno::models::qwen3_5::detail
