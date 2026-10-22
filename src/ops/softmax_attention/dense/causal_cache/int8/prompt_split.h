#pragma once

#include <cstddef>

namespace ninferno::ops::detail {

// The key pieces of the INT8 prompt kernel (int8/prompt_fast.cuh): piece s of every row owns the
// keys [s * kInt8PromptSplitKeys, (s + 1) * kInt8PromptSplitKeys), one CTA each. They follow the
// row's own position, so a call of any width gives a row the same bytes, and a short call over a
// long context still spreads over the SMs (17 columns at 32K keys: 4 pieces per head). 8192 keys:
// a call up to 8K keys runs one piece and stores no partial, and a wide call past it pays the
// partials' traffic for half the pieces 4096 would cut (RTX 5070 Ti, 4096 columns at 32K keys:
// +10 % against +16 %; 17 columns at 32K: 377 against 401 us).
inline constexpr int kInt8PromptSplitShift = 13;
inline constexpr int kInt8PromptSplitKeys  = 1 << kInt8PromptSplitShift;

// The pieces a launch over up to `visible` keys reserves.
constexpr int int8_prompt_pieces(int visible) noexcept {
    return (visible + kInt8PromptSplitKeys - 1) >> kInt8PromptSplitShift;
}

// A launch's FP32 partials (D256 sums, maximum and sum per row, head and piece) stay near this many
// bytes: a wider call runs in chunks of whole 128-row blocks.
inline constexpr std::size_t kInt8PromptPartialBudget = std::size_t{64} << 20;

// The columns one prompt launch of a `width`-column call takes: whole 128-row blocks whose partials
// fit the budget (one block at least), never more than the call. It grows with the width, so the
// widest call of an interval sizes the workspace.
constexpr int int8_prompt_launch_columns(int width, int heads, int pieces) noexcept {
    if (pieces <= 1) return width;
    const std::size_t block  = static_cast<std::size_t>(heads) * static_cast<std::size_t>(pieces) *
                               (256 + 2) * sizeof(float) * 128;
    const std::size_t blocks = kInt8PromptPartialBudget / block;
    const int chunk          = blocks > 0 ? static_cast<int>(blocks) * 128 : 128;
    return width < chunk ? width : chunk;
}

} // namespace ninferno::ops::detail
