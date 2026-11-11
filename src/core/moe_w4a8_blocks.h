#pragma once

// The grouped MoE W4A8's block rule, one definition for the op and the plan that mirrors it
// (ops/moe w4a8_bank_refusal; product/plan's expert route). The tier quantizes the activation a
// stage reads in whole 32-value blocks, and both quantizers load a block's 32 values whole once it
// starts below the width (quantize_x_task, moe_mx8_quantize_kernel): the bank's K and the width of
// the activation must both be whole blocks. A down stage reads the intermediate at its own K; a
// gate/up reads x at the router's width, which a zero-padded bank's K exceeds (a padded x not in
// whole blocks would put the next row's values into its last block's codes, scale and sum).

#include <cstdint>

namespace ninferno {

inline constexpr std::int64_t kMoeW4a8Block = 32;

// Whether a bank of K `bank_k` whose stage reads an activation of `x_width` live values takes the
// grouped W4A8 as far as its blocks go (the down stage: x_width = bank_k).
[[nodiscard]] constexpr bool moe_w4a8_whole_blocks(std::int64_t bank_k,
                                                   std::int64_t x_width) noexcept {
    return bank_k % kMoeW4a8Block == 0 && x_width % kMoeW4a8Block == 0;
}

} // namespace ninferno
