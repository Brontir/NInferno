#pragma once

// Launch-shape rules that read the device's SM count (DeviceClass::sms, a discrete driver fact)
// and the call's shape, nothing else: pure functions a CPU reproduces
// (docs/maintainer/deterministic-decisions.md, principle 8). Each replaced a literal that was the
// RTX 5090's 170 SMs; on 170 SMs every rule answers what the literal did
// (ninferno_sm_wave_rules_test enumerates it).

#include <algorithm>
#include <cstdint>

namespace ninferno::ops::detail::sm_rules {

// The split of an FP8 TMA call's final wave (fp8_tma_split_k_plan): `tiles` output tiles of one
// CTA per SM each, so a wave is `sms` CTAs. A final wave that fills at most half the SMs splits
// its tiles' K into up to `max_parts` parts (no more than K has `block_k` steps), at most one
// wave of split CTAs. No SM count (0): no split.
struct SplitTail {
    int full_tiles = 0;
    int tail_tiles = 0;
    int split_ctas = 0;
};

constexpr SplitTail fp8_split_tail(int tiles, int k, int sms, int block_k, int max_parts) {
    SplitTail plan{tiles, 0, 0};
    if (sms <= 0 || block_k <= 0) return plan;
    const int tail = tiles % sms;
    if (tail != 0 && tail <= sms / 2) {
        const int parts = std::min(max_parts, k / block_k);
        if (parts >= 2) plan = {tiles - tail, tail, std::min(sms, tail * parts)};
    }
    return plan;
}

// Whether a grid of `row_tiles` row tiles by ceil(T / tile_tokens) token tiles passes one wave of
// `ctas_per_sm` resident CTAs an SM. Gemma 4's NVFP4 2048x5376 A4 prefill: the TMA route's grid is
// only 16 row tiles wide and stays under one wave, so it loses to the 128-token MMA tile (one CTA
// an SM) until that tile's grid passes the SMs (RTX 5090: T > 1280). Measured on the 5090 alone;
// on other boards the crossing is extrapolated until measured.
constexpr bool grid_passes_wave(std::int32_t tokens, int sms, int row_tiles, int tile_tokens,
                                int ctas_per_sm = 1) {
    if (sms <= 0 || ctas_per_sm <= 0 || row_tiles <= 0 || tile_tokens <= 0 || tokens <= 0)
        return false;
    const std::int64_t token_tiles =
        (static_cast<std::int64_t>(tokens) + tile_tokens - 1) / tile_tokens;
    return token_tiles * row_tiles > static_cast<std::int64_t>(sms) * ctas_per_sm;
}

} // namespace ninferno::ops::detail::sm_rules
