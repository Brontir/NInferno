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

// Whether a grid of `row_tiles` row tiles by whole `tile_tokens`-token columns fills whole waves:
// a wave holds c = floor(sms / row_tiles) columns, its idle SMs (sms - c x row_tiles) are at most
// a sixteenth of it, and T is a whole number of c-column waves. Gemma 4's Q8 5376x16384 at its
// 256-token tile (42 row tiles of 128): the tile won by 2.6-5.7 % on whole waves of the RTX 5090
// (c = 4, 168 of 170 SMs busy: T a multiple of 1024) and lost 20-30 % elsewhere (768, 1536), so
// a wave more than a sixteenth idle (6 %) gives the gain back.
constexpr bool fills_whole_waves(std::int32_t tokens, int sms, int row_tiles, int tile_tokens) {
    if (sms <= 0 || row_tiles <= 0 || tile_tokens <= 0 || tokens <= 0) return false;
    const int columns = sms / row_tiles;
    if (columns == 0) return false;
    const int idle = sms - columns * row_tiles;
    if (16 * idle > sms) return false;
    return tokens % (static_cast<std::int32_t>(columns) * tile_tokens) == 0;
}

// Whether a grid of `row_tiles` row tiles by ceil(T / tile_tokens) token tiles passes one CTA per
// SM. Gemma 4's NVFP4 2048x5376 A4 prefill: the TMA route's grid is only 16 row tiles wide and
// stays under one wave, so it loses to the 128-token MMA tile until that tile's grid (16 row
// tiles) passes the SMs (RTX 5090: T > 1280).
constexpr bool grid_passes_wave(std::int32_t tokens, int sms, int row_tiles, int tile_tokens) {
    if (sms <= 0 || row_tiles <= 0 || tile_tokens <= 0 || tokens <= 0) return false;
    const std::int64_t token_tiles =
        (static_cast<std::int64_t>(tokens) + tile_tokens - 1) / tile_tokens;
    return token_tiles * row_tiles > static_cast<std::int64_t>(sms);
}

} // namespace ninferno::ops::detail::sm_rules
