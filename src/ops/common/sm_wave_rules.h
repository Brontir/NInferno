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

// Whether a grid of `row_tiles` row tiles by whole `tile_tokens`-token columns fills whole waves
// of `ctas_per_sm` resident CTAs an SM (the wave: sms x ctas_per_sm slots): a wave holds
// c = floor(wave / row_tiles) columns, its idle slots (wave - c x row_tiles) are at most a
// sixteenth of it, and T is a whole number of c-column waves. Gemma 4's Q8 5376x16384 at its
// 256-token tile (42 row tiles of 128, one CTA an SM): the tile won by 2.6-5.7 % on whole waves of
// the RTX 5090 (c = 4, 168 of 170 SMs busy: T a multiple of 1024) and lost 20-30 % elsewhere (768,
// 1536). A necessary condition only: the caller also asks wide_tile_pays, since whole waves alone
// took the tile where it lost (RTX 4090, 128 SMs: T = 1536 3 % slower). The sixteenth is a
// declared estimate (docs/maintainer/deterministic-decisions.md): the 5090's data admits any bound
// from 1.2 % (its own idle share) to 26 % (T = 1792, 0.74 of a wave).
constexpr bool fills_whole_waves(std::int32_t tokens, int sms, int row_tiles, int tile_tokens,
                                 int ctas_per_sm = 1) {
    if (sms <= 0 || ctas_per_sm <= 0 || row_tiles <= 0 || tile_tokens <= 0 || tokens <= 0)
        return false;
    const int wave    = sms * ctas_per_sm;
    const int columns = wave / row_tiles;
    if (columns == 0) return false;
    const int idle = wave - columns * row_tiles;
    if (16 * idle > wave) return false;
    return tokens % (static_cast<std::int32_t>(columns) * tile_tokens) == 0;
}

// Whether a wide output tile (one CTA an SM) beats a narrow one on a call of `tokens` columns of an
// `rows`-row output, by the waves each grid costs. The narrow tile's CTAs are a fraction of the
// wide one's work and its grid spreads evenly over the SMs: it costs its total work over the SMs.
// The wide tile pays every wave it starts, the last one whole: ceil(CTAs / SMs) waves of its work.
// The wide tile streams each weight and activation fewer times, so it takes a tie and a cost up to
// a 64th above the narrow one's: the RTX 5090 (170 SMs) measured Gemma 4's Q8 5376x16384 256-token
// tile 2.6-5.7 % faster at 1.2 % more cost (T a multiple of 1024), the RTX 4090 (128 SMs) 3 %
// slower at 1.6 % more (T = 1536), and the RTX 5070 Ti (70 SMs) 4-20 % slower at 4-25 % more (T =
// 1024.. 4096). The 64th is fitted on those three boards. Only whole wide tiles (T a multiple of
// `wide_tokens`) are considered.
constexpr bool wide_tile_pays(std::int32_t tokens, int sms, int rows, int wide_rows,
                              int wide_tokens, int narrow_rows, int narrow_tokens) {
    if (sms <= 0 || tokens <= 0 || rows <= 0 || wide_rows <= 0 || wide_tokens <= 0 ||
        narrow_rows <= 0 || narrow_tokens <= 0 || tokens % wide_tokens != 0)
        return false;
    const std::int64_t wide_ctas =
        static_cast<std::int64_t>((rows + wide_rows - 1) / wide_rows) * (tokens / wide_tokens);
    const std::int64_t narrow_ctas =
        static_cast<std::int64_t>((rows + narrow_rows - 1) / narrow_rows) *
        ((static_cast<std::int64_t>(tokens) + narrow_tokens - 1) / narrow_tokens);
    // The wide CTA's work in narrow CTAs.
    const std::int64_t ratio = static_cast<std::int64_t>(wide_rows) * wide_tokens /
                               (static_cast<std::int64_t>(narrow_rows) * narrow_tokens);
    const std::int64_t wide_waves = (wide_ctas + sms - 1) / sms;
    // wide: waves x ratio, narrow: narrow_ctas / sms (both in narrow-CTA work an SM), times 64 x
    // sms.
    return 64 * wide_waves * ratio * sms <= 65 * narrow_ctas;
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
