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
// an SM) until that tile's grid passes the SMs (RTX 5090: T > 1280). Fitted on the 5090 and checked
// on the RTX 5070 Ti (70 SMs: TMA from T = 641, 28 % faster there, equal from 768); other boards
// are extrapolated until measured.
constexpr bool grid_passes_wave(std::int32_t tokens, int sms, int row_tiles, int tile_tokens,
                                int ctas_per_sm = 1) {
    if (sms <= 0 || ctas_per_sm <= 0 || row_tiles <= 0 || tile_tokens <= 0 || tokens <= 0)
        return false;
    const std::int64_t token_tiles =
        (static_cast<std::int64_t>(tokens) + tile_tokens - 1) / tile_tokens;
    return token_tiles * row_tiles > static_cast<std::int64_t>(sms) * ctas_per_sm;
}

// The width (tokens) up to which fusing launches pays: a fusion that replaces `saved_launches`
// launches by one kernel whose per-token time is `excess` (a fraction of the fused pass's bytes at
// the DRAM rate) worse than the separate kernels' wins while the launches it saves outweigh the
// slope it loses,
//
//     saved_launches x launch_seconds  >=  T x excess x bytes_per_token / dram_bytes_per_s
//     T*  =  saved_launches x launch_seconds x dram_bytes_per_s / (excess x bytes_per_token)
//
// (the launch measured in bytes the board could have moved, over the bytes the extra slope costs a
// token). Floor of T*, no rounding to a measured grid. 0 where an input is unknown or `excess` is
// not calibrated: the caller then keeps its own default. Used by the fused text qk-norm + RoPE
// (rmsnorm_rope): three kernels become one, two launches saved.
constexpr std::int32_t launch_amortized_tokens(int saved_launches, double launch_seconds,
                                               double dram_bytes_per_s, double bytes_per_token,
                                               double excess) {
    if (saved_launches <= 0 || !(launch_seconds > 0.0) || !(dram_bytes_per_s > 0.0) ||
        !(bytes_per_token > 0.0) || !(excess > 0.0)) {
        return 0;
    }
    const double tokens =
        saved_launches * launch_seconds * dram_bytes_per_s / (excess * bytes_per_token);
    return tokens >= 1.0e9 ? 1000000000 : static_cast<std::int32_t>(tokens);
}

} // namespace ninferno::ops::detail::sm_rules
