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

// CTAs of `threads` threads one SM holds at once, from the device's two per-SM limits (the thread
// count and the CTA slots; DeviceClass::max_threads_per_sm, max_blocks_per_sm). Registers and
// shared memory are the compiled kernel's: a kernel that takes either to a lower count has its own
// occupancy figure (kernel_resident_ctas_per_sm), and this answers the upper bound.
constexpr int resident_ctas_per_sm(int max_blocks, int max_threads, int threads) {
    if (max_blocks <= 0 || max_threads <= 0 || threads <= 0) return 0;
    return std::min(max_blocks, max_threads / threads);
}

// The widest call (tokens) whose grid of `ctas_per_token` CTAs a token stays within `waves` waves
// of `ctas_per_sm` resident CTAs an SM, rounded down to a power of two (the measured widths are
// doublings, so a finer bound claims a resolution no measurement has). 0 where an input is
// unknown or not even one token fits: the caller then keeps its own default. The fused text
// qk-norm + RoPE (rmsnorm_rope): its grid is tokens x ceil((Q + K heads) / 3) CTAs of three warps,
// and past a couple of waves the three separate kernels, each free to pick its own shape, are
// ahead; on the RTX 5090's 170 SMs the widest geometry (12 CTAs a token, 16 resident an SM, two
// waves) gives 453 tokens, so 256: the bound the kernel's measurement set.
constexpr std::int32_t grid_wave_bound_tokens(int sms, int ctas_per_sm, int ctas_per_token,
                                              int waves) {
    if (sms <= 0 || ctas_per_sm <= 0 || ctas_per_token <= 0 || waves <= 0) return 0;
    const std::int64_t fit = static_cast<std::int64_t>(waves) * sms * ctas_per_sm / ctas_per_token;
    std::int32_t bound     = 0;
    for (std::int64_t p = 1; p <= fit && p <= (std::int64_t{1} << 30); p *= 2) {
        bound = static_cast<std::int32_t>(p);
    }
    return bound;
}

} // namespace ninferno::ops::detail::sm_rules
