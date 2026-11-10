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
// count and the CTA slots: DeviceClass::max_threads_per_sm, max_blocks_per_sm). Registers and
// shared memory are the compiled kernel's; this is the upper bound.
constexpr int resident_ctas_per_sm(int max_blocks, int max_threads, int threads) {
    if (max_blocks <= 0 || max_threads <= 0 || threads <= 0) return 0;
    return std::min(max_blocks, max_threads / threads);
}

// The width (tokens) up to which fusing launches pays. A fusion that replaces `saved_launches`
// launches by one kernel wins while the launches it saves outweigh the per-token time it loses:
//
//     saved_launches x launch_seconds  >=  T x delta_per_token
//
// The fused text qk-norm + RoPE (rmsnorm_rope) is a per-warp latency chain (one warp owns one head:
// load, reduce, rotate, store), not a bandwidth stream: the three separate kernels move the same
// DRAM bytes (their intermediates stay in the L2 at these widths). Resident warps run their chains
// side by side, so a token's heads take heads_per_token / (sms x warps_per_sm) chains, each
// `excess_cycles` longer in the fused kernel than the separate kernels' share of it:
//
//     delta_per_token = excess_cycles / sm_clock_hz x heads_per_token / (sms x warps_per_sm)
//     T*  =  saved_launches x launch_seconds x sms x warps_per_sm x sm_clock_hz
//            / (excess_cycles x heads_per_token)
//
// The competing reading normalises the excess by DRAM time instead (T* ~ launch_seconds x DRAM
// rate / bytes a token); on the Ada cards the two differ by under 7 % (DRAM rate per SM is nearly
// constant), so the pair that tells them apart is a board whose DRAM rate per SM differs: an L4
// against an RTX 4060 Ti (1.37 against 1.04), or boards of other generations with the launch
// measured on each (bench/ops/rmsnorm_rope_bench.cu --crossover prints both implied constants).
// Floor of T*, no rounding to a grid. 0 where an input is unknown or `excess_cycles` is not
// calibrated: the caller then keeps its own default. A call wider than a graph is captured for
// (a prefill chunk run eagerly) pays the eager launch, dearer than the node: T* is understated
// there.
constexpr std::int32_t launch_amortized_tokens(int saved_launches, double launch_seconds, int sms,
                                               int warps_per_sm, double sm_clock_hz,
                                               double heads_per_token, double excess_cycles) {
    if (saved_launches <= 0 || !(launch_seconds > 0.0) || sms <= 0 || warps_per_sm <= 0 ||
        !(sm_clock_hz > 0.0) || !(heads_per_token > 0.0) || !(excess_cycles > 0.0)) {
        return 0;
    }
    const double tokens = saved_launches * launch_seconds * sms * warps_per_sm * sm_clock_hz /
                          (excess_cycles * heads_per_token);
    return tokens >= 1.0e9 ? 1000000000 : static_cast<std::int32_t>(tokens);
}

} // namespace ninferno::ops::detail::sm_rules
