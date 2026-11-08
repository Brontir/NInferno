#pragma once

// The two tiles of the W4A8 matrix kernel that a Blackwell prefill product of Q4_K or Q6_K weights
// may take from 128 tokens (ops/generic heuristic_w4a8), and what each costs: one place for the
// kernel's choice and the product's estimate of it (product/choice mmq_linears_seconds), so the
// estimate prices the tile the kernel runs (DECISIONS 75).
//
// Both tiles are 64 weight rows over the whole K. The producer-warp tile (128 x 64) holds 128
// tokens; its producer warps decode the weight rows and copy the activations ahead of the MMA
// warps, so a CTA's K step is nearly the same for 64 and for 128 tokens (RTX 5090 and 5070 Ti,
// Q4_K 4096 x 4096: 41.0 us at T = 64, 43.0 at 128). The classic tile (64 x 64) holds 64 tokens in
// four warps that decode and multiply in turn.
//
// How many CTAs of a tile share an SM follows from its shared memory: an SM holds
// smem_per_sm / (the CTA's shared memory + 1 KiB the hardware reserves a CTA) of them. The classic
// tile's two buffers are 2 x (64 + 64) x (128 + 32) = 40 KiB (generic_w4a8.cuh W4a8Schedule::
// kSmemBytes), the producer ring at most 99 KiB (kW4a8MaxDynamicSmem, the kernel's static_assert;
// Q4_K's is 90 KiB): on an sm_120 SM of 100 KiB two classic CTAs and one producer-warp CTA are
// resident (static_asserts below). Threads (128 a classic CTA against 1536 a SM) and registers
// (64 K a SM) leave room for two; whether they leave room for more was not read, as no board
// with more shared memory was measured.
//
// The costs were read where those counts are 2 and 1 (RTX 5090 with 170 SMs and RTX 5070 Ti with
// 70, Qwen3.5 9B's Q4_K and Q6_K shapes, N = 1024..12288, K = 4096 and 12288, T = 64..512,
// ninferno-serve route-tune --a8-candidates): a classic CTA alone on its SM costs `single` of a
// producer-warp CTA, two co-resident cost `pair` together (a pair is cheaper than two singles: the
// weight rows' decode is the tile's fixed part, which a CTA's tokens do not scale). A call of
// `columns` tokens takes ceil(columns / 128) token tiles of the producer-warp tile or
// ceil(columns / 64) of the classic one; the busiest SM runs ceil(tiles / (SMs x resident)) rounds
// of the first and ceil(tiles / SMs) CTAs of the second, and the call costs, in producer-warp CTAs,
//   producer-warp: its rounds,
//   classic:       CTAs / 2 pairs plus, for an odd count, one single.
// On a board whose shared memory gives other counts the classic tile is not priced and the
// producer-warp tile stays (`narrow_wins` is false).
//
// What the constants fix: `pair` makes the classic tile win where the 128-token tiles' padding
// exceeds the pair's extra (129..192 tokens: three classic tiles against two producer-warp tiles;
// and 257..320), where the grid fills the SMs less than about twice over. At 129..192 tokens on a
// grid of at most one pass (classic CTAs <= SMs, so busy = 1 against 1), the decision rests on
// `single` alone: 0.98 against 1.0, a 2 % margin; the constants fitted per card are 0.95 (RTX
// 5090) and 1.02 (RTX 5070 Ti), so the 5070 Ti's own fit would flip these calls, at about a tie's
// cost (the product A/B at 136..192 tokens still gained 2.7-3.1 %, dominated by the multi-pass
// calls). The model also leaves out that the producer-warp tile's step grows with the tokens in its
// last tile (RTX 5070 Ti, Q4_K 4096 x 4096 at T = 136: 77.8 us for the two tiles; at T = 256: 84.0,
// +8 %; T = 64: 41.0 and T = 128: 43.0 for one tile): three points do not give the tile's fixed
// and per-token parts, so the step is held constant, which biases the choice toward the
// producer-warp tile.
//
// The classic tile computes the same bytes (every W4A8 schedule sums in one order).

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ninferno::w4a8_tiles {

inline constexpr int kWideTokens   = 128; // the producer-warp tile's tokens
inline constexpr int kNarrowTokens = 64;  // the classic tile's tokens
inline constexpr int kRows         = 64;  // both tiles' weight rows

// Shared memory the hardware reserves for each resident CTA.
inline constexpr std::size_t kBlockReserve = 1024;

// The classic tile's two buffers: int8 codes (128 a row) and two floats per 32-value block.
[[nodiscard]] constexpr std::size_t classic_smem(int tokens = kNarrowTokens,
                                                 int rows   = kRows) noexcept {
    return 2 * static_cast<std::size_t>(tokens + rows) * (128 + 32);
}

// The producer ring's bound (generic_w4a8.cuh kW4a8MaxDynamicSmem).
inline constexpr std::size_t kProducerRingBound = 99 * 1024;

// CTAs of a kernel using `block_smem` bytes that an SM with `smem_per_sm` holds.
[[nodiscard]] constexpr int resident(std::size_t smem_per_sm, std::size_t block_smem) noexcept {
    return static_cast<int>(smem_per_sm / (block_smem + kBlockReserve));
}

struct Residency {
    int wide;   // producer-warp CTAs a SM holds
    int narrow; // classic CTAs a SM holds
};

[[nodiscard]] constexpr Residency residency(std::size_t smem_per_sm) noexcept {
    return {resident(smem_per_sm, kProducerRingBound), resident(smem_per_sm, classic_smem())};
}

// The counts the costs below were read at.
[[nodiscard]] constexpr bool priced(Residency r) noexcept { return r.wide == 1 && r.narrow == 2; }

// sm_120: 100 KiB an SM.
static_assert(residency(100 * 1024).wide == 1 && residency(100 * 1024).narrow == 2 &&
                  priced(residency(100 * 1024)),
              "an sm_120 SM holds one producer-warp CTA and two classic CTAs");
// A 164 KiB SM (the A100) holds four classic CTAs: not priced.
static_assert(residency(164 * 1024).wide == 1 && residency(164 * 1024).narrow == 4 &&
              !priced(residency(164 * 1024)));
static_assert(classic_smem() == 40 * 1024);

// A classic CTA's cost against a producer-warp CTA's at the same K: alone on its SM, and two
// co-resident. Fitted over both cards' points of a format: the pair costs 1.16 of a producer-warp
// CTA for Q4_K on both cards and 1.26 / 1.22 for Q6_K; the two constants give the faster tile at
// 235 of 240 (call, card) points and a time within 16 % (rms 5 %) of the measured one.
struct NarrowCost {
    double single, pair;
};

inline constexpr NarrowCost kQ4k{0.98, 1.16};
inline constexpr NarrowCost kQ6k{0.84, 1.24};

// The producer-warp tile's rounds.
[[nodiscard]] constexpr double wide_work(std::int64_t row_tiles, std::int64_t columns, int sms,
                                         Residency r) noexcept {
    const std::int64_t tiles = row_tiles * ((columns + kWideTokens - 1) / kWideTokens);
    const std::int64_t slots = static_cast<std::int64_t>(sms) * std::max(r.wide, 1);
    return static_cast<double>((tiles + slots - 1) / slots);
}

// The classic tile's work, in producer-warp CTAs (priced residency only).
[[nodiscard]] constexpr double narrow_work(std::int64_t row_tiles, std::int64_t columns, int sms,
                                           const NarrowCost& cost) noexcept {
    const std::int64_t tiles = row_tiles * ((columns + kNarrowTokens - 1) / kNarrowTokens);
    const std::int64_t busy  = (tiles + sms - 1) / sms;
    return static_cast<double>(busy / 2) * cost.pair + static_cast<double>(busy % 2) * cost.single;
}

// Whether the classic tile covers the call in less work (on a tie, or where the counts are not
// the priced ones, the producer-warp tile).
[[nodiscard]] constexpr bool narrow_wins(std::int64_t row_tiles, std::int64_t columns, int sms,
                                         Residency r, const NarrowCost& cost) noexcept {
    return priced(r) &&
           narrow_work(row_tiles, columns, sms, cost) < wide_work(row_tiles, columns, sms, r);
}

// The least work of the two, in producer-warp CTAs.
[[nodiscard]] constexpr double work(std::int64_t row_tiles, std::int64_t columns, int sms,
                                    Residency r, const NarrowCost& cost) noexcept {
    const double wide = wide_work(row_tiles, columns, sms, r);
    return priced(r) ? std::min(wide, narrow_work(row_tiles, columns, sms, cost)) : wide;
}

} // namespace ninferno::w4a8_tiles
