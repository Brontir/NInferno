#pragma once

// The two tiles of the W4A8 matrix kernel that a Blackwell prefill product of Q4_K or Q6_K weights
// may take from 128 tokens (ops/generic heuristic_w4a8), and what each costs: one place for the
// kernel's choice and the product's estimate of it (product/choice mmq_linears_seconds), so the
// estimate prices the tile the kernel runs (DECISIONS 75).
//
// Both tiles are 64 weight rows over the whole K. The producer-warp tile (128 x 64) holds 128
// tokens, takes one CTA a SM (86-92 KiB of shared memory) and its producer warps decode the weight
// rows and copy the activations ahead of the MMA warps, so a CTA's K step is the same for 64 and
// for 128 tokens (RTX 5090 and 5070 Ti, Q4_K 4096 x 4096: 41.0 us at T = 64, 43.0 at 128). The
// classic tile (64 x 64) holds 64 tokens in four warps that decode and multiply in turn: a CTA
// alone on its SM costs `single` of a producer-warp CTA, and two CTAs resident on one SM cost
// `pair` together (read below), so a CTA in a pair costs pair / 2 of one, more than the half that
// its half of the tokens would give alone: the weight rows' decode is its fixed part. A call of
// `columns` tokens takes ceil(columns / 128) token tiles of the producer-warp tile or ceil(columns
// / 64) of the classic one; the busiest SM runs ceil(tiles / SMs) CTAs, and the call costs
//   producer-warp: that count in producer-warp CTAs,
//   classic:       count / 2 pairs plus, for an odd count, one single CTA.
// The classic tile wins where the producer-warp tile's token padding exceeds the pair's extra: at
// 129..192 tokens (3 classic tiles against 2) and 257..320, where the grid fills the SMs less than
// twice over; it computes the same bytes (every W4A8 schedule sums in one order).

#include <algorithm>
#include <cstdint>

namespace ninferno::w4a8_tiles {

inline constexpr int kWideTokens   = 128; // the producer-warp tile's tokens
inline constexpr int kNarrowTokens = 64;  // the classic tile's tokens

// A classic CTA's cost against a producer-warp CTA's at the same K: alone on its SM, and two
// co-resident. RTX 5090 (170 SMs) and RTX 5070 Ti (70 SMs), Qwen3.5 9B's Q4_K and Q6_K shapes
// (N = 1024..12288, K = 4096 and 12288, T = 64..512, ninferno-serve route-tune --a8-candidates):
// the pair costs 1.16 of a producer-warp CTA for Q4_K on both cards and 1.22 / 1.26 for Q6_K; the
// two constants fitted over both cards' points of a format give the right tile at 235 of 240
// (call, card) points and a time within 16 % (rms 5 %) of the measured one.
struct NarrowCost {
    double single, pair;
};

inline constexpr NarrowCost kQ4k{0.98, 1.16};
inline constexpr NarrowCost kQ6k{0.84, 1.24};

// The producer-warp tile's CTAs on the busiest SM.
[[nodiscard]] constexpr double wide_work(std::int64_t row_tiles, std::int64_t columns,
                                         int sms) noexcept {
    const std::int64_t tiles = row_tiles * ((columns + kWideTokens - 1) / kWideTokens);
    return static_cast<double>((tiles + sms - 1) / sms);
}

// The classic tile's work, in producer-warp CTAs.
[[nodiscard]] constexpr double narrow_work(std::int64_t row_tiles, std::int64_t columns, int sms,
                                           const NarrowCost& cost) noexcept {
    const std::int64_t tiles = row_tiles * ((columns + kNarrowTokens - 1) / kNarrowTokens);
    const std::int64_t busy  = (tiles + sms - 1) / sms;
    return static_cast<double>(busy / 2) * cost.pair + static_cast<double>(busy % 2) * cost.single;
}

// Whether the classic tile covers the call in less work (on a tie the producer-warp tile).
[[nodiscard]] constexpr bool narrow_wins(std::int64_t row_tiles, std::int64_t columns, int sms,
                                         const NarrowCost& cost) noexcept {
    return narrow_work(row_tiles, columns, sms, cost) < wide_work(row_tiles, columns, sms);
}

// The least work of the two, in producer-warp CTAs.
[[nodiscard]] constexpr double work(std::int64_t row_tiles, std::int64_t columns, int sms,
                                    const NarrowCost& cost) noexcept {
    return std::min(wide_work(row_tiles, columns, sms), narrow_work(row_tiles, columns, sms, cost));
}

} // namespace ninferno::w4a8_tiles
