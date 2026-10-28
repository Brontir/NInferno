#pragma once

// How many rows a draft head restricted to the most frequent token ids keeps (Qwen's proposal
// head: proposal/head with proposal/token_ids, the target's output head gathered over the ids a
// token ranking puts first). From facts only:
//
//   bytes   the draft head reads N/V of the full head's rows each round, so a round saves
//           (1 - N/V) x share of its bytes, share = the full-vocabulary draft head's bytes over a
//           round's (the target's verify pass, the drafter's pass and that head);
//   tokens  a draft whose target token is not among the N kept ids is wrong, so draft j of a
//           round survives with probability c(N)^j, c(N) the ranked ids' share of the tokens a
//           model emits (the ranking's own counts). The bound takes every draft as accepted
//           (a_j = 1): E(N) / E(V) = sum_{j=0..k} c^j / (k + 1), the most a restriction can cost
//           a round of k drafts, so no pair's measured acceptance enters the choice.
//
// N maximizes E(N) / R(N) = [sum c^j / (k + 1)] / [1 - (1 - N/V) share] over steps of kRowStep
// rows. Measured against it (hard50 on the RTX 4070 Ti, Qwen3.5-9B Q4_K_M dflash:7, N 131072):
// rounds +4.7 % (bytes predicted +5.3 %), tokens a round 4.80 -> 4.79 (bound -0.24, the hard50
// outputs' own c predict -0.03), tok/s +4.4 % (predicted +4.6 %).

#include <cstdint>
#include <span>

namespace ninferno::spec {

// The depth a converted head is chosen for: DFlash's default draft count (the built-in default
// dflash:7 and an explicit --spec dflash without --draft-tokens, product/speculative_options.h).
inline constexpr std::int32_t kDraftHeadDepth = 7;

// The granularity of the rows searched (the restricted head's rows are whole multiples of it).
inline constexpr std::uint64_t kDraftHeadRowStep = 8192;

struct DraftHeadRows {
    std::uint64_t rows = 0; // 0: keep the full head (no restriction pays)
    double coverage    = 1; // c(rows): the kept ids' share of the ranking's counts
    double gain        = 0; // E/R over the full head's, minus one (the bound's)
};

// `counts`: the ranking's counts per token id (any order; ids with no count count zero).
// `share`: the full draft head's share of a round's bytes, in (0, 1). `depth`: drafts a round.
[[nodiscard]] DraftHeadRows draft_head_rows(std::span<const std::int64_t> counts, double share,
                                            std::int32_t depth);

} // namespace ninferno::spec
