#pragma once

// The speculation load model: which draft depth pays at each number of rows decoding together,
// from the card's cost of a verify round and the drafter's acceptance, rather than a measured
// table per GPU class (decisions from card properties; measured tables are only
// overrides). Pure functions, tested without a GPU.
//
// A round of B rows at depth k verifies w = B(k+1) columns. Relative to a plain step of one row
// (grids on the RTX 4090, RTX 5090 and V100, 27B, per-request decode speed while in flight):
//   t(B, k) / t0 = 1 + alpha (w - 1) + beta max(0, w - knee) + passes * delta,
// passes k for an MTP round (one drafter pass per draft), 1 for a DFlash round (one block).
// Worst error 5.6-7.8 % over plain, MTP 1/2/3/5 and DFlash2 3/5/7 at C = 1, 2, 4, 8.
//
// The i-th draft of a round is accepted, given the earlier ones, with probability a d^(i-1): the
// acceptance falls along the block (RTX 4090 grid: DFlash2 a 0.75, d 0.97; MTP a 0.77, d 0.91 give
// the measured tokens per round at every depth to 0.02). A round of k drafts then yields
// E(k) = 1 + sum_{j=1..k} prod_{i=1..j} a d^(i-1) tokens, and a row's speed is E(k) / t(B, k).

#include "spec/profile.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninferno::spec {

struct RoundCost {
    double alpha = 0;      // per verify column beyond the first
    double beta  = 0;      // per column beyond `knee`
    std::int32_t knee = 0;
    double delta = 0;      // per drafter pass
    bool pass_per_draft = true; // MTP: k passes; DFlash: one per round
};

// The round cost of `gpu` for a target of `bytes_per_weight` (0.53: Q4_G64), from its rates: a
// verify column costs 2 flops a weight at the tensor cores' peak, a plain step reads every weight
// once, so alpha = 2 dram / (eta * flops_per_clock_sm * sms * clock * bytes_per_weight). eta per
// architecture folds in what the step spends past the weight read; calibrated on measured grids
// (27B Q4_G64, 8K): Ada 0.99 (RTX 4090 alpha 0.0233), consumer Blackwell 1.13 (RTX 5090 0.0285),
// A100 0.60 (0.0312), GeForce Ampere 1.125 (RTX 3090 0.0441). The knee is where the verify GEMM
// takes its next tile (RTX 4090 8 columns, RTX 5090 12, A100 24, RTX 3090 32), delta the drafter's
// pass (0.056-0.080 of a plain step on each of them). nullopt: an architecture no grid measured, or
// unknown rates.
[[nodiscard]] std::optional<RoundCost> round_cost_of(const GpuKey& gpu, double bytes_per_weight,
                                                     bool pass_per_draft);

// The relative time of a round of `rows` rows at `depth` drafts (0: plain decode).
[[nodiscard]] double round_time(const RoundCost& cost, std::int32_t rows, std::int32_t depth);

// E(depth) for first-draft acceptance `a` and decay `d` along the block.
[[nodiscard]] double tokens_per_round(double a, double d, std::int32_t depth);

// The first-draft acceptance that gives `tokens` per round at `depth` with decay `d` (a profile
// measures one depth); clamped to [0, 0.999].
[[nodiscard]] double first_draft_acceptance(double tokens, std::int32_t depth, double d);

// The depth of `depths` (or 0, plain decode) that maximizes a row's speed at each row count
// 1..max_rows, as a load table: an entry {lanes, depth} bounds every step of `lanes` rows or more
// at `depth` (parse_load's "lanes:depth"); one row runs the deepest that pays, unbounded when that
// is the deepest of `depths`. A depth never rises again once it fell (a row turned plain does not
// resume).
[[nodiscard]] std::vector<LoadBound> modeled_load(const RoundCost& cost, double a, double d,
                                                  std::span<const std::int32_t> depths,
                                                  std::int32_t max_rows);

} // namespace ninferno::spec
