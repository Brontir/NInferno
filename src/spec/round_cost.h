#pragma once

// What a speculation round costs, from facts: the weight bytes each pass reads (the model's) and
// the card's memory and tensor rates (the chip's). Measured are only the tokens a round yields
// (E(k), the pair's acceptance: the lab's registry). The speed of a configuration is then
//
//     decode tok/s x E(k) / R(k)
//
// with R(k) a round's price in plain decode steps:
//
//     R(k) = [ C_t(k+1) + n(k) C_d(r_d) ] / C_t(1),     C(r) = B(r) + kappa F(r)
//
//   B(r)   the weight bytes a pass over r rows reads: the dense weights (layers and output head)
//          once, an MoE's routed experts as the rows route them (moe_read_fraction);
//   F(r)   its operations, 2 x r x the active parameters;
//   C_t    the target's pass: plain decode is C_t(1), the verify of k drafts C_t(k + 1);
//   C_d    the drafter's pass of r_d rows (1 for MTP, its block for DFlash), n(k) of them a round
//          (k for MTP, which drafts token by token; 1 for DFlash, which drafts a block at once);
//   kappa  the card's bytes per operation at the small batches a round verifies:
//          DRAM bytes/s / (eta x the generation's int8 tensor peak), eta the share of that peak
//          small-T kernels reach (kSmallTEta: the only fitted constants, one per generation).
//
// Measured against it (docs/maintainer/choice-notes/DECISIONS.md 53, choice-notes/lab/round_cost):
// Qwen dense and MoE, gpt-oss at k <= 7 within 1-2 % on average (4 % rms) on six cards of five
// generations. Where the model does not fit, the estimate is made conservative and says so
// (round_cost_estimate): the lab card's own measurement of the pair, where it found rounds dearer
// than their bytes (Gemma 4's verify, Nemotron's MTP, Q8_0 targets: +10-26 %), scales the estimate
// up, never down; rounds of more than kPortableDepth drafts (from 12 rows verify kernels change
// tiles: +15-50 %) are not estimated away from the lab card at all.

#include "spec/config.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace ninferno::spec {

struct GpuKey; // spec/profile.h

// One pass's weights: its dense part (read once whatever the rows) and its routed experts.
struct PassFacts {
    double bytes         = 0; // dense weights' bytes (layers, output head), as stored
    double params        = 0; // their parameters
    double expert_bytes  = 0; // routed experts' bytes (0: dense)
    double expert_params = 0;
    std::int32_t experts      = 0; // routed experts per layer (0: dense)
    std::int32_t experts_used = 0; // experts a token routes to

    friend bool operator==(const PassFacts&, const PassFacts&) = default;
};

// A drafter's pass: rows a pass runs over, and whether a round runs one per draft (MTP) or one.
struct DraftFacts {
    PassFacts pass;
    std::int32_t rows = 1;
    bool per_draft    = true;

    friend bool operator==(const DraftFacts&, const DraftFacts&) = default;
};

// A target's facts and its drafters' (by mode name: "mtp", "dflash", "dflash2"), as the registry's
// pairs carry them ("round_facts", tools/spec/round_facts.py from the weights' tensor tables).
struct RoundFacts {
    PassFacts target;
    std::map<std::string, DraftFacts> drafters;

    friend bool operator==(const RoundFacts&, const RoundFacts&) = default;
};

[[nodiscard]] nlohmann::json round_facts_json(const RoundFacts& facts);
// Throws std::runtime_error naming what is malformed.
[[nodiscard]] RoundFacts parse_round_facts(const nlohmann::json& doc);

// Rows routed apart read this share of the experts independent routing would add (the
// configuration choice's kMoeRoutedShare, product/choice/choice_constants.h: fitted on the
// registry's speculation floors, audit1018/x3).
inline constexpr double kRoutedExpertShare = 0.30;

// The share of a pass's routed-expert bytes `rows` rows read: k/E for one row, towards the experts
// any of them routes to (1 - (1 - k/E)^rows) at kRoutedExpertShare of independent routing.
[[nodiscard]] double moe_read_fraction(const PassFacts& pass, std::int32_t rows) noexcept;

// The share of its int8 tensor peak (the generation row's s8 rate, core/generation.h; dp4a's where
// it has none) a verify of up to 8 rows reaches, by generation. Fitted per generation on the
// registry's spec-tune rounds of Qwen targets that verify 5-8 rows (k 4..7, the depths profiles
// choose; choice-notes/lab/round_cost/fit_eta.py), as kappa = BW / (eta peak):
//   ada    0.617  RTX 4070 Ti and RTX 4090 (each fitted alone: 0.67 / 0.67 over k <= 7)
//   gb20x  0.189  RTX 5070 Ti; checked on the RTX 5090 (hard50: MTP within 5 %, DFlash 11-14 %
//                 conservative)
//   ga10x  0.420  RTX 3060 (one card)
//   turing 0.168  RTX 2080 Ti (one card)
//   volta  0.233  Quadro GV100 (one card; its verify steps up from 5 rows, so a fit over 5-8 rows
//                 prices 2-4 rows 10-22 % high: conservative)
// One-card rows are architecture facts as DECISIONS 52 takes them (the owner, 27.10): flagged in
// the estimate's note. A generation without a row (ga100, hopper, gb100, thor) has no estimate.
struct SmallTEta {
    std::string_view generation;
    double eta;
    std::int32_t cards; // lab cards the row rests on
};
inline constexpr SmallTEta kSmallTEta[] = {
    {"ada", 0.617, 2}, {"gb20x", 0.189, 1}, {"ga10x", 0.420, 1},
    {"turing", 0.168, 1}, {"volta", 0.233, 1},
};

// dp4a's int8 rate a SM-clock, for a generation without an int8 MMA (Volta): the configuration
// choice's kDp4aOpsPerSmClock (CUDA C++ Programming Guide, compute capability 7.0).
inline constexpr double kDp4aOpsPerSmClockRound = 512;

// A card's rates as the round cost reads them.
struct CardRates {
    double dram_bytes_per_s = 0;
    double int8_ops_per_s   = 0; // the generation's int8 tensor rate x SMs x reference clock
    std::string generation;      // core/generation.h's row name
    double eta              = 0; // kSmallTEta's, 0 when the generation has none
    std::int32_t eta_cards  = 0;

    // DRAM bytes per operation at small T; 0 when unknown.
    [[nodiscard]] double kappa() const noexcept {
        return eta > 0 && int8_ops_per_s > 0 ? dram_bytes_per_s / (eta * int8_ops_per_s) : 0;
    }
    [[nodiscard]] bool known() const noexcept { return kappa() > 0; }
};

// From a compute capability (89: Ada), SMs, reference SM clock and nominal DRAM rate; not known()
// when any is 0 or the generation has no kSmallTEta row.
[[nodiscard]] CardRates card_rates(int compute_capability, int sms, int sm_clock_khz,
                                   double dram_gbps);
// The rates a GpuKey carries (product::choice_card_of's facts).
[[nodiscard]] CardRates card_rates(const GpuKey& gpu);
// A board by its PCI product id (a pair's lab_board): the PCI-ID table's row (core/machine_facts.h).
// Not known() when the table has no row for it.
[[nodiscard]] CardRates card_rates_of_board(std::uint32_t pci_device_id);

// R(k) above for configuration `config` on a card of `kappa`; nullopt when the facts have no
// drafter of its mode, or the configuration drafts nothing.
[[nodiscard]] std::optional<double> round_cost(const RoundFacts& facts, const SpecConfig& config,
                                               double kappa);

} // namespace ninferno::spec
