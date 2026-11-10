#pragma once

// What a speculation round costs, from facts: the weight bytes each pass reads (the model's) and
// the card's memory and tensor rates (the chip's). Measured are only the tokens a round yields
// (E(k), the pair's acceptance: the lab's registry). The speed of a configuration is then
//
//     decode tok/s x E(k) / R(k)
//
// with R(k) a round's price in plain decode steps:
//
//     R(k) = [ C_t(k+1) + n(k) C_d(r_d) + C_h(k) ] / C_t(1),     C(r) = B(r) + kappa F(r)
//
//   B(r)   the weight bytes a pass over r rows reads: the dense weights (layers and output head)
//          once, an MoE's routed experts as the rows route them (moe_read_fraction);
//   F(r)   its operations, 2 x r x the active parameters;
//   C_t    the target's pass: plain decode is C_t(1), the verify of k drafts C_t(k + 1);
//   C_d    the drafter's pass of r_d rows (1 for MTP, its block for DFlash), n(k) of them a round
//          (k for MTP, which drafts token by token; 1 for DFlash, which drafts a block at once),
//          plus, for DFlash, the target's output head over the k drafted rows once a round;
//   kappa  the card's bytes per operation at the small batches a round verifies:
//          DRAM bytes/s / (eta x the generation's int8 tensor peak), eta the share of that peak
//          small-T kernels reach (kSmallTEta: the only fitted constants, one per generation).
//
// Measured against it (docs/maintainer/decisions.md 53, tools/fits/round_cost):
// Qwen dense within 2 % on average (2-5 % rms), Qwen MoE +1 % (Ada) / +12 % (Blackwell), gpt-oss
// +7..16 %, at k <= 7 on seven cards of five generations. Where the model does not fit, the estimate
// is made conservative and says so
// (round_cost_estimate): the lab card's own measurement of the pair, where it found rounds dearer
// than their bytes (Gemma 4's verify, Nemotron's MTP, Q8_0 targets, gpt-oss: +10-25 %), scales the estimate
// up, never down; rounds of more than kPortableDepth drafts (from 12 rows verify kernels change
// tiles: up to +60 %) are not estimated away from the lab card at all.

#include "core/cost_constants.h"
#include "core/holdout.h"
#include "spec/moe_routed_share.h"
#include "spec/config.h"
#include "spec/proposal_rows.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::spec {

struct GpuKey; // spec/profile.h

// Linears a pass computes alike: `count` matrices of n output rows over k input columns in one
// published format (tools/spec/round_facts.py: "gguf:Q4_K", "bf16", "nvfp4", "mxfp4", "int4", "fp8",
// "conv:q8" where the converter re-encodes a drafter's projections), the output head's apart, a
// routed expert bank's with its experts counted. Facts of the weights' headers: the shapes a verify's
// routes are asked for (ops/generic/route_query.h); not yet priced (the round's cost is its bytes).
struct LinearGroup {
    std::string format;
    std::int64_t n     = 0;
    std::int64_t k     = 0;
    std::int32_t count = 0;
    bool head          = false;
    bool expert        = false;

    friend bool operator==(const LinearGroup&, const LinearGroup&) = default;
};

// One pass's weights: its dense part (read once whatever the rows) and its routed experts.
struct PassFacts {
    double bytes         = 0; // dense weights' bytes (layers, output head), as stored
    double params        = 0; // their parameters
    double expert_bytes  = 0; // routed experts' bytes (0: dense)
    double expert_params = 0;
    std::int32_t experts      = 0; // routed experts per layer (0: dense)
    std::int32_t experts_used = 0; // experts a token routes to
    std::vector<LinearGroup> linears; // its linears (empty: facts written before they were listed)

    friend bool operator==(const PassFacts&, const PassFacts&) = default;
};

// A drafter's pass: rows a pass runs over, and whether a round runs one per draft (MTP) or one;
// and the target's output head as the drafter runs it, apart: per draft over the pass's rows (the
// target's own MTP layers), or once a round over the k drafted rows only (DFlash: the target's
// output head scores the drafted positions, not the block's, models/speculation/dflash.cpp). A
// proposal head reads its rows' share of it (ProposalRows; the head's rows are its head linear's n).
struct DraftFacts {
    PassFacts pass;
    std::int32_t rows = 1;
    bool per_draft    = true;
    std::optional<PassFacts> head;

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
// Throws std::runtime_error naming what is malformed. A drafter of a mode this build does not know
// is left out (a later registry's facts read by an earlier build).
[[nodiscard]] RoundFacts parse_round_facts(const nlohmann::json& doc);

// The share of a pass's routed-expert bytes `rows` rows read: k/E for one row, towards the experts
// any of them routes to (1 - (1 - k/E)^rows) at kMoeRoutedShare (spec/moe_routed_share.h) of
// independent routing.
[[nodiscard]] double moe_read_fraction(const PassFacts& pass, std::int32_t rows) noexcept;
// The same over `streams` requests of `rows` rows each, decoded together (#303): the rows of one
// request route alike (kMoeRoutedShare, fitted on one request's rows), different requests'
// independently, so together they read 1 - (1 - f)^streams of the experts, f one request's share.
// RTX 4090, Qwen3.6-35B-A3B UD-Q4_K_M, plain decode, one server, graph durations (a graph-level
// trace): a step of two requests 4.805 ms against one request's 3.964, 1.212 one-request steps;
// this rule gives 1.210, the one-request rule over two rows 1.083. (spec-tune --load's cells,
// 1.356, also hold the requests' turnover: the next request's prefill between steps.)
[[nodiscard]] double moe_read_fraction(const PassFacts& pass, std::int32_t rows,
                                       std::int32_t streams) noexcept;

// The share of its int8 tensor peak (the generation row's s8 rate, core/generation.h; dp4a's where
// it has none) a verify of up to 8 rows reaches, by generation. Fitted per generation on the
// registry's spec-tune rounds of Qwen targets that verify 5-8 rows (k 4..7, the depths profiles
// choose; tools/fits/round_cost/fit_eta.py), as kappa = BW / (eta peak). A generation
// constant rests on two lab cards of the generation or more (AGENTS.md); each card's rounds are
// predicted from the generation's other cards' fit before a row is added (fit_eta.py, item 3):
//   ada    0.148  RTX 4070 Ti and RTX 4090 (each card's rounds predicted from the other's fit:
//                 hard50 within -3..+4 % and +1..+9 %; gpt-oss +16..19 %, its runs drafted BF16)
//   gb20x  0.165  RTX 5070 Ti and RTX 5090 (DECISIONS 56: held out, R within +4.3 % / +1.4 % on
//                 average; the plan's pick on the other card's pairs is the held-out card's
//                 measured best or within 1.5 % of it, gpt-oss 4 %; the 5070 Ti's 9B rounds of
//                 6-8 verify rows cost 7-12 % less than their facts, its speedup predicted low)
// Each other generation needs a second lab card's registry pairs before it gets a row; until then
// it has no estimate, and its cards keep the lab-measured profiles (as before the facts). Their
// one-card fits, for when the second card comes (DECISIONS 53): ga10x 0.197 (RTX 3060), turing
// 0.099 (RTX 2080 Ti), volta 0.179 (Quadro GV100). The rounds are the registry's after #156 (Gemma 4
// assistants and DFlash drafters attached q8): their q8 drafters' block passes are compute-bound
// enough that eta fell from 0.54 (Ada, BF16 drafters) to 0.15.
struct SmallTEta {
    std::string_view generation;
    double eta;
    std::int32_t cards; // lab cards the row rests on (two or more)
    Holdout holdout;    // required: test_spec_round_cost fails on an Unset one
};

inline constexpr SmallTEta kSmallTEta[] = {
    {"ada", 0.148, 2,
     Holdout::checked(
         "registry spec-tune rounds of Qwen targets, k 4..7, RTX 4070 Ti and RTX 4090 together",
         "each card's rounds predicted from the other card's eta alone (fit_eta.py item 3)",
         "R measured / R facts",
         "RTX 4070 Ti from the 4090 -3..+4 %, RTX 4090 from the 4070 Ti +1..+9 % (DECISIONS 53); "
         "+0.1 % and +0.2 % on average, rms 3.7 % and 2.0 % (DECISIONS 56, for scale)",
         "DECISIONS 53 and 56",
         "gpt-oss rounds run +16..19 % over their facts (its runs drafted BF16)")},
    {"gb20x", 0.165, 2,
     Holdout::checked(
         "registry spec-tune rounds of Qwen targets, k <= 7, RTX 5070 Ti and RTX 5090 together",
         "each card's rounds predicted from the other card's eta alone (fit_eta.py item 3)",
         "R measured / R facts",
         "RTX 5070 Ti from the 5090 (0.184) +4.3 % on average, rms 8.4 %; RTX 5090 from the 5070 "
         "Ti "
         "(0.162) +1.4 %, rms 4.4 %",
         "DECISIONS 56",
         "the plan's pick on the other card's pairs equals the held-out card's measured best or is "
         "within 1.5 % of it (gpt-oss 4 %)")},
};

// A kernel's launch time inside a graph replay (t0) and the DRAM rate a decode kernel's bytes reach
// (eta_dram), by generation: a kernel of B bytes takes B / (eta_dram x BW) + t0 (the Ada census,
// Qwen3.5-9B Q4_K_M plain decode on the RTX 4090, 4070 Ti and 4060 Ti, every kernel within 2 %:
// t0 1.5 us without PDL, eta_dram 0.946). A round pays
// t0 for each kernel its graphs launch (Measurement::graph_kernels, counted from the captured
// graphs), a plain step for each of its own: R(k) adds them, in bytes at eta_dram x BW (the
// bytes term's convention; kappa converts operations at the full BW, ~5 % apart, immaterial here).
// Kernel nodes only (a memset or copy node is not counted), and kernels launched outside the graphs
// (a plain multi-step's token feed between replays) are not counted. Ampere and Turing have no row:
// their graph gaps were measured (RTX 2080 Ti 0.38 us a kernel, RTX 3060 ~0) but no intercept was
// fitted, and each has one lab card (a generation constant rests on two, as kSmallTEta does).
// gb20x has none either: PDL chains its launches (t0 about 0.5 us), but no intercept was fitted
// there, and its two lab cards' counted spec-tunes (RTX 5070 Ti, 5090) cannot tell 0.5 from
// 1.5 us (R within +-1.5 % either way, the verify routes dominating the residuals).
struct LaunchTime {
    std::string_view generation;
    double seconds;
    double eta_dram;
};
inline constexpr LaunchTime kLaunchTime[] = {
    {"ada", 1.5e-6, 0.946},
};

// The kernels a round's graphs launch and a plain step's (0: not counted, no launch term).
struct RoundLaunches {
    double round = 0;
    double step  = 0;
    [[nodiscard]] bool known() const noexcept { return round > 0 && step > 0; }
};

// A card's rates as the round cost reads them.
struct CardRates {
    double dram_bytes_per_s = 0;
    double int8_ops_per_s   = 0; // the generation's int8 tensor rate x SMs x reference clock
    std::string generation;      // core/generation.h's row name
    double eta              = 0; // kSmallTEta's, 0 when the generation has none
    std::int32_t eta_cards  = 0;
    double launch_seconds   = 0; // kLaunchTime's, 0 when the generation has none
    double eta_dram         = 0;

    // A kernel launch's time as the bytes it would read in it (round cost's unit); 0 when unknown.
    [[nodiscard]] double launch_bytes() const noexcept {
        return launch_seconds * eta_dram * dram_bytes_per_s;
    }

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

// R(k) above for configuration `config` on a card of `kappa`, the draft head read over the rows
// `head_rows` names for its mode (the full head where none, or where the head's facts list no head
// linear to scale); nullopt when the facts have no drafter of its mode, or the configuration
// drafts nothing.
// With `launches` known and `launch_bytes` > 0 (CardRates::launch_bytes), the round's kernel
// launches and the plain step's are added (each launch_bytes).
// `streams` requests decode together (#303): their round verifies streams x (k + 1) rows in one
// pass and drafts streams x the drafter's rows at once, against a plain step of streams rows; the
// launches are one round's either way.
[[nodiscard]] std::optional<double> round_cost(const RoundFacts& facts, const SpecConfig& config,
                                               double kappa, const ProposalRows& head_rows = {},
                                               double launch_bytes          = 0,
                                               const RoundLaunches& launches = {},
                                               std::int32_t streams         = 1);

// The plain decode step of `streams` requests decoded together, in bytes-equivalent (the
// denominator of round_cost at `streams`); its ratio to one request's is the step's price at load.
[[nodiscard]] double plain_step_cost(const RoundFacts& facts, double kappa, std::int32_t streams);

// The verify rows within which the round cost was measured against the cards (DECISIONS 53, and
// kSmallTEta's fit): from 12 rows verify kernels change tiles. A joint verify of more rows is not
// priced.
inline constexpr std::int32_t kPricedVerifyRows = 8;

// A drafter's gain over plain decode at `streams` streams, from its one-request gain (#303): the
// tokens a round yields are the pair's at any load, so the gain scales by the one-stream round's
// price over the joint one's, speedup x R_1(k) / R_C(k); one stream's is `speedup` itself. At
// two or more streams, nullopt where a joint verify exceeds
// kPricedVerifyRows, where the drafter's joint pass is wider than both that and its own one-stream
// pass (DFlash's block of 16 rows at C >= 2: tiles change past ~12 rows), or either round is not
// priced. An MoE's experts are read as different streams route them, apart (moe_read_fraction).
// The KV and attention bytes and each stream's recurrent state, which grow with C in both, are
// left out as in R_1 (the GDN state is ~3 % of an MoE step; at long context the gain is priced
// low); so are the requests' turnover between steps, which a load cell's tok/s holds.
[[nodiscard]] std::optional<double> load_gain(const RoundFacts& facts, const SpecConfig& config,
                                              double kappa, double speedup, std::int32_t streams,
                                              const ProposalRows& head_rows = {},
                                              double launch_bytes          = 0,
                                              const RoundLaunches& launches = {});

} // namespace ninferno::spec
