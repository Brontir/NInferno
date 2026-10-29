#pragma once

// Speculation measured per pair: a target's weights and the drafter that drafts for them, as the
// lab's spec-tune measured them on one card. A profile (spec/profile.h) is built from a pair's
// measurements for the card a model runs on; nothing is estimated from the card's rates and no
// other model's profile stands in.
//
// What carries from the lab card to another (measured on four cards, four families):
//   - tokens per round (E): the pair's own, on any card (within 2.6 %);
//   - a round's price in plain decode steps (R): computed for the card from the pair's facts
//     (spec/round_cost.h: the bytes each pass reads, the card's memory and tensor rates) while a
//     round verifies at most kPortableDepth drafts of one request, never below what the lab card
//     measured over its own facts (round_cost_estimate); deeper drafts and rounds of several
//     requests are measured only, taken where two lab cards of the card's generation measured the
//     pair and chose alike (a lab constant holds on two cards of a generation:
//     docs/maintainer/deterministic-decisions.md). A pair without facts, or a card whose rates are
//     unknown, takes the median R of the lab cards of its generation that measured the target,
//     else of every lab card that did, said as a transfer (profile_for_target).
//
// A target nobody measured takes its group's choice (profile_for_group): the pairs of the same
// engine and drafter kind, of the same bit class (below 4 bits drafts pay less), measured on two
// lab cards or more, the depth that is best on average over them, no deeper than kPortableDepth,
// one request at a time.

#include "spec/profile.h"
#include "spec/round_cost.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::spec {

// The deepest draft whose round price carries between GPU generations (one request).
inline constexpr std::int32_t kPortableDepth = 7;

struct MeasuredPair {
    // The target's weights: the SHA-256 of its weight files, sorted and comma-joined (the
    // checkpoint's or GGUF's own files, as the hub lists them), so a new commit that changes no
    // weight keeps the pair.
    std::string target;
    std::string target_repo; // org/name@commit: for people and for the download
    std::string engine;      // qwen3_5 | gemma4 | dense | nemotron_h
    double target_bits = 0;  // bits per weight of the target's weights (its quantization)
    // The lab card's architecture generation (DeviceClass's arch key: "sm89", "sm120").
    std::string generation;
    // The lab's spec-tune of the pair, as it wrote it: its key (engine version), drafters (kind,
    // mode, name, repo@commit of a separate one), measurements (one request at a time) and, from
    // spec-tune --load on that card, its load bounds.
    Profile tune;
    // The drafters measured, by kind: a separate drafter's repo@revision ("" or absent: one built
    // into the target, or not recorded). A target carrying another drafter of a kind is not
    // measured for it (pair_choice).
    std::map<std::string, std::string> drafters;
    // The weights' facts a round's cost is computed from (spec/round_cost.h; absent: none recorded,
    // the measured ratios stand) and the lab board's PCI product id ("lab_board": "0x2782"; 0:
    // not recorded), whose rates price the lab's own rounds by those facts.
    std::optional<RoundFacts> round_facts;
    std::uint32_t lab_board = 0;
};

[[nodiscard]] nlohmann::json pair_json(const MeasuredPair& pair);
// Throws std::runtime_error naming what is missing or malformed.
[[nodiscard]] MeasuredPair parse_pair(const nlohmann::json& doc);

// The identity of a set of weight files: their SHA-256 digests (lowercase hex), sorted and
// comma-joined, as MeasuredPair::target holds it; "" (unknown) for no files, or when a digest is
// missing (a file kept in git, not LFS).
[[nodiscard]] std::string weights_identity(std::vector<std::string> sha256s);
// A converted target's weights: its provenance's "base" input files (artifact/source_identity.h).
// nullopt for an artifact converted before the converter recorded its inputs.
[[nodiscard]] std::optional<std::string> artifact_weights(const std::filesystem::path& artifact);

// Whether the pair was measured by this build's speculation engine (its tune's key names
// kSpecEngineVersion): a bump of the version invalidates every earlier measurement.
[[nodiscard]] bool pair_current(const MeasuredPair& pair);

// The generation key of a device (its chain's architecture, "sm89"); "" when the chain names none.
[[nodiscard]] std::string gpu_generation(const GpuKey& gpu);

// The portable profile of one measurement: select() over its configurations of at most
// kPortableDepth drafts (and the serving Engine's support, when given), one request speculating
// and two or more decoding plainly. A pair another engine version measured gives plain decode,
// disabled with the reason.
[[nodiscard]] Profile profile_for_pair(const MeasuredPair& pair, const SelectOptions& options = {});

// How far a lab card's measured round may sit above its facts before the estimate is raised to it
// (round_cost_estimate; the two Ada cards agree within 1.5 % on the 9B MTP pair).
inline constexpr double kRoundCostAgreement = 0.05;
// What scales a pair's rounds, by configuration label (a configuration without one is left out).
struct RoundCostRatios {
    std::map<std::string, double> ratio; // by configuration label
    bool known = false;
};

// A round's cost on a card from the pair's facts (round_cost on card_rates), by configuration label,
// for the measured configurations of at most kPortableDepth drafts: R(k) in plain decode steps,
// raised to the lab card's measured R(k) x R_facts(card) / R_facts(lab card) where the lab card
// measured rounds more than kRoundCostAgreement dearer than their facts (a cost the bytes do not
// show: Gemma 4's verify, Q8_0's; conservative, said in `note`). `known` false when the pair has no
// facts, the card's or the lab board's rates are unknown, or no configuration is priced.
struct RoundCostEstimate {
    std::map<std::string, double> cost; // by configuration label, plain decode steps a round
    bool known = false;
    std::string note; // how it was priced, for the choice's source
};
[[nodiscard]] RoundCostEstimate round_cost_estimate(const MeasuredPair& pair, const GpuKey& gpu);

// What a round costs in a profile's measurements, by configuration label: per corpus section the
// seconds a round over plain decode's seconds a token, the mean over the sections both measured
// (plain decode steps a round). Of a profile priced by the facts (price_by_facts) it is their R(k);
// of a measured one, the measuring card's. The plan reports it (its speculation's "round_cost") so
// that a user's spec-tune can be held to it (tools/kit/validate.py).
[[nodiscard]] std::map<std::string, double> profile_round_costs(const Profile& profile);

// The profile of a target from its pairs measured on the lab's cards (the same target and drafters),
// the same whatever the order of `measured` (records ordered by GPU class id, then content):
//   1. With `gpu`, a card other than the measuring one prices its rounds by the facts
//      (round_cost_estimate) where they are known; `note` says so.
//   2. Two or more lab cards of the card's generation: their median round cost per configuration
//      (on the card's own record when it is one of them, else the first by GPU class id; tokens a
//      round are the pair's on any card); where they chose alike (each default's configuration),
//      every measured depth and their load bounds when equal (else one request at a time), else
//      the portable depths. `note`: "rounds: the median of N lab cards of <generation>".
//   3. One lab card of the generation: its measurement, the portable depths.
//   4. None, and no facts: the median round cost per configuration of every lab card that
//      measured the target, the portable depths; `note`: "transferred: median of N lab cards of
//      other generations (no facts for <generation>)".
// Pairs another engine version measured are left out; none left: plain decode, disabled with the
// reason.
[[nodiscard]] Profile profile_for_target(const std::vector<const MeasuredPair*>& measured,
                                         std::string_view generation,
                                         const SelectOptions& options = {},
                                         const GpuKey* gpu = nullptr, std::string* note = nullptr);

// The group of a target nobody measured: the current pairs (pair_current) of `engine` drafting with
// `kind` (a drafter of that kind, not only of its mode), each target once per card, of the target's
// bit class (below 4 bits, or 4 and above) when any is, else of the other. Its choice is a lab
// constant: empty unless the group's pairs were measured on two cards or more.
[[nodiscard]] std::vector<const MeasuredPair*> pair_group(const std::vector<MeasuredPair>& pairs,
                                                          std::string_view engine,
                                                          std::string_view kind,
                                                          double target_bits);

// The choice for a target nobody measured, from its group: per default (overall, thinking,
// non-thinking) the configuration of that kind up to kPortableDepth, measured on two cards or
// more, whose gain at a fine-tune's acceptance (each pair's drafted tokens accepted at
// kFineTuneAcceptance of its measured rate) is highest on average over the pairs that measured it;
// that gain its tok_s, its floor the highest any of them measured. Plain decode only where no
// configuration gains kPreliminaryGain so (choice/DECISIONS 49). One request speculates (load:
// plain from two rows).
// nullopt for an empty group. With `support`, only the depths the serving Engine runs.
[[nodiscard]] std::optional<Profile> profile_for_group(const std::vector<const MeasuredPair*>& group,
                                                       std::string_view kind,
                                                       const EngineSupport* support = nullptr);

// A target's bits per weight by its quantization label (quant_label_bits): enough to tell the bit
// classes apart. 16 for "" (a checkpoint's own weights) and a label that names none.
[[nodiscard]] double label_bits(std::string_view label);
// A drafter kind's speculation mode ("mtp", "gemma4_assistant", "nemotron_h_mtp" MTP; "dflash",
// "dflash2"); nullopt for a kind no pair drafts with.
[[nodiscard]] std::optional<Mode> drafter_kind_mode(std::string_view kind);
// A converted target's quantization label (its provenance's "quant"; "" for a checkpoint).
[[nodiscard]] std::string artifact_quant(const std::filesystem::path& artifact);

// What the lab's pairs give a target carrying drafters of `kinds` (the kinds attached or built in):
// its own measurements over those drafters (profile_for_target) with basis "measured"; else, for a
// target of `repo` (org/name@commit, the repository revision its quantization was listed at), the
// measurements of the sibling quantization of that revision nearest in bits per weight within its
// bit class (choice/DECISIONS 48) with basis "preliminary" (another commit may hold other weights:
// no sibling, nor without a commit); else the best of its groups' choices (one per kind, the
// fastest overall) with basis "preliminary"; else nullopt. `source` names where it came from (and,
// with `gpu`, how its rounds were priced on that card: profile_for_target).
struct PairChoice {
    Profile profile;
    std::string basis;
    std::string source;
};
[[nodiscard]] std::optional<PairChoice> pair_choice(const std::vector<MeasuredPair>& pairs,
                                                    std::string_view target,
                                                    std::string_view engine,
                                                    const std::vector<std::string>& kinds,
                                                    double target_bits, std::string_view generation,
                                                    const SelectOptions& options = {},
                                                    const std::map<std::string, std::string>&
                                                        attached          = {},
                                                    std::string_view repo = {},
                                                    const GpuKey* gpu     = nullptr);

// A fine-tune's drafts may be accepted less than its base model's: WaveCut's uncensored Qwen3.8-27B
// fine-tune took 2.90 tokens a round with the base's DFlash2 drafter against the base's 4.5-4.6,
// 0.535 of the drafts past the first. A group's choice must still pay at that acceptance.
// A group's choice is a lab constant for targets nobody measured: fitted on this many cards or more.
inline constexpr std::size_t kGroupCards = 2;
inline constexpr double kFineTuneAcceptance = 0.535;
inline constexpr double kPreliminaryGain    = 1.05;

} // namespace ninferno::spec
