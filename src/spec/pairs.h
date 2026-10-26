#pragma once

// Speculation measured per pair: a target's weights and the drafter that drafts for them, as the
// lab's spec-tune measured them on one card. A profile (spec/profile.h) is built from a pair's
// measurements for the card a model runs on; nothing is estimated from the card's rates and no
// other model's profile stands in.
//
// What carries from the lab card to another (measured on four cards, four families):
//   - tokens per round (E): the pair's own, on any card (within 2.6 %);
//   - a round's price in plain decode steps (R): on any card while a round verifies at most
//     kPortableDepth drafts of one request; deeper drafts and rounds of several requests are the
//     generation's (Ada and Blackwell part at k >= 8 and under load), so they are taken only where
//     two lab cards of the card's generation measured the pair and chose alike (a lab constant
//     holds on two cards of a generation: docs/maintainer/deterministic-decisions.md).
//
// A target nobody measured takes its group's choice (profile_for_group): the pairs of the same
// engine and drafter kind, of the same bit class (below 4 bits drafts pay less), measured on two
// lab cards or more, the depth that is best on average over them, no deeper than kPortableDepth,
// one request at a time.

#include "spec/profile.h"

#include <nlohmann/json.hpp>

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

// The profile of a target from its pairs measured on the lab's cards (the same target and drafters):
// on a card of a generation two distinct lab cards (their tunes' GPU class) measured and chose
// alike for (each default's
// configuration), their profile with every measured depth, and their load bounds when they are
// equal (else one request at a time); otherwise the portable one (profile_for_pair away from home)
// from a measurement of this generation if there is one, else the first. Pairs another engine
// version measured are left out; none left: plain decode, disabled with the reason.
[[nodiscard]] Profile profile_for_target(const std::vector<const MeasuredPair*>& measured,
                                         std::string_view generation,
                                         const SelectOptions& options = {});

// The group of a target nobody measured: the current pairs (pair_current) of `engine` drafting with
// `kind` (a drafter of that kind, not only of its mode), each target once per card, of the target's
// bit class (below 4 bits, or 4 and above) when any is, else of the other. Its choice is a lab
// constant: empty unless the group's pairs were measured on two cards or more.
[[nodiscard]] std::vector<const MeasuredPair*> pair_group(const std::vector<MeasuredPair>& pairs,
                                                          std::string_view engine,
                                                          std::string_view kind,
                                                          double target_bits);

// The choice for a target nobody measured, from its group: per default (overall, thinking,
// non-thinking) the configuration of that kind up to kPortableDepth, measured on every pair, with
// the highest mean speed relative to each pair's best; its floor the highest any pair measured for
// it. Plain decode where a pair's tokens per round scaled down by kFineTuneAcceptance would make it
// gain less than kPreliminaryGain on any pair. One request speculates (load: plain from two rows).
// nullopt for an empty group. With `support`, only the depths the serving Engine runs.
[[nodiscard]] std::optional<Profile> profile_for_group(const std::vector<const MeasuredPair*>& group,
                                                       std::string_view kind,
                                                       const EngineSupport* support = nullptr);

// The bits per weight a quantization label names ("Q4_K_M" 4, "UD-IQ2_M" 2, "MXFP4" 4, "BF16" 16):
// enough to tell the bit classes apart. 16 for "" (a checkpoint's own weights).
[[nodiscard]] double label_bits(std::string_view label);
// A converted target's quantization label (its provenance's "quant"; "" for a checkpoint).
[[nodiscard]] std::string artifact_quant(const std::filesystem::path& artifact);

// What the lab's pairs give a target carrying drafters of `kinds` (the kinds attached or built in):
// its own measurements over those drafters (profile_for_target) with basis "measured"; else, for a
// target of `repo` (org/name@commit, the repository revision its quantization was listed at), the
// measurements of the sibling quantization of that revision nearest in bits per weight within its
// bit class (choice/DECISIONS 48) with basis "preliminary" (another commit may hold other weights:
// no sibling, nor without a commit); else the best of its groups' choices (one per kind, the
// fastest overall) with basis "preliminary"; else nullopt. `source` names where it came from.
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
                                                    std::string_view repo = {});
// Whether two drafter identities (repo@revision) name one drafter: the same repository, and the
// same revision where both say one (either may be a prefix of the other). True when either is
// unknown ("").
[[nodiscard]] bool same_drafter(std::string_view a, std::string_view b);

// A fine-tune's drafts may be accepted less than its base model's: WaveCut's uncensored Qwen3.8-27B
// fine-tune took 2.90 tokens a round with the base's DFlash2 drafter against the base's 4.5-4.6,
// 0.535 of the drafts past the first. A group's choice must still pay at that acceptance.
inline constexpr double kFineTuneAcceptance = 0.535;
inline constexpr double kPreliminaryGain    = 1.05;

} // namespace ninferno::spec
