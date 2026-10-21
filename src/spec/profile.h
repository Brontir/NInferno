#pragma once

// Speculation profiles: what spec-tune measured for one (target artifact, drafters, GPU, engine
// version) and which configuration it chose, written as <artifact>.spec_profile.json next to the
// artifact and read by the engines at start. The tune measures the corpus (spec/corpus.h) in its
// sections (math, code, prose, thinking): the sections only make up the mix every choice is
// judged on; no section gets a configuration of its own, since a server does not know what a
// request holds.
//
// Selection rules (select()):
//   overall        the configuration with the lowest total seconds-per-token summed over every
//                  section (each section weighs the same), measured on all of them and no slower
//                  than plain decode on any of them; plain decode unless it beats plain by at
//                  least min_gain
//   non_thinking   the same over the math, code and prose sections
//   thinking       the same over the thinking section (thinking is the request's own flag)
//   a drafter      accepted when one of its configurations beats plain decode by min_gain over
//                  the whole mix; otherwise rejected (a wrong or useless drafter), and none of its
//                  configurations is chosen
// Every choice carries its watchdog floor: the tokens per round at which its rounds cost as
// much as plain decode of what they emit (seconds per round / plain seconds per token).

#include "spec/config.h"
#include "spec/engine_support.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::spec {

// Bumped whenever an engine change moves speculation speed or acceptance enough to invalidate
// earlier profiles (verify routes, drafter kernels, round structure).
inline constexpr const char* kSpecEngineVersion = "ninferno-spec-1";

struct Measurement {
    SpecConfig config;
    std::string family; // the corpus section measured (part of the mix, never a choice of its own)
    std::int64_t prompts  = 0;
    std::int64_t tokens   = 0; // decoded tokens (the timed part)
    double seconds        = 0; // decode wall time
    std::int64_t rounds   = 0;
    std::int64_t drafted  = 0;
    std::int64_t accepted = 0;
    // Greedy parity with plain decode over the same prompts: tokens compared and the identical
    // prefix before the first difference (summed over prompts).
    std::int64_t compared = 0, identical = 0;

    [[nodiscard]] double tok_s() const noexcept { return seconds > 0 ? tokens / seconds : 0.0; }
    [[nodiscard]] double tokens_per_round() const noexcept {
        return rounds > 0 ? double(accepted + rounds) / double(rounds) : 1.0;
    }
    [[nodiscard]] double acceptance() const noexcept {
        return drafted > 0 ? double(accepted) / double(drafted) : 0.0;
    }
};

struct Choice {
    SpecConfig config;
    std::string drafter; // the drafter (by its declared name) the configuration runs
    double tok_s            = 0;
    double plain_tok_s      = 0;
    double tokens_per_round = 1;
    double acceptance       = 0;
    double floor_tokens_per_round = 0; // watchdog floor (0: plain decode)

    [[nodiscard]] double speedup() const noexcept {
        return plain_tok_s > 0 ? tok_s / plain_tok_s : 1.0;
    }
};

// How many columns of its block a DFlash drafter computes in a round (the Engine's
// DFlashDraftWidth): Default leaves it to the Engine (the whole block for a launch of one row,
// the anchor and the drafts from two rows on), Block every column the drafter was trained on,
// Drafts the anchor and the round's drafts only. Only a DFlash round has a width.
enum class DraftWidth : std::uint8_t { Default, Block, Drafts };
// "block" and "drafts" ("" for Default).
[[nodiscard]] std::string_view draft_width_name(DraftWidth width) noexcept;
// The inverse for "block" and "drafts"; nullopt otherwise.
[[nodiscard]] std::optional<DraftWidth> draft_width_from(std::string_view name) noexcept;

// A bound on speculation under decode load, measured by spec-tune --load: a decode step of at
// least `lanes` rows verifies at most `depth` drafts per row, and at depth 0 its rows decode
// plainly (the Engine's SpeculativeOptions::load). A DFlash round whose launch has at least
// `lanes` rows (the rows its graphs were captured for, not the rows a step happens to run) drafts
// at `width`; a bound of depth 0 has none. A profile lists them by ascending lanes; a step of
// fewer rows than the first entry is not bounded, so one request runs as the profile's choices
// say.
struct LoadBound {
    std::int32_t lanes = 0;
    std::int32_t depth = 0;
    DraftWidth width   = DraftWidth::Default;

    friend bool operator==(const LoadBound&, const LoadBound&) = default;
};

// "2:7,3:3,5:0" (lanes:depth, lanes ascending from 2: one request is never bounded), a DFlash
// bound's width after a second colon ("2:3:drafts,4:3:block"); "" or "none": no bound. No nonzero
// depth may follow a bound of 0 (a row turned plain cannot resume), and a bound of 0 has no width.
// Throws on malformed text.
[[nodiscard]] std::vector<LoadBound> parse_load(std::string_view text);
// The inverse of parse_load ("none" when empty).
[[nodiscard]] std::string load_label(const std::vector<LoadBound>& load);

// One concurrency spec-tune --load measured: the bound it chose (0: plain decode) and, for a
// DFlash drafter measured at both, the draft width.
struct LoadChoice {
    std::int32_t concurrency = 0;
    std::int32_t depth       = 0;
    DraftWidth width         = DraftWidth::Default;
};
// The table for choices measured at ascending concurrencies: the bound chosen at C applies from
// the lanes after the previous measured concurrency (the first from 2), so a step never runs
// deeper than measured for its load. An entry equal to the one before (depth and width) is
// dropped, and so are leading entries at `window` with no width (the engine's deepest depth: no
// bound); once a concurrency decodes plainly every higher one does too (a later nonzero choice is
// dropped).
[[nodiscard]] std::vector<LoadBound> load_table(const std::vector<LoadChoice>& choices,
                                                std::int32_t window);

struct DrafterRecord {
    std::string name;        // the declared name (registry / bundle), e.g. assistant, dflash
    std::string kind;        // gemma4_assistant, dflash, dflash2, mtp, ngram
    std::string mode;        // the SpecConfig mode its configurations use
    std::string artifact;    // path (separate drafters)
    std::string artifact_id; // hex
    std::string repo;
    std::string status;      // accepted | rejected | untested
    std::string reason;
    double best_speedup = 0;
    double acceptance   = 0; // pooled over its measured configurations
};

struct ProfileKey {
    std::string artifact_id;
    std::string drafters; // sorted drafter artifact ids joined by ',' ("" when none)
    std::string gpu;      // the device's full class id (core/device_class.h: "gb202:170sm")
    std::string engine = kSpecEngineVersion;

    friend bool operator==(const ProfileKey&, const ProfileKey&) = default;
};

struct RuntimeDisable {
    std::string reason;
    std::string when;
};

// The choices a server makes: the same selection over the configurations the serving Engine runs
// (spec/engine_support.h), from the same measurements.
struct ServedChoices {
    Choice overall, thinking, non_thinking;
};

struct Profile {
    ProfileKey key;
    std::string artifact;
    std::string created;
    std::string settings; // engine settings the tune ran with (context, chunk, KV codec)
    double tune_seconds = 0;
    Choice overall, thinking, non_thinking;
    // Absent when the tune did not know the serving Engine, or it runs every chosen configuration.
    std::optional<ServedChoices> served;
    // The choices for GPUs nobody measured (the registry's "*" profile, tools/spec/registry_add.py
    // --fallback): per default, the shallowest configuration within
    // SelectOptions::portable_within of the best, over what the serving Engine runs when the tune
    // knew it. A deep draft's margin depends most on the GPU's cost of verifying it (a 16 GB card
    // with a 3-bit target lost 17% to MTP k3 with the RTX 5090's mtp:10); a shallow one nearly as
    // fast here is the safer guess there. Absent when select() was asked for none.
    std::optional<ServedChoices> portable;
    // The configuration to run instead when the chosen one's drafter does not fit the requested
    // context beside the model on the device (the Engine refuses the options with CapacityError):
    // a smaller drafter measured on the same profile (the MTP head beside a DFlash2 drafter on a
    // 16 GB card). Taken only for choices from a profile, never for an explicit --spec, and always
    // logged. Absent: the refusal stands.
    std::optional<Choice> fit_fallback;
    // The bounds on its choices' depths under decode load (spec-tune --load), for every choice.
    // Empty: unbounded.
    std::vector<LoadBound> load;
    // The longest prompt its choices speculate on (SpeculativeOptions::context_limit), measured
    // where rounds stop paying at long context. 0: every length.
    std::uint32_t context_limit = 0;
    std::vector<DrafterRecord> drafters;
    std::vector<Measurement> measurements;
    std::vector<std::string> skipped; // configurations the time budget left out
    std::optional<RuntimeDisable> disabled;
};

struct SelectOptions {
    double min_gain = 0.03;
    // Profile::portable: the shallowest configuration within this fraction of the best (0: none).
    double portable_within = 0.02;
    // The serving Engine's support for this model: when given and some choice needs a
    // configuration it cannot run, select() also fills Profile::served.
    std::optional<EngineSupport> served;
};

// Fills overall, non_thinking and thinking, and every drafter's status (drafters are
// matched to measurements by `mode`); with options.served, also Profile::served.
void select(Profile& profile, const SelectOptions& options = {});

[[nodiscard]] nlohmann::json to_json(const Profile& profile);
[[nodiscard]] Profile profile_from_json(const nlohmann::json& doc);

[[nodiscard]] std::filesystem::path default_profile_path(const std::filesystem::path& artifact);
// nullopt when absent or unreadable (the reason in *error).
[[nodiscard]] std::optional<Profile> read_profile(const std::filesystem::path& path,
                                                  std::string* error = nullptr);
// Atomic (temporary file and rename).
void write_profile(const std::filesystem::path& path, const Profile& profile);
// `doc` as write_profile writes a profile: a temporary file renamed over `path` (removed when the
// write or the rename fails); throws when either fails.
void write_json_file(const std::filesystem::path& path, const nlohmann::json& doc);
// Empty when the keys agree, else which field differs.
[[nodiscard]] std::string key_mismatch(const ProfileKey& recorded, const ProfileKey& expected);
[[nodiscard]] std::string drafter_key(std::vector<std::string> artifact_ids);

// The configuration for a request: the thinking or non-thinking default when the request says,
// else the overall default; plain decode when the profile was disabled at run time.
[[nodiscard]] const Choice& pick(const Profile& profile, std::optional<bool> thinking);

// ---------------------------------------------------------------------------------------------
// Published profiles: the "profiles" list of a registry model entry or a bundle declaration, one
// object per (GPU class, applicable repositories), as the maintainer's spec-tune produced them:
//
//   {"gpu_class": "gb202", "gpu": "NVIDIA GeForce RTX 5090",
//    "applies_to": ["RedHatAI/gemma-4-26B-A4B-it-NVFP4"],     // empty or absent: any variant
//    "measured_on": "...", "source": "...",
//    "default": CHOICE, "non_thinking": CHOICE, "thinking": CHOICE,
//    "fit_fallback": CHOICE,                           // optional: Profile::fit_fallback
//    "load": [{"lanes": 2, "depth": 7}, {"lanes": 3, "depth": 3}, {"lanes": 5, "depth": 0}],
//    "load_measured_on": "...",                        // optional: Profile::load
//    "load_partial": true}  // optional: spec-tune --load left a concurrency undecided (not read)
//   CHOICE = {"config": "dflash:7", "drafter": "dflash", "floor_tokens_per_round": 1.9, ...}
//
// gpu_class is a key of core/device_class.h's class chain: a chip ("gb202", measured on one of its
// boards), a chip with its SM variant ("gb202:170sm", only where boards of the chip tune apart), an
// architecture ("sm120"), or "*", the fallback for GPUs nobody measured. A profile that truly
// differs by memory tier adds "mem": a range such as ">=24gib" or "<16gib", never an exact size.

// The device a published profile is chosen for.
struct GpuKey {
    // DeviceClass::similarity_chain(): id, chip, arch, the architecture's similar chips, "*", the
    // other architectures' chips (a plain class_chain() also works: id, chip, arch, "*").
    std::vector<std::string> chain;
    std::uint64_t memory_bytes = 0; // total device memory (0: unknown, no "mem" profile applies)
    bool nvfp4                 = true; // runs NVFP4 weights (.kind::mxf4nvf4); false on Ada, Ampere
    // How many leading keys of `chain` are the device's own (id, chip, arch); a profile found
    // past them is another card's standing in (0: every key counts as the device's own).
    std::size_t own_keys = 0;
    int sms              = 0; // multiprocessors (0: unknown, no stand-in load cap)
    // The rates the speculation load model prices a verify round with (0: unknown, no model).
    int compute_capability = 0;   // 89: Ada
    int sm_clock_khz       = 0;   // reference SM clock (DeviceClass::facts, not the driver's)
    double dram_gbps       = 0;   // nominal memory bandwidth (DeviceClass::facts)
    // mma.sync without tensor cores (DeviceFeatures::mma_slow, board_mma_slow): the verify MMA is
    // emulated, which the load model has no price for.
    bool mma_slow = false;
    // Whether the clock and bandwidth above are not the PCI-ID table's verified row (MachineFacts::
    // source other than Verified, or a driver value read: an unlisted id on the driver's clocks, a
    // laptop, a MIG slice).
    // The load model then prices the card but does not stand behind it.
    bool facts_extrapolated = false;
};

// Whether the published profile's class is one of the device's own keys, and if not, the note
// saying whose measurement stands in ("" when it is its own or the "*" fallback).
[[nodiscard]] std::string similarity_note(const GpuKey& gpu, const nlohmann::json& published);
// Whether `published` was measured on another GPU than this one: the "*" profile, or a class past
// the device's own keys. False when every key counts as the device's own (own_keys 0).
[[nodiscard]] bool measured_elsewhere(const GpuKey& gpu, const nlohmann::json& published);
// The load bounds of a profile another GPU measured, on a device of `sms` multiprocessors: from
// max(2, ceil(sms / 10)) rows on a step decodes plainly (a bound of depth 0 there, the bounds past
// it dropped). Speculation's verify work grows with the rows while a smaller GPU saturates sooner:
// Qwen3.6-35B on the RTX 4090's profile gained 11 % at 6 rows and lost 11 % at 8 on a 66-SM
// RTX 4070 Ti SUPER and 13 % at 8 on an 80-SM RTX 4080 SUPER, while the 128-SM RTX 4090 and the
// 170-SM RTX 5090 gained 29 % and 19 % at 8. Unchanged when sms is 0 or a plain bound already
// starts at or below the cap.
[[nodiscard]] std::vector<LoadBound> stand_in_load(std::vector<LoadBound> load, int sms);

// Whether `memory_bytes` satisfies a "mem" range ("", ">=24gib", ">16gib", "<=12gib", "<16gib").
[[nodiscard]] bool memory_in_range(std::string_view range, std::uint64_t memory_bytes);

// The published profile for a device and a repository: the classes of its chain, closest first;
// within a class, a profile naming the repository before one for any variant. A profile whose
// "requires" names a drafter not in `available` (when given), or an NVFP4 drafter variant on a
// device without NVFP4 (a profile published for "*" from a Blackwell tune), or whose "mem" range
// excludes the device, is skipped. nullptr when none applies.
// Among the profiles that apply at the first matching level, the one requiring the most drafters.
[[nodiscard]] const nlohmann::json* select_published(const nlohmann::json& profiles,
                                                     const GpuKey& gpu, std::string_view repo,
                                                     const std::vector<std::string>* available = nullptr);

// Where a published profile measured without load bounds takes them from, among the profiles of
// the same list with a "load" table whose default drafts with the same kind (an MTP table does not
// bound DFlash rounds) and that requires the same drafters: first a sibling on the same GPU class
// (another artifact of the model, measured on this hardware), else the "*" profile for any
// repository (bounds measured on one GPU class are the best guess for another, as its choices'
// floors are).
// nullptr when none has.
[[nodiscard]] const nlohmann::json* select_load_fallback(const nlohmann::json& profiles,
                                                         const nlohmann::json& selected);

// The choices of a published profile as a Profile (its defaults; no measurements).
[[nodiscard]] Profile profile_from_published(const nlohmann::json& published);

// A tuned profile as a published one, for the registry (tools/spec/registry_add.py merges it).
[[nodiscard]] nlohmann::json published_from_profile(const Profile& profile,
                                                    const std::string& gpu_class,
                                                    const std::string& gpu_name,
                                                    const std::vector<std::string>& applies_to,
                                                    const std::string& measured_on);

// ninferno spec-reselect: a saved tune selected again with this build's select() and published as
// spec-tune --publish would have. `served` is the serving Engine's support (--engine); a saved
// profile with served choices needs it (without it select() would publish, and the "*" fallback
// take, choices the Engine does not run). Throws std::invalid_argument for that, a profile without
// measurements, or an empty GPU class.
struct ReselectOptions {
    std::string gpu_class;
    std::string gpu_name;
    std::vector<std::string> applies_to;
    std::string measured_on;
    std::optional<EngineSupport> served;
};
[[nodiscard]] nlohmann::json reselect(Profile& profile, const ReselectOptions& options);

} // namespace ninferno::spec
