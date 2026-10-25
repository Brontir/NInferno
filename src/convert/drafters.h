#pragma once

// Attaching speculative drafters to a converted model. The drafter comes with the model:
//
//   1. builtin      MTP layers inside the checkpoint (config keys, mtp.* tensors): kept as the
//                   artifact's "mtp" component;
//   2. bundle       the checkpoint declares its drafters and speculation profiles itself, in
//                   ninferno_speculation.json (a registry "model" object without "match");
//   3. registry     otherwise the official registry (spec/registry.h) names them for the model's
//                   repository id, or for a base model the checkpoint's own model card declares;
//   4. none         no drafter: the engines fall back to model-free prompt lookup.
//
// Nothing is searched for and nothing is ranked on the user's machine: the plan is a pure
// function of the checkpoint and the registry. Each declared drafter is still checked against
// the target before it is used (architecture and kind, whether the target's engine runs it,
// vocabulary fingerprint or size, hidden size, target-layer taps, window, block size): a registry
// or bundle error must fail loudly, not run silently with near-zero acceptance.
//
// attach_drafters() fetches and converts each separately packaged drafter (Gemma 4 assistant,
// DFlash) into the hub cache's ninferno/<commit>/ directory, records companions (Qwen DFlash /
// DFlash2) as embedded when the target carries their component, and writes
// <target>.speculation.json (spec/manifest.h) with the drafters and the published profiles.

#include "convert/convert.h"
#include "convert/fetch.h"
#include "spec/registry.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::convert {

struct DrafterTarget {
    std::string repo;                 // hub id ("" for a local model without one)
    std::vector<std::string> lineage; // base models the checkpoint's model card declares
    std::string engine;               // gemma4 | qwen3_5 | "" (no engine family)
    std::string architecture;
    spec::Geometry geometry;
    std::uint64_t sliding_window = 0;
    std::string fingerprint;             // vocabulary fingerprint ("" when no tokenizer.json)
    bool builtin_mtp = false;            // the checkpoint carries MTP layers
    std::vector<std::string> components; // an artifact's components (text, mtp, dflash, ...)
    std::optional<nlohmann::json> declaration; // ninferno_speculation.json of the bundle
    std::filesystem::path checkpoint;   // local checkpoint directory, if any
    std::filesystem::path artifact;     // local artifact, if any
    // An artifact converted from a GGUF model, and the GGUF mapping's version it records
    // (provenance.gguf_mapper; "" when the mapping predates the versions).
    bool gguf_source = false;
    std::string gguf_mapper;
};

struct PlannedDrafter {
    spec::DrafterDecl decl;
    bool supported  = false; // the target's engine runs this kind
    bool compatible = false; // every static check passed
    std::vector<std::string> checks; // "ok ..." / "FAIL ..." / "note ..."
    std::string reason;              // why it is not usable ("" when usable)
    std::string status;   // embedded, converted, cached, attached, incompatible, failed: ...
    std::string artifact; // the converted drafter
    std::string artifact_id;
    std::string commit;
    // The registry entry's geometry (zero when it records none): a drafter published for it
    // that records no num_target_layers (a GGUF drafter) is checked against its layer count.
    spec::Geometry entry_geometry;
    // The target's engine runs this kind, but not for this model: the dense family drafts for the
    // models it was validated on (a registry entry for the dense engine whose geometry the
    // target has). A drafter given by the user (--drafter-artifact) runs anyway.
    bool unvalidated = false;
    // Its num_target_layers may be unrecorded, its taps alone checked against the target's
    // layers: a drafter the user gave (--drafter-artifact) or a bundle declares, which no
    // registry entry's geometry vouches for (a converted GGUF drafter records none).
    bool taps_only = false;
    // DrafterPlan::stale_target, carried by each drafter for check_drafter.
    std::string stale_target;
};

struct DrafterPlan {
    DrafterTarget target;
    std::string source;     // bundle | registry | builtin | none
    std::string entry;      // registry model id
    std::string matched_by; // declaration | repo | lineage | similar | "" (Registry::nearest)
    std::vector<PlannedDrafter> drafters;
    nlohmann::json profiles = nlohmann::json::array();
    std::vector<std::string> notes;
    // Why no drafter runs for this target whatever it is (not even one the user gives, nor its
    // built-in MTP layers): the target was converted wrong and must be converted again. A
    // Nemotron-H GGUF converted before nemotron_h_moe/2 (which records no gguf_mapper) at a depth
    // no nemotron_h model of the registry has counted the GGUF's MTP block as a trunk layer; a
    // target of a listed repository with another layer count than the entry's model is such a
    // target, or not that model. "" otherwise.
    std::string stale_target;

    [[nodiscard]] nlohmann::json to_json() const;
};

// Whether a safetensors index (model.safetensors.index.json) lists MTP layers (mtp.*): the
// weights, not config.json's declaration, say whether a checkpoint has them (a fine-tune saved
// without them keeps mtp_num_hidden_layers).
[[nodiscard]] bool index_lists_mtp(const nlohmann::json& index);

// Whether a sharded checkpoint has built-in MTP layers: the mtp.* tensors its index lists, or the
// NextN layers its config.json declares (numbered past the decoder's, they keep their
// declaration).
[[nodiscard]] bool sharded_checkpoint_mtp(const nlohmann::json& config,
                                          const nlohmann::json& index);

// Whether the checkpoint directory `dir` (its config.json `config`) has built-in MTP layers: what
// its weights hold where their listing is there (a model_mtp.safetensors part, the index, or the
// one weight file's header), else the declaration; NextN layers keep their declaration.
[[nodiscard]] bool checkpoint_mtp(const std::filesystem::path& dir, const nlohmann::json& config);

// The target as the converter sees it: a checkpoint directory (its README's base_model lineage
// and its ninferno_speculation.json are read from disk) or a .ninferno artifact.
[[nodiscard]] DrafterTarget describe_checkpoint(const std::filesystem::path& dir, std::string repo);
[[nodiscard]] DrafterTarget describe_artifact(const std::filesystem::path& artifact,
                                              std::string repo);
// An artifact converted from a GGUF model (which carries no checkpoint to plan from): the
// artifact, with the base model the GGUF names (general.base_model) as its lineage.
[[nodiscard]] DrafterTarget describe_gguf_artifact(const std::filesystem::path& artifact,
                                                   std::string repo,
                                                   const std::filesystem::path& gguf);

// The deterministic plan: builtin MTP, then the bundle's declaration, else the registry entry
// for the repository or a declared base model.
[[nodiscard]] DrafterPlan plan_drafters(DrafterTarget target,
                                        const spec::Registry& registry = spec::Registry::builtin());

// Checks one drafter checkpoint config (and its tokenizer fingerprint, if any) against the target.
void check_drafter(const DrafterTarget& target, const nlohmann::json& config,
                   const std::string& drafter_fingerprint, PlannedDrafter& drafter,
                   const spec::Registry& registry = spec::Registry::builtin());

struct AttachOptions {
    FetchOptions fetch;
    std::filesystem::path repository; // source tree (shortlist rankings)
    // Already converted drafters by declared name (maintainers, offline machines): used instead
    // of fetching; checked against the target like any other. A name the model does not declare
    // adds that drafter (its kind from its architecture).
    std::map<std::string, std::filesystem::path> drafter_artifacts;
    // The run's conversion settings, the base of each separate drafter's conversion: its
    // --drafter-quant (else the registry's "quant"), --threads, --max-file-bytes,
    // --round-tiny-scales and --maintainer-quantize.
    ConvertOptions conversion;
    std::function<void(const std::string&)> log;
};

// Companion drafters (Qwen DFlash/DFlash2) of the plan, fetched and checked so the target's
// conversion can include them: (component, snapshot directory).
std::vector<std::pair<std::string, std::filesystem::path>>
fetch_companions(DrafterPlan& plan, const AttachOptions& options);

// Fetches, checks and converts the separate drafters, records builtin and companion ones, and
// writes <target_artifact>.speculation.json, a drafter that failed (an unreachable hub, an
// offline cache without its file) listed as failed: the others serve meanwhile. Returns the
// updated plan.
DrafterPlan attach_drafters(DrafterPlan plan, const std::filesystem::path& target_artifact,
                            const AttachOptions& options);

// Whether attaching to `target_artifact` is still to do: no manifest, an unreadable one, one planned
// from another drafter registry than this build's (an update may name a new drafter), or one
// listing a published drafter as failed (its own quantizations do not count).
[[nodiscard]] bool drafters_incomplete(const std::filesystem::path& target_artifact);

// A shortlist ranking by file name: tools/spec/rankings (and tools/freq_corpus/fixtures/ranking) of
// `repository`, then the rankings installed beside this binary, then the tree it was built from.
[[nodiscard]] std::optional<std::filesystem::path>
resolve_ranking(const std::string& name, const std::filesystem::path& repository);

// How a converted drafter's provenance names a registry ranking: "<name> sha256:<hex>", the same
// wherever the file lay.
[[nodiscard]] std::string ranking_label(const std::string& name, const std::filesystem::path& path);

// The bytes a companion drafter (converted into the target's artifact, DrafterDecl::component)
// takes there, from its published BF16 checkpoint's `published_bytes`, as the converter stores it:
// - a declared quantization (`declared_quant`, the registry drafter's "quant": convert.cpp
//   quantize_drafter) replaces every plain projection: q8 Q8_G32_FP16 (8.5 bits per weight), q4
//   Q4_G64_FP16 (4.25), nvfp4 (4.5 with its FP8 scales);
// - else a Qwen 3.5 / 3.6 target's DFlash and DFlash2 projections take Q8 in every recipe
//   (qwen3_5.cpp companion_drafter_q8), DFlash2's dynamic-convolution kernel projections and
//   candidate selector keeping BF16 (10.1 % of z-lab/Qwen3.8-27B-DFlash2's bytes: 0.362 of 3.584
//   GiB, its safetensors header);
// - any other: as published.
[[nodiscard]] std::uint64_t companion_drafter_bytes(std::string_view target_model_type,
                                                    std::string_view kind,
                                                    std::string_view declared_quant,
                                                    std::uint64_t published_bytes) noexcept;

} // namespace ninferno::convert
