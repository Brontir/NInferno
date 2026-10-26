#pragma once

// The speculation manifest of a converted model: <artifact>.speculation.json, written by
// ninferno-convert when it attaches the drafters a model bundle declares or the official registry
// names for it. It is the "drafter comes with the model" record the engines read at start:
//
//   {"schema": 1,
//    "target": {"artifact_id": ..., "repo": ..., "engine": ..., "vocab_fingerprint": ...},
//    "source": "bundle" | "registry" | "builtin" | "none",   "entry": registry model id,
//    "matched_by": "repo" | "lineage" | "similar" | "declaration",
//    "drafters": [{"name", "kind", "repo", "revision", "artifact", "artifact_id", "component",
//                  "status", "checks"}],
//    "profiles": [published profiles, see spec/profile.h],
//    "fallback": "ngram" | "none"}
//
// resolve_speculation() turns what lies next to an artifact into the configuration to run:
//   1. <artifact>.spec_profile.json, a maintainer's own spec-tune result, when its key (artifact
//      id, GPU class, engine version) matches;
//   2. the published profile for this GPU along its similarity chain (chip variant, chip,
//      architecture, the architecture's similar chips, the "*" fallback, then the most similar
//      measured GPU of another architecture: select_published, DeviceClass::similarity_chain;
//      a note names a profile that stands in), from the registry this binary
//      carries when the manifest names a registry entry it knows, else from the manifest;
//   3. nothing: the engine's model-free fallback (prompt lookup) or plain decode.

#include "spec/profile.h"

#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace ninferno::spec {

struct ManifestDrafter {
    std::string name, kind, repo, revision, artifact, artifact_id, component, status;
    [[nodiscard]] bool usable() const noexcept {
        return status == "embedded" || status == "converted" || status == "cached" ||
               status == "attached";
    }
    // Usable and its weights here: a component of the target's artifact, or its own artifact on
    // disk.
    [[nodiscard]] bool present() const;
};

struct SpeculationManifest {
    std::string target_artifact_id, repo, engine, vocab_fingerprint;
    std::string source, entry, matched_by, fallback;
    std::vector<ManifestDrafter> drafters;
    nlohmann::json profiles = nlohmann::json::array();

    [[nodiscard]] const ManifestDrafter* drafter(std::string_view name) const;
};

[[nodiscard]] std::filesystem::path manifest_path(const std::filesystem::path& artifact);
[[nodiscard]] std::optional<SpeculationManifest> read_manifest(const std::filesystem::path& path,
                                                               std::string* error = nullptr);
[[nodiscard]] SpeculationManifest manifest_from_json(const nlohmann::json& doc);

struct ResolvedSpeculation {
    std::string source; // "local tune <path>" | "manifest <path> (<gpu class>)" | ""
    // SHA-256 (lowercase hex) of what the profile was read from: the local tune's file, the
    // manifest's, or the drafter registry's (Registry::source_sha256); empty when none was found.
    std::string sha256;
    Profile profile;    // the choices (and, for a local tune, its drafter records)
    // Drafter name -> artifact path (separate drafters) or "component:<name>" (inside the target).
    std::map<std::string, std::string> drafters;
    std::vector<std::string> notes; // why a source was passed over
    std::string fallback = "none";
    // Where the profile came from, as data: "own" (this GPU's class, or a local tune), "similar"
    // (another GPU's standing in), "generic" (the "*" profile), "builtin" (the built-in
    // drafter's untuned default); "" when none was found.
    std::string basis;
    // The profile was measured on another GPU (spec::measured_elsewhere), or is a built-in
    // default measured nowhere: its floors are raised (foreign_floor, raise_foreign_floors).
    bool foreign_floors = false;
    // The model's published profiles (a JSON array: every GPU class it was measured on) whose
    // floors bound a foreign profile's; null for a built-in default (the registry median then).
    nlohmann::json floor_sources;
    // Why the tune named by `local_profile` does not serve (its path and the reason: unreadable,
    // tuned for another artifact, GPU or drafters, or a drafter replaced since); empty when it
    // serves or none was named. What resolves is then the published profile, which an entry point
    // must not run in its place (require_named_tune).
    std::string named_tune_refused;

    [[nodiscard]] bool found() const noexcept { return !source.empty(); }
};

// `artifact_id`: the target's id (hex); `gpu`: this device (a local tune must carry its full class
// id, gpu.chain.front()). `local_profile`: a maintainer's own tune (`ninferno spec-tune`'s output),
// read only when named (serve/CLI --spec-profile): one beside the artifact is never read by
// itself, so a machine's own measurement does not steer what it runs.
[[nodiscard]] ResolvedSpeculation resolve_speculation(const std::filesystem::path& artifact,
                                                      const std::string& artifact_id,
                                                      const GpuKey& gpu,
                                                      const std::filesystem::path& local_profile = {});

// The same resolution for a target not converted yet (a dry run: ninferno-serve --plan --no-load,
// ninferno-ui /api/plan): `manifest` the one the converter would write (convert::plan_drafters,
// each drafter a start would attach counted as attached), else the registry `entry` of the model's
// shape over the drafters it will carry as `components` (no manifest: a GGUF hub model without
// --with-drafters), else nothing. `label` names it in the sources (in place of the artifact). No
// local tune is read; the built-in drafter's default is the caller's (as for an artifact).
struct ModelEntry;
// The lab's pairs answer first (spec/pairs.h): by `weights` (the planned files' identity, "":
// unknown), the drafters the manifest attaches or the components carry, `quant`'s bit class and
// the `engine` family, and `repo` (org/name@commit: its sibling quantizations, spec::pair_choice); then
// the published profiles as above.
[[nodiscard]] ResolvedSpeculation
resolve_planned_speculation(const std::optional<SpeculationManifest>& manifest,
                            const ModelEntry* entry, const std::set<std::string>& components,
                            const GpuKey& gpu, const std::filesystem::path& label,
                            std::string_view weights = {}, std::string_view quant = {},
                            std::string_view engine = {}, std::string_view repo = {});

// Why the tune at `tune` would not serve the artifact `artifact_id` on GPU class `gpu_class` (the
// device's full class id, as resolve_speculation checks it; empty: the GPU is not checked, for a
// caller without a device such as the UI): unreadable, tuned for another artifact, GPU or
// drafters, or a drafter replaced since. Empty when it would serve. For a tune found rather than
// named (spec-tune --load's own beside the artifact): passed over with a note; for the one a UI
// start asks for (spec_profile): that start is refused.
[[nodiscard]] std::string tune_refusal(const std::filesystem::path& tune,
                                       const std::string& artifact_id,
                                       const std::string& gpu_class = {});

// A tune the user named (--spec-profile, spec-tune --load --profile) serves, or the entry point
// refuses to start: throws std::runtime_error naming the tune and why it does not serve
// (resolved.named_tune_refused). Nothing when it serves or none was named.
void require_named_tune(const ResolvedSpeculation& resolved);

// The largest break-even (floor_tokens_per_round) `profiles` (one model's published profiles,
// every GPU class) measured for exactly this configuration (mode, depth, proposal head): lab
// constants of the same model on other GPUs. 0 when none.
[[nodiscard]] double floor_ceiling(const nlohmann::json& profiles, const SpecConfig& config);

// The median break-even of this configuration over every model and GPU class of the registry: the
// floor of a model measured nowhere (a built-in default). 0 when the registry measured none.
[[nodiscard]] double registry_floor_median(const SpecConfig& config);

// The floor a profile measured elsewhere runs `config` at: its own, raised to the same model's
// largest on any GPU (floor_ceiling over resolved.floor_sources) or, for a model measured nowhere,
// to the registry median. `own` itself for a profile measured on this GPU.
[[nodiscard]] double foreign_floor(const ResolvedSpeculation& resolved, const SpecConfig& config,
                                   double own);

// The floor a "+head" `config` runs at drafting with the full head (the artifact has no proposal
// head): foreign_floor's, raised to the full head's ceiling over the model's floors (or the
// registry median) whichever GPU measured the profile, since `own` is the optimized head's, a
// cheaper round. foreign_floor's for a config without the head.
[[nodiscard]] double full_head_floor(const ResolvedSpeculation& resolved, const SpecConfig& config,
                                     double own);

// A profile measured on another GPU, or measured nowhere, carries no break-even of this one: each
// choice's floor is raised to foreign_floor (the conservative side: its watchdog trips sooner).
// A note names each raised choice.
void raise_foreign_floors(Profile& profile, const ResolvedSpeculation& resolved,
                          std::vector<std::string>& notes);

} // namespace ninferno::spec
