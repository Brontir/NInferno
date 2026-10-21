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
#include <string>
#include <vector>

namespace ninferno::spec {

struct ManifestDrafter {
    std::string name, kind, repo, revision, artifact, artifact_id, component, status;
    [[nodiscard]] bool usable() const noexcept {
        return status == "embedded" || status == "converted" || status == "cached" ||
               status == "attached";
    }
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
    // The profile was measured on another GPU (spec::measured_elsewhere): its watchdog floors are
    // that GPU's, so the engine calibrates them here (SpeculativeOptions::calibrate_floor).
    bool foreign_floors = false;

    [[nodiscard]] bool found() const noexcept { return !source.empty(); }
};

// `artifact_id`: the target's id (hex); `gpu`: this device (a local tune must carry its full class
// id, gpu.chain.front()).
[[nodiscard]] ResolvedSpeculation resolve_speculation(const std::filesystem::path& artifact,
                                                      const std::string& artifact_id,
                                                      const GpuKey& gpu);

} // namespace ninferno::spec
