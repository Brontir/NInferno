#pragma once

// Engine auto-conversion: a Hugging Face checkpoint directory stands in for an artifact path.

#include <filesystem>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace ninferno::convert {

// Whether `path` is a checkpoint directory (config.json beside safetensors weights, or beside the
// cache of a checkpoint consumed by an earlier conversion) rather than an artifact.
[[nodiscard]] bool is_checkpoint_directory(const std::filesystem::path& path);

// A checkpoint directory, a GGUF file, or a directory holding one GGUF model.
[[nodiscard]] bool is_convertible(const std::filesystem::path& path);

// Whether a checkpoint directory's config.json describes a Vision encoder: a vision_config
// object (null or absent: none; an unreadable file: none).
[[nodiscard]] bool checkpoint_has_vision(const std::filesystem::path& checkpoint);

// The revision of the engine's cached conversions. Raise it when a converter change must reach
// models converted before it (a recipe's quantization table, a layout, a tokenizer fix): a cached
// artifact of an older revision is converted again where its source is still there, once, and
// kept where it is not (a download removed, a checkpoint consumed by --consume-source).
// 2: a GGUF hub model keeps its MTP layers and takes its companion drafters.
// 3: a gpt-oss (or other dense-converter) checkpoint plans its drafters for the dense engine.
inline constexpr int kCacheRevision = 3;

// The revision recorded beside `artifact` (<artifact>.revision); 1 where none is: converted before
// revisions were kept.
[[nodiscard]] int cached_revision(const std::filesystem::path& artifact);
// Records kCacheRevision beside a conversion just made.
void stamp_revision(const std::filesystem::path& artifact);
// Removes an artifact and its sidecars (<artifact>.conversion.json, .speculation.json, ...).
void remove_artifact(const std::filesystem::path& artifact);

struct CachedArtifact {
    std::filesystem::path path; // the artifact to load
    std::string recipe;         // empty for an artifact path
    bool converted = false;     // converted by this call
    double seconds = 0.0;       // conversion wall time when converted
};

// An artifact path is returned unchanged. A checkpoint directory maps to
// <dir>/.ninferno-cache/<recipe>.ninferno (a GGUF file to <its
// dir>/.ninferno-cache/<stem>.<recipe>.ninferno) under the official recipe for its architecture
// (choose_recipe), converted in-process first when missing; one converter at a time holds the cache
// directory's lock and the next one finds the artifact. `resources` supplies frontend files the
// checkpoint lacks (a drafter's tokenizer comes from its target); a file present in the checkpoint
// wins. `consume_source` converts with --consume-source (destroying the checkpoint's weights); only
// a caller the user explicitly asked for that may pass it. `vision` keeps a Qwen3.5 / 3.6 / 3.8
// model's Vision encoder (a checkpoint's own, or the mmproj-*.gguf beside a GGUF file) in its own
// artifact (<...>.vision.<recipe>.ninferno); a model without one converts as without it.
[[nodiscard]] CachedArtifact
cached_artifact(const std::filesystem::path& path,
                const std::vector<std::pair<std::string, std::filesystem::path>>& resources = {},
                const std::function<void(const std::string&)>& log                          = {},
                bool consume_source = false, bool vision = false);

} // namespace ninferno::convert
