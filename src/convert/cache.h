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
// 4: qwen3_5_published gives a DFlash drafter's projections Q8 (its fused QKV load refused the
//    BF16 ones); only artifacts with such a drafter are converted again (cache_current).
// 5: every recipe does (qwen3_5_bf16, the dense Qwen3.5 GGUF default, left the companion BF16);
//    again only artifacts with such a drafter are converted again.
// 6: a Qwen target with a DFlash companion takes a proposal head by default (its draft head over
//    the ranked rows default_proposal chooses); only Qwen artifacts with such a companion and no
//    proposal head are converted again, where the default ranking is found.
inline constexpr int kCacheRevision = 6;

// Whether an artifact carries the defect revisions 4 and 5 fixed: its report binds a DFlash /
// DFlash2 companion drafter's attention query unquantized (bf16, fp16, fp32).
[[nodiscard]] bool unquantized_drafter_attention(const std::filesystem::path& artifact);
// Whether an artifact lacks what revision 6 added: its report is a Qwen target's (qwen3_5_text,
// qwen3_5_moe_text) binding a DFlash companion and no proposal head, and the default ranking is
// found (without it the conversion adds none).
[[nodiscard]] bool companion_without_proposal(const std::filesystem::path& artifact);

// Whether a cached artifact is served as it is: revision kCacheRevision, or revision 3 or 4
// without the defect revisions 4 and 5 fixed nor revision 6's addition (a text-only artifact, a drafter already in Q8): a
// conversion of many minutes is not repeated for a change that did not touch it. Such an older
// artifact is stamped kCacheRevision (stamp_revision), so its report is read once.
[[nodiscard]] bool cache_current(const std::filesystem::path& artifact);
// cache_current's answer, read only: it never stamps (a plan's, which only reads the cache).
[[nodiscard]] bool cache_serves(const std::filesystem::path& artifact);

// The revision recorded beside `artifact` (<artifact>.revision); 1 where none is: converted before
// revisions were kept.
[[nodiscard]] int cached_revision(const std::filesystem::path& artifact);
// Records kCacheRevision beside a conversion just made.
void stamp_revision(const std::filesystem::path& artifact);
// What a conversion read from its source, cheaply: each file's name (from `root`, the model's
// directory; a file outside it by its full path), size and modification time, one line each in
// name order (a file that is not there has none). A checkpoint replaced at the same path (a newer
// revision copied over it, a re-download) changes it; another spelling of the path does not.
[[nodiscard]] std::string source_stamp(const std::filesystem::path& root,
                                       std::vector<std::filesystem::path> files);
// Records `stamp` beside a conversion just made (<artifact>.source).
void stamp_source(const std::filesystem::path& artifact, const std::string& stamp);
// Whether `stamp` names a file the one recorded beside `artifact` does not, or one of another size
// or time, or lacks a recorded file other than weights (weights that are gone are no change:
// --consume-source removed them). An artifact converted before stamps were kept has none: it is
// stamped `stamp` now (a read-only cache stays unstamped) and is not changed.
[[nodiscard]] bool source_changed(const std::filesystem::path& artifact, const std::string& stamp);
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
// artifact (<...>.vision.<recipe>.ninferno); a model without one converts as without it. A cached
// artifact whose source files changed since it was converted (source_stamp) is converted again
// while the source is there; one whose source is gone is served as it is. `round_tiny_scales`
// converts with --round-tiny-scales (ConvertOptions::round_tiny_scales).
[[nodiscard]] CachedArtifact
cached_artifact(const std::filesystem::path& path,
                const std::vector<std::pair<std::string, std::filesystem::path>>& resources = {},
                const std::function<void(const std::string&)>& log                          = {},
                bool consume_source = false, bool vision = false, bool round_tiny_scales = false);

} // namespace ninferno::convert
