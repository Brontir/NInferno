#pragma once

// The source identity of an artifact (docs/maintainer/deterministic-decisions.md, principle 4): a
// digest of everything that can change its converted bytes, so that every conversion of the same
// sources by the same converter has the same identity on every machine. The artifact id is drawn
// at random per conversion, so a constant keyed by it applies to one file only; constants and the
// decision fingerprint key the artifact by this identity, and the artifact id stays an observed
// field.
//
// The identity is the SHA-256 (64 lowercase hex digits) of a record, as compact JSON with its keys
// sorted (nlohmann::json::dump()):
//
//   {"schema": "ninferno-source/1",
//    "inputs": the provenance's "inputs" (written by the converter): one entry per checkpoint and
//              resource the conversion read, in the order it opened them,
//              {"role": "base" | a --source name | "resource:<role>",
//               "files": [{"repo": "org/name", "commit": 40 hex, "name": its path in the
//                          repository, "sha256": 64 hex} for a Hugging Face file,
//                         {"sha256": 64 hex} for a local file (keyed by its content alone)]},
//              each list of files sorted (by its JSON);
//    "quant": the provenance's "quant" (a GGUF's quantization label, from its file name) or "",
//    "converter": the provenance's "converter" (the converter format version, "ninferno-v3"),
//    "recipe": the provenance's "recipe",
//    "choices": the provenance's other byte-changing choices that are present
//               (drafter_quantization, vocabulary, gguf_mapper, ranking),
//    "tensor_index": tensor_index_sha256 of the directory}
//
// An artifact converted before the converter recorded its inputs has no source identity
// (nullopt): what it was converted from cannot be proven, and a constant keyed by an identity
// reads as unknown for it.

#include "artifact/schema.h"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace ninferno::artifact {

inline constexpr std::string_view kSourceIdentitySchema = "ninferno-source/1";

// The SHA-256 (hex) of the directory's object index: one line per object in id order,
// "tensor\t<id>\t<format>\t<layout>\t<shape, comma-joined>\t<bytes>\t<divisors>\n" or
// "resource\t<id>\t<encoding>\t<bytes>\n". Offsets do not enter (the file split moves them).
// The artifact stores no per-object content hashes, so contents enter through the inputs.
[[nodiscard]] std::string tensor_index_sha256(const Directory& directory);

// The record the identity hashes; nullopt when the provenance records no "inputs" (or no
// converter or recipe).
[[nodiscard]] std::optional<Json> source_record(const Directory& directory);

[[nodiscard]] std::optional<std::string> source_identity(const Directory& directory);
// Reads the artifact's directory (no payload). Throws when the artifact cannot be read.
[[nodiscard]] std::optional<std::string> source_identity(const std::filesystem::path& artifact);

// Whether `text` is a source identity: 64 lowercase hex digits.
[[nodiscard]] bool is_source_identity(std::string_view text) noexcept;

// A file in a Hugging Face hub cache, <hub>/models--<org>--<name>/snapshots/<commit>/<path>:
// its repository, commit and path in the repository; nullopt for any other path. The path is read
// as written (a snapshot's files are symbolic links into blobs/; do not resolve it first).
struct HubFile {
    std::string repo, commit, name;
};

[[nodiscard]] std::optional<HubFile> hub_file(const std::filesystem::path& file);

// One "files" entry of an input: a hub file by repository, commit, path and content, any other by
// its content alone.
[[nodiscard]] Json input_file(const std::filesystem::path& file, const std::string& sha256);

} // namespace ninferno::artifact
