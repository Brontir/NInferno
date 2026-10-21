#pragma once

// Vocabulary fingerprints: two models whose fingerprints agree map every token id to the same
// token text, so a drafter's proposals mean to the target what they meant to the drafter.
//
//   fingerprint = "v1:" + the first 16 hex digits of sha256 over one line per token id,
//                 "<id>\t<token>\n", in ascending id order; the token text of an id is its entry
//                 in tokenizer.json's model.vocab (a map token -> id, or a list of [token, score]
//                 pairs indexed by id), replaced by its added_tokens entry when it has one.
//
// tools/spec/tokenize_corpus.py computes the same value in Python.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace ninferno::spec {

struct VocabFingerprint {
    std::string value;        // "v1:0123456789abcdef"
    std::uint64_t tokens = 0; // ids with a token text
    std::uint64_t max_id = 0; // largest id + 1
};

// Throws when the document has no model.vocab.
[[nodiscard]] VocabFingerprint vocab_fingerprint(const nlohmann::json& tokenizer);
[[nodiscard]] VocabFingerprint vocab_fingerprint_text(std::string_view tokenizer_json);
[[nodiscard]] VocabFingerprint vocab_fingerprint_file(const std::filesystem::path& tokenizer_json);

// The tokenizer.json an artifact carries as a resource (the "text" component first, then any
// other), parsed; nullopt when it carries none.
[[nodiscard]] std::optional<nlohmann::json>
artifact_tokenizer_json(const std::filesystem::path& artifact);

// The fingerprint of that tokenizer.json; nullopt when the artifact carries none.
[[nodiscard]] std::optional<VocabFingerprint>
artifact_vocab_fingerprint(const std::filesystem::path& artifact);

// The artifact's 16-byte id as 32 lowercase hex digits (the conversion report's artifact_id).
[[nodiscard]] std::string artifact_id_hex(const std::filesystem::path& artifact);

} // namespace ninferno::spec
