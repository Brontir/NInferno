#pragma once

#include "text/token_codec.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::text {

struct AddedToken {
    int id = -1;
    std::string content;
    bool special     = false;
    bool single_word = false;
    bool lstrip      = false;
    bool rstrip      = false;
    bool normalized  = false;
};

// A Hugging Face tokenizer.json BPE tokenizer whose encode and decode are byte-identical to
// the Rust `tokenizers` library (0.23): Tokenizer.from_file(...).encode(text,
// add_special_tokens).ids and .decode(ids, skip_special_tokens).
//
// Supported pipeline components; the constructor throws std::invalid_argument naming any other
// component type or option:
//   normalizer:     null, Sequence, NFC, NFKC, Lowercase, Prepend, Replace (String or Regex)
//   pre_tokenizer:  null, Sequence, Split (String or Regex; Isolated, Removed,
//                   MergedWithPrevious, MergedWithNext, Contiguous; invert), ByteLevel
//                   (add_prefix_space, use_regex), Metaspace (prepend_scheme always, first,
//                   never; split), Digits (individual_digits), Punctuation (behavior)
//   model:          BPE (merges as "a b" strings or [a, b] pairs, byte_fallback, ignore_merges,
//                   unk_token, fuse_unk; no dropout, empty continuing_subword_prefix and
//                   end_of_word_suffix)
//   added_tokens:   special and non-special, lstrip, rstrip, single_word, normalized
//   post_processor: null, TemplateProcessing (single), ByteLevel, Sequence of those
//   decoder:        null, ByteLevel, Metaspace, Replace, ByteFallback, Fuse, Strip, Sequence
//   truncation, padding: ignored, as transformers does by default (encode never truncates or
//                   pads; the engine enforces the context limit itself).
// Regex patterns go through text::Pattern, which refuses constructs outside its subset.
//
// Instances are immutable after construction and safe to share between threads.
class Tokenizer {
public:
    // tokenizer_config_json is optional; when given, its bos/eos/pad/unk token names are
    // resolved to ids.
    explicit Tokenizer(std::string_view tokenizer_json,
                       std::string_view tokenizer_config_json = {});

    // Throws std::invalid_argument when text is not valid UTF-8.
    [[nodiscard]] std::vector<int> encode(std::string_view text, bool add_special_tokens) const;

    // Encodes without special-token post-processing and reports where each byte boundary falls
    // in the token stream. A boundary is exact when the tokens of the text before it are a prefix
    // of the whole encoding (always at an added-token edge; inside an ordinary run only when the
    // run encodes the same split there). Added tokens that overlap a literal span are not
    // recognised. max_tokens truncates the ids.
    [[nodiscard]] BoundaryEncodedText
    encode_with_boundaries(std::string_view text, std::span<const std::size_t> byte_boundaries,
                           EncodeOptions options                    = {},
                           std::span<const ByteSpan> literal_spans = {}) const;

    // The bytes one token contributes to an output stream: its string through the token-local
    // decoder steps (ByteLevel, ByteFallback, Metaspace and Replace per token; Fuse and Strip are
    // sequence-level and skipped). A byte-level or byte-fallback token may be a partial UTF-8
    // sequence. Empty for an id outside the vocabulary.
    [[nodiscard]] std::string_view token_bytes(int id) const noexcept;
    // An id that the vocabulary or the added tokens define.
    [[nodiscard]] bool is_valid_token(int id) const noexcept;

    // Ids outside the vocabulary are dropped, as the library does. Invalid byte sequences
    // decode to U+FFFD the way the configured decoder produces them.
    [[nodiscard]] std::string decode(std::span<const int> ids, bool skip_special_tokens) const;

    // Added tokens first, then the model vocabulary.
    [[nodiscard]] std::optional<int> token_id(std::string_view token) const;
    // The string of an id as the decoder sees it (added-token content or vocab entry).
    [[nodiscard]] std::optional<std::string> id_to_token(int id) const;

    // One past the largest id of the model vocabulary and the added tokens.
    [[nodiscard]] std::size_t vocab_size() const noexcept;

    // Sorted by id.
    [[nodiscard]] std::span<const AddedToken> added_tokens() const noexcept;
    [[nodiscard]] bool is_special_token(int id) const noexcept;

    // From tokenizer_config.json, when given and when the named token exists.
    [[nodiscard]] std::optional<int> bos_token_id() const noexcept;
    [[nodiscard]] std::optional<int> eos_token_id() const noexcept;
    [[nodiscard]] std::optional<int> pad_token_id() const noexcept;
    [[nodiscard]] std::optional<int> unk_token_id() const noexcept;

    struct Impl;

private:
    std::shared_ptr<const Impl> impl_;
};

} // namespace ninferno::text
