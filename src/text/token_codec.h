#pragma once

#include "text/byte_span.h"

#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::text {

struct EncodeOptions {
    bool parse_added_tokens = true;
    std::size_t max_tokens  = std::numeric_limits<std::size_t>::max();
    // For a caller that only asks whether the text needs more than max_tokens ids (a prompt
    // against its context): the encoding may stop as soon as the text's bytes prove it does, and
    // the max_tokens ids it then returns are not the text's prefix. A 96 MiB single word took
    // 70 s and 5.7 GB to be found too long; with this it is found before its merges run.
    bool overflow_check_only = false;
};

struct DecodeOptions {
    bool skip_special_tokens = false;
    std::vector<int> stop_token_ids;
};

// The bytes one token contributes to an output stream, and whether it is a control token.
// Bytes may be a partial UTF-8 sequence; the consumer assembles code points across tokens.
struct DecodedTokenView {
    std::string_view bytes;
    bool special = false;
};

struct TokenBoundaryResult {
    // exact_frontier is present when the byte marker is also a boundary in the complete token
    // stream. stable_frontier retains only a normalization- and pre-tokenization-complete prefix.
    std::optional<std::size_t> exact_frontier;
    std::size_t stable_frontier = 0;
};

struct BoundaryEncodedText {
    std::vector<int> input_ids;
    std::vector<TokenBoundaryResult> boundaries;
};

// What a chat frontend needs from a tokenizer. The prompt path never adds special tokens: the
// chat template renders them. Implementations are immutable and shared between threads.
class TokenCodec {
public:
    virtual ~TokenCodec() = default;

    // Literal spans suppress added-token recognition (template-inserted user text cannot forge
    // control tokens) without splitting ordinary runs.
    [[nodiscard]] virtual BoundaryEncodedText
    encode_with_boundaries(std::string_view text, std::span<const std::size_t> byte_boundaries,
                           EncodeOptions options                    = {},
                           std::span<const ByteSpan> literal_spans = {}) const = 0;
    [[nodiscard]] virtual std::vector<int> encode(std::string_view text,
                                                  EncodeOptions options = {}) const {
        return encode_with_boundaries(text, {}, options).input_ids;
    }
    [[nodiscard]] virtual std::string decode(std::span<const int> ids,
                                             DecodeOptions options = {}) const = 0;
    // Throws std::out_of_range for an id outside the vocabulary.
    [[nodiscard]] virtual DecodedTokenView decoded_token(int id) const = 0;

    [[nodiscard]] virtual const std::vector<int>& default_stop_token_ids() const noexcept = 0;
    [[nodiscard]] virtual std::size_t vocab_size() const noexcept                       = 0;
    [[nodiscard]] virtual bool is_special_token(int id) const noexcept                  = 0;
    [[nodiscard]] virtual bool is_valid_token(int id) const noexcept                    = 0;
    [[nodiscard]] virtual bool has_exact_token_domain(std::size_t size) const noexcept  = 0;
    // The id of a token whose whole string is `content` (added tokens first), if any.
    [[nodiscard]] virtual std::optional<int> token_id(std::string_view content) const = 0;
};

} // namespace ninferno::text
