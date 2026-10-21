#pragma once

#include "text/bpe_merge_table.h"
#include "text/byte_span.h"
#include "text/token_codec.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ninferno::models::qwen3_5::frontend {

using text::DecodedTokenView;
using text::DecodeOptions;
using text::EncodeOptions;
using text::TokenBoundaryResult;
using text::BoundaryEncodedText;

struct AddedToken {
    int id = -1;
    std::string content;
    bool single_word = false;
    bool lstrip      = false;
    bool rstrip      = false;
    bool normalized  = false;
    bool special     = false;
};

struct TokenizerResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view generation_config_json;
};

using text::BpeMergeTable;

// Byte-level BPE with NFC normalization and the Qwen pre-tokenizer, specialized for speed: the
// Qwen3.5 Split regex or the Qwen2 one, as tokenizer.json declares.
class Tokenizer final : public text::TokenCodec {
public:
    explicit Tokenizer(TokenizerResources resources);

    [[nodiscard]] std::vector<int> encode(std::string_view text,
                                          EncodeOptions options = {}) const override;
    // Literal spans suppress added-token recognition without splitting ordinary NFC/BPE runs.
    [[nodiscard]] BoundaryEncodedText
    encode_with_boundaries(std::string_view text, std::span<const std::size_t> byte_boundaries,
                           EncodeOptions options                         = {},
                           std::span<const text::ByteSpan> literal_spans = {}) const override;
    [[nodiscard]] std::string decode(std::span<const int> ids,
                                     DecodeOptions options = {}) const override;
    [[nodiscard]] DecodedTokenView decoded_token(int id) const override;
    [[nodiscard]] std::string_view decode_token_bytes(int id,
                                                      bool skip_special_tokens = false) const;

    [[nodiscard]] const std::vector<int>& default_stop_token_ids() const noexcept override {
        return default_stop_token_ids_;
    }

    [[nodiscard]] std::size_t vocab_size() const noexcept override {
        return decoded_token_bytes_.size();
    }

    [[nodiscard]] bool is_special_token(int id) const noexcept override;
    [[nodiscard]] bool is_valid_token(int id) const noexcept override;
    [[nodiscard]] bool has_exact_token_domain(std::size_t size) const noexcept override;
    [[nodiscard]] std::optional<int> token_id(std::string_view content) const override;

private:
    std::vector<std::string> decoded_token_bytes_;
    std::vector<bool> valid_token_ids_;
    std::vector<bool> special_token_ids_;
    std::unordered_map<std::string, int> vocab_token_to_id_;
    BpeMergeTable bpe_merge_rules_;
    std::array<int, 256> byte_token_ids_{};
    // tokenizer.json declares the Qwen2 Split regex: combining marks outside letter runs.
    bool qwen2_split_ = false;
    // The most input bytes one BPE token covers (EncodeOptions::overflow_check_only).
    std::size_t max_token_bytes_ = 0;
    std::vector<AddedToken> added_tokens_;
    std::array<std::vector<std::size_t>, 256> added_token_candidates_;
    std::vector<int> default_stop_token_ids_;
};

} // namespace ninferno::models::qwen3_5::frontend
