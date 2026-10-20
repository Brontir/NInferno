#pragma once

#include "text/token_codec.h"
#include "text/tokenizer.h"

#include <span>
#include <string_view>
#include <vector>

namespace ninferno::text {

// TokenCodec over any Hugging Face tokenizer.json that text::Tokenizer accepts. The default stop
// ids come from generation_config.json eos_token_id (an id or a list), else the tokenizer's eos.
class HfTokenCodec final : public TokenCodec {
public:
    HfTokenCodec(std::string_view tokenizer_json, std::string_view tokenizer_config_json,
                 std::string_view generation_config_json);

    [[nodiscard]] BoundaryEncodedText
    encode_with_boundaries(std::string_view text, std::span<const std::size_t> byte_boundaries,
                           EncodeOptions options                    = {},
                           std::span<const ByteSpan> literal_spans = {}) const override;
    [[nodiscard]] std::string decode(std::span<const int> ids,
                                     DecodeOptions options = {}) const override;
    [[nodiscard]] DecodedTokenView decoded_token(int id) const override;
    [[nodiscard]] const std::vector<int>& default_stop_token_ids() const noexcept override {
        return stops_;
    }
    [[nodiscard]] std::size_t vocab_size() const noexcept override {
        return tokenizer_.vocab_size();
    }
    [[nodiscard]] bool is_special_token(int id) const noexcept override {
        return tokenizer_.is_special_token(id);
    }
    [[nodiscard]] bool is_valid_token(int id) const noexcept override {
        return tokenizer_.is_valid_token(id);
    }
    [[nodiscard]] bool has_exact_token_domain(std::size_t size) const noexcept override;
    [[nodiscard]] std::optional<int> token_id(std::string_view content) const override {
        return tokenizer_.token_id(content);
    }

    [[nodiscard]] const Tokenizer& tokenizer() const noexcept { return tokenizer_; }

private:
    Tokenizer tokenizer_;
    std::vector<int> stops_;
};

} // namespace ninferno::text
