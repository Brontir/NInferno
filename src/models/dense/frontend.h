#pragma once

// The text frontend of a dense-decoder artifact: its own tokenizer.json (text::Tokenizer: byte-level
// BPE of Llama 3 / tekken / GPT-2 / cl100k style, SentencePiece-style BPE with byte fallback) and
// its own chat_template.jinja, both read from the artifact's resources. No model name appears:
// every convention comes from those files.

#include "chat/conventions.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::text {
class JinjaTemplate;
} // namespace ninferno::text

namespace ninferno::models::dense {

// End-of-generation ids: generation_config.json's eos_token_id (one id or a list) and the
// tokenizer's eos token, each once. A fine-tune that keeps its base model's generation_config
// closes its chat turns with the tokenizer's eos (Hermes 3 on Llama 3.2: ChatML <|im_end|>, while
// generation_config lists only Llama's own end tokens), so both end a generation. Throws when
// neither names one.
[[nodiscard]] std::vector<std::int32_t> end_of_generation_ids(
    const nlohmann::json& generation, std::optional<std::int32_t> tokenizer_eos);

class Frontend {
public:
    // Reads tokenizer.json, tokenizer_config.json, chat_template.jinja and generation_config.json
    // from the artifact's text component.
    static Frontend load(const std::filesystem::path& artifact);
    ~Frontend();
    Frontend(Frontend&&) noexcept;
    Frontend& operator=(Frontend&&) noexcept;

    // The chat template rendered over `messages` ([{"role": ..., "content": ...}, ...]) with a
    // generation prompt, tokenized without adding special tokens (the template writes its own),
    // as transformers' apply_chat_template(..., tokenize=True) does.
    [[nodiscard]] std::vector<std::int32_t> render_chat(const nlohmann::ordered_json& messages) const;
    [[nodiscard]] std::string render_chat_text(const nlohmann::ordered_json& messages) const;

    [[nodiscard]] std::vector<std::int32_t> encode(std::string_view text,
                                                   bool add_special_tokens) const;
    [[nodiscard]] std::string decode(std::span<const std::int32_t> ids,
                                     bool skip_special_tokens) const;

    // End-of-generation ids: end_of_generation_ids() of the artifact's files.
    [[nodiscard]] const std::vector<std::int32_t>& stop_ids() const noexcept { return stops_; }

    [[nodiscard]] const text::Tokenizer& tokenizer() const noexcept { return *tokenizer_; }

    // The turn and reasoning markup of the artifact's chat template (chat::resolve_conventions:
    // the data table, by the markers the template contains).
    [[nodiscard]] const chat::ChatConventions& conventions() const noexcept { return conventions_; }

private:
    Frontend() = default;
    std::unique_ptr<text::Tokenizer> tokenizer_;
    std::unique_ptr<text::JinjaTemplate> template_;
    std::string bos_token_, eos_token_;
    std::vector<std::int32_t> stops_;
    chat::ChatConventions conventions_;
};

} // namespace ninferno::models::dense
