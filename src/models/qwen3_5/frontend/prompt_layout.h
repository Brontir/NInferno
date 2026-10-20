#pragma once

#include "models/qwen3_5/frontend/chat_template.h"

#include <span>

namespace ninferno::models::qwen3_5::frontend {

struct MessageLayout {
    ChatRole role             = ChatRole::User;
    std::size_t begin         = 0;
    std::size_t content_begin = 0;
    std::size_t content_end   = 0;
    std::size_t end           = 0;
    bool closed               = false;
};

struct PromptLayout {
    std::vector<MessageLayout> messages;
    std::vector<MediaPlaceholderByteSpec> media_placeholders;
    std::vector<std::size_t> execution_boundaries;
    bool starts_in_reasoning    = false;
    bool reasoning_open_allowed = false;
};

// Turns and reasoning sections follow the template's conventions; a template without turn markup
// has no message layout.
[[nodiscard]] PromptLayout inspect_prompt_layout(const text::TemplateOutput& rendered,
                                                 std::span<const Modality> media,
                                                 const chat::ChatConventions& conventions);
[[nodiscard]] std::optional<text::ByteSpan>
unique_output_region(const text::TemplateOutput& rendered, std::uint32_t tag);
[[nodiscard]] std::optional<std::size_t> source_boundary(const text::TemplateOutput& rendered,
                                                         std::uint32_t tag, std::size_t offset);

} // namespace ninferno::models::qwen3_5::frontend
