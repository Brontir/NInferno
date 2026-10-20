#pragma once

#include "chat/conventions.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "text/jinja.h"

#include "models/qwen3_5/frontend/prepared_prompt.h"
#include <ninferno/types.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninferno::models::qwen3_5::frontend {

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

enum class ChatPartKind {
    Text,
    Image,
    Video,
};

enum class Modality : std::uint8_t {
    Image = 1,
    Video = 2,
};

struct MediaPlaceholderByteSpec {
    text::ByteSpan bytes;
    Modality modality      = Modality::Image;
    std::size_t item_index = 0;
    VisionStyle style      = VisionStyle::Qwen;
};

struct MediaTokenRunByteSpec {
    text::ByteSpan bytes;
    Modality modality       = Modality::Image;
    std::size_t item_index  = 0;
    std::size_t frame_index = 0;
    VisionStyle style       = VisionStyle::Qwen;
};

struct MediaData {
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
    ImageResizePolicy image_resize_policy = ImageResizePolicy::Downsize;
};

struct ChatPart {
    ChatPartKind kind = ChatPartKind::Text;
    std::string text;
    MediaData media;

    static ChatPart text_part(std::string value) {
        ChatPart part;
        part.text = std::move(value);
        return part;
    }

    static ChatPart image(MediaData value) {
        ChatPart part;
        part.kind  = ChatPartKind::Image;
        part.media = std::move(value);
        return part;
    }

    static ChatPart video(MediaData value) {
        ChatPart part;
        part.kind  = ChatPartKind::Video;
        part.media = std::move(value);
        return part;
    }
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<ChatPart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;

    [[nodiscard]] bool has_media() const noexcept;
};

struct ChatRenderOptions {
    PromptContinuationMode continuation = PromptContinuationMode::NewAssistantTurn;
    // Internal renderer control used by frontend qualification. Product PromptInput always
    // selects either a new assistant turn or continuation of the final assistant.
    bool add_generation_prompt = true;
    std::optional<bool> enable_thinking;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::string chat_template_kwargs_json;
    bool add_vision_id = false;
    std::vector<std::string> tool_jsons;
    std::vector<PromptCacheMarker> cache_markers;
};

struct RewriteCheckpointByteSpec {
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::size_t offset         = 0;
};

struct RenderedChat {
    std::string text;
    std::vector<text::ByteSpan> literal_spans;
    bool starts_in_reasoning = false;
    // The generation starts a fresh assistant body that has no reasoning section yet, so the
    // model may open one itself (templates that leave the opener to the model).
    bool reasoning_open_allowed = false;
    // The request turned reasoning off in a template that has the switch (enable_thinking false,
    // or unset where the template's default is off): the model is not to open a section itself.
    bool reasoning_declined = false;
    std::vector<MediaPlaceholderByteSpec> media_placeholders;
    std::vector<MediaTokenRunByteSpec> media_token_runs;
    std::optional<RewriteCheckpointByteSpec> rewrite_checkpoint;
    std::vector<std::size_t> rewrite_execution_boundaries;
    // Index n is the exact byte frontier after serializing the first n input messages. A missing
    // value means the template has no independent boundary there (for example, before a leading
    // instruction message folded into the system preamble).
    std::vector<std::optional<std::size_t>> message_boundaries;
    // One rendered byte boundary per requested cache marker.
    std::vector<std::optional<std::size_t>> cache_boundaries;
};

class CompiledChatTemplate {
public:
    [[nodiscard]] static CompiledChatTemplate
    resolve(std::string_view source, std::string source_name = "chat_template.jinja",
            nlohmann::ordered_json special_tokens = nlohmann::ordered_json::object());
    [[nodiscard]] RenderedChat render(const std::vector<ChatMessage>& messages,
                                      ChatRenderOptions options         = {},
                                      const PreparationControl& control = {}) const;
    // The turn, reasoning and tool-call markup this template uses (chat::resolve_conventions).
    [[nodiscard]] const chat::ChatConventions& conventions() const noexcept { return conventions_; }

private:
    [[nodiscard]] RenderedChat render_once(const std::vector<ChatMessage>& messages,
                                           const ChatRenderOptions& options,
                                           const PreparationControl& control) const;
    CompiledChatTemplate(text::JinjaTemplate compiled, nlohmann::ordered_json special_tokens,
                         chat::ChatConventions conventions)
        : compiled_(std::move(compiled)), special_tokens_(std::move(special_tokens)),
          conventions_(std::move(conventions)) {}

    text::JinjaTemplate compiled_;
    nlohmann::ordered_json special_tokens_;
    chat::ChatConventions conventions_;
    // enable_thinking changes the rendered prompt, and the template's choice when it is unset.
    bool thinking_switch_  = false;
    bool thinking_default_ = true;
};

} // namespace ninferno::models::qwen3_5::frontend
