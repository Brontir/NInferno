#pragma once

#include "product/media_acquire/source.h"

#include <ninferno/types.h>

// Internal, wire-format-independent representation of a generation request.
//
// OpenAI and Anthropic schemas map into this wire-independent value.
// translate.cpp then produces the public PromptInput and RequestOptions consumed
// by Engine; media sources remain unresolved until the product service acquires
// owning bytes.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::serve {

// A structured API error mapped onto an error object + HTTP status. Wire-format
// independent: each protocol layer renders it into its own error body shape.
struct ApiError {
    int status       = 400;
    std::string type = "invalid_request_error";
    std::string message;
    std::string param; // optional
    std::string code;  // optional
};

class ApiException : public std::runtime_error {
public:
    explicit ApiException(ApiError error)
        : std::runtime_error(error.message), error_(std::move(error)) {}

    [[nodiscard]] const ApiError& error() const noexcept { return error_; }

private:
    ApiError error_;
};

// A server-side failure as every endpoint answers it: status 500, type "server_error" (OpenAI's;
// the Anthropic renderer types an error by its status), the failure's text as the message.
[[nodiscard]] inline ApiError internal_error(std::string_view message) {
    ApiError error;
    error.status  = 500;
    error.type    = "server_error";
    error.message = std::string(message);
    return error;
}

[[nodiscard]] inline ApiError internal_error(const std::exception& exception) {
    return internal_error(exception.what());
}

// A request refused, or a run given up, because the server is shutting down: status 503, code
// "server_draining" (the writers add Retry-After).
[[nodiscard]] inline ApiError server_draining_error() {
    ApiError error;
    error.status  = 503;
    error.type    = "server_error";
    error.code    = "server_draining";
    error.message = "the server is shutting down; retry on another instance";
    return error;
}

// The most messages (Responses: input Items) one request may carry, --max-request-messages. A
// longer list is refused with a 400 before any of it is parsed or rendered, so a request of a
// million tiny messages does not hold a worker while the template walks it.
inline constexpr std::size_t kDefaultMaxRequestMessages = 16384;

// Server-side context needed while parsing/validating a request.
struct RequestLimits {
    int default_max_tokens   = 8192;
    std::size_t max_messages = kDefaultMaxRequestMessages;
};

enum class ContentKind {
    Text,
    Image,
    Video,
    // A file the model reads as text (a PDF): acquired like media, then expanded into a
    // <document> text part before templating. It needs no vision.
    Document,
};

struct CacheBoundary {
    enum class Ttl : std::uint8_t {
        Default,
        FiveMinutes,
        OneHour,
    };

    ninferno::PromptCacheMarkerKind kind = ninferno::PromptCacheMarkerKind::SharedStablePrefix;
    ninferno::SharedCandidateEvidence evidence =
        ninferno::SharedCandidateEvidence::ExplicitBoundary;
    Ttl ttl                                  = Ttl::Default;

    [[nodiscard]] friend constexpr bool operator==(CacheBoundary, CacheBoundary) noexcept = default;
};

struct ContentPart {
    ContentKind kind = ContentKind::Text;
    std::string text;     // populated for Text
    std::string type_raw; // original wire "type" string for diagnostics
    ninferno::product::media_acquire::Source source;
    ninferno::ImageResizePolicy image_resize_policy = ninferno::ImageResizePolicy::Downsize;
    std::optional<CacheBoundary> cache_boundary_after;
    // Document: the client's title (or file name) and context, printed with the text.
    std::string document_title;
    std::string document_context;
};

struct ToolDefinition {
    std::string name;
    std::string description;
    std::string input_schema_json;
    std::optional<std::string> input_examples_json;
    std::optional<CacheBoundary> cache_boundary_after;
    // strict=true: generated arguments are constrained to input_schema_json.
    bool strict = false;
    // A custom tool (free-form input, served as one string argument "input",
    // custom_tool_input_schema): its calls are answered in the protocol's custom shape.
    bool custom = false;
    // The prompt-bearing fields the client declared, in its own order ("name", "description",
    // "parameters", "strict", "input_examples"; an Anthropic input_schema is "parameters"). The
    // template receives exactly these, in this order, as a transformers-based server would pass
    // the request's own tool object; empty means the canonical name, description, parameters.
    std::vector<std::string> declared_fields;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

// The calls a generation made, as a dialect's response carries them: call `index` takes the id
// id_of(index) gives it (a fresh one, or one the response already handed out).
template <class IdOf>
[[nodiscard]] std::vector<ToolCall>
tool_calls_of(const std::vector<ninferno::GeneratedToolCall>& generated, IdOf&& id_of) {
    std::vector<ToolCall> calls;
    calls.reserve(generated.size());
    for (std::size_t index = 0; index < generated.size(); ++index) {
        calls.push_back(ToolCall{.id             = id_of(index),
                                 .name           = generated[index].name,
                                 .arguments_json = generated[index].arguments_json});
    }
    return calls;
}

enum class ToolChoiceMode {
    Auto,
    None,
};

struct ToolChoice {
    ToolChoiceMode mode = ToolChoiceMode::Auto;
    // With Auto: whether a call is required, or exactly the named function must be called.
    ninferno::ToolCallRequirement requirement = ninferno::ToolCallRequirement::Optional;
    std::string named;
};

struct ChatTurn {
    ChatRole role = ChatRole::User;
    std::vector<ContentPart> content; // ordered parts; may be empty when wire content is empty
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id; // populated for role=tool
    // Optional protocol assertion for a tool result. Call-graph normalization verifies it against
    // the function identified by tool_call_id before the Engine sees the history.
    std::optional<std::string> tool_result_name;
    bool tool_result_is_error = false;
    std::string reasoning_content; // assistant thinking carried across turns (round-tripped to the
                                   // template)
    std::optional<CacheBoundary> cache_boundary_after;
};

// Sampling overrides that have an executable Engine meaning. Protocol-only
// fields are normalized or rejected before this value is constructed.
struct SamplingParams {
    std::optional<double> temperature;
    std::optional<double> top_p;
    std::optional<double> min_p;
    std::optional<int> top_k;
    std::optional<double> presence_penalty;
    std::optional<double> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// Constrained-decoding request normalized from any wire dialect (OpenAI response_format and
// text.format, vLLM structured_outputs/guided_*, llama.cpp grammar, Anthropic
// output_config.format). Preparation compiles it against the Engine tokenizer; a compile failure
// is reported on `param`.
struct OutputConstraintRequest {
    ninferno::OutputConstraintKind kind = ninferno::OutputConstraintKind::JsonObject;
    std::string definition;
    std::vector<std::string> choices;
    bool compact_json = false;
    std::string param;
    // ToolCalls only.
    std::vector<ninferno::ToolCallConstraintTool> tools;
    ninferno::ToolCallRequirement tool_requirement = ninferno::ToolCallRequirement::Optional;
    std::string named_tool;
    bool parallel_tool_calls = true;
    // ToolCalls only: response_format beside optional tools; a text answer must match it.
    std::string answer_schema;
};

// Maximum tool-name length accepted at every protocol boundary and enforced on
// model-generated tool-call names. The OpenAI schema allows 64 bytes, but agent
// hosts synthesize longer names: VS Code Copilot wraps MCP tools as
// "activate_fallback_mcp_<server>_<tool>" (67 bytes observed), and the Anthropic
// adapter already accepted 128. 256 keeps real client names valid while still
// bounding prompt rendering and the streaming parser.
inline constexpr std::size_t kMaximumToolNameLength = 256;

struct GenerationRequest {
    std::vector<ChatTurn> messages;
    // What the server accepted but did not use as sent (an Item left out of the prompt, say): the
    // request log records each note, and the start line is a warning.
    std::vector<std::string> notes;
    std::optional<OutputConstraintRequest> output_constraint;
    std::vector<ToolDefinition> tools;
    std::size_t tool_name_max_length = kMaximumToolNameLength;
    ToolChoice tool_choice;
    // false: at most one tool call (enforced through a tool-call constraint).
    bool parallel_tool_calls = true;
    std::vector<std::string> stop_strings;
    // Benchmark/serving extension shared with vLLM, SGLang and llama.cpp: suppress the
    // checkpoint's default stop tokens so generation runs to the requested token budget.
    // Caller-supplied stop tokens and stop strings still apply.
    bool ignore_eos = false;
    int max_tokens                       = 0; // resolved budget; zero means immediate output limit
    std::optional<bool> enable_thinking;      // unset => use the server default
    std::optional<std::uint32_t> thinking_budget;
    std::optional<ninferno::ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::string chat_template_kwargs_json;
    ninferno::PromptContinuationMode continuation =
        ninferno::PromptContinuationMode::NewAssistantTurn;
    bool allow_engine_automatic_shared_prefixes = true;
    SamplingParams sampling;

    [[nodiscard]] bool uses_tools() const noexcept {
        return !tools.empty() && tool_choice.mode != ToolChoiceMode::None;
    }

    [[nodiscard]] std::size_t media_item_count() const noexcept {
        std::size_t count = 0;
        for (const ChatTurn& message : messages) {
            for (const ContentPart& part : message.content) {
                if (part.kind == ContentKind::Image || part.kind == ContentKind::Video) { ++count; }
            }
        }
        return count;
    }

    [[nodiscard]] bool has_tool_history() const noexcept {
        for (const ChatTurn& message : messages) {
            if (!message.tool_calls.empty() || message.role == ChatRole::Tool) { return true; }
        }
        return false;
    }
};

} // namespace ninferno::serve
