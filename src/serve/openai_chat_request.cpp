#include "serve/openai_chat.h"
#include "serve/openai_common.h"
#include "serve/output_constraint_request.h"
#include "serve/request_validation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace ninferno::serve {
namespace {

using Json = RequestJson;

void require_object(const Json& value, const char* message, const char* param = nullptr) {
    if (!value.is_object()) { bad_request(message, param == nullptr ? "" : param); }
}

std::optional<std::uint64_t> get_seed(const Json& object) {
    if (!object.contains("seed") || object.at("seed").is_null()) { return std::nullopt; }
    const Json& value = object.at("seed");
    if (!value.is_number_integer()) { bad_request("seed must be an integer", "seed"); }
    if (value.is_number_unsigned()) { return value.get<std::uint64_t>(); }
    // A negative seed (-1, how llama.cpp-style clients and the bundled chat page say "random") is
    // the server's own seed, as when the field is absent: cast to unsigned it would be one fixed
    // seed for every request.
    const std::int64_t seed = value.get<std::int64_t>();
    if (seed < 0) { return std::nullopt; }
    return static_cast<std::uint64_t>(seed);
}

ChatRole parse_message_role(const std::string& role) {
    if (role == "system") { return ChatRole::System; }
    if (role == "developer") { return ChatRole::Developer; }
    if (role == "user") { return ChatRole::User; }
    if (role == "assistant") { return ChatRole::Assistant; }
    if (role == "tool") { return ChatRole::Tool; }
    bad_request("unsupported role: " + ascii_preview(role), "messages", "unsupported_role");
}

// A replayed call's arguments are a JSON string by the protocol; some clients (JetBrains AI
// Assistant) send the object itself, which is taken as its text. Anything else is malformed.
std::string history_arguments(const Json& function, const std::string& param) {
    if (function.contains("arguments") && function.at("arguments").is_string()) {
        return function.at("arguments").get<std::string>();
    }
    if (function.contains("arguments") && function.at("arguments").is_object()) {
        return function.at("arguments").dump();
    }
    bad_request("tool-call arguments must be a JSON string or a JSON object", param);
}

void validate_standard_output_controls(const Json& body) {
    if (body.contains("functions") && !body.at("functions").is_null()) {
        const Json& functions = body.at("functions");
        if (!functions.is_array()) { bad_request("functions must be an array", "functions"); }
        if (!functions.empty()) {
            bad_request(
                "non-empty legacy functions require the legacy prompt and single-function-call "
                "response contract, which NInferno does not expose; use tools instead",
                "functions", "legacy_tools_not_supported");
        }
    }
    if (body.contains("function_call") && !body.at("function_call").is_null()) {
        const Json& choice = body.at("function_call");
        if (choice.is_string()) {
            const std::string value = choice.get<std::string>();
            if (value != "none" && value != "auto") {
                bad_request("function_call must be 'none', 'auto', or a named function choice",
                            "function_call");
            }
        } else if (choice.is_object()) {
            bad_request(
                "a named legacy function_call requires forced tool invocation, which NInferno "
                "cannot guarantee",
                "function_call", "legacy_tools_not_supported");
        } else {
            bad_request("function_call must be 'none', 'auto', or an object", "function_call");
        }
    }

    if (body.contains("logit_bias") && !body.at("logit_bias").is_null()) {
        const Json& biases = body.at("logit_bias");
        if (!biases.is_object()) { bad_request("logit_bias must be an object", "logit_bias"); }
        for (auto iterator = biases.begin(); iterator != biases.end(); ++iterator) {
            if (!iterator.value().is_number()) {
                bad_request("logit_bias values must be numbers", "logit_bias");
            }
            const double value = iterator.value().get<double>();
            if (!std::isfinite(value) || value != 0.0) {
                bad_request(
                    "nonzero logit_bias requires per-token logit modification, which NInferno "
                    "does not provide",
                    "logit_bias", "logit_bias_not_supported");
            }
        }
    }
    if (body.contains("logprobs") && !body.at("logprobs").is_null()) {
        if (!body.at("logprobs").is_boolean()) {
            bad_request("logprobs must be a boolean", "logprobs");
        }
        if (body.at("logprobs").get<bool>()) {
            bad_request("logprobs=true requires per-token log probabilities in the response, which "
                        "NInferno does not provide",
                        "logprobs", "logprobs_not_supported");
        }
    }
    if (const std::optional<int> top_logprobs = optional_int(body, "top_logprobs")) {
        if (*top_logprobs != 0) {
            bad_request(
                "nonzero top_logprobs requires alternative-token probabilities in the response, "
                "which NInferno does not provide",
                "top_logprobs", "logprobs_not_supported");
        }
    }

    if (body.contains("modalities") && !body.at("modalities").is_null()) {
        const Json& modalities = body.at("modalities");
        if (!modalities.is_array() || modalities.empty()) {
            bad_request("modalities must be a non-empty array", "modalities");
        }
        for (const Json& modality : modalities) {
            if (!modality.is_string()) {
                bad_request("modalities entries must be strings", "modalities");
            }
            if (modality.get<std::string>() != "text") {
                bad_request(
                    "the requested output modality requires non-text generation, while NInferno "
                    "produces text only",
                    "modalities", "modality_not_supported");
            }
        }
    }

    if (body.contains("web_search_options") && !body.at("web_search_options").is_null()) {
        bad_request(
            "web_search_options requests hosted web search and citations, which NInferno does not "
            "provide",
            "web_search_options", "web_search_not_supported");
    }
    if (body.contains("moderation") && !body.at("moderation").is_null()) {
        bad_request(
            "moderation requests input/output moderation behavior, which NInferno does not provide",
            "moderation", "moderation_not_supported");
    }
    if (body.contains("verbosity") && !body.at("verbosity").is_null()) {
        if (!body.at("verbosity").is_string()) {
            bad_request("verbosity must be 'low', 'medium', or 'high'", "verbosity");
        }
        const std::string value = body.at("verbosity").get<std::string>();
        if (value != "low" && value != "medium" && value != "high") {
            bad_request("verbosity must be 'low', 'medium', or 'high'", "verbosity");
        }
        if (value != "medium") {
            bad_request(
                "verbosity='" + value +
                    "' requires an output-length style constraint that NInferno cannot guarantee; "
                    "the default 'medium' value is accepted",
                "verbosity", "verbosity_not_supported");
        }
    }

    if (body.contains("store") && !body.at("store").is_null()) {
        if (!body.at("store").is_boolean()) { bad_request("store must be a boolean", "store"); }
        if (body.at("store").get<bool>()) {
            bad_request(
                "store=true requires a retrievable stored Chat Completion, which NInferno does not "
                "provide",
                "store", "store_not_supported");
        }
    }
}

// OpenAI response_format, vLLM structured_outputs (and the guided_* spellings) and llama.cpp
// grammar each promise constrained generation. At most one may be present; each accepted form is
// enforced by the Engine.
std::optional<OutputConstraintRequest> parse_output_constraint(const Json& body) {
    std::optional<OutputConstraintRequest> format;
    if (body.contains("response_format") && !body.at("response_format").is_null()) {
        format = parse_openai_response_format(body.at("response_format"), "response_format", false);
    }
    std::optional<OutputConstraintRequest> extension = parse_structured_output_extensions(body);
    if (format && extension) {
        bad_request("response_format and " + extension->param + " cannot be combined",
                    extension->param, "conflicting_constraints");
    }
    return format ? format : extension;
}

void validate_compatibility_hints(const Json& body) {
    // vLLM exposes repetition_penalty, but NInferno's Engine intentionally has no such sampler.
    // The neutral value is accepted so common client defaults remain harmless.
    if (body.contains("repetition_penalty") && !body.at("repetition_penalty").is_null()) {
        if (!body.at("repetition_penalty").is_number()) {
            bad_request("repetition_penalty must be a number", "repetition_penalty");
        }
        const double value = body.at("repetition_penalty").get<double>();
        if (!std::isfinite(value) || value != 1.0) {
            bad_request(
                "a non-neutral repetition_penalty requires a sampler transform that NInferno does "
                "not provide; only repetition_penalty=1 is accepted",
                "repetition_penalty", "repetition_penalty_not_supported");
        }
    }

    // vLLM/SGLang expose processor-specific kwargs. They cannot be honored by NInferno's fixed
    // Vision frontend, so only omitted/null-valued overrides are semantically neutral.
    if (body.contains("mm_processor_kwargs") && !body.at("mm_processor_kwargs").is_null()) {
        if (!body.at("mm_processor_kwargs").is_object()) {
            bad_request("mm_processor_kwargs must be an object", "mm_processor_kwargs");
        }
        for (auto iterator = body.at("mm_processor_kwargs").begin();
             iterator != body.at("mm_processor_kwargs").end(); ++iterator) {
            if (!iterator.value().is_null()) {
                bad_request("mm_processor_kwargs contains a non-null preprocessing override that "
                            "NInferno's "
                            "fixed Vision frontend cannot apply",
                            "mm_processor_kwargs", "mm_processor_kwargs_not_supported");
            }
        }
    }
}

ninferno::product::media_acquire::Source parse_media_url(const Json& part, const char* field,
                                                       bool image) {
    if (!part.contains(field)) {
        bad_request(std::string(field) + " content part must contain " + field, "messages");
    }
    const Json& value = part.at(field);
    std::string url;
    if (value.is_string()) {
        // vLLM/SGLang accept this shorthand in addition to OpenAI's object form.
        url = value.get<std::string>();
    } else if (value.is_object()) {
        if (!value.contains("url") || !value.at("url").is_string()) {
            bad_request(std::string(field) + " must contain a string url", "messages");
        }
        url = value.at("url").get<std::string>();
        if (image) { check_image_detail(value, "image_url", "messages"); }
    } else {
        bad_request(std::string(field) + " must be a URL string or object", "messages");
    }
    if (url.empty()) { bad_request(std::string(field) + " URL must not be empty", "messages"); }

    return media_source(std::move(url), field, "messages");
}

void parse_content_parts(const Json& content, ChatTurn& turn, std::size_t index) {
    if (content.is_string()) {
        turn.content.push_back(ContentPart{
            .kind = ContentKind::Text, .text = content.get<std::string>(), .type_raw = "text"});
        return;
    }
    if (!content.is_array()) {
        bad_request("message " + std::to_string(index) + " content must be a string or array",
                    "messages");
    }
    for (const Json& part : content) {
        if (!part.is_object() || !part.contains("type") || !part.at("type").is_string()) {
            bad_request("message content parts must contain a string type", "messages");
        }
        const std::string type = part.at("type").get<std::string>();
        ContentPart parsed;
        parsed.type_raw = type;
        if (type == "text") {
            if (!part.contains("text") || !part.at("text").is_string()) {
                bad_request("text content part must contain a string text", "messages");
            }
            parsed.kind = ContentKind::Text;
            parsed.text = part.at("text").get<std::string>();
        } else if (type == "refusal") {
            if (turn.role != ChatRole::Assistant) {
                bad_request("refusal content is only valid on assistant messages", "messages");
            }
            if (!part.contains("refusal") || !part.at("refusal").is_string()) {
                bad_request("refusal content part must contain a string refusal", "messages");
            }
            parsed.kind = ContentKind::Text;
            parsed.text = part.at("refusal").get<std::string>();
        } else if (type == "image_url") {
            // VS Code Copilot-compatible screenshot results use image_url parts on tool turns.
            if (turn.role != ChatRole::User && turn.role != ChatRole::Tool) {
                bad_request("image_url is only supported on user or tool messages", "messages",
                            "modality_not_supported");
            }
            parsed.kind   = ContentKind::Image;
            parsed.source = parse_media_url(part, "image_url", true);
        } else if (type == "video_url") {
            // Qwen, vLLM, and SGLang use video_url as a Chat Completions extension for
            // multimodal models. NInferno maps it to the Engine's native Video input.
            if (turn.role != ChatRole::User) {
                bad_request("video_url is only supported on user messages", "messages",
                            "modality_not_supported");
            }
            parsed.kind   = ContentKind::Video;
            parsed.source = parse_media_url(part, "video_url", false);
        } else if (type == "file") {
            // OpenAI's inline file part: {"file": {"file_data": "data:application/pdf;base64,...",
            // "filename": ...}}. A file_id names an upload in OpenAI's Files API.
            if (turn.role != ChatRole::User && turn.role != ChatRole::Tool) {
                bad_request("file is only supported on user or tool messages", "messages",
                            "modality_not_supported");
            }
            if (!part.contains("file") || !part.at("file").is_object()) {
                bad_request("file content part must contain a file object", "messages");
            }
            const Json& file = part.at("file");
            if (!file.contains("file_data") || !file.at("file_data").is_string()) {
                bad_request(file.contains("file_id")
                                ? "file.file_id requires a Files API, which NInferno does not "
                                  "provide; send file.file_data"
                                : "file content part must contain a string file_data",
                            "messages", "file_inputs_not_supported");
            }
            parsed.kind   = ContentKind::Document;
            parsed.source = inline_file_source(file.at("file_data").get<std::string>(),
                                               "file.file_data", "messages");
            if (file.contains("filename") && file.at("filename").is_string()) {
                parsed.document_title = file.at("filename").get<std::string>();
            }
        } else {
            bad_request("content type '" + type + "' is not supported", "messages",
                        "modality_not_supported");
        }
        if (parse_openai_prompt_cache_breakpoint(part, "messages")) {
            parsed.cache_boundary_after = CacheBoundary{};
        }
        turn.content.push_back(std::move(parsed));
    }
}

std::vector<ToolCall> parse_assistant_tool_calls(const Json& message, std::size_t index) {
    std::vector<ToolCall> calls;
    if (!message.contains("tool_calls") || message.at("tool_calls").is_null()) { return calls; }
    const Json& values = message.at("tool_calls");
    if (!values.is_array()) {
        bad_request("assistant message " + std::to_string(index) + " tool_calls must be an array",
                    "messages");
    }
    calls.reserve(values.size());
    for (std::size_t call_index = 0; call_index < values.size(); ++call_index) {
        const std::string prefix = "messages[" + std::to_string(index) + "].tool_calls[" +
                                   std::to_string(call_index) + "]";
        const Json& value = values.at(call_index);
        if (!value.is_object() || !value.contains("id") || !value.at("id").is_string()) {
            bad_request("tool_calls entries must contain a string id", prefix + ".id");
        }
        if (!value.contains("type") || !value.at("type").is_string() ||
            (value.at("type").get<std::string>() != "function" &&
             value.at("type").get<std::string>() != "custom")) {
            bad_request("only function and custom tool_calls are supported", prefix + ".type",
                        "tool_type_not_supported");
        }
        // A custom tool's call carries its free-form input: the "input" argument of the
        // single-string-input function the tool is served as.
        if (value.at("type").get<std::string>() == "custom") {
            if (!value.contains("custom") || !value.at("custom").is_object()) {
                bad_request("custom tool_calls must contain a custom object", prefix + ".custom");
            }
            const Json& custom = value.at("custom");
            if (!custom.contains("input") || !custom.at("input").is_string()) {
                bad_request("custom tool_calls must contain string input",
                            prefix + ".custom.input");
            }
            calls.push_back(ToolCall{.id   = value.at("id").get<std::string>(),
                                     .name = require_tool_name(custom, prefix + ".custom.name"),
                                     .arguments_json = Json{{"input", custom.at("input")}}.dump()});
            continue;
        }
        if (!value.contains("function") || !value.at("function").is_object()) {
            bad_request("tool_calls entries must contain a function object", prefix + ".function");
        }
        const Json& function = value.at("function");
        calls.push_back(ToolCall{.id   = value.at("id").get<std::string>(),
                                 .name = require_tool_name(function, prefix + ".function.name"),
                                 .arguments_json =
                                     history_arguments(function, prefix + ".function.arguments")});
    }
    return calls;
}

std::optional<ToolCall> parse_legacy_assistant_function_call(const Json& message,
                                                             std::size_t index) {
    if (!message.contains("function_call") || message.at("function_call").is_null()) {
        return std::nullopt;
    }
    const Json& call = message.at("function_call");
    if (!call.is_object()) {
        bad_request("assistant message " + std::to_string(index) +
                        " function_call must be an object",
                    "messages");
    }
    return ToolCall{.id             = {},
                    .name           = require_tool_name(call, "messages[" + std::to_string(index) +
                                                                  "].function_call.name"),
                    .arguments_json = history_arguments(call, "messages[" + std::to_string(index) +
                                                                  "].function_call.arguments")};
}

std::optional<std::string> parse_assistant_reasoning(const Json& message, std::size_t index) {
    auto parse = [&](const char* key) -> std::optional<std::string> {
        if (!message.contains(key) || message.at(key).is_null()) { return std::nullopt; }
        if (!message.at(key).is_string()) {
            bad_request("assistant message " + std::to_string(index) + " " + key +
                            " must be a string",
                        "messages");
        }
        return message.at(key).get<std::string>();
    };
    // vLLM historically used reasoning_content and now also accepts reasoning. They carry the
    // same assistant-history semantics, so equal aliases normalize to one Engine field.
    std::optional<std::string> content   = parse("reasoning_content");
    std::optional<std::string> reasoning = parse("reasoning");
    if (content && content->empty()) { content.reset(); }
    if (reasoning && reasoning->empty()) { reasoning.reset(); }
    if (content && reasoning && *content != *reasoning) {
        bad_request("conflicting assistant reasoning and reasoning_content values", "messages",
                    "conflicting_template_option");
    }
    return reasoning ? reasoning : content;
}

void validate_message_name(const Json& item) {
    if (!item.contains("name") || item.at("name").is_null()) { return; }
    if (!item.at("name").is_string()) { bad_request("message name must be a string", "messages"); }
    // OpenAI's optional participant name: the chat templates have no place for it, and clients
    // send it on user, assistant and tool messages alike (some mirror the function name onto
    // tool results), so it is an ignored, output-neutral hint that never reaches the Engine or
    // the prompt renderer. A 400 here would end the conversation for such a client.
}

void validate_non_assistant_fields(const Json& item, ChatRole role) {
    if (role != ChatRole::Assistant && item.contains("function_call") &&
        !item.at("function_call").is_null()) {
        bad_request("function_call is only valid on assistant messages", "messages");
    }
    if (role == ChatRole::Assistant) { return; }
    for (const char* key : {"reasoning", "reasoning_content"}) {
        if (!item.contains(key) || item.at(key).is_null()) { continue; }
        if (!item.at(key).is_string()) {
            bad_request(std::string(key) + " must be a string", "messages");
        }
        if (!item.at(key).get<std::string>().empty()) {
            bad_request(std::string(key) + " is only valid on assistant messages", "messages");
        }
    }
}

void validate_non_tool_call_id(const Json& item) {
    if (!item.contains("tool_call_id") || item.at("tool_call_id").is_null()) { return; }
    if (!item.at("tool_call_id").is_string()) {
        bad_request("tool_call_id must be a string", "messages");
    }
    if (!item.at("tool_call_id").get<std::string>().empty()) {
        bad_request("a non-empty tool_call_id is only valid on tool messages", "messages");
    }
}

ChatTurn parse_tool_message(const Json& item, std::size_t index, bool legacy_function) {
    if (item.contains("tool_calls") && !item.at("tool_calls").is_null()) {
        if (!item.at("tool_calls").is_array()) {
            bad_request("tool_calls must be an array", "messages");
        }
        if (!item.at("tool_calls").empty()) {
            bad_request("tool messages cannot contain tool_calls", "messages");
        }
    }
    if (!legacy_function &&
        (!item.contains("tool_call_id") || !item.at("tool_call_id").is_string())) {
        bad_request("tool messages must contain a string tool_call_id", "messages");
    }
    if (item.contains("tool_call_id") && !item.at("tool_call_id").is_null() &&
        !item.at("tool_call_id").is_string()) {
        bad_request("tool_call_id must be a string", "messages");
    }
    if (!item.contains("content") || item.at("content").is_null()) {
        bad_request("tool messages must contain content", "messages");
    }

    ChatTurn turn;
    turn.role = ChatRole::Tool;
    if (item.contains("tool_call_id") && item.at("tool_call_id").is_string()) {
        turn.tool_call_id = item.at("tool_call_id").get<std::string>();
    }
    parse_content_parts(item.at("content"), turn, index);
    return turn;
}

ChatTurn parse_assistant_message(const Json& item, std::size_t index) {
    if (item.contains("audio") && !item.at("audio").is_null()) {
        bad_request("assistant audio history requires resolving a previous audio response, which "
                    "NInferno cannot provide",
                    "messages", "assistant_history_not_supported");
    }

    ChatTurn turn;
    turn.role       = ChatRole::Assistant;
    turn.tool_calls = parse_assistant_tool_calls(item, index);
    if (std::optional<ToolCall> legacy_call = parse_legacy_assistant_function_call(item, index)) {
        turn.tool_calls.insert(turn.tool_calls.begin(), std::move(*legacy_call));
    }
    const std::optional<std::string> reasoning = parse_assistant_reasoning(item, index);
    if (item.contains("content") && !item.at("content").is_null()) {
        parse_content_parts(item.at("content"), turn, index);
    }
    if (item.contains("refusal") && !item.at("refusal").is_null()) {
        if (!item.at("refusal").is_string()) {
            bad_request("assistant refusal must be a string", "messages");
        }
        std::string refusal = item.at("refusal").get<std::string>();
        if (!refusal.empty()) {
            turn.content.push_back(ContentPart{
                .kind = ContentKind::Text, .text = std::move(refusal), .type_raw = "refusal"});
        }
    }
    if (reasoning) { turn.reasoning_content = *reasoning; }
    return turn;
}

ChatTurn parse_regular_message(const Json& item, std::size_t index, ChatRole role) {
    if (item.contains("tool_calls") && !item.at("tool_calls").is_null()) {
        if (!item.at("tool_calls").is_array()) {
            bad_request("tool_calls must be an array", "messages");
        }
        if (!item.at("tool_calls").empty()) {
            bad_request("non-empty tool_calls are only valid on assistant messages", "messages");
        }
    }
    if (!item.contains("content") || item.at("content").is_null()) {
        bad_request("message " + std::to_string(index) + " must have content", "messages");
    }

    ChatTurn turn;
    turn.role = role;
    parse_content_parts(item.at("content"), turn, index);
    return turn;
}

ChatTurn parse_message(const Json& item, std::size_t index) {
    if (!item.is_object() || !item.contains("role") || !item.at("role").is_string()) {
        bad_request("message " + std::to_string(index) + " must be an object with a string role",
                    "messages");
    }
    const std::string role_name = item.at("role").get<std::string>();
    const bool legacy_function  = role_name == "function";
    const ChatRole role         = legacy_function ? ChatRole::Tool : parse_message_role(role_name);

    validate_message_name(item);
    if (legacy_function) {
        (void)require_tool_name(item, "messages[" + std::to_string(index) + "].name");
    }
    validate_non_assistant_fields(item, role);

    if (role == ChatRole::Tool) { return parse_tool_message(item, index, legacy_function); }
    validate_non_tool_call_id(item);
    if (role == ChatRole::Assistant) { return parse_assistant_message(item, index); }
    return parse_regular_message(item, index, role);
}

// Tool messages that answer an assistant's calls are put in the order of the calls. Templates
// render results by position (Qwen's <tool_response> carries no ID), and a client that runs the
// calls concurrently may return them as they finish: call A's result read as B's.
// Only a run whose IDs are distinct and all name calls of that assistant turn is
// reordered; any other run is left as sent.
void order_tool_results(std::vector<ChatTurn>& messages) {
    for (std::size_t index = 0; index < messages.size(); ++index) {
        const ChatTurn& calls = messages[index];
        if (calls.role != ChatRole::Assistant || calls.tool_calls.size() < 2) { continue; }
        std::size_t end = index + 1;
        while (end < messages.size() && messages[end].role == ChatRole::Tool) { ++end; }
        std::vector<std::pair<std::size_t, std::size_t>> order; // (call position, message index)
        for (std::size_t result = index + 1; result < end; ++result) {
            const auto call = std::find_if(
                calls.tool_calls.begin(), calls.tool_calls.end(),
                [&](const ToolCall& c) { return c.id == messages[result].tool_call_id; });
            if (call == calls.tool_calls.end()) {
                order.clear();
                break;
            }
            order.emplace_back(static_cast<std::size_t>(call - calls.tool_calls.begin()), result);
        }
        std::sort(order.begin(), order.end());
        const bool distinct =
            std::adjacent_find(order.begin(), order.end(), [](const auto& a, const auto& b) {
                return a.first == b.first;
            }) == order.end();
        if (order.size() < 2 || !distinct) { continue; }
        std::vector<ChatTurn> results;
        results.reserve(order.size());
        for (const auto& [position, message] : order) {
            (void)position;
            results.push_back(std::move(messages[message]));
        }
        std::move(results.begin(), results.end(),
                  messages.begin() + static_cast<std::ptrdiff_t>(index + 1));
        index = end - 1;
    }
}

void parse_messages(const Json& body, const RequestLimits& limits, GenerationRequest& output) {
    if (!body.contains("messages")) { bad_request("missing required field: messages", "messages"); }
    const Json& messages = body.at("messages");
    if (!messages.is_array() || messages.empty()) {
        bad_request("messages must be a non-empty array", "messages");
    }
    check_message_count(messages.size(), limits, "messages");
    output.messages.reserve(messages.size());
    for (std::size_t index = 0; index < messages.size(); ++index) {
        output.messages.push_back(parse_message(messages.at(index), index));
    }
    order_tool_results(output.messages);
}

void parse_tools(const Json& body, GenerationRequest& output) {
    if (!body.contains("tools") || body.at("tools").is_null()) { return; }
    const Json& tools = body.at("tools");
    if (!tools.is_array()) { bad_request("tools must be an array", "tools"); }
    output.tools.reserve(tools.size());
    for (std::size_t index = 0; index < tools.size(); ++index) {
        const std::string prefix = "tools[" + std::to_string(index) + "]";
        const Json& item         = tools.at(index);
        if (!item.is_object() || !item.contains("type") || !item.at("type").is_string()) {
            bad_request("tools entries must contain a string type", prefix + ".type");
        }
        const std::string type = item.at("type").get<std::string>();
        if (type == "custom") {
            if (!item.contains("custom") || !item.at("custom").is_object()) {
                bad_request("custom tools must contain a custom object", prefix + ".custom");
            }
            const Json& custom = item.at("custom");
            ToolDefinition tool;
            tool.name = require_tool_name(custom, prefix + ".custom.name");
            if (custom.contains("description") && !custom.at("description").is_null()) {
                if (!custom.at("description").is_string()) {
                    bad_request("custom tool description must be a string",
                                prefix + ".custom.description");
                }
                tool.description = custom.at("description").get<std::string>();
            }
            tool.input_schema_json = custom_tool_input_schema(custom);
            tool.custom            = true;
            output.tools.push_back(std::move(tool));
            continue;
        }
        if (type != "function") {
            bad_request(
                "tool type '" + ascii_preview(type) +
                    "' requires a non-function output contract that NInferno does not provide",
                prefix + ".type", "tool_type_not_supported");
        }
        if (!item.contains("function") || !item.at("function").is_object()) {
            bad_request("function tools must contain a function object", prefix + ".function");
        }
        const Json& function = item.at("function");
        ToolDefinition tool;
        tool.name = require_tool_name(function, prefix + ".function.name");
        if (function.contains("description") && !function.at("description").is_null()) {
            if (!function.at("description").is_string()) {
                bad_request("function description must be a string",
                            prefix + ".function.description");
            }
            tool.description = function.at("description").get<std::string>();
        }
        if (!function.contains("parameters") || function.at("parameters").is_null()) {
            tool.input_schema_json =
                Json{{"type", "object"}, {"properties", Json::object()}}.dump();
        } else if (!function.at("parameters").is_object()) {
            bad_request("function parameters must be a JSON object",
                        prefix + ".function.parameters");
        } else {
            tool.input_schema_json = function.at("parameters").dump();
        }
        if (function.contains("strict") && !function.at("strict").is_null()) {
            if (!function.at("strict").is_boolean()) {
                bad_request("function strict must be a boolean", prefix + ".function.strict");
            }
            // Enforced by a tool-call constraint (attach_tool_call_constraint).
            tool.strict = function.at("strict").get<bool>();
        }
        for (auto field = function.begin(); field != function.end(); ++field) {
            const std::string& key = field.key();
            if (!field.value().is_null() && (key == "name" || key == "description" ||
                                             key == "parameters" || key == "strict")) {
                tool.declared_fields.push_back(key);
            }
        }
        output.tools.push_back(std::move(tool));
    }
}

void apply_allowed_tools(const Json& config, GenerationRequest& output) {
    if (!config.is_object()) {
        bad_request("tool_choice.allowed_tools must be an object", "tool_choice");
    }
    if (!config.contains("mode") || !config.at("mode").is_string()) {
        bad_request("tool_choice.allowed_tools.mode must be 'auto' or 'required'", "tool_choice");
    }
    const std::string mode = config.at("mode").get<std::string>();
    if (mode != "auto" && mode != "required") {
        bad_request("tool_choice.allowed_tools.mode must be 'auto' or 'required'", "tool_choice");
    }
    if (!config.contains("tools") || !config.at("tools").is_array()) {
        bad_request("tool_choice.allowed_tools.tools must be an array", "tool_choice");
    }

    std::vector<std::string> allowed_names;
    allowed_names.reserve(config.at("tools").size());
    for (std::size_t index = 0; index < config.at("tools").size(); ++index) {
        const std::string prefix = "tool_choice.allowed_tools.tools[" + std::to_string(index) + "]";
        const Json& item         = config.at("tools").at(index);
        if (!item.is_object() || !item.contains("type") || !item.at("type").is_string()) {
            bad_request("allowed tool entries must contain a string type", prefix + ".type");
        }
        if (item.at("type").get<std::string>() != "function" &&
            item.at("type").get<std::string>() != "custom") {
            bad_request(
                "allowed_tools entries must select a function or custom tool, because NInferno "
                "provides no other output contract",
                prefix + ".type", "tool_type_not_supported");
        }
        // Chat Completions nests the name the way its tools do ({"type": "function", "function":
        // {"name": ...}}, the custom object for a custom tool); Responses, and our earlier reading,
        // put it on the entry itself. Both are read.
        const std::string type = item.at("type").get<std::string>();
        const bool nested      = item.contains(type) && item.at(type).is_object();
        const std::string name =
            nested ? require_tool_name(item.at(type), prefix + "." + type + ".name")
                   : require_tool_name(item, prefix + ".name");
        const bool declared =
            std::any_of(output.tools.begin(), output.tools.end(),
                        [&](const ToolDefinition& tool) { return tool.name == name; });
        if (!declared) {
            bad_request("allowed tool '" + name + "' is not present in tools",
                        prefix + (nested ? "." + type : std::string()) + ".name");
        }
        if (std::find(allowed_names.begin(), allowed_names.end(), name) == allowed_names.end()) {
            allowed_names.push_back(name);
        }
    }

    std::erase_if(output.tools, [&](const ToolDefinition& tool) {
        return std::find(allowed_names.begin(), allowed_names.end(), tool.name) ==
               allowed_names.end();
    });
    output.tool_choice.mode        = ToolChoiceMode::Auto;
    output.tool_choice.requirement = mode == "required" ? ninferno::ToolCallRequirement::Required
                                                        : ninferno::ToolCallRequirement::Optional;
}

void parse_tool_choice(const Json& body, GenerationRequest& output) {
    if (!body.contains("tool_choice") || body.at("tool_choice").is_null()) { return; }
    const Json& choice = body.at("tool_choice");
    if (choice.is_string()) {
        const std::string value = choice.get<std::string>();
        if (value == "auto") {
            output.tool_choice.mode = ToolChoiceMode::Auto;
        } else if (value == "none") {
            output.tool_choice.mode = ToolChoiceMode::None;
        } else if (value == "required") {
            if (output.tools.empty()) {
                bad_request("tool_choice='required' requires tools", "tool_choice");
            }
            output.tool_choice.mode        = ToolChoiceMode::Auto;
            output.tool_choice.requirement = ninferno::ToolCallRequirement::Required;
        } else {
            bad_request("tool_choice must be 'auto', 'none', 'required', or a function choice",
                        "tool_choice");
        }
    } else if (choice.is_object()) {
        if (!choice.contains("type") || !choice.at("type").is_string()) {
            bad_request("tool_choice objects must contain a string type", "tool_choice");
        }
        const std::string type = choice.at("type").get<std::string>();
        if (type == "allowed_tools") {
            apply_allowed_tools(
                choice.contains("allowed_tools") ? choice.at("allowed_tools") : choice, output);
        } else if (type == "function") {
            if (!choice.contains("function") || !choice.at("function").is_object()) {
                bad_request("function tool_choice must contain a function object", "tool_choice");
            }
            const std::string name =
                require_tool_name(choice.at("function"), "tool_choice.function.name");
            if (std::none_of(output.tools.begin(), output.tools.end(),
                             [&](const ToolDefinition& tool) { return tool.name == name; })) {
                bad_request("tool_choice names function '" + name + "', which is not in tools",
                            "tool_choice");
            }
            output.tool_choice.mode        = ToolChoiceMode::Auto;
            output.tool_choice.requirement = ninferno::ToolCallRequirement::Named;
            output.tool_choice.named       = name;
        } else if (type == "custom") {
            if (!choice.contains("custom") || !choice.at("custom").is_object()) {
                bad_request("custom tool_choice must contain a custom object", "tool_choice");
            }
            // Custom tools are served as functions with one string input, so a custom choice is
            // the named choice of that function.
            const std::string name =
                require_tool_name(choice.at("custom"), "tool_choice.custom.name");
            if (std::none_of(output.tools.begin(), output.tools.end(),
                             [&](const ToolDefinition& tool) { return tool.name == name; })) {
                bad_request("tool_choice names custom tool '" + name + "', which is not in tools",
                            "tool_choice");
            }
            output.tool_choice.mode        = ToolChoiceMode::Auto;
            output.tool_choice.requirement = ninferno::ToolCallRequirement::Named;
            output.tool_choice.named       = name;
        } else {
            bad_request("unsupported tool_choice type: " + ascii_preview(type), "tool_choice");
        }
    } else {
        bad_request("tool_choice must be a string or object", "tool_choice");
    }
}

void parse_parallel_tool_calls(const Json& body, GenerationRequest& output) {
    if (!body.contains("parallel_tool_calls") || body.at("parallel_tool_calls").is_null()) {
        return;
    }
    if (!body.at("parallel_tool_calls").is_boolean()) {
        bad_request("parallel_tool_calls must be a boolean", "parallel_tool_calls");
    }
    // false is enforced by a tool-call constraint that stops after the first call.
    output.parallel_tool_calls = body.at("parallel_tool_calls").get<bool>();
}

void parse_stop(const Json& body, GenerationRequest& output) {
    output.ignore_eos = optional_bool(body, "ignore_eos", false);
    if (!body.contains("stop") || body.at("stop").is_null()) { return; }
    // Stops end the answer, as on the Anthropic path: a client's "\n\n" or "```" must not end a
    // thinking model's turn inside its reasoning with an empty answer.
    append_stop_strings(body.at("stop"), "stop", true, output.stop_strings);
}

void parse_sampling(const Json& body, GenerationRequest& output) {
    SamplingParams& sampling   = output.sampling;
    sampling.temperature       = optional_number(body, "temperature");
    sampling.top_p             = optional_number(body, "top_p");
    sampling.presence_penalty  = optional_number(body, "presence_penalty");
    sampling.frequency_penalty = optional_number(body, "frequency_penalty");
    sampling.seed              = get_seed(body);

    // vLLM and SGLang expose top_k/min_p on their OpenAI-compatible endpoints; both map directly
    // to NInferno's native sampler and are useful for Qwen's published sampling presets.
    sampling.top_k = optional_int(body, "top_k");
    sampling.min_p = optional_number(body, "min_p");

    if (const std::optional<int> count = optional_int(body, "n")) {
        if (*count != 1) {
            bad_request(
                "n requests multiple completions, while NInferno produces one completion per "
                "request; only n=1 is supported",
                "n", "n_not_supported");
        }
    }
}

struct TemplateOptions {
    std::string kwargs_json;
    std::optional<bool> enable_thinking;
    std::optional<bool> preserve_thinking;
};

TemplateOptions parse_template_options(const Json& body) {
    // Qwen's deployment examples use both top-level aliases and
    // chat_template_kwargs. vLLM/SGLang expose the kwargs form. They are normalized here so the
    // Engine sees one value and conflicting aliases fail before execution.
    TemplateOptions output;
    output.enable_thinking   = nullable_bool(body, "enable_thinking");
    output.preserve_thinking = nullable_bool(body, "preserve_thinking");
    // The same switch as other servers spell it: Ollama's "think": bool and the Anthropic-style
    // "thinking": {"type": "enabled" | "disabled"} that OpenAI-compatible proxies forward.
    std::optional<bool> alias = nullable_bool(body, "think");
    if (body.contains("thinking") && body.at("thinking").is_object() &&
        body.at("thinking").contains("type") && body.at("thinking").at("type").is_string()) {
        const auto type = body.at("thinking").at("type").get<std::string>();
        if (type == "enabled" || type == "adaptive") alias = true;
        if (type == "disabled") alias = false;
    }
    if (alias) {
        if (output.enable_thinking && *output.enable_thinking != *alias) {
            bad_request("conflicting enable_thinking values", "enable_thinking",
                        "conflicting_template_option");
        }
        output.enable_thinking = alias;
    }

    if (!body.contains("chat_template_kwargs") || body.at("chat_template_kwargs").is_null()) {
        return output;
    }
    const Json& kwargs = body.at("chat_template_kwargs");
    if (!kwargs.is_object()) {
        bad_request("chat_template_kwargs must be an object", "chat_template_kwargs");
    }
    output.kwargs_json = kwargs.dump();
    auto merge         = [&](const char* key, std::optional<bool>& top_level) {
        const std::optional<bool> nested =
            nullable_bool(kwargs, key, std::string("chat_template_kwargs.") + key);
        if (top_level && nested && *top_level != *nested) {
            bad_request(std::string("conflicting ") + key + " values", key,
                                "conflicting_template_option");
        }
        if (nested) { top_level = nested; }
    };
    merge("enable_thinking", output.enable_thinking);
    merge("preserve_thinking", output.preserve_thinking);
    return output;
}

void parse_reasoning_effort(const Json& body, GenerationRequest& output) {
    if (!body.contains("reasoning_effort") || body.at("reasoning_effort").is_null()) { return; }
    if (!body.at("reasoning_effort").is_string()) {
        bad_request("reasoning_effort must be a string or null", "reasoning_effort");
    }
    const std::string value = body.at("reasoning_effort").get<std::string>();
    // Client aliases for "use the server-configured level" are accepted as an omitted request.
    if (value == "default" || value == "auto") { return; }
    const std::optional<ninferno::ReasoningEffort> parsed = ninferno::parse_reasoning_effort(value);
    if (!parsed) {
        bad_request("reasoning_effort must be one of none, minimal, low, medium, high, xhigh, or "
                    "max, or the aliases default/auto",
                    "reasoning_effort");
    }
    output.reasoning_effort = *parsed;
}

void parse_stream_options(const Json& body, OpenAIChatRequest& output) {
    output.stream = optional_bool(body, "stream", false);
    if (!body.contains("stream_options") || body.at("stream_options").is_null()) { return; }
    const Json& options = body.at("stream_options");
    if (!options.is_object()) { bad_request("stream_options must be an object", "stream_options"); }
    output.include_usage = optional_bool(options, "include_usage", false);
    if (options.contains("include_obfuscation") && !options.at("include_obfuscation").is_null() &&
        !options.at("include_obfuscation").is_boolean()) {
        bad_request("include_obfuscation must be a boolean", "stream_options");
    }
}

void parse_response_observations(const Json& body, OpenAIChatRequest& output) {
    output.timings_per_token = optional_bool(body, "timings_per_token", false);
    output.return_progress   = optional_bool(body, "return_progress", false);
}

void parse_output_limit(const Json& body, const RequestLimits& limits, OpenAIChatRequest& output) {
    std::optional<int> limit = optional_int(body, "max_completion_tokens");
    const char* param        = "max_completion_tokens";
    if (!limit) {
        limit = optional_int(body, "max_tokens");
        param = "max_tokens";
    }
    // -1 is how llama.cpp-style clients (LM Studio, SillyTavern) say "no limit": the server's
    // default, as when the field is absent.
    if (limit && *limit < 0) { limit.reset(); }
    if (limit) {
        output.generation.max_tokens  = *limit;
        output.output_tokens_explicit = true;
    } else {
        output.generation.max_tokens = limits.default_max_tokens;
    }
}

} // namespace

OpenAIChatRequest parse_chat_completion_request(const Json& body, const RequestLimits& limits,
                                                bool auto_system_shared_prefix) {
    require_object(body, "request body must be a JSON object");
    validate_standard_output_controls(body);
    std::optional<OutputConstraintRequest> constraint = parse_output_constraint(body);
    validate_compatibility_hints(body);

    OpenAIChatRequest output;
    if (!body.contains("model") || !body.at("model").is_string() ||
        body.at("model").get<std::string>().empty()) {
        bad_request("missing required field: model", "model");
    }
    output.model = body.at("model").get<std::string>();

    OpenAIPromptCachePolicy cache_policy   = parse_openai_prompt_cache_policy(body);
    cache_policy.auto_system_shared_prefix = auto_system_shared_prefix;

    parse_tools(body, output.generation);
    parse_tool_choice(body, output.generation);
    parse_parallel_tool_calls(body, output.generation);
    parse_messages(body, limits, output.generation);
    parse_stop(body, output.generation);
    parse_sampling(body, output.generation);
    parse_stream_options(body, output);
    parse_response_observations(body, output);
    parse_output_limit(body, limits, output);
    parse_reasoning_effort(body, output.generation);
    const TemplateOptions template_options      = parse_template_options(body);
    output.generation.enable_thinking           = template_options.enable_thinking;
    output.generation.preserve_thinking         = template_options.preserve_thinking;
    output.generation.chat_template_kwargs_json = template_options.kwargs_json;
    apply_openai_prompt_cache_policy(output.generation, cache_policy);
    check_ninferno_extension(body);
    output.generation.output_constraint = std::move(constraint);
    attach_tool_call_constraint(output.generation);
    // What the parser reads, plus standard fields with no local effect (metadata, prediction,
    // service_tier, a text-only audio configuration).
    static constexpr std::string_view kKnown[] = {"audio",
                                                  "chat_template_kwargs",
                                                  "enable_thinking",
                                                  "think",
                                                  "thinking",
                                                  "frequency_penalty",
                                                  "function_call",
                                                  "functions",
                                                  "grammar",
                                                  "guided_choice",
                                                  "guided_decoding_backend",
                                                  "guided_grammar",
                                                  "guided_json",
                                                  "guided_regex",
                                                  "guided_whitespace_pattern",
                                                  "ignore_eos",
                                                  "logit_bias",
                                                  "logprobs",
                                                  "max_completion_tokens",
                                                  "max_tokens",
                                                  "messages",
                                                  "metadata",
                                                  "min_p",
                                                  "mm_processor_kwargs",
                                                  "modalities",
                                                  "model",
                                                  "moderation",
                                                  "n",
                                                  "ninferno",
                                                  "parallel_tool_calls",
                                                  "prediction",
                                                  "presence_penalty",
                                                  "preserve_thinking",
                                                  "prompt_cache_key",
                                                  "prompt_cache_options",
                                                  "prompt_cache_retention",
                                                  "reasoning_effort",
                                                  "repetition_penalty",
                                                  "response_format",
                                                  "return_progress",
                                                  "safety_identifier",
                                                  "seed",
                                                  "service_tier",
                                                  "stop",
                                                  "store",
                                                  "stream",
                                                  "stream_options",
                                                  "structured_outputs",
                                                  "temperature",
                                                  "timings_per_token",
                                                  "tool_choice",
                                                  "tools",
                                                  "top_k",
                                                  "top_logprobs",
                                                  "top_p",
                                                  "user",
                                                  "verbosity",
                                                  "web_search_options"};
    note_unknown_fields(body, kKnown, output.generation);
    return output;
}

} // namespace ninferno::serve
