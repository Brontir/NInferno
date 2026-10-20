#pragma once

// OpenAI wire objects shared by Chat Completions and Responses HTTP handlers.

#include "serve/request.h"
#include "serve/request_json.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <string_view>

namespace ninferno::serve {

enum class OpenAIPromptCacheAutomatic : std::uint8_t {
    Default,
    Requested,
    Disabled,
};

struct OpenAIPromptCachePolicy {
    OpenAIPromptCacheAutomatic automatic = OpenAIPromptCacheAutomatic::Default;
    // With automatic writes on, also a shared-prefix candidate at the end of the leading
    // system/developer run, so sibling sessions share that head without a client marker.
    bool auto_system_shared_prefix = true;
};

[[nodiscard]] bool parse_openai_prompt_cache_breakpoint(const RequestJson& value,
                                                        std::string_view param);
[[nodiscard]] OpenAIPromptCachePolicy parse_openai_prompt_cache_policy(const RequestJson& body);
void apply_openai_prompt_cache_policy(GenerationRequest& request, OpenAIPromptCachePolicy policy);

std::string make_models_list(const std::string& model_id, std::int64_t created,
                             std::uint32_t max_model_len, const ninferno::ModelMetadata& metadata);
std::string make_model_object(const std::string& model_id, std::int64_t created,
                              std::uint32_t max_model_len, const ninferno::ModelMetadata& metadata);
// GET /props: llama.cpp's server properties, as clients read them for the context and slots.
std::string make_props(const std::string& model_id, std::uint32_t max_context,
                       std::uint32_t slots, const ninferno::ModelMetadata& metadata);
// GET /metrics: Prometheus text exposition under llama.cpp's metric names (dashboards made for it
// read ours), plus ninferno's own counters.
std::string make_prometheus_metrics(const ninferno::RuntimeStats& stats,
                                    std::uint32_t http_active_requests);
std::string make_error_body(const ApiError& error);
std::int64_t unix_time_now();

std::string new_openai_chat_completion_id();
std::string new_openai_chat_tool_call_id();
std::string new_openai_request_id();
std::string new_openai_response_id();
std::string new_openai_response_item_id(std::string_view prefix);

// Custom tools carry no declared JSON Schema: their input is free-form text. NInferno serves them
// as a single-string-input function ("input") under the same name, so callers that dispatch by tool
// name (the GitHub Copilot CLI, Codex, MCP clients) keep working. A declared `format` (text, or a
// grammar: Chat Completions nests it as {"grammar": {"definition", "syntax"}}, Responses writes
// "definition" and "syntax" on the format) is carried as descriptive prompt metadata only, because
// the engine has no constrained decoding for it.
[[nodiscard]] std::string custom_tool_input_schema(const RequestJson& custom);

// The input a custom tool call carries: the "input" string of its arguments, or the arguments as
// written when the model wrote something else.
[[nodiscard]] std::string custom_tool_call_input(std::string_view arguments_json);

} // namespace ninferno::serve
