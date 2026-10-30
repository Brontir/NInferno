#include "serve/anthropic_messages.h"

#include "serve/generation_service.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Json = ninferno::serve::RequestJson;
using namespace ninferno::serve;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

Json base_request() {
    return Json{{"model", "claude-local"},
                {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})},
                {"max_tokens", 4096}};
}

RequestLimits limits() {
    RequestLimits value;
    value.default_max_tokens = 8192;
    return value;
}

AnthropicMessagesRequest parse(const Json& body) {
    return parse_anthropic_messages_request(body, limits());
}

std::string api_code(const std::function<void()>& action) {
    try {
        action();
    } catch (const ApiException& error) { return error.error().code; } catch (...) {
        return "wrong_exception";
    }
    return {};
}

std::string api_param(const std::function<void()>& action) {
    try {
        action();
    } catch (const ApiException& error) { return error.error().param; } catch (...) {
        return "wrong_exception";
    }
    return {};
}

ResolvedPromptSemantics semantics(const GenerationRequest& request, bool default_thinking = true) {
    ServeOptions options;
    options.enable_thinking = default_thinking;
    return resolve_prompt_semantics(request, options);
}

ninferno::PromptInput prompt(const GenerationRequest& request) {
    return to_prompt_input(request, semantics(request), [](const ContentPart& part) {
        ninferno::OwnedMedia media;
        media.kind                = part.kind == ContentKind::Image ? ninferno::MediaKind::Image
                                                                    : ninferno::MediaKind::Video;
        media.bytes               = {1};
        media.image_resize_policy = part.image_resize_policy;
        return media;
    });
}

Json parse_event(const std::string& value) {
    const std::size_t newline = value.find('\n');
    if (!value.starts_with("event: ") || newline == std::string::npos || !value.ends_with("\n\n")) {
        throw std::runtime_error("invalid Anthropic SSE framing");
    }
    const std::string type   = value.substr(7, newline - 7);
    const std::string prefix = "data: ";
    if (value.compare(newline + 1U, prefix.size(), prefix) != 0) {
        throw std::runtime_error("missing Anthropic SSE data");
    }
    Json payload = Json::parse(value.substr(newline + 1U + prefix.size(),
                                            value.size() - newline - 1U - prefix.size() - 2U));
    if (payload.at("type") != type) { throw std::runtime_error("Anthropic SSE type mismatch"); }
    return payload;
}

int test_envelope_and_field_policy() {
    Json body             = base_request();
    body["stream"]        = true;
    body["temperature"]   = 0.7;
    body["top_p"]         = 0.8;
    body["top_k"]         = 20;
    body["metadata"]      = Json{{"user_id", "u"}};
    body["service_tier"]  = "auto";
    body["inference_geo"] = "anywhere";
    body["future_field"]  = Json{{"unknown", true}};
    body["output_config"] = Json{{"effort", "low"}, {"future_option", true}};

    const AnthropicMessagesRequest request = parse(body);
    Json depth                             = base_request();
    depth["ninferno"]                      = Json{{"draft_tokens", 4}};
    Json too_deep                          = base_request();
    too_deep["ninferno"]                   = Json{{"draft_tokens", 16}};
    if (!(parse(depth).generation.draft_tokens == 4U && !request.generation.draft_tokens &&
          api_code([&] { (void)parse(too_deep); }) == "invalid_draft_tokens")) {
        std::cerr << "FAIL: ninferno.draft_tokens through the Messages dialect\n";
        return 1;
    }
    int failures =
        check(request.model == "claude-local" && request.stream && request.output_tokens_explicit &&
                  request.generation.max_tokens == 4096,
              "Messages envelope fields were not preserved");
    failures += check(request.generation.sampling.temperature == 0.7 &&
                          request.generation.sampling.top_p == 0.8 &&
                          request.generation.sampling.top_k == 20 &&
                          request.generation.reasoning_effort == ninferno::ReasoningEffort::Low,
                      "executable Anthropic generation fields were not lowered");

    body.erase("max_tokens");
    const AnthropicMessagesRequest defaulted = parse(body);
    failures += check(!defaulted.output_tokens_explicit && defaulted.generation.max_tokens == 8192,
                      "omitted max_tokens did not use the server default");
    body["max_tokens"] = 0;
    failures += check(api_code([&] { (void)parse(body); }) == "cache_prewarm_not_supported",
                      "max_tokens=0 was accepted as a false cache prewarm");

    body                = base_request();
    body["temperature"] = 1.01;
    failures += check(api_param([&] { (void)parse(body); }) == "temperature",
                      "Anthropic temperature range was not enforced");
    body          = base_request();
    body["top_k"] = ninferno::kMaximumTopK + 1;
    failures += check(api_param([&] { (void)parse(body); }) == "top_k",
                      "Engine top_k range was not enforced");
    body                  = base_request();
    body["output_config"] = Json{{"format", Json{{"type", "json_text"}}}};
    failures += check(api_code([&] { (void)parse(body); }) == "output_config_format_not_supported",
                      "unsupported structured output type was silently downgraded");
    body                  = base_request();
    body["output_config"] = Json{{"format", Json{{"type", "json_schema"}}}};
    failures += check(api_param([&] { (void)parse(body); }) == "output_config.format",
                      "json_schema without a schema was accepted");
    body                  = base_request();
    body["output_config"] = Json{
        {"format", Json{{"type", "json_schema"},
                        {"schema", Json{{"type", "object"},
                                        {"properties", Json{{"z", Json{{"type", "string"}}}}}}}}}};
    {
        const auto constraint = parse(body).generation.output_constraint;
        failures +=
            check(constraint && constraint->kind == ninferno::OutputConstraintKind::JsonSchema &&
                      constraint->param == "output_config.format" &&
                      constraint->definition.find("\"z\"") != std::string::npos,
                  "output_config.format json_schema is enforced");
    }
    body["tools"] = Json::array({Json{{"name", "weather"},
                                      {"input_schema", Json{{"type", "object"}}}}});
    {
        const auto constraint = parse(body).generation.output_constraint;
        failures +=
            check(constraint && constraint->kind == ninferno::OutputConstraintKind::ToolCalls &&
                      constraint->answer_schema.find("\"z\"") != std::string::npos,
                  "output_config.format with enabled tools did not constrain the answer");
    }
    body              = base_request();
    body["container"] = "container_1";
    failures += check(api_code([&] { (void)parse(body); }) == "container_not_supported",
                      "container execution was silently ignored");
    return failures;
}

int test_message_normalization() {
    Json body        = base_request();
    body["thinking"] = Json{{"type", "disabled"}};
    body["system"] =
        Json::array({Json{{"type", "text"}, {"text", "A"}},
                     Json{{"type", "text"},
                          {"text", "B"},
                          {"cache_control", Json{{"type", "ephemeral"}, {"ttl", "1h"}}}}});
    body["messages"] =
        Json::array({Json{{"role", "user"},
                          {"content", Json::array({Json{{"type", "text"}, {"text", "one"}},
                                                   Json{{"type", "text"}, {"text", "two"}}})}},
                     Json{{"role", "user"}, {"content", "three"}},
                     Json{{"role", "assistant"}, {"content", "prefix"}}});

    const GenerationRequest request = parse(body).generation;
    int failures                    = check(request.messages.size() == 3 &&
                                                request.messages[0].role == ninferno::ChatRole::System &&
                                                request.messages[1].role == ninferno::ChatRole::User &&
                                                request.messages[1].content.size() == 3,
                                            "consecutive Anthropic roles were not merged by block concatenation");
    failures += check(request.messages[0].content[0].text == "A" &&
                          request.messages[0].content[1].text == "B" &&
                          request.messages[1].content[0].text == "one" &&
                          request.messages[1].content[1].text == "two" &&
                          request.messages[1].content[2].text == "three",
                      "message normalization inserted or removed text");
    failures +=
        check(request.continuation == ninferno::PromptContinuationMode::ContinueFinalAssistant,
              "final assistant text did not select continuation mode");
    const ninferno::PromptInput translated = prompt(request);
    failures += check(translated.options.continuation ==
                              ninferno::PromptContinuationMode::ContinueFinalAssistant &&
                          translated.context_cache.markers.size() == 1 &&
                          translated.context_cache.markers[0].location ==
                              ninferno::PromptCacheMarkerLocation::LeadingInstructionBoundary,
                      "assistant continuation or system block cache boundary was lost");
    return failures;
}

int test_attribution_system_block() {
    const auto attributed = [](std::string fingerprint) {
        Json body        = base_request();
        body["thinking"] = Json{{"type", "disabled"}};
        body["system"]   = Json::array(
            {Json{{"type", "text"},
                    {"text", "x-anthropic-billing-header: cc_version=2.1; cch=" + fingerprint}},
               Json{{"type", "text"},
                    {"text", "real system"},
                    {"cache_control", Json{{"type", "ephemeral"}}}}});
        return body;
    };

    const Json first_body                 = attributed(".89c");
    const GenerationRequest first_request = parse(first_body).generation;
    const ninferno::PromptInput first       = prompt(first_request);
    const ninferno::PromptInput second      = prompt(parse(attributed(".8f7")).generation);
    const ninferno::PromptInput counted =
        prompt(parse_anthropic_count_tokens_request(first_body).generation);

    const auto has_real_system = [](const ninferno::PromptInput& input) {
        return input.messages.size() == 2 &&
               input.messages.front().role == ninferno::ChatRole::System &&
               input.messages.front().parts.size() == 1 &&
               input.messages.front().parts.front().text == "real system" &&
               input.context_cache.markers.size() == 1 &&
               input.context_cache.markers.front().location ==
                   ninferno::PromptCacheMarkerLocation::LeadingInstructionBoundary;
    };
    int failures =
        check(has_real_system(first) && has_real_system(second) && has_real_system(counted) &&
                  first.messages.front().parts.front().text ==
                      second.messages.front().parts.front().text &&
                  first.context_cache.markers == second.context_cache.markers &&
                  first.context_cache.markers == counted.context_cache.markers,
              "Anthropic attribution changed model input or Count Tokens lowering");

    Json attribution_only        = base_request();
    attribution_only["thinking"] = Json{{"type", "disabled"}};
    attribution_only["system"]   = Json::array(
        {Json{{"type", "text"}, {"text", "x-anthropic-billing-header: cc_version=2.1"}}});
    const GenerationRequest without_empty = parse(attribution_only).generation;
    failures += check(without_empty.messages.size() == 1 &&
                          without_empty.messages.front().role == ninferno::ChatRole::User,
                      "attribution-only system produced an empty model turn");

    Json non_first        = base_request();
    non_first["thinking"] = Json{{"type", "disabled"}};
    non_first["system"] =
        Json::array({Json{{"type", "text"}, {"text", "real system"}},
                     Json{{"type", "text"}, {"text", "x-anthropic-billing-header: user text"}}});
    const GenerationRequest preserved_non_first = parse(non_first).generation;
    failures += check(preserved_non_first.messages.size() == 2 &&
                          preserved_non_first.messages.front().content.size() == 2 &&
                          preserved_non_first.messages.front().content.back().text ==
                              "x-anthropic-billing-header: user text",
                      "non-first attribution-like system text was consumed");

    Json string_system                       = base_request();
    string_system["thinking"]                = Json{{"type", "disabled"}};
    string_system["system"]                  = "x-anthropic-billing-header: ordinary string";
    const GenerationRequest preserved_string = parse(string_system).generation;
    failures += check(preserved_string.messages.size() == 2 &&
                          preserved_string.messages.front().content.size() == 1 &&
                          preserved_string.messages.front().content.front().text ==
                              "x-anthropic-billing-header: ordinary string",
                      "ordinary string system content was consumed as attribution");
    return failures;
}

Json tool_use(std::string id, std::string name = "lookup") {
    return Json{{"type", "tool_use"},
                {"id", std::move(id)},
                {"name", std::move(name)},
                {"input", Json::object()}};
}

Json tool_result(std::string id, std::string content, bool is_error = false) {
    return Json{{"type", "tool_result"},
                {"tool_use_id", std::move(id)},
                {"content", std::move(content)},
                {"is_error", is_error}};
}

int test_tool_history() {
    Json body        = base_request();
    body["messages"] = Json::array(
        {Json{{"role", "assistant"},
              {"content", Json::array({tool_use("toolu_a"), tool_use("toolu_b")})}},
         Json{{"role", "user"},
              {"content", Json::array({Json{{"type", "tool_result"},
                                            {"tool_use_id", "toolu_b"},
                                            {"content", "result B"},
                                            {"is_error", true},
                                            {"cache_control", Json{{"type", "ephemeral"}}}},
                                       tool_result("toolu_a", "result A"),
                                       Json{{"type", "text"}, {"text", "continue"}}})}}});
    const GenerationRequest normalized          = parse(body).generation;
    int failures                                  = check(normalized.messages.size() == 4 &&
                                                              normalized.messages[0].role == ninferno::ChatRole::Assistant &&
                                                              normalized.messages[1].role == ninferno::ChatRole::Tool &&
                                                              normalized.messages[1].tool_call_id == "toolu_a" &&
                                                              normalized.messages[1].content[0].text == "result A" &&
                                                              normalized.messages[2].role == ninferno::ChatRole::Tool &&
                                                              normalized.messages[2].tool_call_id == "toolu_b" &&
                                                              normalized.messages[2].content[0].text == "result B" &&
                                                              normalized.messages[2].tool_result_is_error &&
                                                              normalized.messages[2].cache_boundary_after &&
                                                              normalized.messages[2].cache_boundary_after->kind ==
                                                                  ninferno::PromptCacheMarkerKind::SharedStablePrefix &&
                                                              normalized.messages[3].role == ninferno::ChatRole::User &&
                                                              normalized.messages[3].content[0].text == "continue",
                                                          "valid out-of-order tool results were not associated by ID");
    const ninferno::PromptInput normalized_prompt = prompt(normalized);
    failures += check(normalized_prompt.messages[2].parts.size() == 2 &&
                          normalized_prompt.messages[2].parts[0].text == "[tool_error]\n" &&
                          normalized_prompt.messages[2].parts[1].text == "result B",
                      "tool result error state did not follow its ID during normalization");

    body["messages"][1]["content"] =
        Json::array({tool_result("toolu_unknown", "wrong"), tool_result("toolu_a", "A")});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history" &&
                          api_param([&] { (void)parse(body); }) == "messages",
                      "unknown tool_result ID was accepted");

    body["messages"][1]["content"] =
        Json::array({tool_result("toolu_a", "first"), tool_result("toolu_a", "duplicate")});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history",
                      "duplicate tool_result ID was accepted");

    body["messages"][1]["content"] = Json::array({tool_result("toolu_a", "only one")});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history",
                      "missing tool_result was accepted");

    body["messages"] =
        Json::array({Json{{"role", "assistant"}, {"content", Json::array({tool_use("toolu_a")})}},
                     Json{{"role", "user"},
                          {"content", Json::array({Json{{"type", "text"}, {"text", "before"}},
                                                   tool_result("toolu_a", "result")})}}});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history",
                      "tool_result after ordinary user content was accepted");

    body["messages"] = Json::array(
        {Json{{"role", "assistant"}, {"content", Json::array({tool_use("toolu_a")})}},
         Json{{"role", "user"}, {"content", "before"}},
         Json{{"role", "user"}, {"content", Json::array({tool_result("toolu_a", "result")})}}});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history",
                      "same-role message joining bypassed tool_result ordering");

    body["messages"] = Json::array(
        {Json{{"role", "user"},
              {"content", Json::array({tool_result("toolu_external", "imported"),
                                       Json{{"type", "text"}, {"text", "continue"}}})}}});
    const GenerationRequest truncated = parse(body).generation;
    failures += check(truncated.messages.size() == 2 &&
                          truncated.messages[0].role == ninferno::ChatRole::Tool &&
                          truncated.messages[0].tool_call_id == "toolu_external" &&
                          truncated.messages[1].role == ninferno::ChatRole::User,
                      "leading tool_result from a truncated history was rejected or reordered");

    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "first"}},
         Json{{"role", "assistant"}, {"content", "ordinary"}},
         Json{{"role", "user"}, {"content", Json::array({tool_result("toolu_orphan", "late")})}}});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history",
                      "orphan tool_result inside a visible history was accepted");

    body["messages"] = Json::array(
        {Json{{"role", "assistant"},
              {"content", Json::array({tool_use("toolu_same"), tool_use("toolu_same", "other")})}},
         Json{{"role", "user"}, {"content", Json::array({tool_result("toolu_same", "result")})}}});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history",
                      "duplicate tool_use ID was accepted");
    failures += check(api_code([&] { (void)parse_anthropic_count_tokens_request(body); }) ==
                          "invalid_tool_history",
                      "Count Tokens did not share Messages tool-history validation");
    return failures;
}

Json ordinary_tool(bool strict = false) {
    return Json{{"name", "weather"},
                {"description", "Get weather"},
                {"input_schema", Json{{"type", "object"},
                                      {"properties", Json{{"city", Json{{"type", "string"}}}}}}},
                {"input_examples", Json::array({Json{{"city", "Paris"}}})},
                {"strict", strict}};
}

// System messages are accepted anywhere in `messages`: in place, and after the tool_result turn
// when they sit between a tool_use and its results.
int test_system_messages_anywhere() {
    const auto roles = [](const GenerationRequest& request) {
        std::vector<ninferno::ChatRole> out;
        for (const auto& message : request.messages) { out.push_back(message.role); }
        return out;
    };
    using ninferno::ChatRole;
    Json body        = base_request();
    body["messages"] = Json::array({Json{{"role", "system"}, {"content", "be brief"}},
                                    Json{{"role", "user"}, {"content", "hi"}}});
    int failures     = check(roles(parse(body).generation) ==
                                 std::vector<ChatRole>{ChatRole::System, ChatRole::User},
                             "a leading system message was refused or moved");

    body["messages"] = Json::array({Json{{"role", "user"}, {"content", "hi"}},
                                    Json{{"role", "system"}, {"content", "reminder"}},
                                    Json{{"role", "user"}, {"content", "go on"}}});
    failures += check(roles(parse(body).generation) ==
                          std::vector<ChatRole>{ChatRole::User, ChatRole::System, ChatRole::User},
                      "a system message between user turns was refused or moved");

    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "look it up"}},
         Json{{"role", "assistant"}, {"content", Json::array({tool_use("toolu_a")})}},
         Json{{"role", "system"}, {"content", "reminder"}},
         Json{{"role", "user"}, {"content", Json::array({tool_result("toolu_a", "found")})}}});
    failures += check(roles(parse(body).generation) ==
                          std::vector<ChatRole>{ChatRole::User, ChatRole::Assistant, ChatRole::Tool,
                                                ChatRole::System},
                      "a system message between tool_use and tool_result did not move after it");
    return failures;
}

int test_tools() {
    Json body                       = base_request();
    body["tools"]                   = Json::array({ordinary_tool()});
    body["tool_choice"]             = Json{{"type", "auto"}, {"disable_parallel_tool_use", false}};
    const GenerationRequest request = parse(body).generation;
    const ninferno::PromptInput translated = prompt(request);
    const Json rendered                  = Json::parse(translated.options.tool_jsons.at(0));
    int failures = check(request.uses_tools() && rendered["function"]["name"] == "weather" &&
                             rendered["function"]["input_examples"].is_array(),
                         "Anthropic tool schema/examples did not reach the Qwen prompt");

    body["tools"] = Json::array({ordinary_tool(true)});
    {
        const auto strict = parse(body).generation.output_constraint;
        failures += check(strict && strict->kind == ninferno::OutputConstraintKind::ToolCalls &&
                              strict->tools.size() == 1 && strict->tools[0].strict,
                          "active strict tool is not enforced by a tool-call constraint");
    }
    body["tool_choice"]               = Json{{"type", "none"}, {"disable_parallel_tool_use", true}};
    body["tools"][0]["defer_loading"] = true;
    body["tools"][0]["allowed_callers"] = Json::array({"code_execution"});
    const GenerationRequest disabled    = parse(body).generation;
    failures += check(!disabled.uses_tools() && prompt(disabled).options.tool_jsons.empty(),
                      "tool_choice:none did not neutralize inactive tool guarantees");
    // tool_choice none renders the tools an auto turn renders, with calls disabled.
    Json none_ordinary                    = base_request();
    none_ordinary["tools"]                = Json::array({ordinary_tool()});
    none_ordinary["tool_choice"]          = Json{{"type", "none"}};
    Json auto_ordinary                    = none_ordinary;
    auto_ordinary["tool_choice"]          = Json{{"type", "auto"}};
    const ninferno::PromptInput none_prompt = prompt(parse(none_ordinary).generation);
    failures += check(!none_prompt.options.tool_calls_enabled &&
                          none_prompt.options.tool_jsons ==
                              prompt(parse(auto_ordinary).generation).options.tool_jsons,
                      "tool_choice:none did not render the auto turn's tools");

    body                = base_request();
    body["tools"]       = Json::array({ordinary_tool()});
    body["tool_choice"] = Json{{"type", "any"}};
    failures += check(parse(body).generation.output_constraint->tool_requirement ==
                          ninferno::ToolCallRequirement::Required,
                      "forced any-tool choice is not enforced");
    body["tool_choice"] = Json{{"type", "tool"}, {"name", "weather"}};
    {
        const auto named = parse(body).generation.output_constraint;
        failures += check(named->tool_requirement == ninferno::ToolCallRequirement::Named &&
                              named->named_tool == "weather",
                          "named tool choice is not enforced");
    }
    body["tool_choice"] = Json{{"type", "auto"}, {"disable_parallel_tool_use", true}};
    failures += check(!parse(body).generation.output_constraint->parallel_tool_calls,
                      "active single-tool-call guarantee is not enforced");

    body          = base_request();
    body["tools"] = Json::array({Json{{"type", "web_search_20250305"}, {"name", "web_search"}}});
    failures += check(api_code([&] { (void)parse(body); }) == "anthropic_tools_not_supported",
                      "Anthropic-provided tool was treated as a user tool");
    body["tool_choice"] = Json{{"type", "none"}};
    failures += check(api_code([&] { (void)parse(body); }).empty(),
                      "inactive Anthropic-provided tool unnecessarily blocked generation");

    body          = base_request();
    body["tools"] = Json::array({ordinary_tool(), ordinary_tool()});
    failures += check(api_param([&] { (void)parse(body); }) == "tools",
                      "duplicate tool names were accepted");
    return failures;
}

int test_prompt_object_order() {
    const auto parse_raw = [](std::string_view input) {
        const std::string body =
            R"({"model":"claude-local","thinking":{"type":"disabled"},"tools":[{"name":"probe","input_schema":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"object","properties":{"yankee":{"type":"integer"},"bravo":{"type":"boolean"}}}}}}],"messages":[{"role":"user","content":"run probe"},{"role":"assistant","content":[{"type":"tool_use","id":"toolu_order","name":"probe","input":)" +
            std::string(input) +
            R"(}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_order","content":"done"}]}],"max_tokens":32})";
        return parse(Json::parse(body)).generation;
    };

    const std::string original      = R"({"zeta":"last","alpha":{"yankee":2,"bravo":true}})";
    const GenerationRequest request = parse_raw(original);
    const auto assistant =
        std::find_if(request.messages.begin(), request.messages.end(), [](const ChatTurn& turn) {
            return turn.role == ninferno::ChatRole::Assistant;
        });
    const ninferno::PromptInput translated = prompt(request);
    const std::string expected_tool =
        R"({"type":"function","function":{"name":"probe","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"object","properties":{"yankee":{"type":"integer"},"bravo":{"type":"boolean"}}}}}}})";
    int failures = check(assistant != request.messages.end() && assistant->tool_calls.size() == 1 &&
                             assistant->tool_calls.front().arguments_json == original &&
                             translated.options.tool_jsons.size() == 1 &&
                             translated.options.tool_jsons.front() == expected_tool,
                         "Anthropic request changed prompt-bearing object member order");

    const GenerationRequest reordered =
        parse_raw(R"({"alpha":{"yankee":2,"bravo":true},"zeta":"last"})");
    const auto reordered_assistant = std::find_if(
        reordered.messages.begin(), reordered.messages.end(),
        [](const ChatTurn& turn) { return turn.role == ninferno::ChatRole::Assistant; });
    failures +=
        check(reordered_assistant != reordered.messages.end() &&
                  reordered_assistant->tool_calls.size() == 1 &&
                  reordered_assistant->tool_calls.front().arguments_json != original,
              "Anthropic request treated reordered tool input as the original representation");
    return failures;
}

int test_thinking_and_count_tokens() {
    Json body = base_request();
    body["thinking"] =
        Json{{"type", "enabled"}, {"budget_tokens", 1024}, {"display", "summarized"}};
    const GenerationRequest enabled = parse(body).generation;
    ServeOptions server;
    server.default_thinking_budget = 777;
    const ninferno::RequestOptions options =
        to_request_options(enabled, server, semantics(enabled), true);
    int failures = check(enabled.enable_thinking == true && enabled.thinking_budget == 1024 &&
                             options.execution.thinking.budget == 1024,
                         "request Thinking budget did not reach Engine options");

    // A budget at or above max_tokens is accepted and passed on; the output limit ends thinking
    // first. Qwen Code sends a fixed budget while shrinking max_tokens to the context left.
    for (const int budget : {4096, 128000}) {
        body["thinking"]["budget_tokens"] = budget;
        const GenerationRequest over      = parse(body).generation;
        failures += check(over.enable_thinking == true && over.max_tokens == 4096 &&
                              over.thinking_budget == static_cast<std::uint32_t>(budget),
                          "Thinking budget at or above max_tokens was rejected or altered");
    }
    body["thinking"]["budget_tokens"] = 1023;
    failures += check(api_param([&] { (void)parse(body); }) == "thinking",
                      "Thinking budget below 1024 was accepted");
    body["thinking"] = Json{{"type", "future"}};
    failures += check(api_param([&] { (void)parse(body); }) == "thinking",
                      "unknown Thinking mode defaulted to enabled");
    body["thinking"] = Json{{"type", "adaptive"}, {"display", "omitted"}};
    failures += check(parse(body).thinking_display == AnthropicThinkingDisplay::Omitted &&
                          parse(body).generation.enable_thinking,
                      "hidden adaptive Thinking was not accepted");
    body["thinking"] = Json{{"type", "adaptive"}, {"display", "summarized"}};
    failures += check(parse(body).thinking_display == AnthropicThinkingDisplay::Visible,
                      "summarized Thinking was not visible");

    body["max_tokens"]                        = 0;
    body["temperature"]                       = "ignored for counting";
    body["output_config"]                     = Json{{"format", Json{{"type", "json_schema"}}}};
    const AnthropicCountTokensRequest counted = parse_anthropic_count_tokens_request(body);
    failures += check(counted.generation.enable_thinking == true,
                      "Count Tokens did not share prompt-affecting Thinking parsing");

    body             = base_request();
    body["thinking"] = Json{{"type", "disabled"}};
    body["messages"].push_back(Json{{"role", "assistant"}, {"content", "prefix"}});
    failures += check(api_code([&] { (void)semantics(parse(body).generation); }).empty(),
                      "disabled-Thinking assistant prefill was rejected");
    body.erase("thinking");
    failures += check(api_code([&] { (void)semantics(parse(body).generation); }) ==
                          "assistant_prefill_not_supported",
                      "Thinking-on assistant prefill was not rejected at capability resolution");
    return failures;
}

int test_thinking_history_transport_metadata() {
    Json body = base_request();
    body["messages"] =
        Json::array({Json{{"role", "user"}, {"content", "before"}},
                     Json{{"role", "assistant"},
                          {"content", Json::array({Json{{"type", "thinking"},
                                                        {"thinking", "thought"},
                                                        {"signature", "opaque-from-stopped-serve"}},
                                                   Json{{"type", "text"}, {"text", "answer"}}})}},
                     Json{{"role", "user"}, {"content", "after"}}});
    const GenerationRequest accepted = parse(body).generation;
    int failures                     = check(accepted.messages.size() == 3 &&
                                                 accepted.messages[1].reasoning_content == "thought" &&
                                                 accepted.messages[1].content[0].text == "answer",
                                             "Thinking history with opaque transport metadata was not lowered");

    Json changed_signature                                      = body;
    changed_signature["messages"][1]["content"][0]["signature"] = "different";
    const GenerationRequest changed_metadata = parse(changed_signature).generation;
    failures += check(changed_metadata.messages[1].reasoning_content == "thought",
                      "opaque Thinking signature changed prompt semantics");

    Json missing_signature = body;
    missing_signature["messages"][1]["content"][0].erase("signature");
    const GenerationRequest missing_metadata = parse(missing_signature).generation;
    failures += check(missing_metadata.messages[1].reasoning_content == "thought",
                      "missing Thinking signature blocked visible history");

    Json structured_signature                                      = body;
    structured_signature["messages"][1]["content"][0]["signature"] = Json{{"provider", "external"}};
    const GenerationRequest structured_metadata = parse(structured_signature).generation;
    failures += check(structured_metadata.messages[1].reasoning_content == "thought",
                      "non-semantic Thinking metadata was interpreted by request lowering");

    Json changed_thinking                                     = body;
    changed_thinking["messages"][1]["content"][0]["thinking"] = "changed";
    const GenerationRequest changed_semantics                 = parse(changed_thinking).generation;
    failures += check(changed_semantics.messages[1].reasoning_content == "changed",
                      "visible Thinking text did not own prompt semantics");

    const AnthropicCountTokensRequest counted =
        parse_anthropic_count_tokens_request(missing_signature);
    failures += check(counted.generation.messages[1].reasoning_content == "thought",
                      "Count Tokens assigned semantics to Thinking signature metadata");
    return failures;
}

int test_content_and_cache_hints() {
    Json body                       = base_request();
    body["cache_control"]           = Json{{"type", "ephemeral"}, {"ttl", "5m"}};
    body["messages"]                = Json::array({Json{
                       {"role", "user"},
                       {"content",
                        Json::array({Json{{"type", "text"},
                                          {"text", "look"},
                                          {"cache_control", Json{{"type", "ephemeral"}}}},
                                     Json{{"type", "image"},
                                          {"source", Json{{"type", "url"}, {"url", "https://example/image.png"}}},
                                          {"transformations", Json{{"oversized_image", "error"}}}}})}}});
    const GenerationRequest request = parse(body).generation;
    int failures                    = check(request.messages[0].content[1].cache_boundary_after &&
                                                ninferno::has_shared_candidate_evidence(
                                 request.messages[0].content[1].cache_boundary_after->evidence,
                                 ninferno::SharedCandidateEvidence::RequestedAutomatic) &&
                                                request.media_item_count() == 1 &&
                                                request.messages[0].content[1].image_resize_policy ==
                                                    ninferno::ImageResizePolicy::RejectOversized,
                                            "automatic caching or image transformation policy was lost");
    const ninferno::PromptInput translated = prompt(request);
    failures += check(translated.context_cache.markers.size() == 2 &&
                          translated.context_cache.markers[1].location ==
                              ninferno::PromptCacheMarkerLocation::MessagePartBoundary,
                      "message-part cache boundary was not represented in PromptInput");

    body                           = base_request();
    body["messages"][0]["content"] = Json::array(
        {Json{{"type", "image"}, {"source", Json{{"type", "file"}, {"file_id", "file_1"}}}}});
    failures += check(api_code([&] { (void)parse(body); }) == "files_not_supported",
                      "Files image source was flattened or ignored");
    body["messages"][0]["content"] =
        Json::array({Json{{"type", "document"}, {"source", Json::object()}}});
    failures += check(api_code([&] { (void)parse(body); }) == "documents_not_supported",
                      "a document without a readable source was accepted");
    body["messages"][0]["content"] = Json::array(
        {Json{{"type", "document"},
              {"title", "notes"},
              {"source", Json{{"type", "text"}, {"media_type", "text/plain"}, {"data", "abc"}}}},
         Json{{"type", "document"},
              {"source", Json{{"type", "content"},
                              {"content", Json::array({Json{{"type", "text"}, {"text", "x"}}})}}}},
         Json{{"type", "text"}, {"text", "summarize"}}});
    bool parsed = true;
    try {
        (void)parse(body);
    } catch (const std::exception&) { parsed = false; }
    failures += check(parsed, "text and content documents were refused");
    body["messages"][0]["content"] = Json::array(
        {Json{{"type", "document"},
              {"title", "report.pdf"},
              {"source", Json{{"type", "base64"}, {"media_type", "application/pdf"}, {"data", "JVBE"}}}}});
    {
        const GenerationRequest pdf = parse(body).generation;
        failures += check(pdf.messages[0].content.size() == 1 &&
                              pdf.messages[0].content[0].kind == ContentKind::Document &&
                              pdf.messages[0].content[0].document_title == "report.pdf" &&
                              pdf.messages[0].content[0].source.value == "data:application/pdf;base64,JVBE",
                          "a base64 PDF document did not become a Document part");
        // The test acquirer hands back one byte, not a PDF: refused as not a PDF, not as vision.
        failures += check(api_code([&] { (void)prompt(pdf); }) == "documents_not_supported",
                          "bytes that are not a PDF were read as one");
        failures += check(pdf.media_item_count() == 0, "a PDF document counted as vision media");
    }
    body["messages"][0]["content"] = Json::array(
        {Json{{"type", "document"},
              {"source", Json{{"type", "base64"}, {"media_type", "text/csv"}, {"data", "YQ=="}}}}});
    failures += check(api_code([&] { (void)parse(body); }) == "documents_not_supported",
                      "a base64 document other than PDF was accepted");
    body["messages"][0]["content"] = Json::array(
        {Json{{"type", "document"}, {"source", Json{{"type", "file"}, {"file_id", "file_1"}}}}});
    failures += check(api_code([&] { (void)parse(body); }) == "files_not_supported",
                      "a Files API document was accepted");
    return failures;
}

GenerationOutcome sample_outcome() {
    GenerationOutcome outcome;
    outcome.text                            = "answer";
    outcome.reasoning                       = "thought";
    outcome.prompt_tokens                   = 100;
    outcome.completion_tokens               = 12;
    outcome.reasoning_tokens                = 5;
    outcome.finish_reason                   = ninferno::FinishReason::StopString;
    outcome.matched_stop_string             = "STOP";
    outcome.metrics.prefix_cache_hit_tokens = 60;
    return outcome;
}

int test_aggregate_and_errors() {
    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity("req_test", "claude-local");
    GenerationOutcome outcome = sample_outcome();
    const Json response       = Json::parse(make_anthropic_messages_response(identity, outcome));
    int failures =
        check(response["content"].size() == 2 && response["content"][0]["type"] == "thinking" &&
                  response["content"][0]["signature"] == identity.message_id &&
                  response["stop_reason"] == "stop_sequence" && response["stop_sequence"] == "STOP",
              "aggregate Thinking or stop-sequence presentation is incomplete");
    failures += check(response["usage"]["input_tokens"] == 40 &&
                          response["usage"]["cache_read_input_tokens"] == 60 &&
                          response["usage"]["cache_creation_input_tokens"].is_null() &&
                          response["usage"]["output_tokens_details"]["thinking_tokens"] == 5,
                      "aggregate cache/reasoning usage was fabricated or lost");

    outcome.prompt_tokens                   = 10;
    outcome.metrics.prefix_cache_hit_tokens = 1000;
    const Json clamped = Json::parse(make_anthropic_messages_response(identity, outcome));
    failures += check(clamped["usage"]["input_tokens"] == 0 &&
                          clamped["usage"]["cache_read_input_tokens"] == 10,
                      "cache-read usage was not clamped to the prompt token count");

    failures += check(!response["usage"].contains("ninferno"),
                      "usage carries no extension without a speculation config");
    outcome.metrics.speculative_config    = "dflash:7";
    outcome.metrics.speculative_selection = "fixed";
    outcome.metrics.speculative_reason    = "--spec-select fixed";
    outcome.metrics.speculative_fallback  = "watchdog_plain";
    const Json speculated = Json::parse(make_anthropic_messages_response(identity, outcome));
    const Json& extension = speculated["usage"]["ninferno"]["speculation"];
    failures += check(extension["config"] == "dflash:7" && extension["selection"] == "fixed" &&
                          extension["reason"] == "--spec-select fixed" &&
                          extension["fallback"] == "watchdog_plain" &&
                          speculated["usage"]["input_tokens"] == 0,
                      "usage.ninferno.speculation carries the speculation fallback");

    outcome               = {};
    outcome.finish_reason = ninferno::FinishReason::ContextCapacity;
    const Json empty      = Json::parse(make_anthropic_messages_response(identity, outcome));
    failures +=
        check(empty["content"].empty() && empty["stop_reason"] == "model_context_window_exceeded",
              "empty output was fabricated or context capacity was misclassified");

    ApiError overloaded;
    overloaded.status         = 429;
    overloaded.code           = "server_overloaded";
    overloaded.message        = "full";
    const ApiError normalized = normalize_anthropic_error(overloaded);
    const Json error          = Json::parse(make_anthropic_error_body(overloaded, "req_error"));
    failures += check(normalized.status == 529 && normalized.type == "overloaded_error" &&
                          error["request_id"] == "req_error" &&
                          error["error"]["type"] == "overloaded_error",
                      "Anthropic overload or request-id error mapping is wrong");
    return failures;
}

int test_tool_call_presentation() {
    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity("req_tool", "claude-local");
    GenerationOutcome outcome;
    outcome.text                = "I need one more check.";
    outcome.finish_reason       = ninferno::FinishReason::StopToken;
    const std::string arguments = R"({"zeta":"last","alpha":{"yankee":2,"bravo":true}})";
    outcome.tool_calls.push_back(ninferno::GeneratedToolCall{
        .name           = "Edit",
        .arguments_json = arguments,
    });

    const Json aggregate = Json::parse(make_anthropic_messages_response(identity, outcome));
    int failures =
        check(aggregate.at("stop_reason") == "tool_use" && aggregate.at("content").size() == 2 &&
                  aggregate.at("content").at(0).at("type") == "text" &&
                  aggregate.at("content").at(1).at("type") == "tool_use" &&
                  aggregate.at("content").at(1).at("name") == "Edit" &&
                  aggregate.at("content").at(1).at("input").dump() == arguments &&
                  !aggregate.at("content").at(1).at("input").contains("replace_all"),
              "aggregate Anthropic response changed ordered tool input");

    AnthropicMessagesStream stream(identity, 10);
    std::vector<std::string> events{stream.start()};
    std::vector<std::string> content = stream.content_delta(outcome.text);
    events.insert(events.end(), std::make_move_iterator(content.begin()),
                  std::make_move_iterator(content.end()));
    std::vector<std::string> terminal = stream.finish(outcome);
    events.insert(events.end(), std::make_move_iterator(terminal.begin()),
                  std::make_move_iterator(terminal.end()));

    bool saw_edit_start = false;
    bool saw_arguments  = false;
    bool saw_tool_stop  = false;
    for (const std::string& wire : events) {
        const Json event = parse_event(wire);
        if (event.at("type") == "content_block_start" &&
            event.at("content_block").at("type") == "tool_use") {
            saw_edit_start = event.at("content_block").at("name") == "Edit";
        } else if (event.at("type") == "content_block_delta" &&
                   event.at("delta").at("type") == "input_json_delta") {
            saw_arguments = event.at("delta").at("partial_json").get<std::string>() == arguments;
        } else if (event.at("type") == "message_delta") {
            saw_tool_stop = event.at("delta").at("stop_reason") == "tool_use";
        }
    }
    failures += check(saw_edit_start && saw_arguments && saw_tool_stop,
                      "Anthropic stream did not terminate the recovered Edit as tool_use");
    return failures;
}

// Hidden Thinking (display "omitted", sent by Claude Code in headless mode) returns empty Thinking
// whose signature carries the reasoning; the echoed block restores it exactly, so the next prompt
// equals the visible-Thinking one and its prefix is reused.
int test_hidden_thinking_round_trip() {
    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity("req_hidden", "claude-local");
    GenerationOutcome outcome = sample_outcome();
    outcome.reasoning = "Line one.\nUnicode: \xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82, "
                        "\xe2\x9c\x93 and \"quotes\" {json} \x01.";
    const Json hidden = Json::parse(
        make_anthropic_messages_response(identity, outcome, AnthropicThinkingDisplay::Omitted));
    const Json block = hidden["content"][0];
    int failures     = check(block["type"] == "thinking" && block["thinking"] == "" &&
                                 decode_hidden_thinking_signature(
                                 block["signature"].get<std::string>()) == outcome.reasoning,
                             "hidden Thinking did not carry its reasoning in the signature");

    Json body        = base_request();
    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "before"}},
         Json{{"role", "assistant"},
              {"content", Json::array({block, Json{{"type", "text"}, {"text", "answer"}}})}},
         Json{{"role", "user"}, {"content", "after"}}});
    failures += check(parse(body).generation.messages[1].reasoning_content == outcome.reasoning,
                      "echoed hidden Thinking was not restored");
    Json visible                                      = body;
    visible["messages"][1]["content"][0]["thinking"]  = outcome.reasoning;
    visible["messages"][1]["content"][0]["signature"] = identity.message_id;
    failures += check(parse(visible).generation.messages[1].reasoning_content ==
                          parse(body).generation.messages[1].reasoning_content,
                      "hidden and visible Thinking lowered to different prompts");

    Json foreign                                      = body;
    foreign["messages"][1]["content"][0]["signature"] = "EqQBCkYIBxgCKkCopaque";
    failures += check(parse(foreign).generation.messages[1].reasoning_content.empty(),
                      "a foreign signature was interpreted as reasoning");
    Json corrupt                                      = body;
    corrupt["messages"][1]["content"][0]["signature"] = "ninferno-hidden-thinking-v1:a*b";
    failures += check(parse(corrupt).generation.messages[1].reasoning_content.empty(),
                      "a corrupt hidden signature was decoded");
    for (const std::string sample : {"", "a", "ab", "abc", "abcd"}) {
        failures += check(
            decode_hidden_thinking_signature(encode_hidden_thinking_signature(sample)) == sample,
            "hidden Thinking signature did not round-trip");
    }

    AnthropicMessagesStream stream(identity, 100, AnthropicThinkingDisplay::Omitted);
    std::vector<std::string> events{stream.start()};
    for (auto values : {stream.reasoning_delta("tho"), stream.reasoning_delta("ught"),
                        stream.content_delta("answer"), stream.finish(sample_outcome())}) {
        events.insert(events.end(), values.begin(), values.end());
    }
    bool saw_thinking_text = false;
    std::optional<std::string> streamed;
    for (const std::string& wire : events) {
        const Json parsed = parse_event(wire);
        if (parsed.at("type") != "content_block_delta") { continue; }
        if (parsed["delta"]["type"] == "thinking_delta") { saw_thinking_text = true; }
        if (parsed["delta"]["type"] == "signature_delta") {
            streamed =
                decode_hidden_thinking_signature(parsed["delta"]["signature"].get<std::string>());
        }
    }
    failures += check(!saw_thinking_text && streamed == std::string("thought"),
                      "hidden Thinking stream leaked text or lost its signature");
    return failures;
}

int test_stream() {
    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity("req_stream", "claude-local");
    AnthropicMessagesStream stream(identity, 100);
    const ninferno::GenerationStart warm_start{
        .prompt               = ninferno::PromptSummary{.prompt_tokens = 100},
        .reused_prompt_tokens = 60,
    };
    std::vector<std::string> events{stream.start(warm_start)};
    auto append_events = [&](std::vector<std::string> values) {
        events.insert(events.end(), std::make_move_iterator(values.begin()),
                      std::make_move_iterator(values.end()));
    };
    append_events(stream.reasoning_delta("thought"));
    append_events(stream.content_delta("answer"));
    append_events(stream.finish(sample_outcome()));

    std::vector<std::string> types;
    bool saw_signature        = false;
    bool start_usage_is_exact = false;
    Json terminal_usage;
    for (const std::string& value : events) {
        const Json parsed = parse_event(value);
        types.push_back(parsed.at("type").get<std::string>());
        if (parsed.at("type") == "message_start") {
            start_usage_is_exact = parsed["message"]["usage"]["input_tokens"] == 40 &&
                                   parsed["message"]["usage"]["cache_read_input_tokens"] == 60;
        }
        if (parsed.at("type") == "content_block_delta" &&
            parsed["delta"]["type"] == "signature_delta") {
            const std::string signature = parsed["delta"]["signature"].get<std::string>();
            saw_signature               = signature == identity.message_id;
        }
        if (parsed.at("type") == "message_delta") { terminal_usage = parsed.at("usage"); }
    }
    int failures = check(types.front() == "message_start" && types.back() == "message_stop" &&
                             saw_signature && start_usage_is_exact,
                         "Anthropic stream lifecycle/signature/start usage is incomplete");
    const auto signature_position =
        std::find_if(events.begin(), events.end(), [](const auto& value) {
            return value.find("signature_delta") != std::string::npos;
        });
    const auto text_position = std::find_if(events.begin(), events.end(), [](const auto& value) {
        return value.find("text_delta") != std::string::npos;
    });
    failures += check(signature_position < text_position,
                      "Thinking signature was emitted after the text block began");
    const Json aggregate =
        Json::parse(make_anthropic_messages_response(identity, sample_outcome()));
    failures +=
        check(terminal_usage == aggregate.at("usage") && terminal_usage["input_tokens"] == 40 &&
                  terminal_usage["cache_read_input_tokens"] == 60,
              "terminal stream usage did not match aggregate cache usage");

    AnthropicMessagesStream cold_stream(identity, 25);
    (void)cold_stream.start(ninferno::GenerationStart{
        .prompt               = ninferno::PromptSummary{.prompt_tokens = 25},
        .reused_prompt_tokens = 0,
    });
    GenerationOutcome cold;
    cold.prompt_tokens                         = 25;
    cold.completion_tokens                     = 3;
    const std::vector<std::string> cold_events = cold_stream.finish(cold);
    const Json cold_delta = parse_event(cold_events.at(cold_events.size() - 2U));
    failures += check(cold_delta["usage"]["input_tokens"] == 25 &&
                          cold_delta["usage"]["cache_read_input_tokens"] == 0 &&
                          cold_delta["usage"]["output_tokens"] == 3,
                      "cold terminal stream usage was not cumulative and exact");

    AnthropicMessagesStream pre_admission_error(identity, 25);
    const Json provisional = parse_event(pre_admission_error.start());
    failures += check(provisional["message"]["usage"]["input_tokens"] == 25 &&
                          provisional["message"]["usage"]["cache_read_input_tokens"].is_null(),
                      "pre-admission stream error prefix fabricated cache usage");
    return failures;
}

// The converter turns one engine outcome into content blocks. Thinking
// arrives in two deltas, text in one; the outcome carries three tool calls, one of them with
// arguments that a streaming parser would have seen as '{"a' then '":1}'. Every block must open
// with the next index, carry its deltas and close before the next block opens; a tool block opens
// with input {} and its input_json_delta fragments concatenate to the call's arguments; each call
// appears exactly once, in order, with a unique id; the aggregate lists the same blocks.
int test_converter_block_sequence() {
    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity("req_blocks", "claude-local");
    GenerationOutcome outcome;
    outcome.reasoning     = "step one; step two";
    outcome.text          = "calling tools";
    outcome.finish_reason = ninferno::FinishReason::StopToken;
    const std::vector<std::pair<std::string, std::string>> calls = {
        {"lookup", R"({"a":1})"},
        {"search", R"({"q":"北京 \"quoted\" }","n":[1,2]})"},
        {"lookup", R"({})"},
    };
    for (const auto& [name, arguments] : calls) {
        outcome.tool_calls.push_back(
            ninferno::GeneratedToolCall{.name = name, .arguments_json = arguments});
    }

    AnthropicMessagesStream stream(identity, 10);
    std::vector<std::string> events{stream.start()};
    const auto append = [&](std::vector<std::string> values) {
        events.insert(events.end(), std::make_move_iterator(values.begin()),
                      std::make_move_iterator(values.end()));
    };
    append(stream.reasoning_delta("step one; "));
    append(stream.reasoning_delta("step two"));
    append(stream.content_delta(outcome.text));
    append(stream.finish(outcome));

    int failures = 0;
    int open     = -1; // index of the open block, -1 when none
    int next     = 0;  // index the next content_block_start must carry
    std::vector<std::string> kinds;
    std::vector<std::string> tool_names;
    std::vector<std::string> tool_inputs;
    std::vector<std::string> tool_ids;
    std::string thinking;
    std::string text;
    for (const std::string& wire : events) {
        const Json event = parse_event(wire);
        const std::string type = event.at("type").get<std::string>();
        if (type == "content_block_start") {
            const int index = event.at("index").get<int>();
            failures += check(open == -1, "a block opened before the previous one closed");
            failures += check(index == next, "block indices are not consecutive from 0");
            open = index;
            ++next;
            const Json& block = event.at("content_block");
            kinds.push_back(block.at("type").get<std::string>());
            if (kinds.back() == "tool_use") {
                failures += check(block.at("input") == Json::object(),
                                  "a tool_use block did not open with input {}");
                tool_names.push_back(block.at("name").get<std::string>());
                tool_ids.push_back(block.at("id").get<std::string>());
                tool_inputs.emplace_back();
            }
        } else if (type == "content_block_delta") {
            failures += check(event.at("index").get<int>() == open,
                              "a delta addressed a block that is not open");
            const Json& delta = event.at("delta");
            if (delta.at("type") == "input_json_delta") {
                if (!tool_inputs.empty()) {
                    tool_inputs.back() += delta.at("partial_json").get<std::string>();
                }
            } else if (delta.at("type") == "thinking_delta") {
                thinking += delta.at("thinking").get<std::string>();
            } else if (delta.at("type") == "text_delta") {
                text += delta.at("text").get<std::string>();
            }
        } else if (type == "content_block_stop") {
            failures += check(event.at("index").get<int>() == open,
                              "a stop addressed a block that is not open");
            open = -1;
        }
    }
    failures += check(open == -1, "a block was left open at message_stop");
    failures += check(kinds == std::vector<std::string>{"thinking", "text", "tool_use",
                                                        "tool_use", "tool_use"},
                      "block kinds are not thinking, text, three tool_use in order");
    failures += check(thinking == outcome.reasoning && text == outcome.text,
                      "thinking or text deltas do not reassemble the outcome");
    bool calls_ok = tool_names.size() == calls.size() && tool_inputs.size() == calls.size();
    for (std::size_t i = 0; calls_ok && i < calls.size(); ++i) {
        calls_ok = tool_names[i] == calls[i].first &&
                   Json::parse(tool_inputs[i]) == Json::parse(calls[i].second);
    }
    failures += check(calls_ok, "tool calls were lost, duplicated, reordered or split wrongly");
    std::vector<std::string> unique_ids = tool_ids;
    std::sort(unique_ids.begin(), unique_ids.end());
    failures += check(std::adjacent_find(unique_ids.begin(), unique_ids.end()) ==
                          unique_ids.end(),
                      "tool_use ids are not unique");

    const Json aggregate = Json::parse(make_anthropic_messages_response(identity, outcome));
    bool aggregate_ok = aggregate.at("content").size() == 5 &&
                        aggregate.at("stop_reason") == "tool_use";
    for (std::size_t i = 0; aggregate_ok && i < calls.size(); ++i) {
        const Json& block = aggregate.at("content").at(2 + i);
        aggregate_ok      = block.at("type") == "tool_use" && block.at("name") == calls[i].first &&
                       block.at("input") == Json::parse(calls[i].second);
    }
    failures += check(aggregate_ok, "aggregate blocks differ from the streamed blocks");
    return failures;
}

// An image inside a tool_result reaches the prompt of the tool turn, and
// redacted_thinking in history is accepted and contributes nothing to the prompt.
int test_tool_result_image_and_redacted_thinking() {
    const Json png = Json{{"type", "image"},
                          {"source", Json{{"type", "base64"},
                                          {"media_type", "image/png"},
                                          {"data", "iVBORw0KGgo="}}}};
    Json body        = base_request();
    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "take a screenshot"}},
         Json{{"role", "assistant"},
              {"content",
               Json::array({Json{{"type", "redacted_thinking"}, {"data", "opaque-bytes"}},
                            Json{{"type", "text"}, {"text", "taking it"}},
                            tool_use("toolu_shot", "screenshot")})}},
         Json{{"role", "user"},
              {"content",
               Json::array({Json{{"type", "tool_result"},
                                 {"tool_use_id", "toolu_shot"},
                                 {"content", Json::array({Json{{"type", "text"},
                                                               {"text", "here it is"}},
                                                          png})}},
                            Json{{"type", "text"}, {"text", "what colour is the square?"}}})}}});
    body["thinking"]                = Json{{"type", "disabled"}};
    const GenerationRequest request = parse(body).generation;
    const auto tool =
        std::find_if(request.messages.begin(), request.messages.end(),
                     [](const ChatTurn& t) { return t.role == ninferno::ChatRole::Tool; });
    int failures    = check(tool != request.messages.end() && tool->content.size() == 2 &&
                             tool->content[0].text == "here it is" &&
                             tool->content[1].kind == ContentKind::Image &&
                             request.media_item_count() == 1,
                         "an image inside tool_result did not reach the tool turn");
    const ninferno::PromptInput input = prompt(request);
    bool media_in_tool_turn         = false;
    for (const auto& message : input.messages) {
        if (message.role != ninferno::ChatRole::Tool) { continue; }
        for (const auto& part : message.parts) {
            media_in_tool_turn |= part.kind == ninferno::MessagePartKind::Media;
        }
    }
    failures += check(media_in_tool_turn, "the tool turn's image did not reach the PromptInput");
    const auto assistant =
        std::find_if(request.messages.begin(), request.messages.end(),
                     [](const ChatTurn& t) { return t.role == ninferno::ChatRole::Assistant; });
    failures += check(assistant != request.messages.end() &&
                          assistant->reasoning_content.empty() && assistant->content.size() == 1 &&
                          assistant->content[0].text == "taking it" &&
                          assistant->tool_calls.size() == 1,
                      "redacted_thinking changed the assistant turn");
    failures += check(api_code([&] { (void)parse_anthropic_count_tokens_request(body); }).empty(),
                      "count_tokens refused redacted_thinking or a tool_result image");
    return failures;
}

} // namespace

// An unknown Anthropic field is accepted and noted; standard neutral ones are not.
int test_unknown_fields_note() {
    Json known                = base_request();
    known["metadata"]         = Json{{"user_id", "u"}};
    known["service_tier"]     = "auto";
    int failures              = check(parse(known).generation.notes.empty(),
                                      "standard neutral fields drew an unknown-field note");
    Json misspelt             = base_request();
    misspelt["stop_sequence"] = Json::array({"x"});
    const auto notes          = parse(misspelt).generation.notes;
    failures += check(notes.size() == 1 && notes[0] == "unknown fields ignored: stop_sequence",
                      "an unknown Anthropic field was not noted");
    return failures;
}

int test_message_count_limit() {
    int failures = 0;
    Json body    = base_request();
    body["messages"].push_back(Json{{"role", "assistant"}, {"content", "hi"}});
    body["messages"].push_back(Json{{"role", "user"}, {"content", "again"}});
    RequestLimits narrow = limits();
    narrow.max_messages  = 2;
    failures += check(api_code([&] { (void)parse_anthropic_messages_request(body, narrow); }) ==
                          "too_many_messages",
                      "Messages over the message limit is a 400");
    failures += check(api_code([&] { (void)parse_anthropic_count_tokens_request(body, narrow); }) ==
                          "too_many_messages",
                      "count_tokens over the message limit is a 400");
    narrow.max_messages = 3;
    failures +=
        check(api_code([&] { (void)parse_anthropic_messages_request(body, narrow); }).empty(),
              "Messages at the message limit is accepted");
    return failures;
}

int main() {
    int failures = 0;
    failures += test_message_count_limit();
    failures += test_envelope_and_field_policy();
    failures += test_message_normalization();
    failures += test_attribution_system_block();
    failures += test_tool_history();
    failures += test_system_messages_anywhere();
    failures += test_tools();
    failures += test_unknown_fields_note();
    failures += test_prompt_object_order();
    failures += test_thinking_and_count_tokens();
    failures += test_thinking_history_transport_metadata();
    failures += test_content_and_cache_hints();
    failures += test_aggregate_and_errors();
    failures += test_tool_call_presentation();
    failures += test_stream();
    failures += test_hidden_thinking_round_trip();
    failures += test_converter_block_sequence();
    failures += test_tool_result_image_and_redacted_thinking();
    if (failures != 0) {
        std::cerr << failures << " Anthropic adapter checks failed\n";
        return 1;
    }
    std::cout << "Anthropic adapter checks passed\n";
    return 0;
}
