#include "serve/generation_service.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = ninferno::serve::RequestJson;
using namespace ninferno::serve;

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

template <typename Function>
ApiError api_error(Function&& function) {
    try {
        function();
    } catch (const ApiException& exception) { return exception.error(); }
    return ApiError{.status = 0, .message = "no exception"};
}

template <typename Function>
bool throws_logic(Function&& function) {
    try {
        function();
    } catch (const std::logic_error&) { return true; }
    return false;
}

RequestLimits limits() { return RequestLimits{.default_max_tokens = 512}; }

Json base_request() {
    return Json{{"model", "qwen"},
                {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})}};
}

OpenAIChatRequest parse(Json body) { return parse_chat_completion_request(body, limits()); }

ResolvedPromptSemantics semantics(const GenerationRequest& request) {
    ServeOptions server;
    return resolve_prompt_semantics(request, server);
}

ninferno::PromptInput prompt(const GenerationRequest& request) {
    return to_prompt_input(request, semantics(request), {});
}

ninferno::RequestOptions options(const GenerationRequest& request) {
    ServeOptions server;
    return to_request_options(request, server, semantics(request), true);
}

Json parse_sse(const std::string& event) {
    constexpr std::string_view prefix = "data: ";
    if (!event.starts_with(prefix) || !event.ends_with("\n\n")) {
        throw std::runtime_error("invalid SSE framing");
    }
    return Json::parse(event.substr(prefix.size(), event.size() - prefix.size() - 2));
}

int test_request_envelope_and_sampling() {
    int failures                  = 0;
    Json body                     = base_request();
    body["stream"]                = true;
    body["stream_options"]        = Json{{"include_usage", true}, {"include_obfuscation", false}};
    body["max_completion_tokens"] = 48;
    body["max_tokens"]            = 9;
    body["temperature"]           = 0.7;
    body["top_p"]                 = 0.8;
    body["presence_penalty"]      = 0.3;
    body["frequency_penalty"]     = -0.2;
    body["seed"]                  = -1;
    body["top_k"]                 = 17;
    body["min_p"]                 = 0.05;
    body["timings_per_token"]     = true;
    body["return_progress"]       = true;

    const OpenAIChatRequest request = parse(body);
    failures += check(request.model == "qwen", "model remains in OpenAI envelope");
    failures += check(request.stream && request.include_usage, "stream metadata parsed");
    failures += check(request.timings_per_token && request.return_progress,
                      "llama.cpp response observations remain in the protocol envelope");
    failures += check(request.output_tokens_explicit && request.generation.max_tokens == 48,
                      "max_completion_tokens wins and explicitness stays in envelope");
    failures += check(!request.generation.sampling.seed,
                      "a negative seed is the server's own (llama.cpp's -1 = random)");
    failures +=
        check(request.generation.sampling.top_k == 17 && request.generation.sampling.min_p == 0.05,
              "compatible sampler extensions parsed");
    const ninferno::RequestOptions translated = options(request.generation);
    failures +=
        check(translated.execution.sampling.top_k == 17, "top_k reaches Engine request options");
    failures +=
        check(translated.execution.sampling.min_p && *translated.execution.sampling.min_p == 0.05F,
              "min_p reaches Engine request options");
    failures += check(translated.execution.sampling.seed.has_value(),
                      "a negative seed reaches the Engine as one the server drew (and logs)");

    const OpenAIChatRequest defaults = parse(base_request());
    failures +=
        check(!defaults.stream && !defaults.include_usage && !defaults.output_tokens_explicit &&
                  !defaults.timings_per_token && !defaults.return_progress &&
                  defaults.generation.max_tokens == limits().default_max_tokens,
              "protocol defaults remain outside GenerationRequest");

    Json malformed              = base_request();
    malformed["stream_options"] = true;
    failures += check(api_error([&] { (void)parse(malformed); }).param == "stream_options",
                      "malformed stream_options rejected");
    malformed                      = base_request();
    malformed["timings_per_token"] = "yes";
    failures += check(api_error([&] { (void)parse(malformed); }).param == "timings_per_token",
                      "non-boolean timings_per_token rejected");
    malformed                    = base_request();
    malformed["return_progress"] = 1;
    failures += check(api_error([&] { (void)parse(malformed); }).param == "return_progress",
                      "non-boolean return_progress rejected");
    return failures;
}

int test_standard_field_policy() {
    int failures  = 0;
    auto rejected = [&](const char* key, Json value, const char* code) {
        Json body            = base_request();
        body[key]            = std::move(value);
        const ApiError error = api_error([&] { (void)parse(body); });
        failures += check(error.param == key && error.code == code,
                          std::string(key) + " non-neutral value rejected");
    };

    rejected("n", 2, "n_not_supported");
    rejected("logit_bias", Json{{"12", 1}}, "logit_bias_not_supported");
    rejected("logprobs", true, "logprobs_not_supported");
    rejected("top_logprobs", 2, "logprobs_not_supported");
    rejected("response_format", Json{{"type", "xml"}}, "response_format_not_supported");
    rejected("modalities", Json::array({"text", "audio"}), "modality_not_supported");
    rejected("web_search_options", Json::object(), "web_search_not_supported");
    rejected("moderation", Json::object(), "moderation_not_supported");
    rejected("verbosity", "high", "verbosity_not_supported");
    rejected("store", true, "store_not_supported");
    rejected("functions", Json::array({Json{{"name", "legacy"}}}), "legacy_tools_not_supported");

    Json neutral                      = base_request();
    neutral["n"]                      = 1;
    neutral["logit_bias"]             = Json{{"12", 0}, {"13", 0.0}};
    neutral["logprobs"]               = false;
    neutral["top_logprobs"]           = 0;
    neutral["response_format"]        = Json{{"type", "text"}};
    neutral["modalities"]             = Json::array({"text"});
    neutral["audio"]                  = Json{{"voice", "alloy"}};
    neutral["prediction"]             = Json{{"type", "content"}, {"content", "expected"}};
    neutral["verbosity"]              = "medium";
    neutral["store"]                  = false;
    neutral["functions"]              = Json::array();
    neutral["function_call"]          = "auto";
    neutral["metadata"]               = Json{{"trace", "client"}};
    neutral["user"]                   = "user-1";
    neutral["safety_identifier"]      = "safe-1";
    neutral["prompt_cache_key"]       = "cache-1";
    neutral["prompt_cache_options"]   = Json{{"retention", "24h"}};
    neutral["prompt_cache_retention"] = "24h";
    neutral["service_tier"]           = "priority";
    neutral["future_unknown_field"]   = Json{{"value", 1}};
    failures += check(parse(neutral).generation.messages.size() == 1,
                      "neutral controls and advisory hints are accepted");
    Json empty_format               = base_request();
    empty_format["response_format"] = Json::object();
    failures += check(!parse(empty_format).generation.output_constraint.has_value(),
                      "an empty response_format requests plain text");

    Json zero_limit                     = base_request();
    zero_limit["max_completion_tokens"] = 0;
    const OpenAIChatRequest zero        = parse(zero_limit);
    failures += check(zero.output_tokens_explicit && zero.generation.max_tokens == 0,
                      "an explicit zero output limit reaches Engine's no-generation path");
    return failures;
}

int test_auto_system_shared_prefix() {
    using Evidence = ninferno::SharedCandidateEvidence;
    int failures   = 0;

    Json head             = base_request();
    head["messages"]      = Json::array({Json{{"role", "system"}, {"content", "shared agent head"}},
                                         Json{{"role", "user"}, {"content", "hello"}}});
    const auto system_req = parse(head).generation;
    failures += check(
        system_req.messages.size() == 2 && system_req.messages[0].cache_boundary_after &&
            system_req.messages[0].cache_boundary_after->evidence == Evidence::DefaultAutomatic &&
            !system_req.messages[1].cache_boundary_after &&
            system_req.messages[1].content.back().cache_boundary_after,
        "the leading system turn and the last content are both candidates");
    failures += check(prompt(system_req).context_cache.markers.size() == 2,
                      "the system-head candidate reaches the Engine as its own marker");

    head["messages"]   = Json::array({Json{{"role", "system"}, {"content", "head"}},
                                      Json{{"role", "developer"}, {"content", "config"}},
                                      Json{{"role", "user"}, {"content", "hi"}}});
    const auto dev_req = parse(head).generation;
    failures +=
        check(dev_req.messages.size() == 3 && !dev_req.messages[0].cache_boundary_after &&
                  dev_req.messages[1].cache_boundary_after &&
                  dev_req.messages[1].cache_boundary_after->evidence == Evidence::DefaultAutomatic,
              "a system+developer run is marked at its end");

    head["messages"]    = Json::array({Json{{"role", "user"}, {"content", "no head"}},
                                       Json{{"role", "system"}, {"content", "late"}}});
    const auto late_req = parse(head).generation;
    failures += check(!late_req.messages[0].cache_boundary_after &&
                          !late_req.messages[1].cache_boundary_after,
                      "a system turn after the first user turn is not a shared head");

    // Four explicit breakpoints: with the last-content and system-head candidates the request
    // still carries at most the frontend's four markers.
    Json four = Json::array();
    for (int index = 0; index < 4; ++index) {
        four.push_back(Json{{"type", "text"},
                            {"text", "section " + std::to_string(index)},
                            {"prompt_cache_breakpoint", Json{{"mode", "explicit"}}}});
    }
    Json filled        = base_request();
    filled["messages"] = Json::array(
        {Json{{"role", "system"}, {"content", "head"}}, Json{{"role", "user"}, {"content", four}}});
    const auto capped = prompt(parse(filled).generation);
    failures +=
        check(capped.context_cache.markers.size() <= ninferno::kMaximumExplicitPromptCacheMarkers &&
                  capped.context_cache.markers.front().evidence == Evidence::DefaultAutomatic,
              "four breakpoints and the system head stay within the marker cap");

    Json explicit_mode        = base_request();
    explicit_mode["messages"] = Json::array(
        {Json{{"role", "system"}, {"content", "h"}}, Json{{"role", "user"}, {"content", "hi"}}});
    explicit_mode["prompt_cache_options"] = Json{{"mode", "explicit"}};
    const auto explicit_req               = parse(explicit_mode).generation;
    failures += check(!explicit_req.messages[0].cache_boundary_after,
                      "explicit mode adds no system-head candidate");

    Json switched_off        = base_request();
    switched_off["messages"] = Json::array(
        {Json{{"role", "system"}, {"content", "h"}}, Json{{"role", "user"}, {"content", "hi"}}});
    const auto off = parse_chat_completion_request(switched_off, limits(), false).generation;
    failures += check(!off.messages[0].cache_boundary_after &&
                          off.messages[1].content.back().cache_boundary_after,
                      "--no-auto-system-shared-prefix keeps only the last-content candidate");
    return failures;
}

int test_constrained_decoding_extensions() {
    using Kind   = ninferno::OutputConstraintKind;
    int failures = 0;
    struct Case {
        const char* field;
        Json value;
        Kind kind;
        const char* param;
    };
    const Json schema = Json::parse(R"({"type":"object","properties":{"b":{"type":"string"},
        "a":{"type":"integer"}},"required":["b","a"],"additionalProperties":false})");
    const std::vector<Case> accepted = {
        {"grammar", "root ::= \"yes\" | \"no\"", Kind::Grammar, "grammar"},
        {"structured_outputs", Json{{"json", schema}}, Kind::JsonSchema, "structured_outputs.json"},
        {"structured_outputs", Json{{"json", schema.dump()}}, Kind::JsonSchema,
         "structured_outputs.json"},
        {"structured_outputs", Json{{"regex", "[a-z]+"}}, Kind::Regex, "structured_outputs.regex"},
        {"structured_outputs", Json{{"choice", Json::array({"yes", "no"})}}, Kind::Choice,
         "structured_outputs.choice"},
        {"structured_outputs", Json{{"grammar", "root ::= \"x\""}}, Kind::Grammar,
         "structured_outputs.grammar"},
        {"structured_outputs", Json{{"json_object", true}}, Kind::JsonObject,
         "structured_outputs.json_object"},
        {"guided_json", schema, Kind::JsonSchema, "guided_json"},
        {"guided_regex", "[a-z]+", Kind::Regex, "guided_regex"},
        {"guided_choice", Json::array({"yes", "no"}), Kind::Choice, "guided_choice"},
        {"guided_grammar", "root ::= \"yes\" | \"no\"", Kind::Grammar, "guided_grammar"},
        {"response_format", Json{{"type", "json_object"}}, Kind::JsonObject, "response_format"},
        {"response_format",
         Json{{"type", "json_schema"},
              {"json_schema", Json{{"name", "reply"}, {"schema", schema}, {"strict", true}}}},
         Kind::JsonSchema, "response_format"},
    };
    for (const Case& test_case : accepted) {
        Json body              = base_request();
        body[test_case.field]  = test_case.value;
        const auto constraint  = parse(body).generation.output_constraint;
        const std::string what = std::string(test_case.field) + " " + test_case.value.dump();
        failures += check(constraint.has_value() && constraint->kind == test_case.kind &&
                              constraint->param == test_case.param,
                          what + " is parsed into an enforced constraint");
        if (constraint && test_case.kind == Kind::JsonSchema) {
            failures += check(constraint->definition.find("\"b\"") <
                                  constraint->definition.find("\"a\""),
                              what + " keeps the declared property order");
        }
        if (constraint && test_case.kind == Kind::Choice) {
            failures += check(constraint->choices == std::vector<std::string>{"yes", "no"},
                              what + " keeps its choices");
        }
    }

    Json compact                  = base_request();
    compact["structured_outputs"] = Json{{"json", schema}, {"disable_any_whitespace", true}};
    failures += check(parse(compact).generation.output_constraint->compact_json,
                      "disable_any_whitespace selects compact JSON");

    const auto rejected = [&](Json body, const char* param, const char* code,
                              const std::string& what) {
        const ApiError error = api_error([&] { (void)parse(body); });
        failures += check(error.status == 400 && error.param == param && error.code == code,
                          what + " is rejected with 400 on " + param);
    };
    // JSON carried as text gets the depth limit the request body has.
    const auto nested = [](int depth) {
        std::string text;
        for (int i = 0; i < depth; ++i) { text += "{\"a\":"; }
        text += "1";
        text.append(static_cast<std::size_t>(depth), '}');
        return text;
    };
    Json shallow_text           = base_request();
    shallow_text["guided_json"] = nested(100);
    failures += check(parse(shallow_text).generation.output_constraint.has_value(),
                      "a 100-level guided_json text is not accepted");

    struct DeepCase {
        const char* field;
        Json value;
        const char* param;
    };

    for (const DeepCase& deep_case :
         {DeepCase{"guided_json", nested(200), "guided_json"},
          DeepCase{"structured_outputs", Json{{"structural_tag", nested(200)}},
                   "structured_outputs.structural_tag"}}) {
        Json deep             = base_request();
        deep[deep_case.field] = deep_case.value;
        const ApiError error  = api_error([&] { (void)parse(deep); });
        failures += check(error.status == 400 && error.param == deep_case.param &&
                              error.message.find("nests deeper than 128") != std::string::npos,
                          std::string(deep_case.param) + " text nesting 200 levels is not refused");
    }

    Json two                = base_request();
    two["response_format"]  = Json{{"type", "json_object"}};
    two["guided_regex"]     = "[a-z]+";
    rejected(two, "guided_regex", "conflicting_constraints", "two constraint sources");
    Json both                  = base_request();
    both["structured_outputs"] = Json{{"json", schema}, {"regex", "x"}};
    rejected(both, "structured_outputs", "conflicting_constraints", "two structured_outputs keys");
    Json pattern                  = base_request();
    pattern["structured_outputs"] = Json{{"json", schema}, {"whitespace_pattern", "[ ]*"}};
    rejected(pattern, "structured_outputs", "structured_output_option_not_supported",
             "whitespace_pattern");
    Json guided_pattern                         = base_request();
    guided_pattern["guided_whitespace_pattern"] = "[ ]*";
    rejected(guided_pattern, "guided_whitespace_pattern", "structured_output_option_not_supported",
             "guided_whitespace_pattern");
    Json nameless               = base_request();
    nameless["response_format"] = Json{{"type", "json_schema"}, {"json_schema", {{"schema", schema}}}};
    failures += check(api_error([&] { (void)parse(nameless); }).param == "response_format",
                      "json_schema without a name is rejected");
    Json with_tools               = base_request();
    with_tools["response_format"] = Json{{"type", "json_object"}};
    with_tools["tools"]           = Json::array(
        {Json{{"type", "function"},
              {"function", Json{{"name", "f"}, {"parameters", Json{{"type", "object"}}}}}}});
    // A JSON format beside tools constrains the text answer: calls, or JSON in the format.
    const auto beside = parse(with_tools).generation.output_constraint;
    failures += check(beside && beside->kind == Kind::ToolCalls &&
                          beside->answer_schema == R"({"type":"object"})" &&
                          beside->tool_requirement == ninferno::ToolCallRequirement::Optional,
                      "response_format with enabled tools is not the text-answer alternative");
    Json schema_tools = with_tools;
    schema_tools["response_format"] =
        Json{{"type", "json_schema"},
             {"json_schema",
              {{"name", "a"}, {"schema", Json{{"type", "object"}, {"required", {"x"}}}}}}};
    const auto schema_beside = parse(schema_tools).generation.output_constraint;
    failures += check(schema_beside && schema_beside->kind == Kind::ToolCalls &&
                          Json::parse(schema_beside->answer_schema).contains("required"),
                      "a json_schema beside tools is not the answer schema");
    Json required_tools           = with_tools;
    required_tools["tool_choice"] = "required";
    const auto calls_only         = parse(required_tools).generation.output_constraint;
    failures += check(calls_only && calls_only->kind == Kind::ToolCalls &&
                          calls_only->answer_schema.empty(),
                      "a required call kept a text-answer format");
    Json regex_tools            = base_request();
    regex_tools["guided_regex"] = "[a-z]+";
    regex_tools["tools"]        = with_tools["tools"];
    rejected(regex_tools, "guided_regex", "constraint_with_tools_not_supported",
             "a regex beside enabled tools");
    with_tools["tool_choice"] = "none";
    failures += check(parse(with_tools).generation.output_constraint.has_value(),
                      "response_format with tool_choice none is accepted");

    Json neutral                  = base_request();
    neutral["grammar"]            = "";
    neutral["structured_outputs"] = nullptr;
    neutral["guided_json"]        = nullptr;
    neutral["guided_regex"]       = nullptr;
    neutral["guided_choice"]      = nullptr;
    neutral["guided_grammar"]     = nullptr;
    neutral["response_format"]    = Json{{"type", "text"}};
    const OpenAIChatRequest plain = parse(neutral);
    failures += check(plain.generation.messages.size() == 1 && !plain.generation.output_constraint,
                      "neutral constrained-decoding values request no constraint");
    Json options_only                  = base_request();
    options_only["structured_outputs"] = Json{{"disable_any_whitespace", true}};
    failures += check(!parse(options_only).generation.output_constraint,
                      "structured_outputs options without a constraint request none");
    return failures;
}

Json function_tool(std::string name = "weather", bool strict = false) {
    return Json{{"type", "function"},
                {"function", Json{{"name", std::move(name)},
                                  {"description", "Get weather"},
                                  {"parameters", Json{{"type", "object"}}},
                                  {"strict", strict}}}};
}

int test_tool_name_diagnostics() {
    int failures = 0;

    Json body                            = base_request();
    body["tools"]                        = Json::array({function_tool("weather")});
    body["tools"][0]["function"]["name"] = "mcp weather";
    const std::string limit_spec =
        "[A-Za-z0-9_.:/-]{1," + std::to_string(kMaximumToolNameLength) + "}";
    const ApiError invalid_name = api_error([&] { (void)parse(body); });
    failures +=
        check(invalid_name.status == 400 && invalid_name.param == "tools[0].function.name" &&
                  invalid_name.message.find("'mcp weather'") != std::string::npos &&
                  invalid_name.message.find("11 bytes") != std::string::npos &&
                  invalid_name.message.find(limit_spec) != std::string::npos,
              "invalid tool name rejection names the value, its length, and its location");

    // MCP and namespaced names (one such name in history refused every later turn).
    for (const char* name : {"github.create_issue", "fs:read", "tools/search"}) {
        body          = base_request();
        body["tools"] = Json::array({function_tool(name)});
        failures += check(api_error([&] { (void)parse(body); }).status == 0,
                          std::string("tool name ") + name + " is accepted");
    }

    body["tools"][0]["function"]["name"] = Json::array({"not", "a", "string"});
    const ApiError non_string            = api_error([&] { (void)parse(body); });
    failures += check(non_string.param == "tools[0].function.name" &&
                          non_string.message.find("must be a string") != std::string::npos &&
                          non_string.message.find("array") != std::string::npos,
                      "non-string tool name rejection reports the actual JSON type");

    body                                 = base_request();
    body["tools"]                        = Json::array({function_tool("weather")});
    body["tools"][0]["function"]["name"] = "bad\nname";
    const ApiError escaped               = api_error([&] { (void)parse(body); });
    failures += check(escaped.message.find("bad\\x0aname") != std::string::npos &&
                          escaped.message.find('\n') == std::string::npos,
                      "control characters in a rejected name are escaped, not embedded raw");

    // Regression for VS Code Copilot agent traffic: MCP activation wrappers are
    // synthesized as "activate_fallback_mcp_<server>_<tool>" and exceed the
    // OpenAI 64-byte limit; the Engine must accept them up to the shared cap.
    const std::string mcp_name =
        "activate_fallback_mcp_pgsql-tools_pgsql_get_dashboard_metric_data";
    body["tools"][0]["function"]["name"] = mcp_name;
    const OpenAIChatRequest mcp_ok       = parse(body);
    failures += check(mcp_name.size() == 65 && mcp_ok.generation.tools.size() == 1 &&
                          mcp_ok.generation.tools[0].name == mcp_name,
                      "MCP wrapper names longer than 64 bytes are accepted");

    Json history        = base_request();
    history["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "hi"}},
         Json{{"role", "assistant"},
              {"content", nullptr},
              {"tool_calls", Json::array({Json{{"id", "call_1"},
                                               {"type", "function"},
                                               {"function", Json{{"name", "mcp weather"},
                                                                 {"arguments", "{}"}}}}})}}});
    const ApiError history_error = api_error([&] { (void)parse(history); });
    failures += check(history_error.param == "messages[1].tool_calls[0].function.name" &&
                          history_error.message.find("'mcp weather'") != std::string::npos,
                      "invalid historical tool_call name reports its message location");

    Json allowed     = base_request();
    allowed["tools"] = Json::array({function_tool("weather")});
    allowed["tool_choice"] =
        Json{{"type", "allowed_tools"},
             {"allowed_tools",
              Json{{"mode", "auto"},
                   {"tools", Json::array({Json{{"type", "function"}, {"name", "bad.name"}}})}}}};
    const ApiError allowed_error = api_error([&] { (void)parse(allowed); });
    failures += check(allowed_error.param == "tool_choice.allowed_tools.tools[0].name",
                      "invalid allowed_tools name reports its location");

    return failures;
}

int test_tools() {
    int failures                      = 0;
    Json body                         = base_request();
    body["tools"]                     = Json::array({function_tool()});
    const OpenAIChatRequest automatic = parse(body);
    failures += check(automatic.generation.uses_tools(), "function tools default to auto");
    failures += check(prompt(automatic.generation).options.tool_jsons.size() == 1,
                      "auto tools reach PromptInput");

    // A tool set that comes back in another order renders in the order it was first seen.
    const auto rendered_names = [&](const Json& request) {
        std::vector<std::string> names;
        for (const std::string& tool : prompt(parse(request).generation).options.tool_jsons) {
            names.push_back(Json::parse(tool).at("function").at("name").get<std::string>());
        }
        return names;
    };
    Json reordered     = base_request();
    reordered["tools"] = Json::array({function_tool("order_alpha"), function_tool("order_beta")});
    const std::vector<std::string> first_order = rendered_names(reordered);
    reordered["tools"] = Json::array({function_tool("order_beta"), function_tool("order_alpha")});
    failures += check(first_order == std::vector<std::string>{"order_alpha", "order_beta"} &&
                          rendered_names(reordered) == first_order,
                      "a reordered tool set keeps the order it was first seen in");

    body["tools"][0]["future_item_field"]                 = "ignored";
    body["tools"][0]["function"]["future_function_field"] = "ignored";
    const std::string normalized_definition = prompt(parse(body).generation).options.tool_jsons[0];
    failures += check(normalized_definition.find("future_item_field") == std::string::npos &&
                          normalized_definition.find("future_function_field") == std::string::npos,
                      "unknown tool fields do not silently alter the model prompt");

    // The template receives the declared fields in the client's order and nothing the
    // client did not send, as transformers' apply_chat_template renders the request's tools.
    const auto rendered_tool = [&](const Json& function) {
        Json request     = base_request();
        request["tools"] = Json::array({Json{{"type", "function"}, {"function", function}}});
        return prompt(parse(request).generation).options.tool_jsons.at(0);
    };
    const Json declared = Json::parse(
        R"({"name":"weather","description":"Get weather","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}}}})");
    failures += check(rendered_tool(declared) ==
                          R"({"type":"function","function":)" + declared.dump() + "}",
                      "a declared tool reached the template reordered or with added fields");
    const Json unusual_order = Json::parse(
        R"({"parameters":{"type":"object"},"strict":true,"name":"weather","description":"d"})");
    failures += check(rendered_tool(unusual_order) ==
                          R"({"type":"function","function":)" + unusual_order.dump() + "}",
                      "a declared tool's member order or strict flag was not kept");
    const Json name_only = Json::parse(R"({"name":"ping"})");
    failures += check(rendered_tool(name_only) == R"({"type":"function","function":{"name":"ping"}})",
                      "a tool without description or parameters gained fields in the prompt");

    body["tool_choice"]          = "none";
    body["parallel_tool_calls"]  = false;
    const OpenAIChatRequest none = parse(body);
    const ninferno::PromptInput none_prompt = prompt(none.generation);
    failures += check(!none.generation.uses_tools() && !none_prompt.options.tool_calls_enabled &&
                          none_prompt.options.tool_jsons.size() == none.generation.tools.size() &&
                          !none_prompt.options.tool_jsons.empty(),
                      "tool_choice none did not keep the declarations with calls disabled");

    using Requirement   = ninferno::ToolCallRequirement;
    const auto tool_constraint = [&](const Json& request_body) {
        return parse(request_body).generation.output_constraint;
    };
    body["tool_choice"]  = "required";
    body["parallel_tool_calls"] = true;
    auto required_calls  = tool_constraint(body);
    failures +=
        check(required_calls && required_calls->kind == ninferno::OutputConstraintKind::ToolCalls &&
                  required_calls->tool_requirement == Requirement::Required &&
                  required_calls->param == "tools",
              "required tool choice is enforced by a tool-call constraint");
    body["tool_choice"] = Json{{"type", "function"}, {"function", Json{{"name", "weather"}}}};
    auto named_call     = tool_constraint(body);
    failures += check(named_call && named_call->tool_requirement == Requirement::Named &&
                          named_call->named_tool == "weather",
                      "named tool choice is enforced by a tool-call constraint");
    body["tool_choice"] = Json{{"type", "function"}, {"function", Json{{"name", "missing"}}}};
    failures += check(api_error([&] { (void)parse(body); }).param == "tool_choice",
                      "named tool choice must name a declared tool");

    body          = base_request();
    body["tools"] = Json::array({function_tool(), function_tool("search")});
    body["tool_choice"] =
        Json{{"type", "allowed_tools"},
             {"allowed_tools",
              Json{{"mode", "auto"},
                   {"tools", Json::array({Json{{"type", "function"}, {"name", "search"}}})}}}};
    const GenerationRequest allowed = parse(body).generation;
    failures += check(allowed.tools.size() == 1 && allowed.tools[0].name == "search" &&
                          prompt(allowed).options.tool_jsons.size() == 1,
                      "allowed_tools auto narrows the executable function set");

    body["tool_choice"] =
        Json{{"type", "allowed_tools"},
             {"mode", "auto"},
             {"tools", Json::array({Json{{"type", "function"}, {"name", "weather"}}})}};
    const GenerationRequest direct_allowed = parse(body).generation;
    failures += check(direct_allowed.tools.size() == 1 && direct_allowed.tools[0].name == "weather",
                      "direct allowed_tools compatibility shape is accepted");
    // A custom tool's call: answered as {"type": "custom", "custom": {"name", "input"}} and read
    // back from history in the same shape.
    {
        Json custom_body                       = base_request();
        custom_body["tools"]                   = Json::array({Json{
                              {"type", "custom"},
                              {"custom", Json{{"name", "apply_patch"},
                                              {"format", Json{{"type", "grammar"},
                                                              {"grammar", Json{{"syntax", "lark"},
                                                                               {"definition", "start: x"}}}}}}}}});
        const GenerationRequest custom_request = parse(custom_body).generation;
        failures += check(custom_request.tools.size() == 1 && custom_request.tools[0].custom &&
                              custom_request.tools[0].input_schema_json.find("start: x") !=
                                  std::string::npos,
                          "a chat custom tool lost its custom flag or its grammar");
        OpenAIChatResponseIdentity identity = make_openai_chat_response_identity("m");
        identity.custom_tools.push_back("apply_patch");
        GenerationOutcome outcome;
        outcome.finish_reason = ninferno::FinishReason::StopToken;
        outcome.tool_calls.push_back(ninferno::GeneratedToolCall{
            .name = "apply_patch", .arguments_json = R"({"input":"x"})"});
        const Json answer =
            Json::parse(make_chat_completion_response(identity, outcome)).at("choices")[0];
        const Json& call = answer.at("message").at("tool_calls")[0];
        failures +=
            check(call.at("type") == "custom" && call.at("custom").at("name") == "apply_patch" &&
                      call.at("custom").at("input") == "x",
                  "a chat custom tool's call was not answered in the custom shape");
        Json replay        = custom_body;
        replay["messages"] = Json::array(
            {Json{{"role", "user"}, {"content", "edit"}},
             Json{{"role", "assistant"},
                  {"content", nullptr},
                  {"tool_calls",
                   Json::array({Json{{"id", "call_1"},
                                     {"type", "custom"},
                                     {"custom", Json{{"name", "apply_patch"}, {"input", "x"}}}}})}},
             Json{{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "ok"}}});
        const GenerationRequest replayed = parse(replay).generation;
        failures +=
            check(replayed.messages.size() == 3 && replayed.messages[1].tool_calls.size() == 1 &&
                      replayed.messages[1].tool_calls[0].arguments_json == R"({"input":"x"})",
                  "a chat custom tool call in history was not read back");
    }
    // The Chat Completions shape nests the name: {"type": "function", "function": {"name": ...}}.
    body["tool_choice"] =
        Json{{"type", "allowed_tools"},
             {"allowed_tools",
              Json{{"mode", "auto"},
                   {"tools", Json::array({Json{{"type", "function"},
                                               {"function", Json{{"name", "search"}}}}})}}}};
    const GenerationRequest nested_allowed = parse(body).generation;
    failures += check(nested_allowed.tools.size() == 1 && nested_allowed.tools[0].name == "search",
                      "the nested Chat Completions allowed_tools shape was refused");
    body["tool_choice"]["allowed_tools"]["tools"][0]["function"]["name"] = "missing";
    failures += check(api_error([&] { (void)parse(body); }).param ==
                          "tool_choice.allowed_tools.tools[0].function.name",
                      "an undeclared nested allowed tool does not report its location");
    body["tool_choice"] =
        Json{{"type", "allowed_tools"},
             {"mode", "auto"},
             {"tools", Json::array({Json{{"type", "function"}, {"name", "weather"}}})}};
    body["tool_choice"]["mode"]   = "required";
    const auto required_allowed = parse(body).generation;
    failures += check(required_allowed.output_constraint &&
                          required_allowed.output_constraint->tool_requirement ==
                              ninferno::ToolCallRequirement::Required &&
                          required_allowed.output_constraint->tools.size() == 1,
                      "required allowed_tools is enforced over the allowed set");
    body["tool_choice"]["mode"]             = "auto";
    body["tool_choice"]["tools"][0]["name"] = "missing";
    failures += check(api_error([&] { (void)parse(body); }).param ==
                          "tool_choice.allowed_tools.tools[0].name",
                      "allowed_tools rejects names absent from the declared tool set");

    body          = base_request();
    body["tools"] = Json::array({function_tool("weather", true)});
    {
        const auto strict = parse(body).generation.output_constraint;
        failures += check(strict && strict->tools.size() == 1 && strict->tools[0].strict &&
                              strict->tool_requirement == ninferno::ToolCallRequirement::Optional,
                          "strict tools are enforced by a tool-call constraint");
    }
    body["response_format"] = Json{{"type", "json_object"}};
    {
        const auto answer = parse(body).generation.output_constraint;
        failures += check(answer && answer->tools.size() == 1 &&
                              answer->answer_schema == R"({"type":"object"})",
                          "json_object beside strict tools constrains the answer");
    }
    body.erase("response_format");
    body["tools"]                        = Json::array({Json{
                               {"type", "custom"},
                               {"custom", Json{{"name", "shell"},
                                               {"description", "Run a shell command"},
                                               {"format", Json{{"type", "grammar"}, {"grammar", "start: /.+/"}}}}}}});
    const GenerationRequest custom_tools = parse(body).generation;
    failures +=
        check(custom_tools.tools.size() == 1 && custom_tools.tools[0].name == "shell" &&
                  custom_tools.tools[0].input_schema_json.find("\"input\"") != std::string::npos &&
                  custom_tools.tools[0].input_schema_json.find("start: /.+/") != std::string::npos,
              "custom tools are served as a single-string-input function");

    body                        = base_request();
    body["tools"]               = Json::array({function_tool()});
    body["parallel_tool_calls"] = false;
    {
        const auto single = parse(body).generation.output_constraint;
        failures += check(single && !single->parallel_tool_calls && !single->tools[0].strict,
                          "parallel_tool_calls=false is enforced by a tool-call constraint");
    }
    body.erase("tools");
    failures += check(parse(body).generation.tools.empty(),
                      "parallel_tool_calls=false is neutral without tools");
    body["tool_choice"] = "auto";
    failures +=
        check(parse(body).generation.tools.empty(), "tool_choice auto is neutral without tools");

    Json history = base_request();
    history["messages"] =
        Json::array({Json{{"role", "user"}, {"content", "weather?"}},
                     Json{{"role", "assistant"},
                          {"content", nullptr},
                          {"tool_calls",
                           Json::array({Json{{"id", "call_1"},
                                             {"type", "function"},
                                             {"function", Json{{"name", "weather"},
                                                               {"arguments", "not-json-yet"}}}}})}},
                     Json{{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "sunny"}}});
    failures += check(parse(history).generation.has_tool_history(),
                      "tool-call history follows wire types without inventing JSON validation");

    Json mixed_assistant        = base_request();
    mixed_assistant["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "inspect"}},
         Json{{"role", "assistant"},
              {"content", "I will inspect it"},
              {"tool_calls",
               Json::array({Json{
                   {"id", "call_2"},
                   {"type", "function"},
                   {"function", Json{{"name", "inspect"}, {"arguments", R"({"path":"a"})"}}}}})}}});
    const GenerationRequest mixed_request  = parse(mixed_assistant).generation;
    const ninferno::PromptInput mixed_prompt = prompt(mixed_request);
    failures += check(mixed_request.messages[1].cache_boundary_after &&
                          !mixed_request.messages[1].content[0].cache_boundary_after &&
                          !mixed_prompt.context_cache.markers.empty() &&
                          mixed_prompt.context_cache.markers.back().location ==
                              ninferno::PromptCacheMarkerLocation::MessageBoundary &&
                          mixed_prompt.context_cache.markers.back().after_message_count == 2,
                      "automatic caching stops after a complete assistant text/tool-call turn");

    const Json ordered = Json::parse(
        R"({"model":"qwen","messages":[{"role":"user","content":"probe"}],"tools":[{"type":"function","function":{"name":"probe","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}}}}}]})");
    const ninferno::PromptInput ordered_prompt = prompt(parse(ordered).generation);
    failures += check(
        ordered_prompt.options.tool_jsons.size() == 1 &&
            ordered_prompt.options.tool_jsons.front() ==
                R"({"type":"function","function":{"name":"probe","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}}}}})",
        "OpenAI Chat changed tool-schema member order before PromptInput");
    return failures;
}

// Results a client returns as its concurrent calls finish are rendered in the calls' order; a run
// naming a call of another turn is left as sent.
int test_tool_result_order() {
    int failures    = 0;
    Json body       = base_request();
    const auto call = [](const char* id, const char* item) {
        return Json{{"id", id},
                    {"type", "function"},
                    {"function", Json{{"name", "read_item"},
                                      {"arguments", std::string("{\"item\": \"") + item + "\"}"}}}};
    };
    const auto result = [](const char* id, const char* text) {
        return Json{{"role", "tool"}, {"tool_call_id", id}, {"content", text}};
    };
    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "read a, b and c"}},
         Json{{"role", "assistant"},
              {"content", nullptr},
              {"tool_calls",
               Json::array({call("call_a", "a"), call("call_b", "b"), call("call_c", "c")})}},
         result("call_c", "C"), result("call_a", "A"), result("call_b", "B")});
    const GenerationRequest ordered = parse(body).generation;
    failures +=
        check(ordered.messages.size() == 5 && ordered.messages[2].tool_call_id == "call_a" &&
                  ordered.messages[3].tool_call_id == "call_b" &&
                  ordered.messages[4].tool_call_id == "call_c" &&
                  ordered.messages[2].content[0].text == "A",
              "tool results returned out of order are put in the calls' order");
    body["messages"][2]["tool_call_id"] = "call_other";
    const GenerationRequest kept        = parse(body).generation;
    failures += check(kept.messages[2].tool_call_id == "call_other" &&
                          kept.messages[3].tool_call_id == "call_a",
                      "a result run naming an unknown call is left as sent");
    return failures;
}

int test_messages_and_media() {
    int failures                   = 0;
    Json body                      = base_request();
    body["messages"][0]["content"] = Json::array(
        {Json{{"type", "text"}, {"text", "alpha"}}, Json{{"type", "text"}, {"text", "beta"}}});
    const ninferno::PromptInput translated = prompt(parse(body).generation);
    failures += check(translated.messages[0].parts.size() == 2 &&
                          translated.messages[0].parts[0].text == "alpha" &&
                          translated.messages[0].parts[1].text == "beta",
                      "adjacent text parts preserve exact text without inserted newline");

    body                           = base_request();
    body["messages"][0]["content"] = Json::array(
        {Json{{"type", "image_url"},
              {"image_url", Json{{"url", "https://example.test/a.png"}, {"detail", "auto"}}}},
         Json{{"type", "video_url"}, {"video_url", "https://example.test/a.mp4"}}});
    const GenerationRequest media = parse(body).generation;
    failures += check(media.media_item_count() == 2 &&
                          media.messages[0].content[0].kind == ContentKind::Image &&
                          media.messages[0].content[1].kind == ContentKind::Video,
                      "image and video compatibility inputs normalize to Engine media");

    for (const char* level : {"low", "high"}) {
        body["messages"][0]["content"][0]["image_url"]["detail"] = level;
        failures += check(parse(body).generation.media_item_count() == 2,
                          "OpenAI low/high image detail takes the fixed Vision profile");
    }
    body["messages"][0]["content"][0]["image_url"]["detail"] = "original";
    failures += check(api_error([&] { (void)parse(body); }).code == "image_detail_not_supported",
                      "unknown image detail level rejected");

    auto content_rejected = [&](const char* role, const char* type) {
        Json invalid                   = base_request();
        invalid["messages"][0]["role"] = role;
        invalid["messages"][0]["content"] =
            Json::array({Json{{"type", type}, {type, "https://example.test/x"}}});
        return api_error([&] { (void)parse(invalid); }).code == "modality_not_supported";
    };
    failures +=
        check(content_rejected("assistant", "image_url"), "assistant media history rejected");
    failures += check(content_rejected("system", "image_url"),
                      "system media rejected at protocol boundary");

    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "capture it"}},
         Json{{"role", "assistant"},
              {"content", nullptr},
              {"tool_calls",
               Json::array({Json{{"id", "call_capture"},
                                 {"type", "function"},
                                 {"function", Json{{"name", "capture"}, {"arguments", "{}"}}}}})}},
         Json{{"role", "tool"},
              {"tool_call_id", "call_capture"},
              {"content",
               Json::array({Json{{"type", "text"}, {"text", "captured"}},
                            Json{{"type", "image_url"},
                                 {"image_url", Json{{"url", "https://example.test/capture.png"},
                                                    {"detail", "auto"}}}}})}}});
    const GenerationRequest tool_image = parse(body).generation;
    failures += check(tool_image.messages.back().role == ninferno::ChatRole::Tool &&
                          tool_image.messages.back().tool_call_id == "call_capture" &&
                          tool_image.messages.back().content.size() == 2 &&
                          tool_image.messages.back().content[0].kind == ContentKind::Text &&
                          tool_image.messages.back().content[1].kind == ContentKind::Image,
                      "tool result text and image parts normalize to one tool turn");

    body["messages"].back()["content"] = Json::array(
        {Json{{"type", "video_url"}, {"video_url", "https://example.test/capture.mp4"}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "modality_not_supported",
                      "tool result video remains outside the Chat compatibility extension");

    body = base_request();
    body["messages"][0]["content"] =
        Json::array({Json{{"type", "input_audio"}, {"input_audio", Json::object()}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "modality_not_supported",
                      "input audio rejected");
    body["messages"][0]["content"] =
        Json::array({Json{{"type", "file"}, {"file", Json::object()}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "file_inputs_not_supported",
                      "a file part without file_data rejected");
    body["messages"][0]["content"] = Json::array(
        {Json{{"type", "file"},
              {"file", Json{{"file_data", "data:application/pdf;base64,JVBE"}, {"filename", "a.pdf"}}}}});
    {
        const auto file = parse(body).generation;
        failures += check(file.messages[0].content.size() == 1 &&
                              file.messages[0].content[0].kind == ContentKind::Document &&
                              file.messages[0].content[0].document_title == "a.pdf",
                          "an inline file part did not become a Document part");
    }

    body                        = base_request();
    body["messages"][0]["name"] = "speaker";
    failures += check(parse(body).generation.messages.size() == 1,
                      "a user message name is an ignored compatibility hint");

    body                      = base_request();
    body["safety_identifier"] = std::string(128, 's');
    body["prompt_cache_key"]  = std::string(256, 'k');
    failures += check(parse(body).generation.messages.size() == 1,
                      "OpenAI-length safety_identifier and a session-length cache key");
    body["safety_identifier"] = std::string(129, 's');
    failures += check(api_error([&] { (void)parse(body); }).message ==
                          "safety_identifier must be at most 128 characters",
                      "over-long safety_identifier rejected");

    body = base_request();
    body["messages"].push_back(Json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls",
         Json::array({Json{{"id", "call_1"},
                           {"type", "function"},
                           {"function", Json{{"name", "get_status"}, {"arguments", "{}"}}}}})}});
    body["messages"].push_back(Json{
        {"role", "tool"}, {"name", "get_status"}, {"tool_call_id", "call_1"}, {"content", "ok"}});
    const GenerationRequest named_tool_history = parse(body).generation;
    const ChatTurn& named_tool                 = named_tool_history.messages.back();
    failures += check(named_tool.role == ninferno::ChatRole::Tool &&
                          named_tool.tool_call_id == "call_1" && !named_tool.tool_result_name &&
                          named_tool.content.size() == 1 && named_tool.content[0].text == "ok",
                      "tool message name is an ignored compatibility extension");

    body["messages"].back()["name"] = Json::array();
    failures +=
        check(api_error([&] { (void)parse(body); }).message == "message name must be a string",
              "tool message name remains type checked");

    body                           = base_request();
    body["messages"][0]["name"]    = "";
    body["messages"][0]["content"] = Json::array();
    failures += check(parse(body).generation.messages[0].content.empty(),
                      "empty names and empty content arrays remain neutral");

    body = base_request();
    body["messages"].push_back(
        Json{{"role", "assistant"},
             {"content", Json::array({Json{{"type", "refusal"}, {"refusal", "part"}}})},
             {"refusal", "top-level"}});
    const GenerationRequest refusal_history = parse(body).generation;
    const ChatTurn& refusal                 = refusal_history.messages.back();
    failures += check(refusal.content.size() == 2 && refusal.content[0].text == "part" &&
                          refusal.content[1].text == "top-level",
                      "assistant refusal history is preserved as assistant text");

    body = base_request();
    body["messages"].push_back(Json{{"role", "assistant"}});
    failures += check(parse(body).generation.messages.back().content.empty(),
                      "an empty assistant history turn is representable");

    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "run it"}},
         Json{{"role", "assistant"},
              {"content", nullptr},
              {"function_call", Json{{"name", "legacy"}, {"arguments", R"({"value":1})"}}}},
         Json{{"role", "function"}, {"name", "legacy"}, {"content", "done"}}});
    const GenerationRequest legacy = parse(body).generation;
    failures += check(legacy.messages[1].tool_calls.size() == 1 &&
                          legacy.messages[1].tool_calls[0].name == "legacy" &&
                          legacy.messages[2].role == ninferno::ChatRole::Tool,
                      "legacy function-call history lowers to Engine tool history");

    body["messages"] = Json::array(
        {Json{{"role", "assistant"},
              {"content", nullptr},
              {"tool_calls",
               Json::array({Json{{"id", ""},
                                 {"type", "function"},
                                 {"function", Json{{"name", "weather"}, {"arguments", "{}"}}}}})}},
         Json{{"role", "tool"}, {"tool_call_id", ""}, {"content", "done"}}});
    failures += check(parse(body).generation.has_tool_history(),
                      "string tool-call identifiers may be empty without changing history");

    // Some clients replay a call's arguments as the JSON object instead of its text.
    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "run it"}},
         Json{{"role", "assistant"},
              {"content", nullptr},
              {"tool_calls",
               Json::array({Json{{"id", "call_1"},
                                 {"type", "function"},
                                 {"function", Json{{"name", "weather"},
                                                   {"arguments", Json{{"city", "Oslo"}}}}}}})}},
         Json{{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "done"}}});
    const GenerationRequest object_args = parse(body).generation;
    failures += check(object_args.messages[1].tool_calls.size() == 1 &&
                          Json::parse(object_args.messages[1].tool_calls[0].arguments_json) ==
                              Json{{"city", "Oslo"}},
                      "object tool-call arguments in history are taken as their JSON text");
    body["messages"][1]["tool_calls"][0]["function"]["arguments"] = 7;
    const ApiError number_args = api_error([&] { (void)parse(body); });
    failures += check(number_args.param == "messages[1].tool_calls[0].function.arguments",
                      "non-object, non-string tool-call arguments are rejected at their location");
    body["messages"][1] =
        Json{{"role", "assistant"},
             {"content", nullptr},
             {"function_call", Json{{"name", "legacy"}, {"arguments", Json{{"value", 1}}}}}};
    body["messages"][2] = Json{{"role", "function"}, {"name", "legacy"}, {"content", "done"}};
    failures +=
        check(Json::parse(parse(body).generation.messages[1].tool_calls[0].arguments_json) ==
                  Json{{"value", 1}},
              "object legacy function_call arguments are taken as their JSON text");
    return failures;
}

int test_reasoning_and_extensions() {
    int failures = 0;
    Json body    = base_request();
    body["reasoning_effort"] = "default";
    failures += check(!parse(body).generation.reasoning_effort.has_value(),
                      "reasoning_effort default alias resolves to the server-configured level");
    body["reasoning_effort"] = "auto";
    failures += check(!parse(body).generation.reasoning_effort.has_value(),
                      "reasoning_effort auto alias resolves to the server-configured level");
    body["reasoning_effort"] = "xhigh";
    failures += check(parse(body).generation.reasoning_effort == ninferno::ReasoningEffort::XHigh,
                      "explicit reasoning_effort still reaches the Engine");
    body = base_request();
    body["messages"].push_back(Json{{"role", "assistant"},
                                    {"content", "answer"},
                                    {"reasoning_content", "thought"},
                                    {"reasoning", "thought"}});
    failures += check(parse(body).generation.messages.back().reasoning_content == "thought",
                      "assistant reasoning aliases normalize");
    body["messages"].back()["reasoning"] = "different";
    failures += check(api_error([&] { (void)parse(body); }).code == "conflicting_template_option",
                      "conflicting assistant reasoning aliases rejected");
    body["messages"].back()["reasoning_content"] = "";
    failures += check(parse(body).generation.messages.back().reasoning_content == "different",
                      "an empty reasoning alias does not conflict with a meaningful alias");
    body = base_request();
    body["messages"].push_back(Json{
        {"role", "assistant"}, {"content", nullptr}, {"reasoning_content", "unfinished thought"}});
    failures +=
        check(parse(body).generation.messages.back().reasoning_content == "unfinished thought",
              "reasoning-only assistant history is preserved");

    body                      = base_request();
    body["enable_thinking"]      = true;
    body["preserve_thinking"]    = false;
    body["chat_template_kwargs"] = Json{{"enable_thinking", true}, {"preserve_thinking", false}};
    const GenerationRequest normalized = parse(body).generation;
    failures += check(normalized.enable_thinking == true && normalized.preserve_thinking == false,
                      "Qwen/vLLM template aliases normalize");
    body["chat_template_kwargs"]["enable_thinking"] = false;
    failures += check(api_error([&] { (void)parse(body); }).code == "conflicting_template_option",
                      "conflicting thinking aliases rejected");
    body                      = base_request();
    body["chat_template_kwargs"] = Json{{"future", 1}};
    failures +=
        check(Json::parse(parse(body).generation.chat_template_kwargs_json).at("future") == 1,
              "custom template keyword did not survive protocol parsing");
    body["chat_template_kwargs"] = Json{{"future", nullptr}};
    failures += check(parse(body).generation.messages.size() == 1,
                      "null unknown template option is neutral");

    // The ninferno extension object has no members: null, absent or empty; anything else is a 400.
    body           = base_request();
    body["ninferno"] = Json::object();
    failures +=
        check(parse(body).generation.messages.size() == 1, "an empty ninferno object is neutral");
    body["ninferno"] = nullptr;
    failures +=
        check(parse(body).generation.messages.size() == 1, "a null ninferno object is neutral");
    for (const Json& bad :
         {Json{{"spec_family", "code"}}, Json{{"family", "code"}}, Json("code")}) {
        body["ninferno"] = bad;
        failures += check(api_error([&] { (void)parse(body); }).status == 400,
                          "a bad ninferno extension is a 400: " + bad.dump());
    }

    body                        = base_request();
    body["repetition_penalty"]  = 1.0;
    body["mm_processor_kwargs"] = Json{{"max_pixels", nullptr}};
    failures +=
        check(parse(body).generation.messages.size() == 1, "neutral ecosystem defaults accepted");
    body["repetition_penalty"] = 1.1;
    failures +=
        check(api_error([&] { (void)parse(body); }).code == "repetition_penalty_not_supported",
              "non-neutral repetition penalty rejected");
    body                        = base_request();
    body["mm_processor_kwargs"] = Json{{"max_pixels", 100}};
    failures +=
        check(api_error([&] { (void)parse(body); }).code == "mm_processor_kwargs_not_supported",
              "non-empty media processor kwargs rejected");
    return failures;
}

int test_stops_and_ranges() {
    int failures                            = 0;
    Json body                               = base_request();
    body["stop"]                            = Json::array({"A", "B"});
    const ninferno::RequestOptions translated = options(parse(body).generation);
    failures += check(translated.stop.strings.size() == 2,
                      "each stop string applies to Content only");
    failures += check(translated.stop.strings[0].channel == ninferno::OutputChannel::Content &&
                          translated.stop.strings[1].channel == ninferno::OutputChannel::Content,
                      "a stop string never ends the reasoning");

    body["stop"] = Json::array({"1", "2", "3", "4", "5"});
    failures += check(options(parse(body).generation).stop.strings.size() == 5,
                      "more than four stop strings are accepted");
    Json many = Json::array();
    for (int i = 0; i < 17; ++i) { many.push_back(std::to_string(i)); }
    body["stop"] = many;
    failures += check(api_error([&] { (void)parse(body); }).param == "stop",
                      "more than sixteen stop strings rejected");
    body["stop"] = "";
    failures += check(options(parse(body).generation).stop.strings.empty(),
                      "an empty stop string means none");
    body["stop"] = Json::array({"", "X"});
    failures += check(options(parse(body).generation).stop.strings.size() == 1,
                      "empty entries are dropped");
    body = base_request();
    body["max_tokens"] = -1;
    failures += check(!parse(body).output_tokens_explicit, "max_tokens -1 means the default");
    body = base_request();
    body["think"] = false;
    failures += check(parse(body).generation.enable_thinking == std::optional<bool>(false),
                      "Ollama's think switches thinking");
    body = base_request();
    body["thinking"] = Json{{"type", "disabled"}};
    failures += check(parse(body).generation.enable_thinking == std::optional<bool>(false),
                      "thinking.type disabled switches thinking off");
    body["enable_thinking"] = true;
    failures += check(api_error([&] { (void)parse(body); }).code == "conflicting_template_option",
                      "conflicting thinking switches are refused");

    body = base_request();
    failures += check(options(parse(body).generation).stop.include_model_defaults,
                      "an omitted ignore_eos keeps the checkpoint's own stop tokens");
    body["ignore_eos"] = false;
    failures += check(options(parse(body).generation).stop.include_model_defaults,
                      "ignore_eos false keeps the checkpoint's own stop tokens");
    body["ignore_eos"] = true;
    failures += check(parse(body).generation.ignore_eos &&
                          !options(parse(body).generation).stop.include_model_defaults,
                      "ignore_eos suppresses the checkpoint's own stop tokens");
    body["stop"] = Json::array({"A"});
    failures += check(options(parse(body).generation).stop.strings.size() == 1 &&
                          !options(parse(body).generation).stop.include_model_defaults,
                      "ignore_eos leaves caller stop strings in place");
    body.erase("stop");
    body["ignore_eos"] = "true";
    failures += check(api_error([&] { (void)parse(body); }).param == "ignore_eos",
                      "a non-boolean ignore_eos is rejected");

    body                                  = base_request();
    body["top_k"]                         = ninferno::kMaximumTopK + 1;
    const GenerationRequest invalid_top_k = parse(body).generation;
    failures += check(api_error([&] { (void)options(invalid_top_k); }).param == "top_k",
                      "Engine translator owns sampler value range");
    body["top_k"]                         = 5;
    body["min_p"]                         = 1.1;
    const GenerationRequest invalid_min_p = parse(body).generation;
    failures += check(api_error([&] { (void)options(invalid_min_p); }).param == "min_p",
                      "min_p range enforced by common Engine translator");
    return failures;
}

GenerationOutcome sample_outcome() {
    GenerationOutcome outcome;
    outcome.text                                = "answer";
    outcome.reasoning                           = "thought";
    outcome.prompt_tokens                       = 20;
    outcome.completion_tokens                   = 7;
    outcome.reasoning_tokens                    = 3;
    outcome.finish_reason                       = ninferno::FinishReason::StopToken;
    outcome.metrics.prefix_cache_hit_tokens     = 12;
    outcome.metrics.prompt_wall_seconds         = 0.04;
    outcome.metrics.generation_wall_seconds     = 0.03;
    outcome.metrics.speculative_draft_tokens    = 9;
    outcome.metrics.speculative_accepted_tokens = 6;
    return outcome;
}

OpenAIChatResponseIdentity identity() {
    return OpenAIChatResponseIdentity{.id = "chatcmpl-test", .model = "qwen", .created = 42};
}

int test_aggregate_response() {
    int failures              = 0;
    GenerationOutcome outcome = sample_outcome();
    Json response             = Json::parse(make_chat_completion_response(identity(), outcome));
    failures += check(response["choices"][0]["message"]["content"] == "answer" &&
                          response["choices"][0]["message"]["reasoning_content"] == "thought" &&
                          response["choices"][0]["message"]["refusal"].is_null(),
                      "aggregate response separates reasoning and content");
    failures += check(response["choices"][0]["logprobs"].is_null(),
                      "aggregate choice carries nullable logprobs");
    failures += check(response["usage"]["prompt_tokens_details"]["cached_tokens"] == 12 &&
                          response["usage"]["completion_tokens_details"]["reasoning_tokens"] == 3,
                      "aggregate usage exposes cache hits and reasoning tokens");
    failures += check(
        response["timings"]["cache_n"] == 12 && response["timings"]["prompt_n"] == 8 &&
            response["timings"]["prompt_ms"] == 40.0 &&
            response["timings"]["prompt_per_second"] == 200.0 &&
            response["timings"]["predicted_n"] == 7 &&
            response["timings"]["predicted_ms"] == 30.0 &&
            response["timings"]["predicted_per_second"] == 200.0 &&
            response["timings"]["draft_n"] == 9 && response["timings"]["draft_n_accepted"] == 6,
        "aggregate timings use exact cache and N-1 generation intervals");

    outcome.text.clear();
    outcome.tool_calls.push_back(ninferno::GeneratedToolCall{
        .name = "Edit",
        .arguments_json =
            R"({"file_path":"/tmp/probe.cpp","old_string":"old","new_string":"new"})"});
    response         = Json::parse(make_chat_completion_response(identity(), outcome));
    const Json& call = response["choices"][0]["message"]["tool_calls"][0];
    failures += check(response["choices"][0]["finish_reason"] == "tool_calls" &&
                          response["choices"][0]["message"]["content"].is_null(),
                      "aggregate tool call has OpenAI terminal shape");
    failures += check(
        call["id"].get<std::string>().starts_with("call_") && call["function"]["name"] == "Edit" &&
            !Json::parse(call["function"]["arguments"].get<std::string>()).contains("replace_all"),
        "OpenAI adapter owns wire tool-call identifiers");
    return failures;
}

int test_stream_response() {
    int failures = 0;
    OpenAIChatStream stream(identity(), true);
    failures += check(stream.start().starts_with(":"),
                      "stream opens with an SSE comment, not a chunk a client would time");
    Json reasoning = parse_sse(stream.reasoning_delta("thought"));
    Json content   = parse_sse(stream.content_delta("ans"));
    failures +=
        check(reasoning["choices"][0]["delta"]["role"] == "assistant" &&
                  reasoning["choices"][0]["logprobs"].is_null() && reasoning["usage"].is_null(),
              "the first token's chunk carries the role, nullable logprobs and null usage");
    failures += check(reasoning["choices"][0]["delta"]["reasoning_content"] == "thought" &&
                          content["choices"][0]["delta"]["content"] == "ans" &&
                          !content["choices"][0]["delta"].contains("role"),
                      "stream separates reasoning and content deltas; the role is sent once");

    GenerationOutcome outcome             = sample_outcome();
    const std::vector<std::string> events = stream.finish(outcome);
    failures +=
        check(events.size() == 4, "finish emits buffered suffix, terminal, usage, and done");
    failures += check(parse_sse(events[0])["choices"][0]["delta"]["content"] == "wer",
                      "terminal content suffix is emitted exactly once");
    failures += check(parse_sse(events[1])["choices"][0]["finish_reason"] == "stop",
                      "stream terminal finish reason emitted");
    const Json usage = parse_sse(events[2]);
    failures += check(usage["choices"].empty() &&
                          usage["usage"]["prompt_tokens_details"]["cached_tokens"] == 12 &&
                          usage["usage"]["completion_tokens_details"]["reasoning_tokens"] == 3 &&
                          usage["timings"]["predicted_n"] == 7,
                      "dedicated stream usage carries token accounting and terminal timings");
    failures += check(events.back() == "data: [DONE]\n\n", "stream ends with DONE sentinel");

    OpenAIChatStream choiced_stream(identity(), true, false, false, true);
    (void)choiced_stream.start();
    (void)choiced_stream.reasoning_delta("thought");
    (void)choiced_stream.content_delta("ans");
    const std::vector<std::string> choiced_events = choiced_stream.finish(outcome);
    const Json choiced_usage                      = parse_sse(choiced_events[2]);
    failures += check(
        choiced_usage["choices"].size() == 1 && choiced_usage["choices"][0]["delta"].empty() &&
            choiced_usage["choices"][0]["finish_reason"].is_null() &&
            choiced_usage["usage"]["completion_tokens"] == usage["usage"]["completion_tokens"],
        "usage-chunk-choice keeps usage accounting with a zero-delta choice");

    OpenAIChatStream mismatch(identity(), false);
    (void)mismatch.start();
    (void)mismatch.content_delta("different");
    failures += check(throws_logic([&] { (void)mismatch.finish(outcome); }),
                      "stream encoder rejects terminal/content divergence");

    OpenAIChatStream tool_stream(identity(), false);
    (void)tool_stream.start();
    GenerationOutcome tool_outcome;
    tool_outcome.tool_calls.push_back(ninferno::GeneratedToolCall{
        .name = "Edit", .arguments_json = R"({"file_path":"/tmp/probe.cpp"})"});
    tool_outcome.finish_reason                 = ninferno::FinishReason::StopToken;
    const std::vector<std::string> tool_events = tool_stream.finish(tool_outcome);
    const Json tool_delta                      = parse_sse(tool_events[0]);
    failures += check(
        tool_delta["choices"][0]["delta"]["tool_calls"][0]["id"].get<std::string>().starts_with(
            "call_") &&
            tool_delta["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "Edit" &&
            tool_delta["choices"][0]["delta"]["role"] == "assistant" &&
            parse_sse(tool_events[1])["choices"][0]["finish_reason"] == "tool_calls" &&
            !parse_sse(tool_events[1])["choices"][0]["delta"].contains("role"),
        "stream encoder owns stable OpenAI tool-call shape; a call-only turn carries the role");
    return failures;
}

int test_stream_observations() {
    int failures = 0;
    OpenAIChatStream stream(identity(), true, true, true);
    failures +=
        check(stream.start().starts_with(":"), "the opening comment precedes Engine observations");

    stream.note_start(
        ninferno::GenerationStart{.prompt = {.prompt_tokens = 32}, .reused_prompt_tokens = 12});
    const Json initial = parse_sse(stream.initial_prompt_progress());
    failures += check(initial["choices"][0]["delta"] == Json{{"role", "assistant"}} &&
                          initial["prompt_progress"]["total"] == 32 &&
                          initial["prompt_progress"]["cache"] == 12 &&
                          initial["prompt_progress"]["processed"] == 12 &&
                          initial["prompt_progress"]["time_ms"] == 0,
                      "initial prompt progress begins at the admitted cache frontier");
    failures += check(initial["timings"]["prompt_n"] == 20 && initial["timings"]["prompt_ms"] == 0.0 &&
                          initial["timings"]["prompt_per_second"] == 0.0 &&
                          initial["timings"]["predicted_n"] == 0,
                      "initial prompt progress carries a zero prompt-only timing snapshot");

    const Json middle = parse_sse(stream.prompt_progress(ninferno::PromptProgress{
        .total_prompt_tokens     = 32,
        .reused_prompt_tokens    = 12,
        .processed_prompt_tokens = 20,
        .elapsed_ns              = 57000000,
    }));
    failures += check(middle["prompt_progress"]["processed"] == 20 &&
                          middle["prompt_progress"]["time_ms"] == 57,
                      "prompt progress exposes a cumulative completed frontier");
    failures += check(middle["timings"]["prompt_ms"] == 57.0 &&
                          middle["timings"]["prompt_per_token_ms"] == 7.125 &&
                          middle["timings"]["prompt_per_second"] == 1000.0 * 8.0 / 57.0 &&
                          middle["timings"]["predicted_n"] == 0,
                      "prompt progress timings observe the processed suffix so far");
    const Json complete = parse_sse(stream.prompt_progress(ninferno::PromptProgress{
        .total_prompt_tokens     = 32,
        .reused_prompt_tokens    = 12,
        .processed_prompt_tokens = 32,
        .elapsed_ns              = 100000000,
    }));
    failures +=
        check(complete["prompt_progress"]["processed"] == complete["prompt_progress"]["total"],
              "final prompt progress reaches the complete prompt");
    failures += check(complete["timings"]["prompt_per_token_ms"] == 5.0 &&
                          complete["timings"]["prompt_per_second"] == 200.0,
                      "final prompt progress timings span the complete prompt suffix");

    OpenAIChatStream plain(identity(), false, false, true);
    (void)plain.start();
    plain.note_start(
        ninferno::GenerationStart{.prompt = {.prompt_tokens = 8}, .reused_prompt_tokens = 0});
    const Json plain_progress = parse_sse(plain.initial_prompt_progress());
    failures += check(!plain_progress.contains("timings"),
                      "prompt progress omits timings without timings_per_token");

    stream.note_timing(ninferno::GenerationTimingObservation{
        .generated_tokens = 1, .prompt_elapsed_ns = 110000000, .generation_elapsed_ns = 0});
    stream.note_timing(ninferno::GenerationTimingObservation{
        .generated_tokens      = 3,
        .prompt_elapsed_ns     = 110000000,
        .generation_elapsed_ns = 20000000,
    });
    const Json content = parse_sse(stream.content_delta("answer"));
    failures +=
        check(content["timings"]["prompt_n"] == 20 && content["timings"]["predicted_n"] == 3 &&
                  content["timings"]["predicted_per_second"] == 100.0,
              "visible output uses the latest independent commit observation");

    GenerationOutcome outcome = sample_outcome();
    outcome.reasoning.clear();
    const std::vector<std::string> terminal = stream.finish(outcome);
    failures += check(parse_sse(terminal[1])["timings"]["predicted_n"] == 7,
                      "terminal usage replaces live timing with exact final accounting");
    return failures;
}

int test_common_objects() {
    int failures      = 0;
    const ninferno::ModelMetadata metadata{
        .model_id       = "qwen3.6-27b",
        .weights_id     = "groupwise-int",
        .vocab_size     = 248077,
        .embedding_size = 5120,
        .native_context = 262144,
        .parameters     = 27000000000ULL,
        .weight_bytes   = 17000000000ULL,
    };
    const Json models = Json::parse(make_models_list("qwen", 7, 240000, metadata));
    failures +=
        check(models["data"][0]["id"] == "qwen" && models["data"][0]["max_model_len"] == 240000,
              "models list advertises the configured context limit");
    failures += check(models["models"].is_array() && models["models"].empty(),
                      "models list carries the empty Codex catalog array");
    const Json list_meta = models["data"][0]["meta"];
    failures +=
        check(list_meta["n_vocab"] == 248077 && list_meta["n_ctx"] == 240000 &&
                  list_meta["n_ctx_train"] == 262144 && list_meta["n_embd"] == 5120 &&
                  list_meta["n_params"] == 27000000000ULL && list_meta["size"] == 17000000000ULL &&
                  list_meta["ftype"] == "groupwise-int",
              "models list exposes the llama.cpp-compatible model meta");
    const Json model = Json::parse(make_model_object("qwen", 7, 240000, metadata));
    failures += check(model["max_model_len"] == 240000,
                      "model lookup advertises the configured context limit");
    failures += check(model["meta"]["n_embd"] == 5120 && model["meta"]["n_ctx_train"] == 262144 &&
                          model["meta"]["ftype"] == "groupwise-int",
                      "model lookup exposes the llama.cpp-compatible model meta");
    failures += check(!list_meta.contains("published_activations"),
                      "models list says nothing of activations a checkpoint does not describe");
    auto described                  = metadata;
    described.published_activations = {{16, 31000}, {8, 60}, {0, 2}};
    const Json described_meta =
        Json::parse(make_model_object("qwen", 7, 240000, described))["meta"]["published_activations"];
    failures += check(described_meta == Json{{"16", 31000}, {"8", 60}, {"unknown", 2}},
                      "model lookup gives the publisher's activation widths per matrix count");
    const Json error = Json::parse(make_error_body(
        ApiError{.status = 400, .message = "bad", .param = "messages", .code = "invalid"}));
    failures += check(error["error"]["param"] == "messages" && error["error"]["code"] == "invalid",
                      "OpenAI common error shape remains stable");
    return failures;
}

} // namespace

// A field the parser neither reads nor knows as neutral is accepted, and one request note lists
// such fields (the response carries it as X-NInferno-Warning); known fields add no note.
int test_unknown_fields_note() {
    Json known             = base_request();
    known["user"]          = "u";
    known["metadata"]      = Json{{"k", "v"}};
    known["service_tier"]  = "auto";
    known["prediction"]    = Json{{"type", "content"}, {"content", ""}};
    int failures           = check(parse(known).generation.notes.empty(),
                                   "standard neutral fields drew an unknown-field note");
    Json misspelt          = base_request();
    misspelt["max_token"]  = 16;
    misspelt["min_tokens"] = 4;
    misspelt["bad\nname"]  = 1;
    const auto notes       = parse(misspelt).generation.notes;
    failures += check(notes.size() == 1 && notes[0].starts_with("unknown fields ignored: ") &&
                          notes[0].find("max_token") != std::string::npos &&
                          notes[0].find("min_tokens") != std::string::npos &&
                          notes[0].find("bad\\x0aname") != std::string::npos &&
                          notes[0].find('\n') == std::string::npos,
                      "unknown fields were not listed, or a name was not escaped");
    return failures;
}

int test_message_count_limit() {
    int failures = 0;
    Json body    = base_request();
    body["messages"].push_back(Json{{"role", "assistant"}, {"content", "hi"}});
    body["messages"].push_back(Json{{"role", "user"}, {"content", "again"}});
    RequestLimits narrow = limits();
    narrow.max_messages  = 3;
    failures += check(parse_chat_completion_request(body, narrow).generation.messages.size() == 3,
                      "a request at the message limit is accepted");
    narrow.max_messages  = 2;
    const ApiError error = api_error([&] { (void)parse_chat_completion_request(body, narrow); });
    failures += check(error.status == 400 && error.code == "too_many_messages" &&
                          error.param == "messages" &&
                          error.message.find("--max-request-messages") != std::string::npos,
                      "a request over the message limit is a 400 naming the flag");
    return failures;
}

int main() {
    int failures = 0;
    failures += test_message_count_limit();
    failures += test_request_envelope_and_sampling();
    failures += test_standard_field_policy();
    failures += test_auto_system_shared_prefix();
    failures += test_constrained_decoding_extensions();
    failures += test_unknown_fields_note();
    failures += test_tools();
    failures += test_tool_name_diagnostics();
    failures += test_messages_and_media();
    failures += test_tool_result_order();
    failures += test_reasoning_and_extensions();
    failures += test_stops_and_ranges();
    failures += test_aggregate_response();
    failures += test_stream_response();
    failures += test_stream_observations();
    failures += test_common_objects();
    if (failures == 0) { std::cout << "OpenAI Chat protocol tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
