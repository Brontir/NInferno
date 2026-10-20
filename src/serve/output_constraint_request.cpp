#include "serve/output_constraint_request.h"

#include "serve/request_validation.h"
#include "text/json_depth.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ninferno::serve {
namespace {

using Json = RequestJson;
using Kind = ninferno::OutputConstraintKind;

void reject_unknown_members(const Json& object, const std::unordered_set<std::string>& allowed,
                            const std::string& param) {
    for (auto it = object.begin(); it != object.end(); ++it) {
        if (allowed.contains(it.key()) || it.value().is_null()) { continue; }
        bad_request(param + "." + it.key() +
                        " is not supported; NInferno rejects structured-output options it cannot "
                        "enforce",
                    param, "structured_output_option_not_supported");
    }
}

bool valid_format_name(std::string_view name) {
    if (name.empty() || name.size() > 64) { return false; }
    for (const char c : name) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                        c == '_' || c == '-';
        if (!ok) { return false; }
    }
    return true;
}

OutputConstraintRequest json_schema_request(const Json& holder, const std::string& param,
                                            bool require_name) {
    if (require_name) {
        if (!holder.contains("name") || !holder.at("name").is_string() ||
            !valid_format_name(holder.at("name").get_ref<const std::string&>())) {
            bad_request(param + ".name must match [A-Za-z0-9_-]{1,64}", param);
        }
    }
    if (holder.contains("description") && !holder.at("description").is_null() &&
        !holder.at("description").is_string()) {
        bad_request(param + ".description must be a string", param);
    }
    if (holder.contains("strict") && !holder.at("strict").is_null() &&
        !holder.at("strict").is_boolean()) {
        bad_request(param + ".strict must be a boolean", param);
    }
    if (!holder.contains("schema") || !holder.at("schema").is_object()) {
        bad_request(param + ".schema must be a JSON Schema object", param);
    }
    // strict=true and strict=false are enforced identically: the schema is always applied to
    // every generated token, which is at least the guarantee strict mode promises.
    OutputConstraintRequest out;
    out.kind       = Kind::JsonSchema;
    out.definition = holder.at("schema").dump();
    out.param      = param;
    return out;
}

OutputConstraintRequest json_object_request(const std::string& param) {
    OutputConstraintRequest out;
    out.kind  = Kind::JsonObject;
    out.param = param;
    return out;
}

std::string require_string(const Json& value, const std::string& param, const char* what) {
    if (!value.is_string() || value.get_ref<const std::string&>().empty()) {
        bad_request(param + " must be a non-empty " + what + " string", param);
    }
    return value.get<std::string>();
}

// JSON a field carries as text was not under the body's depth check.
void refuse_deep_text_json(const Json& parsed, const std::string& param) {
    if (text::json_nests_deeper_than(parsed, kMaximumRequestJsonDepth)) {
        bad_request(param + " nests deeper than " + std::to_string(kMaximumRequestJsonDepth) +
                        " levels",
                    param);
    }
}

OutputConstraintRequest schema_value_request(const Json& value, const std::string& param) {
    OutputConstraintRequest out;
    out.kind  = Kind::JsonSchema;
    out.param = param;
    if (value.is_object()) {
        out.definition = value.dump();
    } else if (value.is_string()) {
        const Json parsed = Json::parse(value.get<std::string>(), nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) {
            bad_request(param + " must be a JSON Schema object or its JSON text", param,
                        "invalid_schema");
        }
        refuse_deep_text_json(parsed, param);
        out.definition = parsed.dump();
    } else {
        bad_request(param + " must be a JSON Schema object or its JSON text", param);
    }
    return out;
}

OutputConstraintRequest choice_request(const Json& value, const std::string& param) {
    if (!value.is_array() || value.empty()) {
        bad_request(param + " must be a non-empty array of strings", param);
    }
    OutputConstraintRequest out;
    out.kind  = Kind::Choice;
    out.param = param;
    for (const Json& choice : value) {
        if (!choice.is_string()) { bad_request(param + " entries must be strings", param); }
        out.choices.push_back(choice.get<std::string>());
    }
    return out;
}

OutputConstraintRequest structural_tag_request(const Json& value, const std::string& param) {
    OutputConstraintRequest out;
    out.kind  = Kind::StructuralTag;
    out.param = param;
    if (value.is_object()) {
        out.definition = value.dump();
    } else if (value.is_string() && !value.get_ref<const std::string&>().empty()) {
        const Json parsed = Json::parse(value.get<std::string>(), nullptr, false);
        if (!parsed.is_discarded()) { refuse_deep_text_json(parsed, param); }
        out.definition = value.get<std::string>();
    } else {
        bad_request(param + " must be a structural-tag object or its JSON text", param);
    }
    return out;
}

std::optional<OutputConstraintRequest> parse_structured_outputs_object(const Json& value) {
    const std::string param = "structured_outputs";
    if (!value.is_object()) { bad_request("structured_outputs must be an object", param); }
    reject_unknown_members(value,
                           {"json", "regex", "choice", "grammar", "json_object", "structural_tag",
                            "disable_fallback", "disable_any_whitespace",
                            "disable_additional_properties", "whitespace_pattern", "backend"},
                           param);
    std::optional<OutputConstraintRequest> out;
    const auto set = [&](OutputConstraintRequest request) {
        if (out) {
            bad_request("structured_outputs must set exactly one of json, regex, choice, grammar, "
                        "json_object or structural_tag",
                        param, "conflicting_constraints");
        }
        out = std::move(request);
    };
    if (value.contains("json") && !value.at("json").is_null()) {
        set(schema_value_request(value.at("json"), param + ".json"));
    }
    if (value.contains("regex") && !value.at("regex").is_null()) {
        OutputConstraintRequest request;
        request.kind       = Kind::Regex;
        request.param      = param + ".regex";
        request.definition = require_string(value.at("regex"), request.param, "regex");
        set(std::move(request));
    }
    if (value.contains("choice") && !value.at("choice").is_null()) {
        set(choice_request(value.at("choice"), param + ".choice"));
    }
    if (value.contains("grammar") && !value.at("grammar").is_null()) {
        OutputConstraintRequest request;
        request.kind       = Kind::Grammar;
        request.param      = param + ".grammar";
        request.definition = require_string(value.at("grammar"), request.param, "EBNF grammar");
        set(std::move(request));
    }
    if (value.contains("json_object") && !value.at("json_object").is_null()) {
        if (!value.at("json_object").is_boolean()) {
            bad_request("structured_outputs.json_object must be a boolean", param);
        }
        if (value.at("json_object").get<bool>()) {
            set(json_object_request(param + ".json_object"));
        }
    }
    if (value.contains("structural_tag") && !value.at("structural_tag").is_null()) {
        set(structural_tag_request(value.at("structural_tag"), param + ".structural_tag"));
    }
    for (const char* flag :
         {"disable_fallback", "disable_any_whitespace", "disable_additional_properties"}) {
        if (value.contains(flag) && !value.at(flag).is_null() && !value.at(flag).is_boolean()) {
            bad_request(param + "." + flag + " must be a boolean", param);
        }
    }
    if (value.contains("whitespace_pattern") && !value.at("whitespace_pattern").is_null()) {
        bad_request("structured_outputs.whitespace_pattern is not supported; NInferno bounds JSON "
                    "whitespace itself (disable_any_whitespace selects compact JSON)",
                    param, "structured_output_option_not_supported");
    }
    if (value.contains("backend") && !value.at("backend").is_null()) {
        const Json& backend = value.at("backend");
        if (!backend.is_string() || (backend.get<std::string>() != "xgrammar" &&
                                     backend.get<std::string>() != "auto")) {
            bad_request("structured_outputs.backend must be 'auto' or 'xgrammar'", param,
                        "structured_output_option_not_supported");
        }
    }
    if (!out) {
        // Options without a constraint request nothing.
        return std::nullopt;
    }
    if (value.contains("disable_any_whitespace") && value.at("disable_any_whitespace").is_boolean()) {
        out->compact_json = value.at("disable_any_whitespace").get<bool>();
    }
    return out;
}

} // namespace

std::optional<OutputConstraintRequest>
parse_openai_response_format(const RequestJson& format, const char* param_name,
                             bool flat_json_schema) {
    const std::string param = param_name;
    // An empty object carries no request for structure; clients send it as "default".
    if (format.is_object() && format.empty()) { return std::nullopt; }
    if (!format.is_object() || !format.contains("type") || !format.at("type").is_string()) {
        bad_request(param + " must be an object with a string type", param);
    }
    const std::string type = format.at("type").get<std::string>();
    if (type == "text") {
        reject_unknown_members(format, {"type"}, param);
        return std::nullopt;
    }
    if (type == "json_object") {
        reject_unknown_members(format, {"type"}, param);
        return json_object_request(param);
    }
    if (type == "json_schema") {
        if (flat_json_schema) {
            reject_unknown_members(format, {"type", "name", "schema", "strict", "description"},
                                   param);
            return json_schema_request(format, param, true);
        }
        reject_unknown_members(format, {"type", "json_schema"}, param);
        if (!format.contains("json_schema") || !format.at("json_schema").is_object()) {
            bad_request(param + ".json_schema must be an object", param);
        }
        const Json& holder = format.at("json_schema");
        reject_unknown_members(holder, {"name", "schema", "strict", "description"},
                               param + ".json_schema");
        return json_schema_request(holder, param, true);
    }
    bad_request(param + " type '" + type +
                    "' is not supported; use text, json_object or json_schema",
                param, "response_format_not_supported");
}

OutputConstraintRequest parse_anthropic_output_format(const RequestJson& format,
                                                      const char* param_name) {
    const std::string param = param_name;
    if (!format.is_object() || !format.contains("type") || !format.at("type").is_string()) {
        bad_request(param + " must be an object with a string type", param);
    }
    if (format.at("type").get<std::string>() != "json_schema") {
        bad_request(param + ".type must be 'json_schema'", param,
                    "output_config_format_not_supported");
    }
    reject_unknown_members(format, {"type", "schema"}, param);
    return json_schema_request(format, param, false);
}

// A JSON response format beside enabled tools (Agent SDKs send both) constrains the text answer:
// the model makes calls, or answers in the format. When a call is required the answer is calls
// only, so the format has nothing to apply to. Other formats (regex, grammar, choice) cannot be
// told apart from call markup and stay refused.
void attach_tool_call_constraint(GenerationRequest& request) {
    if (!request.uses_tools()) { return; }
    std::string answer_schema;
    bool compact_answer = false;
    if (request.output_constraint) {
        const OutputConstraintRequest& format = *request.output_constraint;
        if (format.kind != Kind::JsonSchema && format.kind != Kind::JsonObject) {
            bad_request(format.param +
                            " with enabled tools must be a JSON schema or json_object format; "
                            "disable tools (tool_choice none) or omit the format",
                        format.param, "constraint_with_tools_not_supported");
        }
        if (request.tool_choice.requirement == ninferno::ToolCallRequirement::Optional) {
            answer_schema  = format.kind == Kind::JsonObject ? std::string(R"({"type":"object"})")
                                                             : format.definition;
            compact_answer = format.compact_json;
        }
        request.output_constraint.reset();
    }
    const bool strict = std::any_of(request.tools.begin(), request.tools.end(),
                                    [](const ToolDefinition& tool) { return tool.strict; });
    if (!strict && request.tool_choice.requirement == ninferno::ToolCallRequirement::Optional &&
        request.parallel_tool_calls && answer_schema.empty()) {
        return;
    }
    OutputConstraintRequest out;
    out.answer_schema       = std::move(answer_schema);
    out.compact_json        = compact_answer;
    out.kind                = Kind::ToolCalls;
    out.param               = "tools";
    out.tool_requirement    = request.tool_choice.requirement;
    out.named_tool          = request.tool_choice.named;
    out.parallel_tool_calls = request.parallel_tool_calls;
    out.tools.reserve(request.tools.size());
    for (const ToolDefinition& tool : request.tools) {
        out.tools.push_back(ninferno::ToolCallConstraintTool{
            .name = tool.name, .parameters_json = tool.input_schema_json, .strict = tool.strict});
    }
    request.output_constraint = std::move(out);
}

std::optional<OutputConstraintRequest> parse_structured_output_extensions(const RequestJson& body) {
    std::optional<OutputConstraintRequest> out;
    const auto set = [&](std::optional<OutputConstraintRequest> request) {
        if (!request) { return; }
        if (out) {
            bad_request("only one structured-output request is allowed; got " + out->param +
                            " and " + request->param,
                        request->param, "conflicting_constraints");
        }
        out = std::move(request);
    };

    if (body.contains("structured_outputs") && !body.at("structured_outputs").is_null()) {
        set(parse_structured_outputs_object(body.at("structured_outputs")));
    }
    if (body.contains("guided_json") && !body.at("guided_json").is_null()) {
        set(schema_value_request(body.at("guided_json"), "guided_json"));
    }
    if (body.contains("guided_regex") && !body.at("guided_regex").is_null()) {
        OutputConstraintRequest request;
        request.kind       = Kind::Regex;
        request.param      = "guided_regex";
        request.definition = require_string(body.at("guided_regex"), request.param, "regex");
        set(std::move(request));
    }
    if (body.contains("guided_choice") && !body.at("guided_choice").is_null()) {
        set(choice_request(body.at("guided_choice"), "guided_choice"));
    }
    if (body.contains("guided_grammar") && !body.at("guided_grammar").is_null()) {
        OutputConstraintRequest request;
        request.kind       = Kind::Grammar;
        request.param      = "guided_grammar";
        request.definition =
            require_string(body.at("guided_grammar"), request.param, "EBNF grammar");
        set(std::move(request));
    }
    if (body.contains("grammar") && !body.at("grammar").is_null()) {
        const Json& grammar = body.at("grammar");
        if (!grammar.is_string()) { bad_request("grammar must be a string", "grammar"); }
        // llama.cpp treats an empty grammar as no grammar.
        if (!grammar.get_ref<const std::string&>().empty()) {
            OutputConstraintRequest request;
            request.kind       = Kind::Grammar;
            request.param      = "grammar";
            request.definition = grammar.get<std::string>();
            set(std::move(request));
        }
    }
    if (body.contains("guided_whitespace_pattern") &&
        !body.at("guided_whitespace_pattern").is_null()) {
        bad_request("guided_whitespace_pattern is not supported; NInferno bounds JSON whitespace "
                    "itself (use structured_outputs.disable_any_whitespace for compact JSON)",
                    "guided_whitespace_pattern", "structured_output_option_not_supported");
    }
    if (body.contains("guided_decoding_backend") &&
        !body.at("guided_decoding_backend").is_null()) {
        const Json& backend = body.at("guided_decoding_backend");
        if (!backend.is_string() || (backend.get<std::string>() != "xgrammar" &&
                                     backend.get<std::string>() != "auto")) {
            bad_request("guided_decoding_backend must be 'auto' or 'xgrammar'",
                        "guided_decoding_backend", "structured_output_option_not_supported");
        }
    }
    return out;
}

} // namespace ninferno::serve
