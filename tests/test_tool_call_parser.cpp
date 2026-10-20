#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "structured/constraint.h"
#include "structured/request_constraint.h"

#include <nlohmann/json.hpp>

#include <initializer_list>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Json   = nlohmann::json;
namespace fi = ninferno::models::qwen3_5::frontend;

const fi::ToolCallOutputContract kLegacyContract;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

std::string tool_definition(const std::string& tool_name, Json properties,
                            Json required = Json::array()) {
    Json parameters{{"type", "object"}, {"properties", std::move(properties)}};
    if (!required.empty()) { parameters["required"] = std::move(required); }
    return Json{{"type", "function"},
                {"function", Json{{"name", tool_name}, {"parameters", std::move(parameters)}}}}
        .dump();
}

std::shared_ptr<const fi::ToolCallOutputContract>
contract_from_definitions(const std::vector<std::string>& definitions) {
    return fi::build_tool_call_output_contract(
        std::span<const std::string>(definitions.data(), definitions.size()), true);
}

std::shared_ptr<const fi::ToolCallOutputContract> output_contract_for(const std::string& tool_name,
                                                                      Json properties) {
    const std::vector<std::string> definitions = {
        tool_definition(tool_name, std::move(properties))};
    return contract_from_definitions(definitions);
}

fi::ToolCallOutputContract contract_for(const std::string& tool_name, Json properties) {
    return *output_contract_for(tool_name, std::move(properties));
}

std::string
tool_call(std::string_view tool_name,
          std::initializer_list<std::pair<std::string_view, std::string_view>> parameters = {}) {
    std::string text = "<tool_call>\n<function=";
    text.append(tool_name);
    text += ">\n";
    for (const auto& [name, value] : parameters) {
        text += "<parameter=";
        text.append(name);
        text += ">\n";
        text.append(value);
        text += "\n</parameter>\n";
    }
    text += "</function>\n</tool_call>";
    return text;
}

int check_rejected(const std::string& text, const fi::ToolCallOutputContract& contract,
                   ninferno::ToolCallParseFallbackReason reason, std::string_view message) {
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);
    return check(!parsed.is_tool_call_response && parsed.content == text &&
                     parsed.tool_calls.empty() && parsed.diagnostics.marker_seen &&
                     parsed.diagnostics.fallback_reason == reason,
                 std::string(message));
}

int check_parameter_schema_mismatch(const fi::ToolCallOutputContract& contract,
                                    std::string_view parameter_name, std::string_view value,
                                    std::string_view expected_json_value,
                                    std::string_view message) {
    const auto parsed = fi::parse_qwen_tool_call_output(
        tool_call("configure", {{parameter_name, value}}), 64, contract);
    const std::string expected_arguments = "{" + Json(std::string(parameter_name)).dump() + ":" +
                                           std::string(expected_json_value) + "}";
    return check(
        parsed.is_tool_call_response && parsed.content.empty() && parsed.tool_calls.size() == 1 &&
            parsed.tool_calls.front().arguments_json == expected_arguments &&
            parsed.diagnostics.marker_seen && parsed.diagnostics.structured_call_count == 1 &&
            parsed.diagnostics.schema_mismatch_arguments == 1 &&
            parsed.diagnostics.fallback_reason == ninferno::ToolCallParseFallbackReason::None,
        std::string(message));
}

int test_basic_legacy_parsing() {
    const auto parsed = fi::parse_qwen_tool_call_output("Calling weather.\n"
                                                        "<tool_call>\n"
                                                        "<function=get_weather>\n"
                                                        "<parameter=city>\nParis\n</parameter>\n"
                                                        "<parameter=days>\n2\n</parameter>\n"
                                                        "</function>\n"
                                                        "</tool_call>",
                                                        64, kLegacyContract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "legacy call was not parsed");
    failures += check(parsed.content == "Calling weather.", "content prefix was not trimmed");
    failures += check(parsed.tool_calls.size() == 1, "legacy call count changed");
    if (parsed.tool_calls.size() != 1) { return failures; }
    failures += check(parsed.tool_calls.front().name == "get_weather", "function name changed");
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("city") == "Paris", "legacy string inference changed");
    failures += check(args.at("days") == 2, "legacy JSON inference changed");
    return failures;
}

// A function element holding one JSON object (fork igorls/ninferno#9: 3 of 31 production turns
// returned such a call as text) is a call; a body that is not one whole object stays malformed.
int test_multiple_calls() {
    const std::string text = tool_call("first", {{"payload", "{\"ok\":true,\"items\":[1,2]}"}}) +
                             "\n" + tool_call("second", {{"value", "plain text"}});
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, kLegacyContract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 2,
                      "multiple complete calls were not parsed");
    if (parsed.tool_calls.size() != 2) { return failures; }
    const Json first  = Json::parse(parsed.tool_calls[0].arguments_json);
    const Json second = Json::parse(parsed.tool_calls[1].arguments_json);
    failures +=
        check(first.at("payload").at("ok") == true && first.at("payload").at("items").at(1) == 2,
              "legacy object value changed");
    failures += check(second.at("value") == "plain text", "legacy plain text value changed");
    return failures;
}

int test_json_argument_body() {
    int failures      = 0;
    const auto parsed = fi::parse_qwen_tool_call_output(
        "Looking it up.\n<tool_call>\n<function=get_weather>\n{\"city\": \"Paris\", "
        "\"days\": 3, \"note\": \"a } and a </parameter>\"}\n</function>\n</tool_call>",
        64, kLegacyContract);
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                          parsed.content == "Looking it up.",
                      "a JSON-object function body was not parsed as a call");
    if (parsed.tool_calls.size() == 1) {
        failures += check(parsed.tool_calls.front().name == "get_weather" &&
                              parsed.tool_calls.front().arguments_json ==
                                  "{\"city\":\"Paris\",\"days\":3,\"note\":\"a } and a "
                                  "</parameter>\"}",
                          "a JSON-object function body changed its arguments");
    }
    const auto cut = fi::parse_qwen_tool_call_output(
        "<tool_call>\n<function=get_weather>\n{\"city\": \"Paris\"}\n</func", 64, kLegacyContract);
    failures += check(cut.is_tool_call_response && cut.tool_calls.size() == 1,
                      "a JSON-object body before a cut closer was not parsed");
    for (const std::string text :
         {std::string("<tool_call>\n<function=get_weather>\n{\"city\": \"Par\n</function>\n"
                      "</tool_call>"),
          std::string("<tool_call>\n<function=get_weather>\n[1, 2]\n</function>\n</tool_call>"),
          std::string("<tool_call>\n<function=get_weather>\n{\"city\": \"Paris\"} trailing\n"
                      "</function>\n</tool_call>")}) {
        const auto rejected = fi::parse_qwen_tool_call_output(text, 64, kLegacyContract);
        failures += check(!rejected.is_tool_call_response && rejected.content == text,
                          "a malformed JSON function body was accepted: " + text);
    }
    return failures;
}

// A small quant writes a space after "<function=" now and then (Qwen3.8 IQ2_XS: "<function=
// capture>"); the name is the call's, as it is around a parameter's name.
int test_spaced_names() {
    int failures      = 0;
    const auto parsed = fi::parse_qwen_tool_call_output(
        "<tool_call>\n<function= get_weather >\n<parameter= city >\nParis\n</parameter>\n"
        "</function>\n</tool_call>",
        64, kLegacyContract);
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "a function name with surrounding spaces was not parsed as a call");
    if (parsed.tool_calls.size() == 1) {
        failures += check(parsed.tool_calls.front().name == "get_weather" &&
                              parsed.tool_calls.front().arguments_json == "{\"city\":\"Paris\"}",
                          "spaces around a function or parameter name were kept in it");
    }
    const auto blank = fi::parse_qwen_tool_call_output(
        "<tool_call>\n<function= >\n</function>\n</tool_call>", 64, kLegacyContract);
    failures += check(!blank.is_tool_call_response, "a blank function name was accepted");
    return failures;
}

int test_declared_strings_preserve_text() {
    const auto contract =
        contract_for("TaskUpdate",
                     Json{{"taskId", Json{{"type", "string"}}},
                          {"content", Json{{"type", "string"}}},
                          {"truthy", Json{{"type", "string"}}},
                          {"nullish", Json{{"type", "string"}}},
                          {"quoted", Json{{"type", "string"}}},
                          {"windows", Json{{"type", "string"}}},
                          {"string_or_number", Json{{"type", Json::array({"number", "string"})}}}});
    const auto parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=TaskUpdate>\n"
                                        "<parameter=taskId>\n1\n</parameter>\n"
                                        "<parameter=content>\n  {\"x\":1}\n\n</parameter>\n"
                                        "<parameter=truthy>\ntrue\n</parameter>\n"
                                        "<parameter=nullish>\nnull\n</parameter>\n"
                                        "<parameter=quoted>\n\"literal\"\n</parameter>\n"
                                        "<parameter=windows>\r\n  value  \r\n</parameter>\n"
                                        "<parameter=string_or_number>\n7\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        128, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "declared string call was rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("taskId") == "1", "numeric-shaped string was promoted");
    failures +=
        check(args.at("content") == "  {\"x\":1}\n", "string whitespace or content changed");
    failures += check(args.at("truthy") == "true" && args.at("nullish") == "null",
                      "boolean/null-shaped string was promoted");
    failures +=
        check(args.at("quoted") == "\"literal\"", "quoted string was reinterpreted as JSON");
    failures += check(args.at("windows") == "  value  ", "CRLF framing changed string content");
    failures +=
        check(args.at("string_or_number") == "7", "string-admitting union did not preserve text");
    return failures;
}

int test_string_values_preserve_embedded_tool_markup() {
    const auto contract       = contract_for("bash", Json{{"command", Json{{"type", "string"}}},
                                                          {"timeout", Json{{"type", "integer"}}}});
    const std::string command = "python3 - <<'PY'\n"
                                "import re\n"
                                "pattern = r'<parameter=edits>\\n(.*?)\\n</parameter>'\n"
                                "print(pattern)\n"
                                "PY";
    const std::string text    = tool_call("bash", {{"command", command}, {"timeout", "30"}});
    const auto parsed         = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "balanced parameter markup inside a string broke the tool call");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("command") == command,
                      "embedded parameter markup was removed from the string value");
    failures +=
        check(args.at("timeout") == 30, "sibling parameter after embedded markup was not parsed");

    const std::string nested_markup =
        "literal closes: </function> and </tool_call>\n"
        "<function=fake>body</function>\n"
        "<tool_call>body</tool_call>\n"
        "<parameter=outer>before<parameter=inner>value</parameter>after</parameter>";
    const std::string nested_text = tool_call("bash", {{"command", nested_markup}});
    const auto nested             = fi::parse_qwen_tool_call_output(nested_text, 64, contract);
    failures += check(nested.is_tool_call_response && nested.tool_calls.size() == 1,
                      "nested tool markup inside a string broke outer structure");
    if (nested.tool_calls.size() == 1) {
        const Json nested_args = Json::parse(nested.tool_calls.front().arguments_json);
        failures += check(nested_args.at("command") == nested_markup,
                          "nested function/tool/parameter markup was not preserved exactly");
    }
    return failures;
}

int test_unrepresentable_parameter_delimiters_fall_back() {
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string unmatched_open =
        tool_call("bash", {{"command", "echo '<parameter=unterminated>'"}});
    // Only a closer followed by the next grammar token can end the value.
    const std::string closer_before_structure =
        tool_call("bash", {{"command", "echo\n</parameter>\n</function>\ntail"}});

    int failures = 0;
    failures += check_rejected(unmatched_open, contract,
                               ninferno::ToolCallParseFallbackReason::MalformedStructure,
                               "unbalanced nested parameter open was silently repaired");
    failures += check_rejected(closer_before_structure, contract,
                               ninferno::ToolCallParseFallbackReason::MalformedStructure,
                               "structurally ambiguous parameter close was guessed");
    return failures;
}

int test_quoted_parameter_close_stays_in_value() {
    const auto contract = contract_for(
        "agent", Json{{"task", Json{{"type", "string"}}}, {"mode", Json{{"type", "string"}}}});
    // Shape of an observed advisor call: prose quotes the closer sequence with escaped newlines.
    const std::string task =
        "Failures:\n"
        "   2. `</parameter>\\n</function>\\n</tool_call>` - stopped mid value\n"
        "   4. `>\\n</function>` - stopped mid `</parameter>`\n"
        "echo '</parameter>'";
    const std::string text = tool_call("agent", {{"task", task}, {"mode", "read"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                          parsed.content.empty(),
                      "quoted parameter close demoted a valid call to text");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.at("task") == task && args.at("mode") == "read",
                          "quoted parameter close changed parameter values");
    }
    const std::string truncated = "<tool_call>\n<function=agent>\n<parameter=task>\n" + task;
    failures += check_rejected(truncated, contract,
                               ninferno::ToolCallParseFallbackReason::MalformedStructure,
                               "truncated value with quoted closers was accepted");
    return failures;
}

int test_quoted_marker_before_real_call() {
    const auto contract = contract_for("todo_write", Json{{"todos", Json{{"type", "array"}}}});
    // Shape of an observed turn: reasoning quoted the opener, then the model made the real call.
    const std::string prose = "Avoid `<tool_call>\n<function=NAME>`; set continuation.\n"
                              "Design ready; recording the plan.";
    const std::string call  = tool_call("todo_write", {{"todos", "[{\"id\":\"a\"}]"}});
    const std::string text  = prose + "\n\n" + call;

    int failures      = 0;
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);
    failures +=
        check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                  parsed.content == prose && parsed.diagnostics.marker_seen &&
                  parsed.diagnostics.fallback_reason == ninferno::ToolCallParseFallbackReason::None,
              "quoted marker swallowed the real call");

    // Every call after the quote is kept, not only the last one.
    const std::string two = prose + "\n" + call + "\n" + call;
    const auto both       = fi::parse_qwen_tool_call_output(two, 64, contract);
    failures += check(both.tool_calls.size() == 2 && both.content == prose,
                      "recovery dropped an earlier call of a multi-call suffix");

    // A complete quoted example naming an undeclared tool still yields to the declared call after
    // it: declared names pick the region before an undeclared call is taken as one.
    const std::string example = "Format: " + tool_call("NAME", {{"x", "1"}}) + "\nNow:";
    const auto after_example = fi::parse_qwen_tool_call_output(example + "\n" + call, 64, contract);
    failures += check(after_example.tool_calls.size() == 1 &&
                          after_example.tool_calls.front().name == "todo_write" &&
                          after_example.content == example &&
                          after_example.diagnostics.undeclared_calls == 0,
                      "a quoted undeclared example was taken before the declared call");

    // A marker quoted inside the real call's value does not hide the call itself.
    const auto bash = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string quoted_value =
        prose + "\n" + tool_call("bash", {{"command", "grep '<tool_call>'"}});
    const auto in_value = fi::parse_qwen_tool_call_output(quoted_value, 64, bash);
    failures += check(in_value.tool_calls.size() == 1 && in_value.content == prose,
                      "marker quoted in the recovered call's value was chosen as the start");

    // A complete first call is not a quote, so a later call is not executed on its own.
    failures += check_rejected(call + "\nNow also:\n" + call, contract,
                               ninferno::ToolCallParseFallbackReason::TrailingContent,
                               "a later call was recovered after a complete first call");

    // Every candidate fails: raw bytes and the first marker's reason are kept. The quoted
    // `<function=NAME>` fails on its structure once its undeclared name no longer does.
    failures += check_rejected(prose + "\nthen <tool_call> again", contract,
                               ninferno::ToolCallParseFallbackReason::MalformedStructure,
                               "failed recovery lost raw bytes or the primary reason");

    // Streamed in uneven chunks, including a split inside the quoted marker.
    for (const std::size_t chunk : {std::size_t{1}, std::size_t{3}, std::size_t{7}}) {
        fi::ToolCallOutputDecoder decoder(
            output_contract_for("todo_write", Json{{"todos", Json{{"type", "array"}}}}), 64);
        std::string visible;
        for (std::size_t offset = 0; offset < text.size(); offset += chunk) {
            visible += decoder.feed(std::string_view(text).substr(offset, chunk));
        }
        auto terminal = decoder.finish();
        failures += check(visible + terminal.content == prose && terminal.tool_calls.size() == 1,
                          "streamed quoted marker lost prose bytes or the real call");
    }
    return failures;
}

int test_declared_json_types() {
    const auto contract = contract_for(
        "configure", Json{{"count", Json{{"type", "integer"}}},
                          {"total", Json{{"type", "number"}}},
                          {"ratio", Json{{"type", "number"}}},
                          {"enabled", Json{{"type", "boolean"}}},
                          {"payload", Json{{"type", "object"}}},
                          {"items", Json{{"type", "array"}}},
                          {"unset", Json{{"type", "null"}}},
                          {"optional", Json{{"type", Json::array({"integer", "null"})}}}});
    const std::string text = tool_call("configure", {{"count", "7"},
                                                     {"total", "8"},
                                                     {"ratio", "1.5"},
                                                     {"enabled", "true"},
                                                     {"payload", "{\"x\":1}"},
                                                     {"items", "[\"a\",2]"},
                                                     {"unset", "null"},
                                                     {"optional", "null"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "valid declared JSON values were rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("count") == 7, "integer was not decoded");
    failures += check(args.at("total") == 8, "integer did not satisfy number");
    failures += check(args.at("ratio") == 1.5, "fractional number was not decoded");
    failures += check(args.at("enabled") == true, "JSON boolean was not decoded");
    failures += check(args.at("payload").is_object() && args.at("payload").at("x") == 1,
                      "object was not decoded");
    failures +=
        check(args.at("items").is_array() && args.at("items").at(1) == 2, "array was not decoded");
    failures += check(args.at("unset").is_null() && args.at("optional").is_null(),
                      "declared null was not decoded");
    return failures;
}

// An untyped parameter whose text is not JSON keeps its whitespace past the framing newlines
// (indented file content lost its indentation and final newline).
int test_untyped_text_keeps_whitespace() {
    const auto parsed = fi::parse_qwen_tool_call_output(
        tool_call("write", {{"content", "    def f():\n        return 1\n"}, {"count", " 2 "}}), 64,
        kLegacyContract);
    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "untyped call was not parsed");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("content") == "    def f():\n        return 1\n",
                      "untyped text lost its indentation or final newline");
    failures += check(args.at("count") == 2, "untyped JSON inference changed");
    return failures;
}

int test_boolean_boundary() {
    const auto contract =
        contract_for("configure", Json{{"lower_true", Json{{"type", "boolean"}}},
                                       {"title_true", Json{{"type", "boolean"}}},
                                       {"upper_true", Json{{"type", "boolean"}}},
                                       {"mixed_false", Json{{"type", "boolean"}}},
                                       {"spaced_true", Json{{"type", "boolean"}}},
                                       {"windows_false", Json{{"type", "boolean"}}}});
    const auto parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=configure>\n"
                                        "<parameter=lower_true>\ntrue\n</parameter>\n"
                                        "<parameter=title_true>\nTrue\n</parameter>\n"
                                        "<parameter=upper_true>\nTRUE\n</parameter>\n"
                                        "<parameter=mixed_false>\nfAlSe\n</parameter>\n"
                                        "<parameter=spaced_true>\n \tTrUe \n</parameter>\n"
                                        "<parameter=windows_false>\r\nFaLsE\r\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "case-insensitive booleans were rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("lower_true") == true && args.at("title_true") == true &&
                          args.at("upper_true") == true && args.at("spaced_true") == true,
                      "true variants were not canonicalized");
    failures += check(args.at("mixed_false") == false && args.at("windows_false") == false,
                      "false variants were not canonicalized");

    const auto one_flag = contract_for("configure", Json{{"flag", Json{{"type", "boolean"}}}});
    failures += check_parameter_schema_mismatch(one_flag, "flag", "1", "1",
                                                "integer boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "0", "0",
                                                "zero boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "\"true\"", "\"true\"",
                                                "string boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "yes", "\"yes\"",
                                                "plain boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "None", "\"None\"",
                                                "Python null mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "null", "null",
                                                "null boolean mismatch was not structured");
    return failures;
}

int test_exact_integer_boundary() {
    const auto integer_contract =
        contract_for("configure", Json{{"decimal", Json{{"type", "integer"}}},
                                       {"exponent", Json{{"type", "integer"}}},
                                       {"scaled", Json{{"type", "integer"}}},
                                       {"negative_zero", Json{{"type", "integer"}}},
                                       {"large", Json{{"type", "integer"}}}});
    const std::string valid = tool_call("configure", {{"decimal", "7.0"},
                                                      {"exponent", "1e2"},
                                                      {"scaled", "100e-2"},
                                                      {"negative_zero", "-0.0"},
                                                      {"large", "9007199254740992.0"}});
    const auto parsed       = fi::parse_qwen_tool_call_output(valid, 64, integer_contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "mathematically integral JSON numbers were rejected");
    if (parsed.tool_calls.size() == 1) {
        failures += check(parsed.tool_calls.front().arguments_json ==
                              "{\"decimal\":7.0,\"exponent\":1e2,\"scaled\":100e-2,"
                              "\"negative_zero\":-0.0,\"large\":9007199254740992.0}",
                          "integer JSON lexemes were rewritten");
    }

    const auto one_integer = contract_for("configure", Json{{"value", Json{{"type", "integer"}}}});
    failures += check_parameter_schema_mismatch(one_integer, "value", "7.5", "7.5",
                                                "fractional integer mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_integer, "value", "1e-1", "1e-1",
                                                "fractional exponent mismatch was not structured");
    failures += check_parameter_schema_mismatch(
        one_integer, "value", "9007199254740992.5", "9007199254740992.5",
        "large fractional integer mismatch lost its exact lexeme");

    const auto one_number = contract_for("configure", Json{{"value", Json{{"type", "number"}}}});
    const std::string large_fraction = tool_call("configure", {{"value", "9007199254740992.5"}});
    const auto number_parsed = fi::parse_qwen_tool_call_output(large_fraction, 64, one_number);
    failures += check(number_parsed.is_tool_call_response && number_parsed.tool_calls.size() == 1 &&
                          number_parsed.tool_calls.front().arguments_json ==
                              "{\"value\":9007199254740992.5}",
                      "valid number was rejected or lost its original precision");
    return failures;
}

int test_composed_schema_types() {
    const auto contract = contract_for(
        "configure",
        Json{{"flag",
              Json{{"anyOf", Json::array({Json{{"type", "boolean"}}, Json{{"type", "null"}}})}}},
             {"unset",
              Json{{"oneOf", Json::array({Json{{"type", "null"}}, Json{{"type", "boolean"}}})}}},
             {"count",
              Json{{"anyOf", Json::array({Json{{"type", "integer"}}, Json{{"type", "null"}}})}}},
             {"nested",
              Json{{"anyOf", Json::array({Json{{"oneOf", Json::array({Json{{"type", "boolean"}},
                                                                      Json{{"type", "null"}}})}},
                                          Json{{"type", "integer"}}})}}},
             {"string_or_number",
              Json{{"oneOf", Json::array({Json{{"type", "string"}}, Json{{"type", "number"}}})}}}});
    const std::string text = tool_call("configure", {{"flag", "False"},
                                                     {"unset", "null"},
                                                     {"count", "7.0"},
                                                     {"nested", "TRUE"},
                                                     {"string_or_number", "7"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "explicit anyOf/oneOf primitive union was rejected");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.at("flag") == false && args.at("unset").is_null(),
                          "nullable boolean composition was decoded incorrectly");
        failures += check(args.at("count") == 7.0 && args.at("nested") == true,
                          "nested primitive composition was decoded incorrectly");
        failures += check(args.at("string_or_number") == "7",
                          "string-admitting composition did not preserve text");
    }
    failures += check_parameter_schema_mismatch(
        contract, "count", "7.5", "7.5",
        "fractional anyOf integer/null mismatch was not structured");
    return failures;
}

// Pydantic and the OpenAI SDKs emit parameters as local references into $defs or
// definitions. A string field behind a reference must stay a string ("12345" is a zip code, not a
// number); unresolvable and cyclic references keep the legacy policy.
int test_local_reference_schema_types() {
    const Json parameters{
        {"type", "object"},
        {"properties",
         Json{{"zip", Json{{"$ref", "#/$defs/Zip"}}},
              {"version", Json{{"$ref", "#/definitions/Version"}}},
              {"count", Json{{"$ref", "#/$defs/Count"}}},
              {"maybe_zip",
               Json{{"anyOf", Json::array({Json{{"$ref", "#/$defs/Zip"}}, Json{{"type", "null"}}})}}},
              {"alias", Json{{"$ref", "#/$defs/Alias"}}},
              {"loop", Json{{"$ref", "#/$defs/Loop"}}},
              {"missing", Json{{"$ref", "#/$defs/Nowhere"}}},
              {"remote", Json{{"$ref", "https://example.com/schema.json"}}}}},
        {"$defs", Json{{"Zip", Json{{"type", "string"}, {"pattern", "^[0-9]{5}$"}}},
                       {"Count", Json{{"type", "integer"}}},
                       {"Alias", Json{{"$ref", "#/$defs/Zip"}}},
                       {"Loop", Json{{"$ref", "#/$defs/Loop"}}}}},
        {"definitions", Json{{"Version", Json{{"type", "string"}}}}}};
    const std::vector<std::string> definitions = {
        Json{{"type", "function"},
             {"function", Json{{"name", "ship"}, {"parameters", parameters}}}}
            .dump()};
    const auto contract    = contract_from_definitions(definitions);
    const std::string text = tool_call("ship", {{"zip", "12345"},
                                                {"version", "1.0"},
                                                {"count", "3"},
                                                {"maybe_zip", "00501"},
                                                {"alias", "90210"},
                                                {"loop", "7"},
                                                {"missing", "8"},
                                                {"remote", "9"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, *contract);
    int failures           = check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                                   "referenced parameter schemas were rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("zip") == "12345", "a $defs string reference became " +
                                                     args.at("zip").dump());
    failures += check(args.at("version") == "1.0", "a definitions string reference became " +
                                                        args.at("version").dump());
    failures += check(args.at("count") == 3, "an integer reference was not decoded");
    failures += check(args.at("maybe_zip") == "00501", "anyOf over a reference became " +
                                                           args.at("maybe_zip").dump());
    failures += check(args.at("alias") == "90210", "a chained reference became " +
                                                       args.at("alias").dump());
    failures += check(args.at("loop") == 7 && args.at("missing") == 8 && args.at("remote") == 9,
                      "cyclic, dangling or remote references left the legacy policy");
    return failures;
}

int test_empty_declared_non_string_is_omitted() {
    const Json properties = {
        {"file_path", Json{{"type", "string"}}},
        {"new_string", Json{{"type", "string"}}},
        {"old_string", Json{{"type", "string"}}},
        {"replace_all", Json{{"type", "boolean"}}},
    };
    const std::string text = "I need one more check.\n\n"
                             "<tool_call>\n"
                             "<function=Edit>\n"
                             "<parameter=file_path>\n/tmp/probe.cpp\n</parameter>\n"
                             "<parameter=new_string>\n"
                             "std::map<std::uint32_t, int> counts;\n"
                             "</parameter>\n"
                             "<parameter=old_string>\nold line\n</parameter>\n"
                             "<parameter=replace_all>\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>";

    const std::vector<std::string> definitions = {tool_definition(
        "Edit", properties, Json::array({"file_path", "new_string", "old_string"}))};
    const auto contract                        = contract_from_definitions(definitions);
    const auto parsed = fi::parse_qwen_tool_call_output(text, 128, *contract);

    int failures = 0;
    failures +=
        check(parsed.is_tool_call_response && parsed.content == "I need one more check." &&
                  parsed.tool_calls.size() == 1 && parsed.diagnostics.marker_seen &&
                  parsed.diagnostics.structured_call_count == 1 &&
                  parsed.diagnostics.empty_arguments_omitted == 1 &&
                  parsed.diagnostics.schema_mismatch_arguments == 0 &&
                  parsed.diagnostics.fallback_reason == ninferno::ToolCallParseFallbackReason::None,
              "empty optional boolean demoted a complete Edit call to text");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.size() == 3 && args.at("file_path") == "/tmp/probe.cpp" &&
                              args.at("new_string") == "std::map<std::uint32_t, int> counts;" &&
                              args.at("old_string") == "old line" && !args.contains("replace_all"),
                          "empty optional boolean was not omitted from Edit arguments");
    }

    bool every_split_matches = true;
    for (std::size_t split = 0; split <= text.size(); ++split) {
        fi::ToolCallOutputDecoder decoder(contract, 128);
        std::string visible = decoder.feed(std::string_view(text).substr(0, split));
        visible += decoder.feed(std::string_view(text).substr(split));
        auto terminal = decoder.finish();
        if (visible != "I need one more check." || !terminal.content.empty() ||
            terminal.tool_calls.size() != 1 ||
            terminal.tool_calls.front().arguments_json !=
                parsed.tool_calls.front().arguments_json ||
            terminal.diagnostics != parsed.diagnostics) {
            every_split_matches = false;
            break;
        }
    }
    failures += check(every_split_matches,
                      "incremental Edit parsing depends on the transport chunk boundary");

    fi::ToolCallOutputDecoder bytewise(contract, 128);
    std::string bytewise_visible;
    for (const char byte : text) { bytewise_visible += bytewise.feed(std::string_view(&byte, 1)); }
    auto bytewise_terminal = bytewise.finish();
    failures +=
        check(bytewise_visible == "I need one more check." && bytewise_terminal.content.empty() &&
                  bytewise_terminal.tool_calls.size() == 1 &&
                  bytewise_terminal.diagnostics == parsed.diagnostics,
              "bytewise Edit parsing changed the terminal tool-call semantics");

    const auto string_contract =
        contract_for("configure", Json{{"label", Json{{"type", "string"}}}});
    const auto empty_string = fi::parse_qwen_tool_call_output(
        tool_call("configure", {{"label", ""}}), 64, string_contract);
    failures +=
        check(empty_string.is_tool_call_response && empty_string.tool_calls.size() == 1 &&
                  Json::parse(empty_string.tool_calls.front().arguments_json).at("label") == "",
              "empty declared string was incorrectly omitted");
    return failures;
}

int test_schema_mismatches_remain_structured() {
    const auto contract =
        contract_for("configure", Json{{"integer_value", Json{{"type", "integer"}}},
                                       {"number_value", Json{{"type", "number"}}},
                                       {"boolean_value", Json{{"type", "boolean"}}},
                                       {"object_value", Json{{"type", "object"}}},
                                       {"array_value", Json{{"type", "array"}}},
                                       {"null_value", Json{{"type", "null"}}}});

    int failures = 0;
    failures += check_parameter_schema_mismatch(contract, "number_value", "\"1\"", "\"1\"",
                                                "string number mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "boolean_value", "[]", "[]",
                                                "array boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "object_value", "[]", "[]",
                                                "array object mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "array_value", "{}", "{}",
                                                "object array mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "null_value", "false", "false",
                                                "boolean null mismatch was not structured");
    failures += check_parameter_schema_mismatch(
        contract, "object_value", "{'x': True}", "\"{'x': True}\"",
        "Python object mismatch was not preserved for client validation");
    failures += check_parameter_schema_mismatch(
        contract, "array_value", "['a', None]", "\"['a', None]\"",
        "Python array mismatch was not preserved for client validation");
    return failures;
}

int test_unsupported_schema_uses_legacy_policy() {
    const auto contract = contract_for(
        "configure",
        Json{{"missing_type", Json::object()},
             {"alias", Json{{"type", "int"}}},
             {"invalid_type_array", Json{{"type", Json::array({"integer", "int"})}}},
             {"partial_anyof", Json{{"anyOf", Json::array({Json{{"type", "integer"}},
                                                           Json{{"enum", Json::array({1, 2})}}})}}},
             {"mixed_composition", Json{{"anyOf", Json::array({Json{{"type", "boolean"}}})},
                                        {"oneOf", Json::array({Json{{"type", "null"}}})}}}});
    const std::string text = tool_call("configure", {{"missing_type", "7"},
                                                     {"alias", "8"},
                                                     {"invalid_type_array", "9"},
                                                     {"partial_anyof", "7.5"},
                                                     {"mixed_composition", "True"},
                                                     {"undeclared", "{\"x\":1}"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                          parsed.diagnostics.schema_mismatch_arguments == 1,
                      "unsupported schema did not retain legacy policy");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("missing_type") == 7 && args.at("alias") == 8 &&
                          args.at("invalid_type_array") == 9,
                      "legacy numeric inference changed");
    failures += check(args.at("partial_anyof") == 7.5 && args.at("mixed_composition") == "True",
                      "unsupported composition was partially inferred");
    failures +=
        check(args.at("undeclared").at("x") == 1, "undeclared parameter legacy inference changed");
    return failures;
}

int test_strict_structure_and_active_tool_set() {
    const auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    int failures        = 0;

    const std::string malformed = "<tool_call>\n<function=configure>\n";
    failures += check_rejected(malformed, contract,
                               ninferno::ToolCallParseFallbackReason::MalformedStructure,
                               "missing structural tags were accepted");

    // Text after complete calls is content after them; the calls stay calls.
    const std::string suffix =
        "Setting it.\n" + tool_call("configure", {{"value", "x"}}) + "\nextra answer\n";
    const auto kept = fi::parse_qwen_tool_call_output(suffix, 64, contract);
    failures +=
        check(kept.is_tool_call_response && kept.tool_calls.size() == 1 &&
                  kept.content == "Setting it.\nextra answer" &&
                  kept.diagnostics.trailing_content_bytes == 12 &&
                  kept.diagnostics.fallback_reason == ninferno::ToolCallParseFallbackReason::None,
              "a call followed by text lost the call or the text");

    // The output ending inside the closers of a complete call (the turn ended there) keeps the
    // call; ending before the function's closer starts does not.
    for (const char* tail :
         {"</function>\n</tool_call", "</function>\n", "</function>", "</function", "</"}) {
        const std::string cut = "<tool_call>\n<function=configure>\n<parameter=value>\nx\n"
                                "</parameter>\n" +
                                std::string(tail);
        const auto parsed = fi::parse_qwen_tool_call_output(cut, 64, contract);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.content.empty() &&
                              parsed.tool_calls.front().arguments_json == "{\"value\":\"x\"}",
                          "a complete call cut in its closers was not kept");
    }
    const std::string cut_before_closer =
        "<tool_call>\n<function=configure>\n<parameter=value>\nx\n</parameter>\n<";
    failures += check_rejected(cut_before_closer, contract,
                               ninferno::ToolCallParseFallbackReason::MalformedStructure,
                               "a call cut before its function closer was accepted");

    const std::string missing_parameter_close =
        "<tool_call>\n<function=configure>\n<parameter=value>\nx\n"
        "</function>\n</tool_call>";
    failures += check_rejected(missing_parameter_close, contract,
                               ninferno::ToolCallParseFallbackReason::MalformedStructure,
                               "missing parameter close was repaired");

    // A repeated parameter keeps its last value, as a repeated JSON key would, and is counted.
    const std::string duplicate = tool_call("configure", {{"value", "first"}, {"value", "second"}});
    const auto repeated         = fi::parse_qwen_tool_call_output(duplicate, 64, contract);
    failures +=
        check(repeated.is_tool_call_response && repeated.tool_calls.size() == 1 &&
                  repeated.diagnostics.repeated_parameters == 1 &&
                  Json::parse(repeated.tool_calls.front().arguments_json).at("value") == "second",
              "a repeated parameter did not keep its last value");

    // An undeclared name is still a call, counted (the client answers an unknown tool); with a
    // declared call beside it, neither is lost.
    const std::string unknown_tool = tool_call("other", {{"value", "x"}});
    const auto unknown             = fi::parse_qwen_tool_call_output(unknown_tool, 64, contract);
    failures += check(
        unknown.is_tool_call_response && unknown.tool_calls.size() == 1 &&
            unknown.tool_calls.front().name == "other" &&
            Json::parse(unknown.tool_calls.front().arguments_json).at("value") == "x" &&
            unknown.diagnostics.undeclared_calls == 1 &&
            unknown.diagnostics.fallback_reason == ninferno::ToolCallParseFallbackReason::None,
        "an undeclared tool call was not published");
    const auto mixed = fi::parse_qwen_tool_call_output(
        tool_call("configure", {{"value", "y"}}) + "\n" + unknown_tool, 64, contract);
    failures +=
        check(mixed.tool_calls.size() == 2 && mixed.tool_calls[0].name == "configure" &&
                  mixed.tool_calls[1].name == "other" && mixed.diagnostics.undeclared_calls == 1,
              "an undeclared call dropped the declared call beside it");

    const std::string invalid_name = tool_call("bad name", {{"value", "x"}});
    failures += check_rejected(invalid_name, kLegacyContract,
                               ninferno::ToolCallParseFallbackReason::InvalidToolName,
                               "invalid function-name character was accepted");
    return failures;
}

// A generated call to an MCP-style name is a call, as its declaration is accepted.
int test_dotted_tool_name() {
    const auto parsed = fi::parse_qwen_tool_call_output(
        tool_call("github.create_issue", {{"title", "Bug"}}), 64,
        contract_for("github.create_issue", Json{{"title", Json{{"type", "string"}}}}));
    return check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                     parsed.tool_calls.front().name == "github.create_issue",
                 "a dotted tool name was not parsed as a call");
}

int test_name_limits_and_non_strict_omissions() {
    const std::string name(128, 'a');
    const std::string text          = tool_call(name);
    const auto anthropic            = fi::parse_qwen_tool_call_output(text, 128, kLegacyContract);
    const auto openai               = fi::parse_qwen_tool_call_output(text, 64, kLegacyContract);
    const std::string too_long_text = tool_call(std::string(129, 'a'));
    const auto too_long = fi::parse_qwen_tool_call_output(too_long_text, 128, kLegacyContract);

    int failures = 0;
    failures += check(anthropic.is_tool_call_response && anthropic.tool_calls.size() == 1,
                      "128-character Anthropic tool name was rejected");
    failures += check(!openai.is_tool_call_response, "128-character OpenAI tool name was accepted");
    failures +=
        check(!too_long.is_tool_call_response, "129-character Anthropic tool name was accepted");

    const std::string definition = tool_definition(
        "optional", Json{{"value", Json{{"type", "string"}}}}, Json::array({"value"}));
    const std::vector<std::string> definitions = {definition};
    const auto contract                        = contract_from_definitions(definitions);
    const auto omitted = fi::parse_qwen_tool_call_output(tool_call("optional"), 64, *contract);
    failures += check(omitted.is_tool_call_response && omitted.tool_calls.size() == 1 &&
                          omitted.tool_calls.front().arguments_json == "{}",
                      "non-strict parser enforced required parameters");
    return failures;
}

int test_conflicting_duplicate_tool_contracts_use_legacy_normalization() {
    const std::string integer_definition =
        tool_definition("configure", Json{{"value", Json{{"type", "integer"}}}});
    const std::string string_definition =
        tool_definition("configure", Json{{"value", Json{{"type", "string"}}}});

    const std::vector<std::string> identical_definitions = {integer_definition, integer_definition};
    const auto identical = contract_from_definitions(identical_definitions);
    const auto accepted =
        fi::parse_qwen_tool_call_output(tool_call("configure", {{"value", "7"}}), 64, *identical);

    const std::vector<std::string> conflicting_definitions = {integer_definition,
                                                              string_definition};
    const auto conflicting      = contract_from_definitions(conflicting_definitions);
    const std::string ambiguous = tool_call("configure", {{"value", "7"}});
    const auto ambiguous_parsed = fi::parse_qwen_tool_call_output(ambiguous, 64, *conflicting);

    int failures = 0;
    failures += check(accepted.is_tool_call_response && accepted.tool_calls.size() == 1,
                      "identical duplicate tool contracts became ambiguous");
    failures +=
        check(ambiguous_parsed.is_tool_call_response && ambiguous_parsed.tool_calls.size() == 1 &&
                  ambiguous_parsed.tool_calls.front().arguments_json == "{\"value\":7}" &&
                  ambiguous_parsed.diagnostics.schema_mismatch_arguments == 0,
              "conflicting duplicate tool contracts did not use legacy normalization");
    return failures;
}

int test_all_or_nothing_structural_commit() {
    const auto contract    = contract_for("configure", Json{{"flag", Json{{"type", "boolean"}}}});
    const std::string text = tool_call("configure", {{"flag", "true"}}) +
                             "\n<tool_call>\n<function=configure>\n<parameter=flag>\nfalse\n";
    return check_rejected(text, contract, ninferno::ToolCallParseFallbackReason::MalformedStructure,
                          "partially valid tool-call region was partially committed");
}

// The output limit cut the second call off: the first, complete call is kept and the unfinished
// markup dropped and counted. Without the cut the region stays text, as before; a cut region with
// no complete call before its last marker stays text too.
int test_cut_off_last_call() {
    int failures          = 0;
    const auto contract   = output_contract_for("search", Json{{"q", Json{{"type", "string"}}}});
    const std::string one = "<tool_call>\n<function=search>\n<parameter=q>\nfirst\n</parameter>\n"
                            "</function>\n</tool_call>";
    const std::string cut = "\n<tool_call>\n<function=search>\n<parameter=q>\nsec";
    for (const bool cut_off : {true, false}) {
        fi::ToolCallOutputDecoder decoder(contract, 64);
        const std::string visible = decoder.feed("Checking.\n" + one + cut);
        auto terminal             = decoder.finish(cut_off);
        if (cut_off) {
            failures += check(visible == "Checking." && terminal.content.empty() &&
                                  terminal.tool_calls.size() == 1 &&
                                  terminal.tool_calls[0].arguments_json == R"({"q":"first"})" &&
                                  terminal.diagnostics.cut_calls_dropped == 1,
                              "a call before the cut-off call was not kept");
        } else {
            failures += check(terminal.tool_calls.empty() &&
                                  visible + terminal.content == "Checking.\n" + one + cut &&
                                  terminal.diagnostics.cut_calls_dropped == 0,
                              "an uncut malformed region was not restored as text");
        }
    }
    // A complete last call followed by more text is not a cut call: the calls are kept and the
    // text is content after them.
    fi::ToolCallOutputDecoder trailing(contract, 64);
    const std::string complete_then_text = one + "\n" + one + "\nNow I";
    const std::string seen               = trailing.feed(complete_then_text);
    auto trailing_end                    = trailing.finish(true);
    failures += check(trailing_end.tool_calls.size() == 2 && seen.empty() &&
                          trailing_end.content == "Now I" &&
                          trailing_end.diagnostics.cut_calls_dropped == 0,
                      "a complete last call followed by text was dropped as a cut call");
    fi::ToolCallOutputDecoder lone(contract, 64);
    const std::string shown = lone.feed(cut.substr(1));
    auto lone_end           = lone.finish(true);
    failures += check(lone_end.tool_calls.empty() && shown + lone_end.content == cut.substr(1),
                      "a lone cut-off call was not restored as text");
    return failures;
}

int test_incremental_valid_and_boolean() {
    fi::ToolCallOutputDecoder legacy(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string visible;
    visible += legacy.feed("Calling weather.  \n<tool_");
    visible += legacy.feed("call>\n<function=get_weather>");
    visible += legacy.feed("\n</function>\n</tool_call>");
    auto legacy_terminal = legacy.finish();
    visible += legacy_terminal.content;

    auto bool_contract =
        output_contract_for("configure", Json{{"enabled", Json{{"type", "boolean"}}}});
    fi::ToolCallOutputDecoder boolean(std::move(bool_contract), 64);
    std::string boolean_visible;
    boolean_visible += boolean.feed("<tool_call>\n<function=configure>\n<parameter=enabled>\nT");
    boolean_visible += boolean.feed("r");
    boolean_visible += boolean.feed("ue\n</parameter>\n</function>\n</tool_call>");
    auto boolean_terminal = boolean.finish();

    int failures = 0;
    failures += check(visible == "Calling weather." && legacy_terminal.tool_calls.size() == 1,
                      "incremental valid call was not committed");
    failures += check(boolean_visible.empty() && boolean_terminal.content.empty() &&
                          boolean_terminal.tool_calls.size() == 1,
                      "incremental boolean call leaked as content");
    if (boolean_terminal.tool_calls.size() == 1) {
        const Json args = Json::parse(boolean_terminal.tool_calls.front().arguments_json);
        failures += check(args.at("enabled") == true,
                          "split case-insensitive boolean was not canonicalized");
    }
    return failures;
}

int test_incremental_fallback_preserves_bytes() {
    const std::string original = "prefix  \n<tool_call>\n<function=broken>";
    fi::ToolCallOutputDecoder malformed(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string restored;
    restored += malformed.feed(original.substr(0, 10));
    restored += malformed.feed(original.substr(10));
    auto malformed_terminal = malformed.finish();
    restored += malformed_terminal.content;

    fi::ToolCallOutputDecoder ordinary(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string ordinary_text;
    ordinary_text += ordinary.feed("ordinary text  ");
    ordinary_text += ordinary.finish().content;

    const std::string partial_original = "  <tool_x then <tool_";
    fi::ToolCallOutputDecoder partial(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string partial_restored;
    partial_restored += partial.feed("  <too");
    partial_restored += partial.feed("l_x then <tool_");
    partial_restored += partial.finish().content;

    int failures = 0;
    failures += check(restored == original && malformed_terminal.diagnostics.marker_seen &&
                          malformed_terminal.diagnostics.fallback_reason ==
                              ninferno::ToolCallParseFallbackReason::MalformedStructure,
                      "malformed incremental call lost raw bytes or fallback diagnostics");
    failures += check(ordinary_text == "ordinary text  ",
                      "ordinary incremental output lost trailing whitespace");
    failures +=
        check(partial_restored == partial_original, "partial marker mismatch lost raw bytes");
    return failures;
}

int test_incremental_embedded_parameter_markup() {
    auto contract = output_contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string command = "pattern='<parameter=inner>value</parameter>'\n"
                                "printf '%s' \"$pattern\"";
    const std::string text    = tool_call("bash", {{"command", command}});

    fi::ToolCallOutputDecoder decoder(std::move(contract), 64);
    std::string visible;
    constexpr std::size_t kChunk = 7;
    for (std::size_t offset = 0; offset < text.size(); offset += kChunk) {
        visible += decoder.feed(std::string_view(text).substr(offset, kChunk));
    }
    auto terminal = decoder.finish();

    int failures = 0;
    failures +=
        check(visible.empty() && terminal.content.empty() && terminal.tool_calls.size() == 1,
              "chunked embedded parameter markup was not committed as a tool call");
    if (terminal.tool_calls.size() == 1) {
        const Json args = Json::parse(terminal.tool_calls.front().arguments_json);
        failures += check(args.at("command") == command,
                          "chunked embedded parameter markup changed string bytes");
    }
    return failures;
}

} // namespace

// ---- Strict tool calls: the grammar and the parser accept one language -------------------------

ninferno::structured::GrammarVocabulary byte_vocabulary() {
    ninferno::structured::GrammarVocabulary out;
    out.token_domain = 258;
    out.token_bytes.resize(out.token_domain);
    out.special.assign(out.token_domain, false);
    for (int byte = 0; byte < 256; ++byte) {
        out.token_bytes[static_cast<std::size_t>(byte)] = std::string(1, static_cast<char>(byte));
    }
    out.token_bytes[256] = "<tool_call>";
    out.special[256]     = true;
    out.token_bytes[257] = "<|im_end|>";
    out.special[257]     = true;
    out.stop_token_ids   = {257};
    out.reasoning_close  = "</think>";
    return out;
}

std::vector<ninferno::TokenId> byte_tokens(std::string_view text) {
    std::vector<ninferno::TokenId> out;
    for (const char c : text) {
        out.push_back(static_cast<ninferno::TokenId>(static_cast<unsigned char>(c)));
    }
    return out;
}

// Commits `text` byte by byte, then the stop token. Returns whether the grammar accepted all of
// it as a complete output.
bool grammar_accepts(
    const std::shared_ptr<const ninferno::structured::CompiledConstraint>& compiled,
    std::string_view text) {
    ninferno::structured::RequestConstraint request(compiled, false);
    request.commit(byte_tokens(text));
    if (request.violated()) { return false; }
    std::vector<std::uint32_t> row(request.mask_words());
    request.fill_round({}, row.data(), row.size());
    if (((row[257U >> 5U] >> (257U & 31U)) & 1U) == 0U) { return false; }
    const ninferno::TokenId stop[] = {257};
    request.commit(stop);
    return request.stats().complete;
}

std::string xml_call(std::string_view name,
                     std::initializer_list<std::pair<std::string_view, std::string_view>> args) {
    // Rendered as the Qwen3.5 chat template renders a call in history.
    std::string text = "<tool_call>\n<function=";
    text.append(name);
    text += ">\n";
    for (const auto& [key, value] : args) {
        text += "<parameter=";
        text.append(key);
        text += ">\n";
        text.append(value);
        text += "\n</parameter>\n";
    }
    text += "</function>\n</tool_call>";
    return text;
}

int test_strict_tool_grammar_matches_parser() {
    using ninferno::ToolCallConstraintTool;
    using ninferno::ToolCallRequirement;
    int failures = 0;
    const Json weather_schema =
        Json::parse(R"({"type":"object","properties":{"city":{"type":"string"},
            "days":{"type":"integer","minimum":1,"maximum":14},
            "units":{"type":"string","enum":["c","f"]},
            "opts":{"type":"object","properties":{"hourly":{"type":"boolean"}},
                    "additionalProperties":false}},
            "required":["city","days"],"additionalProperties":false})");
    const std::vector<ToolCallConstraintTool> tools = {
        {.name = "weather", .parameters_json = weather_schema.dump(), .strict = true},
        {.name = "note", .parameters_json = R"({"type":"object"})", .strict = false},
    };
    const ninferno::structured::ConstraintCompiler compiler(byte_vocabulary());
    const auto compile = [&](ToolCallRequirement requirement, std::string_view named,
                             bool parallel) {
        ninferno::OutputConstraintSpec spec;
        spec.kind                   = ninferno::OutputConstraintKind::StructuralTag;
        spec.special_tokens_visible = true;
        spec.definition = fi::build_tool_call_structural_tag(tools, requirement, named, parallel);
        return compiler.compile(spec);
    };
    const auto optional = compile(ToolCallRequirement::Optional, "", true);
    const std::vector<std::string> definitions = {
        Json{{"type", "function"}, {"function", Json{{"name", "weather"}, {"parameters", weather_schema}}}}
            .dump(),
        Json{{"type", "function"},
             {"function", Json{{"name", "note"}, {"parameters", Json{{"type", "object"}}}}}}
            .dump()};
    const auto contract = contract_from_definitions(definitions);

    struct Case {
        std::string text;
        Json arguments; // expected parsed arguments of the first call
    };
    const std::vector<Case> accepted = {
        {xml_call("weather", {{"city", "Paris"}, {"days", "3"}}), Json{{"city", "Paris"}, {"days", 3}}},
        {xml_call("weather", {{"city", "Нью-Йорк 東京"}, {"days", "14"}, {"units", "f"}}),
         Json{{"city", "Нью-Йорк 東京"}, {"days", 14}, {"units", "f"}}},
        {xml_call("weather", {{"city", "a\n"}, {"days", "1"}}), Json{{"city", "a\n"}, {"days", 1}}},
        {xml_call("weather", {{"city", "x</param"}, {"days", "2"}}),
         Json{{"city", "x</param"}, {"days", 2}}},
        {xml_call("weather", {{"city", R"(f(){ return "}\"{"; })"}, {"days", "2"}}),
         Json{{"city", R"(f(){ return "}\"{"; })"}, {"days", 2}}},
        {xml_call("weather", {{"city", "Oslo"}, {"days", "2"}, {"opts", R"({"hourly":true})"}}),
         Json{{"city", "Oslo"}, {"days", 2}, {"opts", Json{{"hourly", true}}}}},
        {"Checking.\n\n" + xml_call("weather", {{"city", "Rome"}, {"days", "5"}}) + "\n" +
             xml_call("note", {{"text", "anything <b>goes</b>"}}),
         Json{{"city", "Rome"}, {"days", 5}}},
    };
    for (const Case& test_case : accepted) {
        failures += check(grammar_accepts(optional, test_case.text),
                          "strict tool grammar accepts: " + test_case.text);
        const auto parsed = fi::parse_qwen_tool_call_output(test_case.text, 64, *contract);
        failures += check(parsed.is_tool_call_response && !parsed.tool_calls.empty() &&
                              Json::parse(parsed.tool_calls.front().arguments_json) ==
                                  test_case.arguments,
                          "parser recovers the grammar's arguments: " + test_case.text);
    }

    const std::vector<std::string> rejected = {
        xml_call("weather", {{"city", "Paris"}, {"days", "three"}}),
        xml_call("weather", {{"city", "Paris"}, {"days", "0"}}),
        xml_call("weather", {{"city", "Paris"}}),
        xml_call("weather", {{"city", "Paris"}, {"days", "2"}, {"units", "k"}}),
        xml_call("weather", {{"city", "Paris"}, {"days", "2"}, {"extra", "1"}}),
        xml_call("unknown", {{"a", "1"}}),
        xml_call("weather", {{"city", "Paris"}, {"days", "2"}}) + "\ntrailing text",
    };
    for (const std::string& text : rejected) {
        failures += check(!grammar_accepts(optional, text), "strict tool grammar rejects: " + text);
    }
    failures += check(grammar_accepts(optional, "plain answer without a call"),
                      "optional tool calls admit plain text");

    const auto required = compile(ToolCallRequirement::Required, "", true);
    failures += check(!grammar_accepts(required, "plain answer without a call"),
                      "required tool call rejects plain text");
    failures += check(grammar_accepts(required, xml_call("note", {{"x", "y"}})),
                      "required tool call accepts a call");
    // Text before the call left a model with nothing to call writing until max_tokens.
    failures += check(!grammar_accepts(required, "Hello! " + xml_call("note", {{"x", "y"}})),
                      "required tool call rejects leading text");
    // A few bytes of whitespace may end the output after the calls, no more: a model that ranked
    // a forbidden token above its stop token after its call wrote newlines until max_tokens.
    failures += check(grammar_accepts(required, xml_call("note", {{"x", "y"}}) + "\n\n"),
                      "required tool call admits trailing whitespace");
    failures +=
        check(!grammar_accepts(required, xml_call("note", {{"x", "y"}}) + std::string(5, '\n')),
              "required tool call bounds trailing whitespace");

    const auto single = compile(ToolCallRequirement::Optional, "", false);
    failures += check(!grammar_accepts(single, xml_call("note", {{"x", "1"}}) + "\n" +
                                                   xml_call("note", {{"x", "2"}})),
                      "parallel_tool_calls=false admits one call");

    const auto named = compile(ToolCallRequirement::Named, "weather", true);
    failures += check(grammar_accepts(named, xml_call("weather", {{"city", "Lima"}, {"days", "4"}})),
                      "named tool call accepts that tool");
    failures += check(!grammar_accepts(named, xml_call("note", {{"x", "1"}})),
                      "named tool call rejects another tool");
    failures += check(!grammar_accepts(named, "text " + xml_call("weather", {{"city", "Lima"}, {"days", "4"}})),
                      "named tool call rejects leading text");
    return failures;
}

// Qwen XML calls written as bare <function=NAME> blocks are calls when every name is declared; a
// last call the stop token ended before its </function> is closed, one the output limit cut is not.
int test_bare_and_unclosed_qwen_calls() {
    const Json props    = Json{{"value", Json{{"type", "string"}}}};
    const auto contract = contract_for("configure", props);
    const auto shared   = output_contract_for("configure", props);
    const std::string block =
        "<function=configure>\n<parameter=value>\nx\n</parameter>\n</function>";
    int failures = 0;

    const auto bare = fi::parse_qwen_tool_call_output("Checking.\n" + block, 64, contract);
    failures += check(bare.is_tool_call_response && bare.tool_calls.size() == 1 &&
                          bare.content == "Checking." && bare.diagnostics.unwrapped_calls == 1,
                      "a bare declared <function=> block was not a call");
    const auto two = fi::parse_qwen_tool_call_output(block + "\n" + block, 64, contract);
    failures += check(two.tool_calls.size() == 2 && two.diagnostics.unwrapped_calls == 2,
                      "two bare blocks were not two calls");
    const std::string undeclared =
        "<function=launch>\n<parameter=value>\nx\n</parameter>\n</function>";
    const auto quoted = fi::parse_qwen_tool_call_output("Use " + undeclared, 64, contract);
    failures += check(!quoted.is_tool_call_response && quoted.content == "Use " + undeclared,
                      "a bare block naming no declared tool was taken");

    fi::ToolCallOutputDecoder streamed(shared, 64);
    const std::string visible = streamed.feed("Sure.\n" + block);
    auto streamed_end         = streamed.finish();
    // The newline before the marker is held with the region, as before a <tool_call>.
    failures += check(visible == "Sure." && streamed_end.tool_calls.size() == 1,
                      "a streamed bare block was published as text");

    const std::string unclosed = "<tool_call>\n<function=configure>\n<parameter=value>\nx\n"
                                 "</parameter>\n";
    fi::ToolCallOutputDecoder stopped(shared, 64);
    (void)stopped.feed(unclosed);
    auto stopped_end = stopped.finish(false);
    failures +=
        check(stopped_end.tool_calls.size() == 1 && stopped_end.diagnostics.closed_calls == 1,
              "a call the stop token ended before </function> was not closed");
    fi::ToolCallOutputDecoder cut(shared, 64);
    (void)cut.feed(unclosed);
    auto cut_end = cut.finish(true);
    failures += check(cut_end.tool_calls.empty() && cut_end.diagnostics.closed_calls == 0,
                      "a call the output limit cut before </function> was closed");
    return failures;
}

int main() {
    int failures = 0;
    failures += test_strict_tool_grammar_matches_parser();
    failures += test_basic_legacy_parsing();
    failures += test_multiple_calls();
    failures += test_json_argument_body();
    failures += test_spaced_names();
    failures += test_declared_strings_preserve_text();
    failures += test_string_values_preserve_embedded_tool_markup();
    failures += test_unrepresentable_parameter_delimiters_fall_back();
    failures += test_quoted_parameter_close_stays_in_value();
    failures += test_quoted_marker_before_real_call();
    failures += test_declared_json_types();
    failures += test_untyped_text_keeps_whitespace();
    failures += test_boolean_boundary();
    failures += test_exact_integer_boundary();
    failures += test_composed_schema_types();
    failures += test_local_reference_schema_types();
    failures += test_empty_declared_non_string_is_omitted();
    failures += test_schema_mismatches_remain_structured();
    failures += test_unsupported_schema_uses_legacy_policy();
    failures += test_strict_structure_and_active_tool_set();
    failures += test_bare_and_unclosed_qwen_calls();
    failures += test_name_limits_and_non_strict_omissions();
    failures += test_dotted_tool_name();
    failures += test_conflicting_duplicate_tool_contracts_use_legacy_normalization();
    failures += test_all_or_nothing_structural_commit();
    failures += test_cut_off_last_call();
    failures += test_incremental_valid_and_boolean();
    failures += test_incremental_fallback_preserves_bytes();
    failures += test_incremental_embedded_parameter_markup();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
