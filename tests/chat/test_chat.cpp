// The shared chat layer: conventions resolved from template markers, every tool-call syntax
// (parser, streaming decoder and structural tag agree on one language) and the generic
// tokenizer's boundary encoding and per-token bytes.

#include "chat/conventions.h"
#include "chat/tool_calls.h"
#include "structured/constraint.h"
#include "structured/request_constraint.h"
#include "text/hf_token_codec.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

using Json = nlohmann::json;
using ninferno::ChatRole;
using ninferno::ToolCallConstraintTool;
using ninferno::ToolCallRequirement;
using ninferno::chat::ToolCallFormat;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

std::string tool_definition(const std::string& name, const Json& parameters) {
    return Json{{"type", "function"},
                {"function", Json{{"name", name}, {"parameters", parameters}}}}
        .dump();
}

const Json kWeatherSchema = Json::parse(R"({"type":"object","properties":{
    "city":{"type":"string"},"days":{"type":"integer"}},"required":["city","days"],
    "additionalProperties":false})");

std::shared_ptr<const ninferno::chat::ToolCallOutputContract> contract(ToolCallFormat format) {
    const std::vector<std::string> tools = {tool_definition("weather", kWeatherSchema),
                                            tool_definition("note", Json{{"type", "object"}})};
    return ninferno::chat::build_tool_call_output_contract(tools, true, format);
}

int test_conventions() {
    int failures = 0;
    using ninferno::chat::resolve_conventions;
    const auto qwen = resolve_conventions(
        "<|im_start|>{{ m }}<|im_end|><think></think><tool_call>\n<function=x>");
    failures += check(qwen.turns && qwen.turns->name == "chatml", "chatml turns");
    failures += check(qwen.reasoning && qwen.reasoning->close == "</think>", "think reasoning");
    failures += check(qwen.tools.format == ToolCallFormat::QwenXml, "qwen xml tools");
    failures += check(qwen.tools.response_boundary == "\n</tool_response>", "qwen boundary");

    const auto hermes =
        resolve_conventions("<|im_start|>x<|im_end|><think></think><tool_call>{json}</tool_call>");
    failures += check(hermes.tools.format == ToolCallFormat::HermesJson, "hermes tools");

    const auto gemma = resolve_conventions(
        "<|turn>model\n<|channel>thought\n<channel|><turn|><|tool_call>call:<tool_call|>");
    failures += check(gemma.turns && gemma.turns->role("model") == ChatRole::Assistant,
                      "gemma model role");
    failures += check(gemma.reasoning && gemma.reasoning->open == "<|channel>thought" &&
                          gemma.reasoning->close == "<channel|>",
                      "gemma channel reasoning");
    failures += check(gemma.tools.format == ToolCallFormat::Gemma4, "gemma tools");

    const auto llama = resolve_conventions(
        "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>");
    failures += check(llama.turns && llama.turns->header_end == "<|end_header_id|>\n\n" &&
                          !llama.reasoning && llama.tools.format == ToolCallFormat::None,
                      "llama3 turns only");

    // Llama 3.1's own template writes a call as the reply: '{"name": "' + name + '", ' and
    // '"parameters": ' + arguments.
    const auto llama_tools = resolve_conventions(
        "<|start_header_id|>assistant<|end_header_id|>\n\n{{- '{\"name\": \"' + name + '\", ' "
        "}}{{- '\"parameters\": ' }}<|eot_id|>");
    failures += check(llama_tools.turns && llama_tools.turns->name == "llama3" &&
                          llama_tools.tools.format == ToolCallFormat::Llama3Json,
                      "llama3 json tools");
    // Qwen3 writes '<tool_call>\n{"name": "' too: the first matching format wins.
    const auto qwen3 = resolve_conventions(
        "<|im_start|>x<|im_end|><tool_call>\n{\"name\": \"x\", \"arguments\": ");
    failures += check(qwen3.tools.format == ToolCallFormat::HermesJson, "qwen3 stays hermes");

    // gpt-oss (harmony): role headers carry the channel, reasoning is the analysis channel and
    // the answer's final-channel header is dropped from the content.
    const auto harmony = resolve_conventions(
        "<|start|>system<|message|>x<|end|><|start|>assistant<|channel|>analysis<|message|>t"
        "<|end|><|start|>assistant<|channel|>final<|message|>a<|end|><|start|>assistant to="
        "{{ name }}<|channel|>commentary json<|message|>{}<|call|><|start|>functions.f to="
        "assistant<|channel|>commentary<|message|>r<|end|>");
    failures += check(harmony.turns && harmony.turns->name == "harmony" &&
                          harmony.turns->role("assistant<|channel|>final") == ChatRole::Assistant,
                      "harmony turns");
    failures += check(harmony.reasoning && harmony.reasoning->close == "<|end|>" &&
                          harmony.reasoning->content_open.size() == 2,
                      "harmony analysis reasoning");
    failures += check(harmony.tools.format == ToolCallFormat::Harmony, "harmony tools");

    // Generation stops come from the matched turn markup, the tool hand-off among them. A close
    // that ends one message of a reply (harmony's <|end|>) and the reasoning markers are inside a
    // reply, never stops.
    using Stops = std::vector<std::string>;
    failures +=
        check(harmony.generation_stops() == Stops{"<|return|>", "<|call|>", "<|endoftext|>"},
              "harmony stops");
    failures += check(harmony.within_reply("<|end|>") && !harmony.within_reply("<|return|>"),
                      "harmony <|end|> is within a reply");
    failures += check(gemma.generation_stops() == Stops{"<turn|>", "<|tool_response>", "<eos>"},
                      "gemma stops");
    failures += check(!gemma.within_reply("<turn|>") && gemma.within_reply("<channel|>"),
                      "gemma turn close ends the reply, the channel close does not");
    failures += check(qwen.generation_stops() == Stops{"<|im_end|>", "<|endoftext|>"} &&
                          qwen.within_reply("</think>") && !qwen.within_reply("<|im_end|>"),
                      "chatml stops");
    const auto unknown = resolve_conventions("{{ messages }}");
    failures += check(unknown.generation_stops().empty() && !unknown.within_reply("<|end|>") &&
                          !ninferno::chat::generic_generation_stops().empty(),
                      "an unknown template has no stops of its own");

    // OLMo 3: chatml turns (tool results as "environment"), Python-style calls.
    const auto olmo = resolve_conventions(
        "<|im_start|>environment\nr<|im_end|><function_calls>f(a=1)</function_calls>");
    failures += check(olmo.turns && olmo.turns->role("environment") == ChatRole::Tool &&
                          olmo.tools.format == ToolCallFormat::Pythonic,
                      "olmo pythonic tools");

    // Granite 3.x: <|tool_call|> then a JSON list (not Gemma 4's <|tool_call>).
    const auto granite = resolve_conventions(
        "<|start_of_role|>available_tools<|end_of_role|>respond only with <|tool_call|> followed "
        "by a JSON list");
    failures += check(granite.tools.format == ToolCallFormat::GraniteJson, "granite tools");

    const auto mistral = resolve_conventions("[INST]x[/INST][TOOL_CALLS]");
    failures += check(!mistral.turns && mistral.tools.format == ToolCallFormat::MistralJson,
                      "mistral tools without turn markup");

    // A start/message template whose assistant reasons in a message to itself (Muse Glimmer:
    // generation prompt "<|start|>assistant"; the model writes " to=self<|message|>", reasons,
    // ends that message with <|eom|> and answers in "<|start|>assistant to=user<|message|>").
    const auto muse = resolve_conventions(
        "<|start|>system<|message|>x<|eot|>{%- if r -%}<|start|>assistant to=self<|message|>"
        "{{ r }}<|eom|>{%- endif -%}<|start|>assistant to={{ recipient }}<|message|>{{ c }}<|eot|>");
    failures += check(muse.turns && muse.turns->name == "start_message" &&
                          muse.turns->header_end == "<|message|>" &&
                          muse.turns->close == "<|eot|>",
                      "start/message turns");
    failures += check(muse.reasoning && muse.reasoning->name == "self_recipient" &&
                          muse.reasoning->close == muse.reasoning->canonical_close,
                      "self-recipient reasoning");
    if (muse.reasoning) {
        const auto split = ninferno::chat::split_reasoning(
            " to=self<|message|>Two sentences. Rayleigh.<|eom|><|start|>assistant to=user"
            "<|message|>Air scatters blue light more.",
            *muse.reasoning);
        failures += check(split.opened && split.closed &&
                              split.reasoning == "Two sentences. Rayleigh." &&
                              split.content == "Air scatters blue light more.",
                          "self-recipient split");
        const auto open_only =
            ninferno::chat::split_reasoning(" to=self<|message|>Still thinking", *muse.reasoning);
        failures += check(open_only.opened && !open_only.closed &&
                              open_only.reasoning == "Still thinking" && open_only.content.empty(),
                          "unclosed reasoning keeps no answer");
        // An answer without reasoning continues the generation prompt's header.
        const auto direct =
            ninferno::chat::split_reasoning(" to=user<|message|>Blue light.", *muse.reasoning);
        failures += check(!direct.opened && direct.content == "Blue light.",
                          "self-recipient answer without reasoning drops its header");
        // A budget that ends inside the answer's header leaves no answer; plain text stays.
        const auto cut = ninferno::chat::split_reasoning(
            " to=self<|message|>R.<|eom|><|start|>assistant to=user", *muse.reasoning);
        failures += check(cut.closed && cut.reasoning == "R." && cut.content.empty(),
                          "answer header cut by the budget became content");
        const auto cut_direct = ninferno::chat::split_reasoning(" to=user", *muse.reasoning);
        failures += check(cut_direct.content.empty(), "direct answer header cut became content");
        failures +=
            check(!ninferno::chat::is_cut_content_header(" to", muse.reasoning->content_open),
                  "a plain-text header prefix was dropped");
        // Reasoning ends at <|eom|> whatever message follows: a call is addressed to its function.
        const auto then_call = ninferno::chat::split_reasoning(
            " to=self<|message|>Need weather.<|eom|><|start|>assistant to=weather<|message|>"
            "<atem:function_calls>",
            *muse.reasoning);
        failures += check(then_call.closed && then_call.reasoning == "Need weather." &&
                              then_call.content ==
                                  "<|start|>assistant to=weather<|message|><atem:function_calls>",
                          "self-recipient reasoning before a call");
    }
    // Muse Glimmer's ATEM tool calls: its template writes <atem:function_calls> blocks.
    const auto atem = resolve_conventions(
        "<|start|>assistant to=self<|message|>r<|eom|><|start|>assistant to={{ name }}<|message|>"
        "<atem:function_calls>\n<atem:invoke name=\"{{ name }}\">\n</atem:invoke>\n"
        "</atem:function_calls><|eot|>");
    failures += check(atem.tools.format == ToolCallFormat::Atem, "atem tools");
    if (qwen.reasoning) {
        const auto split =
            ninferno::chat::split_reasoning("<think>\nplan\n</think>\n\nanswer", *qwen.reasoning);
        failures += check(split.closed && split.reasoning == "plan\n" &&
                              split.content == "\n\nanswer",
                          "think split");
        const auto plain = ninferno::chat::split_reasoning("just an answer", *qwen.reasoning);
        failures += check(!plain.opened && plain.content == "just an answer", "no reasoning");
    }

    const auto markers = gemma.structural_markers();
    failures += check(std::find(markers.begin(), markers.end(), "<|\"|>") != markers.end(),
                      "gemma string quote is structural");
    return failures;
}

struct ParseCase {
    ToolCallFormat format;
    std::string text;
    bool structured;
    std::string content;
    std::vector<std::pair<std::string, Json>> calls;
};

int test_parsers() {
    int failures = 0;
    const std::vector<ParseCase> cases = {
        {ToolCallFormat::HermesJson,
         "Checking.\n<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Kyiv\", "
         "\"days\": 3}}\n</tool_call>",
         true,
         "Checking.",
         {{"weather", Json{{"city", "Kyiv"}, {"days", 3}}}}},
        {ToolCallFormat::HermesJson,
         "<tool_call>\n{\"name\": \"note\", \"arguments\": \"{\\\"a\\\": 1}\"}\n</tool_call>\n"
         "<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Lviv\", \"days\": 1}}"
         "\n</tool_call>",
         true,
         "",
         {{"note", Json{{"a", 1}}}, {"weather", Json{{"city", "Lviv"}, {"days", 1}}}}},
        {ToolCallFormat::HermesJson,
         "<tool_call>\n{\"name\": \"weather\", \"arguments\": {",
         false,
         "",
         {}},
        // An undeclared name is still a call: the client answers an unknown tool.
        {ToolCallFormat::HermesJson,
         "<tool_call>\n{\"name\": \"launch\", \"arguments\": {}}\n</tool_call>",
         true,
         "",
         {{"launch", Json::object()}}},
        {ToolCallFormat::Gemma4,
         "<|tool_call>call:weather{city:<|\"|>Kyiv, UA<|\"|>,days:3}<tool_call|>",
         true,
         "",
         {{"weather", Json{{"city", "Kyiv, UA"}, {"days", 3}}}}},
        {ToolCallFormat::Gemma4,
         "Let me note.<|tool_call>call:note{tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>],deep:{on:true,"
         "n:null},x:1.5}<tool_call|><|tool_call>call:weather{city:<|\"|>Rome<|\"|>,days:2}"
         "<tool_call|>",
         true,
         "Let me note.",
         {{"note", Json::parse(R"({"tags":["a","b"],"deep":{"on":true,"n":null},"x":1.5})")},
          {"weather", Json{{"city", "Rome"}, {"days", 2}}}}},
        {ToolCallFormat::Gemma4,
         "<|tool_call>call:weather{city:<|\"|>Kyiv}<tool_call|>",
         false,
         "",
         {}},
        // 4-bit Gemma 4 drops the call keyword and keeps its colon.
        {ToolCallFormat::Gemma4,
         "<|tool_call>:weather{city:<|\"|>Kyiv<|\"|>,days:3}<tool_call|>",
         true,
         "",
         {{"weather", Json{{"city", "Kyiv"}, {"days", 3}}}}},
        {ToolCallFormat::Gemma4,
         "<|tool_call>weather{city:<|\"|>Kyiv<|\"|>}<tool_call|>",
         false,
         "",
         {}},
        // A complete call whose turn ended in or before its closer is kept (Hermes, Gemma 4).
        {ToolCallFormat::HermesJson,
         "<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Kyiv\", \"days\": 3}}"
         "\n</tool",
         true,
         "",
         {{"weather", Json{{"city", "Kyiv"}, {"days", 3}}}}},
        {ToolCallFormat::HermesJson,
         "Checking.\n<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Kyiv\", "
         "\"days\": 3}}\n",
         true,
         "Checking.",
         {{"weather", Json{{"city", "Kyiv"}, {"days", 3}}}}},
        {ToolCallFormat::HermesJson,
         "<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Kyiv\", \"days\": 3}}"
         "\nok",
         false,
         "",
         {}},
        {ToolCallFormat::Gemma4,
         "<|tool_call>call:weather{city:<|\"|>Kyiv<|\"|>,days:3}",
         true,
         "",
         {{"weather", Json{{"city", "Kyiv"}, {"days", 3}}}}},
        {ToolCallFormat::Gemma4,
         "<|tool_call>call:weather{city:<|\"|>Kyiv<|\"|>,days:3} done",
         false,
         "",
         {}},
        {ToolCallFormat::MistralJson,
         "[TOOL_CALLS]weather[ARGS]{\"city\": \"Oslo\", \"days\": 5}",
         true,
         "",
         {{"weather", Json{{"city", "Oslo"}, {"days", 5}}}}},
        {ToolCallFormat::MistralJson,
         "[TOOL_CALLS][{\"name\": \"note\", \"arguments\": {\"t\": \"x\"}}]",
         true,
         "",
         {{"note", Json{{"t", "x"}}}}},
        {ToolCallFormat::Llama3Json,
         "{\"name\": \"weather\", \"parameters\": {\"city\": \"Kyiv\", \"days\": 3}}",
         true,
         "",
         {{"weather", Json{{"city", "Kyiv"}, {"days", 3}}}}},
        {ToolCallFormat::Llama3Json,
         "Sure.\n{\"name\": \"note\", \"parameters\": {\"t\": \"}\"}}; {\"name\": \"weather\", "
         "\"parameters\": {\"city\": \"Rome\", \"days\": 2}}",
         true,
         "Sure.",
         {{"note", Json{{"t", "}"}}}, {"weather", Json{{"city", "Rome"}, {"days", 2}}}}},
        {ToolCallFormat::Llama3Json,
         "<|python_tag|>{\"name\": \"weather\", \"parameters\": {\"city\": \"Kyiv\"}}",
         true,
         "",
         {{"weather", Json{{"city", "Kyiv"}}}}},
        {ToolCallFormat::Llama3Json,
         "{\"name\": \"weather\", \"parameters\": {\"city\": ",
         false,
         "",
         {}},
        // Llama 3's call opens like any JSON answer: only a declared name makes it a call.
        {ToolCallFormat::Llama3Json, "{\"name\": \"launch\", \"parameters\": {}}", false, "", {}},
        {ToolCallFormat::Harmony,
         "<|start|>assistant<|channel|>commentary to=functions.weather <|constrain|>json"
         "<|message|>{\"city\": \"Kyiv\", \"days\": 3}",
         true,
         "",
         {{"weather", Json{{"city", "Kyiv"}, {"days", 3}}}}},
        {ToolCallFormat::Harmony,
         "<|start|>assistant to=functions.note<|channel|>commentary json<|message|>{\"t\": \"x\"}"
         "<|call|>",
         true,
         "",
         {{"note", Json{{"t", "x"}}}}},
        {ToolCallFormat::Harmony,
         "<|start|>assistant<|channel|>commentary to=functions.weather <|constrain|>json"
         "<|message|>{\"city\": ",
         false,
         "",
         {}},
        {ToolCallFormat::Harmony,
         "<|start|>assistant<|channel|>final<|message|>{\"city\": \"Kyiv\"}",
         false,
         "",
         {}},
        {ToolCallFormat::Pythonic,
         "Let me check.<function_calls>weather(city=\"Kyiv, UA\", days=3)\nnote(t='x y', "
         "deep={\"on\": true}, flag=True, none=None)</function_calls>",
         true,
         "Let me check.",
         {{"weather", Json{{"city", "Kyiv, UA"}, {"days", 3}}},
          {"note", Json::parse(R"({"t":"x y","deep":{"on":true},"flag":true,"none":null})")}}},
        {ToolCallFormat::Pythonic,
         "<function_calls>weather(city=\"Rome\", days=2)",
         true,
         "",
         {{"weather", Json{{"city", "Rome"}, {"days", 2}}}}},
        {ToolCallFormat::Pythonic,
         "<function_calls>weather(city=\"Rome\", days=)</function_calls>",
         false,
         "",
         {}},
        // A positional argument has no name to give it: not a call (was the key "\"Paris\",
        // units").
        {ToolCallFormat::Pythonic,
         "<function_calls>weather(\"Paris\", units=\"c\")</function_calls>",
         false,
         "",
         {}},
        // The close marker and a parenthesis inside string arguments stay in them.
        {ToolCallFormat::Pythonic,
         "<function_calls>note(t=\"see </function_calls> (x)\")\nnote(t='a b',)</function_calls>",
         true,
         "",
         {{"note", Json{{"t", "see </function_calls> (x)"}}}, {"note", Json{{"t", "a b"}}}}},
        {ToolCallFormat::Pythonic,
         "<function_calls>note()</function_calls>",
         true,
         "",
         {{"note", Json::object()}}},
        {ToolCallFormat::HermesJson,
         "<tool_call>\n{\"name\": \"note\", \"arguments\": {\"t\": \"ends with </tool_call> "
         "here\"}}"
         "\n</tool_call>",
         true,
         "",
         {{"note", Json{{"t", "ends with </tool_call> here"}}}}},
        {ToolCallFormat::GraniteJson,
         "<|tool_call|>[{\"name\": \"weather\", \"arguments\": {\"city\": \"Oslo\", \"days\": 5}}, "
         "{\"name\": \"note\", \"arguments\": {\"t\": \"]\"}}]",
         true,
         "",
         {{"weather", Json{{"city", "Oslo"}, {"days", 5}}}, {"note", Json{{"t", "]"}}}}},
        {ToolCallFormat::GraniteJson,
         "Checking.<|tool_call|>[{\"name\": \"weather\", \"arguments\": {\"city\": \"Kyiv\"}}]"
         "<|tool_call|>[{\"name\": \"note\", \"arguments\": {}}]",
         true,
         "Checking.",
         {{"weather", Json{{"city", "Kyiv"}}}, {"note", Json::object()}}},
        {ToolCallFormat::GraniteJson,
         "<|tool_call|>{\"name\": \"weather\", \"arguments\": {}}",
         false,
         "",
         {}},
        {ToolCallFormat::GraniteJson,
         "<|tool_call|>[{\"name\": \"weather\", \"arguments\": {",
         false,
         "",
         {}},
        // ATEM: a call right after the generation prompt continues its header.
        {ToolCallFormat::Atem,
         " to=weather<|message|><atem:function_calls>\n<atem:invoke name=\"weather\">\n"
         "<atem:parameter name=\"city\">Kyiv, UA</atem:parameter>\n"
         "<atem:parameter name=\"days\">3</atem:parameter>\n</atem:invoke>\n"
         "</atem:function_calls>",
         true,
         "",
         {{"weather", Json{{"city", "Kyiv, UA"}, {"days", 3}}}}},
        // Two calls, one message each; strings as is, objects as JSON.
        {ToolCallFormat::Atem,
         "Checking.<|start|>assistant to=note<|message|><atem:function_calls>\n"
         "<atem:invoke name=\"note\">\n<atem:parameter name=\"t\"> x y \n</atem:parameter>\n"
         "<atem:parameter name=\"deep\">{\"on\": true}</atem:parameter>\n</atem:invoke>\n"
         "</atem:function_calls><|eom|><|start|>assistant to=weather<|message|>"
         "<atem:function_calls>\n<atem:invoke name=\"weather\">\n"
         "<atem:parameter name=\"city\"> Rome </atem:parameter>\n"
         "<atem:parameter name=\"days\"> 2 </atem:parameter>\n</atem:invoke>\n"
         "</atem:function_calls>",
         true,
         "Checking.",
         {{"note", Json::parse(R"({"t":" x y \n","deep":{"on":true}})")},
          {"weather", Json{{"city", " Rome "}, {"days", 2}}}}},
        // A parameter named "" (a schema may declare that key) is kept.
        {ToolCallFormat::Atem,
         " to=note<|message|><atem:function_calls>\n<atem:invoke name=\"note\">\n"
         "<atem:parameter name=\"\">x</atem:parameter>\n</atem:invoke>\n</atem:function_calls>",
         true,
         "",
         {{"note", Json{{"", "x"}}}}},
        // The recipient must be the invoked function.
        {ToolCallFormat::Atem,
         " to=note<|message|><atem:function_calls>\n<atem:invoke name=\"weather\">\n"
         "</atem:invoke>\n</atem:function_calls>",
         false,
         "",
         {}},
        // Cut off before the block closes.
        {ToolCallFormat::Atem,
         " to=weather<|message|><atem:function_calls>\n<atem:invoke name=\"weather\">\n"
         "<atem:parameter name=\"city\">Ky",
         false,
         "",
         {}},
        // A block without its message header.
        {ToolCallFormat::Atem,
         "<atem:function_calls>\n<atem:invoke name=\"weather\">\n</atem:invoke>\n"
         "</atem:function_calls>",
         false,
         "",
         {}},
    };
    for (std::size_t index = 0; index < cases.size(); ++index) {
        const ParseCase& item = cases[index];
        const std::string label =
            std::string(ninferno::chat::tool_call_format_name(item.format)) + " case " +
            std::to_string(index);
        const auto parsed = ninferno::chat::tool_call_syntax(item.format)
                                .parse(item.text, 64, *contract(item.format));
        failures += check(parsed.is_tool_call_response == item.structured, label + " structured");
        if (!item.structured) {
            failures += check(parsed.content == item.text, label + " restores the text");
            continue;
        }
        failures += check(parsed.content == item.content, label + " content");
        failures += check(parsed.tool_calls.size() == item.calls.size(), label + " call count");
        for (std::size_t call = 0; call < std::min(parsed.tool_calls.size(), item.calls.size());
             ++call) {
            failures += check(parsed.tool_calls[call].name == item.calls[call].first,
                              label + " name");
            failures += check(Json::parse(parsed.tool_calls[call].arguments_json) ==
                                  item.calls[call].second,
                              label + " arguments " + parsed.tool_calls[call].arguments_json);
        }
    }
    return failures;
}

// The streaming decoder publishes the text before the call and keeps the call structural, one
// byte at a time.
int test_streaming_decoder() {
    int failures = 0;
    ninferno::chat::ToolCallOutputDecoder decoder(contract(ToolCallFormat::Gemma4), 64);
    const std::string text =
        "Sure.\n<|tool_call>call:weather{city:<|\"|>Oslo<|\"|>,days:1}<tool_call|>";
    std::string visible;
    for (const char byte : text) { visible += decoder.feed(std::string_view(&byte, 1)); }
    auto terminal = decoder.finish();
    failures += check(visible == "Sure.", "gemma decoder visible text: " + visible);
    failures += check(terminal.content.empty() && terminal.tool_calls.size() == 1 &&
                          terminal.tool_calls[0].name == "weather",
                      "gemma decoder structured call");

    // The output limit cut the second Gemma call off: the first is kept, the cut one dropped.
    ninferno::chat::ToolCallOutputDecoder cut(contract(ToolCallFormat::Gemma4), 64);
    (void)cut.feed("<|tool_call>call:weather{city:<|\"|>Oslo<|\"|>,days:1}<tool_call|>"
                   "<|tool_call>call:weather{city:<|\"|>Ber");
    const auto cut_end = cut.finish(true);
    failures += check(cut_end.tool_calls.size() == 1 && cut_end.content.empty() &&
                          cut_end.diagnostics.cut_calls_dropped == 1,
                      "gemma decoder lost the call before a cut-off call");

    ninferno::chat::ToolCallOutputDecoder plain(contract(ToolCallFormat::Gemma4), 64);
    const std::string prose = "a <|tool b";
    std::string shown;
    for (const char byte : prose) { shown += plain.feed(std::string_view(&byte, 1)); }
    auto end = plain.finish();
    failures += check(shown + end.content == prose && end.tool_calls.empty(),
                      "gemma decoder restores a broken marker");

    // Llama 3.1: <|python_tag|> before the JSON call belongs to the call; without a call after it
    // it is content, restored verbatim.
    for (const auto& [feed_text, want_visible, want_calls] :
         std::vector<std::tuple<std::string, std::string, std::size_t>>{
             {"Sure.\n<|python_tag|>{\"name\": \"weather\", \"parameters\": {\"city\": \"Oslo\"}}",
              "Sure.", 1},
             {"<|python_tag|> {\"name\": \"weather\", \"parameters\": {}}", "", 1},
             {"a <|python_tag|>b <|pyt{\"name\": \"weather\", \"parameters\": {}}", "a <|python_tag|>b <|pyt", 1},
             {"tag <|python_tag|>", "tag", 0},
         }) {
        ninferno::chat::ToolCallOutputDecoder llama(contract(ToolCallFormat::Llama3Json), 64);
        std::string seen;
        for (const char byte : feed_text) { seen += llama.feed(std::string_view(&byte, 1)); }
        auto done = llama.finish();
        failures += check(seen == want_visible && done.tool_calls.size() == want_calls &&
                              (want_calls != 0 || seen + done.content == feed_text) &&
                              (want_calls == 0 || done.content.empty()),
                          "llama decoder on " + feed_text + ": visible " + seen + " | " +
                              done.content);
    }
    // ATEM: the call region begins at its message header, which names the function; prose that
    // merely contains "to=" is published.
    const std::string atem_call =
        "<atem:function_calls>\n<atem:invoke name=\"weather\">\n"
        "<atem:parameter name=\"city\">Oslo</atem:parameter>\n</atem:invoke>\n"
        "</atem:function_calls>";
    for (const auto& [feed_text, want_visible, want_calls] :
         std::vector<std::tuple<std::string, std::string, std::size_t>>{
             {" to=weather<|message|>" + atem_call, "", 1},
             {"Set x to=5.\n<|start|>assistant to=weather<|message|>" + atem_call, "Set x to=5.",
              1},
             {"Go to=town<|message|>no call", "Go to=town<|message|>no call", 0},
         }) {
        ninferno::chat::ToolCallOutputDecoder atem(contract(ToolCallFormat::Atem), 64);
        std::string seen;
        for (const char byte : feed_text) { seen += atem.feed(std::string_view(&byte, 1)); }
        auto done = atem.finish();
        const std::string shown = want_calls == 0 ? seen + done.content : seen;
        failures += check(shown == want_visible && done.tool_calls.size() == want_calls &&
                              (want_calls == 0 || done.content.empty()),
                          "atem decoder on " + feed_text + ": visible " + seen + " | " +
                              done.content);
    }
    return failures;
}

ninferno::structured::GrammarVocabulary byte_vocabulary() {
    ninferno::structured::GrammarVocabulary out;
    out.token_domain = 257;
    out.token_bytes.resize(out.token_domain);
    out.special.assign(out.token_domain, false);
    for (int byte = 0; byte < 256; ++byte) {
        out.token_bytes[static_cast<std::size_t>(byte)] = std::string(1, static_cast<char>(byte));
    }
    out.token_bytes[256] = "<eos>";
    out.special[256]     = true;
    out.stop_token_ids   = {256};
    return out;
}

bool grammar_accepts(
    const std::shared_ptr<const ninferno::structured::CompiledConstraint>& compiled,
    std::string_view text) {
    ninferno::structured::RequestConstraint request(compiled, false);
    std::vector<ninferno::TokenId> tokens;
    for (const char c : text) { tokens.push_back(static_cast<unsigned char>(c)); }
    request.commit(tokens);
    if (request.violated()) { return false; }
    std::vector<std::uint32_t> row(request.mask_words());
    request.fill_round({}, row.data(), row.size());
    if (((row[256U >> 5U] >> (256U & 31U)) & 1U) == 0U) { return false; }
    const ninferno::TokenId stop[] = {256};
    request.commit(stop);
    return request.stats().complete;
}

// What a syntax's grammar admits, its parser turns into calls.
int test_structural_tags() {
    int failures = 0;
    const std::vector<ToolCallConstraintTool> tools = {
        {.name = "weather", .parameters_json = kWeatherSchema.dump(), .strict = true},
        {.name = "note", .parameters_json = R"({"type":"object"})", .strict = false},
    };
    const ninferno::structured::ConstraintCompiler compiler(byte_vocabulary());
    struct Sample {
        ToolCallFormat format;
        std::string call;     // a weather call the grammar must admit
        std::string invalid;  // a weather call the strict schema must refuse
    };

    const std::vector<Sample> samples = {
        {ToolCallFormat::QwenXml,
         "<tool_call>\n<function=weather>\n<parameter=city>\nOslo\n</parameter>\n"
         "<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>",
         "<tool_call>\n<function=weather>\n<parameter=city>\nOslo\n</parameter>\n"
         "<parameter=days>\ntwo\n</parameter>\n</function>\n</tool_call>"},
        {ToolCallFormat::HermesJson,
         "<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Oslo\", \"days\": 2}}\n"
         "</tool_call>",
         "<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": 2, \"days\": 2}}\n"
         "</tool_call>"},
        {ToolCallFormat::QwenXml,
         "<tool_call>\n<function=weather>\n<parameter=city>\nOslo\n</parameter>\n"
         "<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>",
         ""},
        {ToolCallFormat::Gemma4,
         "<|tool_call>call:weather{city:<|\"|>Oslo<|\"|>,days:2}<tool_call|>",
         "<|tool_call>call:weather{city:<|\"|>Oslo<|\"|>,days:<|\"|>two<|\"|>}<tool_call|>"},
        {ToolCallFormat::MistralJson, "[TOOL_CALLS]weather[ARGS]{\"city\": \"Oslo\", \"days\": 2}",
         "[TOOL_CALLS]weather[ARGS]{\"city\": 2, \"days\": 2}"},
        {ToolCallFormat::Llama3Json,
         "{\"name\": \"weather\", \"parameters\": {\"city\": \"Oslo\", \"days\": 2}}",
         "{\"name\": \"weather\", \"parameters\": {\"city\": 2, \"days\": 2}}"},
        {ToolCallFormat::Harmony,
         "<|start|>assistant<|channel|>commentary to=functions.weather <|constrain|>json"
         "<|message|>{\"city\": \"Oslo\", \"days\": 2}<|call|>",
         "<|start|>assistant<|channel|>commentary to=functions.weather <|constrain|>json"
         "<|message|>{\"city\": 2, \"days\": 2}<|call|>"},
        {ToolCallFormat::Pythonic,
         "<function_calls>weather(city=\"Oslo\", days=2)</function_calls>",
         "<function_calls>weather(city=\"Oslo\", days=\"two\")</function_calls>"},
        {ToolCallFormat::GraniteJson,
         "<|tool_call|>[{\"name\": \"weather\", \"arguments\": {\"city\": \"Oslo\", \"days\": 2}}]",
         "<|tool_call|>[{\"name\": \"weather\", \"arguments\": {\"city\": 2, \"days\": 2}}]"},
        {ToolCallFormat::Atem,
         " to=weather<|message|><atem:function_calls>\n<atem:invoke name=\"weather\">\n"
         "<atem:parameter name=\"city\">Oslo</atem:parameter>\n"
         "<atem:parameter name=\"days\">2</atem:parameter>\n</atem:invoke>\n"
         "</atem:function_calls>",
         " to=weather<|message|><atem:function_calls>\n<atem:invoke name=\"weather\">\n"
         "<atem:parameter name=\"city\">Oslo</atem:parameter>\n"
         "<atem:parameter name=\"days\">two</atem:parameter>\n</atem:invoke>\n"
         "</atem:function_calls>"},
    };
    for (const Sample& sample : samples) {
        const std::string name(ninferno::chat::tool_call_format_name(sample.format));
        const auto& syntax = ninferno::chat::tool_call_syntax(sample.format);
        for (const ToolCallRequirement requirement :
             {ToolCallRequirement::Optional, ToolCallRequirement::Required,
              ToolCallRequirement::Named}) {
            ninferno::OutputConstraintSpec spec;
            spec.kind                   = ninferno::OutputConstraintKind::StructuralTag;
            spec.special_tokens_visible = true;
            spec.definition = syntax.structural_tag(tools, requirement, "weather", true);
            std::shared_ptr<const ninferno::structured::CompiledConstraint> compiled;
            try {
                compiled = compiler.compile(spec);
            } catch (const std::exception& error) {
                failures += check(false, name + " structural tag compiles: " + error.what());
                continue;
            }
            const std::string text =
                requirement == ToolCallRequirement::Optional ? "Hi. " + sample.call : sample.call;
            failures += check(grammar_accepts(compiled, text), name + " grammar admits a call");
            if (requirement == ToolCallRequirement::Required) {
                failures += check(!grammar_accepts(compiled, "Hi. " + sample.call),
                                  name + " required grammar rejects text before the call");
            }
            if (!sample.invalid.empty()) {
                failures += check(!grammar_accepts(compiled, sample.invalid),
                                  name + " grammar enforces the strict schema");
            }
            const auto parsed = syntax.parse(text, 64, *contract(sample.format));
            failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                              name + " parser reads what the grammar admits");
            if (requirement != ToolCallRequirement::Named) {
                // A newline after the last call ends the output; it does not open another call.
                failures += check(grammar_accepts(compiled, text + "\n"),
                                  name + " grammar ends on whitespace after the calls");
                // A few bytes of it, no more: past them only the stop token remains.
                failures += check(!grammar_accepts(compiled, text + std::string(5, '\n')),
                                  name + " grammar bounds the whitespace after the calls");
            }
            if (sample.format == ToolCallFormat::Gemma4) {
                // Strict arguments refuse an undeclared key; the checks below are of a non-strict
                // tool's free arguments.
                failures += check(!grammar_accepts(compiled,
                                                   "<|tool_call>call:weather{city:<|\"|>Oslo<|\"|>,"
                                                   "days:2,tags:[3]}<tool_call|>"),
                                  name + " strict grammar refuses an undeclared argument");
                const std::vector<ToolCallConstraintTool> free_tools = {
                    {.name = "weather", .parameters_json = kWeatherSchema.dump(), .strict = false}};
                spec.definition = syntax.structural_tag(free_tools, requirement, "weather", true);
                compiled        = compiler.compile(spec);
                // Gemma's free-text arguments never hold a turn or channel marker.
                failures += check(!grammar_accepts(compiled,
                                                   "<|tool_call>call:weather{city:<|\"|>Oslo<|\"|>}"
                                                   "<|channel>thought\n<channel|>Hi}<tool_call|>"),
                                  name + " grammar rejects a channel inside the arguments");
                // Nor text past the arguments' brace (the same model wrote "<|thought" there).
                failures += check(!grammar_accepts(compiled,
                                                   "<|tool_call>call:weather{city:<|\"|>Oslo<|\"|>}"
                                                   "<|thought}<tool_call|>"),
                                  name + " grammar rejects text after the arguments");
                failures += check(grammar_accepts(compiled,
                                                  "<|tool_call>call:weather{city:<|\"|>Oslo<|\"|>,"
                                                  "days:2,tags:[<|\"|>a<|\"|>,3],"
                                                  "at:{lat:1.5,ok:true}}<tool_call|>"),
                                  name + " grammar admits nested arguments");
                // Whitespace runs inside the arguments are bounded like JSON's.
                const auto spaced = [](std::size_t run) {
                    return "<|tool_call>call:weather{" + std::string(run, ' ') +
                           "city:<|\"|>Oslo<|\"|>,days:2}<tool_call|>";
                };
                failures += check(grammar_accepts(compiled, spaced(64)) &&
                                      !grammar_accepts(compiled, spaced(65)),
                                  name + " grammar bounds whitespace inside the arguments");
            }
        }
    }
    // A forced call of a non-strict Qwen tool keeps the parameter syntax: declared names of any
    // spelling, in any order, with any value (lines, markup, JSON) or none, and other names are
    // admitted; a body the parser cannot read (JSON in place of parameters) is not.
    const Json search_schema = Json::parse(R"({"type":"object","properties":{
        "max-results":{"type":"integer"},"запрос":{"type":"string"}},"required":["запрос"]})");
    const std::vector<ToolCallConstraintTool> loose_tools = {
        {.name = "search", .parameters_json = search_schema.dump(), .strict = false}};
    const std::vector<std::string> loose_definitions = {tool_definition("search", search_schema)};
    const auto loose_contract = ninferno::chat::build_tool_call_output_contract(
        loose_definitions, true, ToolCallFormat::QwenXml);
    const auto& qwen = ninferno::chat::tool_call_syntax(ToolCallFormat::QwenXml);
    ninferno::OutputConstraintSpec loose;
    loose.kind                   = ninferno::OutputConstraintKind::StructuralTag;
    loose.special_tokens_visible = true;
    loose.definition  = qwen.structural_tag(loose_tools, ToolCallRequirement::Required, "", true);
    const auto search = compiler.compile(loose);
    const auto call   = [](std::string_view body) {
        return "<tool_call>\n<function=search>\n" + std::string(body) +
               "\n</function>\n</tool_call>";
    };
    for (const std::string& admitted :
         {call("<parameter=запрос>\ntwo lines,\n<b>markup</b> and {code}\n</parameter>\n"
               "<parameter=max-results>\n5\n</parameter>"),
          call("<parameter=max-results>\n[1, 2]\n</parameter>\n<parameter=extra_key>\nx\n"
               "</parameter>"),
          call("")}) {
        failures +=
            check(grammar_accepts(search, admitted), "qwen non-strict call admits: " + admitted);
        failures += check(qwen.parse(admitted, 64, *loose_contract).tool_calls.size() == 1,
                          "qwen parser reads the non-strict call: " + admitted);
    }
    failures += check(!grammar_accepts(search, call("{\"запрос\": \"x\"}")),
                      "qwen non-strict call refuses a body the parser cannot read");
    // A strict tool declaring a name <parameter=NAME> cannot carry is refused up front.
    for (const char* name : {"", "a>b", "two\nlines"}) {
        Json schema = Json::parse(R"({"type":"object","properties":{}})");
        schema["properties"][name] = Json{{"type", "string"}};
        const std::vector<ToolCallConstraintTool> strict_tools = {
            {.name = "search", .parameters_json = schema.dump(), .strict = true}};
        bool refused = false;
        try {
            (void)qwen.structural_tag(strict_tools, ToolCallRequirement::Required, "", true);
        } catch (const std::invalid_argument&) { refused = true; }
        failures += check(refused, std::string("qwen strict tool with parameter name '") + name +
                                       "' is refused");
    }
    return failures;
}

// Optional arguments in the order the prompt asks for them, not the schema's (the "schema-order"
// bug class: a grammar that compiles optional properties in schema order with each skippable
// makes every earlier optional key unwritable once a later one is written, so the call stays valid
// JSON while fields silently vanish). Every syntax, strict or not, forced or named, admits the
// five keys in any order, and its parser reads all five back. A strict tool's arguments hold each
// key at most once and no undeclared key.
int test_argument_key_order() {
    int failures = 0;
    const Json schema = Json::parse(R"({"type":"object","properties":{
        "A":{"type":"string"},"B":{"type":"string"},"C":{"type":"string"},
        "D":{"type":"string"},"E":{"type":"string"}},"additionalProperties":false})");
    const std::vector<std::string> order = {"C", "A", "E", "B", "D"};
    const auto json_args = [](const std::vector<std::string>& keys) {
        std::string out = "{";
        for (std::size_t i = 0; i < keys.size(); ++i) {
            out += (i == 0 ? "\"" : ", \"") + keys[i] + "\": \"v" + keys[i] + "\"";
        }
        return out + "}";
    };
    using Render = std::function<std::string(const std::vector<std::string>&)>;
    struct Case {
        ToolCallFormat format;
        Render call;
        bool shaped; // the syntax holds a strict tool's arguments to its schema
    };
    const std::vector<Case> cases = {
        {ToolCallFormat::QwenXml,
         [](const std::vector<std::string>& keys) {
             std::string body;
             for (const std::string& key : keys) {
                 body += "<parameter=" + key + ">\nv" + key + "\n</parameter>\n";
             }
             return "<tool_call>\n<function=fill>\n" + body + "</function>\n</tool_call>";
         },
         true},
        {ToolCallFormat::HermesJson,
         [&](const std::vector<std::string>& keys) {
             return "<tool_call>\n{\"name\": \"fill\", \"arguments\": " + json_args(keys) +
                    "}\n</tool_call>";
         },
         true},
        {ToolCallFormat::MistralJson,
         [&](const std::vector<std::string>& keys) {
             return "[TOOL_CALLS]fill[ARGS]" + json_args(keys);
         },
         true},
        {ToolCallFormat::Llama3Json,
         [&](const std::vector<std::string>& keys) {
             return "{\"name\": \"fill\", \"parameters\": " + json_args(keys) + "}";
         },
         true},
        {ToolCallFormat::Harmony,
         [&](const std::vector<std::string>& keys) {
             return "<|start|>assistant<|channel|>commentary to=functions.fill <|constrain|>json"
                    "<|message|>" +
                    json_args(keys) + "<|call|>";
         },
         true},
        {ToolCallFormat::GraniteJson,
         [&](const std::vector<std::string>& keys) {
             return "<|tool_call|>[{\"name\": \"fill\", \"arguments\": " + json_args(keys) + "}]";
         },
         true},
        {ToolCallFormat::Gemma4,
         [](const std::vector<std::string>& keys) {
             std::string body;
             for (std::size_t i = 0; i < keys.size(); ++i) {
                 body += (i == 0 ? "" : ",") + keys[i] + ":<|\"|>v" + keys[i] + "<|\"|>";
             }
             return "<|tool_call>call:fill{" + body + "}<tool_call|>";
         },
         true},
        {ToolCallFormat::Pythonic,
         [](const std::vector<std::string>& keys) {
             std::string body;
             for (std::size_t i = 0; i < keys.size(); ++i) {
                 body += (i == 0 ? "" : ", ") + keys[i] + "=\"v" + keys[i] + "\"";
             }
             return "<function_calls>fill(" + body + ")</function_calls>";
         },
         true},
        {ToolCallFormat::Atem,
         [](const std::vector<std::string>& keys) {
             std::string body;
             for (const std::string& key : keys) {
                 body += "<atem:parameter name=\"" + key + "\">v" + key + "</atem:parameter>\n";
             }
             return " to=fill<|message|><atem:function_calls>\n<atem:invoke name=\"fill\">\n" +
                    body + "</atem:invoke>\n</atem:function_calls>";
         },
         true},
    };
    const ninferno::structured::ConstraintCompiler compiler(byte_vocabulary());
    const std::vector<std::string> definitions = {tool_definition("fill", schema)};
    for (const Case& c : cases) {
        const std::string name(ninferno::chat::tool_call_format_name(c.format));
        const auto& syntax = ninferno::chat::tool_call_syntax(c.format);
        const auto parse_contract =
            ninferno::chat::build_tool_call_output_contract(definitions, true, c.format);
        for (const bool strict : {true, false}) {
            const std::vector<ToolCallConstraintTool> tools = {
                {.name = "fill", .parameters_json = schema.dump(), .strict = strict}};
            for (const ToolCallRequirement requirement :
                 {ToolCallRequirement::Required, ToolCallRequirement::Named}) {
                ninferno::OutputConstraintSpec spec;
                spec.kind                   = ninferno::OutputConstraintKind::StructuralTag;
                spec.special_tokens_visible = true;
                spec.definition     = syntax.structural_tag(tools, requirement, "fill", true);
                const auto compiled = compiler.compile(spec);
                const std::string label =
                    name + (strict ? " strict" : " non-strict") +
                    (requirement == ToolCallRequirement::Named ? " named" : " required");
                const std::string text = c.call(order);
                failures += check(grammar_accepts(compiled, text),
                                  label + " admits keys C,A,E,B,D: " + text);
                failures += check(grammar_accepts(compiled, c.call({"E", "D", "C", "B", "A"})),
                                  label + " admits keys in reverse schema order");
                failures += check(grammar_accepts(compiled, c.call({"D"})),
                                  label + " admits one late key alone");
                failures += check(grammar_accepts(compiled, c.call({"A", "B", "C", "D", "E"})),
                                  label + " admits keys in schema order");
                if (strict && c.shaped) {
                    failures += check(!grammar_accepts(compiled, c.call({"C", "A", "C"})),
                                      label + " refuses a key written twice");
                    failures += check(!grammar_accepts(compiled, c.call({"C", "F"})),
                                      label + " refuses an undeclared key");
                }
                const auto parsed = syntax.parse(text, 64, *parse_contract);
                bool all          = parsed.tool_calls.size() == 1;
                if (all) {
                    const Json arguments = Json::parse(parsed.tool_calls[0].arguments_json);
                    for (const std::string& key : order) {
                        all = all && arguments.value(key, std::string()) == "v" + key;
                    }
                }
                failures += check(all, label + " parser reads all five keys back");
                if (!strict || !c.shaped) { continue; }
                // Required A and D keep their order; the optional keys go anywhere around them.
                Json mixed        = schema;
                mixed["required"] = Json::array({"A", "D"});
                const std::vector<ToolCallConstraintTool> spined = {
                    {.name = "fill", .parameters_json = mixed.dump(), .strict = true}};
                spec.definition    = syntax.structural_tag(spined, requirement, "fill", true);
                const auto ordered = compiler.compile(spec);
                failures += check(grammar_accepts(ordered, text), label + " mixed: C,A,E,B,D");
                failures += check(grammar_accepts(ordered, c.call({"A", "D"})),
                                  label + " mixed: the required keys only");
                failures += check(!grammar_accepts(ordered, c.call({"D", "A"})),
                                  label + " mixed: required keys keep their order");
                failures += check(!grammar_accepts(ordered, c.call({"C", "A", "E"})),
                                  label + " mixed: a required key cannot be skipped");
            }
        }
    }
    // A strict schema a syntax cannot write is refused, not silently left unenforced.
    const auto named = [](const std::string& property) {
        Json schema                     = Json::parse(R"({"type":"object","properties":{}})");
        schema["properties"][property] = Json{{"type", "string"}};
        return schema;
    };
    const Json recursive = Json::parse(R"({"type":"object","properties":{
        "name":{"type":"string"},"children":{"type":"array","items":{"$ref":"#"}}}})");
    const Json patterned = Json::parse(
        R"({"type":"object","patternProperties":{"^x":{"type":"string"}}})");
    for (const auto& [format, refused] : std::vector<std::pair<ToolCallFormat, Json>>{
             {ToolCallFormat::Pythonic, named("max-results")},
             {ToolCallFormat::Atem, named("say \"hi\"")},
             {ToolCallFormat::Pythonic, recursive},
             {ToolCallFormat::Atem, recursive},
             {ToolCallFormat::Pythonic, patterned},
             {ToolCallFormat::Gemma4, Json::parse(R"({"anyOf":[{"type":"object"},{"type":"null"}]})")},
             {ToolCallFormat::Gemma4, named("a<|\"|>b")},
             {ToolCallFormat::Gemma4, Json::parse(
                  R"({"type":"object","properties":{"v":{"type":"string","pattern":"^.+$"}}})")},
             {ToolCallFormat::Atem, Json::parse(
                  R"({"type":"object","properties":{"v":{"type":"string","pattern":"^[^x]+$"}}})")},
             {ToolCallFormat::Atem,
              Json::parse(R"({"type":"object","properties":{"v":{"enum":["a</atem:parameter>b"]}}})")},
         }) {
        const std::vector<ToolCallConstraintTool> tools = {
            {.name = "fill", .parameters_json = refused.dump(), .strict = true}};
        ninferno::OutputConstraintSpec spec;
        spec.kind                   = ninferno::OutputConstraintKind::StructuralTag;
        spec.special_tokens_visible = true;
        spec.definition             = ninferno::chat::tool_call_syntax(format).structural_tag(
            tools, ToolCallRequirement::Required, "", true);
        std::string code = "accepted";
        try {
            (void)compiler.compile(spec);
        } catch (const ninferno::OutputConstraintError& error) { code = error.code(); }
        failures += check(code == "unsupported_schema",
                          std::string(ninferno::chat::tool_call_format_name(format)) +
                              " strict schema refused: " + refused.dump() + ": " + code);
    }
    // A root typed ["object"] is the same object root.
    for (const ToolCallFormat format :
         {ToolCallFormat::Gemma4, ToolCallFormat::Pythonic, ToolCallFormat::Atem}) {
        const std::vector<ToolCallConstraintTool> tools = {
            {.name            = "fill",
             .parameters_json = R"({"type":["object"],"properties":{"a":{"type":"integer"}}})",
             .strict          = true}};
        ninferno::OutputConstraintSpec spec;
        spec.kind                   = ninferno::OutputConstraintKind::StructuralTag;
        spec.special_tokens_visible = true;
        spec.definition             = ninferno::chat::tool_call_syntax(format).structural_tag(
            tools, ToolCallRequirement::Required, "", true);
        bool compiled = true;
        try {
            (void)compiler.compile(spec);
        } catch (const std::exception&) { compiled = false; }
        failures += check(compiled, std::string(ninferno::chat::tool_call_format_name(format)) +
                                        " accepts a root typed [\"object\"]");
    }
    // ATEM: a constrained string is the text between the tags, read back as written; an additional
    // key never spells a declared one.
    {
        const Json unit = Json::parse(R"({"type":"object","properties":{
            "unit":{"enum":["celsius","fahrenheit"]},"days":{"type":"integer"}},
            "required":["unit"],"additionalProperties":{"type":"string"}})");
        const std::vector<ToolCallConstraintTool> tools = {
            {.name = "fill", .parameters_json = unit.dump(), .strict = true}};
        const auto& atem = ninferno::chat::tool_call_syntax(ToolCallFormat::Atem);
        ninferno::OutputConstraintSpec spec;
        spec.kind                   = ninferno::OutputConstraintKind::StructuralTag;
        spec.special_tokens_visible = true;
        spec.definition = atem.structural_tag(tools, ToolCallRequirement::Named, "fill", true);
        const auto compiled = compiler.compile(spec);
        const auto call     = [](const std::string& body) {
            return " to=fill<|message|><atem:function_calls>\n<atem:invoke name=\"fill\">\n" + body +
                   "</atem:invoke>\n</atem:function_calls>";
        };
        const std::string good = call("<atem:parameter name=\"unit\">celsius</atem:parameter>\n"
                                      "<atem:parameter name=\"note\">x</atem:parameter>\n");
        failures += check(grammar_accepts(compiled, good), "atem strict enum and additional key");
        failures += check(
            !grammar_accepts(compiled, call("<atem:parameter name=\"unit\">\ncelsius\n"
                                            "</atem:parameter>\n")),
            "atem strict enum refuses whitespace around the value");
        failures += check(
            !grammar_accepts(compiled, call("<atem:parameter name=\"unit\">celsius</atem:parameter>\n"
                                            "<atem:parameter name=\"days\">1</atem:parameter>\n"
                                            "<atem:parameter name=\"days\">x</atem:parameter>\n")),
            "atem additional key cannot spell a declared one");
        const std::vector<std::string> unit_definitions = {tool_definition("fill", unit)};
        const auto parsed = atem.parse(
            good, 64, *ninferno::chat::build_tool_call_output_contract(unit_definitions, true,
                                                                       ToolCallFormat::Atem));
        failures += check(parsed.tool_calls.size() == 1 &&
                              Json::parse(parsed.tool_calls[0].arguments_json).value("unit", "") ==
                                  "celsius",
                          "atem parser reads the enum value back");
    }
    // Gemma 4 and Pythonic read a repeated key as its last value (a key past the tracked ones may
    // repeat under the grammar), as the JSON and XML parsers do.
    {
        const auto gemma = ninferno::chat::tool_call_syntax(ToolCallFormat::Gemma4)
                               .parse("<|tool_call>call:fill{A:1,A:2}<tool_call|>", 64,
                                      *ninferno::chat::build_tool_call_output_contract(
                                          definitions, true, ToolCallFormat::Gemma4));
        failures += check(gemma.tool_calls.size() == 1 &&
                              Json::parse(gemma.tool_calls[0].arguments_json).value("A", 0) == 2,
                          "gemma4 parser keeps the last of a repeated key");
        const auto python = ninferno::chat::tool_call_syntax(ToolCallFormat::Pythonic)
                                .parse("<function_calls>fill(A=1, A=2)</function_calls>", 64,
                                       *ninferno::chat::build_tool_call_output_contract(
                                           definitions, true, ToolCallFormat::Pythonic));
        failures += check(python.tool_calls.size() == 1 &&
                              Json::parse(python.tool_calls[0].arguments_json).value("A", 0) == 2,
                          "pythonic parser keeps the last of a repeated key");
    }
    // Gemma 4 writes a key of other characters quoted, and its parser reads it back.
    {
        const Json spaced = Json::parse(
            R"({"type":"object","properties":{"two words":{"type":"integer"}},"required":["two words"],"additionalProperties":false})");
        const std::vector<ToolCallConstraintTool> tools = {
            {.name = "fill", .parameters_json = spaced.dump(), .strict = true}};
        ninferno::OutputConstraintSpec spec;
        spec.kind                   = ninferno::OutputConstraintKind::StructuralTag;
        spec.special_tokens_visible = true;
        spec.definition = ninferno::chat::tool_call_syntax(ToolCallFormat::Gemma4)
                              .structural_tag(tools, ToolCallRequirement::Named, "fill", true);
        const auto compiled = compiler.compile(spec);
        failures += check(
            grammar_accepts(compiled, "<|tool_call>call:fill{<|\"|>two words<|\"|>:3}<tool_call|>"),
            "gemma4 strict quoted key");
        failures += check(
            !grammar_accepts(compiled, "<|tool_call>call:fill{two words:3}<tool_call|>"),
            "gemma4 strict refuses a bare key with a space");
    }
    return failures;
}

// A tiny BPE: "a"+"b" merge, one special added token, byte-fallback decoding.
constexpr std::string_view kTinyTokenizer = R"({
  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2, " ": 3, "<": 4, "T": 5, ">": 6,
            "<0x41>": 7, "<0xE2>": 8}, "merges": ["a b"], "byte_fallback": true},
  "added_tokens": [{"id": 9, "content": "<T>", "special": true}],
  "decoder": {"type": "Sequence", "decoders": [{"type": "ByteFallback"}, {"type": "Fuse"}]}
})";

int test_codec() {
    int failures = 0;
    const ninferno::text::HfTokenCodec codec(kTinyTokenizer, "", R"({"eos_token_id": 9})");
    const std::string text = "ab<T>ab a";
    const std::vector<std::size_t> boundaries = {0, 1, 2, 5, 8, 9};
    const auto encoded = codec.encode_with_boundaries(text, boundaries);
    failures += check(encoded.input_ids == std::vector<int>({2, 9, 2, 3, 0}), "tiny encode");
    const auto exact = [&](std::size_t index) { return encoded.boundaries[index].exact_frontier; };
    failures += check(exact(0) == 0U, "boundary at start");
    failures += check(!exact(1) && encoded.boundaries[1].stable_frontier == 0U,
                      "boundary inside a merged pair is not exact");
    failures += check(exact(2) == 1U, "boundary before an added token");
    failures += check(exact(3) == 2U, "boundary after an added token");
    failures += check(exact(4) == 4U, "exact boundary inside an ordinary run");
    failures += check(exact(5) == 5U, "boundary at the end");

    const std::vector<ninferno::text::ByteSpan> literal = {{2, 5}};
    const auto guarded = codec.encode_with_boundaries(text, {}, {}, literal);
    failures += check(guarded.input_ids == std::vector<int>({2, 4, 5, 6, 2, 3, 0}),
                      "literal span suppresses the added token");

    failures += check(codec.decoded_token(7).bytes == "A", "byte-fallback token bytes");
    failures += check(codec.decoded_token(8).bytes == "\xE2", "partial UTF-8 token bytes");
    failures += check(codec.decoded_token(9).special && codec.decoded_token(9).bytes == "<T>",
                      "special token bytes");
    failures += check(codec.default_stop_token_ids() == std::vector<int>({9}),
                      "generation config stops");
    failures += check(codec.token_id("<T>") == 9 && codec.token_id("ab") == 2, "token ids");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_conventions();
    failures += test_parsers();
    failures += test_streaming_decoder();
    failures += test_structural_tags();
    failures += test_argument_key_order();
    failures += test_codec();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "chat: all checks passed\n";
    return 0;
}
