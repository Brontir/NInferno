// Compile cost of strict tool-call grammars with a real vocabulary: twenty strict tools, each with
// required and optional arguments, a length-bounded string, an enum, an array and a nested object
// of six optional properties, in every call syntax that holds strict arguments to their schema.
// Bounds per syntax: the compile ends within half the default compile timeout, the compiled grammar
// stays within 1/8 of the default cache, and a token mask along a call costs at most 20 ms.
//
// Prediction (key order rule, structured/README): a tool's object has 3 required and 5 optional
// properties, (3 + 1) * 2^5 = 128 states, the nested object 2^6 = 64; states hold only rule
// references, so the vocabulary-wide mask work grows with the property terminals (about 30 per
// tool), not the states. Expected: a compile of a few seconds or less, tens of MB at most.
//
// Needs NINFERNO_GRAMMAR_TOKENIZER (a tokenizer.json, e.g. Qwen3.5's); skipped (77) without it.
#include "chat/conventions.h"
#include "chat/tool_calls.h"
#include "structured/constraint.h"
#include "text/hf_token_codec.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;
using ninferno::ToolCallConstraintTool;
using ninferno::ToolCallRequirement;
using ninferno::chat::ToolCallFormat;
using Clock = std::chrono::steady_clock;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

Json tool_schema(int index) {
    Json nested = Json::parse(R"({"type":"object","properties":{
        "street":{"type":"string"},"city":{"type":"string"},"zip":{"type":"string"},
        "country":{"type":"string"},"lat":{"type":"number"},"lon":{"type":"number"}},
        "additionalProperties":false})");
    Json schema = Json::parse(R"({"type":"object","properties":{
        "query":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":100},
        "mode":{"enum":["fast","exact","fuzzy"]},
        "note":{"type":"string","maxLength":200},"weight":{"type":"number"},
        "verbose":{"type":"boolean"},"tags":{"type":"array","items":{"type":"string"}}},
        "required":["query","limit","mode"],"additionalProperties":false})");
    schema["properties"]["address"]                = nested;
    schema["properties"]["query"]["description"]   = "query " + std::to_string(index);
    return schema;
}

// One call of tool_0 in each syntax: the required arguments after two optional ones.
std::string call_text(ToolCallFormat format) {
    switch (format) {
    case ToolCallFormat::QwenXml:
        return "<tool_call>\n<function=tool_0>\n<parameter=verbose>\ntrue\n</parameter>\n"
               "<parameter=address>\n{\"city\": \"Oslo\", \"street\": \"Main\"}\n</parameter>\n"
               "<parameter=query>\nweather\n</parameter>\n<parameter=limit>\n5\n</parameter>\n"
               "<parameter=mode>\nfast\n</parameter>\n</function>\n</tool_call>";
    case ToolCallFormat::HermesJson:
        return "<tool_call>\n{\"name\": \"tool_0\", \"arguments\": {\"verbose\": true, "
               "\"address\": {\"city\": \"Oslo\", \"street\": \"Main\"}, \"query\": \"weather\", "
               "\"limit\": 5, \"mode\": \"fast\"}}\n</tool_call>";
    case ToolCallFormat::Harmony:
        return "<|start|>assistant<|channel|>commentary to=functions.tool_0 <|constrain|>json"
               "<|message|>{\"verbose\": true, \"address\": {\"city\": \"Oslo\", \"street\": "
               "\"Main\"}, \"query\": \"weather\", \"limit\": 5, \"mode\": \"fast\"}<|call|>";
    case ToolCallFormat::Gemma4:
        return "<|tool_call>call:tool_0{verbose:true,address:{city:<|\"|>Oslo<|\"|>,"
               "street:<|\"|>Main<|\"|>},query:<|\"|>weather<|\"|>,limit:5,"
               "mode:<|\"|>fast<|\"|>}<tool_call|>";
    case ToolCallFormat::Pythonic:
        return "<function_calls>tool_0(verbose=true, address={\"city\": \"Oslo\", \"street\": "
               "\"Main\"}, query=\"weather\", limit=5, mode=\"fast\")</function_calls>";
    case ToolCallFormat::Atem:
        return " to=tool_0<|message|><atem:function_calls>\n<atem:invoke name=\"tool_0\">\n"
               "<atem:parameter name=\"verbose\">true</atem:parameter>\n"
               "<atem:parameter name=\"address\">{\"city\": \"Oslo\", \"street\": \"Main\"}"
               "</atem:parameter>\n<atem:parameter name=\"query\">weather</atem:parameter>\n"
               "<atem:parameter name=\"limit\">5</atem:parameter>\n"
               "<atem:parameter name=\"mode\">fast</atem:parameter>\n</atem:invoke>\n"
               "</atem:function_calls>";
    default:
        return {};
    }
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

} // namespace

int main() {
    const char* path = std::getenv("NINFERNO_GRAMMAR_TOKENIZER");
    if (path == nullptr || *path == '\0') {
        std::cout << "tool grammar budget: NINFERNO_GRAMMAR_TOKENIZER not set, skipped\n";
        return 77;
    }
    const ninferno::text::HfTokenCodec codec(read_file(path), "{}", "{}");
    ninferno::structured::GrammarVocabulary vocabulary;
    vocabulary.token_domain = static_cast<std::uint32_t>(codec.vocab_size());
    vocabulary.token_bytes.resize(vocabulary.token_domain);
    vocabulary.special.assign(vocabulary.token_domain, false);
    for (std::uint32_t token = 0; token < vocabulary.token_domain; ++token) {
        const int id = static_cast<int>(token);
        if (!codec.is_valid_token(id)) {
            vocabulary.special[token] = true;
            continue;
        }
        const auto decoded              = codec.decoded_token(id);
        vocabulary.token_bytes[token]   = std::string(decoded.bytes);
        vocabulary.special[token]       = decoded.special;
    }
    for (const int stop : codec.default_stop_token_ids()) { vocabulary.stop_token_ids.push_back(stop); }
    std::cout << "vocabulary " << vocabulary.token_domain << " tokens\n";

    std::vector<ToolCallConstraintTool> tools;
    for (int index = 0; index < 20; ++index) {
        tools.push_back({.name            = "tool_" + std::to_string(index),
                         .parameters_json = tool_schema(index).dump(),
                         .strict          = true});
    }
    const ninferno::structured::CompilerOptions options;
    for (const ToolCallFormat format :
         {ToolCallFormat::QwenXml, ToolCallFormat::HermesJson, ToolCallFormat::Harmony,
          ToolCallFormat::Gemma4, ToolCallFormat::Pythonic, ToolCallFormat::Atem}) {
        const std::string name(ninferno::chat::tool_call_format_name(format));
        // A compiler per syntax: nothing is shared through its cache.
        const ninferno::structured::ConstraintCompiler compiler(vocabulary, options);
        ninferno::OutputConstraintSpec spec;
        spec.kind                   = ninferno::OutputConstraintKind::StructuralTag;
        spec.special_tokens_visible = true;
        spec.definition = ninferno::chat::tool_call_syntax(format).structural_tag(
            tools, ToolCallRequirement::Required, "", true);
        const auto start = Clock::now();
        std::shared_ptr<const ninferno::structured::CompiledConstraint> compiled;
        try {
            compiled = compiler.compile(spec);
        } catch (const std::exception& error) {
            check(false, name + ": compile failed: " + error.what());
            continue;
        }
        const double compile_s = std::chrono::duration<double>(Clock::now() - start).count();
        const std::size_t bytes = ninferno::structured::memory_bytes(*compiled);

        ninferno::structured::ConstraintMatcher matcher(compiled);
        std::vector<std::uint32_t> row(ninferno::structured::vocabulary_view(*compiled).mask_words);
        double mask_total = 0;
        double mask_max   = 0;
        int masks         = 0;
        bool accepted     = true;
        for (const char c : call_text(format)) {
            const auto mask_start = Clock::now();
            matcher.fill_mask(row.data());
            const double took = std::chrono::duration<double>(Clock::now() - mask_start).count();
            mask_total += took;
            mask_max = std::max(mask_max, took);
            ++masks;
            if (!matcher.accept_bytes(std::string_view(&c, 1))) {
                accepted = false;
                break;
            }
        }
        std::cout << name << ": compile " << compile_s << " s, " << (bytes >> 20U)
                  << " MiB, mask mean " << (mask_total / masks) * 1e3 << " ms, max "
                  << mask_max * 1e3 << " ms over " << masks << " steps\n";
        check(accepted && matcher.completed(), name + ": the grammar admits a tool_0 call");
        check(compile_s < options.compile_timeout.count() / 2000.0,
              name + ": compile within half the compile timeout");
        check(bytes < options.cache_bytes / 8, name + ": compiled grammar within 1/8 of the cache");
        check(mask_max < 0.020, name + ": a token mask within 20 ms");
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " tool grammar budget (" << failures
              << " failures)\n";
    return failures == 0 ? 0 : 1;
}
