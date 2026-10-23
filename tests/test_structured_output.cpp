// Host-side structured-output contracts: JSON Schema support policy, grammar compilation, the
// per-request mask rows used by speculative verification, the reasoning gate, violation handling,
// and the declared mask-row class with the step rule it feeds (runtime/contract/token_mask.h).
// CPU only; uses a synthetic byte-level vocabulary.
#include "runtime/contract/token_mask.h"
#include "structured/constraint.h"
#include "structured/request_constraint.h"
#include "structured/schema_support.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninferno;
using namespace ninferno::structured;
using Json = nlohmann::ordered_json;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

// Tokens 0..255 are single bytes; the rest are multi-byte and control tokens.
constexpr TokenId kObjectOpenQuote = 256; // {"
constexpr TokenId kQuoteColon      = 257; // ":
constexpr TokenId kThinkClose      = 258; // </think>
constexpr TokenId kDoubleNewline   = 259; // \n\n
constexpr TokenId kImEnd           = 260; // <|im_end|> (special, stop)
constexpr TokenId kToolCall        = 261; // <tool_call> (special)
constexpr TokenId kTrue            = 262; // true
constexpr TokenId kCloseThenText   = 263; // </think>{  (marker completes mid-token)
constexpr TokenId kThinkOpen       = 264; // <think> (a vocabulary whose model opens reasoning)
constexpr std::uint32_t kDomain    = 300;

GrammarVocabulary vocabulary() {
    GrammarVocabulary out;
    out.token_domain = kDomain;
    out.token_bytes.resize(kDomain);
    out.special.assign(kDomain, false);
    for (int byte = 0; byte < 256; ++byte) {
        out.token_bytes[static_cast<std::size_t>(byte)] = std::string(1, static_cast<char>(byte));
    }
    out.token_bytes[kObjectOpenQuote] = "{\"";
    out.token_bytes[kQuoteColon]      = "\":";
    out.token_bytes[kThinkClose]      = "</think>";
    out.token_bytes[kDoubleNewline]   = "\n\n";
    out.token_bytes[kImEnd]           = "<|im_end|>";
    out.special[kImEnd]               = true;
    out.token_bytes[kToolCall]        = "<tool_call>";
    out.special[kToolCall]            = true;
    out.token_bytes[kTrue]            = "true";
    out.token_bytes[kCloseThenText]   = "</think>{";
    out.stop_token_ids                = {kImEnd};
    out.reasoning_close               = "</think>";
    return out;
}

bool allowed(const std::uint32_t* row, TokenId token) {
    return ((row[static_cast<std::uint32_t>(token) >> 5U] >> (static_cast<std::uint32_t>(token) & 31U)) &
            1U) != 0U;
}

std::vector<TokenId> bytes_of(std::string_view text) {
    std::vector<TokenId> out;
    for (const char c : text) { out.push_back(static_cast<TokenId>(static_cast<unsigned char>(c))); }
    return out;
}

OutputConstraintSpec schema_spec(const Json& schema, bool compact = true) {
    OutputConstraintSpec spec;
    spec.kind         = OutputConstraintKind::JsonSchema;
    spec.definition   = schema.dump();
    spec.compact_json = compact;
    return spec;
}

std::string expect_error(const ConstraintCompiler& compiler, const OutputConstraintSpec& spec) {
    try {
        (void)compiler.compile(spec);
    } catch (const OutputConstraintError& error) { return error.code(); }
    return "accepted";
}

// Drives one request through `text` one byte token at a time: each byte must be admitted by the
// row computed for it and the final row must admit the stop token. `closed`: no proper prefix is
// complete and nothing but whitespace may follow the text.
void walk_text(RequestConstraint& request, std::string_view text, const std::string& label,
               bool closed = true) {
    const std::size_t words = request.mask_words();
    std::vector<std::uint32_t> row(words);
    for (const TokenId token : bytes_of(text)) {
        request.fill_round({}, row.data(), words);
        check(allowed(row.data(), token), label + ": byte token " + std::to_string(token) +
                                              " rejected");
        check(!closed || !allowed(row.data(), kImEnd),
              label + ": stop admitted before completion");
        const TokenId one[] = {token};
        request.commit(one);
    }
    request.fill_round({}, row.data(), words);
    check(allowed(row.data(), kImEnd), label + ": stop not admitted at completion");
    for (TokenId token = 0; closed && token < 256; ++token) {
        check(!allowed(row.data(), token) || token == ' ' || token == '\n' || token == '\t' ||
                  token == '\r',
              label + ": completed grammar admits content token " + std::to_string(token));
    }
    const TokenId stop[] = {kImEnd};
    request.commit(stop);
    const OutputConstraintStats stats = request.stats();
    check(stats.complete && !stats.violated, label + ": not complete after stop");
}

void schema_policy_contract() {
    const auto code = [](const Json& schema) -> std::string {
        try {
            (void)normalize_supported_json_schema(schema);
        } catch (const OutputConstraintError& error) { return error.code(); }
        return "ok";
    };
    check(code(Json::parse(R"({"type":"object","properties":{"a":{"type":"string"}},
        "required":["a"],"additionalProperties":false})")) == "ok",
          "plain object schema");
    check(code(Json::parse(R"({"anyOf":[{"type":"string"},{"type":"null"}],"default":null,
        "title":"x"})")) == "ok",
          "pydantic optional with annotations");
    check(code(Json::parse(R"({"$defs":{"P":{"type":"integer","minimum":0}},
        "type":"object","properties":{"p":{"$ref":"#/$defs/P","description":"d"}}})")) == "ok",
          "$ref with an annotation sibling");
    check(code(Json::parse(R"({"type":"string","enum":["a","b"]})")) == "ok", "typed enum");
    check(code(Json::parse(R"({"type":"object","x-order":1})")) == "ok", "x- extension");
    check(code(Json::parse(R"({"not":{"type":"string"}})")) == "unsupported_schema", "not");
    check(code(Json::parse(R"({"type":"array","uniqueItems":true})")) == "unsupported_schema",
          "uniqueItems");
    check(code(Json::parse(R"({"type":"string","pattern":"^a+$","maxLength":3})")) ==
              "unsupported_schema",
          "pattern with maxLength");
    check(code(Json::parse(R"({"type":"string","format":"ipv4"})")) == "unsupported_schema",
          "unknown format");
    check(code(Json::parse(R"({"$ref":"#/$defs/P","type":"integer"})")) == "unsupported_schema",
          "$ref with an assertion sibling");
    check(code(Json::parse(R"({"type":"object","anyOf":[{"required":["a"]}]})")) ==
              "unsupported_schema",
          "anyOf next to type");
    check(code(Json::parse(R"({"type":"number","multipleOf":0.5})")) == "unsupported_schema",
          "number multipleOf");
    check(code(Json::parse(R"({"allOf":[{"type":"object"},{"required":["a"]}]})")) ==
              "unsupported_schema",
          "allOf with two schemas");
    check(code(Json::parse(R"({"minimum":1})")) == "unsupported_schema", "minimum without type");
    check(code(Json::parse(R"({"type":"object","required":["a"]})")) == "unsupported_schema",
          "required without a property");
    check(code(Json::parse(R"({"type":"string","nullable":true})")) == "ok",
          "non-vocabulary keyword is ignored");
    check(code(Json::parse(R"({"type":"number","divisibleBy":2})")) == "unsupported_schema",
          "draft-03 assertion");
    check(code(Json::parse(R"({"type":"array","items":{"type":"string"},
        "additionalItems":false})")) == "ok",
          "additionalItems without tuple items has no effect");
    check(code(Json::parse(R"({"type":"array","items":[{"type":"string"}],
        "additionalItems":false})")) == "unsupported_schema",
          "additionalItems with tuple items");
    check(code(Json::parse(R"({"type":"object","properties":{"a":{"$ref":"#/components/A"}},
        "components":{"A":{"type":"string"}}})")) == "unsupported_schema",
          "$ref into a dropped keyword");
    check(code(Json::parse(R"({"type":"object","properties":{"a":{"id":"http://x/a",
        "type":"object","properties":{"b":{"$ref":"#/definitions/B"}}}},
        "definitions":{"B":{"type":"string"}}})")) == "unsupported_schema",
          "$ref below a nested base URI");
    check(code(Json::parse(R"({"type":"object","properties":{"a":{"id":"#/properties/a",
        "type":"object","properties":{"b":{"$ref":"#/definitions/B"}}}},
        "definitions":{"B":{"type":"string"}}})")) == "ok",
          "fragment-only nested id keeps the base URI");
    check(code(Json::parse(R"({"type":"integer","exclusiveMinimum":true})")) == "invalid_schema",
          "draft-04 exclusive bound without a bound");
    check(code(Json::parse(R"({"$ref":"http://example.com/s"})")) == "unsupported_schema",
          "remote $ref");
    check(code(Json::parse(R"({"type":"strin"})")) == "invalid_schema", "unknown type name");
    check(code(Json::parse(R"({"type":"integer","enum":["a"]})")) == "invalid_schema",
          "enum outside type");

    const Json normalized =
        normalize_supported_json_schema(Json::parse(R"({"type":"object","properties":{
            "tags":{"type":"array"},"meta":{"type":"object"}}})"));
    check(normalized["properties"]["tags"]["items"] == true, "array without items is opened");
    check(normalized["properties"]["meta"]["additionalProperties"] == true,
          "object without property space is opened");
    check(!normalized.contains("additionalProperties"), "declared object stays strict");

    const Json dialects = normalize_supported_json_schema(Json::parse(R"({"type":"object",
        "javaName":"X","properties":{
        "n":{"type":"integer","format":"int32"},
        "m":{"type":"integer","format":"int64","minimum":0},
        "f":{"type":"number","format":"float"},
        "s":{"type":"string","format":"password","maxLength":8},
        "e":{"type":"number","minimum":1,"exclusiveMinimum":true,"maximum":5,
             "exclusiveMaximum":false}}})"));
    const Json& props = dialects["properties"];
    check(!dialects.contains("javaName"), "non-vocabulary keyword is dropped");
    check(props["n"]["minimum"] == -2147483648LL && props["n"]["maximum"] == 2147483647LL &&
              !props["n"].contains("format"),
          "int32 becomes range bounds");
    check(props["m"]["minimum"] == 0 && props["m"].contains("maximum"),
          "int64 keeps the declared lower bound");
    check(!props["f"].contains("format"), "number format is an annotation");
    check(!props["s"].contains("format") && props["s"]["maxLength"] == 8,
          "vendor string format is an annotation");
    check(props["e"]["exclusiveMinimum"] == 1 && !props["e"].contains("minimum") &&
              props["e"]["maximum"] == 5 && !props["e"].contains("exclusiveMaximum"),
          "draft-04 boolean exclusive bounds");

    const Json composite = normalize_supported_json_schema(Json::parse(R"({"$defs":{
        "P":{"type":"integer"}},"type":"object","properties":{
        "o":{"type":"object","oneOf":[{"properties":{"a":{"type":"string"}},"required":["a"]},
             {"type":"string"},{"type":["object","null"],"properties":{"b":{"type":"null"}}}]},
        "k":{"type":["integer","string"],"anyOf":[{"type":"number"},true]},
        "e":{"type":"string","enum":["a",1,"b"]},
        "r":{"type":"number","$ref":"#/$defs/P"},
        "big":{"type":"integer","maximum":9223372036854775807,"minimum":-1e30}}})"));
    const Json& cp = composite["properties"];
    check(!cp["o"].contains("type") && cp["o"]["oneOf"].size() == 2 &&
              cp["o"]["oneOf"][0]["type"] == "object" && cp["o"]["oneOf"][1]["type"] == "object",
          "type is distributed into oneOf branches and a string branch is dropped");
    check(cp["k"]["anyOf"][0]["type"] == "integer" &&
              cp["k"]["anyOf"][1]["type"] == Json::parse(R"(["integer","string"])"),
          "type intersection keeps integer inside number");
    check(cp["e"]["enum"] == Json::parse(R"(["a","b"])"), "enum values outside the type");
    check(!cp["r"].contains("type") && cp["r"]["$ref"] == "#/$defs/P",
          "type implied by the $ref target");
    check(cp["big"]["maximum"] == 9007199254740991LL &&
              cp["big"]["minimum"] == -9007199254740991LL,
          "integer bounds narrowed to the I-JSON range");
    check(code(Json::parse(R"({"type":"integer","minimum":1e30})")) == "unsupported_schema",
          "integer lower bound past the I-JSON range");
    check(code(Json::parse(R"({"$defs":{"P":{"type":"number"}},"type":"integer",
        "$ref":"#/$defs/P"})")) == "unsupported_schema",
          "type next to a wider $ref target");
    check(code(Json::parse(R"({"type":"string","anyOf":[{"type":"integer"}]})")) ==
              "invalid_schema",
          "no branch admits the declared type");
}

void compile_error_contract(const ConstraintCompiler& compiler) {
    OutputConstraintSpec bad_json;
    bad_json.kind       = OutputConstraintKind::JsonSchema;
    bad_json.definition = "{not json";
    check(expect_error(compiler, bad_json) == "invalid_schema", "schema that is not JSON");

    OutputConstraintSpec bad_regex;
    bad_regex.kind       = OutputConstraintKind::Regex;
    bad_regex.definition = "(unclosed";
    check(expect_error(compiler, bad_regex) == "invalid_regex", "unbalanced regex");

    OutputConstraintSpec lookahead;
    lookahead.kind       = OutputConstraintKind::Regex;
    lookahead.definition = "a(?=b)b";
    check(expect_error(compiler, lookahead) == "invalid_regex", "lookahead is rejected");

    // An unrecognized escape is a converter warning (the character would be matched literally);
    // the warning capture turns it into an error instead of a silently different language.
    OutputConstraintSpec escape;
    escape.kind       = OutputConstraintKind::Regex;
    escape.definition = "a\\qb";
    check(expect_error(compiler, escape) == "unsupported_regex",
          "a degrading converter warning is reported instead of ignored");

    OutputConstraintSpec bad_grammar;
    bad_grammar.kind       = OutputConstraintKind::Grammar;
    bad_grammar.definition = "root ::= missing_rule";
    check(expect_error(compiler, bad_grammar) == "invalid_grammar", "undefined EBNF rule");

    OutputConstraintSpec empty_choice;
    empty_choice.kind = OutputConstraintKind::Choice;
    check(expect_error(compiler, empty_choice) == "invalid_choice", "empty choice list");

    OutputConstraintSpec overlapping_one_of = schema_spec(
        Json::parse(R"({"oneOf":[{"type":"object","properties":{"a":{"type":"string"}}},
                     {"type":"object","properties":{"b":{"type":"string"}}}]})"));
    check(expect_error(compiler, overlapping_one_of) == "unsupported_schema",
          "oneOf the compiler would weaken to anyOf");

    OutputConstraintSpec huge;
    huge.kind       = OutputConstraintKind::Regex;
    huge.definition = std::string(300U * 1024U, 'a');
    check(expect_error(compiler, huge) == "constraint_too_large", "oversized definition");
}

void busy_contract() {
    CompilerOptions options;
    options.max_concurrent  = 0;
    options.compile_timeout = std::chrono::milliseconds(20);
    const ConstraintCompiler busy(vocabulary(), options);
    OutputConstraintSpec spec;
    spec.kind = OutputConstraintKind::JsonObject;
    check(expect_error(busy, spec) == "constraint_compiler_busy", "no free compiler slot");
}

void json_walk_contract(const ConstraintCompiler& compiler) {
    const Json schema = Json::parse(R"({"type":"object","properties":{
        "name":{"type":"string"},"n":{"type":"integer","minimum":0},"ok":{"type":"boolean"}},
        "required":["name","n"],"additionalProperties":false})");
    auto compiled = compiler.compile(schema_spec(schema));
    check(compiler.compile(schema_spec(schema)) == compiled, "identical spec is cached");
    {
        RequestConstraint request(compiled, false);
        walk_text(request, R"({"name":"a\"b","n":12,"ok":true})", "strict object");
    }
    {
        RequestConstraint request(compiled, false);
        std::vector<std::uint32_t> row(request.mask_words());
        request.fill_round({}, row.data(), row.size());
        check(allowed(row.data(), '{') && allowed(row.data(), kObjectOpenQuote),
              "object start and {\" admitted");
        check(!allowed(row.data(), ' ') && !allowed(row.data(), '['),
              "compact schema rejects whitespace and arrays at the root");
        check(!allowed(row.data(), kToolCall) && !allowed(row.data(), kImEnd),
              "hidden special and stop tokens rejected");
    }
    {
        // Required property missing: the grammar cannot end.
        RequestConstraint request(compiled, false);
        std::vector<std::uint32_t> row(request.mask_words());
        const auto tokens = bytes_of(R"({"name":"x")");
        request.commit(tokens);
        request.fill_round({}, row.data(), row.size());
        check(!allowed(row.data(), '}'), "required property cannot be skipped");
        check(allowed(row.data(), ','), "required property continues");
    }

    OutputConstraintSpec spaced = schema_spec(schema, false);
    RequestConstraint spaced_request(compiler.compile(spaced), false);
    walk_text(spaced_request, "{ \"name\": \"x\",\n  \"n\": 0 }", "whitespace object");
}

void other_kinds_contract(const ConstraintCompiler& compiler) {
    OutputConstraintSpec choice;
    choice.kind    = OutputConstraintKind::Choice;
    choice.choices = {"yes", "no", "a \"quoted\"\nline"};
    for (const std::string& value : choice.choices) {
        RequestConstraint request(compiler.compile(choice), false);
        walk_text(request, value, "choice " + value);
    }

    OutputConstraintSpec regex;
    regex.kind       = OutputConstraintKind::Regex;
    regex.definition = "[0-9]{3}-[a-z]+";
    RequestConstraint regex_request(compiler.compile(regex), false);
    walk_text(regex_request, "123-abc", "regex", false);

    OutputConstraintSpec grammar;
    grammar.kind       = OutputConstraintKind::Grammar;
    grammar.definition = R"EBNF(root ::= "call(" arg ("," arg)* ")"
arg ::= [0-9]+)EBNF";
    RequestConstraint grammar_request(compiler.compile(grammar), false);
    walk_text(grammar_request, "call(1,22,333)", "ebnf");

    OutputConstraintSpec object;
    object.kind = OutputConstraintKind::JsonObject;
    RequestConstraint object_request(compiler.compile(object), false);
    walk_text(object_request, R"({"free":[1,{"x":null}],"y":"z"})", "json_object");
}

// Speculative rows: row j masks the position after drafts[0..j).
void speculative_rows_contract(const ConstraintCompiler& compiler) {
    const Json schema = Json::parse(R"({"type":"object","properties":{"k":{"type":"boolean"}},
        "required":["k"],"additionalProperties":false})");
    auto compiled     = compiler.compile(schema_spec(schema));
    RequestConstraint request(compiled, false);
    const std::size_t words = request.mask_words();

    const std::vector<TokenId> drafts = {kObjectOpenQuote, 'k', kQuoteColon, kTrue};
    std::vector<std::uint32_t> rows((drafts.size() + 1) * words);
    request.fill_round(drafts, rows.data(), words);
    const auto row = [&](std::size_t index) { return rows.data() + index * words; };
    check(allowed(row(0), kObjectOpenQuote), "row 0 admits {\"");
    check(allowed(row(1), 'k') && !allowed(row(1), 'x'), "row 1 admits only the property name");
    check(allowed(row(2), kQuoteColon), "row 2 admits \":");
    check(allowed(row(3), kTrue) && !allowed(row(3), '"'), "row 3 admits a boolean only");
    check(allowed(row(4), '}') && !allowed(row(4), ','), "row 4 closes the object");

    // Rows are recomputed from the restored matcher: a second call is identical.
    std::vector<std::uint32_t> again(rows.size());
    request.fill_round(drafts, again.data(), words);
    check(again == rows, "fill_round restores the matcher");

    // A draft the grammar rejects: later rows repeat the last consumable row.
    const std::vector<TokenId> bad = {kObjectOpenQuote, 'x', 'y'};
    std::vector<std::uint32_t> bad_rows((bad.size() + 1) * words);
    request.fill_round(bad, bad_rows.data(), words);
    check(std::equal(bad_rows.begin() + static_cast<std::ptrdiff_t>(words),
                     bad_rows.begin() + static_cast<std::ptrdiff_t>(2 * words),
                     bad_rows.begin() + static_cast<std::ptrdiff_t>(2 * words)),
          "row after a rejected draft repeats the previous row");
    check(!allowed(bad_rows.data() + words, 'x'), "rejected draft is disallowed in its row");

    // Commit a prefix the verifier accepted, then the next round starts from it.
    const TokenId accepted[] = {kObjectOpenQuote, 'k'};
    request.commit(accepted);
    std::vector<std::uint32_t> next(words);
    request.fill_round({}, next.data(), words);
    check(std::equal(next.begin(), next.end(), rows.begin() + static_cast<std::ptrdiff_t>(2 * words)),
          "committed prefix continues where the draft row predicted");

    // A stop token accepted mid-window ends consumable rows.
    RequestConstraint done(compiled, false);
    done.commit(bytes_of(R"({"k":true})"));
    const std::vector<TokenId> stop_first = {kImEnd, 'a'};
    std::vector<std::uint32_t> stop_rows(3 * words);
    done.fill_round(stop_first, stop_rows.data(), words);
    check(allowed(stop_rows.data(), kImEnd), "completed grammar admits stop");
    check(std::equal(stop_rows.begin() + static_cast<std::ptrdiff_t>(words),
                     stop_rows.begin() + static_cast<std::ptrdiff_t>(2 * words),
                     stop_rows.begin() + static_cast<std::ptrdiff_t>(2 * words)),
          "rows after an accepted stop repeat");
    const TokenId stop[] = {kImEnd};
    done.commit(stop);
    check(done.stats().complete, "stop completes the constraint");
}

// The constraint starts after the reasoning terminator, also inside one speculative window.
void reasoning_gate_contract(const ConstraintCompiler& compiler) {
    OutputConstraintSpec spec = schema_spec(Json::parse(R"({"type":"object","properties":{
        "a":{"type":"integer"}},"required":["a"],"additionalProperties":false})"));
    spec.follows_reasoning    = true;
    auto compiled             = compiler.compile(spec);
    RequestConstraint request(compiled, true);
    const std::size_t words = request.mask_words();

    std::vector<std::uint32_t> row(words);
    request.fill_round({}, row.data(), words);
    check(allowed(row.data(), 'x') && allowed(row.data(), kThinkClose) &&
              allowed(row.data(), kToolCall),
          "reasoning is unconstrained");
    check(!allowed(row.data(), kImEnd), "reasoning cannot end the request");

    // One window: reasoning byte, the terminator, blank line, object start.
    const std::vector<TokenId> drafts = {'x', kThinkClose, kDoubleNewline, '{'};
    std::vector<std::uint32_t> rows((drafts.size() + 1) * words);
    request.fill_round(drafts, rows.data(), words);
    const auto at = [&](std::size_t index) { return rows.data() + index * words; };
    check(allowed(at(1), kThinkClose) && !allowed(at(1), kImEnd), "row 1 still in reasoning");
    check(allowed(at(2), kDoubleNewline) && allowed(at(2), '{') && !allowed(at(2), 'x'),
          "row 2 follows the terminator: blank run or the object");
    check(allowed(at(3), '{') && !allowed(at(3), 'x'), "row 3 after the blank line");
    check(allowed(at(4), '"') && !allowed(at(4), '}'), "row 4 inside the object");

    const TokenId committed[] = {'x', kThinkClose, kDoubleNewline};
    request.commit(committed);
    walk_text(request, R"({"a":7})", "after reasoning");

    // The terminator completes inside a token whose tail is content.
    RequestConstraint mid(compiled, true);
    const TokenId close_then_text[] = {kCloseThenText};
    mid.commit(close_then_text);
    check(!mid.violated(), "tail after the terminator is matched");
    std::vector<std::uint32_t> tail_row(words);
    mid.fill_round({}, tail_row.data(), words);
    check(allowed(tail_row.data(), '"') && !allowed(tail_row.data(), '{'),
          "tail bytes advanced the grammar");
}

// A model that opens its reasoning itself (Gemma 4, gpt-oss, Qwen3 with thinking): before its
// first token a row admits the open marker beside the grammar; a token on both keeps both; the
// marker completed rolls the grammar back to the start, which resumes after the terminator and a
// short whitespace run. Speculative rounds leave the committed matcher as they found it.
void self_opened_reasoning_contract() {
    GrammarVocabulary words_with_open       = vocabulary();
    words_with_open.token_bytes[kThinkOpen] = "<think>";
    words_with_open.reasoning_open          = "<think>";
    const ConstraintCompiler compiler(words_with_open);
    OutputConstraintSpec choice;
    choice.kind         = OutputConstraintKind::Choice;
    choice.choices      = {"<tag>", "yes"};
    const auto compiled = compiler.compile(choice);

    RequestConstraint fresh(compiled, false, true);
    const std::size_t words = fresh.mask_words();
    std::vector<std::uint32_t> row(words);
    fresh.fill_round({}, row.data(), words);
    check(allowed(row.data(), kThinkOpen) && allowed(row.data(), '<') && allowed(row.data(), 'y'),
          "before the first token: the open marker and the grammar");
    check(!allowed(row.data(), 'x') && !allowed(row.data(), kImEnd),
          "before the first token: nothing else, no stop");

    RequestConstraint direct(compiled, false, true);
    walk_text(direct, "yes", "no reasoning");
    RequestConstraint shared_prefix(compiled, false, true);
    walk_text(shared_prefix, "<tag>", "text sharing the marker's first bytes");

    RequestConstraint reasoned(compiled, false, true);
    const TokenId open[] = {kThinkOpen, 'x', kThinkClose, '\n'};
    reasoned.commit(open);
    check(!reasoned.violated(), "a reasoning section the model opened is admitted");
    reasoned.fill_round({}, row.data(), words);
    check(allowed(row.data(), 'y') && allowed(row.data(), ' ') && !allowed(row.data(), 'x'),
          "after the terminator: the grammar or a short blank run");
    walk_text(reasoned, "yes", "after self-opened reasoning");

    // The marker spelled byte by byte: the grammar took '<' and 't', then gave way; it starts
    // over after the terminator.
    RequestConstraint spelled(compiled, false, true);
    const TokenId bytes[] = {'<', 't', 'h', 'i', 'n', 'k', '>', 'x', kThinkClose};
    spelled.commit(bytes);
    check(!spelled.violated(), "a marker spelled in bytes opens the reasoning");
    walk_text(spelled, "<tag>", "grammar from its start after a spelled marker");

    // One speculative window through the marker and the terminator.
    RequestConstraint window(compiled, false, true);
    const TokenId first[] = {'<'};
    window.commit(first);
    const std::vector<TokenId> drafts = {'t', 'h', 'i', 'n', 'k', '>', 'x', kThinkClose, 'y'};
    std::vector<std::uint32_t> rows((drafts.size() + 1) * words);
    window.fill_round(drafts, rows.data(), words);
    const auto at = [&](std::size_t index) { return rows.data() + index * words; };
    check(allowed(at(0), 't') && !allowed(at(0), 'y'), "after '<': marker or grammar 't'");
    check(allowed(at(2), 'i') && !allowed(at(2), 'a'), "after '<th': the marker only");
    check(allowed(at(6), 'x') && !allowed(at(6), kImEnd), "inside the reasoning");
    check(allowed(at(8), 'y') && allowed(at(8), '<') && !allowed(at(8), 'x') &&
              !allowed(at(8), kImEnd),
          "after the terminator in the same window: the grammar from its start");
    check(allowed(at(9), 'e') && !allowed(at(9), 'y'), "the grammar advances in the window");
    window.fill_round({}, row.data(), words);
    check(allowed(row.data(), 't') && !allowed(row.data(), 'y'),
          "a speculative round leaves the committed state");
    const TokenId rest[] = {'t', 'h', 'i', 'n', 'k', '>', 'x', kThinkClose};
    window.commit(rest);
    walk_text(window, "yes", "after a window through the marker");

    // A prompt that already holds its reasoning (thinking off) or a raw output: no marker.
    RequestConstraint closed(compiled, false, false);
    closed.fill_round({}, row.data(), words);
    check(!allowed(row.data(), kThinkOpen) && allowed(row.data(), 'y'),
          "no open marker where the model may not open reasoning");
}

void violation_contract(const ConstraintCompiler& compiler) {
    OutputConstraintSpec spec;
    spec.kind = OutputConstraintKind::JsonObject;
    RequestConstraint request(compiler.compile(spec), false);
    const TokenId bad[] = {'x'};
    request.commit(bad);
    check(request.violated() && !request.stats().complete, "disallowed commit is a violation");
    std::vector<std::uint32_t> rows(2 * request.mask_words());
    const TokenId draft[] = {'{'};
    request.fill_round(draft, rows.data(), request.mask_words());
    for (TokenId token = 0; token < static_cast<TokenId>(kDomain); ++token) {
        check(allowed(rows.data(), token) == (token == kImEnd),
              "violated request admits only stop, token " + std::to_string(token));
    }

    RequestConstraint faulted(compiler.compile(spec), false);
    faulted.record_device_fault();
    check(faulted.violated() && !faulted.stats().violation.empty(), "device fault is reported");
}

// A step's mask time is declared, never measured. A client's EBNF grammar, as a grammar or as a
// format of its structural tag, declares free-grammar rows (runtime::MaskRowClass::FreeGrammar)
// outside a reasoning section; every other constraint, the Engine's own tool-call markup among
// them, and a free grammar inside its reasoning declare shaped ones. The class follows the spec
// and the committed tokens only: no round a request computes moves it. The step rule the Program
// applies (runtime::plan_step_masks): a step of shaped rows overlaps its masks with its launch
// even at the widest shape, so its constrained DFlash rows draft; one free-grammar row, also
// beside shaped ones, fills the step's masks before its launch and stops their DFlash drafts.
void mask_row_class_contract(const ConstraintCompiler& compiler) {
    using runtime::MaskRowClass;
    using runtime::StepMaskRow;
    const auto class_of = [](const RequestConstraint& request) {
        return runtime::mask_row_class(request.free_grammar_rows());
    };

    OutputConstraintSpec grammar;
    grammar.kind       = OutputConstraintKind::Grammar;
    grammar.definition = R"EBNF(root ::= "call(" arg ("," arg)* ")"
arg ::= [0-9]+)EBNF";
    check(declares_free_grammar(grammar) && declares_free_grammar(*compiler.compile(grammar)),
          "an EBNF grammar declares shaped mask rows");

    // A client's structural tag around an EBNF format holds a free grammar; the same tag rendered
    // by the Engine (Gemma 4's tool-call markup holds its argument grammar) does not, and the
    // compile cache keeps the two apart.
    OutputConstraintSpec client_tag;
    client_tag.kind       = OutputConstraintKind::StructuralTag;
    client_tag.definition = R"({"type":"structural_tag","format":{"type":"tag","begin":"<t>",
        "content":{"type":"grammar","grammar":"root ::= [0-9]+"},"end":"</t>"}})";
    OutputConstraintSpec engine_tag = client_tag;
    engine_tag.engine_rendered      = true;
    check(declares_free_grammar(client_tag) && declares_free_grammar(*compiler.compile(client_tag)),
          "a client's structural tag around an EBNF format declares shaped mask rows");
    check(!declares_free_grammar(engine_tag) &&
              !declares_free_grammar(*compiler.compile(engine_tag)) &&
              declares_free_grammar(*compiler.compile(client_tag)),
          "the Engine's tool-call markup declares free-grammar mask rows, or shares the client's "
          "compiled tag");

    OutputConstraintSpec object;
    object.kind = OutputConstraintKind::JsonObject;
    OutputConstraintSpec choice;
    choice.kind    = OutputConstraintKind::Choice;
    choice.choices = {"yes", "no"};
    OutputConstraintSpec regex;
    regex.kind       = OutputConstraintKind::Regex;
    regex.definition = "[0-9]{3}-[a-z]+";
    // A structural tag of other formats is shaped, also where a value in its JSON Schema reads
    // like a grammar format.
    OutputConstraintSpec schema_tag;
    schema_tag.kind       = OutputConstraintKind::StructuralTag;
    schema_tag.definition = R"({"type":"structural_tag","format":{"type":"tag","begin":"<t>",
        "content":{"type":"json_schema","json_schema":{"type":"object","properties":{"k":{
        "type":"object","default":{"type":"grammar","grammar":"root ::= [0-9]+"}}}}},
        "end":"</t>"}})";
    check(!declares_free_grammar(schema_tag),
          "a JSON Schema value in a structural tag declares free-grammar mask rows");

    OutputConstraintSpec schema = schema_spec(Json::parse(R"({"type":"object",
        "properties":{"k":{"type":"boolean"}},"required":["k"],"additionalProperties":false})"));
    for (const OutputConstraintSpec* spec : {&object, &choice, &regex, &schema}) {
        check(!declares_free_grammar(*spec) && !declares_free_grammar(*compiler.compile(*spec)),
              "kind " + std::to_string(static_cast<int>(spec->kind)) +
                  " declares free-grammar mask rows");
    }

    // No round a request computes moves its class, nor does a commit outside a reasoning section.
    RequestConstraint free_request(compiler.compile(grammar), false);
    RequestConstraint shaped_request(compiler.compile(schema), false);
    const std::size_t words = free_request.mask_words();
    std::vector<std::uint32_t> rows(4 * words);
    const std::vector<TokenId> drafts = {'c', 'a', 'l'};
    for (int round = 0; round < 3; ++round) {
        free_request.fill_round(drafts, rows.data(), words);
        shaped_request.fill_round({}, rows.data(), words);
        check(class_of(free_request) == MaskRowClass::FreeGrammar &&
                  class_of(shaped_request) == MaskRowClass::Shaped,
              "a request's mask row class moved after round " + std::to_string(round));
    }
    free_request.commit(bytes_of("call(1"));
    check(class_of(free_request) == MaskRowClass::FreeGrammar, "a commit moved the class");

    // Inside a reasoning section a free grammar's rows copy the no-stop mask: shaped, also over a
    // round whose drafts cross the terminator, until a committed token passes it.
    RequestConstraint reasoning(compiler.compile(grammar), true);
    std::vector<std::uint32_t> crossing(4 * words);
    const std::vector<TokenId> across = {'x', kThinkClose, 'c'};
    reasoning.fill_round(across, crossing.data(), words);
    check(class_of(reasoning) == MaskRowClass::Shaped,
          "a free grammar inside its reasoning declares free-grammar mask rows");
    const TokenId thought[] = {'x'};
    reasoning.commit(thought);
    check(class_of(reasoning) == MaskRowClass::Shaped, "a reasoning token moved the class");
    const TokenId terminator[] = {kThinkClose};
    reasoning.commit(terminator);
    check(class_of(reasoning) == MaskRowClass::FreeGrammar,
          "a free grammar past its reasoning terminator declares shaped mask rows");

    // A model that opens its reasoning itself: the grammar's rows before its first token, none in
    // the reasoning it opened, the grammar's again from the gap after its terminator.
    GrammarVocabulary words_with_open       = vocabulary();
    words_with_open.token_bytes[kThinkOpen] = "<think>";
    words_with_open.reasoning_open          = "<think>";
    const ConstraintCompiler opening(words_with_open);
    RequestConstraint self_opened(opening.compile(grammar), false, true);
    check(class_of(self_opened) == MaskRowClass::FreeGrammar,
          "a free grammar before its first token declares shaped mask rows");
    const TokenId opener[] = {kThinkOpen};
    self_opened.commit(opener);
    check(class_of(self_opened) == MaskRowClass::Shaped,
          "a free grammar in the reasoning it opened declares free-grammar mask rows");
    self_opened.commit(terminator);
    check(!self_opened.violated() && class_of(self_opened) == MaskRowClass::FreeGrammar,
          "a free grammar in the gap after its reasoning declares shaped mask rows");

    // The step rule. The Program's widest launch verifies 15 drafts a row
    // (kDFlashDecodeMaximumDrafts and kMtpDecodeMaximumDrafts; decode.cpp asserts the same at
    // kMaximumConcurrency rows).
    constexpr std::uint32_t kWidestDrafts = 15;
    const StepMaskRow shaped_row{.row_class = class_of(shaped_request), .speculative = true};
    std::vector<StepMaskRow> step(kMaximumConcurrency, shaped_row);
    const runtime::StepMaskPlan dflash = runtime::plan_step_masks(step, false, true, kWidestDrafts);
    check(dflash.mask_ns == kMaximumConcurrency * runtime::kShapedMaskRowNs &&
              dflash.drafted_ns == kMaximumConcurrency * 16U * runtime::kShapedMaskRowNs &&
              !dflash.masks_first && dflash.dflash_masks_follow_drafts,
          "a DFlash step of shaped rows at the widest launch fills its masks first or does not "
          "draft");
    const runtime::StepMaskPlan mtp = runtime::plan_step_masks(step, true, true, kWidestDrafts);
    check(mtp.mask_ns == kMaximumConcurrency * 16U * runtime::kShapedMaskRowNs &&
              !mtp.masks_first && !mtp.dflash_masks_follow_drafts,
          "an MTP step of shaped rows at the widest launch fills its masks first");

    // A plain free-grammar row beside drafting shaped rows: the step's masks come first and its
    // DFlash rows verify without drafts. Inside its reasoning the same request lets them draft.
    step.back() = StepMaskRow{.row_class = class_of(free_request), .speculative = false};
    const runtime::StepMaskPlan beside = runtime::plan_step_masks(step, false, true, kWidestDrafts);
    check(beside.masks_first && !beside.dflash_masks_follow_drafts,
          "a free-grammar row beside drafting shaped rows overlaps the step's masks");
    RequestConstraint thinking(compiler.compile(grammar), true);
    step.back().row_class = class_of(thinking);
    const runtime::StepMaskPlan reasoning_step =
        runtime::plan_step_masks(step, false, true, kWidestDrafts);
    check(!reasoning_step.masks_first && reasoning_step.dflash_masks_follow_drafts,
          "a free grammar inside its reasoning stops its step's DFlash drafts");

    // Nothing drafts without a speculative row, and a step without a constrained row has no masks.
    const std::vector<StepMaskRow> plain_only = {
        StepMaskRow{.row_class = MaskRowClass::Shaped, .speculative = false}};
    check(!runtime::plan_step_masks(plain_only, false, false, kWidestDrafts)
               .dflash_masks_follow_drafts,
          "a step without a speculative row drafts its constrained rows");
    const runtime::StepMaskPlan none = runtime::plan_step_masks({}, false, true, kWidestDrafts);
    check(none.mask_ns == 0 && !none.masks_first && !none.dflash_masks_follow_drafts,
          "a step without a constrained row plans masks");
    check(runtime::masks_overlap_launch(runtime::kLaunchMaskOverlapNs) &&
              !runtime::masks_overlap_launch(runtime::kLaunchMaskOverlapNs + 1),
          "the overlap bound is not inclusive");
}

void visible_special_contract(const ConstraintCompiler& compiler) {
    OutputConstraintSpec spec;
    spec.kind                   = OutputConstraintKind::Regex;
    spec.definition             = "<tool_call>[a-z]+";
    spec.special_tokens_visible = true;
    RequestConstraint visible(compiler.compile(spec), false);
    std::vector<std::uint32_t> row(visible.mask_words());
    visible.fill_round({}, row.data(), row.size());
    check(allowed(row.data(), kToolCall), "visible special token matches its bytes");
    spec.special_tokens_visible = false;
    RequestConstraint hidden(compiler.compile(spec), false);
    hidden.fill_round({}, row.data(), row.size());
    check(!allowed(row.data(), kToolCall) && allowed(row.data(), '<'),
          "hidden special token is never admitted");
}

} // namespace

// An integer without a declared bound (on either side) must not admit an endless digit run: the
// open side closes at the I-JSON limit 2^53-1, so the grammar ends after 16 digits and every
// exactly representable value stays admitted. multipleOf keeps its open sides.
void integer_digit_bound_contract(const ConstraintCompiler& compiler) {
    const Json normalized =
        normalize_supported_json_schema(Json::parse(R"({"type":"object","properties":{
        "free":{"type":"integer"},"low":{"type":"integer","minimum":0},
        "high":{"type":"integer","exclusiveMaximum":10},"nullable":{"type":["integer","null"]},
        "step":{"type":"integer","multipleOf":3},"num":{"type":"number"}}})"));
    const Json& p = normalized["properties"];
    check(p["free"]["minimum"] == -9007199254740991LL && p["free"]["maximum"] == 9007199254740991LL,
          "an unbounded integer is closed at +-(2^53-1)");
    check(p["low"]["minimum"] == 0 && p["low"]["maximum"] == 9007199254740991LL,
          "a declared lower bound is kept and the open upper side closed");
    check(p["high"]["exclusiveMaximum"] == 10 && !p["high"].contains("maximum") &&
              p["high"]["minimum"] == -9007199254740991LL,
          "a declared exclusive upper bound is kept and the open lower side closed");
    check(p["nullable"]["maximum"] == 9007199254740991LL, "integer-or-null is bounded too");
    check(!p["step"].contains("minimum") && !p["step"].contains("maximum"),
          "multipleOf keeps its open sides");
    check(!p["num"].contains("minimum"), "number is not given integer bounds");

    for (const char* schema : {R"({"type":"integer"})", R"({"type":"integer","minimum":0})"}) {
        const auto compiled = compiler.compile(schema_spec(Json::parse(schema)));
        // The limit itself and sixteen digits below it are admitted and complete; no further
        // digit is admitted after either.
        for (const char* digits : {"9007199254740991", "1234567890123456"}) {
            RequestConstraint request(compiled, false);
            walk_text(request, digits, std::string(schema) + " " + digits, false);
            RequestConstraint again(compiled, false);
            again.commit(bytes_of(digits));
            std::vector<std::uint32_t> row(again.mask_words());
            again.fill_round({}, row.data(), row.size());
            bool digit = false;
            for (char c = '0'; c <= '9'; ++c) { digit = digit || allowed(row.data(), c); }
            check(!digit, std::string(schema) + ": a digit is admitted after " + digits);
        }
    }
    {
        RequestConstraint request(compiler.compile(schema_spec(Json::parse(R"({"type":"integer"})"))),
                                  false);
        walk_text(request, "-9007199254740991", "negative I-JSON limit", false);
    }
}

// JSON inside a structural tag (tool-call arguments) has its whitespace runs bounded like a JSON
// response format (CompilerOptions::json_whitespace_run, 24 by default).
void structural_tag_whitespace_contract(const ConstraintCompiler& compiler) {
    OutputConstraintSpec spec;
    spec.kind           = OutputConstraintKind::StructuralTag;
    spec.definition     = R"({"type":"structural_tag","format":{"type":"tag","begin":"<t>",
        "content":{"type":"json_schema","json_schema":{"type":"object"}},"end":"</t>"}})";
    const auto compiled = compiler.compile(spec);
    for (const int run : {23, 24}) {
        RequestConstraint request(compiled, false);
        request.commit(bytes_of("<t>{" + std::string(static_cast<std::size_t>(run), ' ')));
        std::vector<std::uint32_t> row(request.mask_words());
        request.fill_round({}, row.data(), row.size());
        check(allowed(row.data(), ' ') == (run < 24) && allowed(row.data(), '}'),
              "structural tag JSON whitespace run of " + std::to_string(run));
    }
}

// response_format beside tools: the output is the call markup or a JSON answer in the format, and
// nothing else; the answer schema enters the cache key.
void answer_schema_contract(const ConstraintCompiler& compiler) {
    OutputConstraintSpec spec;
    spec.kind           = OutputConstraintKind::StructuralTag;
    spec.definition     = R"({"type":"structural_tag","format":{"type":"tag","begin":"<t>",
        "content":{"type":"json_schema","json_schema":{"type":"object"}},"end":"</t>"}})";
    spec.compact_json   = true;
    spec.answer_schema  = R"({"type":"object","properties":{"x":{"type":"integer"}},
        "required":["x"],"additionalProperties":false})";
    const auto compiled = compiler.compile(spec);
    {
        RequestConstraint request(compiled, false);
        walk_text(request, "<t>{}</t>", "answer schema: the call markup");
    }
    {
        RequestConstraint request(compiled, false);
        walk_text(request, R"({"x":7})", "answer schema: the JSON answer");
    }
    {
        RequestConstraint request(compiled, false);
        std::vector<std::uint32_t> row(request.mask_words());
        request.fill_round({}, row.data(), row.size());
        check(!allowed(row.data(), 'h') && allowed(row.data(), '{') && allowed(row.data(), '<'),
              "answer schema: free text admitted beside the markup and the JSON answer");
        request.commit(bytes_of(R"({"y")"));
        request.fill_round({}, row.data(), row.size());
        check(request.stats().violated || !allowed(row.data(), ':'),
              "answer schema: an answer outside the schema admitted");
    }
    OutputConstraintSpec plain = spec;
    plain.answer_schema.clear();
    check(compiler.compile(plain) != compiled, "the answer schema is not part of the cache key");
    OutputConstraintSpec misplaced = schema_spec(Json::parse(R"({"type":"object"})"));
    misplaced.answer_schema        = R"({"type":"object"})";
    check(expect_error(compiler, misplaced) == "invalid_tool_constraint",
          "an answer schema outside tool-call markup was accepted");
}

// A strict tool's arguments are one JSON object in every tool-call syntax: the root is held to
// one (an untyped plain root becomes an object, a combinator root is left to its branches) and
// another declared type is refused.
void tool_parameters_contract() {
    const auto code = [](const Json& schema) -> std::string {
        try {
            (void)normalize_tool_parameters_schema(schema);
        } catch (const OutputConstraintError& error) { return error.code(); }
        return "ok";
    };
    check(code(Json::parse(R"({"type":"string"})")) == "invalid_schema", "string-rooted tool");
    check(code(Json::parse(R"({"type":["object","null"]})")) == "invalid_schema",
          "nullable-rooted tool");
    check(code(Json::parse(R"({"type":["object"]})")) == "ok", "object type list");
    check(normalize_tool_parameters_schema(Json::object()).value("type", "") == "object",
          "an empty tool schema describes an object");
    check(normalize_tool_parameters_schema(Json::parse(R"({"properties":{"a":{}}})"))
                  .value("type", "") == "object",
          "an untyped tool schema with properties describes an object");
    check(!normalize_tool_parameters_schema(
               Json::parse(R"({"anyOf":[{"type":"object","properties":{"a":{"type":"string"}}},
                 {"type":"object","properties":{"b":{"type":"string"}}}]})"))
               .contains("type"),
          "a combinator root is left to its branches");
}

int main() {
    const ConstraintCompiler compiler(vocabulary());
    tool_parameters_contract();
    integer_digit_bound_contract(compiler);
    schema_policy_contract();
    compile_error_contract(compiler);
    busy_contract();
    json_walk_contract(compiler);
    other_kinds_contract(compiler);
    structural_tag_whitespace_contract(compiler);
    answer_schema_contract(compiler);
    speculative_rows_contract(compiler);
    reasoning_gate_contract(compiler);
    self_opened_reasoning_contract();
    violation_contract(compiler);
    mask_row_class_contract(compiler);
    visible_special_contract(compiler);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " structured output (" << failures
              << " failures)\n";
    return failures == 0 ? 0 : 1;
}
