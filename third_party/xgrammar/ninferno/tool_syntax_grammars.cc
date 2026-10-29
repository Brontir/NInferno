// NInferno-owned file (not upstream): see tool_syntax_grammars.h.

#include "tool_syntax_grammars.h"

#include <picojson.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "grammar_functor.h"
#include "json_schema_converter.h"
#include "json_schema_converter_ext.h"
#include "support/encoding.h"
#include "support/json_parse.h"

namespace ninferno_xgrammar {
namespace {

using xgrammar::JSONFormat;
using xgrammar::JSONSchemaConverter;
using xgrammar::SchemaSpecPtr;
using Range = xgrammar::GrammarBuilder::CharacterClassElement;

// Code points of a UTF-8 string; nullopt when it is not valid UTF-8.
std::optional<std::vector<int32_t>> codepoints(const std::string& text) {
    std::vector<int32_t> out;
    for (std::size_t at = 0; at < text.size();) {
        const auto byte = static_cast<unsigned char>(text[at]);
        int length      = 0;
        int32_t value   = 0;
        if (byte < 0x80) {
            length = 1;
            value  = byte;
        } else if ((byte & 0xE0) == 0xC0) {
            length = 2;
            value  = byte & 0x1F;
        } else if ((byte & 0xF0) == 0xE0) {
            length = 3;
            value  = byte & 0x0F;
        } else if ((byte & 0xF8) == 0xF0) {
            length = 4;
            value  = byte & 0x07;
        } else {
            return std::nullopt;
        }
        if (at + length > text.size()) { return std::nullopt; }
        for (int i = 1; i < length; ++i) {
            const auto next = static_cast<unsigned char>(text[at + i]);
            if ((next & 0xC0) != 0x80) { return std::nullopt; }
            value = (value << 6) | (next & 0x3F);
        }
        out.push_back(value);
        at += length;
    }
    return out;
}

// `ranges` (sorted, disjoint) without the code points in `removed`.
std::vector<Range> subtract(const std::vector<Range>& ranges, std::vector<int32_t> removed) {
    std::sort(removed.begin(), removed.end());
    std::vector<Range> out;
    for (const Range& range : ranges) {
        int32_t lower = range.lower;
        for (const int32_t point : removed) {
            if (point < lower || point > range.upper) { continue; }
            if (point > lower) { out.push_back({lower, point - 1}); }
            lower = point + 1;
        }
        if (lower <= range.upper) { out.push_back({lower, range.upper}); }
    }
    return out;
}

bool admits(const std::vector<Range>& ranges, int32_t point) {
    return std::any_of(ranges.begin(), ranges.end(), [point](const Range& range) {
        return point >= range.lower && point <= range.upper;
    });
}

// A converter that can spell "a key over these code points, none of these names": the additional
// keys of an object whose declared keys must not be written a second time through them.
template <class Base>
class KeyedConverter : public Base {
public:
    using Base::Base;

protected:
    // `end`: what follows a key, asserted after it so that its tokens settle at compile time.
    int32_t keys_excluding(const std::vector<std::string>& names, const std::vector<Range>& first,
                           const std::vector<Range>& rest, const std::string& hint, int32_t end) {
        struct Node {
            std::map<int32_t, std::size_t> children;
            bool terminal = false;
        };
        std::vector<Node> nodes(1);
        for (const std::string& name : names) {
            const std::optional<std::vector<int32_t>> points = codepoints(name);
            if (!points || points->empty()) { continue; }
            bool spellable = true;
            for (std::size_t i = 0; i < points->size(); ++i) {
                spellable = spellable && admits(i == 0 ? first : rest, (*points)[i]);
            }
            if (!spellable) { continue; }
            std::size_t at = 0;
            for (const int32_t point : *points) {
                const auto found = nodes[at].children.find(point);
                if (found != nodes[at].children.end()) {
                    at = found->second;
                } else {
                    nodes.emplace_back();
                    nodes[at].children.emplace(point, nodes.size() - 1);
                    at = nodes.size() - 1;
                }
            }
            nodes[at].terminal = true;
        }
        const int32_t tail = this->builder_.AddCharacterClassStar(rest);
        std::vector<int32_t> rules(nodes.size());
        for (std::size_t index = 0; index < nodes.size(); ++index) {
            rules[index] = this->builder_.AddEmptyRuleWithHint(hint + "_" + std::to_string(index));
        }
        for (std::size_t index = 0; index < nodes.size(); ++index) {
            const Node& node                 = nodes[index];
            const std::vector<Range>& admitted = index == 0 ? first : rest;
            std::vector<int32_t> choices;
            std::vector<int32_t> taken;
            for (const auto& [point, child] : node.children) {
                taken.push_back(point);
                choices.push_back(
                    this->Sequence({this->ByteString(xgrammar::CharToUTF8(point)), this->RuleRef(rules[child])}));
            }
            const std::vector<Range> others = subtract(admitted, taken);
            if (!others.empty()) {
                choices.push_back(this->Sequence({this->builder_.AddCharacterClass(others), tail}));
            }
            if (index != 0 && !node.terminal) { choices.push_back(this->Empty()); }
            this->builder_.UpdateRuleBody(rules[index], this->Choice(choices));
        }
        this->builder_.UpdateLookaheadAssertion(rules[0], end);
        return this->RuleRef(rules[0]);
    }

    static std::vector<std::string> names_of(const std::vector<xgrammar::ObjectSpec::Property>& p) {
        std::vector<std::string> out;
        out.reserve(p.size());
        for (const auto& property : p) { out.push_back(property.name); }
        return out;
    }
};

// ---------------------------------------------------------------------------------------------
// Gemma 4

const std::string kGemmaQuote = "<|\"|>";
const std::string kGemmaKey   = "ninferno_gemma_key";

// A bare key: what Gemma 4's template writes for a schema's property names. Other keys are written
// <|"|>-quoted, which Gemma4Syntax::Reader reads as well. Keeping bare keys to these characters
// keeps their masks on the fast path (any character but the delimiters cost up to 30 ms per mask at
// a key's start on Gemma 4's 262k vocabulary).
const std::vector<Range> kGemmaBareKey = {{'$', '$'}, {'-', '.'}, {'0', '9'}, {'@', 'Z'},
                                          {'_', '_'}, {'a', 'z'}};

bool gemma_bare_key(const std::string& key) {
    if (key.empty()) { return false; }
    return std::all_of(key.begin(), key.end(),
                       [](char c) { return admits(kGemmaBareKey, static_cast<unsigned char>(c)); });
}

std::string gemma_key_text(const std::string& key) {
    return gemma_bare_key(key) ? key : kGemmaQuote + key + kGemmaQuote;
}

// A JSON value written the way Gemma 4 writes it.
std::string gemma_literal(const picojson::value& value) {
    if (value.is<std::string>()) {
        if (value.get<std::string>().find(kGemmaQuote) != std::string::npos) {
            throw std::invalid_argument("a const or enum string holds Gemma's quote <|\"|>");
        }
        return kGemmaQuote + value.get<std::string>() + kGemmaQuote;
    }
    if (value.is<picojson::array>()) {
        std::string out = "[";
        bool first      = true;
        for (const picojson::value& item : value.get<picojson::array>()) {
            out += (first ? "" : ",") + gemma_literal(item);
            first = false;
        }
        return out + "]";
    }
    if (value.is<picojson::object>()) {
        std::string out = "{";
        bool first      = true;
        for (const auto& [key, item] : value.get<picojson::object>()) {
            out += (first ? "" : ",") + gemma_key_text(key) + ":" + gemma_literal(item);
            first = false;
        }
        return out + "}";
    }
    return value.serialize();
}

std::string gemma_literal(const std::string& json) {
    picojson::value value;
    const std::string error = xgrammar::ParseJSON(value, json);
    if (!error.empty()) { throw std::invalid_argument("a const or enum value is not JSON"); }
    return gemma_literal(value);
}

class Gemma4Converter final : public KeyedConverter<JSONSchemaConverter> {
public:
    Gemma4Converter(int max_whitespace, RefResolver resolver)
        : KeyedConverter<JSONSchemaConverter>(std::nullopt, std::nullopt, true, max_whitespace,
                                              std::move(resolver)) {}

protected:
    void AddBasicRules() override {
        JSONSchemaConverter::AddBasicRules({kGemmaKey});
        // A quoted string: any text up to the closing <|"|> (Reader::quoted takes the first one).
        // A tag dispatch, as XGrammar's XML strings, keeps its masks on the fast path.
        builder_.UpdateRuleBody(kBasicString,
                                Sequence({ByteString(kGemmaQuote), TagDispatch(false, {kGemmaQuote}),
                                          ByteString(kGemmaQuote)}));
        const std::vector<Range>& bare = kGemmaBareKey;
        const int32_t bare_key        = builder_.AddRuleWithHint(
            "ninferno_gemma_bare_key",
            Sequence({builder_.AddCharacterClass(bare), builder_.AddCharacterClassStar(bare)}));
        // What follows a bare key: a run of almost any characters settles its tokens only so.
        builder_.UpdateLookaheadAssertion(bare_key, KeyEnd());
        builder_.UpdateRuleBody(kGemmaKey, Choice({RuleRef(bare_key), RuleRef(kBasicString)}));
    }

    int32_t KeyEnd() { return Sequence({WhitespaceExpression(), ByteString(":")}); }

    int32_t GenerateString(const xgrammar::StringSpec& spec, const std::string& rule_name) override {
        if (!spec.pattern.has_value() && !spec.format.has_value() && spec.min_length == 0 &&
            spec.max_length == -1) {
            return RuleRef(kBasicString);
        }
        int32_t body = -1;
        if (spec.format.has_value()) {
            if (const auto regex = JSONFormatToRegexPattern(*spec.format)) {
                body = RegexExpression(*regex, false, true);
            }
        }
        if (body == -1 && spec.pattern.has_value()) {
            body = RegexExpression(*spec.pattern, false, true);
        }
        if (body == -1) {
            // Length in code points; '<' is left out so no run of them can form the closing quote.
            body = Repeat(rule_name + "_characters", builder_.AddCharacterClass({{'<', '<'}}, true),
                          static_cast<int32_t>(spec.min_length),
                          static_cast<int32_t>(spec.max_length));
        }
        return Sequence({ByteString(kGemmaQuote), body, ByteString(kGemmaQuote)});
    }

    int32_t GenerateConst(const xgrammar::ConstSpec& spec, const std::string&) override {
        return ByteString(gemma_literal(spec.json_value));
    }

    int32_t GenerateEnum(const xgrammar::EnumSpec& spec, const std::string&) override {
        std::vector<int32_t> values;
        for (const std::string& value : spec.json_values) {
            values.push_back(ByteString(gemma_literal(value)));
        }
        return Choice(values);
    }

    int32_t FormatPropertyKey(const std::string& key, const SchemaSpecPtr&) override {
        if (key.find(kGemmaQuote) != std::string::npos) {
            throw std::invalid_argument("argument name '" + key + "' holds Gemma's quote <|\"|>");
        }
        return ByteString(gemma_key_text(key));
    }

    std::string GetKeyPattern() const override { return kGemmaKey; }

    int32_t GetKeyPatternExcluding(const std::vector<xgrammar::ObjectSpec::Property>& properties,
                                   const std::string& rule_name) override {
        // Bare keys only: a quoted one could spell a declared name again.
        const std::vector<Range>& bare = kGemmaBareKey;
        return keys_excluding(names_of(properties), bare, bare, rule_name + "_addl_key", KeyEnd());
    }
};

// ---------------------------------------------------------------------------------------------
// Pythonic

const std::string kPythonIdentifier = "ninferno_python_identifier";
const std::vector<Range> kIdentifierFirst = {{'A', 'Z'}, {'_', '_'}, {'a', 'z'}};
const std::vector<Range> kIdentifierRest  = {{'0', '9'}, {'A', 'Z'}, {'_', '_'}, {'a', 'z'}};

bool python_identifier(const std::string& name) {
    if (name.empty()) { return false; }
    for (std::size_t i = 0; i < name.size(); ++i) {
        if (!admits(i == 0 ? kIdentifierFirst : kIdentifierRest,
                    static_cast<unsigned char>(name[i]))) {
            return false;
        }
    }
    return true;
}

// The arguments object is NAME(key=value, ...)'s keyword list (level 1, no braces); every value,
// objects included, is JSON.
class PythonicConverter final : public KeyedConverter<JSONSchemaConverter> {
public:
    PythonicConverter(int max_whitespace, RefResolver resolver)
        : KeyedConverter<JSONSchemaConverter>(std::nullopt, std::nullopt, true, max_whitespace,
                                              std::move(resolver)) {}

protected:
    void AddBasicRules() override {
        // The basic rules are values: build them as nested (JSON, with braces).
        level_ = 1;
        JSONSchemaConverter::AddBasicRules({kPythonIdentifier});
        builder_.UpdateRuleBody(kPythonIdentifier,
                                Sequence({builder_.AddCharacterClass(kIdentifierFirst),
                                          builder_.AddCharacterClassStar(kIdentifierRest)}));
        level_ = 0;
    }

    int32_t GenerateObject(const xgrammar::ObjectSpec& spec, const std::string& rule_name,
                           bool) override {
        if (level_ == 0) {
            for (const auto& property : spec.properties) {
                if (!python_identifier(property.name)) {
                    throw std::invalid_argument("argument name '" + property.name +
                                                "' is not a Python identifier");
                }
            }
        }
        ++level_;
        const int32_t result = JSONSchemaConverter::GenerateObject(spec, rule_name, level_ > 1);
        --level_;
        return result;
    }

    // The root is the keyword list: never shared with a JSON object value through the cache.
    void AddCache(const std::string& key, int32_t rule_id) override {
        if (level_ > 0) { JSONSchemaConverter::AddCache(key, rule_id); }
    }
    std::optional<int32_t> GetCache(const std::string& key) const override {
        return level_ > 0 ? JSONSchemaConverter::GetCache(key) : std::nullopt;
    }

    int32_t FormatPropertyKey(const std::string& key, const SchemaSpecPtr& schema) override {
        return level_ == 1 ? ByteString(key) : JSONSchemaConverter::FormatPropertyKey(key, schema);
    }

    int32_t FormatProperty(const std::string& key, int32_t value_rule_id,
                           const std::string& rule_name, int64_t idx,
                           const SchemaSpecPtr& schema) override {
        if (level_ != 1) {
            return JSONSchemaConverter::FormatProperty(key, value_rule_id, rule_name, idx, schema);
        }
        return Sequence({ByteString(key + "="), RuleRef(value_rule_id)});
    }

    int32_t FormatOtherProperty(int32_t key_pattern_expr, int32_t value_rule_id,
                                const std::string& rule_name, const std::string& suffix,
                                const SchemaSpecPtr& schema) override {
        if (level_ != 1) {
            return JSONSchemaConverter::FormatOtherProperty(key_pattern_expr, value_rule_id,
                                                            rule_name, suffix, schema);
        }
        return Sequence({key_pattern_expr, ByteString("="), RuleRef(value_rule_id)});
    }

    std::string GetKeyPattern() const override {
        return level_ == 1 ? kPythonIdentifier : JSONSchemaConverter::GetKeyPattern();
    }

    int32_t GetKeyPatternExcluding(const std::vector<xgrammar::ObjectSpec::Property>& properties,
                                   const std::string& rule_name) override {
        if (level_ != 1) {
            return JSONSchemaConverter::GetKeyPatternExcluding(properties, rule_name);
        }
        return keys_excluding(names_of(properties), kIdentifierFirst, kIdentifierRest,
                              rule_name + "_addl_key", ByteString("="));
    }

private:
    int level_ = 0;
};

// ---------------------------------------------------------------------------------------------
// ATEM: MiniMax's XML parameters under the atem: prefix.

const std::string kAtemParameter = "<atem:parameter name=\"";
const std::string kAtemNameEnd   = "\">";
const std::string kAtemClose     = "</atem:parameter>";

class AtemConverter final : public KeyedConverter<xgrammar::XMLToolCallingConverter> {
public:
    AtemConverter(int max_whitespace, RefResolver resolver)
        : KeyedConverter<xgrammar::XMLToolCallingConverter>(
              std::nullopt, std::nullopt, true, max_whitespace, std::move(resolver),
              JSONFormat::kMiniMaxXML) {}

protected:
    void AddBasicRules() override {
        XMLToolCallingConverter::AddBasicRules();
        builder_.UpdateRuleBody(kXMLString, TagDispatch(false, {kAtemClose}));
    }

    // A parameter value never holds the close tag, where the parser ends it: a bounded string
    // leaves out '<', a const or enum string holding the tag is refused. (A pattern or format is
    // the client's regex; one that admits the tag is not caught.)
    int32_t GenerateString(const xgrammar::StringSpec& spec, const std::string& rule_name) override {
        if (nested_object_level_ <= 1 && !spec.pattern.has_value() && !spec.format.has_value() &&
            (spec.min_length != 0 || spec.max_length != -1)) {
            return Repeat(rule_name + "_characters", builder_.AddCharacterClass({{'<', '<'}}, true),
                          static_cast<int32_t>(spec.min_length),
                          static_cast<int32_t>(spec.max_length));
        }
        return XMLToolCallingConverter::GenerateString(spec, rule_name);
    }
    int32_t GenerateConst(const xgrammar::ConstSpec& spec, const std::string& rule_name) override {
        refuse_close(spec.json_value);
        return XMLToolCallingConverter::GenerateConst(spec, rule_name);
    }
    int32_t GenerateEnum(const xgrammar::EnumSpec& spec, const std::string& rule_name) override {
        for (const std::string& value : spec.json_values) { refuse_close(value); }
        return XMLToolCallingConverter::GenerateEnum(spec, rule_name);
    }
    void refuse_close(const std::string& json_value) const {
        if (nested_object_level_ <= 1 && json_value.find(kAtemClose) != std::string::npos) {
            throw std::invalid_argument("a const or enum value holds " + kAtemClose);
        }
    }

    int32_t FormatPropertyKey(const std::string& key, const SchemaSpecPtr& schema) override {
        if (nested_object_level_ > 1) {
            return XMLToolCallingConverter::FormatPropertyKey(key, schema);
        }
        if (key.find('"') != std::string::npos) {
            throw std::invalid_argument("argument name '" + key + "' holds a quote mark");
        }
        return ByteString(kAtemParameter + key + kAtemNameEnd);
    }

    int32_t FormatProperty(const std::string& key, int32_t value_rule_id,
                           const std::string& rule_name, int64_t idx,
                           const SchemaSpecPtr& schema) override {
        if (nested_object_level_ > 1) {
            return XMLToolCallingConverter::FormatProperty(key, value_rule_id, rule_name, idx,
                                                           schema);
        }
        return parameter(FormatPropertyKey(key, schema), value_rule_id);
    }

    int32_t FormatOtherProperty(int32_t key_pattern_expr, int32_t value_rule_id,
                                const std::string& rule_name, const std::string& suffix,
                                const SchemaSpecPtr& schema) override {
        if (nested_object_level_ > 1) {
            return XMLToolCallingConverter::FormatOtherProperty(key_pattern_expr, value_rule_id,
                                                                rule_name, suffix, schema);
        }
        return parameter(
            Sequence({ByteString(kAtemParameter), key_pattern_expr, ByteString(kAtemNameEnd)}),
            value_rule_id);
    }

    // An additional key never spells a declared name (that would write it a second time).
    int32_t GetKeyPatternExcluding(const std::vector<xgrammar::ObjectSpec::Property>& properties,
                                   const std::string& rule_name) override {
        if (nested_object_level_ > 1) {
            return XMLToolCallingConverter::GetKeyPatternExcluding(properties, rule_name);
        }
        return keys_excluding(names_of(properties), kIdentifierFirst, kIdentifierRest,
                              rule_name + "_addl_key", ByteString(kAtemNameEnd));
    }

private:
    int32_t parameter(int32_t key, int32_t value_rule_id) {
        // The value is the text between the tags as the parser keeps it (ValueFraming::AsIs): no
        // whitespace around it, or a constrained string would read back with it.
        return Sequence({key, RuleRef(value_rule_id), ByteString(kAtemClose)});
    }
};

// The arguments root every syntax here writes: an object of declared properties. A keyword list
// (Pythonic) or a parameter list (ATEM) has no braces, so a reference back to the root, or keys by
// pattern at the root, cannot be written there.
void check_root(const std::string& schema, bool list_root) {
    picojson::value value;
    if (!xgrammar::ParseJSON(value, schema).empty() || !value.is<picojson::object>()) {
        throw std::invalid_argument("the arguments schema is not a JSON object");
    }
    const picojson::object& root = value.get<picojson::object>();
    const auto type              = root.find("type");
    const bool object_type =
        type != root.end() &&
        ((type->second.is<std::string>() && type->second.get<std::string>() == "object") ||
         (type->second.is<picojson::array>() && type->second.get<picojson::array>().size() == 1 &&
          type->second.get<picojson::array>()[0].is<std::string>() &&
          type->second.get<picojson::array>()[0].get<std::string>() == "object"));
    if (!object_type) {
        throw std::invalid_argument("the arguments schema must be a plain {\"type\": \"object\"}");
    }
    if (!list_root) { return; }
    for (const char* key : {"patternProperties", "propertyNames", "$ref", "anyOf", "oneOf", "allOf"}) {
        if (root.count(key) != 0) {
            throw std::invalid_argument(std::string("\"") + key +
                                        "\" at the arguments root cannot be written");
        }
    }
    const auto self = [](const auto& recurse, const picojson::value& node) -> bool {
        if (node.is<picojson::array>()) {
            for (const picojson::value& item : node.get<picojson::array>()) {
                if (recurse(recurse, item)) { return true; }
            }
        } else if (node.is<picojson::object>()) {
            for (const auto& [key, item] : node.get<picojson::object>()) {
                if (key == "$ref" && item.is<std::string>() && item.get<std::string>() == "#") {
                    return true;
                }
                if (recurse(recurse, item)) { return true; }
            }
        }
        return false;
    };
    if (self(self, value)) {
        throw std::invalid_argument("a reference to the arguments root (\"$ref\": \"#\")");
    }
}

template <class Converter>
std::string ebnf(const std::string& schema, JSONFormat parse_format, int max_whitespace) {
    const xgrammar::Grammar grammar = xgrammar::JSONSchemaToGrammarWithConverter(
        schema, parse_format, [max_whitespace](JSONSchemaConverter::RefResolver resolver) {
            return std::unique_ptr<JSONSchemaConverter>(
                new Converter(max_whitespace, std::move(resolver)));
        });
    return xgrammar::GrammarNormalizer::Apply(grammar).ToString();
}

} // namespace

std::string gemma4_arguments_ebnf(const std::string& schema, int max_whitespace) {
    check_root(schema, false);
    return ebnf<Gemma4Converter>(schema, JSONFormat::kJSON, max_whitespace);
}

std::string pythonic_arguments_ebnf(const std::string& schema, int max_whitespace) {
    check_root(schema, true);
    return ebnf<PythonicConverter>(schema, JSONFormat::kJSON, max_whitespace);
}

std::string atem_arguments_ebnf(const std::string& schema, int max_whitespace) {
    check_root(schema, true);
    return ebnf<AtemConverter>(schema, JSONFormat::kMiniMaxXML, max_whitespace);
}

} // namespace ninferno_xgrammar
