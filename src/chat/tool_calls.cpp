#include "chat/tool_calls.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninferno::chat {
namespace {

using Json                = nlohmann::json;
using Contract            = ToolCallOutputContract;
using FallbackReason      = ToolCallParseFallbackReason;
using NormalizationPolicy = Contract::NormalizationPolicy;
using SchemaType          = Contract::SchemaType;
using TypeSet             = Contract::TypeSet;

constexpr std::string_view kToolOpen      = "<tool_call>";
constexpr std::string_view kToolClose     = "</tool_call>";
constexpr std::string_view kFunctionOpen  = "<function=";
constexpr std::string_view kFunctionClose = "</function>";
constexpr std::string_view kParamOpen     = "<parameter=";
constexpr std::string_view kParamClose    = "</parameter>";

struct RawParameter {
    std::string_view name;
    std::string_view value;
};

struct RawToolCall {
    std::string_view name;
    std::vector<RawParameter> parameters;
    std::uint32_t repeated_parameters = 0; // repeats set_parameter folded into an earlier one
    // The arguments when the function element holds one JSON object instead of parameters.
    std::optional<std::string> json_arguments;
};

// A region's calls and, when complete calls are followed by text (TrailingContent), where that
// text starts in the region.
struct RawCalls : std::vector<RawToolCall> {
    std::size_t trailing = 0;
};

// A repeated parameter keeps its last value, as a repeated key does in a JSON object, rather than
// dropping an otherwise well-formed call: models do repeat one, with a changed value.
void set_parameter(RawToolCall& call, std::string_view name, std::string_view value) {
    const auto existing = std::find_if(call.parameters.begin(), call.parameters.end(),
                                       [&](const RawParameter& p) { return p.name == name; });
    if (existing == call.parameters.end()) {
        call.parameters.push_back(RawParameter{.name = name, .value = value});
        return;
    }
    existing->value = value;
    ++call.repeated_parameters;
}

enum class JsonValueKind : std::uint8_t {
    Null,
    Boolean,
    Integer,
    Number,
    String,
    Object,
    Array,
};

enum class ParameterNormalization : std::uint8_t {
    Emitted,
    Omitted,
    SchemaMismatch,
};

struct NormalizedParameter {
    ParameterNormalization disposition = ParameterNormalization::Emitted;
    std::string json_value;
};

constexpr bool is_format_whitespace(char byte) {
    return byte == ' ' || byte == '\t' || byte == '\r' || byte == '\n';
}

constexpr bool is_ascii_digit(char byte) { return byte >= '0' && byte <= '9'; }

constexpr bool is_ascii_alphanumeric(char byte) {
    return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') || is_ascii_digit(byte);
}

std::string_view trim_format_whitespace(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && is_format_whitespace(text[begin])) { ++begin; }
    std::size_t end = text.size();
    while (end > begin && is_format_whitespace(text[end - 1])) { --end; }
    return text.substr(begin, end - begin);
}

std::string rtrim_format_whitespace(std::string_view text) {
    std::size_t end = text.size();
    while (end != 0 && is_format_whitespace(text[end - 1])) { --end; }
    return std::string(text.substr(0, end));
}

void skip_format_whitespace(std::string_view text, std::size_t& pos) {
    while (pos < text.size() && is_format_whitespace(text[pos])) { ++pos; }
}

bool starts_with_at(std::string_view text, std::size_t pos, std::string_view prefix) {
    return pos <= text.size() && text.substr(pos, prefix.size()) == prefix;
}

constexpr bool tool_name_byte(char byte) {
    return is_ascii_alphanumeric(byte) || byte == '_' || byte == '-' || byte == '.' ||
           byte == ':' || byte == '/';
}

bool valid_function_name(std::string_view name, std::size_t max_name_length) {
    return valid_tool_name(name, max_name_length);
}

constexpr std::uint8_t type_bit(SchemaType type) { return static_cast<std::uint8_t>(type); }

constexpr bool admits_type(TypeSet types, SchemaType type) {
    return (types.bits & type_bit(type)) != 0;
}

bool schema_type(std::string_view name, SchemaType& type) {
    if (name == "null") {
        type = SchemaType::Null;
    } else if (name == "boolean") {
        type = SchemaType::Boolean;
    } else if (name == "integer") {
        type = SchemaType::Integer;
    } else if (name == "number") {
        type = SchemaType::Number;
    } else if (name == "string") {
        type = SchemaType::String;
    } else if (name == "object") {
        type = SchemaType::Object;
    } else if (name == "array") {
        type = SchemaType::Array;
    } else {
        return false;
    }
    return true;
}

bool compile_direct_types(const Json& type_definition, TypeSet& types) {
    types = {};
    if (type_definition.is_string()) {
        SchemaType type;
        if (!schema_type(type_definition.get_ref<const std::string&>(), type)) { return false; }
        types.bits = type_bit(type);
        return true;
    }
    if (!type_definition.is_array() || type_definition.empty()) { return false; }
    for (const Json& member : type_definition) {
        if (!member.is_string()) { return false; }
        SchemaType type;
        if (!schema_type(member.get_ref<const std::string&>(), type)) { return false; }
        types.bits |= type_bit(type);
    }
    return types.bits != 0;
}

// A local reference ("#/$defs/Name", "#/definitions/Name" or any JSON Pointer into the tool's
// parameters schema) resolves within that schema; remote, dangling and cyclic references do not.
constexpr int kMaximumReferenceDepth = 16;

const Json* resolve_local_reference(const Json& root, const Json& reference) {
    if (!reference.is_string()) { return nullptr; }
    const std::string& text = reference.get_ref<const std::string&>();
    if (text.empty() || text.front() != '#') { return nullptr; }
    try {
        const Json::json_pointer pointer(text.substr(1));
        return root.contains(pointer) ? &root.at(pointer) : nullptr;
    } catch (const Json::exception&) { return nullptr; }
}

bool compile_schema_types(const Json& schema, TypeSet& types, const Json& root, int depth) {
    if (!schema.is_object() || depth > kMaximumReferenceDepth) { return false; }
    const auto direct = schema.find("type");
    if (direct != schema.end()) { return compile_direct_types(*direct, types); }
    if (const auto reference = schema.find("$ref"); reference != schema.end()) {
        const Json* target = resolve_local_reference(root, *reference);
        return target != nullptr && compile_schema_types(*target, types, root, depth + 1);
    }

    const auto any_of     = schema.find("anyOf");
    const auto one_of     = schema.find("oneOf");
    const bool has_any_of = any_of != schema.end();
    const bool has_one_of = one_of != schema.end();
    if (has_any_of == has_one_of) { return false; }

    const Json& alternatives = has_any_of ? *any_of : *one_of;
    if (!alternatives.is_array() || alternatives.empty()) { return false; }

    TypeSet combined;
    for (const Json& alternative : alternatives) {
        TypeSet branch;
        if (!compile_schema_types(alternative, branch, root, depth + 1)) { return false; }
        combined.bits |= branch.bits;
    }
    if (combined.bits == 0) { return false; }
    types = combined;
    return true;
}

Contract::Tool compile_tool_contract(const Json& definition) {
    Contract::Tool contract;
    if (!definition.is_object()) { return contract; }
    const auto function = definition.find("function");
    if (function == definition.end() || !function->is_object()) { return contract; }
    const auto name = function->find("name");
    if (name == function->end() || !name->is_string()) { return contract; }
    contract.name = name->get<std::string>();

    const auto schema = function->find("parameters");
    if (schema == function->end() || !schema->is_object()) { return contract; }
    const auto properties = schema->find("properties");
    if (properties == schema->end() || !properties->is_object()) { return contract; }

    contract.parameters.reserve(properties->size());
    for (const auto& [parameter_name, property] : properties->items()) {
        Contract::Parameter parameter;
        parameter.name = parameter_name;
        if (compile_schema_types(property, parameter.types, *schema, 0)) {
            parameter.policy = NormalizationPolicy::DeclaredTypes;
        }
        contract.parameters.push_back(std::move(parameter));
    }
    return contract;
}

bool same_contract(const Contract::Tool& lhs, const Contract::Tool& rhs) {
    if (lhs.parameters.size() != rhs.parameters.size()) { return false; }
    for (std::size_t i = 0; i < lhs.parameters.size(); ++i) {
        const Contract::Parameter& left  = lhs.parameters[i];
        const Contract::Parameter& right = rhs.parameters[i];
        if (left.name != right.name || left.policy != right.policy ||
            left.types.bits != right.types.bits) {
            return false;
        }
    }
    return true;
}

void append_tool_contract(Contract& contracts, const Json& definition) {
    Contract::Tool compiled = compile_tool_contract(definition);
    if (compiled.name.empty()) { return; }
    const auto existing =
        std::find_if(contracts.tools.begin(), contracts.tools.end(),
                     [&](const auto& tool) { return tool.name == compiled.name; });
    if (existing == contracts.tools.end()) {
        contracts.tools.push_back(std::move(compiled));
        return;
    }
    if (existing->unambiguous && !same_contract(*existing, compiled)) {
        existing->parameters.clear();
        existing->unambiguous = false;
    }
}

const Contract::Tool* find_tool_contract(const Contract& contract, std::string_view tool_name) {
    const auto tool =
        std::find_if(contract.tools.begin(), contract.tools.end(),
                     [&](const auto& candidate) { return candidate.name == tool_name; });
    return tool == contract.tools.end() ? nullptr : &*tool;
}

const Contract::Parameter* find_parameter_contract(const Contract::Tool& tool,
                                                   std::string_view parameter_name) {
    const auto parameter =
        std::find_if(tool.parameters.begin(), tool.parameters.end(),
                     [&](const auto& candidate) { return candidate.name == parameter_name; });
    return parameter == tool.parameters.end() ? nullptr : &*parameter;
}

std::string_view remove_parameter_framing_newlines(std::string_view text) {
    std::size_t begin = 0;
    std::size_t end   = text.size();
    if (text.starts_with("\r\n")) {
        begin = 2;
    } else if (text.starts_with('\n')) {
        begin = 1;
    }
    if (end >= begin + 2 && text.substr(end - 2, 2) == "\r\n") {
        end -= 2;
    } else if (end > begin && text[end - 1] == '\n') {
        --end;
    }
    return text.substr(begin, end - begin);
}

bool ascii_case_equal(std::string_view text, std::string_view lowercase) {
    if (text.size() != lowercase.size()) { return false; }
    for (std::size_t i = 0; i < text.size(); ++i) {
        char byte = text[i];
        if (byte >= 'A' && byte <= 'Z') { byte = static_cast<char>(byte + ('a' - 'A')); }
        if (byte != lowercase[i]) { return false; }
    }
    return true;
}

bool json_number_is_integer(std::string_view number) {
    std::size_t pos = number.starts_with('-') ? 1 : 0;
    if (pos >= number.size()) { return false; }

    const std::size_t integer_begin = pos;
    while (pos < number.size() && is_ascii_digit(number[pos])) { ++pos; }
    const std::size_t integer_end = pos;

    std::size_t fraction_begin = pos;
    std::size_t fraction_end   = pos;
    if (pos < number.size() && number[pos] == '.') {
        fraction_begin = ++pos;
        while (pos < number.size() && is_ascii_digit(number[pos])) { ++pos; }
        fraction_end = pos;
    }

    bool exponent_negative     = false;
    std::size_t exponent_value = 0;
    if (pos < number.size() && (number[pos] == 'e' || number[pos] == 'E')) {
        ++pos;
        if (pos < number.size() && (number[pos] == '+' || number[pos] == '-')) {
            exponent_negative = number[pos] == '-';
            ++pos;
        }
        const std::size_t cap = number.size();
        while (pos < number.size() && is_ascii_digit(number[pos])) {
            const std::size_t digit = static_cast<std::size_t>(number[pos] - '0');
            if (exponent_value != cap) {
                if (exponent_value > cap / 10 || (exponent_value == cap / 10 && digit > cap % 10)) {
                    exponent_value = cap;
                } else {
                    exponent_value = exponent_value * 10 + digit;
                }
            }
            ++pos;
        }
    }
    if (integer_begin == integer_end || pos != number.size()) { return false; }

    bool coefficient_is_zero   = true;
    std::size_t trailing_zeros = 0;
    const auto observe_digit   = [&](char digit) {
        if (digit == '0') {
            ++trailing_zeros;
        } else {
            coefficient_is_zero = false;
            trailing_zeros      = 0;
        }
    };
    for (std::size_t i = integer_begin; i < integer_end; ++i) { observe_digit(number[i]); }
    for (std::size_t i = fraction_begin; i < fraction_end; ++i) { observe_digit(number[i]); }
    if (coefficient_is_zero) { return true; }

    const std::size_t fraction_digits = fraction_end - fraction_begin;
    if (!exponent_negative) {
        if (exponent_value >= fraction_digits) { return true; }
        return fraction_digits - exponent_value <= trailing_zeros;
    }
    if (exponent_value > trailing_zeros) { return false; }
    return fraction_digits <= trailing_zeros - exponent_value;
}

bool classify_json_value(std::string_view value, JsonValueKind& kind) {
    if (value.empty() || !Json::accept(value.begin(), value.end())) { return false; }
    switch (value.front()) {
    case 'n':
        kind = JsonValueKind::Null;
        return true;
    case 't':
    case 'f':
        kind = JsonValueKind::Boolean;
        return true;
    case '"':
        kind = JsonValueKind::String;
        return true;
    case '{':
        kind = JsonValueKind::Object;
        return true;
    case '[':
        kind = JsonValueKind::Array;
        return true;
    default:
        if (value.front() == '-' || is_ascii_digit(value.front())) {
            kind = json_number_is_integer(value) ? JsonValueKind::Integer : JsonValueKind::Number;
            return true;
        }
        return false;
    }
}

bool admits_value(TypeSet types, JsonValueKind kind) {
    switch (kind) {
    case JsonValueKind::Null:
        return admits_type(types, SchemaType::Null);
    case JsonValueKind::Boolean:
        return admits_type(types, SchemaType::Boolean);
    case JsonValueKind::Integer:
        return admits_type(types, SchemaType::Integer) || admits_type(types, SchemaType::Number);
    case JsonValueKind::Number:
        return admits_type(types, SchemaType::Number);
    case JsonValueKind::String:
        return admits_type(types, SchemaType::String);
    case JsonValueKind::Object:
        return admits_type(types, SchemaType::Object);
    case JsonValueKind::Array:
        return admits_type(types, SchemaType::Array);
    }
    return false;
}

std::string encode_json_string(std::string_view value) { return Json(std::string(value)).dump(); }

NormalizedParameter normalize_declared_parameter(std::string_view encoded_value, TypeSet types) {
    const std::string_view framed = remove_parameter_framing_newlines(encoded_value);
    if (admits_type(types, SchemaType::String)) {
        return {.json_value = encode_json_string(framed)};
    }

    const std::string_view value = trim_format_whitespace(framed);
    if (value.empty()) { return {.disposition = ParameterNormalization::Omitted}; }

    JsonValueKind kind;
    if (classify_json_value(value, kind)) {
        return {.disposition = admits_value(types, kind) ? ParameterNormalization::Emitted
                                                         : ParameterNormalization::SchemaMismatch,
                .json_value  = std::string(value)};
    }

    if (admits_type(types, SchemaType::Boolean)) {
        if (ascii_case_equal(value, "true")) { return {.json_value = "true"}; }
        if (ascii_case_equal(value, "false")) { return {.json_value = "false"}; }
    }
    return {.disposition = ParameterNormalization::SchemaMismatch,
            .json_value  = encode_json_string(framed)};
}

// How a syntax writes an argument's text: Qwen XML frames it in newlines; ATEM writes a string
// as is (its template: "spaces for string values are not stripped").
enum class ValueFraming : std::uint8_t { NewlineFramed, AsIs };

NormalizedParameter normalize_parameter(std::string_view encoded_value,
                                        const Contract::Parameter* parameter,
                                        ValueFraming framing = ValueFraming::NewlineFramed) {
    if (parameter != nullptr && parameter->policy == NormalizationPolicy::DeclaredTypes) {
        if (framing == ValueFraming::AsIs && admits_type(parameter->types, SchemaType::String)) {
            return {.json_value = encode_json_string(encoded_value)};
        }
        return normalize_declared_parameter(encoded_value, parameter->types);
    }

    // Text that is not JSON is a string: only the format's framing newlines are removed, so code
    // keeps its indentation and final newline (as Qwen's reference parser keeps them).
    const std::string_view framed = framing == ValueFraming::AsIs
                                        ? encoded_value
                                        : remove_parameter_framing_newlines(encoded_value);
    const std::string_view value  = trim_format_whitespace(framed);
    if (Json::accept(value.begin(), value.end())) { return {.json_value = std::string(value)}; }
    return {.json_value = encode_json_string(framed)};
}

class QwenToolRegionParser {
public:
    QwenToolRegionParser(std::string_view text, std::size_t max_name_length,
                         const Contract& contract, bool enforce_names)
        : text_(text), max_name_length_(max_name_length), contract_(contract),
          enforce_names_(enforce_names && contract.enforce_declared_names) {}

    FallbackReason parse(RawCalls& calls) const {
        std::size_t pos = 0;
        for (;;) {
            skip_format_whitespace(text_, pos);
            if (pos == text_.size()) {
                return calls.empty() ? FallbackReason::MalformedStructure : FallbackReason::None;
            }
            if (!starts_with_at(text_, pos, kToolOpen)) {
                calls.trailing = pos;
                return calls.empty() ? FallbackReason::MalformedStructure
                                     : FallbackReason::TrailingContent;
            }

            RawToolCall call;
            const FallbackReason failure = parse_tool_call(pos, call);
            if (failure != FallbackReason::None) { return failure; }
            calls.push_back(std::move(call));
        }
    }

private:
    bool consume(std::size_t& pos, std::string_view token) const {
        if (!starts_with_at(text_, pos, token)) { return false; }
        pos += token.size();
        return true;
    }

    FallbackReason parse_tool_call(std::size_t& pos, RawToolCall& call) const {
        if (!consume(pos, kToolOpen)) { return FallbackReason::MalformedStructure; }
        skip_format_whitespace(text_, pos);
        const FallbackReason failure = parse_function(pos, call);
        if (failure != FallbackReason::None) { return failure; }
        skip_format_whitespace(text_, pos);
        if (consume(pos, kToolClose) || ends_in_closer(pos, kToolClose, 0)) {
            return FallbackReason::None;
        }
        return FallbackReason::MalformedStructure;
    }

    // The output ends at pos with a closer that was cut short (none of it, or a prefix of at
    // least min_prefix bytes). A small quant ends its turn there now and then: Qwen3.8-27B
    // UD-IQ2_XXS stopped on <|im_end|> after "</function" or "</function>", every parameter
    // closed, and the call reached the client as raw markup. A call cut inside a parameter, or
    // before the function's closer starts, stays malformed.
    bool ends_in_closer(std::size_t& pos, std::string_view closer, std::size_t min_prefix) const {
        const std::string_view rest = text_.substr(pos);
        if (rest.size() < min_prefix || rest.size() >= closer.size() ||
            closer.substr(0, rest.size()) != rest) {
            return false;
        }
        pos = text_.size();
        return true;
    }

    FallbackReason parse_function(std::size_t& pos, RawToolCall& call) const {
        if (!consume(pos, kFunctionOpen)) { return FallbackReason::MalformedStructure; }
        const std::size_t name_begin = pos;
        const std::size_t name_end   = text_.find('>', name_begin);
        if (name_end == std::string_view::npos || name_end == name_begin) {
            return FallbackReason::InvalidToolName;
        }
        // A small quant writes "<function= NAME>" now and then: the spaces are not the name's.
        call.name = trim_format_whitespace(text_.substr(name_begin, name_end - name_begin));
        if (!valid_function_name(call.name, max_name_length_)) {
            return FallbackReason::InvalidToolName;
        }
        if (enforce_names_ && find_tool_contract(contract_, call.name) == nullptr) {
            return FallbackReason::UndeclaredTool;
        }
        pos = name_end + 1;

        std::size_t body = pos;
        skip_format_whitespace(text_, body);
        if (body < text_.size() && text_[body] == '{') { return parse_json_body(body, pos, call); }
        for (;;) {
            skip_format_whitespace(text_, pos);
            if (consume(pos, kFunctionClose) || ends_in_closer(pos, kFunctionClose, 2)) {
                return FallbackReason::None;
            }
            const FallbackReason failure = parse_parameter(pos, call);
            if (failure != FallbackReason::None) { return failure; }
        }
    }

    // Some models write the arguments as one JSON object in the function element
    // (<function=NAME>{"key": value}</function>) instead of parameter elements. The call is kept
    // when that object parses whole up to the function's closer; otherwise it stays malformed.
    FallbackReason parse_json_body(std::size_t body, std::size_t& pos, RawToolCall& call) const {
        std::size_t end  = text_.find(kFunctionClose, body);
        std::size_t next = end == std::string_view::npos ? end : end + kFunctionClose.size();
        if (end == std::string_view::npos) {
            // A closer cut short at the end of the output, as ends_in_closer accepts.
            const std::size_t open = text_.rfind('<');
            if (open == std::string_view::npos || open < body || text_.size() - open < 2 ||
                kFunctionClose.substr(0, text_.size() - open) != text_.substr(open)) {
                return FallbackReason::MalformedStructure;
            }
            end  = open;
            next = text_.size();
        }
        const std::string_view object = trim_format_whitespace(text_.substr(body, end - body));
        const nlohmann::ordered_json parsed =
            nlohmann::ordered_json::parse(object.begin(), object.end(), nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) {
            return FallbackReason::MalformedStructure;
        }
        call.json_arguments = parsed.dump();
        pos                 = next;
        return FallbackReason::None;
    }

    FallbackReason parse_parameter(std::size_t& pos, RawToolCall& call) const {
        if (!consume(pos, kParamOpen)) { return FallbackReason::MalformedStructure; }
        const std::size_t name_begin = pos;
        const std::size_t name_end   = text_.find('>', name_begin);
        if (name_end == std::string_view::npos || name_end == name_begin) {
            return FallbackReason::MalformedStructure;
        }
        const std::string_view name =
            trim_format_whitespace(text_.substr(name_begin, name_end - name_begin));
        if (name.empty()) { return FallbackReason::MalformedStructure; }

        const std::size_t value_begin = name_end + 1;
        std::size_t value_end         = 0;
        if (!find_parameter_close(value_begin, value_end)) {
            return FallbackReason::MalformedStructure;
        }
        set_parameter(call, name, text_.substr(value_begin, value_end - value_begin));
        pos = value_end + kParamClose.size();
        return FallbackReason::None;
    }

    bool find_parameter_open_before(std::size_t scan, std::size_t limit,
                                    std::size_t& open_end) const {
        std::size_t candidate = text_.find(kParamOpen, scan);
        while (candidate != std::string_view::npos && candidate < limit) {
            const std::size_t name_begin = candidate + kParamOpen.size();
            const std::size_t name_end   = text_.find('>', name_begin);
            if (name_end != std::string_view::npos && name_end < limit && name_end != name_begin) {
                open_end = name_end + 1;
                return true;
            }
            candidate = text_.find(kParamOpen, candidate + 1);
        }
        return false;
    }

    bool closes_parameter(std::size_t pos) const {
        skip_format_whitespace(text_, pos);
        return starts_with_at(text_, pos, kParamOpen) ||
               starts_with_at(text_, pos, kFunctionClose) || ends_in_closer(pos, kFunctionClose, 2);
    }

    bool find_parameter_close(std::size_t value_begin, std::size_t& value_end) const {
        std::size_t depth = 1;
        std::size_t scan  = value_begin;
        for (;;) {
            const std::size_t close = text_.find(kParamClose, scan);
            if (close == std::string_view::npos) { return false; }

            std::size_t nested_open_end = 0;
            if (find_parameter_open_before(scan, close, nested_open_end)) {
                ++depth;
                scan = nested_open_end;
                continue;
            }

            // An outer closer the grammar cannot continue from (whitespace, then the next
            // parameter or the function's closer) is value text quoting the markup.
            if (depth != 1 || closes_parameter(close + kParamClose.size())) { --depth; }
            if (depth == 0) {
                value_end = close;
                return true;
            }
            scan = close + kParamClose.size();
        }
    }

    std::string_view text_;
    std::size_t max_name_length_;
    const Contract& contract_;
    bool enforce_names_;
};

GeneratedToolCall normalize_raw_tool_call(const RawToolCall& raw, const Contract& contract,
                                          ToolCallParseDiagnostics& diagnostics,
                                          ValueFraming framing = ValueFraming::NewlineFramed) {
    const Contract::Tool* tool = find_tool_contract(contract, raw.name);
    if (tool != nullptr && !tool->unambiguous) { tool = nullptr; }

    diagnostics.repeated_parameters += raw.repeated_parameters;
    if (raw.json_arguments) {
        return GeneratedToolCall{.name           = std::string(raw.name),
                                 .arguments_json = *raw.json_arguments};
    }
    std::string arguments = "{";
    bool first            = true;
    for (const RawParameter& raw_parameter : raw.parameters) {
        const Contract::Parameter* parameter =
            tool == nullptr ? nullptr : find_parameter_contract(*tool, raw_parameter.name);
        NormalizedParameter normalized =
            normalize_parameter(raw_parameter.value, parameter, framing);
        if (tool != nullptr && parameter == nullptr) {
            normalized.disposition = ParameterNormalization::SchemaMismatch;
        }
        if (normalized.disposition == ParameterNormalization::Omitted) {
            ++diagnostics.empty_arguments_omitted;
            continue;
        }
        if (normalized.disposition == ParameterNormalization::SchemaMismatch) {
            ++diagnostics.schema_mismatch_arguments;
        }

        if (!first) { arguments.push_back(','); }
        first = false;
        arguments += encode_json_string(raw_parameter.name);
        arguments.push_back(':');
        arguments += normalized.json_value;
    }
    arguments.push_back('}');

    return GeneratedToolCall{.name = std::string(raw.name), .arguments_json = std::move(arguments)};
}

// No tool-call syntax escapes its markers, so the first open marker may be prose quoting one (a
// model reasoning about tool calls before making one). When the region from it fails for any
// reason but text after complete calls (content_around_calls keeps those calls), the parse is
// retried from the last kQuotedMarkerRetries later markers, earliest first so that every call of a
// multi-call suffix is kept. `parse_from` parses the region from an offset, replacing its calls;
// after a success they are that region's.
constexpr std::size_t kQuotedMarkerRetries = 4;

struct CallRegion {
    std::size_t start;
    FallbackReason failure; // of the first marker's region when no retry parses
};

template <class ParseFrom>
CallRegion find_call_region_pass(const std::string& text, std::size_t first, std::string_view open,
                                 ParseFrom&& parse_from) {
    const FallbackReason failure = parse_from(first);
    if (failure == FallbackReason::None || failure == FallbackReason::TrailingContent) {
        return {first, failure};
    }
    std::vector<std::size_t> later;
    for (std::size_t pos = text.find(open, first + open.size()); pos != std::string::npos;
         pos             = text.find(open, pos + open.size())) {
        later.push_back(pos);
    }
    const std::size_t skip =
        later.size() > kQuotedMarkerRetries ? later.size() - kQuotedMarkerRetries : 0;
    for (std::size_t i = skip; i < later.size(); ++i) {
        if (parse_from(later[i]) == FallbackReason::None) {
            return {later[i], FallbackReason::None};
        }
    }
    return {first, failure};
}

// `parse_from(start, enforce_names)`. Declared names pick the region first, so a quoted example
// naming no declared tool still yields to a later call that does. When only an undeclared name
// stands in the way, the calls are taken as they are (vLLM and llama.cpp do the same): the client
// answers an unknown tool and the model recovers, where text would end the agent's turn and lose
// every parallel call of it.
template <class ParseFrom>
CallRegion find_call_region(const std::string& text, std::size_t first, std::string_view open,
                            ParseFrom&& parse_from) {
    const CallRegion declared = find_call_region_pass(
        text, first, open, [&](std::size_t start) { return parse_from(start, true); });
    if (declared.failure != FallbackReason::UndeclaredTool) { return declared; }
    return find_call_region_pass(text, first, open,
                                 [&](std::size_t start) { return parse_from(start, false); });
}

template <class Calls>
std::uint32_t count_undeclared(const Contract& contract, const Calls& calls) {
    if (!contract.enforce_declared_names) { return 0; }
    return static_cast<std::uint32_t>(
        std::count_if(calls.begin(), calls.end(),
                      [&](const auto& call) { return !contract.declares(call.name); }));
}

// The content of a parsed region: the text before it and, when complete calls are followed by
// text (TrailingContent at `trailing` within the region), that text after it. The calls stay calls
// (vLLM keeps them too): a model that adds a remark after its call would otherwise lose the call
// and end the agent's turn. Text that opens another call (one the output limit cut off, or a
// call after prose) is not a remark: then nothing is returned and the whole output stays text.
std::optional<std::string> content_around_calls(const std::string& text, const CallRegion& region,
                                                std::size_t trailing, std::string_view open,
                                                ToolCallParseDiagnostics& diagnostics) {
    std::string content(rtrim_format_whitespace(std::string_view(text).substr(0, region.start)));
    if (region.failure != FallbackReason::TrailingContent) { return content; }
    const std::string_view tail =
        trim_format_whitespace(std::string_view(text).substr(region.start + trailing));
    if (tail.find(open) != std::string_view::npos) { return std::nullopt; }
    diagnostics.trailing_content_bytes = static_cast<std::uint32_t>(tail.size());
    if (!content.empty() && !tail.empty()) { content += '\n'; }
    content += tail;
    return content;
}

ParsedToolCallOutput fallback(const std::string& text, ToolCallParseDiagnostics diagnostics = {}) {
    ParsedToolCallOutput out;
    out.content     = text;
    out.diagnostics = diagnostics;
    return out;
}

// A Qwen 3.x model now and then writes its calls as bare <function=NAME>...</function> blocks
// without the <tool_call> wrapper; vLLM's Qwen parser takes them as calls. Here each block is
// wrapped and the whole is parsed as usual, and the result counts only when every call names a
// declared tool, so prose quoting the markup stays text.
ParsedToolCallOutput parse_bare_function_calls(const std::string& text,
                                               std::size_t max_tool_name_length,
                                               const Contract& contract) {
    const std::size_t first = text.find(kFunctionOpen);
    if (first == std::string::npos || !contract.enforce_declared_names) { return fallback(text); }
    std::string wrapped = text.substr(0, first);
    std::size_t pos     = first;
    for (;;) {
        wrapped.append(kToolOpen);
        wrapped.push_back('\n');
        const std::size_t close = text.find(kFunctionClose, pos);
        if (close == std::string::npos) {
            wrapped.append(text, pos);
            break;
        }
        const std::size_t end = close + kFunctionClose.size();
        wrapped.append(text, pos, end - pos);
        wrapped.push_back('\n');
        wrapped.append(kToolClose);
        std::size_t next = end;
        skip_format_whitespace(text, next);
        wrapped.append(text, end, next - end);
        if (!starts_with_at(text, next, kFunctionOpen)) {
            wrapped.append(text, next);
            break;
        }
        pos = next;
    }
    ParsedToolCallOutput out = parse_qwen_tool_call_output(wrapped, max_tool_name_length, contract);
    if (!out.is_tool_call_response ||
        std::any_of(out.tool_calls.begin(), out.tool_calls.end(),
                    [&](const GeneratedToolCall& call) { return !contract.declares(call.name); })) {
        return fallback(text);
    }
    out.diagnostics.unwrapped_calls = static_cast<std::uint32_t>(out.tool_calls.size());
    return out;
}

} // namespace

bool valid_tool_name(std::string_view name, std::size_t max_length) noexcept {
    if (name.empty() || name.size() > max_length) { return false; }
    return std::all_of(name.begin(), name.end(), tool_name_byte);
}

bool ToolCallOutputContract::declares(std::string_view name) const noexcept {
    return std::any_of(tools.begin(), tools.end(),
                       [&](const Tool& tool) { return tool.name == name; });
}

std::shared_ptr<const ToolCallOutputContract>
build_tool_call_output_contract(std::span<const std::string> tool_jsons, bool enabled,
                                ToolCallFormat format) {
    if (!enabled || format == ToolCallFormat::None) { return {}; }
    auto contract                    = std::make_shared<ToolCallOutputContract>();
    contract->enforce_declared_names = true;
    contract->format                 = format;
    contract->tools.reserve(tool_jsons.size());
    for (const std::string& tool_json : tool_jsons) {
        const Json definition = Json::parse(tool_json, nullptr, false);
        if (!definition.is_discarded()) { append_tool_contract(*contract, definition); }
    }
    return contract;
}

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length,
                                                 const ToolCallOutputContract& contract) {
    const std::size_t first = text.find(kToolOpen);
    if (first == std::string::npos) {
        return parse_bare_function_calls(text, max_tool_name_length, contract);
    }

    ParsedToolCallOutput out;
    out.diagnostics.marker_seen = true;

    RawCalls raw_calls;
    const CallRegion region =
        find_call_region(text, first, kToolOpen, [&](std::size_t start, bool enforce_names) {
            raw_calls.clear();
            raw_calls.trailing = 0;
            return QwenToolRegionParser(std::string_view(text).substr(start), max_tool_name_length,
                                        contract, enforce_names)
                .parse(raw_calls);
        });
    std::optional<std::string> content =
        region.failure == FallbackReason::None || region.failure == FallbackReason::TrailingContent
            ? content_around_calls(text, region, raw_calls.trailing, kToolOpen, out.diagnostics)
            : std::nullopt;
    if (!content) {
        out.diagnostics.fallback_reason = region.failure;
        return fallback(text, out.diagnostics);
    }
    out.content = std::move(*content);

    out.tool_calls.reserve(raw_calls.size());
    for (const RawToolCall& raw : raw_calls) {
        out.tool_calls.push_back(normalize_raw_tool_call(raw, contract, out.diagnostics));
    }

    out.diagnostics.undeclared_calls      = count_undeclared(contract, out.tool_calls);
    out.diagnostics.structured_call_count = static_cast<std::uint32_t>(out.tool_calls.size());
    out.is_tool_call_response             = true;
    return out;
}

namespace {

// Whitespace that may end the output after the calls, a few bytes of it. A model that writes "\n"
// after its last call (Nemotron 3) otherwise had that newline taken as the separator and was made
// to open one more call, again and again until max_tokens. Unbounded, it let a model that ranks a
// forbidden token above its stop token after the call (SmolLM3: its answer's text) write newlines
// until max_tokens instead.
constexpr std::string_view kCallsTrailingWhitespace = "[ \\t\\r\\n]{1,4}";

// Whether <parameter=NAME> carries a name the parser reads back: it ends at the first '>' and is
// never empty; a line break is kept out as well.
bool qwen_parameter_name(std::string_view name) {
    return !name.empty() && name.find_first_of(">\n\r") == std::string_view::npos;
}

// A non-strict tool's arguments under a tool-call constraint: its declared parameter names as
// literal keys with any value, in any order, none required (other names still admitted), so the
// call keeps the syntax the parser reads without holding to the schema. Names the syntax cannot
// carry are left to the generic name rule.
nlohmann::ordered_json loose_qwen_parameters(const ToolCallConstraintTool& tool) {
    using Ordered        = nlohmann::ordered_json;
    Ordered properties   = Ordered::object();
    const Ordered schema = Ordered::parse(tool.parameters_json, nullptr, false);
    if (schema.is_object()) {
        const auto declared = schema.find("properties");
        if (declared != schema.end() && declared->is_object()) {
            for (const auto& [name, unused] : declared->items()) {
                if (qwen_parameter_name(name)) { properties[name] = Ordered{{"type", "string"}}; }
            }
        }
    }
    // Every value is the raw text up to </parameter> (a "string" in the XML style): the language of
    // "any value" (that text, a JSON array or a JSON object) minus JSON holding "</parameter>",
    // whose masks cost up to 50 ms at a value's start on a 262k vocabulary, and under 1 ms here.
    return Ordered{{"type", "object"},
                   {"properties", std::move(properties)},
                   {"additionalProperties", Ordered{{"type", "string"}}}};
}

// A strict tool's arguments hold to its schema, so every declared name must be one the syntax
// carries: a grammar forcing another would produce a call the parser reads as malformed.
nlohmann::ordered_json strict_qwen_parameters(const ToolCallConstraintTool& tool) {
    nlohmann::ordered_json schema = nlohmann::ordered_json::parse(tool.parameters_json);
    if (const auto declared = schema.find("properties");
        declared != schema.end() && declared->is_object()) {
        for (const auto& [name, unused] : declared->items()) {
            if (!qwen_parameter_name(name)) {
                throw std::invalid_argument("tool '" + tool.name + "' parameter name '" + name +
                                            "' cannot be written as a Qwen XML parameter");
            }
        }
    }
    return schema;
}

} // namespace

std::string build_tool_call_structural_tag(std::span<const ToolCallConstraintTool> tools,
                                           ToolCallRequirement requirement,
                                           std::string_view named_tool, bool parallel_tool_calls) {
    using Ordered  = nlohmann::ordered_json;
    const auto tag = [](const ToolCallConstraintTool& tool) {
        // Strict arguments hold to the declared schema; non-strict ones keep only the syntax.
        Ordered content{{"type", "json_schema"},
                        {"json_schema", tool.strict ? strict_qwen_parameters(tool)
                                                    : loose_qwen_parameters(tool)},
                        {"style", "qwen_xml"},
                        {"any_order", !tool.strict}};
        return Ordered{{"type", "tag"},
                       {"begin", std::string(kToolOpen) + "\n" + std::string(kFunctionOpen) +
                                     tool.name + ">\n"},
                       {"content", std::move(content)},
                       {"end", "\n" + std::string(kFunctionClose) + "\n" + std::string(kToolClose)}};
    };
    Ordered format;
    if (requirement == ToolCallRequirement::Named) {
        const auto it = std::find_if(tools.begin(), tools.end(), [&](const auto& tool) {
            return tool.name == named_tool;
        });
        if (it == tools.end()) {
            throw std::invalid_argument("named tool '" + std::string(named_tool) +
                                        "' is not among the constrained tools");
        }
        format = tag(*it);
    } else {
        if (tools.empty()) { throw std::invalid_argument("tool-call constraint has no tools"); }
        Ordered tags = Ordered::array();
        for (const ToolCallConstraintTool& tool : tools) { tags.push_back(tag(tool)); }
        Ordered text = Ordered{{"type", "any_text"},
                               {"excludes", Ordered::array({std::string(kToolOpen)})}};
        Ordered calls{{"type", "tags_with_separator"},
                      {"tags", std::move(tags)},
                      {"separator", "\n"},
                      {"at_least_one", requirement == ToolCallRequirement::Required},
                      {"stop_after_first", !parallel_tool_calls}};
        // Whitespace may end the output after the calls (kCallsTrailingWhitespace).
        Ordered tail{{"type", "optional"},
                     {"content", Ordered{{"type", "regex"},
                                         {"pattern", std::string(kCallsTrailingWhitespace)}}}};
        // tool_choice "required": the answer is the call(s), as for a named tool. Text first let
        // a model with nothing to call write until max_tokens, since the text cannot end
        // without a call.
        format =
            requirement == ToolCallRequirement::Required
                ? Ordered{{"type", "sequence"},
                          {"elements", Ordered::array({std::move(calls), std::move(tail)})}}
                : Ordered{{"type", "sequence"},
                          {"elements",
                           Ordered::array({std::move(text), std::move(calls), std::move(tail)})}};
    }
    return Ordered{{"type", "structural_tag"}, {"format", std::move(format)}}.dump();
}

namespace {

using CallStart = ToolCallStart;

// Whether held bytes can still begin a tool-call suffix: whitespace, an optional lead marker and
// whitespace, then the open marker (Complete once the marker is whole).
CallStart match_call_start(std::string_view held, std::string_view lead, std::string_view open) {
    std::size_t at = 0;
    while (at < held.size() && is_format_whitespace(held[at])) { ++at; }
    std::string_view rest = held.substr(at);
    if (!lead.empty()) {
        if (rest.size() < lead.size()) {
            if (lead.starts_with(rest)) { return CallStart::Partial; }
        } else if (rest.starts_with(lead)) {
            rest.remove_prefix(lead.size());
            std::size_t gap = 0;
            while (gap < rest.size() && is_format_whitespace(rest[gap])) { ++gap; }
            rest.remove_prefix(gap);
        }
    }
    if (rest.size() < open.size()) {
        return open.starts_with(rest) ? CallStart::Partial : CallStart::No;
    }
    return rest == open ? CallStart::Complete : CallStart::No;
}

} // namespace

ToolCallStart ToolCallSyntax::match_start(std::string_view held) const {
    return match_call_start(held, lead(), open());
}

ToolCallOutputDecoder::ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                                             std::size_t max_tool_name_length)
    : contract_(std::move(contract)),
      syntax_(contract_ ? &tool_call_syntax(contract_->format) : nullptr),
      max_tool_name_length_(max_tool_name_length) {}

std::string ToolCallOutputDecoder::feed(std::string_view text) {
    if (finished_) { throw std::logic_error("tool-call output decoder is already finished"); }
    if (text.empty()) { return {}; }
    if (!contract_) { return std::string(text); }
    if (saw_tool_marker_) {
        tool_region_.append(text);
        return {};
    }

    std::string visible;
    for (std::size_t index = 0; index < text.size(); ++index) {
        held_.push_back(text[index]);
        CallStart start = syntax_->match_start(held_);
        // Publish the held bytes that can no longer begin the suffix, oldest first.
        while (start == CallStart::No) {
            visible.push_back(held_.front());
            held_.erase(0, 1);
            start = held_.empty() ? CallStart::Partial : syntax_->match_start(held_);
        }
        if (start == CallStart::Complete) {
            tool_region_ = std::move(held_);
            held_.clear();
            tool_region_.append(text.substr(index + 1));
            saw_tool_marker_ = true;
            break;
        }
    }
    return visible;
}

ToolCallOutputDecoder::Terminal ToolCallOutputDecoder::finish(bool cut_off) {
    if (finished_) { throw std::logic_error("tool-call output decoder is already finished"); }
    finished_ = true;
    if (!contract_) { return {}; }

    ParsedToolCallOutput parsed =
        tool_call_syntax(contract_->format).parse(tool_region_, max_tool_name_length_, *contract_);
    if (saw_tool_marker_ && parsed.is_tool_call_response) {
        held_.clear();
        tool_region_.clear();
        // The prose before a recovered later marker, and text after complete calls.
        return Terminal{.content     = std::move(parsed.content),
                        .tool_calls  = std::move(parsed.tool_calls),
                        .diagnostics = parsed.diagnostics};
    }
    // Qwen XML: the stop token ended the turn after the last call's closed parameter, before its
    // </function> (vLLM closes such a call at the end of the stream too). The call is closed; one
    // the output limit cut off is not, since more of its parameters may have followed.
    if (!cut_off && saw_tool_marker_ && contract_->format == ToolCallFormat::QwenXml) {
        std::string closed = rtrim_format_whitespace(tool_region_);
        if (closed.ends_with(kParamClose)) {
            closed.append("\n").append(kFunctionClose);
            if (closed.find(kToolOpen) != std::string::npos) {
                closed.append("\n").append(kToolClose);
            }
            ParsedToolCallOutput completed =
                syntax_->parse(closed, max_tool_name_length_, *contract_);
            if (completed.is_tool_call_response && !completed.tool_calls.empty()) {
                held_.clear();
                tool_region_.clear();
                completed.diagnostics.closed_calls = 1;
                return Terminal{.content     = std::move(completed.content),
                                .tool_calls  = std::move(completed.tool_calls),
                                .diagnostics = completed.diagnostics};
            }
        }
    }
    // The output limit cut the last call off (its open marker has no close marker after it): the
    // calls before that marker were complete and are the turn's calls (as an all-or-nothing parse,
    // every one of them was lost as text). The unfinished call cannot run; its markup is dropped
    // and counted. A region that fails for any other reason (a malformed call, a call after
    // prose that follows complete calls), or does not parse up to that marker, stays text.
    const std::string_view close = syntax_->close();
    if (cut_off && saw_tool_marker_ && !close.empty()) {
        const std::size_t last = tool_region_.rfind(syntax_->open());
        if (last != std::string::npos && last != 0 &&
            tool_region_.find(close, last) == std::string::npos) {
            ParsedToolCallOutput complete =
                syntax_->parse(tool_region_.substr(0, last), max_tool_name_length_, *contract_);
            if (complete.is_tool_call_response && !complete.tool_calls.empty()) {
                held_.clear();
                tool_region_.clear();
                complete.diagnostics.cut_calls_dropped = 1;
                return Terminal{.content     = std::move(complete.content),
                                .tool_calls  = std::move(complete.tool_calls),
                                .diagnostics = complete.diagnostics};
            }
        }
    }

    std::string tail = std::move(held_);
    held_.clear();
    tail += tool_region_;
    tool_region_.clear();
    return Terminal{
        .content = std::move(tail), .tool_calls = {}, .diagnostics = parsed.diagnostics};
}

// ---------------------------------------------------------------------------------------------
// Tool-call syntaxes. Qwen XML is the parser above; the JSON-argument syntaxes share one frame.

namespace {

using OrderedJson = nlohmann::ordered_json;

struct ParsedCall {
    std::string name;
    std::string arguments_json;
};

// A region's calls and, when complete calls are followed by text (TrailingContent), where that
// text starts in the region.
struct ParsedCalls : std::vector<ParsedCall> {
    std::size_t trailing = 0;
};

// Text before the call region is content; the region is parsed into calls by the syntax, from the
// first open marker or, past a quoted one, a later marker (find_call_region); any failure restores
// the whole text as content, as the Qwen parser does.
template <class RegionParser>
ParsedToolCallOutput parse_calls(const std::string& text, std::string_view open,
                                 std::size_t max_tool_name_length, const Contract& contract,
                                 RegionParser&& region_parser) {
    const std::size_t first = text.find(open);
    if (first == std::string::npos) { return fallback(text); }
    ParsedToolCallOutput out;
    out.diagnostics.marker_seen = true;
    ParsedCalls calls;
    const CallRegion region =
        find_call_region(text, first, open, [&](std::size_t start, bool enforce_names) {
            calls.clear();
            calls.trailing         = 0;
            FallbackReason failure = region_parser(std::string_view(text).substr(start), calls);
            for (const ParsedCall& call : calls) {
                if (failure != FallbackReason::None && failure != FallbackReason::TrailingContent) {
                    break;
                }
                if (!valid_function_name(call.name, max_tool_name_length)) {
                    failure = FallbackReason::InvalidToolName;
                } else if (enforce_names && contract.enforce_declared_names &&
                           !contract.declares(call.name)) {
                    failure = FallbackReason::UndeclaredTool;
                }
            }
            return failure;
        });
    std::optional<std::string> content =
        region.failure == FallbackReason::None || region.failure == FallbackReason::TrailingContent
            ? content_around_calls(text, region, calls.trailing, open, out.diagnostics)
            : std::nullopt;
    if (!content) {
        out.diagnostics.fallback_reason = region.failure;
        return fallback(text, out.diagnostics);
    }
    out.content                      = std::move(*content);
    out.diagnostics.undeclared_calls = count_undeclared(contract, calls);
    out.tool_calls.reserve(calls.size());
    for (ParsedCall& call : calls) {
        out.tool_calls.push_back(GeneratedToolCall{.name           = std::move(call.name),
                                                   .arguments_json = std::move(call.arguments_json)});
    }
    out.diagnostics.structured_call_count = static_cast<std::uint32_t>(out.tool_calls.size());
    out.is_tool_call_response             = true;
    return out;
}

// Arguments as a JSON object: an object, a string holding one, or absent (no arguments).
std::optional<std::string> arguments_object(const OrderedJson* arguments) {
    if (arguments == nullptr || arguments->is_null()) { return std::string("{}"); }
    if (arguments->is_object()) { return arguments->dump(); }
    if (arguments->is_string()) {
        const OrderedJson parsed = OrderedJson::parse(arguments->get<std::string>(), nullptr, false);
        if (!parsed.is_discarded() && parsed.is_object()) { return parsed.dump(); }
    }
    return std::nullopt;
}

// {"name": ..., "arguments"|"parameters": {...}}
std::optional<ParsedCall> json_call(const OrderedJson& value) {
    if (!value.is_object()) { return std::nullopt; }
    const auto name = value.find("name");
    if (name == value.end() || !name->is_string()) { return std::nullopt; }
    const OrderedJson* arguments = nullptr;
    for (const char* key : {"arguments", "parameters"}) {
        if (const auto it = value.find(key); it != value.end()) {
            arguments = &*it;
            break;
        }
    }
    std::optional<std::string> object = arguments_object(arguments);
    if (!object) { return std::nullopt; }
    return ParsedCall{name->get<std::string>(), std::move(*object)};
}

// End of the JSON object or array starting at pos, honouring strings.
std::optional<std::size_t> json_value_end(std::string_view text, std::size_t pos) {
    std::size_t depth = 0;
    bool in_string    = false;
    for (std::size_t index = pos; index < text.size(); ++index) {
        const char byte = text[index];
        if (in_string) {
            if (byte == '\\') {
                ++index;
            } else if (byte == '"') {
                in_string = false;
            }
            continue;
        }
        if (byte == '"') {
            in_string = true;
        } else if (byte == '{' || byte == '[') {
            ++depth;
        } else if (byte == '}' || byte == ']') {
            if (depth == 0) { return std::nullopt; }
            if (--depth == 0) { return index + 1; }
        }
    }
    return std::nullopt;
}

// Tool-call arguments in a call syntax XGrammar has no style for: the constraint compiler
// turns this format into that syntax's grammar held to the schema (ninferno/types.h).
OrderedJson argument_style_format(OrderedJson schema, std::string_view style) {
    return OrderedJson{
        {"type", "json_schema"}, {"json_schema", std::move(schema)}, {"style", std::string(style)}};
}

OrderedJson schema_or_object(const ToolCallConstraintTool& tool) {
    return tool.strict ? OrderedJson::parse(tool.parameters_json)
                       : OrderedJson{{"type", "object"}};
}

// The structural tag frame shared by the syntaxes: free text that never opens a call, then calls.
template <class MakeTag>
std::string calls_structural_tag(std::span<const ToolCallConstraintTool> tools,
                                 ToolCallRequirement requirement, std::string_view named_tool,
                                 bool parallel_tool_calls, std::string_view open,
                                 std::string_view separator, MakeTag&& make_tag) {
    OrderedJson format;
    if (requirement == ToolCallRequirement::Named) {
        const auto it = std::find_if(tools.begin(), tools.end(), [&](const auto& tool) {
            return tool.name == named_tool;
        });
        if (it == tools.end()) {
            throw std::invalid_argument("named tool '" + std::string(named_tool) +
                                        "' is not among the constrained tools");
        }
        format = make_tag(*it);
    } else {
        if (tools.empty()) { throw std::invalid_argument("tool-call constraint has no tools"); }
        OrderedJson tags = OrderedJson::array();
        for (const ToolCallConstraintTool& tool : tools) { tags.push_back(make_tag(tool)); }
        OrderedJson text = OrderedJson{{"type", "any_text"},
                                       {"excludes", OrderedJson::array({std::string(open)})}};
        OrderedJson calls{{"type", "tags_with_separator"},
                          {"tags", std::move(tags)},
                          {"separator", std::string(separator)},
                          {"at_least_one", requirement == ToolCallRequirement::Required},
                          {"stop_after_first", !parallel_tool_calls}};
        // Whitespace may end the output after the calls (kCallsTrailingWhitespace).
        OrderedJson tail{
            {"type", "optional"},
            {"content",
             OrderedJson{{"type", "regex"}, {"pattern", std::string(kCallsTrailingWhitespace)}}}};
        // tool_choice "required": the answer is the call(s), as for a named tool (see above).
        format =
            requirement == ToolCallRequirement::Required
                ? OrderedJson{{"type", "sequence"},
                              {"elements", OrderedJson::array({std::move(calls), std::move(tail)})}}
                : OrderedJson{{"type", "sequence"},
                              {"elements", OrderedJson::array({std::move(text), std::move(calls),
                                                               std::move(tail)})}};
    }
    return OrderedJson{{"type", "structural_tag"}, {"format", std::move(format)}}.dump();
}

class QwenXmlSyntax final : public ToolCallSyntax {
public:
    [[nodiscard]] std::string_view open() const noexcept override { return kToolOpen; }

    [[nodiscard]] std::string_view close() const noexcept override { return kToolClose; }

    // A bare <function=NAME> block opens the region too (parse_bare_function_calls).
    [[nodiscard]] ToolCallStart match_start(std::string_view held) const override {
        const ToolCallStart wrapped = match_call_start(held, {}, kToolOpen);
        return wrapped != ToolCallStart::No ? wrapped : match_call_start(held, {}, kFunctionOpen);
    }
    [[nodiscard]] ParsedToolCallOutput parse(const std::string& text, std::size_t max_length,
                                             const Contract& contract) const override {
        return parse_qwen_tool_call_output(text, max_length, contract);
    }
    [[nodiscard]] std::string structural_tag(std::span<const ToolCallConstraintTool> tools,
                                             ToolCallRequirement requirement,
                                             std::string_view named_tool,
                                             bool parallel_tool_calls) const override {
        return build_tool_call_structural_tag(tools, requirement, named_tool, parallel_tool_calls);
    }
};

// <tool_call>\n{"name": NAME, "arguments": {...}}\n</tool_call>, calls separated by whitespace.
class HermesJsonSyntax final : public ToolCallSyntax {
public:
    [[nodiscard]] std::string_view open() const noexcept override { return kToolOpen; }

    [[nodiscard]] std::string_view close() const noexcept override { return kToolClose; }
    [[nodiscard]] ParsedToolCallOutput parse(const std::string& text, std::size_t max_length,
                                             const Contract& contract) const override {
        return parse_calls(text, kToolOpen, max_length, contract,
                           [](std::string_view region, ParsedCalls& calls) {
            std::size_t pos = 0;
            for (;;) {
                skip_format_whitespace(region, pos);
                if (pos == region.size()) {
                    return calls.empty() ? FallbackReason::MalformedStructure
                                         : FallbackReason::None;
                }
                if (!starts_with_at(region, pos, kToolOpen)) {
                    calls.trailing = pos;
                    return calls.empty() ? FallbackReason::MalformedStructure
                                         : FallbackReason::TrailingContent;
                }
                pos += kToolOpen.size();
                // The call's JSON value ends the body, so a closer inside one of its strings
                // stays in the argument; the closer follows the value.
                skip_format_whitespace(region, pos);
                const std::optional<std::size_t> end = json_value_end(region, pos);
                if (!end) { return FallbackReason::MalformedStructure; }
                std::size_t next = *end;
                skip_format_whitespace(region, next);
                if (starts_with_at(region, next, kToolClose)) {
                    next += kToolClose.size();
                } else {
                    // The turn ended inside or before the closer of a complete call (a small
                    // quant stops there now and then, as with the Qwen XML closers): keep it.
                    const std::string_view rest = region.substr(next);
                    if (rest.size() >= kToolClose.size() ||
                        kToolClose.substr(0, rest.size()) != rest) {
                        return FallbackReason::MalformedStructure;
                    }
                    next = region.size();
                }
                const std::string_view body = region.substr(pos, *end - pos);
                std::optional<ParsedCall> call =
                    json_call(OrderedJson::parse(body, nullptr, false));
                if (!call) { return FallbackReason::MalformedStructure; }
                calls.push_back(std::move(*call));
                pos = next;
            }
        });
    }
    [[nodiscard]] std::string structural_tag(std::span<const ToolCallConstraintTool> tools,
                                             ToolCallRequirement requirement,
                                             std::string_view named_tool,
                                             bool parallel_tool_calls) const override {
        return calls_structural_tag(
            tools, requirement, named_tool, parallel_tool_calls, kToolOpen, "\n",
            [](const ToolCallConstraintTool& tool) {
                return OrderedJson{
                    {"type", "tag"},
                    {"begin", std::string(kToolOpen) + "\n{\"name\": " +
                                  OrderedJson(tool.name).dump() + ", \"arguments\": "},
                    {"content",
                     OrderedJson{{"type", "json_schema"}, {"json_schema", schema_or_object(tool)}}},
                    {"end", "}\n" + std::string(kToolClose)}};
            });
    }
};

// <|tool_call>call:NAME{key:value,...}<tool_call|>; strings are wrapped in <|"|>, keys are bare.
class Gemma4Syntax final : public ToolCallSyntax {
public:
    static constexpr std::string_view kOpen  = "<|tool_call>";
    static constexpr std::string_view kClose = "<tool_call|>";
    static constexpr std::string_view kQuote = "<|\"|>";
    static constexpr std::string_view kCall  = "call:";


    [[nodiscard]] std::string_view open() const noexcept override { return kOpen; }

    [[nodiscard]] std::string_view close() const noexcept override { return kClose; }
    [[nodiscard]] ParsedToolCallOutput parse(const std::string& text, std::size_t max_length,
                                             const Contract& contract) const override {
        return parse_calls(text, kOpen, max_length, contract,
                           [](std::string_view region, ParsedCalls& calls) {
            std::size_t pos = 0;
            for (;;) {
                skip_format_whitespace(region, pos);
                if (pos == region.size()) {
                    return calls.empty() ? FallbackReason::MalformedStructure
                                         : FallbackReason::None;
                }
                if (!starts_with_at(region, pos, kOpen)) {
                    calls.trailing = pos;
                    return calls.empty() ? FallbackReason::MalformedStructure
                                         : FallbackReason::TrailingContent;
                }
                pos += kOpen.size();
                if (starts_with_at(region, pos, kCall)) {
                    pos += kCall.size();
                } else if (starts_with_at(region, pos, ":")) {
                    // 4-bit Gemma 4 drops the call keyword and keeps its colon,
                    // <|tool_call>:name{...}: the same call.
                    ++pos;
                } else {
                    return FallbackReason::MalformedStructure;
                }
                const std::size_t brace = region.find('{', pos);
                if (brace == std::string_view::npos || brace == pos) {
                    return FallbackReason::InvalidToolName;
                }
                ParsedCall call{std::string(region.substr(pos, brace - pos)), {}};
                pos = brace;
                Reader reader{region, pos};
                std::optional<OrderedJson> arguments = reader.object();
                if (!arguments) { return FallbackReason::MalformedStructure; }
                pos = reader.pos;
                if (starts_with_at(region, pos, kClose)) {
                    pos += kClose.size();
                } else {
                    // The turn ended where the closer belongs, after a complete call: keep it.
                    std::size_t tail = pos;
                    skip_format_whitespace(region, tail);
                    if (tail != region.size()) { return FallbackReason::MalformedStructure; }
                    pos = tail;
                }
                call.arguments_json = arguments->dump();
                calls.push_back(std::move(call));
            }
        });
    }
    [[nodiscard]] std::string structural_tag(std::span<const ToolCallConstraintTool> tools,
                                             ToolCallRequirement requirement,
                                             std::string_view named_tool,
                                             bool parallel_tool_calls) const override {
        // Arguments are Gemma's own value syntax, not JSON: an object of bare (or <|"|>-quoted)
        // keys, <|"|>-quoted strings and bare numbers, booleans and null, braces included
        // (kGemma4ArgumentStyle), held to a strict tool's schema and otherwise any such object.
        // Free text up to the close let a model made to call with nothing to call (tool_choice
        // "required") open a thought channel inside the arguments, or write on past their brace,
        // and never close the call (gemma-4-26B-A4B-it Q4_0, "Hello there!").
        return calls_structural_tag(
            tools, requirement, named_tool, parallel_tool_calls, kOpen, "",
            [](const ToolCallConstraintTool& tool) {
                return OrderedJson{
                    {"type", "tag"},
                    {"begin", std::string(kOpen) + std::string(kCall) + tool.name},
                    {"content", argument_style_format(tool.strict ? OrderedJson::parse(
                                                                        tool.parameters_json)
                                                                  : OrderedJson{{"type", "object"}},
                                                      kGemma4ArgumentStyle)},
                    {"end", std::string(kClose)}};
            });
    }

private:
    struct Reader {
        std::string_view text;
        std::size_t pos = 0;

        void skip() { skip_format_whitespace(text, pos); }

        std::optional<std::string> quoted() {
            if (!starts_with_at(text, pos, kQuote)) { return std::nullopt; }
            const std::size_t begin = pos + kQuote.size();
            const std::size_t end   = text.find(kQuote, begin);
            if (end == std::string_view::npos) { return std::nullopt; }
            pos = end + kQuote.size();
            return std::string(text.substr(begin, end - begin));
        }

        std::optional<OrderedJson> value() {
            skip();
            if (pos >= text.size()) { return std::nullopt; }
            if (starts_with_at(text, pos, kQuote)) {
                std::optional<std::string> string = quoted();
                if (!string) { return std::nullopt; }
                return OrderedJson(std::move(*string));
            }
            if (text[pos] == '{') { return object(); }
            if (text[pos] == '[') { return array(); }
            const std::size_t begin = pos;
            while (pos < text.size() && text[pos] != ',' && text[pos] != '}' && text[pos] != ']') {
                ++pos;
            }
            const std::string_view bare = trim_format_whitespace(text.substr(begin, pos - begin));
            if (bare.empty()) { return std::nullopt; }
            OrderedJson literal = OrderedJson::parse(bare, nullptr, false);
            if (!literal.is_discarded() && !literal.is_structured() && !literal.is_string()) {
                return literal;
            }
            return OrderedJson(std::string(bare));
        }

        std::optional<OrderedJson> object() {
            if (pos >= text.size() || text[pos] != '{') { return std::nullopt; }
            ++pos;
            OrderedJson result = OrderedJson::object();
            skip();
            if (pos < text.size() && text[pos] == '}') {
                ++pos;
                return result;
            }
            for (;;) {
                skip();
                std::string key;
                if (starts_with_at(text, pos, kQuote)) {
                    std::optional<std::string> quoted_key = quoted();
                    if (!quoted_key) { return std::nullopt; }
                    key = std::move(*quoted_key);
                    skip();
                    if (pos >= text.size() || text[pos] != ':') { return std::nullopt; }
                } else {
                    const std::size_t colon = text.find(':', pos);
                    if (colon == std::string_view::npos) { return std::nullopt; }
                    key = std::string(trim_format_whitespace(text.substr(pos, colon - pos)));
                    pos = colon;
                }
                if (key.empty() || result.contains(key)) { return std::nullopt; }
                ++pos;  // ':'
                std::optional<OrderedJson> item = value();
                if (!item) { return std::nullopt; }
                result[key] = std::move(*item);
                skip();
                if (pos < text.size() && text[pos] == ',') {
                    ++pos;
                    continue;
                }
                if (pos < text.size() && text[pos] == '}') {
                    ++pos;
                    return result;
                }
                return std::nullopt;
            }
        }

        std::optional<OrderedJson> array() {
            ++pos;  // '['
            OrderedJson result = OrderedJson::array();
            skip();
            if (pos < text.size() && text[pos] == ']') {
                ++pos;
                return result;
            }
            for (;;) {
                std::optional<OrderedJson> item = value();
                if (!item) { return std::nullopt; }
                result.push_back(std::move(*item));
                skip();
                if (pos < text.size() && text[pos] == ',') {
                    ++pos;
                    continue;
                }
                if (pos < text.size() && text[pos] == ']') {
                    ++pos;
                    return result;
                }
                return std::nullopt;
            }
        }
    };
};

// [TOOL_CALLS]NAME[ARGS]{...} per call, or the older [TOOL_CALLS][{"name":...,"arguments":...}].
class MistralJsonSyntax final : public ToolCallSyntax {
public:
    static constexpr std::string_view kOpen = "[TOOL_CALLS]";
    static constexpr std::string_view kArgs = "[ARGS]";

    [[nodiscard]] std::string_view open() const noexcept override { return kOpen; }
    [[nodiscard]] ParsedToolCallOutput parse(const std::string& text, std::size_t max_length,
                                             const Contract& contract) const override {
        return parse_calls(text, kOpen, max_length, contract,
                           [](std::string_view region, ParsedCalls& calls) {
            std::size_t pos = 0;
            for (;;) {
                skip_format_whitespace(region, pos);
                if (pos == region.size()) {
                    return calls.empty() ? FallbackReason::MalformedStructure
                                         : FallbackReason::None;
                }
                if (!starts_with_at(region, pos, kOpen)) {
                    calls.trailing = pos;
                    return calls.empty() ? FallbackReason::MalformedStructure
                                         : FallbackReason::TrailingContent;
                }
                pos += kOpen.size();
                skip_format_whitespace(region, pos);
                if (pos < region.size() && region[pos] == '[') {
                    const std::optional<std::size_t> end = json_value_end(region, pos);
                    if (!end) { return FallbackReason::MalformedStructure; }
                    const OrderedJson list =
                        OrderedJson::parse(region.substr(pos, *end - pos), nullptr, false);
                    if (list.is_discarded() || !list.is_array() || list.empty()) {
                        return FallbackReason::MalformedStructure;
                    }
                    for (const OrderedJson& item : list) {
                        std::optional<ParsedCall> call = json_call(item);
                        if (!call) { return FallbackReason::MalformedStructure; }
                        calls.push_back(std::move(*call));
                    }
                    pos = *end;
                    continue;
                }
                const std::size_t args = region.find(kArgs, pos);
                if (args == std::string_view::npos || args == pos) {
                    return FallbackReason::InvalidToolName;
                }
                ParsedCall call{std::string(region.substr(pos, args - pos)), {}};
                pos = args + kArgs.size();
                skip_format_whitespace(region, pos);
                const std::optional<std::size_t> end = json_value_end(region, pos);
                if (!end || region[pos] != '{') { return FallbackReason::MalformedStructure; }
                const OrderedJson arguments =
                    OrderedJson::parse(region.substr(pos, *end - pos), nullptr, false);
                if (arguments.is_discarded() || !arguments.is_object()) {
                    return FallbackReason::MalformedStructure;
                }
                call.arguments_json = arguments.dump();
                calls.push_back(std::move(call));
                pos = *end;
            }
        });
    }
    [[nodiscard]] std::string structural_tag(std::span<const ToolCallConstraintTool> tools,
                                             ToolCallRequirement requirement,
                                             std::string_view named_tool,
                                             bool parallel_tool_calls) const override {
        return calls_structural_tag(tools, requirement, named_tool, parallel_tool_calls, kOpen, "",
                                    [](const ToolCallConstraintTool& tool) {
            return OrderedJson{
                {"type", "tag"},
                {"begin", std::string(kOpen) + tool.name + std::string(kArgs)},
                {"content",
                 OrderedJson{{"type", "json_schema"}, {"json_schema", schema_or_object(tool)}}},
                {"end", ""}};
        });
    }
};

// {"name": NAME, "parameters": {...}}: Llama 3.1+'s JSON tool calling, where the call is the reply
// itself, with no marker around it; calls are separated by whitespace or ';'.
class Llama3JsonSyntax final : public ToolCallSyntax {
public:
    static constexpr std::string_view kOpen = "{\"name\": ";
    // Llama 3.1 marks a tool call with <|python_tag|> (the ipython environment); the JSON form
    // may follow it, so the tag belongs to the call.
    static constexpr std::string_view kLead = "<|python_tag|>";

    [[nodiscard]] std::string_view open() const noexcept override { return kOpen; }
    [[nodiscard]] std::string_view lead() const noexcept override { return kLead; }
    [[nodiscard]] ParsedToolCallOutput parse(const std::string& text, std::size_t max_length,
                                             const Contract& contract) const override {
        ParsedToolCallOutput out = parse_json_calls(text, max_length, contract);
        // The open marker is any JSON answer's: only declared names tell a call from one.
        if (out.diagnostics.undeclared_calls != 0) {
            return fallback(
                text, ToolCallParseDiagnostics{.marker_seen     = true,
                                               .fallback_reason = FallbackReason::UndeclaredTool});
        }
        if (out.is_tool_call_response && out.content.ends_with(kLead)) {
            out.content = rtrim_format_whitespace(
                std::string_view(out.content).substr(0, out.content.size() - kLead.size()));
        }
        return out;
    }
    [[nodiscard]] std::string structural_tag(std::span<const ToolCallConstraintTool> tools,
                                             ToolCallRequirement requirement,
                                             std::string_view named_tool,
                                             bool parallel_tool_calls) const override {
        return calls_structural_tag(
            tools, requirement, named_tool, parallel_tool_calls, kOpen, "; ",
            [](const ToolCallConstraintTool& tool) {
                return OrderedJson{
                    {"type", "tag"},
                    {"begin", std::string(kOpen) + OrderedJson(tool.name).dump() +
                                  ", \"parameters\": "},
                    {"content",
                     OrderedJson{{"type", "json_schema"}, {"json_schema", schema_or_object(tool)}}},
                    {"end", "}"}};
            });
    }

private:
    static ParsedToolCallOutput parse_json_calls(const std::string& text, std::size_t max_length,
                                                 const Contract& contract) {
        return parse_calls(text, kOpen, max_length, contract,
                           [](std::string_view region, ParsedCalls& calls) {
            std::size_t pos = 0;
            for (;;) {
                while (pos < region.size() &&
                       (is_format_whitespace(region[pos]) || region[pos] == ';')) {
                    ++pos;
                }
                if (pos == region.size()) {
                    return calls.empty() ? FallbackReason::MalformedStructure
                                         : FallbackReason::None;
                }
                if (!starts_with_at(region, pos, kOpen)) {
                    calls.trailing = pos;
                    return calls.empty() ? FallbackReason::MalformedStructure
                                         : FallbackReason::TrailingContent;
                }
                const std::optional<std::size_t> end = json_value_end(region, pos);
                if (!end) { return FallbackReason::MalformedStructure; }
                std::optional<ParsedCall> call =
                    json_call(OrderedJson::parse(region.substr(pos, *end - pos), nullptr, false));
                if (!call) { return FallbackReason::MalformedStructure; }
                calls.push_back(std::move(*call));
                pos = *end;
            }
        });
    }
};

// OpenAI harmony (gpt-oss): a call is a message of its own to a function recipient,
//   <|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{...}<|call|>
// with the recipient written in the channel or in the role (" to=functions.NAME<|channel|>...");
// <|call|> ends generation, so a reply's last call may arrive without it.
class HarmonySyntax final : public ToolCallSyntax {
public:
    static constexpr std::string_view kOpen      = "<|start|>assistant";
    static constexpr std::string_view kRecipient = "to=functions.";
    static constexpr std::string_view kMessage   = "<|message|>";
    static constexpr std::string_view kCall      = "<|call|>";

    [[nodiscard]] std::string_view open() const noexcept override { return kOpen; }
    [[nodiscard]] ParsedToolCallOutput parse(const std::string& text, std::size_t max_length,
                                             const Contract& contract) const override {
        return parse_calls(text, kOpen, max_length, contract,
                           [](std::string_view region, ParsedCalls& calls) {
            std::size_t pos = 0;
            for (;;) {
                skip_format_whitespace(region, pos);
                if (pos == region.size()) {
                    return calls.empty() ? FallbackReason::MalformedStructure
                                         : FallbackReason::None;
                }
                if (!starts_with_at(region, pos, kOpen)) {
                    calls.trailing = pos;
                    return calls.empty() ? FallbackReason::MalformedStructure
                                         : FallbackReason::TrailingContent;
                }
                const std::size_t header_end = region.find(kMessage, pos);
                if (header_end == std::string_view::npos) {
                    return FallbackReason::MalformedStructure;
                }
                const std::string_view header = region.substr(pos, header_end - pos);
                const std::size_t recipient   = header.find(kRecipient);
                if (recipient == std::string_view::npos ||
                    header.find("<|channel|>commentary") == std::string_view::npos) {
                    return FallbackReason::MalformedStructure;
                }
                std::size_t name_end = recipient + kRecipient.size();
                while (name_end < header.size() && !is_format_whitespace(header[name_end]) &&
                       header[name_end] != '<') {
                    ++name_end;
                }
                std::string name(header.substr(recipient + kRecipient.size(),
                                               name_end - recipient - kRecipient.size()));
                pos = header_end + kMessage.size();
                skip_format_whitespace(region, pos);
                if (pos == region.size() || region[pos] != '{') {
                    return FallbackReason::MalformedStructure;
                }
                const std::optional<std::size_t> end = json_value_end(region, pos);
                if (!end) { return FallbackReason::MalformedStructure; }
                const OrderedJson arguments =
                    OrderedJson::parse(region.substr(pos, *end - pos), nullptr, false);
                if (arguments.is_discarded() || !arguments.is_object()) {
                    return FallbackReason::MalformedStructure;
                }
                calls.push_back(ParsedCall{std::move(name), arguments.dump()});
                pos = *end;
                skip_format_whitespace(region, pos);
                if (starts_with_at(region, pos, kCall)) { pos += kCall.size(); }
            }
        });
    }
    [[nodiscard]] std::string structural_tag(std::span<const ToolCallConstraintTool> tools,
                                             ToolCallRequirement requirement,
                                             std::string_view named_tool,
                                             bool parallel_tool_calls) const override {
        return calls_structural_tag(
            tools, requirement, named_tool, parallel_tool_calls, kOpen, "",
            [](const ToolCallConstraintTool& tool) {
                return OrderedJson{
                    {"type", "tag"},
                    {"begin", std::string(kOpen) + "<|channel|>commentary " +
                                  std::string(kRecipient) + tool.name +
                                  " <|constrain|>json" + std::string(kMessage)},
                    {"content",
                     OrderedJson{{"type", "json_schema"}, {"json_schema", schema_or_object(tool)}}},
                    {"end", std::string(kCall)}};
            });
    }
};

// One argument value of a Python-style call: a JSON value, or Python's True, False, None or a
// single-quoted string, read as JSON. Returns the value and advances pos past it.
std::optional<OrderedJson> pythonic_value(std::string_view text, std::size_t& pos) {
    if (pos >= text.size()) { return std::nullopt; }
    const char first = text[pos];
    std::size_t end  = pos;
    if (first == '{' || first == '[') {
        const std::optional<std::size_t> close = json_value_end(text, pos);
        if (!close) { return std::nullopt; }
        end = *close;
    } else if (first == '"' || first == '\'') {
        std::string value;
        for (end = pos + 1; end < text.size() && text[end] != first; ++end) {
            if (text[end] == '\\' && end + 1 < text.size()) { value.push_back(text[end++]); }
            value.push_back(text[end]);
        }
        if (end >= text.size()) { return std::nullopt; }
        ++end;
        if (first == '"') {
            OrderedJson parsed =
                OrderedJson::parse(text.substr(pos, end - pos), nullptr, false);
            if (parsed.is_discarded()) { return std::nullopt; }
            pos = end;
            return parsed;
        }
        // A single-quoted string: its escapes are JSON's once the quote is.
        std::string json = "\"";
        for (std::size_t i = 0; i < value.size(); ++i) {
            if (value[i] == '\\' && i + 1 < value.size() && value[i + 1] == '\'') { continue; }
            if (value[i] == '"') { json.push_back('\\'); }
            json.push_back(value[i]);
        }
        json.push_back('"');
        OrderedJson parsed = OrderedJson::parse(json, nullptr, false);
        if (parsed.is_discarded()) { return std::nullopt; }
        pos = end;
        return parsed;
    } else {
        while (end < text.size() && text[end] != ',' && text[end] != ')' &&
               !is_format_whitespace(text[end])) {
            ++end;
        }
        const std::string_view word = text.substr(pos, end - pos);
        OrderedJson parsed;
        if (word == "True") {
            parsed = true;
        } else if (word == "False") {
            parsed = false;
        } else if (word == "None") {
            parsed = nullptr;
        } else {
            parsed = OrderedJson::parse(word, nullptr, false);
            if (parsed.is_discarded() || parsed.is_object() || parsed.is_array() ||
                parsed.is_string()) {
                return std::nullopt;
            }
        }
        pos = end;
        return parsed;
    }
    OrderedJson parsed = OrderedJson::parse(text.substr(pos, end - pos), nullptr, false);
    if (parsed.is_discarded()) { return std::nullopt; }
    pos = end;
    return parsed;
}

bool python_identifier_byte(char byte, bool first) noexcept {
    return byte == '_' || (byte >= 'A' && byte <= 'Z') || (byte >= 'a' && byte <= 'z') ||
           (!first && byte >= '0' && byte <= '9');
}

// NAME(key=value, ...) at pos, read value by value so a parenthesis, a newline or the close
// marker inside a string stays in it; advances pos past the closing parenthesis. A positional
// argument (no name to give it) or a key that is not a Python identifier is malformed.
std::optional<ParsedCall> pythonic_call(std::string_view text, std::size_t& pos) {
    const std::size_t open = text.find('(', pos);
    if (open == std::string_view::npos) { return std::nullopt; }
    const std::string_view name = trim_format_whitespace(text.substr(pos, open - pos));
    if (name.empty() || name.find_first_of(" \t\r\n") != std::string_view::npos) {
        return std::nullopt;
    }
    ParsedCall call{std::string(name), {}};
    OrderedJson object = OrderedJson::object();
    std::size_t at     = open + 1;
    for (;;) {
        skip_format_whitespace(text, at);
        if (at < text.size() && text[at] == ')') { break; }
        const std::size_t key_begin = at;
        while (at < text.size() && python_identifier_byte(text[at], at == key_begin)) { ++at; }
        const std::string key(text.substr(key_begin, at - key_begin));
        skip_format_whitespace(text, at);
        if (key.empty() || at >= text.size() || text[at] != '=' || object.contains(key)) {
            return std::nullopt;
        }
        ++at;
        skip_format_whitespace(text, at);
        std::optional<OrderedJson> value = pythonic_value(text, at);
        if (!value) { return std::nullopt; }
        object[key] = std::move(*value);
        skip_format_whitespace(text, at);
        if (at >= text.size()) { return std::nullopt; }
        if (text[at] == ')') { break; }
        if (text[at] != ',') { return std::nullopt; }
        ++at;
    }
    pos                 = at + 1;
    call.arguments_json = object.dump();
    return call;
}

// <function_calls>NAME(key=value, ...)\nNAME(...)</function_calls>: OLMo 3's calls, one per line,
// each argument a JSON value (its template writes value | tojson) or a Python literal.
class PythonicSyntax final : public ToolCallSyntax {
public:
    static constexpr std::string_view kOpen  = "<function_calls>";
    static constexpr std::string_view kClose = "</function_calls>";

    [[nodiscard]] std::string_view open() const noexcept override { return kOpen; }
    [[nodiscard]] ParsedToolCallOutput parse(const std::string& text, std::size_t max_length,
                                             const Contract& contract) const override {
        return parse_calls(text, kOpen, max_length, contract,
                           [](std::string_view region, ParsedCalls& calls) {
            std::size_t pos = 0;
            for (;;) {
                skip_format_whitespace(region, pos);
                if (pos == region.size()) {
                    return calls.empty() ? FallbackReason::MalformedStructure
                                         : FallbackReason::None;
                }
                if (!starts_with_at(region, pos, kOpen)) {
                    calls.trailing = pos;
                    return calls.empty() ? FallbackReason::MalformedStructure
                                         : FallbackReason::TrailingContent;
                }
                pos += kOpen.size();
                // Calls one per line up to the close, which may be missing when the reply ends
                // right after the last call.
                bool any = false;
                for (;;) {
                    skip_format_whitespace(region, pos);
                    if (pos == region.size()) { break; }
                    if (starts_with_at(region, pos, kClose)) {
                        pos += kClose.size();
                        break;
                    }
                    std::optional<ParsedCall> call = pythonic_call(region, pos);
                    if (!call) { return FallbackReason::MalformedStructure; }
                    calls.push_back(std::move(*call));
                    any = true;
                }
                if (!any) { return FallbackReason::MalformedStructure; }
            }
        });
    }
    [[nodiscard]] std::string structural_tag(std::span<const ToolCallConstraintTool> tools,
                                             ToolCallRequirement requirement,
                                             std::string_view named_tool,
                                             bool parallel_tool_calls) const override {
        // Arguments are Python-style keywords: free text, or for a strict tool the keywords held
        // to its schema (kPythonicArgumentStyle).
        return calls_structural_tag(tools, requirement, named_tool, parallel_tool_calls, kOpen,
                                    "\n", [](const ToolCallConstraintTool& tool) {
            return OrderedJson{{"type", "tag"},
                               {"begin", std::string(kOpen) + tool.name + "("},
                               {"content", tool.strict
                                               ? argument_style_format(OrderedJson::parse(tool.parameters_json),
                                                                       kPythonicArgumentStyle)
                                               : OrderedJson{{"type", "any_text"}}},
                               {"end", ")" + std::string(kClose)}};
        });
    }
};

// <|tool_call|> then a JSON list of {"name": NAME, "arguments": {...}} (Granite 3.x; vLLM's granite
// parser). Several lists may follow each other: a constrained reply writes one list per call.
class GraniteJsonSyntax final : public ToolCallSyntax {
public:
    static constexpr std::string_view kOpen = "<|tool_call|>";

    [[nodiscard]] std::string_view open() const noexcept override { return kOpen; }
    [[nodiscard]] ParsedToolCallOutput parse(const std::string& text, std::size_t max_length,
                                             const Contract& contract) const override {
        return parse_calls(text, kOpen, max_length, contract,
                           [](std::string_view region, ParsedCalls& calls) {
            std::size_t pos = 0;
            for (;;) {
                skip_format_whitespace(region, pos);
                if (pos == region.size()) {
                    return calls.empty() ? FallbackReason::MalformedStructure
                                         : FallbackReason::None;
                }
                if (!starts_with_at(region, pos, kOpen)) {
                    calls.trailing = pos;
                    return calls.empty() ? FallbackReason::MalformedStructure
                                         : FallbackReason::TrailingContent;
                }
                pos += kOpen.size();
                skip_format_whitespace(region, pos);
                if (pos == region.size() || region[pos] != '[') {
                    return FallbackReason::MalformedStructure;
                }
                const std::optional<std::size_t> end = json_value_end(region, pos);
                if (!end) { return FallbackReason::MalformedStructure; }
                const OrderedJson list =
                    OrderedJson::parse(region.substr(pos, *end - pos), nullptr, false);
                if (list.is_discarded() || !list.is_array() || list.empty()) {
                    return FallbackReason::MalformedStructure;
                }
                for (const OrderedJson& item : list) {
                    std::optional<ParsedCall> call = json_call(item);
                    if (!call) { return FallbackReason::MalformedStructure; }
                    calls.push_back(std::move(*call));
                }
                pos = *end;
            }
        });
    }
    [[nodiscard]] std::string structural_tag(std::span<const ToolCallConstraintTool> tools,
                                             ToolCallRequirement requirement,
                                             std::string_view named_tool,
                                             bool parallel_tool_calls) const override {
        return calls_structural_tag(
            tools, requirement, named_tool, parallel_tool_calls, kOpen, "",
            [](const ToolCallConstraintTool& tool) {
                return OrderedJson{
                    {"type", "tag"},
                    {"begin", std::string(kOpen) + "[{\"name\": " + OrderedJson(tool.name).dump() +
                                  ", \"arguments\": "},
                    {"content",
                     OrderedJson{{"type", "json_schema"}, {"json_schema", schema_or_object(tool)}}},
                    {"end", "}]"}};
            });
    }
};

// ATEM (Muse Glimmer): every call is an assistant message addressed to the function,
//   <|start|>assistant to=NAME<|message|><atem:function_calls>
//   <atem:invoke name="NAME">
//   <atem:parameter name="K">V</atem:parameter>
//   </atem:invoke>
//   </atem:function_calls>
// and consecutive calls are separated by <|eom|>; the last ends with <|eot|>, a stop token. The
// generation prompt ends at "<|start|>assistant", so a reply that calls at once begins with
// " to=NAME<|message|>". V is a string or scalar as is, or a list or object as JSON. The header's
// recipient must be the invoked function: the template writes a call only that way.
class AtemSyntax final : public ToolCallSyntax {
public:
    static constexpr std::string_view kStart       = "<|start|>assistant";
    static constexpr std::string_view kTo          = "to=";
    static constexpr std::string_view kMessage     = "<|message|>";
    static constexpr std::string_view kEom         = "<|eom|>";
    static constexpr std::string_view kOpen        = "<atem:function_calls>";
    static constexpr std::string_view kClose       = "</atem:function_calls>";
    static constexpr std::string_view kInvoke      = "<atem:invoke name=\"";
    static constexpr std::string_view kInvokeClose = "</atem:invoke>";
    static constexpr std::string_view kParam       = "<atem:parameter name=\"";
    static constexpr std::string_view kParamClose  = "</atem:parameter>";
    static constexpr std::string_view kNameEnd     = "\">";

    [[nodiscard]] std::string_view open() const noexcept override { return kOpen; }

    [[nodiscard]] ToolCallStart match_start(std::string_view held) const override {
        std::size_t pos = 0;
        skip_format_whitespace(held, pos);
        const Header header = match_header(held, pos);
        if (header.state != ToolCallStart::Complete) { return header.state; }
        pos = header.end;
        skip_format_whitespace(held, pos);
        const std::string_view rest = held.substr(pos);
        if (rest.size() < kOpen.size()) {
            return kOpen.starts_with(rest) ? ToolCallStart::Partial : ToolCallStart::No;
        }
        return rest.starts_with(kOpen) ? ToolCallStart::Complete : ToolCallStart::No;
    }

    [[nodiscard]] ParsedToolCallOutput parse(const std::string& text, std::size_t max_length,
                                             const Contract& contract) const override {
        const std::size_t block = text.find(kOpen);
        if (block == std::string::npos) { return fallback(text); }
        ParsedToolCallOutput out;
        out.diagnostics.marker_seen = true;
        const std::optional<std::size_t> start = header_before(text, block);
        if (!start) {
            out.diagnostics.fallback_reason = FallbackReason::MalformedStructure;
            return fallback(text, out.diagnostics);
        }
        RawCalls raw_calls;
        FallbackReason failure =
            parse_region(std::string_view(text).substr(*start), raw_calls);
        // One block, no quoted-marker retry: an undeclared name is a call (find_call_region).
        for (const RawToolCall& call : raw_calls) {
            if (failure != FallbackReason::None && failure != FallbackReason::TrailingContent) {
                break;
            }
            if (!valid_function_name(call.name, max_length)) {
                failure = FallbackReason::InvalidToolName;
            }
        }
        std::optional<std::string> content =
            failure == FallbackReason::None || failure == FallbackReason::TrailingContent
                ? content_around_calls(text, CallRegion{*start, failure}, raw_calls.trailing, kOpen,
                                       out.diagnostics)
                : std::nullopt;
        if (!content) {
            out.diagnostics.fallback_reason = failure;
            return fallback(text, out.diagnostics);
        }
        out.content = std::move(*content);
        out.diagnostics.undeclared_calls = count_undeclared(contract, raw_calls);
        out.tool_calls.reserve(raw_calls.size());
        for (const RawToolCall& raw : raw_calls) {
            out.tool_calls.push_back(
                normalize_raw_tool_call(raw, contract, out.diagnostics, ValueFraming::AsIs));
        }
        out.diagnostics.structured_call_count = static_cast<std::uint32_t>(out.tool_calls.size());
        out.is_tool_call_response             = true;
        return out;
    }

    [[nodiscard]] std::string structural_tag(std::span<const ToolCallConstraintTool> tools,
                                             ToolCallRequirement requirement,
                                             std::string_view named_tool,
                                             bool parallel_tool_calls) const override {
        // Arguments are tagged text: free, or for a strict tool the parameters held to its schema
        // (kAtemArgumentStyle). A call right after the generation prompt continues its
        // "<|start|>assistant" header; later calls open their own message.
        return calls_structural_tag(
            tools, requirement, named_tool, parallel_tool_calls, kOpen,
            std::string(kEom) + std::string(kStart), [](const ToolCallConstraintTool& tool) {
                return OrderedJson{
                    {"type", "tag"},
                    {"begin", " " + std::string(kTo) + tool.name + std::string(kMessage) +
                                  std::string(kOpen) + "\n" + std::string(kInvoke) + tool.name +
                                  std::string(kNameEnd) + "\n"},
                    {"content", tool.strict ? argument_style_format(OrderedJson::parse(tool.parameters_json),
                                                                    kAtemArgumentStyle)
                                            : OrderedJson{{"type", "any_text"}}},
                    {"end", std::string(kInvokeClose) + "\n" + std::string(kClose)}};
            });
    }

private:
    struct Header {
        ToolCallStart state = ToolCallStart::No;
        std::size_t end     = 0;
        std::string_view recipient;
    };

    static constexpr bool name_byte(char byte) { return tool_name_byte(byte); }

    // A call header at pos: an optional "<|start|>assistant", whitespace, then
    // "to=NAME<|message|>". Partial when the text ends inside one.
    static Header match_header(std::string_view text, std::size_t pos) {
        std::string_view rest = text.substr(pos);
        if (rest.starts_with(kStart)) {
            pos += kStart.size();
        } else if (kStart.starts_with(rest)) {
            return {.state = ToolCallStart::Partial};
        }
        skip_format_whitespace(text, pos);
        rest = text.substr(pos);
        if (rest.size() < kTo.size()) {
            return {.state = kTo.starts_with(rest) ? ToolCallStart::Partial : ToolCallStart::No};
        }
        if (!rest.starts_with(kTo)) { return {}; }
        pos += kTo.size();
        const std::size_t name_begin = pos;
        while (pos < text.size() && name_byte(text[pos])) { ++pos; }
        if (pos == text.size()) { return {.state = ToolCallStart::Partial}; }
        if (pos == name_begin) { return {}; }
        const std::string_view recipient = text.substr(name_begin, pos - name_begin);
        rest                             = text.substr(pos);
        if (rest.size() < kMessage.size()) {
            return {.state = kMessage.starts_with(rest) ? ToolCallStart::Partial : ToolCallStart::No};
        }
        if (!rest.starts_with(kMessage)) { return {}; }
        return {.state = ToolCallStart::Complete, .end = pos + kMessage.size(), .recipient = recipient};
    }

    // Where the header of the call block at `block` begins: "to=NAME<|message|>" and whitespace
    // right before the block, with the whitespace and "<|start|>assistant" before it.
    static std::optional<std::size_t> header_before(std::string_view text, std::size_t block) {
        std::size_t at = block;
        while (at > 0 && is_format_whitespace(text[at - 1])) { --at; }
        if (at < kMessage.size() || text.substr(at - kMessage.size(), kMessage.size()) != kMessage) {
            return std::nullopt;
        }
        at -= kMessage.size();
        const std::size_t name_end = at;
        while (at > 0 && name_byte(text[at - 1])) { --at; }
        if (at == name_end || at < kTo.size() || text.substr(at - kTo.size(), kTo.size()) != kTo) {
            return std::nullopt;
        }
        at -= kTo.size();
        while (at > 0 && is_format_whitespace(text[at - 1])) { --at; }
        if (at >= kStart.size() && text.substr(at - kStart.size(), kStart.size()) == kStart) {
            at -= kStart.size();
        }
        return at;
    }

    // Calls from the region start: header, block of invokes, optional <|eom|>, repeated.
    static FallbackReason parse_region(std::string_view region, RawCalls& calls) {
        std::size_t pos = 0;
        for (;;) {
            skip_format_whitespace(region, pos);
            if (pos == region.size()) {
                return calls.empty() ? FallbackReason::MalformedStructure : FallbackReason::None;
            }
            const Header header = match_header(region, pos);
            if (header.state != ToolCallStart::Complete) {
                calls.trailing = pos;
                return calls.empty() ? FallbackReason::MalformedStructure
                                     : FallbackReason::TrailingContent;
            }
            pos = header.end;
            skip_format_whitespace(region, pos);
            if (!starts_with_at(region, pos, kOpen)) { return FallbackReason::MalformedStructure; }
            pos += kOpen.size();
            bool any = false;
            for (;;) {
                skip_format_whitespace(region, pos);
                if (!starts_with_at(region, pos, kInvoke)) { break; }
                pos += kInvoke.size();
                const std::size_t name_end = region.find(kNameEnd, pos);
                if (name_end == std::string_view::npos || name_end == pos) {
                    return FallbackReason::MalformedStructure;
                }
                RawToolCall call{.name = region.substr(pos, name_end - pos)};
                if (call.name != header.recipient) { return FallbackReason::MalformedStructure; }
                pos = name_end + kNameEnd.size();
                for (;;) {
                    skip_format_whitespace(region, pos);
                    if (!starts_with_at(region, pos, kParam)) { break; }
                    pos += kParam.size();
                    // An empty name (name="") is well formed: a schema may declare the key "".
                    const std::size_t key_end = region.find(kNameEnd, pos);
                    if (key_end == std::string_view::npos) {
                        return FallbackReason::MalformedStructure;
                    }
                    const std::string_view key    = region.substr(pos, key_end - pos);
                    const std::size_t value_begin = key_end + kNameEnd.size();
                    const std::size_t value_end   = region.find(kParamClose, value_begin);
                    if (value_end == std::string_view::npos) {
                        return FallbackReason::MalformedStructure;
                    }
                    set_parameter(call, key, region.substr(value_begin, value_end - value_begin));
                    pos = value_end + kParamClose.size();
                }
                if (!starts_with_at(region, pos, kInvokeClose)) {
                    return FallbackReason::MalformedStructure;
                }
                pos += kInvokeClose.size();
                calls.push_back(std::move(call));
                any = true;
            }
            if (!any || !starts_with_at(region, pos, kClose)) {
                return FallbackReason::MalformedStructure;
            }
            pos += kClose.size();
            skip_format_whitespace(region, pos);
            if (starts_with_at(region, pos, kEom)) { pos += kEom.size(); }
        }
    }
};

} // namespace

const ToolCallSyntax& tool_call_syntax(ToolCallFormat format) {
    static const QwenXmlSyntax qwen_xml;
    static const HermesJsonSyntax hermes_json;
    static const Gemma4Syntax gemma4;
    static const MistralJsonSyntax mistral_json;
    static const Llama3JsonSyntax llama3_json;
    static const HarmonySyntax harmony;
    static const PythonicSyntax pythonic;
    static const GraniteJsonSyntax granite_json;
    static const AtemSyntax atem;
    switch (format) {
    case ToolCallFormat::QwenXml:
        return qwen_xml;
    case ToolCallFormat::HermesJson:
        return hermes_json;
    case ToolCallFormat::Gemma4:
        return gemma4;
    case ToolCallFormat::MistralJson:
        return mistral_json;
    case ToolCallFormat::Llama3Json:
        return llama3_json;
    case ToolCallFormat::Harmony:
        return harmony;
    case ToolCallFormat::Pythonic:
        return pythonic;
    case ToolCallFormat::GraniteJson:
        return granite_json;
    case ToolCallFormat::Atem:
        return atem;
    case ToolCallFormat::None:
        break;
    }
    throw std::invalid_argument("this chat template declares no tool-call syntax");
}

} // namespace ninferno::chat
