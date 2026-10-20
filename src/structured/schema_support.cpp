#include "structured/schema_support.h"

#include "ninferno/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ninferno::structured {
namespace {

using Json = nlohmann::ordered_json;

constexpr std::size_t kMaximumDepth = 64;
constexpr std::size_t kMaximumNodes = 20000;

const std::unordered_set<std::string_view> kContainers = {"$defs", "definitions"};

// Every keyword the compiler implements. Anything else that is a JSON Schema assertion in any
// draft is rejected by name below. Other keywords are not JSON Schema vocabulary (vendor fields
// such as "javaName", draft-04 "id", Snowplow "self"); the specification requires validators to
// ignore them, so they are dropped and cannot make a standard validator reject the output.
const std::unordered_set<std::string_view> kEnforced = {
    "type",           "enum",          "const",         "$ref",           "anyOf",
    "oneOf",          "allOf",         "properties",    "required",       "additionalProperties",
    "unevaluatedProperties",           "patternProperties",               "propertyNames",
    "minProperties",  "maxProperties", "items",         "prefixItems",    "unevaluatedItems",
    "minItems",       "maxItems",      "uniqueItems",   "minimum",        "maximum",
    "exclusiveMinimum",                "exclusiveMaximum",                "multipleOf",
    "minLength",      "maxLength",     "pattern",       "format",
};

const std::unordered_set<std::string_view> kRejected = {
    "not",           "if",            "then",           "else",           "dependentRequired",
    "dependentSchemas",               "dependencies",   "contains",       "minContains",
    "maxContains",   "$dynamicRef",   "$dynamicAnchor", "$recursiveRef",  "$recursiveAnchor",
    "$vocabulary",   "contentSchema", "divisibleBy",    "disallow",       "extends",
};

const std::unordered_set<std::string_view> kFormats = {
    "email",    "date",          "time",         "date-time",   "duration",
    "hostname", "uuid",          "uri",          "uri-reference", "uri-template",
    "json-pointer",              "relative-json-pointer",
};

// Formats the JSON Schema specification defines but the grammar cannot enforce exactly: rejected.
// The ipv4 grammar admits leading zeros, which RFC 2673 (and Python's ipaddress) reject.
const std::unordered_set<std::string_view> kSpecFormatsNotEnforced = {
    "ipv4", "ipv6", "idn-email", "idn-hostname", "iri", "iri-reference", "regex",
};

// Every other "format" value is not defined by JSON Schema (OpenAPI "int32", "float", "byte",
// "password", vendor names); validators treat it as an annotation. int32/int64 on an integer
// schema become the matching range bounds, which only narrows the admitted values.
bool spec_defined_format(const Json& format) {
    if (!format.is_string()) { return true; }
    const std::string& name = format.get_ref<const std::string&>();
    return kFormats.contains(name) || kSpecFormatsNotEnforced.contains(name);
}

const std::array<std::string_view, 7> kTypes = {"null",   "boolean", "object", "array",
                                                "number", "integer", "string"};

[[noreturn]] void fail(const char* code, const std::string& pointer, const std::string& message) {
    throw OutputConstraintError(code, "JSON Schema at " + (pointer.empty() ? "#" : pointer) +
                                          ": " + message);
}

std::string escape_pointer(std::string_view token) {
    std::string out;
    for (const char c : token) {
        if (c == '~') {
            out += "~0";
        } else if (c == '/') {
            out += "~1";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

struct TypeSet {
    bool declared = false;
    std::unordered_set<std::string> names;

    [[nodiscard]] bool admits(std::string_view name) const {
        return !declared || names.contains(std::string(name));
    }
};

bool value_matches_type(const Json& value, std::string_view type) {
    if (type == "null") { return value.is_null(); }
    if (type == "boolean") { return value.is_boolean(); }
    if (type == "object") { return value.is_object(); }
    if (type == "array") { return value.is_array(); }
    if (type == "string") { return value.is_string(); }
    if (type == "number") { return value.is_number(); }
    if (type == "integer") {
        if (value.is_number_integer()) { return true; }
        if (!value.is_number_float()) { return false; }
        const double number = value.get<double>();
        return number == static_cast<double>(static_cast<long long>(number));
    }
    return false;
}

class Walker {
public:
    // allow_false: the position is an "additional" schema (items, additionalProperties, ...) where
    // false means "none", rather than a position that must admit a value.
    Json walk(const Json& node, const std::string& pointer, std::size_t depth,
              bool allow_false = false) {
        if (++nodes_ > kMaximumNodes) {
            fail("constraint_too_large", pointer,
                 "schema has more than " + std::to_string(kMaximumNodes) + " subschemas");
        }
        if (depth > kMaximumDepth) {
            fail("constraint_too_large", pointer,
                 "schema nesting exceeds " + std::to_string(kMaximumDepth) + " levels");
        }
        if (node.is_boolean()) {
            if (!node.get<bool>() && !allow_false) {
                fail("invalid_schema", pointer, "false admits no value");
            }
            return node;
        }
        if (!node.is_object()) { fail("invalid_schema", pointer, "a schema must be an object"); }

        // A subschema with its own base URI resolves "#/..." references against that URI, not
        // against this document's root, which is where the grammar compiler resolves them.
        const RebaseScope scope(*this, depth > 0 && changes_base_uri(node));

        // Keep only keywords the compiler acts on; annotations and non-vocabulary keywords are
        // dropped so they can never reach the compiler.
        Json kept = Json::object();
        for (auto it = node.begin(); it != node.end(); ++it) {
            const std::string& key = it.key();
            if (kRejected.contains(key)) {
                fail("unsupported_schema", pointer + "/" + escape_pointer(key),
                     "keyword '" + key + "' cannot be enforced by constrained decoding");
            }
            if (key == "additionalItems") {
                // Without array-form "items" the keyword has no effect (draft-07 section 9.3.1.2).
                if (node.contains("items") && node.at("items").is_array()) {
                    fail("unsupported_schema", pointer + "/additionalItems",
                         "keyword 'additionalItems' cannot be enforced by constrained decoding");
                }
                continue;
            }
            if (kEnforced.contains(key) || kContainers.contains(key)) { kept[key] = it.value(); }
        }
        upgrade_draft4_exclusive_bounds(kept, pointer);
        const TypeSet types = parse_type(kept, pointer);
        if (kept.contains("format") && !spec_defined_format(kept.at("format"))) {
            const std::string format = kept.at("format").get<std::string>();
            kept.erase("format");
            if (types.declared && types.names.size() == 1 && types.admits("integer")) {
                apply_integer_format_bounds(kept, format);
            }
        }
        if (kept.contains("$ref") && rebased_ > 0) {
            fail("unsupported_schema", pointer + "/$ref",
                 "$ref inside a subschema that declares its own base URI ('$id'/'id') cannot be "
                 "resolved against this document");
        }
        return walk_kept(kept, pointer, depth, types);
    }

private:
    struct RebaseScope {
        RebaseScope(Walker& walker, bool active) : walker_(walker), active_(active) {
            if (active_) { ++walker_.rebased_; }
        }
        ~RebaseScope() {
            if (active_) { --walker_.rebased_; }
        }
        RebaseScope(const RebaseScope&)            = delete;
        RebaseScope& operator=(const RebaseScope&) = delete;
        Walker& walker_;
        bool active_;
    };

    static bool changes_base_uri(const Json& node) {
        for (const char* key : {"$id", "id"}) {
            if (node.contains(key) && node.at(key).is_string() &&
                !node.at(key).get_ref<const std::string&>().starts_with("#")) {
                return true;
            }
        }
        return false;
    }

    // Draft-04 spells exclusive bounds as booleans modifying minimum/maximum.
    static void upgrade_draft4_exclusive_bounds(Json& kept, const std::string& pointer) {
        for (const auto& [exclusive, bound] :
             {std::pair{"exclusiveMinimum", "minimum"}, std::pair{"exclusiveMaximum", "maximum"}}) {
            if (!kept.contains(exclusive) || !kept.at(exclusive).is_boolean()) { continue; }
            const bool flag = kept.at(exclusive).get<bool>();
            kept.erase(exclusive);
            if (!flag) { continue; }
            if (!kept.contains(bound)) {
                fail("invalid_schema", pointer + "/" + exclusive,
                     std::string("boolean '") + exclusive + "' requires '" + bound + "'");
            }
            kept[exclusive] = kept.at(bound);
            kept.erase(bound);
        }
    }

    static void apply_integer_format_bounds(Json& kept, const std::string& format) {
        long long low = 0;
        long long high = 0;
        if (format == "int32") {
            low  = -2147483648LL;
            high = 2147483647LL;
        } else if (format == "int64") {
            // Narrowed to the I-JSON range; see clamp_integer_bounds.
            low  = -9007199254740991LL;
            high = 9007199254740991LL;
        } else {
            return;
        }
        if (!kept.contains("minimum") && !kept.contains("exclusiveMinimum")) {
            kept["minimum"] = low;
        }
        if (!kept.contains("maximum") && !kept.contains("exclusiveMaximum")) {
            kept["maximum"] = high;
        }
    }

    Json walk_kept(const Json& node, const std::string& pointer, std::size_t depth,
                   const TypeSet& types) {
        Json out = node;
        for (const char* container : {"$defs", "definitions"}) {
            if (!node.contains(container)) { continue; }
            const Json& defs = node.at(container);
            const std::string base = pointer + "/" + container;
            if (!defs.is_object()) { fail("invalid_schema", base, "must be an object"); }
            for (auto it = defs.begin(); it != defs.end(); ++it) {
                out[container][it.key()] =
                    walk(it.value(), base + "/" + escape_pointer(it.key()), depth + 1);
            }
        }

        std::vector<std::string> composites;
        for (const char* key : {"$ref", "const", "enum", "anyOf", "oneOf", "allOf"}) {
            if (node.contains(key)) { composites.emplace_back(key); }
        }
        if (!composites.empty()) {
            walk_composite(node, out, pointer, depth, composites, types);
            return out;
        }

        check_numeric(node, pointer, types);
        clamp_integer_bounds(out, pointer, types);
        bound_open_integer_sides(out, types);
        check_string(node, pointer, types);
        walk_array(node, out, pointer, depth, types);
        walk_object(node, out, pointer, depth, types);
        return out;
    }

    static TypeSet parse_type(const Json& node, const std::string& pointer) {
        TypeSet types;
        if (!node.contains("type")) { return types; }
        const Json& type = node.at("type");
        const auto add   = [&](const Json& name) {
            if (!name.is_string() ||
                std::find(kTypes.begin(), kTypes.end(), name.get<std::string>()) == kTypes.end()) {
                fail("invalid_schema", pointer + "/type", "unknown type " + name.dump());
            }
            types.names.insert(name.get<std::string>());
        };
        if (type.is_array()) {
            if (type.empty()) { fail("invalid_schema", pointer + "/type", "type array is empty"); }
            for (const Json& name : type) { add(name); }
        } else {
            add(type);
        }
        types.declared = true;
        return types;
    }

    void walk_composite(const Json& node, Json& out, const std::string& pointer, std::size_t depth,
                        const std::vector<std::string>& composites, const TypeSet& types) {
        if (composites.size() > 1) {
            fail("unsupported_schema", pointer,
                 "'" + composites[0] + "' and '" + composites[1] +
                     "' in one schema object cannot be enforced together; the grammar compiler "
                     "honours only one of them");
        }
        const std::string& composite = composites.front();
        const bool branching         = composite == "anyOf" || composite == "oneOf";
        for (auto it = node.begin(); it != node.end(); ++it) {
            const std::string& key = it.key();
            if (key == composite || kContainers.contains(key)) { continue; }
            // "type" beside enum/const filters the values; beside anyOf/oneOf it is distributed
            // into every branch; beside $ref it is dropped when the target already implies it.
            if (key == "type" && composite != "allOf") { continue; }
            fail("unsupported_schema", pointer + "/" + escape_pointer(key),
                 "keyword '" + key + "' next to '" + composite +
                     "' is not enforced by the grammar compiler; move it inside the '" +
                     composite + "' branches");
        }

        const std::string base = pointer + "/" + composite;
        if (composite == "$ref") {
            const Json& ref = node.at("$ref");
            if (!ref.is_string()) { fail("invalid_schema", base, "$ref must be a string"); }
            const std::string& uri = ref.get_ref<const std::string&>();
            if (uri != "#" && !uri.starts_with("#/")) {
                fail("unsupported_schema", base,
                     "only JSON-pointer references into this document ('#' or '#/...') are "
                     "supported, got '" +
                         uri + "'");
            }
            if (types.declared) {
                if (!reference_implies_type(uri, types)) {
                    fail("unsupported_schema", pointer + "/type",
                         "keyword 'type' next to '$ref' is enforced only when the referenced "
                         "schema declares a type within it");
                }
                out.erase("type");
            }
            refs_.emplace_back(base, uri);
            return;
        }
        if (composite == "const" || composite == "enum") {
            if (composite == "enum" && (!node.at("enum").is_array() || node.at("enum").empty())) {
                fail("invalid_schema", base, "enum must be a non-empty array");
            }
            if (!types.declared) { return; }
            const auto matches = [&](const Json& value) {
                return std::any_of(
                    types.names.begin(), types.names.end(),
                    [&](const std::string& name) { return value_matches_type(value, name); });
            };
            if (composite == "const") {
                if (!matches(node.at("const"))) {
                    fail("invalid_schema", base,
                         "value " + node.at("const").dump() + " does not match the declared type");
                }
                return;
            }
            // An enum value outside the declared type can never validate; only the rest remain.
            Json admitted = Json::array();
            for (const Json& value : node.at("enum")) {
                if (matches(value)) { admitted.push_back(value); }
            }
            if (admitted.empty()) {
                fail("invalid_schema", base, "no enum value matches the declared type");
            }
            out["enum"] = std::move(admitted);
            return;
        }
        const Json& branches = node.at(composite);
        if (!branches.is_array() || branches.empty()) {
            fail("invalid_schema", base, composite + " must be a non-empty array");
        }
        if (composite == "allOf" && branches.size() > 1) {
            fail("unsupported_schema", base,
                 "allOf with more than one schema cannot be enforced by constrained decoding");
        }
        Json walked = Json::array();
        for (std::size_t index = 0; index < branches.size(); ++index) {
            const std::string branch_pointer = base + "/" + std::to_string(index);
            if (branching && types.declared) {
                std::optional<Json> typed = distribute_type(branches[index], types, branch_pointer);
                if (!typed.has_value()) { continue; } // branch outside the declared type
                walked.push_back(walk(*typed, branch_pointer, depth + 1));
            } else {
                walked.push_back(walk(branches[index], branch_pointer, depth + 1));
            }
        }
        if (walked.empty()) {
            fail("invalid_schema", base, "no branch admits the declared type");
        }
        out[composite] = std::move(walked);
        if (branching) { out.erase("type"); }
    }

    static std::unordered_set<std::string> intersect_types(const std::unordered_set<std::string>& a,
                                                           const std::unordered_set<std::string>& b) {
        std::unordered_set<std::string> both;
        for (const std::string& name : a) {
            if (b.contains(name)) {
                both.insert(name);
            } else if (name == "integer" && b.contains("number")) {
                both.insert("integer");
            } else if (name == "number" && b.contains("integer")) {
                both.insert("integer");
            }
        }
        return both;
    }

    static Json type_json(const std::unordered_set<std::string>& names) {
        std::vector<std::string> ordered;
        for (const std::string_view name : kTypes) {
            if (names.contains(std::string(name))) { ordered.emplace_back(name); }
        }
        if (ordered.size() == 1) { return ordered.front(); }
        return ordered;
    }

    // {type: T, anyOf: [B...]} is anyOf over (T and B). Returns the branch with T folded in, or
    // nothing when the branch cannot hold a value of T.
    static std::optional<Json> distribute_type(const Json& branch, const TypeSet& types,
                                               const std::string& pointer) {
        if (branch.is_boolean()) {
            if (!branch.get<bool>()) { return std::nullopt; }
            Json typed    = Json::object();
            typed["type"] = type_json(types.names);
            return typed;
        }
        if (!branch.is_object()) { fail("invalid_schema", pointer, "a schema must be an object"); }
        for (const char* nested : {"$ref", "anyOf", "oneOf", "allOf"}) {
            if (branch.contains(nested)) {
                fail("unsupported_schema", pointer,
                     std::string("a 'type' next to a composite cannot be combined with a branch "
                                 "using '") +
                         nested + "'");
            }
        }
        Json typed = branch;
        if (!branch.contains("type")) {
            typed["type"] = type_json(types.names);
            return typed;
        }
        const TypeSet own                        = parse_type(branch, pointer);
        const std::unordered_set<std::string> both = intersect_types(own.names, types.names);
        if (both.empty()) { return std::nullopt; }
        typed["type"] = type_json(both);
        return typed;
    }

    bool reference_implies_type(const std::string& uri, const TypeSet& types) const {
        if (root_ == nullptr || uri == "#") { return false; }
        try {
            const Json::json_pointer target_pointer(uri.substr(1));
            if (!root_->contains(target_pointer)) { return false; }
            const Json& target = root_->at(target_pointer);
            if (!target.is_object() || !target.contains("type")) { return false; }
            const TypeSet own = parse_type(target, uri);
            for (const std::string& name : own.names) {
                if (!types.names.contains(name) &&
                    !(name == "integer" && types.names.contains("number"))) {
                    return false;
                }
            }
            return true;
        } catch (const nlohmann::json::exception&) { return false; }
    }

    // A type-specific keyword in a schema without "type" is dropped by the grammar compiler (it
    // falls back to any value) unless the schema's shape implies the type.
    static void require_type(const Json& node, const std::string& pointer, const TypeSet& types,
                             const char* keyword, bool implied_by_shape = false) {
        if (!node.contains(keyword)) { return; }
        if (!types.declared) {
            if (implied_by_shape) { return; }
            fail("unsupported_schema", pointer + "/" + keyword,
                 std::string("'") + keyword +
                     "' without a declared 'type' is not enforced by the grammar compiler; add "
                     "the type");
        }
    }

    // The grammar compiler reads integer bounds through a double and refuses values beyond 2^63.
    // Integer bounds past the I-JSON interoperable range (RFC 7493, +-(2^53-1)) are therefore
    // tightened to it, which only narrows the admitted values; a bound that would have to widen is
    // rejected.
    static void clamp_integer_bounds(Json& out, const std::string& pointer, const TypeSet& types) {
        if (!types.declared || !types.names.contains("integer") || types.names.contains("number")) {
            return;
        }
        constexpr double kSafe = 9007199254740991.0;
        for (const char* keyword : {"minimum", "exclusiveMinimum"}) {
            if (!out.contains(keyword)) { continue; }
            const double value = out.at(keyword).get<double>();
            if (value > kSafe) {
                fail("unsupported_schema", pointer + "/" + keyword,
                     "integer bounds beyond +-(2^53-1) are not supported");
            }
            if (value < -kSafe) { out[keyword] = -9007199254740991LL; }
        }
        for (const char* keyword : {"maximum", "exclusiveMaximum"}) {
            if (!out.contains(keyword)) { continue; }
            const double value = out.at(keyword).get<double>();
            if (value < -kSafe) {
                fail("unsupported_schema", pointer + "/" + keyword,
                     "integer bounds beyond +-(2^53-1) are not supported");
            }
            if (value > kSafe) { out[keyword] = 9007199254740991LL; }
        }
    }

    // An integer side without a bound is closed at the I-JSON limit, +-(2^53-1): the grammar
    // compiler emits "-"? [1-9] [0-9]* for an open side, so a model that keeps writing digits is
    // held only by max_tokens (outlines#1785; llama.cpp caps the digits for the same reason,
    // ggml-org/llama.cpp#6555). Every value an interoperable JSON consumer can read exactly stays
    // admitted, the grammar ends after at most 16 digits, and a declared bound is kept. A
    // multipleOf schema keeps its open sides: the compiler enumerates the multiples of a closed
    // range, which this bound would make enormous.
    static void bound_open_integer_sides(Json& out, const TypeSet& types) {
        if (!types.declared || !types.names.contains("integer") ||
            types.names.contains("number") || out.contains("multipleOf")) {
            return;
        }
        constexpr long long kSafe = 9007199254740991LL;
        if (!out.contains("minimum") && !out.contains("exclusiveMinimum")) {
            out["minimum"] = -kSafe;
        }
        if (!out.contains("maximum") && !out.contains("exclusiveMaximum")) {
            out["maximum"] = kSafe;
        }
    }

    static void check_numeric(const Json& node, const std::string& pointer, const TypeSet& types) {
        for (const char* keyword :
             {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}) {
            require_type(node, pointer, types, keyword);
            if (node.contains(keyword) && !node.at(keyword).is_number()) {
                fail("invalid_schema", pointer + "/" + keyword,
                     std::string("'") + keyword + "' must be a number");
            }
        }
        if (node.contains("multipleOf") && types.admits("number")) {
            fail("unsupported_schema", pointer + "/multipleOf",
                 "multipleOf is enforced only for type 'integer'");
        }
    }

    static void check_string(const Json& node, const std::string& pointer, const TypeSet& types) {
        int string_forms = 0;
        for (const char* keyword : {"minLength", "maxLength", "pattern", "format"}) {
            require_type(node, pointer, types, keyword);
        }
        if (node.contains("format")) {
            ++string_forms;
            const Json& format = node.at("format");
            if (!format.is_string() || !kFormats.contains(format.get<std::string>())) {
                fail("unsupported_schema", pointer + "/format",
                     "format " + format.dump() +
                         " is not enforced; supported formats: email, date, time, date-time, "
                         "duration, hostname, uuid, uri, uri-reference, uri-template, "
                         "json-pointer, relative-json-pointer");
            }
        }
        if (node.contains("pattern")) {
            ++string_forms;
            if (!node.at("pattern").is_string()) {
                fail("invalid_schema", pointer + "/pattern", "pattern must be a string");
            }
        }
        if (node.contains("minLength") || node.contains("maxLength")) { ++string_forms; }
        if (string_forms > 1 && types.admits("string")) {
            fail("unsupported_schema", pointer,
                 "only one of 'format', 'pattern' and 'minLength'/'maxLength' can be enforced on "
                 "one string schema");
        }
    }

    void walk_array(const Json& node, Json& out, const std::string& pointer, std::size_t depth,
                    const TypeSet& types) {
        const bool shaped = node.contains("items") || node.contains("prefixItems") ||
                            node.contains("unevaluatedItems");
        for (const char* keyword : {"minItems", "maxItems", "uniqueItems"}) {
            require_type(node, pointer, types, keyword, shaped);
        }
        if (node.contains("uniqueItems")) {
            const Json& unique = node.at("uniqueItems");
            if (!unique.is_boolean()) {
                fail("invalid_schema", pointer + "/uniqueItems", "uniqueItems must be a boolean");
            }
            if (unique.get<bool>()) {
                fail("unsupported_schema", pointer + "/uniqueItems",
                     "uniqueItems=true cannot be enforced by constrained decoding");
            }
            out.erase("uniqueItems");
        }
        if (node.contains("items")) {
            const Json& items = node.at("items");
            if (items.is_array()) {
                fail("unsupported_schema", pointer + "/items",
                     "array-form items (draft-07 tuples) is not supported; use prefixItems");
            }
            out["items"] = walk(items, pointer + "/items", depth + 1, true);
        }
        if (node.contains("unevaluatedItems")) {
            out["unevaluatedItems"] =
                walk(node.at("unevaluatedItems"), pointer + "/unevaluatedItems", depth + 1, true);
        }
        if (node.contains("prefixItems")) {
            const Json& prefix = node.at("prefixItems");
            if (!prefix.is_array()) {
                fail("invalid_schema", pointer + "/prefixItems", "prefixItems must be an array");
            }
            for (std::size_t index = 0; index < prefix.size(); ++index) {
                out["prefixItems"][index] = walk(
                    prefix[index], pointer + "/prefixItems/" + std::to_string(index), depth + 1);
            }
        }
        if (types.admits("array") && types.declared && !shaped) { out["items"] = true; }
    }

    void walk_object(const Json& node, Json& out, const std::string& pointer, std::size_t depth,
                     const TypeSet& types) {
        const bool shaped = node.contains("properties") ||
                            node.contains("additionalProperties") ||
                            node.contains("unevaluatedProperties");
        for (const char* keyword :
             {"required", "patternProperties", "propertyNames", "minProperties", "maxProperties"}) {
            require_type(node, pointer, types, keyword, shaped);
        }
        if (node.contains("properties")) {
            const Json& properties = node.at("properties");
            if (!properties.is_object()) {
                fail("invalid_schema", pointer + "/properties", "properties must be an object");
            }
            for (auto it = properties.begin(); it != properties.end(); ++it) {
                out["properties"][it.key()] = walk(
                    it.value(), pointer + "/properties/" + escape_pointer(it.key()), depth + 1);
            }
        }
        if (node.contains("required")) {
            const Json& required = node.at("required");
            if (!required.is_array()) {
                fail("invalid_schema", pointer + "/required", "required must be an array");
            }
            for (const Json& name : required) {
                if (!name.is_string()) {
                    fail("invalid_schema", pointer + "/required",
                         "required entries must be strings");
                }
                if (!node.contains("properties") ||
                    !node.at("properties").contains(name.get<std::string>())) {
                    fail("unsupported_schema", pointer + "/required",
                         "required property '" + name.get<std::string>() +
                             "' is not declared in properties");
                }
            }
        }
        for (const char* keyword : {"additionalProperties", "unevaluatedProperties"}) {
            if (node.contains(keyword)) {
                out[keyword] = walk(node.at(keyword), pointer + "/" + keyword, depth + 1, true);
            }
        }
        if (node.contains("propertyNames")) {
            out["propertyNames"] =
                walk(node.at("propertyNames"), pointer + "/propertyNames", depth + 1);
        }
        if (node.contains("patternProperties")) {
            const Json& patterns = node.at("patternProperties");
            if (!patterns.is_object()) {
                fail("invalid_schema", pointer + "/patternProperties",
                     "patternProperties must be an object");
            }
            for (auto it = patterns.begin(); it != patterns.end(); ++it) {
                out["patternProperties"][it.key()] =
                    walk(it.value(),
                         pointer + "/patternProperties/" + escape_pointer(it.key()), depth + 1);
            }
        }
        const bool declares_space = shaped || node.contains("patternProperties");
        if (types.admits("object") && types.declared && !declares_space) {
            out["additionalProperties"] = true;
        }
    }

public:
    // Every reference must land on a subschema this walker checked and kept; a target under a
    // dropped keyword (for example OpenAPI "components") was never validated.
    void check_references(const Json& normalized) const {
        for (const auto& [pointer, uri] : refs_) {
            if (uri == "#") { continue; }
            bool found = false;
            try {
                found = normalized.contains(Json::json_pointer(uri.substr(1)));
            } catch (const nlohmann::json::exception&) {
                found = false;
            }
            if (!found) {
                fail("unsupported_schema", pointer,
                     "reference '" + uri +
                         "' does not point at a subschema of a keyword NInferno enforces");
            }
        }
    }

    void set_root(const Json* root) { root_ = root; }

private:
    const Json* root_    = nullptr;
    std::size_t nodes_   = 0;
    std::size_t rebased_ = 0;
    std::vector<std::pair<std::string, std::string>> refs_;
};

} // namespace

nlohmann::ordered_json normalize_supported_json_schema(const nlohmann::ordered_json& schema) {
    Walker walker;
    walker.set_root(&schema);
    Json normalized = walker.walk(schema, "", 0);
    walker.check_references(normalized);
    return normalized;
}

nlohmann::ordered_json normalize_tool_parameters_schema(const nlohmann::ordered_json& schema) {
    nlohmann::ordered_json rooted = schema;
    if (const auto type = rooted.find("type"); type == rooted.end()) {
        constexpr std::array<std::string_view, 6> kShaping = {"$ref",  "const", "enum",
                                                              "anyOf", "oneOf", "allOf"};
        if (std::none_of(kShaping.begin(), kShaping.end(),
                         [&](std::string_view key) { return rooted.contains(key); })) {
            rooted["type"] = "object";
        }
    } else if (!(type->is_string() && *type == "object") &&
               !(type->is_array() && type->size() == 1 && type->front() == "object")) {
        throw OutputConstraintError("invalid_schema", "must describe a JSON object (type object)");
    }
    return normalize_supported_json_schema(rooted);
}

} // namespace ninferno::structured
