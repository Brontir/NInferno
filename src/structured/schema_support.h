#pragma once

#include <nlohmann/json.hpp>

namespace ninferno::structured {

// Validates a JSON Schema against the subset the grammar compiler enforces exactly, and returns
// the normalized schema handed to it. Throws ninferno::OutputConstraintError with code
// "invalid_schema", "unsupported_schema" or "constraint_too_large".
//
// The compiler weakens some constructs silently (it ignores keywords it does not implement, and a
// $ref, const, enum, anyOf, oneOf or allOf hides every sibling keyword). Accepting such a schema
// would return output that looks conforming but is not checked, so every such construct is
// rejected with the JSON pointer of the offending keyword.
//
// Normalization keeps generation useful under the compiler's strict object/array mode: an object
// that declares no property space (no properties, patternProperties, additionalProperties or
// unevaluatedProperties) receives "additionalProperties": true, and an array with no item schema
// (no items, prefixItems or unevaluatedItems) receives "items": true. Objects that declare
// properties admit only declared properties unless additionalProperties says otherwise, as in
// OpenAI strict mode; every accepted output is still valid against the original schema.
[[nodiscard]] nlohmann::ordered_json
normalize_supported_json_schema(const nlohmann::ordered_json& schema);

// A strict tool's parameters, normalized as above once the root is held to an object: every
// tool-call syntax carries a call's arguments as one JSON object, the shape its parser reads back.
// A root without a type and without $ref, const, enum, anyOf, oneOf or allOf receives
// "type": "object"; a root that declares another type throws "invalid_schema".
[[nodiscard]] nlohmann::ordered_json
normalize_tool_parameters_schema(const nlohmann::ordered_json& schema);

} // namespace ninferno::structured
