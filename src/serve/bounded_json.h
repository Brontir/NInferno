#pragma once

#include "serve/request_json.h"

#include <cstddef>
#include <string_view>

namespace ninferno::serve {

enum class JsonTextStatus { Ok, Invalid, TooManyValues, TooDeep };

// Parses JSON text into `out` while counting what it builds: the parse stops, before the next node
// is allocated, once the text holds more than `max_values` values (scalars, containers and object
// keys) or nests deeper than kMaximumRequestJsonDepth. A document costs about 100 bytes per value
// whatever its text size (`{}` is three bytes), so this, not the text length, bounds the memory.
// Used for request bodies and for JSON a field carries as text (a schema, a structural tag), which
// the body's own count sees as one string.
[[nodiscard]] JsonTextStatus parse_bounded_json(std::string_view text, std::size_t max_values,
                                                RequestJson& out);

} // namespace ninferno::serve
