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

// Whether the calling thread has built a document that left a lot of freed memory in its malloc
// arena (one of 10,000 values or more, built from text of 1 MiB or more, or stopped at a limit
// after building up to it) since the last call; the call clears it. See release_request_memory.
[[nodiscard]] bool take_large_parse_seen() noexcept;

// The same bounds, checked without building the document: nothing is allocated, so any text up to
// the body limit can be measured. Used for text that is kept as it is and parsed later (a replayed
// tool call's arguments, which the chat template parses): Invalid is not an error here, since the
// renderer shows text that is not JSON as it is.
// `values_seen`, when given, receives how many values were counted.
[[nodiscard]] JsonTextStatus check_bounded_json(std::string_view text, std::size_t max_values,
                                                std::size_t* values_seen = nullptr);

// Records that the calling thread is about to build, or has built, a document of `values` values,
// for take_large_parse_seen (the chat template parses replayed tool-call arguments on its own).
void note_parse_size(std::size_t values) noexcept;

} // namespace ninferno::serve
