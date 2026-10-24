#pragma once

#include "serve/request.h"
#include "serve/request_json.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::serve {

// Deeper request JSON is refused before anything walks it: tool schemas, response formats and
// history are later converted and rendered recursively, and a tool schema nested 5,000 levels
// deep overflowed the stack. It bounds the body, and the JSON a field carries as text (a schema,
// a structural tag) as well.
inline constexpr std::size_t kMaximumRequestJsonDepth = 128;

// A per-thread engine for identifiers and seeds, seeded from 256 bits of std::random_device:
// one 32-bit seed per thread gave two threads the same sequence (duplicate response ids) about
// once in four billion starts per thread pair, and a guessable one.
inline std::mt19937_64& unpredictable_engine() {
    thread_local std::mt19937_64 engine = [] {
        std::random_device device;
        std::seed_seq seed{device(), device(), device(), device(),
                           device(), device(), device(), device()};
        return std::mt19937_64(seed);
    }();
    return engine;
}

// `prefix` and `words` random 64-bit words in hex (16 digits each): response, message and tool
// call identifiers.
inline std::string random_identifier(std::string_view prefix, int words = 1) {
    std::string out(prefix);
    for (int i = 0; i < words; ++i) {
        std::array<char, 17> word{};
        std::snprintf(word.data(), word.size(), "%016llx",
                      static_cast<unsigned long long>(unpredictable_engine()()));
        out += word.data();
    }
    return out;
}

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {});

std::optional<int> optional_int(const RequestJson& object, const char* key);
std::optional<double> optional_number(const RequestJson& object, const char* key);
bool optional_bool(const RequestJson& object, const char* key, bool fallback);
// A boolean that may be absent or null (nullopt), as `name` (default: the key) in a refusal.
std::optional<bool> nullable_bool(const RequestJson& object, const char* key,
                                  std::string_view name = {});

[[nodiscard]] bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept;
// The rule valid_tool_name checks, as error messages state it.
[[nodiscard]] std::string tool_name_rule(std::size_t maximum_length);
// Where a media part's bytes come from: a data URI or an HTTP(S) URL (with allow_data false, a
// URL only), refused under `param` as "<field> must use HTTP(S)[ or a data URI]".
[[nodiscard]] ninferno::product::media_acquire::Source
media_source(std::string value, std::string_view field, std::string param, bool allow_data = true);
// An inline file's bytes (file_data): a data URI, or bare base64 of a PDF.
[[nodiscard]] ninferno::product::media_acquire::Source
inline_file_source(std::string data, std::string_view field, std::string param);
// An image's detail level in object["detail"], when given: 'auto', 'low' or 'high', which the one
// Vision preprocessing profile all serves ('low' and 'high' are cost hints SDKs send by default).
void check_image_detail(const RequestJson& object, std::string_view field, std::string param);
// The most stop strings a request may name: OpenAI documents four; llama.cpp-style clients
// (Continue, SillyTavern) send more.
inline constexpr std::size_t kMaximumStopStrings = 16;
// The stop strings in `value` (an array of strings, or with `single` one string) refused under
// `param`, appended to `out`: at most kMaximumStopStrings, an empty one dropped (it matches
// nothing, and clients send "" for none).
void append_stop_strings(const RequestJson& value, std::string param, bool single,
                         std::vector<std::string>& out);
// The tool name in object[key]: a string valid_tool_name accepts (kMaximumToolNameLength bytes at
// most), refused as `what` ("function name", "tool name", "namespace name") under `param`, with
// code invalid_tool_name when it breaks the rule. With `nullable`, an absent or null field is
// nullopt; without it, the field is required.
[[nodiscard]] std::optional<std::string> tool_name_field(const RequestJson& object, const char* key,
                                                         std::string_view what, std::string param,
                                                         bool nullable);
[[nodiscard]] std::string require_tool_name(const RequestJson& object, std::string param,
                                            std::string_view what = "function name");

// The refusal of a top_k above kMaximumTopK, naming the field as its caller spells it ("top_k",
// "top-k") and telling it what to send instead ("send", "use").
[[nodiscard]] std::string top_k_limit_message(std::string_view field, std::string_view verb);

// A 400 (code too_many_messages) when a request's `field` list holds more than
// limits.max_messages entries.
void check_message_count(std::size_t count, const RequestLimits& limits, const char* field);

// The server extension object shared by every dialect: {"ninferno": {...}}. Absent or null is
// accepted; anything else that is not a known member with a known value is a 400. Its members:
//   draft_tokens  this request's speculative draft depth, 0..kMaximumRequestDraftTokens: 0 decodes
//                 it plainly; the server refuses a depth past the window it captured at startup.
inline constexpr std::uint32_t kMaximumRequestDraftTokens = 15;
struct NinfernoExtension {
    std::optional<std::uint32_t> draft_tokens;
};
[[nodiscard]] NinfernoExtension parse_ninferno_extension(const RequestJson& body);

// Top-level fields of `body` outside `known` (what the dialect reads, plus fields it knows to be
// neutral): one note lists them, so a misspelt option (max_token) is not silently without effect.
// The names are ascii_preview'd, at most 16 of them.
void note_unknown_fields(const RequestJson& body, std::span<const std::string_view> known,
                         GenerationRequest& output);

// The response warning carrying a request's notes: printable ASCII only (a header value cannot
// hold a line break), empty when there are none.
[[nodiscard]] std::string notes_header_value(const GenerationRequest& request);

// Name of a JSON value's type for error messages ("string", "array", ...).
[[nodiscard]] const char* request_json_type_name(const RequestJson& value) noexcept;

// ASCII-safe, bounded preview of a client-supplied value for error messages. Every byte outside
// printable ASCII is escaped as \xNN, so the result can be embedded in the JSON error body and in
// logs without control characters or invalid UTF-8.
[[nodiscard]] std::string ascii_preview(std::string_view value, std::size_t limit = 64);

} // namespace ninferno::serve
