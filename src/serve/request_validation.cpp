#include "serve/request_validation.h"

#include "chat/tool_calls.h"

#include <algorithm>
#include <cctype>
#include <iterator>
#include <string_view>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace ninferno::serve {

[[noreturn]] void bad_request(std::string message, std::string param, std::string code) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

std::optional<int> optional_int(const RequestJson& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    const RequestJson& value = object.at(key);
    if (!value.is_number_integer()) { bad_request(std::string(key) + " must be an integer", key); }
    if (value.is_number_unsigned()) {
        const std::uint64_t converted = value.get<std::uint64_t>();
        if (converted > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            bad_request(std::string(key) + " is out of range", key);
        }
        return static_cast<int>(converted);
    }
    const std::int64_t converted = value.get<std::int64_t>();
    if (converted < std::numeric_limits<int>::min() ||
        converted > std::numeric_limits<int>::max()) {
        bad_request(std::string(key) + " is out of range", key);
    }
    return static_cast<int>(converted);
}

std::optional<double> optional_number(const RequestJson& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    if (!object.at(key).is_number()) { bad_request(std::string(key) + " must be a number", key); }
    const double value = object.at(key).get<double>();
    if (!std::isfinite(value)) { bad_request(std::string(key) + " must be finite", key); }
    return value;
}

bool optional_bool(const RequestJson& object, const char* key, bool fallback) {
    if (!object.contains(key) || object.at(key).is_null()) { return fallback; }
    if (!object.at(key).is_boolean()) { bad_request(std::string(key) + " must be a boolean", key); }
    return object.at(key).get<bool>();
}

std::optional<bool> nullable_bool(const RequestJson& object, const char* key,
                                  std::string_view name) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    if (!object.at(key).is_boolean()) {
        const std::string field = name.empty() ? std::string(key) : std::string(name);
        bad_request(field + " must be a boolean or null", field);
    }
    return object.at(key).get<bool>();
}

NinfernoExtension parse_ninferno_extension(const RequestJson& body) {
    NinfernoExtension out;
    if (!body.contains("ninferno") || body.at("ninferno").is_null()) { return out; }
    const RequestJson& extension = body.at("ninferno");
    if (!extension.is_object()) { bad_request("ninferno must be an object", "ninferno"); }
    for (auto it = extension.begin(); it != extension.end(); ++it) {
        if (it.key() != "draft_tokens") {
            bad_request("unknown ninferno member: " + it.key(), "ninferno." + it.key(),
                        "unknown_parameter");
        }
    }
    if (extension.contains("draft_tokens") && !extension.at("draft_tokens").is_null()) {
        const RequestJson& value = extension.at("draft_tokens");
        const std::string range  = "[0," + std::to_string(kMaximumRequestDraftTokens) + "]";
        if (!value.is_number_integer()) {
            bad_request("ninferno.draft_tokens must be an integer in " + range,
                        "ninferno.draft_tokens", "invalid_draft_tokens");
        }
        const bool out_of_range =
            value.is_number_unsigned()
                ? value.get<std::uint64_t>() > kMaximumRequestDraftTokens
                : (value.get<std::int64_t>() < 0 ||
                   value.get<std::int64_t>() > std::int64_t{kMaximumRequestDraftTokens});
        if (out_of_range) {
            bad_request("ninferno.draft_tokens must be in " + range + ": 0 decodes the request "
                        "plainly, a depth drafts that many tokens a round",
                        "ninferno.draft_tokens", "invalid_draft_tokens");
        }
        out.draft_tokens = static_cast<std::uint32_t>(value.get<std::int64_t>());
    }
    return out;
}

bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept {
    return chat::valid_tool_name(name, maximum_length);
}

std::string tool_name_rule(std::size_t maximum_length) {
    return std::string(chat::kToolNameBytes) + "{1," + std::to_string(maximum_length) + "}";
}

std::string top_k_limit_message(std::string_view field, std::string_view verb) {
    const std::string limit = std::to_string(kMaximumTopK);
    return std::string(field) + " must be in [0," + limit + "]: the sampler keeps at most " +
           limit + " candidates, so " + std::string(verb) + " " + limit +
           " (or 0, the same) for the widest; a model card's larger top_k differs only in the far "
           "tail under top_p";
}

void check_message_count(std::size_t count, const RequestLimits& limits, const char* field) {
    if (count <= limits.max_messages) { return; }
    bad_request(std::string(field) + " has " + std::to_string(count) +
                    " entries; this server accepts at most " + std::to_string(limits.max_messages) +
                    " (--max-request-messages)",
                field, "too_many_messages");
}

ninferno::product::media_acquire::Source media_source(std::string value, std::string_view field,
                                                      std::string param, bool allow_data) {
    using ninferno::product::media_acquire::SourceKind;
    ninferno::product::media_acquire::Source source;
    if (allow_data && value.starts_with("data:")) {
        source.kind = SourceKind::Data;
    } else if (value.starts_with("http://") || value.starts_with("https://")) {
        source.kind = SourceKind::Url;
    } else {
        bad_request(std::string(field) +
                        (allow_data ? " must use HTTP(S) or a data URI" : " must use HTTP(S)"),
                    std::move(param));
    }
    source.value = std::move(value);
    return source;
}

ninferno::product::media_acquire::Source
inline_file_source(std::string data, std::string_view field, std::string param) {
    if (data.empty()) { bad_request(std::string(field) + " must not be empty", std::move(param)); }
    ninferno::product::media_acquire::Source source;
    source.kind = ninferno::product::media_acquire::SourceKind::Data;
    source.value =
        data.starts_with("data:") ? std::move(data) : "data:application/pdf;base64," + data;
    return source;
}

void check_image_detail(const RequestJson& object, std::string_view field, std::string param) {
    if (!object.contains("detail") || object.at("detail").is_null()) { return; }
    if (!object.at("detail").is_string()) {
        bad_request(std::string(field) + ".detail must be a string", std::move(param));
    }
    const std::string detail = object.at("detail").get<std::string>();
    if (detail != "auto" && detail != "low" && detail != "high") {
        bad_request(std::string(field) + ".detail='" + ascii_preview(detail) +
                        "' is not an OpenAI image detail level; use 'auto', 'low' or 'high'",
                    std::move(param), "image_detail_not_supported");
    }
}

void append_stop_strings(const RequestJson& value, std::string param, bool single,
                         std::vector<std::string>& out) {
    const auto append = [&](const RequestJson& item) {
        if (!item.is_string()) { bad_request(param + " entries must be strings", param); }
        if (!item.get_ref<const std::string&>().empty()) { out.push_back(item.get<std::string>()); }
    };
    if (single && value.is_string()) { return append(value); }
    if (!value.is_array()) {
        bad_request(param + (single ? " must be a string or array of strings"
                                    : " must be an array of strings"),
                    param);
    }
    if (value.size() > kMaximumStopStrings) {
        bad_request(param + " supports at most " + std::to_string(kMaximumStopStrings) + " strings",
                    param);
    }
    for (const RequestJson& item : value) { append(item); }
}

std::optional<std::string> tool_name_field(const RequestJson& object, const char* key,
                                           std::string_view what, std::string param,
                                           bool nullable) {
    const bool present = object.contains(key) && !object.at(key).is_null();
    if (nullable && !present) { return std::nullopt; }
    if (!present || !object.at(key).is_string()) {
        const std::string got = object.contains(key) ? request_json_type_name(object.at(key))
                                                     : "no " + std::string(key) + " field";
        bad_request(std::string(what) +
                        (nullable ? " must be a string or null, got " : " must be a string, got ") +
                        got,
                    std::move(param));
    }
    std::string name = object.at(key).get<std::string>();
    if (!valid_tool_name(name, kMaximumToolNameLength)) {
        bad_request(std::string(what) + " '" + ascii_preview(name) + "' (" +
                        std::to_string(name.size()) + " bytes) must match " +
                        tool_name_rule(kMaximumToolNameLength),
                    std::move(param), "invalid_tool_name");
    }
    return name;
}

std::string require_tool_name(const RequestJson& object, std::string param, std::string_view what) {
    return *tool_name_field(object, "name", what, std::move(param), false);
}

const char* request_json_type_name(const RequestJson& value) noexcept {
    switch (value.type()) {
    case RequestJson::value_t::null:
        return "null";
    case RequestJson::value_t::object:
        return "object";
    case RequestJson::value_t::array:
        return "array";
    case RequestJson::value_t::string:
        return "string";
    case RequestJson::value_t::boolean:
        return "boolean";
    case RequestJson::value_t::number_integer:
    case RequestJson::value_t::number_unsigned:
    case RequestJson::value_t::number_float:
        return "number";
    case RequestJson::value_t::binary:
        return "binary";
    case RequestJson::value_t::discarded:
        return "discarded";
    }
    return "value";
}

std::string ascii_preview(std::string_view value, std::size_t limit) {
    static constexpr char kHexDigits[] = "0123456789abcdef";
    std::string out;
    out.reserve(value.size() < limit ? value.size() : limit);
    std::size_t shown = 0;
    for (const unsigned char character : value) {
        if (shown == limit) {
            out += "...";
            break;
        }
        if (character >= 0x20 && character <= 0x7e) {
            out.push_back(static_cast<char>(character));
        } else {
            out += "\\x";
            out.push_back(kHexDigits[character >> 4]);
            out.push_back(kHexDigits[character & 0x0f]);
        }
        ++shown;
    }
    return out;
}

void note_unknown_fields(const RequestJson& body, std::span<const std::string_view> known,
                         GenerationRequest& output) {
    constexpr std::size_t kListed = 16;
    std::string names;
    std::size_t unknown = 0;
    for (auto it = body.begin(); it != body.end(); ++it) {
        if (std::find(known.begin(), known.end(), std::string_view(it.key())) != known.end()) {
            continue;
        }
        if (unknown < kListed) { names += (unknown == 0 ? "" : ", ") + ascii_preview(it.key()); }
        ++unknown;
    }
    if (unknown == 0) { return; }
    if (unknown > kListed) { names += " and " + std::to_string(unknown - kListed) + " more"; }
    output.notes.push_back("unknown fields ignored: " + names);
}

std::string notes_header_value(const GenerationRequest& request) {
    std::string value;
    for (const std::string& note : request.notes) {
        if (!value.empty()) { value += "; "; }
        value += ascii_preview(note, 256);
    }
    return value;
}

} // namespace ninferno::serve
