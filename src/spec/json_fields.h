#pragma once

// Lenient reads of one field of a JSON object (manifests, configs, the registry): a field that is
// missing or of another type reads as empty or zero.

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>

namespace ninferno::spec {

// The string at `key`; empty when it is missing or not a string.
[[nodiscard]] inline std::string json_text(const nlohmann::json& object, const char* key) {
    const auto it = object.find(key);
    return it == object.end() || !it->is_string() ? std::string() : it->get<std::string>();
}

// The non-negative integer at `key`; 0 when it is missing, negative or not an integer.
[[nodiscard]] inline std::uint64_t json_unsigned(const nlohmann::json& object, const char* key) {
    const auto it = object.find(key);
    if (it == object.end() || !it->is_number_integer() || it->get<std::int64_t>() < 0) { return 0; }
    return it->get<std::uint64_t>();
}

} // namespace ninferno::spec
