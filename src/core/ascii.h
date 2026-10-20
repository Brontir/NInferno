#pragma once

#include <cctype>
#include <string>
#include <string_view>

namespace ninferno {

// `text` with ASCII letters lowered (bytes outside ASCII unchanged).
[[nodiscard]] inline std::string ascii_lower(std::string_view text) {
    std::string out(text);
    for (auto& c : out) { c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }
    return out;
}

// `text` with ASCII letters raised (bytes outside ASCII unchanged).
[[nodiscard]] inline std::string ascii_upper(std::string_view text) {
    std::string out(text);
    for (auto& c : out) { c = static_cast<char>(std::toupper(static_cast<unsigned char>(c))); }
    return out;
}

} // namespace ninferno
