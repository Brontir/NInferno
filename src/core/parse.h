#pragma once

// A whole string as a number: the text holds the number and nothing else.

#include <charconv>
#include <optional>
#include <string_view>
#include <system_error>

namespace ninferno {

// `text` as a T (std::from_chars); nullopt when it is not a number of that type, overflows it, or
// has anything after the number.
template <class T>
[[nodiscard]] std::optional<T> parse_exact(std::string_view text) noexcept {
    T value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) { return std::nullopt; }
    return value;
}

} // namespace ninferno
