#pragma once

// Option values shared by the products and tools (numbers, KV choices, lists): every refusal names
// the option as "invalid <label>: <text>".

#include "ninferno/types.h"

#include "core/units.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::product {

[[noreturn]] inline void invalid_option(std::string_view label, const char* text) {
    throw std::invalid_argument("invalid " + std::string(label) + ": " +
                                (text == nullptr ? "" : text));
}

// A decimal count: digits only (no sign, no space), within 64 bits.
[[nodiscard]] inline std::uint64_t parse_u64(const char* text, std::string_view label) {
    if (text == nullptr || *text < '0' || *text > '9') { invalid_option(label, text); }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || *end != '\0') { invalid_option(label, text); }
    return static_cast<std::uint64_t>(value);
}

[[nodiscard]] inline std::uint32_t parse_u32(const char* text, std::string_view label,
                                             bool allow_zero = false) {
    const std::uint64_t value = parse_u64(text, label);
    if ((!allow_zero && value == 0) || value > std::numeric_limits<std::uint32_t>::max()) {
        invalid_option(label, text);
    }
    return static_cast<std::uint32_t>(value);
}

// A count from 0 to INT_MAX.
[[nodiscard]] inline int parse_nonnegative_int(const char* text, std::string_view label) {
    const std::uint64_t value = parse_u64(text, label);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        invalid_option(label, text);
    }
    return static_cast<int>(value);
}

// A size in MiB as bytes; zero only where allow_zero.
[[nodiscard]] inline std::size_t parse_mib(const char* text, std::string_view label,
                                           bool allow_zero) {
    const std::uint64_t mib = parse_u64(text, label);
    if ((!allow_zero && mib == 0) || mib > std::numeric_limits<std::size_t>::max() / kMiB) {
        throw std::invalid_argument("--" + std::string(label) + " is out of range");
    }
    return static_cast<std::size_t>(mib << 20);
}

// A finite decimal in [minimum, maximum].
[[nodiscard]] inline float parse_float_in(const char* text, std::string_view label, float minimum,
                                          float maximum) {
    if (text == nullptr) { invalid_option(label, text); }
    errno              = 0;
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' || !std::isfinite(value) ||
        value < static_cast<double>(minimum) || value > static_cast<double>(maximum)) {
        invalid_option(label, text);
    }
    return static_cast<float>(value);
}

// The items of a list option ("a,b,c"), empty items dropped.
[[nodiscard]] inline std::vector<std::string> option_list(std::string_view text, char separator) {
    std::vector<std::string> out;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t end = std::min(text.find(separator, begin), text.size());
        if (end > begin) { out.emplace_back(text.substr(begin, end - begin)); }
        begin = end + 1;
    }
    return out;
}

[[nodiscard]] inline KvCacheStorage parse_kv_dtype(std::string_view text) {
    if (const auto storage = parse_kv_cache_storage(text)) { return *storage; }
    throw std::invalid_argument("invalid kv-dtype: " + std::string(text));
}

// --kv-capacity auto|N (N positive).
[[nodiscard]] inline KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (text != nullptr && std::string_view(text) == "auto") {
        return KvCapacityPolicy::automatic();
    }
    return KvCapacityPolicy::explicit_capacity(parse_u32(text, "kv-capacity"));
}

} // namespace ninferno::product
