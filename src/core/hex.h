#pragma once

#include <cstddef>
#include <string>

namespace ninferno {

// The lower-case hexadecimal of `size` bytes, two digits a byte.
[[nodiscard]] inline std::string hex_encode(const void* data, std::size_t size) {
    static constexpr char kDigits[] = "0123456789abcdef";
    const auto* bytes               = static_cast<const unsigned char*>(data);
    std::string out(size * 2, '\0');
    for (std::size_t i = 0; i < size; ++i) {
        out[2 * i]     = kDigits[bytes[i] >> 4];
        out[2 * i + 1] = kDigits[bytes[i] & 15];
    }
    return out;
}

} // namespace ninferno
