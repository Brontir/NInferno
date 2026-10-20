#pragma once

// Strict UTF-8 validation: no overlong forms, no surrogates, nothing above U+10FFFF.

#include <cstddef>
#include <string_view>

namespace ninferno {

[[nodiscard]] inline bool valid_utf8(std::string_view text) noexcept {
    const auto* s       = reinterpret_cast<const unsigned char*>(text.data());
    const std::size_t n = text.size();
    for (std::size_t i = 0; i < n;) {
        const unsigned char b = s[i];
        if (b < 0x80) {
            ++i;
            continue;
        }
        // The lead byte fixes the length and narrows the first continuation byte's range.
        std::size_t need = 0;
        unsigned char lo = 0x80, hi = 0xBF;
        if (b >= 0xC2 && b <= 0xDF) {
            need = 1;
        } else if (b >= 0xE0 && b <= 0xEF) {
            need = 2;
            if (b == 0xE0) { lo = 0xA0; }
            if (b == 0xED) { hi = 0x9F; }
        } else if (b >= 0xF0 && b <= 0xF4) {
            need = 3;
            if (b == 0xF0) { lo = 0x90; }
            if (b == 0xF4) { hi = 0x8F; }
        } else {
            return false;
        }
        if (i + need >= n) { return false; }
        if (s[i + 1] < lo || s[i + 1] > hi) { return false; }
        for (std::size_t k = 2; k <= need; ++k) {
            if ((s[i + k] & 0xC0) != 0x80) { return false; }
        }
        i += need + 1;
    }
    return true;
}

} // namespace ninferno
