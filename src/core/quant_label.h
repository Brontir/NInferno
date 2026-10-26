#pragma once

#include "core/ascii.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace ninferno {

// The bits per weight a quantization label names: "UD-Q4_K_XL" 4, "IQ2_XXS" 2, the ternary
// "TQ1_0" 1, "MXFP4_MOE" 4, "BF16" 16, "F32" 32, "Q8_0" 8; nullopt when it names none. The one
// reading of a label's width: the fetch's width tags (--quant 4bit), a repository's listing and the
// lab pairs' bit classes all take it.
[[nodiscard]] inline std::optional<int> quant_label_bits(std::string_view label) {
    const std::string text = ascii_lower(label);
    if (text.find("f16") != std::string::npos) { return 16; } // F16 and BF16
    if (text.find("f32") != std::string::npos) { return 32; }
    if (text.find("fp4") != std::string::npos) { return 4; }
    // A "q<digit>" word, its "i" (IQ2_M) or "t" (TQ1_0) kept: at the start or after a '-' or '_'.
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (i > 0 && text[i - 1] != '-' && text[i - 1] != '_') { continue; }
        const std::size_t q = i + (text[i] == 'i' || text[i] == 't' ? 1 : 0);
        if (q + 1 < text.size() && text[q] == 'q' && text[q + 1] >= '0' && text[q + 1] <= '9') {
            return text[q + 1] - '0';
        }
    }
    return std::nullopt;
}

} // namespace ninferno
