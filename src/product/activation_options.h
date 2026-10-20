#pragma once

// --activations exact|fast|int8, shared by ninferno and ninferno-serve (ActivationMode,
// ninferno/types.h).

#include "ninferno/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno::product {

[[nodiscard]] inline ActivationMode parse_activation_mode(std::string_view value) {
    if (value == "exact") { return ActivationMode::Exact; }
    if (value == "fast") { return ActivationMode::Fast; }
    if (value == "int8") { return ActivationMode::Int8; }
    throw std::invalid_argument("--activations must be exact, fast or int8, not " +
                                std::string(value));
}

[[nodiscard]] inline const char* activation_mode_name(ActivationMode mode) noexcept {
    switch (mode) {
    case ActivationMode::Exact:
        return "exact";
    case ActivationMode::Fast:
        return "fast";
    case ActivationMode::Int8:
        return "int8";
    }
    return "unknown";
}

} // namespace ninferno::product
