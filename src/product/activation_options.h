#pragma once

// --activations auto|16|8|4, shared by ninferno, ninferno-serve and the desktop UI (ActivationMode,
// ninferno/types.h). The old spellings stay accepted for one release: exact = 16, fast = auto,
// int8 = 8 (activation_mode_alias names the new one).

#include "ninferno/types.h"

#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno::product {

// The --help paragraph of ninferno and ninferno-serve, each line indented by seven spaces.
inline constexpr std::string_view kActivationHelp =
    "       --activations auto|16|8|4 is the narrowest activation width a quantized linear may\n"
    "       compute with (the KV cache is a separate choice: --kv-dtype). auto (the default)\n"
    "       runs what the artifact recorded: four bits on calibrated NVFP4 weights of an\n"
    "       sm_120a build (RTX 50), eight on wide calls of the integer formats. 16 keeps BF16\n"
    "       activations everywhere under the invariant summation profile: an answer does not\n"
    "       depend on batch neighbours, prompt cuts, the prefix cache or speculation. 8 allows\n"
    "       int8 activations on every quantized linear with the route (a head or embedding the\n"
    "       artifact kept 16-bit keeps it); 4 also allows four bits on calibrated NVFP4 weights\n"
    "       of an sm_120a build. 8 and 4 may move quality. A width no matrix can take refuses to\n"
    "       start; one only some can take warns and names the rest. The startup line\n"
    "       'activations |' says what the mode resolved to. Deprecated: exact, fast and int8,\n"
    "       the old names of 16, auto and 8.\n";

// The new spelling a deprecated one stands for ("16" for "exact"); nullopt for any other text.
[[nodiscard]] constexpr std::optional<std::string_view>
activation_mode_alias(std::string_view value) noexcept {
    if (value == "exact") { return "16"; }
    if (value == "fast") { return "auto"; }
    if (value == "int8") { return "8"; }
    return std::nullopt;
}

[[nodiscard]] inline ActivationMode parse_activation_mode(std::string_view value) {
    const std::string_view name = activation_mode_alias(value).value_or(value);
    if (name == "auto") { return ActivationMode::Auto; }
    if (name == "16") { return ActivationMode::Bits16; }
    if (name == "8") { return ActivationMode::Bits8; }
    if (name == "4") { return ActivationMode::Bits4; }
    throw std::invalid_argument("--activations must be auto, 16, 8 or 4, not " +
                                std::string(value));
}

// parse_activation_mode, saying on stderr ("<tool>: --activations exact is deprecated; use
// --activations 16") when `value` is an old spelling, so a script learns before the alias goes.
[[nodiscard]] inline ActivationMode parse_activation_option(std::string_view value,
                                                            std::string_view tool) {
    const ActivationMode mode = parse_activation_mode(value);
    if (const auto alias = activation_mode_alias(value)) {
        std::cerr << tool << ": --activations " << value << " is deprecated; use --activations "
                  << *alias << '\n';
    }
    return mode;
}

// The canonical spelling: what --activations takes, the logs print and /v1/models reports.
[[nodiscard]] constexpr const char* activation_mode_name(ActivationMode mode) noexcept {
    switch (mode) {
    case ActivationMode::Auto:
        return "auto";
    case ActivationMode::Bits16:
        return "16";
    case ActivationMode::Bits8:
        return "8";
    case ActivationMode::Bits4:
        return "4";
    }
    return "unknown";
}

// The narrowest width a mode asks for (16 for Auto, which asks for none).
[[nodiscard]] constexpr int activation_mode_bits(ActivationMode mode) noexcept {
    switch (mode) {
    case ActivationMode::Bits8:
        return 8;
    case ActivationMode::Bits4:
        return 4;
    case ActivationMode::Auto:
    case ActivationMode::Bits16:
        break;
    }
    return 16;
}

// The startup line's summary of a resolution: "auto -> 4-bit on 312, 8-bit on 48, 16-bit on 6
// matrices"; the widths no matrix admits are left out.
[[nodiscard]] inline std::string describe_activation_precision(const ActivationPrecision& p) {
    std::string widths;
    const auto add = [&](std::uint64_t count, const char* bits) {
        if (count == 0) { return; }
        widths +=
            (widths.empty() ? "" : ", ") + std::string(bits) + "-bit on " + std::to_string(count);
    };
    add(p.bits4, "4");
    add(p.bits8, "8");
    add(p.bits16, "16");
    return std::string(activation_mode_name(p.requested)) + " -> " +
           (widths.empty() ? std::string("no quantized matrices") : widths + " matrices");
}

// "4-bit on 0, 8-bit on 120, 16-bit on 2 matrices" for the in-artifact drafter's line.
[[nodiscard]] inline std::string describe_drafter_precision(const ActivationPrecision& p) {
    std::string widths;
    const auto add = [&](std::uint64_t count, const char* bits) {
        if (count == 0) { return; }
        widths +=
            (widths.empty() ? "" : ", ") + std::string(bits) + "-bit on " + std::to_string(count);
    };
    add(p.drafter_bits4, "4");
    add(p.drafter_bits8, "8");
    add(p.drafter_bits16, "16");
    return widths.empty() ? std::string("no quantized matrices") : widths + " matrices";
}

} // namespace ninferno::product
