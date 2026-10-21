#pragma once

#include "core/sha256.h"

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>

namespace ninferno::models::qwen3_5::frontend {

using Sha256Digest = ninferno::Sha256Digest;

[[nodiscard]] Sha256Digest sha256(std::span<const std::uint8_t> input);
[[nodiscard]] Sha256Digest sha256(std::span<const std::uint8_t> input,
                                  const std::function<void()>& checkpoint);
[[nodiscard]] Sha256Digest sha256(std::string_view input);
[[nodiscard]] std::string sha256_hex(const Sha256Digest& digest);

} // namespace ninferno::models::qwen3_5::frontend
