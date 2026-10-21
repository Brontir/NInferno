#pragma once

// SHA-256 (FIPS 180-4) in plain C++, for content digests that must be the same on every host: the
// decision fingerprint (runtime/engine/decision_fingerprint.h), the files it names, and the media
// digests of the Qwen frontend.

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace ninferno {

using Sha256Digest = std::array<std::uint8_t, 32>;

[[nodiscard]] Sha256Digest sha256(std::span<const std::uint8_t> input);
// `checkpoint` runs before every MiB of input (a long hash that can be cancelled).
[[nodiscard]] Sha256Digest sha256(std::span<const std::uint8_t> input,
                                  const std::function<void()>& checkpoint);
[[nodiscard]] Sha256Digest sha256(std::string_view input);
// Lowercase hex, 64 digits.
[[nodiscard]] std::string sha256_hex(const Sha256Digest& digest);
[[nodiscard]] std::string sha256_hex(std::string_view input);
// The digest of a file's bytes; nullopt when it cannot be read.
[[nodiscard]] std::optional<std::string> file_sha256_hex(const std::filesystem::path& path);

} // namespace ninferno
