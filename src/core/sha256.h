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
// A path's file name as UTF-8 (path::string() throws on Windows for a name outside the ANSI code
// page; this never throws for a name the file system gave).
[[nodiscard]] std::string utf8_file_name(const std::filesystem::path& path);
// How the decision fingerprint names a source file: "<file name> sha256:<hex of its bytes>";
// nullopt when it cannot be read. Never throws.
[[nodiscard]] std::optional<std::string> named_file_sha256(const std::filesystem::path& path);

} // namespace ninferno
