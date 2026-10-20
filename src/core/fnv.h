#pragma once

// FNV-1a, 64-bit: the hash the engine keys its identities and digests with and the converter its
// checksums. mix() folds one value in (a byte, or a whole word, as the planners fold fields).

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace ninferno::fnv {

inline constexpr std::uint64_t kOffset = 0xcbf29ce484222325ULL; // 14695981039346656037
inline constexpr std::uint64_t kPrime  = 0x100000001b3ULL;      // 1099511628211

[[nodiscard]] constexpr std::uint64_t mix(std::uint64_t hash, std::uint64_t value) noexcept {
    return (hash ^ value) * kPrime;
}

[[nodiscard]] constexpr std::uint64_t bytes(const unsigned char* data, std::size_t size,
                                            std::uint64_t hash = kOffset) noexcept {
    for (std::size_t i = 0; i < size; ++i) { hash = mix(hash, data[i]); }
    return hash;
}

[[nodiscard]] constexpr std::uint64_t bytes(std::string_view text,
                                            std::uint64_t hash = kOffset) noexcept {
    for (const char c : text) { hash = mix(hash, static_cast<unsigned char>(c)); }
    return hash;
}

} // namespace ninferno::fnv
