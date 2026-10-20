#pragma once

// Binary byte units, and byte counts as MiB or GiB for messages and size estimates.

#include <cstdint>

namespace ninferno {

inline constexpr std::uint64_t kMiB = 1ULL << 20;
inline constexpr std::uint64_t kGiB = 1ULL << 30;

template <class T>
[[nodiscard]] constexpr double to_mib(T bytes) noexcept {
    return static_cast<double>(bytes) / static_cast<double>(kMiB);
}

template <class T>
[[nodiscard]] constexpr double to_gib(T bytes) noexcept {
    return static_cast<double>(bytes) / static_cast<double>(kGiB);
}

} // namespace ninferno
