#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninferno::artifact {

inline constexpr std::array<std::byte, 8> magic8(const char (&text)[9]) {
    std::array<std::byte, 8> out{};
    for (std::size_t i = 0; i < out.size(); ++i) { out[i] = static_cast<std::byte>(text[i]); }
    return out;
}

// The first 8 bytes of an entry file and of each continuation part.
inline constexpr auto kEntryMagic = magic8("NInferno");
inline constexpr auto kPartMagic  = magic8("NInfPart");

inline bool has_magic(const std::array<std::byte, 8>& magic, const std::byte* header) noexcept {
    for (std::size_t i = 0; i < magic.size(); ++i) {
        if (header[i] != magic[i]) { return false; }
    }
    return true;
}
inline constexpr std::size_t kHeaderBytes        = 32;
inline constexpr std::uint64_t kPayloadAlignment = 4096;
using ArtifactId                                 = std::array<std::byte, 16>;

inline std::uint64_t read_u64_le(const std::byte* bytes) noexcept {
    std::uint64_t out = 0;
    for (unsigned i = 0; i < 8; ++i) {
        out |= std::uint64_t(std::to_integer<unsigned>(bytes[i])) << (8 * i);
    }
    return out;
}

inline std::uint32_t read_u32_le(const std::byte* bytes) noexcept {
    std::uint32_t out = 0;
    for (unsigned i = 0; i < 4; ++i) {
        out |= std::uint32_t(std::to_integer<unsigned>(bytes[i])) << (8 * i);
    }
    return out;
}

} // namespace ninferno::artifact
