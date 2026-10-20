#include "core/ggml_frag.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ninferno {
namespace {

// Chunk q of a 64-byte span at `base` takes word q of each of the span's four 16-byte rows.
template <std::size_t N>
constexpr void transpose64(std::array<std::uint8_t, N>& m, std::uint32_t base) {
    for (std::uint32_t q = 0; q < 4; ++q)
        for (std::uint32_t i = 0; i < 4; ++i)
            for (std::uint32_t t = 0; t < 4; ++t)
                m[base + 16 * q + 4 * i + t] = static_cast<std::uint8_t>(base + 16 * i + 4 * q + t);
}

constexpr std::array<std::uint8_t, 64> q40_map() {
    std::array<std::uint8_t, 64> m{};
    transpose64(m, 0);
    return m;
}

constexpr std::array<std::uint8_t, 144> q4k_map() {
    std::array<std::uint8_t, 144> m{};
    for (std::uint32_t i = 0; i < 16; ++i) m[i] = static_cast<std::uint8_t>(i);
    transpose64(m, 16);
    transpose64(m, 80);
    return m;
}

constexpr std::array<std::uint8_t, 64> iq3xxs_map() {
    std::array<std::uint8_t, 64> m{};
    for (std::uint32_t h = 0; h < 2; ++h)
        for (std::uint32_t q = 0; q < 4; ++q)
            for (std::uint32_t j = 0; j < 8; ++j)
                m[32 * h + 8 * q + j] = static_cast<std::uint8_t>(32 * h + 4 * j + q);
    return m;
}

constexpr std::array<std::uint8_t, 72> iq2s_map() {
    std::array<std::uint8_t, 72> m{};
    for (std::uint32_t h = 0; h < 2; ++h)
        for (std::uint32_t b = 0; b < 2; ++b)
            for (std::uint32_t j = 0; j < 8; ++j) {
                m[32 * h + 16 * b + j]     = static_cast<std::uint8_t>(16 * h + b + 2 * j);
                m[32 * h + 16 * b + 8 + j] = static_cast<std::uint8_t>(32 + 16 * h + b + 2 * j);
            }
    for (std::uint32_t i = 64; i < 72; ++i) m[i] = static_cast<std::uint8_t>(i);
    return m;
}

constexpr auto kQ40    = q40_map();
constexpr auto kQ4K    = q4k_map();
constexpr auto kIq3Xxs = iq3xxs_map();
constexpr auto kIq2S   = iq2s_map();

const std::uint8_t* map_of(QType format) noexcept {
    switch (format) {
    case QType::GGML_Q4_0: return kQ40.data();
    case QType::GGML_Q4_K: return kQ4K.data();
    case QType::GGML_IQ3_XXS: return kIq3Xxs.data();
    case QType::GGML_IQ2_S: return kIq2S.data();
    default: return nullptr;
    }
}

} // namespace

bool ggml_frag_format(QType format) noexcept { return map_of(format) != nullptr; }

QuantLayout ggml_published_layout(QType format) noexcept {
    return format == QType::GGML_Q4_K || format == QType::GGML_Q5_K ? QuantLayout::GgmlBlocks
                                                                     : QuantLayout::GgmlPlanes;
}

std::uint32_t ggml_frag_unit_bytes(QType format) noexcept {
    switch (format) {
    case QType::GGML_Q4_0: return 64;
    case QType::GGML_Q4_K: return 144;
    case QType::GGML_IQ3_XXS: return 64;
    case QType::GGML_IQ2_S: return 72;
    default: return 0;
    }
}

std::uint32_t ggml_frag_source(QType format, std::uint32_t i) noexcept {
    const auto* map = map_of(format);
    return map != nullptr && i < ggml_frag_unit_bytes(format) ? map[i] : 0;
}

void ggml_frag_row(QType format, std::uint64_t k, const std::uint8_t* in, std::uint8_t* out,
                   bool to_frag) {
    const auto* map   = map_of(format);
    const auto values = ggml_block_values(format);
    if (map == nullptr || values == 0 || k == 0 || k % values != 0) {
        throw std::invalid_argument("ggml_frag_v1: no map for format " +
                                    std::to_string(static_cast<int>(format)) + " at K " +
                                    std::to_string(k));
    }
    const std::uint64_t unit   = ggml_frag_unit_bytes(format);
    const std::uint64_t blocks = k / values;
    // Q4_0 maps whole four-block spans; the row's other formats map every superblock.
    const std::uint64_t units     = format == QType::GGML_Q4_0 ? blocks / 4 : blocks;
    const std::uint64_t row_bytes =
        format == QType::GGML_Q4_K ? blocks * 144 : blocks * ggml_plane_bytes(format).code;
    for (std::uint64_t u = 0; u < units; ++u) {
        const std::uint8_t* src = in + u * unit;
        std::uint8_t* dst       = out + u * unit;
        for (std::uint64_t i = 0; i < unit; ++i) {
            if (to_frag) {
                dst[i] = src[map[i]];
            } else {
                dst[map[i]] = src[i];
            }
        }
    }
    const std::uint64_t mapped = units * unit;
    if (mapped < row_bytes) { std::memcpy(out + mapped, in + mapped, row_bytes - mapped); }
}

} // namespace ninferno
