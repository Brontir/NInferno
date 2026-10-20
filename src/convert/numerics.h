#pragma once

// Numerical transforms of the converter, bit-exact with tools/convert and tools/artifact/codecs.
// Nothing here chooses a format or a tensor: callers pass values, words and geometry.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ninferno::convert {

[[nodiscard]] inline std::uint32_t f32_bits(float value) noexcept {
    std::uint32_t out;
    std::memcpy(&out, &value, 4);
    return out;
}

[[nodiscard]] inline float f32_from_bits(std::uint32_t bits) noexcept {
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

[[nodiscard]] inline float bf16_to_f32(std::uint16_t word) noexcept {
    return f32_from_bits(std::uint32_t(word) << 16);
}

// torch's float -> bfloat16: round to nearest even, NaN to the canonical quiet NaN.
[[nodiscard]] inline std::uint16_t f32_to_bf16(float value) noexcept {
    const auto bits = f32_bits(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) { return 0x7fc0; }
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

// IEEE binary32 -> binary16, round to nearest even (numpy astype(float16)).
[[nodiscard]] std::uint16_t f32_to_f16(float value) noexcept;
[[nodiscard]] float f16_to_f32(std::uint16_t word) noexcept;

// E4M3FN words (bias 7, no infinities, 0x7f/0xff NaN) and E2M1 nibbles.
extern const std::array<float, 256> kE4m3Values;
inline constexpr std::array<float, 16> kE2m1Values = {
    0.0F,  0.5F,  1.0F,  1.5F,  2.0F,  3.0F,  4.0F,  6.0F,
    -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F,
};

// Signed grouped integer formats of row_split_k128_v1.
struct GroupedFormat {
    int bits  = 8;
    int group = 32;
    int qmin  = -127;
    int qmax  = 127;

    [[nodiscard]] std::size_t base_bytes_per_group() const noexcept {
        return bits == 8 ? std::size_t(group) : std::size_t(group) / 2;
    }

    [[nodiscard]] std::size_t high_bytes_per_group() const noexcept {
        return (bits == 4 || bits == 8) ? 0 : std::size_t(group) * std::size_t(bits - 4) / 8;
    }
};

inline constexpr GroupedFormat kQ4{4, 64, -8, 7};
inline constexpr GroupedFormat kQ5{5, 64, -16, 15};
inline constexpr GroupedFormat kQ6{6, 64, -32, 31};
inline constexpr GroupedFormat kQ8{8, 32, -127, 127};

// grouped_absmax on `rows` rows of `k_pad` binary32 values (logical K zero-padded to k_pad).
// Writes each row's base/high plane bytes and its binary16 scale words, rows consecutive.
// Throws on a non-finite value or an unrepresentable scale, as quantize_matrix does.
void quantize_grouped_rows(const GroupedFormat& format, const float* values, std::size_t rows,
                           std::size_t k_pad, std::uint8_t* base, std::uint8_t* high,
                           std::uint16_t* scales);

// fp8_row_maxabs (quantization/fp8_row.py) on `rows` rows of k binary32 values: each value is
// rounded to BF16 first, then a row takes the BF16 scale max|x|/448 (rounded to nearest even,
// never below the smallest subnormal) and E4M3FN codes of x/scale rounded to nearest even.
// Writes codes [rows, k] and one BF16 scale word per row. Throws on a non-finite value.
void quantize_fp8_rows(const float* values, std::size_t rows, std::size_t k, std::uint8_t* codes,
                       std::uint16_t* scales);

// Round-to-nearest NVFP4 of rows of binary32 values (the compressed-tensors / llm-compressor
// weight scheme without calibration): the matrix divisor is 448 * 6 / max|W| (nvfp4_divisor), each
// 16-value block takes the E4M3 scale RNE(max|block| / 6 * divisor), and each value the E2M1 code
// RNE(x * divisor / scale) (saturating at +-6; a zero scale gives zero codes). Writes packed codes
// [rows, k/2] (element 2i in the low nibble) and natural E4M3 scales [rows, k/16]; k % 16 == 0.
[[nodiscard]] float nvfp4_divisor(float max_abs) noexcept;
void quantize_nvfp4_rows(const float* values, std::size_t rows, std::size_t k, float divisor,
                         std::uint8_t* codes, std::uint8_t* scales);

// Natural [rows, k/16] E4M3 scale words to block_scale_k16_m128x4_v1 order: whole 128-row tiles,
// ceil(rows / 128) * 128 * k/16 words, a last tile rows do not fill completed with zero words.
void swizzle_nvfp4_scales(const std::uint8_t* natural, std::size_t rows, std::size_t k,
                          std::uint8_t* out) noexcept;

// One NVFP4 row as the Python sources read it: f32(e2m1 * e4m3) / divisor, true division.
void dequantize_nvfp4_row(const std::uint8_t* codes, const std::uint8_t* scales, float divisor,
                          std::size_t k, float* out) noexcept;

// A finite positive binary32 word, the only divisor the artifact admits.
[[nodiscard]] inline bool valid_positive_f32_word(std::uint32_t word) noexcept {
    const auto value = f32_from_bits(word);
    return (word & 0x7f800000U) != 0x7f800000U && value > 0.0F;
}

} // namespace ninferno::convert
