#include "convert/numerics.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ninferno::convert {
namespace {

// Round half to even for |v| < 2^22 without a libm call: adding 1.5 * 2^23 leaves no fraction bits,
// so the addition itself rounds (to nearest even, the default mode) and the subtraction is exact.
// Quantized codes are bounded by qmax times a scale rounding factor, far inside that range.
inline float round_even(float v) noexcept {
    constexpr float kMagic = 12582912.0F;
    return (v + kMagic) - kMagic;
}

std::array<float, 256> e4m3_table() {
    std::array<float, 256> out{};
    for (unsigned word = 0; word < 256; ++word) {
        const float sign    = (word & 0x80U) ? -1.0F : 1.0F;
        const unsigned exp  = (word >> 3) & 0xFU;
        const unsigned frac = word & 0x7U;
        if ((word & 0x7FU) == 0x7FU) {
            out[word] = std::copysign(NAN, sign);
        } else if (exp == 0) {
            out[word] = sign * static_cast<float>(frac) * 0x1p-9F;
        } else {
            out[word] = sign * (1.0F + static_cast<float>(frac) / 8.0F) *
                        std::ldexp(1.0F, static_cast<int>(exp) - 7);
        }
    }
    return out;
}

} // namespace

const std::array<float, 256> kE4m3Values = e4m3_table();

std::uint16_t f32_to_f16(float value) noexcept {
    const auto bits = f32_bits(value);
    const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000U);
    const auto ax   = bits & 0x7fffffffU;
    if (ax >= 0x7f800000U) { return sign | (ax > 0x7f800000U ? 0x7e00 : 0x7c00); }
    if (ax >= 0x477ff000U) { return sign | 0x7c00; } // rounds past 65504
    if (ax < 0x38800000U) {
        // Below the smallest normal binary16: the result counts units of 2^-24. Scaling by a
        // power of two is exact, so nearbyint performs the only (ties-to-even) rounding.
        const float scaled = f32_from_bits(ax) * 16777216.0F;
        return sign | static_cast<std::uint16_t>(std::nearbyint(scaled));
    }
    const auto rebased = ax - (112U << 23);
    return sign | static_cast<std::uint16_t>((rebased + 0xfffU + ((ax >> 13) & 1U)) >> 13);
}

float f16_to_f32(std::uint16_t word) noexcept {
    const std::uint32_t sign = std::uint32_t(word & 0x8000U) << 16;
    const std::uint32_t exp  = (word >> 10) & 0x1fU;
    const std::uint32_t frac = word & 0x3ffU;
    if (exp == 0) {
        const float magnitude = static_cast<float>(frac) * 0x1p-24F;
        return sign ? -magnitude : magnitude;
    }
    if (exp == 31) { return f32_from_bits(sign | 0x7f800000U | (frac << 13)); }
    return f32_from_bits(sign | ((exp + 112U) << 23) | (frac << 13));
}

void quantize_grouped_rows(const GroupedFormat& format, const float* values, std::size_t rows,
                           std::size_t k_pad, std::uint8_t* base, std::uint8_t* high,
                           std::uint16_t* scales) {
    const std::size_t group    = static_cast<std::size_t>(format.group);
    const std::size_t groups   = k_pad / group;
    const std::size_t base_bpg = format.base_bytes_per_group();
    const std::size_t high_bpg = format.high_bytes_per_group();
    const int high_bits        = format.bits - 4;
    const int mask             = (1 << format.bits) - 1;
    const float qmin           = static_cast<float>(format.qmin);
    const float qmax           = static_cast<float>(format.qmax);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t g = 0; g < groups; ++g) {
            const float* x = values + r * k_pad + g * group;
            float max_abs  = 0.0F;
            bool finite    = true;
            for (std::size_t i = 0; i < group; ++i) {
                const float a = std::fabs(x[i]);
                finite        = finite && a <= FLT_MAX;
                max_abs       = a > max_abs ? a : max_abs;
            }
            if (!finite) {
                throw std::invalid_argument("grouped quantization source contains NaN or infinity");
            }
            const auto raw  = static_cast<float>(static_cast<double>(max_abs) / format.qmax);
            std::uint16_t h = f32_to_f16(raw);
            if (h == 0 && max_abs > 0.0F) { h = 1; } // binary16 minimum subnormal
            if (max_abs > 0.0F && h >= 0x7c00) {
                throw std::invalid_argument(
                    "grouped quantization scale is not finite and positive");
            }
            scales[r * groups + g] = h;
            const float scale      = f16_to_f32(h);
            const float reciprocal =
                scale > 0.0F ? static_cast<float>(1.0 / static_cast<double>(scale)) : 0.0F;
            std::uint8_t* b  = base + (r * groups + g) * base_bpg;
            std::uint8_t* hi = high_bpg ? high + (r * groups + g) * high_bpg : nullptr;
            if (format.bits == 8) {
                for (std::size_t i = 0; i < group; ++i) {
                    float q = round_even(x[i] * reciprocal);
                    q       = q < qmin ? qmin : (q > qmax ? qmax : q);
                    b[i]    = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
                }
                continue;
            }
            for (std::size_t i = 0; i < base_bpg; ++i) { b[i] = 0; }
            for (std::size_t i = 0; i < high_bpg; ++i) { hi[i] = 0; }
            for (std::size_t i = 0; i < group; ++i) {
                float q     = round_even(x[i] * reciprocal);
                q           = q < qmin ? qmin : (q > qmax ? qmax : q);
                const int c = static_cast<int>(q);
                b[i / 2] |= static_cast<std::uint8_t>((c & 0x0F) << (4 * (i & 1)));
                if (hi) {
                    const int upper = ((c & mask) >> 4) & ((1 << high_bits) - 1);
                    for (int s = 0; s < high_bits; ++s) {
                        const std::size_t bit = i * std::size_t(high_bits) + std::size_t(s);
                        hi[bit / 8] |= static_cast<std::uint8_t>(((upper >> s) & 1) << (bit % 8));
                    }
                }
            }
        }
    }
}

void swizzle_nvfp4_scales(const std::uint8_t* natural, std::size_t rows, std::size_t k,
                          std::uint8_t* out) noexcept {
    const std::size_t per_row = k / 16;
    const std::size_t k_tiles = k / 64;
    std::size_t o             = 0;
    for (std::size_t nt = 0; nt < (rows + 127) / 128; ++nt) {
        for (std::size_t kt = 0; kt < k_tiles; ++kt) {
            for (std::size_t r32 = 0; r32 < 32; ++r32) {
                for (std::size_t r4 = 0; r4 < 4; ++r4) {
                    const std::size_t row = nt * 128 + r4 * 32 + r32;
                    if (row >= rows) {
                        for (int i = 0; i < 4; ++i) { out[o++] = 0; }
                        continue;
                    }
                    const std::uint8_t* src = natural + row * per_row + kt * 4;
                    out[o++]                = src[0];
                    out[o++]                = src[1];
                    out[o++]                = src[2];
                    out[o++]                = src[3];
                }
            }
        }
    }
}

void dequantize_nvfp4_row(const std::uint8_t* codes, const std::uint8_t* scales, float divisor,
                          std::size_t k, float* out) noexcept {
    for (std::size_t j = 0; j < k / 2; ++j) {
        const float scale = kE4m3Values[scales[(2 * j) / 16]];
        const auto byte   = codes[j];
        out[2 * j]        = (kE2m1Values[byte & 15U] * scale) / divisor;
        out[2 * j + 1]    = (kE2m1Values[byte >> 4] * scale) / divisor;
    }
}

namespace {

// E4M3FN magnitudes of words 0x00..0x7e, ascending (the finite positive codes).
const std::array<float, 127>& positive_e4m3() {
    static const auto table = [] {
        std::array<float, 127> out{};
        for (std::size_t i = 0; i < out.size(); ++i) { out[i] = kE4m3Values[i]; }
        return out;
    }();
    return table;
}

// _round_e4m3fn_rne on one bounded finite binary32 value.
std::uint8_t round_e4m3fn(float value) {
    const auto& table     = positive_e4m3();
    const float magnitude = std::fabs(value);
    auto upper =
        static_cast<int>(std::lower_bound(table.begin(), table.end(), magnitude) - table.begin());
    upper                      = std::min(upper, 0x7e);
    const int lower            = std::max(upper - 1, 0);
    const float lower_distance = magnitude - table[static_cast<std::size_t>(lower)];
    const float upper_distance = table[static_cast<std::size_t>(upper)] - magnitude;
    const bool choose_upper =
        upper_distance < lower_distance || (upper_distance == lower_distance && (upper & 1) == 0);
    auto word = static_cast<std::uint8_t>(choose_upper ? upper : lower);
    if (std::signbit(value)) { word |= 0x80; }
    return word;
}

// The E2M1 code of a value already divided by its block scale: nearest of 0, .5, 1, 1.5, 2, 3,
// 4, 6 with ties to the even code, saturating at 6.
std::uint8_t round_e2m1(float value) {
    const float magnitude = std::min(std::fabs(value), 6.0F);
    int code              = 0;
    for (int c = 1; c < 8; ++c) {
        const float below = kE2m1Values[static_cast<std::size_t>(c - 1)];
        const float above = kE2m1Values[static_cast<std::size_t>(c)];
        if (magnitude <= below) { break; }
        const float mid = (below + above) * 0.5F;
        if (magnitude > mid || (magnitude == mid && (c & 1) == 0)) {
            code = c;
        } else {
            break;
        }
    }
    return static_cast<std::uint8_t>(code | (std::signbit(value) && code != 0 ? 8 : 0));
}

} // namespace

float nvfp4_divisor(float max_abs) noexcept {
    return max_abs > 0.0F && std::isfinite(max_abs) ? 448.0F * 6.0F / max_abs : 1.0F;
}

void quantize_nvfp4_rows(const float* values, std::size_t rows, std::size_t k, float divisor,
                         std::uint8_t* codes, std::uint8_t* scales) {
    if (k % 16 != 0) { throw std::runtime_error("NVFP4 quantization needs K % 16 == 0"); }
    for (std::size_t r = 0; r < rows; ++r) {
        const float* row = values + r * k;
        for (std::size_t b = 0; b < k / 16; ++b) {
            float block_max = 0.0F;
            for (std::size_t j = 0; j < 16; ++j) {
                const float v = row[b * 16 + j];
                if (!std::isfinite(v)) {
                    throw std::runtime_error("NVFP4 quantization source contains NaN or infinity");
                }
                block_max = std::max(block_max, std::fabs(v));
            }
            const auto word =
                round_e4m3fn(std::min(block_max / 6.0F * divisor, 448.0F));
            scales[r * (k / 16) + b] = word;
            const float scale        = kE4m3Values[word];
            for (std::size_t j = 0; j < 16; j += 2) {
                std::uint8_t lo = 0, hi = 0;
                if (scale > 0.0F) {
                    lo = round_e2m1(row[b * 16 + j] * divisor / scale);
                    hi = round_e2m1(row[b * 16 + j + 1] * divisor / scale);
                }
                codes[r * (k / 2) + (b * 16 + j) / 2] = static_cast<std::uint8_t>(lo | (hi << 4));
            }
        }
    }
}

void quantize_fp8_rows(const float* values, std::size_t rows, std::size_t k, std::uint8_t* codes,
                       std::uint16_t* scales) {
    std::vector<float> host(k);
    for (std::size_t r = 0; r < rows; ++r) {
        float max_abs = 0.0F;
        for (std::size_t c = 0; c < k; ++c) {
            host[c] = bf16_to_f32(f32_to_bf16(values[r * k + c]));
            if (!std::isfinite(host[c])) {
                throw std::runtime_error("FP8 quantization source contains NaN or infinity");
            }
            max_abs = std::max(max_abs, std::fabs(host[c]));
        }
        auto* row_codes = codes + r * k;
        if (max_abs == 0.0F) {
            std::memset(row_codes, 0, k);
            scales[r] = 0;
            continue;
        }
        const auto raw  = static_cast<float>(static_cast<double>(max_abs) / 448.0);
        const auto bits = static_cast<std::uint64_t>(f32_bits(raw));
        auto word       = static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
        if (word == 0) { word = 0x0001; }
        const float scale = bf16_to_f32(word);
        if (!std::isfinite(scale) || scale <= 0.0F) {
            throw std::runtime_error("FP8 quantization row scale is not finite and positive");
        }
        const auto reciprocal = static_cast<double>(static_cast<float>(1.0 / scale));
        for (std::size_t c = 0; c < k; ++c) {
            const auto normalized = static_cast<float>(static_cast<double>(host[c]) * reciprocal);
            if (!std::isfinite(normalized)) {
                throw std::runtime_error(
                    "FP8 quantization normalization produced a non-finite value");
            }
            row_codes[c] = round_e4m3fn(std::clamp(normalized, -448.0F, 448.0F));
        }
        scales[r] = word;
    }
}

} // namespace ninferno::convert
