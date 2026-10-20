// The native converter's numerical kernels against the Python codecs on fixed inputs.
//
// Golden hashes come from numerics_golden.py (tools/convert quantize_matrix + encode_row_split,
// tools/artifact encode_nvfp4) over the same LCG inputs; the rounding helpers are also checked
// against the compiler's own IEEE conversions over a stride of all binary32 words.

#include "convert/numerics.h"
#include "core/weight_view.h"
#include "core/fnv.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__)
#include <immintrin.h>
#endif

using namespace ninferno;
using namespace ninferno::convert;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

struct Lcg {
    std::uint64_t state;

    std::uint32_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>(state >> 40);
    }

    float unit() { return static_cast<float>(next()) * 0x1p-24F - 0.5F; }
};

std::uint64_t fnv1a(const std::vector<std::uint8_t>& data) {
    std::uint64_t h = ninferno::fnv::kOffset;
    for (const auto byte : data) { h = ninferno::fnv::mix(h, byte); }
    return h;
}

struct GroupedCase {
    const char* format;
    std::size_t rows, k;
    std::uint64_t seed, golden;
};

// numerics_golden.py output.
constexpr GroupedCase kGrouped[] = {
    {"q8_g32_fp16", 37, 200, 1, 0x79e5975e65ae2081ULL},
    {"q6_g64_fp16", 19, 200, 2, 0x144930b77295bfabULL},
    {"q5_g64_fp16", 11, 256, 3, 0x19919c20df427e82ULL},
    {"q4_g64_fp16", 7, 130, 4, 0x798a804a55bba8e7ULL},
};
constexpr std::uint64_t kNvfp4Golden = 0x66dcf9e0799bcd85ULL;

void grouped(const GroupedCase& c) {
    const std::string name      = c.format;
    const auto format           = name == "q8_g32_fp16"   ? kQ8
                                  : name == "q6_g64_fp16" ? kQ6
                                  : name == "q5_g64_fp16" ? kQ5
                                                          : kQ4;
    const auto qtype            = name == "q8_g32_fp16"   ? QType::Q8_G32_FP16
                                  : name == "q6_g64_fp16" ? QType::Q6_G64_FP16
                                  : name == "q5_g64_fp16" ? QType::Q5_G64_FP16
                                                          : QType::Q4_G64_FP16;
    const std::uint64_t shape[] = {c.rows, c.k};
    const auto g                = weight_geometry(qtype, QuantLayout::RowSplit, shape);
    const auto k_pad            = g.padded_columns;
    Lcg lcg{c.seed};
    std::vector<float> values(c.rows * k_pad, 0.0F);
    for (std::size_t r = 0; r < c.rows; ++r) {
        for (std::size_t i = 0; i < c.k; ++i) { values[r * k_pad + i] = lcg.unit() * 8.0F; }
    }
    for (std::size_t i = 0; i < 32; ++i) {
        values[i] = 0.0F;
        values[k_pad + i] *= 0x1p-110F;
    }
    const float ties[] = {
        static_cast<float>(format.qmax), 2.5F, -3.5F, 0.5F, -0.5F, 1.5F, -1.5F, 4.5F};
    for (std::size_t i = 0; i < 32; ++i) { values[2 * k_pad + i] = i < 8 ? ties[i] : 0.25F; }

    std::vector<std::uint8_t> base(c.rows * g.code_bytes_per_row);
    std::vector<std::uint8_t> high(c.rows * g.high_bytes_per_row);
    std::vector<std::uint16_t> scales(c.rows * g.scale_bytes_per_row / 2);
    quantize_grouped_rows(format, values.data(), c.rows, k_pad, base.data(), high.data(),
                          scales.data());
    std::vector<std::uint8_t> payload(g.bytes, 0);
    std::memcpy(payload.data(), base.data(), base.size());
    if (!high.empty()) { std::memcpy(payload.data() + g.high_offset, high.data(), high.size()); }
    std::memcpy(payload.data() + g.scale_offset, scales.data(), scales.size() * 2);
    char text[64];
    std::snprintf(text, sizeof(text), "0x%016llx", static_cast<unsigned long long>(fnv1a(payload)));
    check(fnv1a(payload) == c.golden, name + " encoding differs from Python: " + text);
}

void nvfp4() {
    constexpr std::size_t rows = 256;
    constexpr std::size_t k    = 128;
    Lcg lcg{5};
    std::vector<std::uint8_t> codes(rows * k / 2);
    std::vector<std::uint8_t> scales(rows * k / 16);
    // Every code first, then every scale, as numerics_golden.py draws them.
    for (auto& code : codes) { code = lcg.next() & 0xFF; }
    for (auto& scale : scales) { scale = lcg.next() % 0x7F; }
    const std::uint64_t shape[] = {rows, k};
    const auto g = weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, shape);
    std::vector<std::uint8_t> payload(g.bytes, 0);
    std::memcpy(payload.data(), codes.data(), codes.size());
    swizzle_nvfp4_scales(scales.data(), rows, k, payload.data() + g.scale_offset);
    const float divisor = 0.123F;
    std::memcpy(payload.data() + g.divisor_offset, &divisor, 4);
    char text[64];
    std::snprintf(text, sizeof(text), "0x%016llx", static_cast<unsigned long long>(fnv1a(payload)));
    check(fnv1a(payload) == kNvfp4Golden, std::string("nvfp4 repack differs from Python: ") + text);
}

// A plane whose rows end inside a 128-row scale tile (a 202048-row vocabulary): the geometry
// holds the whole last tile, every row's words land where the layout addresses them and the
// words of the rows past N are zero.
void nvfp4_ragged() {
    constexpr std::size_t rows = 200;
    constexpr std::size_t k    = 128;
    Lcg lcg{7};
    std::vector<std::uint8_t> scales(rows * k / 16);
    for (auto& scale : scales) { scale = 1 + lcg.next() % 0x7E; }
    const std::uint64_t shape[] = {rows, k};
    const auto g = weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, shape);
    check(g.scale_bytes == 256 * k / 16 && g.code_bytes == rows * k / 2 &&
              g.bytes == g.scale_offset + g.scale_bytes + 4,
          "ragged nvfp4 geometry holds whole scale tiles");
    std::vector<std::uint8_t> payload(g.bytes, 0xAB);
    swizzle_nvfp4_scales(scales.data(), rows, k, payload.data() + g.scale_offset);
    std::size_t placed = 0;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t group = 0; group < k / 16; ++group) {
            placed += payload[weight_scale_offset(g, row, group)] == scales[row * (k / 16) + group];
        }
    }
    check(placed == scales.size(), "ragged nvfp4 scale words at their layout offsets");
    std::size_t zero = 0;
    for (std::size_t i = 0; i < g.scale_bytes; ++i) { zero += payload[g.scale_offset + i] == 0; }
    check(zero == (256 - rows) * (k / 16), "ragged nvfp4 scale words past N are zero");
}

// f32_to_f16 against the compiler's IEEE conversion; bf16 and the dequantized value table too.
void conversions() {
    for (std::uint64_t word = 0; word <= 0xffffffffULL; word += 997) {
        const float value = f32_from_bits(static_cast<std::uint32_t>(word));
        if (std::isnan(value)) { continue; }
#if defined(_MSC_VER) && !defined(__clang__)
        // MSVC has no _Float16: the processor's F16C conversion (round to nearest even).
        const auto expected = static_cast<std::uint16_t>(
            _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(value), _MM_FROUND_TO_NEAREST_INT), 0));
        const float back = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(expected)));
#else
        const auto reference = static_cast<_Float16>(value);
        std::uint16_t expected;
        std::memcpy(&expected, &reference, 2);
        const auto back = static_cast<float>(reference);
#endif
        if (f32_to_f16(value) != expected) {
            check(false, "f32_to_f16 word " + std::to_string(word));
            return;
        }
        if (!std::isnan(back) && f16_to_f32(expected) != back) {
            check(false, "f16_to_f32 word " + std::to_string(expected));
            return;
        }
    }
    check(f32_to_bf16(1.00390625F) == 0x3f80, "bf16 tie rounds to even (down)");
    check(f32_to_bf16(1.01171875F) == 0x3f82, "bf16 tie rounds to even (up)");
    check(kE4m3Values[0x7e] == 448.0F && kE4m3Values[0x01] == 0x1p-9F, "e4m3 table");
    // One NVFP4 row: e2m1 * e4m3 / divisor, true division.
    const std::uint8_t row_codes[8]  = {0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe};
    const std::uint8_t row_scales[1] = {0x38}; // 1.0
    float out[16];
    dequantize_nvfp4_row(row_codes, row_scales, 3.0F, 16, out);
    check(out[0] == 0.0F && out[1] == 0.5F / 3.0F && out[15] == -6.0F / 3.0F, "nvfp4 dequant");
}

} // namespace

int main() {
    for (const auto& c : kGrouped) { grouped(c); }
    nvfp4();
    nvfp4_ragged();
    conversions();
    if (failures) { return 1; }
    std::printf("convert numerics: all checks passed\n");
    return 0;
}
