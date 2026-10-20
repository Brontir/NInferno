// ggml block decoders against hand-computed expectations, and the NVFP4 / MXFP4 repacks
// against the values their blocks decode to. Every type is also cross-checked bit for bit against
// gguf-py's independent numpy decoders by gguf_pyref.py.

#include "convert/ggml_quants.h"
#include "convert/numerics.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace ninferno::convert;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

void put16(std::vector<std::uint8_t>& b, std::size_t at, std::uint16_t v) {
    std::memcpy(b.data() + at, &v, 2);
}

std::vector<float> decode(std::uint32_t type, const std::vector<std::uint8_t>& block,
                          std::uint64_t elements) {
    std::vector<float> out(elements);
    ggml::dequantize(type, reinterpret_cast<const std::byte*>(block.data()), elements, out.data());
    return out;
}

bool same_bits(float a, float b) { return f32_bits(a) == f32_bits(b); }

constexpr float kE2m1[16] = {0, 0.5F, 1, 1.5F, 2, 3, 4, 6, -0.0F, -0.5F, -1, -1.5F, -2, -3, -4, -6};

void test_traits() {
    struct Case {
        std::uint32_t type;
        std::uint32_t block, bytes;
    };
    // ggml-common.h static_asserts / GGML_QUANT_SIZES.
    const Case cases[] = {
        {ggml::Q4_0, 32, 18},    {ggml::Q4_1, 32, 20},    {ggml::Q5_0, 32, 22},
        {ggml::Q5_1, 32, 24},    {ggml::Q8_0, 32, 34},    {ggml::Q2_K, 256, 84},
        {ggml::Q3_K, 256, 110},  {ggml::Q4_K, 256, 144},  {ggml::Q5_K, 256, 176},
        {ggml::Q6_K, 256, 210},  {ggml::IQ2_XXS, 256, 66}, {ggml::IQ2_XS, 256, 74},
        {ggml::IQ2_S, 256, 82},  {ggml::IQ3_XXS, 256, 98}, {ggml::IQ3_S, 256, 110},
        {ggml::IQ1_S, 256, 50},  {ggml::IQ1_M, 256, 56},  {ggml::IQ4_NL, 32, 18},
        {ggml::IQ4_XS, 256, 136}, {ggml::TQ1_0, 256, 54}, {ggml::TQ2_0, 256, 66},
        {ggml::MXFP4, 32, 17},   {ggml::NVFP4, 64, 36},   {ggml::Q1_0, 128, 18},
        {ggml::Q2_0, 64, 18},    {ggml::BF16, 1, 2},      {ggml::F16, 1, 2},
        {ggml::F32, 1, 4},
    };
    for (const auto& c : cases) {
        const auto t = ggml::traits(c.type);
        check(t && t->block == c.block && t->bytes == c.bytes,
              "traits of " + ggml::type_name(c.type));
    }
    check(!ggml::traits(4), "type 4 (removed Q4_2) is unknown");
    check(ggml::type_name(ggml::Q4_K) == "q4_k", "lowercase type name");
    bool threw = false;
    try {
        (void)ggml::bytes_for(ggml::Q4_K, 100);
    } catch (const std::exception&) { threw = true; }
    check(threw, "partial blocks are rejected");
    check(ggml::bytes_for(ggml::Q4_K, 512) == 288, "bytes of two Q4_K blocks");
}

void test_q8_0() {
    std::vector<std::uint8_t> b(34);
    put16(b, 0, 0x3800); // 0.5
    for (int j = 0; j < 32; ++j) { b[2 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(j * 8 - 128)); }
    const auto y = decode(ggml::Q8_0, b, 32);
    bool ok = true;
    for (int j = 0; j < 32; ++j) { ok = ok && y[j] == static_cast<float>(j * 8 - 128) * 0.5F; }
    check(ok, "q8_0 values are q * d");
}

void test_q4_0() {
    std::vector<std::uint8_t> b(18);
    put16(b, 0, 0x3C00); // 1.0
    for (int j = 0; j < 16; ++j) { b[2 + j] = static_cast<std::uint8_t>(j | ((15 - j) << 4)); }
    const auto y = decode(ggml::Q4_0, b, 32);
    bool ok = true;
    for (int j = 0; j < 16; ++j) {
        ok = ok && y[j] == static_cast<float>(j - 8) && y[j + 16] == static_cast<float>(7 - j);
    }
    check(ok, "q4_0 low nibbles are the first half, high nibbles the second");
}

void test_q4_k() {
    std::vector<std::uint8_t> b(144);
    put16(b, 0, 0x3C00); // d = 1
    put16(b, 2, 0x3800); // dmin = 0.5
    // Sub-block scale 2 and min 3 for all eight sub-blocks (6-bit packing of get_scale_min_k4).
    const std::uint8_t scales[12] = {2, 2, 2, 2, 3, 3, 3, 3, 0x32, 0x32, 0x32, 0x32};
    std::memcpy(b.data() + 4, scales, 12);
    for (int i = 0; i < 128; ++i) { b[16 + i] = static_cast<std::uint8_t>((i % 16) | ((15 - i % 16) << 4)); }
    const auto y = decode(ggml::Q4_K, b, 256);
    bool ok = true;
    for (int chunk = 0; chunk < 4; ++chunk) {
        for (int l = 0; l < 32; ++l) {
            const int lo = (chunk * 32 + l) % 16;
            ok = ok && y[chunk * 64 + l] == 2.0F * static_cast<float>(lo) - 1.5F;
            ok = ok && y[chunk * 64 + 32 + l] == 2.0F * static_cast<float>(15 - lo) - 1.5F;
        }
    }
    check(ok, "q4_k: d*sc*q - dmin*m over 64-value chunks");
}

void test_q6_k() {
    std::vector<std::uint8_t> b(210);
    // ql = 0x21 everywhere, qh = 0: low codes 1, high codes 2; scales 1..16; d = 0.25.
    for (int i = 0; i < 128; ++i) { b[i] = 0x21; }
    for (int i = 0; i < 16; ++i) { b[192 + i] = static_cast<std::uint8_t>(i + 1); }
    put16(b, 208, 0x3400); // 0.25
    const auto y = decode(ggml::Q6_K, b, 256);
    bool ok = true;
    for (int n = 0; n < 2; ++n) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const auto s = [&](int k) { return static_cast<float>(n * 8 + is + k + 1); };
            ok = ok && y[n * 128 + l] == 0.25F * s(0) * -31.0F;
            ok = ok && y[n * 128 + l + 32] == 0.25F * s(2) * -31.0F;
            ok = ok && y[n * 128 + l + 64] == 0.25F * s(4) * -30.0F;
            ok = ok && y[n * 128 + l + 96] == 0.25F * s(6) * -30.0F;
        }
    }
    check(ok, "q6_k: d * scale * (q - 32)");
}

void test_iq4_nl() {
    static constexpr int kValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                        1,    13,   25,  38,  53,  69,  89,  113};
    std::vector<std::uint8_t> b(18);
    put16(b, 0, 0x4000); // 2.0
    for (int j = 0; j < 16; ++j) { b[2 + j] = static_cast<std::uint8_t>(j | ((15 - j) << 4)); }
    const auto y = decode(ggml::IQ4_NL, b, 32);
    bool ok = true;
    for (int j = 0; j < 16; ++j) {
        ok = ok && y[j] == 2.0F * kValues[j] && y[j + 16] == 2.0F * kValues[15 - j];
    }
    check(ok, "iq4_nl: d * codebook");
}

void test_mxfp4() {
    std::vector<std::uint8_t> b(17);
    b[0] = 130; // 2^3
    for (int j = 0; j < 16; ++j) { b[1 + j] = static_cast<std::uint8_t>(j | ((15 - j) << 4)); }
    const auto y = decode(ggml::MXFP4, b, 32);
    bool ok = true;
    // ggml's doubled integer codebook has no negative zero: code 8 decodes to +0.
    for (int j = 0; j < 16; ++j) {
        ok = ok && y[j] == kE2m1[j] * 8.0F && y[j + 16] == kE2m1[15 - j] * 8.0F;
    }
    ok = ok && !std::signbit(y[8]);
    check(ok, "mxfp4: e2m1 * 2^(e-127), element j and j+16 share a byte");
    std::uint8_t codes[16];
    const auto e = ggml::mxfp4_block_natural(b.data(), codes);
    ok = e == 130;
    for (int i = 0; i < 16; ++i) {
        const float lo = kE2m1[codes[i] & 15] * 8.0F;
        const float hi = kE2m1[codes[i] >> 4] * 8.0F;
        ok = ok && lo == y[2 * i] && hi == y[2 * i + 1];
    }
    check(ok, "mxfp4 repack: natural nibble order reproduces every value");
}

void test_nvfp4() {
    std::vector<std::uint8_t> b(36);
    const std::uint8_t scales[4] = {0x38, 0x40, 0x7F, 0x01}; // 1.0, 2.0, ggml's zero, 2^-9
    std::memcpy(b.data(), scales, 4);
    for (int j = 0; j < 32; ++j) { b[4 + j] = static_cast<std::uint8_t>((j * 7) & 0xFF); }
    const auto y = decode(ggml::NVFP4, b, 64);
    const float expect_scale[4] = {1.0F, 2.0F, 0.0F, std::ldexp(1.0F, -9)};
    bool ok = true;
    for (int s = 0; s < 4; ++s) {
        for (int j = 0; j < 8; ++j) {
            const auto byte = b[4 + s * 8 + j];
            ok = ok && y[s * 16 + j] == kE2m1[byte & 15] * expect_scale[s];
            ok = ok && y[s * 16 + j + 8] == kE2m1[byte >> 4] * expect_scale[s];
        }
    }
    check(ok, "nvfp4: e2m1 * e4m3 per 16 values, element j and j+8 share a byte");
    std::uint8_t codes[32], words[4];
    ggml::nvfp4_block_natural(b.data(), codes, words);
    ok = words[0] == 0x38 && words[1] == 0x40 && words[2] == 0 && words[3] == 0x01;
    for (int i = 0; i < 32; ++i) {
        const float scale = kE4m3Values[words[(2 * i) / 16]];
        ok = ok && kE2m1[codes[i] & 15] * scale == y[2 * i];
        ok = ok && kE2m1[codes[i] >> 4] * scale == y[2 * i + 1];
    }
    check(ok, "nvfp4 repack: natural words reproduce every value");
}

void test_plain() {
    std::vector<std::uint8_t> b(4);
    put16(b, 0, 0x3F80); // bf16 1.0
    put16(b, 2, 0xC000); // bf16 -2.0
    const auto y = decode(ggml::BF16, b, 2);
    check(y[0] == 1.0F && y[1] == -2.0F, "bf16 values");
    put16(b, 0, 0x3C00);
    put16(b, 2, 0x8001); // f16 1.0, smallest negative subnormal
    const auto h = decode(ggml::F16, b, 2);
    check(h[0] == 1.0F && h[1] == -std::ldexp(1.0F, -24), "f16 values");
}

} // namespace

int main() {
    test_traits();
    test_q8_0();
    test_q4_0();
    test_q4_k();
    test_q6_k();
    test_iq4_nl();
    test_mxfp4();
    test_nvfp4();
    test_plain();
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ggml decoders: ok\n");
    return 0;
}
