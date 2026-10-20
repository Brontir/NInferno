// Decode tables, instruction-set detection and the scalar row dots (host_kernels.h). Compiled
// with -ffp-contract=off: every multiply and add below is the operation it spells, and each fma
// is std::fma, so the scalar path computes what the vector paths do.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include "ops/host_experts/host_kernels.h"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninferno::ops::host {

namespace {

struct Tables {
    std::array<float, 16> e2m1{};
    std::array<float, 256> e8m0{};
    std::array<float, 256> e4m3{};
    Tables() {
        constexpr float kMagnitudes[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
        for (int c = 0; c < 16; ++c) {
            e2m1[static_cast<std::size_t>(c)] = (c & 8) != 0 ? -kMagnitudes[c & 7] : kMagnitudes[c & 7];
        }
        for (int e = 0; e < 255; ++e) {
            e8m0[static_cast<std::size_t>(e)] = std::ldexp(1.0F, e - 127);
        }
        e8m0[255] = std::numeric_limits<float>::quiet_NaN();
        for (int w = 0; w < 256; ++w) {
            const int exponent = (w >> 3) & 15;
            const int mantissa = w & 7;
            float magnitude    = 0.0F;
            if (exponent == 15 && mantissa == 7) {
                magnitude = std::numeric_limits<float>::quiet_NaN();
            } else if (exponent == 0) {
                magnitude = std::ldexp(static_cast<float>(mantissa), -9);
            } else {
                magnitude = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, exponent - 7);
            }
            e4m3[static_cast<std::size_t>(w)] = (w & 0x80) != 0 ? -magnitude : magnitude;
        }
    }
};

const Tables& tables() {
    static const Tables t;
    return t;
}

// The 16 lanes folded in the canonical tree.
float fold(const float* acc) {
    float r8[8];
    for (int i = 0; i < 8; ++i) { r8[i] = acc[i] + acc[i + 8]; }
    float r4[4];
    for (int i = 0; i < 4; ++i) { r4[i] = r8[i] + r8[i + 4]; }
    const float r20 = r4[0] + r4[2];
    const float r21 = r4[1] + r4[3];
    return r20 + r21;
}

template <class Scale>
void row_dot_scalar(const std::uint8_t* codes, int chunks, Scale scale, const float* const* x,
                    int columns, float* out) {
    const float* lut = tables().e2m1.data();
    float acc[kMaxColumns][16] = {};
    for (int c = 0; c < chunks; ++c) {
        const std::uint8_t* b = codes + static_cast<std::ptrdiff_t>(c) * 16;
        float s[16];
        scale(c, s);
        for (int i = 0; i < columns; ++i) {
            const float* even = x[i] + static_cast<std::ptrdiff_t>(c) * kChunk;
            const float* odd  = even + 16;
            for (int j = 0; j < 16; ++j) {
                const float lo = lut[b[j] & 15];
                const float hi = lut[b[j] >> 4];
                float p        = lo * even[j];
                p              = std::fma(hi, odd[j], p);
                acc[i][j]      = std::fma(p, s[j], acc[i][j]);
            }
        }
    }
    for (int i = 0; i < columns; ++i) { out[i] = fold(acc[i]); }
}

} // namespace

const float* e2m1_values() noexcept { return tables().e2m1.data(); }
const float* e8m0_values() noexcept { return tables().e8m0.data(); }
const float* e4m3_values() noexcept { return tables().e4m3.data(); }

void permute_input(const float* x, int k, float* permuted) noexcept {
    for (int c = 0; c < k; c += kChunk) {
        for (int j = 0; j < kChunk / 2; ++j) {
            permuted[c + j]              = x[c + 2 * j];
            permuted[c + kChunk / 2 + j] = x[c + 2 * j + 1];
        }
    }
}

namespace detail {

void mxfp4_scalar(const std::uint8_t* codes, const std::uint8_t* scales, int chunks,
                  const float* const* x, int columns, float* out) {
    const float* e8m0 = tables().e8m0.data();
    row_dot_scalar(
        codes, chunks,
        [&](int c, float* s) {
            const float v = e8m0[scales[c]];
            for (int j = 0; j < 16; ++j) { s[j] = v; }
        },
        x, columns, out);
}

void nvfp4_scalar(const std::uint8_t* codes, const std::uint8_t* scales, int chunks,
                  const float* const* x, int columns, float* out) {
    const float* e4m3 = tables().e4m3.data();
    row_dot_scalar(
        codes, chunks,
        [&](int c, float* s) {
            const std::uint8_t* pair = scales + static_cast<std::ptrdiff_t>(c / 2) * 512 + (c % 2) * 2;
            for (int j = 0; j < 8; ++j) {
                s[j]     = e4m3[pair[0]];
                s[j + 8] = e4m3[pair[1]];
            }
        },
        x, columns, out);
}

} // namespace detail

bool isa_available(Isa isa) noexcept {
    switch (isa) {
    case Isa::Scalar:
        return true;
    case Isa::Avx2:
#if defined(_WIN32) && defined(_M_X64)
        return IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE) != 0;
#elif defined(__x86_64__) || defined(_M_X64)
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
        return false;
#endif
    case Isa::Avx512:
#if defined(_WIN32) && defined(_M_X64)
        return IsProcessorFeaturePresent(PF_AVX512F_INSTRUCTIONS_AVAILABLE) != 0;
#elif defined(__x86_64__) || defined(_M_X64)
        return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
               __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
#else
        return false;
#endif
    }
    return false;
}

Isa detect_isa() noexcept {
    if (isa_available(Isa::Avx512)) { return Isa::Avx512; }
    if (isa_available(Isa::Avx2)) { return Isa::Avx2; }
    return Isa::Scalar;
}

const char* isa_name(Isa isa) noexcept {
    switch (isa) {
    case Isa::Scalar:
        return "scalar";
    case Isa::Avx2:
        return "AVX2";
    case Isa::Avx512:
        return "AVX-512";
    }
    return "?";
}

RowKernels row_kernels(Isa isa) {
    if (!isa_available(isa)) {
        throw std::invalid_argument(std::string("host experts: this CPU has no ") + isa_name(isa));
    }
    switch (isa) {
    case Isa::Scalar:
        return {detail::mxfp4_scalar, detail::nvfp4_scalar};
    case Isa::Avx2:
        return {detail::mxfp4_avx2, detail::nvfp4_avx2};
    case Isa::Avx512:
        return {detail::mxfp4_avx512, detail::nvfp4_avx512};
    }
    throw std::invalid_argument("host experts: unknown instruction set");
}

} // namespace ninferno::ops::host
