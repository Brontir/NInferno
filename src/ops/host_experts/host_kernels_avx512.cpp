// AVX-512 row dots (host_kernels.h): the 16 canonical lanes in one register. Compiled with
// -mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma -ffp-contract=off; reached only when the CPU
// has them.
#include "ops/host_experts/host_kernels.h"

#include <immintrin.h>

#include <cstddef>

namespace ninferno::ops::host::detail {

namespace {

inline float fold(__m512 acc) {
    const __m256 r8 = _mm256_add_ps(_mm512_castps512_ps256(acc), _mm512_extractf32x8_ps(acc, 1));
    const __m128 r4 = _mm_add_ps(_mm256_castps256_ps128(r8), _mm256_extractf128_ps(r8, 1));
    const __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
    return _mm_cvtss_f32(_mm_add_ss(r2, _mm_shuffle_ps(r2, r2, 1)));
}

template <int C, class Scale>
void group(const std::uint8_t* codes, int chunks, Scale scale, const float* const* x,
           float* out) {
    const __m512 lut = _mm512_loadu_ps(e2m1_values());
    __m512 acc[C];
    for (int i = 0; i < C; ++i) { acc[i] = _mm512_setzero_ps(); }
    for (int c = 0; c < chunks; ++c) {
        const __m512i bytes = _mm512_cvtepu8_epi32(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + static_cast<std::ptrdiff_t>(c) * 16)));
        // permutexvar reads bits 0-3 of each index: the low nibble as it is.
        const __m512 lo = _mm512_permutexvar_ps(bytes, lut);
        const __m512 hi = _mm512_permutexvar_ps(_mm512_srli_epi32(bytes, 4), lut);
        const __m512 s  = scale(c);
        for (int i = 0; i < C; ++i) {
            const float* in = x[i] + static_cast<std::ptrdiff_t>(c) * kChunk;
            __m512 p        = _mm512_mul_ps(lo, _mm512_loadu_ps(in));
            p               = _mm512_fmadd_ps(hi, _mm512_loadu_ps(in + 16), p);
            acc[i]          = _mm512_fmadd_ps(p, s, acc[i]);
        }
    }
    for (int i = 0; i < C; ++i) { out[i] = fold(acc[i]); }
}

template <class Scale>
void row_dot(const std::uint8_t* codes, int chunks, Scale scale, const float* const* x, int columns,
             float* out) {
    int i = 0;
    for (; i + 8 <= columns; i += 8) { group<8>(codes, chunks, scale, x + i, out + i); }
    const int rest = columns - i;
    if (rest >= 4) {
        group<4>(codes, chunks, scale, x + i, out + i);
        i += 4;
    }
    switch (columns - i) {
    case 3:
        group<3>(codes, chunks, scale, x + i, out + i);
        break;
    case 2:
        group<2>(codes, chunks, scale, x + i, out + i);
        break;
    case 1:
        group<1>(codes, chunks, scale, x + i, out + i);
        break;
    default:
        break;
    }
}

} // namespace

void mxfp4_avx512(const std::uint8_t* codes, const std::uint8_t* scales, int chunks,
                  const float* const* x, int columns, float* out) {
    const float* e8m0 = e8m0_values();
    row_dot(
        codes, chunks, [&](int c) { return _mm512_set1_ps(e8m0[scales[c]]); }, x, columns, out);
}

void nvfp4_avx512(const std::uint8_t* codes, const std::uint8_t* scales, int chunks,
                  const float* const* x, int columns, float* out) {
    const float* e4m3 = e4m3_values();
    row_dot(
        codes, chunks,
        [&](int c) {
            const std::uint8_t* pair = scales + static_cast<std::ptrdiff_t>(c / 2) * 512 + (c % 2) * 2;
            return _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(e4m3[pair[0]])),
                                      _mm256_set1_ps(e4m3[pair[1]]), 1);
        },
        x, columns, out);
}

} // namespace ninferno::ops::host::detail
