// AVX2 row dots (host_kernels.h): the 16 canonical lanes as two 8-lane halves, A = lanes 0-7 and
// B = lanes 8-15. Compiled with -mavx2 -mfma -ffp-contract=off; reached only when the CPU has both.
#include "ops/host_experts/host_kernels.h"

#include <immintrin.h>

#include <cstddef>

namespace ninferno::ops::host::detail {

namespace {

// Eight E2M1 codes (bits 0-3 of each lane) -> values: the magnitude from bits 0-2, the sign bit 3.
inline __m256 decode8(__m256i codes, __m256 magnitudes) {
    const __m256 magnitude = _mm256_permutevar8x32_ps(magnitudes, codes);
    const __m256i sign     = _mm256_slli_epi32(_mm256_and_si256(codes, _mm256_set1_epi32(8)), 28);
    return _mm256_xor_ps(magnitude, _mm256_castsi256_ps(sign));
}

inline float fold(__m256 a, __m256 b) {
    const __m256 r8 = _mm256_add_ps(a, b);
    const __m128 r4 = _mm_add_ps(_mm256_castps256_ps128(r8), _mm256_extractf128_ps(r8, 1));
    const __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
    return _mm_cvtss_f32(_mm_add_ss(r2, _mm_shuffle_ps(r2, r2, 1)));
}

template <int C, class Scale>
void group(const std::uint8_t* codes, int chunks, Scale scale, const float* const* x,
           float* out) {
    const __m256 magnitudes = _mm256_setr_ps(0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F);
    const __m256i low_mask  = _mm256_set1_epi32(15);
    __m256 acc_a[C], acc_b[C];
    for (int i = 0; i < C; ++i) {
        acc_a[i] = _mm256_setzero_ps();
        acc_b[i] = _mm256_setzero_ps();
    }
    for (int c = 0; c < chunks; ++c) {
        const std::uint8_t* b = codes + static_cast<std::ptrdiff_t>(c) * 16;
        const __m256i bytes_a =
            _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(b)));
        const __m256i bytes_b =
            _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + 8)));
        const __m256 lo_a = decode8(_mm256_and_si256(bytes_a, low_mask), magnitudes);
        const __m256 lo_b = decode8(_mm256_and_si256(bytes_b, low_mask), magnitudes);
        const __m256 hi_a = decode8(_mm256_srli_epi32(bytes_a, 4), magnitudes);
        const __m256 hi_b = decode8(_mm256_srli_epi32(bytes_b, 4), magnitudes);
        __m256 s_a, s_b;
        scale(c, s_a, s_b);
        for (int i = 0; i < C; ++i) {
            const float* in = x[i] + static_cast<std::ptrdiff_t>(c) * kChunk;
            __m256 p_a      = _mm256_mul_ps(lo_a, _mm256_loadu_ps(in));
            __m256 p_b      = _mm256_mul_ps(lo_b, _mm256_loadu_ps(in + 8));
            p_a             = _mm256_fmadd_ps(hi_a, _mm256_loadu_ps(in + 16), p_a);
            p_b             = _mm256_fmadd_ps(hi_b, _mm256_loadu_ps(in + 24), p_b);
            acc_a[i]        = _mm256_fmadd_ps(p_a, s_a, acc_a[i]);
            acc_b[i]        = _mm256_fmadd_ps(p_b, s_b, acc_b[i]);
        }
    }
    for (int i = 0; i < C; ++i) { out[i] = fold(acc_a[i], acc_b[i]); }
}

template <class Scale>
void row_dot(const std::uint8_t* codes, int chunks, Scale scale, const float* const* x, int columns,
             float* out) {
    int i = 0;
    for (; i + 4 <= columns; i += 4) { group<4>(codes, chunks, scale, x + i, out + i); }
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

void mxfp4_avx2(const std::uint8_t* codes, const std::uint8_t* scales, int chunks,
                const float* const* x, int columns, float* out) {
    const float* e8m0 = e8m0_values();
    row_dot(
        codes, chunks,
        [&](int c, __m256& a, __m256& b) {
            a = _mm256_set1_ps(e8m0[scales[c]]);
            b = a;
        },
        x, columns, out);
}

void nvfp4_avx2(const std::uint8_t* codes, const std::uint8_t* scales, int chunks,
                const float* const* x, int columns, float* out) {
    const float* e4m3 = e4m3_values();
    row_dot(
        codes, chunks,
        [&](int c, __m256& a, __m256& b) {
            const std::uint8_t* pair = scales + static_cast<std::ptrdiff_t>(c / 2) * 512 + (c % 2) * 2;
            a                        = _mm256_set1_ps(e4m3[pair[0]]);
            b                        = _mm256_set1_ps(e4m3[pair[1]]);
        },
        x, columns, out);
}

} // namespace ninferno::ops::host::detail
