#pragma once

// Row dot products of block-scaled FP4 expert weights on the CPU (MoE expert offload's decode
// path, src/ops/host_experts/host_experts.h). One stored row against up to kMaxColumns inputs.
//
// The arithmetic is canonical, so every instruction set gives the same bytes. A row is read in
// chunks of 32 values: 16 code bytes, byte j holding value 2j in its low nibble and value 2j + 1
// in its high one. The input is permuted to match (permute_input): per chunk its 16 even values,
// then its 16 odd values. Sixteen FP32 lanes accumulate; lane j of chunk c computes
//
//     p     = lo_j * even_j              (lo_j, hi_j: the E2M1 values of byte j)
//     p     = fma(hi_j, odd_j, p)
//     acc_j = fma(p, scale_j, acc_j)
//
// with scale_j the chunk's scale (MXFP4: its E8M0 exponent; NVFP4: the E4M3 scale of the value's
// 16-value group, lanes 0-7 the first group and 8-15 the second). The row's dot is the lanes
// folded in one fixed tree: r8_i = acc_i + acc_{i+8}, r4_i = r8_i + r8_{i+4}, r2_i = r4_i +
// r4_{i+2}, dot = r2_0 + r2_1. AVX2 carries the 16 lanes as two 8-lane halves, AVX-512 as one
// register, the scalar path as an array: the same operations in the same order. NVFP4's global
// divisor is not applied here.

#include <cstdint>

namespace ninferno::ops::host {

enum class Isa : std::uint8_t { Scalar, Avx2, Avx512 };

// The widest instruction set this CPU and this build both support.
[[nodiscard]] Isa detect_isa() noexcept;
[[nodiscard]] const char* isa_name(Isa isa) noexcept;
[[nodiscard]] bool isa_available(Isa isa) noexcept;

inline constexpr int kChunk      = 32;
inline constexpr int kMaxColumns = 16;

// Decode tables shared by every instruction set: E2M1 (16 codes, bit 3 the sign), E8M0 (2^(e-127),
// 255 NaN) and E4M3FN (bias 7, no infinities, 0x7f/0xff NaN).
[[nodiscard]] const float* e2m1_values() noexcept;
[[nodiscard]] const float* e8m0_values() noexcept;
[[nodiscard]] const float* e4m3_values() noexcept;

// x (K values) -> the chunk permutation the kernels read. K is a multiple of 32.
void permute_input(const float* x, int k, float* permuted) noexcept;
// The permuted position of value j.
[[nodiscard]] constexpr int permuted_index(int j) noexcept {
    const int in_chunk = j % kChunk;
    return j - in_chunk + (in_chunk % 2 == 0 ? in_chunk / 2 : kChunk / 2 + in_chunk / 2);
}

// One MXFP4 row: codes (K/2 bytes), exponents (K/32 bytes, one per chunk).
// NVFP4 row: codes (K/2 bytes); the scales of chunk c are the two bytes at
// scale_row + (c / 2) * 512 + (c % 2) * 2 (the 128x4 block-scale swizzle, see nvfp4_scale_row).
// x[i]: column i's permuted input (K floats); out[i]: its dot. 1 <= columns <= kMaxColumns.
using RowDot = void (*)(const std::uint8_t* codes, const std::uint8_t* scales, int chunks,
                        const float* const* x, int columns, float* out);

struct RowKernels {
    RowDot mxfp4 = nullptr;
    RowDot nvfp4 = nullptr;
};

[[nodiscard]] RowKernels row_kernels(Isa isa);

// The first scale byte of `row` of an NVFP4 plane [N, K] in QuantLayout::BlockScaleK16M128x4.
[[nodiscard]] inline const std::uint8_t* nvfp4_scale_row(const std::uint8_t* scales,
                                                         std::int64_t row, int k) noexcept {
    const std::int64_t tile  = row / 128;
    const std::int64_t inner = row % 128;
    return scales + tile * (k / 64) * 512 + (inner % 32) * 16 + (inner / 32) * 4;
}

namespace detail {
void mxfp4_scalar(const std::uint8_t*, const std::uint8_t*, int, const float* const*, int, float*);
void nvfp4_scalar(const std::uint8_t*, const std::uint8_t*, int, const float* const*, int, float*);
void mxfp4_avx2(const std::uint8_t*, const std::uint8_t*, int, const float* const*, int, float*);
void nvfp4_avx2(const std::uint8_t*, const std::uint8_t*, int, const float* const*, int, float*);
void mxfp4_avx512(const std::uint8_t*, const std::uint8_t*, int, const float* const*, int, float*);
void nvfp4_avx512(const std::uint8_t*, const std::uint8_t*, int, const float* const*, int, float*);
} // namespace detail

} // namespace ninferno::ops::host
