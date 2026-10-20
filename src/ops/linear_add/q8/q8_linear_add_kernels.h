#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>

namespace ninferno::ops::detail {

// The split-K exact-T launcher: one kernel instance per T in this band.
inline constexpr std::int32_t kQ8LinearAddFirstExactCols = 2;
inline constexpr std::int32_t kQ8LinearAddLastExactCols  = 48;

// The medium split-K launcher (T = 49..128) picks its tile by band: band i serves the widths
// above kQ8LinearAddMediumBandLast[i - 1] through kQ8LinearAddMediumBandLast[i].
inline constexpr std::int32_t kQ8LinearAddMediumFirstCols = kQ8LinearAddLastExactCols + 1;
inline constexpr std::array<std::int32_t, 9> kQ8LinearAddMediumBandLast{64,  65,  72,  80, 96,
                                                                        112, 120, 125, 128};

// The medium launcher's band of `cols` (kQ8LinearAddMediumFirstCols..128), -1 outside it.
[[nodiscard]] constexpr int q8_linear_add_medium_band(std::int32_t cols) noexcept {
    if (cols < kQ8LinearAddMediumFirstCols) { return -1; }
    for (std::size_t band = 0; band < kQ8LinearAddMediumBandLast.size(); ++band) {
        if (cols <= kQ8LinearAddMediumBandLast[band]) { return static_cast<int>(band); }
    }
    return -1;
}

// The prefill-profile small-T launcher (17..128 columns): the 16-row tile through this width, the
// 32-row one above; both tiles are this many columns wide.
inline constexpr std::int32_t kQ8LinearAddPrefillSmallR16MaxCols = 64;
inline constexpr std::int32_t kQ8LinearAddPrefillSmallTileCols   = 32;

void q8_linear_add_decode_k4096_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       int rows_per_cta, cudaStream_t stream);
void q8_linear_add_decode_r4_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                    cudaStream_t stream);
void q8_linear_add_decode_r8_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                    cudaStream_t stream);
void q8_linear_add_decode_r16_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                     cudaStream_t stream);
void q8_linear_add_simt_r8_c4_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                     cudaStream_t stream);
void q8_linear_add_simt_r8_c8_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                     cudaStream_t stream);
void q8_linear_add_splitk_mma_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                     cudaStream_t stream);
void q8_linear_add_medium_splitk_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        cudaStream_t stream);
void q8_linear_add_mma_r32_c128_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);
void q8_linear_add_mma_r32_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q8_linear_add_mma_r32_c80_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q8_linear_add_mma_r32_c96_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q8_linear_add_mma_r48_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q8_linear_add_mma_r48_c96_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q8_linear_add_mma_r48_c112_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);
void q8_linear_add_mma_r48_c128_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);
void q8_linear_add_mma_r64_c96_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q8_linear_add_mma_r64_c112_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);
void q8_linear_add_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);
void q8_linear_add_mma_r128_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);
void q8_linear_add_mma_r128_c80_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);

// Prefill profile, 17..128 columns: the MMA contraction on a 16-row tile chosen by T.
void q8_linear_add_mma_prefill_small_t_launch(const Tensor& x, const Weight& w,
                                              Tensor& residual_out, cudaStream_t stream);

void q8_linear_add_splitk_capacity_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                          cudaStream_t stream);

void q8_linear_add_grouped_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                  cudaStream_t stream);

} // namespace ninferno::ops::detail
