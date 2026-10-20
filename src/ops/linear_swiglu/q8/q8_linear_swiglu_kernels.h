#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninferno::ops::detail {

// The widest T of the split-K exact-T launcher (one kernel instance per T = 2..48).
inline constexpr std::int32_t kQ8LinearSwiGluLastExactT = 48;

void q8_linear_swiglu_decode_pair_r16_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             cudaStream_t stream);
void q8_linear_swiglu_splitk_exact_t_launch(const Tensor& x, const Weight& w, Tensor& out,
                                            cudaStream_t stream);
void q8_linear_swiglu_mma_r32_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void q8_linear_swiglu_mma_r32_c80_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void q8_linear_swiglu_mma_r32_c96_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void q8_linear_swiglu_mma_r32_c128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void q8_linear_swiglu_mma_r64_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void q8_linear_swiglu_mma_r64_c96_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void q8_linear_swiglu_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void q8_linear_swiglu_mma_r128_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void q8_linear_swiglu_mma_r128_c80_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void q8_dflash2_linear_swiglu_small_t_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             cudaStream_t stream);

void q8_dflash2_linear_swiglu_mma_r32_c64_k128_launch(const Tensor&, const Weight&, Tensor&,
                                                      cudaStream_t);

void q8_dflash2_linear_swiglu_mma_r64_c64_k128_launch(const Tensor&, const Weight&, Tensor&,
                                                      cudaStream_t);

void q8_dflash2_linear_swiglu_mma_r64_c80_k128_launch(const Tensor&, const Weight&, Tensor&,
                                                      cudaStream_t);

void q8_dflash2_linear_swiglu_mma_r64_c96_k128_launch(const Tensor&, const Weight&, Tensor&,
                                                      cudaStream_t);

} // namespace ninferno::ops::detail
