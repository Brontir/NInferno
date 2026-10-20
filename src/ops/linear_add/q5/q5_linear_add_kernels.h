#pragma once
#include "core/weight.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninferno::ops::detail {
void q5_linear_add_split2_exact_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r16_t8_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r16_t8_c4_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r16_t16_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r16_t24_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r32_t32_w4_s2_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r32_t24_pairwise_launch(const Tensor&, const Weight&, Tensor&,
                                                  cudaStream_t);
void q5_linear_add_sliced_r32_t32_w4_s1_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r32_t32_w2_s2_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r32_t64_w2_s1_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_mma_r32_t32_k128_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_mma_r32_t128_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_mma_r64_t128_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
bool q5_linear_add_small_t_mma_operands(const Tensor& x, const Tensor& residual_out) noexcept;
void q5_linear_add_small_t_mma_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
} // namespace ninferno::ops::detail
