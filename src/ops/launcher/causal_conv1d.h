#pragma once

// ninferno::ops::detail - private launch prototypes for causal_conv1d.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninferno::ops::detail {

inline constexpr std::int32_t kCausalConvParallelMaxTokens = 32;

void causal_conv1d_prefill_launch(const Tensor& x, const Tensor& weight,
                                  const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                                  cudaStream_t stream);
void causal_conv1d_prefill_split_launch(const Tensor& x, const Tensor& weight,
                                        const Tensor& conv_state_in, Tensor& conv_state_out,
                                        Tensor& out0, Tensor& out1, Tensor& out2,
                                        cudaStream_t stream);
void causal_conv1d_smallt_launch(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                                 Tensor& conv_state_out, Tensor& out, cudaStream_t stream);
void causal_conv1d_smallt_split_launch(const Tensor& x, const Tensor& weight,
                                       const Tensor& conv_state_in, Tensor& conv_state_out,
                                       Tensor& out0, Tensor& out1, Tensor& out2,
                                       cudaStream_t stream);
void causal_conv1d_decode_launch(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                                 Tensor& conv_state_out, Tensor& out, cudaStream_t stream);
void causal_conv1d_snapshot_launch(const Tensor& x, const Tensor& weight, Tensor& conv_states,
                                   const Tensor& valid_columns, const Tensor& initial_state_slots,
                                   const Tensor& snapshot_base_slots, Tensor& out,
                                   cudaStream_t stream);

} // namespace ninferno::ops::detail
