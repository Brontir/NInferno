#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninferno::ops::detail {

// T in [1,32]: Q5 gate/value and Q4 query/key through the small-T MMA kernels.
void q4_q5_attn_input_small_t_mma_launch(const Tensor& x, const Weight& query_key_weight,
                                         const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                         Tensor& k, Tensor& v, cudaStream_t stream);

void q4_q5_attn_input_small_t_launch(const Tensor& x, const Weight& query_key_weight,
                                     const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream);

// Both parents on the sliced-K MMA kernels at T in [1,32].
void q4_q5_attn_input_sliced_launch(const Tensor& x, const Weight& qk_weight,
                                    const Weight& gv_weight, Tensor& q, Tensor& gate, Tensor& k,
                                    Tensor& v, int variant, cudaStream_t stream);

void q4_q5_attn_input_mixed_r32_c32_s2_launch(const Tensor& x, const Weight& w0, const Weight& w1,
                                              Tensor& q, Tensor& g, Tensor& k, Tensor& v,
                                              cudaStream_t stream);
void q4_q5_attn_input_mixed_r32_c64_s3_launch(const Tensor& x, const Weight& w0, const Weight& w1,
                                              Tensor& q, Tensor& g, Tensor& k, Tensor& v,
                                              cudaStream_t stream);
void q4_q5_attn_input_pair_r32_c64_s3_launch(const Tensor& x, const Weight& w0, const Weight& w1,
                                             Tensor& q, Tensor& g, Tensor& k, Tensor& v,
                                             cudaStream_t stream);
void q4_q5_attn_input_mixed_r64_c128_s2_launch(const Tensor& x, const Weight& w0, const Weight& w1,
                                               Tensor& q, Tensor& g, Tensor& k, Tensor& v,
                                               cudaStream_t stream);

void q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(const Tensor& x, const Weight& query_key_weight,
                                                    const Weight& gate_value_weight, Tensor& q,
                                                    Tensor& gate, Tensor& k, Tensor& v,
                                                    cudaStream_t stream);

} // namespace ninferno::ops::detail
