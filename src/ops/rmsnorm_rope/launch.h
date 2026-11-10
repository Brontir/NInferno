#pragma once

#include "core/device_class.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninferno::ops::detail {

// The text form: one warp per head, three heads per block (launch.cu). Its grid is
// tokens x ceil((Q + K heads) / 3) blocks of three warps.
inline constexpr int kTextHeadsPerBlock = 3;

// Resident blocks an SM holds of the widest text kernel instance on the current device (the
// compiled kernel's occupancy: registers and shared memory included).
int text_kernel_resident_ctas();

// The launch-amortisation law with an explicit chain excess in SM cycles
// (rmsnorm_rope_text_fused_max_tokens passes the pending Fitted constant once calibrated): 0 where
// the generation has no launch node, the geometry is not instantiated or the excess is not
// positive.
std::int32_t text_fused_max_tokens_with_excess(const DeviceClass& device, std::int32_t query_heads,
                                               std::int32_t key_heads, double excess);

void rmsnorm_rope_pair_launch(const Tensor& positions, const Tensor& q_norm_weight,
                              const Tensor& k_norm_weight, Tensor& q, Tensor& k,
                              std::int32_t tokens, cudaStream_t stream);

void rmsnorm_rope_single_launch(const Tensor& positions, const Tensor& norm_weight, Tensor& x,
                                std::int32_t tokens, cudaStream_t stream);

void rmsnorm_rope_text_launch(const Tensor& positions, const Tensor& q_norm_weight,
                              const Tensor& k_norm_weight, const Tensor& q_in, const Tensor& k_in,
                              Tensor& q_out, Tensor& k_out, std::int32_t tokens,
                              cudaStream_t stream);

} // namespace ninferno::ops::detail
