#pragma once

// ninferno::ops::detail - private launch prototype for rope. Included by the wrapper
// and defined by the CUDA launcher.

#include "core/device.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninferno::ops::detail {

void rope_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                 DeviceExecutionView execution);

void rope_single_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                        DeviceExecutionView execution);

// rope_scaled: the generic kernel over the caller's frequencies; q or k may be null.
void rope_scaled_launch(const Tensor& positions, int rotary_dim, const Tensor& inverse_frequencies,
                        float attention_scale, Tensor* q, Tensor* k, cudaStream_t stream);

} // namespace ninferno::ops::detail
