#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninferno::ops::detail {

void layer_norm_launch(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps,
                       Tensor& out, cudaStream_t stream);

} // namespace ninferno::ops::detail
