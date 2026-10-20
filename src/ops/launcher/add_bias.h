#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninferno::ops::detail {

void add_bias_launch(const Tensor& bias, Tensor& x, cudaStream_t stream);

} // namespace ninferno::ops::detail
