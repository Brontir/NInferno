#pragma once

#include "core/tensor.h"
#include "ninferno/ops/gelu.h"

#include <cuda_runtime.h>

namespace ninferno::ops::detail {

void gelu_launch(Tensor& x, GeluMode mode, cudaStream_t stream);

} // namespace ninferno::ops::detail
