#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninferno::ops::detail {

void prepare_ragged_prefix_launch(const Tensor& source, const Tensor& lanes, const Tensor& starts,
                                  const Tensor& ends, Tensor& destination, Tensor& positions,
                                  Tensor& counts, cudaStream_t stream);

} // namespace ninferno::ops::detail
