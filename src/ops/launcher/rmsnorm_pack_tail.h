#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninferno::ops::detail {

void rmsnorm_pack_tail_launch(const Tensor& input, const Tensor& weight, Tensor& output,
                              cudaStream_t stream);

} // namespace ninferno::ops::detail
