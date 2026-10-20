#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninferno::ops::detail {

void vision_pos_embed_add_launch(const Tensor& table, const Tensor& indices, const Tensor& weights,
                                 Tensor& x, cudaStream_t stream);

} // namespace ninferno::ops::detail
