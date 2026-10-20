#pragma once

// ninferno::ops::detail - private launch prototype for rmsnorm.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninferno::ops::detail {

// heads, groups: the head regroup of the output (groups 0: none; launch_rmsnorm's warp route).
void rmsnorm_launch(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                    const Tensor* z, Tensor& out, std::int32_t multiprocessor_count,
                    std::int32_t heads, std::int32_t groups, cudaStream_t stream);

} // namespace ninferno::ops::detail
