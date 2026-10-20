#pragma once

// ninferno::ops::detail - private launch prototype for finite_check.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninferno::ops::detail {

void finite_check_launch(const Tensor& values, std::int32_t hidden, std::int32_t columns,
                         std::int32_t rows, const std::int32_t* valid_columns,
                         std::uint32_t* status, cudaStream_t stream);

} // namespace ninferno::ops::detail
