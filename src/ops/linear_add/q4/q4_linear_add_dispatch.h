#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninferno::ops::detail {

using Q4LinearAddLaunch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// The (rows, k) problems this plan registers; ops/wrapper/linear_add.cpp asks it.
bool q4_linear_add_registered(std::int32_t rows, std::int32_t k) noexcept;
Q4LinearAddLaunch select_q4_linear_add(std::int32_t rows, std::int32_t k, std::int32_t tokens);

} // namespace ninferno::ops::detail
