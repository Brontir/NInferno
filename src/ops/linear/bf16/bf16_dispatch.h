#pragma once

#include "core/weight.h"
#include "ninferno/ops/linear.h"
#include "ops/linear/bf16/bf16_launch.h"

#include <cuda_runtime.h>

namespace ninferno::ops::detail {

// Whether the shape-owned table registers this exact [N,K] (the fast path).
bool bf16_linear_registered(std::int32_t n, std::int32_t k);
Bf16Launch select_bf16_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t);
Bf16Launch select_bf16_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy);

void bf16_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   cudaStream_t stream);

} // namespace ninferno::ops::detail
