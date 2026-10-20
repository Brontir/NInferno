#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninferno::ops::detail {

using Bf16Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

} // namespace ninferno::ops::detail
