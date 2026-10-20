#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninferno::ops::detail {

void logit_softcap_launch(Tensor& logits, float cap, float scale, cudaStream_t stream);

} // namespace ninferno::ops::detail
