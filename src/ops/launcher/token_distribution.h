#pragma once

// ninferno::ops::detail - private launch prototypes for logprob_topk and gather_logprobs.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninferno::ops::detail {

void logprob_topk_launch(const Tensor& logits, std::int32_t valid_rows, Tensor& ids,
                         Tensor& logprobs, cudaStream_t stream);

void gather_logprobs_launch(const Tensor& logits, std::int32_t valid_rows, const Tensor& ids,
                            Tensor& out, cudaStream_t stream);

} // namespace ninferno::ops::detail
