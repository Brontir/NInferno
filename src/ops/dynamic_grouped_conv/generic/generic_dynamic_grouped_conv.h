#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninferno::ops::detail {

// n BF16 [5120, W*B] (normalized residual), d BF16 [1280, W*B] (its kernel projection):
// prepared BF16 [5120, W*B] and finish_delta BF16 [320, 2, W*B].
void generic_dynamic_grouped_conv_prepare_launch(const Tensor& n, const Tensor& d,
                                                 const Tensor& base_kernel, Tensor& prepared,
                                                 Tensor& finish_delta, int width, int columns,
                                                 cudaStream_t stream);

// z BF16 [5120, W*B] (the projected input): residual += the convolution of z.
void generic_linear_dynamic_grouped_conv_add_launch(const Tensor& z, const Tensor& base_kernel,
                                                    const Tensor& finish_delta, Tensor& residual,
                                                    int width, int columns, cudaStream_t stream);

} // namespace ninferno::ops::detail
