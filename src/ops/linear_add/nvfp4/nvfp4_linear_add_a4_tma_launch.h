#pragma once
#include "ops/linear/nvfp4/nvfp4_operands.h"

namespace ninferno::ops::detail {
void launch_nvfp4_a4_tma_linear_add(const Nvfp4A4Operands& p, __nv_bfloat16* residual,
                                    cudaStream_t stream);
} // namespace ninferno::ops::detail
