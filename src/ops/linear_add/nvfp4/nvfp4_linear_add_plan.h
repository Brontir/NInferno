#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninferno/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninferno::ops::detail {

// The (rows, k) problems this plan registers; ops/wrapper/linear_add.cpp asks it.
// Defined here, not in the plan's TU: a build without the NVFP4 family (Ada, Ampere) drops
// that TU, and the wrapper still asks.
[[nodiscard]] inline bool nvfp4_linear_add_registered(std::int32_t rows, std::int32_t k) noexcept {
    return rows == 5120 && (k == 6144 || k == 17408);
}
[[nodiscard]] std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                                    std::int32_t input_rows,
                                                                    LinearPolicy policy,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

void nvfp4_linear_add_a16_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                 cudaStream_t stream);

void nvfp4_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream);

void nvfp4_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                     cudaStream_t stream);

// LinearPolicy::A16Only at any T: tensor-core A16, token invariant (nvfp4_a16_exact.cuh).
void nvfp4_linear_add_exact_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                   cudaStream_t stream);

void nvfp4_linear_add_a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                Nvfp4A4Workspace workspace, cudaStream_t stream);

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninferno::ops::detail
