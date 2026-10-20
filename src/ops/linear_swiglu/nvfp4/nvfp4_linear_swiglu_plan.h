#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninferno/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninferno::ops::detail {

// The (gate+up rows, k) problems this plan registers (the linear registry's N34816 K5120);
// ops/wrapper/linear_swiglu.cpp asks it.
// Defined here, not in the plan's TU: a build without the NVFP4 family (Ada, Ampere) drops
// that TU, and the wrapper still asks.
[[nodiscard]] inline bool nvfp4_linear_swiglu_registered(std::int32_t gate_up_rows,
                                                         std::int32_t k) noexcept {
    return gate_up_rows == 34816 && k == 5120;
}
[[nodiscard]] std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                                       std::int32_t min_tokens,
                                                                       std::int32_t max_tokens);

void nvfp4_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream);
void nvfp4_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
// LinearPolicy::A16Only at any T: tensor-core A16 with SiLU in the epilogue, token invariant
// (nvfp4_a16_exact.cuh).
void nvfp4_linear_swiglu_exact_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream);
void nvfp4_linear_swiglu_a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                   WorkspaceArena& workspace, cudaStream_t stream);

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, WorkspaceArena& workspace,
                                  cudaStream_t stream);

} // namespace ninferno::ops::detail
