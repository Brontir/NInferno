#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninferno/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninferno::ops::detail {

// The (gate+up rows, k) problems this plan registers; ops/wrapper/linear_swiglu.cpp asks it.
bool fp8_linear_swiglu_registered(std::int32_t gate_up_rows, std::int32_t k) noexcept;
[[nodiscard]] std::size_t fp8_linear_swiglu_partial_capacity_bytes(std::int32_t max_tokens);

[[nodiscard]] std::size_t fp8_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                                     std::int32_t min_tokens,
                                                                     std::int32_t max_tokens);

void fp8_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream);
void fp8_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream);
void fp8_linear_swiglu_matrix_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void fp8_linear_swiglu_a8_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                 WorkspaceArena& workspace, cudaStream_t stream);

void fp8_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream);

} // namespace ninferno::ops::detail
