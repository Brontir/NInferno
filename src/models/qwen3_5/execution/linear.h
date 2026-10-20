#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/linear_add.h"
#include "ninferno/ops/linear_swiglu.h"

namespace ninferno::models::qwen3_5::execution {

inline void project(const Tensor& input, const LinearParameters& p, Tensor& output,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    ops::linear(input, p.weight, output, p.policy, workspace, stream);
}

inline void project_add(const Tensor& input, const LinearParameters& p, Tensor& residual,
                        WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    ops::linear_add(input, p.weight, residual, p.policy, workspace, stream);
}

inline void project_swiglu(const Tensor& input, const LinearParameters& p, Tensor& output,
                           WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    ops::linear_swiglu(input, p.weight, output, p.policy, workspace, stream);
}

} // namespace ninferno::models::qwen3_5::execution
