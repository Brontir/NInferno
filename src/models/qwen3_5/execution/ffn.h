#pragma once

#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"

namespace ninferno::models::qwen3_5::execution {

[[nodiscard]] std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                              std::int32_t last, bool mtp = false);
// `experts`: where a sparse MoE reads its routed experts (MoE expert offload).
void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, DeviceExecutionView execution,
         const ExpertPlacement& experts, bool mtp = false);

} // namespace ninferno::models::qwen3_5::execution
