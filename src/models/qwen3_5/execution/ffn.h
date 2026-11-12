#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "models/qwen3_5/execution/parameter_types.h"

namespace ninferno::models::qwen3_5::execution {

[[nodiscard]] std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                              std::int32_t last, bool mtp = false);
void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, DeviceExecutionView execution,
         bool mtp = false);

} // namespace ninferno::models::qwen3_5::execution
