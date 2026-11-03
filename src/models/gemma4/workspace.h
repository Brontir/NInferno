#pragma once

// The text target's workspace (Target::workspace_bytes) from what the bytes depend on: the
// geometry (Config), every projection's (format, rows, inputs, policy) and the routed experts'
// capacity. A start passes the bound weights' projections; the plan without the weights
// (runtime/engine/plan_dry_run.h set_plan_workspace_query) passes the same from the model's facts,
// so the two size the same arena (#302).

#include "models/gemma4/config.h"
#include "ninferno/ops/linear.h"
#include "ninferno/types.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninferno::models::gemma4 {

struct WorkspaceLinear {
    QType qtype                = QType::BF16;
    std::int32_t n             = 0; // output rows
    std::int32_t k             = 0; // inputs
    ops::LinearPolicy policy   = ops::LinearPolicy::A16Only;
    bool geglu                 = false; // the fused gate/up GeGLU's own query too
};

// The fixed reserve the target's arena keeps beside its rows and Ops.
inline constexpr std::size_t kTargetWorkspaceReserve = std::size_t(64) << 20;

// One call's activation rows over `columns` tokens and the head's logits over `head_columns`.
[[nodiscard]] std::size_t target_activation_bytes(const Config& c, std::int32_t columns,
                                                  std::int32_t head_columns);

// The widest Op's transient capacity in a call of up to `columns` tokens: the projections given,
// the attention of either kind over `max_context`, the Gemma MoE kernel, the head (its policy and
// A16 for the exact logits) and the sampler. `routed_experts`: a generic MoE layer's own capacity
// (moe_workspace_bytes over its weights; 0 when none or unknown).
[[nodiscard]] std::size_t target_op_bytes(const Config& c, std::span<const WorkspaceLinear> linears,
                                          const WorkspaceLinear& head, std::int32_t max_context,
                                          std::int32_t columns, std::int32_t head_columns,
                                          std::size_t routed_experts);

// A separate assistant's pass over `batch` rows beside the target (DrafterPass::workspace_bytes):
// its activations, the widest of its projections (each with its fused GeGLU where `geglu`), its
// head and its attention over the target's caches at `max_context`, and the pass's 16 MiB
// reserve. The plan without the weights passes the same from the assistant's config (#302).
[[nodiscard]] std::size_t drafter_pass_bytes(const DrafterConfig& dc,
                                             std::span<const WorkspaceLinear> projections,
                                             const WorkspaceLinear& head, std::int32_t max_context,
                                             std::int32_t batch);

} // namespace ninferno::models::gemma4
