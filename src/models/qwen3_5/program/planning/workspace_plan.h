#pragma once

// The Qwen 3.5 start's workspace (WorkspacePlan) from what its bytes depend on, without the Model:
// the text geometry, the public token count and the prepared form of every projection the planner
// sizes. Each projection's bytes follow from its format, rows, inputs and policy and from the form
// its load gave it (one parent, a pair, a segmented bank; a dense FFN's split gate/up; the MTP
// layer's packed parent or rows; a fused or generic MoE), never from its address or its bytes.
// A start passes the forms it bound (qwen3_5_workspace_plan, over workspace_inputs below); a plan
// without the weights may pass the same forms prepared over described weights (#615), and the two
// size the same arena. This header and the planner's own source include no loader and no device
// context.

#include "models/qwen3_5/config.h"
#include "models/qwen3_5/execution/parameter_types.h"
#include "models/qwen3_5/program/planning/sequence_plan.h"

#include <cstdint>
#include <optional>

namespace ninferno::models::qwen3_5::detail {

// What the workspace planner reads; it borrows every part. The pointers a plan's features need
// must be set (build_workspace_plan refuses a plan that reads one left null):
// - `mtp` under plan.features.mtp();
// - `draft_config` and `draft` under plan.features.masked_draft();
// - `proposal` under either of them with ProposalHead::Optimized.
// The planner reads no Tensor of these forms (norms, biases, token ids): only the projections'
// Weight (qtype, n, k, layout) and policy, and the variants and optionals that pick their routes.
struct WorkspaceInputs {
    const TextConfig& config;
    std::int32_t public_tokens = 0; // the ids a request may carry or sample (the sampler's width)
    const execution::TextParameters& text;
    const execution::MtpParameters* mtp           = nullptr;
    const execution::ProposalParameters* proposal = nullptr;
    const DraftConfig* draft_config               = nullptr;
    const execution::DraftParameters* draft       = nullptr;
};

// The view of a start's own prepared forms: what qwen3_5_workspace_plan passes for its Parameters.
[[nodiscard]] inline WorkspaceInputs
workspace_inputs(const Config& config, std::int32_t public_tokens,
                 const execution::TextParameters& text,
                 const std::optional<execution::MtpParameters>& mtp,
                 const std::optional<execution::ProposalParameters>& proposal,
                 const std::optional<execution::DraftParameters>& draft) {
    return {.config        = config.text,
            .public_tokens = public_tokens,
            .text          = text,
            .mtp           = mtp ? &*mtp : nullptr,
            .proposal      = proposal ? &*proposal : nullptr,
            .draft_config  = config.draft ? &*config.draft : nullptr,
            .draft         = draft ? &*draft : nullptr};
}

// The arenas of `plan` over `inputs`: the text prefill (and the causal score), the ordinary round,
// under MTP its prefill and rounds, under a masked drafter its context appends and rounds, and
// their widest (general_capacity, capacity). A Vision start adds the encoder's region above
// general_capacity (qwen3_5_workspace_plan): `vision` is left empty here. No device: the capacity
// queries answer for this thread's device class (core/device_class.h, ScopedDeviceClass) and
// plan.multiprocessor_count, under the caller's route profile (a start under --activations exact
// calls this inside ops::ScopedInvariantRoutes, Qwen35Execution::workspace_plan).
[[nodiscard]] WorkspacePlan build_workspace_plan(const WorkspaceInputs& inputs,
                                                 const SequencePlanImpl& plan);

} // namespace ninferno::models::qwen3_5::detail
