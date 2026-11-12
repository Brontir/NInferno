#pragma once

// The forms a Qwen 3.5 start prepares, over an artifact's directory without its payload: what the
// plan without the weights (--plan, runtime/engine/plan_workspace.cpp) sizes the workspace over
// (#615). The model is bound as plan_load binds it (load.h, describe_model), its forms prepared by
// the start's own function (execution/prepare.h, prepare_parameters), and the arenas laid out by
// the start's planner (prepared_workspace_plan): the start's three steps, the payload left unread.

#include "models/load_options.h"
#include "models/qwen3_5/execution/parameter_types.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/planning/sequence_plan.h"

#include <cstdint>
#include <memory>

namespace ninferno::artifact {
struct Directory;
} // namespace ninferno::artifact

namespace ninferno::models::qwen3_5::detail {

struct DescribedStart {
    std::unique_ptr<DescribedModel> model;
    execution::PreparedParameters forms; // their views point into `model`

    // The arenas a start's planner sizes for `plan` over these forms (qwen3_5_workspace_plan):
    // the capacity queries answer for this thread's device class, the plan's multiprocessor count
    // and the caller's route profile, as a start's do.
    [[nodiscard]] WorkspacePlan workspace_plan(std::int32_t public_tokens,
                                               const SequencePlanImpl& plan) const;
};

// Throws where describe_model does (plan_load would refuse the directory for `options`, or a
// binding reads the payload) and where a form refuses the bound weights (a start refuses them
// too).
[[nodiscard]] std::unique_ptr<const DescribedStart>
describe_start(const artifact::Directory& directory, const LoadOptions& options);

} // namespace ninferno::models::qwen3_5::detail
