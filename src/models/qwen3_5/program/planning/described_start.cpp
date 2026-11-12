#include "models/qwen3_5/program/planning/described_start.h"

#include "models/qwen3_5/execution/prepare.h"
#include "models/qwen3_5/program/qwen3_5_execution.h"

namespace ninferno::models::qwen3_5::detail {

WorkspacePlan DescribedStart::workspace_plan(std::int32_t public_tokens,
                                             const SequencePlanImpl& plan) const {
    return prepared_workspace_plan(model->config(), public_tokens, forms, plan);
}

std::unique_ptr<const DescribedStart> describe_start(const artifact::Directory& directory,
                                                     const LoadOptions& options) {
    auto out   = std::make_unique<DescribedStart>();
    out->model = describe_model(directory, options);
    // A plan asks each MoE layer's path as the start does and leaves the start's ledger alone.
    out->forms = execution::prepare_parameters(*out->model, false);
    return out;
}

} // namespace ninferno::models::qwen3_5::detail
