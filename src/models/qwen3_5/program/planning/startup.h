#pragma once
#include "models/qwen3_5/program/internal.h"

#include "models/qwen3_5/program/planning/sequence_bytes.h"
#include "models/qwen3_5/program/planning/sequence_plan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>

namespace ninferno::models::qwen3_5::detail {

struct SequencePlannerImpl {
    SequencePlanningInputs inputs;
    runtime::SequenceCapacityCurve curve;
    std::unique_ptr<SequencePlanImpl> minimum;
};

[[nodiscard]] std::unique_ptr<qwen3_5::detail::SequencePlannerImpl>
make_sequence_planner_impl(const ModelExecution& model, DeviceContext& device,
                           const EngineOptions& options, bool graph_memory_held = false);

// Planning pieces every family's Program shares.
struct ModelTraits;

// The Engine options any family's Program checks: the context against the model's position
// capacity, the chunk, the concurrency, the KV capacity policy and the speculative depths (MTP up
// to the family's `maximum_mtp_draft_tokens`, masked drafts up to 15). Copy rounds
// (SpeculativeOptions::ngram_draft_tokens) need a family whose MTP round runs the copy window
// (`mtp_copy_rounds`).
void validate_program_options(const EngineOptions& options, std::uint64_t max_positions,
                              std::uint32_t maximum_mtp_draft_tokens, bool mtp_copy_rounds = false);

// Whether `steps` (SpeculativeOptions::step_down) descend strictly below `depth`, each at least 1,
// with nonnegative floors (empty: always).
[[nodiscard]] inline bool speculation_steps_descend(std::uint32_t depth,
                                                    const std::vector<SpeculationStep>& steps) {
    for (const auto& step : steps) {
        if (step.draft_tokens == 0 || step.draft_tokens >= depth || !(step.acceptance_floor >= 0.0)) {
            return false;
        }
        depth = step.draft_tokens;
    }
    return true;
}

[[nodiscard]] std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_5::detail::SequencePlannerImpl> planner,
                            std::uint32_t main_page_groups);

} // namespace ninferno::models::qwen3_5::detail
