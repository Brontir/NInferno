#pragma once

// The byte arithmetic of a sequence plan that needs no bound weights, shared by a start and the
// plan without the weights (runtime/engine/plan_dry_run.h):
//
//   sequence_planning_inputs   the planner's inputs from the Engine options (the copy window
//                              resolved, the launch windows, the load bounds' batches, the
//                              codecs);
//   sequence_page_range        the Main KV page groups a plan of a context and concurrency spans;
//   sequence_plan_shell        a candidate plan at a page count, before its layouts;
//   plan_persistent_layout     every family's persistent state (ModelExecution::persistent_layout
//                              lays its model's StateShape out here): the paged KV of the full
//                              layers and of the MTP layer, the StateImage pool (recurrent states,
//                              sliding rings, a DFlash drafter's ring), the replay records, a
//                              DFlash drafter's paged KV and features, the Program's buffers;
//   plan_graph_allowance       the CUDA graph allowance of the plan's launch configurations.
//
// What is not here: the workspace (ModelExecution::workspace_plan), whose transient Op scratch the
// bound weights' routes size. No CUDA, no weights.

#include "models/qwen3_5/program/planning/sequence_plan.h"
#include "models/qwen3_5/program/planning/state_shape.h"

#include <cstdint>
#include <memory>
#include <string_view>

namespace ninferno::models::qwen3_5::detail {

// The Program buffers' share of a family's traits (ModelTraits: hidden, output_rows,
// public_tokens).
struct ProgramBufferShape {
    std::int32_t hidden        = 0;
    std::int32_t output_rows   = 0;
    std::int32_t public_tokens = 0;
};

// The Program's own persistent buffers, planned after a family's state: the round state, the
// prefill hidden, the causal-score staging and the sampling state. Finishes the layout (out.bytes).
void plan_program_buffers(LayoutBuilder& builder, const SequencePlanImpl& plan,
                          const ProgramBufferShape& traits, PersistentLayout& out);

// The persistent layout of `plan` for a model of `shape` (its family's rules).
[[nodiscard]] PersistentLayout plan_persistent_layout(const StateShape& shape,
                                                      const SequencePlanImpl& plan);

// The storage a ring decoder (the dense family, Gemma 4) offers: BF16 or FP8 keys and values in
// the paged pool and the sliding rings, and no recurrent state. `family` names it in the refusal.
void validate_ring_decoder_storage(const EngineOptions& options, std::string_view family);

// The KV and ring storage a family lays out for a model of `shape` (the family's validate_options
// refuses the rest with these words): Qwen 3.5 any KV codec and no ring; the ring decoders BF16 or
// FP8 (a dense decoder without sliding layers no ring codec); NemotronH BF16 or FP8 and no ring.
void validate_state_storage(const StateShape& shape, const EngineOptions& options);

// The recurrent state's element type a --state-dtype names.
[[nodiscard]] DType recurrent_state_dtype(RecurrentStateStorage storage) noexcept;

// Whether a family's MTP round runs copy rounds (ModelTraits::copy_rounds: the families' traits
// take it from here): Qwen 3.5 / 3.6 and Gemma 4.
[[nodiscard]] constexpr bool family_copy_rounds(models::StateFamily family) noexcept {
    return family == models::StateFamily::Qwen35 || family == models::StateFamily::Gemma4;
}

// The copy window (SpeculativeOptions::ngram_draft_tokens) a plan takes: an automatic one
// (kCopyWindowAuto) is the MTP backend's widest window (kMtpDecodeMaximumDrafts) on a family whose
// round runs copy rounds (`copy_rounds`), where it is deeper than every draft depth and
// ngram_min_match is positive, and none otherwise; an explicit one stays.
[[nodiscard]] SpeculativeOptions resolve_copy_window(SpeculativeOptions speculative,
                                                     bool copy_rounds);

// The planner's inputs of `options` (validated by the family) on a family that runs copy rounds
// or not (`copy_rounds`, ModelTraits::copy_rounds): the copy window resolved here
// (resolve_copy_window), for a start and the plan without the weights alike. No model (`model`
// null: the caller sets it).
[[nodiscard]] SequencePlanningInputs sequence_planning_inputs(const EngineOptions& options,
                                                              bool copy_rounds,
                                                              std::int32_t multiprocessor_count,
                                                              bool graph_memory_held);

// The Main KV page groups of a plan: every lane's whole context at most, at least one context and
// a page a lane.
struct SequencePageRange {
    std::uint32_t minimum = 0;
    std::uint32_t maximum = 0;
};

[[nodiscard]] SequencePageRange sequence_page_range(std::uint32_t capacity,
                                                    std::uint32_t max_concurrency);

// A candidate plan of `inputs` at `main_page_groups`, its layouts and allowance not yet planned.
[[nodiscard]] std::unique_ptr<SequencePlanImpl>
sequence_plan_shell(const SequencePlanningInputs& inputs, std::uint32_t main_page_groups);

// Plans the CUDA graph allowance of `plan` (graph_allowance_bytes, launch_window_graph_bytes):
// none when graphs are off or the driver already holds their memory.
void plan_graph_allowance(SequencePlanImpl& plan, bool graph_memory_held);

// The widest call a ring decoder's sliding ring holds beyond its window: a prefill chunk or a
// speculative verify of the draft window.
[[nodiscard]] std::uint32_t ring_decoder_widest_call(const SequencePlanImpl& plan) noexcept;

} // namespace ninferno::models::qwen3_5::detail
