#include "models/qwen3_5/program/qwen3_5_execution.h"

#include "models/qwen3_5/execution/linear.h"
#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/sampling.h"
#include "spec/engine_support.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninferno::models::qwen3_5::detail {

VisionControlPlan ModelExecution::plan_vision_control(const PreparedPromptData&) const {
    throw std::invalid_argument("this model has no Vision encoder");
}

std::size_t ModelExecution::vision_item_workspace_bytes(std::size_t, std::size_t,
                                                        const VisionWorkspacePlan&) const {
    throw std::invalid_argument("this model has no Vision encoder");
}

std::unique_ptr<execution::VisionPrefill>
ModelExecution::begin_vision_prefill(DeviceContext&, DeviceSpan, const VisionWorkspacePlan&,
                                     PreparedPromptData&, const VisionPrefillPlan&,
                                     std::size_t&) const {
    throw std::invalid_argument("this model has no Vision encoder");
}

execution::PrefillChunkResult
ModelExecution::prefill_multimodal_chunk(execution::PrefillContext&, const PreparedPromptData&,
                                         execution::VisionPrefill&, std::uint32_t,
                                         std::optional<std::uint32_t>, bool) const {
    throw std::invalid_argument("this model has no Vision encoder");
}

void ModelExecution::sample_from_hidden(execution::PrefillContext& state, const Tensor& hidden,
                                        std::int32_t absolute_position,
                                        std::int32_t purpose) const {
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != traits().hidden || hidden.ne[1] != 1 ||
        hidden.ne[2] != 1 || hidden.ne[3] != 1 || hidden.data == nullptr) {
        throw std::invalid_argument("sample_from_hidden requires BF16 [hidden,1]");
    }
    execution::ExecutionCore& core = state.execution;
    core.work.reset();
    Tensor logits = core.io.logits.slice(1, 0, 1);
    output_logits(hidden, logits, core.work, core.device.stream);
    CUDA_CHECK(cudaMemcpyAsync(core.io.pos.data, &absolute_position, sizeof(absolute_position),
                               cudaMemcpyHostToDevice, core.device.stream));
    ops::sample(logits, core.io.token, traits().public_tokens, state.sampling, core.io.pos, purpose,
                core.work, core.device.stream);
    core.work.reset();
}

namespace {

[[noreturn]] void no_backend(const char* backend) {
    throw std::logic_error(std::string("this model family does not run the ") + backend +
                           " speculative backend");
}

} // namespace

void ModelExecution::mtp_bridge_and_propose(execution::PrefillContext&, const Tensor&,
                                            const Tensor&, std::int32_t,
                                            std::span<const std::int32_t>, bool) const {
    no_backend("MTP");
}

void ModelExecution::mtp_bridge_multimodal(execution::PrefillContext&, const PreparedPromptData&,
                                           execution::VisionPrefill&,
                                           const execution::MtpBridgeInput&) const {
    no_backend("MTP");
}

void ModelExecution::capture_mtp_decode_batch(execution::MtpBatchContext&, std::int32_t,
                                              std::uint32_t, execution::MtpCausalAttentionEnvelopes,
                                              DecodeGraphDefinition&) const {
    no_backend("MTP");
}

void ModelExecution::mtp_decode_batch(execution::MtpBatchContext&, std::int32_t, std::uint32_t,
                                      execution::MtpCausalAttentionEnvelopes,
                                      DecodeGraphExecutable*) const {
    no_backend("MTP");
}

void ModelExecution::dflash_append_context(execution::DFlashAppendContext&, const Tensor&,
                                           const Tensor&, const Tensor&, const Tensor&,
                                           const Tensor&,
                                           ops::KVCacheAppendPrefixExecutionEnvelope) const {
    no_backend("DFlash");
}

void ModelExecution::capture_dflash_decode_batch(execution::DFlashBatchContext&, std::int32_t,
                                                 std::uint32_t, execution::DFlashEnvelopes,
                                                 ops::CausalAttentionExecutionEnvelope,
                                                 DecodeGraphDefinition&) const {
    no_backend("DFlash");
}

void ModelExecution::dflash_decode_batch(execution::DFlashBatchContext&, std::int32_t,
                                         std::uint32_t, execution::DFlashEnvelopes,
                                         ops::CausalAttentionExecutionEnvelope,
                                         DecodeGraphExecutable*) const {
    no_backend("DFlash");
}

Qwen35Execution::Qwen35Execution(const execution::Parameters& parameters)
    : parameters_(parameters),
      traits_{
          .hidden        = execution::dimension(parameters.model.config().text.hidden_size),
          .output_rows   = execution::dimension(parameters.model.config().text.vocab_size),
          .public_tokens = execution::dimension(parameters.model.resources().public_token_count),
          .weight_bytes  = parameters.model.storage_stats().device_capacity_bytes,
          .draft_feature_rows =
              parameters.draft ? parameters.draft->feature_projection.weight.k : 0,
          .max_positions = parameters.draft
                               ? std::min<std::uint64_t>(
                                     parameters.model.config().text.max_position_embeddings,
                                     parameters.model.config().draft->max_position_embeddings)
                               : parameters.model.config().text.max_position_embeddings,
          .copy_rounds = true,
      },
      invariant_(parameters.model.options().activations == ActivationMode::Exact) {}

models::LoadOptions Qwen35Execution::load_options() const { return parameters_.model.options(); }

void Qwen35Execution::validate_options(DeviceContext& device, const EngineOptions& options) const {
    validate_qwen3_5_options(parameters_, device, options);
    // The DFlash and DFlash2 rounds draft the anchor and the round's K drafts only
    // (execution/draft.cpp), at every width; spec/engine_support.h tells the products so.
    static_assert(!spec::engine_support("qwen3_5_text").dflash_block_width &&
                  !spec::engine_support("qwen3_5_moe_text").dflash_block_width);
    const auto block = [](DFlashDraftWidth width) { return width == DFlashDraftWidth::Block; };
    if (block(options.speculative.draft_width) ||
        std::any_of(options.speculative.load.begin(), options.speculative.load.end(),
                    [&](const SpeculationLoad& bound) { return block(bound.draft_width); })) {
        throw std::invalid_argument(
            "Qwen 3.5's DFlash drafters draft only the round's K + 1 columns (draft width "
            "drafts)");
    }
}

PersistentLayout Qwen35Execution::persistent_layout(const SequencePlanImpl& plan) const {
    return qwen3_5_persistent_layout(parameters_, plan);
}

// --activations exact: every call under the invariant route profile
// (ninferno/ops/prefill_routes.h), the capacity queries of the workspace plan included, as the
// Gemma 4, dense and Nemotron-H families size theirs. Without it a projection on the generic path
// (NVFP4 outside sm_120a, GGUF weights on every card) sums a column in another order at a decode or
// verify width than in a prefill.
WorkspacePlan Qwen35Execution::workspace_plan(const SequencePlanImpl& plan) const {
    const ops::ScopedInvariantRoutes routes(invariant_);
    return qwen3_5_workspace_plan(parameters_, plan);
}

VisionControlPlan Qwen35Execution::plan_vision_control(const PreparedPromptData& prompt) const {
    return qwen3_5::plan_vision_control(prompt, *parameters_.model.config().vision);
}

std::size_t Qwen35Execution::vision_item_workspace_bytes(std::size_t patches,
                                                         std::size_t merged_tokens,
                                                         const VisionWorkspacePlan& plan) const {
    return execution::VisionContext::workspace_bytes(
        *parameters_.model.config().vision, *parameters_.vision, patches, merged_tokens, plan);
}

std::unique_ptr<execution::VisionPrefill>
Qwen35Execution::begin_vision_prefill(DeviceContext& device, DeviceSpan workspace,
                                      const VisionWorkspacePlan& workspace_plan,
                                      PreparedPromptData& prompt, const VisionPrefillPlan& plan,
                                      std::size_t& handoff_peak_bytes) const {
    return std::make_unique<execution::VisionPrefillSession>(
        device, parameters_, workspace, workspace_plan, prompt, plan, handoff_peak_bytes);
}

std::uint32_t Qwen35Execution::prefill_call_tokens(std::uint32_t chunk, std::uint32_t cursor,
                                                   std::uint32_t end) const {
    return execution::prefill_call_tokens(chunk, cursor, end);
}

execution::PrefillChunkResult Qwen35Execution::prefill_text_chunk(
    execution::PrefillContext& state, std::span<const TokenId> ids, std::uint32_t nominal_length,
    std::optional<std::uint32_t> split_frontier, bool finalize_at_end) const {
    const ops::ScopedInvariantRoutes routes(invariant_);
    return execution::prefill_text_chunk(state, ids, nominal_length, split_frontier,
                                         finalize_at_end);
}

execution::PrefillChunkResult Qwen35Execution::prefill_multimodal_chunk(
    execution::PrefillContext& state, const PreparedPromptData& prompt,
    execution::VisionPrefill& vision, std::uint32_t nominal_length,
    std::optional<std::uint32_t> split_frontier, bool finalize_at_end) const {
    const ops::ScopedInvariantRoutes routes(invariant_);
    return execution::prefill_multimodal_chunk(
        state, prompt, static_cast<execution::VisionPrefillSession&>(vision), nominal_length,
        split_frontier, finalize_at_end);
}

void Qwen35Execution::output_logits(const Tensor& hidden, Tensor& logits, WorkspaceArena& work,
                                    cudaStream_t stream) const {
    const ops::ScopedInvariantRoutes routes(invariant_);
    execution::project(hidden, parameters_.text.output_head, logits, work, stream);
}

void Qwen35Execution::mtp_bridge_and_propose(execution::PrefillContext& state,
                                             const Tensor& next_token,
                                             const Tensor& previous_hidden, std::int32_t position,
                                             std::span<const std::int32_t> rope_position,
                                             bool build_proposal) const {
    const ops::ScopedInvariantRoutes routes(invariant_);
    execution::mtp_bridge_and_propose(state, next_token, previous_hidden, position, rope_position,
                                      build_proposal);
}

void Qwen35Execution::mtp_bridge_multimodal(execution::PrefillContext& state,
                                            const PreparedPromptData& prompt,
                                            execution::VisionPrefill& vision,
                                            const execution::MtpBridgeInput& bridge) const {
    const ops::ScopedInvariantRoutes routes(invariant_);
    execution::mtp_bridge_multimodal(
        state, prompt, static_cast<execution::VisionPrefillSession&>(vision), bridge);
}

void Qwen35Execution::capture_ordinary_decode_batch(execution::OrdinaryBatchContext& state,
                                                    std::int32_t batch_size,
                                                    ops::CausalAttentionExecutionEnvelope envelope,
                                                    DecodeGraphDefinition& definition) const {
    const ops::ScopedInvariantRoutes routes(invariant_);
    execution::capture_ordinary_decode_batch(state, batch_size, envelope, definition);
}

void Qwen35Execution::ordinary_decode_batch(execution::OrdinaryBatchContext& state,
                                            std::int32_t batch_size,
                                            ops::CausalAttentionExecutionEnvelope envelope,
                                            DecodeGraphExecutable* executable) const {
    const ops::ScopedInvariantRoutes routes(invariant_);
    execution::ordinary_decode_batch(state, batch_size, envelope, executable);
}

void Qwen35Execution::capture_mtp_decode_batch(execution::MtpBatchContext& state,
                                               std::int32_t batch_size, std::uint32_t k,
                                               execution::MtpCausalAttentionEnvelopes envelopes,
                                               DecodeGraphDefinition& definition) const {
    const ops::ScopedInvariantRoutes routes(invariant_);
    const ops::ScopedRoundLanes lanes(batch_size); // the drafter's calls too
    execution::capture_mtp_decode_batch(state, batch_size, k, envelopes, definition);
}

void Qwen35Execution::mtp_decode_batch(execution::MtpBatchContext& state, std::int32_t batch_size,
                                       std::uint32_t k,
                                       execution::MtpCausalAttentionEnvelopes envelopes,
                                       DecodeGraphExecutable* executable) const {
    const ops::ScopedInvariantRoutes routes(invariant_);
    const ops::ScopedRoundLanes lanes(batch_size); // the drafter's calls too
    execution::mtp_decode_batch(state, batch_size, k, envelopes, executable);
}

void Qwen35Execution::dflash_append_context(
    execution::DFlashAppendContext& state, const Tensor& features, const Tensor& positions,
    const Tensor& commit_counts, const Tensor& lanes, const Tensor& table_rows,
    ops::KVCacheAppendPrefixExecutionEnvelope envelope) const {
    const ops::ScopedInvariantRoutes routes(invariant_);
    execution::dflash_append_context(state, features, positions, commit_counts, lanes, table_rows,
                                     envelope);
}

void Qwen35Execution::capture_dflash_decode_batch(
    execution::DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
    execution::DFlashEnvelopes envelopes, ops::CausalAttentionExecutionEnvelope target_envelope,
    DecodeGraphDefinition& definition) const {
    const ops::ScopedInvariantRoutes routes(invariant_);
    const ops::ScopedRoundLanes lanes(batch_size); // the drafter's calls too
    execution::capture_dflash_decode_batch(state, batch_size, k, envelopes, target_envelope,
                                           definition);
}

void Qwen35Execution::dflash_decode_batch(execution::DFlashBatchContext& state,
                                          std::int32_t batch_size, std::uint32_t k,
                                          execution::DFlashEnvelopes envelopes,
                                          ops::CausalAttentionExecutionEnvelope target_envelope,
                                          DecodeGraphExecutable* executable) const {
    const ops::ScopedInvariantRoutes routes(invariant_);
    const ops::ScopedRoundLanes lanes(batch_size); // the drafter's calls too
    execution::dflash_decode_batch(state, batch_size, k, envelopes, target_envelope, executable);
}

std::unique_ptr<const ModelExecution>
make_qwen3_5_execution(const execution::Parameters& parameters) {
    return std::make_unique<const Qwen35Execution>(parameters);
}

} // namespace ninferno::models::qwen3_5::detail
