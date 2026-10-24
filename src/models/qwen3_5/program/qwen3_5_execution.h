#pragma once
#include "models/qwen3_5/program/internal.h"

#include "models/qwen3_5/program/model_family.h"

#include <span>

namespace ninferno::models::qwen3_5::detail {

// The Qwen 3.5 family: its planning over the loaded parameters and the execution functions of
// execution/ and program/ that read them. Those functions reach the parameters through the
// family that called them (qwen_parameters below).
class Qwen35Execution final : public ModelExecution {
public:
    explicit Qwen35Execution(const execution::Parameters& parameters);

    [[nodiscard]] const execution::Parameters& parameters() const noexcept { return parameters_; }

    [[nodiscard]] const ModelTraits& traits() const noexcept override { return traits_; }
    [[nodiscard]] std::vector<std::string> tune_routes(cudaStream_t stream) const override {
        return tune_moe_layers(stream, [&](const auto& tune) {
            const auto ffn = [&](const execution::FfnParameters& p) {
                if (const auto* moe = std::get_if<execution::GenericMoeParameters>(&p)) {
                    tune(moe->weights, moe->config);
                }
            };
            for (const auto& layer : parameters_.text.layers) { ffn(layer.ffn); }
            if (parameters_.mtp) { ffn(parameters_.mtp->ffn); }
        });
    }

    [[nodiscard]] models::LoadOptions load_options() const override;
    void validate_options(DeviceContext& device, const EngineOptions& options) const override;
    [[nodiscard]] PersistentLayout persistent_layout(const SequencePlanImpl& plan) const override;
    [[nodiscard]] WorkspacePlan workspace_plan(const SequencePlanImpl& plan) const override;

    [[nodiscard]] VisionControlPlan
    plan_vision_control(const PreparedPromptData& prompt) const override;
    [[nodiscard]] std::size_t
    vision_item_workspace_bytes(std::size_t patches, std::size_t merged_tokens,
                                const VisionWorkspacePlan& plan) const override;
    [[nodiscard]] std::unique_ptr<execution::VisionPrefill>
    begin_vision_prefill(DeviceContext& device, DeviceSpan workspace,
                         const VisionWorkspacePlan& workspace_plan, PreparedPromptData& prompt,
                         const VisionPrefillPlan& plan,
                         std::size_t& handoff_peak_bytes) const override;

    [[nodiscard]] std::uint32_t prefill_call_tokens(std::uint32_t chunk, std::uint32_t cursor,
                                                    std::uint32_t end) const override;
    [[nodiscard]] execution::PrefillChunkResult
    prefill_text_chunk(execution::PrefillContext& state, std::span<const TokenId> ids,
                       std::uint32_t nominal_length, std::optional<std::uint32_t> split_frontier,
                       bool finalize_at_end) const override;
    [[nodiscard]] execution::PrefillChunkResult
    prefill_multimodal_chunk(execution::PrefillContext& state, const PreparedPromptData& prompt,
                             execution::VisionPrefill& vision, std::uint32_t nominal_length,
                             std::optional<std::uint32_t> split_frontier,
                             bool finalize_at_end) const override;
    void output_logits(const Tensor& hidden, Tensor& logits, WorkspaceArena& work,
                       cudaStream_t stream) const override;
    void mtp_bridge_and_propose(execution::PrefillContext& state, const Tensor& next_token,
                                const Tensor& previous_hidden, std::int32_t position,
                                std::span<const std::int32_t> rope_position,
                                bool build_proposal) const override;
    void mtp_bridge_multimodal(execution::PrefillContext& state, const PreparedPromptData& prompt,
                               execution::VisionPrefill& vision,
                               const execution::MtpBridgeInput& bridge) const override;
    void capture_ordinary_decode_batch(execution::OrdinaryBatchContext& state,
                                       std::int32_t batch_size,
                                       ops::CausalAttentionExecutionEnvelope envelope,
                                       DecodeGraphDefinition& definition) const override;
    void ordinary_decode_batch(execution::OrdinaryBatchContext& state, std::int32_t batch_size,
                               ops::CausalAttentionExecutionEnvelope envelope,
                               DecodeGraphExecutable* executable) const override;
    void capture_mtp_decode_batch(execution::MtpBatchContext& state, std::int32_t batch_size,
                                  std::uint32_t k, execution::MtpCausalAttentionEnvelopes envelopes,
                                  DecodeGraphDefinition& definition) const override;
    void mtp_decode_batch(execution::MtpBatchContext& state, std::int32_t batch_size,
                          std::uint32_t k, execution::MtpCausalAttentionEnvelopes envelopes,
                          DecodeGraphExecutable* executable) const override;
    void dflash_append_context(execution::DFlashAppendContext& state, const Tensor& features,
                               const Tensor& positions, const Tensor& commit_counts,
                               const Tensor& lanes, const Tensor& table_rows,
                               ops::KVCacheAppendPrefixExecutionEnvelope envelope) const override;
    void capture_dflash_decode_batch(execution::DFlashBatchContext& state, std::int32_t batch_size,
                                     std::uint32_t k, execution::DFlashEnvelopes envelopes,
                                     ops::CausalAttentionExecutionEnvelope target_envelope,
                                     DecodeGraphDefinition& definition) const override;
    void dflash_decode_batch(execution::DFlashBatchContext& state, std::int32_t batch_size,
                             std::uint32_t k, execution::DFlashEnvelopes envelopes,
                             ops::CausalAttentionExecutionEnvelope target_envelope,
                             DecodeGraphExecutable* executable) const override;

private:
    const execution::Parameters& parameters_;
    ModelTraits traits_;
    bool invariant_; // ActivationMode::Exact: the invariant route profile
};

// Qwen 3.5 planning over its parameters (planning/startup.cpp).
void validate_qwen3_5_options(const execution::Parameters& parameters, DeviceContext& device,
                              const EngineOptions& options);
[[nodiscard]] PersistentLayout qwen3_5_persistent_layout(const execution::Parameters& parameters,
                                                         const SequencePlanImpl& plan);
[[nodiscard]] WorkspacePlan qwen3_5_workspace_plan(const execution::Parameters& parameters,
                                                   const SequencePlanImpl& plan);
// The shape of the persistent state a Qwen 3.5 config lays out (sequence_bytes.h,
// plan_persistent_layout): `public_tokens` the ids a request may carry or sample.
[[nodiscard]] StateShape qwen3_5_state_shape(const qwen3_5::Config& config,
                                             std::int32_t public_tokens);

// The parameters of the Qwen 3.5 family that built an execution context. Only the Qwen 3.5
// execution functions call this, and only the Qwen 3.5 family calls them.
[[nodiscard]] inline const execution::Parameters&
qwen_parameters(const execution::ExecutionCore& core) noexcept {
    return static_cast<const Qwen35Execution&>(core.model).parameters();
}

} // namespace ninferno::models::qwen3_5::detail
