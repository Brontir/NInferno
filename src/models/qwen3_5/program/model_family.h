#pragma once
#include "models/qwen3_5/program/internal.h"

#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/vision_control.h"

#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "ninferno/ops/gdn_replay.h"
#include "ninferno/ops/moe.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <memory>
#include <optional>
#include <span>

namespace ninferno::models::qwen3_5::detail {

// The facts of a loaded model the Program's orchestration reads: widths of the buffers it owns,
// the token domain it validates and samples over, and what the weights occupy.
struct ModelTraits {
    std::int32_t hidden        = 0; // the target's residual width (prefill and carry hidden)
    std::int32_t output_rows   = 0; // rows of the output head's logits
    std::int32_t public_tokens = 0; // ids a request may carry or sample
    std::size_t weight_bytes   = 0; // device bytes the loaded weights occupy
    // Rows of one DFlash context feature column (target taps x hidden); 0 without a drafter.
    std::int32_t draft_feature_rows = 0;
    // Positions one sequence may address (the model's max_position_embeddings, and its drafter's):
    // the largest max_context validate_options admits, and an automatic max_context's ceiling.
    std::uint64_t max_positions = 0;
    // Its DFlash rounds draft the drafter's whole block where the round's width is Block
    // (dflash_draft_width); without it every round drafts the anchor and its K drafts only.
    bool dflash_block_width = false;
    // Its MTP round runs copy rounds (SpeculativeOptions::ngram_draft_tokens): the drafter
    // proposes a depth below the window's (MtpBatchContext::draft_depth).
    bool copy_rounds = false;
};

// The model under the Program: its planning facts and every call the Program's orchestration makes
// into it. The Program owns the mechanism (paged KV, state images, transactions, graphs, the
// resource policy's physical side); a family lays out the state and workspace its math needs and
// implements the entry points over the resources the contexts carry. Qwen 3.5 is the first family.
// Tunes every generic MoE layer the visitor names (ops::tune_moe): one line per layer shape.
template <class Visit>
std::vector<std::string> tune_moe_layers(cudaStream_t stream, Visit&& visit) {
    std::vector<std::string> lines;
    visit([&](const ops::MoeWeights& weights, const ops::MoeConfig& config) {
        const auto report = ops::tune_moe(weights, config, 16, stream);
        if ((report.ran || !report.cached) &&
            std::find(lines.begin(), lines.end(), report.summary) == lines.end()) {
            lines.push_back(report.summary);
        }
    });
    return lines;
}

class ModelExecution {
public:
    // Commits a speculative launch's accepted columns (per row: source slot, destination slot,
    // committed columns) into the linear state images from that launch's replay records.
    using ReplayFold = std::function<void(std::span<const ops::GdnReplayFoldRow>, cudaStream_t)>;

    virtual ~ModelExecution() = default;

    // Tunes the model's routes on this GPU at load (ops::tune_moe for every generic MoE layer)
    // before the free memory is measured: one line per layer shape for the startup log. The
    // default has nothing to tune.
    [[nodiscard]] virtual std::vector<std::string> tune_routes(cudaStream_t stream) const {
        (void)stream;
        return {};
    }

    [[nodiscard]] virtual const ModelTraits& traits() const noexcept = 0;
    // The components the model was loaded with (MTP, drafter, Vision); a plan must match them.
    [[nodiscard]] virtual models::LoadOptions load_options() const = 0;

    // Planning. validate_options rejects Engine options the model cannot run under;
    // persistent_layout and workspace_plan size one candidate plan (whose model is this one).
    virtual void validate_options(DeviceContext& device, const EngineOptions& options) const = 0;
    [[nodiscard]] virtual PersistentLayout
    persistent_layout(const SequencePlanImpl& plan) const                                  = 0;
    [[nodiscard]] virtual WorkspacePlan workspace_plan(const SequencePlanImpl& plan) const = 0;

    // Vision. A family without an encoder is never planned with Vision enabled.
    [[nodiscard]] virtual VisionControlPlan
    plan_vision_control(const PreparedPromptData& prompt) const;
    [[nodiscard]] virtual std::size_t
    vision_item_workspace_bytes(std::size_t patches, std::size_t merged_tokens,
                                const VisionWorkspacePlan& plan) const;
    [[nodiscard]] virtual std::unique_ptr<execution::VisionPrefill>
    begin_vision_prefill(DeviceContext& device, DeviceSpan workspace,
                         const VisionWorkspacePlan& workspace_plan, PreparedPromptData& prompt,
                         const VisionPrefillPlan& plan, std::size_t& handoff_peak_bytes) const;

    // Absolute end of the next prefill call over [cursor, end) when at most `chunk` fit in one
    // call. The family's rule keeps every call on the routes whose per-token result does not
    // depend on how a prompt is cut into calls.
    [[nodiscard]] virtual std::uint32_t prefill_call_tokens(std::uint32_t chunk,
                                                            std::uint32_t cursor,
                                                            std::uint32_t end) const = 0;

    [[nodiscard]] virtual execution::PrefillChunkResult
    prefill_text_chunk(execution::PrefillContext& state, std::span<const TokenId> ids,
                       std::uint32_t nominal_length, std::optional<std::uint32_t> split_frontier,
                       bool finalize_at_end) const = 0;
    // Vision prompts; a family without an encoder is never planned with Vision enabled.
    [[nodiscard]] virtual execution::PrefillChunkResult
    prefill_multimodal_chunk(execution::PrefillContext& state, const PreparedPromptData& prompt,
                             execution::VisionPrefill& vision, std::uint32_t nominal_length,
                             std::optional<std::uint32_t> split_frontier,
                             bool finalize_at_end) const;

    // Samples the next token from one final hidden state (a prefill that reused the whole
    // prompt): the output head, then the request's sampler keyed by `absolute_position`.
    virtual void sample_from_hidden(execution::PrefillContext& state, const Tensor& hidden,
                                    std::int32_t absolute_position, std::int32_t purpose) const;
    // Output-head logits of final hidden states (teacher-forced scoring).
    virtual void output_logits(const Tensor& hidden, Tensor& logits, WorkspaceArena& work,
                               cudaStream_t stream) const = 0;

    // Speculation. A family is planned with a backend only when it implements that backend's
    // entry points; the defaults reject the call.
    virtual void mtp_bridge_and_propose(execution::PrefillContext& state, const Tensor& next_token,
                                        const Tensor& previous_hidden, std::int32_t position,
                                        std::span<const std::int32_t> rope_position,
                                        bool build_proposal) const;
    virtual void mtp_bridge_multimodal(execution::PrefillContext& state,
                                       const PreparedPromptData& prompt,
                                       execution::VisionPrefill& vision,
                                       const execution::MtpBridgeInput& bridge) const;
    // The fold over one launch window's records (`records`, narrowed to its width) into `linear`.
    // Empty, the default: the ReplaySSM fold of Gated DeltaNet layers (ops::GdnReplayFoldPlan).
    // A family whose linear layers record something else (Mamba2) returns its own; it runs in the
    // round's tail, outside the captured graphs, with the rows the host resolved.
    [[nodiscard]] virtual ReplayFold replay_fold(const GdnReplayRecords& records,
                                                 LinearAttentionStatePool& linear) const {
        (void)records;
        (void)linear;
        return {};
    }

    // How many rows a sequence may execute past the frontier its request commits and leave no
    // trace a later row of that sequence reads: a key past a row's position is never visible to
    // it, and a ring holds that many rows beyond its window before one overwrites a visible key.
    // Ordinary decode may then run up to this many steps more than it commits, as a round of
    // several steps that ends at a stop. 0, the default: a row advances a recurrent state in place
    // (linear layers), so every executed row must be committed.
    [[nodiscard]] virtual std::uint32_t uncommitted_row_limit(const SequencePlanImpl& plan) const {
        (void)plan;
        return 0;
    }

    virtual void capture_ordinary_decode_batch(execution::OrdinaryBatchContext& state,
                                               std::int32_t batch_size,
                                               ops::CausalAttentionExecutionEnvelope envelope,
                                               DecodeGraphDefinition& definition) const = 0;
    virtual void ordinary_decode_batch(execution::OrdinaryBatchContext& state,
                                       std::int32_t batch_size,
                                       ops::CausalAttentionExecutionEnvelope envelope,
                                       DecodeGraphExecutable* executable) const = 0;

    virtual void capture_mtp_decode_batch(execution::MtpBatchContext& state,
                                          std::int32_t batch_size, std::uint32_t k,
                                          execution::MtpCausalAttentionEnvelopes envelopes,
                                          DecodeGraphDefinition& definition) const;
    virtual void mtp_decode_batch(execution::MtpBatchContext& state, std::int32_t batch_size,
                                  std::uint32_t k, execution::MtpCausalAttentionEnvelopes envelopes,
                                  DecodeGraphExecutable* executable) const;

    virtual void dflash_append_context(execution::DFlashAppendContext& state,
                                       const Tensor& features, const Tensor& positions,
                                       const Tensor& commit_counts, const Tensor& lanes,
                                       const Tensor& table_rows,
                                       ops::KVCacheAppendPrefixExecutionEnvelope envelope) const;
    virtual void capture_dflash_decode_batch(execution::DFlashBatchContext& state,
                                             std::int32_t batch_size, std::uint32_t k,
                                             execution::DFlashEnvelopes envelopes,
                                             ops::CausalAttentionExecutionEnvelope target_envelope,
                                             DecodeGraphDefinition& definition) const;
    virtual void dflash_decode_batch(execution::DFlashBatchContext& state, std::int32_t batch_size,
                                     std::uint32_t k, execution::DFlashEnvelopes envelopes,
                                     ops::CausalAttentionExecutionEnvelope target_envelope,
                                     DecodeGraphExecutable* executable) const;
};

// Qwen 3.5 / 3.6 (dense and MoE, with MTP, DFlash and Vision) over its execution parameters,
// which must outlive it.
[[nodiscard]] std::unique_ptr<const ModelExecution>
make_qwen3_5_execution(const execution::Parameters& parameters);

} // namespace ninferno::models::qwen3_5::detail
