#pragma once

// The dense family's execution under the Program (models/dense/program_family.h builds it): the
// target's forward over the Program's state (program_family.cpp) and, with a DFlash drafter, the
// drafter's side of the Program's DFlash rounds (dflash.cpp). Internal to the dense family.

#include "core/device.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "models/dense/config.h"
#include "models/dense/model.h"
#include "models/dense/target.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/model_family.h"
#include "models/speculation/dflash.h"
#include "ninferno/ops/dense_fused.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace ninferno::models::dense::detail {

using qwen3_5::detail::DFlashPersistentState;
using qwen3_5::detail::ModelExecution;
using qwen3_5::detail::ModelTraits;
using qwen3_5::detail::PersistentLayout;
using qwen3_5::detail::SequencePlanImpl;
using qwen3_5::detail::WorkspacePlan;
namespace program = qwen3_5::execution;

class DenseExecution final : public ModelExecution {
public:
    DenseExecution(const Model& model, const speculation::DFlashDrafter* drafter,
                   DeviceContext& device, std::int32_t public_tokens, ActivationMode activations);

    [[nodiscard]] const ModelTraits& traits() const noexcept override { return traits_; }
    [[nodiscard]] std::vector<std::string> tune_routes(cudaStream_t stream) const override {
        if (!model_.config().moe) { return {}; }
        return qwen3_5::detail::tune_moe_layers(stream, [&](const auto& tune) {
            for (const auto& layer : model_.weights().layers) {
                if (layer.moe) { tune(*layer.moe, model_.config().moe->op); }
            }
        });
    }
    [[nodiscard]] models::LoadOptions load_options() const override;

    void validate_options(DeviceContext& device, const EngineOptions& options) const override;
    [[nodiscard]] PersistentLayout persistent_layout(const SequencePlanImpl& plan) const override;
    [[nodiscard]] WorkspacePlan workspace_plan(const SequencePlanImpl& plan) const override;

    [[nodiscard]] std::uint32_t prefill_call_tokens(std::uint32_t chunk, std::uint32_t cursor,
                                                    std::uint32_t end) const override;
    [[nodiscard]] program::PrefillChunkResult
    prefill_text_chunk(program::PrefillContext& state, std::span<const TokenId> ids,
                       std::uint32_t nominal_length, std::optional<std::uint32_t> split_frontier,
                       bool finalize_at_end) const override;

    void output_logits(const Tensor& hidden, Tensor& logits, WorkspaceArena& work,
                       cudaStream_t) const override;

    // Full layers never read a key past the reader; a sliding ring holds the widest call's rows
    // beyond the largest window (ring_capacity).
    [[nodiscard]] std::uint32_t
    uncommitted_row_limit(const SequencePlanImpl& plan) const override;

    void capture_ordinary_decode_batch(program::OrdinaryBatchContext& state,
                                       std::int32_t batch_size,
                                       ops::CausalAttentionExecutionEnvelope envelope,
                                       DecodeGraphDefinition& definition) const override;
    void ordinary_decode_batch(program::OrdinaryBatchContext& state, std::int32_t batch_size,
                               ops::CausalAttentionExecutionEnvelope envelope,
                               DecodeGraphExecutable* executable) const override;

    // The DFlash hooks (dflash.cpp).
    void dflash_append_context(program::DFlashAppendContext& state, const Tensor& features,
                               const Tensor& positions, const Tensor& commit_counts,
                               const Tensor& lanes, const Tensor& table_rows,
                               ops::KVCacheAppendPrefixExecutionEnvelope envelope) const override;
    void capture_dflash_decode_batch(program::DFlashBatchContext& state, std::int32_t batch_size,
                                     std::uint32_t k, program::DFlashEnvelopes envelopes,
                                     ops::CausalAttentionExecutionEnvelope target_envelope,
                                     DecodeGraphDefinition& definition) const override;
    void dflash_decode_batch(program::DFlashBatchContext& state, std::int32_t batch_size,
                             std::uint32_t k, program::DFlashEnvelopes envelopes,
                             ops::CausalAttentionExecutionEnvelope target_envelope,
                             DecodeGraphExecutable* executable) const override;

private:
    // --- the target (program_family.cpp) ---

    // The context of the session the Program serves (its plan's capacity, the resolved
    // max_context): a LongRoPE model's rotation mode follows it, as under ninferno-dense.
    static std::int32_t session_context(const program::ExecutionCore& core);

    // Sliding rings keep every key the widest call can see while it appends its own rows.
    [[nodiscard]] std::uint32_t ring_capacity(const SequencePlanImpl& plan) const;
    // The most rows one call appends: a prefill chunk, or a speculative verify.
    [[nodiscard]] static std::uint32_t widest_call(const SequencePlanImpl& plan);

    // One binding per layer: a full layer reads the paged pool through `tables` (row rows_full[b]
    // of batch row b), a sliding layer its ring in state image rows_ring[b]. With `stored`, batch
    // row b writes the keys and values of its first stored[b] columns only (a verify's padding).
    [[nodiscard]] std::vector<ops::Gemma4KvLayer>
    bindings(const qwen3_5::PagedKVCache& cache, const qwen3_5::StateImageDevicePool& images,
             const Tensor& tables, const Tensor& rows_full, const Tensor& rows_ring,
             const Tensor& stored = {}) const;

    // The one-sequence bindings of a prefill: its table as a one-row table, its rings in state
    // image `slot`. The two row indices live in `rows` (I32 [2], carved by the caller).
    [[nodiscard]] std::vector<ops::Gemma4KvLayer>
    sequence_bindings(program::PrefillContext& state, std::int32_t slot, Tensor& rows) const;

    std::function<void()> ordinary_body(program::OrdinaryBatchContext& state,
                                        std::int32_t batch_size,
                                        ops::CausalAttentionExecutionEnvelope envelope) const;

    // --- the DFlash drafter (dflash.cpp) ---

    // Refuses a drafter the Program's DFlash round cannot run: a block wider than the round's 16
    // columns, sliding layers causal inside the block (the round's sliding attention is the block
    // mask), or a pairing its loader did not establish.
    void check_drafter() const;

    // The drafter's local (sliding) layers as fixed rings of every state image; none without a
    // sliding layer.
    [[nodiscard]] std::optional<qwen3_5::DFlashLocalStateSpec> dflash_local_spec() const;
    // The rest of the drafter's persistent state: its full layers' paged pool (when it has any),
    // the prefill chunk's tapped features and every lane's pending features.
    [[nodiscard]] qwen3_5::detail::DFlashPersistentLayout
    dflash_layout(LayoutBuilder& builder, const SequencePlanImpl& plan) const;
    // dflash_context (a prefill chunk's append) and dflash_round.
    void dflash_workspace(const SequencePlanImpl& plan, WorkspacePlan& out) const;

    // The tap of the drafter's feature layers into `sink` (BF16 [taps*H, T]): a round's verify.
    [[nodiscard]] FeatureTap feature_tap(const Tensor& sink) const;
    // A prefill chunk's tap into `sink`: its feature projection BF16 [H, T] when the drafter's is
    // split by tap (accumulated tap by tap as the forward reaches each tapped layer, in layer
    // order: the first tap's slice stores, each later one adds with linear_add, one BF16 rounding
    // of the sum per tap), else the tapped features themselves as feature_tap().
    [[nodiscard]] FeatureTap prefill_tap(const Tensor& sink, WorkspaceArena& work,
                                         cudaStream_t stream) const;
    // Rows of prefill_features: H when the projection is split, else taps*H.
    [[nodiscard]] std::int32_t prefill_feature_rows() const;
    // The workspace a prefill chunk's split projection adds to its forward (a slice's linear or
    // linear_add); 0 when the projection is not split.
    [[nodiscard]] std::size_t prefill_tap_bytes(std::int32_t chunk) const;
    // feature_projection(features) for features BF16 [taps*H, C] from the split projection:
    // each tap's rows through its slice, summed in layer order into `out` BF16 [H, C].
    void project_features(WorkspaceArena& work, const Tensor& features, Tensor& out,
                          cudaStream_t stream) const;

    // A drafter layer's q/k/v preparation (dense_qkv_prepare) at `positions`: its key and value
    // biases, the per-head key norm and the drafter's rotation; the caller adds the queries.
    [[nodiscard]] ops::DenseQkvPrepare qkv_prepare(const speculation::DFlashLayer& weights,
                                                   const Tensor& positions) const;
    // Context keys and values [D, Hkv, C] of a layer from its projected ones [Hkv*D, C] (row
    // slices allowed): bias, key norm and rotation of the keys, bias of the values.
    void prepare_keys_values(const speculation::DFlashLayer& weights, const Tensor& key_src,
                             const Tensor& value_src, const Tensor& positions, Tensor key,
                             Tensor value, cudaStream_t stream) const;

    // Appends committed target features BF16 [taps*H, W, B] (a prefill chunk, B = 1, or a round's
    // compacted ragged batch) to the drafter's context: context_norm(feature_projection(x)), then
    // each layer's context key (key norm, RoPE) and value, the first commit_counts[b] columns of
    // row b at positions[:, b] into lane lanes[b] (local layers) or KV row table_rows[b] (full).
    // `projected`: `features` is already feature_projection(x) BF16 [H, W, 1] (prefill_tap()).
    void append_context(program::ExecutionCore& core, DFlashPersistentState& dflash,
                        const Tensor& features, const Tensor& positions,
                        const Tensor& commit_counts, const Tensor& lanes,
                        const Tensor& table_rows,
                        ops::KVCacheAppendPrefixExecutionEnvelope envelope,
                        bool projected = false) const;

    // The drafter's block pass: the anchor and block_size - 1 mask tokens at each row's frontier
    // through every drafter layer over its committed context, then the target's head on columns
    // 1..k, whose argmax rows are the round's drafts.
    void propose(program::DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                 program::DFlashEnvelopes envelopes) const;

    // The round: the append of the last round's accepted features, the proposal, the target's
    // verify with the feature tap and the greedy accept.
    std::function<void()> dflash_body(program::DFlashBatchContext& state, std::int32_t batch_size,
                                      std::uint32_t k, program::DFlashEnvelopes envelopes,
                                      ops::CausalAttentionExecutionEnvelope target_envelope) const;

    [[nodiscard]] std::size_t dflash_context_bytes(std::int32_t width, std::int32_t batch,
                                                   bool compact_input) const;
    [[nodiscard]] std::size_t dflash_proposal_bytes(std::uint32_t capacity, std::int32_t drafts,
                                                    std::int32_t batch) const;

    const Model& model_;
    const Config& c_;
    DensePass pass_;
    const speculation::DFlashDrafter* drafter_;
    bool invariant_; // ActivationMode::Bits16: the invariant route profile
    // The mode the model was loaded under: what load_options() reports to the frozen sequence
    // plan, which compares it with the startup options (program_impl.cpp).
    ActivationMode activations_;
    ModelTraits traits_;
    std::vector<std::uint32_t> index_; // each target layer's place among its kind
    std::uint32_t full_layers_    = 0;
    std::uint32_t sliding_layers_ = 0;
    // With a drafter: each target layer's row block in the tapped features (-1: untapped), and
    // each drafter layer's place among its kind (its ring, or its plane of the full pool).
    std::vector<std::int32_t> tap_slots_;
    // The tap slots in layer order, and whether the drafter's feature projection is split by tap
    // (DFlashWeights::feature_slices).
    std::vector<std::int32_t> tap_order_;
    bool split_features_ = false;
    std::vector<std::uint32_t> drafter_index_;
    std::uint32_t drafter_local_layers_ = 0;
    std::uint32_t drafter_full_layers_  = 0;
};

} // namespace ninferno::models::dense::detail
