#include "models/nemotron_h/program_family.h"

#include "core/device_class.h"
#include "models/nemotron_h/pass.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/prefill_text_call.h"
#include "models/qwen3_5/program/graph_execution.h"
#include "models/qwen3_5/program/ordinary_decode.h"
#include "models/qwen3_5/program/speculative/mtp_verify_accept.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "ninferno/ops/argmax.h"
#include "ninferno/ops/mamba2.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/mtp_round.h"
#include "ninferno/ops/position.h"
#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/sampling.h"
#include "ninferno/ops/scalar.h"
#include "ninferno/ops/scatter.h"
#include "ninferno/ops/speculative_round.h"
#include "runtime/contract/timing.h"
#include "spec/engine_support.h"

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninferno::models::nemotron_h {
namespace {

using qwen3_5::detail::ModelExecution;
using qwen3_5::detail::ModelTraits;
using qwen3_5::detail::PersistentLayout;
using qwen3_5::detail::SequencePlanImpl;
using qwen3_5::detail::WorkspacePlan;
namespace program = qwen3_5::execution;

static_assert(kMaximumMtpDraftTokens <= qwen3_5::kMtpDecodeMaximumDrafts,
              "a NemotronH MTP round must stay inside the Program's MTP round domain");
static_assert(kMaximumConcurrency <= ops::kMamba2FoldMaxRows,
              "the Mamba2 fold takes every row of the widest launch");

// NemotronH as a family of the one Program: its attention layers keep their keys and values in the
// paged pool, its Mamba2 layers their convolution windows and recurrent states in the linear
// state of each sequence's state image (a window [C, 3] and a state [N, P, H] per layer), so
// prefix checkpoints, forks and fresh sequences are the Program's own slot mechanics. A fork runs
// in place after the destination slot takes the source's state.
//
// With the MTP head a round verifies the pending token and k drafts in
// one target call whose Mamba2 layers only record (the replay records) and whose attention stores
// only the real columns; replay_fold then commits each row's accepted columns from its source slot
// into its destination. The head keeps its own keys and values in the MTP pool: position p holds
// the pair (target hidden at p, token at p + 1), written for the prompt by the prefill, for the
// licensed columns by each round's alignment, and speculatively by the draft chain.
class NemotronExecution final : public ModelExecution {
public:
    NemotronExecution(const Model& model, const MtpWeights* mtp, DeviceContext& device,
                      std::int32_t public_tokens, ActivationMode activations)
        : model_(model), c_(model.config()), pass_(model, device), mtp_(mtp),
          invariant_(activations == ActivationMode::Exact),
          traits_{.hidden        = c_.hidden_size,
                  .output_rows   = c_.vocab_size,
                  .public_tokens = public_tokens,
                  .weight_bytes  = model.device_bytes(),
                  .max_positions = static_cast<std::uint64_t>(c_.max_positions)} {
        if (public_tokens <= 0 || public_tokens > c_.vocab_size) {
            throw std::invalid_argument("nemotron_h public token domain exceeds the embedding rows");
        }
        if (mtp_ != nullptr) { mtp_pass_.emplace(model, *mtp_, device); }
    }

    [[nodiscard]] const ModelTraits& traits() const noexcept override { return traits_; }

    // The trunk's MoE layers share one shape and the MTP block's is the same: one timing.
    [[nodiscard]] std::vector<std::string> tune_routes(cudaStream_t stream) const override {
        return qwen3_5::detail::tune_moe_layers(stream, [&](const auto& tune) {
            for (const auto& layer : model_.weights().layers) {
                if (layer.moe) { tune(*layer.moe, c_.moe.op); }
            }
            if (mtp_ != nullptr) { tune(mtp_->moe, c_.moe.op); }
        });
    }

    [[nodiscard]] models::LoadOptions load_options() const override {
        return {.speculative = mtp_ ? SpeculativeBackend::Mtp : SpeculativeBackend::None,
                .activations = invariant_ ? ActivationMode::Exact : ActivationMode::Fast};
    }

    void validate_options(DeviceContext& device, const EngineOptions& options) const override {
        if (options.purpose != EnginePurpose::Generation) {
            throw std::invalid_argument(
                "Nemotron-H serves generation only; ninferno-perplexity scores Qwen models");
        }
        if (options.enable_vision) {
            throw std::invalid_argument("Nemotron-H artifacts carry no Vision encoder");
        }
        static_assert(spec::engine_support("nemotron_h").mtp_depth ==
                          static_cast<std::int32_t>(kMaximumMtpDraftTokens) &&
                      spec::engine_support("nemotron_h").dflash_depth == 0 &&
                      !spec::engine_support("nemotron_h").proposal_head);
        if (options.speculative.backend !=
            (mtp_ ? SpeculativeBackend::Mtp : SpeculativeBackend::None)) {
            throw std::invalid_argument("Nemotron-H speculates with its MTP head only (--spec mtp "
                                        "on an artifact with one)");
        }
        if (options.speculative.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument("Nemotron-H's MTP head scores with the full head "
                                        "(--lm-head-draft is for Qwen models)");
        }
        qwen3_5::detail::validate_state_storage(nemotron_h::state_shape(c_, traits_.public_tokens),
                                                options);
        qwen3_5::detail::validate_program_options(options, traits_.max_positions,
                                                  mtp_ ? kMaximumMtpDraftTokens : 0);
        try {
            require_device_support(device.device_class(), {.mma_bf16 = true}, "nemotron_h runtime");
        } catch (const std::runtime_error& e) { throw std::invalid_argument(e.what()); }
    }

    [[nodiscard]] PersistentLayout persistent_layout(const SequencePlanImpl& plan) const override {
        return qwen3_5::detail::plan_persistent_layout(
            nemotron_h::state_shape(c_, traits_.public_tokens), plan);
    }

    // Every Mamba2 layer's fold (mamba2_verify_fold) over the window's records.
    [[nodiscard]] ReplayFold replay_fold(const GdnReplayRecords& records,
                                         LinearAttentionStatePool& linear) const override {
        return [this, records, &linear](std::span<const ops::GdnReplayFoldRow> rows,
                                        cudaStream_t stream) {
            if (rows.empty() || rows.size() > ops::kMamba2FoldMaxRows) {
                throw std::logic_error("nemotron_h fold rows are out of range");
            }
            std::array<ops::Mamba2FoldRow, ops::kMamba2FoldMaxRows> fold{};
            for (std::size_t i = 0; i < rows.size(); ++i) {
                fold[i] = {.source_slot = rows[i].source_state_slot,
                           .dest_slot   = rows[i].destination_state_slot,
                           .commit      = rows[i].commit_columns};
            }
            const auto batch = static_cast<std::int32_t>(rows.size());
            const std::int32_t width    = records.spec.width;
            const std::int32_t columns  = width * batch;
            const std::int32_t channels = c_.mamba.conv_channels();
            for (const auto& layer : model_.weights().layers) {
                if (!layer.mamba) { continue; }
                const auto& m                   = *layer.mamba;
                const GdnReplayRecordLayer rec =
                    records.layer(static_cast<std::int32_t>(layer.index), batch);
                LinearAttentionStateLayerView v = linear.layer_view(layer.index);
                ops::mamba2_verify_fold(rec.conv.view({channels, columns}),
                                        rec.value.view({channels, columns}),
                                        rec.key.view({c_.mamba.geometry.heads, columns}), width,
                                        std::span<const ops::Mamba2FoldRow>(fold.data(), rows.size()),
                                        m.dt_bias, m.a_log, m.d, v.conv, v.recurrent,
                                        c_.mamba.geometry, stream);
            }
        };
    }

    [[nodiscard]] WorkspacePlan workspace_plan(const SequencePlanImpl& plan) const override {
        const auto chunk   = static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
        const auto batch   = static_cast<std::int32_t>(plan.max_concurrency);
        const auto context = static_cast<std::int32_t>(plan.capacity);
        // Every capacity query under the route profiles its calls run under
        // (ninferno/ops/prefill_routes.h): the invariant one under --activations exact, and the
        // prefill one for a prefill chunk (its MTP alignment and chain included).
        const ops::ScopedInvariantRoutes routes(invariant_);
        WorkspacePlan out;
        {
            const ops::ScopedPrefillRoutes prefill(true);
            out.text_prefill = Pass::workspace_bytes(model_, context, chunk, 1) +
                               ops::sampling_workspace_capacity_bytes(traits_.public_tokens, 1, 1);
        }
        out.ordinary_round =
            Pass::workspace_bytes(model_, context, batch, batch) +
            ops::sampling_workspace_capacity_bytes(traits_.public_tokens, 1, batch);
        if (mtp_ != nullptr) {
            const auto width = static_cast<std::int32_t>(plan.draft_window) + 1;
            // A speculative round runs at most the widest speculative launch's rows.
            const auto rows = static_cast<std::int32_t>(
                qwen3_5::detail::speculative_round_batches(plan));
            // The most verify columns a round runs: under a load bound, a batch at least its lanes
            // wide runs no window deeper than the bound.
            const auto columns = static_cast<std::int32_t>(
                qwen3_5::detail::speculative_round_columns(plan));
            // The prompt's alignment over one chunk (and its draft chain, one column at a time).
            {
                const ops::ScopedPrefillRoutes prefill(true);
                out.mtp_prefill = out.text_prefill +
                                  MtpPass::workspace_bytes(model_, *mtp_, context, chunk) +
                                  static_cast<std::size_t>(c_.hidden_size) * 2 * (chunk + 2) + 4096;
            }
            // The verify, the acceptance, the alignment over every column and the chain.
            out.mtp_round =
                Pass::workspace_bytes(model_, context, columns, columns) +
                ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                    traits_.public_tokens, 1, width - 1, 1, rows) +
                MtpPass::workspace_bytes(model_, *mtp_, context, columns) + 4096;
        }
        out.general_capacity = std::max(
            {out.text_prefill, out.ordinary_round, out.mtp_prefill, out.mtp_round});
        out.capacity         = out.general_capacity;
        return out;
    }

    [[nodiscard]] std::uint32_t prefill_call_tokens(std::uint32_t chunk, std::uint32_t cursor,
                                                    std::uint32_t end) const override {
        return program::ungridded_call_tokens(chunk, end - cursor);
    }

    [[nodiscard]] program::PrefillChunkResult
    prefill_text_chunk(program::PrefillContext& state, std::span<const TokenId> ids,
                       std::uint32_t nominal_length, std::optional<std::uint32_t> split_frontier,
                       bool finalize_at_end) const override;

    void output_logits(const Tensor& hidden, Tensor& logits, WorkspaceArena& work,
                       cudaStream_t) const override {
        const ops::ScopedInvariantRoutes routes(invariant_);
        pass_.head(work, hidden, logits);
    }

    void capture_ordinary_decode_batch(program::OrdinaryBatchContext& state,
                                       std::int32_t batch_size,
                                       ops::CausalAttentionExecutionEnvelope envelope,
                                       DecodeGraphDefinition& definition) const override {
        const ops::ScopedInvariantRoutes routes(invariant_);
        program::capture_graph(state, definition, ordinary_body(state, batch_size, envelope));
    }

    void ordinary_decode_batch(program::OrdinaryBatchContext& state, std::int32_t batch_size,
                               ops::CausalAttentionExecutionEnvelope envelope,
                               DecodeGraphExecutable* executable) const override {
        const ops::ScopedInvariantRoutes routes(invariant_);
        // A row that forks from a checkpoint writes its first token in this launch: its
        // destination takes the source's Mamba2 state first, and the launch then runs in place.
        auto& linear = state.execution.state_images.linear();
        for (std::int32_t row = 0; row < batch_size; ++row) {
            const std::int32_t from = state.host_ingress.state_source_slots[row];
            const std::int32_t to   = state.host_ingress.state_destination_slots[row];
            if (from != to) { linear.copy_slot(from, to, state.execution.device.stream); }
        }
        program::run_prepared(state, executable, ordinary_body(state, batch_size, envelope));
    }

    void mtp_bridge_and_propose(program::PrefillContext& state, const Tensor& next_token,
                                const Tensor& previous_hidden, std::int32_t position,
                                std::span<const std::int32_t>,
                                bool build_proposal) const override;

    void capture_mtp_decode_batch(program::MtpBatchContext& state, std::int32_t batch_size,
                                  std::uint32_t k, program::MtpCausalAttentionEnvelopes envelopes,
                                  DecodeGraphDefinition& definition) const override {
        const ops::ScopedInvariantRoutes routes(invariant_);
        const ops::ScopedRoundLanes lanes(batch_size);
        program::capture_graph(state, definition, mtp_body(state, batch_size, k, envelopes));
    }

    // No fork settles here: the verify reads each row's source slot and the fold writes its
    // destination.
    void mtp_decode_batch(program::MtpBatchContext& state, std::int32_t batch_size,
                          std::uint32_t k, program::MtpCausalAttentionEnvelopes envelopes,
                          DecodeGraphExecutable* executable) const override {
        const ops::ScopedInvariantRoutes routes(invariant_);
        const ops::ScopedRoundLanes lanes(batch_size);
        program::run_prepared(state, executable, mtp_body(state, batch_size, k, envelopes));
    }

private:
    // One attention layer of a pool through `tables`; an FP8 pool's scale planes [1, 64, pages,
    // heads] are the code planes without their D axis.
    [[nodiscard]] static ops::Gemma4KvLayer layer_binding(const PagedKVBatchLayerView& view,
                                                          const Tensor& tables, const Tensor& rows,
                                                          const Tensor& stored) {
        ops::Gemma4KvLayer layer{.k            = view.k_pages,
                                 .v            = view.v_pages,
                                 .window       = 0,
                                 .block_tables = tables,
                                 .rows         = rows,
                                 .stored       = stored};
        if (view.k_scale_pages.data != nullptr) {
            const auto& k   = view.k_pages;
            const auto n    = static_cast<std::int32_t>(k.ne[1]);
            const auto pgs  = static_cast<std::int32_t>(k.ne[2]);
            const auto hds  = static_cast<std::int32_t>(k.ne[3]);
            layer.k_scale   = view.k_scale_pages.view({n, pgs, hds});
            layer.v_scale   = view.v_scale_pages.view({n, pgs, hds});
        }
        return layer;
    }

    // The MTP head's one layer of the MTP pool through `tables` (row rows[b] of batch row b),
    // storing the first stored[b] columns of each row when `stored` is given.
    [[nodiscard]] static ops::Gemma4KvLayer mtp_binding(const qwen3_5::PagedKVCache& cache,
                                                        const Tensor& tables, const Tensor& rows,
                                                        const Tensor& stored = {}) {
        return layer_binding(cache.batch_layer_view(0), tables, rows, stored);
    }

    // One sequence's MTP binding in a prefill or bridge: its MTP table as a one-row table, read at
    // row `row0` (I32 [1] holding 0).
    [[nodiscard]] static ops::Gemma4KvLayer sequence_mtp_binding(program::PrefillContext& state,
                                                                 const Tensor& row0) {
        if (!state.mtp_kv.valid() || state.mtp_cache == nullptr) {
            throw std::logic_error("nemotron_h MTP prefill has no MTP pool");
        }
        const Tensor row_table = state.mtp_kv.layer_view(0).block_table;
        const Tensor table(row_table.data, DType::I32, {row_table.ne[0], 1});
        return mtp_binding(*state.mtp_cache, table, row0);
    }

    // Drafts `extent` tokens for one sequence into the prefill proposal from u [H, 1], the head's
    // output at position - 1: the first by u itself, each next by one head step at the following
    // position on the previous draft (its keys and values stored there until a round rewrites
    // them).
    void propose_chain(program::ExecutionCore& core, const ops::Gemma4KvLayer& kv,
                       const Tensor& u, std::int32_t position, std::uint32_t extent) const {
        if (!mtp_pass_ || !core.io.mtp || extent == 0 ||
            extent > static_cast<std::uint32_t>(core.io.mtp->draft_tokens.ne[0])) {
            throw std::logic_error("nemotron_h MTP proposal has no head or exceeds its window");
        }
        const cudaStream_t s = core.device.stream;
        auto scope           = core.work.scope();
        const Tensor drafts  = core.io.mtp->draft_tokens;
        Tensor logits        = core.io.logits.slice(1, 0, 1);
        Tensor first         = drafts.slice(0, 0, 1);
        mtp_pass_->propose(core.work, u, logits, first, traits_.public_tokens);
        if (extent == 1) { return; }
        Tensor positions = core.work.alloc(DType::I32, {1});
        ops::set_i32_scalar(positions, position, s);
        Tensor carry = core.work.alloc(DType::BF16, {c_.hidden_size, 1});
        Tensor next  = core.work.alloc(DType::BF16, {c_.hidden_size, 1});
        CUDA_CHECK(cudaMemcpyAsync(carry.data, u.data, carry.bytes(), cudaMemcpyDeviceToDevice, s));
        for (std::uint32_t i = 1; i < extent; ++i) {
            const auto step    = static_cast<std::int32_t>(i);
            const Tensor input = drafts.slice(0, step - 1, 1);
            Tensor draft       = drafts.slice(0, step, 1);
            mtp_pass_->forward(core.work, input, carry, positions, kv, 1,
                               {.max_visible_keys = static_cast<std::uint32_t>(position + step),
                                .max_tokens       = 1},
                               next);
            mtp_pass_->propose(core.work, next, logits, draft, traits_.public_tokens);
            CUDA_CHECK(cudaMemcpyAsync(carry.data, next.data, carry.bytes(),
                                       cudaMemcpyDeviceToDevice, s));
            ops::increment_i32_scalar(positions, s);
        }
    }

    std::function<void()> mtp_body(program::MtpBatchContext& state, std::int32_t batch_size,
                                   std::uint32_t k,
                                   program::MtpCausalAttentionEnvelopes envelopes) const {
        return [this, &state, batch_size, k, envelopes] {
            if (!mtp_pass_ || state.mtp_cache == nullptr ||
                state.execution.replay_records == nullptr || batch_size <= 0 ||
                batch_size > static_cast<std::int32_t>(kMaximumConcurrency) || k == 0 ||
                k > kMaximumMtpDraftTokens) {
                throw std::logic_error("nemotron_h MTP decode batch state is incomplete");
            }
            program::ExecutionCore& core   = state.execution;
            const cudaStream_t s           = core.device.stream;
            qwen3_5::MtpDecodeState& frame = state.frame;
            const std::int32_t b           = batch_size;
            const std::int32_t width       = static_cast<std::int32_t>(k) + 1;
            const std::int32_t columns     = width * b;
            const std::int32_t H           = c_.hidden_size;
            CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                       sizeof(qwen3_5::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                       s));
            // The frame's first b rows.
            qwen3_5::MtpDecodeState fb = frame.rows(b);

            // Verify: the pending token and the current drafts of every row, B runs of k + 1
            // columns. The Mamba2 layers read the rows' source slots and record every column;
            // attention stores only each row's extent + 1 real columns.
            ops::speculative_prepare_verify_inputs(fb.anchors, fb.current_drafts, fb.base_frontiers,
                                                   fb.current_extents, fb.verify_ids,
                                                   fb.target_positions, s);
            const auto layers =
                bindings(state.text_cache, state.text_cache.batch_layer_view(0).block_tables,
                         fb.text_kv_table_rows, fb.target_valid_columns);
            const GdnReplayRecords records = core.replay_records->narrowed(width);
            {
                auto scope = core.work.scope();
                const Bindings bound{.attention = layers,
                                     .mamba     = &core.state_images.linear(),
                                     .slots     = fb.state_source_slots,
                                     .batch     = b,
                                     .records   = &records};
                const Tensor normed =
                    pass_.forward(core.work, fb.verify_ids.view({columns}),
                                  fb.target_positions.view({columns}), bound,
                                  {.max_visible_keys = envelopes.target_verify.max_visible_keys,
                                   .max_tokens       = width});
                CUDA_CHECK(cudaMemcpyAsync(fb.target_hidden.data, normed.data,
                                           fb.target_hidden.bytes(), cudaMemcpyDeviceToDevice, s));
            }
            program::mtp_verify_accept(fb, frame.sampling, H, columns, traits_.public_tokens,
                                       static_cast<std::int32_t>(state.text_cache.max_context()),
                                       state.continuation_hidden_store, core.work, s,
                                       [this](auto& work, const Tensor& hidden, Tensor& logits) {
                                           pass_.head(work, hidden, logits);
                                       });

            // Alignment: the head over every verify column, the target hidden at p paired with
            // the licensed token at p + 1; only the licensed columns store their keys and values.
            const Tensor mtp_tables = state.mtp_cache->batch_layer_view(0).block_tables;
            {
                Tensor aligned = fb.alignment_hidden.view({H, columns});
                mtp_pass_->forward(
                    core.work, fb.alignment_ids.view({columns}),
                    fb.target_hidden.view({H, columns}), fb.target_positions.view({columns}),
                    mtp_binding(*state.mtp_cache, mtp_tables, fb.mtp_kv_table_rows,
                                fb.licensed_counts),
                    b, {.max_visible_keys = envelopes.batch.max_visible_keys, .max_tokens = width},
                    aligned);
            }
            ops::speculative_select_accepted_hidden(fb.alignment_hidden, fb.accepted_drafts,
                                                    fb.ar_hidden, s);

            // The next round's drafts: the first from the fb.accepted_drafts column's head output,
            // each next by one head step on the previous draft at the following position.
            Tensor draft0 = fb.next_drafts.slice(1, 0, 1).view({b});
            mtp_pass_->propose(core.work, fb.ar_hidden, fb.proposal_logits, draft0,
                               traits_.public_tokens);
            // Each step reads one hidden buffer and writes the other, then they trade places;
            // the last step's hidden ends in fb.ar_hidden (copied once when it lands in the other).
            Tensor current = fb.ar_hidden;
            Tensor spare   = fb.next_hidden;
            for (std::uint32_t step = 0; step + 1 < k; ++step) {
                const auto i          = static_cast<std::int32_t>(step);
                const Tensor previous = fb.next_drafts.slice(1, i, 1).view({b});
                Tensor next           = fb.next_drafts.slice(1, i + 1, 1).view({b});
                const Tensor position = fb.ar_positions.slice(1, i, 1).view({b});
                const Tensor valid    = fb.ar_valid_columns.slice(1, i, 1).view({b});
                mtp_pass_->forward(
                    core.work, previous, current, position,
                    mtp_binding(*state.mtp_cache, mtp_tables, fb.mtp_kv_table_rows, valid), b,
                    {.max_visible_keys = envelopes.ar[step].max_visible_keys, .max_tokens = 1},
                    spare);
                mtp_pass_->propose(core.work, spare, fb.proposal_logits, next,
                                   traits_.public_tokens);
                std::swap(current, spare);
            }
            if (current.data != fb.ar_hidden.data) {
                CUDA_CHECK(cudaMemcpyAsync(fb.ar_hidden.data, current.data, fb.ar_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice, s));
            }
            ops::copy_to_pinned_host(&state.host_egress, frame.egress.data,
                                     sizeof(qwen3_5::MtpDecodeEgress), s);
        };
    }

    // One binding per attention layer, reading the paged pool through `tables` (row rows[b] of
    // batch row b).
    // `stored`: a verify's real columns per row (none: every column stores).
    [[nodiscard]] std::vector<ops::Gemma4KvLayer> bindings(const qwen3_5::PagedKVCache& cache,
                                                           const Tensor& tables,
                                                           const Tensor& rows,
                                                           const Tensor& stored = {}) const {
        std::vector<ops::Gemma4KvLayer> out;
        for (std::uint32_t i = 0; i < c_.count(Block::Attention); ++i) {
            out.push_back(layer_binding(cache.batch_layer_view(i), tables, rows, stored));
        }
        return out;
    }

    std::function<void()> ordinary_body(program::OrdinaryBatchContext& state,
                                        std::int32_t batch_size,
                                        ops::CausalAttentionExecutionEnvelope envelope) const {
        return program::ordinary_decode_body(
            state, batch_size, traits_.public_tokens, "nemotron_h",
            [this, &state, batch_size, envelope](program::ExecutionCore& core, const Tensor& tokens,
                                                 const Tensor& positions, const Tensor& table_rows,
                                                 const Tensor& destinations) {
                const auto layers =
                    bindings(state.text_cache, state.text_cache.batch_layer_view(0).block_tables,
                             table_rows);
                const Bindings bound{.attention = layers,
                                     .mamba     = &core.state_images.linear(),
                                     .slots     = destinations,
                                     .batch     = batch_size};
                return pass_.forward(
                    core.work, tokens, positions, bound,
                    {.max_visible_keys = envelope.max_visible_keys, .max_tokens = 1});
            },
            [this](auto& work, const Tensor& hidden, Tensor& logits) {
                pass_.head(work, hidden, logits);
            });
    }

    const Model& model_;
    const Config& c_;
    Pass pass_;
    const MtpWeights* mtp_ = nullptr;
    std::optional<MtpPass> mtp_pass_;
    bool invariant_;
    ModelTraits traits_;
};

program::PrefillChunkResult NemotronExecution::prefill_text_chunk(
    program::PrefillContext& state, std::span<const TokenId> ids, std::uint32_t nominal_length,
    std::optional<std::uint32_t> split_frontier, bool finalize_at_end) const {
    runtime::ExecutionTimingRecorder timing;
    program::ExecutionCore& core = state.execution;
    const cudaStream_t s         = core.device.stream;
    const program::PrefillTextCall call = program::prefill_text_call(
        state, ids, nominal_length, split_frontier, finalize_at_end, "nemotron_h",
        [this](std::uint32_t chunk, std::uint32_t cursor, std::uint32_t frontier) {
            return prefill_call_tokens(chunk, cursor, frontier);
        });
    const auto [base, len, t, is_last] = call;
    const ops::ScopedPrefillRoutes route_profile(true);
    const ops::ScopedInvariantRoutes invariant_routes(invariant_);

    core.work.reset();
    if (state.state_source_slot != state.state_destination_slot) {
        core.state_images.linear().copy_slot(state.state_source_slot, state.state_destination_slot,
                                             s);
    }
    auto [device_ids, positions] = program::upload_prefill_text(core, ids, call);
    // rows[0]: the one-row table's row; rows[1]: the sequence's state slot.
    Tensor rows = core.work.alloc(DType::I32, {2});
    const std::array<std::int32_t, 2> rows_host{0, state.state_destination_slot};
    CUDA_CHECK(cudaMemcpyAsync(rows.data, rows_host.data(), sizeof(rows_host),
                               cudaMemcpyHostToDevice, s));
    const Tensor row_table = state.text_kv.layer_view(0).block_table;
    const Tensor table(row_table.data, DType::I32, {row_table.ne[0], 1});
    const auto layers = bindings(state.text_cache, table, rows.slice(0, 0, 1));
    ops::set_i32_scalar(core.io.rope_delta, 0, s);

    Tensor hidden = core.prefill_hidden.slice(1, 0, t);
    {
        auto scope = core.work.scope();
        const Bindings bound{.attention = layers,
                             .mamba     = &core.state_images.linear(),
                             .slots     = rows.slice(0, 1, 1),
                             .batch     = 1};
        const Tensor normed = pass_.forward(core.work, device_ids, positions, bound,
                                            {.max_visible_keys = base + len, .max_tokens = t});
        CUDA_CHECK(cudaMemcpyAsync(hidden.data, normed.data, hidden.bytes(),
                                   cudaMemcpyDeviceToDevice, s));
    }
    const Tensor last = hidden.slice(1, t - 1, 1);
    program::finish_prefill_text(state, call, split_frontier, last, traits_.public_tokens,
                                 [this](auto& work, const Tensor& x, Tensor& logits) {
                                     pass_.head(work, x, logits);
                                 });
    if (mtp_pass_ && state.mtp_kv.valid()) {
        // The head's keys and values over the chunk: position p pairs the target hidden at p with
        // the prompt's token at p + 1, the drawn token past the prompt's end.
        auto scope                        = core.work.scope();
        const bool generated              = static_cast<std::size_t>(base) + len == ids.size();
        const std::int32_t prompt_columns = t - (generated ? 1 : 0);
        Tensor mtp_ids                    = core.work.alloc(DType::I32, {t});
        if (prompt_columns > 0) {
            CUDA_CHECK(cudaMemcpyAsync(mtp_ids.data, ids.data() + base + 1,
                                       sizeof(TokenId) * static_cast<std::size_t>(prompt_columns),
                                       cudaMemcpyHostToDevice, s));
        }
        if (generated) {
            CUDA_CHECK(cudaMemcpyAsync(mtp_ids.slice(0, t - 1, 1).data, core.io.token.data,
                                       sizeof(TokenId), cudaMemcpyDeviceToDevice, s));
        }
        const ops::Gemma4KvLayer kv = sequence_mtp_binding(state, rows.slice(0, 0, 1));
        Tensor u                    = core.work.alloc(DType::BF16, {c_.hidden_size, t});
        mtp_pass_->forward(core.work, mtp_ids, hidden, positions, kv, 1,
                           {.max_visible_keys = base + len, .max_tokens = t}, u);
        if (is_last && state.mtp_proposal_extent != 0) {
            propose_chain(core, kv, u.slice(1, t - 1, 1), static_cast<std::int32_t>(base + len),
                          state.mtp_proposal_extent);
        }
    }
    timing.begin_wait();
    core.device.synchronize();
    timing.end_wait();
    core.work.reset();
    return {.processed_tokens = len, .finalized = is_last, .timing = timing.finish()};
}

void NemotronExecution::mtp_bridge_and_propose(program::PrefillContext& state,
                                               const Tensor& next_token,
                                               const Tensor& previous_hidden,
                                               std::int32_t position, std::span<const std::int32_t>,
                                               bool build_proposal) const {
    // A reused prefix's last pair: the head at `position` on the target hidden there and the next
    // token, stored in the sequence's MTP table; then, when asked, the proposal from it.
    if (!mtp_pass_) { throw std::logic_error("nemotron_h MTP bridge has no head"); }
    const ops::ScopedInvariantRoutes routes(invariant_);
    program::ExecutionCore& core = state.execution;
    const cudaStream_t s         = core.device.stream;
    core.work.reset();
    Tensor row0 = core.work.alloc(DType::I32, {1});
    ops::set_i32_scalar(row0, 0, s);
    Tensor pos = core.work.alloc(DType::I32, {1});
    ops::set_i32_scalar(pos, position, s);
    const ops::Gemma4KvLayer kv = sequence_mtp_binding(state, row0);
    Tensor u                    = core.work.alloc(DType::BF16, {c_.hidden_size, 1});
    mtp_pass_->forward(core.work, next_token, previous_hidden.view({c_.hidden_size, 1}), pos, kv,
                       1, {.max_visible_keys = static_cast<std::uint32_t>(position + 1),
                           .max_tokens       = 1},
                       u);
    if (build_proposal) { propose_chain(core, kv, u, position + 1, state.mtp_proposal_extent); }
    core.work.reset();
}

} // namespace

StateShape state_shape(const Config& config, std::int32_t public_tokens) {
    const auto& g = config.mamba.geometry;
    return StateShape{
        .family           = StateFamily::NemotronH,
        .full_layers      = config.count(Block::Attention),
        .kv_heads         = config.kv_heads,
        .head_dim         = config.head_dim,
        .recurrent_layers = config.count(Block::Mamba),
        .value_heads      = g.heads,
        .value_dim        = g.head_dim,
        .key_heads        = g.groups,
        .key_dim          = g.state,
        .hidden           = config.hidden_size,
        .output_rows      = config.vocab_size,
        .public_tokens    = public_tokens,
    };
}

std::unique_ptr<const ModelExecution>
make_program_execution(const Model& model, const MtpWeights* mtp, DeviceContext& device,
                       std::int32_t public_tokens, ActivationMode activations) {
    return std::make_unique<const NemotronExecution>(model, mtp, device, public_tokens,
                                                     activations);
}

} // namespace ninferno::models::nemotron_h
