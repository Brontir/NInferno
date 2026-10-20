#include "ninferno/ops/speculative_round.h"
#include "models/qwen3_5/program/qwen3_5_execution.h"
#include "models/qwen3_5/program/graph_execution.h"
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/context.h"
#include "core/nvtx.h"
#include "ninferno/ops/mtp_round.h"
#include "ninferno/ops/scatter.h"
#include "ninferno/ops/scalar.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>

namespace ninferno::models::qwen3_5::execution {
void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, std::int32_t position,
                            std::span<const std::int32_t> rope_position, bool build_proposal,
                            const Tensor* next_embedding) {
    if (!state.mtp_kv.valid() || !state.execution.io.mtp) {
        throw std::logic_error("MTP bridge requires MTP storage");
    }
    if (rope_position.size() != 3) {
        throw std::invalid_argument("MTP bridge requires one three-axis rope position");
    }
    state.execution.work.reset();
    TextContext card(state.execution.device, qwen3_5::detail::qwen_parameters(state.execution),
                     state.execution.work, state.text_kv, state.execution.state_images.linear(),
                     state.execution.io, state.execution.prefill_hidden,
                     state.execution.prefill_chunk, state.text_kv_base, state.mtp_kv,
                     &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);

    Tensor position_view = state.execution.io.mtp->target_positions.slice(0, 0, 1);
    ops::set_i32_scalar(position_view, position, state.execution.device.stream);
    Tensor mtp_hidden         = state.execution.io.mtp->ar_hidden;
    Tensor logits             = state.execution.io.logits.slice(1, 0, 1);
    Tensor draft0             = state.execution.io.mtp->draft_tokens.slice(0, 0, 1);
    Tensor rope_position_view = state.execution.work.alloc(DType::I32, {1, 3});
    CUDA_CHECK(cudaMemcpyAsync(rope_position_view.data, rope_position.data(),
                               rope_position.size_bytes(), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    const auto bridge_visible = static_cast<std::uint32_t>(position + 1);
    const ops::CausalAttentionExecutionEnvelope bridge_envelope{bridge_visible, bridge_visible};
    card.mtp_forward_batch(next_token, previous_hidden, position_view, bridge_envelope, mtp_hidden,
                           build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr, &rope_position_view, next_embedding);
    if (!build_proposal) { return; }

    if (state.mtp_proposal_extent == 0 ||
        state.mtp_proposal_extent >
            static_cast<std::uint32_t>(state.execution.io.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP bridge proposal extent is outside the configured window");
    }

    Tensor ar_position = state.execution.io.mtp->position.slice(0, 0, 1);
    ops::set_i32_scalar(ar_position, position + 1, state.execution.device.stream);
    for (int i = 1; i < static_cast<int>(state.mtp_proposal_extent); ++i) {
        Tensor previous_token = state.execution.io.mtp->draft_tokens.slice(0, i - 1, 1);
        Tensor next_draft     = state.execution.io.mtp->draft_tokens.slice(0, i, 1);
        Tensor next_hidden    = state.execution.prefill_hidden.slice(1, i, 1);
        const auto visible    = static_cast<std::uint32_t>(position + i + 1);
        const ops::CausalAttentionExecutionEnvelope envelope{visible, visible};
        card.mtp_forward_ar_step(previous_token, state.execution.io.mtp->ar_hidden, ar_position,
                                 envelope, next_hidden, logits, next_draft);
        CUDA_CHECK(cudaMemcpyAsync(state.execution.io.mtp->ar_hidden.data, next_hidden.data,
                                   state.execution.io.mtp->ar_hidden.bytes(),
                                   cudaMemcpyDeviceToDevice, state.execution.device.stream));
        ops::increment_i32_scalar(ar_position, state.execution.device.stream);
    }
}

auto mtp_decode_batch_body(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                           MtpCausalAttentionEnvelopes envelopes) {
    return [&state, batch_size, k, envelopes] {
        const std::uint32_t depth = state.draft_depth == 0 ? k : state.draft_depth;
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kMtpDecodeMaximumDrafts || depth > std::min(k, kMaximumMtpDraftTokens) ||
            state.mtp_cache == nullptr) {
            throw std::logic_error("MTP decode batch state is incomplete");
        }

        qwen3_5::MtpDecodeState& frame = state.frame;
        const std::int32_t width       = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_5::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        TextContext card(state.execution.device, qwen3_5::detail::qwen_parameters(state.execution),
                         state.execution.work, {}, state.execution.state_images.linear(),
                         state.execution.io, state.execution.prefill_hidden,
                         state.execution.prefill_chunk, 0, {}, &state.text_cache, state.mtp_cache);
        // The frame's first batch_size rows.
        qwen3_5::MtpDecodeState fb = frame.rows(batch_size);

        ops::speculative_prepare_verify_inputs(fb.anchors, fb.current_drafts, fb.base_frontiers,
                                               fb.current_extents, fb.verify_ids,
                                               fb.target_positions, state.execution.device.stream);
        {
            nvtx::ScopedRange target_range(nvtx::Name::DecodeMtpTarget, nvtx::Category::Mtp,
                                           static_cast<std::uint64_t>(width) * batch_size);
            target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                 TargetVerifyFrameView{
                                     .ids                     = fb.verify_ids,
                                     .cache_positions         = fb.target_positions,
                                     .rope_positions          = fb.target_rope_positions,
                                     .valid_columns           = fb.target_valid_columns,
                                     .kv_table_rows           = fb.text_kv_table_rows,
                                     .state_source_slots      = fb.state_source_slots,
                                     .state_destination_slots = fb.state_destination_slots,
                                     .target_hidden           = fb.target_hidden,
                                     .target_logits           = fb.target_logits,
                                     .target_tokens           = fb.target_argmax,
                                     .drafts                  = fb.current_drafts,
                                     .current_extents         = fb.current_extents,
                                     .frontiers               = fb.base_frontiers,
                                     .anchors                 = fb.anchors,
                                     .licensed_tokens         = fb.licensed_tokens,
                                     .licensed_counts         = fb.licensed_counts,
                                     .accepted_drafts         = fb.accepted_drafts,
                                     .selected_hidden         = fb.target_continuation_hidden,
                                     .replay_records          = state.execution.replay_records,
                                     .sampling                = frame.sampling,
                                 },
                                 envelopes.target_verify);
        }

        {
            nvtx::ScopedRange draft_range(nvtx::Name::DecodeMtpDraft, nvtx::Category::Mtp,
                                          static_cast<std::uint64_t>(depth) * batch_size);
            // The drafter proposes `depth` drafts: its AR steps are the first depth - 1 columns.
            const auto ar_steps    = static_cast<std::int32_t>(std::max(depth, 2U) - 1U);
            Tensor depth_positions = fb.ar_positions.slice(1, 0, ar_steps);
            Tensor depth_rope      = fb.ar_rope_positions.slice(1, 0, ar_steps);
            Tensor depth_valid     = fb.ar_valid_columns.slice(1, 0, ar_steps);
            ops::mtp_prepare_next_round(fb.verify_ids, fb.anchors, fb.accepted_drafts,
                                        fb.base_frontiers, fb.remaining_budgets, fb.licensed_counts,
                                        fb.rope_deltas, fb.alignment_ids, fb.next_extents,
                                        depth_positions, depth_rope, depth_valid,
                                        static_cast<std::int32_t>(state.text_cache.max_context()),
                                        static_cast<std::int32_t>(depth),
                                        state.execution.device.stream);
            card.mtp_forward_decode_batch(
                fb.alignment_ids, fb.target_hidden, fb.target_positions, fb.target_rope_positions,
                fb.licensed_counts, fb.mtp_kv_table_rows, envelopes.batch, fb.alignment_hidden);
            ops::speculative_select_accepted_hidden(fb.alignment_hidden, fb.accepted_drafts,
                                                    fb.ar_hidden, state.execution.device.stream);

            Tensor proposal_logits = fb.proposal_logits;
            Tensor draft0          = fb.next_drafts.slice(1, 0, 1).view({batch_size});
            card.mtp_propose_batch(fb.ar_hidden, proposal_logits, draft0);
            // Each step reads one hidden buffer and writes the other, then they trade places;
            // the last step's hidden ends in fb.ar_hidden (copied once when it lands in the other).
            Tensor current = fb.ar_hidden;
            Tensor spare   = fb.next_hidden;
            for (std::uint32_t step = 0; step + 1 < depth; ++step) {
                Tensor previous =
                    fb.next_drafts.slice(1, static_cast<std::int32_t>(step), 1).view({batch_size});
                Tensor next = fb.next_drafts.slice(1, static_cast<std::int32_t>(step + 1), 1)
                                  .view({batch_size});
                Tensor position = fb.ar_positions.slice(1, static_cast<std::int32_t>(step), 1)
                                      .view({1, batch_size});
                Tensor rope = fb.ar_rope_positions.slice(1, static_cast<std::int32_t>(step), 1)
                                  .view({1, batch_size});
                Tensor valid = fb.ar_valid_columns.slice(1, static_cast<std::int32_t>(step), 1)
                                   .view({batch_size});
                Tensor previous_batch = previous.view({1, batch_size});
                Tensor hidden_batch =
                    current.view({dimension(qwen3_5::detail::qwen_parameters(state.execution)
                                                .model.config()
                                                .text.hidden_size),
                                  1, batch_size});
                Tensor next_hidden_batch =
                    spare.view({dimension(qwen3_5::detail::qwen_parameters(state.execution)
                                              .model.config()
                                              .text.hidden_size),
                                1, batch_size});
                card.mtp_forward_decode_batch(previous_batch, hidden_batch, position, rope, valid,
                                              fb.mtp_kv_table_rows, envelopes.ar[step],
                                              next_hidden_batch);
                card.mtp_propose_batch(spare, proposal_logits, next);
                std::swap(current, spare);
            }
            if (current.data != fb.ar_hidden.data) {
                CUDA_CHECK(cudaMemcpyAsync(fb.ar_hidden.data, current.data, fb.ar_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice,
                                           state.execution.device.stream));
            }
        }

        ops::copy_to_pinned_host(&state.host_egress, frame.egress.data,
                                 sizeof(qwen3_5::MtpDecodeEgress), state.execution.device.stream);
    };
}

void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              MtpCausalAttentionEnvelopes envelopes,
                              DecodeGraphDefinition& definition) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    capture_graph(state, definition, body);
}

void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      MtpCausalAttentionEnvelopes envelopes, DecodeGraphExecutable* executable) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    run_prepared(state, executable, body);
}

} // namespace ninferno::models::qwen3_5::execution
