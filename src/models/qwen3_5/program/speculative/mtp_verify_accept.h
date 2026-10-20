#pragma once

// The verify tail every MTP family under the Program shares: the head over the verify columns,
// greedy acceptance, the accepted hidden kept for the rows' continuations, and the next round's
// inputs.

#include "core/tensor.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "ninferno/ops/argmax.h"
#include "ninferno/ops/mtp_round.h"
#include "ninferno/ops/scatter.h"
#include "ninferno/ops/speculative_round.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace ninferno::models::qwen3_5::execution {

// `fb` is the frame's batch rows (MtpDecodeState::rows) with the verify's hidden in
// fb.target_hidden ([hidden_size, columns]); head(work, hidden, logits) is the family's head.
// `depth`: the drafts the drafter proposes next (below the window's k in a copy window), whose
// first depth - 1 columns are its AR steps; 0 is the window's own depth.
template <class Work, class Head>
void mtp_verify_accept(MtpDecodeState& fb, const ops::SamplingConfig* sampling,
                       std::int32_t hidden_size, std::int32_t columns, std::int32_t public_tokens,
                       std::int32_t max_context, Tensor& continuation_hidden_store, Work& work,
                       cudaStream_t s, Head head, std::int32_t depth = 0) {
    {
        auto scope    = work.scope();
        Tensor logits = fb.target_logits.view({fb.target_logits.ne[0], columns});
        head(work, fb.target_hidden.view({hidden_size, columns}), logits);
        Tensor argmax = fb.target_argmax.view({columns});
        ops::argmax(logits, argmax, public_tokens, s);
    }
    {
        auto scope = work.scope();
        ops::speculative_accept_greedy_drafts(fb.target_argmax, fb.target_logits, fb.current_drafts,
                                              fb.current_extents, fb.base_frontiers, fb.anchors,
                                              fb.licensed_tokens, fb.licensed_counts,
                                              fb.accepted_drafts, public_tokens, sampling, work, s);
    }
    ops::speculative_select_accepted_hidden(fb.target_hidden, fb.accepted_drafts,
                                            fb.target_continuation_hidden, s);
    ops::scatter(fb.target_continuation_hidden, fb.state_destination_slots,
                 continuation_hidden_store, s);
    if (depth == 0) {
        ops::mtp_prepare_next_round(fb.verify_ids, fb.anchors, fb.accepted_drafts,
                                    fb.base_frontiers, fb.remaining_budgets, fb.licensed_counts,
                                    fb.rope_deltas, fb.alignment_ids, fb.next_extents,
                                    fb.ar_positions, fb.ar_rope_positions, fb.ar_valid_columns,
                                    max_context, s);
        return;
    }
    const auto ar_steps    = std::max(depth, 2) - 1;
    Tensor depth_positions = fb.ar_positions.slice(1, 0, ar_steps);
    Tensor depth_rope      = fb.ar_rope_positions.slice(1, 0, ar_steps);
    Tensor depth_valid     = fb.ar_valid_columns.slice(1, 0, ar_steps);
    ops::mtp_prepare_next_round(fb.verify_ids, fb.anchors, fb.accepted_drafts, fb.base_frontiers,
                                fb.remaining_budgets, fb.licensed_counts, fb.rope_deltas,
                                fb.alignment_ids, fb.next_extents, depth_positions, depth_rope,
                                depth_valid, max_context, depth, s);
}

} // namespace ninferno::models::qwen3_5::execution
