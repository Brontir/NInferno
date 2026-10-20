#pragma once

// What greedy speculative decoding needs from a target model, whatever its family: one forward
// over a batch of columns that can also copy chosen layers' outputs (the DFlash drafter's context
// features), logits whose argmax is the target's greedy token, and the target's token table as a
// drafter's block reads it. Gemma 4 and the dense decoder implement it on their own Target; the
// rounds (models/speculation/decoder.h), the DFlash drafter and prompt lookup run on any of them.

#include "core/tensor.h"
#include "ninferno/ops/gemma4_attention.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace ninferno::models::speculation {

// Where a forward copies the residual stream after chosen layers: layer i's output lands in row
// block slots[i] of `sink`, one column per token; a negative slot leaves the layer untapped. With
// `project`, the forward hands each tapped layer's output BF16 [H, T] to it instead, once the
// layer's own buffers are released (a drafter that accumulates its feature projection tap by tap
// into a sink of its own shape).
struct FeatureTap {
    std::span<const std::int32_t> slots; // one per target layer
    Tensor sink;                         // BF16 [taps*H, T], or project's own [N, T]
    std::function<void(std::int32_t slot, const Tensor& output)> project;
};

// The widths a drafter must pair with.
struct TargetShape {
    std::int32_t hidden_size = 0;
    std::int32_t vocab_size  = 0;
    std::int32_t layers      = 0;
};

class DraftTarget {
public:
    virtual ~DraftTarget() = default;

    [[nodiscard]] virtual TargetShape shape() const noexcept = 0;

    // Embeds the T ids already on the device and runs every layer over them at the absolute
    // positions `pos`, appending their keys and values; returns final_norm(x), carved from the
    // session's workspace. `a16`: every projection with BF16 activations (a verify pass must
    // compute what k+1 single-token steps would). `tap`: copy the tapped layers' outputs.
    virtual Tensor draft_forward(const Tensor& ids, const Tensor& pos,
                                 ops::Gemma4AttentionEnvelope envelope, bool a16,
                                 const FeatureTap* tap) = 0;

    // Logits over the columns of `normed` whose argmax is the target's greedy token: a monotonic
    // soft cap or a positive logit multiplier may be left out.
    virtual void greedy_logits(const Tensor& normed, Tensor& out, bool a16) = 0;

    // The rows of the target's input embedding for `ids` as a drafter's block reads them: the
    // table's own values, before any embedding norm the target applies after it.
    virtual void draft_embedding(const Tensor& ids, Tensor& out) = 0;
};

// The session's device-side decode state, which every round reads and advances. The addresses
// are fixed for the session's life: captured rounds replay against them.
struct DecodeState {
    std::int32_t* token;      // the pending token
    std::int32_t* position;   // its absolute position
    std::int32_t* out_tokens; // tokens emitted by the current call
    std::int32_t* step;       // how many of them
};

struct SpeculationStats {
    std::int64_t rounds   = 0;
    std::int64_t drafted  = 0;
    std::int64_t accepted = 0;

    [[nodiscard]] double tokens_per_round() const noexcept {
        return rounds ? double(accepted + rounds) / double(rounds) : 0.0;
    }
};

// One speculative round as an audit sees it (speculate with a trace).
struct SpeculationRoundTrace {
    std::int32_t position = 0; // the device's pending position when the round opened
    std::int32_t depth    = 0; // drafts verified this round
    std::int32_t accepted = 0;
    // Every draft the drafter produced after the pending token: `depth` for a step-by-step
    // drafter, the whole block's block_size - 1 for DFlash (of which the first `depth` were
    // verified).
    std::vector<std::int32_t> drafts;
    std::vector<std::int32_t> target; // the target's argmax of each of the depth + 1 columns
    // Device time of the round's phases, from event records captured into the round's graph:
    // the draft phase, the target's forward over the verify columns, its head and argmax, and
    // acceptance with the drafter's commit. Zero unless the trace was profiled.
    float draft_ms = 0, forward_ms = 0, head_ms = 0, commit_ms = 0;
};

// The adaptive schedule's depth after a round of depth k accepted `accepted` of its drafts: deeper
// by two after a fully accepted round, one shallower otherwise, within [lo, hi].
[[nodiscard]] constexpr std::int32_t next_draft_depth(std::int32_t k, std::int32_t accepted,
                                                      std::int32_t lo, std::int32_t hi) {
    return accepted == k ? std::min(k + 2, hi) : std::max(k - 1, lo);
}

} // namespace ninferno::models::speculation
