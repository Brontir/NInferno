#pragma once

// The draft phase of a speculative round with Google's MTP drafter: k single-token steps, each
// attending over the target's last sliding and last full layer caches from the pending token's
// position and handing its post-projected state to the next step.

#include "models/gemma4/model.h"
#include "models/gemma4/target.h"
#include "models/speculation/draft_round.h"

#include <cstdint>

namespace ninferno::models::gemma4 {

// The drafter's step over caches it does not own: B rows at once, each attending the target's
// last sliding and last full layer through its binding row (ops::Gemma4KvLayer). A step consumes
// one token and one hidden state per row and yields the next draft and the state the next step
// reads.
class DrafterPass {
public:
    DrafterPass(const Drafter& drafter, const Model& target, DeviceContext& device);

    // Activations and every Op's transient capacity of one step over `batch` rows.
    [[nodiscard]] static std::size_t workspace_bytes(const Drafter& drafter,
                                                     std::int32_t max_context, std::int32_t batch);

    // The target layers the drafter reads: its last sliding and last full layer.
    [[nodiscard]] std::int32_t sliding_source() const noexcept { return sliding_source_; }

    [[nodiscard]] std::int32_t full_source() const noexcept { return full_source_; }

    // tokens, next: I32 [B]; hidden_in, carry_out: BF16 [H, B] (they may alias: the input is
    // consumed before the carry is written). query_positions rotate the queries; each row sees the
    // target's keys through key_positions[b]. Both I32 [B]. dump_step >= 0 names the step for
    // NINFERNO_GEMMA4_DUMP (an eager step dumps its input and logits as D<dump_step>).
    void step(DeviceArena& work, const Tensor& tokens, const Tensor& hidden_in,
              const Tensor& query_positions, const Tensor& key_positions,
              const ops::Gemma4KvLayer& sliding, const ops::Gemma4KvLayer& full,
              ops::Gemma4AttentionEnvelope envelope, Tensor& next, Tensor& carry_out,
              std::int32_t dump_step = -1) const;

private:
    const Drafter& drafter_;
    const Model& target_;
    DeviceContext& device_;
    std::int32_t sliding_source_ = -1, full_source_ = -1;
};

class MtpRound final : public speculation::DraftRound {
public:
    // Buffers for rounds of up to `max_drafts` drafts; `work` is the session's workspace.
    MtpRound(const Drafter& drafter, const Target& target, DeviceContext& device, DeviceArena& work,
             std::int32_t max_context, std::int32_t max_drafts);

    // Opens a round from the device-side position P of the pending `token` (the verify batch at
    // `verify_positions` covers P..P+k) and drafts k tokens after it into ids().
    void draft(const std::int32_t* position, const std::int32_t* token,
               std::int32_t* verify_positions, std::int32_t k) override;

    // I32 [columns]: the pending token, then the drafts; the verify batch's ids.
    [[nodiscard]] Tensor ids(std::int32_t columns) const override;

    // BF16 [H, 1]: the target's final-normed state behind the pending token, which the first
    // draft step reads. The prefill writes it, and so does every accepted round.
    [[nodiscard]] Tensor carry() const override;

private:
    const Target& target_;
    DeviceContext& device_;
    DeviceArena& work_;
    std::int32_t max_context_;
    DrafterPass pass_;
    DeviceBuffer draft_ids_;    // I32 [k+1]: pending token, then drafts
    DeviceBuffer carry_hidden_; // BF16 [H]
    DeviceBuffer query_position_, key_position_;
    DeviceBuffer draft_hidden_; // BF16 [H]: post-projection output carried between draft steps
};

} // namespace ninferno::models::gemma4
