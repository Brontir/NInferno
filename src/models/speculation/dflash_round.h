#pragma once

// The draft phase of a speculative round with a DFlash drafter: one pass drafts a whole block
// over the drafter's own key/value cache of committed tokens. That cache is built from the
// target's residual stream after the tapped layers, captured by every forward that commits
// tokens: each prefill chunk and each verify pass.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/speculation/dflash.h"
#include "models/speculation/draft_round.h"
#include "models/speculation/draft_target.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninferno::models::speculation {

class DFlashRound final : public DraftRound {
public:
    // `work` is the session's workspace, sized with workspace_bytes() on top of the target's.
    DFlashRound(const DFlashDrafter& drafter, DraftTarget& target, DeviceContext& device,
                DeviceArena& work, std::int32_t max_context, std::int32_t prefill_chunk);

    // What a context append over a whole chunk and a draft pass hold at once beyond the target's
    // own chunk: projected and normed context, all layers' key/value rows, one layer's prepared
    // keys and values, and the draft block's activations and logits.
    [[nodiscard]] static std::size_t workspace_bytes(const DFlashDrafter& drafter,
                                                     std::int32_t max_context,
                                                     std::int32_t prefill_chunk);

    // The target's forward over a prefill chunk. Every prompt token is committed context: the
    // forward captures the tapped layers' outputs and the drafter appends the chunk's keys and
    // values. Returns the forward's final-normed output.
    Tensor prefill(const Tensor& ids, const Tensor& pos, ops::Gemma4AttentionEnvelope envelope);

    // Opens a round from the device-side position P of the pending `token` (the verify batch at
    // `verify_positions` covers P..P+k) and drafts the rest of the block after it into ids().
    void draft(const std::int32_t* position, const std::int32_t* token,
               std::int32_t* verify_positions, std::int32_t k) override;

    // I32 [columns]: the pending token, then the drafts; the verify batch's ids.
    [[nodiscard]] Tensor ids(std::int32_t columns) const override;

    // A draft pass always drafts the whole block.
    [[nodiscard]] std::int32_t drafted(std::int32_t) const override { return block_size() - 1; }

    // The verify forward captures the tapped layers; commit() appends the verified columns.
    [[nodiscard]] std::optional<FeatureTap> tap(std::int32_t columns) const override {
        return capture(columns);
    }

    void commit(const FeatureTap& tap, const Tensor& pos) override {
        append_context(tap.sink, pos);
    }

    // Positions one draft pass covers: the pending token and block_size() - 1 drafts.
    [[nodiscard]] std::int32_t block_size() const noexcept { return drafter_.config().block_size; }

    // The capture of a forward over `t` tokens into the drafter's features.
    [[nodiscard]] FeatureTap capture(std::int32_t t) const;

    // Keys and values of committed tokens for every drafter layer, from their captured features:
    // context_norm(feature_projection(features)), then each layer's key/value rows of one packed
    // projection, key_norm and RoPE on the keys, appended at the tokens' positions.
    void append_context(const Tensor& features, const Tensor& pos);

private:
    // Positions the drafter's cache must address. A round opens at P <= max_context - 2 (the
    // session admits its verify columns P..P+k) but always drafts the whole block P..P+block-1,
    // whose tail can pass the session's last position: the drafter's full layers and its
    // attention envelope take a block of slack, so those keys never wrap onto committed context.
    [[nodiscard]] static std::int32_t key_span(const DFlashDrafter& drafter,
                                               std::int32_t max_context);

    // One draft pass: the block [pending token, mask...] at positions P..P+block-1 through every
    // drafter layer over the committed context, then the target's head on columns 1.. — ids()[i]
    // for i >= 1 is the drafter's token for position P+i.
    void block_pass();

    struct Planes {
        DeviceBuffer k, v;
        ops::Gemma4KvLayer view;
    };

    const DFlashDrafter& drafter_;
    DraftTarget& target_;
    TargetShape shape_;
    DeviceContext& device_;
    DeviceArena& work_;
    std::int32_t max_context_;
    std::vector<Planes> kv_;             // every committed position
    DeviceBuffer draft_ids_;             // I32 [block]: pending token, then the drafts
    DeviceBuffer features_;              // BF16 [taps*H, max(chunk, block)]
    DeviceBuffer block_ids_;             // I32 [block]: the pending token, then mask tokens
    DeviceBuffer block_positions_;       // I32 [block]: P..P+block-1
    std::vector<std::int32_t> tap_slot_; // per target layer: its row block in the features, or -1
};

} // namespace ninferno::models::speculation
