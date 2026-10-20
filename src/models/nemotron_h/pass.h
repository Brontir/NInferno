#pragma once

// The NemotronH forward pass over state it does not own: the attention layers' keys and values in
// the Program's paged pool, the Mamba2 layers' convolution windows and recurrent states in the
// Program's linear state pool. It holds only the model; every call carves its activations from
// the arena it is given.

#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "models/nemotron_h/model.h"
#include "ninferno/ops/gemma4_attention.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninferno::models::nemotron_h {

// What one forward reads and updates. A call of T columns runs `batch` rows of T / batch tokens;
// row b's attention binding row is attention[i].rows[b] and its Mamba2 state is slot slots[b].
//
// A speculative verify passes `records` (at the call's row length): the Mamba2 layers then only
// read their slots and record every column (Mamba2 layer i at record layer i, in the planes
// mamba2_records_spec names), and replay_fold later commits the accepted columns.
struct Bindings {
    std::span<const ops::Gemma4KvLayer> attention; // one per attention layer, by its index
    LinearAttentionStatePool* mamba = nullptr;     // one pool layer per Mamba2 layer
    Tensor slots;                                  // I32 [batch]
    std::int32_t batch = 1;
    const GdnReplayRecords* records = nullptr;
};

// The replay records of a Mamba2 stack, in the Program's record planes: conv = the conv inputs
// (xBC before the convolution) [C, width, rows], value = the conv outputs [C, 1, width, rows],
// key = dt [H, 1, width, rows] (the gate plane is unused).
[[nodiscard]] GdnReplayRecordSpec mamba2_records_spec(const Config& c, std::int32_t rows,
                                                      std::int32_t width);

// One NoPE attention sub-layer (the trunk's and the MTP head's): input [H, T] (normed), keys and
// values appended through `kv`, the output projection added into residual [H, T] in its epilogue.
void attention_block(DeviceArena& work, const Config& c, const AttentionWeights& a,
                     const Tensor& input, const Tensor& pos, const ops::Gemma4KvLayer& kv,
                     std::int32_t batch, ops::Gemma4AttentionEnvelope envelope, Tensor& residual,
                     cudaStream_t stream);

class Pass {
public:
    Pass(const Model& model, DeviceContext& device);

    // Activations of one call of `tokens` columns and a head over `head_columns` columns, plus
    // every Op's own transient capacity at its worst T, for attention up to `max_context` keys.
    [[nodiscard]] static std::size_t workspace_bytes(const Model& model, std::int32_t max_context,
                                                     std::int32_t tokens,
                                                     std::int32_t head_columns);

    // Embeds the T ids on the device and runs every block at the absolute positions `pos`;
    // returns final_norm(x), carved from `work`.
    Tensor forward(DeviceArena& work, const Tensor& ids, const Tensor& pos, const Bindings& state,
                   ops::Gemma4AttentionEnvelope envelope) const;

    void head(DeviceArena& work, const Tensor& normed, Tensor& out) const;

private:
    void project(DeviceArena& work, const Tensor& x, const Projection& p, Tensor& out) const;
    // The mixers add their output projection into `residual` (the projection's epilogue).
    void mamba(DeviceArena& work, const Layer& layer, const Tensor& input, const Bindings& state,
               Tensor& residual) const;
    void attention(DeviceArena& work, const Layer& layer, const Tensor& input, const Tensor& pos,
                   const Bindings& state, ops::Gemma4AttentionEnvelope envelope,
                   Tensor& residual) const;

    const Model& model_;
    const Config& c_;
    DeviceContext& device_;
};

/**
 * The MTP head's forward. For T columns (batch rows of T / batch):
 *
 *   u   = eh_proj . [rmsnorm(embed(ids), enorm); rmsnorm(hidden, hnorm)]
 *   u  += attention(rmsnorm(u, attn_norm))         (NoPE, keys and values through `kv`)
 *   u  += moe(rmsnorm(u, post_attention_norm))
 *
 * `hidden` is the previous position's hidden: the target's final-normed hidden for the first step,
 * the head's own u after that. propose() then takes LayerNorm(u, final_norm) through the target's
 * LM head to the argmax token. The RMSNorms take the drafter's rms_norm_eps, the LayerNorm its
 * layer_norm_epsilon (MtpWeights), as llama.cpp does.
 */
class MtpPass {
public:
    MtpPass(const Model& target, const MtpWeights& mtp, DeviceContext& device);

    // Activations and every Op's transient capacity of a call of `tokens` columns with keys up to
    // `max_context`, and of propose() over `tokens` columns.
    [[nodiscard]] static std::size_t workspace_bytes(const Model& target, const MtpWeights& mtp,
                                                     std::int32_t max_context,
                                                     std::int32_t tokens);

    void forward(DeviceArena& work, const Tensor& ids, const Tensor& hidden, const Tensor& pos,
                 const ops::Gemma4KvLayer& kv, std::int32_t batch,
                 ops::Gemma4AttentionEnvelope envelope, Tensor& out) const;

    // logits [vocab, B] and the argmax tokens [B] (over the first public_tokens rows) of u [H, B].
    void propose(DeviceArena& work, const Tensor& u, Tensor& logits, Tensor& tokens,
                 std::int32_t public_tokens) const;

private:
    const Model& target_;
    const MtpWeights& mtp_;
    const Config& c_;
    DeviceContext& device_;
};

} // namespace ninferno::models::nemotron_h
