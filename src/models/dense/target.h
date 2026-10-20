#pragma once

// The dense decoder's forward pass: one KV cache per layer (a ring on sliding layers), every layer
// over a batch of tokens at given positions, and the head. Prefill, scoring and decode all run
// this one pass and differ only in what they read from its final-normed output.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/dense/model.h"
#include "models/speculation/draft_target.h"
#include "ninferno/ops/gemma4_attention.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninferno::models::dense {

// Every projection's activations: the Use's recorded policy, or BF16 whatever it recorded.
enum class Activations : std::uint8_t { Recorded, A16Only };

// Where a forward copies the residual stream after chosen layers (a DFlash drafter's context
// features).
using speculation::FeatureTap;

// The keys and values one forward reads and appends: one binding per layer. batch > 1 runs B
// sequences at once: the T columns of the call are B consecutive runs of T/B tokens, run b at the
// positions pos[b*T/B .. (b+1)*T/B) through each layer's binding rows[b] (see ops::Gemma4KvLayer).
struct KvBindings {
    std::span<const ops::Gemma4KvLayer> layers;
    std::int32_t batch = 1;
};

// The forward pass over caches it does not own: the Program's paged pool and state-image rings,
// or a Target's own planes. It holds only the model; every call carves its activations from the
// arena it is given.
class DensePass {
public:
    DensePass(const Model& model, DeviceContext& device);

    // Activations of one call of `tokens` columns and of a head over `head_columns` columns, plus
    // every Op's own transient capacity at its worst T, for attention up to `max_context` keys.
    [[nodiscard]] static std::size_t workspace_bytes(const Model& model, std::int32_t max_context,
                                                     std::int32_t tokens,
                                                     std::int32_t head_columns);

    // Embeds the T ids already on the device and runs every layer over them at the absolute
    // positions `pos`; returns final_norm(x), the head's input, carved from `work`.
    // `session_context` is the context of the session the call belongs to (its max_context),
    // which fixes a LongRoPE group's rotation mode (RopeGroup::long_mode); a model with LongRoPE
    // refuses a call without one (0). With `tap`, each tapped layer's output is copied into its
    // row block of the sink.
    Tensor forward(DeviceArena& work, const Tensor& ids, const Tensor& pos, KvBindings kv,
                   std::int32_t session_context, ops::Gemma4AttentionEnvelope envelope,
                   Activations activations = Activations::Recorded,
                   const FeatureTap* tap   = nullptr) const;

    // The head (soft-capped when the model caps) over the columns of `normed`.
    void head(DeviceArena& work, const Tensor& normed, Tensor& out,
              Activations activations = Activations::Recorded) const;

    // The head without its logit multiplier and soft cap, for callers that only take an argmax:
    // both are monotonic, so the argmax is the same.
    void uncapped_head(DeviceArena& work, const Tensor& normed, Tensor& out,
                       Activations activations) const;

    [[nodiscard]] const Model& model() const noexcept { return model_; }

private:
    void project(DeviceArena& work, const Tensor& x, const Projection& p, Tensor& out,
                 Activations activations) const;
    // Several projections of one x, each into its own output, side by side (project_side_by_side,
    // target.cu); false, with nothing launched, where they must run apart.
    bool project_side_by_side(DeviceArena& work, const Tensor& x,
                              std::span<const Projection* const> projections,
                              std::span<Tensor* const> outputs, Activations activations) const;
    // The dense feed-forward (gate/up, activation, down) of one layer into `branch`, or (add)
    // added into it in the down projection's epilogue.
    // With `release`, the down projection into the branch may release the next kernel: the
    // return says whether it did (ops::linear_release_next), and that kernel must wait.
    bool dense_mlp(DeviceArena& work, const Layer& layer, const Tensor& mlp_input, Tensor& branch,
                   std::int32_t t, Activations activations, bool add = false,
                   bool release = false) const;
    bool project_release_next(DeviceArena& work, const Tensor& x, const Projection& p, Tensor& out,
                              Activations activations) const;

    const Model& model_;
    const Config& c_;
    DeviceContext& device_;
};

// The target of speculative rounds too (speculation::DraftTarget): the verify pass, the uncapped
// head and the token table.
class Target final : public speculation::DraftTarget {
public:
    // `work` is the session's workspace, sized with workspace_bytes() among its other users.
    // kv_fp8: FP8 keys and values with a scale per row (Gemma4KvLayer's FP8 storage), half the
    // BF16 cache bytes.
    Target(const Model& model, DeviceContext& device, DeviceArena& work, std::int32_t max_context,
           std::int32_t prefill_chunk, bool kv_fp8 = false);

    // Activations of one whole chunk and of a head over `head_columns` columns, plus every Op's
    // own transient capacity at its worst T.
    [[nodiscard]] static std::size_t workspace_bytes(const Model& model, std::int32_t max_context,
                                                     std::int32_t prefill_chunk,
                                                     std::int32_t head_columns);

    // Embeds the T ids already on the device and runs every layer over them at the absolute
    // positions `pos`; returns final_norm(x), the head's input, carved from the workspace.
    Tensor forward(const Tensor& ids, const Tensor& pos, ops::Gemma4AttentionEnvelope envelope,
                   Activations activations = Activations::Recorded,
                   const FeatureTap* tap   = nullptr);

    // The head (soft-capped when the model caps) over the columns of `normed`.
    void head(const Tensor& normed, Tensor& out, Activations activations = Activations::Recorded);

    [[nodiscard]] std::uint64_t kv_bytes() const noexcept;

    // speculation::DraftTarget.
    [[nodiscard]] speculation::TargetShape shape() const noexcept override;
    Tensor draft_forward(const Tensor& ids, const Tensor& pos,
                         ops::Gemma4AttentionEnvelope envelope, bool a16,
                         const FeatureTap* tap) override;
    void greedy_logits(const Tensor& normed, Tensor& out, bool a16) override;
    void draft_embedding(const Tensor& ids, Tensor& out) override;

private:
    struct KvPlanes {
        DeviceBuffer k, v, k_scale, v_scale; // scales: FP8 planes only
    };

    const Model& model_;
    DeviceContext& device_;
    DeviceArena& work_;
    std::int32_t max_context_;
    std::vector<KvPlanes> kv_;
    std::vector<ops::Gemma4KvLayer> views_;
    DensePass pass_;
};

} // namespace ninferno::models::dense
