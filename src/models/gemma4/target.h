#pragma once

// The Gemma 4 target's forward pass: its two KV pools, every layer over a batch of tokens at given
// positions, and the head. Prefill, scoring, decode and a speculative verify all run this one pass
// and differ only in what they read from its final-normed output.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/gemma4/model.h"
#include "models/speculation/draft_target.h"
#include "ninferno/ops/gemma4_attention.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninferno::models::gemma4 {

[[nodiscard]] inline std::size_t bf16_bytes(std::int64_t elements) {
    return static_cast<std::size_t>(elements) * 2;
}

// One layer's [head_dim, capacity, kv_heads] key and value planes and the Op's view of them:
// BF16, or FP8 E4M3 codes with FP32 [capacity, kv_heads] row scales (Gemma4KvLayer's FP8 storage).
struct KvPlanes {
    DeviceBuffer k, v, k_scale, v_scale;
    ops::Gemma4KvLayer view;
};

// `window` is the sliding window of a ring layer, 0 for a full one. `fp8`: FP8 planes, about half
// the BF16 bytes.
[[nodiscard]] KvPlanes allocate_kv_planes(std::int32_t head_dim, std::int32_t capacity,
                                          std::int32_t kv_heads, std::int32_t window,
                                          bool fp8 = false);

// The activation route of a projection. A verify pass keeps every activation in BF16: it must
// compute what k+1 single-token steps would, and a W4A4 route at that width would accept drafts
// against a coarser model than plain decode runs.
enum class Activations : std::uint8_t { Recorded, A16Only };

// One projection whose transient bytes are released back to `work` on return.
void project(const Tensor& x, const Projection& p, Tensor& out, DeviceArena& work,
             cudaStream_t stream, Activations activations = Activations::Recorded);

// Where a forward copies the residual stream after chosen layers (the DFlash drafter's context
// features).
using speculation::FeatureTap;

// The keys and values one forward reads and appends: one binding per target layer. batch > 1
// runs B sequences at once: the T columns of the call are B consecutive runs of T/B tokens, run b
// at the positions pos[b*T/B .. (b+1)*T/B) through each layer's binding rows[b] (see
// ops::Gemma4KvLayer).
struct KvBindings {
    std::span<const ops::Gemma4KvLayer> layers;
    std::int32_t batch = 1;
    // The call is one image's soft tokens (batch 1): the sliding layers attend them to one another
    // (Gemma4AttentionMask::Block, use_bidirectional_attention "vision"); full layers stay causal.
    bool image_block = false;
};

// The forward pass over caches it does not own: the Program's paged pools and state-image rings,
// or a Target's own planes. It holds only the model and the side stream of the expert branch;
// every call carves its activations from the arena it is given.
class TargetPass {
public:
    TargetPass(const Model& model, DeviceContext& device);

    // Embeds the T ids already on the device and runs every layer over them at the absolute
    // positions `pos`; returns final_norm(x), the head's input, carved from `work`. `embeddings`
    // (BF16 [hidden, T]) replaces the ids' embedding rows: an image's soft tokens.
    Tensor forward(DeviceArena& work, const Tensor& ids, const Tensor& pos, KvBindings kv,
                   ops::Gemma4AttentionEnvelope envelope,
                   Activations activations = Activations::Recorded,
                   const FeatureTap* tap = nullptr, const Tensor* embeddings = nullptr) const;

    // Soft-capped head over the columns of `normed`, which already carries final_norm.
    void head(DeviceArena& work, const Tensor& normed, Tensor& out) const;

    // The head without its soft cap (see Target::uncapped_head).
    void uncapped_head(DeviceArena& work, const Tensor& normed, Tensor& out,
                       Activations activations) const;

    [[nodiscard]] const Model& model() const noexcept { return model_; }

private:
    void layers(DeviceArena& work, Tensor& x, Tensor& h, const Tensor& pos, KvBindings kv,
                ops::Gemma4AttentionEnvelope envelope, Activations activations,
                const FeatureTap* tap, const Tensor* ple) const;

    const Model& model_;
    const Config& c_;
    DeviceContext& device_;
    // The expert bank's parallel branch, and the two events that fork and join it per layer:
    // synchronization handles, not state, so a const pass records on them.
    CudaStream side_;
    mutable CudaCompletionEvent fork_, join_;
};

// The target of speculative rounds too (speculation::DraftTarget): the verify pass, the uncapped
// head and the pre-scaled token table.
class Target final : public speculation::DraftTarget {
public:
    // `work` is the session's workspace, sized with workspace_bytes() among its other users.
    // kv_fp8: FP8 key and value planes on every layer (allocate_kv_planes), or with kv_fp8_full
    // on the full layers alone (the sliding rings stay BF16, as ninferno-serve keeps them).
    Target(const Model& model, DeviceContext& device, DeviceArena& work, std::int32_t max_context,
           std::int32_t prefill_chunk, bool kv_fp8 = false, bool kv_fp8_full = false);

    // Activations of one whole chunk and of a head over `head_columns` columns, plus every Op's
    // own transient capacity taken at its worst T.
    [[nodiscard]] static std::size_t workspace_bytes(const Model& model, std::int32_t max_context,
                                                     std::int32_t prefill_chunk,
                                                     std::int32_t head_columns);

    // Embeds the T ids already on the device and runs every layer over them at the absolute
    // positions `pos`; returns final_norm(x), the head's input, carved from the workspace.
    Tensor forward(const Tensor& ids, const Tensor& pos, ops::Gemma4AttentionEnvelope envelope,
                   Activations activations = Activations::Recorded,
                   const FeatureTap* tap   = nullptr);

    // Soft-capped head over the columns of `normed`, which already carries final_norm.
    void head(const Tensor& normed, Tensor& out);

    // The head without its soft cap, for callers that only take an argmax: the cap is monotonic,
    // so the argmax of the capped and uncapped logits is the same.
    void uncapped_head(const Tensor& normed, Tensor& out, Activations activations);

    [[nodiscard]] const Model& model() const noexcept { return model_; }

    // speculation::DraftTarget.
    [[nodiscard]] speculation::TargetShape shape() const noexcept override;
    Tensor draft_forward(const Tensor& ids, const Tensor& pos,
                         ops::Gemma4AttentionEnvelope envelope, bool a16,
                         const FeatureTap* tap) override;
    void greedy_logits(const Tensor& normed, Tensor& out, bool a16) override;
    void draft_embedding(const Tensor& ids, Tensor& out) override;

    // Layer i's cache; the MTP drafter attends over the last sliding and last full layers'.
    [[nodiscard]] const ops::Gemma4KvLayer& kv(std::int32_t layer) const noexcept {
        return views_[static_cast<std::size_t>(layer)];
    }

    [[nodiscard]] std::uint64_t kv_bytes() const noexcept;

    // Debug poisoning (NINFERNO_DEBUG_POISON): overwrites every key and value
    // slot with the poison byte on the device stream, so a read of a slot the current sequence
    // has not written (outside the window, past the position, a ring slot of an earlier sequence)
    // turns into NaN instead of plausible stale data. No-op when poisoning is off.
    void poison_kv() noexcept;

private:
    const Model& model_;
    DeviceContext& device_;
    DeviceArena& work_;
    std::vector<KvPlanes> kv_;
    std::vector<ops::Gemma4KvLayer> views_;
    TargetPass pass_;
};

// NINFERNO_GEMMA4_DUMP (target.cu): whether a pass on `stream` dumps (the switch is on and the
// stream is not being captured), and appending `bytes` of device memory to <dir>/<name>.bin.
// Drafter rounds use them to expose what an eager step reads and produces.
[[nodiscard]] bool debug_dumping(cudaStream_t stream);
void debug_dump(const void* device_data, std::size_t bytes, const char* name);

} // namespace ninferno::models::gemma4
