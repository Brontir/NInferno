#pragma once

// The DFlash block-diffusion drafter (z-lab's DFlash; Meta's Muse Glimmer assistant is the same
// design): a few Qwen3-style layers (D = 128, or 64 for gpt-oss's) that draft a whole block in one
// pass. It owns no embedding and no head (it uses the target's), and keeps its own key/value cache
// of committed tokens, computed from the target's residual stream after the tapped layers. Any
// target that implements DraftTarget runs it.

#include "artifact/schema.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/speculation/draft_target.h"
#include "ninferno/ops/weight_input.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::models {
struct ArtifactBacking; // models/common/loading.h
} // namespace ninferno::models

namespace ninferno::models::speculation {

struct DFlashConfig {
    std::int32_t hidden_size       = 0; // equals the target's
    std::int32_t vocab_size        = 0; // after load(): the target's when unrecorded
    std::int32_t layers            = 0;
    std::int32_t target_layers     = 0; // likewise
    std::int32_t query_heads       = 0;
    std::int32_t kv_heads          = 0;
    std::int32_t head_dim          = 0;
    std::int32_t intermediate_size = 0;
    std::int32_t sliding_window    = 0; // 0: no sliding layer
    std::int32_t block_size        = 0;
    std::int32_t mask_token        = 0;
    // Positions the drafter addresses (max_position_embeddings) when its artifact records them;
    // 0: unrecorded, and the target's limit stands.
    std::uint32_t max_positions    = 0;
    float rope_theta               = 0;
    float rms_norm_eps             = 0;
    // The rotary embedding over the whole head: pair i turns at rope_inv_freq[i] (head_dim/2
    // values: the artifact's resolved table, e.g. YaRN, else theta^(-2i/D) as transformers
    // computes it in FP32), cos and sin scaled by rope_attention_scaling (YaRN's attention factor).
    std::vector<float> rope_inv_freq;
    float rope_attention_scaling = 1;
    bool rope_scaled             = false; // the table is the artifact's (not rope_theta's)
    // q/k/v/o projections carry biases (gpt-oss's drafter); the key and value biases apply to the
    // context keys and values as well.
    bool attention_bias = false;
    // Per layer: the sliding window, 0 for a full layer.
    std::vector<std::int32_t> windows;
    // The block attends bidirectionally inside itself on the sliding layers too (Muse Glimmer's
    // assistant, llama.cpp's DFlash graph); false: sliding layers are causal inside the block
    // (z-lab's Gemma 4 drafters). Full layers always see the whole block.
    bool sliding_block = false;
    std::vector<std::int32_t> taps; // target layers whose outputs feed the context, in order
};

// model_type "gemma4_dflash" (z-lab's Gemma 4 drafters, causal sliding layers) or "dflash" (the
// block mask named by dflash_config.sliding_attention_mask).
[[nodiscard]] DFlashConfig parse_dflash_config(const artifact::Json& config);

using Projection = ops::SingleProjectionWeight;

// Siblings the artifact packed into one plane are one projection; siblings it kept apart (a GGUF
// drafter mixes K-quant formats: q/k Q4_K, v Q6_K) are projected one by one.
struct DFlashLayer {
    std::int32_t window = 0; // 0: full
    Tensor input_norm, post_attention_norm, query_norm, key_norm;
    // With DFlashConfig::attention_bias: BF16 [Hq*D], [Hkv*D], [Hkv*D], [H]; else empty.
    Tensor query_bias, key_bias, value_bias, output_bias;
    std::optional<Projection> qkv; // rows [q; k; v] over the block, when packed
    Projection query, key, value;  // otherwise
    Projection context_key, context_value; // when DFlashWeights::context_kv is not packed
    Projection output;
    std::optional<Projection> gate_up; // rows [gate; up], when packed
    Projection gate, up, down;
};

struct DFlashWeights {
    // [H, taps*H]. After DFlashDrafter::split_feature_projection() its bytes are the slices' and
    // its data pointers are null: only its shape remains.
    Projection feature_projection;
    // The feature projection by tap, when split: slice i is the [H, H] block of columns
    // [i*H, (i+1)*H), so feature_projection(features) = sum_i slice_i(tap i's features).
    std::vector<Projection> feature_slices;
    std::optional<Projection> context_kv; // rows [k0; v0; k1; v1; ...] over the normed context
    Tensor context_norm, final_norm;
    // DFlashConfig::rope_inv_freq on the device, FP32 [head_dim/2].
    Tensor rope_inv_freq;
    std::vector<DFlashLayer> layers;
};

// Reorders a [N, parts*K] projection's device bytes in place into `parts` contiguous [N, K]
// matrices, slice i holding columns [i*K, (i+1)*K) of every row, so that the projection of
// x = [x_0; ...; x_{parts-1}] is the sum of slice i's projection of x_i. It takes a transient
// device copy of the matrix (cudaMalloc, freed before it returns) and no memory after; the
// projection keeps only its shape. BF16 (Contiguous) and the ggml block layouts whose K splits into
// whole blocks are split; for any other, a matrix that shares its storage, or a failed allocation
// of the copy, it returns no slices, changes nothing and says why in `why`.
[[nodiscard]] std::vector<Projection> split_projection_columns(Projection& projection,
                                                               std::uint64_t parts,
                                                               cudaStream_t stream,
                                                               std::string* why = nullptr);

class DFlashDrafter {
public:
    // Binds a DFlash artifact and checks it pairs with `target`: width, vocabulary, tapped layers.
    static std::unique_ptr<DFlashDrafter> load(const std::filesystem::path& path,
                                               DeviceContext& device, const TargetShape& target);

    // The same for a family's target config (any with hidden_size, vocab_size and layers).
    template <class TargetConfig>
    static std::unique_ptr<DFlashDrafter> load(const std::filesystem::path& path,
                                               DeviceContext& device, const TargetConfig& target) {
        return load(path, device,
                    TargetShape{target.hidden_size, target.vocab_size, target.layers});
    }

    ~DFlashDrafter();

    [[nodiscard]] const DFlashConfig& config() const noexcept { return config_; }

    [[nodiscard]] const DFlashWeights& weights() const noexcept { return weights_; }

    [[nodiscard]] std::uint64_t device_bytes() const noexcept;

    // Splits the feature projection by tap (DFlashWeights::feature_slices,
    // split_projection_columns), for a caller that accumulates the projection tap by tap instead
    // of keeping every tap's features. false: it stays whole, and `why` says why.
    bool split_feature_projection(DeviceContext& device, std::string* why = nullptr);

private:
    DFlashDrafter() = default;
    std::unique_ptr<ArtifactBacking> backing_;
    DFlashConfig config_;
    DFlashWeights weights_;
};

} // namespace ninferno::models::speculation
