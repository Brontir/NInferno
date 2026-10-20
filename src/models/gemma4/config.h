#pragma once

// The instance config of a Gemma 4 target and of its two drafters, as the converter writes it
// into each artifact's text component. Parsing validates what the family's mathematics fixes;
// the pairing of a drafter with its target is checked by the drafter's loader.

#include "artifact/schema.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace ninferno::models::gemma4 {

enum class LayerKind : std::uint8_t { Sliding, Full };

struct AttentionGeometry {
    std::int32_t head_dim     = 0;
    std::int32_t kv_heads     = 0;
    float rope_theta          = 0;
    std::int32_t rotary_pairs = 0;
};

struct Config {
    std::int32_t hidden_size       = 0;
    std::int32_t vocab_size        = 0;
    std::int32_t layers            = 0;
    std::int32_t max_positions     = 0;
    std::int32_t query_heads       = 0;
    std::int32_t sliding_window    = 0;
    std::int32_t intermediate_size = 0; // the widest dense MLP of any layer
    float rms_norm_eps             = 0;
    float attention_scale          = 1;
    std::optional<float> softcap;
    bool k_eq_v = false;
    std::vector<LayerKind> kinds;
    AttentionGeometry sliding, full;
    // Mixture of experts; zero experts means a dense model.
    std::int32_t experts             = 0;
    std::int32_t experts_per_token   = 0;
    std::int32_t expert_intermediate = 0;
    // Each layer's dense MLP width (a layer that reads shared keys and values may be wider).
    std::vector<std::int32_t> intermediate;
    // The layer whose keys and values each layer attends over: itself, or (KV sharing) the last
    // layer of the same kind before the first sharing layer. A sharing layer computes queries
    // only and writes no cache.
    std::vector<std::int32_t> kv_source;
    // Per-layer embeddings (PLE): a P-wide input per layer and token, from a [ple_vocab, L*P]
    // table and a projection of the embedded token; P = 0 means none.
    // The factor on gathered token-embedding rows; 1 when the table carries it folded in.
    float embedding_scale = 1.0F;
    std::int32_t ple_width     = 0;
    std::int32_t ple_vocab     = 0;
    float ple_embedding_scale  = 0; // on the table rows (sqrt(P) in BF16)
    float ple_projection_scale = 0; // on the projection, before its norm (H^-1/2)
    float ple_input_scale      = 0; // on the sum of the two (2^-1/2)

    [[nodiscard]] const AttentionGeometry& geometry(LayerKind kind) const noexcept {
        return kind == LayerKind::Sliding ? sliding : full;
    }

    [[nodiscard]] bool moe() const noexcept { return experts > 0; }

    [[nodiscard]] bool ple() const noexcept { return ple_width > 0; }

    [[nodiscard]] bool owns_kv(std::int32_t layer) const noexcept {
        return kv_source[static_cast<std::size_t>(layer)] == layer;
    }
};

[[nodiscard]] Config parse_config(const artifact::Json& text_config);

// Google's MTP drafter for a Gemma 4 target ("assistant"): a small Gemma 4 stack that owns no
// keys or values and reads the target's last sliding and last full layer caches.
struct DrafterConfig {
    std::int32_t hidden_size       = 0;
    std::int32_t backbone_size     = 0; // the target's hidden size
    std::int32_t vocab_size        = 0;
    std::int32_t head_rows         = 0; // vocab_size, or the shortlist size of a restricted head
    std::int32_t layers            = 0;
    std::int32_t query_heads       = 0;
    std::int32_t intermediate_size = 0;
    std::int32_t sliding_window    = 0;
    float rms_norm_eps             = 0;
    std::vector<LayerKind> kinds;
    AttentionGeometry sliding, full;

    [[nodiscard]] const AttentionGeometry& geometry(LayerKind kind) const noexcept {
        return kind == LayerKind::Sliding ? sliding : full;
    }
};

[[nodiscard]] DrafterConfig parse_drafter_config(const artifact::Json& config);

// The vision tower of a 31B / 26B-A4B artifact (its "vision" component; modeling_gemma4.py
// Gemma4VisionModel): a ViT over 16-px patches with a learned 2-D position table and an axial 2-D
// rope, pooled k x k into soft tokens and projected to the text width.
struct VisionConfig {
    std::int32_t hidden_size       = 0; // 1152
    std::int32_t layers            = 0; // 27
    std::int32_t heads             = 0; // 16
    std::int32_t head_dim          = 0; // 72
    std::int32_t intermediate_size = 0; // 4304 (GeGLU)
    std::int32_t patch_size        = 0; // 16
    std::int32_t pooling           = 0; // 3: soft tokens are k x k patch means
    std::int32_t table_size        = 0; // 10240 rows per axis of the position table
    std::int32_t default_tokens    = 0; // 280 soft tokens per image
    std::int32_t output_hidden     = 0; // the text model's hidden size
    float rope_theta               = 0;
    float rms_norm_eps             = 0;
    std::int32_t boi_token         = 0; // <|image>
    std::int32_t eoi_token         = 0; // <image|>
    std::int32_t image_token       = 0; // <|image|>, one per soft token

    [[nodiscard]] std::int32_t patch_width() const noexcept { return 3 * patch_size * patch_size; }
};

[[nodiscard]] VisionConfig parse_vision_config(const artifact::Json& config);

// z-lab's DFlash drafter for a Gemma 4 target: models/speculation/dflash.h (any target runs it).

} // namespace ninferno::models::gemma4
