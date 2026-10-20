#pragma once

// The instance config of a dense decoder (the Llama lineage: Llama, Mistral/Ministral, Granite,
// OLMo, Seed-OSS, Qwen2/Qwen3 dense, SmolLM3, ...) as the converter writes it into the artifact's
// text component (model_type "dense_decoder"). The differences between those models are fields
// here, so the family's mathematics needs no model name (a few tuned launch choices still name
// the model they were measured on). The converter resolves each checkpoint's
// conventions into these fields (tools and src/convert/dense.cpp); parsing validates what the
// family's mathematics and Ops admit.

#include "artifact/schema.h"
#include "ninferno/ops/dense_fused.h"
#include "ninferno/ops/moe.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::models::dense {

// Where a layer's RMSNorms sit around its two sublayers f (attention, then the MLP):
//   Pre      x += f(norm(x))                  (Llama, Mistral, Granite, Seed-OSS, Qwen)
//   Post     x += norm(f(x))                  (OLMo 2/3)
//   Sandwich x += post_norm(f(pre_norm(x)))   (Gemma 2/3-style)
enum class NormPlacement : std::uint8_t { Pre, Post, Sandwich };

enum class Activation : std::uint8_t { SiLU, GeluTanh };

// One rotary embedding: the frequency of each rotated pair (HF's inv_freq, resolved from the
// checkpoint's rope type - default, linear, llama3, yarn, longrope, ... - at conversion) and the
// factor that scales cos and sin (YaRN's and LongRoPE's attention factor).
//
// LongRoPE has two modes: transformers rotates a sequence of up to long_after positions with the
// short frequencies (inv_freq) and a longer one with long_inv_freq. A session fixes the mode by its
// capacity (llama.cpp's rule): a context of at most long_after positions runs the short mode, so
// every sequence it holds rotates exactly as in transformers, and a longer context the long mode
// throughout (as transformers does once a sequence passes long_after). Rotation then depends on
// the position alone, whatever the chunking.
struct RopeGroup {
    std::string name;
    std::int32_t rotary_dim = 0;
    std::vector<float> inv_freq;
    float attention_scaling = 1;
    std::vector<float> long_inv_freq; // empty: one mode
    std::int32_t long_after = 0;

    // Whether a session of `max_context` positions runs the long mode.
    [[nodiscard]] bool long_mode(std::int32_t max_context) const noexcept {
        return !long_inv_freq.empty() && max_context > long_after;
    }
};

// The sparse feed-forward of a MoE decoder (model_type "moe_decoder"): every layer's MLP is a
// routed expert bank, optionally with one shared expert. Each field is one of the generic MoE Op's
// knobs (ninferno/ops/moe.h); the converter resolves the checkpoint's conventions into them.
struct MoeSpec {
    std::int32_t experts      = 0;
    std::int32_t top_k        = 0;
    std::int32_t inter        = 0; // one expert's intermediate width F
    std::int32_t shared_inter = 0; // 0: no shared expert
    bool shared_gate          = false;
    bool router_bias          = false;
    bool expert_bias          = false;
    ops::MoeConfig op;
};

struct LayerSpec {
    std::int32_t window = 0; // sliding window; 0: full attention
    std::int32_t rope   = -1; // index into rope_groups; -1: no rotary embedding (NoPE)
};

struct Config {
    std::int32_t hidden_size       = 0;
    std::int32_t vocab_size        = 0;
    std::int32_t layers            = 0;
    std::int32_t max_positions     = 0;
    std::int32_t query_heads       = 0;
    std::int32_t kv_heads          = 0;
    std::int32_t head_dim          = 0;
    std::int32_t intermediate_size = 0;
    float rms_norm_eps             = 0;
    Activation activation          = Activation::SiLU;
    NormPlacement norms            = NormPlacement::Pre;
    // Every layer RMSNorm's gain is 1 + weight (Gemma), not the weight.
    bool norm_unit_offset          = false;
    // The final norm's gain convention (it may differ from the layers', Muse Glimmer).
    bool final_norm_unit_offset    = false;
    // The epsilon of the post-norms of a sandwich or post placement (rms_norm_eps unless given).
    float post_norm_eps            = 0;
    ops::QkNorm qk_norm            = ops::QkNorm::None;
    // The q/k norm has a learned gain; false: a norm without weight (gain 1).
    bool qk_norm_scaled            = true;
    // The embedded token is RMS-normalized without weight before the first layer.
    bool embedding_norm            = false;
    // The attention context is gated before the output projection: context * sigmoid(W_g x),
    // W_g [Hq*D, H] reading the attention input.
    bool attention_output_gate     = false;
    // Logits are multiplied by this before the soft cap (1 without one).
    float logit_multiplier         = 1;
    bool qkv_bias                  = false;
    bool output_bias               = false;
    // Scores are scale * q.k (1/sqrt(head_dim) unless the checkpoint says otherwise).
    float attention_scale = 1;
    // x += residual_multiplier * branch (Granite's muP multiplier; 1 elsewhere).
    float residual_multiplier = 1;
    std::optional<float> softcap; // final logit soft cap
    // Query temperature by position (Llama 4 / Mistral 3): period 0 is off.
    float query_beta          = 0;
    std::int32_t query_period = 0;
    std::vector<LayerSpec> layer_specs;
    std::vector<RopeGroup> rope_groups;
    // A MoE decoder: the routed feed-forward (intermediate_size is then unused).
    std::optional<MoeSpec> moe;
    // Per-head attention sink logits in every layer (gpt-oss).
    bool attention_sinks = false;

    [[nodiscard]] bool pre_norms() const noexcept { return norms != NormPlacement::Post; }
    [[nodiscard]] bool post_norms() const noexcept { return norms != NormPlacement::Pre; }
    [[nodiscard]] std::int32_t max_window() const noexcept;
};

[[nodiscard]] Config parse_config(const artifact::Json& text_config);

} // namespace ninferno::models::dense
