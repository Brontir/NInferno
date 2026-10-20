#pragma once

// The prepared parameters of a Gemma 4 target and of its two drafters: every projection already
// in the form its Op consumes, every norm weight a device vector. They borrow device storage from
// the loaded model's materialized artifact and live exactly as long as it.

#include "core/expert_residency.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/gemma4/config.h"
#include "ninferno/ops/gemma4_moe.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/weight_input.h"

#include <optional>
#include <vector>

namespace ninferno::models::gemma4 {

// One projection as the Op consumes it: the shared weight-plus-policy pair.
using Projection = ops::SingleProjectionWeight;

struct Layer {
    LayerKind kind = LayerKind::Sliding;
    Tensor input_norm, post_attention_norm, pre_ffn_norm, post_ffn_norm;
    Tensor query_norm, key_norm;
    // Either one plane whose rows are [q; k; v] (sliding) or [q; k] (full), or q and k apart.
    // [q; k; v] in one plane, or [q; k] with v apart (a GGUF layer whose v takes a wider type),
    // or [q; k] alone under k_eq_v; q, k and v apart when the artifact could not pack them.
    std::optional<Projection> qkv;
    Projection query, key, value; // value: beside a [q; k] qkv too; unused under k_eq_v
    Projection output;
    Projection gate_up, down;
    float output_scale = 1;
    // Expert branch (26B-A4B only), and the activation encoding of its prefill route.
    Tensor pre_moe_norm, post_mlp_norm, post_moe_norm;
    std::optional<ops::Gemma4MoeWeights> moe;
    ops::Gemma4MoeActivations moe_activations = ops::Gemma4MoeActivations::Recorded;
    // Experts in any other format (a GGUF's ggml blocks) take the generic MoE Op; its router reads
    // rmsnorm(x) * router_norm, router_norm = BF16(router_scale * hidden^-0.5).
    std::optional<ops::MoeWeights> generic_moe;
    ops::MoeConfig generic_moe_config;
    Tensor router_norm;
    [[nodiscard]] bool has_moe() const noexcept { return moe.has_value() || generic_moe.has_value(); }
    // Per-layer embedding block (Config::ple()): gate [P, H] over the residual, output [H, P].
    Projection ple_gate, ple_output;
    Tensor post_ple_norm;
};

struct Weights {
    Weight token_embedding; // pre-scaled by sqrt(hidden_size) at conversion
    Projection output_head; // tied to the embedding source, unscaled
    Tensor final_norm;
    std::vector<Layer> layers;
    // MoE expert offload: where each expert-bank call reads its bank (null when off), and the
    // CPU that computes the offloaded layers' experts in decode calls (null: the GPU does).
    ExpertResidency* experts       = nullptr;
    ops::HostExperts* host_experts = nullptr;
    // Per-layer embeddings (Config::ple()): the [vocab, L*P] table as published (unscaled), the
    // [L*P, H] projection of the embedded token, and the norm of each P-wide projection slice.
    Weight ple_table;
    Projection ple_projection;
    Tensor ple_norm;
};

// The vision tower's parameters (VisionConfig). q and k rows carry the converter's half-head swap
// for the engine's Vision 2-D rope; gate_up is one [2I, H] plane (gate rows first).
struct VisionLayer {
    Tensor input_norm, post_attention_norm, pre_ffn_norm, post_ffn_norm;
    Tensor query_norm, key_norm; // [head_dim], shared by every head
    std::optional<Projection> qkv; // [q; k; v] in one plane when the artifact packed them
    Projection query, key, value, output;
    Projection gate_up, down;
};

struct VisionWeights {
    Projection patch_embedding; // [H, 3 p^2] over a patch's (y, x, channel) raster
    Tensor position_table;      // BF16 [2 * table, H]: the x rows, then the y rows
    Tensor std_bias, std_scale; // [H]
    std::vector<VisionLayer> layers;
    Projection projection; // [text hidden, H]
};

struct DrafterLayer {
    LayerKind kind = LayerKind::Sliding;
    Tensor input_norm, post_attention_norm, pre_ffn_norm, post_ffn_norm, query_norm;
    Projection query, output, gate_up, down;
    float output_scale = 1;
};

struct DrafterWeights {
    Projection pre_projection, post_projection, head;
    Tensor head_token_ids; // I32 [head_rows] vocabulary id of each head row; empty for a full head
    Tensor final_norm;
    std::vector<DrafterLayer> layers;
};

} // namespace ninferno::models::gemma4
