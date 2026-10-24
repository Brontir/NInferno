#pragma once

// What a family's persistent state reads of its model: the shape its ModelExecution::
// persistent_layout lays out (the paged KV of the full layers and of the MTP layer, the sliding
// rings, the recurrent states and their replay records, a DFlash drafter's own KV, the Program's
// buffers). A family fills it from its bound config; the plan without the weights
// (runtime/engine/plan_dry_run.h) from a GGUF header, a config.json or an artifact's text config.
// Both then lay it out with the same function (sequence_bytes.h, plan_persistent_layout), whose
// per-family rules (codecs, ring capacities, convolution widths) live there alone. Plain data: no
// CUDA, no weights.

#include <cstdint>
#include <optional>

namespace ninferno::models {

// The Program family whose layout the shape takes.
enum class StateFamily : std::uint8_t {
    Qwen35,       // Qwen 3.5 / 3.6, dense and MoE (qwen3_5_text, qwen3_5_moe_text)
    DenseDecoder, // the dense decoder family (dense_decoder, moe_decoder: gpt-oss among them)
    Gemma4,       // Gemma 4 (gemma4_text)
    NemotronH,    // NemotronH hybrids (nemotron_h)
};

// A DFlash drafter's own state: its full layers' paged KV and, per StateImage slot, its sliding
// layers' ring; the target features it reads (taps x hidden rows a column).
struct DrafterStateShape {
    std::uint32_t full_layers  = 0;
    std::uint32_t local_layers = 0;
    std::uint32_t window       = 0; // the sliding layers' window (0: none)
    std::int32_t kv_heads      = 0;
    std::int32_t head_dim      = 0;
    // Rows of one pending feature column (target taps x hidden), and of one prefill feature
    // column (the same, or the hidden width when the drafter projects each tap on its own).
    std::int32_t feature_rows         = 0;
    std::int32_t prefill_feature_rows = 0;
    bool dflash2                      = false; // Qwen 3.8's DFlash2 drafter

    friend bool operator==(const DrafterStateShape&, const DrafterStateShape&) = default;
};

struct StateShape {
    StateFamily family = StateFamily::Qwen35;
    // Full-attention layers with keys and values of their own (one geometry for all of them).
    std::uint32_t full_layers = 0;
    std::int32_t kv_heads     = 0;
    std::int32_t head_dim     = 0;
    // Sliding-window layers (a ring of window + the widest call - 1 rows per StateImage slot).
    std::uint32_t sliding_layers  = 0;
    std::int32_t sliding_kv_heads = 0;
    std::int32_t sliding_head_dim = 0;
    std::uint32_t window          = 0;
    // Recurrent mixers: GatedDeltaNet (value heads x value width, key heads x key width) or Mamba2
    // (heads x head width; key_heads the groups, key_dim the state), and GatedDeltaNet's
    // convolution kernel (Mamba2's layout keeps its own three rows: 0).
    std::uint32_t recurrent_layers = 0;
    std::int32_t value_heads       = 0;
    std::int32_t value_dim         = 0;
    std::int32_t key_heads         = 0;
    std::int32_t key_dim           = 0;
    std::int32_t conv_kernel       = 0;
    // The Program's buffers: the residual width, the head's rows, the ids a request may carry.
    std::int32_t hidden        = 0;
    std::int32_t output_rows   = 0;
    std::int32_t public_tokens = 0;
    // A DFlash drafter the plan runs (none: no DFlash, or a family without one).
    std::optional<DrafterStateShape> drafter;

    friend bool operator==(const StateShape&, const StateShape&) = default;
};

} // namespace ninferno::models
