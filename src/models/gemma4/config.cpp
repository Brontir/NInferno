#include "models/gemma4/config.h"

#include "models/common/config_json.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninferno::models::gemma4 {
namespace {

float positive_real(const artifact::Json& json, const char* key) {
    return json_positive_float(json, key);
}

AttentionGeometry geometry(const artifact::Json& config, const char* kind) {
    const auto& shape = config.at("attention_geometry").at(kind);
    const auto& rope  = config.at("rope_parameters").at(kind);
    AttentionGeometry out;
    out.head_dim     = json_dimension_i32(shape, "head_dim");
    out.kv_heads     = json_dimension_i32(shape, "num_key_value_heads");
    out.rope_theta   = positive_real(rope, "rope_theta");
    out.rotary_pairs = json_dimension_i32(rope, "rotary_pairs");
    if (json_dimension_i32(rope, "head_dim") != out.head_dim || out.rotary_pairs > out.head_dim / 2) {
        throw std::invalid_argument(std::string("gemma4 config: inconsistent RoPE for ") + kind);
    }
    return out;
}

} // namespace

Config parse_config(const artifact::Json& c) {
    if (c.value("model_type", std::string()) != "gemma4_text") {
        throw std::invalid_argument("gemma4 config: model_type must be gemma4_text");
    }
    Config out;
    out.hidden_size       = json_dimension_i32(c, "hidden_size");
    out.vocab_size        = json_dimension_i32(c, "vocab_size");
    out.layers            = json_dimension_i32(c, "num_hidden_layers");
    out.max_positions     = json_dimension_i32(c, "max_position_embeddings");
    out.query_heads       = json_dimension_i32(c, "num_attention_heads");
    out.intermediate_size = json_dimension_i32(c, "intermediate_size");
    out.rms_norm_eps      = positive_real(c, "rms_norm_eps");
    out.attention_scale   = positive_real(c, "attention_scale");
    out.k_eq_v            = c.value("attention_k_eq_v", false);
    // A table kept as its publisher quantized it carries no folded scale: its rows are scaled
    // after the gather (sqrt(H) in BF16).
    if (c.value("embedding_scale_at_run_time", false)) {
        out.embedding_scale = positive_real(c, "embedding_scale");
    }
    if (c.contains("final_logit_softcapping")) {
        out.softcap = positive_real(c, "final_logit_softcapping");
    }
    if (!c.value("tie_word_embeddings", false)) {
        throw std::invalid_argument("gemma4 config: only the tied output head is implemented");
    }
    const auto& kinds = c.at("layer_types");
    if (!kinds.is_array() || static_cast<std::int32_t>(kinds.size()) != out.layers) {
        throw std::invalid_argument("gemma4 config: layer_types must cover every layer");
    }
    for (const auto& kind : kinds) {
        const auto name = kind.get<std::string>();
        if (name == "sliding_attention") {
            out.kinds.push_back(LayerKind::Sliding);
        } else if (name == "full_attention") {
            out.kinds.push_back(LayerKind::Full);
        } else {
            throw std::invalid_argument("gemma4 config: unknown layer type " + name);
        }
    }
    if (out.kinds.back() != LayerKind::Full) {
        throw std::invalid_argument("gemma4 config: the last layer must be full attention");
    }
    out.sliding_window = json_dimension_i32(c, "sliding_window");
    out.sliding        = geometry(c, "sliding_attention");
    out.full           = geometry(c, "full_attention");
    if (out.sliding.rotary_pairs != out.sliding.head_dim / 2) {
        throw std::invalid_argument("gemma4 config: sliding layers rotate every pair");
    }
    for (const auto* g : {&out.sliding, &out.full}) {
        if (out.query_heads % g->kv_heads != 0) {
            throw std::invalid_argument("gemma4 config: query heads must divide by KV heads");
        }
    }
    if (c.contains("num_experts")) {
        out.experts             = json_dimension_i32(c, "num_experts");
        out.experts_per_token   = json_dimension_i32(c, "num_experts_per_tok");
        out.expert_intermediate = json_dimension_i32(c, "moe_intermediate_size");
    }
    const auto layers = static_cast<std::size_t>(out.layers);
    // Per-layer MLP widths: absent means every layer has intermediate_size.
    out.intermediate.assign(layers, out.intermediate_size);
    if (c.contains("layer_intermediate_sizes")) {
        const auto& widths = c.at("layer_intermediate_sizes");
        if (!widths.is_array() || widths.size() != layers) {
            throw std::invalid_argument(
                "gemma4 config: layer_intermediate_sizes must cover every layer");
        }
        for (std::size_t i = 0; i < layers; ++i) {
            out.intermediate[i] = static_cast<std::int32_t>(
                json_u32(widths[i], "layer_intermediate_sizes"));
        }
        out.intermediate_size = *std::max_element(out.intermediate.begin(), out.intermediate.end());
    }
    // KV sharing: absent means every layer owns its keys and values. A sharing layer reads an
    // earlier owning layer of its own kind (same geometry, same window).
    out.kv_source.resize(layers);
    for (std::size_t i = 0; i < layers; ++i) { out.kv_source[i] = static_cast<std::int32_t>(i); }
    if (c.contains("kv_source_layers")) {
        const auto& sources = c.at("kv_source_layers");
        if (!sources.is_array() || sources.size() != layers) {
            throw std::invalid_argument("gemma4 config: kv_source_layers must cover every layer");
        }
        for (std::size_t i = 0; i < layers; ++i) {
            const auto source = json_u32(sources[i], "kv_source_layers", false);
            if (source > i || out.kinds[source] != out.kinds[i] ||
                (source != i && out.kv_source[source] != static_cast<std::int32_t>(source))) {
                throw std::invalid_argument(
                    "gemma4 config: layer " + std::to_string(i) +
                    " must read its own keys or an earlier owning layer of its kind");
            }
            out.kv_source[i] = static_cast<std::int32_t>(source);
        }
    }
    if (c.contains("hidden_size_per_layer_input")) {
        out.ple_width            = json_dimension_i32(c, "hidden_size_per_layer_input");
        out.ple_vocab            = json_dimension_i32(c, "vocab_size_per_layer_input");
        out.ple_embedding_scale  = positive_real(c, "per_layer_embedding_scale");
        out.ple_projection_scale = positive_real(c, "per_layer_projection_scale");
        out.ple_input_scale      = positive_real(c, "per_layer_input_scale");
        // Every token id the embedding accepts must have a per-layer row.
        if (out.ple_vocab != out.vocab_size) {
            throw std::invalid_argument(
                "gemma4 config: vocab_size_per_layer_input must equal vocab_size");
        }
    }
    return out;
}

VisionConfig parse_vision_config(const artifact::Json& c) {
    if (c.value("model_type", std::string()) != "gemma4_vision") {
        throw std::invalid_argument("gemma4 vision config: model_type must be gemma4_vision");
    }
    VisionConfig out;
    out.hidden_size       = json_dimension_i32(c, "hidden_size");
    out.layers            = json_dimension_i32(c, "num_hidden_layers");
    out.heads             = json_dimension_i32(c, "num_attention_heads");
    out.head_dim          = json_dimension_i32(c, "head_dim");
    out.intermediate_size = json_dimension_i32(c, "intermediate_size");
    out.patch_size        = json_dimension_i32(c, "patch_size");
    out.pooling           = json_dimension_i32(c, "pooling_kernel_size");
    out.table_size        = json_dimension_i32(c, "position_embedding_size");
    out.default_tokens    = json_dimension_i32(c, "default_output_length");
    out.output_hidden     = json_dimension_i32(c, "output_hidden_size");
    out.rope_theta        = positive_real(c, "rope_theta");
    out.rms_norm_eps      = positive_real(c, "rms_norm_eps");
    out.boi_token         = json_dimension_i32(c, "boi_token_id");
    out.eoi_token         = json_dimension_i32(c, "eoi_token_id");
    out.image_token       = json_dimension_i32(c, "image_token_id");
    if (out.heads * out.head_dim != out.hidden_size || out.head_dim % 4 != 0) {
        throw std::invalid_argument("gemma4 vision config: head geometry is not an axial 2-D rope's");
    }
    return out;
}

DrafterConfig parse_drafter_config(const artifact::Json& c) {
    if (c.value("model_type", std::string()) != "gemma4_assistant") {
        throw std::invalid_argument("gemma4 drafter config: model_type must be gemma4_assistant");
    }
    DrafterConfig out;
    out.hidden_size   = json_dimension_i32(c, "hidden_size");
    out.backbone_size = json_dimension_i32(c, "backbone_hidden_size");
    out.vocab_size    = json_dimension_i32(c, "vocab_size");
    out.head_rows     = c.contains("head_rows") ? json_dimension_i32(c, "head_rows") : out.vocab_size;
    if (out.head_rows > out.vocab_size) {
        throw std::invalid_argument("gemma4 drafter config: head_rows exceeds vocab_size");
    }
    out.layers            = json_dimension_i32(c, "num_hidden_layers");
    out.query_heads       = json_dimension_i32(c, "num_attention_heads");
    out.intermediate_size = json_dimension_i32(c, "intermediate_size");
    out.sliding_window    = json_dimension_i32(c, "sliding_window");
    out.rms_norm_eps      = positive_real(c, "rms_norm_eps");
    if (positive_real(c, "attention_scale") != 1.0F) {
        throw std::invalid_argument("gemma4 drafter config: attention_scale must be 1");
    }
    const auto& kinds = c.at("layer_types");
    if (!kinds.is_array() || static_cast<std::int32_t>(kinds.size()) != out.layers) {
        throw std::invalid_argument("gemma4 drafter config: layer_types must cover every layer");
    }
    for (const auto& kind : kinds) {
        const auto name = kind.get<std::string>();
        out.kinds.push_back(name == "full_attention" ? LayerKind::Full : LayerKind::Sliding);
        if (name != "full_attention" && name != "sliding_attention") {
            throw std::invalid_argument("gemma4 drafter config: unknown layer type " + name);
        }
    }
    out.sliding = geometry(c, "sliding_attention");
    out.full    = geometry(c, "full_attention");
    return out;
}

} // namespace ninferno::models::gemma4
