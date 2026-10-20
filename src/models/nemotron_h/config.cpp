#include "models/nemotron_h/config.h"

#include "models/common/config_json.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninferno::models::nemotron_h {
namespace {

[[noreturn]] void fail(const std::string& what) {
    throw std::invalid_argument("nemotron_h config: " + what);
}

float number(const artifact::Json& c, const char* key, float fallback) {
    if (!c.contains(key)) { return fallback; }
    if (!c.at(key).is_number()) { fail(std::string(key) + " must be a number"); }
    const double v = c.at(key).get<double>();
    if (!std::isfinite(v) || v < 0) { fail(std::string(key) + " must be finite and >= 0"); }
    return static_cast<float>(v);
}

const artifact::Json& object(const artifact::Json& c, const char* key) {
    if (!c.contains(key) || !c.at(key).is_object()) { fail(std::string(key) + " must be an object"); }
    return c.at(key);
}

// The routed MoE of the trunk and of the MTP drafter (resolve_nemotron_h_moe in the converter).
MoeSpec parse_moe(const artifact::Json& e, std::int32_t hidden_size) {
    MoeSpec moe;
    moe.experts = json_dimension_i32(e, "num_experts");
    moe.inter   = json_dimension_i32(e, "moe_intermediate_size");
    moe.input   = e.contains("expert_input_size") ? json_dimension_i32(e, "expert_input_size") : hidden_size;
    if (moe.input < hidden_size) { fail("expert_input_size must be >= hidden_size"); }
    moe.shared_inter    = json_dimension_i32(e, "shared_intermediate_size");
    moe.op.top_k        = json_dimension_i32(e, "num_experts_per_tok");
    moe.op.scoring      = ops::MoeScoring::Sigmoid;
    moe.op.renormalize  = e.value("renormalize", true);
    moe.op.routed_scale = number(e, "routed_scaling_factor", 1.0F);
    moe.op.activation   = ops::MoeActivation::Relu2;
    if (e.value("scoring", std::string("sigmoid")) != "sigmoid" ||
        e.value("activation", std::string("relu2")) != "relu2") {
        fail("the MoE must score by sigmoid and activate by relu2");
    }
    return moe;
}

} // namespace

std::uint32_t Config::count(Block kind) const noexcept {
    return static_cast<std::uint32_t>(std::count(blocks.begin(), blocks.end(), kind));
}

Config parse_config(const artifact::Json& c) {
    if (!c.is_object() || c.value("model_type", std::string()) != "nemotron_h") {
        fail("model_type must be nemotron_h");
    }
    Config out;
    out.hidden_size     = json_dimension_i32(c, "hidden_size");
    out.vocab_size      = json_dimension_i32(c, "vocab_size");
    out.layers          = json_dimension_i32(c, "num_hidden_layers");
    out.max_positions   = json_dimension_i32(c, "max_position_embeddings");
    out.rms_norm_eps    = json_positive_float(c, "rms_norm_eps");
    out.query_heads     = json_dimension_i32(c, "num_attention_heads");
    out.kv_heads        = json_dimension_i32(c, "num_key_value_heads");
    out.head_dim        = json_dimension_i32(c, "head_dim");
    out.attention_scale = json_positive_float(c, "attention_scale");
    if (out.query_heads % out.kv_heads != 0) { fail("query heads must be a multiple of kv heads"); }

    const auto& types = c.at("layer_types");
    if (!types.is_array() || static_cast<std::int32_t>(types.size()) != out.layers) {
        fail("layer_types must name every layer");
    }
    for (const auto& t : types) {
        const auto kind = t.get<std::string>();
        if (kind == "mamba") {
            out.blocks.push_back(Block::Mamba);
        } else if (kind == "attention") {
            out.blocks.push_back(Block::Attention);
        } else if (kind == "moe") {
            out.blocks.push_back(Block::Moe);
        } else {
            fail("unknown block type " + kind);
        }
    }

    const auto& m          = object(c, "mamba");
    auto& g                = out.mamba.geometry;
    g.heads                = json_dimension_i32(m, "num_heads");
    g.head_dim             = json_dimension_i32(m, "head_dim");
    g.groups               = json_dimension_i32(m, "n_groups");
    g.state                = json_dimension_i32(m, "state_size");
    g.dt_min               = number(m, "dt_min", 0.0F);
    g.dt_max               = number(m, "dt_max", 0.0F);
    out.mamba.conv_bias    = m.value("conv_bias", true);
    if (json_dimension_i32(m, "conv_kernel") != 4) { fail("conv_kernel must be 4"); }
    if (g.heads % g.groups != 0) { fail("mamba heads must be a multiple of n_groups"); }

    out.moe = parse_moe(object(c, "moe"), out.hidden_size);
    if (out.count(Block::Attention) == 0) {
        fail("the Program keeps a sequence's context in attention layers; this model has none");
    }
    return out;
}

MtpConfig parse_mtp_config(const artifact::Json& c) {
    if (!c.is_object() || c.value("model_type", std::string()) != "nemotron_h_mtp") {
        fail("model_type must be nemotron_h_mtp");
    }
    MtpConfig out;
    out.hidden_size     = json_dimension_i32(c, "hidden_size");
    out.vocab_size      = json_dimension_i32(c, "vocab_size");
    out.max_positions   = json_dimension_i32(c, "max_position_embeddings");
    out.rms_norm_eps    = json_positive_float(c, "rms_norm_eps");
    out.layer_norm_eps  = json_positive_float(c, "layer_norm_eps");
    out.query_heads     = json_dimension_i32(c, "num_attention_heads");
    out.kv_heads        = json_dimension_i32(c, "num_key_value_heads");
    out.head_dim        = json_dimension_i32(c, "head_dim");
    if (out.query_heads % out.kv_heads != 0) {
        fail("query heads must be a multiple of kv heads");
    }

    out.moe = parse_moe(object(c, "moe"), out.hidden_size);
    return out;
}

} // namespace ninferno::models::nemotron_h
