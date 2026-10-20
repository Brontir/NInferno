#include "models/dense/config.h"

#include "models/common/config_json.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninferno::models::dense {
namespace {

[[noreturn]] void fail(const std::string& what) {
    throw std::invalid_argument("dense_decoder config: " + what);
}

std::string text(const artifact::Json& c, const char* key, const char* fallback) {
    if (!c.contains(key)) { return fallback; }
    if (!c.at(key).is_string()) { fail(std::string(key) + " must be a string"); }
    return c.at(key).get<std::string>();
}

bool flag(const artifact::Json& c, const char* key) {
    if (!c.contains(key)) { return false; }
    if (!c.at(key).is_boolean()) { fail(std::string(key) + " must be boolean"); }
    return c.at(key).get<bool>();
}

float real(const artifact::Json& c, const char* key, float fallback) {
    return c.contains(key) ? json_positive_float(c, key) : fallback;
}

RopeGroup rope_group(const std::string& name, const artifact::Json& g, std::int32_t head_dim) {
    RopeGroup out;
    out.name       = name;
    out.rotary_dim = json_dimension_i32(g, "rotary_dim");
    if (out.rotary_dim % 2 != 0 || out.rotary_dim > head_dim) {
        fail("rope group " + name + ": rotary_dim must be even and at most head_dim");
    }
    const auto frequencies = [&](const char* key) {
        const auto& freq = g.at(key);
        if (!freq.is_array() || static_cast<std::int32_t>(freq.size()) != out.rotary_dim / 2) {
            fail("rope group " + name + ": " + key + " must hold rotary_dim/2 values");
        }
        std::vector<float> values;
        for (const auto& value : freq) {
            const double f = value.get<double>();
            if (!std::isfinite(f) || f <= 0) {
                fail("rope group " + name + ": " + key + " must be positive");
            }
            values.push_back(static_cast<float>(f));
        }
        return values;
    };
    out.inv_freq          = frequencies("inv_freq");
    out.attention_scaling = real(g, "attention_scaling", 1.0F);
    if (g.contains("long_inv_freq")) {
        out.long_inv_freq = frequencies("long_inv_freq");
        out.long_after    = json_dimension_i32(g, "long_after");
    }
    return out;
}

MoeSpec moe_spec(const artifact::Json& m) {
    MoeSpec out;
    out.experts      = json_dimension_i32(m, "num_experts");
    out.top_k        = json_dimension_i32(m, "num_experts_per_tok");
    out.inter        = json_dimension_i32(m, "moe_intermediate_size");
    out.shared_inter = m.contains("shared_intermediate_size")
                           ? static_cast<std::int32_t>(m.at("shared_intermediate_size").get<std::int64_t>())
                           : 0;
    if (out.shared_inter < 0) { fail("moe.shared_intermediate_size must not be negative"); }
    out.shared_gate = flag(m, "shared_gate");
    out.router_bias = flag(m, "router_bias");
    out.expert_bias = flag(m, "expert_bias");
    if (out.top_k > out.experts) { fail("moe.num_experts_per_tok exceeds num_experts"); }
    auto& op = out.op;
    op.top_k = out.top_k;
    const auto scoring = text(m, "scoring", "softmax");
    if (scoring == "softmax") {
        op.scoring = ops::MoeScoring::Softmax;
    } else if (scoring == "sigmoid") {
        op.scoring = ops::MoeScoring::Sigmoid;
    } else {
        fail("unknown moe.scoring " + scoring);
    }
    op.renormalize  = flag(m, "renormalize");
    op.routed_scale = real(m, "routed_scale", 1.0F);
    const auto activation = text(m, "activation", "swiglu");
    if (activation == "swiglu") {
        op.activation = ops::MoeActivation::SwiGlu;
    } else if (activation == "geglu_tanh") {
        op.activation = ops::MoeActivation::GeGluTanh;
    } else if (activation == "clamped_swiglu") {
        op.activation = ops::MoeActivation::ClampedSwiGlu;
        op.alpha      = json_positive_float(m, "alpha");
        op.limit      = json_positive_float(m, "limit");
    } else {
        fail("unknown moe.activation " + activation);
    }
    const auto layout = text(m, "gate_up_layout", "concatenated");
    if (layout == "concatenated") {
        op.gate_up_layout = ops::MoeGateUpLayout::Concatenated;
    } else if (layout == "interleaved") {
        op.gate_up_layout = ops::MoeGateUpLayout::Interleaved;
    } else {
        fail("unknown moe.gate_up_layout " + layout);
    }
    if (out.shared_inter > 0 && op.activation != ops::MoeActivation::SwiGlu) {
        fail("a shared expert is implemented with the SwiGLU activation only");
    }
    return out;
}

} // namespace

std::int32_t Config::max_window() const noexcept {
    std::int32_t w = 0;
    for (const auto& layer : layer_specs) { w = std::max(w, layer.window); }
    return w;
}

Config parse_config(const artifact::Json& c) {
    const auto model_type = c.value("model_type", std::string());
    if (model_type != "dense_decoder" && model_type != "moe_decoder") {
        fail("model_type must be dense_decoder or moe_decoder");
    }
    Config out;
    if (model_type == "moe_decoder") {
        if (!c.contains("moe") || !c.at("moe").is_object()) { fail("a moe_decoder needs a moe object"); }
        out.moe = moe_spec(c.at("moe"));
    }
    out.attention_sinks = flag(c, "attention_sinks");
    out.hidden_size       = json_dimension_i32(c, "hidden_size");
    out.vocab_size        = json_dimension_i32(c, "vocab_size");
    out.layers            = json_dimension_i32(c, "num_hidden_layers");
    out.max_positions     = json_dimension_i32(c, "max_position_embeddings");
    out.query_heads       = json_dimension_i32(c, "num_attention_heads");
    out.kv_heads          = json_dimension_i32(c, "num_key_value_heads");
    out.head_dim          = json_dimension_i32(c, "head_dim");
    out.intermediate_size = out.moe ? 0 : json_dimension_i32(c, "intermediate_size");
    out.rms_norm_eps      = json_positive_float(c, "rms_norm_eps");
    out.attention_scale   = json_positive_float(c, "attention_scale");
    out.residual_multiplier = real(c, "residual_multiplier", 1.0F);
    out.qkv_bias          = flag(c, "attention_bias");
    out.output_bias       = flag(c, "attention_output_bias");
    if (c.contains("final_logit_softcapping") && !c.at("final_logit_softcapping").is_null()) {
        out.softcap = json_positive_float(c, "final_logit_softcapping");
    }
    const auto act = text(c, "hidden_act", "silu");
    if (act == "silu") {
        out.activation = Activation::SiLU;
    } else if (act == "gelu_pytorch_tanh") {
        out.activation = Activation::GeluTanh;
    } else {
        fail("hidden_act " + act + " is not implemented");
    }
    const auto placement = text(c, "norm_placement", "pre");
    if (placement == "pre") {
        out.norms = NormPlacement::Pre;
    } else if (placement == "post") {
        out.norms = NormPlacement::Post;
    } else if (placement == "sandwich") {
        out.norms = NormPlacement::Sandwich;
    } else {
        fail("unknown norm_placement " + placement);
    }
    out.norm_unit_offset = flag(c, "norm_unit_offset");
    out.final_norm_unit_offset = c.contains("final_norm_unit_offset")
                                     ? flag(c, "final_norm_unit_offset")
                                     : out.norm_unit_offset;
    out.post_norm_eps          = real(c, "post_norm_eps", out.rms_norm_eps);
    out.embedding_norm         = flag(c, "embedding_norm");
    out.attention_output_gate  = flag(c, "attention_output_gate");
    out.logit_multiplier       = real(c, "logit_multiplier", 1.0F);
    if (out.logit_multiplier != 1.0F && !out.softcap) {
        fail("logit_multiplier is implemented with a final logit soft cap only");
    }
    const auto norm = text(c, "qk_norm", "none");
    if (norm == "none") {
        out.qk_norm = ops::QkNorm::None;
    } else if (norm == "head") {
        out.qk_norm = ops::QkNorm::Head;
    } else if (norm == "head_unscaled") {
        out.qk_norm        = ops::QkNorm::Head;
        out.qk_norm_scaled = false;
    } else if (norm == "projection") {
        out.qk_norm = ops::QkNorm::Projection;
    } else {
        fail("unknown qk_norm " + norm);
    }
    if (c.contains("query_scale") && !c.at("query_scale").is_null()) {
        const auto& q    = c.at("query_scale");
        out.query_beta   = json_positive_float(q, "beta");
        out.query_period = json_dimension_i32(q, "period");
    }
    // What the Ops admit: the prepare Op's head widths, and the attention's GQA groups.
    if (out.head_dim != 64 && out.head_dim != 128 && out.head_dim != 256) {
        fail("head_dim " + std::to_string(out.head_dim) + " is not implemented (64, 128 or 256)");
    }
    if (out.query_heads % out.kv_heads != 0) { fail("query heads must divide by KV heads"); }
    const auto group = out.query_heads / out.kv_heads;
    if (group < 1 || group > 16) {
        fail("GQA group " + std::to_string(group) + " is not implemented (1..16)");
    }
    if (out.hidden_size % 8 != 0 || out.hidden_size > 8192) {
        fail("hidden_size must be a multiple of 8 up to 8192");
    }

    if (c.contains("rope_groups")) {
        const auto& groups = c.at("rope_groups");
        if (!groups.is_object()) { fail("rope_groups must be an object"); }
        for (auto it = groups.begin(); it != groups.end(); ++it) {
            out.rope_groups.push_back(rope_group(it.key(), it.value(), out.head_dim));
        }
    }
    const auto& kinds = c.at("layer_types");
    const auto& ropes = c.at("layer_rope");
    if (!kinds.is_array() || static_cast<std::int32_t>(kinds.size()) != out.layers ||
        !ropes.is_array() || ropes.size() != kinds.size()) {
        fail("layer_types and layer_rope must cover every layer");
    }
    std::int32_t window = 0;
    for (std::int32_t i = 0; i < out.layers; ++i) {
        LayerSpec spec;
        const auto kind = kinds[i].get<std::string>();
        if (kind == "sliding_attention") {
            if (window == 0) { window = json_dimension_i32(c, "sliding_window"); }
            spec.window = window;
        } else if (kind != "full_attention") {
            fail("unknown layer type " + kind);
        }
        if (!ropes[i].is_null()) {
            const auto name = ropes[i].get<std::string>();
            const auto it = std::find_if(out.rope_groups.begin(), out.rope_groups.end(),
                                         [&](const RopeGroup& g) { return g.name == name; });
            if (it == out.rope_groups.end()) { fail("layer_rope names an unknown group " + name); }
            spec.rope = static_cast<std::int32_t>(it - out.rope_groups.begin());
        }
        out.layer_specs.push_back(spec);
    }
    return out;
}

} // namespace ninferno::models::dense
