#include "models/inkling2/config.h"

#include "models/common/config_json.h"

#include <stdexcept>
#include <string>

namespace ninferno::models::inkling2 {

Config parse_config(const artifact::Json& c) {
    if (c.value("model_type", std::string()) != "inkling2_text") {
        throw std::invalid_argument("inkling2 config: model_type must be inkling2_text");
    }
    Config out;
    out.hidden_size      = json_dimension_i32(c, "hidden_size");
    out.vocab_rows       = json_dimension_i32(c, "vocab_size");
    out.vocab_size       = json_dimension_i32(c, "unpadded_vocab_size");
    out.layers           = json_dimension_i32(c, "num_hidden_layers");
    out.max_positions    = json_dimension_i32(c, "max_position_embeddings");
    out.query_heads      = json_dimension_i32(c, "num_attention_heads");
    out.kv_heads         = json_dimension_i32(c, "num_key_value_heads");
    out.head_dim         = json_dimension_i32(c, "head_dim");
    out.sliding_window   = json_dimension_i32(c, "sliding_window");
    out.rel_width        = json_dimension_i32(c, "d_rel");
    out.rel_extent       = json_dimension_i32(c, "rel_extent");
    out.conv_taps        = json_dimension_i32(c, "sconv_kernel_size");
    out.block_size       = json_dimension_i32(c, "n_layers_per_attn_res_block");
    out.experts          = json_dimension_i32(c, "n_routed_experts");
    out.experts_per_tok  = json_dimension_i32(c, "num_experts_per_tok");
    out.expert_width     = json_dimension_i32(c, "intermediate_size");
    out.rms_norm_eps     = json_positive_float(c, "rms_norm_eps");
    out.qk_clip          = json_positive_float(c, "qk_l2_clip_threshold");
    out.rel_clip         = json_positive_float(c, "rel_l2_clip_threshold");
    out.log_scale_floor  = static_cast<float>(json_dimension_i32(c, "log_scaling_n_floor"));
    out.log_scale_alpha  = c.at("log_scaling_alpha").get<float>();
    out.route_scale      = json_positive_float(c, "route_scale");
    out.situ_beta        = json_positive_float(c, "mlp_gate_softcap");
    out.situ_linear_beta = json_positive_float(c, "mlp_up_softcap");
    out.softcap          = json_positive_float(c, "final_logit_softcapping");
    if (out.vocab_size > out.vocab_rows || out.query_heads % out.kv_heads || out.conv_taps != 4 ||
        out.experts_per_tok > out.experts || out.log_scale_alpha < 0 ||
        out.layers % out.block_size) {
        throw std::invalid_argument(
            "inkling2 config: vocabulary, head grouping, convolution width, "
            "experts or residual blocks out of the family's domain");
    }
    const auto& kinds = c.at("layer_types");
    if (!kinds.is_array() || static_cast<std::int32_t>(kinds.size()) != out.layers) {
        throw std::invalid_argument("inkling2 config: layer_types must cover every layer");
    }
    for (std::int32_t i = 0; i < out.layers; ++i) {
        const auto name = kinds[static_cast<std::size_t>(i)].get<std::string>();
        if (name != "sliding_attention" && name != "full_attention") {
            throw std::invalid_argument("inkling2 config: unknown layer type " + name);
        }
        out.kinds.push_back(name == "full_attention" ? LayerKind::Full : LayerKind::Sliding);
    }
    if (c.contains("loop_config")) {
        const auto& loop = c.at("loop_config");
        if (loop.value("is_varloop", false)) {
            throw std::invalid_argument(
                "inkling2 config: the variable-pass research loop is not implemented");
        }
        out.core_start = json_dimension_i32(loop, "core_start");
        out.core_end   = json_dimension_i32(loop, "core_end");
        out.passes     = json_dimension_i32(loop, "num_passes");
        out.gate_rank  = json_dimension_i32(loop, "gate_rank");
        if (out.core_start >= out.core_end || out.core_end > out.layers ||
            out.core_start % out.block_size || out.core_end % out.block_size || out.passes < 2) {
            throw std::invalid_argument(
                "inkling2 config: the loop core must lie on residual-block bounds "
                "and run at least twice");
        }
    }
    return out;
}

std::vector<Visit> visit_schedule(const Config& c) {
    std::vector<Visit> out;
    out.reserve(static_cast<std::size_t>(c.visits()));
    const auto start = c.looped() ? c.core_start : c.layers;
    const auto end   = c.looped() ? c.core_end : c.layers;
    for (std::int32_t i = 0; i < start; ++i) { out.push_back({i, i, 0}); }
    for (std::int32_t pass = 0; pass < c.passes && c.looped(); ++pass) {
        for (std::int32_t i = start; i < end; ++i) {
            const auto slot = pass == 0 ? i : c.layers + (pass - 1) * c.core_layers() + (i - start);
            out.push_back({i, slot, pass});
        }
    }
    for (std::int32_t i = end; i < c.layers; ++i) { out.push_back({i, i, 0}); }
    return out;
}

} // namespace ninferno::models::inkling2
