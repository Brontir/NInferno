#include "convert/inkling2.h"

#include "convert/checks.h"
#include "convert/formats.h"
#include "convert/sources.h"

#include <algorithm>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace ninferno::convert {
namespace {

std::uint64_t count(const OJson& config, const char* key) {
    const auto* value = member(config, key);
    if (!value || !value->is_number_unsigned() || value->get<std::uint64_t>() == 0) {
        throw std::runtime_error(std::string("inkling2: ") + key + " must be a positive integer");
    }
    return value->get<std::uint64_t>();
}

// The mapped config (gguf_model.cpp map_inkling2), checked against what the family's
// mathematics fixes: one head geometry, a full layer closing every residual block, a loop core
// on block boundaries.
OJson text_config(const OJson& source) {
    if (source.value("model_type", std::string()) != "inkling2_text") {
        throw std::runtime_error("inkling2: model_type must be inkling2_text");
    }
    const auto layers = count(source, "num_hidden_layers");
    const auto block  = count(source, "n_layers_per_attn_res_block");
    const auto* types = member(source, "layer_types");
    if (!types || !types->is_array() || types->size() != layers || layers % block) {
        throw std::runtime_error("inkling2: layer_types must cover whole residual blocks");
    }
    for (std::uint64_t i = 0; i < layers; ++i) {
        const auto type = (*types)[i].get<std::string>();
        if ((type == "full_attention") != (i % block == block - 1)) {
            throw std::runtime_error(
                "inkling2: the last layer of each residual block, and only it, "
                "is a full-attention layer");
        }
    }
    for (const char* key :
         {"hidden_size", "vocab_size", "unpadded_vocab_size", "num_attention_heads",
          "num_key_value_heads", "head_dim", "sliding_window", "d_rel", "rel_extent",
          "log_scaling_n_floor", "sconv_kernel_size", "n_routed_experts", "num_experts_per_tok",
          "intermediate_size"}) {
        (void)count(source, key);
    }
    if (count(source, "num_attention_heads") % count(source, "num_key_value_heads") ||
        count(source, "num_experts_per_tok") > count(source, "n_routed_experts") ||
        count(source, "unpadded_vocab_size") > count(source, "vocab_size") ||
        count(source, "sconv_kernel_size") != 4) {
        throw std::runtime_error("inkling2: head grouping, expert count, vocabulary or convolution "
                                 "width out of the family's domain");
    }
    if (const auto* loop = member(source, "loop_config")) {
        const auto start = (*loop)["core_start"].get<std::uint64_t>();
        const auto end   = (*loop)["core_end"].get<std::uint64_t>();
        if (!start || start >= end || end > layers || start % block || end % block ||
            (*loop)["num_passes"].get<std::uint64_t>() < 1 ||
            !(*loop)["gate_rank"].get<std::uint64_t>()) {
            throw std::runtime_error(
                "inkling2: loop_config core must lie on residual-block bounds");
        }
    }
    return source;
}

class Builder {
public:
    Builder(Model& model, const Checkpoint& base) : model_(model), base_(base) {}

    void matrix(const std::string& name, const std::string& tensor, Shape shape,
                std::vector<std::string> inputs, std::string probe_leaf = {}) {
        auto factory = [tensor, shape](const Checkpoint& base, MatrixFormat format) -> SourcePtr {
            return matrix_source(base, tensor, shape, format);
        };
        add(name, shape, factory(base_, MatrixFormat::Auto), factory, std::move(inputs),
            QType::BF16, std::move(probe_leaf));
    }

    void vector(const std::string& name, const std::string& tensor, Shape shape,
                QType direct = QType::BF16) {
        auto factory = [name](const Checkpoint&, MatrixFormat) -> SourcePtr {
            throw std::runtime_error(name + ": provide an explicit encoded source mapping");
        };
        add(name, shape, tensor_source(base_, tensor, shape), factory, {}, direct, {});
    }

    // An expert's gate and up as one matrix (gate rows first); NVFP4 halves share one divisor.
    void stacked(const std::string& name, std::vector<std::string> members, Shape shape,
                 std::vector<std::string> inputs, std::string probe_leaf) {
        auto factory = [name, members, shape](const Checkpoint& base,
                                              MatrixFormat format) -> SourcePtr {
            const auto rows_each = shape[0] / members.size();
            std::vector<SourcePtr> parts;
            bool encoded = true;
            for (const auto& part : members) {
                parts.push_back(matrix_source(base, part, {rows_each, shape[1]}, format));
                auto prefix = part;
                if (prefix.ends_with(".weight")) { prefix.resize(prefix.size() - 7); }
                const auto kept = kept_encoding(base, prefix, rows_each, shape[1]);
                encoded         = encoded && kept && kept->format == QType::NVFP4;
            }
            if (format == MatrixFormat::Fp8Row || (format == MatrixFormat::Auto && !encoded)) {
                return concat_source(std::move(parts), shape, name);
            }
            return stacked_source(std::move(parts), name);
        };
        add(name, shape, factory(base_, MatrixFormat::Auto), factory, std::move(inputs),
            QType::BF16, std::move(probe_leaf));
    }

    // An expert's down projection read against its gate_up plane's adopted divisor.
    void compensated(const std::string& name, const std::string& tensor, Shape shape,
                     std::string gate, std::string up, Shape pair, std::vector<std::string> inputs,
                     std::string probe_leaf) {
        auto factory = [tensor, shape, gate, up, pair](const Checkpoint& base,
                                                       MatrixFormat format) -> SourcePtr {
            auto ref         = matrix_source(base, tensor, shape, format);
            const auto strip = [](std::string value) {
                if (value.ends_with(".weight")) { value.resize(value.size() - 7); }
                return value;
            };
            const auto nvfp4 = [&](const std::string& matrix, const Shape& dims) {
                const auto kept = kept_encoding(base, strip(matrix), dims[0], dims[1]);
                return kept && kept->format == QType::NVFP4;
            };
            if (!(nvfp4(tensor, shape) && nvfp4(gate, pair) && nvfp4(up, pair))) { return ref; }
            return divisor_scaled_source(ref, matrix_source(base, gate, pair, MatrixFormat::Nvfp4),
                                         matrix_source(base, up, pair, MatrixFormat::Nvfp4));
        };
        add(name, shape, factory(base_, MatrixFormat::Auto), factory, std::move(inputs),
            QType::BF16, std::move(probe_leaf));
    }

    std::size_t group(const std::vector<std::string>& names) {
        return model_.add_packing_group(names);
    }

private:
    void add(const std::string& name, Shape shape, SourcePtr source,
             std::function<SourcePtr(const Checkpoint&, MatrixFormat)> factory,
             std::vector<std::string> inputs, QType direct, std::string probe_leaf) {
        Parameter parameter;
        parameter.name           = name;
        parameter.shape          = std::move(shape);
        parameter.source         = std::move(source);
        parameter.encoded_source = std::move(factory);
        parameter.inputs         = std::move(inputs);
        parameter.direct         = direct;
        parameter.residency      = name.substr(0, name.find('/'));
        parameter.probe_leaf     = std::move(probe_leaf);
        model_.add(std::move(parameter));
    }

    Model& model_;
    const Checkpoint& base_;
};

void text_block(Builder& b, Model& model, const OJson& c, std::uint64_t layer) {
    const auto h     = c["hidden_size"].get<std::uint64_t>();
    const auto d     = c["head_dim"].get<std::uint64_t>();
    const auto q     = c["num_attention_heads"].get<std::uint64_t>() * d;
    const auto heads = c["num_attention_heads"].get<std::uint64_t>();
    const auto kvh   = c["num_key_value_heads"].get<std::uint64_t>();
    const auto k     = kvh * d;
    const auto rel   = c["d_rel"].get<std::uint64_t>();
    const auto taps  = c["sconv_kernel_size"].get<std::uint64_t>();
    const bool full  = c["layer_types"][layer].get<std::string>() == "full_attention";
    const auto extent =
        full ? c["rel_extent"].get<std::uint64_t>() : c["sliding_window"].get<std::uint64_t>();
    const auto p    = "text/layers/" + std::to_string(layer) + "/";
    const auto leaf = "layers." + std::to_string(layer) + ".";
    const auto s    = "model." + leaf;
    const auto f32  = [&](const std::string& name, const std::string& tensor, Shape shape) {
        b.vector(p + name, s + tensor, std::move(shape), QType::FP32);
    };

    if (full) {
        for (const char* role : {"q", "k", "v"}) {
            b.vector(p + role + "_input_norm", s + role + "_input_layernorm.weight", {h});
        }
    } else {
        b.vector(p + "input_norm", s + "input_layernorm.weight", {h});
    }
    const auto attn = p + "attention/";
    for (const auto& [role, field, rows, input] :
         {std::tuple{"query", "q_proj", q, "query_input"},
          std::tuple{"key", "k_proj", k, "key_input"},
          std::tuple{"value", "v_proj", k, "value_input"},
          std::tuple{"gate", "gate_proj", q, "query_input"},
          std::tuple{"relative", "rel_proj", rel * heads, "query_input"},
          std::tuple{"sink", "sink_proj", heads, "query_input"}}) {
        b.matrix(attn + role, s + "self_attn." + field + ".weight", {rows, h}, {attn + input},
                 leaf + "self_attn." + field);
    }
    if (full) {
        b.matrix(attn + "key_taps", s + "self_attn.k_conv_proj.weight", {kvh * taps, h},
                 {attn + "key_input"}, leaf + "self_attn.k_conv_proj");
        b.matrix(attn + "value_taps", s + "self_attn.v_conv_proj.weight", {kvh * taps, h},
                 {attn + "value_input"}, leaf + "self_attn.v_conv_proj");
    }
    b.matrix(attn + "output", s + "self_attn.o_proj.weight", {h, q}, {attn + "context"},
             leaf + "self_attn.o_proj");
    f32("attention/query_clip", "self_attn.q_clip", {heads, d});
    f32("attention/relative_clip", "self_attn.rel_clip", {heads, rel});
    f32("attention/relative_bias", "self_attn.rel_bias", {extent, rel});
    f32("attention/key_conv", "self_attn.k_conv", {k, taps});
    f32("attention/value_conv", "self_attn.v_conv", {k, taps});
    f32("attention/output_conv", "self_attn.post_conv", {h, taps});
    if (layer > 0) { f32("attention_query", "attention_input.query", {full ? 3U : 1U, h}); }
    f32("ffn_query", "ffn_input.query", {1, h});
    b.vector(p + "ffn_norm", s + "post_attention_layernorm.weight", {h});

    const auto experts = c["n_routed_experts"].get<std::uint64_t>();
    const auto ir      = c["intermediate_size"].get<std::uint64_t>();
    const auto m       = p + "moe/";
    b.matrix(m + "router", s + "mlp.gate.weight", {experts, h}, {m + "router_input"},
             leaf + "mlp.gate");
    b.vector(m + "selection_bias", s + "mlp.gate.e_score_correction_bias", {experts}, QType::FP32);
    f32("moe/output_conv", "mlp.post_conv", {h, taps});
    std::vector<std::string> gate_up, downs;
    for (std::uint64_t e = 0; e < experts; ++e) {
        const auto ep    = m + "experts/" + std::to_string(e) + "/";
        const auto es    = s + "mlp.experts." + std::to_string(e) + ".";
        const auto eleaf = leaf + "mlp.experts." + std::to_string(e) + ".";
        b.stacked(ep + "gate_up", {es + "gate_proj.weight", es + "up_proj.weight"}, {2 * ir, h},
                  {m + "expert_input"}, eleaf + "gate_proj");
        model.parameters.back().probe_shape = {ir, h};
        gate_up.push_back(ep + "gate_up");
        b.compensated(ep + "down", es + "down_proj.weight", {h, ir}, es + "gate_proj.weight",
                      es + "up_proj.weight", {ir, h}, {ep + "product"}, eleaf + "down_proj");
        downs.push_back(ep + "down");
    }
    model.expert_banks.push_back(b.group(gate_up));
    model.expert_banks.push_back(b.group(downs));

    if (const auto* loop = member(c, "loop_config")) {
        const auto start = (*loop)["core_start"].get<std::uint64_t>();
        const auto end   = (*loop)["core_end"].get<std::uint64_t>();
        const auto rank  = (*loop)["gate_rank"].get<std::uint64_t>();
        if (layer < start || layer >= end) { return; }
        f32("loop/message_query", "loop.message_query", {4, h});
        for (const char* role : {"q", "k", "v", "ffn"}) {
            const auto lp = p + "loop/" + role + "/";
            const auto ls = s + "loop." + role + "_control.";
            const auto ll = leaf + "loop." + role + "_control.";
            b.matrix(lp + "down", ls + "down.weight", {rank, 2 * h}, {lp + "pair"}, ll + "down");
            b.matrix(lp + "gain", ls + "gain.weight", {h, rank}, {lp + "low_rank"}, ll + "gain");
            b.matrix(lp + "forget", ls + "forget.weight", {h, rank}, {lp + "low_rank"},
                     ll + "forget");
            b.group({lp + "gain", lp + "forget"});
        }
    }
}

// The NVFP4 sites the checkpoint carries per projection (the expert banks), found by probe leaf.
std::map<std::size_t, KeptEncoding> encoded_sites(const Model& model, const Checkpoint& base) {
    std::map<std::size_t, KeptEncoding> found;
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        const auto& parameter = model.parameters[i];
        if (!parameter.projection() || parameter.probe_leaf.empty()) { continue; }
        const auto& dims = parameter.probe_shape.empty() ? parameter.shape : parameter.probe_shape;
        if (const auto kept =
                kept_encoding(base, "model." + parameter.probe_leaf, dims[0], dims[1])) {
            found[i] = *kept;
        }
    }
    return found;
}

} // namespace

bool is_inkling2_architecture(std::string_view architecture) noexcept {
    return architecture == "Inkling2LoopForCausalLM" ||
           architecture == "Inkling2LoopForConditionalGeneration";
}

Model build_inkling2_model(const Checkpoint& base, const ResourceOverrides& overrides) {
    Model model;
    model.components["text"]["config"] = text_config(base.config());
    load_resources(model, base.root(), model.config()["unpadded_vocab_size"].get<std::uint64_t>(),
                   overrides, nullptr, resource_fallbacks(base));
    const auto& config = model.config();
    Builder b(model, base);
    const auto h = config["hidden_size"].get<std::uint64_t>();
    const auto r = config["vocab_size"].get<std::uint64_t>();
    // The table and the head keep the padded rows the GGUF stores; the head's logits past the
    // tokenizer's vocabulary are dropped by the engine.
    b.matrix("text/token_embedding", "model.embed_tokens.weight", {r, h}, {});
    b.vector("text/embedding_norm", "model.embed_norm.weight", {h});
    b.matrix("text/output_head", "lm_head.weight", {r, h}, {"text/final_hidden"});
    b.vector("text/final_norm", "model.norm.weight", {h});
    b.vector("text/output_query", "model.output_res_query", {1, h}, QType::FP32);
    for (std::uint64_t i = 0; i < config["num_hidden_layers"].get<std::uint64_t>(); ++i) {
        text_block(b, model, config, i);
    }
    return model;
}

bool is_inkling2_recipe(std::string_view name) noexcept { return name == "inkling2_published"; }

void apply_inkling2_recipe(std::string_view name, Model& model, Recipe& recipe,
                           const Checkpoint& base) {
    if (name != "inkling2_published" ||
        model.config().value("model_type", std::string()) != "inkling2_text") {
        throw std::runtime_error("inkling2_published requires an Inkling 2 text model");
    }
    // NVFP4 expert banks keep their words and per-expert divisors; every other tensor keeps its
    // published encoding (the GGUF's Q8_0 words, BF16 and FP32 values) through keep_published.
    const auto encoded = encoded_sites(model, base);
    for (const auto& [index, site] : encoded) {
        const auto& parameter = model.parameters[index];
        recipe.assign(index, site.format, Method::ImportEncoded,
                      parameter.encoded_source(base, encoded_matrix_format(site.format)));
        recipe.set_policy(index, kept_policy(site));
    }
    for (const auto bank : model.expert_banks) {
        const auto& names = model.packing_groups[bank];
        if (std::all_of(names.begin(), names.end(), [&](std::size_t i) {
                const auto it = encoded.find(i);
                return it != encoded.end() && it->second.format == QType::NVFP4;
            })) {
            recipe.explicit_groups.push_back(names);
        }
    }
}

std::string default_inkling2_recipe(const OJson& config) {
    (void)text_config(config);
    return "inkling2_published";
}

} // namespace ninferno::convert
