#include "spec/dspine.h"

#include "spec/json_fields.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace ninferno::spec {

namespace {

using nlohmann::json;

[[noreturn]] void refuse(const std::string& what) {
    throw std::runtime_error("DSpine draft config: " + what);
}

std::uint32_t dimension(const json& object, const char* key) {
    const auto value = json_unsigned(object, key);
    if (value == 0 || value > 0x7fffffffULL) {
        refuse(std::string(key) + " must be a positive integer");
    }
    return static_cast<std::uint32_t>(value);
}

float positive_float(const json& object, const char* key) {
    const auto it = object.find(key);
    if (it == object.end() || !it->is_number() || !(it->get<double>() > 0)) {
        refuse(std::string(key) + " must be a positive number");
    }
    return static_cast<float>(it->get<double>());
}

bool flag_is(const json& object, const char* key, bool expected) {
    const auto it = object.find(key);
    return it != object.end() && it->is_boolean() && it->get<bool>() == expected;
}

// A boolean / number that is missing or null reads as `fallback`; another type is refused.
bool bool_or(const json& object, const char* key, bool fallback) {
    const auto it = object.find(key);
    if (it == object.end() || it->is_null()) { return fallback; }
    if (!it->is_boolean()) { refuse(std::string(key) + " must be a boolean"); }
    return it->get<bool>();
}

double number_or(const json& object, const char* key, double fallback) {
    const auto it = object.find(key);
    if (it == object.end() || it->is_null()) { return fallback; }
    if (!it->is_number()) { refuse(std::string(key) + " must be a number"); }
    return it->get<double>();
}

void require_absent_or_null(const json& object, const char* key, const char* where) {
    const auto it = object.find(key);
    if (it != object.end() && !it->is_null()) {
        refuse(std::string(where) + "." + key + " is not part of the DSpine path");
    }
}

} // namespace

DSpineConfig parse_dspine_config(const json& config) {
    if (!config.is_object()) { refuse("not an object"); }
    const auto architectures = config.find("architectures");
    if (architectures == config.end() || !architectures->is_array() || architectures->empty() ||
        (*architectures)[0] != "DFlashDraftModel") {
        refuse("architectures must be [DFlashDraftModel]");
    }
    if (json_text(config, "model_type") != "qwen3") { refuse("model_type must be qwen3"); }
    if (!is_dspine_config(config)) {
        refuse("dflash_config.selected_prefix_code_bridge is missing");
    }
    const auto& draft = config.at("dflash_config");

    DSpineConfig out;
    out.hidden_size       = dimension(config, "hidden_size");
    out.intermediate_size = dimension(config, "intermediate_size");
    out.layers            = dimension(config, "num_hidden_layers");
    out.heads             = dimension(config, "num_attention_heads");
    out.kv_heads          = dimension(config, "num_key_value_heads");
    out.head_dim          = dimension(config, "head_dim");
    out.vocab_size        = dimension(config, "vocab_size");
    out.block_size        = dimension(config, "block_size");
    out.mask_token_id     = static_cast<std::uint32_t>(json_unsigned(draft, "mask_token_id"));
    out.rms_norm_eps      = positive_float(config, "rms_norm_eps");
    if (out.heads % out.kv_heads || out.head_dim % 2) { refuse("invalid attention geometry"); }
    if (out.block_size < 2 || out.block_size > 16) { refuse("block_size outside [2, 16]"); }
    if (out.mask_token_id >= out.vocab_size) { refuse("mask_token_id outside the vocabulary"); }
    if (const auto* rope =
            config.contains("rope_parameters") ? &config.at("rope_parameters") : nullptr;
        rope && rope->is_object()) {
        out.rope_theta = positive_float(*rope, "rope_theta");
    } else {
        refuse("rope_parameters.rope_theta is missing");
    }
    // The draft attention rotates and normalizes Q/K in one fused kernel with these constants.
    if (out.rope_theta != 1.0e7F || out.rms_norm_eps != 1.0e-6F) {
        refuse("RoPE theta " + std::to_string(out.rope_theta) + " and RMSNorm epsilon " +
               std::to_string(out.rms_norm_eps) +
               " are not the draft attention kernel's fixed 1e7 and 1e-6 (Tencent's Qwen3-8B "
               "DSpine trains with theta 1e6)");
    }
    if (config.contains("rope_scaling") && !config.at("rope_scaling").is_null()) {
        refuse("rope_scaling is not part of the DSpine path");
    }
    if (number_or(config, "partial_rotary_factor", 1.0) != 1.0) {
        refuse("partial_rotary_factor != 1");
    }
    // The backbone's math is fixed (a bias-free Qwen3 layer with SiLU and plain RoPE, as
    // dspine_tensors inventories it): another one is refused, not run on the wrong contract.
    if (bool_or(config, "attention_bias", false)) { refuse("attention_bias is not supported"); }
    if (bool_or(config, "mlp_bias", false)) { refuse("mlp_bias is not supported"); }
    if (const auto act = config.find("hidden_act");
        act != config.end() && !act->is_null() && *act != "silu") {
        refuse("hidden_act " + act->dump() + " is not silu");
    }
    if (const auto& rope = config.at("rope_parameters"); rope.contains("rope_type") &&
                                                         !rope.at("rope_type").is_null() &&
                                                         rope.at("rope_type") != "default") {
        refuse("rope_parameters.rope_type " + rope.at("rope_type").dump() + " is not default");
    }
    if (bool_or(config, "has_embed_tokens", false)) {
        refuse("has_embed_tokens: the target's embedding is used");
    }
    // The causal block is dflash_config.draft_block_causal (checked below); a top-level is_causal
    // may be absent (the reference runtime sets it from that), never false.
    if (!bool_or(config, "is_causal", true)) { refuse("is_causal is false (causal draft block)"); }
    if (!flag_is(draft, "draft_block_causal", true)) {
        refuse("dflash_config.draft_block_causal must be true");
    }

    const auto& layers = config.value("layer_types", json::array());
    if (!layers.is_array() || layers.size() != out.layers) {
        refuse("layer_types length mismatch");
    }
    bool sliding = false;
    for (const auto& layer : layers) {
        if (layer == "sliding_attention") {
            sliding = true;
        } else if (layer != "full_attention") {
            refuse("unknown layer_type");
        }
    }
    if (sliding) {
        out.sliding_window = dimension(config, "sliding_window");
        // A layer is local or full as the checkpoint says; a mix has no measured geometry here.
        if (std::any_of(layers.begin(), layers.end(),
                        [](const json& layer) { return layer != "sliding_attention"; })) {
            refuse("a mix of sliding and full layers is not part of the DSpine path");
        }
    }

    // The keys DFlash2 and the other drafters add; a DSpine draft is plain DFlash layers.
    for (const char* key : {"conv_kernel_size", "conv_group_size", "selector_rank",
                            "selector_top_k", "projector_type", "attention_sink_bias",
                            "attention_value_scale", "selected_prefix_feature_reliability"}) {
        require_absent_or_null(draft, key, "dflash_config");
    }
    if (bool_or(draft, "selected_prefix_feature_next_token_rows", false)) {
        refuse("next-token rows are a DSpark layout; the refinement is defined for anchor rows");
    }
    if (!flag_is(draft, "selected_prefix_terminal", false)) {
        refuse(
            "only terminal-disabled code bridges are supported (selected_prefix_terminal=false)");
    }
    if (number_or(draft, "selected_prefix_terminal_residual_rank", 0) != 0) {
        refuse("selected_prefix_terminal_residual_rank must be 0");
    }
    if (json_text(draft, "selected_prefix_feature_write_mode") != "independent" ||
        !flag_is(draft, "selected_prefix_feature_writes", true)) {
        refuse("only independent learned feature writes are supported");
    }

    // Every draft layer carries a write; the last one is redone per candidate predecessor.
    const auto active = draft.value("selected_prefix_feature_layers", json::array());
    if (!active.is_array() || active.size() != out.layers) {
        refuse("selected_prefix_feature_layers must name every draft layer");
    }
    for (std::uint32_t i = 0; i < out.layers; ++i) {
        if (!active[i].is_number_integer() || active[i].get<std::int64_t>() != i + 1) {
            refuse("selected_prefix_feature_layers must be 1..num_hidden_layers");
        }
    }

    const auto& bridge = draft.at("selected_prefix_code_bridge");
    if (!bridge.is_object()) { refuse("selected_prefix_code_bridge must be an object"); }
    out.code_size  = dimension(bridge, "code_size");
    out.rank       = dimension(bridge, "rank");
    out.bridge_eps = positive_float(bridge, "eps");

    // The serial last-write refinement: its width, stated one way or both and equal.
    std::set<std::uint32_t> widths;
    if (const auto second = config.find("code_bridge_second_pass"); second != config.end()) {
        if (!second->is_object() || json_text(*second, "mode") != "serial_table") {
            refuse("code_bridge_second_pass.mode must be serial_table");
        }
        if (json_unsigned(*second, "first_exact") != out.layers - 1 ||
            !second->contains("first_exact")) {
            refuse("code_bridge_second_pass.first_exact must be num_hidden_layers - 1");
        }
        widths.insert(dimension(*second, "head_topk"));
    }
    if (const auto refine = draft.find("selected_prefix_last_write_refine");
        refine != draft.end()) {
        if (!refine->is_object() || json_text(*refine, "kind") != "exact_sender" ||
            number_or(*refine, "version", 1) != 1) {
            refuse("selected_prefix_last_write_refine must be {kind: exact_sender, version: 1}");
        }
        widths.insert(dimension(*refine, "head_topk"));
    }
    if (widths.size() != 1) { refuse("the refinement width (head_topk) is missing or disagrees"); }
    out.head_topk = *widths.begin();
    if ((out.head_topk & (out.head_topk - 1)) || out.head_topk < 2 || out.head_topk > 64) {
        refuse("head_topk must be a power of two in [2, 64]");
    }

    const auto& taps = draft.value("target_layer_ids", json::array());
    if (!taps.is_array() || taps.empty()) { refuse("target_layer_ids is missing"); }
    for (const auto& tap : taps) {
        if (!tap.is_number_integer() || tap.get<std::int64_t>() < 0) {
            refuse("invalid target_layer_ids");
        }
        out.target_layer_ids.push_back(tap.get<std::uint32_t>());
    }
    return out;
}

std::uint64_t DSpineTensor::elements() const noexcept {
    std::uint64_t n = 1;
    for (const auto d : shape) { n *= d; }
    return n;
}

std::vector<DSpineTensor> dspine_tensors(const DSpineConfig& c) {
    const std::uint64_t h = c.hidden_size, ff = c.intermediate_size,
                        q    = std::uint64_t{c.heads} * c.head_dim,
                        kv   = std::uint64_t{c.kv_heads} * c.head_dim,
                        taps = c.target_layer_ids.size(), code = c.code_size, rank = c.rank,
                        layers = c.layers, vocab = c.vocab_size;
    std::vector<DSpineTensor> out{
        {"fc.weight", {h, taps * h}, false},
        {"hidden_norm.weight", {h}, false},
        {"norm.weight", {h}, false},
    };
    for (std::uint32_t i = 0; i < c.layers; ++i) {
        const auto p = "layers." + std::to_string(i) + ".";
        out.push_back({p + "input_layernorm.weight", {h}, false});
        out.push_back({p + "post_attention_layernorm.weight", {h}, false});
        out.push_back({p + "self_attn.q_norm.weight", {c.head_dim}, false});
        out.push_back({p + "self_attn.k_norm.weight", {c.head_dim}, false});
        out.push_back({p + "self_attn.q_proj.weight", {q, h}, false});
        out.push_back({p + "self_attn.k_proj.weight", {kv, h}, false});
        out.push_back({p + "self_attn.v_proj.weight", {kv, h}, false});
        out.push_back({p + "self_attn.o_proj.weight", {h, q}, false});
        out.push_back({p + "mlp.gate_proj.weight", {ff, h}, false});
        out.push_back({p + "mlp.up_proj.weight", {ff, h}, false});
        out.push_back({p + "mlp.down_proj.weight", {h, ff}, false});
    }
    const std::string b = "selected_prefix_bridge.";
    out.push_back({b + "codebook", {vocab, code}, true});
    out.push_back({b + "early_read_weights", {layers - 1, code, h}, true});
    out.push_back({b + "read_weight", {code, h}, true});
    out.push_back({b + "feature_write_weights", {layers, h, rank}, true});
    out.push_back({b + "gate.weight", {rank, code}, true});
    out.push_back({b + "gate.bias", {rank}, true});
    out.push_back({b + "predecessor_gate.weight", {rank, code}, true});
    out.push_back({b + "prev.weight", {rank, code}, true});
    // The terminal projection: all zero in a terminal-disabled checkpoint, never read.
    out.push_back({b + "out.weight", {code, rank}, true});
    return out;
}

std::uint64_t dspine_parameter_count(const DSpineConfig& config) {
    std::uint64_t n = 0;
    for (const auto& tensor : dspine_tensors(config)) { n += tensor.elements(); }
    return n;
}

DSpineRoundFacts dspine_round_facts(const DSpineConfig& c, std::uint32_t width) {
    if (width < 2 || width > c.block_size) { throw std::invalid_argument("DSpine round width"); }
    const std::uint64_t h = c.hidden_size, ff = c.intermediate_size,
                        q  = std::uint64_t{c.heads} * c.head_dim,
                        kv = std::uint64_t{c.kv_heads} * c.head_dim, code = c.code_size,
                        rank = c.rank, k = c.head_topk, rows = width, proposals = width - 1;
    DSpineRoundFacts out;
    for (const auto& tensor : dspine_tensors(c)) {
        if (tensor.name == "selected_prefix_bridge.codebook" ||
            tensor.name == "selected_prefix_bridge.out.weight") {
            continue; // gathered rows below; the terminal projection is never read
        }
        (tensor.bridge ? out.bridge_matrices : out.backbone_params) += tensor.elements();
    }
    out.codebook_values       = (1 + k * (proposals - 1)) * code;
    out.head_values           = k * proposals * h;
    const std::uint64_t layer = h * (q + 2 * kv) + q * h + 3 * h * ff;
    const std::uint64_t write = h * code + 3 * rank * code + rank * h;
    out.multiply_adds         = rows * c.layers * (layer + write) +
                        proposals * k * (2 * rank * code + rank * h) + proposals * k * k * h;
    return out;
}

void dspine_unit_code(std::span<const float> z, float eps, std::span<float> out) {
    double norm = 0;
    for (const float v : z) { norm += double(v) * v; }
    const double scale = std::sqrt(double(z.size())) / (std::sqrt(norm) + eps);
    for (std::size_t i = 0; i < z.size(); ++i) { out[i] = static_cast<float>(z[i] * scale); }
}

void dspine_anchor_code(std::span<const float> row, std::span<float> out) {
    const float scale = std::sqrt(static_cast<float>(row.size()));
    for (std::size_t i = 0; i < row.size(); ++i) { out[i] = row[i] * scale; }
}

void dspine_message(std::span<const float> query, std::span<const float> sender,
                    std::span<const float> value, std::span<const float> bias,
                    std::span<float> out) {
    for (std::size_t i = 0; i < out.size(); ++i) {
        const float gate = 1.0F / (1.0F + std::exp(-(query[i] + sender[i] + bias[i])));
        out[i]           = gate * value[i];
    }
}

void dspine_write(std::span<const float> raw, std::span<const float> delta, float rms,
                  std::span<float> out) {
    for (std::size_t i = 0; i < out.size(); ++i) { out[i] = raw[i] + rms * delta[i]; }
}

std::vector<std::uint32_t> dspine_walk_greedy(std::span<const float> edges, std::uint32_t k,
                                              std::uint32_t proposals) {
    std::vector<std::uint32_t> lanes(proposals);
    std::uint32_t previous = 0;
    for (std::uint32_t i = 0; i < proposals; ++i) {
        const float* row   = edges.data() + (std::size_t{i} * k + (i == 0 ? 0 : previous)) * k;
        std::uint32_t best = 0;
        for (std::uint32_t c = 1; c < k; ++c) {
            if (row[c] > row[best]) { best = c; }
        }
        lanes[i] = previous = best;
    }
    return lanes;
}

std::vector<std::uint32_t> dspine_walk_sample(std::span<const float> edges, std::uint32_t k,
                                              std::uint32_t proposals, float temperature,
                                              std::span<const float> uniforms, std::span<float> q) {
    if (!(temperature > 0)) {
        // A cold row is the greedy walk with a one-hot distribution.
        auto lanes = dspine_walk_greedy(edges, k, proposals);
        std::fill(q.begin(), q.begin() + std::size_t{proposals} * k, 0.0F);
        for (std::uint32_t i = 0; i < proposals; ++i) { q[std::size_t{i} * k + lanes[i]] = 1.0F; }
        return lanes;
    }
    std::vector<std::uint32_t> lanes(proposals);
    std::uint32_t previous = 0;
    for (std::uint32_t i = 0; i < proposals; ++i) {
        const float* row = edges.data() + (std::size_t{i} * k + (i == 0 ? 0 : previous)) * k;
        float top        = row[0];
        for (std::uint32_t c = 1; c < k; ++c) { top = std::max(top, row[c]); }
        double total = 0;
        for (std::uint32_t c = 0; c < k; ++c) {
            total += std::exp((double(row[c]) - top) / temperature);
        }
        double mass        = 0;
        std::uint32_t pick = k;
        std::uint32_t last = 0; // the last lane with mass: where rounding past the end lands
        for (std::uint32_t c = 0; c < k; ++c) {
            const double p            = std::exp((double(row[c]) - top) / temperature) / total;
            q[std::size_t{i} * k + c] = static_cast<float>(p);
            mass += p;
            if (p > 0) { last = c; }
            if (pick == k && uniforms[i] < mass) { pick = c; }
        }
        if (pick == k) { pick = last; }
        lanes[i] = previous = pick;
    }
    return lanes;
}

} // namespace ninferno::spec
