#include "convert/dense.h"

#include "convert/checks.h"
#include "convert/formats.h"
#include "convert/numerics.h"
#include "core/yarn.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <mutex>
#include <numbers>
#include <tuple>
#include <optional>
#include <set>
#include <stdexcept>

namespace ninferno::convert {
namespace {

// ---------------------------------------------------------------------------------------------
// Architectures and their transformers conventions.

struct KnownSource {
    const char* architecture;
    const char* model_type; // the text decoder's model_type
};

constexpr KnownSource kSources[] = {
    {"LlamaForCausalLM", "llama"},
    {"MistralForCausalLM", "mistral"},
    {"Ministral3ForCausalLM", "ministral3"},
    {"Mistral3ForConditionalGeneration", "ministral3"}, // or "mistral" (Mistral Small 3.x)
    {"GraniteForCausalLM", "granite"},
    {"Olmo2ForCausalLM", "olmo2"},
    {"Olmo3ForCausalLM", "olmo3"},
    {"SeedOssForCausalLM", "seed_oss"},
    {"Qwen2ForCausalLM", "qwen2"},
    {"Qwen3ForCausalLM", "qwen3"},
    {"SmolLM3ForCausalLM", "smollm3"},
    {"Phi3ForCausalLM", "phi3"},
    {"Gemma3ForCausalLM", "gemma3_text"},
    {"Gemma3ForConditionalGeneration", "gemma3"}, // its text_config: gemma3_text
    {"MuseGlimmerForConditionalGeneration", "muse_glimmer"}, // text_config: muse_glimmer_text
    // MoE decoders: the same layer with a routed feed-forward (model_type "moe_decoder").
    {"GptOssForCausalLM", "gpt_oss"},
    {"Qwen3MoeForCausalLM", "qwen3_moe"},
    // Hybrid Mamba2 / attention / MoE decoders (model_type "nemotron_h", models/nemotron_h).
    {"NemotronHForCausalLM", "nemotron_h"},
    // The MTP (NextN) drafter of a NemotronH MoE target (model_type "nemotron_h_mtp").
    {"NemotronHMtpForCausalLM", "nemotron_h_mtp"},
};

const std::set<std::string> kTextModelTypes = {"llama",   "mistral",  "ministral3", "granite",
                                               "olmo2",   "olmo3",    "seed_oss",   "qwen2",
                                               "qwen3",   "smollm3",  "gpt_oss",    "qwen3_moe",
                                               "phi3",    "gemma3_text", "muse_glimmer_text",
                                               "nemotron_h", "nemotron_h_mtp"};

bool is_moe_type(const std::string& model_type) {
    return model_type == "gpt_oss" || model_type == "qwen3_moe" || model_type == "nemotron_h";
}

[[noreturn]] void refuse(const std::string& what) {
    throw std::runtime_error("dense decoder: " + what);
}

bool truthy(const OJson* value) {
    return value != nullptr && !value->is_null() &&
           !(value->is_boolean() && !value->get<bool>()) &&
           !(value->is_number() && value->get<double>() == 0.0);
}

bool flag(const OJson& raw, const char* key) {
    const auto* v = member(raw, key);
    if (v == nullptr || v->is_null()) { return false; }
    if (!v->is_boolean()) { refuse(std::string(key) + " must be boolean"); }
    return v->get<bool>();
}

double number(const OJson& raw, const char* key, double fallback) {
    const auto* v = member(raw, key);
    if (v == nullptr || v->is_null()) { return fallback; }
    if (!v->is_number()) { refuse(std::string(key) + " must be a number"); }
    return v->get<double>();
}

// ---------------------------------------------------------------------------------------------
// Rotary frequencies, as transformers' ROPE_INIT_FUNCTIONS compute them in FP32
// (modeling_rope_utils.py): default, linear, llama3, yarn.

std::vector<float> base_frequencies(double theta, std::uint64_t dim) {
    // 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)), in binary32.
    std::vector<float> out;
    for (std::uint64_t i = 0; i < dim; i += 2) {
        const float exponent = static_cast<float>(i) / static_cast<float>(dim);
        out.push_back(1.0F / std::pow(static_cast<float>(theta), exponent));
    }
    return out;
}

struct Rotary {
    std::vector<float> inv_freq;
    double attention_scaling = 1.0;
    std::string type;
    // LongRoPE: the frequencies of the long-context mode and the position count past which it
    // applies (original_max_position_embeddings); empty when the rope has one mode.
    std::vector<float> long_inv_freq;
    std::uint64_t long_after = 0;
};

// LongRoPE's per-pair rescale: 1 / (factor[i] * theta^(2i/dim)), transformers'
// _compute_longrope_parameters in binary32.
std::vector<float> longrope_frequencies(const OJson& factors, double theta, std::uint64_t dim,
                                        const std::string& where) {
    if (!factors.is_array() || factors.size() != dim / 2) {
        refuse(where + ": longrope factors must hold rotary_dim/2 values");
    }
    std::vector<float> out;
    for (std::uint64_t i = 0; i < dim; i += 2) {
        const auto factor = static_cast<float>(factors[i / 2].get<double>());
        if (!(factor > 0)) { refuse(where + ": longrope factors must be positive"); }
        const float exponent = static_cast<float>(i) / static_cast<float>(dim);
        out.push_back(1.0F / (factor * std::pow(static_cast<float>(theta), exponent)));
    }
    return out;
}

Rotary rotary(const OJson& p, std::uint64_t dim, std::uint64_t max_positions,
              const std::string& where) {
    std::string type = "default";
    if (const auto* t = member(p, "rope_type"); t && t->is_string()) {
        type = t->get<std::string>();
    } else if (const auto* t2 = member(p, "type"); t2 && t2->is_string()) {
        type = t2->get<std::string>();
    }
    const double theta = number(p, "rope_theta", 10000.0);
    if (!(theta > 0)) { refuse(where + ".rope_theta must be positive"); }
    Rotary out;
    out.type     = type;
    out.inv_freq = base_frequencies(theta, dim);
    if (type == "default") { return out; }
    if (type == "frequency_factors") {
        // llama.cpp's rope_freqs tensor (a GGUF Llama 3.x): pair i turns at inv_freq_i / factor_i.
        const auto* factors = member(p, "factors");
        if (factors == nullptr || !factors->is_array() ||
            factors->size() != out.inv_freq.size()) {
            refuse(where + ": frequency_factors needs one factor per rotated pair");
        }
        for (std::size_t i = 0; i < out.inv_freq.size(); ++i) {
            const auto& factor = (*factors)[i];
            if (!factor.is_number() || !(factor.get<double>() > 0)) {
                refuse(where + ": frequency factors must be positive");
            }
            out.inv_freq[i] /= static_cast<float>(factor.get<double>());
        }
        return out;
    }
    if (type == "longrope" || type == "su") {
        // transformers: original_max_position_embeddings sets the factor (max / original), else
        // the rope's own factor; attention factor sqrt(1 + ln(factor) / ln(original)) unless given.
        // Sequences up to `original` positions use the short factors, longer ones the long.
        const auto* orig = member(p, "original_max_position_embeddings");
        if (!truthy(orig)) { refuse(where + ": longrope needs original_max_position_embeddings"); }
        const double original = orig->get<double>();
        const double factor   = static_cast<double>(max_positions) / original;
        const auto* attention = member(p, "attention_factor");
        out.attention_scaling =
            truthy(attention) ? attention->get<double>()
            : factor <= 1     ? 1.0
                              : std::sqrt(1 + std::log(factor) / std::log(original));
        const auto* short_f = member(p, "short_factor");
        const auto* long_f  = member(p, "long_factor");
        if (short_f == nullptr || long_f == nullptr) {
            refuse(where + ": longrope needs short_factor and long_factor");
        }
        out.type          = "longrope";
        out.inv_freq      = longrope_frequencies(*short_f, theta, dim, where);
        out.long_inv_freq = longrope_frequencies(*long_f, theta, dim, where);
        out.long_after    = static_cast<std::uint64_t>(original);
        return out;
    }
    if (type == "linear") {
        const auto factor = static_cast<float>(number(p, "factor", 1.0));
        for (float& f : out.inv_freq) { f /= factor; }
        return out;
    }
    if (type == "llama3") {
        const double factor = number(p, "factor", 0);
        const double low    = number(p, "low_freq_factor", 0);
        const double high   = number(p, "high_freq_factor", 0);
        const double old    = number(p, "original_max_position_embeddings", 0);
        if (!(factor > 0 && low > 0 && high > low && old > 0)) {
            refuse(where + ": incomplete llama3 rope parameters");
        }
        const double low_wavelen  = old / low;
        const double high_wavelen = old / high;
        for (float& f : out.inv_freq) {
            const double wavelen = 2 * std::numbers::pi / f;
            const float scaled   = wavelen > low_wavelen ? static_cast<float>(f / factor) : f;
            const double smooth  = (old / wavelen - low) / (high - low);
            const bool medium    = !(wavelen < high_wavelen) && !(wavelen > low_wavelen);
            if (medium) {
                f = static_cast<float>((1 - smooth) * scaled / factor + smooth * scaled);
            } else {
                f = scaled;
            }
        }
        return out;
    }
    if (type == "yarn") {
        const auto* orig = member(p, "original_max_position_embeddings");
        YarnParameters yarn;
        yarn.original = truthy(orig) ? orig->get<double>() : static_cast<double>(max_positions);
        yarn.factor   = number(p, "factor", 0);
        if (!(yarn.factor > 0)) { yarn.factor = static_cast<double>(max_positions) / yarn.original; }
        if (const auto* attention = member(p, "attention_factor"); truthy(attention)) {
            yarn.attention_factor = attention->get<double>();
        }
        const auto* m  = member(p, "mscale");
        const auto* ma = member(p, "mscale_all_dim");
        if (truthy(m) && truthy(ma)) {
            yarn.mscale         = m->get<double>();
            yarn.mscale_all_dim = ma->get<double>();
        }
        yarn.beta_fast = truthy(member(p, "beta_fast")) ? number(p, "beta_fast", 32) : 32;
        yarn.beta_slow = truthy(member(p, "beta_slow")) ? number(p, "beta_slow", 1) : 1;
        const auto* truncate = member(p, "truncate");
        yarn.truncate        = truncate == nullptr || truncate->is_null() || truthy(truncate);
        YarnRope rope         = yarn_rope(theta, dim, yarn);
        out.inv_freq          = std::move(rope.inv_freq);
        out.attention_scaling = rope.attention_scaling;
        return out;
    }
    refuse(where + ": rope_type " + type + " is not implemented");
}

// The rope parameters that apply to a layer type: transformers-5 rope_parameters (flat, or keyed
// by layer type), else the legacy rope_theta + rope_scaling pair. OLMo 3 applies its scaling to
// full-attention layers only (Olmo3Config.convert_rope_params_to_dict).
OJson rope_parameters(const OJson& raw, const std::string& model_type,
                      const std::string& layer_type) {
    if (const auto* p = member(raw, "rope_parameters"); p && p->is_object()) {
        if (const auto* typed = member(*p, layer_type); typed && typed->is_object()) {
            return *typed;
        }
        if (member(*p, "rope_theta") || member(*p, "rope_type")) { return *p; }
    }
    OJson out = OJson::object();
    // Gemma 3 before transformers 5: sliding layers rotate unscaled at rope_local_base_freq
    // (Gemma3TextConfig's conversion), full layers take rope_theta and rope_scaling.
    if (const auto* local = member(raw, "rope_local_base_freq");
        model_type == "gemma3_text" && layer_type == "sliding_attention" && truthy(local)) {
        out["rope_type"]  = "default";
        out["rope_theta"] = local->get<double>();
        return out;
    }
    const auto* scaling = member(raw, "rope_scaling");
    const bool scaled   = scaling != nullptr && scaling->is_object() &&
                        !(model_type == "olmo3" && layer_type != "full_attention");
    if (scaled) { out = *scaling; }
    out["rope_theta"] = number(raw, "rope_theta", 10000.0);
    // Phi-3 keeps original_max_position_embeddings beside rope_scaling, not in it.
    if (const auto* orig = member(raw, "original_max_position_embeddings");
        scaled && truthy(orig) && !out.contains("original_max_position_embeddings")) {
        out["original_max_position_embeddings"] = *orig;
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// config.json -> dense_decoder

struct Resolved {
    OJson config;
    std::string root;     // checkpoint prefix of the decoder ("model.", "language_model.model.")
    std::string head;     // lm_head tensor name
    double embedding_multiplier = 1.0;
    double logits_scaling       = 1.0;
};

const OJson& text_part(const OJson& source) {
    const auto* nested = member(source, "text_config");
    return nested != nullptr && nested->is_object() ? *nested : source;
}

// The routed feed-forward, in the generic MoE Op's terms (models/dense/config.h MoeSpec).
OJson moe_block(const OJson& raw, const std::string& model_type) {
    OJson m = OJson::object();
    if (model_type == "gpt_oss") {
        // transformers GptOssTopKRouter / GptOssExperts: top-k of the biased logits, softmax over
        // the k (= the renormalized full softmax), clamped SwiGLU with alpha 1.702 (a constant of
        // the modeling code) and limit swiglu_limit, gate/up rows interleaved, both biased.
        const auto* per_tok = member(raw, "num_experts_per_tok");
        m["num_experts"] = positive(member(raw, "num_local_experts"), "text.num_local_experts");
        m["num_experts_per_tok"] =
            present(per_tok) ? positive(per_tok, "text.num_experts_per_tok")
                             : positive(member(raw, "experts_per_token"), "text.experts_per_token");
        m["moe_intermediate_size"] = positive(member(raw, "intermediate_size"), "text.intermediate_size");
        m["scoring"]        = "softmax";
        m["renormalize"]    = true;
        m["activation"]     = "clamped_swiglu";
        m["alpha"]          = 1.702;
        m["limit"]          = number(raw, "swiglu_limit", 7.0);
        m["gate_up_layout"] = "interleaved";
        m["router_bias"]    = true;
        m["expert_bias"]    = true;
    } else if (model_type == "qwen3_moe") {
        // transformers Qwen3MoeSparseMoeBlock: softmax over every expert, top-k, renormalized
        // when norm_topk_prob (its default is false), SwiGLU experts, no shared expert.
        if (const auto* only = member(raw, "mlp_only_layers");
            only && only->is_array() && !only->empty()) {
            refuse("mlp_only_layers (dense layers inside a MoE decoder) is not implemented");
        }
        if (number(raw, "decoder_sparse_step", 1.0) != 1.0) {
            refuse("decoder_sparse_step other than 1 is not implemented");
        }
        m["num_experts"]           = positive(member(raw, "num_experts"), "text.num_experts");
        m["num_experts_per_tok"]   = positive(member(raw, "num_experts_per_tok"), "text.num_experts_per_tok");
        m["moe_intermediate_size"] = positive(member(raw, "moe_intermediate_size"), "text.moe_intermediate_size");
        m["scoring"]        = "softmax";
        m["renormalize"]    = flag(raw, "norm_topk_prob");
        m["activation"]     = "swiglu";
        m["gate_up_layout"] = "concatenated";
        m["router_bias"]    = false;
        m["expert_bias"]    = false;
    }
    m["shared_intermediate_size"] = 0;
    return m;
}

// ---------------------------------------------------------------------------------------------
// config.json -> nemotron_h (transformers NemotronHForCausalLM)
//
// One mixer per block, x += mixer(RMSNorm(x)): a Mamba2 mixer, a NoPE attention (no q/k norm,
// scores scaled by head_dim^-1/2) or a MoE (sigmoid router with a selection-only correction bias,
// top-k, renormalized, times routed_scaling_factor; non-gated relu^2 experts and one relu^2
// shared expert). The Mamba2 time step is softplus(dt + dt_bias) clamped to time_step_limit, as
// the mamba_ssm kernels the model was trained with clamp it (transformers' CPU fallback clamps to
// time_step_min instead; that is not the model's arithmetic).

double limit_value(const OJson& v, double infinite) {
    if (v.is_number()) { return v.get<double>(); }
    if (v.is_object() && v.contains("__float__")) {
        const auto text = v["__float__"].get<std::string>();
        if (text == "Infinity") { return infinite; }
    }
    refuse("time_step_limit entries must be numbers");
}

// The routed MoE of a NemotronH trunk and of its MTP drafter: sigmoid-scored, relu^2 experts
// and one shared expert. A zero-padded routed up bank (gguf_model.cpp) states its K past
// hidden_size as moe_expert_input_size.
OJson resolve_nemotron_h_moe(const OJson& raw) {
    OJson m          = OJson::object();
    m["num_experts"] = positive(member(raw, "n_routed_experts"), "text.n_routed_experts");
    m["num_experts_per_tok"] =
        positive(member(raw, "num_experts_per_tok"), "text.num_experts_per_tok");
    m["moe_intermediate_size"] =
        positive(member(raw, "moe_intermediate_size"), "text.moe_intermediate_size");
    m["shared_intermediate_size"] = positive(member(raw, "moe_shared_expert_intermediate_size"),
                                             "text.moe_shared_expert_intermediate_size");
    if (const OJson* input = member(raw, "moe_expert_input_size")) {
        m["expert_input_size"] = positive(input, "text.moe_expert_input_size");
    }
    m["scoring"]               = "sigmoid";
    m["renormalize"]           = flag(raw, "norm_topk_prob");
    m["routed_scaling_factor"] = number(raw, "routed_scaling_factor", 1.0);
    m["activation"]            = "relu2";
    return m;
}

Resolved resolve_nemotron_h(const OJson& raw, const std::string& architecture,
                            bool checkpoint_has_head) {
    for (const char* key : {"mamba_proj_bias", "use_bias", "attention_bias", "mlp_bias"}) {
        if (flag(raw, key)) { refuse(std::string(key) + " is not implemented"); }
    }
    if (truthy(member(raw, "moe_latent_size"))) { refuse("moe_latent_size is not implemented"); }
    if (number(raw, "n_group", 1.0) != 1.0 || number(raw, "topk_group", 1.0) != 1.0) {
        refuse("grouped expert selection (n_group, topk_group) is not implemented");
    }
    if (number(raw, "n_shared_experts", 1.0) != 1.0) { refuse("n_shared_experts must be 1"); }
    if (const auto* v = member(raw, "mlp_hidden_act");
        v && v->is_string() && v->get<std::string>() != "relu2") {
        refuse("mlp_hidden_act " + v->get<std::string>() + " is not implemented");
    }
    if (const auto* v = member(raw, "mamba_hidden_act");
        v && v->is_string() && v->get<std::string>() != "silu") {
        refuse("mamba_hidden_act " + v->get<std::string>() + " is not implemented");
    }
    OJson kinds = OJson::array();
    if (const auto* types = member(raw, "layers_block_type"); types && types->is_array()) {
        for (const auto& kind : *types) { kinds.push_back(kind); }
    } else if (const auto* pattern = member(raw, "hybrid_override_pattern");
               pattern && pattern->is_string()) {
        for (const char k : pattern->get<std::string>()) {
            kinds.push_back(k == 'M' ? "mamba" : k == '*' ? "attention" : k == 'E' ? "moe" : "mlp");
        }
    } else {
        refuse("neither layers_block_type nor hybrid_override_pattern is given");
    }
    // transformers 5 derives num_hidden_layers from the pattern; an explicit count must agree.
    const std::uint64_t layers = kinds.size();
    if (present(member(raw, "num_hidden_layers")) &&
        positive(member(raw, "num_hidden_layers"), "text.num_hidden_layers") != layers) {
        refuse("the block pattern must cover every layer");
    }
    for (const auto& kind : kinds) {
        if (kind != "mamba" && kind != "attention" && kind != "moe") {
            refuse("block type " + kind.dump() + " is not implemented");
        }
    }
    const auto hidden   = positive(member(raw, "hidden_size"), "text.hidden_size");
    const auto heads    = positive(member(raw, "num_attention_heads"), "text.num_attention_heads");
    const auto kv       = positive(member(raw, "num_key_value_heads"), "text.num_key_value_heads");
    const auto head_dim = present(member(raw, "head_dim"))
                              ? positive(member(raw, "head_dim"), "text.head_dim")
                              : hidden / heads;
    const auto m_heads  = positive(member(raw, "mamba_num_heads"), "text.mamba_num_heads");
    const auto m_dim    = positive(member(raw, "mamba_head_dim"), "text.mamba_head_dim");
    const auto groups   = positive(member(raw, "n_groups"), "text.n_groups");
    const auto state    = positive(member(raw, "ssm_state_size"), "text.ssm_state_size");
    const auto conv     = positive(member(raw, "conv_kernel"), "text.conv_kernel");
    if (conv != 4) { refuse("conv_kernel other than 4 is not implemented"); }
    if (m_heads % groups != 0) { refuse("mamba_num_heads must be a multiple of n_groups"); }
    double dt_min = 0.0, dt_max = 0.0; // dt_max 0: no upper clamp
    if (const auto* limit = member(raw, "time_step_limit"); limit && limit->is_array()) {
        if (limit->size() != 2) { refuse("time_step_limit must be a pair"); }
        dt_min             = limit_value((*limit)[0], 0.0);
        const double upper = limit_value((*limit)[1], -1.0);
        dt_max             = upper < 0 ? 0.0 : upper;
    }

    Resolved out;
    OJson c                      = OJson::object();
    c["architectures"]           = OJson::array({"NemotronHForCausalLM"});
    c["model_type"]              = "nemotron_h";
    c["source_architecture"]     = architecture;
    c["source_model_type"]       = "nemotron_h";
    c["hidden_size"]             = hidden;
    c["vocab_size"]              = positive(member(raw, "vocab_size"), "text.vocab_size");
    c["num_hidden_layers"]       = layers;
    c["max_position_embeddings"] =
        positive(member(raw, "max_position_embeddings"), "text.max_position_embeddings");
    c["rms_norm_eps"] =
        f32_value(member(raw, "layer_norm_epsilon"), 1e-5, "text.layer_norm_epsilon");
    c["layer_types"]         = std::move(kinds);
    c["num_attention_heads"] = heads;
    c["num_key_value_heads"] = kv;
    c["head_dim"]            = head_dim;
    c["attention_scale"] =
        static_cast<double>(static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim))));
    OJson mamba          = OJson::object();
    mamba["num_heads"]   = m_heads;
    mamba["head_dim"]    = m_dim;
    mamba["n_groups"]    = groups;
    mamba["state_size"]  = state;
    mamba["conv_kernel"] = conv;
    mamba["conv_bias"]   = flag(raw, "use_conv_bias");
    mamba["dt_min"]      = dt_min;
    mamba["dt_max"]      = dt_max;
    c["mamba"]           = std::move(mamba);
    c["moe"]             = resolve_nemotron_h_moe(raw);
    const OJson* tied = member(raw, "tie_word_embeddings");
    c["tie_word_embeddings"] =
        tied != nullptr && tied->is_boolean() ? tied->get<bool>() : !checkpoint_has_head;
    out.config = std::move(c);
    return out;
}

// The MTP drafter's config, defined with the other NemotronH builders below.
Resolved resolve_nemotron_h_mtp(const OJson& raw, const std::string& architecture);

Resolved resolve(const OJson& source, bool checkpoint_has_head) {
    const auto architecture = declared_architecture(source);
    if (architecture == "NemotronHMtpForCausalLM") {
        return resolve_nemotron_h_mtp(source, architecture);
    }
    if (!is_dense_architecture(architecture)) {
        refuse("unsupported architecture " + architecture);
    }
    const OJson& raw = text_part(source);
    const auto* type_value = member(raw, "model_type");
    const std::string model_type =
        type_value && type_value->is_string() ? type_value->get<std::string>() : "";
    if (!kTextModelTypes.contains(model_type)) {
        refuse("text model_type '" + model_type + "' is not a dense decoder this adapter knows");
    }
    if (model_type == "nemotron_h") {
        return resolve_nemotron_h(raw, architecture, checkpoint_has_head);
    }
    if (flag(raw, "mlp_bias")) { refuse("mlp_bias is not implemented"); }
    if (truthy(member(raw, "attn_logit_softcapping"))) {
        refuse("attn_logit_softcapping (a capped attention score) is not implemented");
    }
    if (flag(raw, "use_sliding_window") && model_type != "mistral") {
        refuse("use_sliding_window (a partial sliding-window layout) is not implemented");
    }
    const auto hidden = positive(member(raw, "hidden_size"), "text.hidden_size");
    const auto heads  = positive(member(raw, "num_attention_heads"), "text.num_attention_heads");
    const auto* kv_value = member(raw, "num_key_value_heads");
    const auto kv =
        present(kv_value) ? positive(kv_value, "text.num_key_value_heads") : heads;
    const auto* head_dim_value = member(raw, "head_dim");
    const auto head_dim = present(head_dim_value) ? positive(head_dim_value, "text.head_dim")
                                                  : hidden / heads;
    const auto layers = positive(member(raw, "num_hidden_layers"), "text.num_hidden_layers");
    const auto max_positions =
        positive(member(raw, "max_position_embeddings"), "text.max_position_embeddings");
    const auto* act = member(raw, "hidden_act");
    if (act == nullptr || act->is_null()) { act = member(raw, "hidden_activation"); } // Gemma
    const std::string activation = act && act->is_string() ? act->get<std::string>() : "silu";
    if (activation != "silu" && activation != "gelu_pytorch_tanh") {
        refuse("hidden_act " + activation + " is not implemented");
    }

    Resolved out;
    OJson c                    = OJson::object();
    const bool moe             = is_moe_type(model_type);
    c["architectures"] = OJson::array({moe ? "MoeDecoderForCausalLM" : "DenseDecoderForCausalLM"});
    c["model_type"]    = moe ? "moe_decoder" : "dense_decoder";
    c["source_architecture"]   = architecture;
    c["source_model_type"]     = model_type;
    c["hidden_size"]           = hidden;
    c["vocab_size"]            = positive(member(raw, "vocab_size"), "text.vocab_size");
    c["num_hidden_layers"]     = layers;
    c["max_position_embeddings"] = max_positions;
    c["num_attention_heads"]   = heads;
    c["num_key_value_heads"]   = kv;
    c["head_dim"]              = head_dim;
    c["intermediate_size"] = positive(member(raw, "intermediate_size"), "text.intermediate_size");
    c["rms_norm_eps"]      = f32_value(member(raw, "rms_norm_eps"), 1e-6, "text.rms_norm_eps");
    c["hidden_act"]        = activation;
    const bool olmo        = model_type == "olmo2" || model_type == "olmo3";
    const bool gemma       = model_type == "gemma3_text";
    // Muse Glimmer (transformers MuseGlimmerTextModel): Gemma 2's sandwich of (1 + w) norms with
    // its own post-norm epsilon, a final norm with the plain gain w, q/k norms without weight, an
    // RMS-normalized (weightless) embedding, a sigmoid gate on the attention context
    // (self_attn.gate_proj) and logits multiplied by output_multiplier before the soft cap.
    const bool muse        = model_type == "muse_glimmer_text";
    // Gemma 3: pre- and post-norms around both sublayers, every gain 1 + w (Gemma3RMSNorm).
    c["norm_placement"]    = olmo ? "post" : gemma || muse ? "sandwich" : "pre";
    c["norm_unit_offset"]  = gemma || muse;
    c["qk_norm"]           = olmo ? "projection"
                             : model_type == "qwen3" || model_type == "qwen3_moe" || gemma ? "head"
                             : muse                                                        ? "head_unscaled"
                                                                                           : "none";
    if (muse) {
        c["final_norm_unit_offset"] = false;
        c["post_norm_eps"]          = f32_value(member(raw, "post_norm_eps"), 1e-8, "text.post_norm_eps");
        c["embedding_norm"]         = true;
        c["attention_output_gate"]  = true;
        c["logit_multiplier"] =
            f32_value(member(raw, "output_multiplier"), 1.0, "text.output_multiplier");
    }
    if (moe) { c["moe"] = moe_block(raw, model_type); }
    // gpt-oss: a learned sink logit per query head joins every softmax denominator.
    c["attention_sinks"] = model_type == "gpt_oss";
    if (const auto* cap = member(raw, "final_logit_softcapping"); truthy(cap)) {
        c["final_logit_softcapping"] = cap->get<double>();
    }

    // Biases, by each model's attention module.
    bool qkv_bias = false, output_bias = false;
    if (model_type == "qwen2") {
        qkv_bias = true;
    } else if (model_type == "seed_oss") {
        qkv_bias    = flag(raw, "attention_bias");
        output_bias = flag(raw, "attention_out_bias");
    } else if (model_type != "mistral" && model_type != "ministral3") {
        qkv_bias = output_bias = flag(raw, "attention_bias");
    }
    c["attention_bias"]        = qkv_bias;
    c["attention_output_bias"] = output_bias;

    // Scales and the muP multipliers (Granite); the embedding and logit ones are folded into the
    // tables below and recorded here for provenance.
    double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    if (model_type == "granite") {
        scale                    = number(raw, "attention_multiplier", scale);
        out.embedding_multiplier = number(raw, "embedding_multiplier", 1.0);
        out.logits_scaling       = number(raw, "logits_scaling", 1.0);
        c["residual_multiplier"] = number(raw, "residual_multiplier", 1.0);
    }
    // Gemma: scores scaled by query_pre_attn_scalar^-1/2; embeddings by sqrt(hidden_size), which
    // transformers rounds to the weights' BF16 before multiplying (Gemma3TextScaledWordEmbedding).
    if (const auto* scalar = member(raw, "query_pre_attn_scalar"); truthy(scalar)) {
        scale = 1.0 / std::sqrt(scalar->get<double>());
    }
    // Muse Glimmer multiplies the normalized queries by qk_scale_factor before the standard
    // 1/sqrt(head_dim) scores: one scale on the scores.
    if (const auto* factor = member(raw, "qk_scale_factor"); muse && truthy(factor)) {
        scale *= factor->get<double>();
    }
    if (gemma) {
        out.embedding_multiplier =
            static_cast<double>(bf16_to_f32(f32_to_bf16(static_cast<float>(
                std::sqrt(static_cast<double>(hidden))))));
    }
    c["attention_scale"]      = static_cast<double>(static_cast<float>(scale));
    c["embedding_multiplier"] = out.embedding_multiplier;
    c["logits_scaling"]       = out.logits_scaling;
    if (!(out.logits_scaling > 0) || !(out.embedding_multiplier > 0)) {
        refuse("embedding_multiplier and logits_scaling must be positive");
    }

    // Layer pattern: explicit layer_types, or one window on every layer (Mistral's
    // sliding_window, which Ministral 3 and Mistral 3.x leave null), and NoPE layers (SmolLM3).
    OJson kinds = OJson::array();
    if (const auto* types = member(raw, "layer_types"); types && types->is_array()) {
        if (types->size() != layers) { refuse("layer_types must cover every layer"); }
        for (const auto& kind : *types) {
            if (kind != "sliding_attention" && kind != "full_attention") {
                refuse("unknown layer type " + kind.dump());
            }
            kinds.push_back(kind);
        }
    } else if (const auto* pattern = member(raw, "sliding_window_pattern");
               truthy(pattern) && !member(raw, "layer_types")) {
        // Gemma 3 before transformers 5: every pattern-th layer full, the others sliding.
        const auto every = positive(pattern, "text.sliding_window_pattern");
        for (std::uint64_t i = 0; i < layers; ++i) {
            kinds.push_back((i + 1) % every ? "sliding_attention" : "full_attention");
        }
    } else {
        // Mistral's sliding_window (Ministral 3 and Mistral 3.x leave it null) and Phi-3's (every
        // layer; transformers' Phi3Attention passes it to each). A window covering every position
        // excludes no key: those layers are full ones.
        const auto* window = member(raw, "sliding_window");
        const bool slides =
            present(window) && !window->is_null() &&
            (model_type == "mistral" || model_type == "ministral3" || model_type == "phi3") &&
            (model_type != "mistral" || flag(raw, "use_sliding_window") ||
             !member(raw, "use_sliding_window")) &&
            window->get<double>() < static_cast<double>(max_positions);
        for (std::uint64_t i = 0; i < layers; ++i) {
            kinds.push_back(slides ? "sliding_attention" : "full_attention");
        }
    }
    if (std::any_of(kinds.begin(), kinds.end(),
                    [](const OJson& k) { return k == "sliding_attention"; })) {
        c["sliding_window"] = positive(member(raw, "sliding_window"), "text.sliding_window");
    }
    c["layer_types"] = kinds;

    std::vector<bool> rotated(layers, true);
    if (const auto* nope = member(raw, "no_rope_layers"); nope && nope->is_array()) {
        if (nope->size() != layers) { refuse("no_rope_layers must cover every layer"); }
        for (std::uint64_t i = 0; i < layers; ++i) { rotated[i] = truthy(&(*nope)[i]); }
    }
    // A per-layer rope base (Muse Glimmer's layer_rope_theta): 0 is a NoPE layer; any other value
    // must be the base its layer type's rope parameters already carry (one group per type).
    if (const auto* thetas = member(raw, "layer_rope_theta"); thetas && thetas->is_array()) {
        if (thetas->size() != layers) { refuse("layer_rope_theta must cover every layer"); }
        for (std::uint64_t i = 0; i < layers; ++i) {
            const double theta = (*thetas)[i].get<double>();
            if (theta == 0) {
                rotated[i] = false;
                continue;
            }
            const auto p = rope_parameters(raw, model_type, kinds[i].get<std::string>());
            if (number(p, "rope_theta", 10000.0) != theta) {
                refuse("layer_rope_theta differs from its layer type's rope_theta at layer " +
                       std::to_string(i));
            }
        }
    }
    // One rope group per layer type (their parameters may differ, OLMo 3), NoPE layers none.
    OJson groups = OJson::object();
    OJson layer_rope = OJson::array();
    std::optional<std::pair<double, std::uint64_t>> temperature;
    for (std::uint64_t i = 0; i < layers; ++i) {
        const auto kind = kinds[i].get<std::string>();
        if (!rotated[i]) {
            layer_rope.push_back(nullptr);
            continue;
        }
        if (!groups.contains(kind)) {
            const auto p       = rope_parameters(raw, model_type, kind);
            const double partial =
                number(p, "partial_rotary_factor", number(raw, "partial_rotary_factor", 1.0));
            const auto dim = static_cast<std::uint64_t>(static_cast<double>(head_dim) * partial);
            if (dim == 0 || dim % 2 || dim > head_dim) { refuse("invalid partial rotary width"); }
            const auto r = rotary(p, dim, max_positions, "text.rope(" + kind + ")");
            OJson g                = OJson::object();
            g["rope_type"]         = r.type;
            g["rope_theta"]        = number(p, "rope_theta", 10000.0);
            g["rotary_dim"]        = dim;
            g["attention_scaling"] = r.attention_scaling;
            OJson freq             = OJson::array();
            for (const float f : r.inv_freq) { freq.push_back(static_cast<double>(f)); }
            g["inv_freq"] = std::move(freq);
            if (!r.long_inv_freq.empty()) {
                OJson long_freq = OJson::array();
                for (const float f : r.long_inv_freq) { long_freq.push_back(static_cast<double>(f)); }
                g["long_inv_freq"] = std::move(long_freq);
                g["long_after"]    = r.long_after;
            }
            groups[kind]  = std::move(g);
            if (const auto* beta = member(p, "llama_4_scaling_beta"); truthy(beta)) {
                const auto period = positive(member(p, "original_max_position_embeddings"),
                                             "text.rope.original_max_position_embeddings");
                temperature.emplace(beta->get<double>(), period);
            }
        }
        layer_rope.push_back(kind);
    }
    c["layer_rope"]  = std::move(layer_rope);
    c["rope_groups"] = std::move(groups);
    if (temperature) {
        c["query_scale"] = OJson{{"beta", temperature->first}, {"period", temperature->second}};
    }

    const OJson* tied = member(raw, "tie_word_embeddings");
    if (tied == nullptr || tied->is_null()) { tied = member(source, "tie_word_embeddings"); }
    // Left unset (absent, or null as in RedHatAI's SmolLM3-3B-FP8-dynamic re-save), the flag is
    // the checkpoint's to answer, as for GGUF: no output head tensor means the embedding is the
    // head. A declared value is taken as declared.
    if (tied != nullptr && !tied->is_null() && !tied->is_boolean()) {
        refuse("tie_word_embeddings must be boolean");
    }
    c["tie_word_embeddings"] =
        tied != nullptr && tied->is_boolean() ? tied->get<bool>() : !checkpoint_has_head;
    out.config               = std::move(c);
    return out;
}

// ---------------------------------------------------------------------------------------------
// Parameters

class Builder {
public:
    Builder(Model& model, const Checkpoint& base) : model_(model), base_(base) {}

    void matrix(const std::string& name, const std::string& tensor, Shape shape,
                std::vector<std::string> inputs, std::string probe) {
        auto factory = [tensor, shape](const Checkpoint& base, MatrixFormat format) -> SourcePtr {
            return matrix_source(base, tensor, shape, format);
        };
        add(name, shape, factory(base_, MatrixFormat::Auto), factory, std::move(inputs),
            QType::BF16, std::move(probe));
    }

    // Rows [first, last) of the fused checkpoint matrix `tensor` of shape `fused` (Phi-3's
    // qkv_proj and gate_up_proj): the split is a row selection of the matrix as published (its
    // encoded form too), probed for encoded sites through the fused matrix.
    void rows(const std::string& name, const std::string& tensor, Shape fused, std::uint64_t first,
              std::uint64_t last, std::vector<std::string> inputs, std::string probe) {
        auto factory = [tensor, fused, first, last](const Checkpoint& base,
                                                    MatrixFormat format) -> SourcePtr {
            return select_rows(matrix_source(base, tensor, fused, format), {{first, last}});
        };
        add(name, {last - first, fused[1]}, factory(base_, MatrixFormat::Auto), factory,
            std::move(inputs), QType::BF16, std::move(probe));
        model_.parameters.back().probe_shape = fused;
    }

    // A vector that is an element range of a fused checkpoint vector (a fused bias).
    void vector_range(const std::string& name, const std::string& tensor, std::uint64_t first,
                      std::uint64_t last, std::uint64_t total) {
        auto factory = [name](const Checkpoint&, MatrixFormat) -> SourcePtr {
            throw std::runtime_error(name + ": provide an explicit encoded source mapping");
        };
        add(name, {last - first}, tensor_source(base_, tensor, {last - first}, first, {total}),
            factory, {}, QType::BF16, {});
    }

    // A tied output head: it reads the embedding table `table` itself, so its default source is
    // the table's (one source: a recipe that encodes both alike binds them to one object, see
    // share_tied_head); the factory still maps `tensor` for probed encoded sites.
    void tied(const std::string& name, const std::string& table, const std::string& tensor,
              Shape shape, std::vector<std::string> inputs, std::string probe) {
        auto factory = [tensor, shape](const Checkpoint& base, MatrixFormat format) -> SourcePtr {
            return matrix_source(base, tensor, shape, format);
        };
        add(name, shape, model_.parameters[model_.at(table)].source, factory, std::move(inputs),
            QType::BF16, std::move(probe));
    }

    // A matrix or table scaled by a constant at conversion (a folded muP multiplier).
    void scaled(const std::string& name, const std::string& tensor, Shape shape, float factor,
                std::vector<std::string> inputs) {
        auto factory = [name](const Checkpoint&, MatrixFormat) -> SourcePtr {
            throw std::runtime_error(name + ": a scaled source keeps no encoded form");
        };
        add(name, shape, scaled_source(tensor_source(base_, tensor, shape), factor), factory,
            std::move(inputs), QType::BF16, {});
    }

    void vector(const std::string& name, const std::string& tensor, Shape shape) {
        auto factory = [name](const Checkpoint&, MatrixFormat) -> SourcePtr {
            throw std::runtime_error(name + ": provide an explicit encoded source mapping");
        };
        add(name, shape, tensor_source(base_, tensor, shape), factory, {}, QType::BF16, {});
    }

    // A parameter with an explicit source (and encoded-source factory): an expert's rows of a
    // stacked checkpoint tensor.
    void custom(const std::string& name, Shape shape, SourcePtr source,
                std::function<SourcePtr(const Checkpoint&, MatrixFormat)> factory,
                std::vector<std::string> inputs, QType direct = QType::BF16) {
        add(name, std::move(shape), std::move(source), std::move(factory), std::move(inputs),
            direct, {});
    }

    // A vector stored in FP32 (attention sinks, which the attention Op reads as FP32).
    void vector_f32(const std::string& name, const std::string& tensor, Shape shape) {
        auto factory = [name](const Checkpoint&, MatrixFormat) -> SourcePtr {
            throw std::runtime_error(name + ": provide an explicit encoded source mapping");
        };
        add(name, shape, tensor_source(base_, tensor, shape), factory, {}, QType::FP32, {});
    }

    void group(const std::vector<std::string>& names) { model_.add_packing_group(names); }

private:
    void add(const std::string& name, Shape shape, SourcePtr source,
             std::function<SourcePtr(const Checkpoint&, MatrixFormat)> factory,
             std::vector<std::string> inputs, QType direct, std::string probe) {
        Parameter parameter;
        parameter.name           = name;
        parameter.shape          = std::move(shape);
        parameter.source         = std::move(source);
        parameter.encoded_source = std::move(factory);
        parameter.inputs         = std::move(inputs);
        parameter.direct         = direct;
        parameter.residency      = "text";
        parameter.probe_leaf     = std::move(probe);
        model_.add(std::move(parameter));
    }

    Model& model_;
    const Checkpoint& base_;
};

std::string find_root(const Checkpoint& base) {
    for (const char* root : {"model.language_model.", "language_model.model.", "model."}) {
        if (base.has(std::string(root) + "embed_tokens.weight")) { return root; }
    }
    throw std::runtime_error("dense decoder: no embed_tokens.weight under a known prefix");
}

std::optional<std::string> head_tensor(const Checkpoint& base) {
    for (const char* name : {"lm_head.weight", "language_model.lm_head.weight"}) {
        if (base.has(name) || base.has(std::string(name).substr(0, std::string(name).size() - 7) +
                                       ".weight_packed")) {
            return name;
        }
    }
    return std::nullopt;
}

// A projection matrix at `prefix` as published: plain (.weight) or packed (.weight_packed).
bool has_matrix(const Checkpoint& base, const std::string& prefix) {
    return base.has(prefix + ".weight") || base.has(prefix + ".weight_packed");
}

std::string find_head(const Checkpoint& base) {
    if (auto name = head_tensor(base)) { return *name; }
    throw std::runtime_error("dense decoder: untied model without lm_head.weight");
}


// NemotronH parameters. Checkpoint names under backbone.: embedding, layers.<i>.norm,
// layers.<i>.mixer.{in_proj, conv1d, dt_bias, A_log, D, norm, out_proj} (Mamba2),
// {q,k,v,o}_proj (attention), gate.{weight, e_score_correction_bias}, experts.<e>.{up,down}_proj,
// shared_experts.{up,down}_proj (MoE), norm_f; lm_head.
Model build_nemotron_h(Model model, const Checkpoint& base) {
    const auto& c = model.config();
    Builder builder(model, base);
    std::string root;
    for (const char* r : {"backbone.", "model."}) {
        if (base.has(std::string(r) + "embeddings.weight") ||
            base.has(std::string(r) + "embedding.weight")) {
            root = r;
            break;
        }
    }
    if (root.empty()) { throw std::runtime_error("nemotron_h: no backbone.embedding.weight"); }
    const auto embed = base.has(root + "embeddings.weight") ? root + "embeddings.weight"
                                                            : root + "embedding.weight";
    const auto h     = c["hidden_size"].get<std::uint64_t>();
    const auto v     = c["vocab_size"].get<std::uint64_t>();
    builder.matrix("text/token_embedding", embed, {v, h}, {}, {});
    if (c["tie_word_embeddings"].get<bool>()) {
        builder.tied("text/output_head", "text/token_embedding", embed, {v, h},
                     {"text/final_hidden"}, embed.substr(0, embed.size() - 7));
    } else {
        const auto head = find_head(base);
        builder.matrix("text/output_head", head, {v, h}, {"text/final_hidden"},
                       head.substr(0, head.size() - 7));
    }
    builder.vector("text/final_norm", root + "norm_f.weight", {h});
    const auto& mb    = c["mamba"];
    const auto heads  = mb["num_heads"].get<std::uint64_t>();
    const auto inner  = heads * mb["head_dim"].get<std::uint64_t>();
    const auto convc  = inner + 2 * mb["n_groups"].get<std::uint64_t>() *
                                   mb["state_size"].get<std::uint64_t>();
    const auto d      = c["head_dim"].get<std::uint64_t>();
    const auto q      = c["num_attention_heads"].get<std::uint64_t>() * d;
    const auto kv     = c["num_key_value_heads"].get<std::uint64_t>() * d;
    const auto& m     = c["moe"];
    const auto e      = m["num_experts"].get<std::uint64_t>();
    const auto f      = m["moe_intermediate_size"].get<std::uint64_t>();
    const auto fs     = m["shared_intermediate_size"].get<std::uint64_t>();
    const auto up_k        = m.value("expert_input_size", h);
    const auto layers = c["num_hidden_layers"].get<std::uint64_t>();
    const auto no_encoding = [](const std::string& name) {
        return [name](const Checkpoint&, MatrixFormat) -> SourcePtr {
            throw std::runtime_error(name + ": no encoded form");
        };
    };
    for (std::uint64_t i = 0; i < layers; ++i) {
        const auto p    = "text/layers/" + std::to_string(i) + "/";
        const auto s    = root + "layers." + std::to_string(i) + ".";
        const auto x    = s + "mixer.";
        const auto kind = c["layer_types"][i].get<std::string>();
        const auto in   = p + "mixer_input";
        builder.vector(p + "norm", s + "norm.weight", {h});
        if (kind == "mamba") {
            builder.matrix(p + "mamba/in_proj", x + "in_proj.weight", {inner + convc + heads, h},
                           {in}, x + "in_proj");
            builder.custom(p + "mamba/conv_weight", {convc, 4},
                           tensor_source(base, x + "conv1d.weight", {convc, 4}, 0, {convc, 1, 4}),
                           no_encoding(p + "mamba/conv_weight"), {});
            if (mb["conv_bias"].get<bool>()) {
                builder.vector(p + "mamba/conv_bias", x + "conv1d.bias", {convc});
            }
            builder.vector_f32(p + "mamba/dt_bias", x + "dt_bias", {heads});
            builder.vector_f32(p + "mamba/a_log", x + "A_log", {heads});
            builder.vector_f32(p + "mamba/d", x + "D", {heads});
            builder.vector(p + "mamba/norm", x + "norm.weight", {inner});
            builder.matrix(p + "mamba/out_proj", x + "out_proj.weight", {h, inner},
                           {p + "mamba/gated"}, x + "out_proj");
        } else if (kind == "attention") {
            std::vector<std::string> packed;
            for (const auto& [role, field, rows] :
                 {std::tuple{"query", "q_proj", q}, std::tuple{"key", "k_proj", kv},
                  std::tuple{"value", "v_proj", kv}}) {
                builder.matrix(p + "attention/" + role, x + field + ".weight", {rows, h}, {in},
                               x + field);
                packed.push_back(p + "attention/" + role);
            }
            builder.group(packed);
            builder.matrix(p + "attention/output", x + "o_proj.weight", {h, q},
                           {p + "attention/context"}, x + "o_proj");
        } else {
            builder.matrix(p + "mlp/router", x + "gate.weight", {e, h}, {in}, x + "gate");
            builder.vector_f32(p + "mlp/selection_bias", x + "gate.e_score_correction_bias", {e});
            std::vector<std::string> up, down;
            for (std::uint64_t k = 0; k < e; ++k) {
                const auto ep = p + "mlp/experts/" + std::to_string(k) + "/";
                const auto sp = x + "experts." + std::to_string(k) + ".";
                builder.matrix(ep + "up", sp + "up_proj.weight", {f, up_k}, {in}, sp + "up_proj");
                builder.matrix(ep + "down", sp + "down_proj.weight", {h, f},
                               {p + "mlp/expert_product"}, sp + "down_proj");
                up.push_back(ep + "up");
                down.push_back(ep + "down");
            }
            builder.group(up);
            model.expert_banks.push_back(model.packing_groups.size() - 1);
            builder.group(down);
            model.expert_banks.push_back(model.packing_groups.size() - 1);
            builder.matrix(p + "mlp/shared_up", x + "shared_experts.up_proj.weight", {fs, h}, {in},
                           x + "shared_experts.up_proj");
            builder.matrix(p + "mlp/shared_down", x + "shared_experts.down_proj.weight", {h, fs},
                           {p + "mlp/shared_product"}, x + "shared_experts.down_proj");
        }
    }
    return model;
}

// ---------------------------------------------------------------------------------------------
// The NemotronH MTP (NextN) drafter (a nemotron-h-mtp GGUF sidecar, model_type "nemotron_h_mtp"):
// one block of a NoPE attention sub-layer and a MoE sub-layer over eh_proj of
// [rmsnorm(embedding); rmsnorm(previous hidden)], the pre-MoE RMSNorm and the final LayerNorm
// weight. The embedding and the LM head are the target's; the drafter artifact carries no copy.

Resolved resolve_nemotron_h_mtp(const OJson& raw, const std::string& architecture) {
    const auto count = [&](const char* key) {
        return positive(member(raw, key), std::string("text.") + key);
    };
    OJson c                      = OJson::object();
    c["architectures"]           = OJson::array({"NemotronHMtpForCausalLM"});
    c["model_type"]              = "nemotron_h_mtp";
    c["source_architecture"]     = architecture;
    c["hidden_size"]             = count("hidden_size");
    c["vocab_size"]              = count("vocab_size");
    c["num_hidden_layers"]       = 1;
    c["max_position_embeddings"] = count("max_position_embeddings");
    c["rms_norm_eps"]   = f32_value(member(raw, "rms_norm_eps"), 1e-5, "text.rms_norm_eps");
    c["layer_norm_eps"] =
        f32_value(member(raw, "layer_norm_epsilon"), 1e-5, "text.layer_norm_epsilon");
    c["num_attention_heads"] = count("num_attention_heads");
    c["num_key_value_heads"] = count("num_key_value_heads");
    c["head_dim"]            = count("head_dim");
    c["attention_scale"]     = static_cast<double>(
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(count("head_dim")))));
    c["moe"] = resolve_nemotron_h_moe(raw);
    Resolved out;
    out.config = std::move(c);
    return out;
}

// A llama.cpp MTP sidecar ships no tokenizer files of its own: the drafter artifact takes the
// target's, from the sidecar's directory or given with --resource.
void require_sidecar_resources(const Checkpoint& base, const ResourceOverrides& overrides,
                               const ResourceFallbacks& fallbacks) {
    std::string missing;
    for (const std::string role : {"tokenizer.json", "tokenizer_config.json",
                                   "chat_template.jinja", "generation_config.json"}) {
        const bool given = std::any_of(overrides.begin(), overrides.end(),
                                       [&](const auto& pair) { return pair.first == role; });
        if (!given && !fallbacks.contains(role) && !std::filesystem::exists(base.root() / role)) {
            missing += (missing.empty() ? "" : ", ") + role;
        }
    }
    if (!missing.empty()) {
        throw std::runtime_error(
            "the NemotronH MTP sidecar has no " + missing + " beside it in " +
            base.root().string() + ": it ships none, the drafter takes the target's; put the "
            "target's files beside it or give each with --resource ROLE=PATH (e.g. --resource "
            "tokenizer.json=TARGET_DIR/tokenizer.json)");
    }
}

// NemotronH MTP drafter parameters. Checkpoint names: mtp.layers.0.{enorm,hnorm} and
// mtp.layers.0.mixer.eh_proj (the fusion), mtp.layers.0.norm with the attention mixer
// {q,k,v,o}_proj, mtp.layers.1.norm with the MoE mixer (gate, experts, shared experts) and
// mtp.layers.1.final_layernorm. The eh_proj input rows are [norm(e); norm(h)] in that order, as
// stored. The attention reads mtp/attention_input (rmsnorm(u, attn_norm)), the MoE mtp/ffn_input
// (rmsnorm(u, post_attention_norm)).
Model build_nemotron_h_mtp(Model model, const Checkpoint& base) {
    const auto& c = model.config();
    Builder builder(model, base);
    const auto h    = c["hidden_size"].get<std::uint64_t>();
    const auto d    = c["head_dim"].get<std::uint64_t>();
    const auto q    = c["num_attention_heads"].get<std::uint64_t>() * d;
    const auto kv   = c["num_key_value_heads"].get<std::uint64_t>() * d;
    const auto& m   = c["moe"];
    const auto e    = m["num_experts"].get<std::uint64_t>();
    const auto f    = m["moe_intermediate_size"].get<std::uint64_t>();
    const auto fs   = m["shared_intermediate_size"].get<std::uint64_t>();
    const auto up_k = m.value("expert_input_size", h);
    const auto x0   = std::string("mtp.layers.0.mixer.");
    const auto x1   = std::string("mtp.layers.1.mixer.");
    const auto attn = std::string("mtp/attention_input");
    const auto ffn  = std::string("mtp/ffn_input");
    builder.matrix("mtp/eh_proj", x0 + "eh_proj.weight", {h, 2 * h}, {"mtp/fusion_input"},
                   x0 + "eh_proj");
    builder.vector("mtp/embedding_norm", "mtp.layers.0.enorm.weight", {h});
    builder.vector("mtp/hidden_norm", "mtp.layers.0.hnorm.weight", {h});
    builder.vector("mtp/attention_norm", "mtp.layers.0.norm.weight", {h});
    std::vector<std::string> packed;
    for (const auto& [role, field, rows] :
         {std::tuple{std::string("query"), std::string("q_proj"), q},
          std::tuple{std::string("key"), std::string("k_proj"), kv},
          std::tuple{std::string("value"), std::string("v_proj"), kv}}) {
        builder.matrix("mtp/attention/" + role, x0 + field + ".weight", {rows, h}, {attn},
                       x0 + field);
        packed.push_back("mtp/attention/" + role);
    }
    builder.group(packed);
    builder.matrix("mtp/attention/output", x0 + "o_proj.weight", {h, q}, {"mtp/attention/context"},
                   x0 + "o_proj");
    builder.vector("mtp/post_attention_norm", "mtp.layers.1.norm.weight", {h});
    builder.matrix("mtp/mlp/router", x1 + "gate.weight", {e, h}, {ffn}, x1 + "gate");
    builder.vector_f32("mtp/mlp/selection_bias", x1 + "gate.e_score_correction_bias", {e});
    std::vector<std::string> up, down;
    for (std::uint64_t k = 0; k < e; ++k) {
        const auto ep = "mtp/mlp/experts/" + std::to_string(k) + "/";
        const auto sp = x1 + "experts." + std::to_string(k) + ".";
        builder.matrix(ep + "up", sp + "up_proj.weight", {f, up_k}, {ffn}, sp + "up_proj");
        builder.matrix(ep + "down", sp + "down_proj.weight", {h, f}, {"mtp/mlp/expert_product"},
                       sp + "down_proj");
        up.push_back(ep + "up");
        down.push_back(ep + "down");
    }
    builder.group(up);
    model.expert_banks.push_back(model.packing_groups.size() - 1);
    builder.group(down);
    model.expert_banks.push_back(model.packing_groups.size() - 1);
    builder.matrix("mtp/mlp/shared_up", x1 + "shared_experts.up_proj.weight", {fs, h}, {ffn},
                   x1 + "shared_experts.up_proj");
    builder.matrix("mtp/mlp/shared_down", x1 + "shared_experts.down_proj.weight", {h, fs},
                   {"mtp/mlp/shared_product"}, x1 + "shared_experts.down_proj");
    builder.vector("mtp/final_norm", "mtp.layers.1.final_layernorm.weight", {h});
    return model;
}

// ---------------------------------------------------------------------------------------------
// Recipes

constexpr QType kQ8 = QType::Q8_G32_FP16;

// --vocab q8: the token embedding and the output head take q8_g32_fp16 where the checkpoint left
// them unquantized. A vocabulary the publisher quantized (a GGUF's Q8_0/K-quant table) is kept as
// published either way: quantizing it again would be a second quantization. Returns how many of
// the two were quantized.
std::size_t assign_vocabulary(Model& model, Recipe& recipe) {
    std::size_t assigned = 0;
    for (const char* name : {"text/token_embedding", "text/output_head"}) {
        const auto i = model.at(name);
        if (model.parameters[i].source && model.parameters[i].source->dequantizes()) { continue; }
        recipe.assign(i, kQ8, Method::GroupedAbsmax);
        ++assigned;
    }
    return assigned;
}

// An output head the checkpoint publishes encoded keeps its words under every recipe and --vocab
// (as a quantized table does above): ModelOpt quantizes lm_head too (nvidia/Muse-Glimmer-30B-NVFP4:
// W4A16 NVFP4, 202048 rows, whose last 128-row scale tile the layout completes). An NVFP4 or FP8
// head keeps BF16 activations whatever the site calibrated (A16Only), so decode and scoring of any
// token count compute it on one route; an integer head (GPTQ/AWQ u4) keeps its AllowA8.
void keep_encoded_head(Model& model, Recipe& recipe, const Checkpoint& base) {
    const auto i  = model.at("text/output_head");
    const auto& p = model.parameters[i];
    if (p.probe_leaf.empty() || !p.encoded_source) { return; }
    const auto site = kept_encoding(base, p.probe_leaf, p.shape[0], p.shape[1]);
    if (!site) { return; }
    recipe.assign(i, site->format, Method::ImportEncoded,
                  p.encoded_source(base, encoded_matrix_format(site->format)));
    recipe.set_policy(i, head_policy(*site));
}

void dense_q8(Model& model, Recipe& recipe) {
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        const auto& p = model.parameters[i];
        if (p.name.starts_with("text/layers/") && p.projection()) {
            recipe.assign(i, kQ8, Method::GroupedAbsmax);
        }
    }
}

// Every projection in row-scaled FP8 (max|row|/448 as a BF16 multiplier) with FP8 activations
// allowed: the tensor-core prefill of an 8-bit weight at about Q8's quality.
void dense_fp8(Model& model, Recipe& recipe) {
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        const auto& p = model.parameters[i];
        if (p.name.starts_with("text/layers/") && p.projection()) {
            recipe.assign(i, QType::FP8_E4M3FN_ROW_BF16, Method::Fp8RowMaxabs);
            recipe.set_policy(i, ActivationPolicy::AllowA8);
        }
    }
}

// The checkpoint's own NVFP4 (with its calibrated activation divisor: W4A4 allowed) or row/tensor
// FP8 (A8 allowed) projections are kept. Every other projection takes Q8 (dense_encoded) or is
// rounded to NVFP4 here (dense_nvfp4: weight-only, BF16 activations; a projection whose rows are
// not whole 128-row scale tiles takes Q8). A kept NVFP4 matrix of any row count keeps its words:
// the layout completes a last scale tile its rows do not fill. Q8 of a projection the publisher
// quantized in an encoding kept nowhere is a second quantization, which convert() refuses.
// dense_published quantizes nothing: every projection the checkpoint left unquantized stays BF16.
// The vocabulary follows --vocab (apply_dense_recipe): native unless asked; an encoded head is
// kept either way (keep_encoded_head).
enum class Plain { Q8, RoundNvfp4, Published };

// The shape of the checkpoint matrix a parameter is probed through (the fused matrix of a split).
const Shape& probed(const Parameter& p) { return p.probe_shape.empty() ? p.shape : p.probe_shape; }

void dense_kept(Model& model, Recipe& recipe, const Checkpoint& base, Plain plain) {
    const bool round_plain = plain == Plain::RoundNvfp4;
    // Matrices one fused parent will hold (q/k/v, gate/up) are rounded against one divisor.
    std::map<std::size_t, std::vector<SourcePtr>> siblings;
    for (const auto& group : model.packing_groups) {
        std::vector<SourcePtr> sources;
        for (const auto i : group) { sources.push_back(model.parameters[i].source); }
        for (const auto i : group) { siblings[i] = sources; }
    }
    const auto site_of = [&](std::size_t i) -> std::optional<KeptEncoding> {
        const auto& p = model.parameters[i];
        return p.probe_leaf.empty() ? std::nullopt
                                    : kept_encoding(base, p.probe_leaf, probed(p)[0], probed(p)[1]);
    };
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        const auto& p = model.parameters[i];
        if (!p.name.starts_with("text/layers/") || !p.projection()) { continue; }
        const auto site = site_of(i);
        if (!site) {
            if (plain == Plain::Published) { continue; } // BF16 as published
            if (round_plain && p.shape[0] % 128 == 0 && p.shape[1] % 64 == 0) {
                recipe.assign(i, QType::NVFP4, Method::ImportEncoded, nvfp4_rounded_source(p.source, siblings[i]));
                recipe.set_policy(i, ActivationPolicy::A16Only);
            } else {
                recipe.assign(i, kQ8, Method::GroupedAbsmax);
            }
            continue;
        }
        recipe.assign(i, site->format, Method::ImportEncoded,
                      p.encoded_source(base, encoded_matrix_format(site->format)));
        recipe.set_policy(i, kept_policy(*site));
    }
}

// A MoE layer's router and expert banks. Every expert's matrices are parameters of their own
// (the checkpoint may quantize each one against its own divisor), grouped into one packing group
// per bank in expert order, which the loader binds as one stacked plane.
//   gpt-oss   one stacked MXFP4 tensor per bank (<name>_blocks / _scales, leading dims [E, rows]):
//             expert e is rows [e*R, (e+1)*R) of it; gate/up rows stay interleaved; the biases
//             are one [E*R] vector per bank.
//   qwen3_moe experts.<e>.{gate,up,down}_proj, each a matrix of its own (BF16, NVFP4, FP8).
void add_experts(Builder& builder, Model& model, const Checkpoint& base, const OJson& c,
                 const std::string& p, const std::string& s) {
    const auto& m   = c["moe"];
    const auto h    = c["hidden_size"].get<std::uint64_t>();
    const auto e    = m["num_experts"].get<std::uint64_t>();
    const auto f    = m["moe_intermediate_size"].get<std::uint64_t>();
    const auto type = c["source_model_type"].get<std::string>();
    const std::string in = p + "ffn_input";
    std::vector<std::string> gate_up, down;
    if (type == "gpt_oss") {
        builder.matrix(p + "mlp/router", s + "mlp.router.weight", {e, h}, {in}, s + "mlp.router");
        builder.vector(p + "mlp/router_bias", s + "mlp.router.bias", {e});
        // One source per (checkpoint, stacked tensor, format), shared by every expert's rows: an
        // MXFP4 source reads its whole exponent plane once.
        using Key  = std::tuple<const Checkpoint*, std::string, int>;
        auto cache = std::make_shared<std::map<Key, SourcePtr>>();
        auto guard = std::make_shared<std::mutex>();
        const auto rows_of = [cache, guard](const std::string& tensor, Shape stacked,
                                            std::uint64_t first, std::uint64_t last) {
            return [cache, guard, tensor, stacked, first,
                    last](const Checkpoint& checkpoint, MatrixFormat format) -> SourcePtr {
                SourcePtr matrix;
                {
                    std::lock_guard lock(*guard);
                    const Key key{&checkpoint, tensor, static_cast<int>(format)};
                    auto it = cache->find(key);
                    if (it == cache->end()) {
                        it = cache->emplace(key, matrix_source(checkpoint, tensor, stacked, format))
                                 .first;
                    }
                    matrix = it->second;
                }
                return select_rows(matrix, {{first, last}});
            };
        };
        for (std::uint64_t x = 0; x < e; ++x) {
            const auto ep = p + "mlp/experts/" + std::to_string(x) + "/";
            const auto gu = rows_of(s + "mlp.experts.gate_up_proj", {e * 2 * f, h}, x * 2 * f,
                                    (x + 1) * 2 * f);
            const auto dn = rows_of(s + "mlp.experts.down_proj", {e * h, f}, x * h, (x + 1) * h);
            builder.custom(ep + "gate_up", {2 * f, h}, gu(base, MatrixFormat::Auto), gu, {in});
            builder.custom(ep + "down", {h, f}, dn(base, MatrixFormat::Auto), dn,
                           {p + "mlp/expert_product"});
            gate_up.push_back(ep + "gate_up");
            down.push_back(ep + "down");
        }
        builder.custom(p + "mlp/experts/gate_up_bias", {e * 2 * f},
                       tensor_source(base, s + "mlp.experts.gate_up_proj_bias", {e * 2 * f}, 0,
                                     {e, 2 * f}),
                       [](const Checkpoint&, MatrixFormat) -> SourcePtr {
                           throw std::runtime_error("expert bias: no encoded form");
                       },
                       {});
        builder.custom(p + "mlp/experts/down_bias", {e * h},
                       tensor_source(base, s + "mlp.experts.down_proj_bias", {e * h}, 0, {e, h}),
                       [](const Checkpoint&, MatrixFormat) -> SourcePtr {
                           throw std::runtime_error("expert bias: no encoded form");
                       },
                       {});
    } else {
        builder.matrix(p + "mlp/router", s + "mlp.gate.weight", {e, h}, {in}, s + "mlp.gate");
        for (std::uint64_t x = 0; x < e; ++x) {
            const auto ep = p + "mlp/experts/" + std::to_string(x) + "/";
            const auto sp = s + "mlp.experts." + std::to_string(x) + ".";
            builder.matrix(ep + "gate", sp + "gate_proj.weight", {f, h}, {in}, sp + "gate_proj");
            builder.matrix(ep + "up", sp + "up_proj.weight", {f, h}, {in}, sp + "up_proj");
            builder.matrix(ep + "down", sp + "down_proj.weight", {h, f}, {p + "mlp/expert_product"},
                           sp + "down_proj");
            gate_up.push_back(ep + "gate");
            gate_up.push_back(ep + "up");
            down.push_back(ep + "down");
        }
    }
    builder.group(gate_up);
    model.expert_banks.push_back(model.packing_groups.size() - 1);
    builder.group(down);
    model.expert_banks.push_back(model.packing_groups.size() - 1);
}

// MoE recipes. The experts keep the checkpoint's own encoding (MXFP4 as mxfp4, NVFP4 with its
// divisors, row-scaled FP8; a bank whose experts end inside a scale tile shares one divisor or is
// refused: refuse_split_nvfp4_tiles), BF16 experts stay BF16; the Op computes them with BF16
// activations.
// The router stays BF16. moe_published keeps every other tensor as published (BF16 unless the
// checkpoint encoded it); moe_q8 takes Q8 for the attention projections where the checkpoint left
// them in BF16. The vocabulary follows --vocab (apply_dense_recipe).
bool is_expert(const Parameter& p) { return p.name.find("/mlp/experts/") != std::string::npos; }

void moe_recipe(Model& model, Recipe& recipe, const Checkpoint& base, bool q8) {
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        const auto& p = model.parameters[i];
        if (!p.projection() || !p.name.starts_with("text/layers/")) { continue; }
        if (p.name.ends_with("/mlp/router")) { continue; } // BF16, as published
        if (is_expert(p)) {
            const auto encoded = p.encoded_source(base, MatrixFormat::Mxfp4);
            std::optional<QType> kept;
            try {
                kept = encoded->encoded_format();
            } catch (const std::exception&) {}
            if (kept == QType::MXFP4) {
                recipe.assign(i, QType::MXFP4, Method::ImportEncoded, encoded);
                recipe.set_policy(i, kept_block_policy(QType::MXFP4));
                continue;
            }
            const auto site = p.probe_leaf.empty()
                                  ? std::nullopt
                                  : kept_encoding(base, p.probe_leaf, probed(p)[0], probed(p)[1]);
            if (site) {
                recipe.assign(i, site->format, Method::ImportEncoded,
                              p.encoded_source(base, encoded_matrix_format(site->format)));
            }
            // BF16 experts keep the default (cast_direct BF16).
            continue;
        }
        // Attention projections: the checkpoint's NVFP4/FP8 kept (as dense_encoded does), else
        // BF16 (published) or Q8.
        const auto site = p.probe_leaf.empty()
                              ? std::nullopt
                              : kept_encoding(base, p.probe_leaf, probed(p)[0], probed(p)[1]);
        if (site) {
            recipe.assign(i, site->format, Method::ImportEncoded,
                          p.encoded_source(base, encoded_matrix_format(site->format)));
            recipe.set_policy(i, kept_policy(*site));
        } else if (q8) {
            recipe.assign(i, kQ8, Method::GroupedAbsmax);
        }
    }
    // One stacked plane per bank, even where the automatic packing would keep NVFP4 parents with
    // different divisors apart: only the MoE Op reads a stack, and it reads its divisors.
    for (const auto bank : model.expert_banks) {
        refuse_split_nvfp4_tiles(model, recipe, model.packing_groups[bank]);
        recipe.explicit_groups.push_back(model.packing_groups[bank]);
    }
}

} // namespace

std::vector<std::string_view> dense_architectures() {
    std::vector<std::string_view> out;
    for (const auto& source : kSources) { out.emplace_back(source.architecture); }
    return out;
}

bool is_dense_architecture(std::string_view architecture) noexcept {
    return std::any_of(std::begin(kSources), std::end(kSources),
                       [&](const KnownSource& s) { return architecture == s.architecture; });
}

OJson dense_text_config(const OJson& source, bool checkpoint_has_head) {
    return resolve(source, checkpoint_has_head).config;
}

Model build_dense_model(const Checkpoint& base, const ResourceOverrides& overrides) {
    auto resolved = resolve(base.config(), head_tensor(base).has_value());
    Model model;
    model.components["text"]["config"] = resolved.config;
    const auto type      = resolved.config["model_type"].get<std::string>();
    const auto fallbacks = resource_fallbacks(base);
    if (type == "nemotron_h_mtp") { require_sidecar_resources(base, overrides, fallbacks); }
    load_resources(model, base.root(), resolved.config["vocab_size"].get<std::uint64_t>(),
                   overrides, nullptr, fallbacks);
    if (type == "nemotron_h") { return build_nemotron_h(std::move(model), base); }
    if (type == "nemotron_h_mtp") { return build_nemotron_h_mtp(std::move(model), base); }
    const auto& c = model.config(); // after load_resources: adding resources moves the config
    Builder builder(model, base);
    const auto root  = find_root(base);
    const auto h     = c["hidden_size"].get<std::uint64_t>();
    const auto v     = c["vocab_size"].get<std::uint64_t>();
    const auto d     = c["head_dim"].get<std::uint64_t>();
    const auto q     = c["num_attention_heads"].get<std::uint64_t>() * d;
    const auto kv    = c["num_key_value_heads"].get<std::uint64_t>() * d;
    const auto inter = c["intermediate_size"].get<std::uint64_t>();
    const auto embed = root + "embed_tokens.weight";
    if (resolved.embedding_multiplier != 1.0) {
        builder.scaled("text/token_embedding", embed, {v, h},
                       static_cast<float>(resolved.embedding_multiplier), {});
    } else {
        // Read as a matrix, so a published encoding of the table (GGUF Q8_0, K-quants) is kept.
        builder.matrix("text/token_embedding", embed, {v, h}, {}, {});
    }
    const bool tied = c["tie_word_embeddings"].get<bool>();
    const auto head = tied ? embed : find_head(base);
    if (resolved.logits_scaling != 1.0) {
        builder.scaled("text/output_head", head, {v, h},
                       static_cast<float>(1.0 / resolved.logits_scaling), {"text/final_hidden"});
    } else if (tied && resolved.embedding_multiplier == 1.0) {
        builder.tied("text/output_head", "text/token_embedding", head, {v, h},
                     {"text/final_hidden"}, head.substr(0, head.size() - 7));
    } else {
        builder.matrix("text/output_head", head, {v, h}, {"text/final_hidden"},
                       head.substr(0, head.size() - 7));
    }
    builder.vector("text/final_norm", root + "norm.weight", {h});
    const bool post       = c["norm_placement"] == "post";
    const bool sandwich   = c["norm_placement"] == "sandwich";
    const auto qk_norm    = c["qk_norm"].get<std::string>();
    const bool qkv_bias   = c["attention_bias"].get<bool>();
    const bool out_bias   = c["attention_output_bias"].get<bool>();
    const auto layers     = c["num_hidden_layers"].get<std::uint64_t>();
    for (std::uint64_t i = 0; i < layers; ++i) {
        const auto p = "text/layers/" + std::to_string(i) + "/";
        const auto s = root + "layers." + std::to_string(i) + ".";
        if (sandwich) {
            // Gemma 3: input_layernorm, post_attention_layernorm, pre_ and post_feedforward_layernorm.
            builder.vector(p + "attention_norm", s + "input_layernorm.weight", {h});
            builder.vector(p + "attention_post_norm", s + "post_attention_layernorm.weight", {h});
            builder.vector(p + "ffn_norm", s + "pre_feedforward_layernorm.weight", {h});
            builder.vector(p + "ffn_post_norm", s + "post_feedforward_layernorm.weight", {h});
        } else if (post) {
            builder.vector(p + "attention_post_norm", s + "post_attention_layernorm.weight", {h});
            builder.vector(p + "ffn_post_norm", s + "post_feedforward_layernorm.weight", {h});
        } else {
            builder.vector(p + "attention_norm", s + "input_layernorm.weight", {h});
            builder.vector(p + "ffn_norm", s + "post_attention_layernorm.weight", {h});
        }
        std::vector<std::string> packed;
        // q/k/v as three matrices, or one fused qkv_proj (rows q, then k, then v) split by rows.
        const auto fused_qkv = s + "self_attn.qkv_proj";
        const bool fused     = has_matrix(base, fused_qkv);
        std::uint64_t first  = 0;
        for (const auto& [role, field, rows] :
             {std::tuple{"query", "q_proj", q}, std::tuple{"key", "k_proj", kv},
              std::tuple{"value", "v_proj", kv}}) {
            if (std::string(role) == "value" && c.value("attention_output_gate", false)) {
                // The output gate reads the attention input too: packed after k ([q; k; gate; v]),
                // so a GGUF layer whose v takes a wider type still packs [q; k; gate].
                builder.matrix(p + "attention/gate", s + "self_attn.gate_proj.weight", {q, h},
                               {p + "attention_input"}, s + "self_attn.gate_proj");
                packed.push_back(p + "attention/gate");
            }
            if (fused) {
                builder.rows(p + "attention/" + role, fused_qkv + ".weight", {q + 2 * kv, h}, first,
                             first + rows, {p + "attention_input"}, fused_qkv);
                if (qkv_bias) {
                    builder.vector_range(p + "attention/" + role + "_bias", fused_qkv + ".bias",
                                         first, first + rows, q + 2 * kv);
                }
                first += rows;
            } else {
                builder.matrix(p + "attention/" + role, s + "self_attn." + field + ".weight",
                               {rows, h}, {p + "attention_input"}, s + "self_attn." + field);
                if (qkv_bias) {
                    builder.vector(p + "attention/" + role + "_bias",
                                   s + "self_attn." + field + ".bias", {rows});
                }
            }
            packed.push_back(p + "attention/" + role);
        }
        builder.group(packed);
        builder.matrix(p + "attention/output", s + "self_attn.o_proj.weight", {h, q},
                       {p + "attention/context"}, s + "self_attn.o_proj");
        if (out_bias) {
            builder.vector(p + "attention/output_bias", s + "self_attn.o_proj.bias", {h});
        }
        if (qk_norm == "head" || qk_norm == "projection") {
            const bool head_scope = qk_norm == "head";
            builder.vector(p + "attention/query_norm", s + "self_attn.q_norm.weight",
                           {head_scope ? d : q});
            builder.vector(p + "attention/key_norm", s + "self_attn.k_norm.weight",
                           {head_scope ? d : kv});
        }
        if (c.value("attention_sinks", false)) {
            builder.vector_f32(p + "attention/sinks", s + "self_attn.sinks",
                               {c["num_attention_heads"].get<std::uint64_t>()});
        }
        if (c.contains("moe")) {
            add_experts(builder, model, base, c, p, s);
            continue;
        }
        // gate and up as two matrices, or one fused gate_up_proj (gate rows, then up) split by rows.
        const auto fused_gate_up = s + "mlp.gate_up_proj";
        const bool fused_mlp     = has_matrix(base, fused_gate_up);
        std::uint64_t row        = 0;
        for (const char* role : {"gate", "up"}) {
            if (fused_mlp) {
                builder.rows(p + "mlp/" + role, fused_gate_up + ".weight", {2 * inter, h}, row,
                             row + inter, {p + "ffn_input"}, fused_gate_up);
                row += inter;
            } else {
                builder.matrix(p + "mlp/" + role, s + "mlp." + role + "_proj.weight", {inter, h},
                               {p + "ffn_input"}, s + "mlp." + role + "_proj");
            }
        }
        builder.group({p + "mlp/gate", p + "mlp/up"});
        builder.matrix(p + "mlp/down", s + "mlp.down_proj.weight", {h, inter},
                       {p + "mlp/product"}, s + "mlp.down_proj");
    }
    return model;
}

bool is_dense_recipe(std::string_view name) noexcept {
    return name == "dense_encoded" || name == "dense_published" || name == "dense_nvfp4" ||
           name == "dense_fp8" ||
           name == "dense_q8" || name == "dense_bf16" || name == "moe_published" ||
           name == "moe_q8" || name == "nemotron_h_mtp_published";
}

std::string default_vocabulary(std::string_view recipe) {
    // The recipes that quantize the model themselves quantize its vocabulary too; every recipe
    // that runs a checkpoint as published keeps its vocabulary as published.
    return recipe == "dense_q8" || recipe == "dense_fp8" || recipe == "dense_nvfp4" ||
                   recipe == "moe_q8"
               ? "q8"
               : "native";
}

std::string apply_dense_recipe(std::string_view name, Model& model, Recipe& recipe,
                               const Checkpoint& base, std::string_view vocabulary) {
    const std::string vocab =
        vocabulary.empty() ? default_vocabulary(name) : std::string(vocabulary);
    if (vocab != "native" && vocab != "q8") {
        throw std::runtime_error("--vocab must be native or q8, not '" + vocab + "'");
    }
    const auto type = model.config().value("model_type", std::string());
    if (name == "nemotron_h_mtp_published") {
        // The MTP drafter as published: every tensor keeps its published encoding (a GGUF's Q4_0
        // words are kept word for word by keep_published; the BF16 gate stays BF16), nothing
        // quantized here. There is no q8 variant: a drafter is never re-quantized by NInferno.
        if (type != "nemotron_h_mtp") {
            throw std::runtime_error("requires a NemotronH MTP drafter");
        }
        return "native";
    }
    if (name.starts_with("moe_")) {
        if (type != "moe_decoder" && type != "nemotron_h") {
            throw std::runtime_error("requires a MoE decoder");
        }
        moe_recipe(model, recipe, base, name == "moe_q8");
        keep_encoded_head(model, recipe, base);
        if (vocab == "q8") { assign_vocabulary(model, recipe); }
        return vocab;
    }
    if (type != "dense_decoder") { throw std::runtime_error("requires a dense decoder"); }
    if (name == "dense_q8") {
        dense_q8(model, recipe);
    } else if (name == "dense_fp8") {
        dense_fp8(model, recipe);
    } else if (name == "dense_encoded") {
        dense_kept(model, recipe, base, Plain::Q8);
    } else if (name == "dense_published") {
        dense_kept(model, recipe, base, Plain::Published);
    } else if (name == "dense_nvfp4") {
        dense_kept(model, recipe, base, Plain::RoundNvfp4);
    } else if (name != "dense_bf16") {
        throw std::runtime_error("not a dense recipe");
    }
    keep_encoded_head(model, recipe, base);
    if (vocab == "q8") { assign_vocabulary(model, recipe); }
    return vocab;
}

std::string default_dense_recipe(const OJson& config, bool quantized) {
    const OJson& raw = text_part(config);
    if (const auto* type = member(raw, "model_type");
        type && type->is_string() && type->get<std::string>() == "nemotron_h_mtp") {
        return "nemotron_h_mtp_published";
    }
    if (const auto* type = member(raw, "model_type");
        type && type->is_string() && is_moe_type(type->get<std::string>())) {
        return "moe_published";
    }
    return quantized ? "dense_encoded" : "dense_bf16";
}

OJson resolved_rope(const OJson& source, std::uint64_t dim, std::uint64_t max_positions,
                    const std::string& where) {
    const auto model_type = source.value("model_type", std::string());
    const auto p          = rope_parameters(source, model_type, "full_attention");
    const auto r          = rotary(p, dim, max_positions, where);
    if (!r.long_inv_freq.empty()) {
        refuse(where + ": a rope with a long-context mode (" + r.type + ") is not supported here");
    }
    OJson out                = OJson::object();
    out["rope_type"]         = r.type;
    out["rope_theta"]        = number(p, "rope_theta", 10000.0);
    out["rotary_dim"]        = dim;
    out["attention_scaling"] = r.attention_scaling;
    OJson freq               = OJson::array();
    for (const float f : r.inv_freq) { freq.push_back(static_cast<double>(f)); }
    out["inv_freq"] = std::move(freq);
    return out;
}

} // namespace ninferno::convert
