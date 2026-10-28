#include "core/device_traits.h"
#include "product/plan/plan_facts.h"

#include "artifact/reader.h"
#include "convert/dense.h"
#include "convert/ggml_quants.h"
#include "convert/gguf.h"
#include "core/device_class.h"
#include "core/machine_facts.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace ninferno::product {
namespace {

using nlohmann::json;

// ---- JSON readers: a metadata or config value that is not a number reads as the fallback (a
// null sliding_window, a string where a count was expected), never as an error: these facts plan,
// the converter refuses.

double number(const json& j, const std::string& key, double fallback = 0) {
    const auto it = j.find(key);
    return it != j.end() && it->is_number() ? it->get<double>() : fallback;
}

// A count that may be a per-layer array (GGUF head_count_kv, feed_forward_length): the layer's
// value, the scalar for every layer, else the fallback.
double at_layer(const json& j, const std::string& key, std::size_t layer, double fallback = 0) {
    const auto it = j.find(key);
    if (it == j.end()) { return fallback; }
    if (it->is_array()) {
        return layer < it->size() && (*it)[layer].is_number() ? (*it)[layer].get<double>() : 0.0;
    }
    return it->is_number() ? it->get<double>() : fallback;
}

// The largest value of a per-layer count (the widest layer), or the scalar.
double widest(const json& j, const std::string& key, double fallback = 0) {
    const auto it = j.find(key);
    if (it == j.end()) { return fallback; }
    if (it->is_array()) {
        double out = 0;
        for (const auto& v : *it) { out = std::max(out, v.is_number() ? v.get<double>() : 0.0); }
        return out;
    }
    return it->is_number() ? it->get<double>() : fallback;
}

bool truthy(const json& v) {
    if (v.is_boolean()) { return v.get<bool>(); }
    if (v.is_number()) { return v.get<double>() != 0; }
    return false;
}

std::string lower(std::string_view text) {
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

std::string trim(std::string_view text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) { return {}; }
    const auto last = text.find_last_not_of(" \t\r\n");
    return std::string(text.substr(first, last - first + 1));
}

bool starts_with(std::string_view text, std::string_view prefix) {
    return text.substr(0, prefix.size()) == prefix;
}

std::uint32_t u32(double value) { return value > 0 ? static_cast<std::uint32_t>(value) : 0; }

// ---- The shape every source reads into, then one arithmetic for all of them.

struct Geometry {
    std::string architecture, model_type;
    std::uint32_t context = 0, layers = 0, hidden = 0, vocab = 0;
    // The attention layers' projections, each at its own widths (add_attention): Gemma 4's
    // full layers are twice as wide as its sliding ones.
    double attention_params = 0;
    // What a token attends with: heads x (key + value) summed over the full layers and over the
    // sliding ones (PlanTargetShape).
    double full_attention_width = 0, sliding_attention_width = 0;
    std::uint32_t full_layers = 0, full_kv_total = 0, full_key = 0, full_value = 0;
    std::uint32_t sliding_layers = 0, sliding_kv_total = 0, sliding_key = 0, sliding_value = 0;
    std::uint32_t window = 0;
    // Layers by kind for the parameter count: attention (a layer reading another's KV counts
    // here, with no KV of its own), recurrent mixers, feed-forward.
    std::uint32_t attention_layers = 0, recurrent_layers = 0, ffn_layers = 0;
    double ssm_inner = 0, ssm_state = 0, ssm_groups = 0, ssm_heads = 0, ssm_conv = 0;
    double ffn = 0, experts = 0, experts_used = 0, expert_ffn = 0, shared_ffn = 0;
    double ffn_matrices       = 3;     // gate, up, down; NemotronH's relu^2 experts have no gate
    bool dense_beside_experts = false; // Gemma 4 MoE: every layer runs its MLP and its experts
    bool tied = false, mtp = false;
};

// One attention layer: its projections (queries and output at heads x key and heads x value; keys
// and values, one projection when the keys are the values, where it keeps its own) and its KV.
void add_attention(Geometry& g, bool sliding, bool own_kv, double heads, double kv_heads,
                   double key, double value, bool keys_are_values = false) {
    ++g.attention_layers;
    g.attention_params += g.hidden * heads * (key + value);
    (sliding ? g.sliding_attention_width : g.full_attention_width) += heads * (key + value);
    if (!own_kv || kv_heads <= 0) { return; } // reads an earlier layer's keys and values
    g.attention_params += g.hidden * kv_heads * (keys_are_values ? key : key + value);
    if (sliding) {
        ++g.sliding_layers;
        g.sliding_kv_total += u32(kv_heads);
        g.sliding_key   = u32(key);
        g.sliding_value = u32(value);
    } else {
        ++g.full_layers;
        g.full_kv_total += u32(kv_heads);
        g.full_key   = u32(key);
        g.full_value = u32(value);
    }
}

// The state the family's planner lays out (models::StateShape), from the geometry; an empty
// reason when the geometry names it whole.
std::string state_of(const Geometry& g, models::StateShape& s) {
    const auto per_layer = [](std::uint32_t total, std::uint32_t layers) -> std::int32_t {
        return layers != 0 && total % layers == 0 ? static_cast<std::int32_t>(total / layers) : 0;
    };
    const std::string& type = g.model_type;
    if (type == "qwen3_5_text" || type == "qwen3_5_moe_text") {
        s.family = models::StateFamily::Qwen35;
    } else if (type == "dense_decoder" || type == "moe_decoder") {
        s.family = models::StateFamily::DenseDecoder;
    } else if (type == "gemma4_text") {
        s.family = models::StateFamily::Gemma4;
    } else if (type == "nemotron_h") {
        s.family = models::StateFamily::NemotronH;
    } else {
        return "no family serves the model";
    }
    if (g.hidden == 0 || g.vocab == 0) { return "the residual width or the vocabulary is unknown"; }
    s.hidden        = static_cast<std::int32_t>(g.hidden);
    s.output_rows   = static_cast<std::int32_t>(g.vocab);
    s.public_tokens = static_cast<std::int32_t>(g.vocab);
    s.full_layers   = g.full_layers;
    s.kv_heads      = per_layer(g.full_kv_total, g.full_layers);
    s.head_dim      = static_cast<std::int32_t>(g.full_key);
    if (g.full_layers == 0 || s.kv_heads == 0 || s.head_dim == 0 || g.full_value != g.full_key) {
        return "the full-attention layers' geometry is unknown or not one for all of them";
    }
    if (g.sliding_layers != 0) {
        s.sliding_layers   = g.sliding_layers;
        s.sliding_kv_heads = per_layer(g.sliding_kv_total, g.sliding_layers);
        s.sliding_head_dim = static_cast<std::int32_t>(g.sliding_key);
        s.window           = g.window;
        if (s.sliding_kv_heads == 0 || s.sliding_head_dim == 0 ||
            g.sliding_value != g.sliding_key || g.window == 0) {
            return "the sliding layers' geometry is unknown or not one for all of them";
        }
    }
    switch (s.family) {
    case models::StateFamily::DenseDecoder:
        // One geometry for the pool and the rings.
        if (s.sliding_layers != 0 &&
            (s.sliding_kv_heads != s.kv_heads || s.sliding_head_dim != s.head_dim)) {
            return "the sliding layers' geometry differs from the full layers'";
        }
        break;
    case models::StateFamily::Gemma4:
        if (s.sliding_layers == 0) { return "a Gemma 4 model without sliding layers"; }
        break;
    case models::StateFamily::Qwen35:
    case models::StateFamily::NemotronH:
        if (s.sliding_layers != 0) { return "sliding layers in a family without rings"; }
        break;
    }
    if (s.family == models::StateFamily::Qwen35 || s.family == models::StateFamily::NemotronH) {
        const auto heads = static_cast<std::uint64_t>(g.ssm_heads);
        if (g.recurrent_layers == 0 || heads == 0 || g.ssm_state <= 0 || g.ssm_groups <= 0 ||
            static_cast<std::uint64_t>(g.ssm_inner) % heads != 0) {
            return "the recurrent mixers' geometry is unknown";
        }
        s.recurrent_layers = g.recurrent_layers;
        s.value_heads      = static_cast<std::int32_t>(heads);
        s.value_dim = static_cast<std::int32_t>(static_cast<std::uint64_t>(g.ssm_inner) / heads);
        s.key_heads = static_cast<std::int32_t>(g.ssm_groups);
        s.key_dim   = static_cast<std::int32_t>(g.ssm_state);
        // GatedDeltaNet's convolution kernel (Mamba2's layout keeps its own rows).
        s.conv_kernel =
            s.family == models::StateFamily::Qwen35 ? static_cast<std::int32_t>(g.ssm_conv) : 0;
        if (s.family == models::StateFamily::Qwen35 && s.conv_kernel < 2) {
            return "the recurrent mixers' convolution kernel is unknown";
        }
    } else if (g.recurrent_layers != 0) {
        return "recurrent layers in a family without them";
    }
    return {};
}

// count_params of ui/catalog.cpp, the same arithmetic (a CPU test of the UI checks its figures):
// attention, recurrent mixers (in-projection to x, z, B, C and the heads' gates, out) and
// feed-forward (dense or experts) each over the layers that have them, and the vocabulary tables.
PlanSourceFacts finish(const Geometry& g, std::uint64_t weight_bytes, runtime::WeightBasis basis) {
    PlanSourceFacts out;
    auto& m                  = out.model;
    m.model_type             = g.model_type;
    m.architecture           = g.architecture;
    m.native_context         = g.context;
    m.full_layers            = g.full_layers;
    m.full_kv_heads_total    = g.full_kv_total;
    m.full_key_dim           = g.full_key;
    m.full_value_dim         = g.full_value;
    m.sliding_layers         = g.sliding_layers;
    m.sliding_kv_heads_total = g.sliding_kv_total;
    m.sliding_key_dim        = g.sliding_key;
    m.sliding_value_dim      = g.sliding_value;
    m.window                 = g.sliding_layers != 0 ? g.window : 0;
    m.recurrent_layers       = g.recurrent_layers;
    m.hidden                 = g.hidden;
    m.vocab                  = g.vocab;
    m.weight_bytes           = weight_bytes;
    m.weight_basis           = basis;
    m.offline_reason         = state_of(g, m.state);
    m.offline_bytes          = m.offline_reason.empty();
    // One sequence's state: heads x key x value per layer (GatedDeltaNet v_heads * dk * dv, Mamba2
    // heads * head_dim * state: inner x state either way), and the BF16 convolution window over
    // x, B and C (inner + 2 * groups * state channels, conv_kernel - 1 rows).
    if (g.recurrent_layers != 0 && g.ssm_inner > 0 && g.ssm_state > 0) {
        const auto inner = static_cast<std::uint64_t>(g.ssm_inner);
        const auto state = static_cast<std::uint64_t>(g.ssm_state);
        const auto conv_rows =
            g.ssm_conv > 1 ? static_cast<std::uint64_t>(g.ssm_conv) - 1 : std::uint64_t{0};
        const auto channels        = inner + 2 * static_cast<std::uint64_t>(g.ssm_groups) * state;
        m.recurrent_state_elements = std::uint64_t{g.recurrent_layers} * inner * state;
        m.recurrent_conv_bytes     = std::uint64_t{g.recurrent_layers} * conv_rows * channels * 2;
    }
    const bool moe           = g.experts > 1 && g.experts_used > 0 && g.expert_ffn > 0;
    const double dense_width = !moe || g.dense_beside_experts ? g.ffn : 0;
    m.ffn_active = u32(moe ? g.experts_used * g.expert_ffn + g.shared_ffn + dense_width : g.ffn);

    auto& s             = out.shape;
    s.hidden_size       = g.hidden;
    s.num_hidden_layers = g.layers;
    s.vocab_size        = g.vocab;
    s.experts           = moe ? static_cast<int>(g.experts) : 0;
    s.experts_per_token = moe ? static_cast<int>(g.experts_used) : 0;
    s.builtin_mtp       = g.mtp;
    s.tied_embeddings   = g.tied;
    s.full_attention_width    = g.full_attention_width;
    s.sliding_attention_width = g.sliding_attention_width;

    if (g.layers != 0 && g.hidden != 0) {
        const double hidden    = g.hidden;
        const double mixer =
            hidden * (2 * g.ssm_inner + 2 * g.ssm_groups * g.ssm_state + g.ssm_heads) +
            g.ssm_inner * hidden;
        const double gated  = g.ffn_matrices * hidden;
        const double shared = gated * g.shared_ffn;
        const double routed = moe ? g.experts * gated * g.expert_ffn : 0;
        const double taken  = moe ? g.experts_used * gated * g.expert_ffn : 0;
        const double dense  = gated * dense_width;
        const double vocab  = g.vocab > 0 ? g.vocab : 152064; // catalog.cpp's default
        const double table  = vocab * hidden;
        const double r      = g.recurrent_layers;
        const double f      = g.ffn_layers;
        const double a      = g.attention_params;
        m.params = a + r * mixer + f * (dense + shared + routed) + table * (g.tied ? 1 : 2);
        s.active_params        = a + r * mixer + f * (dense + shared + taken) + table;
        s.routed_expert_params = f * routed;
    }
    return out;
}

// ---- GGUF.

// The dense architectures of gguf_model.cpp's kDenseArchs (their GGUF names).
constexpr std::string_view kDenseGgufArchs[] = {
    "llama", "qwen2", "qwen3", "granite", "olmo2", "smollm3", "mistral3", "phi3", "muse-glimmer"};

// Which layers slide, as llama.cpp's load_swa_pattern and gguf_model.cpp read it: one flag per
// layer, or a period n (every n-th layer full; 0: all sliding), else the period llama.cpp sets for
// the architecture (gpt-oss: every other layer from the first; Gemma 3: five of six; OLMo 3 and
// Muse Glimmer: three of four), else none. Only with a window.
std::vector<bool> gguf_sliding(const json& metadata, const std::string& arch, std::size_t layers,
                               double window) {
    std::vector<bool> out(layers, false);
    if (!(window > 0)) { return out; }
    const auto it = metadata.find(arch + ".attention.sliding_window_pattern");
    long period   = 0;
    if (it != metadata.end() && it->is_array()) {
        for (std::size_t i = 0; i < it->size() && i < layers; ++i) { out[i] = truthy((*it)[i]); }
        return out;
    }
    if (it != metadata.end() && it->is_number()) {
        period = it->get<long>();
        if (period == 0) { return std::vector<bool>(layers, true); }
    } else {
        period = arch == "gpt-oss"                           ? 2
                 : arch == "gemma3"                          ? 6
                 : arch == "olmo2" || arch == "muse-glimmer" ? 4
                                                             : 0;
    }
    for (std::size_t i = 0; period > 0 && i < layers; ++i) {
        out[i] = static_cast<long>(i) % period < period - 1;
    }
    return out;
}

Geometry geometry_of_gguf(const json& md) {
    Geometry g;
    g.architecture = md.value("general.architecture", std::string());
    if (g.architecture.empty()) { return g; }
    g.model_type            = plan_model_type_of_gguf(md);
    const std::string& arch = g.architecture;
    const std::string p     = arch + ".";
    // block_count counts the MTP (NextN) blocks after the trunk too (gguf_model.cpp trunk_blocks).
    const double blocks = number(md, p + "block_count");
    const double nextn  = number(md, p + "nextn_predict_layers");
    g.mtp               = nextn > 0;
    g.layers            = u32(nextn > 0 && nextn < blocks ? blocks - nextn : blocks);
    g.context           = u32(number(md, p + "context_length"));
    g.hidden            = u32(number(md, p + "embedding_length"));
    g.vocab             = u32(number(md, p + "vocab_size"));
    if (const auto it = md.find("tokenizer.ggml.tokens");
        g.vocab == 0 && it != md.end() && it->is_array()) {
        g.vocab = static_cast<std::uint32_t>(it->size());
    }
    const std::size_t layers = g.layers;
    const bool has_kv        = md.contains(p + "attention.head_count_kv");
    std::vector<double> heads(layers), kv(layers);
    double first_heads = 0;
    for (std::size_t i = 0; i < layers; ++i) {
        heads[i] = at_layer(md, p + "attention.head_count", i);
        kv[i]    = has_kv ? at_layer(md, p + "attention.head_count_kv", i) : heads[i];
        if (first_heads == 0) { first_heads = heads[i]; }
    }
    const double head_dim  = first_heads > 0 && g.hidden > 0 ? g.hidden / first_heads : 128.0;
    const double key       = number(md, p + "attention.key_length", head_dim);
    const double value     = number(md, p + "attention.value_length", key);
    const double key_swa   = number(md, p + "attention.key_length_swa", key);
    const double value_swa = number(md, p + "attention.value_length_swa", key_swa);
    g.window               = u32(number(md, p + "attention.sliding_window"));
    const auto sliding     = gguf_sliding(md, arch, layers, g.window);
    // Gemma 4 E2B/E4B: the last shared_kv_layers layers read an earlier layer's keys and values.
    const double shared_kv = number(md, p + "attention.shared_kv_layers");
    const std::size_t first_shared =
        shared_kv > 0 && shared_kv < layers ? layers - static_cast<std::size_t>(shared_kv) : layers;

    // Recurrent layers: Qwen3.5 by attention.recurrent_layers, else full_attention_interval (4 its
    // default); another hybrid by its interval; else a layer without attention is a recurrent
    // mixer when the model has a state and no feed-forward of its own there (NemotronH's MoE
    // layers have neither attention nor a state).
    const bool qwen35          = arch == "qwen35" || arch == "qwen35moe";
    const bool has_state       = number(md, p + "ssm.state_size") > 0;
    const auto ffn_it          = md.find(p + "feed_forward_length");
    const bool ffn_per_layer   = ffn_it != md.end() && ffn_it->is_array();
    const double interval      = number(md, p + "full_attention_interval", qwen35 ? 4 : 0);
    const auto recurrent_flags = md.find(p + "attention.recurrent_layers");
    for (std::size_t i = 0; i < layers; ++i) {
        const double ffn_i = at_layer(md, p + "feed_forward_length", i);
        const bool has_ffn = !ffn_per_layer || ffn_i > 0;
        bool recurrent     = false;
        if (qwen35 && recurrent_flags != md.end() && recurrent_flags->is_array()) {
            recurrent = i < recurrent_flags->size() && truthy((*recurrent_flags)[i]);
        } else if (interval > 1) {
            recurrent = (i + 1) % static_cast<std::size_t>(interval) != 0;
        } else {
            recurrent = kv[i] <= 0 && has_state && !has_ffn;
        }
        g.ffn_layers += has_ffn ? 1 : 0;
        if (recurrent) {
            ++g.recurrent_layers;
            continue;
        }
        if (kv[i] <= 0) { continue; } // a feed-forward-only layer (NemotronH MoE)
        add_attention(g, sliding[i], i < first_shared, heads[i], kv[i],
                      sliding[i] ? key_swa : key, sliding[i] ? value_swa : value);
    }
    g.ssm_inner    = number(md, p + "ssm.inner_size");
    g.ssm_state    = number(md, p + "ssm.state_size");
    g.ssm_groups   = number(md, p + "ssm.group_count");
    g.ssm_heads    = number(md, p + "ssm.time_step_rank");
    g.ssm_conv     = number(md, p + "ssm.conv_kernel");
    g.ffn          = widest(md, p + "feed_forward_length");
    g.ffn_matrices = starts_with(arch, "nemotron_h") ? 2 : 3;
    g.experts      = number(md, p + "expert_count");
    g.experts_used = number(md, p + "expert_used_count");
    g.expert_ffn   = widest(md, p + "expert_feed_forward_length");
    g.shared_ffn   = number(md, p + "expert_shared_feed_forward_length") *
                   std::max(1.0, number(md, p + "expert_shared_count", 1));
    g.dense_beside_experts = arch == "gemma4";
    return g;
}

// A ggml type as artifact::format_name spells what the converter keeps it as: the k-, i- and
// small block quants unchanged (ggml_*), Q8_0 as q8_g32_fp16 (the same words), MXFP4 and NVFP4 as
// the engine's, the float types as bf16 / fp32.
std::string format_of_ggml(std::uint32_t type) {
    namespace g = convert::ggml;
    switch (type) {
    case g::F32:
        return "fp32";
    case g::F16:
    case g::BF16:
        return "bf16";
    case g::Q8_0:
        return "q8_g32_fp16";
    case g::MXFP4:
        return "mxfp4";
    case g::NVFP4:
        return "nvfp4";
    case g::I32:
        return "int32";
    default:
        return "ggml_" + g::type_name(type);
    }
}

// ---- config.json and the artifact's text config.

Geometry geometry_of_config(const json& config) {
    Geometry g;
    if (const auto it = config.find("architectures");
        it != config.end() && it->is_array() && !it->empty() && (*it)[0].is_string()) {
        g.architecture = (*it)[0].get<std::string>();
    } else {
        g.architecture = config.value("model_type", std::string());
    }
    g.model_type           = plan_model_type_of_config(config);
    const auto text        = config.find("text_config");
    const json& c          = text != config.end() && text->is_object() ? *text : config;
    const std::string type = c.value("model_type", config.value("model_type", std::string()));
    const double heads     = number(c, "num_attention_heads");
    const double kv_heads  = number(c, "num_key_value_heads", heads);
    g.hidden               = u32(number(c, "hidden_size"));
    const double head_dim = number(c, "head_dim", heads > 0 && g.hidden > 0 ? g.hidden / heads : 0);
    g.context             = u32(number(c, "max_position_embeddings"));
    g.vocab               = u32(number(c, "vocab_size"));
    double layers         = number(c, "num_hidden_layers");
    {
        const json* tied = c.contains("tie_word_embeddings")        ? &c["tie_word_embeddings"]
                           : config.contains("tie_word_embeddings") ? &config["tie_word_embeddings"]
                                                                    : nullptr;
        g.tied           = tied != nullptr && tied->is_boolean() && tied->get<bool>();
    }
    g.mtp = number(c, "mtp_num_hidden_layers", number(config, "mtp_num_hidden_layers")) > 0 ||
            number(c, "num_nextn_predict_layers", number(config, "num_nextn_predict_layers")) > 0;
    g.window = u32(number(c, "sliding_window"));

    // Layer kinds: layer_types (full / sliding / linear attention; the converter's NemotronH
    // mamba / attention / moe), else layers_block_type or hybrid_override_pattern (NemotronH), else
    // Gemma 3's older sliding_window_pattern, else full_attention_interval (Qwen3.5), else one
    // window on every layer (Mistral, Ministral 3, Phi-3: dense.cpp's rule), else every layer full.
    std::vector<std::string> kinds;
    const auto strings = [&](const json& array) {
        for (const auto& t : array) { kinds.push_back(t.is_string() ? t.get<std::string>() : ""); }
    };
    if (const auto it = c.find("layer_types"); it != c.end() && it->is_array()) {
        strings(*it);
    } else if (const auto blocks = c.find("layers_block_type");
               blocks != c.end() && blocks->is_array()) {
        strings(*blocks);
    } else if (const auto pattern = c.find("hybrid_override_pattern");
               pattern != c.end() && pattern->is_string()) {
        for (const char ch : pattern->get<std::string>()) {
            kinds.emplace_back(ch == 'M'   ? "mamba"
                               : ch == '*' ? "attention"
                               : ch == 'E' ? "moe"
                                           : "mlp");
        }
    } else if (const double every = number(c, "sliding_window_pattern");
               every > 0 && g.window > 0) {
        for (long i = 0; i < static_cast<long>(layers); ++i) {
            kinds.emplace_back((i + 1) % static_cast<long>(every) != 0 ? "sliding_attention"
                                                                       : "full_attention");
        }
    } else if (const double interval = number(c, "full_attention_interval"); interval > 1) {
        for (long i = 0; i < static_cast<long>(layers); ++i) {
            kinds.emplace_back((i + 1) % static_cast<long>(interval) == 0 ? "full_attention"
                                                                          : "linear_attention");
        }
    } else {
        const bool mistral = type == "mistral" || type == "ministral3" || type == "phi3";
        const bool use     = type != "mistral" || !c.contains("use_sliding_window") ||
                         truthy(c["use_sliding_window"]);
        const bool slides = mistral && use && g.window > 0 && g.window < g.context;
        for (long i = 0; i < static_cast<long>(layers); ++i) {
            kinds.emplace_back(slides ? "sliding_attention" : "full_attention");
        }
    }
    if (layers <= 0) { layers = static_cast<double>(kinds.size()); }
    kinds.resize(static_cast<std::size_t>(layers));
    g.layers = u32(layers);

    // Latent attention (kv_lora_rank) caches what this estimate does not model: no attention.
    if (c.contains("kv_lora_rank")) { kinds.assign(kinds.size(), ""); }

    // Gemma 4: the sliding layers' geometry is head_dim / num_key_value_heads, the full layers'
    // global_head_dim and, with attention_k_eq_v, num_global_key_value_heads (gemma4.cpp
    // text_config); the artifact spells both in attention_geometry.
    double sliding_dim = head_dim, sliding_kv = kv_heads;
    double full_dim = number(c, "global_head_dim", head_dim);
    // attention_k_eq_v: the full layers' keys are their values, one projection for both.
    const bool k_eq_v = c.contains("attention_k_eq_v") && truthy(c["attention_k_eq_v"]);
    double full_kv =
        c.contains("attention_k_eq_v")
            ? (k_eq_v ? number(c, "num_global_key_value_heads", kv_heads) : kv_heads)
            : number(c, "num_global_key_value_heads", kv_heads);
    if (const auto geo = c.find("attention_geometry"); geo != c.end() && geo->is_object()) {
        if (const auto s = geo->find("sliding_attention"); s != geo->end() && s->is_object()) {
            sliding_dim = number(*s, "head_dim", sliding_dim);
            sliding_kv  = number(*s, "num_key_value_heads", sliding_kv);
        }
        if (const auto f = geo->find("full_attention"); f != geo->end() && f->is_object()) {
            full_dim = number(*f, "head_dim", full_dim);
            full_kv  = number(*f, "num_key_value_heads", full_kv);
        }
    }
    // KV sharing (Gemma 4): the artifact's kv_source_layers, the checkpoint's last
    // num_kv_shared_layers.
    std::vector<bool> own_kv(kinds.size(), true);
    if (const auto src = c.find("kv_source_layers"); src != c.end() && src->is_array()) {
        for (std::size_t i = 0; i < own_kv.size() && i < src->size(); ++i) {
            own_kv[i] = !(*src)[i].is_number() || (*src)[i].get<double>() == static_cast<double>(i);
        }
    } else if (const double shared = number(c, "num_kv_shared_layers");
               shared > 0 && shared < static_cast<double>(own_kv.size())) {
        for (std::size_t i = own_kv.size() - static_cast<std::size_t>(shared); i < own_kv.size();
             ++i) {
            own_kv[i] = false;
        }
    }
    bool ffn_kinds           = false; // NemotronH: feed-forward on its own layers
    std::uint32_t ffn_layers = 0;
    for (std::size_t i = 0; i < kinds.size(); ++i) {
        const auto& kind = kinds[i];
        if (kind == "sliding_attention" && g.window > 0) {
            add_attention(g, true, own_kv[i], heads, sliding_kv, sliding_dim, sliding_dim);
        } else if (kind == "full_attention" || kind == "attention" || kind == "sliding_attention") {
            add_attention(g, false, own_kv[i], heads, full_kv, full_dim, full_dim, k_eq_v);
        } else if (kind == "linear_attention" || kind == "mamba") {
            ++g.recurrent_layers;
        } else if (kind == "moe" || kind == "mlp") {
            ffn_kinds = true;
            ++ffn_layers;
        }
    }
    g.ffn_layers = ffn_kinds ? ffn_layers : g.layers;

    // Recurrent geometry: Qwen3.5's GatedDeltaNet (linear_*), NemotronH's Mamba2 (the artifact's
    // mamba object, the checkpoint's mamba_* / ssm_state_size).
    if (c.contains("linear_num_value_heads")) {
        g.ssm_heads  = number(c, "linear_num_value_heads");
        g.ssm_inner  = g.ssm_heads * number(c, "linear_value_head_dim");
        g.ssm_state  = number(c, "linear_key_head_dim");
        g.ssm_groups = number(c, "linear_num_key_heads");
        g.ssm_conv   = number(c, "linear_conv_kernel_dim");
    } else if (const auto mamba = c.find("mamba"); mamba != c.end() && mamba->is_object()) {
        g.ssm_heads  = number(*mamba, "num_heads");
        g.ssm_inner  = g.ssm_heads * number(*mamba, "head_dim");
        g.ssm_state  = number(*mamba, "state_size");
        g.ssm_groups = number(*mamba, "n_groups");
        g.ssm_conv   = number(*mamba, "conv_kernel");
    } else if (c.contains("mamba_num_heads")) {
        g.ssm_heads  = number(c, "mamba_num_heads");
        g.ssm_inner  = g.ssm_heads * number(c, "mamba_head_dim");
        g.ssm_state  = number(c, "ssm_state_size");
        g.ssm_groups = number(c, "n_groups");
        g.ssm_conv   = number(c, "conv_kernel");
    }

    // Feed-forward: the dense width, and the experts (the artifact's moe object; the checkpoint's
    // num_experts / num_local_experts / n_routed_experts, gpt-oss's expert width being its
    // intermediate_size).
    g.ffn = number(c, "intermediate_size");
    if (const auto moe = c.find("moe"); moe != c.end() && moe->is_object()) {
        g.experts      = number(*moe, "num_experts");
        g.experts_used = number(*moe, "num_experts_per_tok");
        g.expert_ffn   = number(*moe, "moe_intermediate_size");
        g.shared_ffn   = number(*moe, "shared_intermediate_size");
    } else {
        g.experts =
            number(c, "num_experts", number(c, "num_local_experts", number(c, "n_routed_experts")));
        g.experts_used = number(c, "num_experts_per_tok",
                                number(c, "experts_per_token", number(c, "top_k_experts")));
        g.expert_ffn   = number(c, "moe_intermediate_size", g.ffn);
        g.shared_ffn   = number(c, "shared_expert_intermediate_size",
                                number(c, "moe_shared_expert_intermediate_size",
                                       number(c, "n_shared_experts") * g.expert_ffn));
    }
    g.ffn_matrices = starts_with(type, "nemotron_h") || g.model_type == "nemotron_h" ? 2 : 3;
    g.dense_beside_experts = g.model_type == "gemma4_text";
    return g;
}

bool is_family(std::string_view type) {
    return type == "qwen3_5_text" || type == "qwen3_5_moe_text" || type == "gemma4_text" ||
           type == "dense_decoder" || type == "moe_decoder" || type == "nemotron_h";
}

std::string normalized_board(std::string_view name) {
    std::string out = trim(lower(name));
    for (const std::string_view prefix : {"nvidia ", "geforce "}) {
        if (starts_with(out, prefix)) { out = trim(out.substr(prefix.size())); }
    }
    return out;
}

} // namespace

std::string plan_model_type_of_gguf(const json& metadata) {
    const std::string arch = metadata.value("general.architecture", std::string());
    if (arch == "qwen35") { return "qwen3_5_text"; }
    if (arch == "qwen35moe") { return "qwen3_5_moe_text"; }
    if (arch == "gemma4") { return "gemma4_text"; }
    if (arch == "gpt-oss" || arch == "qwen3moe") { return "moe_decoder"; }
    if (arch == "nemotron_h_moe") { return "nemotron_h"; }
    if (std::find(std::begin(kDenseGgufArchs), std::end(kDenseGgufArchs), arch) !=
        std::end(kDenseGgufArchs)) {
        return number(metadata, arch + ".expert_count") > 1 ? "moe_decoder" : "dense_decoder";
    }
    return {};
}

std::string plan_model_type_of_config(const json& config) {
    // An artifact's text config declares its family already.
    const std::string own = config.value("model_type", std::string());
    if (is_family(own)) { return own; }
    std::string arch;
    if (const auto it = config.find("architectures");
        it != config.end() && it->is_array() && !it->empty() && (*it)[0].is_string()) {
        arch = (*it)[0].get<std::string>();
    }
    if (starts_with(arch, "Qwen3_5Moe")) { return "qwen3_5_moe_text"; }
    if (starts_with(arch, "Qwen3_5")) { return "qwen3_5_text"; }
    if (starts_with(arch, "Gemma4")) {
        return arch.find("Assistant") == std::string::npos ? "gemma4_text" : "";
    }
    if (starts_with(arch, "GptOss")) { return "moe_decoder"; }
    if (starts_with(arch, "NemotronH")) {
        return arch.find("Mtp") == std::string::npos ? "nemotron_h" : ""; // the drafter
    }
    if (!arch.empty() && convert::is_dense_architecture(arch)) {
        const auto text = config.find("text_config");
        const json& c   = text != config.end() && text->is_object() ? *text : config;
        const double experts =
            number(c, "num_experts", number(c, "num_local_experts", number(c, "n_routed_experts")));
        return experts > 1 ? "moe_decoder" : "dense_decoder";
    }
    return {};
}

std::optional<models::DrafterStateShape> plan_drafter_state_of_config(const json& config,
                                                                      std::uint32_t target_hidden) {
    const auto draft = config.find("dflash_config");
    const auto kinds = config.find("layer_types");
    if (draft == config.end() || !draft->is_object() || kinds == config.end() ||
        !kinds->is_array() || target_hidden == 0) {
        return std::nullopt;
    }
    models::DrafterStateShape out;
    for (const auto& kind : *kinds) {
        const std::string name = kind.is_string() ? kind.get<std::string>() : std::string();
        if (name == "sliding_attention") {
            ++out.local_layers;
        } else if (name == "full_attention") {
            ++out.full_layers;
        } else {
            return std::nullopt;
        }
    }
    out.window      = out.local_layers != 0 ? u32(number(config, "sliding_window")) : 0;
    out.kv_heads    = static_cast<std::int32_t>(u32(number(config, "num_key_value_heads")));
    out.head_dim    = static_cast<std::int32_t>(u32(number(config, "head_dim")));
    const auto taps = draft->find("target_layer_ids");
    if (taps == draft->end() || !taps->is_array() || taps->empty() || out.kv_heads == 0 ||
        out.head_dim == 0 || (out.local_layers != 0 && out.window == 0)) {
        return std::nullopt;
    }
    const std::uint64_t rows = std::uint64_t{target_hidden} * taps->size();
    if (rows > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        return std::nullopt;
    }
    out.feature_rows         = static_cast<std::int32_t>(rows);
    out.prefill_feature_rows = out.feature_rows;
    std::string arch;
    if (const auto it = config.find("architectures");
        it != config.end() && it->is_array() && !it->empty() && (*it)[0].is_string()) {
        arch = (*it)[0].get<std::string>();
    }
    out.dflash2 = arch == "DFlash2DraftModel" || draft->contains("selector_rank");
    return out;
}

PlanSourceFacts plan_facts_of_gguf_metadata(const json& metadata, std::uint64_t file_bytes) {
    auto out = finish(geometry_of_gguf(metadata), file_bytes, runtime::WeightBasis::FileSize);
    if (auto base = convert::gguf_metadata_base_repository(metadata)) {
        out.shape.base_models = {std::move(*base)};
    }
    return out;
}

PlanSourceFacts plan_facts_of_gguf(const convert::GgufFile& head, std::uint64_t file_bytes) {
    Geometry g      = geometry_of_gguf(head.metadata);
    bool has_output = false;
    std::set<std::string> formats;
    std::uint64_t tensor_bytes = 0;
    // The NextN blocks after the trunk (blk.<b>.* for b at least the trunk's layers): the MTP
    // layers the converter moves to the mtp component, uploaded under MTP only.
    std::uint64_t mtp_bytes = 0;
    const auto block_of     = [](const std::string& name) -> std::optional<std::uint64_t> {
        if (!starts_with(name, "blk.")) { return std::nullopt; }
        const auto dot = name.find('.', 4);
        if (dot == std::string::npos || dot == 4) { return std::nullopt; }
        std::uint64_t block = 0;
        for (std::size_t i = 4; i < dot; ++i) {
            if (name[i] < '0' || name[i] > '9') { return std::nullopt; }
            block = block * 10 + static_cast<std::uint64_t>(name[i] - '0');
        }
        return block;
    };
    double elements = 0;
    std::uint64_t input_table_bytes = 0;
    for (const auto& t : head.tensors) {
        tensor_bytes += t.bytes;
        elements += static_cast<double>(t.elements());
        if (t.name == "token_embd.weight") { input_table_bytes = t.bytes; }
        if (const auto block = block_of(t.name); g.mtp && block && *block >= g.layers) {
            mtp_bytes += t.bytes;
        }
        has_output = has_output || t.name == "output.weight";
        if (t.name == "token_embd.weight" && g.vocab == 0 && t.ne.size() >= 2) {
            g.vocab = static_cast<std::uint32_t>(t.ne[1]);
        }
        formats.insert(format_of_ggml(t.type));
    }
    // The tensor table names the whole model only in an unsplit file: a split part's head lists
    // its own tensors, so the listed file sizes stand for the rest.
    const double splits      = number(head.metadata, "split.count");
    const bool whole         = splits <= 1 && !head.tensors.empty();
    // Tied embeddings only from a whole tensor table: a split part may hold output.weight in a
    // later file (unknown: the parameter count is left out, the label's bits rank it).
    g.tied                   = whole && !has_output;
    auto out                 = finish(g, whole ? tensor_bytes : file_bytes,
                      whole ? runtime::WeightBasis::GgufTensors : runtime::WeightBasis::FileSize);
    out.shape.weight_formats = std::vector<std::string>(formats.begin(), formats.end());
    if (auto base = convert::gguf_metadata_base_repository(head.metadata)) {
        out.shape.base_models = {std::move(*base)};
    }
    out.model.mtp_bytes      = whole ? mtp_bytes : 0;
    // The input table apart from the output head (a part that lists output.weight says so).
    out.shape.input_table_bytes = has_output ? input_table_bytes : 0;
    // A split part's table lists its own tensors only: the elements of the whole are not known.
    out.model.weight_elements = whole ? elements : 0;
    if (!whole && !has_output) { out.model.params = 0; }
    return out;
}

PlanSourceFacts plan_facts_of_config(const json& config, std::uint64_t weight_bytes,
                                     runtime::WeightBasis basis) {
    return finish(geometry_of_config(config), weight_bytes, basis);
}

PlanSourceFacts plan_facts_of_artifact(const std::filesystem::path& artifact) {
    const artifact::Reader reader(artifact);
    const auto& dir = reader.directory();
    const auto text = dir.components.find("text");
    if (text == dir.components.end()) {
        throw std::runtime_error(artifact.string() + ": the artifact has no text component");
    }

    // The vision encoder's tensors stay off the Device for a text plan: the objects only
    // "vision/..." parameters bind (a parameter is named <component>/<name>).
    // An object only a component the options may leave unloaded binds: the vision encoder's (under
    // --vision), the MTP layers' (under MTP), the own DFlash drafter's (under DFlash).
    enum Part : std::uint8_t { Vision, Mtp, Drafter, Other, Parts };

    std::vector<std::array<bool, Parts>> marks(dir.objects.size(), std::array<bool, Parts>{});
    // The input table's objects that the output head does not share (an untied table: a decode
    // step gathers one row of it).
    std::vector<std::uint8_t> table(dir.objects.size(), 0); // bit 0: input table, bit 1: head
    for (const auto& [name, binding] : dir.bindings) {
        const std::uint8_t bit = name == "text/token_embedding" ? 1 : name == "text/output_head" ? 2 : 0;
        for (const auto& piece : binding.parts) {
            if (bit != 0 && piece.object.index < table.size()) { table[piece.object.index] |= bit; }
        }
    }
    std::uint64_t input_table_bytes = 0;
    for (const auto& [name, binding] : dir.bindings) {
        const Part part = starts_with(name, "vision/")                                    ? Vision
                          : starts_with(name, "mtp/")                                     ? Mtp
                          : starts_with(name, "dflash/") || starts_with(name, "dflash2/") ? Drafter
                                                                                          : Other;
        for (const auto& piece : binding.parts) {
            if (piece.object.index < marks.size()) { marks[piece.object.index][part] = true; }
        }
    }
    std::uint64_t bytes = 0, vision_bytes = 0, mtp_bytes = 0, drafter_bytes = 0;
    std::set<std::string> formats;
    for (std::size_t i = 0; i < dir.objects.size(); ++i) {
        const auto* tensor = std::get_if<artifact::TensorObject>(&dir.objects[i]);
        if (tensor == nullptr) { continue; } // resources
        const auto& m    = marks[i];
        const bool alone = !m[Other] && (int(m[Vision]) + int(m[Mtp]) + int(m[Drafter]) == 1);
        if (alone && m[Vision]) {
            vision_bytes += tensor->bytes; // off the text plan's weights
            continue;
        }
        bytes += tensor->bytes;
        formats.insert(tensor->format);
        if (table[i] == 1) { input_table_bytes += tensor->bytes; }
        if (alone && m[Mtp]) { mtp_bytes += tensor->bytes; }
        if (alone && m[Drafter]) { drafter_bytes += tensor->bytes; }
    }
    auto out =
        finish(geometry_of_config(text->second.config), bytes, runtime::WeightBasis::Artifact);
    out.model.vision_bytes      = vision_bytes;
    out.model.mtp_bytes         = mtp_bytes;
    out.model.own_drafter_bytes = drafter_bytes;
    out.shape.input_table_bytes = input_table_bytes;
    out.shape.builtin_mtp       = out.shape.builtin_mtp || dir.components.contains("mtp");
    // A Qwen artifact's own DFlash drafter (its dflash or dflash2 component).
    for (const char* component : {"dflash", "dflash2"}) {
        const auto it = dir.components.find(component);
        if (it == dir.components.end() || out.model.state.drafter) { continue; }
        out.model.state.drafter = plan_drafter_state_of_config(it->second.config, out.model.hidden);
    }
    out.shape.weight_formats = std::vector<std::string>(formats.begin(), formats.end());
    return out;
}

PlanSourceFacts with_weights(PlanSourceFacts facts, std::uint64_t bytes,
                             runtime::WeightBasis basis) {
    // The speculation-only shares at the other quantization: in proportion (a listing names only
    // the file's size).
    const auto scaled = [&](std::uint64_t part) -> std::uint64_t {
        const std::uint64_t whole = facts.model.weight_bytes;
        if (whole == 0 || part == 0) { return 0; }
        return static_cast<std::uint64_t>(static_cast<long double>(part) * bytes / whole);
    };
    facts.model.mtp_bytes         = scaled(facts.model.mtp_bytes);
    facts.shape.input_table_bytes = scaled(facts.shape.input_table_bytes);
    facts.model.own_drafter_bytes = scaled(facts.model.own_drafter_bytes);
    facts.model.weight_bytes      = bytes;
    facts.model.weight_basis      = basis;
    return facts;
}

bool host_paravirtual() {
#ifdef _WIN32
    return true;
#else
    static const bool present = [] {
        std::error_code error;
        return std::filesystem::exists("/dev/dxg", error);
    }();
    return present;
#endif
}

std::optional<runtime::PlanDeviceFacts> plan_device_of_board(std::string_view spec,
                                                             std::string* error) {
    const auto fail = [&](std::string message) -> std::optional<runtime::PlanDeviceFacts> {
        if (error != nullptr) { *error = std::move(message); }
        return std::nullopt;
    };
    std::vector<std::string> parts;
    for (std::size_t begin = 0;;) {
        const auto comma = spec.find(',', begin);
        parts.push_back(trim(spec.substr(
            begin, comma == std::string_view::npos ? std::string_view::npos : comma - begin)));
        if (comma == std::string_view::npos) { break; }
        begin = comma + 1;
    }
    const std::string query = parts.front();
    if (query.empty()) { return fail("no board named: give a GPU name or a PCI id (0x2684)"); }
    std::optional<bool> display, paravirtual;
    for (std::size_t i = 1; i < parts.size(); ++i) {
        const auto fact = lower(parts[i]);
        if (fact == "headless") {
            display = false;
        } else if (fact == "display") {
            display = true;
        } else if (fact == "linux") {
            paravirtual = false;
        } else if (fact == "wddm") {
            paravirtual = true;
        } else {
            return fail("unknown board fact '" + parts[i] +
                        "' (headless | display | linux | wddm)");
        }
    }

    const auto table       = gpu_facts_table();
    const GpuFactsRow* row = nullptr;
    if (const auto q = lower(query); starts_with(q, "0x") && q.size() > 2) {
        char* end     = nullptr;
        const auto id = std::strtoul(q.c_str() + 2, &end, 16);
        if (end != nullptr && *end == '\0') {
            row = find_gpu_facts(static_cast<std::uint32_t>(id));
        }
        if (row == nullptr) { return fail("no board with PCI id " + query + " in gpu_facts.json"); }
    } else {
        // The exact name first, else the shortest name containing the query ("4090" finds
        // "GeForce RTX 4090", not "... 4090 D" or "... 4090 Laptop GPU"); the table's order
        // breaks a tie.
        const auto wanted = normalized_board(query);
        for (const auto& r : table) {
            if (normalized_board(r.name) == wanted) {
                row = &r;
                break;
            }
        }
        if (row == nullptr && !wanted.empty()) {
            for (const auto& c : table) {
                const auto name = normalized_board(c.name);
                if (name.find(wanted) != std::string::npos &&
                    (row == nullptr || name.size() < normalized_board(row->name).size())) {
                    row = &c;
                }
            }
        }
    }
    if (row == nullptr) {
        // Close names: those containing any word of the query.
        std::vector<std::string> words;
        for (std::size_t begin = 0; begin < query.size();) {
            const auto space = query.find(' ', begin);
            const auto word  = normalized_board(query.substr(begin, space - begin));
            if (!word.empty() && word != "rtx" && word != "nvidia" && word != "geforce") {
                words.push_back(word);
            }
            if (space == std::string::npos) { break; }
            begin = space + 1;
        }
        std::vector<std::string> close;
        for (const auto& r : table) {
            const auto name = normalized_board(r.name);
            const bool near = std::any_of(words.begin(), words.end(), [&](const std::string& w) {
                return name.find(w) != std::string::npos;
            });
            if (near && std::find(close.begin(), close.end(), r.name) == close.end()) {
                close.push_back(r.name);
                if (close.size() == 8) { break; }
            }
        }
        std::string message = "no board named '" + query + "' in gpu_facts.json";
        for (std::size_t i = 0; i < close.size(); ++i) {
            message += (i == 0 ? "; close names: " : ", ") + close[i];
        }
        return fail(message);
    }
    runtime::PlanDeviceFacts out;
    out.name               = row->name;
    out.source             = "named board " + row->name;
    out.total_bytes        = runtime::plan_board_total_bytes(row->vram_gib);
    out.compute_capability = row->major * 10 + row->minor;
    out.pci_device_id      = row->pci_device_id;
    out.sms                = row->sms;
    out.paravirtual        = paravirtual.value_or(host_paravirtual());
    // Unknown: a display under WDDM (a desktop's GPU, the conservative side), none on Linux.
    out.drives_display = display.value_or(out.paravirtual);
    return out;
}

runtime::PlanDeviceFacts plan_device_of_cuda(int device, bool measure_free, bool stack_reset) {
    const auto check = [&](cudaError_t status, const char* what) {
        if (status != cudaSuccess) {
            throw std::runtime_error("GPU " + std::to_string(device) + ": " + what + ": " +
                                     cudaGetErrorString(status));
        }
    };
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, device), "cudaGetDeviceProperties");
    runtime::PlanDeviceFacts out;
    out.name               = props.name;
    out.source             = "GPU " + std::to_string(device);
    out.total_bytes        = props.totalGlobalMem;
    out.compute_capability = props.major * 10 + props.minor;
    out.integrated         = props.integrated != 0;
    // The driver model and the display as a start reads them (core/device_traits.h, the
    // functions model_instance.cpp calls).
    out.paravirtual    = paravirtualized_gpu(device);
    out.drives_display = drives_display(device);
    // The PCI product id and the multiprocessors as the device class reads them (the
    // pciDeviceID property is the bus slot, not the product).
    const auto& cls   = device_class(device);
    out.pci_device_id = cls.pci_device_id;
    out.sms           = cls.sms;
    if (measure_free) {
        check(cudaSetDevice(device), "cudaSetDevice");
        // As a start reads it (model_instance.cpp, construct_or_plan): when its plan has an
        // automatic axis, with the CUDA stack limit at 0, which returns the driver's default
        // per-thread stack reservation (about 1 KiB for each resident thread: 0.09-0.19 GiB on
        // the lab's cards) before the free memory is read.
        if (stack_reset) { check(cudaDeviceSetLimit(cudaLimitStackSize, 0), "cudaDeviceSetLimit"); }
        std::size_t free_bytes = 0, total_bytes = 0;
        check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
        out.free_bytes = free_bytes;
    }
    return out;
}

runtime::PlanDeviceFacts plan_device_of_stats(const std::string& name, std::uint64_t total_bytes,
                                              int cc_major, int cc_minor,
                                              std::optional<bool> display,
                                              std::uint32_t pci_device_id, int sms) {
    runtime::PlanDeviceFacts out;
    out.name               = name;
    out.source             = "GPU (NVML)";
    out.total_bytes        = total_bytes;
    out.compute_capability = cc_major * 10 + cc_minor;
    out.pci_device_id      = pci_device_id;
    out.sms                = sms;
    out.paravirtual        = host_paravirtual();
    out.drives_display     = display.value_or(out.paravirtual);
    return out;
}

} // namespace ninferno::product
