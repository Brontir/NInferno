#include "ui/catalog.h"

#include "core/device_class.h"
#include "core/units.h"
#include "spec/registry.h"
#include "ui/embedded_catalog.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ninferno::ui {

namespace {

// Bits per weight a quantization stores on average (scales included), for a size estimate
// before the listing arrives.
double effective_bits(const CatalogSource& source) {
    if (source.format == "nvfp4") { return 4.5; }
    if (source.quant.find("MXFP4") != std::string::npos) { return 4.25; }
    return source.bits + 0.85;
}

} // namespace

Catalog Catalog::parse(const nlohmann::json& document) {
    Catalog out;
    for (const auto& g : document.at("gpus")) {
        CatalogGpu gpu;
        gpu.key           = g.at("key").get<std::string>();
        gpu.match         = g.value("match", std::vector<std::string>{gpu.key});
        gpu.memory_gib    = g.value("memory_gib", 0.0);
        gpu.bandwidth_gbs = g.value("bandwidth_gbs", 0.0);
        gpu.fp4           = g.value("fp4", false);
        out.gpus_.push_back(std::move(gpu));
    }
    for (const auto& m : document.at("models")) {
        CatalogModel model;
        model.id       = m.at("id").get<std::string>();
        model.name     = m.value("name", model.id);
        model.kind     = m.value("kind", std::string());
        model.use      = m.value("use", std::string());
        model.spec     = m.value("spec", std::string());
        model.params_b = m.value("params_b", 0.0);
        model.active_b = m.value("active_b", model.params_b);
        for (const auto& s : m.at("sources")) {
            CatalogSource source;
            source.repo   = s.at("repo").get<std::string>();
            source.quant  = s.value("quant", std::string());
            source.format = s.value("format", std::string("gguf"));
            source.bits   = s.value("bits", 4);
            if (source.bits > 8) {
                throw std::runtime_error("catalog: " + model.id + " lists a source above 8 bits");
            }
            model.sources.push_back(std::move(source));
        }
        for (const auto& r : m.value("measured", nlohmann::json::array())) {
            CatalogMeasure measure;
            measure.gpu       = r.at("gpu").get<std::string>();
            measure.tps       = r.value("tps", 0.0);
            measure.llama_cpp = r.value("llama_cpp", 0.0);
            measure.context   = r.value("context", std::string());
            measure.quant     = r.value("quant", std::string());
            model.measured.push_back(std::move(measure));
        }
        model.load = m.value("load", nlohmann::json::array());
        if (!model.load.is_array()) {
            throw std::runtime_error("catalog: " + model.id + "'s load is not a list");
        }
        model.drafters   = m.value("drafters", nlohmann::json::array());
        model.spec_suite = m.value("spec_suite", nlohmann::json::array());
        if (!model.drafters.is_array() || !model.spec_suite.is_array()) {
            throw std::runtime_error("catalog: " + model.id +
                                     "'s drafters or spec_suite is not a list");
        }
        if (model.sources.empty()) {
            throw std::runtime_error("catalog: " + model.id + " has no source");
        }
        out.models_.push_back(std::move(model));
    }
    return out;
}

const Catalog& Catalog::builtin() {
    static const Catalog catalog = parse(nlohmann::json::parse(embedded::kCatalogJson));
    return catalog;
}

const CatalogGpu* Catalog::gpu_for(const std::string& nvml_name) const {
    for (const auto& gpu : gpus_) {
        for (const auto& pattern : gpu.match) {
            if (nvml_name.find(pattern) != std::string::npos) { return &gpu; }
        }
    }
    return nullptr;
}

const CatalogGpu* Catalog::gpu_by_key(const std::string& key) const {
    for (const auto& gpu : gpus_) {
        if (gpu.key == key) { return &gpu; }
    }
    return nullptr;
}

std::string size_key(const std::string& repo, const std::string& quant) {
    return repo + "|" + quant;
}

TargetGpu target_of(const GpuStat& stat) {
    TargetGpu out;
    out.name          = stat.name;
    out.memory_gib    = to_gib(stat.memory_total);
    out.bandwidth_gbs = stat.bandwidth_gbs();
    // Native NVFP4 tensor cores in this release: its sm_120a engine (compute capability 12.0).
    // The 10.x and 11.x engines are portable builds that decode NVFP4 in software, so a GGUF
    // source suits them better (a 12.1 GB10 is aarch64, which the x86_64 release does not run).
    out.fp4 = stat.cc_major == 12 && stat.cc_minor == 0;
    if (stat.cc_major > 0) {
        out.chip          = gpu_chip(stat.pci_device_id, stat.cc_major, stat.cc_minor, stat.sms);
        out.profile_chain = gpu_similarity_chain(out.chip, stat.sms, stat.cc_major, stat.cc_minor);
    }
    return out;
}

SpeculationPlan speculation_plan(const std::string& registry_id, const TargetGpu& gpu) {
    SpeculationPlan out;
    out.profile                   = "none";
    const spec::ModelEntry* entry = spec::Registry::builtin().by_id(registry_id);
    if (entry == nullptr) { return out; }
    for (const auto& drafter : entry->drafters) {
        if (std::find(out.drafters.begin(), out.drafters.end(), drafter.kind) ==
            out.drafters.end()) {
            out.drafters.push_back(drafter.kind);
        }
    }
    std::vector<std::string> classes;
    for (const auto& profile : entry->profiles) {
        classes.push_back(profile.value("gpu_class", std::string()));
    }
    auto chain = gpu.profile_chain;
    if (std::find(chain.begin(), chain.end(), "*") == chain.end()) { chain.push_back("*"); }
    for (const auto& key : chain) {
        if (std::find(classes.begin(), classes.end(), key) == classes.end()) { continue; }
        out.profile_class = key;
        if (key == "*") {
            out.profile = "generic";
        } else if (!gpu.chip.empty() && (key == gpu.chip || key.rfind(gpu.chip + ":", 0) == 0)) {
            out.profile = "own";
        } else {
            out.profile = "similar";
        }
        break;
    }
    return out;
}

void attach_speculation(nlohmann::json& model_json, const SpeculationPlan& plan) {
    model_json["speculation"] = {{"drafters", plan.drafters},
                                 {"profile", plan.profile},
                                 {"profile_class", plan.profile_class}};
}

CatalogModel with_fitting_quant(const CatalogModel& model, const TargetGpu& gpu,
                                const std::map<std::string, nlohmann::json>& listings) {
    if (gpu.memory_gib <= 0) { return model; }
    const double budget = gpu.memory_gib * 0.95 - kReserveGib;
    for (const auto& source : model.sources) {
        const auto it = listings.find(source.repo);
        if (it == listings.end()) { continue; }
        for (const auto& q : it->second) {
            if (q.value("quant", std::string()) == source.quant && !q.value("drafter", false) &&
                to_gib(q.value("bytes", 0.0)) <= budget) {
                return model; // a listed source fits already
            }
        }
    }
    const nlohmann::json* best = nullptr;
    std::string best_repo;
    for (const auto& source : model.sources) {
        if (source.format != "gguf") { continue; }
        const auto it = listings.find(source.repo);
        if (it == listings.end()) { continue; }
        for (const auto& q : it->second) {
            const int bits = q.value("bits", 0);
            if (q.value("drafter", false) || bits < 2 || bits > 4 ||
                to_gib(q.value("bytes", 0.0)) > budget) {
                continue;
            }
            if (best == nullptr || q.value("bytes", 0.0) > best->value("bytes", 0.0)) {
                best      = &q;
                best_repo = source.repo;
            }
        }
    }
    if (best == nullptr) { return model; }
    CatalogModel out = model;
    out.sources.push_back(
        {best_repo, best->value("quant", std::string()), "gguf", best->value("bits", 4)});
    return out;
}

namespace {

// A metadata number, or the per-layer array's values (an int becomes one value per layer).
std::vector<double> per_layer(const nlohmann::json& metadata, const std::string& key, long layers) {
    const auto it = metadata.find(key);
    if (it == metadata.end()) { return {}; }
    if (it->is_array()) {
        std::vector<double> out;
        for (const auto& v : *it) { out.push_back(v.is_number() ? v.get<double>() : 0.0); }
        return out;
    }
    if (!it->is_number() || layers <= 0) { return {}; }
    return std::vector<double>(static_cast<std::size_t>(layers), it->get<double>());
}

double number(const nlohmann::json& metadata, const std::string& key, double fallback = 0) {
    const auto it = metadata.find(key);
    return it != metadata.end() && it->is_number() ? it->get<double>() : fallback;
}

} // namespace

void count_params(KvModel& kv, const ModelShape& s) {
    if (s.layers <= 0 || s.hidden <= 0) { return; }
    const double head_dim  = s.head_dim > 0 ? s.head_dim : (s.heads > 0 ? s.hidden / s.heads : 0);
    const double attention = s.hidden * head_dim * (2 * s.heads + 2 * s.kv_heads);
    // A Mamba2 / GatedDeltaNet mixer: in-projection to x, z, B, C and the heads' gates, out.
    const double mixer =
        s.hidden * (2 * s.ssm_inner + 2 * s.ssm_groups * s.ssm_state + s.ssm_heads) +
        s.ssm_inner * s.hidden;
    const double gated  = s.ffn_matrices * s.hidden; // projections per unit of width
    const double shared = gated * s.shared_ffn;
    const bool moe      = s.experts > 1 && s.experts_used > 0 && s.expert_ffn > 0;
    const double routed = moe ? s.experts * gated * s.expert_ffn : 0;
    const double taken  = moe ? s.experts_used * gated * s.expert_ffn : 0;
    const double dense  = moe ? 0 : gated * s.ffn;
    const double vocab  = s.vocab > 0 ? s.vocab : 152064;
    const double table  = vocab * s.hidden;
    const double a = s.attention_layers > 0 || s.mixer_layers > 0 ? s.attention_layers : s.layers;
    const double f = s.ffn_layers > 0 ? s.ffn_layers : s.layers;
    const double m = s.mixer_layers;
    kv.params =
        a * attention + m * mixer + f * (dense + shared + routed) + table * (s.tied ? 1 : 2);
    kv.active_params = a * attention + m * mixer + f * (dense + shared + taken) + table;
    kv.routed_share  = kv.params > 0 ? f * routed / kv.params : 0;
    kv.experts_used  = moe ? s.experts_used / s.experts : 0;
    // Prefill activations at one chunk: about twelve hidden-wide bf16 rows (norms, residual,
    // attention projections) and three of the widest feed-forward a token runs.
    const double ffn_active = moe ? s.experts_used * s.expert_ffn + s.shared_ffn : s.ffn;
    kv.workspace_bytes = static_cast<double>(kPrefillChunk) * (12 * s.hidden + 3 * ffn_active) * 2;
    // One sequence's recurrent state: fp32 heads x key x value (inner x state), and the bf16
    // convolution window over x, B and C.
    if (m > 0 && s.ssm_inner > 0 && s.ssm_state > 0) {
        kv.state_bytes = m * (s.ssm_inner * s.ssm_state * 4 +
                              std::max(0.0, s.ssm_conv - 1) *
                                  (s.ssm_inner + 2 * s.ssm_groups * s.ssm_state) * 2);
    }
}

KvModel kv_model_of_gguf(const nlohmann::json& metadata) {
    KvModel out;
    const std::string arch = metadata.value("general.architecture", std::string());
    if (arch.empty()) { return out; }
    out.architecture    = arch;
    const std::string p = arch + ".";
    const long layers   = static_cast<long>(number(metadata, p + "block_count"));
    out.native_context  = static_cast<long>(number(metadata, p + "context_length"));
    auto kv_heads       = per_layer(metadata, p + "attention.head_count_kv", layers);
    const auto heads    = per_layer(metadata, p + "attention.head_count", layers);
    if (layers <= 0 || (kv_heads.empty() && heads.empty())) { return out; }
    if (kv_heads.empty()) { kv_heads = heads; }
    const double width = number(metadata, p + "embedding_length");
    const double head_dim =
        !heads.empty() && heads.front() > 0 && width > 0 ? width / heads.front() : 128.0;
    const double key       = number(metadata, p + "attention.key_length", head_dim);
    const double value     = number(metadata, p + "attention.value_length", key);
    const double key_swa   = number(metadata, p + "attention.key_length_swa", key);
    const double value_swa = number(metadata, p + "attention.value_length_swa", key_swa);
    out.window             = static_cast<long>(number(metadata, p + "attention.sliding_window"));
    // Which layers slide: a per-layer pattern (true: sliding), else the one llama.cpp sets for the
    // architecture (gpt-oss: every other layer from the first; Gemma 3: five of six), else none.
    std::vector<bool> sliding(static_cast<std::size_t>(layers), false);
    if (const auto it = metadata.find(p + "attention.sliding_window_pattern");
        it != metadata.end() && it->is_array() && out.window > 0) {
        for (std::size_t i = 0; i < it->size() && i < sliding.size(); ++i) {
            sliding[i] = (*it)[i].is_boolean()
                             ? (*it)[i].get<bool>()
                             : (*it)[i].is_number() && (*it)[i].get<double>() != 0;
        }
    } else if (out.window > 0) {
        const long period = arch == "gpt-oss" ? 2 : arch == "gemma3" ? 6 : 0;
        for (long layer = 0; period > 0 && layer < layers; ++layer) {
            sliding[static_cast<std::size_t>(layer)] = layer % period < period - 1;
        }
    }
    // Hybrid stacks (Qwen3.5: one attention layer in full_attention_interval) keep KV only there.
    // A layer without attention is a recurrent mixer when the model has a state (ssm.state_size)
    // and no feed-forward of its own there (NemotronH's MoE layers have neither).
    const long interval = static_cast<long>(number(metadata, p + "full_attention_interval"));
    const bool recurrent     = number(metadata, p + "ssm.state_size") > 0;
    const auto ffn           = per_layer(metadata, p + "feed_forward_length", layers);
    const bool ffn_per_layer = [&] {
        const auto it = metadata.find(p + "feed_forward_length");
        return it != metadata.end() && it->is_array();
    }();
    long attention_layers = 0, mixer_layers = 0, ffn_layers = 0;
    for (long layer = 0; layer < layers; ++layer) {
        const auto l       = static_cast<std::size_t>(layer);
        const double h     = l < kv_heads.size() ? kv_heads[l] : 0;
        const bool has_ffn = !ffn_per_layer || (l < ffn.size() && ffn[l] > 0);
        ffn_layers += has_ffn ? 1 : 0;
        if (h <= 0 || (interval > 1 && (layer + 1) % interval != 0)) {
            mixer_layers += recurrent && (h <= 0 ? !ffn_per_layer || !has_ffn : true) ? 1 : 0;
            continue;
        }
        ++attention_layers;
        if (sliding[l]) {
            out.window_bytes_per_token += h * (key_swa + value_swa) * kKvElementBytes;
        } else {
            out.full_bytes_per_token += h * (key + value) * kKvElementBytes;
            out.kv_head_dim = key;
            ++out.full_layers;
        }
    }
    out.known  = out.full_bytes_per_token + out.window_bytes_per_token > 0;
    out.layers = layers;
    ModelShape shape;
    shape.layers       = static_cast<double>(layers);
    shape.hidden       = width;
    shape.heads        = heads.empty() ? 0 : *std::max_element(heads.begin(), heads.end());
    shape.kv_heads     = kv_heads.empty() ? 0 : *std::max_element(kv_heads.begin(), kv_heads.end());
    shape.head_dim     = key;
    shape.attention_layers = static_cast<double>(attention_layers);
    shape.mixer_layers     = static_cast<double>(mixer_layers);
    shape.ffn_layers       = static_cast<double>(ffn_layers);
    shape.ffn              = ffn.empty() ? 0 : *std::max_element(ffn.begin(), ffn.end());
    shape.ssm_inner        = number(metadata, p + "ssm.inner_size");
    shape.ssm_state        = number(metadata, p + "ssm.state_size");
    shape.ssm_groups       = number(metadata, p + "ssm.group_count");
    shape.ssm_heads        = number(metadata, p + "ssm.time_step_rank");
    shape.ssm_conv         = number(metadata, p + "ssm.conv_kernel");
    shape.ffn_matrices     = arch.rfind("nemotron_h", 0) == 0 ? 2 : 3;
    shape.experts      = number(metadata, p + "expert_count");
    shape.experts_used = number(metadata, p + "expert_used_count");
    shape.expert_ffn   = number(metadata, p + "expert_feed_forward_length");
    shape.shared_ffn   = number(metadata, p + "expert_shared_feed_forward_length") *
                       std::max(1.0, number(metadata, p + "expert_shared_count", 1));
    shape.vocab = number(metadata, p + "vocab_size");
    if (const auto it = metadata.find("tokenizer.ggml.tokens");
        shape.vocab <= 0 && it != metadata.end() && it->is_array()) {
        shape.vocab = static_cast<double>(it->size());
    }
    count_params(out, shape);
    return out;
}

KvModel kv_model_of_config(const nlohmann::json& config) {
    KvModel out;
    if (const auto archs = config.value("architectures", nlohmann::json::array());
        !archs.empty() && archs[0].is_string()) {
        out.architecture = archs[0].get<std::string>();
    } else {
        out.architecture = config.value("model_type", std::string());
    }
    const auto text         = config.find("text_config");
    const nlohmann::json& c = text != config.end() && text->is_object() ? *text : config;
    const long layers       = static_cast<long>(number(c, "num_hidden_layers"));
    const double heads      = number(c, "num_attention_heads");
    const double kv_heads   = number(c, "num_key_value_heads", heads);
    const double width      = number(c, "hidden_size");
    const double head_dim   = number(c, "head_dim", heads > 0 && width > 0 ? width / heads : 0);
    out.native_context      = static_cast<long>(number(c, "max_position_embeddings"));
    if (layers <= 0 || kv_heads <= 0 || head_dim <= 0 || c.contains("kv_lora_rank")) { return out; }
    const double full_dim   = number(c, "global_head_dim", head_dim);
    const double full_heads = number(c, "num_global_key_value_heads", kv_heads);
    out.window              = static_cast<long>(number(c, "sliding_window"));
    std::vector<std::string> types;
    if (const auto it = c.find("layer_types"); it != c.end() && it->is_array()) {
        for (const auto& t : *it) { types.push_back(t.is_string() ? t.get<std::string>() : ""); }
    } else if (const auto pattern = c.find("hybrid_override_pattern");
               pattern != c.end() && pattern->is_string()) {
        for (const char ch : pattern->get<std::string>()) {
            types.emplace_back(ch == '*' ? "full_attention" : ch == 'M' ? "mamba" : "none");
        }
    } else {
        const long interval = static_cast<long>(number(c, "full_attention_interval"));
        for (long layer = 0; layer < layers; ++layer) {
            types.emplace_back(interval <= 1 || (layer + 1) % interval == 0 ? "full_attention"
                                                                            : "linear_attention");
        }
    }
    long attention_layers = 0, mixer_layers = 0;
    for (const auto& type : types) {
        if (type == "sliding_attention" && out.window > 0) {
            out.window_bytes_per_token += kv_heads * 2 * head_dim * kKvElementBytes;
            ++attention_layers;
        } else if (type == "full_attention" || type == "attention" || type == "sliding_attention") {
            out.full_bytes_per_token += full_heads * 2 * full_dim * kKvElementBytes;
            out.kv_head_dim = full_dim;
            ++out.full_layers;
            ++attention_layers;
        } else if (type == "linear_attention" || type == "mamba") {
            ++mixer_layers;
        }
    }
    out.known  = out.full_bytes_per_token + out.window_bytes_per_token > 0;
    out.layers = layers;
    ModelShape shape;
    shape.layers   = static_cast<double>(layers);
    shape.hidden   = width;
    shape.heads    = heads;
    shape.kv_heads = kv_heads;
    shape.head_dim = head_dim;
    shape.ffn      = number(c, "intermediate_size");
    shape.experts =
        number(c, "num_experts", number(c, "num_local_experts", number(c, "n_routed_experts")));
    shape.experts_used = number(c, "num_experts_per_tok");
    shape.expert_ffn   = number(c, "moe_intermediate_size", shape.ffn);
    shape.shared_ffn   = number(c, "shared_expert_intermediate_size",
                                number(c, "n_shared_experts") * shape.expert_ffn);
    shape.vocab        = number(c, "vocab_size");
    shape.tied         = c.value("tie_word_embeddings", config.value("tie_word_embeddings", false));
    // Qwen3.5 GatedDeltaNet (linear_*) or NemotronH Mamba2 (mamba_*, ssm_state_size).
    if (mixer_layers > 0) {
        shape.attention_layers = static_cast<double>(attention_layers);
        shape.mixer_layers     = static_cast<double>(mixer_layers);
        const double v_heads   = number(c, "linear_num_value_heads", number(c, "mamba_num_heads"));
        shape.ssm_heads        = v_heads;
        shape.ssm_inner = v_heads * number(c, "linear_value_head_dim", number(c, "mamba_head_dim"));
        shape.ssm_state = number(c, "linear_key_head_dim", number(c, "ssm_state_size"));
        shape.ssm_groups = number(c, "linear_num_key_heads", number(c, "n_groups"));
        shape.ssm_conv   = number(c, "linear_conv_kernel_dim", number(c, "conv_kernel"));
        shape.ffn_matrices =
            c.value("model_type", std::string()).rfind("nemotron_h", 0) == 0 ? 2 : 3;
        if (const auto pattern = c.find("hybrid_override_pattern");
            pattern != c.end() && pattern->is_string()) {
            const auto p     = pattern->get<std::string>();
            shape.ffn_layers = static_cast<double>(
                std::count_if(p.begin(), p.end(), [](char ch) { return ch == 'E' || ch == '-'; }));
        }
    }
    count_params(out, shape);
    return out;
}

double kv_bytes(const KvModel& kv, long context) {
    const long windowed = kv.window > 0 ? std::min(context, kv.window) : context;
    return kv.full_bytes_per_token * static_cast<double>(context) +
           kv.window_bytes_per_token * static_cast<double>(windowed);
}

nlohmann::json to_json(const KvModel& kv) {
    return {{"known", kv.known},
            {"architecture", kv.architecture},
            {"full_bytes_per_token", kv.full_bytes_per_token},
            {"window_bytes_per_token", kv.window_bytes_per_token},
            {"window", kv.window},
            {"native_context", kv.native_context},
            {"layers", kv.layers},
            {"full_layers", kv.full_layers},
            {"kv_head_dim", kv.kv_head_dim},
            {"state_bytes", kv.state_bytes},
            {"workspace_bytes", kv.workspace_bytes},
            {"params", kv.params},
            {"active_params", kv.active_params},
            {"routed_share", kv.routed_share},
            {"experts_used", kv.experts_used}};
}

ModelEstimate estimate(const Catalog& catalog, const CatalogModel& model, const TargetGpu& gpu,
                       const SizeTable& sizes) {
    ModelEstimate out;
    out.model               = &model;
    const CatalogGpu* known = catalog.gpu_for(gpu.name);
    const double bandwidth  = gpu.bandwidth_gbs > 0 ? gpu.bandwidth_gbs
                              : known != nullptr    ? known->bandwidth_gbs
                                                    : 0.0;

    // The measurement on this GPU model, else the one on the measured card nearest in bandwidth.
    const CatalogMeasure* here    = nullptr;
    const CatalogMeasure* nearest = nullptr;
    double nearest_bandwidth      = 0;
    for (const auto& measure : model.measured) {
        const CatalogGpu* card = catalog.gpu_by_key(measure.gpu);
        if (card == nullptr) { continue; }
        if (known != nullptr && card->key == known->key) { here = &measure; }
        if (bandwidth > 0 && (nearest == nullptr || std::abs(card->bandwidth_gbs - bandwidth) <
                                                        std::abs(nearest_bandwidth - bandwidth))) {
            nearest           = &measure;
            nearest_bandwidth = card->bandwidth_gbs;
        }
    }

    for (std::size_t i = 0; i < model.sources.size(); ++i) {
        const auto& source = model.sources[i];
        SourceEstimate e;
        e.source = i;
        if (const auto it = sizes.find(size_key(source.repo, source.quant)); it != sizes.end()) {
            e.size_gib   = to_gib(it->second);
            e.size_known = true;
        } else {
            e.size_gib = to_gib(model.params_b * 1e9 * effective_bits(source) / 8.0);
        }
        const double need     = e.size_gib + kReserveGib;
        e.fit                 = gpu.memory_gib <= 0             ? "no"
                                : need <= gpu.memory_gib * 0.95 ? "fits"
                                : need <= gpu.memory_gib        ? "tight"
                                                                : "no";
        const bool same_quant = [&](const CatalogMeasure* m) {
            return m != nullptr && (m->quant.empty() || m->quant == source.quant);
        }(here);
        if (here != nullptr && same_quant) {
            e.tps       = here->tps;
            e.llama_cpp = here->llama_cpp;
            e.tps_basis = "measured";
            e.context   = here->context.empty() ? "auto" : here->context;
        } else if (nearest != nullptr && nearest_bandwidth > 0) {
            // Decode follows bandwidth only so far: the measured cards gain at most ~1.5x from a
            // faster bus (Qwen3.8 27B: 1.55x for 1.78x from the 4090 to the 5090), small models
            // less (host and launch overheads); a slower bus loses in proportion.
            e.tps       = nearest->tps * std::min(bandwidth / nearest_bandwidth, 1.5);
            e.tps_basis = "scaled from " + nearest->gpu;
            e.context   = "auto";
        } else if (bandwidth > 0 && model.active_b > 0) {
            // Active weight bytes read per token, at the 55 % of peak bandwidth the measured cards
            // reach without speculation.
            const double bytes = model.active_b * 1e9 * effective_bits(source) / 8.0;
            e.tps              = 0.55 * bandwidth * 1e9 / bytes;
            e.tps_basis        = "bandwidth";
            e.context          = "auto";
        } else {
            e.context = "auto";
        }
        out.sources.push_back(std::move(e));
    }

    // Preferred: NVFP4 on an FP4 card, GGUF elsewhere; among those that fit, the most bits
    // (higher quality) first; else whatever fits.
    const auto rank = [&](const SourceEstimate& e) {
        const auto& source = model.sources[e.source];
        int score          = 0;
        if (e.fit == "fits") {
            score += 100;
        } else if (e.fit == "tight") {
            score += 50;
        }
        if (gpu.fp4 ? source.format == "nvfp4" : source.format == "gguf") { score += 10; }
        score += source.bits;
        return score;
    };
    for (std::size_t i = 0; i < out.sources.size(); ++i) {
        if (out.sources[i].fit == "no") { continue; }
        if (!out.preferred || rank(out.sources[i]) > rank(out.sources[*out.preferred])) {
            out.preferred = i;
        }
    }
    out.recommended = out.preferred.has_value() && out.sources[*out.preferred].fit == "fits";
    return out;
}

nlohmann::json to_json(const ModelEstimate& e) {
    const auto& model      = *e.model;
    nlohmann::json sources = nlohmann::json::array();
    for (const auto& s : e.sources) {
        const auto& source = model.sources[s.source];
        sources.push_back(
            {{"repo", source.repo},
             {"quant", source.quant},
             {"format", source.format},
             {"bits", source.bits},
             {"arg", source.quant.empty() ? source.repo : source.repo + ":" + source.quant},
             {"size_gib", std::round(s.size_gib * 10) / 10},
             {"size_known", s.size_known},
             {"fit", s.fit},
             {"tps", std::round(s.tps * 10) / 10},
             {"tps_basis", s.tps_basis},
             {"llama_cpp", s.llama_cpp},
             {"context", s.context}});
    }
    return {{"id", model.id},
            {"name", model.name},
            {"kind", model.kind},
            {"use", model.use},
            {"spec", model.spec},
            {"params_b", model.params_b},
            {"active_b", model.active_b},
            {"sources", sources},
            {"load", model.load},
            {"drafters", model.drafters},
            {"spec_suite", model.spec_suite},
            {"preferred", e.preferred ? nlohmann::json(*e.preferred) : nlohmann::json(nullptr)},
            {"recommended", e.recommended}};
}

} // namespace ninferno::ui
