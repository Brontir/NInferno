#include "convert/qwen3_5.h"

#include "convert/formats.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>

namespace ninferno::convert {
namespace {

// ---------------------------------------------------------------------------------------------
// qwen3_5.py configs

// Qwen3_5 text architectures and whether they are MoE.
const std::map<std::string, bool, std::less<>>& text_architectures() {
    static const std::map<std::string, bool, std::less<>> table = {
        {"Qwen3_5ForCausalLM", false},
        {"Qwen3_5ForConditionalGeneration", false},
        {"Qwen3_5MoeForCausalLM", true},
        {"Qwen3_5MoeForConditionalGeneration", true},
    };
    return table;
}

bool contains(const OJson& list, std::string_view value) {
    return std::any_of(list.begin(), list.end(),
                       [&](const OJson& item) { return item.is_string() && item.get_ref<const std::string&>() == value; });
}

OJson get_or(const OJson& object, std::string_view key, const OJson& fallback) {
    const auto* value = member(object, key);
    return value ? *value : fallback;
}

// A strict Python int (a bool is not one).
bool is_int(const OJson& value) { return value.is_number_integer(); }

std::int64_t as_int(const OJson& value) {
    return value.is_number_unsigned() ? static_cast<std::int64_t>(value.get<std::uint64_t>())
                                      : value.get<std::int64_t>();
}

OJson rope_source(const OJson& raw, const std::string& label) {
    OJson rope = OJson::object();
    for (const char* field : {"rope_scaling", "rope_parameters"}) {
        const auto* value = member(raw, field);
        if (!present(value)) { continue; }
        if (!value->is_object()) {
            throw std::runtime_error(label + "." + field + ": expected an object");
        }
        // YaRN (transformers' "yarn" rope_type, as Qwen documents it for contexts past the trained
        // one): carried as yarn_factor and yarn_original, which the load applies as
        // --rope-yarn-factor does. Only the default ramp (beta 32 / 1) and attention scale.
        bool yarn = false;
        for (const char* alias : {"type", "rope_type"}) {
            const auto* kind = member(*value, alias);
            if (present(kind) && kind->is_string() && kind->get<std::string>() == "yarn") {
                yarn = true;
            } else {
                fixed(*value, alias, "default", label + "." + field);
            }
        }
        OJson own = *value;
        if (yarn) {
            const double factor = f32(member(*value, "factor"), label + ".factor");
            if (!(factor > 1.0) || factor > 64.0) {
                throw std::runtime_error(label + ": YaRN factor must be in (1, 64]");
            }
            rope["yarn_factor"]   = factor;
            rope["yarn_original"] = positive(member(*value, "original_max_position_embeddings"),
                                             label + ".original_max_position_embeddings");
            for (const char* name : {"beta_fast", "beta_slow", "attention_factor", "mscale",
                                     "mscale_all_dim", "truncate"}) {
                const auto* v = member(*value, name);
                const double standard = std::string(name) == "beta_fast" ? 32.0
                                      : std::string(name) == "beta_slow" ? 1.0
                                                                         : -1.0;
                if (present(v) && !(standard > 0 && v->is_number() && v->get<double>() == standard)) {
                    throw std::runtime_error(label + ": YaRN " + name +
                                             " other than the default is not implemented");
                }
            }
            for (const char* name : {"type", "rope_type", "factor", "original_max_position_embeddings",
                                     "beta_fast", "beta_slow"}) {
                own.erase(name);
            }
        } else if (const auto* factor = member(*value, "factor");
                   factor && f32(factor, label + ".factor") != 1.0) {
            throw std::runtime_error(label + ": scaled RoPE other than YaRN is not implemented");
        }
        for (auto it = own.begin(); it != own.end(); ++it) {
            if (rope.contains(it.key()) && !py_equal(rope[it.key()], it.value())) {
                throw std::runtime_error(label + ": conflicting RoPE field " + it.key());
            }
            rope[it.key()] = it.value();
        }
    }
    return rope;
}

OJson text_config(const OJson& source, bool mtp) {
    const auto* architectures = member(source, "architectures");
    if (!architectures || !architectures->is_array() || architectures->size() != 1 ||
        !(*architectures)[0].is_string() ||
        !text_architectures().contains((*architectures)[0].get_ref<const std::string&>())) {
        throw std::runtime_error("unsupported Qwen text architecture " +
                                 (architectures ? architectures->dump() : std::string("None")));
    }
    const bool moe = text_architectures().at((*architectures)[0].get_ref<const std::string&>());
    const auto* raw_ptr = member(source, "text_config");
    const OJson& raw    = raw_ptr ? *raw_ptr : source;
    fixed(raw, "hidden_act", "silu", "text");
    fixed(raw, "attention_bias", false, "text");
    fixed(raw, "attn_output_gate", true, "text");
    OJson result            = OJson::object();
    result["architectures"] = OJson::array({moe ? "Qwen3_5MoeForCausalLM" : "Qwen3_5ForCausalLM"});
    result["model_type"]    = moe ? "qwen3_5_moe_text" : "qwen3_5_text";
    for (const char* key :
         {"hidden_size", "vocab_size", "num_hidden_layers", "max_position_embeddings"}) {
        result[key] = positive(member(raw, key), std::string("text.") + key);
    }
    const auto tied =
        get_or(raw, "tie_word_embeddings", get_or(source, "tie_word_embeddings", false));
    if (!tied.is_boolean()) {
        throw std::runtime_error("text.tie_word_embeddings must be boolean");
    }
    result["tie_word_embeddings"] = tied;
    result["rms_norm_eps"] = f32_value(member(raw, "rms_norm_eps"), 1e-6, "text.rms_norm_eps");
    const auto count       = result["num_hidden_layers"].get<std::uint64_t>();
    OJson layers           = get_or(raw, "layer_types", nullptr);
    if (layers.is_null()) {
        const OJson fallback = 4;
        const auto* given    = member(raw, "full_attention_interval");
        const auto interval  = positive(given ? given : &fallback, "full_attention_interval");
        layers               = OJson::array();
        for (std::uint64_t i = 0; i < count; ++i) {
            layers.push_back((i + 1) % interval == 0 ? "full_attention" : "linear_attention");
        }
    }
    if (!layers.is_array() || layers.size() != count ||
        std::any_of(layers.begin(), layers.end(), [](const OJson& kind) {
            return !(kind == "full_attention" || kind == "linear_attention");
        })) {
        throw std::runtime_error("text.layer_types must describe every block");
    }
    result["layer_types"] = layers;
    if (contains(layers, "full_attention") || mtp) {
        for (const char* key : {"num_attention_heads", "num_key_value_heads", "head_dim"}) {
            result[key] = positive(member(raw, key), std::string("text.") + key);
        }
        if (result["num_attention_heads"].get<std::uint64_t>() %
            result["num_key_value_heads"].get<std::uint64_t>()) {
            throw std::runtime_error("text attention heads must be divisible by KV heads");
        }
        const auto rope = rope_source(raw, "text");
        fixed(rope, "mrope_interleaved", true, "text.rope_parameters");
        const auto factor_value =
            get_or(rope, "partial_rotary_factor", get_or(raw, "partial_rotary_factor", 0.25));
        const auto factor      = f32(&factor_value, "partial_rotary_factor");
        const auto theta_value = get_or(rope, "rope_theta", get_or(raw, "rope_theta", 10'000'000));
        const auto theta       = f32(&theta_value, "rope_theta");
        const auto* sections   = member(rope, "mrope_section");
        if (!sections || !sections->is_array() || sections->size() != 3 ||
            std::any_of(sections->begin(), sections->end(),
                        [](const OJson& v) { return !is_int(v) || as_int(v) < 0; })) {
            throw std::runtime_error("mrope_section must contain three nonnegative integers");
        }
        std::array<std::int64_t, 3> section{};
        for (std::size_t i = 0; i < 3; ++i) { section[i] = as_int((*sections)[i]); }
        const auto rotary = static_cast<std::int64_t>(
            static_cast<double>(result["head_dim"].get<std::uint64_t>()) * factor);
        if (factor > 1 || rotary <= 0 || rotary % 2 ||
            section[0] + section[1] + section[2] != rotary / 2) {
            throw std::runtime_error("MRoPE sections and rotary width disagree");
        }
        std::array<std::int64_t, 3> counts{};
        for (std::int64_t r = 0; r < rotary / 2; ++r) {
            const int axis = (r % 3 == 1 && r < 3 * section[1])   ? 1
                             : (r % 3 == 2 && r < 3 * section[2]) ? 2
                                                                  : 0;
            ++counts[static_cast<std::size_t>(axis)];
        }
        if (counts != section) {
            throw std::runtime_error("MRoPE sections exceed the interleaved axis ranges");
        }
        OJson parameters                    = OJson::object();
        parameters["rope_theta"]            = theta;
        parameters["partial_rotary_factor"] = factor;
        parameters["mrope_section"]         = OJson::array({section[0], section[1], section[2]});
        if (rope.contains("yarn_factor")) {
            parameters["yarn_factor"]   = rope["yarn_factor"];
            parameters["yarn_original"] = rope["yarn_original"];
        }
        result["rope_parameters"]           = std::move(parameters);
    }
    if (contains(layers, "linear_attention")) {
        fixed(raw, "mamba_ssm_dtype", "float32", "text");
        for (const char* key :
             {"linear_num_key_heads", "linear_key_head_dim", "linear_num_value_heads",
              "linear_value_head_dim", "linear_conv_kernel_dim"}) {
            result[key] = positive(member(raw, key), std::string("text.") + key);
        }
        if (result["linear_num_value_heads"].get<std::uint64_t>() %
            result["linear_num_key_heads"].get<std::uint64_t>()) {
            throw std::runtime_error("linear value heads must be divisible by key heads");
        }
        // GGUF K-quant out_proj kept in llama.cpp's tiled value-head column order.
        if (const auto* order = member(raw, "linear_output_head_order")) {
            if (*order != "grouped" && *order != "tiled") {
                throw std::runtime_error("text.linear_output_head_order must be grouped or tiled");
            }
            if (*order == "tiled") { result["linear_output_head_order"] = "tiled"; }
        }
    }
    const std::vector<const char*> fields =
        moe ? std::vector<const char*>{"num_experts", "num_experts_per_tok",
                                       "moe_intermediate_size", "shared_expert_intermediate_size"}
            : std::vector<const char*>{"intermediate_size"};
    for (const char* key : fields) {
        result[key] = positive(member(raw, key), std::string("text.") + key);
    }
    if (moe && result["num_experts_per_tok"].get<std::uint64_t>() >
                   result["num_experts"].get<std::uint64_t>()) {
        throw std::runtime_error("text.num_experts_per_tok " +
                                 result["num_experts_per_tok"].dump() +
                                 " exceeds text.num_experts " + result["num_experts"].dump());
    }
    if (mtp) {
        fixed(raw, "mtp_num_hidden_layers", 1, "text");
        fixed(raw, "mtp_use_dedicated_embeddings", false, "text");
    }
    return result;
}

OJson vision_config(const OJson& source, const OJson& text) {
    const auto* raw_ptr = member(source, "vision_config");
    if (!raw_ptr || !raw_ptr->is_object()) {
        throw std::runtime_error("Vision requires vision_config");
    }
    const auto& raw = *raw_ptr;
    fixed(raw, "in_channels", 3, "vision");
    fixed(raw, "hidden_act", "gelu_pytorch_tanh", "vision");
    fixed(raw, "deepstack_visual_indexes", OJson::array(), "vision");
    fixed(raw, "out_hidden_size", text["hidden_size"], "vision");
    OJson result         = OJson::object();
    result["model_type"] = text.contains("num_experts") ? "qwen3_5_moe_vision" : "qwen3_5_vision";
    for (const char* key :
         {"depth", "hidden_size", "intermediate_size", "num_heads", "patch_size",
          "temporal_patch_size", "spatial_merge_size", "num_position_embeddings"}) {
        result[key] = positive(member(raw, key), std::string("vision.") + key);
    }
    const auto h         = result["hidden_size"].get<std::uint64_t>();
    const auto heads     = result["num_heads"].get<std::uint64_t>();
    const auto positions = result["num_position_embeddings"].get<std::uint64_t>();
    auto root            = static_cast<std::uint64_t>(std::sqrt(static_cast<double>(positions)));
    while (root * root > positions) { --root; }
    while ((root + 1) * (root + 1) <= positions) { ++root; }
    if (h % heads || (h / heads) % 4 || root * root != positions) {
        throw std::runtime_error("Vision head or square position geometry is invalid");
    }
    return result;
}

OJson draft_config(const OJson& raw, const OJson& target, const std::string& backend) {
    const std::string architecture =
        backend == "dflash2" ? "DFlash2DraftModel" : "DFlashDraftModel";
    fixed(raw, "architectures", OJson::array({architecture}), backend);
    fixed(raw, "model_type", "qwen3", backend);
    fixed(raw, "hidden_act", "silu", backend);
    fixed(raw, "attention_bias", false, backend);
    fixed(raw, "is_causal", false, backend);
    fixed(raw, "hidden_size", target["hidden_size"], backend);
    fixed(raw, "vocab_size", target["vocab_size"], backend);
    fixed(raw, "num_target_layers", target["num_hidden_layers"], backend);
    OJson result            = OJson::object();
    result["architectures"] = OJson::array({architecture});
    result["model_type"]    = "qwen3";
    for (const char* key : {"intermediate_size", "num_attention_heads", "num_key_value_heads",
                            "head_dim", "num_hidden_layers", "max_position_embeddings"}) {
        result[key] = positive(member(raw, key), backend + "." + key);
    }
    if (result["num_attention_heads"].get<std::uint64_t>() %
            result["num_key_value_heads"].get<std::uint64_t>() ||
        result["head_dim"].get<std::uint64_t>() % 2) {
        throw std::runtime_error(backend + ": invalid attention geometry");
    }
    result["rms_norm_eps"] =
        f32_value(member(raw, "rms_norm_eps"), 1e-6, backend + ".rms_norm_eps");
    const auto rope           = rope_source(raw, backend);
    if (rope.contains("yarn_factor")) {
        throw std::runtime_error(backend + ": a drafter rotates with its trained frequencies; "
                                           "YaRN is not implemented");
    }
    const auto theta_value    = get_or(rope, "rope_theta", get_or(raw, "rope_theta", 10'000'000));
    result["rope_parameters"] = OJson{{"rope_theta", f32(&theta_value, backend + ".rope_theta")}};
    const auto count          = result["num_hidden_layers"].get<std::uint64_t>();
    OJson layers              = get_or(raw, "layer_types", nullptr);
    if (layers.is_null()) {
        const auto enabled = get_or(raw, "use_sliding_window", false);
        if (!enabled.is_boolean()) {
            throw std::runtime_error(backend + ".use_sliding_window must be boolean");
        }
        // Qwen3Config applies sliding attention at and above this layer index.
        const auto threshold = get_or(raw, "max_window_layers", 28);
        if (!is_int(threshold) || as_int(threshold) < 0) {
            throw std::runtime_error(backend + ".max_window_layers must be nonnegative");
        }
        layers = OJson::array();
        for (std::uint64_t i = 0; i < count; ++i) {
            layers.push_back(enabled.get<bool>() &&
                                     static_cast<std::int64_t>(i) >= as_int(threshold)
                                 ? "sliding_attention"
                                 : "full_attention");
        }
    }
    if (!layers.is_array() || layers.size() != count ||
        std::any_of(layers.begin(), layers.end(), [](const OJson& kind) {
            return !(kind == "sliding_attention" || kind == "full_attention");
        })) {
        throw std::runtime_error(backend + ": invalid layer_types");
    }
    result["layer_types"] = layers;
    if (contains(layers, "sliding_attention")) {
        result["sliding_window"] =
            positive(member(raw, "sliding_window"), backend + ".sliding_window");
    }
    const auto draft = get_or(raw, "dflash_config", OJson::object());
    const auto* taps = member(draft, "target_layer_ids");
    const auto layers_of_target =
        static_cast<std::int64_t>(target["num_hidden_layers"].get<std::uint64_t>());
    bool valid_taps = taps && taps->is_array() && !taps->empty();
    if (valid_taps) {
        std::set<std::int64_t> seen;
        for (const auto& tap : *taps) {
            if (!is_int(tap) || as_int(tap) < 0 || as_int(tap) >= layers_of_target ||
                !seen.insert(as_int(tap)).second) {
                valid_taps = false;
            }
        }
    }
    if (!valid_taps) { throw std::runtime_error(backend + ": invalid target_layer_ids"); }
    const auto* mask = member(draft, "mask_token_id");
    if (!mask || !is_int(*mask) || as_int(*mask) < 0 ||
        as_int(*mask) >= static_cast<std::int64_t>(target["vocab_size"].get<std::uint64_t>())) {
        throw std::runtime_error(backend + ": mask token outside target embedding");
    }
    OJson output               = OJson::object();
    output["target_layer_ids"] = *taps;
    output["mask_token_id"]    = *mask;
    if (backend == "dflash2") {
        for (const char* key :
             {"conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k"}) {
            output[key] = positive(member(draft, key), backend + "." + key);
        }
        if (target["hidden_size"].get<std::uint64_t>() %
            output["conv_group_size"].get<std::uint64_t>()) {
            throw std::runtime_error("DFlash2 hidden width must be divisible by conv_group_size");
        }
        for (const OJson* scope : {&raw, &draft}) {
            fixed(*scope, "sample_from_anchor", false, backend);
            fixed(*scope, "input_embedding_scale", 1.0, backend);
            fixed(*scope, "output_multiplier", 1.0, backend);
            const auto* cap = member(*scope, "final_logit_softcapping");
            if (present(cap) && !(cap->is_number() && cap->get<double>() == 0.0)) {
                throw std::runtime_error("DFlash2 final_logit_softcapping must be disabled");
            }
        }
    }
    result["dflash_config"] = std::move(output);
    return result;
}

// ---------------------------------------------------------------------------------------------
// qwen3_5.py _Builder

using Spans = std::vector<std::pair<std::uint64_t, std::uint64_t>>;

// _Builder.validate_source: an additional source must agree with the model on every field the
// site depends on.
void validate_source(const OJson& components, const Checkpoint& selected,
                     const Checkpoint& original, const std::string& name) {
    if (&selected == &original || !has_model_config(selected.config())) { return; }
    const auto component = name.substr(0, name.find('/'));
    const auto& target   = components.at("text").at("config");
    const auto ends      = [&](std::string_view suffix) { return name.ends_with(suffix); };
    const auto has = [&](std::string_view part) { return name.find(part) != std::string::npos; };
    OJson actual;
    OJson expected;
    std::set<std::string> fields;
    if (component == "text" || component == "mtp") {
        actual   = text_config(selected.config(), component == "mtp");
        expected = target;
        fields   = {"hidden_size"};
        if (ends("token_embedding") || ends("output_head")) { fields.insert("vocab_size"); }
        if (name == "text/output_head") { fields.insert("tie_word_embeddings"); }
        if (has("/attention/")) {
            fields.insert({"num_attention_heads", "num_key_value_heads", "head_dim"});
        }
        if (has("/gdn/")) {
            fields.insert({"linear_num_key_heads", "linear_key_head_dim", "linear_num_value_heads",
                           "linear_value_head_dim"});
            if (ends("/convolution")) { fields.insert("linear_conv_kernel_dim"); }
        }
        if (has("/mlp/")) { fields.insert("intermediate_size"); }
        if (has("/moe/experts/")) { fields.insert({"num_experts", "moe_intermediate_size"}); }
        if (has("/moe/shared/")) { fields.insert("shared_expert_intermediate_size"); }
        if (ends("/moe/router")) { fields.insert("num_experts"); }
    } else if (component == "vision") {
        actual   = vision_config(selected.config(), target);
        expected = components.at(component).at("config");
        fields   = {"hidden_size"};
        if (has("/attention/")) { fields.insert("num_heads"); }
        if (has("/mlp/")) { fields.insert("intermediate_size"); }
        if (name == "vision/patch_embedding") {
            fields.insert({"patch_size", "temporal_patch_size"});
        }
        if (name.starts_with("vision/merger/fc")) { fields.insert("spatial_merge_size"); }
        if (has("position_embedding")) { fields.insert("num_position_embeddings"); }
    } else {
        actual   = draft_config(selected.config(), target, component);
        expected = components.at(component).at("config");
        if (has("/attention/")) {
            fields.insert({"num_attention_heads", "num_key_value_heads", "head_dim"});
        }
        if (has("/mlp/")) { fields.insert("intermediate_size"); }
        std::set<std::string> draft_fields;
        if (ends("/feature_projection")) { draft_fields.insert("target_layer_ids"); }
        if (has("conv/")) {
            draft_fields.insert("conv_kernel_size");
            if (ends("/kernel_projection")) { draft_fields.insert("conv_group_size"); }
        }
        if (has("/candidate_selector/")) { draft_fields.insert("selector_rank"); }
        check_source_fields(name, actual["dflash_config"], expected["dflash_config"], draft_fields);
    }
    if (const auto at = name.find("/layers/"); at != std::string::npos) {
        const auto rest  = name.substr(at + 8);
        const auto index = std::stoull(rest.substr(0, rest.find('/')));
        bool compatible  = false;
        if (component == "vision") {
            compatible = index < actual["depth"].get<std::uint64_t>();
        } else if (component == "mtp") {
            compatible = index == 0;
        } else {
            compatible = index < actual["layer_types"].size() &&
                         actual["layer_types"][index] == expected["layer_types"][index];
        }
        if (!compatible) {
            throw std::runtime_error(name + ": additional source layer topology differs; provide "
                                            "an explicit logical source");
        }
    }
    check_source_fields(name, actual, expected, fields);
}

void replace_first(std::string& text, std::string_view from, std::string_view to) {
    const auto at = text.find(from);
    if (at != std::string::npos) { text.replace(at, from.size(), to); }
}

struct Site {
    std::vector<std::string> inputs;
    QType direct = QType::BF16;
    std::optional<Shape> source_shape;
    std::uint64_t offset = 0;
    std::optional<Spans> rows;
    std::optional<std::vector<std::size_t>> transpose;
};

// Whether a checkpoint names its text model model.language_model.* (a composite checkpoint, or a
// text-only config over composite names, as some quantized exports publish) or model.*.
bool composite_names(const Checkpoint& store) {
    return store.config().contains("text_config") ||
           store.has("model.language_model.embed_tokens.weight");
}

// Whether a checkpoint names its MoE experts one matrix each (experts.0.gate_proj with plain,
// GPTQ/AWQ/AutoRound or compressed-tensors fields) under `mlp` (the text model under either name
// prefix; an MTP layer under its own).
bool per_expert_names(const Checkpoint& store, const std::string& mlp) {
    std::vector<std::string> prefixes{mlp};
    for (const auto& [from, to] : {std::pair{"model.language_model.", "model."},
                                   std::pair{"model.", "model.language_model."}}) {
        if (mlp.starts_with(from)) { prefixes.push_back(to + mlp.substr(std::string(from).size())); }
    }
    for (const auto& prefix : prefixes) {
        for (const char* field : {".weight", ".qweight", ".weight_packed"}) {
            if (store.has(prefix + "experts.0.gate_proj" + field)) { return true; }
        }
    }
    return false;
}

class Builder {
public:
    Builder(Model& model) : model_(model), components_(std::make_shared<OJson>(model.components)) {}

    void add(const std::string& name, const Checkpoint& store, const std::string& source_name,
             Shape shape, Site site = {}) {
        const auto original   = site.source_shape ? *site.source_shape : shape;
        const auto components = components_;
        const auto* origin    = &store;
        auto factory          = [=](const Checkpoint& selected, MatrixFormat format) -> SourcePtr {
            validate_source(*components, selected, *origin, name);
            auto selected_name = source_name;
            // The text model's tensors sit under model.language_model. (a composite checkpoint) or
            // model. (a text-only one); a text-only config over composite names (some quantized
            // exports) keeps the names the checkpoint has.
            if (name.starts_with("text/") && has_model_config(selected.config())) {
                if (!composite_names(selected)) {
                    replace_first(selected_name, "model.language_model.", "model.");
                } else if (!composite_names(*origin) && selected_name.starts_with("model.")) {
                    replace_first(selected_name, "model.", "model.language_model.");
                }
            }
            if (site.transpose) {
                if (format != MatrixFormat::Auto) {
                    throw std::runtime_error(name + ": transpose source requires value access");
                }
                return transpose_source(tensor_source(selected, selected_name, original),
                                                 *site.transpose, shape);
            }
            if (original.size() == 2 && site.offset == 0) {
                auto ref = matrix_source(selected, selected_name, original, format);
                return site.rows ? select_rows(std::move(ref), *site.rows) : ref;
            }
            if (format != MatrixFormat::Auto) {
                throw std::runtime_error(name + ": provide an explicit encoded source mapping");
            }
            return tensor_source(selected, selected_name, shape, site.offset, original);
        };
        Parameter parameter;
        parameter.name           = name;
        parameter.shape          = shape;
        parameter.source         = factory(store, MatrixFormat::Auto);
        parameter.encoded_source = factory;
        parameter.inputs         = site.inputs;
        parameter.direct         = site.direct;
        parameter.residency      = name.substr(0, name.find('/'));
        // A projection read whole (or as a row selection) from one checkpoint matrix is probed
        // there for the encoding the checkpoint published it in.
        constexpr std::string_view weight = ".weight";
        if (!site.inputs.empty() && original.size() == 2 && site.offset == 0 && !site.transpose &&
            source_name.ends_with(weight)) {
            parameter.probe_leaf = source_name.substr(0, source_name.size() - weight.size());
            if (original != shape) { parameter.probe_shape = original; }
        }
        model_.add(std::move(parameter));
    }

    void group(const std::vector<std::string>& names) { model_.add_packing_group(names); }

    void attention(const std::string& prefix, const std::string& source_prefix,
                   const Checkpoint& store, const OJson& config) {
        const auto h     = config["hidden_size"].get<std::uint64_t>();
        const auto d     = config["head_dim"].get<std::uint64_t>();
        const auto heads = config["num_attention_heads"].get<std::uint64_t>();
        const auto q     = heads * d;
        const auto k     = config["num_key_value_heads"].get<std::uint64_t>() * d;
        for (const auto& [role, gate] : {std::pair{"query", 0ULL}, std::pair{"gate", 1ULL}}) {
            Spans spans;
            for (std::uint64_t head = 0; head < heads; ++head) {
                spans.emplace_back(head * 2 * d + gate * d, head * 2 * d + (gate + 1) * d);
            }
            add(prefix + "attention/" + role, store, source_prefix + "self_attn.q_proj.weight",
                {q, h},
                {.inputs       = {prefix + "mixer_input"},
                 .source_shape = Shape{2 * q, h},
                 .rows         = spans});
        }
        for (const auto& [role, source] :
             {std::pair{"key", "k_proj"}, std::pair{"value", "v_proj"}}) {
            add(prefix + "attention/" + role, store,
                source_prefix + "self_attn." + source + ".weight", {k, h},
                {.inputs = {prefix + "mixer_input"}});
        }
        for (const auto& [role, source] :
             {std::pair{"query_norm", "q_norm"}, std::pair{"key_norm", "k_norm"}}) {
            add(prefix + "attention/" + role, store,
                source_prefix + "self_attn." + source + ".weight", {d});
        }
        add(prefix + "attention/output", store, source_prefix + "self_attn.o_proj.weight", {h, q},
            {.inputs = {prefix + "attention/gated_output"}});
        std::vector<std::string> roles;
        for (const char* role : {"query", "key", "gate", "value"}) {
            roles.push_back(prefix + "attention/" + role);
        }
        group(roles);
        group({roles[0], roles[1]});
        group({roles[2], roles[3]});
    }

    void gdn(const std::string& prefix, std::string source_prefix, const Checkpoint& store,
             const OJson& config) {
        const auto h  = config["hidden_size"].get<std::uint64_t>();
        const auto kg = config["linear_num_key_heads"].get<std::uint64_t>() *
                        config["linear_key_head_dim"].get<std::uint64_t>();
        const auto nv       = config["linear_num_value_heads"].get<std::uint64_t>();
        const auto dv       = config["linear_value_head_dim"].get<std::uint64_t>();
        const auto vg       = nv * dv;
        const auto taps     = config["linear_conv_kernel_dim"].get<std::uint64_t>();
        const auto channels = 2 * kg + vg;
        source_prefix += "linear_attn.";
        for (const auto& [role, field] :
             {std::pair{"a_log", "A_log"}, std::pair{"dt_bias", "dt_bias"}}) {
            add(prefix + "gdn/" + role, store, source_prefix + field, {nv},
                {.direct = QType::FP32});
        }
        add(prefix + "gdn/convolution", store, source_prefix + "conv1d.weight", {taps, channels},
            {.source_shape = Shape{channels, 1, taps},
             .transpose    = std::vector<std::size_t>{2, 0, 1}});
        for (const auto& [role, field] :
             {std::pair{"a_projection", "in_proj_a"}, std::pair{"b_projection", "in_proj_b"}}) {
            add(prefix + "gdn/" + role, store, source_prefix + field + ".weight", {nv, h},
                {.inputs = {prefix + "mixer_input"}});
        }
        for (const auto& [role, start, count] :
             {std::tuple{"query", std::uint64_t{0}, kg}, std::tuple{"key", kg, kg},
              std::tuple{"value", 2 * kg, vg}}) {
            add(prefix + "gdn/" + role, store, source_prefix + "in_proj_qkv.weight", {count, h},
                {.inputs       = {prefix + "mixer_input"},
                 .source_shape = Shape{channels, h},
                 .rows         = Spans{{start, start + count}}});
        }
        add(prefix + "gdn/z", store, source_prefix + "in_proj_z.weight", {vg, h},
            {.inputs = {prefix + "mixer_input"}});
        add(prefix + "gdn/norm", store, source_prefix + "norm.weight", {dv});
        add(prefix + "gdn/output", store, source_prefix + "out_proj.weight", {h, vg},
            {.inputs = {prefix + "gdn/gated_output"}});
        std::vector<std::string> roles;
        for (const char* role : {"query", "key", "value", "z"}) {
            roles.push_back(prefix + "gdn/" + role);
        }
        group(roles);
        group({roles[0], roles[1]});
        group({roles[2], roles[3]});
        group({prefix + "gdn/a_projection", prefix + "gdn/b_projection"});
    }

    void dense(const std::string& prefix, const std::string& source_prefix, const Checkpoint& store,
               std::uint64_t h, std::uint64_t intermediate, bool draft = false) {
        const auto stem_input = prefix + (draft ? "mlp_input" : "ffn_input");
        for (const char* role : {"gate", "up"}) {
            add(prefix + "mlp/" + role, store, source_prefix + "mlp." + role + "_proj.weight",
                {intermediate, h}, {.inputs = {stem_input}});
        }
        add(prefix + "mlp/down", store, source_prefix + "mlp.down_proj.weight", {h, intermediate},
            {.inputs = {prefix + (draft ? "mlp_product" : "mlp/product")}});
        group({prefix + "mlp/gate", prefix + "mlp/up"});
    }

    void moe(const std::string& prefix, const std::string& source_prefix, const Checkpoint& store,
             const OJson& config) {
        const auto h      = config["hidden_size"].get<std::uint64_t>();
        const auto e      = config["num_experts"].get<std::uint64_t>();
        const auto ir     = config["moe_intermediate_size"].get<std::uint64_t>();
        const auto shared = config["shared_expert_intermediate_size"].get<std::uint64_t>();
        const auto sp     = source_prefix + "mlp.";
        const auto p      = prefix + "moe/";
        add(p + "router", store, sp + "gate.weight", {e, h}, {.inputs = {prefix + "ffn_input"}});
        add(p + "shared_score", store, sp + "shared_expert_gate.weight", {1, h},
            {.inputs = {prefix + "ffn_input"}});
        group({p + "router", p + "shared_score"});
        std::vector<std::string> gate_up;
        std::vector<std::string> downs;
        // A quantized export (GPTQ, AWQ, AutoRound, compressed-tensors) publishes every expert
        // matrix under its own name, experts.<e>.{gate,up,down}_proj, where the BF16 checkpoint has
        // the fused experts.gate_up_proj / experts.down_proj: each is then read (and probed for a
        // kept encoding) there.
        const bool separate = per_expert_names(store, sp);
        if (separate) {
            for (std::uint64_t expert = 0; expert < e; ++expert) {
                const auto ep = p + "experts/" + std::to_string(expert) + "/";
                const auto sx = sp + "experts." + std::to_string(expert) + ".";
                for (const char* role : {"gate", "up"}) {
                    add(ep + role, store, sx + role + "_proj.weight", {ir, h},
                        {.inputs = {prefix + "ffn_input"}});
                    gate_up.push_back(ep + role);
                }
                add(ep + "down", store, sx + "down_proj.weight", {h, ir},
                    {.inputs = {ep + "product"}});
                downs.push_back(ep + "down");
            }
        }
        for (std::uint64_t expert = 0; !separate && expert < e; ++expert) {
            const auto ep = p + "experts/" + std::to_string(expert) + "/";
            for (const auto& [role, half] : {std::pair{"gate", 0ULL}, std::pair{"up", 1ULL}}) {
                add(ep + role, store, sp + "experts.gate_up_proj", {ir, h},
                    {.inputs       = {prefix + "ffn_input"},
                     .source_shape = Shape{e, 2 * ir, h},
                     .offset       = (expert * 2 + half) * ir * h});
                gate_up.push_back(ep + role);
            }
            add(ep + "down", store, sp + "experts.down_proj", {h, ir},
                {.inputs       = {ep + "product"},
                 .source_shape = Shape{e, h, ir},
                 .offset       = expert * h * ir});
            downs.push_back(ep + "down");
        }
        group(gate_up);
        group(downs);
        for (const char* role : {"gate", "up"}) {
            add(p + "shared/" + role, store, sp + "shared_expert." + role + "_proj.weight",
                {shared, h}, {.inputs = {prefix + "ffn_input"}});
        }
        add(p + "shared/down", store, sp + "shared_expert.down_proj.weight", {h, shared},
            {.inputs = {p + "shared/product"}});
        group({p + "shared/gate", p + "shared/up"});
    }

    void block(const std::string& prefix, const std::string& source_prefix, const Checkpoint& store,
               const OJson& config, const std::string& mixer) {
        const auto h = config["hidden_size"].get<std::uint64_t>();
        add(prefix + "input_norm", store, source_prefix + "input_layernorm.weight", {h});
        if (mixer == "full_attention") {
            attention(prefix, source_prefix, store, config);
        } else {
            gdn(prefix, source_prefix, store, config);
        }
        add(prefix + "post_attention_norm", store,
            source_prefix + "post_attention_layernorm.weight", {h});
        if (config.contains("num_experts")) {
            moe(prefix, source_prefix, store, config);
        } else {
            dense(prefix, source_prefix, store, h,
                  config["intermediate_size"].get<std::uint64_t>());
        }
    }

    void vision(const Checkpoint& store, const OJson& config, std::uint64_t output_width) {
        const auto h            = config["hidden_size"].get<std::uint64_t>();
        const auto intermediate = config["intermediate_size"].get<std::uint64_t>();
        const auto ps           = config["patch_size"].get<std::uint64_t>();
        const auto pt           = config["temporal_patch_size"].get<std::uint64_t>();
        add("vision/patch_embedding", store, "model.visual.patch_embed.proj.weight",
            {h, 3 * pt * ps * ps},
            {.inputs = {"vision/patch_input"}, .source_shape = Shape{h, 3, pt, ps, ps}});
        add("vision/patch_embedding_bias", store, "model.visual.patch_embed.proj.bias", {h});
        add("vision/position_embedding", store, "model.visual.pos_embed.weight",
            {config["num_position_embeddings"].get<std::uint64_t>(), h});
        const auto depth = config["depth"].get<std::uint64_t>();
        for (std::uint64_t i = 0; i < depth; ++i) {
            const auto p  = "vision/layers/" + std::to_string(i) + "/";
            const auto sp = "model.visual.blocks." + std::to_string(i) + ".";
            for (const char* role : {"norm1", "norm2"}) {
                for (const char* kind : {"weight", "bias"}) {
                    add(p + role + "_" + kind, store, sp + role + "." + kind, {h});
                }
            }
            std::uint64_t number = 0;
            for (const char* role : {"query", "key", "value"}) {
                add(p + "attention/" + role, store, sp + "attn.qkv.weight", {h, h},
                    {.inputs       = {p + "attention_input"},
                     .source_shape = Shape{3 * h, h},
                     .rows         = Spans{{number * h, (number + 1) * h}}});
                add(p + "attention/" + role + "_bias", store, sp + "attn.qkv.bias", {h},
                    {.source_shape = Shape{3 * h}, .offset = number * h});
                ++number;
            }
            group({p + "attention/query", p + "attention/key", p + "attention/value"});
            group(
                {p + "attention/query_bias", p + "attention/key_bias", p + "attention/value_bias"});
            add(p + "attention/output", store, sp + "attn.proj.weight", {h, h},
                {.inputs = {p + "attention_output"}});
            add(p + "attention/output_bias", store, sp + "attn.proj.bias", {h});
            add(p + "mlp/fc1", store, sp + "mlp.linear_fc1.weight", {intermediate, h},
                {.inputs = {p + "mlp_input"}});
            add(p + "mlp/fc1_bias", store, sp + "mlp.linear_fc1.bias", {intermediate});
            add(p + "mlp/fc2", store, sp + "mlp.linear_fc2.weight", {h, intermediate},
                {.inputs = {p + "mlp_activation"}});
            add(p + "mlp/fc2_bias", store, sp + "mlp.linear_fc2.bias", {h});
        }
        const auto merge  = config["spatial_merge_size"].get<std::uint64_t>();
        const auto merger = merge * merge * h;
        add("vision/merger/norm_weight", store, "model.visual.merger.norm.weight", {h});
        add("vision/merger/norm_bias", store, "model.visual.merger.norm.bias", {h});
        add("vision/merger/fc1", store, "model.visual.merger.linear_fc1.weight", {merger, merger},
            {.inputs = {"vision/merger/input"}});
        add("vision/merger/fc1_bias", store, "model.visual.merger.linear_fc1.bias", {merger});
        add("vision/merger/fc2", store, "model.visual.merger.linear_fc2.weight",
            {output_width, merger}, {.inputs = {"vision/merger/activation"}});
        add("vision/merger/fc2_bias", store, "model.visual.merger.linear_fc2.bias", {output_width});
    }

    void draft(const Checkpoint& store, const OJson& config, const OJson& target,
               const std::string& backend) {
        const auto h         = target["hidden_size"].get<std::uint64_t>();
        const auto d         = config["head_dim"].get<std::uint64_t>();
        const auto q         = config["num_attention_heads"].get<std::uint64_t>() * d;
        const auto k         = config["num_key_value_heads"].get<std::uint64_t>() * d;
        const auto& settings = config["dflash_config"];
        add(backend + "/feature_projection", store, "fc.weight",
            {h, settings["target_layer_ids"].size() * h},
            {.inputs = {backend + "/target_features"}});
        add(backend + "/context_norm", store, "hidden_norm.weight", {h});
        add(backend + "/final_norm", store, "norm.weight", {h});
        const auto layers = config["num_hidden_layers"].get<std::uint64_t>();
        for (std::uint64_t i = 0; i < layers; ++i) {
            const auto p  = backend + "/layers/" + std::to_string(i) + "/";
            const auto sp = "layers." + std::to_string(i) + ".";
            add(p + "input_norm", store, sp + "input_layernorm.weight", {h});
            add(p + "post_attention_norm", store, sp + "post_attention_layernorm.weight", {h});
            for (const auto& [role, field, rows] :
                 {std::tuple{"query", "q_proj", q}, std::tuple{"key", "k_proj", k},
                  std::tuple{"value", "v_proj", k}}) {
                add(p + "attention/" + role, store, sp + "self_attn." + field + ".weight",
                    {rows, h}, {.inputs = {p + "query_projection_input"}});
                if (std::string_view(role) != "query") {
                    add(p + "attention/context_" + role, store,
                        sp + "self_attn." + field + ".weight", {rows, h},
                        {.inputs = {backend + "/context_input"}});
                }
            }
            group({p + "attention/query", p + "attention/key", p + "attention/value"});
            for (const auto& [role, field] :
                 {std::pair{"query_norm", "q_norm"}, std::pair{"key_norm", "k_norm"}}) {
                add(p + "attention/" + role, store, sp + "self_attn." + field + ".weight", {d});
            }
            add(p + "attention/output", store, sp + "self_attn.o_proj.weight", {h, q},
                {.inputs = {p + "attention_output"}});
            dense(p, sp, store, h, config["intermediate_size"].get<std::uint64_t>(), true);
            if (backend == "dflash2") {
                const auto taps   = settings["conv_kernel_size"].get<std::uint64_t>();
                const auto groups = h / settings["conv_group_size"].get<std::uint64_t>();
                for (const char* branch : {"attention_conv", "mlp_conv"}) {
                    add(p + branch + "/base_kernel", store, sp + branch + ".base_kernel",
                        {2, taps, h});
                    add(p + branch + "/kernel_projection", store,
                        sp + branch + ".kernel_projection.weight", {2 * taps * groups, h},
                        {.inputs = {p + branch + "_input"}});
                }
            }
        }
        if (backend == "dflash2") {
            const auto rank = settings["selector_rank"].get<std::uint64_t>();
            add(backend + "/candidate_selector/hidden_projection", store,
                "candidate_selector.hidden_projection.weight", {rank, h},
                {.inputs = {backend + "/final_hidden"}});
            for (const char* role : {"predecessor_codebook", "successor_codebook"}) {
                add(backend + "/candidate_selector/" + role, store,
                    std::string("candidate_selector.") + role,
                    {target["vocab_size"].get<std::uint64_t>(), rank});
            }
        }
    }

private:
    Model& model_;
    // The component configs as the factories validate other sources against (fixed before any
    // parameter exists).
    std::shared_ptr<const OJson> components_;
};

// ---------------------------------------------------------------------------------------------
// official_recipes.py (Qwen)

constexpr QType kQ4  = QType::Q4_G64_FP16;
constexpr QType kQ5  = QType::Q5_G64_FP16;
constexpr QType kQ6  = QType::Q6_G64_FP16;
constexpr QType kQ8  = QType::Q8_G32_FP16;
constexpr QType kFp8 = QType::FP8_E4M3FN_ROW_BF16;

bool ends_with_any(const std::string& name, std::initializer_list<std::string_view> suffixes) {
    return std::any_of(suffixes.begin(), suffixes.end(),
                       [&](std::string_view suffix) { return name.ends_with(suffix); });
}

bool has_part(const std::string& name, std::string_view part) {
    return name.find(part) != std::string::npos;
}

// _assign with a grouped format: grouped_absmax.
void assign_grouped(const Model& model, Recipe& recipe, std::string_view name, QType format) {
    recipe.assign(model.at(name), format, Method::GroupedAbsmax);
}

std::uint64_t layer_of(const std::string& name) {
    // text/layers/<n>/...
    const auto first = name.find('/', name.find('/') + 1) + 1;
    return std::stoull(name.substr(first, name.find('/', first) - first));
}

// A DFlash layer projects the context with its own key/value matrices: one stored object each.
void share_draft_context(const Model& model, Recipe& recipe) {
    for (const char* backend : {"dflash", "dflash2"}) {
        if (!model.components.contains(backend)) { continue; }
        const auto layers =
            model.components[backend]["config"]["num_hidden_layers"].get<std::uint64_t>();
        for (std::uint64_t layer = 0; layer < layers; ++layer) {
            const auto prefix =
                std::string(backend) + "/layers/" + std::to_string(layer) + "/attention/";
            for (const char* role : {"key", "value"}) {
                recipe.share(model, model.at(prefix + "context_" + role), model.at(prefix + role));
            }
        }
    }
}

void optional_components(const Model& model, Recipe& recipe) {
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        const auto& parameter = model.parameters[i];
        if (!parameter.projection()) { continue; }
        const auto& name = parameter.name;
        if (name.starts_with("vision/")) {
            QType format = kQ5;
            if (name == "vision/patch_embedding") {
                format = kQ6;
            } else if (name.starts_with("vision/merger/")) {
                format = kQ8;
            } else if (ends_with_any(name, {"/attention/query", "/attention/key",
                                            "/attention/value", "/mlp/fc1"})) {
                format = kQ4;
            }
            recipe.assign(i, format, Method::GroupedAbsmax);
        } else if (name.starts_with("mtp/") || name.starts_with("dflash/") ||
                   name.starts_with("dflash2/")) {
            if (ends_with_any(name,
                              {"/moe/router", "/moe/shared_score",
                               "/attention_conv/kernel_projection", "/mlp_conv/kernel_projection",
                               "/candidate_selector/hidden_projection"})) {
                continue;
            }
            recipe.assign(i, kQ8, Method::GroupedAbsmax);
        }
    }
    share_draft_context(model, recipe);
}

bool is_moe(const Model& model) { return model.config().contains("num_experts"); }

void require_dense(const Model& model) {
    if (is_moe(model)) { throw std::runtime_error("it requires a Qwen3.5 dense model"); }
}

void require_moe(const Model& model) {
    if (!is_moe(model)) { throw std::runtime_error("it requires a Qwen3.5 MoE model"); }
}

void dense_groupwise(Model& model, Recipe& recipe, QType vocabulary) {
    require_dense(model);
    optional_components(model, recipe);
    assign_grouped(model, recipe, "text/token_embedding", vocabulary);
    assign_grouped(model, recipe, "text/output_head", vocabulary);
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        const auto& parameter = model.parameters[i];
        const auto& name      = parameter.name;
        if (!name.starts_with("text/layers/") || !parameter.projection()) { continue; }
        if (ends_with_any(name, {"/gdn/a_projection", "/gdn/b_projection"})) {
            recipe.separate_parameter(i);
            continue;
        }
        const bool narrow = ends_with_any(name, {"/attention/query", "/attention/key", "/gdn/query",
                                                 "/gdn/key", "/mlp/gate", "/mlp/up"});
        recipe.assign(i, narrow ? kQ4 : kQ5, Method::GroupedAbsmax);
    }
}

// Any Qwen3.5 Dense size, every text projection and the vocabulary in Q8 (8.5 bits): the
// near-lossless groupwise form for sizes where 4-bit costs too much. The GDN control projections
// keep their direct BF16 form, as in every official recipe.
void dense_q8(Model& model, Recipe& recipe) {
    require_dense(model);
    optional_components(model, recipe);
    assign_grouped(model, recipe, "text/token_embedding", kQ8);
    assign_grouped(model, recipe, "text/output_head", kQ8);
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        const auto& parameter = model.parameters[i];
        const auto& name      = parameter.name;
        if (!name.starts_with("text/layers/") || !parameter.projection()) { continue; }
        if (ends_with_any(name, {"/gdn/a_projection", "/gdn/b_projection"})) {
            recipe.separate_parameter(i);
            continue;
        }
        recipe.assign(i, kQ8, Method::GroupedAbsmax);
    }
}

// Any Qwen3.5 Dense size with every tensor in its direct BF16 form: the unquantized artifact, a
// reference for the engine's arithmetic against the source checkpoint.
void dense_bf16(Model& model, Recipe& recipe) {
    require_dense(model);
    share_draft_context(model, recipe);
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        const auto& name = model.parameters[i].name;
        if (ends_with_any(name, {"/gdn/a_projection", "/gdn/b_projection"})) {
            recipe.separate_parameter(i);
        }
    }
}

// Any Qwen3.5 Dense size as published, quantizing nothing: a layer projection (text and MTP) or
// an untied output head the checkpoint holds in an encoding an engine format keeps (NVFP4 with its
// calibrated activation divisor when it has one, row/tensor-scaled FP8, int4 GPTQ/AWQ/compressed-
// tensors as u4 with zero points, int8 as q8_g32_fp16: formats.h kept_encoding) is imported as it
// is; every other tensor keeps its direct BF16 form (the token embedding, and a head published in
// BF16), but a DFlash drafter's projections, NInferno's own companion, take Q8. An NVFP4 matrix of any row count keeps its words: the layout completes a last 128-row
// scale tile its rows do not fill. The GDN control projections stay separate, as in every recipe.
// A MoE model's experts follow the same rule (an expert bank the checkpoint publishes in int4 stays
// int4); every routed expert bank and shared gate/up pair becomes one parent, as the MoE ops
// address them (NVFP4 matrices of such a parent keep a divisor each).
// (breadth's qwen3_5_published, with every kept encoding rather than NVFP4 and FP8 only.)
void dense_published(Model& model, Recipe& recipe, const Checkpoint& base) {
    // A DFlash drafter is NInferno's companion, not the published model: it takes the Q8 every
    // recipe gives it (the drafter's fused QKV kernels read Q8).
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        const auto& name = model.parameters[i].name;
        if (model.parameters[i].projection() &&
            (name.starts_with("dflash/") || name.starts_with("dflash2/")) &&
            !ends_with_any(name, {"/attention_conv/kernel_projection", "/mlp_conv/kernel_projection",
                                  "/candidate_selector/hidden_projection"})) {
            recipe.assign(i, kQ8, Method::GroupedAbsmax);
        }
    }
    share_draft_context(model, recipe);
    std::size_t kept = 0, published = 0;
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        const auto& parameter = model.parameters[i];
        const auto& name      = parameter.name;
        if (ends_with_any(name, {"/gdn/a_projection", "/gdn/b_projection"})) {
            recipe.separate_parameter(i);
            continue;
        }
        const bool layer = name.starts_with("text/layers/") || name.starts_with("mtp/layers/");
        // A tied head reads the token embedding, which keeps its BF16 form.
        const bool head = name == "text/output_head" &&
                          !model.config().value("tie_word_embeddings", false);
        if (!(layer || head) || !parameter.projection() || parameter.probe_leaf.empty()) {
            continue;
        }
        // A routed expert published in MXFP4 (llama.cpp's MXFP4_MOE) keeps its words as mxfp4,
        // as gpt-oss's: the MoE ops run it (W4A8 or the block-scaled MMA). Mapping it onto NVFP4
        // holds only for experts whose block exponents fit E4M3's range, and one bank mixing
        // NVFP4 and BF16 experts cannot be one parent.
        if (has_part(name, "/moe/experts/")) {
            const auto encoded = parameter.encoded_source
                                     ? parameter.encoded_source(base, MatrixFormat::Mxfp4)
                                     : SourcePtr{};
            std::optional<QType> format;
            try {
                format = encoded ? encoded->encoded_format() : std::nullopt;
            } catch (const std::exception&) {}
            if (format == QType::MXFP4) {
                recipe.assign(i, QType::MXFP4, Method::ImportEncoded, encoded);
                recipe.set_policy(i, kept_block_policy(QType::MXFP4));
                ++kept;
                continue;
            }
        }
        const auto& probed =
            parameter.probe_shape.empty() ? parameter.shape : parameter.probe_shape;
        const auto site = kept_encoding(base, parameter.probe_leaf, probed[0], probed[1]);
        // Published but not assigned here: groupwise int8 / Q8 the planner keeps wherever it is
        // decoded (keep_published), or a site the fidelity check refuses by name.
        published += parameter.source && parameter.source->dequantizes() ? 1 : 0;
        if (!site) { continue; }
        recipe.assign(i, site->format, Method::ImportEncoded,
                      parameter.encoded_source(base, encoded_matrix_format(site->format)));
        recipe.set_policy(i, head ? head_policy(*site) : kept_policy(*site));
        ++kept;
    }
    if (kept == 0 && published == 0) {
        throw std::runtime_error("qwen3_5_published: the checkpoint publishes no quantized "
                                 "projection (use qwen3_5_bf16)");
    }
    if (!is_moe(model)) { return; }
    for (const auto& names : model.packing_groups) {
        if (std::all_of(names.begin(), names.end(), [&](std::size_t i) {
                const auto& name = model.parameters[i].name;
                return has_part(name, "/moe/experts/") || has_part(name, "/moe/shared/");
            })) {
            recipe.explicit_groups.push_back(names);
        }
    }
}

void qwen3_6_35b_a3b(Model& model, Recipe& recipe) {
    require_moe(model);
    optional_components(model, recipe);
    assign_grouped(model, recipe, "text/token_embedding", kQ8);
    assign_grouped(model, recipe, "text/output_head", kQ6);
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        const auto& parameter = model.parameters[i];
        const auto& name      = parameter.name;
        if (!name.starts_with("text/layers/") || !parameter.projection()) { continue; }
        if (ends_with_any(name, {"/gdn/a_projection", "/gdn/b_projection", "/moe/router",
                                 "/moe/shared_score"})) {
            continue;
        }
        QType format = kQ8;
        if (has_part(name, "/moe/experts/")) {
            const auto layer = layer_of(name);
            format           = name.ends_with("/down")
                                   ? ((layer == 34 || layer == 38 || layer == 39) ? kQ6 : kQ5)
                                   : kQ4;
        }
        recipe.assign(i, format, Method::GroupedAbsmax);
    }
}

void qwen3_6_27b_nvfp4(Model& model, Recipe& recipe, SourceInputs& sources) {
    require_dense(model);
    optional_components(model, recipe);
    const auto& quantized = sources.get("quantized");
    assign_grouped(model, recipe, "text/token_embedding", kQ8);
    assign_grouped(model, recipe, "text/output_head", kQ8);
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        const auto& parameter = model.parameters[i];
        const auto& name      = parameter.name;
        if (!name.starts_with("text/layers/") || !parameter.projection()) { continue; }
        const auto layer = layer_of(name);
        if (ends_with_any(name, {"/gdn/a_projection", "/gdn/b_projection"})) {
            recipe.separate_parameter(i);
            continue;
        }
        const bool direct =
            (has_part(name, "/attention/") && !name.ends_with("/output") && layer < 24) ||
            (name.ends_with("/attention/output") && (layer == 3 || layer == 7)) ||
            (name.ends_with("/gdn/output") && layer == 4);
        if (direct) { continue; }
        recipe.assign(i, QType::NVFP4, Method::ImportEncoded,
                      parameter.encoded_source(quantized, MatrixFormat::Nvfp4));
        recipe.set_policy(i, ActivationPolicy::AllowA4);
    }
}

void qwen3_8_27b_nvfp4(Model& model, Recipe& recipe, SourceInputs& sources) {
    require_dense(model);
    optional_components(model, recipe);
    const auto& quantized = sources.get("quantized");
    recipe.assign(model.at("text/token_embedding"), kFp8, Method::Fp8RowMaxabs);
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        const auto& parameter = model.parameters[i];
        const auto& name      = parameter.name;
        if (!name.starts_with("text/") || name == "text/token_embedding") { continue; }
        auto source = parameter.encoded_source(quantized, MatrixFormat::Auto);
        if (!parameter.projection() ||
            ends_with_any(name, {"/gdn/a_projection", "/gdn/b_projection"})) {
            recipe.assign_source(i, std::move(source));
            continue;
        }
        const auto layer = name.starts_with("text/layers/")
                               ? static_cast<std::int64_t>(layer_of(name))
                               : std::int64_t{-1};
        const bool nvfp4 = has_part(name, "/mlp/") && layer < 56;
        recipe.assign(i, nvfp4 ? QType::NVFP4 : kFp8, Method::ImportEncoded,
                      parameter.encoded_source(quantized,
                                               nvfp4 ? MatrixFormat::Nvfp4 : MatrixFormat::Fp8Row));
        recipe.set_policy(i, nvfp4 ? ActivationPolicy::AllowA4 : ActivationPolicy::AllowA8);
    }
}

// _moe_encoded_source: one routed or shared expert matrix at its per-expert NVFP4 prefix.
SourcePtr moe_encoded_source(const Checkpoint& store, const Parameter& parameter) {
    const auto& name = parameter.name;
    const auto at    = name.find("/moe/");
    const auto head  = name.substr(0, at);
    const auto tail  = name.substr(at + 5);
    const auto layer = head.substr(head.rfind('/') + 1);
    std::string prefix;
    std::string role;
    if (tail.starts_with("experts/")) {
        const auto rest   = tail.substr(8);
        const auto slash  = rest.find('/');
        const auto expert = rest.substr(0, slash);
        role              = slash == std::string::npos ? std::string() : rest.substr(slash + 1);
        prefix            = "mlp.experts." + expert + ".";
    } else {
        prefix = "mlp.shared_expert.";
        role   = tail.substr(tail.rfind('/') + 1);
    }
    std::string field;
    if (role == "gate") {
        field = "gate_proj";
    } else if (role == "up") {
        field = "up_proj";
    } else if (role == "down") {
        field = "down_proj";
    } else {
        throw std::runtime_error(name + ": unknown expert matrix '" + role + "'");
    }
    const auto leaf = prefix + field;
    for (const char* root : {"model.language_model.layers.", "model.layers."}) {
        const auto full = std::string(root) + layer + "." + leaf;
        if (store.has(full + ".weight_packed")) {
            return compressed_source(store, full, parameter.shape, MatrixFormat::Nvfp4);
        }
        // A GGUF checkpoint exposes every expert of its NVFP4 banks as a 2-D matrix.
        if (store.has(full + ".weight") && store.describe(full + ".weight").view) {
            return matrix_source(store, full + ".weight", parameter.shape, MatrixFormat::Nvfp4);
        }
    }
    throw std::runtime_error(name + ": no NVFP4 source for " + leaf + " in " + store.path_string());
}

void qwen3_6_35b_a3b_nvfp4(Model& model, Recipe& recipe, SourceInputs& sources) {
    require_moe(model);
    const auto& quantized = sources.get("quantized");
    optional_components(model, recipe);
    assign_grouped(model, recipe, "text/token_embedding", kQ8);
    assign_grouped(model, recipe, "text/output_head", kQ6);
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        const auto& parameter = model.parameters[i];
        const auto& name      = parameter.name;
        if (!name.starts_with("text/layers/") || !parameter.projection()) { continue; }
        if (ends_with_any(name, {"/gdn/a_projection", "/gdn/b_projection", "/moe/router",
                                 "/moe/shared_score"})) {
            continue;
        }
        if (has_part(name, "/moe/experts/") || has_part(name, "/moe/shared/")) {
            recipe.assign(i, QType::NVFP4, Method::ImportEncoded,
                          moe_encoded_source(quantized, parameter));
            recipe.set_policy(i, ActivationPolicy::AllowA4);
        } else {
            recipe.assign(i, kQ8, Method::GroupedAbsmax);
        }
    }
    // The op takes the routed experts as one plane per bank, so every text-layer expert bank
    // (and shared gate/up pair) becomes one parent; the MTP experts keep their Q8 packing.
    for (const auto& names : model.packing_groups) {
        if (std::all_of(names.begin(), names.end(), [&](std::size_t i) {
                const auto& name = model.parameters[i].name;
                return name.starts_with("text/layers/") &&
                       (has_part(name, "/moe/experts/") || has_part(name, "/moe/shared/"));
            })) {
            recipe.explicit_groups.push_back(names);
        }
    }
}

} // namespace

bool is_qwen3_5_architecture(std::string_view architecture) noexcept {
    return text_architectures().contains(architecture);
}

Model build_qwen3_5_model(const Checkpoint& base, const std::vector<std::string>& components,
                          const std::map<std::string, const Checkpoint*>& companions,
                          const ResourceOverrides& overrides) {
    const std::set<std::string> selected(components.begin(), components.end());
    const std::set<std::string> supported = {"text", "vision", "mtp", "dflash", "dflash2"};
    if (!selected.contains("text") ||
        std::any_of(selected.begin(), selected.end(),
                    [&](const std::string& c) { return !supported.contains(c); })) {
        throw std::runtime_error("--components: text and any of vision, mtp, dflash, dflash2");
    }
    Model model;
    model.components["text"]["config"] = text_config(base.config(), selected.contains("mtp"));
    const auto config                  = model.config();
    // The Vision encoder: the checkpoint's own, or a projector beside a GGUF model (its mmproj,
    // the companion "vision").
    const auto vision_companion = companions.find("vision");
    const Checkpoint& vision_store =
        vision_companion != companions.end() ? *vision_companion->second : base;
    if (selected.contains("vision")) {
        model.components["vision"] =
            OJson{{"config", vision_config(vision_store.config(), config)}, {"target", "text"}};
    }
    if (selected.contains("mtp")) {
        const auto* architecture = config.contains("num_experts") ? "Qwen3_5MoeMTP" : "Qwen3_5MTP";
        model.components["mtp"]  = OJson{
             {"config", OJson{{"architectures", OJson::array({architecture})}}}, {"target", "text"}};
    }
    for (const char* backend : {"dflash", "dflash2"}) {
        if (!selected.contains(backend)) { continue; }
        const auto it = companions.find(backend);
        if (it == companions.end()) {
            throw std::runtime_error(std::string("selected ") + backend + " requires its source");
        }
        model.components[backend] = OJson{
            {"config", draft_config(it->second->config(), config, backend)}, {"target", "text"}};
    }
    const auto vision =
        selected.contains("vision") ? model.components["vision"]["config"] : OJson(nullptr);
    auto fallbacks = resource_fallbacks(base);
    if (selected.contains("vision") && vision_companion != companions.end()) {
        // A projector carries no preprocessor files: the Qwen3.5 / 3.6 / 3.8 ones (every model of
        // the family publishes the same two), with the projector's own geometry.
        const auto patch = vision["patch_size"].get<std::uint64_t>();
        const auto merge = vision["spatial_merge_size"].get<std::uint64_t>();
        const auto files = [&](std::uint64_t shortest, std::uint64_t longest, const char* type,
                               const char* kind) {
            OJson j            = OJson::object();
            j["size"]          = OJson{{"longest_edge", longest}, {"shortest_edge", shortest}};
            j["patch_size"]    = patch;
            j["temporal_patch_size"] = vision["temporal_patch_size"];
            j["merge_size"]    = merge;
            j["image_mean"]    = OJson::array({0.5, 0.5, 0.5});
            j["image_std"]     = OJson::array({0.5, 0.5, 0.5});
            j["processor_class"] = "Qwen3VLProcessor";
            j[type]            = kind;
            return j.dump(2) + "\n";
        };
        fallbacks.try_emplace("preprocessor_config.json",
                              files(65536, 16777216, "image_processor_type",
                                    "Qwen2VLImageProcessorFast"));
        fallbacks.try_emplace("video_preprocessor_config.json",
                              files(4096, 25165824, "video_processor_type",
                                    "Qwen3VLVideoProcessor"));
    }
    load_resources(model, base.root(), config["vocab_size"].get<std::uint64_t>(), overrides,
                   selected.contains("vision") ? &vision : nullptr, fallbacks);

    Builder builder(model);
    const auto h           = config["hidden_size"].get<std::uint64_t>();
    const auto r           = config["vocab_size"].get<std::uint64_t>();
    const auto text_prefix =
        composite_names(base) ? std::string("model.language_model.") : std::string("model.");
    builder.add("text/token_embedding", base, text_prefix + "embed_tokens.weight", {r, h});
    const auto head_source               = config["tie_word_embeddings"].get<bool>()
                                               ? text_prefix + "embed_tokens.weight"
                                               : std::string("lm_head.weight");
    std::vector<std::string> head_inputs = {"text/final_hidden"};
    for (const char* component : {"mtp", "dflash", "dflash2"}) {
        if (selected.contains(component)) {
            head_inputs.push_back(std::string(component) + "/final_hidden");
        }
    }
    builder.add("text/output_head", base, head_source, {r, h}, {.inputs = head_inputs});
    builder.add("text/final_norm", base, text_prefix + "norm.weight", {h});
    const auto& layers = config["layer_types"];
    for (std::size_t i = 0; i < layers.size(); ++i) {
        builder.block("text/layers/" + std::to_string(i) + "/",
                      text_prefix + "layers." + std::to_string(i) + ".", base, config,
                      layers[i].get<std::string>());
    }
    if (selected.contains("mtp")) {
        builder.add("mtp/input_projection", base, "mtp.fc.weight", {h, 2 * h},
                    {.inputs = {"mtp/stem_input"}});
        for (const auto& [role, field] :
             {std::pair{"embedding_norm", "pre_fc_norm_embedding"},
              std::pair{"hidden_norm", "pre_fc_norm_hidden"}, std::pair{"final_norm", "norm"}}) {
            builder.add(std::string("mtp/") + role, base, std::string("mtp.") + field + ".weight",
                        {h});
        }
        builder.block("mtp/layers/0/", "mtp.layers.0.", base, config, "full_attention");
    }
    if (selected.contains("vision")) { builder.vision(vision_store, vision, h); }
    for (const char* backend : {"dflash", "dflash2"}) {
        if (!selected.contains(backend)) { continue; }
        const auto& draft = model.components[backend]["config"];
        if (std::string_view(backend) == "dflash2" &&
            draft["dflash_config"]["selector_top_k"].get<std::uint64_t>() > model.token_count) {
            throw std::runtime_error("DFlash2 selector_top_k exceeds public token domain");
        }
        builder.draft(*companions.at(backend), OJson(draft), config, backend);
    }
    return model;
}

bool is_qwen3_5_recipe(std::string_view name) noexcept {
    return name == "qwen3_6_27b" || name == "qwen3_6_27b_nvfp4" || name == "qwen3_8_27b" ||
           name == "qwen3_8_27b_nvfp4" || name == "qwen3_6_35b_a3b" ||
           name == "qwen3_6_35b_a3b_nvfp4" || name == "qwen3_5_q8" || name == "qwen3_5_bf16" ||
           name == "qwen3_5_published";
}

void apply_qwen3_5_recipe(std::string_view name, Model& model, Recipe& recipe,
                          SourceInputs& sources) {
    if (name == "qwen3_6_27b") {
        dense_groupwise(model, recipe, kQ6);
    } else if (name == "qwen3_8_27b") {
        dense_groupwise(model, recipe, kQ8);
    } else if (name == "qwen3_6_35b_a3b") {
        qwen3_6_35b_a3b(model, recipe);
    } else if (name == "qwen3_6_27b_nvfp4") {
        qwen3_6_27b_nvfp4(model, recipe, sources);
    } else if (name == "qwen3_8_27b_nvfp4") {
        qwen3_8_27b_nvfp4(model, recipe, sources);
    } else if (name == "qwen3_6_35b_a3b_nvfp4") {
        qwen3_6_35b_a3b_nvfp4(model, recipe, sources);
    } else if (name == "qwen3_5_q8") {
        dense_q8(model, recipe);
    } else if (name == "qwen3_5_bf16") {
        dense_bf16(model, recipe);
    } else if (name == "qwen3_5_published") {
        dense_published(model, recipe, sources.base());
    } else {
        throw std::runtime_error("not an official Qwen3.5 recipe");
    }
}

} // namespace ninferno::convert
