#include "convert/gemma4.h"

#include "convert/checks.h"
#include "convert/dense.h"
#include "convert/formats.h"
#include "convert/numerics.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace ninferno::convert {
namespace {

// ---------------------------------------------------------------------------------------------
// gemma4.py text_config / _rope_group

OJson rope_group(const OJson& raw, const std::string& layer_type, std::uint64_t head_dim,
                 const std::string& label) {
    const auto* group = member(raw, layer_type);
    if (!group || !group->is_object()) {
        throw std::runtime_error(label + ": rope_parameters must describe " + layer_type);
    }
    const auto where       = label + "." + layer_type;
    const auto* type_value = member(*group, "rope_type");
    const OJson rope_type  = type_value ? *type_value : OJson("default");
    if (rope_type != "default" && rope_type != "proportional") {
        throw std::runtime_error(where + ": unsupported rope_type " + rope_type.dump());
    }
    if (f32_value(member(*group, "factor"), 1.0, where + ".factor") != 1.0) {
        throw std::runtime_error(where + ": scaled RoPE is not implemented");
    }
    const double theta = f32(member(*group, "rope_theta"), where + ".rope_theta");
    if (rope_type == "default" && member(*group, "partial_rotary_factor")) {
        throw std::runtime_error(where + ": default RoPE takes no partial factor");
    }
    const double proportion =
        f32_value(member(*group, "partial_rotary_factor"), 1.0, where + ".partial_rotary_factor");
    if (proportion > 1.0) {
        throw std::runtime_error(where + ": partial_rotary_factor exceeds one");
    }
    if (head_dim % 2) { throw std::runtime_error(where + ": head_dim must be even"); }
    const auto pairs =
        static_cast<std::int64_t>(std::floor(proportion * static_cast<double>(head_dim) / 2.0));
    if (pairs <= 0) { throw std::runtime_error(where + ": rotary width is empty"); }
    OJson out                    = OJson::object();
    out["rope_type"]             = rope_type;
    out["rope_theta"]            = theta;
    out["partial_rotary_factor"] = proportion;
    out["rotary_pairs"]          = pairs;
    out["head_dim"]              = head_dim;
    return out;
}

// The text decoder's model_type. Gemma 4 12B ships as the "unified" family (text, vision and
// audio towers in one checkpoint); its text decoder is the Gemma 4 text model under another name,
// so both names read the same configuration.
void text_model_type(const OJson& raw, const std::string& where) {
    const auto* type = member(raw, "model_type");
    if (!type || !type->is_string() ||
        (*type != "gemma4_text" && *type != "gemma4_unified_text")) {
        throw std::runtime_error(where +
                                 ".model_type: expected gemma4_text or gemma4_unified_text, got " +
                                 (type ? type->dump() : "None"));
    }
}

bool text_architecture(const OJson& name) {
    return name == "Gemma4ForConditionalGeneration" || name == "Gemma4ForCausalLM" ||
           name == "Gemma4UnifiedForConditionalGeneration";
}

// An optional count: absent, null or 0 is none.
std::uint64_t optional_count(const OJson& raw, const char* key) {
    const auto* value = member(raw, key);
    if (!present(value) || py_equal(*value, 0)) { return 0; }
    return positive(value, std::string("text.") + key);
}

// transformers' Gemma 4 KV sharing: the last num_kv_shared_layers layers compute no keys or values
// and attend over those of the last earlier layer of their own type; with use_double_wide_mlp
// their MLP is twice intermediate_size. Written as explicit per-layer sources and widths, and
// only when present, so a model without them keeps its config unchanged.
void kv_sharing_and_mlp_widths(const OJson& raw, const OJson& layer_types, OJson& result) {
    const auto layers = layer_types.size();
    const auto shared = optional_count(raw, "num_kv_shared_layers");
    const auto* wide  = member(raw, "use_double_wide_mlp");
    if (present(wide) && !wide->is_boolean()) {
        throw std::runtime_error("text.use_double_wide_mlp must be boolean");
    }
    if (shared == 0) { return; }
    if (shared >= layers) {
        throw std::runtime_error("text.num_kv_shared_layers: at least one layer must compute its "
                                 "own keys and values");
    }
    const auto first = layers - shared;
    OJson sources    = OJson::array();
    for (std::size_t i = 0; i < layers; ++i) {
        if (i < first) {
            sources.push_back(i);
            continue;
        }
        std::optional<std::size_t> source;
        for (std::size_t j = first; j-- > 0;) {
            if (layer_types[j] == layer_types[i]) {
                source = j;
                break;
            }
        }
        if (!source) {
            throw std::runtime_error("text.num_kv_shared_layers: layer " + std::to_string(i) +
                                     " shares keys of a type no earlier layer computes");
        }
        sources.push_back(*source);
    }
    result["kv_source_layers"] = std::move(sources);
    if (present(wide) && wide->get<bool>()) {
        const auto base = result["intermediate_size"].get<std::uint64_t>();
        OJson widths    = OJson::array();
        for (std::size_t i = 0; i < layers; ++i) { widths.push_back(i < first ? base : 2 * base); }
        result["layer_intermediate_sizes"] = std::move(widths);
    }
}

// transformers' Gemma 4 per-layer embeddings (PLE): a [vocab_size_per_layer_input, L*P] table
// scaled by sqrt(P) in its BF16 dtype, a projection of the embedded token scaled by H^-1/2 and
// normed per P-wide slice, their sum scaled by 2^-1/2 (Python floats applied to BF16 tensors,
// so binary32 factors).
void per_layer_embeddings(const OJson& raw, std::uint64_t hidden, OJson& result) {
    const auto width = optional_count(raw, "hidden_size_per_layer_input");
    if (width == 0) { return; }
    result["hidden_size_per_layer_input"] = width;
    result["vocab_size_per_layer_input"] =
        positive(member(raw, "vocab_size_per_layer_input"), "text.vocab_size_per_layer_input");
    if (result["vocab_size_per_layer_input"] != result["vocab_size"]) {
        throw std::runtime_error(
            "text.vocab_size_per_layer_input: a per-layer table narrower or wider than the "
            "vocabulary is not implemented");
    }
    const auto as_f32 = [](double value, const char* name) {
        const OJson json = static_cast<double>(static_cast<float>(value));
        return f32(&json, std::string("text.") + name);
    };
    const auto root = static_cast<float>(std::sqrt(static_cast<double>(width)));
    result["per_layer_embedding_scale"] =
        as_f32(bf16_to_f32(f32_to_bf16(root)), "per_layer_embedding_scale");
    result["per_layer_projection_scale"] =
        as_f32(1.0 / std::sqrt(static_cast<double>(hidden)), "per_layer_projection_scale");
    result["per_layer_input_scale"] = as_f32(1.0 / std::sqrt(2.0), "per_layer_input_scale");
}

OJson text_config(const OJson& source) {
    const auto* architectures = member(source, "architectures");
    if (!architectures || !architectures->is_array() || architectures->size() != 1 ||
        !text_architecture((*architectures)[0])) {
        throw std::runtime_error("unsupported Gemma text architecture " +
                                 (architectures ? architectures->dump() : "None"));
    }
    const auto* nested = member(source, "text_config");
    const OJson& raw   = nested ? *nested : source;
    text_model_type(raw, "text");
    fixed(raw, "hidden_activation", "gelu_pytorch_tanh", "text");
    fixed(raw, "attention_bias", false, "text");
    fixed(raw, "attention_dropout", 0.0, "text");
    if (const auto* bidirectional = member(raw, "use_bidirectional_attention");
        present(bidirectional) && *bidirectional != "vision") {
        throw std::runtime_error("text.use_bidirectional_attention: only 'vision' is implemented");
    }
    OJson result            = OJson::object();
    result["architectures"] = OJson::array({"Gemma4ForCausalLM"});
    result["model_type"]    = "gemma4_text";
    for (const char* key :
         {"hidden_size", "vocab_size", "num_hidden_layers", "max_position_embeddings"}) {
        result[key] = positive(member(raw, key), std::string("text.") + key);
    }
    const OJson* tied = member(raw, "tie_word_embeddings");
    if (!tied) { tied = member(source, "tie_word_embeddings"); }
    const OJson tied_value = tied ? *tied : OJson(false);
    if (!tied_value.is_boolean()) {
        throw std::runtime_error("text.tie_word_embeddings must be boolean");
    }
    result["tie_word_embeddings"] = tied_value;
    result["rms_norm_eps"] = f32_value(member(raw, "rms_norm_eps"), 1e-6, "text.rms_norm_eps");

    const auto layers_count = result["num_hidden_layers"].get<std::uint64_t>();
    const auto* layers      = member(raw, "layer_types");
    if (!layers || !layers->is_array() || layers->size() != layers_count ||
        std::any_of(layers->begin(), layers->end(), [](const OJson& kind) {
            return kind != "sliding_attention" && kind != "full_attention";
        })) {
        throw std::runtime_error("text.layer_types must cover every text block");
    }
    if (layers->back() != "full_attention") {
        throw std::runtime_error("text.layer_types: the last Gemma 4 block must be full_attention");
    }
    result["layer_types"] = *layers;
    const auto heads = positive(member(raw, "num_attention_heads"), "text.num_attention_heads");
    result["num_attention_heads"] = heads;
    const auto* k_eq_v            = member(raw, "attention_k_eq_v");
    if (k_eq_v && !k_eq_v->is_boolean()) {
        throw std::runtime_error("text.attention_k_eq_v must be boolean");
    }
    const bool shares_value    = k_eq_v && k_eq_v->get<bool>();
    result["attention_k_eq_v"] = shares_value;

    std::vector<std::tuple<std::string, std::uint64_t, std::uint64_t>> geometry;
    const auto has = [&](const char* kind) {
        return std::any_of(layers->begin(), layers->end(),
                           [&](const OJson& value) { return value == kind; });
    };
    if (has("sliding_attention")) {
        result["sliding_window"] = positive(member(raw, "sliding_window"), "text.sliding_window");
        geometry.emplace_back(
            "sliding_attention", positive(member(raw, "head_dim"), "text.head_dim"),
            positive(member(raw, "num_key_value_heads"), "text.num_key_value_heads"));
    }
    if (has("full_attention")) {
        geometry.emplace_back("full_attention",
                              positive(member(raw, "global_head_dim"), "text.global_head_dim"),
                              positive(member(raw, shares_value ? "num_global_key_value_heads"
                                                                : "num_key_value_heads"),
                                       "text.num_global_key_value_heads"));
    }
    const auto* rope_source = member(raw, "rope_parameters");
    if (!rope_source || !rope_source->is_object()) {
        throw std::runtime_error("text.rope_parameters must be an object keyed by layer type");
    }
    result["attention_geometry"] = OJson::object();
    result["rope_parameters"]    = OJson::object();
    for (const auto& [kind, head_dim, kv_heads] : geometry) {
        if (heads % kv_heads) {
            throw std::runtime_error("text." + kind +
                                     ": attention query heads must be divisible by KV heads");
        }
        OJson item                         = OJson::object();
        item["head_dim"]                   = head_dim;
        item["num_key_value_heads"]        = kv_heads;
        result["attention_geometry"][kind] = std::move(item);
        result["rope_parameters"][kind] =
            rope_group(*rope_source, kind, head_dim, "text.rope_parameters");
    }
    result["attention_scale"] = 1.0;
    if (const auto* softcap = member(raw, "final_logit_softcapping"); present(softcap)) {
        result["final_logit_softcapping"] = f32(softcap, "text.final_logit_softcapping");
    }
    result["intermediate_size"] =
        positive(member(raw, "intermediate_size"), "text.intermediate_size");
    const auto* moe = member(raw, "enable_moe_block");
    if (moe && !moe->is_boolean()) {
        throw std::runtime_error("text.enable_moe_block must be boolean");
    }
    if (moe && moe->get<bool>()) {
        result["num_experts"] = positive(member(raw, "num_experts"), "text.num_experts");
        result["num_experts_per_tok"] =
            positive(member(raw, "top_k_experts"), "text.top_k_experts");
        result["moe_intermediate_size"] =
            positive(member(raw, "moe_intermediate_size"), "text.moe_intermediate_size");
        if (result["num_experts_per_tok"].get<std::uint64_t>() >
            result["num_experts"].get<std::uint64_t>()) {
            throw std::runtime_error("text.num_experts_per_tok " +
                                     result["num_experts_per_tok"].dump() +
                                     " exceeds text.num_experts " + result["num_experts"].dump());
        }
    } else if (const auto* experts = member(raw, "num_experts");
               present(experts) && !py_equal(*experts, 0) && !py_equal(*experts, false) &&
               !(experts->is_string() && experts->get_ref<const std::string&>().empty()) &&
               !(experts->is_array() && experts->empty()) &&
               !(experts->is_object() && experts->empty())) {
        throw std::runtime_error("text: Dense config carries MoE fields");
    }
    // Gemma's reference normalizer is sqrt(hidden_size) in the embedding's BF16 compute dtype
    // (transformers: embed_scale.to(weight.dtype); 53.0 for H=2816, not 53.066), so the folded
    // factor is the BF16 root, not the binary32 one.
    const auto hidden = result["hidden_size"].get<std::uint64_t>();
    const auto root   = static_cast<float>(std::sqrt(static_cast<double>(hidden)));
    const OJson scale         = static_cast<double>(bf16_to_f32(f32_to_bf16(root)));
    result["embedding_scale"] = f32(&scale, "text.embedding_scale");
    kv_sharing_and_mlp_widths(raw, *layers, result);
    per_layer_embeddings(raw, hidden, result);
    return result;
}

// ---------------------------------------------------------------------------------------------
// gemma4_assistant.py / gemma4_dflash.py configs

OJson assistant_config(const OJson& source) {
    const auto* architectures = member(source, "architectures");
    if (!architectures || (*architectures != OJson::array({"Gemma4AssistantForCausalLM"}) &&
                           *architectures != OJson::array({"Gemma4UnifiedAssistantForCausalLM"}))) {
        throw std::runtime_error("expected a Gemma4AssistantForCausalLM checkpoint");
    }
    if (const auto* ordered = member(source, "use_ordered_embeddings");
        ordered && !py_equal(*ordered, false) && !ordered->is_null()) {
        throw std::runtime_error("assistant: the ordered-embedding head is not implemented");
    }
    const auto* raw_ptr = member(source, "text_config");
    if (!raw_ptr || !raw_ptr->is_object()) {
        throw std::runtime_error("assistant: text_config is missing");
    }
    const auto& raw = *raw_ptr;
    text_model_type(raw, "assistant.text");
    fixed(raw, "hidden_activation", "gelu_pytorch_tanh", "assistant.text");
    fixed(raw, "enable_moe_block", false, "assistant.text");
    fixed(raw, "hidden_size_per_layer_input", 0, "assistant.text");
    const auto layers  = positive(member(raw, "num_hidden_layers"), "assistant.num_hidden_layers");
    const auto* shared = member(raw, "num_kv_shared_layers");
    if (!shared || !py_equal(*shared, layers)) {
        throw std::runtime_error("assistant: every layer must read the target's key/value cache");
    }
    const auto* kinds = member(raw, "layer_types");
    if (!kinds || !kinds->is_array() || kinds->size() != layers) {
        throw std::runtime_error("assistant: layer_types must cover every layer");
    }
    for (const auto& kind : *kinds) {
        if (kind != "sliding_attention" && kind != "full_attention") {
            throw std::runtime_error("assistant: unknown layer type");
        }
    }
    const auto heads =
        positive(member(raw, "num_attention_heads"), "assistant.num_attention_heads");
    const std::pair<std::uint64_t, std::uint64_t> sliding = {
        positive(member(raw, "head_dim"), "assistant.head_dim"),
        positive(member(raw, "num_key_value_heads"), "assistant.num_key_value_heads")};
    const std::pair<std::uint64_t, std::uint64_t> full = {
        positive(member(raw, "global_head_dim"), "assistant.global_head_dim"),
        positive(member(raw, "num_global_key_value_heads"),
                 "assistant.num_global_key_value_heads")};
    OJson result            = OJson::object();
    result["architectures"] = OJson::array({"Gemma4AssistantForCausalLM"});
    result["model_type"]    = "gemma4_assistant";
    result["hidden_size"]   = positive(member(raw, "hidden_size"), "assistant.hidden_size");
    result["backbone_hidden_size"] =
        positive(member(source, "backbone_hidden_size"), "assistant.backbone_hidden_size");
    result["vocab_size"]        = positive(member(raw, "vocab_size"), "assistant.vocab_size");
    result["num_hidden_layers"] = layers;
    result["max_position_embeddings"] =
        positive(member(raw, "max_position_embeddings"), "assistant.max_position_embeddings");
    result["intermediate_size"] =
        positive(member(raw, "intermediate_size"), "assistant.intermediate_size");
    result["num_attention_heads"] = heads;
    result["layer_types"]         = *kinds;
    result["sliding_window"] = positive(member(raw, "sliding_window"), "assistant.sliding_window");
    result["rms_norm_eps"] = f32_value(member(raw, "rms_norm_eps"), 1e-6, "assistant.rms_norm_eps");
    result["attention_scale"]     = 1.0;
    result["tie_word_embeddings"] = true;
    result["attention_geometry"]  = OJson::object();
    result["rope_parameters"]     = OJson::object();
    const auto* rope              = member(raw, "rope_parameters");
    const OJson empty             = OJson::object();
    for (const auto& [kind, geometry] : {std::pair{std::string("sliding_attention"), sliding},
                                         std::pair{std::string("full_attention"), full}}) {
        if (heads % geometry.second) {
            throw std::runtime_error("assistant." + kind +
                                     ": query heads must divide by the target's KV heads");
        }
        OJson item                         = OJson::object();
        item["head_dim"]                   = geometry.first;
        item["num_key_value_heads"]        = geometry.second;
        result["attention_geometry"][kind] = std::move(item);
        result["rope_parameters"][kind] =
            rope_group(rope ? *rope : empty, kind, geometry.first, "assistant.rope_parameters");
    }
    return result;
}

OJson dflash_config(const OJson& source) {
    const auto* architectures = member(source, "architectures");
    if (!architectures || *architectures != OJson::array({"DFlashDraftModel"})) {
        throw std::runtime_error("expected a DFlashDraftModel checkpoint");
    }
    fixed(source, "model_type", "qwen3", "dflash");
    fixed(source, "hidden_act", "silu", "dflash");
    fixed(source, "attention_bias", false, "dflash");
    const auto* softcap = member(source, "final_logit_softcapping");
    if (!present(softcap)) {
        throw std::runtime_error(
            "dflash: final_logit_softcapping is missing; not a Gemma 4 drafter");
    }
    const auto* draft_ptr = member(source, "dflash_config");
    const OJson draft     = draft_ptr ? *draft_ptr : OJson::object();
    for (const char* key : {"sample_from_anchor", "input_embedding_scale", "output_multiplier"}) {
        if (member(source, key) || member(draft, key)) {
            throw std::runtime_error(std::string("dflash: ") + key +
                                     " is a DFlash2 field this adapter does not implement");
        }
    }
    const auto layers = positive(member(source, "num_hidden_layers"), "dflash.num_hidden_layers");
    const auto* kinds = member(source, "layer_types");
    if (!kinds || !kinds->is_array() || kinds->size() != layers) {
        throw std::runtime_error("dflash: layer_types must cover every layer");
    }
    for (const auto& kind : *kinds) {
        if (kind != "sliding_attention" && kind != "full_attention") {
            throw std::runtime_error("dflash: unknown layer type");
        }
    }
    if (const auto* scaling = member(source, "rope_scaling");
        present(scaling) && !(scaling->is_object() && scaling->empty())) {
        throw std::runtime_error("dflash: rope_scaling is not implemented");
    }
    const auto heads =
        positive(member(source, "num_attention_heads"), "dflash.num_attention_heads");
    const auto kv_heads =
        positive(member(source, "num_key_value_heads"), "dflash.num_key_value_heads");
    if (heads % kv_heads) {
        throw std::runtime_error("dflash: query heads must divide by key/value heads");
    }
    const auto* taps = member(draft, "target_layer_ids");
    const auto target_layers =
        positive(member(source, "num_target_layers"), "dflash.num_target_layers");
    std::set<std::int64_t> seen;
    bool valid = taps && taps->is_array() && !taps->empty();
    if (valid) {
        for (const auto& tap : *taps) {
            if (!tap.is_number_integer() || tap.get<std::int64_t>() < 0 ||
                tap.get<std::uint64_t>() >= target_layers ||
                !seen.insert(tap.get<std::int64_t>()).second) {
                valid = false;
            }
        }
    }
    if (!valid) { throw std::runtime_error("dflash: invalid target_layer_ids"); }
    const auto vocab = positive(member(source, "vocab_size"), "dflash.vocab_size");
    const auto* mask = member(draft, "mask_token_id");
    if (!mask || !mask->is_number_integer() || mask->get<std::int64_t>() < 0 ||
        mask->get<std::uint64_t>() >= vocab) {
        throw std::runtime_error("dflash: mask token outside the vocabulary");
    }
    OJson result                = OJson::object();
    result["architectures"]     = OJson::array({"DFlashDraftModel"});
    result["model_type"]        = "gemma4_dflash";
    result["hidden_size"]       = positive(member(source, "hidden_size"), "dflash.hidden_size");
    result["vocab_size"]        = vocab;
    result["num_hidden_layers"] = layers;
    result["num_target_layers"] = target_layers;
    result["intermediate_size"] =
        positive(member(source, "intermediate_size"), "dflash.intermediate_size");
    result["num_attention_heads"] = heads;
    result["num_key_value_heads"] = kv_heads;
    result["head_dim"]            = positive(member(source, "head_dim"), "dflash.head_dim");
    result["layer_types"]         = *kinds;
    result["sliding_window"] = positive(member(source, "sliding_window"), "dflash.sliding_window");
    // Newer configs carry the rotary base under rope_parameters (plain RoPE only).
    const OJson* theta = member(source, "rope_theta");
    if (!theta || theta->is_null()) {
        if (const auto* rope = member(source, "rope_parameters"); rope && rope->is_object()) {
            if (const auto* type = member(*rope, "rope_type"); type && *type != "default") {
                throw std::runtime_error("dflash.rope_parameters.rope_type must be default");
            }
            theta = member(*rope, "rope_theta");
        }
    }
    result["rope_theta"] = f32(theta, "dflash.rope_theta");
    result["rms_norm_eps"] = f32_value(member(source, "rms_norm_eps"), 1e-6, "dflash.rms_norm_eps");
    result["block_size"]   = positive(member(source, "block_size"), "dflash.block_size");
    result["final_logit_softcapping"] = f32(softcap, "dflash.final_logit_softcapping");
    OJson dflash                      = OJson::object();
    dflash["target_layer_ids"]        = *taps;
    dflash["mask_token_id"]           = *mask;
    result["dflash_config"]           = std::move(dflash);
    return result;
}

// The DFlash drafters whose block attends bidirectionally on every layer: Meta's Muse Glimmer
// assistant (transformers' MuseGlimmerAssistantModel: its mask is bidirectional, sliding on the
// sliding layers), a GGUF DFlash drafter read back (model_type "dflash": llama.cpp's graph runs
// every layer non-causal) and z-lab's Qwen3-style drafters without a soft cap and without sliding
// layers (gpt-oss's: dflash.py attends the block with is_causal=False and no mask). The Muse and
// GGUF configs name no target depth; the Muse assistant's names no vocabulary either (it owns no
// table, and the target's is padded past the tokenizer): the artifact then records none, its mask
// token is checked against the tokenizer, and the loader pairs both with the target.
bool block_dflash(const OJson& source) {
    const auto* architectures = member(source, "architectures");
    if (!architectures) { return false; }
    if (*architectures == OJson::array({"MuseGlimmerAssistantModel"})) { return true; }
    const auto* type = member(source, "model_type");
    if (*architectures != OJson::array({"DFlashDraftModel"}) || !type) { return false; }
    if (*type == "dflash") { return true; }
    // z-lab's: a Gemma 4 drafter carries the target's soft cap (dflash_config below).
    if (*type != "qwen3" || present(member(source, "final_logit_softcapping"))) { return false; }
    const auto* kinds = member(source, "layer_types");
    if (!kinds || !kinds->is_array()) { return false; }
    return std::all_of(kinds->begin(), kinds->end(),
                       [](const OJson& kind) { return kind == "full_attention"; });
}

OJson block_dflash_config(const OJson& source, std::optional<std::uint64_t> tokenizer_vocab) {
    const bool muse = *member(source, "architectures") ==
                      OJson::array({"MuseGlimmerAssistantModel"});
    const std::string label = muse ? "muse assistant" : "dflash";
    fixed(source, "hidden_act", "silu", label);
    // q/k/v/o biases (gpt-oss's drafter); the others have none.
    const auto* bias_value = member(source, "attention_bias");
    const bool bias        = present(bias_value) && bias_value->is_boolean() && bias_value->get<bool>();
    const auto layers = positive(member(source, "num_hidden_layers"), label + ".num_hidden_layers");
    const auto* kinds = member(source, "layer_types");
    if (!kinds || !kinds->is_array() || kinds->size() != layers) {
        throw std::runtime_error(label + ": layer_types must cover every layer");
    }
    bool sliding = false;
    for (const auto& kind : *kinds) {
        if (kind != "sliding_attention" && kind != "full_attention") {
            throw std::runtime_error(label + ": unknown layer type");
        }
        sliding = sliding || kind == "sliding_attention";
    }
    const auto heads = positive(member(source, "num_attention_heads"),
                                label + ".num_attention_heads");
    const auto kv_heads = positive(member(source, "num_key_value_heads"),
                                   label + ".num_key_value_heads");
    if (heads % kv_heads) {
        throw std::runtime_error(label + ": query heads must divide by key/value heads");
    }
    // Muse's config carries the taps and the mask token at the top level; a GGUF's under
    // dflash_config, as z-lab's.
    const auto* draft_ptr = member(source, "dflash_config");
    const OJson draft     = draft_ptr ? *draft_ptr : OJson::object();
    const OJson& fields   = muse ? source : draft;
    const auto* taps      = member(fields, "target_layer_ids");
    const auto* depth     = member(source, "num_target_layers");
    const std::uint64_t target_layers =
        present(depth) ? positive(depth, label + ".num_target_layers") : UINT32_MAX;
    std::set<std::int64_t> seen;
    bool valid = taps && taps->is_array() && !taps->empty();
    if (valid) {
        for (const auto& tap : *taps) {
            if (!tap.is_number_integer() || tap.get<std::int64_t>() < 0 ||
                tap.get<std::uint64_t>() >= target_layers ||
                !seen.insert(tap.get<std::int64_t>()).second) {
                valid = false;
            }
        }
    }
    if (!valid) { throw std::runtime_error(label + ": invalid target_layer_ids"); }
    const auto* vocab_value = member(source, "vocab_size");
    std::uint64_t vocab     = 0;
    if (present(vocab_value)) {
        vocab = positive(vocab_value, label + ".vocab_size");
    } else if (tokenizer_vocab) {
        vocab = *tokenizer_vocab;
    } else {
        throw std::runtime_error(label + ": no vocabulary: the config names none and no "
                                         "tokenizer.json was given (--resource)");
    }
    const auto* mask = member(fields, "mask_token_id");
    if (!mask || !mask->is_number_integer() || mask->get<std::int64_t>() < 0 ||
        mask->get<std::uint64_t>() >= vocab) {
        throw std::runtime_error(label + ": mask token outside the vocabulary");
    }
    OJson result                = OJson::object();
    result["architectures"]     = *member(source, "architectures");
    result["model_type"]        = "dflash";
    result["hidden_size"]       = positive(member(source, "hidden_size"), label + ".hidden_size");
    if (present(vocab_value)) { result["vocab_size"] = vocab; }
    result["num_hidden_layers"] = layers;
    if (present(depth)) { result["num_target_layers"] = target_layers; }
    result["intermediate_size"] =
        positive(member(source, "intermediate_size"), label + ".intermediate_size");
    result["num_attention_heads"] = heads;
    result["num_key_value_heads"] = kv_heads;
    result["head_dim"]            = positive(member(source, "head_dim"), label + ".head_dim");
    result["layer_types"]         = *kinds;
    // Only a drafter with sliding layers records a window (gpt-oss's has none: null).
    if (sliding || present(member(source, "sliding_window"))) {
        result["sliding_window"] =
            positive(member(source, "sliding_window"), label + ".sliding_window");
    }
    if (bias) { result["attention_bias"] = true; }
    const OJson* theta = member(source, "rope_theta");
    if (!theta || theta->is_null()) {
        if (const auto* rope = member(source, "rope_parameters"); rope && rope->is_object()) {
            if (const auto* type = member(*rope, "rope_type"); type && *type != "default") {
                throw std::runtime_error(label + ".rope_parameters.rope_type must be default");
            }
            theta = member(*rope, "rope_theta");
        }
    }
    result["rope_theta"]   = f32(theta, label + ".rope_theta");
    // A scaled rope (gpt-oss's drafter: YaRN, as its target) is resolved into the frequency table
    // over the whole head, as transformers computes it.
    if (const auto* scaling = member(source, "rope_scaling");
        present(scaling) && !(scaling->is_object() && scaling->empty())) {
        const auto* limit = member(source, "max_position_embeddings");
        result["rope"]    = resolved_rope(source, result["head_dim"].get<std::uint64_t>(),
                                          present(limit) ? limit->get<std::uint64_t>() : 0,
                                          label + ".rope_scaling");
    }
    result["rms_norm_eps"] = f32(member(source, "rms_norm_eps"), label + ".rms_norm_eps");
    result["block_size"]   = positive(member(source, "block_size"), label + ".block_size");
    // The positions the drafter was trained to address, when its config records them (a serving
    // Engine bounds its context by them).
    if (const auto* limit = member(source, "max_position_embeddings"); present(limit)) {
        result["max_position_embeddings"] =
            positive(limit, label + ".max_position_embeddings");
    }
    OJson dflash                     = OJson::object();
    dflash["target_layer_ids"]       = *taps;
    dflash["mask_token_id"]          = *mask;
    dflash["sliding_attention_mask"] = "block";
    result["dflash_config"]          = std::move(dflash);
    return result;
}

// The largest token id tokenizer.json names (vocabulary and added tokens), plus one.
std::optional<std::uint64_t> tokenizer_vocabulary(const Checkpoint& base,
                                                  const ResourceOverrides& overrides) {
    std::string data;
    for (const auto& [role, path] : overrides) {
        if (role == "tokenizer.json") { data = read_file(path); }
    }
    if (data.empty()) {
        const auto path = std::filesystem::path(base.root()) / "tokenizer.json";
        if (std::filesystem::exists(path)) {
            data = read_file(path);
        } else {
            const auto fallbacks = resource_fallbacks(base);
            const auto it        = fallbacks.find("tokenizer.json");
            if (it == fallbacks.end()) { return std::nullopt; }
            data = it->second;
        }
    }
    const auto tokenizer = Json::parse(data);
    std::uint64_t top    = 0;
    if (const auto* model = member(tokenizer, "model"); model && model->contains("vocab")) {
        for (const auto& [text, id] : model->at("vocab").items()) {
            top = std::max<std::uint64_t>(top, id.get<std::uint64_t>() + 1);
        }
    }
    if (const auto* added = member(tokenizer, "added_tokens"); added && added->is_array()) {
        for (const auto& token : *added) {
            top = std::max<std::uint64_t>(top, token.at("id").get<std::uint64_t>() + 1);
        }
    }
    return top ? std::optional<std::uint64_t>(top) : std::nullopt;
}

// ---------------------------------------------------------------------------------------------
// gemma4.py _Builder: one method per source kind the adapters use.

// ---------------------------------------------------------------------------------------------
// Vision: Gemma4ForConditionalGeneration's vision tower (31B, 26B-A4B), modeling_gemma4.py
// Gemma4VisionModel and Gemma4MultimodalEmbedder.

OJson vision_config(const OJson& source, std::uint64_t text_hidden) {
    const auto* raw_ptr = member(source, "vision_config");
    if (!raw_ptr || !raw_ptr->is_object()) { throw std::runtime_error("Vision requires vision_config"); }
    const auto& raw = *raw_ptr;
    fixed(raw, "model_type", "gemma4_vision", "vision");
    fixed(raw, "hidden_activation", "gelu_pytorch_tanh", "vision");
    fixed(raw, "attention_bias", false, "vision");
    fixed(raw, "standardize", true, "vision");
    fixed(raw, "use_clipped_linears", false, "vision");
    OJson result         = OJson::object();
    result["model_type"] = "gemma4_vision";
    for (const char* key : {"hidden_size", "num_hidden_layers", "num_attention_heads", "head_dim",
                            "intermediate_size", "patch_size", "pooling_kernel_size",
                            "position_embedding_size", "default_output_length"}) {
        result[key] = positive(member(raw, key), std::string("vision.") + key);
    }
    if (positive(member(raw, "num_key_value_heads"), "vision.num_key_value_heads") !=
        result["num_attention_heads"].get<std::uint64_t>()) {
        throw std::runtime_error("vision: grouped key/value heads are not supported");
    }
    const auto* rope = member(raw, "rope_parameters");
    result["rope_theta"] = rope ? f32(member(*rope, "rope_theta"), "vision.rope_theta") : 100.0;
    result["rms_norm_eps"] = f32(member(raw, "rms_norm_eps"), "vision.rms_norm_eps");
    const auto h     = result["hidden_size"].get<std::uint64_t>();
    const auto heads = result["num_attention_heads"].get<std::uint64_t>();
    const auto d     = result["head_dim"].get<std::uint64_t>();
    // Axial 2-D rope: each head's two halves rotate by x and by y, in pairs within a half.
    if (heads * d != h || d % 4 != 0) {
        throw std::runtime_error("vision: head geometry is not an axial 2-D rope's");
    }
    result["output_hidden_size"] = text_hidden;
    // The prompt's image tokens: <|image> boi, the soft token, <image|> eoi (config.json's).
    for (const char* key : {"boi_token_id", "eoi_token_id", "image_token_id"}) {
        result[key] = positive(member(source, key), key);
    }
    return result;
}

// Within each head, rows [d/4, d/2) and [d/2, 3d/4) swap: transformers rotates the pairs
// (i, i + d/4) of each half (x half, then y half), the engine's Vision 2-D rope the pairs
// (i, i + d/2) with the first d/4 of them by x and the rest by y. Applied alike to q and k (and
// their norm weights), so scores and RMS statistics are unchanged.
std::vector<std::pair<std::uint64_t, std::uint64_t>> axial_rope_rows(std::uint64_t heads,
                                                                      std::uint64_t d) {
    const auto q = d / 4;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> spans;
    for (std::uint64_t h = 0; h < heads; ++h) {
        const auto base = h * d;
        for (const auto first : {0 * q, 2 * q, 1 * q, 3 * q}) {
            spans.emplace_back(base + first, base + first + q);
        }
    }
    return spans;
}

class Builder {
public:
    Builder(Model& model, const Checkpoint& base) : model_(model), base_(base) {}

    // A 2-D projection or table read through matrix_source.
    void matrix(const std::string& name, const std::string& source_name, Shape shape,
                std::vector<std::string> inputs, std::string probe_leaf = {}) {
        auto factory = [source_name, shape](const Checkpoint& base,
                                            MatrixFormat format) -> SourcePtr {
            return matrix_source(base, source_name, shape, format);
        };
        add(name, shape, factory(base_, MatrixFormat::Auto), factory, std::move(inputs),
            QType::BF16, std::move(probe_leaf));
    }

    // A plain matrix read from `store` with its rows reordered by `spans` (an axial rope's
    // half-head swap).
    void permuted(const std::string& name, const Checkpoint& store, const std::string& source_name,
                  Shape shape, std::vector<std::pair<std::uint64_t, std::uint64_t>> spans,
                  std::vector<std::string> inputs) {
        auto factory = [&store, source_name, shape, spans](const Checkpoint&,
                                                          MatrixFormat) -> SourcePtr {
            return select_rows(tensor_source(store, source_name, shape), spans);
        };
        add(name, shape, factory(store, MatrixFormat::Auto), factory, std::move(inputs),
            QType::BF16, {});
    }

    // A plain matrix from `store` (a companion such as a vision projector file).
    void matrix_from(const std::string& name, const Checkpoint& store,
                     const std::string& source_name, Shape shape, std::vector<std::string> inputs) {
        auto factory = [&store, source_name, shape](const Checkpoint&,
                                                   MatrixFormat format) -> SourcePtr {
            return matrix_source(store, source_name, shape, format);
        };
        add(name, shape, factory(store, MatrixFormat::Auto), factory, std::move(inputs),
            QType::BF16, {});
    }

    // A vector from `store`, optionally through a permutation of its elements; `stored` is the
    // checkpoint's own shape when it differs from `shape` (same elements, row-major).
    void vector_from(const std::string& name, const Checkpoint& store,
                     const std::string& source_name, Shape shape,
                     std::vector<std::pair<std::uint64_t, std::uint64_t>> spans = {},
                     Shape stored = {}) {
        auto factory = [name](const Checkpoint&, MatrixFormat) -> SourcePtr {
            throw std::runtime_error(name + ": provide an explicit encoded source mapping");
        };
        if (stored.empty()) { stored = shape; }
        SourcePtr source = tensor_source(store, source_name, shape, 0, stored);
        if (!spans.empty()) {
            // Rows of an [n, 1] view, read back as the vector the parameter declares.
            source = transpose_source(
                select_rows(tensor_source(store, source_name, {shape[0], 1}, 0, stored), spans),
                {0, 1}, shape);
        }
        add(name, shape, std::move(source), factory, {}, QType::BF16, {});
    }

    // Gate and up of a GeGLU MLP stacked by rows (gate first), from `store`.
    void stacked_from(const std::string& name, const Checkpoint& store,
                      std::vector<std::string> members, Shape shape,
                      std::vector<std::string> inputs) {
        auto factory = [&store, name, members, shape](const Checkpoint&,
                                                     MatrixFormat format) -> SourcePtr {
            const auto rows_each = shape[0] / members.size();
            std::vector<SourcePtr> parts;
            for (const auto& part : members) {
                parts.push_back(matrix_source(store, part, {rows_each, shape[1]}, format));
            }
            return concat_source(std::move(parts), shape, name);
        };
        add(name, shape, factory(store, MatrixFormat::Auto), factory, std::move(inputs),
            QType::BF16, {});
    }

    // A vector or scalar read through tensor_source; it has no encoded representation.
    void vector(const std::string& name, const std::string& source_name, Shape shape,
                QType direct = QType::BF16) {
        auto factory = [name](const Checkpoint&, MatrixFormat) -> SourcePtr {
            throw std::runtime_error(name + ": provide an explicit encoded source mapping");
        };
        add(name, shape, tensor_source(base_, source_name, shape), factory, {}, direct, {});
    }

    // The token embedding, scaled by a constant at conversion time.
    void scaled(const std::string& name, const std::string& source_name, Shape shape,
                float factor) {
        auto factory = [name](const Checkpoint&, MatrixFormat) -> SourcePtr {
            throw std::runtime_error(name + ": scaled source requires value access");
        };
        add(name, shape, scaled_source(tensor_source(base_, source_name, shape), factor), factory,
            {}, QType::BF16, {});
    }

    // The token embedding: the scaled values by default, the unscaled matrix as its encoded
    // source (a table kept as published, scaled at run time).
    void table(const std::string& name, const std::string& source_name, Shape shape,
               float factor) {
        auto factory = [source_name, shape](const Checkpoint& base,
                                            MatrixFormat format) -> SourcePtr {
            return matrix_source(base, source_name, shape, format);
        };
        add(name, shape, scaled_source(tensor_source(base_, source_name, shape), factor), factory,
            {}, QType::BF16, {});
    }

    // Gemma 4 expert gate and up as one logical matrix (stack= in gemma4.py).
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
            // Row-scaled FP8 halves stack without a shared value; NVFP4 halves share a divisor.
            if (format == MatrixFormat::Fp8Row || (format == MatrixFormat::Auto && !encoded)) {
                return concat_source(std::move(parts), shape, name);
            }
            return stacked_source(std::move(parts), name);
        };
        add(name, shape, factory(base_, MatrixFormat::Auto), factory, std::move(inputs),
            QType::BF16, std::move(probe_leaf));
    }

    // An expert's down projection compensating its gate_up plane's adopted divisor.
    void compensated(const std::string& name, const std::string& source_name, Shape shape,
                     std::string gate, std::string up, Shape pair_shape,
                     std::vector<std::string> inputs, std::string probe_leaf) {
        auto factory = [source_name, shape, gate, up,
                        pair_shape](const Checkpoint& base, MatrixFormat format) -> SourcePtr {
            auto ref         = matrix_source(base, source_name, shape, format);
            const auto strip = [](std::string value) {
                if (value.ends_with(".weight")) { value.resize(value.size() - 7); }
                return value;
            };
            const auto nvfp4 = [&](const std::string& matrix, const Shape& dims) {
                const auto kept = kept_encoding(base, strip(matrix), dims[0], dims[1]);
                return kept && kept->format == QType::NVFP4;
            };
            const bool encoded =
                nvfp4(source_name, shape) && nvfp4(gate, pair_shape) && nvfp4(up, pair_shape);
            if (!encoded) { return ref; }
            return divisor_scaled_source(ref,
                                         matrix_source(base, gate, pair_shape, MatrixFormat::Nvfp4),
                                         matrix_source(base, up, pair_shape, MatrixFormat::Nvfp4));
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

std::string layer_prefix(std::size_t layer) { return "text/layers/" + std::to_string(layer) + "/"; }

void text_block(Builder& builder, Model& model, const OJson& config, const std::string& root,
                std::size_t layer, const std::string& layer_type) {
    const auto h    = config["hidden_size"].get<std::uint64_t>();
    const auto p    = layer_prefix(layer);
    const auto leaf = "layers." + std::to_string(layer) + ".";
    const auto s    = root + leaf;
    const bool moe  = config.contains("num_experts");
    builder.vector(p + "input_norm", s + "input_layernorm.weight", {h});

    // attention
    const auto& geometry = config["attention_geometry"][layer_type];
    const auto d         = geometry["head_dim"].get<std::uint64_t>();
    const auto kv        = geometry["num_key_value_heads"].get<std::uint64_t>();
    const auto q         = config["num_attention_heads"].get<std::uint64_t>() * d;
    const bool shares_value =
        config["attention_k_eq_v"].get<bool>() && layer_type == "full_attention";
    // A layer that reads another layer's keys and values (KV sharing) has no key or value
    // projection; the checkpoint's unused copies are not converted.
    const bool owns_kv = !config.contains("kv_source_layers") ||
                         config["kv_source_layers"][layer].get<std::uint64_t>() == layer;
    std::vector<std::tuple<std::string, std::string, std::uint64_t>> roles = {
        {"query", "q_proj", q}};
    if (owns_kv) { roles.emplace_back("key", "k_proj", kv * d); }
    if (owns_kv && !shares_value) { roles.emplace_back("value", "v_proj", kv * d); }
    std::vector<std::string> packed;
    for (const auto& [role, field, rows] : roles) {
        builder.matrix(p + "attention/" + role, s + "self_attn." + field + ".weight", {rows, h},
                       {p + "mixer_input"}, leaf + "self_attn." + field);
        packed.push_back(p + "attention/" + role);
    }
    builder.vector(p + "attention/query_norm", s + "self_attn.q_norm.weight", {d});
    if (owns_kv) { builder.vector(p + "attention/key_norm", s + "self_attn.k_norm.weight", {d}); }
    builder.matrix(p + "attention/output", s + "self_attn.o_proj.weight", {h, q},
                   {p + "attention/context"}, leaf + "self_attn.o_proj");
    if (packed.size() > 1) { builder.group(packed); }
    if (packed.size() == 3) {
        builder.group({packed[0], packed[1]});
        builder.group({packed[1], packed[2]});
    }

    builder.vector(p + "post_attention_norm", s + "post_attention_layernorm.weight", {h});
    builder.vector(p + "pre_ffn_norm", s + "pre_feedforward_layernorm.weight", {h});

    // dense MLP
    const auto intermediate = config.contains("layer_intermediate_sizes")
                                  ? config["layer_intermediate_sizes"][layer].get<std::uint64_t>()
                                  : config["intermediate_size"].get<std::uint64_t>();
    for (const char* role : {"gate", "up"}) {
        builder.matrix(p + "mlp/" + role, s + "mlp." + role + "_proj.weight", {intermediate, h},
                       {p + "ffn_input"}, leaf + "mlp." + role + "_proj");
    }
    builder.matrix(p + "mlp/down", s + "mlp.down_proj.weight", {h, intermediate},
                   {p + "mlp/product"}, leaf + "mlp.down_proj");
    builder.group({p + "mlp/gate", p + "mlp/up"});

    builder.vector(p + "post_ffn_norm", s + "post_feedforward_layernorm.weight", {h});
    if (moe) {
        builder.vector(p + "pre_moe_norm", s + "pre_feedforward_layernorm_2.weight", {h});
        builder.vector(p + "post_mlp_norm", s + "post_feedforward_layernorm_1.weight", {h});
        builder.vector(p + "post_moe_norm", s + "post_feedforward_layernorm_2.weight", {h});
        const auto experts = config["num_experts"].get<std::uint64_t>();
        const auto ir      = config["moe_intermediate_size"].get<std::uint64_t>();
        const auto m       = p + "moe/";
        builder.matrix(m + "router", s + "router.proj.weight", {experts, h}, {m + "router_input"},
                       leaf + "router.proj");
        builder.vector(m + "router_scale", s + "router.scale", {h}, QType::FP32);
        builder.vector(m + "expert_scale", s + "router.per_expert_scale", {experts}, QType::FP32);
        std::vector<std::string> gate_up;
        std::vector<std::string> downs;
        for (std::uint64_t e = 0; e < experts; ++e) {
            const auto ep    = m + "experts/" + std::to_string(e) + "/";
            const auto es    = s + "experts." + std::to_string(e) + ".";
            const auto eleaf = leaf + "experts." + std::to_string(e) + ".";
            builder.stacked(ep + "gate_up", {es + "gate_proj.weight", es + "up_proj.weight"},
                            {2 * ir, h}, {m + "expert_input"}, eleaf + "gate_proj");
            model.parameters.back().probe_shape = {ir, h};
            gate_up.push_back(ep + "gate_up");
            builder.compensated(ep + "down", es + "down_proj.weight", {h, ir},
                                es + "gate_proj.weight", es + "up_proj.weight", {ir, h},
                                {ep + "product"}, eleaf + "down_proj");
            downs.push_back(ep + "down");
        }
        model.expert_banks.push_back(builder.group(gate_up));
        model.expert_banks.push_back(builder.group(downs));
    }
    // Per-layer embedding block: gate [P, H] over the stream after the feed-forward update,
    // output [H, P] of the gated product, and its post norm.
    if (config.contains("hidden_size_per_layer_input")) {
        const auto width = config["hidden_size_per_layer_input"].get<std::uint64_t>();
        builder.matrix(p + "ple/gate", s + "per_layer_input_gate.weight", {width, h},
                       {p + "ple_gate_input"}, leaf + "per_layer_input_gate");
        builder.matrix(p + "ple/output", s + "per_layer_projection.weight", {h, width},
                       {p + "ple/product"}, leaf + "per_layer_projection");
        builder.vector(p + "post_ple_norm", s + "post_per_layer_input_norm.weight", {h});
    }
    builder.vector(p + "output_scale", s + "layer_scalar", {1}, QType::FP32);
}

void vision_tower(Builder& builder, const Checkpoint& store, const OJson& v) {
    const auto h      = v["hidden_size"].get<std::uint64_t>();
    const auto heads  = v["num_attention_heads"].get<std::uint64_t>();
    const auto d      = v["head_dim"].get<std::uint64_t>();
    const auto ff     = v["intermediate_size"].get<std::uint64_t>();
    const auto ps     = v["patch_size"].get<std::uint64_t>();
    const auto table  = v["position_embedding_size"].get<std::uint64_t>();
    const auto out    = v["output_hidden_size"].get<std::uint64_t>();
    const auto rows   = axial_rope_rows(heads, d);
    const auto in_head = axial_rope_rows(1, d);
    const std::string t = "model.vision_tower.";
    builder.matrix_from("vision/patch_embedding", store, t + "patch_embedder.input_proj.weight",
                        {h, 3 * ps * ps}, {"vision/patch_input"});
    // [2, table, h]: the x table, then the y table.
    builder.vector_from("vision/position_table", store,
                        t + "patch_embedder.position_embedding_table", {2 * table, h}, {},
                        {2, table, h});
    builder.vector_from("vision/std_bias", store, t + "std_bias", {h});
    builder.vector_from("vision/std_scale", store, t + "std_scale", {h});
    for (std::uint64_t i = 0; i < v["num_hidden_layers"].get<std::uint64_t>(); ++i) {
        const auto p = "vision/layers/" + std::to_string(i) + "/";
        const auto s = t + "encoder.layers." + std::to_string(i) + ".";
        for (const auto& [role, field] :
             {std::pair{"input_norm", "input_layernorm"},
              std::pair{"post_attention_norm", "post_attention_layernorm"},
              std::pair{"pre_feedforward_norm", "pre_feedforward_layernorm"},
              std::pair{"post_feedforward_norm", "post_feedforward_layernorm"}}) {
            builder.vector_from(p + role, store, s + field + ".weight", {h});
        }
        builder.permuted(p + "attention/query", store, s + "self_attn.q_proj.linear.weight",
                         {h, h}, rows, {p + "attention_input"});
        builder.permuted(p + "attention/key", store, s + "self_attn.k_proj.linear.weight", {h, h},
                         rows, {p + "attention_input"});
        builder.matrix_from(p + "attention/value", store, s + "self_attn.v_proj.linear.weight",
                            {h, h}, {p + "attention_input"});
        builder.group({p + "attention/query", p + "attention/key", p + "attention/value"});
        builder.vector_from(p + "attention/query_norm", store, s + "self_attn.q_norm.weight", {d},
                            in_head);
        builder.vector_from(p + "attention/key_norm", store, s + "self_attn.k_norm.weight", {d},
                            in_head);
        builder.matrix_from(p + "attention/output", store, s + "self_attn.o_proj.linear.weight",
                            {h, h}, {p + "attention_output"});
        builder.stacked_from(p + "mlp/gate_up", store,
                             {s + "mlp.gate_proj.linear.weight", s + "mlp.up_proj.linear.weight"},
                             {2 * ff, h}, {p + "mlp_input"});
        builder.matrix_from(p + "mlp/down", store, s + "mlp.down_proj.linear.weight", {h, ff},
                            {p + "mlp_activation"});
    }
    builder.matrix_from("vision/projection", store, "model.embed_vision.embedding_projection.weight",
                        {out, h}, {"vision/projection_input"});
}

Model build_text(const Checkpoint& base, const ResourceOverrides& overrides,
                 const Checkpoint* vision_store) {
    Model model;
    model.components["text"]["config"] = text_config(base.config());
    if (vision_store != nullptr) {
        const auto text_hidden = model.config()["hidden_size"].get<std::uint64_t>();
        // A projector file carries no token ids: the checkpoint's config (or its GGUF's) does.
        OJson source = vision_store->config();
        for (const char* key : {"boi_token_id", "eoi_token_id", "image_token_id"}) {
            if (!member(source, key) && member(base.config(), key)) {
                source[key] = base.config()[key];
            }
        }
        model.components["vision"] =
            OJson{{"config", vision_config(source, text_hidden)}, {"target", "text"}};
    }
    load_resources(model, base.root(), model.config()["vocab_size"].get<std::uint64_t>(),
                   overrides, nullptr, resource_fallbacks(base));
    const auto& config = model.config(); // after load_resources: adding resources moves the config
    Builder builder(model, base);
    const auto h    = config["hidden_size"].get<std::uint64_t>();
    const auto r    = config["vocab_size"].get<std::uint64_t>();
    const auto root = base.config().contains("text_config") ? std::string("model.language_model.")
                                                            : std::string("model.");
    // A plain table has the scale folded in at conversion time (exactly transformers' BF16
    // product); the tied head reads the same bytes unscaled. A table the publisher quantized
    // (GGUF Q8_0, Q4_0, K-quants) is kept as published and its rows are scaled at run time:
    // folding would quantize it a second time.
    // Which of the two is the recipe's: gemma4_published imports a quantized table (the encoded
    // source is the unscaled matrix), the other recipes quantize the folded values.
    builder.table("text/token_embedding", root + "embed_tokens.weight", {r, h},
                  static_cast<float>(config["embedding_scale"].get<double>()));
    builder.matrix("text/output_head",
                   config["tie_word_embeddings"].get<bool>() ? root + "embed_tokens.weight"
                                                             : std::string("lm_head.weight"),
                   {r, h}, {"text/final_hidden"});
    builder.vector("text/final_norm", root + "norm.weight", {h});
    const auto& layers = config["layer_types"];
    // Per-layer embeddings: the table as published (its sqrt(P) scale is applied at run time),
    // the projection of the embedded token, and the norm of each of its P-wide slices.
    if (config.contains("hidden_size_per_layer_input")) {
        const auto width = config["hidden_size_per_layer_input"].get<std::uint64_t>();
        const auto rows  = width * layers.size();
        // Read as a matrix without inputs, so a published encoding (GGUF Q8_0) is kept.
        builder.matrix("text/ple_table", root + "embed_tokens_per_layer.weight",
                       {config["vocab_size_per_layer_input"].get<std::uint64_t>(), rows}, {});
        builder.matrix("text/ple_projection", root + "per_layer_model_projection.weight",
                       {rows, h}, {"text/ple_projection_input"}, "per_layer_model_projection");
        builder.vector("text/ple_norm", root + "per_layer_projection_norm.weight", {width});
    }
    for (std::size_t i = 0; i < layers.size(); ++i) {
        text_block(builder, model, config, root, i, layers[i].get<std::string>());
    }
    if (vision_store != nullptr) {
        vision_tower(builder, *vision_store, model.components["vision"]["config"]);
    }
    return model;
}

Model build_assistant(const Checkpoint& base, const ResourceOverrides& overrides) {
    Model model;
    model.components["text"]["config"] = assistant_config(base.config());
    load_resources(model, base.root(), model.config()["vocab_size"].get<std::uint64_t>(),
                   overrides, nullptr, resource_fallbacks(base));
    const auto& config = model.config(); // after load_resources: adding resources moves the config
    Builder builder(model, base);
    const auto h     = config["hidden_size"].get<std::uint64_t>();
    const auto b     = config["backbone_hidden_size"].get<std::uint64_t>();
    const auto v     = config["vocab_size"].get<std::uint64_t>();
    const auto heads = config["num_attention_heads"].get<std::uint64_t>();
    const auto inter = config["intermediate_size"].get<std::uint64_t>();
    builder.matrix("text/pre_projection", "pre_projection.weight", {h, 2 * b},
                   {"text/draft_input"});
    builder.matrix("text/post_projection", "post_projection.weight", {b, h}, {"text/final_hidden"});
    builder.matrix("text/output_head", "model.embed_tokens.weight", {v, h}, {"text/final_hidden"});
    builder.vector("text/final_norm", "model.norm.weight", {h});
    const auto& kinds = config["layer_types"];
    for (std::size_t i = 0; i < kinds.size(); ++i) {
        const auto p = layer_prefix(i);
        const auto s = "model.layers." + std::to_string(i) + ".";
        const auto d = config["attention_geometry"][kinds[i].get<std::string>()]["head_dim"]
                           .get<std::uint64_t>();
        for (const auto& [role, field] :
             {std::pair{"input_norm", "input_layernorm"},
              std::pair{"post_attention_norm", "post_attention_layernorm"},
              std::pair{"pre_ffn_norm", "pre_feedforward_layernorm"},
              std::pair{"post_ffn_norm", "post_feedforward_layernorm"}}) {
            builder.vector(p + role, s + field + ".weight", {h});
        }
        builder.matrix(p + "attention/query", s + "self_attn.q_proj.weight", {heads * d, h},
                       {p + "mixer_input"});
        builder.vector(p + "attention/query_norm", s + "self_attn.q_norm.weight", {d});
        builder.matrix(p + "attention/output", s + "self_attn.o_proj.weight", {h, heads * d},
                       {p + "attention/context"});
        for (const char* role : {"gate", "up"}) {
            builder.matrix(p + "mlp/" + role, s + "mlp." + role + "_proj.weight", {inter, h},
                           {p + "ffn_input"});
        }
        builder.matrix(p + "mlp/down", s + "mlp.down_proj.weight", {h, inter}, {p + "mlp/product"});
        builder.group({p + "mlp/gate", p + "mlp/up"});
        builder.vector(p + "output_scale", s + "layer_scalar", {1}, QType::FP32);
    }
    return model;
}

Model build_dflash(const Checkpoint& base, const ResourceOverrides& overrides) {
    Model model;
    const bool block = block_dflash(base.config());
    model.components["text"]["config"] =
        block ? block_dflash_config(base.config(), tokenizer_vocabulary(base, overrides))
              : dflash_config(base.config());
    // Muse's assistant (and a GGUF read back) names its context projection and norm under
    // encoder., z-lab's drafters at the top level.
    const bool encoder     = base.has("encoder.fc.weight");
    const std::string fc   = encoder ? "encoder.fc.weight" : "fc.weight";
    const std::string norm = encoder ? "encoder.output_norm_enc.weight" : "hidden_norm.weight";
    for (const char* name :
         {"embed_tokens.weight", "lm_head.weight", "model.embed_tokens.weight"}) {
        if (base.has(name)) {
            throw std::runtime_error(std::string("dflash: ") + name +
                                     " present; this adapter uses the target's table");
        }
    }
    const auto& named = model.config();
    const std::uint64_t vocab =
        named.contains("vocab_size") ? named["vocab_size"].get<std::uint64_t>()
                                     : *tokenizer_vocabulary(base, overrides);
    load_resources(model, base.root(), vocab, overrides, nullptr, resource_fallbacks(base));
    const auto& config = model.config(); // after load_resources: adding resources moves the config
    Builder builder(model, base);
    const auto h     = config["hidden_size"].get<std::uint64_t>();
    const auto d     = config["head_dim"].get<std::uint64_t>();
    const auto q     = config["num_attention_heads"].get<std::uint64_t>() * d;
    const auto kv    = config["num_key_value_heads"].get<std::uint64_t>() * d;
    const auto inter = config["intermediate_size"].get<std::uint64_t>();
    const auto taps  = config["dflash_config"]["target_layer_ids"].size();
    const bool bias  = config.value("attention_bias", false);
    builder.matrix("text/feature_projection", fc, {h, taps * h}, {"text/target_features"});
    builder.vector("text/context_norm", norm, {h});
    builder.vector("text/final_norm", "norm.weight", {h});
    std::vector<std::string> context;
    const auto layers = config["num_hidden_layers"].get<std::uint64_t>();
    for (std::uint64_t i = 0; i < layers; ++i) {
        const auto p = layer_prefix(i);
        const auto s = "layers." + std::to_string(i) + ".";
        builder.vector(p + "input_norm", s + "input_layernorm.weight", {h});
        builder.vector(p + "post_attention_norm", s + "post_attention_layernorm.weight", {h});
        for (const auto& [role, field, rows] :
             {std::tuple{"query", "q_proj", q}, std::tuple{"key", "k_proj", kv},
              std::tuple{"value", "v_proj", kv}}) {
            builder.matrix(p + "attention/" + role, s + "self_attn." + field + ".weight", {rows, h},
                           {p + "mixer_input"});
        }
        builder.group({p + "attention/query", p + "attention/key", p + "attention/value"});
        // The same key/value weights also read the context: a second use with its own input.
        for (const auto& [role, field] :
             {std::pair{"key", "k_proj"}, std::pair{"value", "v_proj"}}) {
            const auto name = p + "attention/context_" + role;
            builder.matrix(name, s + "self_attn." + field + ".weight", {kv, h},
                           {"text/context_input"});
            context.push_back(name);
        }
        builder.vector(p + "attention/query_norm", s + "self_attn.q_norm.weight", {d});
        builder.vector(p + "attention/key_norm", s + "self_attn.k_norm.weight", {d});
        builder.matrix(p + "attention/output", s + "self_attn.o_proj.weight", {h, q},
                       {p + "attention/context"});
        if (bias) {
            // The key and value biases apply to the context's keys and values too.
            for (const auto& [role, field, rows] :
                 {std::tuple{"query", "q_proj", q}, std::tuple{"key", "k_proj", kv},
                  std::tuple{"value", "v_proj", kv}, std::tuple{"output", "o_proj", h}}) {
                builder.vector(p + "attention/" + role + "_bias", s + "self_attn." + field + ".bias",
                               {rows});
            }
        }
        for (const char* role : {"gate", "up"}) {
            builder.matrix(p + "mlp/" + role, s + "mlp." + role + "_proj.weight", {inter, h},
                           {p + "ffn_input"});
        }
        builder.group({p + "mlp/gate", p + "mlp/up"});
        builder.matrix(p + "mlp/down", s + "mlp.down_proj.weight", {h, inter}, {p + "mlp/product"});
    }
    builder.group(context);
    return model;
}

// ---------------------------------------------------------------------------------------------
// official_recipes.py

// gemma4.encoded_sites: which encoded representation the checkpoint carries per projection, and
// keeps. The Python NVFP4 and FP8 sites are found as Python finds them; encodings Python cannot
// read count when an engine format keeps them (MXFP4 as NVFP4, FP8 with row or tensor scales)
// and are otherwise dequantized like any unencoded site.
std::map<std::size_t, KeptEncoding> encoded_sites(const Model& model, const Checkpoint& base) {
    std::map<std::size_t, KeptEncoding> found;
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        const auto& parameter = model.parameters[i];
        if (!parameter.projection() || parameter.probe_leaf.empty()) { continue; }
        const auto& dims = parameter.probe_shape.empty() ? parameter.shape : parameter.probe_shape;
        for (const char* root : {"model.language_model.", "model."}) {
            const auto prefix = root + parameter.probe_leaf;
            const auto kept   = kept_encoding(base, prefix, dims[0], dims[1]);
            if (kept) {
                found[i] = *kept;
                break;
            }
        }
    }
    return found;
}

// The official NVFP4 recipes (`published` false) keep the checkpoint's encoded projections and
// quantize everything else themselves: Q8 projections, a Q8 table and a Q6 head.
// gemma4_published (`published` true) quantizes nothing: every encoded projection the engine
// holds (NVFP4 with or without its calibrated activation divisor, row/tensor FP8) is imported as
// it is, and every other tensor keeps its direct BF16 form (table, head, per-layer embeddings,
// unencoded projections).
void gemma4_kept(Model& model, Recipe& recipe, const Checkpoint& base, bool published) {
    const auto encoded = encoded_sites(model, base);
    if (!published) {
        recipe.assign(model.at("text/token_embedding"), QType::Q8_G32_FP16,
                      Method::GroupedAbsmax);
        recipe.assign(model.at("text/output_head"), QType::Q6_G64_FP16, Method::GroupedAbsmax);
    } else {
        // A table its publisher quantized (GGUF Q8_0, Q4_0, K-quants) is imported word for word;
        // folding the scale in would decode and quantize it again. The engine scales its rows.
        const auto index    = model.at("text/token_embedding");
        const auto unscaled = model.parameters[index].encoded_source(base, MatrixFormat::Auto);
        unscaled->resolve();
        const auto format = unscaled->encoded_format();
        if (unscaled->dequantizes() && format && kept_where_decoded(*format) &&
            unscaled->keeps_losslessly()) {
            recipe.assign(index, *format, Method::ImportEncoded, unscaled);
            model.components["text"]["config"]["embedding_scale_at_run_time"] = true;
            // The tied head reads the very same published words: one object serves both (a
            // second copy of a 262144-row table is a gigabyte of device memory on 31B).
            const auto head = model.at("text/output_head");
            const auto& tied = model.parameters[head].source;
            tied->resolve();
            if (model.config().value("tie_word_embeddings", false) && tied->dequantizes() &&
                tied->encoded_format() == format && tied->label() == unscaled->label()) {
                recipe.share(model, head, index);
            }
        }
    }
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        const auto& parameter = model.parameters[i];
        const bool text_projection =
            parameter.name.starts_with("text/layers/") || parameter.name == "text/ple_projection";
        if (!text_projection || !parameter.projection()) { continue; }
        const auto it = encoded.find(i);
        std::optional<KeptEncoding> site;
        if (it != encoded.end()) { site = it->second; }
        // An NVFP4 site of any row count keeps its words (26B's shared MLP gate/up, 2112 rows
        // each): the layout completes a last 128-row scale tile its rows do not fill, and default
        // packing keeps sites of different divisors apart.
        if (site) {
            recipe.assign(i, site->format, Method::ImportEncoded,
                          parameter.encoded_source(base, encoded_matrix_format(site->format)));
            // An NVFP4 plane without a calibrated activation divisor (MXFP4) keeps BF16 inputs.
            recipe.set_policy(i, kept_policy(*site));
            continue;
        }
        // The router decides which experts run at all; it keeps its direct representation.
        if (published || parameter.name.ends_with("/moe/router")) { continue; }
        recipe.assign(i, QType::Q8_G32_FP16, Method::GroupedAbsmax);
    }
    for (const auto bank : model.expert_banks) {
        const auto& names = model.packing_groups[bank];
        if (std::all_of(names.begin(), names.end(), [&](std::size_t i) {
                const auto it = encoded.find(i);
                return it != encoded.end() && it->second.format == QType::NVFP4;
            })) {
            refuse_split_nvfp4_tiles(model, recipe, names);
            recipe.explicit_groups.push_back(names);
        }
    }
}

void drafter_q8(Model& model, Recipe& recipe, const char* model_type, const char* label) {
    if (model.config().value("model_type", std::string()) != model_type) {
        throw std::runtime_error(std::string("it requires a Gemma 4 ") + label);
    }
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        if (model.parameters[i].projection()) {
            recipe.assign(i, QType::Q8_G32_FP16, Method::GroupedAbsmax);
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------------------------

Architecture architecture_of(const OJson& config) {
    const auto name = declared_architecture(config);
    if (name == "Gemma4ForCausalLM" || name == "Gemma4ForConditionalGeneration" ||
        name == "Gemma4UnifiedForConditionalGeneration") {
        return Architecture::Gemma4Text;
    }
    if (name == "Gemma4AssistantForCausalLM" || name == "Gemma4UnifiedAssistantForCausalLM") {
        return Architecture::Gemma4Assistant;
    }
    if (name == "DFlashDraftModel" || name == "MuseGlimmerAssistantModel") {
        return Architecture::Gemma4DFlash;
    }
    throw std::runtime_error("unsupported architecture '" + name + "'");
}

Model build_model(const Checkpoint& base, const ResourceOverrides& overrides,
                  const Checkpoint* vision) {
    if (vision != nullptr && architecture_of(base.config()) != Architecture::Gemma4Text) {
        throw std::runtime_error("only a Gemma 4 text model takes a Vision encoder");
    }
    switch (architecture_of(base.config())) {
    case Architecture::Gemma4Text:
        return build_text(base, overrides, vision);
    case Architecture::Gemma4Assistant:
        return build_assistant(base, overrides);
    case Architecture::Gemma4DFlash:
        return build_dflash(base, overrides);
    }
    throw std::logic_error("unreachable");
}

void restrict_head(Model& model, const std::filesystem::path& ranking,
                   const std::string& ranking_text, std::uint64_t rows) {
    const auto ids = shortlist_ids(model, ranking, rows);

    const auto index    = model.at("text/output_head");
    auto& head          = model.parameters[index];
    auto source         = gather_source(head.source, ids);
    head.shape          = source->shape();
    head.source         = std::move(source);
    head.encoded_source = nullptr;
    head.direct         = QType::BF16;
    head.residency      = "text";
    head.probe_leaf.clear();

    std::vector<std::int32_t> words(ids.begin(), ids.end());
    Parameter token_ids;
    token_ids.name   = "text/head_token_ids";
    token_ids.shape  = {rows};
    token_ids.source = int32_source(std::move(words), "shortlist(" + ranking_text + ")");
    token_ids.direct = QType::INT32;
    model.add(std::move(token_ids));
    model.config()["head_rows"] = rows;
}

void apply_recipe(std::string_view name, Model& model, Recipe& recipe, const Checkpoint& base) {
    if (name == "gemma4_published") {
        if (model.config().value("model_type", std::string()) != "gemma4_text") {
            throw std::runtime_error("gemma4_published requires a Gemma 4 text model");
        }
        gemma4_kept(model, recipe, base, true);
    } else if (name == "gemma4_26b_a4b_nvfp4") {
        if (!model.config().contains("num_experts")) {
            throw std::runtime_error("it requires a Gemma 4 MoE model");
        }
        gemma4_kept(model, recipe, base, false);
    } else if (name == "gemma4_31b_nvfp4") {
        if (model.config().contains("num_experts")) {
            throw std::runtime_error("it requires a Gemma 4 dense model");
        }
        gemma4_kept(model, recipe, base, false);
    } else if (name == "gemma4_assistant_published") {
        // The drafter as published: nothing quantized here. A BF16 checkpoint stays BF16; a
        // published encoding (a GGUF's Q8_0) is kept word for word by keep_published.
        if (model.config().value("model_type", std::string()) != "gemma4_assistant") {
            throw std::runtime_error("it requires a Gemma 4 assistant");
        }
    } else if (name == "gemma4_assistant_q8") {
        // NInferno's own Q8 of the drafter: only when named (--recipe), never the default.
        drafter_q8(model, recipe, "gemma4_assistant", "assistant");
    } else if (name == "gemma4_dflash_published") {
        // The DFlash drafter as published (BF16 as is): nothing quantized here.
        if (model.config().value("model_type", std::string()) != "gemma4_dflash") {
            throw std::runtime_error("it requires a Gemma 4 DFlash drafter");
        }
    } else if (name == "gemma4_dflash_q8") {
        // NInferno's own Q8 of the DFlash drafter: only when named (--recipe), never the default.
        drafter_q8(model, recipe, "gemma4_dflash", "DFlash drafter");
    } else if (name == "dflash_published") {
        // Every weight as published: BF16 as BF16, a GGUF's K-quants as their own words.
        if (model.config().value("model_type", std::string()) != "dflash") {
            throw std::runtime_error("it requires a block-mask DFlash "
                                     "drafter (Muse Glimmer's assistant or a GGUF DFlash)");
        }
    } else {
        throw std::runtime_error("not an official recipe of the native converter (ninferno-convert "
                                 "--help lists them)");
    }
}

std::string default_recipe(const OJson& config) {
    switch (architecture_of(config)) {
    case Architecture::Gemma4Text:
        // Weights run as published: the checkpoint's own encodings, nothing quantized here.
        (void)text_config(config); // validates the text model before anything is fetched
        return "gemma4_published";
    case Architecture::Gemma4Assistant:
        // As published (BF16 as is); gemma4_assistant_q8 only when asked for.
        return "gemma4_assistant_published";
    case Architecture::Gemma4DFlash:
        // As published (BF16 as is); a block-mask drafter (Muse Glimmer's) takes the generic
        // dflash_published; gemma4_dflash_q8 only when asked for.
        return block_dflash(config) ? "dflash_published" : "gemma4_dflash_published";
    }
    throw std::logic_error("unreachable");
}

} // namespace ninferno::convert
