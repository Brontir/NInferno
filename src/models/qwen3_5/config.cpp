#include "models/qwen3_5/config.h"

#include "core/yarn.h"
#include "models/common/config_json.h"

#include "artifact/schema.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <string>

namespace ninferno::models::qwen3_5 {
namespace {

using artifact::ArtifactError;
using artifact::Json;
using artifact::require_members;

std::string architecture(const Json& config) {
    const auto& names = config.at("architectures");
    if (!names.is_array() || names.size() != 1) {
        throw ArtifactError("architectures must name one implementation");
    }
    return artifact::require_id(names[0], "architecture");
}

const artifact::Component& companion(const artifact::Directory& directory, std::string_view name) {
    const auto& component = directory.component(name);
    if (component.target != "text") {
        throw ArtifactError(std::string(name) + ": target must be text");
    }
    return component;
}

AttentionConfig attention(const Json& value) {
    AttentionConfig out{json_dimension(value, "num_attention_heads"),
                        json_dimension(value, "num_key_value_heads"),
                        json_dimension(value, "head_dim")};
    if (out.num_attention_heads % out.num_key_value_heads) {
        throw ArtifactError("attention query heads must be divisible by KV heads");
    }
    return out;
}

RopeConfig rope(const Json& value, std::uint32_t head_dim) {
    require_members(value, {"rope_theta", "partial_rotary_factor", "mrope_section"},
                    {"yarn_factor", "yarn_original"}, "text RoPE");
    RopeConfig out;
    if (value.contains("yarn_factor")) {
        out.artifact_yarn_factor   = json_positive_float(value, "yarn_factor");
        out.artifact_yarn_original = json_u32(value.at("yarn_original"), "yarn_original", false);
    }
    out.rope_theta            = json_positive_float(value, "rope_theta");
    out.partial_rotary_factor = json_positive_float(value, "partial_rotary_factor");
    if (out.partial_rotary_factor > 1) { throw ArtifactError("partial_rotary_factor exceeds one"); }
    out.rotary_dim = static_cast<std::uint32_t>(double(head_dim) * out.partial_rotary_factor);
    if (!out.rotary_dim || out.rotary_dim % 2) {
        throw ArtifactError("rotary dimension must be positive and even");
    }
    const auto& sections = value.at("mrope_section");
    if (!sections.is_array() || sections.size() != 3) {
        throw ArtifactError("MRoPE requires three sections");
    }
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        out.mrope_section[i] = json_u32(sections[i], "MRoPE section", false);
        sum += out.mrope_section[i];
    }
    if (sum != out.rotary_dim / 2) {
        throw ArtifactError("MRoPE sections differ from the rotary width");
    }
    std::array<std::uint32_t, 3> counts{};
    for (std::uint32_t pair = 0; pair < out.rotary_dim / 2; ++pair) {
        const std::uint8_t axis = pair % 3 == 1 && pair < 3ULL * out.mrope_section[1]   ? 1
                                  : pair % 3 == 2 && pair < 3ULL * out.mrope_section[2] ? 2
                                                                                        : 0;
        out.pair_axes.push_back(axis);
        ++counts[axis];
    }
    if (counts != out.mrope_section) {
        throw ArtifactError("MRoPE sections exceed interleaved axis ranges");
    }
    return out;
}

TextConfig text(const Json& value, bool mtp) {
    require_members(
        value,
        {"architectures", "model_type", "hidden_size", "vocab_size", "num_hidden_layers",
         "max_position_embeddings", "tie_word_embeddings", "rms_norm_eps", "layer_types"},
        {"num_attention_heads", "num_key_value_heads", "head_dim", "rope_parameters",
         "linear_num_key_heads", "linear_key_head_dim", "linear_num_value_heads",
         "linear_value_head_dim", "linear_conv_kernel_dim", "linear_output_head_order",
         "intermediate_size", "num_experts",
         "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size"},
        "text config");
    TextConfig out;
    out.architecture = resolve_architecture(
        architecture(value), artifact::require_id(value.at("model_type"), "model_type"));
    out.hidden_size             = json_dimension(value, "hidden_size");
    out.vocab_size              = json_dimension(value, "vocab_size");
    out.num_hidden_layers       = json_dimension(value, "num_hidden_layers");
    out.max_position_embeddings = json_dimension(value, "max_position_embeddings");
    out.rms_norm_eps            = json_positive_float(value, "rms_norm_eps");
    if (!value.at("tie_word_embeddings").is_boolean()) {
        throw ArtifactError("tie_word_embeddings must be boolean");
    }
    out.tie_word_embeddings = value.at("tie_word_embeddings").get<bool>();
    const auto& layers      = value.at("layer_types");
    if (!layers.is_array() || layers.size() != out.num_hidden_layers) {
        throw ArtifactError("layer_types must cover every text block");
    }
    for (const auto& layer : layers) {
        if (layer == "full_attention") {
            out.layer_types.push_back(MixerKind::FullAttention);
            out.compact_layer_indices.push_back(out.full_attention_layers++);
        } else if (layer == "linear_attention") {
            out.layer_types.push_back(MixerKind::LinearAttention);
            out.compact_layer_indices.push_back(out.linear_attention_layers++);
        } else {
            throw ArtifactError("unknown text layer_type");
        }
    }
    if (out.full_attention_layers || mtp || value.contains("head_dim") ||
        value.contains("rope_parameters") || value.contains("num_attention_heads") ||
        value.contains("num_key_value_heads")) {
        out.attention       = attention(value);
        out.rope_parameters = rope(value.at("rope_parameters"), out.attention->head_dim);
    }
    if (out.linear_attention_layers || value.contains("linear_num_key_heads") ||
        value.contains("linear_key_head_dim") || value.contains("linear_num_value_heads") ||
        value.contains("linear_value_head_dim") || value.contains("linear_conv_kernel_dim")) {
        GdnConfig gdn{json_dimension(value, "linear_num_key_heads"),
                      json_dimension(value, "linear_key_head_dim"),
                      json_dimension(value, "linear_num_value_heads"),
                      json_dimension(value, "linear_value_head_dim"),
                      json_dimension(value, "linear_conv_kernel_dim")};
        if (gdn.linear_num_value_heads % gdn.linear_num_key_heads) {
            throw ArtifactError("GDN value heads must be divisible by key heads");
        }
        if (value.contains("linear_output_head_order")) {
            const auto& order = value.at("linear_output_head_order");
            if (order != "grouped" && order != "tiled") {
                throw ArtifactError("linear_output_head_order must be grouped or tiled");
            }
            gdn.output_heads_tiled = order == "tiled";
        }
        (void)gdn.conv_channels();
        out.gdn = gdn;
    }
    if (out.architecture == Architecture::Qwen3_5Moe) {
        if (value.contains("intermediate_size")) {
            throw ArtifactError("MoE config carries Dense FFN width");
        }
        MoeConfig moe{json_dimension(value, "num_experts"),
                      json_dimension(value, "num_experts_per_tok"),
                      json_dimension(value, "moe_intermediate_size"),
                      json_dimension(value, "shared_expert_intermediate_size")};
        if (moe.num_experts_per_tok > moe.num_experts) {
            throw ArtifactError("selected experts exceed expert count");
        }
        out.ffn = moe;
    } else {
        for (const auto* key : {"num_experts", "num_experts_per_tok", "moe_intermediate_size",
                                "shared_expert_intermediate_size"}) {
            if (value.contains(key)) { throw ArtifactError("Dense config carries MoE fields"); }
        }
        out.ffn = DenseConfig{json_dimension(value, "intermediate_size")};
    }
    return out;
}

VisionConfig vision(const Json& value) {
    require_members(value,
                    {"model_type", "depth", "hidden_size", "intermediate_size", "num_heads",
                     "patch_size", "temporal_patch_size", "spatial_merge_size",
                     "num_position_embeddings"},
                    {}, "vision config");
    if (value.at("model_type") != "qwen3_5_vision" &&
        value.at("model_type") != "qwen3_5_moe_vision") {
        throw ArtifactError("unknown Vision model_type");
    }
    VisionConfig out;
    out.depth                   = json_dimension(value, "depth");
    out.hidden_size             = json_dimension(value, "hidden_size");
    out.intermediate_size       = json_dimension(value, "intermediate_size");
    out.num_heads               = json_dimension(value, "num_heads");
    out.patch_size              = json_dimension(value, "patch_size");
    out.temporal_patch_size     = json_dimension(value, "temporal_patch_size");
    out.spatial_merge_size      = json_dimension(value, "spatial_merge_size");
    out.num_position_embeddings = json_dimension(value, "num_position_embeddings");
    out.position_grid_side =
        static_cast<std::uint32_t>(std::sqrt(double(out.num_position_embeddings)));
    if (out.hidden_size % out.num_heads || (out.hidden_size / out.num_heads) % 4 ||
        std::uint64_t(out.position_grid_side) * out.position_grid_side !=
            out.num_position_embeddings) {
        throw ArtifactError("invalid Vision head or square position geometry");
    }
    // The encoder's packed attention runs heads of 64 (Qwen3.5 0.8B to 9B) or 72 values (27B,
    // Qwen3.6 35B); any other width is refused here, not on the first image.
    if (const std::uint64_t head_dim = out.hidden_size / out.num_heads;
        head_dim != 64 && head_dim != 72) {
        throw ArtifactError("Vision heads of " + std::to_string(head_dim) +
                            " values are not supported (64 or 72)");
    }
    (void)out.patch_width();
    (void)out.merger_width();
    return out;
}

DraftConfig draft(const Json& value, const TextConfig& target, bool dflash2) {
    require_members(value,
                    {"architectures", "model_type", "intermediate_size", "num_attention_heads",
                     "num_key_value_heads", "head_dim", "num_hidden_layers",
                     "max_position_embeddings", "rms_norm_eps", "rope_parameters", "layer_types",
                     "dflash_config"},
                    {"sliding_window"}, "draft config");
    if (architecture(value) != (dflash2 ? "DFlash2DraftModel" : "DFlashDraftModel") ||
        value.at("model_type") != "qwen3") {
        throw ArtifactError("draft architecture/model_type mismatch");
    }
    DraftConfig out;
    out.attention = attention(value);
    if (out.attention.head_dim % 2) { throw ArtifactError("draft head dimension must be even"); }
    out.intermediate_size       = json_dimension(value, "intermediate_size");
    out.num_hidden_layers       = json_dimension(value, "num_hidden_layers");
    out.max_position_embeddings = json_dimension(value, "max_position_embeddings");
    out.rms_norm_eps            = json_positive_float(value, "rms_norm_eps");
    require_members(value.at("rope_parameters"), {"rope_theta"}, {}, "draft RoPE");
    out.rope_theta     = json_positive_float(value.at("rope_parameters"), "rope_theta");
    // The draft attention normalizes and rotates Q/K in one fused kernel whose epsilon and RoPE
    // frequencies are fixed; a draft with other constants would run silently wrong.
    if (out.rope_theta != 1.0e7F || out.rms_norm_eps != 1.0e-6F) {
        throw ArtifactError("draft RoPE theta and RMSNorm epsilon must be 1e7 and 1e-6");
    }
    const auto& layers = value.at("layer_types");
    if (!layers.is_array() || layers.size() != out.num_hidden_layers) {
        throw ArtifactError("draft layer_types length mismatch");
    }
    bool sliding = false;
    for (const auto& layer : layers) {
        if (layer == "full_attention") {
            out.layer_types.push_back(DraftAttentionKind::FullAttention);
        } else if (layer == "sliding_attention") {
            out.layer_types.push_back(DraftAttentionKind::SlidingAttention);
            sliding = true;
        } else {
            throw ArtifactError("unknown draft layer_type");
        }
    }
    if (sliding || value.contains("sliding_window")) {
        out.sliding_window = json_dimension(value, "sliding_window");
    }
    const auto& config = value.at("dflash_config");
    if (dflash2) {
        require_members(config,
                        {"target_layer_ids", "mask_token_id", "conv_kernel_size", "conv_group_size",
                         "selector_rank", "selector_top_k"},
                        {}, "DFlash2 config");
    } else {
        require_members(config, {"target_layer_ids", "mask_token_id"}, {}, "DFlash config");
    }
    for (const auto index : json_layer_indices(config.at("target_layer_ids"),
                                               target.num_hidden_layers, "target_layer_ids")) {
        out.target_layer_ids.push_back(index);
    }
    out.mask_token_id = json_u32(config.at("mask_token_id"), "mask token", false);
    if (out.mask_token_id >= target.vocab_size) {
        throw ArtifactError("mask token exceeds target embedding rows");
    }
    if (dflash2) {
        DFlash2Config extra{
            json_dimension(config, "conv_kernel_size"), json_dimension(config, "conv_group_size"),
            json_dimension(config, "selector_rank"), json_dimension(config, "selector_top_k")};
        if (target.hidden_size % extra.conv_group_size) {
            throw ArtifactError("DFlash2 conv group does not divide hidden width");
        }
        out.dflash2 = extra;
    }
    return out;
}

} // namespace

std::uint64_t GdnConfig::conv_channels() const {
    return artifact::checked_add(artifact::checked_mul(2, key_width(), "GDN key width"),
                                 value_width(), "GDN convolution channels");
}

std::uint64_t VisionConfig::patch_width() const {
    return artifact::checked_mul(artifact::checked_mul(3, temporal_patch_size, "patch channels"),
                                 std::uint64_t(patch_size) * patch_size, "patch width");
}

std::uint64_t VisionConfig::merger_width() const {
    return artifact::checked_mul(std::uint64_t(spatial_merge_size) * spatial_merge_size,
                                 hidden_size, "merger width");
}

Config parse_config(const artifact::Directory& directory, const LoadOptions& options) {
    try {
        if (options.purpose != EnginePurpose::Generation &&
            options.purpose != EnginePurpose::CausalScoring) {
            throw ArtifactError("unknown loading purpose");
        }
        if (options.purpose == EnginePurpose::CausalScoring &&
            (options.vision || options.speculative != SpeculativeBackend::None)) {
            throw ArtifactError("CausalScoring loads the Text backbone only");
        }
        if (options.speculative != SpeculativeBackend::None &&
            options.speculative_component().empty()) {
            throw ArtifactError("unknown speculative backend");
        }
        if (options.proposal_head != ProposalHead::Full &&
            options.proposal_head != ProposalHead::Optimized) {
            throw ArtifactError("unknown proposal head selection");
        }
        Config out;
        out.mtp  = options.speculative == SpeculativeBackend::Mtp;
        out.text = text(directory.component("text").config, out.mtp);
        const RopeConfig* own_rope =
            out.text.rope_parameters ? &*out.text.rope_parameters : nullptr;
        const double yarn_factor = options.rope_yarn_factor != 1.0F
                                       ? static_cast<double>(options.rope_yarn_factor)
                                       : (own_rope ? own_rope->artifact_yarn_factor : 1.0);
        if (yarn_factor != 1.0) {
            // YaRN over the checkpoint's own context: its rotated pairs blend the original and
            // the stretched frequency ladders, and the model admits factor times the context.
            if (options.masked_draft()) {
                throw ArtifactError("--rope-yarn-factor runs no DFlash drafter: they rotate with "
                                    "their own trained frequencies (MTP and plain decode do)");
            }
            RopeConfig& rope    = *out.text.rope_parameters;
            const auto original = rope.artifact_yarn_original != 0 ? rope.artifact_yarn_original
                                                                   : out.text.max_position_embeddings;
            const YarnRope yarn = yarn_rope(rope.rope_theta, rope.rotary_dim,
                                            {.factor   = yarn_factor,
                                             .original = static_cast<double>(original)});
            rope.yarn_inv_freq        = yarn.inv_freq;
            rope.yarn_attention_scale = static_cast<float>(yarn.attention_scaling);
            rope.yarn_original        = original;
            out.text.max_position_embeddings = static_cast<std::uint32_t>(
                std::min<double>(double(original) * yarn_factor, 1 << 20));
        }
        if (options.vision) {
            // A text-only artifact is complete as it is; asking it for Vision is refused.
            if (!directory.components.contains("vision")) {
                throw ArtifactError("no vision component: this artifact is text-only (load it "
                                    "without Vision)");
            }
            out.vision = vision(companion(directory, "vision").config);
        }
        if (out.mtp) {
            const auto& config = companion(directory, "mtp").config;
            require_members(config, {"architectures"}, {}, "MTP config");
            if (architecture(config) != (out.text.architecture == Architecture::Qwen3_5Moe
                                             ? "Qwen3_5MoeMTP"
                                             : "Qwen3_5MTP")) {
                throw ArtifactError("MTP architecture differs from target mathematics");
            }
        }
        if (options.speculative == SpeculativeBackend::DFlash ||
            options.speculative == SpeculativeBackend::DFlash2) {
            out.draft = draft(companion(directory, options.speculative_component()).config,
                              out.text, options.speculative == SpeculativeBackend::DFlash2);
        }
        return out;
    } catch (const std::exception& error) {
        throw ArtifactError(std::string("Qwen3.5 config: ") + error.what());
    }
}

} // namespace ninferno::models::qwen3_5
