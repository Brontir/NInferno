#include "models/gemma4/load.h"

#include "artifact/reader.h"
#include "models/common/activation_census.h"
#include "core/arena.h"
#include "core/weight_view.h"
#include "ninferno/ops/gemma4_moe.h"
#include "ninferno/ops/moe.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>
#include <deque>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::models::gemma4 {

using loading::Demands;
using loading::Id;
using loading::open_text_component;
using loading::Prepare;

Model::~Model() = default;

std::uint64_t Model::device_bytes() const noexcept {
    return backing_ ? backing_->materialized.stats().device_capacity_bytes : 0;
}

double Model::upload_seconds() const noexcept {
    return backing_ ? backing_->materialized.stats().upload_seconds : 0;
}

std::uint64_t Model::host_to_device_bytes() const noexcept {
    return backing_ ? backing_->materialized.stats().h2d_bytes : 0;
}

namespace {

struct LayerIds {
    Id input_norm, post_attention_norm, pre_ffn_norm, post_ffn_norm, query_norm, key_norm;
    Id query, key, output, gate, up, down;
    std::optional<Id> value;
    float output_scale = 1;
    Id pre_moe_norm = 0, post_mlp_norm = 0, post_moe_norm = 0, router = 0, router_scale = 0,
       expert_scale = 0;
    Id ple_gate = 0, ple_output = 0, post_ple_norm = 0;
    std::vector<Id> expert_gate_up, expert_down;
};

} // namespace

namespace {

struct VisionLayerIds {
    Id input_norm, post_attention_norm, pre_ffn_norm, post_ffn_norm, query_norm, key_norm;
    Id query, key, value, output, gate_up, down;
};

struct VisionIds {
    Id patch_embedding, position_table, std_bias, std_scale, projection;
    std::vector<VisionLayerIds> layers;
};

VisionIds demand_vision(Demands& demands, const VisionConfig& v) {
    const auto h  = static_cast<std::uint64_t>(v.hidden_size);
    const auto d  = static_cast<std::uint64_t>(v.head_dim);
    const auto ff = static_cast<std::uint64_t>(v.intermediate_size);
    VisionIds ids;
    ids.patch_embedding = demands.matrix("vision/patch_embedding",
                                         {h, static_cast<std::uint64_t>(v.patch_width())},
                                         "vision/patch_input");
    ids.position_table = demands.direct("vision/position_table",
                                        {2 * static_cast<std::uint64_t>(v.table_size), h},
                                        QType::BF16);
    ids.std_bias  = demands.direct("vision/std_bias", {h}, QType::BF16);
    ids.std_scale = demands.direct("vision/std_scale", {h}, QType::BF16);
    for (std::int32_t i = 0; i < v.layers; ++i) {
        const auto p = "vision/layers/" + std::to_string(i) + "/";
        VisionLayerIds l;
        l.input_norm          = demands.direct(p + "input_norm", {h}, QType::BF16);
        l.post_attention_norm = demands.direct(p + "post_attention_norm", {h}, QType::BF16);
        l.pre_ffn_norm        = demands.direct(p + "pre_feedforward_norm", {h}, QType::BF16);
        l.post_ffn_norm       = demands.direct(p + "post_feedforward_norm", {h}, QType::BF16);
        l.query_norm          = demands.direct(p + "attention/query_norm", {d}, QType::BF16);
        l.key_norm            = demands.direct(p + "attention/key_norm", {d}, QType::BF16);
        l.query  = demands.matrix(p + "attention/query", {h, h}, p + "attention_input");
        l.key    = demands.matrix(p + "attention/key", {h, h}, p + "attention_input");
        l.value  = demands.matrix(p + "attention/value", {h, h}, p + "attention_input");
        l.output = demands.matrix(p + "attention/output", {h, h}, p + "attention_output");
        l.gate_up = demands.matrix(p + "mlp/gate_up", {2 * ff, h}, p + "mlp_input");
        l.down    = demands.matrix(p + "mlp/down", {h, ff}, p + "mlp_activation");
        ids.layers.push_back(l);
    }
    ids.projection = demands.matrix("vision/projection",
                                    {static_cast<std::uint64_t>(v.output_hidden), h},
                                    "vision/projection_input");
    return ids;
}

VisionWeights prepare_vision(const Prepare& prepare, const VisionIds& ids) {
    VisionWeights w;
    w.patch_embedding = prepare.projection(ids.patch_embedding);
    // [2 * table, hidden]: x rows, then y rows.
    const auto& table = prepare.view(ids.position_table);
    w.position_table   = weight_tensor(table, {static_cast<std::int32_t>(table.shape[1]),
                                               static_cast<std::int32_t>(table.shape[0])});
    w.std_bias        = prepare.tensor(ids.std_bias);
    w.std_scale       = prepare.tensor(ids.std_scale);
    for (const auto& l : ids.layers) {
        VisionLayer layer;
        layer.input_norm          = prepare.tensor(l.input_norm);
        layer.post_attention_norm = prepare.tensor(l.post_attention_norm);
        layer.pre_ffn_norm        = prepare.tensor(l.pre_ffn_norm);
        layer.post_ffn_norm       = prepare.tensor(l.post_ffn_norm);
        layer.query_norm          = prepare.tensor(l.query_norm);
        layer.key_norm            = prepare.tensor(l.key_norm);
        const std::array<Id, 3> qkv{l.query, l.key, l.value};
        layer.qkv = prepare.packed(qkv);
        if (!layer.qkv) {
            layer.query = prepare.projection(l.query);
            layer.key   = prepare.projection(l.key);
            layer.value = prepare.projection(l.value);
        }
        layer.output  = prepare.projection(l.output);
        layer.gate_up = prepare.projection(l.gate_up);
        layer.down    = prepare.projection(l.down);
        w.layers.push_back(layer);
    }
    w.projection = prepare.projection(ids.projection);
    return w;
}

} // namespace

std::unique_ptr<Model> Model::load(const std::filesystem::path& path, DeviceContext& device,
                                   ActivationProfile profile, bool vision) {
    artifact::Reader reader(path);
    const auto& text = open_text_component(reader, "gemma4");
    auto model       = std::unique_ptr<Model>(new Model());
    model->config_   = parse_config(text);
    const auto& c    = model->config_;
    const auto h     = static_cast<std::uint64_t>(c.hidden_size);

    Demands demands(reader);
    std::optional<VisionIds> vision_ids;
    if (vision) {
        const auto& components = reader.directory().components;
        const auto it          = components.find("vision");
        if (it == components.end()) {
            throw std::invalid_argument("--vision: this Gemma 4 artifact carries no vision tower "
                                        "(convert it with its vision component)");
        }
        model->vision_config_ = parse_vision_config(it->second.config);
        if (model->vision_config_->output_hidden != c.hidden_size) {
            throw std::invalid_argument("gemma4 vision: the projector's width is not the text's");
        }
        vision_ids = demand_vision(demands, *model->vision_config_);
    }
    // Tables are bound in whatever format the recipe stored; ops::embedding accepts or refuses it.
    const Id embedding =
        demands.direct("text/token_embedding", {std::uint64_t(c.vocab_size), h}, std::nullopt);
    const Id head =
        demands.matrix("text/output_head", {std::uint64_t(c.vocab_size), h}, "text/final_hidden");
    const Id final_norm = demands.direct("text/final_norm", {h}, QType::BF16);
    const auto ple_rows = std::uint64_t(c.layers) * std::uint64_t(c.ple_width);
    Id ple_table = 0, ple_projection = 0, ple_norm = 0;
    if (c.ple()) {
        // Each token gathers one row of L*P values: the table is read in place from pinned host
        // memory (E4B: 5.6 GB), not copied into the device arena.
        ple_table = demands.mapped("text/ple_table", {std::uint64_t(c.ple_vocab), ple_rows},
                                   std::nullopt);
        ple_projection =
            demands.matrix("text/ple_projection", {ple_rows, h}, "text/ple_projection_input");
        ple_norm = demands.direct("text/ple_norm", {std::uint64_t(c.ple_width)}, QType::BF16);
    }

    std::vector<LayerIds> layer_ids(c.layers);
    for (std::int32_t i = 0; i < c.layers; ++i) {
        const std::string p     = "text/layers/" + std::to_string(i) + "/";
        const auto& g           = c.geometry(c.kinds[i]);
        const auto d            = static_cast<std::uint64_t>(g.head_dim);
        const auto q_rows       = std::uint64_t(c.query_heads) * d;
        const auto kv_rows      = std::uint64_t(g.kv_heads) * d;
        auto& ids               = layer_ids[i];
        ids.input_norm          = demands.direct(p + "input_norm", {h}, QType::BF16);
        ids.post_attention_norm = demands.direct(p + "post_attention_norm", {h}, QType::BF16);
        ids.pre_ffn_norm        = demands.direct(p + "pre_ffn_norm", {h}, QType::BF16);
        ids.post_ffn_norm       = demands.direct(p + "post_ffn_norm", {h}, QType::BF16);
        ids.query_norm          = demands.direct(p + "attention/query_norm", {d}, QType::BF16);
        ids.query = demands.matrix(p + "attention/query", {q_rows, h}, p + "mixer_input");
        // A layer that reads another layer's keys and values owns no key or value projection.
        if (c.owns_kv(i)) {
            ids.key_norm = demands.direct(p + "attention/key_norm", {d}, QType::BF16);
            ids.key      = demands.matrix(p + "attention/key", {kv_rows, h}, p + "mixer_input");
            if (demands.has(p + "attention/value")) {
                ids.value = demands.matrix(p + "attention/value", {kv_rows, h}, p + "mixer_input");
            } else if (!(c.k_eq_v && c.kinds[i] == LayerKind::Full)) {
                throw artifact::ArtifactError(p + "attention/value is missing");
            }
        }
        ids.output = demands.matrix(p + "attention/output", {h, q_rows}, p + "attention/context");
        if (c.ple()) {
            const auto pw     = static_cast<std::uint64_t>(c.ple_width);
            ids.ple_gate      = demands.matrix(p + "ple/gate", {pw, h}, p + "ple_gate_input");
            ids.ple_output    = demands.matrix(p + "ple/output", {h, pw}, p + "ple/product");
            ids.post_ple_norm = demands.direct(p + "post_ple_norm", {h}, QType::BF16);
        }
        const auto inter = static_cast<std::uint64_t>(c.intermediate[i]);
        ids.gate         = demands.matrix(p + "mlp/gate", {inter, h}, p + "ffn_input");
        ids.up           = demands.matrix(p + "mlp/up", {inter, h}, p + "ffn_input");
        ids.down         = demands.matrix(p + "mlp/down", {h, inter}, p + "mlp/product");
        ids.output_scale = demands.scalar(p + "output_scale");
        if (c.moe()) {
            const auto e      = static_cast<std::uint64_t>(c.experts);
            const auto ir     = static_cast<std::uint64_t>(c.expert_intermediate);
            ids.pre_moe_norm  = demands.direct(p + "pre_moe_norm", {h}, QType::BF16);
            ids.post_mlp_norm = demands.direct(p + "post_mlp_norm", {h}, QType::BF16);
            ids.post_moe_norm = demands.direct(p + "post_moe_norm", {h}, QType::BF16);
            ids.router        = demands.matrix(p + "moe/router", {e, h}, p + "moe/router_input");
            ids.router_scale  = demands.direct(p + "moe/router_scale", {h}, QType::FP32);
            ids.expert_scale  = demands.direct(p + "moe/expert_scale", {e}, QType::FP32);
            for (std::uint64_t x = 0; x < e; ++x) {
                const std::string ep = p + "moe/experts/" + std::to_string(x) + "/";
                ids.expert_gate_up.push_back(
                    demands.matrix(ep + "gate_up", {2 * ir, h}, p + "moe/expert_input"));
                ids.expert_down.push_back(demands.matrix(ep + "down", {h, ir}, ep + "product"));
            }
        }
    }

    model->backing_ = demands.materialize(device);
    auto& backing   = *model->backing_;
    const Prepare prepare(demands, backing, profile);
    if (vision_ids) { model->vision_weights_ = prepare_vision(prepare, *vision_ids); }

    auto& w           = model->weights_;
    w.token_embedding = native_weight(prepare.view(embedding));
    {
        // The head's recorded policy is A16Only; the Q6 route has no other.
        const auto prepared = ops::prepare_linear_weight(prepare.input(head));
        w.output_head       = {prepared.weight, prepared.policy};
    }
    w.final_norm = prepare.tensor(final_norm);
    if (c.ple()) {
        w.ple_table      = native_weight(prepare.view(ple_table));
        w.ple_projection = prepare.projection(ple_projection);
        w.ple_norm       = prepare.tensor(ple_norm);
    }
    w.layers.resize(c.layers);
    for (std::int32_t i = 0; i < c.layers; ++i) {
        const auto& ids = layer_ids[i];
        auto& layer     = w.layers[i];
        try {
            layer.kind                = c.kinds[i];
            layer.input_norm          = prepare.tensor(ids.input_norm);
            layer.post_attention_norm = prepare.tensor(ids.post_attention_norm);
            layer.pre_ffn_norm        = prepare.tensor(ids.pre_ffn_norm);
            layer.post_ffn_norm       = prepare.tensor(ids.post_ffn_norm);
            layer.query_norm          = prepare.tensor(ids.query_norm);
            if (c.owns_kv(i)) {
                layer.key_norm = prepare.tensor(ids.key_norm);
                std::vector<Id> qkv{ids.query, ids.key};
                if (ids.value) { qkv.push_back(*ids.value); }
                layer.qkv = prepare.packed(qkv);
                if (!layer.qkv && ids.value) {
                    // q and k packed without v (the converter packs a group's longest run).
                    const std::array qk{ids.query, ids.key};
                    layer.qkv = prepare.packed(qk);
                    if (layer.qkv) { layer.value = prepare.projection(*ids.value); }
                }
                if (!layer.qkv) {
                    layer.query = prepare.projection(ids.query);
                    layer.key   = prepare.projection(ids.key);
                    if (ids.value) { layer.value = prepare.projection(*ids.value); }
                }
            } else {
                layer.query = prepare.projection(ids.query);
            }
            layer.output = prepare.projection(ids.output);
            if (c.ple()) {
                layer.ple_gate      = prepare.projection(ids.ple_gate);
                layer.ple_output    = prepare.projection(ids.ple_output);
                layer.post_ple_norm = prepare.tensor(ids.post_ple_norm);
            }
            const std::array gate_up{ids.gate, ids.up};
            const auto fused = prepare.packed(gate_up);
            if (!fused) { throw std::invalid_argument("gate and up must be one packed plane"); }
            layer.gate_up      = *fused;
            layer.down         = prepare.projection(ids.down);
            layer.output_scale = ids.output_scale;
            // Calibrated NVFP4 experts take the native Gemma4Moe where this build has it: its
            // four-bit prefill encodes each expert's input against the expert's recorded
            // activation divisor. Everything else (a GGUF's ggml blocks, NVFP4 below sm_120a, or
            // NVFP4 without divisors: unsloth's gemma-4-26B-A4B-it MXFP4_MOE, refused with "gate_up
            // requires a positive activation divisor") takes the generic moe Op.
            const auto calibrated = [&](const std::vector<Id>& experts) {
                return std::all_of(experts.begin(), experts.end(), [&](Id id) {
                    const auto divisor = prepare.recorded_input(id).activation_input_divisor;
                    return divisor.has_value() && std::isfinite(*divisor) && *divisor > 0;
                });
            };
            // --activations 8 holds the experts at eight bits: the native bank encodes its prefill
            // in four, so they take the generic moe Op's W4A8 prefill (as Qwen's SparseMoe does).
            const bool nvfp4_experts =
                c.moe() && profile != ActivationProfile::A8 && ops::gemma4_moe_available() &&
                contiguous_weight_region(prepare.view(ids.expert_gate_up.front()))
                        .parent->geometry.format == QType::NVFP4 &&
                calibrated(ids.expert_gate_up) && calibrated(ids.expert_down);
            if (c.moe() && !nvfp4_experts) {
                // The generic MoE Op: the experts as stored (a GGUF's ggml blocks), the router
                // reading rmsnorm(x) with router_scale * hidden^-0.5 folded into its BF16 weight.
                layer.pre_moe_norm  = prepare.tensor(ids.pre_moe_norm);
                layer.post_mlp_norm = prepare.tensor(ids.post_mlp_norm);
                layer.post_moe_norm = prepare.tensor(ids.post_moe_norm);
                ops::MoeWeights m;
                m.router       = prepare.projection(ids.router).weight;
                m.expert_scale = prepare.tensor(ids.expert_scale);
                const auto gate_up = prepare.packed(ids.expert_gate_up);
                const auto down    = prepare.packed(ids.expert_down);
                if (!gate_up || !down) {
                    throw std::invalid_argument(
                        "moe/experts: gate_up and down must each be one stacked parent in expert order");
                }
                m.gate_up       = gate_up->weight;
                m.down          = down->weight;
                m.expert_policy = models::loading::expert_policy(*gate_up, *down);
                std::vector<Id> experts(ids.expert_gate_up);
                experts.insert(experts.end(), ids.expert_down.begin(), ids.expert_down.end());
                models::loading::generic_moe(m, prepare, experts, {}, {});
                const Tensor scale = prepare.tensor(ids.router_scale);
                std::vector<float> values(static_cast<std::size_t>(c.hidden_size));
                if (cudaMemcpy(values.data(), scale.data, values.size() * sizeof(float),
                               cudaMemcpyDeviceToHost) != cudaSuccess) {
                    throw std::runtime_error("moe/router_scale: download failed");
                }
                const float root = 1.0F / std::sqrt(static_cast<float>(c.hidden_size));
                std::vector<std::uint16_t> norm(values.size());
                for (std::size_t k = 0; k < values.size(); ++k) {
                    std::uint32_t bits;
                    const float v = values[k] * root;
                    std::memcpy(&bits, &v, sizeof bits);
                    bits += 0x7fffU + ((bits >> 16) & 1U); // round to nearest even
                    norm[k] = static_cast<std::uint16_t>(bits >> 16);
                }
                auto& buffer = backing.derived.emplace_back(norm.size() * sizeof(std::uint16_t));
                buffer.copy_from_host(norm.data(), buffer.bytes);
                layer.router_norm = Tensor(buffer.p, DType::BF16, {c.hidden_size});
                layer.generic_moe = m;
                ops::MoeConfig& mc = layer.generic_moe_config;
                mc.top_k           = c.experts_per_token;
                mc.scoring         = ops::MoeScoring::Softmax;
                mc.renormalize     = true;
                mc.routed_scale    = 1.0F;
                mc.activation      = ops::MoeActivation::GeGluTanh;
                mc.gate_up_layout  = ops::MoeGateUpLayout::Concatenated;
            } else if (c.moe()) {
                layer.pre_moe_norm  = prepare.tensor(ids.pre_moe_norm);
                layer.post_mlp_norm = prepare.tensor(ids.post_mlp_norm);
                layer.post_moe_norm = prepare.tensor(ids.post_moe_norm);
                std::vector<ops::WeightInput> gate_up, down;
                for (const Id id : ids.expert_gate_up) {
                    gate_up.push_back(prepare.recorded_input(id));
                }
                for (const Id id : ids.expert_down) { down.push_back(prepare.recorded_input(id)); }
                // Every expert keeps its own recorded activation divisor; the four-bit route
                // encodes each expert's input against it.
                const auto divisors = ops::gemma4_moe_input_divisors(gate_up, down);
                // Uploaded once; the Op reads them on device.
                const auto upload = [&](const std::vector<float>& values) {
                    auto& buffer = backing.derived.emplace_back(values.size() * sizeof(float));
                    buffer.copy_from_host(values.data(), buffer.bytes);
                    return Tensor(buffer.p, DType::FP32,
                                  {static_cast<std::int32_t>(values.size())});
                };
                const Tensor gate_up_divisors = upload(divisors.gate_up);
                const Tensor down_divisors    = upload(divisors.down);
                layer.moe                     = ops::prepare_gemma4_moe_weights(
                    prepare.recorded_input(ids.router), prepare.tensor(ids.router_scale),
                    prepare.tensor(ids.expert_scale), gate_up, down, gate_up_divisors,
                    down_divisors);
                // The experts' recorded W4A4 use stays on the bank (the divisors above are its
                // calibration); only the full-A16 profile moves its prefill route to BF16.
                layer.moe_activations = profile == ActivationProfile::A16Full
                                            ? ops::Gemma4MoeActivations::A16Only
                                            : ops::Gemma4MoeActivations::Recorded;
                // The activation census (models/common/activation_census.h): the bank's
                // recorded four-bit prefill, or BF16 at every width under the full-A16 profile.
                const bool a16 = layer.moe_activations == ops::Gemma4MoeActivations::A16Only;
                for (const auto* bank : {&ids.expert_gate_up, &ids.expert_down}) {
                    for (const Id id : *bank) {
                        census_width(prepare.name(id), a16 ? 16 : 4,
                                     a16 ? "the native Gemma 4 MoE runs BF16 activations under "
                                           "this profile"
                                         : "");
                    }
                }
            }
        } catch (const std::exception& error) {
            throw std::runtime_error("text/layers/" + std::to_string(i) + ": " + error.what());
        }
    }
    return model;
}

} // namespace ninferno::models::gemma4
