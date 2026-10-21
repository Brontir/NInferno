#include "models/common/loading.h"
#include "models/dense/model.h"

#include "artifact/reader.h"

#include <array>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::models::dense {

using loading::Demands;
using loading::Id;

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

// A MoE layer's demands: the router, every expert's matrices in bank order (Interleaved: one
// gate_up per expert; Concatenated: gate then up per expert), the stacked biases, the shared
// expert.
struct MoeIds {
    Id router = 0;
    std::optional<Id> router_bias, gate_up_bias, down_bias, shared_gate_up, shared_down, shared_gate;
    std::vector<Id> gate_up, down;
};

struct LayerIds {
    std::optional<Id> attention_norm, attention_post_norm, ffn_norm, ffn_post_norm;
    std::optional<Id> query_norm, key_norm, query_bias, key_bias, value_bias, output_bias, sinks;
    std::optional<Id> attention_gate;
    Id query = 0, key = 0, value = 0, output = 0, gate = 0, up = 0, down = 0;
    std::optional<MoeIds> moe;
};

} // namespace

std::unique_ptr<Model> Model::load(const std::filesystem::path& path, DeviceContext& device,
                                   ActivationProfile profile) {
    artifact::Reader reader(path);
    auto model     = std::unique_ptr<Model>(new Model());
    model->config_ = parse_config(loading::open_text_component(reader, "dense decoder"));
    const auto& c  = model->config_;
    const auto h   = static_cast<std::uint64_t>(c.hidden_size);
    const auto d   = static_cast<std::uint64_t>(c.head_dim);
    const auto qr  = std::uint64_t(c.query_heads) * d;
    const auto kvr = std::uint64_t(c.kv_heads) * d;
    const auto ir  = static_cast<std::uint64_t>(c.intermediate_size);
    const auto v   = static_cast<std::uint64_t>(c.vocab_size);

    Demands demands(reader);
    // The recipe chooses the table's format (Q8, BF16, ...); the embedding Op reads any of them.
    const Id embedding  = demands.direct("text/token_embedding", {v, h}, std::nullopt);
    const Id head       = demands.matrix("text/output_head", {v, h}, "text/final_hidden");
    const Id final_norm = demands.direct("text/final_norm", {h}, QType::BF16);
    std::vector<LayerIds> ids(c.layers);
    for (std::int32_t i = 0; i < c.layers; ++i) {
        const std::string p = "text/layers/" + std::to_string(i) + "/";
        auto& l             = ids[i];
        const auto vector   = [&](const std::string& name, std::uint64_t n) {
            return demands.direct(p + name, {n}, QType::BF16);
        };
        if (c.pre_norms()) {
            l.attention_norm = vector("attention_norm", h);
            l.ffn_norm       = vector("ffn_norm", h);
        }
        if (c.post_norms()) {
            l.attention_post_norm = vector("attention_post_norm", h);
            l.ffn_post_norm       = vector("ffn_post_norm", h);
        }
        const std::string in = p + "attention_input";
        l.query  = demands.matrix(p + "attention/query", {qr, h}, in);
        l.key    = demands.matrix(p + "attention/key", {kvr, h}, in);
        l.value  = demands.matrix(p + "attention/value", {kvr, h}, in);
        l.output = demands.matrix(p + "attention/output", {h, qr}, p + "attention/context");
        if (c.attention_output_gate) {
            l.attention_gate = demands.matrix(p + "attention/gate", {qr, h}, in);
        }
        if (c.qk_norm != ops::QkNorm::None && c.qk_norm_scaled) {
            const bool head_scope = c.qk_norm == ops::QkNorm::Head;
            l.query_norm = vector("attention/query_norm", head_scope ? d : qr);
            l.key_norm   = vector("attention/key_norm", head_scope ? d : kvr);
        }
        if (c.qkv_bias) {
            l.query_bias = vector("attention/query_bias", qr);
            l.key_bias   = vector("attention/key_bias", kvr);
            l.value_bias = vector("attention/value_bias", kvr);
        }
        if (c.output_bias) { l.output_bias = vector("attention/output_bias", h); }
        if (c.attention_sinks) {
            l.sinks = demands.direct(p + "attention/sinks", {std::uint64_t(c.query_heads)},
                                     QType::FP32);
        }
        if (c.moe) {
            const auto& m  = *c.moe;
            const auto e   = static_cast<std::uint64_t>(m.experts);
            const auto f   = static_cast<std::uint64_t>(m.inter);
            const auto in  = p + "ffn_input";
            MoeIds ids;
            ids.router = demands.matrix(p + "mlp/router", {e, h}, in);
            if (m.router_bias) { ids.router_bias = vector("mlp/router_bias", e); }
            const bool interleaved = m.op.gate_up_layout == ops::MoeGateUpLayout::Interleaved;
            for (std::uint64_t x = 0; x < e; ++x) {
                const std::string ep = p + "mlp/experts/" + std::to_string(x) + "/";
                if (interleaved) {
                    ids.gate_up.push_back(demands.matrix(ep + "gate_up", {2 * f, h}, in));
                } else {
                    ids.gate_up.push_back(demands.matrix(ep + "gate", {f, h}, in));
                    ids.gate_up.push_back(demands.matrix(ep + "up", {f, h}, in));
                }
                ids.down.push_back(demands.matrix(ep + "down", {h, f}, p + "mlp/expert_product"));
            }
            if (m.expert_bias) {
                ids.gate_up_bias = demands.direct(p + "mlp/experts/gate_up_bias", {e * 2 * f}, std::nullopt);
                ids.down_bias    = demands.direct(p + "mlp/experts/down_bias", {e * h}, std::nullopt);
            }
            if (m.shared_inter > 0) {
                const auto fs      = static_cast<std::uint64_t>(m.shared_inter);
                ids.shared_gate_up = demands.matrix(p + "mlp/shared/gate_up", {2 * fs, h}, in);
                ids.shared_down =
                    demands.matrix(p + "mlp/shared/down", {h, fs}, p + "mlp/shared_product");
                if (m.shared_gate) {
                    ids.shared_gate = demands.matrix(p + "mlp/shared/gate", {1, h}, in);
                }
            }
            l.moe = std::move(ids);
        } else {
            l.gate = demands.matrix(p + "mlp/gate", {ir, h}, p + "ffn_input");
            l.up   = demands.matrix(p + "mlp/up", {ir, h}, p + "ffn_input");
            l.down = demands.matrix(p + "mlp/down", {h, ir}, p + "mlp/product");
        }
    }

    model->backing_ = demands.materialize(device);
    auto& backing   = *model->backing_;
    const loading::Prepare prepare(demands, backing, profile == ActivationProfile::Artifact);
    auto& w           = model->weights_;
    w.token_embedding = native_weight(prepare.view(embedding));
    w.output_head     = prepare.projection(head);
    w.final_norm      = prepare.tensor(final_norm);
    const auto frequencies = [&](const std::vector<float>& values) {
        if (values.empty()) { return Tensor{}; }
        auto& buffer = backing.derived.emplace_back(values.size() * sizeof(float));
        buffer.copy_from_host(values.data(), buffer.bytes);
        return Tensor(buffer.p, DType::FP32, {static_cast<std::int32_t>(values.size())});
    };
    // A norm without weight runs with gain 1 (unit offset off): one BF16 vector of ones.
    const auto ones = [&](std::int32_t n) {
        const std::vector<std::uint16_t> values(static_cast<std::size_t>(n), 0x3F80); // BF16 1.0
        auto& buffer = backing.derived.emplace_back(values.size() * sizeof(std::uint16_t));
        buffer.copy_from_host(values.data(), buffer.bytes);
        return Tensor(buffer.p, DType::BF16, {n});
    };
    if (c.embedding_norm) { w.unit_gain_hidden = ones(c.hidden_size); }
    if (c.qk_norm == ops::QkNorm::Head && !c.qk_norm_scaled) { w.unit_gain_head = ones(c.head_dim); }
    for (const auto& group : c.rope_groups) {
        w.inv_freq.push_back(frequencies(group.inv_freq));
        w.long_inv_freq.push_back(frequencies(group.long_inv_freq));
    }
    w.layers.resize(c.layers);
    const auto optional_tensor = [&](const std::optional<Id>& id) {
        return id ? prepare.tensor(*id) : Tensor{};
    };
    for (std::int32_t i = 0; i < c.layers; ++i) {
        const auto& l = ids[i];
        auto& layer   = w.layers[i];
        try {
            layer.attention_norm      = optional_tensor(l.attention_norm);
            layer.attention_post_norm = optional_tensor(l.attention_post_norm);
            layer.ffn_norm            = optional_tensor(l.ffn_norm);
            layer.ffn_post_norm       = optional_tensor(l.ffn_post_norm);
            layer.query_norm          = optional_tensor(l.query_norm);
            layer.key_norm            = optional_tensor(l.key_norm);
            layer.query_bias          = optional_tensor(l.query_bias);
            layer.key_bias            = optional_tensor(l.key_bias);
            layer.value_bias          = optional_tensor(l.value_bias);
            layer.output_bias         = optional_tensor(l.output_bias);
            if (l.attention_gate) {
                // [q; k; gate; v], else [q; k; gate] with v apart (the converter's group order).
                const std::array qkgv{l.query, l.key, *l.attention_gate, l.value};
                layer.qkv = prepare.packed(qkgv);
                if (!layer.qkv) {
                    const std::array qkg{l.query, l.key, *l.attention_gate};
                    layer.qkv = prepare.packed(qkg);
                    if (layer.qkv) { layer.value = prepare.projection(l.value); }
                }
                layer.gate_in_qkv = layer.qkv.has_value();
            }
            const std::array qkv{l.query, l.key, l.value};
            if (!layer.qkv) { layer.qkv = prepare.packed(qkv); }
            if (!layer.qkv) {
                // q and k packed without v (the converter packs a group's longest run).
                const std::array qk{l.query, l.key};
                layer.qkv = prepare.packed(qk);
                if (layer.qkv) { layer.value = prepare.projection(l.value); }
            }
            if (!layer.qkv) {
                layer.query = prepare.projection(l.query);
                layer.key   = prepare.projection(l.key);
                layer.value = prepare.projection(l.value);
            }
            layer.output = prepare.projection(l.output);
            if (l.attention_gate && !layer.gate_in_qkv) {
                layer.attention_gate = prepare.projection(*l.attention_gate);
            }
            layer.sinks  = optional_tensor(l.sinks);
            if (l.moe) {
                const auto& m = *l.moe;
                // Each bank is one stacked plane: the Op addresses expert e's rows in it.
                const auto bank = [&](const std::vector<Id>& ids, const char* what) {
                    const auto joined = prepare.packed(ids);
                    if (!joined) {
                        throw std::runtime_error(std::string("mlp/experts: the ") + what +
                                                 " bank is not one complete parent in bank order");
                    }
                    return *joined;
                };
                ops::MoeWeights w;
                w.router       = prepare.projection(m.router).weight;
                w.router_bias  = optional_tensor(m.router_bias);
                const auto gate_up = bank(m.gate_up, "gate/up");
                const auto down    = bank(m.down, "down");
                w.gate_up       = gate_up.weight;
                w.down          = down.weight;
                w.expert_policy = loading::expert_policy(gate_up, down);
                w.gate_up_bias = optional_tensor(m.gate_up_bias);
                w.down_bias    = optional_tensor(m.down_bias);
                if (m.shared_gate_up) {
                    const auto shared_up   = prepare.projection(*m.shared_gate_up);
                    const auto shared_down = prepare.projection(*m.shared_down);
                    w.shared_gate_up       = shared_up.weight;
                    w.shared_down          = shared_down.weight;
                    w.shared_policy = ops::shared_moe_policy(shared_up.policy, shared_down.policy);
                }
                if (m.shared_gate) { w.shared_gate = prepare.projection(*m.shared_gate).weight; }
                layer.moe = w;
            } else {
                const std::array gate_up{l.gate, l.up};
                layer.gate_up = prepare.packed(gate_up);
                if (!layer.gate_up) {
                    layer.gate = prepare.projection(l.gate);
                    layer.up   = prepare.projection(l.up);
                }
                layer.down = prepare.projection(l.down);
            }
        } catch (const std::exception& error) {
            throw std::runtime_error("text/layers/" + std::to_string(i) + ": " + error.what());
        }
    }
    return model;
}

} // namespace ninferno::models::dense
