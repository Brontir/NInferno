#include "models/common/loading.h"
#include "models/nemotron_h/load_moe.h"
#include "models/nemotron_h/model.h"

#include "artifact/reader.h"
#include "core/weight_view.h"

#include <array>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::models::nemotron_h {

using loading::Demands;
using loading::Id;

ops::MoeWeights prepare_moe(const loading::Prepare& prepare, const MoeIds& ids,
                            const std::string& where) {
    const auto bank = [&](const std::vector<Id>& bank_ids, const char* what) {
        auto joined = prepare.packed(bank_ids);
        if (!joined) {
            throw std::runtime_error(where + ": the " + what +
                                     " bank is not one complete parent in bank order");
        }
        return *std::move(joined);
    };
    ops::MoeWeights m;
    m.router         = prepare.projection(ids.router).weight;
    m.selection_bias = prepare.tensor(ids.selection_bias);
    const auto up    = bank(ids.up, "up");
    const auto down  = bank(ids.down, "down");
    m.gate_up        = up.weight;
    m.down           = down.weight;
    m.expert_policy  = loading::expert_policy(up, down);
    const auto shared_up   = prepare.projection(ids.shared_up);
    const auto shared_down = prepare.projection(ids.shared_down);
    m.shared_gate_up       = shared_up.weight;
    m.shared_down          = shared_down.weight;
    m.shared_policy        = ops::shared_moe_policy(shared_up.policy, shared_down.policy);
    return m;
}

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
    Id norm = 0;
    // Mamba
    Id in_proj = 0, conv_weight = 0, dt_bias = 0, a_log = 0, d = 0, mamba_norm = 0, out_proj = 0;
    std::optional<Id> conv_bias;
    // Attention
    Id query = 0, key = 0, value = 0, output = 0;
    // MoE
    Id router = 0, selection_bias = 0, shared_up = 0, shared_down = 0;
    std::vector<Id> up, down;
};

} // namespace

std::unique_ptr<Model> Model::load(const std::filesystem::path& path, DeviceContext& device,
                                   ActivationProfile profile) {
    artifact::Reader reader(path);
    auto model     = std::unique_ptr<Model>(new Model());
    model->config_ = parse_config(loading::open_text_component(reader, "nemotron_h decoder"));
    const auto& c  = model->config_;
    const auto h   = static_cast<std::uint64_t>(c.hidden_size);
    const auto v   = static_cast<std::uint64_t>(c.vocab_size);
    const auto d   = static_cast<std::uint64_t>(c.head_dim);
    const auto qr  = std::uint64_t(c.query_heads) * d;
    const auto kvr = std::uint64_t(c.kv_heads) * d;
    const auto& mb = c.mamba;
    const auto inner = static_cast<std::uint64_t>(mb.inner());
    const auto convc = static_cast<std::uint64_t>(mb.conv_channels());
    const auto mh    = static_cast<std::uint64_t>(mb.geometry.heads);
    const auto e     = static_cast<std::uint64_t>(c.moe.experts);
    const auto f     = static_cast<std::uint64_t>(c.moe.inter);
    const auto up_k  = static_cast<std::uint64_t>(c.moe.input);
    const auto fs    = static_cast<std::uint64_t>(c.moe.shared_inter);

    Demands demands(reader);
    const Id embedding  = demands.direct("text/token_embedding", {v, h}, std::nullopt);
    const Id head       = demands.matrix("text/output_head", {v, h}, "text/final_hidden");
    const Id final_norm = demands.direct("text/final_norm", {h}, QType::BF16);
    std::vector<LayerIds> ids(c.layers);
    for (std::int32_t i = 0; i < c.layers; ++i) {
        const std::string p  = "text/layers/" + std::to_string(i) + "/";
        const std::string in = p + "mixer_input";
        auto& l              = ids[i];
        l.norm               = demands.direct(p + "norm", {h}, QType::BF16);
        switch (c.blocks[i]) {
        case Block::Mamba:
            l.in_proj     = demands.matrix(p + "mamba/in_proj", {static_cast<std::uint64_t>(mb.in_rows()), h}, in);
            l.conv_weight = demands.direct(p + "mamba/conv_weight", {convc, 4}, QType::BF16);
            if (mb.conv_bias) { l.conv_bias = demands.direct(p + "mamba/conv_bias", {convc}, QType::BF16); }
            l.dt_bias     = demands.direct(p + "mamba/dt_bias", {mh}, QType::FP32);
            l.a_log       = demands.direct(p + "mamba/a_log", {mh}, QType::FP32);
            l.d           = demands.direct(p + "mamba/d", {mh}, QType::FP32);
            l.mamba_norm  = demands.direct(p + "mamba/norm", {inner}, QType::BF16);
            l.out_proj    = demands.matrix(p + "mamba/out_proj", {h, inner}, p + "mamba/gated");
            break;
        case Block::Attention:
            l.query  = demands.matrix(p + "attention/query", {qr, h}, in);
            l.key    = demands.matrix(p + "attention/key", {kvr, h}, in);
            l.value  = demands.matrix(p + "attention/value", {kvr, h}, in);
            l.output = demands.matrix(p + "attention/output", {h, qr}, p + "attention/context");
            break;
        case Block::Moe:
            l.router         = demands.matrix(p + "mlp/router", {e, h}, in);
            l.selection_bias = demands.direct(p + "mlp/selection_bias", {e}, QType::FP32);
            for (std::uint64_t x = 0; x < e; ++x) {
                const std::string ep = p + "mlp/experts/" + std::to_string(x) + "/";
                l.up.push_back(demands.matrix(ep + "up", {f, up_k}, in));
                l.down.push_back(demands.matrix(ep + "down", {h, f}, p + "mlp/expert_product"));
            }
            l.shared_up   = demands.matrix(p + "mlp/shared_up", {fs, h}, in);
            l.shared_down = demands.matrix(p + "mlp/shared_down", {h, fs}, p + "mlp/shared_product");
            break;
        }
    }

    model->backing_ = demands.materialize(device);
    auto& backing   = *model->backing_;
    const loading::Prepare prepare(demands, backing, profile == ActivationProfile::Artifact);
    auto& w           = model->weights_;
    w.token_embedding = native_weight(prepare.view(embedding));
    w.output_head     = prepare.projection(head);
    w.final_norm      = prepare.tensor(final_norm);
    w.layers.resize(c.layers);
    std::array<std::uint32_t, 3> seen{};
    for (std::int32_t i = 0; i < c.layers; ++i) {
        const auto& l = ids[i];
        auto& layer   = w.layers[i];
        try {
            layer.kind  = c.blocks[i];
            layer.index = seen[static_cast<std::size_t>(layer.kind)]++;
            layer.norm  = prepare.tensor(l.norm);
            switch (layer.kind) {
            case Block::Mamba: {
                MambaWeights m;
                m.in_proj     = prepare.projection(l.in_proj);
                m.conv_weight = weight_tensor(prepare.view(l.conv_weight),
                                              {4, static_cast<std::int32_t>(convc)});
                if (l.conv_bias) { m.conv_bias = prepare.tensor(*l.conv_bias); }
                m.dt_bias  = prepare.tensor(l.dt_bias);
                m.a_log    = prepare.tensor(l.a_log);
                m.d        = prepare.tensor(l.d);
                m.norm     = prepare.tensor(l.mamba_norm);
                m.out_proj = prepare.projection(l.out_proj);
                layer.mamba = std::move(m);
                break;
            }
            case Block::Attention: {
                AttentionWeights a;
                const std::array qkv{l.query, l.key, l.value};
                a.qkv = prepare.packed(qkv);
                if (!a.qkv) {
                    a.query = prepare.projection(l.query);
                    a.key   = prepare.projection(l.key);
                    a.value = prepare.projection(l.value);
                }
                a.output        = prepare.projection(l.output);
                layer.attention = std::move(a);
                break;
            }
            case Block::Moe: {
                layer.moe = prepare_moe(prepare,
                                        {l.router, l.selection_bias, l.shared_up, l.shared_down,
                                         l.up, l.down},
                                        "mlp/experts");
                break;
            }
            }
        } catch (const std::exception& error) {
            throw std::runtime_error("text/layers/" + std::to_string(i) + ": " + error.what());
        }
    }
    return model;
}

} // namespace ninferno::models::nemotron_h
