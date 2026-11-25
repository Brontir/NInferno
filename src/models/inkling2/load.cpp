// Loads an Inkling 2 artifact (convert/inkling2.cpp's parameters) into the family's Model.
#include "models/inkling2/model.h"

#include "artifact/reader.h"
#include "core/weight_view.h"
#include "models/common/loading.h"
#include "ninferno/ops/linear.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::models::inkling2 {

using loading::Demands;
using loading::Id;
using loading::Prepare;

namespace {

struct ControllerIds {
    Id down = 0, gain = 0, forget = 0;
};

struct LayerIds {
    Id input_norm = 0, q_norm = 0, k_norm = 0, v_norm = 0;
    Id query = 0, key = 0, value = 0, gate = 0, relative = 0, sink = 0, output = 0, key_taps = 0,
       value_taps = 0;
    Id query_clip = 0, relative_clip = 0, relative_bias = 0, key_conv = 0, value_conv = 0,
       output_conv = 0, ffn_conv = 0;
    Id attention_query = 0, ffn_query = 0, ffn_norm = 0, router = 0, selection_bias = 0,
       message_query         = 0;
    bool has_attention_query = false, looped = false;
    std::vector<Id> gate_up, down;
    std::array<ControllerIds, 4> control{};
};

// A direct parameter as a tensor of its stored shape, innermost dimension first (Prepare::tensor
// takes vectors only): the FP32 tables (clips, biases, taps, mix queries) are matrices.
Tensor table(const Prepare& prepare, Id id) {
    const auto& v = prepare.view(id);
    if (v.shape.size() == 1) { return weight_tensor(v, {static_cast<std::int32_t>(v.shape[0])}); }
    if (v.shape.size() == 2) {
        return weight_tensor(
            v, {static_cast<std::int32_t>(v.shape[1]), static_cast<std::int32_t>(v.shape[0])});
    }
    throw std::invalid_argument("inkling2: a direct parameter of " +
                                std::to_string(v.shape.size()) + " dimensions");
}

} // namespace

Model::~Model() = default;

std::unique_ptr<Model> Model::load(const std::filesystem::path& path, DeviceContext& device) {
    artifact::Reader reader(path);
    const auto& text = loading::open_text_component(reader, "inkling2");
    auto model       = std::unique_ptr<Model>(new Model());
    model->config_   = parse_config(text);
    const auto& c    = model->config_;
    using U          = std::uint64_t;
    const U h = U(c.hidden_size), d = U(c.head_dim), hq = U(c.query_heads), hkv = U(c.kv_heads);
    const U dr = U(c.rel_width), taps = U(c.conv_taps), e = U(c.experts), f = U(c.expert_width);
    const U r = U(c.vocab_rows);

    Demands demands(reader);
    const Id embedding  = demands.direct("text/token_embedding", {r, h}, std::nullopt);
    const Id embed_norm = demands.direct("text/embedding_norm", {h}, QType::BF16);
    const Id head       = demands.matrix("text/output_head", {r, h}, "text/final_hidden");
    const Id final_norm = demands.direct("text/final_norm", {h}, QType::BF16);
    const Id out_query  = demands.direct("text/output_query", {1, h}, QType::FP32);
    std::vector<LayerIds> ids(static_cast<std::size_t>(c.layers));
    for (std::int32_t i = 0; i < c.layers; ++i) {
        auto& l         = ids[static_cast<std::size_t>(i)];
        const auto p    = "text/layers/" + std::to_string(i) + "/";
        const auto a    = p + "attention/";
        const bool full = c.kinds[static_cast<std::size_t>(i)] == LayerKind::Full;
        const U extent  = U(c.extent(i));
        if (full) {
            l.q_norm     = demands.direct(p + "q_input_norm", {h}, QType::BF16);
            l.k_norm     = demands.direct(p + "k_input_norm", {h}, QType::BF16);
            l.v_norm     = demands.direct(p + "v_input_norm", {h}, QType::BF16);
            l.key_taps   = demands.matrix(a + "key_taps", {hkv * taps, h}, a + "key_input");
            l.value_taps = demands.matrix(a + "value_taps", {hkv * taps, h}, a + "value_input");
        } else {
            l.input_norm = demands.direct(p + "input_norm", {h}, QType::BF16);
        }
        l.query               = demands.matrix(a + "query", {hq * d, h}, a + "query_input");
        l.key                 = demands.matrix(a + "key", {hkv * d, h}, a + "key_input");
        l.value               = demands.matrix(a + "value", {hkv * d, h}, a + "value_input");
        l.gate                = demands.matrix(a + "gate", {hq * d, h}, a + "query_input");
        l.relative            = demands.matrix(a + "relative", {dr * hq, h}, a + "query_input");
        l.sink                = demands.matrix(a + "sink", {hq, h}, a + "query_input");
        l.output              = demands.matrix(a + "output", {h, hq * d}, a + "context");
        l.query_clip          = demands.direct(a + "query_clip", {hq, d}, QType::FP32);
        l.relative_clip       = demands.direct(a + "relative_clip", {hq, dr}, QType::FP32);
        l.relative_bias       = demands.direct(a + "relative_bias", {extent, dr}, QType::FP32);
        l.key_conv            = demands.direct(a + "key_conv", {hkv * d, taps}, QType::FP32);
        l.value_conv          = demands.direct(a + "value_conv", {hkv * d, taps}, QType::FP32);
        l.output_conv         = demands.direct(a + "output_conv", {h, taps}, QType::FP32);
        l.ffn_conv            = demands.direct(p + "moe/output_conv", {h, taps}, QType::FP32);
        l.has_attention_query = i > 0;
        if (i > 0) {
            l.attention_query =
                demands.direct(p + "attention_query", {full ? 3U : 1U, h}, QType::FP32);
        }
        l.ffn_query      = demands.direct(p + "ffn_query", {1, h}, QType::FP32);
        l.ffn_norm       = demands.direct(p + "ffn_norm", {h}, QType::BF16);
        l.router         = demands.matrix(p + "moe/router", {e, h}, p + "moe/router_input");
        l.selection_bias = demands.direct(p + "moe/selection_bias", {e}, QType::FP32);
        for (U x = 0; x < e; ++x) {
            const auto ep = p + "moe/experts/" + std::to_string(x) + "/";
            l.gate_up.push_back(demands.matrix(ep + "gate_up", {2 * f, h}, p + "moe/expert_input"));
            l.down.push_back(demands.matrix(ep + "down", {h, f}, ep + "product"));
        }
        l.looped = c.looped() && i >= c.core_start && i < c.core_end;
        if (l.looped) {
            const U rank         = U(c.gate_rank);
            l.message_query      = demands.direct(p + "loop/message_query", {4, h}, QType::FP32);
            const char* roles[4] = {"q", "k", "v", "ffn"};
            for (int j = 0; j < 4; ++j) {
                const auto lp                          = p + "loop/" + roles[j] + "/";
                l.control[static_cast<std::size_t>(j)] = {
                    demands.matrix(lp + "down", {rank, 2 * h}, lp + "pair"),
                    demands.matrix(lp + "gain", {h, rank}, lp + "low_rank"),
                    demands.matrix(lp + "forget", {h, rank}, lp + "low_rank")};
            }
        }
    }

    model->backing_ = demands.materialize(device);
    const Prepare prepare(demands, *model->backing_, /*recorded*/ false);
    auto& w                  = model->weights_;
    w.token_embedding.weight = native_weight(prepare.view(embedding));
    w.embedding_norm         = table(prepare, embed_norm);
    w.final_norm             = table(prepare, final_norm);
    w.output_query           = table(prepare, out_query);
    {
        const auto prepared = ops::prepare_linear_weight(prepare.input(head));
        w.output_head       = {prepared.weight, prepared.policy};
    }
    w.layers.resize(static_cast<std::size_t>(c.layers));
    for (std::int32_t i = 0; i < c.layers; ++i) {
        const auto& l   = ids[static_cast<std::size_t>(i)];
        auto& layer     = w.layers[static_cast<std::size_t>(i)];
        const bool full = c.kinds[static_cast<std::size_t>(i)] == LayerKind::Full;
        layer.kind      = c.kinds[static_cast<std::size_t>(i)];
        if (full) {
            layer.q_norm     = table(prepare, l.q_norm);
            layer.k_norm     = table(prepare, l.k_norm);
            layer.v_norm     = table(prepare, l.v_norm);
            layer.key_taps   = prepare.projection(l.key_taps);
            layer.value_taps = prepare.projection(l.value_taps);
        } else {
            layer.input_norm = table(prepare, l.input_norm);
        }
        layer.query         = prepare.projection(l.query);
        layer.key           = prepare.projection(l.key);
        layer.value         = prepare.projection(l.value);
        layer.gate          = prepare.projection(l.gate);
        layer.relative      = prepare.projection(l.relative);
        layer.sink          = prepare.projection(l.sink);
        layer.output        = prepare.projection(l.output);
        layer.query_clip    = table(prepare, l.query_clip);
        layer.relative_clip = table(prepare, l.relative_clip);
        layer.relative_bias = table(prepare, l.relative_bias);
        layer.key_conv      = table(prepare, l.key_conv);
        layer.value_conv    = table(prepare, l.value_conv);
        layer.output_conv   = table(prepare, l.output_conv);
        layer.ffn_conv      = table(prepare, l.ffn_conv);
        if (l.has_attention_query) { layer.attention_query = table(prepare, l.attention_query); }
        layer.ffn_query = table(prepare, l.ffn_query);
        layer.ffn_norm  = table(prepare, l.ffn_norm);

        ops::MoeWeights& m = layer.moe;
        m.router           = prepare.projection(l.router).weight;
        m.selection_bias   = table(prepare, l.selection_bias);
        const auto gate_up = prepare.packed(l.gate_up);
        const auto down    = prepare.packed(l.down);
        if (!gate_up || !down) {
            throw std::invalid_argument(
                "inkling2 layer " + std::to_string(i) +
                ": the experts' gate_up and down must each be one stacked parent in expert order");
        }
        m.gate_up          = gate_up->weight;
        m.down             = down->weight;
        m.expert_policy    = loading::expert_policy(*gate_up, *down);
        ops::MoeConfig& mc = layer.moe_config;
        mc.top_k           = c.experts_per_tok;
        mc.scoring         = ops::MoeScoring::Sigmoid;
        mc.renormalize     = true;
        mc.routed_scale    = c.route_scale;
        mc.activation      = ops::MoeActivation::Situ;
        mc.alpha           = c.situ_beta;
        mc.limit           = c.situ_linear_beta;
        mc.gate_up_layout  = ops::MoeGateUpLayout::Concatenated;
        if (l.looped) {
            layer.message_query = table(prepare, l.message_query);
            for (std::size_t j = 0; j < 4; ++j) {
                layer.control[j] = {prepare.projection(l.control[j].down),
                                    prepare.projection(l.control[j].gain),
                                    prepare.projection(l.control[j].forget)};
            }
        }
    }
    return model;
}

} // namespace ninferno::models::inkling2
