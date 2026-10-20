#include "models/gemma4/load.h"

#include "artifact/reader.h"
#include "models/common/config_json.h"

#include <array>
#include <deque>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::models::gemma4 {

using loading::Demands;
using loading::Id;
using loading::open_text_component;
using loading::Prepare;

Drafter::~Drafter() = default;

std::uint64_t Drafter::device_bytes() const noexcept {
    return backing_ ? backing_->materialized.stats().device_capacity_bytes : 0;
}

std::unique_ptr<Drafter> Drafter::load(const std::filesystem::path& path, DeviceContext& device,
                                       const Config& target) {
    artifact::Reader reader(path);
    const auto& text = open_text_component(reader, "gemma4 drafter");
    auto drafter     = std::unique_ptr<Drafter>(new Drafter());
    drafter->config_ = parse_drafter_config(text);
    const auto& c    = drafter->config_;
    // The drafter reads the target's caches with the target's own head geometry: every check
    // below is a pairing the attention call would otherwise get wrong in silence.
    const auto pairing = [](const char* field, auto mine, auto theirs) {
        if (mine != theirs) {
            throw artifact::ArtifactError(std::string("gemma4 drafter: its ") + field + " " +
                                          std::to_string(mine) + " differs from the target's " +
                                          std::to_string(theirs));
        }
    };
    pairing("backbone width", c.backbone_size, target.hidden_size);
    pairing("vocabulary", c.vocab_size, target.vocab_size);
    pairing("sliding window", c.sliding_window, target.sliding_window);
    pairing("query heads", c.query_heads, target.query_heads);
    for (const LayerKind kind : {LayerKind::Sliding, LayerKind::Full}) {
        const auto& mine   = c.geometry(kind);
        const auto& theirs = target.geometry(kind);
        pairing(kind == LayerKind::Sliding ? "sliding head size" : "full head size", mine.head_dim,
                theirs.head_dim);
        pairing(kind == LayerKind::Sliding ? "sliding KV heads" : "full KV heads", mine.kv_heads,
                theirs.kv_heads);
        pairing(kind == LayerKind::Sliding ? "sliding rotary pairs" : "full rotary pairs",
                mine.rotary_pairs, theirs.rotary_pairs);
        pairing(kind == LayerKind::Sliding ? "sliding rope theta" : "full rope theta",
                mine.rope_theta, theirs.rope_theta);
    }
    const auto h  = static_cast<std::uint64_t>(c.hidden_size);
    const auto b  = static_cast<std::uint64_t>(c.backbone_size);
    const auto ir = static_cast<std::uint64_t>(c.intermediate_size);

    Demands demands(reader);
    const Id pre  = demands.matrix("text/pre_projection", {h, 2 * b}, "text/draft_input");
    const Id post = demands.matrix("text/post_projection", {b, h}, "text/final_hidden");
    const Id head =
        demands.matrix("text/output_head", {std::uint64_t(c.head_rows), h}, "text/final_hidden");
    // A restricted head drafts only its shortlist; each row names the token it scores.
    std::optional<Id> head_ids;
    if (c.head_rows != c.vocab_size) {
        head_ids =
            demands.direct("text/head_token_ids", {std::uint64_t(c.head_rows)}, QType::INT32);
        const auto values = demands.values(*head_ids, QType::INT32).integers();
        validate_token_ids(values, static_cast<std::uint32_t>(c.vocab_size),
                           "gemma4 drafter head_token_ids");
    }
    const Id final_norm = demands.direct("text/final_norm", {h}, QType::BF16);

    struct Ids {
        Id input_norm, post_attention_norm, pre_ffn_norm, post_ffn_norm, query_norm;
        Id query, output, gate, up, down;
        float output_scale;
    };

    std::vector<Ids> ids(c.layers);
    for (std::int32_t i = 0; i < c.layers; ++i) {
        const std::string p   = "text/layers/" + std::to_string(i) + "/";
        const auto d          = static_cast<std::uint64_t>(c.geometry(c.kinds[i]).head_dim);
        const auto q_rows     = std::uint64_t(c.query_heads) * d;
        auto& l               = ids[i];
        l.input_norm          = demands.direct(p + "input_norm", {h}, QType::BF16);
        l.post_attention_norm = demands.direct(p + "post_attention_norm", {h}, QType::BF16);
        l.pre_ffn_norm        = demands.direct(p + "pre_ffn_norm", {h}, QType::BF16);
        l.post_ffn_norm       = demands.direct(p + "post_ffn_norm", {h}, QType::BF16);
        l.query_norm          = demands.direct(p + "attention/query_norm", {d}, QType::BF16);
        l.query  = demands.matrix(p + "attention/query", {q_rows, h}, p + "mixer_input");
        l.output = demands.matrix(p + "attention/output", {h, q_rows}, p + "attention/context");
        l.gate   = demands.matrix(p + "mlp/gate", {ir, h}, p + "ffn_input");
        l.up     = demands.matrix(p + "mlp/up", {ir, h}, p + "ffn_input");
        l.down   = demands.matrix(p + "mlp/down", {h, ir}, p + "mlp/product");
        l.output_scale = demands.scalar(p + "output_scale");
    }
    drafter->backing_ = demands.materialize(device);
    const Prepare prepare(demands, *drafter->backing_, ActivationProfile::A16);
    auto& w           = drafter->weights_;
    w.pre_projection  = prepare.projection(pre);
    w.post_projection = prepare.projection(post);
    w.head            = prepare.projection(head);
    if (head_ids) { w.head_token_ids = prepare.tensor(*head_ids); }
    w.final_norm = prepare.tensor(final_norm);
    w.layers.resize(c.layers);
    for (std::int32_t i = 0; i < c.layers; ++i) {
        auto& layer               = w.layers[i];
        const auto& l             = ids[i];
        layer.kind                = c.kinds[i];
        layer.input_norm          = prepare.tensor(l.input_norm);
        layer.post_attention_norm = prepare.tensor(l.post_attention_norm);
        layer.pre_ffn_norm        = prepare.tensor(l.pre_ffn_norm);
        layer.post_ffn_norm       = prepare.tensor(l.post_ffn_norm);
        layer.query_norm          = prepare.tensor(l.query_norm);
        layer.query               = prepare.projection(l.query);
        layer.output              = prepare.projection(l.output);
        const std::array gate_up{l.gate, l.up};
        const auto fused = prepare.packed(gate_up);
        if (!fused) { throw std::invalid_argument("drafter gate and up must be one packed plane"); }
        layer.gate_up      = *fused;
        layer.down         = prepare.projection(l.down);
        layer.output_scale = l.output_scale;
    }
    return drafter;
}

} // namespace ninferno::models::gemma4
