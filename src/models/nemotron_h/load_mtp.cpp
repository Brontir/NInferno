#include "models/common/loading.h"
#include "models/nemotron_h/load_moe.h"
#include "models/nemotron_h/model.h"

#include "artifact/reader.h"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::models::nemotron_h {

using loading::Demands;
using loading::Id;
using loading::Prepare;

MtpDrafter::~MtpDrafter() = default;

std::uint64_t MtpDrafter::device_bytes() const noexcept {
    return backing_ ? backing_->materialized.stats().device_capacity_bytes : 0;
}

std::uint64_t MtpDrafter::host_to_device_bytes() const noexcept {
    return backing_ ? backing_->materialized.stats().h2d_bytes : 0;
}

namespace {

std::string listed(std::initializer_list<std::int64_t> values) {
    std::string out;
    for (const auto value : values) { out += (out.empty() ? "" : " / ") + std::to_string(value); }
    return out;
}

std::string routing(const ops::MoeConfig& m) {
    return "top-k " + std::to_string(m.top_k) + ", " +
           (m.scoring == ops::MoeScoring::Sigmoid ? "sigmoid" : "softmax") + ", renormalize " +
           (m.renormalize ? "on" : "off") + ", routed scale " + std::to_string(m.routed_scale) +
           ", " +
           (m.activation == ops::MoeActivation::Relu2
                ? std::string("relu2")
                : "activation " + std::to_string(static_cast<int>(m.activation)));
}

// Every pairing the head would otherwise get wrong in silence: the fusion, the attention geometry
// it drafts through and the MoE the Program's round math assumes identical to the trunk's. The
// head's norms take the drafter's own epsilons (MtpWeights), so those are not paired.
void check_pairs_with_target(const MtpConfig& c, const Config& target) {
    const auto mismatch = [](const char* what, const std::string& mine,
                             const std::string& theirs) {
        throw std::invalid_argument(std::string("nemotron_h MTP drafter: ") + what + " " + mine +
                                    " differs from the target's " + theirs);
    };
    if (c.hidden_size != target.hidden_size) {
        mismatch("hidden width", listed({c.hidden_size}), listed({target.hidden_size}));
    }
    if (c.vocab_size != target.vocab_size) {
        mismatch("vocabulary", listed({c.vocab_size}), listed({target.vocab_size}));
    }
    if (c.query_heads != target.query_heads || c.kv_heads != target.kv_heads ||
        c.head_dim != target.head_dim) {
        mismatch("attention geometry (query heads / kv heads / head_dim)",
                 listed({c.query_heads, c.kv_heads, c.head_dim}),
                 listed({target.query_heads, target.kv_heads, target.head_dim}));
    }
    if (c.moe.experts != target.moe.experts || c.moe.inter != target.moe.inter ||
        c.moe.shared_inter != target.moe.shared_inter) {
        mismatch("MoE shape (experts / expert width / shared expert width)",
                 listed({c.moe.experts, c.moe.inter, c.moe.shared_inter}),
                 listed({target.moe.experts, target.moe.inter, target.moe.shared_inter}));
    }
    const ops::MoeConfig& mine   = c.moe.op;
    const ops::MoeConfig& theirs = target.moe.op;
    if (mine.top_k != theirs.top_k || mine.scoring != theirs.scoring ||
        mine.renormalize != theirs.renormalize || mine.activation != theirs.activation ||
        mine.routed_scale != theirs.routed_scale) {
        mismatch("MoE routing", "(" + routing(mine) + ")", "(" + routing(theirs) + ")");
    }
}

} // namespace

std::unique_ptr<MtpDrafter> MtpDrafter::load(const std::filesystem::path& path,
                                             DeviceContext& device, const Config& target) {
    artifact::Reader reader(path);
    const auto& text = loading::open_text_component(reader, "nemotron_h MTP drafter");
    const auto c     = parse_mtp_config(text);
    check_pairs_with_target(c, target);
    const auto h   = static_cast<std::uint64_t>(c.hidden_size);
    const auto d   = static_cast<std::uint64_t>(c.head_dim);
    const auto qr  = std::uint64_t(c.query_heads) * d;
    const auto kvr = std::uint64_t(c.kv_heads) * d;
    const auto e   = static_cast<std::uint64_t>(c.moe.experts);
    const auto f    = static_cast<std::uint64_t>(c.moe.inter);
    const auto up_k = static_cast<std::uint64_t>(c.moe.input);
    const auto fs   = static_cast<std::uint64_t>(c.moe.shared_inter);

    // The inputs the projections read: the fusion's [norm(e); norm(h)], the attention sub-layer's
    // normed input, the MoE sub-layer's normed input and the MoE's intermediates.
    const std::string fusion = "mtp/fusion_input";
    const std::string attn   = "mtp/attention_input";
    const std::string ffn    = "mtp/ffn_input";

    Demands demands(reader);
    const Id embedding_norm      = demands.direct("mtp/embedding_norm", {h}, QType::BF16);
    const Id hidden_norm         = demands.direct("mtp/hidden_norm", {h}, QType::BF16);
    // eh_proj reads the [norm(e); norm(h)] concatenation as one [2H, T] input.
    const Id eh_proj             = demands.matrix("mtp/eh_proj", {h, 2 * h}, fusion);
    const Id attention_norm      = demands.direct("mtp/attention_norm", {h}, QType::BF16);
    const Id query               = demands.matrix("mtp/attention/query", {qr, h}, attn);
    const Id key                 = demands.matrix("mtp/attention/key", {kvr, h}, attn);
    const Id value               = demands.matrix("mtp/attention/value", {kvr, h}, attn);
    const Id attention_output    = demands.matrix("mtp/attention/output", {h, qr},
                                                  "mtp/attention/context");
    const Id post_attention_norm = demands.direct("mtp/post_attention_norm", {h}, QType::BF16);
    const Id router              = demands.matrix("mtp/mlp/router", {e, h}, ffn);
    const Id selection_bias      = demands.direct("mtp/mlp/selection_bias", {e}, QType::FP32);
    std::vector<Id> up, down;
    for (std::uint64_t x = 0; x < e; ++x) {
        const std::string ep = "mtp/mlp/experts/" + std::to_string(x) + "/";
        up.push_back(demands.matrix(ep + "up", {f, up_k}, ffn));
        down.push_back(demands.matrix(ep + "down", {h, f}, "mtp/mlp/expert_product"));
    }
    const Id shared_up   = demands.matrix("mtp/mlp/shared_up", {fs, h}, ffn);
    const Id shared_down = demands.matrix("mtp/mlp/shared_down", {h, fs}, "mtp/mlp/shared_product");
    const Id final_norm  = demands.direct("mtp/final_norm", {h}, QType::BF16);

    auto drafter      = std::unique_ptr<MtpDrafter>(new MtpDrafter());
    drafter->backing_ = demands.materialize(device);
    const Prepare prepare(demands, *drafter->backing_, false); // A16: a drafter keeps BF16 inputs
    auto& w          = drafter->weights_;
    w.rms_norm_eps   = c.rms_norm_eps;
    w.layer_norm_eps = c.layer_norm_eps;
    w.enorm          = prepare.tensor(embedding_norm);
    w.hnorm          = prepare.tensor(hidden_norm);
    w.eh_proj        = prepare.projection(eh_proj);
    w.attn_norm      = prepare.tensor(attention_norm);
    {
        AttentionWeights a;
        const std::array qkv{query, key, value};
        a.qkv = prepare.packed(qkv);
        if (!a.qkv) {
            a.query  = prepare.projection(query);
            a.key    = prepare.projection(key);
            a.value  = prepare.projection(value);
        }
        a.output     = prepare.projection(attention_output);
        w.attention  = std::move(a);
    }
    w.post_attention_norm = prepare.tensor(post_attention_norm);
    w.moe = prepare_moe(prepare, {router, selection_bias, shared_up, shared_down, up, down},
                        "mtp/mlp/experts");
    w.final_norm = prepare.tensor(final_norm);
    return drafter;
}

} // namespace ninferno::models::nemotron_h
