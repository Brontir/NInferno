#include "models/nemotron_h/pass.h"

#include "core/nvtx.h"
#include "ninferno/ops/dense_fused.h"
#include "ninferno/ops/argmax.h"
#include "ninferno/ops/embedding.h"
#include "ninferno/ops/layer_norm.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/linear_add.h"
#include "ninferno/ops/mamba2.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/rmsnorm.h"
#include "ninferno/ops/sampling.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::models::nemotron_h {
namespace {

std::size_t bf16_bytes(std::int64_t elements) { return static_cast<std::size_t>(elements) * 2; }

void project_add(DeviceArena& work, const Tensor& x, const Projection& p, Tensor& residual,
                 cudaStream_t stream) {
    auto scope = work.scope();
    ops::linear_add(x, p.weight, residual, p.policy, work, stream);
}

} // namespace

Pass::Pass(const Model& model, DeviceContext& device)
    : model_(model), c_(model.config()), device_(device) {}

std::size_t Pass::workspace_bytes(const Model& model, std::int32_t max_context, std::int32_t tokens,
                                  std::int32_t head_columns) {
    const auto& c        = model.config();
    const std::int64_t t = tokens;
    const std::int64_t h = c.hidden_size;
    const std::int64_t q = std::int64_t(c.query_heads) * c.head_dim;
    const std::int64_t kv = std::int64_t(c.kv_heads) * c.head_dim;
    const auto& m        = c.mamba;
    std::size_t bytes    = 0;
    const auto add       = [&](std::int64_t elements) { bytes += bf16_bytes(elements) + 256; };
    add(h * t); // x
    add(h * t); // normed input
    // The widest mixer's activations: Mamba2 (in_proj, conv output, y, gated) or attention
    // (projections, prepared queries, context).
    const std::int64_t mamba_act = m.in_rows() + m.conv_channels() + 2LL * m.inner();
    const std::int64_t attention_act = (q + 2 * kv) + 2 * q;
    bytes += bf16_bytes(std::max(mamba_act, attention_act) * t) + 4 * 256;
    add(std::int64_t(c.vocab_size) * std::max<std::int64_t>(head_columns, 1));
    std::size_t op = 0;
    for (const auto& layer : model.weights().layers) {
        if (layer.mamba) {
            op =
                std::max(op, ops::linear_workspace_capacity_bytes(layer.mamba->in_proj, 1, tokens));
            op = std::max(
                op, ops::linear_add_workspace_capacity_bytes(layer.mamba->out_proj, 1, tokens));
        } else if (layer.attention) {
            const auto& a = *layer.attention;
            if (a.qkv) {
                op = std::max(op, ops::linear_workspace_capacity_bytes(*a.qkv, 1, tokens));
            } else {
                for (const Projection* p : {&a.query, &a.key, &a.value}) {
                    op = std::max(op, ops::linear_workspace_capacity_bytes(*p, 1, tokens));
                }
            }
            op = std::max(op, ops::linear_add_workspace_capacity_bytes(a.output, 1, tokens));
        } else if (layer.moe) {
            op = std::max(op, ops::moe_workspace_bytes(*layer.moe, c.moe.op, 1, tokens));
        }
    }
    op = std::max(op, ops::linear_workspace_capacity_bytes(model.weights().output_head, 1,
                                                           std::max(head_columns, 1)));
    const ops::Gemma4AttentionEnvelope envelope{static_cast<std::uint32_t>(max_context), tokens};
    op = std::max(op, ops::gemma4_attention_workspace_bytes(c.head_dim, c.query_heads, c.kv_heads,
                                                            envelope, 0));
    op = std::max(op, ops::sampling_workspace_capacity_bytes(c.vocab_size, 1, 1));
    return bytes + op + (std::size_t(64) << 20);
}

void Pass::project(DeviceArena& work, const Tensor& x, const Projection& p, Tensor& out) const {
    auto scope = work.scope();
    ops::linear(x, p.weight, out, p.policy, work, device_.stream);
}

void Pass::mamba(DeviceArena& work, const Layer& layer, const Tensor& input, const Bindings& state,
                 Tensor& residual) const {
    const cudaStream_t s = device_.stream;
    const auto& m        = *layer.mamba;
    const auto& spec     = c_.mamba;
    const std::int32_t t = input.ne[1];
    const std::int32_t inner = spec.inner();
    const std::int32_t convc = spec.conv_channels();
    auto scope               = work.scope();
    Tensor proj              = work.alloc(DType::BF16, {spec.in_rows(), t});
    // The conv joins the input projection when its launch released it (programmatic dependent
    // launch): the conv's window and taps load under the projection's tail.
    bool released = false;
    {
        auto proj_scope = work.scope();
        released =
            ops::linear_release_next(input, m.in_proj.weight, proj, m.in_proj.policy, work, s);
    }
    const Tensor z   = proj.slice(0, 0, inner);
    const Tensor xbc = proj.slice(0, inner, convc);
    const Tensor dt  = proj.slice(0, inner + convc, spec.geometry.heads);
    const LinearAttentionStateLayerView pool = state.mamba->layer_view(layer.index);
    Tensor conv_states = pool.conv;
    Tensor recurrent   = pool.recurrent;
    Tensor conv_out, input_record, dt_record;
    if (state.records != nullptr) {
        // Verify: the conv writes its output into the records and records its inputs, the scan
        // records dt; neither touches the slots (replay_fold commits the accepted columns).
        const GdnReplayRecordLayer rec =
            state.records->layer(static_cast<std::int32_t>(layer.index), state.batch);
        if (static_cast<std::int64_t>(rec.conv.ne[1]) * state.batch != t) {
            throw std::invalid_argument("nemotron_h verify: record width * batch != T");
        }
        input_record = rec.conv.view({convc, t});
        conv_out     = rec.value.view({convc, t});
        dt_record    = rec.key.view({spec.geometry.heads, t});
    } else {
        conv_out = work.alloc(DType::BF16, {convc, t});
    }
    const bool record = state.records != nullptr;
    ops::mamba2_conv(xbc, m.conv_weight, m.conv_bias, conv_states, state.slots, state.batch,
                     conv_out, s, record ? &input_record : nullptr, released);
    Tensor y = work.alloc(DType::BF16, {inner, t});
    ops::mamba2_scan(conv_out, dt, m.dt_bias, m.a_log, m.d, recurrent, state.slots, state.batch,
                     spec.geometry, y, s, !record, record ? &dt_record : nullptr);
    Tensor gated = work.alloc(DType::BF16, {inner, t});
    ops::mamba2_gated_norm(y, z, m.norm, spec.norm_group(), c_.rms_norm_eps, gated, s);
    project_add(work, gated, m.out_proj, residual, s);
}

void attention_block(DeviceArena& work, const Config& c, const AttentionWeights& a,
                     const Tensor& input, const Tensor& pos, const ops::Gemma4KvLayer& kv,
                     std::int32_t batch, ops::Gemma4AttentionEnvelope envelope, Tensor& residual,
                     cudaStream_t s) {
    const std::int32_t t   = input.ne[1];
    const std::int32_t d   = c.head_dim;
    const std::int32_t qr  = c.query_heads * d;
    const std::int32_t kvr = c.kv_heads * d;
    const auto project     = [&](const Tensor& x, const Projection& p, Tensor& out) {
        auto scope = work.scope();
        ops::linear(x, p.weight, out, p.policy, work, s);
    };
    auto scope = work.scope();
    ops::DenseQkvPrepare prep;
    if (a.qkv) {
        Tensor packed = work.alloc(DType::BF16, {a.qkv->weight.n, t});
        project(input, *a.qkv, packed);
        prep.q_src = packed.slice(0, 0, qr);
        prep.k_src = packed.slice(0, qr, kvr);
        prep.v_src = packed.slice(0, qr + kvr, kvr);
    } else {
        prep.q_src = work.alloc(DType::BF16, {qr, t});
        prep.k_src = work.alloc(DType::BF16, {kvr, t});
        prep.v_src = work.alloc(DType::BF16, {kvr, t});
        project(input, a.query, prep.q_src);
        project(input, a.key, prep.k_src);
        project(input, a.value, prep.v_src);
    }
    // NoPE, no q/k norm: the prepare only lays the queries out and appends the keys and values.
    prep.qk_norm   = ops::QkNorm::None;
    prep.eps       = c.rms_norm_eps;
    prep.positions = pos;
    prep.q_out     = work.alloc(DType::BF16, {d, c.query_heads, t});
    prep.cache     = kv;
    ops::dense_qkv_prepare(prep, s);
    Tensor context = work.alloc(DType::BF16, {d, c.query_heads, t});
    {
        auto attention_scope = work.scope();
        WorkspaceArena scratch(work.alloc_bytes(
            std::max<std::size_t>(ops::gemma4_attention_batch_workspace_bytes(
                                      work, batch, d, c.query_heads, c.kv_heads, envelope, 0),
                                  256)));
        if (batch == 1) {
            ops::gemma4_attention(prep.q_out, pos, prep.cache, c.attention_scale, envelope,
                                  scratch, context, s);
        } else {
            const std::int32_t row_tokens = t / batch;
            const Tensor q_rows = prep.q_out.view({d, c.query_heads, row_tokens, batch});
            Tensor context_rows = context.view({d, c.query_heads, row_tokens, batch});
            ops::gemma4_attention(q_rows, pos.view({row_tokens, batch}), prep.cache,
                                  c.attention_scale, envelope, scratch, context_rows, s);
        }
    }
    project_add(work, context.view({qr, t}), a.output, residual, s);
}

void Pass::attention(DeviceArena& work, const Layer& layer, const Tensor& input, const Tensor& pos,
                     const Bindings& state, ops::Gemma4AttentionEnvelope envelope,
                     Tensor& residual) const {
    attention_block(work, c_, *layer.attention, input, pos, state.attention[layer.index],
                    state.batch, envelope, residual, device_.stream);
}

Tensor Pass::forward(DeviceArena& work, const Tensor& ids, const Tensor& pos, const Bindings& state,
                     ops::Gemma4AttentionEnvelope envelope) const {
    const cudaStream_t s = device_.stream;
    const std::int32_t t = ids.ne[0];
    if (state.batch < 1 || t % state.batch != 0 || state.mamba == nullptr ||
        state.attention.size() != c_.count(Block::Attention)) {
        throw std::invalid_argument("nemotron_h forward: bindings do not match the layers or T");
    }
    const auto& w = model_.weights();
    Tensor x      = work.alloc(DType::BF16, {c_.hidden_size, t});
    Tensor h      = work.alloc(DType::BF16, {c_.hidden_size, t});
    ops::embedding(ids, w.token_embedding, x, s);
    ops::rmsnorm(x, w.layers.front().norm, c_.rms_norm_eps, false, h, s);
    for (std::int32_t i = 0; i < c_.layers; ++i) {
        const auto& layer = w.layers[i];
        auto scope        = work.scope();
        try {
            nvtx::ScopedRange range(layer.kind == Block::Attention ? nvtx::Name::PrefillLayerFull
                                                                   : nvtx::Name::PrefillLayerGdn,
                                    nvtx::Category::Attention, i);
            // Every block adds into x in its last kernel's epilogue (the output projection's, the
            // MoE combine's); the next block's input norm then reads x alone.
            switch (layer.kind) {
            case Block::Mamba:
                mamba(work, layer, h, state, x);
                break;
            case Block::Attention:
                attention(work, layer, h, pos, state, envelope, x);
                break;
            case Block::Moe: {
                auto moe_scope = work.scope();
                WorkspaceArena scratch(work.alloc_bytes(std::max<std::size_t>(
                    ops::moe_workspace_bytes(*layer.moe, c_.moe.op, t, t), 256)));
                // A speculative verify keeps plain launches in its MoE, as the drafter does:
                // with PDL the round lost 2.7 % at C=4 on the RTX 5090.
                ops::MoeConfig op = c_.moe.op;
                op.pdl            = state.records == nullptr;
                ops::moe(h, *layer.moe, op, ops::MoeEpilogue::AddResidual, x, scratch, s);
                break;
            }
            }
            const bool last     = i + 1 == c_.layers;
            const Tensor& next  = last ? w.final_norm : w.layers[i + 1].norm;
            ops::rmsnorm(x, next, c_.rms_norm_eps, false, h, s);
        } catch (const std::exception&) {
            rethrow_with_context("text/layers/" + std::to_string(i) + " T=" + std::to_string(t));
        }
    }
    return h;
}

void Pass::head(DeviceArena& work, const Tensor& normed, Tensor& out) const {
    project(work, normed, model_.weights().output_head, out);
}

MtpPass::MtpPass(const Model& target, const MtpWeights& mtp, DeviceContext& device)
    : target_(target), mtp_(mtp), c_(target.config()), device_(device) {}

std::size_t MtpPass::workspace_bytes(const Model& target, const MtpWeights& mtp,
                                     std::int32_t max_context, std::int32_t tokens) {
    const auto& c        = target.config();
    const std::int64_t t = tokens;
    const std::int64_t h = c.hidden_size;
    const std::int64_t q = std::int64_t(c.query_heads) * c.head_dim;
    const std::int64_t kv = std::int64_t(c.kv_heads) * c.head_dim;
    std::size_t bytes    = 0;
    const auto add       = [&](std::int64_t elements) { bytes += bf16_bytes(elements) + 256; };
    add(2 * h * t);                    // [norm(e); norm(h)]
    add(3 * h * t);                    // e and the two norms
    add(h * t);                        // normed
    add((q + 2 * kv) * t + 2 * q * t); // attention projections, prepared queries, context
    add(h * t);                        // LayerNorm output (propose)
    bytes += 512;                      // zero LayerNorm bias
    std::size_t op = ops::linear_workspace_capacity_bytes(mtp.eh_proj, 1, tokens);
    const auto& a  = mtp.attention;
    if (a.qkv) {
        op = std::max(op, ops::linear_workspace_capacity_bytes(*a.qkv, 1, tokens));
    } else {
        for (const Projection* p : {&a.query, &a.key, &a.value}) {
            op = std::max(op, ops::linear_workspace_capacity_bytes(*p, 1, tokens));
        }
    }
    op = std::max(op, ops::linear_add_workspace_capacity_bytes(a.output, 1, tokens));
    op = std::max(op, ops::moe_workspace_bytes(mtp.moe, c.moe.op, 1, tokens));
    op =
        std::max(op, ops::linear_workspace_capacity_bytes(target.weights().output_head, 1, tokens));
    const ops::Gemma4AttentionEnvelope envelope{static_cast<std::uint32_t>(max_context), tokens};
    op = std::max(op, ops::gemma4_attention_workspace_bytes(c.head_dim, c.query_heads, c.kv_heads,
                                                            envelope, 0));
    return bytes + op + 4096;
}

void MtpPass::forward(DeviceArena& work, const Tensor& ids, const Tensor& hidden,
                      const Tensor& pos, const ops::Gemma4KvLayer& kv, std::int32_t batch,
                      ops::Gemma4AttentionEnvelope envelope, Tensor& out) const {
    const cudaStream_t s = device_.stream;
    const std::int32_t t = ids.ne[0];
    const std::int32_t H = c_.hidden_size;
    if (batch < 1 || t % batch != 0 || hidden.ne[0] != H || hidden.ne[1] != t ||
        out.ne[0] != H || out.ne[1] != t) {
        throw std::invalid_argument("nemotron_h MTP forward: shapes do not match T");
    }
    const float eps = mtp_.rms_norm_eps; // the drafter's own
    auto scope      = work.scope();
    // [norm(e); norm(h)] as one [2H, T] input of the fusion projection (the norms write
    // contiguous planes; one strided copy interleaves them per column).
    Tensor fused = work.alloc(DType::BF16, {2 * H, t});
    {
        auto embed_scope = work.scope();
        Tensor e         = work.alloc(DType::BF16, {H, t});
        Tensor e_normed  = work.alloc(DType::BF16, {H, t});
        Tensor h_normed  = work.alloc(DType::BF16, {H, t});
        ops::embedding(ids, target_.weights().token_embedding, e, s);
        ops::rmsnorm(e, mtp_.enorm, eps, false, e_normed, s);
        ops::rmsnorm(hidden, mtp_.hnorm, eps, false, h_normed, s);
        const std::size_t row = sizeof(std::uint16_t) * static_cast<std::size_t>(H);
        auto* base            = static_cast<std::uint8_t*>(fused.data);
        CUDA_CHECK(cudaMemcpy2DAsync(base, 2 * row, e_normed.data, row, row, t,
                                     cudaMemcpyDeviceToDevice, s));
        CUDA_CHECK(cudaMemcpy2DAsync(base + row, 2 * row, h_normed.data, row, row, t,
                                     cudaMemcpyDeviceToDevice, s));
    }
    {
        auto p = work.scope();
        ops::linear(fused, mtp_.eh_proj.weight, out, mtp_.eh_proj.policy, work, s);
    }
    Tensor normed = work.alloc(DType::BF16, {H, t});
    ops::rmsnorm(out, mtp_.attn_norm, eps, false, normed, s);
    // u += attention (the output projection's epilogue); normed = rmsnorm(u, post_attention_norm)
    attention_block(work, c_, mtp_.attention, normed, pos, kv, batch, envelope, out, s);
    ops::rmsnorm(out, mtp_.post_attention_norm, eps, false, normed, s);
    {
        auto moe_scope = work.scope();
        WorkspaceArena scratch(work.alloc_bytes(std::max<std::size_t>(
            ops::moe_workspace_bytes(mtp_.moe, c_.moe.op, t, t), 256)));
        // Plain launches in the drafter's MoE: with PDL the speculative round lost 1-3 % on the
        // RTX 5090, the waits outweighing the overlap in a one-layer pass.
        ops::MoeConfig op = c_.moe.op;
        op.pdl            = false;
        ops::moe(normed, mtp_.moe, op, ops::MoeEpilogue::AddResidual, out, scratch, s);
    }
}

void MtpPass::propose(DeviceArena& work, const Tensor& u, Tensor& logits, Tensor& tokens,
                      std::int32_t public_tokens) const {
    const cudaStream_t s = device_.stream;
    const std::int32_t b = u.ne[1];
    auto scope           = work.scope();
    Tensor bias          = work.alloc(DType::BF16, {c_.hidden_size});
    CUDA_CHECK(cudaMemsetAsync(bias.data, 0, bias.bytes(), s));
    Tensor normed = work.alloc(DType::BF16, {c_.hidden_size, b});
    ops::layer_norm(u, mtp_.final_norm, bias, mtp_.layer_norm_eps, normed, s);
    {
        auto p          = work.scope();
        const auto& out = target_.weights().output_head;
        ops::linear(normed, out.weight, logits, out.policy, work, s);
    }
    ops::argmax(logits, tokens, public_tokens, s);
}

} // namespace ninferno::models::nemotron_h
