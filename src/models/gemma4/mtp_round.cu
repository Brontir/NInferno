#include "models/gemma4/mtp_round.h"

#include "ninferno/ops/argmax.h"
#include "ninferno/ops/embedding.h"
#include "ninferno/ops/gemma4_fused.h"
#include "ninferno/ops/linear_swiglu.h"
#include "ninferno/ops/rmsnorm.h"
#include "ninferno/ops/speculative_round.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <string>

namespace ninferno::models::gemma4 {
namespace {

// Opens an MTP round from the device-side position P of the pending token: the verify batch
// covers P..P+k, the drafter's queries rotate at P and see the target's keys through P-1.
__global__ void mtp_begin(const std::int32_t* position, const std::int32_t* token,
                          std::int32_t* verify_positions, std::int32_t* draft_ids,
                          std::int32_t* query_position, std::int32_t* key_position,
                          std::int32_t columns) {
    const std::int32_t p = *position;
    for (std::int32_t i = threadIdx.x; i < columns; i += blockDim.x) {
        verify_positions[i] = p + i;
    }
    if (threadIdx.x == 0) {
        draft_ids[0]    = *token;
        *query_position = p;
        *key_position   = p - 1;
    }
}

} // namespace

DrafterPass::DrafterPass(const Drafter& drafter, const Model& target, DeviceContext& device)
    : drafter_(drafter), target_(target), device_(device) {
    const auto& c = target.config();
    // The drafter shares the target's KV of its last sliding and last full layers.
    for (std::int32_t i = c.layers - 1; i >= 0; --i) {
        if (c.kinds[i] == LayerKind::Sliding && sliding_source_ < 0) { sliding_source_ = i; }
        if (c.kinds[i] == LayerKind::Full && full_source_ < 0) { full_source_ = i; }
    }
}

std::size_t DrafterPass::workspace_bytes(const Drafter& drafter, std::int32_t max_context,
                                         std::int32_t batch) {
    const auto& dc     = drafter.config();
    const auto& dw     = drafter.weights();
    const std::int64_t n = batch;
    std::size_t bytes  = 0;
    const auto add     = [&](std::int64_t elements) { bytes += bf16_bytes(elements) + 256; };
    add(2 * std::int64_t(dc.backbone_size) * n); // input
    add(std::int64_t(dc.backbone_size) * n);     // embedded
    add(2 * std::int64_t(dc.hidden_size) * n);   // x, hn
    std::int64_t layer = 0;
    for (const auto* g : {&dc.sliding, &dc.full}) {
        layer = std::max<std::int64_t>(layer, 3 * std::int64_t(dc.query_heads) * g->head_dim);
    }
    add(layer * n);                                        // q_raw, q, context
    add(2 * std::int64_t(dc.hidden_size) * n);             // branch, ffn input
    add(3 * std::int64_t(dc.intermediate_size) * n);       // gate/up, product
    add(std::int64_t(dw.head.weight.n) * n);               // logits
    std::size_t op = 0;
    const auto linear = [&](const Projection& p) {
        op = std::max(op, ops::linear_workspace_capacity_bytes(p, 1, batch));
    };
    linear(dw.pre_projection);
    linear(dw.post_projection);
    linear(dw.head);
    for (const auto& l : dw.layers) {
        for (const Projection* p : {&l.query, &l.output, &l.gate_up, &l.down}) { linear(*p); }
        if (ops::linear_geglu_supports(l.gate_up.weight)) {
            op = std::max(op, ops::linear_geglu_workspace_capacity_bytes(
                                  l.gate_up.weight.qtype, l.gate_up.weight.n, l.gate_up.weight.k,
                                  l.gate_up.policy, 1, batch) +
                                  256);
        }
    }
    const ops::Gemma4AttentionEnvelope envelope{static_cast<std::uint32_t>(max_context), 1};
    for (const auto* g : {&dc.sliding, &dc.full}) {
        op = std::max(op, ops::gemma4_attention_workspace_bytes(g->head_dim, dc.query_heads,
                                                                g->kv_heads, envelope));
    }
    return bytes + op + (std::size_t(16) << 20);
}

MtpRound::MtpRound(const Drafter& drafter, const Target& target, DeviceContext& device,
                   DeviceArena& work, std::int32_t max_context, std::int32_t max_drafts)
    : target_(target), device_(device), work_(work), max_context_(max_context),
      pass_(drafter, target.model(), device) {
    const auto& c   = target.model().config();
    draft_ids_      = DeviceBuffer(sizeof(std::int32_t) * (max_drafts + 1));
    carry_hidden_   = DeviceBuffer(bf16_bytes(c.hidden_size));
    query_position_ = DeviceBuffer(sizeof(std::int32_t));
    key_position_   = DeviceBuffer(sizeof(std::int32_t));
    draft_hidden_   = DeviceBuffer(bf16_bytes(c.hidden_size));
}

Tensor MtpRound::ids(std::int32_t columns) const {
    return Tensor(draft_ids_.p, DType::I32, {columns});
}

Tensor MtpRound::carry() const {
    return Tensor(carry_hidden_.p, DType::BF16, {target_.model().config().hidden_size, 1});
}

void MtpRound::draft(const std::int32_t* position, const std::int32_t* token,
                     std::int32_t* verify_positions, std::int32_t k) {
    mtp_begin<<<1, 32, 0, device_.stream>>>(position, token, verify_positions,
                                            static_cast<std::int32_t*>(draft_ids_.p),
                                            static_cast<std::int32_t*>(query_position_.p),
                                            static_cast<std::int32_t*>(key_position_.p), k + 1);
    CUDA_CHECK(cudaGetLastError());
    const Tensor carried = carry();
    Tensor drafted(draft_hidden_.p, DType::BF16, {carried.ne[0], 1});
    const Tensor qpos(query_position_.p, DType::I32, {1});
    const Tensor kpos(key_position_.p, DType::I32, {1});
    const ops::Gemma4AttentionEnvelope envelope{static_cast<std::uint32_t>(max_context_), 1};
    for (std::int32_t i = 0; i < k; ++i) {
        const Tensor token(static_cast<std::int32_t*>(draft_ids_.p) + i, DType::I32, {1});
        Tensor next(static_cast<std::int32_t*>(draft_ids_.p) + i + 1, DType::I32, {1});
        pass_.step(work_, token, i == 0 ? carried : drafted, qpos, kpos,
                   target_.kv(pass_.sliding_source()), target_.kv(pass_.full_source()), envelope,
                   next, drafted, i);
    }
}

void DrafterPass::step(DeviceArena& work_, const Tensor& tokens, const Tensor& hidden_in,
                       const Tensor& qpos, const Tensor& kpos, const ops::Gemma4KvLayer& sliding,
                       const ops::Gemma4KvLayer& full, ops::Gemma4AttentionEnvelope envelope,
                       Tensor& next, Tensor& carry_out, std::int32_t dump_step) const {
    const cudaStream_t s     = device_.stream;
    const auto& dc           = drafter_.config();
    const auto& dw           = drafter_.weights();
    const auto eps           = dc.rms_norm_eps;
    const std::int32_t h     = dc.hidden_size;
    const std::int32_t b     = dc.backbone_size;
    const std::int32_t batch = tokens.ne[0];
    auto scope               = work_.scope();
    // Each column's input is [target embedding of its token; its hidden state].
    Tensor input    = work_.alloc(DType::BF16, {2 * b, batch});
    Tensor embedded = work_.alloc(DType::BF16, {b, batch});
    target_.embed(tokens, embedded, s);
    const std::size_t row = bf16_bytes(b);
    CUDA_CHECK(cudaMemcpy2DAsync(input.data, 2 * row, embedded.data, row, row, batch,
                                 cudaMemcpyDeviceToDevice, s));
    CUDA_CHECK(cudaMemcpy2DAsync(static_cast<char*>(input.data) + row, 2 * row, hidden_in.data, row,
                                 row, batch, cudaMemcpyDeviceToDevice, s));
    Tensor x  = work_.alloc(DType::BF16, {h, batch});
    Tensor hn = work_.alloc(DType::BF16, {h, batch});
    project(input, dw.pre_projection, x, work_, s);
    ops::rmsnorm(x, dw.layers.front().input_norm, eps, false, hn, s);
    const Tensor attention_kpos = batch == 1 ? kpos : kpos.view({1, batch});
    for (std::int32_t l = 0; l < dc.layers; ++l) {
        const auto& layer       = dw.layers[l];
        const auto& g           = dc.geometry(layer.kind);
        const auto d            = g.head_dim;
        const Tensor& next_norm = l + 1 < dc.layers ? dw.layers[l + 1].input_norm : dw.final_norm;
        auto layer_scope        = work_.scope();
        Tensor q_raw            = work_.alloc(DType::BF16, {dc.query_heads * d, batch});
        project(hn, layer.query, q_raw, work_, s);
        Tensor q = work_.alloc(DType::BF16, {d, dc.query_heads, batch});
        ops::gemma4_q_prepare(q_raw, d, layer.query_norm, eps, qpos, g.rope_theta, g.rotary_pairs,
                              q, s);
        const ops::Gemma4KvLayer& source = layer.kind == LayerKind::Sliding ? sliding : full;
        Tensor context                   = work_.alloc(DType::BF16, {d, dc.query_heads, batch});
        {
            auto attention_scope = work_.scope();
            WorkspaceArena scratch(work_.alloc_bytes(
                std::max<std::size_t>(ops::gemma4_attention_workspace_bytes(
                                          d, dc.query_heads, g.kv_heads, envelope, source.window),
                                      256)));
            if (batch == 1) {
                ops::gemma4_attention(q, kpos, source, 1.0F, envelope, scratch, context, s);
            } else {
                const Tensor q_rows = q.view({d, dc.query_heads, 1, batch});
                Tensor context_rows = context.view({d, dc.query_heads, 1, batch});
                ops::gemma4_attention(q_rows, attention_kpos, source, 1.0F, envelope, scratch,
                                      context_rows, s);
            }
        }
        Tensor branch = work_.alloc(DType::BF16, {h, batch});
        project(context.view({dc.query_heads * d, batch}), layer.output, branch, work_, s);
        Tensor ffn_input = work_.alloc(DType::BF16, {h, batch});
        ops::gemma4_residual_norm(x, {branch}, layer.post_attention_norm, eps, 1.0F,
                                  {{layer.pre_ffn_norm, ffn_input}}, s);
        Tensor product = work_.alloc(DType::BF16, {dc.intermediate_size, batch});
        if (ops::linear_geglu_supports(layer.gate_up.weight)) {
            // GeGLU in the projection's epilogue, as the target's layers: no BF16 gate/up plane,
            // no second kernel.
            auto geglu_scope = work_.scope();
            WorkspaceArena scratch(work_.alloc_bytes(std::max<std::size_t>(
                ops::linear_geglu_workspace_capacity_bytes(
                    layer.gate_up.weight.qtype, layer.gate_up.weight.n, layer.gate_up.weight.k,
                    layer.gate_up.policy, batch, batch),
                256)));
            ops::linear_geglu(ffn_input, layer.gate_up.weight, product, layer.gate_up.policy,
                              scratch, s);
        } else {
            Tensor gate_up = work_.alloc(DType::BF16, {layer.gate_up.weight.n, batch});
            project(ffn_input, layer.gate_up, gate_up, work_, s);
            ops::gelu_tanh_mul(gate_up, product, s);
        }
        project(product, layer.down, branch, work_, s);
        ops::gemma4_residual_norm(x, {branch}, layer.post_ffn_norm, eps, layer.output_scale,
                                  {{next_norm, hn}}, s);
    }
    // hn now holds final_norm(x): the head input and the post-projection input.
    Tensor logits = work_.alloc(DType::BF16, {dw.head.weight.n, batch});
    project(hn, dw.head, logits, work_, s);
    if (dump_step >= 0 && debug_dumping(s)) { // an eager step (the round's warm-up): what the drafter read and drew
        const std::string step = "D" + std::to_string(dump_step);
        debug_dump(input.data, input.bytes(), (step + ".input").c_str());
        debug_dump(logits.data, logits.bytes(), (step + ".logits").c_str());
    }
    ops::argmax(logits, next, dw.head.weight.n, s);
    if (dw.head_token_ids.data != nullptr) {
        ops::proposal_remap_token_ids(
            next, static_cast<const std::int32_t*>(dw.head_token_ids.data), dw.head.weight.n, s);
    }
    project(hn, dw.post_projection, carry_out, work_, s);
}

} // namespace ninferno::models::gemma4
