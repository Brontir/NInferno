#include "models/speculation/dflash_round.h"

#include "core/debug_poison.h"
#include "ninferno/ops/argmax.h"
#include "ninferno/ops/gemma4_fused.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/residual_add.h"
#include "ninferno/ops/rmsnorm.h"
#include "ninferno/ops/silu_mul.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace ninferno::models::speculation {
namespace {

std::size_t bf16_bytes(std::int64_t elements) { return static_cast<std::size_t>(elements) * 2; }

// One drafter projection; every activation stays BF16 (the loader prepared no W4A4 route).
void project(const Tensor& x, const Projection& p, Tensor& out, DeviceArena& work,
             cudaStream_t stream) {
    auto scope = work.scope();
    ops::linear(x, p.weight, out, p.policy, work, stream);
}

// Opens a DFlash round from the pending token's position P: the drafter's block sits at
// P..P+block-1 and starts with that token; the verify batch covers P..P+k.
__global__ void dflash_begin(const std::int32_t* position, const std::int32_t* token,
                             std::int32_t* verify_positions, std::int32_t columns,
                             std::int32_t* draft_ids, std::int32_t* block_ids,
                             std::int32_t* block_positions, std::int32_t block) {
    const std::int32_t p = *position;
    for (std::int32_t i = threadIdx.x; i < block; i += blockDim.x) {
        block_positions[i] = p + i;
        if (i < columns) { verify_positions[i] = p + i; }
    }
    if (threadIdx.x == 0) {
        draft_ids[0] = *token;
        block_ids[0] = *token;
    }
}

} // namespace

DFlashRound::DFlashRound(const DFlashDrafter& drafter, DraftTarget& target, DeviceContext& device,
                         DeviceArena& work, std::int32_t max_context, std::int32_t prefill_chunk)
    : drafter_(drafter), target_(target), shape_(target.shape()), device_(device), work_(work),
      max_context_(key_span(drafter, max_context)) {
    const auto& dc = drafter.config();
    if (dc.hidden_size != shape_.hidden_size || dc.vocab_size != shape_.vocab_size ||
        dc.target_layers != shape_.layers) {
        throw std::invalid_argument("dflash: the drafter does not pair with this target");
    }
    // This round prepares q/k with a plain rope of rope_theta over D128 heads without biases; a
    // drafter with biases, a resolved (scaled) rope or 64-wide heads (gpt-oss's) runs in the
    // serving Engine's round (models/dense/dflash.cpp).
    if (dc.attention_bias || dc.rope_scaled || dc.head_dim != 128) {
        throw std::invalid_argument(
            "dflash: this drafter (attention biases, a scaled rope or D64 heads) runs through the "
            "serving Engine (ninferno serve / the ninferno CLI), not this session");
    }
    // One append call carries a prefill chunk or a verified block; a sliding ring keeps
    // window + that many slots, as the target's rings do.
    const std::int32_t widest = std::max(prefill_chunk, dc.block_size);
    // DFlash drafts a whole block, of which a round verifies the first k.
    draft_ids_ = DeviceBuffer(sizeof(std::int32_t) * dc.block_size);
    kv_.reserve(dc.layers);
    for (std::int32_t i = 0; i < dc.layers; ++i) {
        const std::int32_t window = dc.windows[i];
        // As the target's rings: window + widest slots even for a shorter context.
        const std::int32_t capacity = window > 0 ? window + widest : max_context_;
        const auto bytes = bf16_bytes(std::int64_t(dc.head_dim) * capacity * dc.kv_heads);
        Planes planes;
        planes.k = DeviceBuffer(bytes);
        planes.v = DeviceBuffer(bytes);
        if (const auto poison = debug_poison_byte()) {
            planes.k.fill(*poison);
            planes.v.fill(*poison);
        }
        planes.view.k      = Tensor(planes.k.p, DType::BF16, {dc.head_dim, capacity, dc.kv_heads});
        planes.view.v      = Tensor(planes.v.p, DType::BF16, {dc.head_dim, capacity, dc.kv_heads});
        planes.view.window = window;
        kv_.push_back(std::move(planes));
    }
    const auto taps = static_cast<std::int64_t>(dc.taps.size());
    features_       = DeviceBuffer(bf16_bytes(taps * shape_.hidden_size * widest));
    tap_slot_.assign(shape_.layers, -1);
    for (std::size_t j = 0; j < dc.taps.size(); ++j) {
        tap_slot_[dc.taps[j]] = static_cast<std::int32_t>(j);
    }
    std::vector<std::int32_t> ids(dc.block_size, dc.mask_token);
    block_ids_ = DeviceBuffer(sizeof(std::int32_t) * dc.block_size);
    block_ids_.copy_from_host(ids.data(), ids.size() * sizeof(std::int32_t));
    block_positions_ = DeviceBuffer(sizeof(std::int32_t) * dc.block_size);
}

std::int32_t DFlashRound::key_span(const DFlashDrafter& drafter, std::int32_t max_context) {
    return max_context + drafter.config().block_size;
}

std::size_t DFlashRound::workspace_bytes(const DFlashDrafter& drafter, std::int32_t max_context,
                                         std::int32_t prefill_chunk) {
    max_context          = key_span(drafter, max_context);
    const auto& dc       = drafter.config();
    const auto& dw       = drafter.weights();
    const std::int64_t t = std::max(prefill_chunk, dc.block_size);
    const std::int64_t h = dc.hidden_size;
    const std::int64_t b = dc.block_size;
    std::size_t bytes    = 0;
    const auto add       = [&](std::int64_t elements) { bytes += bf16_bytes(elements) + 256; };
    const std::int64_t kv_rows = std::int64_t(dc.kv_heads) * dc.head_dim;
    const std::int64_t q_rows  = std::int64_t(dc.query_heads) * dc.head_dim;
    add(2 * h * t);
    add(2 * kv_rows * dc.layers * t);
    add(2 * kv_rows * t);
    add(4 * h * b + 2 * std::int64_t(dc.intermediate_size) * b * 2);
    add((q_rows + 2 * kv_rows) * b * 2);
    add(std::int64_t(dc.vocab_size) * b);
    // Unpacked siblings each take their own output (at most three at once); the carving slack.
    bytes += 3 * 256;
    std::size_t op = 0;
    std::vector<const Projection*> wide{&dw.feature_projection};
    if (dw.context_kv) {
        wide.push_back(&*dw.context_kv);
    } else {
        for (const auto& layer : dw.layers) {
            wide.push_back(&layer.context_key);
            wide.push_back(&layer.context_value);
        }
    }
    for (const Projection* p : wide) {
        op = std::max(op, ops::linear_workspace_capacity_bytes(p->weight.qtype, p->weight.n,
                                                               p->weight.k, p->policy, 1,
                                                               static_cast<std::int32_t>(t)));
    }
    const ops::Gemma4AttentionEnvelope envelope{static_cast<std::uint32_t>(max_context),
                                                dc.block_size};
    for (const std::int32_t window : {dc.sliding_window, 0}) {
        op = std::max(op, ops::gemma4_attention_workspace_bytes(dc.head_dim, dc.query_heads,
                                                                dc.kv_heads, envelope, window));
    }
    return bytes + op;
}

FeatureTap DFlashRound::capture(std::int32_t t) const {
    const auto taps = static_cast<std::int32_t>(drafter_.config().taps.size());
    return FeatureTap{tap_slot_, Tensor(features_.p, DType::BF16, {taps * shape_.hidden_size, t})};
}

Tensor DFlashRound::ids(std::int32_t columns) const {
    return Tensor(draft_ids_.p, DType::I32, {columns});
}

Tensor DFlashRound::prefill(const Tensor& ids, const Tensor& pos,
                            ops::Gemma4AttentionEnvelope envelope) {
    const FeatureTap tapped = capture(ids.ne[0]);
    Tensor normed = target_.draft_forward(ids, pos, envelope, /*a16=*/false, &tapped);
    append_context(tapped.sink, pos);
    return normed;
}

void DFlashRound::append_context(const Tensor& features, const Tensor& pos) {
    const cudaStream_t s       = device_.stream;
    const auto& dc             = drafter_.config();
    const auto& dw             = drafter_.weights();
    const std::int32_t h       = shape_.hidden_size;
    const std::int32_t t       = features.ne[1];
    const std::int32_t kv_rows = dc.kv_heads * dc.head_dim;
    auto scope                 = work_.scope();
    Tensor projected           = work_.alloc(DType::BF16, {h, t});
    project(features, dw.feature_projection, projected, work_, s);
    Tensor context = work_.alloc(DType::BF16, {h, t});
    ops::rmsnorm(projected, dw.context_norm, dc.rms_norm_eps, false, context, s);
    // Packed: every layer's keys and values in one projection, rows [k0; v0; k1; v1; ...].
    Tensor kv_rows_all;
    if (dw.context_kv) {
        kv_rows_all = work_.alloc(DType::BF16, {2 * kv_rows * dc.layers, t});
        project(context, *dw.context_kv, kv_rows_all, work_, s);
    }
    for (std::int32_t l = 0; l < dc.layers; ++l) {
        auto layer_scope = work_.scope();
        Tensor k_src, v_src;
        if (dw.context_kv) {
            k_src = kv_rows_all.slice(0, 2 * l * kv_rows, kv_rows);
            v_src = kv_rows_all.slice(0, (2 * l + 1) * kv_rows, kv_rows);
        } else {
            // Each its own contiguous output, as the projection writes it.
            k_src = work_.alloc(DType::BF16, {kv_rows, t});
            v_src = work_.alloc(DType::BF16, {kv_rows, t});
            project(context, dw.layers[l].context_key, k_src, work_, s);
            project(context, dw.layers[l].context_value, v_src, work_, s);
        }
        ops::gemma4_dflash_qkv_prepare(nullptr, k_src, v_src, nullptr, dw.layers[l].key_norm,
                                       dc.rms_norm_eps, pos, dc.rope_theta, nullptr, kv_[l].view,
                                       s);
    }
}

void DFlashRound::draft(const std::int32_t* position, const std::int32_t* token,
                        std::int32_t* verify_positions, std::int32_t k) {
    const auto block = drafter_.config().block_size;
    dflash_begin<<<1, 32, 0, device_.stream>>>(
        position, token, verify_positions, k + 1, static_cast<std::int32_t*>(draft_ids_.p),
        static_cast<std::int32_t*>(block_ids_.p), static_cast<std::int32_t*>(block_positions_.p),
        block);
    CUDA_CHECK(cudaGetLastError());
    block_pass();
}

void DFlashRound::block_pass() {
    const cudaStream_t s       = device_.stream;
    const auto& dc             = drafter_.config();
    const auto& dw             = drafter_.weights();
    const std::int32_t b       = dc.block_size;
    const std::int32_t d       = dc.head_dim;
    const std::int32_t hidden  = shape_.hidden_size;
    const std::int32_t q_rows  = dc.query_heads * d;
    const std::int32_t kv_rows = dc.kv_heads * d;
    const auto eps             = dc.rms_norm_eps;
    auto scope                 = work_.scope();
    Tensor ids(block_ids_.p, DType::I32, {b});
    Tensor pos(block_positions_.p, DType::I32, {b});
    Tensor x  = work_.alloc(DType::BF16, {hidden, b});
    Tensor hn = work_.alloc(DType::BF16, {hidden, b});
    // The target's table as the drafter reads it (Gemma 4's is stored pre-scaled by sqrt(H);
    // an embedding norm the target applies after the table is not part of it).
    target_.draft_embedding(ids, x);
    const ops::Gemma4AttentionEnvelope envelope{static_cast<std::uint32_t>(max_context_), b};
    const float scale = 1.0F / std::sqrt(static_cast<float>(d));
    for (std::int32_t l = 0; l < dc.layers; ++l) {
        const auto& layer = dw.layers[l];
        auto layer_scope  = work_.scope();
        ops::rmsnorm(x, layer.input_norm, eps, false, hn, s);
        Tensor q_src, k_src, v_src;
        if (layer.qkv) {
            Tensor packed = work_.alloc(DType::BF16, {q_rows + 2 * kv_rows, b});
            project(hn, *layer.qkv, packed, work_, s);
            q_src = packed.slice(0, 0, q_rows);
            k_src = packed.slice(0, q_rows, kv_rows);
            v_src = packed.slice(0, q_rows + kv_rows, kv_rows);
        } else {
            q_src = work_.alloc(DType::BF16, {q_rows, b});
            k_src = work_.alloc(DType::BF16, {kv_rows, b});
            v_src = work_.alloc(DType::BF16, {kv_rows, b});
            project(hn, layer.query, q_src, work_, s);
            project(hn, layer.key, k_src, work_, s);
            project(hn, layer.value, v_src, work_, s);
        }
        // The block's own keys and values go where the committed rows of P.. will later be
        // written; the next context append overwrites them before any round reads them as
        // context.
        Tensor q = work_.alloc(DType::BF16, {d, dc.query_heads, b});
        ops::gemma4_dflash_qkv_prepare(&q_src, k_src, v_src, &layer.query_norm, layer.key_norm, eps,
                                       pos, dc.rope_theta, &q, kv_[l].view, s);
        Tensor context = work_.alloc(DType::BF16, {d, dc.query_heads, b});
        {
            auto attention_scope = work_.scope();
            WorkspaceArena scratch(work_.alloc_bytes(std::max<std::size_t>(
                std::max(ops::gemma4_attention_workspace_bytes(d, dc.query_heads, dc.kv_heads,
                                                               envelope, dc.sliding_window),
                         ops::gemma4_attention_workspace_bytes(d, dc.query_heads, dc.kv_heads,
                                                               envelope, 0)),
                256)));
            // Full layers see the whole block; sliding layers see it too when the drafter was
            // trained so (sliding_block), else they are causal inside it.
            const auto mask = layer.window == 0 || dc.sliding_block
                                  ? ops::Gemma4AttentionMask::Block
                                  : ops::Gemma4AttentionMask::Causal;
            ops::gemma4_attention(q, pos, kv_[l].view, scale, envelope, scratch, context, s, mask);
        }
        Tensor branch = work_.alloc(DType::BF16, {hidden, b});
        project(context.view({q_rows, b}), layer.output, branch, work_, s);
        ops::residual_add(branch, x, s);
        ops::rmsnorm(x, layer.post_attention_norm, eps, false, hn, s);
        const std::int32_t inter = dc.intermediate_size;
        Tensor gate_up           = work_.alloc(DType::BF16, {2 * inter, b});
        Tensor gate              = gate_up.slice(0, 0, inter);
        Tensor up                = gate_up.slice(0, inter, inter);
        if (layer.gate_up) {
            project(hn, *layer.gate_up, gate_up, work_, s);
        } else {
            // Contiguous halves of the same buffer: [gate | up] by columns, not rows.
            gate = Tensor(gate_up.data, DType::BF16, {inter, b});
            up   = Tensor(static_cast<char*>(gate_up.data) + bf16_bytes(std::int64_t(inter) * b),
                          DType::BF16, {inter, b});
            project(hn, layer.gate, gate, work_, s);
            project(hn, layer.up, up, work_, s);
        }
        Tensor product = work_.alloc(DType::BF16, {inter, b});
        ops::silu_mul(gate, up, product, s);
        project(product, layer.down, branch, work_, s);
        ops::residual_add(branch, x, s);
    }
    // Column 0 re-reads the pending token; columns 1.. are the drafts.
    Tensor tail   = x.slice(1, 1, b - 1);
    Tensor normed = work_.alloc(DType::BF16, {hidden, b - 1});
    ops::rmsnorm(tail, dw.final_norm, eps, false, normed, s);
    Tensor logits = work_.alloc(DType::BF16, {shape_.vocab_size, b - 1});
    target_.greedy_logits(normed, logits, /*a16=*/false);
    Tensor drafts(static_cast<std::int32_t*>(draft_ids_.p) + 1, DType::I32, {b - 1});
    ops::argmax(logits, drafts, shape_.vocab_size, s);
}

} // namespace ninferno::models::speculation
