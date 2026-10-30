#include "models/gemma4/target.h"

#include "core/debug_poison.h"
#include "core/nvtx.h"
#include "models/common/layer_dump.h"
#include "ninferno/ops/embedding.h"
#include "ninferno/ops/gemma4_fused.h"
#include "ninferno/ops/linear_swiglu.h"
#include "ninferno/ops/logit_softcap.h"
#include "ninferno/ops/rmsnorm.h"
#include "ninferno/ops/sampling.h"
#include "ninferno/ops/segmented_projection.h"
#include "ops/linear/linear_registry.h"

#include <algorithm>
#include <cstdio>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::models::gemma4 {
namespace {

// Debug aid: NINFERNO_GEMMA4_DUMP=<dir> appends, per layer and per intermediate, the token columns
// whose absolute position lies in NINFERNO_GEMMA4_DUMP_ROWS=lo:hi (default: all) to
// <dir>/L<layer>.<kind>.bin. Eager passes only (a captured graph is skipped); it synchronizes
// the device, so it is for localizing numerics, never for timing.
const LayerDump* dump_config() {
    static const std::optional<LayerDump> config = read_layer_dump("NINFERNO_GEMMA4_DUMP");
    return config ? &*config : nullptr;
}

// A pass dumps when the switch is on and its stream is not being captured.
bool dumping(cudaStream_t stream) { return dumps_on(dump_config(), stream); }

// Appends the columns of a token-major buffer (`column_bytes` per token, `t` tokens starting at
// absolute position p0) that fall in the configured row window.
void dump_columns(const void* data, std::size_t column_bytes, std::int32_t t, long long p0,
                  std::int32_t layer, const char* kind) {
    char name[64];
    std::snprintf(name, sizeof(name), "/L%02d.%s.bin", layer, kind);
    write_device_rows(*dump_config(), data, column_bytes, t, p0, dump_config()->dir + name, true);
}

void dump_tensor(const Tensor& tensor, std::int32_t t, long long p0, std::int32_t layer,
                 const char* kind) {
    dump_columns(tensor.data, tensor.bytes() / static_cast<std::size_t>(t), t, p0, layer, kind);
}

} // namespace

bool debug_dumping(cudaStream_t stream) { return dumping(stream); }

void debug_dump(const void* device_data, std::size_t bytes, const char* name) {
    write_device_bytes(device_data, bytes, dump_config()->dir + "/" + name + ".bin", true);
}

KvPlanes allocate_kv_planes(std::int32_t head_dim, std::int32_t capacity, std::int32_t kv_heads,
                            std::int32_t window, bool fp8) {
    const std::int64_t values = std::int64_t(head_dim) * capacity * kv_heads;
    const auto bytes          = fp8 ? static_cast<std::size_t>(values) : bf16_bytes(values);
    const DType dtype         = fp8 ? DType::FP8_E4M3FN : DType::BF16;
    KvPlanes planes;
    planes.k           = DeviceBuffer(bytes);
    planes.v           = DeviceBuffer(bytes);
    if (fp8) {
        const std::size_t scale_bytes = sizeof(float) * std::size_t(capacity) * kv_heads;
        planes.k_scale      = DeviceBuffer(scale_bytes);
        planes.v_scale      = DeviceBuffer(scale_bytes);
        planes.view.k_scale = Tensor(planes.k_scale.p, DType::FP32, {capacity, kv_heads});
        planes.view.v_scale = Tensor(planes.v_scale.p, DType::FP32, {capacity, kv_heads});
    }
    if (const auto poison = debug_poison_byte()) {
        planes.k.fill(*poison);
        planes.v.fill(*poison);
        if (fp8) {
            planes.k_scale.fill(*poison);
            planes.v_scale.fill(*poison);
        }
    }
    planes.view.k      = Tensor(planes.k.p, dtype, {head_dim, capacity, kv_heads});
    planes.view.v      = Tensor(planes.v.p, dtype, {head_dim, capacity, kv_heads});
    planes.view.window = window;
    return planes;
}

void project(const Tensor& x, const Projection& p, Tensor& out, DeviceArena& work,
             cudaStream_t stream, Activations activations) {
    auto scope = work.scope();
    ops::linear(x, p.weight, out,
                activations == Activations::Verify ? ops::verify_linear_policy(p.policy) : p.policy, work,
                stream);
}

// project with Call::release_next (ops::linear_release_next): true when the launch lets the next
// kernel start early, which must then wait before it reads `out`.
static bool project_release_next(const Tensor& x, const Projection& p, Tensor& out,
                                 DeviceArena& work, cudaStream_t stream, Activations activations) {
    auto scope = work.scope();
    return ops::linear_release_next(
        x, p.weight, out,
        activations == Activations::Verify ? ops::verify_linear_policy(p.policy) : p.policy, work, stream);
}

namespace {

// Projections of one x side by side (ops::project_side_by_side: a GGUF layer's [q; k] and its v
// in a wider type on the R8 kernel). False leaves them to the caller.
bool project_side_by_side(const Tensor& x, std::span<const Projection* const> projections,
                          std::span<Tensor* const> outputs, DeviceArena& work, cudaStream_t stream,
                          Activations activations) {
    if (projections.size() != outputs.size() ||
        projections.size() > ops::kSideBySideMaxProjections) {
        return false;
    }
    ops::SideBySideProjection requests[ops::kSideBySideMaxProjections];
    for (std::size_t i = 0; i < projections.size(); ++i) {
        const Projection& p = *projections[i];
        requests[i]         = {&p.weight,
                       activations == Activations::Verify ? ops::verify_linear_policy(p.policy) : p.policy,
                               outputs[i]};
    }
    return ops::project_side_by_side(x, std::span(requests, projections.size()), work, stream);
}

} // namespace

TargetPass::TargetPass(const Model& model, DeviceContext& device)
    : model_(model), c_(model.config()), device_(device), side_(device), fork_(device),
      join_(device) {}

Target::Target(const Model& model, DeviceContext& device, DeviceArena& work,
               std::int32_t max_context, std::int32_t prefill_chunk, bool kv_fp8, bool kv_fp8_full)
    : model_(model), device_(device), work_(work), pass_(model, device) {
    const Config& c = model.config();
    kv_.reserve(c.layers);
    std::vector<std::size_t> plane(c.layers);
    for (std::int32_t i = 0; i < c.layers; ++i) {
        // A layer that reads shared keys and values binds its source's planes.
        if (!c.owns_kv(i)) {
            plane[i] = plane[c.kv_source[i]];
            continue;
        }
        const auto& g      = c.geometry(c.kinds[i]);
        const bool sliding = c.kinds[i] == LayerKind::Sliding;
        // A sliding layer keeps a ring: a chunk of T queries reaches back window-1 keys before its
        // first query, so window + chunk slots hold every key any of them can see. The ring keeps
        // that size even when the context is shorter: the attention op needs window + T - 1
        // slots for a T-token pass whatever the context (a 1024-token context with a 512 chunk
        // failed its first prefill with a 1024-slot ring).
        const std::int32_t capacity = sliding ? c.sliding_window + prefill_chunk : max_context;
        plane[i] = kv_.size();
        kv_.push_back(allocate_kv_planes(g.head_dim, capacity, g.kv_heads,
                                         sliding ? c.sliding_window : 0,
                                         kv_fp8 || (kv_fp8_full && !sliding)));
    }
    for (std::int32_t i = 0; i < c.layers; ++i) { views_.push_back(kv_[plane[i]].view); }
}

void Target::poison_kv() noexcept {
    const auto poison = debug_poison_byte();
    if (!poison) { return; }
    // Debug only: errors are sticky and surface at the next checked call.
    for (auto& layer : kv_) {
        (void)cudaMemsetAsync(layer.k.p, *poison, layer.k.bytes, device_.stream);
        (void)cudaMemsetAsync(layer.v.p, *poison, layer.v.bytes, device_.stream);
        if (layer.k_scale.p != nullptr) {
            (void)cudaMemsetAsync(layer.k_scale.p, *poison, layer.k_scale.bytes, device_.stream);
            (void)cudaMemsetAsync(layer.v_scale.p, *poison, layer.v_scale.bytes, device_.stream);
        }
    }
}

std::uint64_t Target::kv_bytes() const noexcept {
    std::uint64_t bytes = 0;
    for (const auto& layer : kv_) {
        bytes += layer.k.bytes + layer.v.bytes + layer.k_scale.bytes + layer.v_scale.bytes;
    }
    return bytes;
}

std::size_t Target::workspace_bytes(const Model& model, std::int32_t max_context,
                                    std::int32_t prefill_chunk, std::int32_t head_columns) {
    const auto& c        = model.config();
    const std::int64_t t = prefill_chunk;
    const std::int64_t h = c.hidden_size;
    std::int64_t proj = 0, attn = 0;
    for (const auto* g : {&c.sliding, &c.full}) {
        proj = std::max<std::int64_t>(proj,
                                      std::int64_t(c.query_heads + 2 * g->kv_heads) * g->head_dim);
        attn = std::max<std::int64_t>(attn, std::int64_t(c.query_heads) * g->head_dim);
    }
    std::size_t bytes = 0;
    const auto add    = [&](std::int64_t elements) { bytes += bf16_bytes(elements) + 256; };
    add(h * t);                                     // x
    add(h * t);                                     // normed input
    add(proj * t);                                  // raw projection(s)
    add(proj * t);                                  // prepared q/k/v
    add(attn * t);                                  // attention context
    add(h * t);                                     // branch output
    add(2 * std::int64_t(c.intermediate_size) * t); // gate/up
    add(std::int64_t(c.intermediate_size) * t);     // GeGLU product
    add(h * t);                                     // dense branch
    add(h * t);                                     // expert input
    add(h * t);                                     // router input (generic MoE)
    add(h * t);                                     // expert output
    add(h * t);                                     // branch join
    if (c.ple()) {
        const std::int64_t rows = std::int64_t(c.layers) * c.ple_width;
        add(rows * t);                 // per-layer inputs (the projection, combined in place)
        add(rows * t);                 // gathered table rows
        add(2 * std::int64_t(c.ple_width) * t); // gate, product
    }
    add(std::int64_t(c.vocab_size) * std::max<std::int64_t>(head_columns, 1));
    std::size_t op = 0;
    if (c.ple()) {
        const auto& p = model.weights().ple_projection;
        op            = std::max(op, ops::linear_workspace_capacity_bytes(p, 1, prefill_chunk));
    }
    for (const auto& layer : model.weights().layers) {
        if (c.ple()) {
            for (const Projection* p : {&layer.ple_gate, &layer.ple_output}) {
                op = std::max(op, ops::linear_workspace_capacity_bytes(
                                      p->weight.qtype, p->weight.n, p->weight.k, p->policy, 1,
                                      prefill_chunk));
            }
        }
        for (const Projection* p : {&layer.output, &layer.gate_up, &layer.down}) {
            op = std::max(op, ops::linear_workspace_capacity_bytes(p->weight.qtype, p->weight.n,
                                                                   p->weight.k, p->policy, 1,
                                                                   prefill_chunk));
        }
        if (layer.gate_up.weight.n > 0 && ops::linear_geglu_supports(layer.gate_up.weight)) {
            for (const auto policy : {layer.gate_up.policy, ops::LinearPolicy::A16Only}) {
                op = std::max(op, ops::linear_geglu_workspace_capacity_bytes(
                                      layer.gate_up.weight.qtype, layer.gate_up.weight.n,
                                      layer.gate_up.weight.k, policy, 1, prefill_chunk) +
                                      256);
            }
        }
        if (layer.qkv) {
            op = std::max(op, ops::linear_workspace_capacity_bytes(
                                  layer.qkv->weight.qtype, layer.qkv->weight.n, layer.qkv->weight.k,
                                  layer.qkv->policy, 1, prefill_chunk));
            if (const Projection& v = layer.value; v.weight.n != 0) {
                op = std::max(op, ops::linear_workspace_capacity_bytes(v, 1, prefill_chunk));
            }
        } else {
            for (const Projection* p : {&layer.query, &layer.key, &layer.value}) {
                if (p->weight.n == 0) { continue; } // a sharing layer's key; k_eq_v's value
                op = std::max(op, ops::linear_workspace_capacity_bytes(p->weight.qtype, p->weight.n,
                                                                       p->weight.k, p->policy, 1,
                                                                       prefill_chunk));
            }
        }
    }
    for (const auto* g : {&c.sliding, &c.full}) {
        const ops::Gemma4AttentionEnvelope envelope{static_cast<std::uint32_t>(max_context),
                                                    prefill_chunk};
        const std::int32_t window = g == &c.sliding ? c.sliding_window : 0;
        op = std::max(op, ops::gemma4_attention_workspace_bytes(g->head_dim, c.query_heads,
                                                                g->kv_heads, envelope, window));
    }
    if (c.moe()) { op = std::max(op, ops::gemma4_moe_workspace_bytes(1, prefill_chunk)); }
    for (const auto& layer : model.weights().layers) {
        if (!layer.generic_moe) { continue; }
        op = std::max(op, ops::moe_workspace_bytes(*layer.generic_moe, layer.generic_moe_config, 1,
                                                   prefill_chunk));
    }
    // The output head runs on this arena too (TargetPass::head, uncapped_head on A16 for the exact
    // logits), over up to head_columns columns; the dense and Nemotron targets count it as well.
    const Projection& head = model.weights().output_head;
    for (const auto policy : {head.policy, ops::LinearPolicy::A16Only}) {
        op = std::max(op, ops::linear_workspace_capacity_bytes(head.weight.qtype, head.weight.n,
                                                               head.weight.k, policy, 1,
                                                               std::max(head_columns, 1)));
    }
    op = std::max(op, ops::sampling_workspace_capacity_bytes(c.vocab_size, 1, 1));
    return bytes + op + (std::size_t(64) << 20);
}

// One pass of every layer over the T columns of x whose absolute positions are in `pos`.
// `h` enters holding input_norm(x) of layer 0 and leaves holding final_norm(x): each layer's
// closing residual update writes the next reader's normalized input in the same launch.
void TargetPass::layers(DeviceArena& work_, Tensor& x, Tensor& h, const Tensor& pos, KvBindings kv,
                        ops::Gemma4AttentionEnvelope envelope, Activations activations,
                        const FeatureTap* tap, const Tensor* ple) const {
    const cudaStream_t s     = device_.stream;
    const std::int32_t t     = x.ne[1];
    const std::int32_t batch = kv.batch;
    if (static_cast<std::int32_t>(kv.layers.size()) != c_.layers || batch < 1 || t % batch != 0) {
        throw std::invalid_argument("Gemma 4 forward: KV bindings do not match the layers or T");
    }
    // A batched call runs attention over [.., T/B, B]; every other step is per column.
    const std::int32_t row_tokens = t / batch;
    const Tensor attention_pos    = batch == 1 ? pos : pos.view({row_tokens, batch});
    const auto eps       = c_.rms_norm_eps;
    const bool dump      = dumping(s);
    long long p0         = 0;
    if (dump) {
        std::int32_t first = 0;
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaMemcpy(&first, pos.data, sizeof(first), cudaMemcpyDeviceToHost));
        p0 = first;
        dump_tensor(h, t, p0, 0, "in");
    }
    for (std::int32_t i = 0; i < c_.layers; ++i) {
        const auto& layer       = model_.weights().layers[i];
        const auto& g           = c_.geometry(layer.kind);
        const auto d            = g.head_dim;
        const auto q_rows       = c_.query_heads * d;
        const auto kv_rows      = g.kv_heads * d;
        const Tensor& next_norm = i + 1 < c_.layers ? model_.weights().layers[i + 1].input_norm
                                                    : model_.weights().final_norm;
        auto scope              = work_.scope();
        try {
            nvtx::ScopedRange range(layer.kind == LayerKind::Full ? nvtx::Name::PrefillLayerFull
                                                                  : nvtx::Name::PrefillLayerGdn,
                                    nvtx::Category::Attention, i);

            // Raw projections, then norms, RoPE and v_norm in one launch.
            Tensor q_src, k_src, v_src;
            Tensor q = work_.alloc(DType::BF16, {d, c_.query_heads, t});
            // Whether the last projection released the preparation (project_release_next; a
            // side-by-side launch does not). Not under a dump, which copies between them.
            bool qkv_released       = false;
            const auto project_last = [&](const Projection& p, Tensor& out) {
                if (dump) {
                    project(h, p, out, work_, s, activations);
                } else {
                    qkv_released = project_release_next(h, p, out, work_, s, activations);
                }
            };
            if (!c_.owns_kv(i)) {
                // KV sharing: queries only, over the keys and values its source layer wrote
                // (kv.layers[i] binds the source's cache).
                q_src = work_.alloc(DType::BF16, {q_rows, t});
                project(h, layer.query, q_src, work_, s, activations);
                ops::gemma4_q_prepare(q_src, d, layer.query_norm, eps, pos, g.rope_theta,
                                      g.rotary_pairs, q, s);
            } else if (layer.qkv) {
                Tensor packed = work_.alloc(DType::BF16, {layer.qkv->weight.n, t});
                q_src = packed.slice(0, 0, q_rows);
                k_src = packed.slice(0, q_rows, kv_rows);
                if (layer.qkv->weight.n == q_rows + 2 * kv_rows) {
                    project_last(*layer.qkv, packed);
                    v_src = packed.slice(0, q_rows + kv_rows, kv_rows);
                } else if (layer.value.weight.n != 0) {
                    // [q; k] packed, v apart in its own type (beside it in one launch where the
                    // side-by-side route admits both).
                    v_src = work_.alloc(DType::BF16, {kv_rows, t});
                    const Projection* p[] = {&*layer.qkv, &layer.value};
                    Tensor* o[]           = {&packed, &v_src};
                    if (!project_side_by_side(h, p, o, work_, s, activations)) {
                        project(h, *layer.qkv, packed, work_, s, activations);
                        project_last(layer.value, v_src);
                    }
                } else {
                    project_last(*layer.qkv, packed);
                    v_src = k_src; // k_eq_v
                }
            } else {
                q_src = work_.alloc(DType::BF16, {q_rows, t});
                k_src = work_.alloc(DType::BF16, {kv_rows, t});
                const bool has_v = layer.value.weight.n != 0;
                if (has_v) { v_src = work_.alloc(DType::BF16, {kv_rows, t}); }
                const Projection* p[] = {&layer.query, &layer.key, &layer.value};
                Tensor* o[]           = {&q_src, &k_src, &v_src};
                const std::size_t n   = has_v ? 3 : 2;
                if (!project_side_by_side(h, std::span(p).first(n), std::span(o).first(n), work_, s,
                                          activations)) {
                    project(h, layer.query, q_src, work_, s, activations);
                    if (has_v) {
                        project(h, layer.key, k_src, work_, s, activations);
                        project_last(layer.value, v_src);
                    } else {
                        project_last(layer.key, k_src);
                    }
                }
                if (!has_v) { v_src = k_src; }
            }
            // Norms, RoPE and v_norm in one launch; keys and values land in this layer's cache
            // slots directly.
            if (c_.owns_kv(i)) {
                ops::gemma4_qkv_prepare(q_src, k_src, v_src, d, layer.query_norm, layer.key_norm,
                                        eps, pos, g.rope_theta, g.rotary_pairs, q, kv.layers[i], s,
                                        qkv_released);
            }
            Tensor context = work_.alloc(DType::BF16, {d, c_.query_heads, t});
            {
                // Ops that own a workspace see all of it as theirs; give them a carved span.
                auto attention_scope = work_.scope();
                WorkspaceArena scratch(work_.alloc_bytes(std::max<std::size_t>(
                    ops::gemma4_attention_batch_workspace_bytes(
                        work_, batch, d, c_.query_heads, g.kv_heads, envelope, kv.layers[i].window),
                    256)));
                if (batch == 1) {
                    const auto mask = kv.image_block && kv.layers[i].window > 0
                                          ? ops::Gemma4AttentionMask::Block
                                          : ops::Gemma4AttentionMask::Causal;
                    ops::gemma4_attention(q, pos, kv.layers[i], c_.attention_scale, envelope,
                                          scratch, context, s, mask);
                } else {
                    const Tensor q_rows = q.view({d, c_.query_heads, row_tokens, batch});
                    Tensor context_rows = context.view({d, c_.query_heads, row_tokens, batch});
                    ops::gemma4_attention(q_rows, attention_pos, kv.layers[i], c_.attention_scale,
                                          envelope, scratch, context_rows, s);
                }
            }
            if (dump) {
                dump_tensor(q, t, p0, i, "q");
                dump_tensor(context, t, p0, i, "attn");
            }
            Tensor branch = work_.alloc(DType::BF16, {c_.hidden_size, t});
            // The residual update below joins the output projection when its launch released it
            // (not under a dump: the copy of `branch` would sit between them).
            const bool released =
                !dump && project_release_next(context.view({q_rows, t}), layer.output, branch,
                                              work_, s, activations);
            if (dump)
                project(context.view({q_rows, t}), layer.output, branch, work_, s, activations);
            if (dump) { dump_tensor(branch, t, p0, i, "ao"); }

            // Residual update after attention, emitting the feed-forward inputs.
            Tensor ffn_input = work_.alloc(DType::BF16, {c_.hidden_size, t});
            Tensor expert_input =
                layer.has_moe() ? work_.alloc(DType::BF16, {c_.hidden_size, t}) : Tensor{};
            // The generic MoE's router reads rmsnorm(x) with the folded router scale: a third
            // output of the same update, off the expert branch's critical path.
            Tensor router_input =
                layer.generic_moe ? work_.alloc(DType::BF16, {c_.hidden_size, t}) : Tensor{};
            const ops::ResidualNormOutput ffn_inputs[] = {{layer.pre_ffn_norm, ffn_input},
                                                          {layer.pre_moe_norm, expert_input},
                                                          {layer.router_norm, router_input}};
            const std::size_t outputs = layer.generic_moe ? 3 : layer.has_moe() ? 2 : 1;
            ops::gemma4_residual_norm(x, {branch}, layer.post_attention_norm, eps, 1.0F,
                                      std::span(ffn_inputs, outputs), s, released);
            if (dump) {
                dump_tensor(x, t, p0, i, "h1");
                dump_tensor(ffn_input, t, p0, i, "ffin");
                if (layer.has_moe()) { dump_tensor(expert_input, t, p0, i, "ein"); }
            }

            // Feed-forward: a dense GeGLU MLP, joined on the MoE model by the expert bank.
            // The two branches read only what the update above wrote, so the expert bank runs
            // on a side stream (a parallel branch of the captured graph) while the dense MLP
            // runs here. Its output and workspace are carved before either starts, so the two
            // branches never share arena bytes.
            Tensor experts;
            std::optional<WorkspaceArena> moe_scratch;
            if (layer.generic_moe) {
                // The generic MoE Op (experts not in NVFP4): its router reads router_input (the
                // residual update's third output), the experts read expert_input.
                const auto& gm = *layer.generic_moe;
                experts        = work_.alloc(DType::BF16, {c_.hidden_size, t});
                moe_scratch.emplace(work_.alloc_bytes(
                    ops::moe_workspace_bytes(gm, layer.generic_moe_config, t, t)));
                fork_.record(s);
                fork_.wait(side_.get());
                ops::moe(expert_input, gm, layer.generic_moe_config, ops::MoeEpilogue::Store,
                         experts, *moe_scratch, side_.get(), &router_input);
                join_.record(side_.get());
            } else if (layer.moe) {
                experts = work_.alloc(DType::BF16, {c_.hidden_size, t});
                moe_scratch.emplace(work_.alloc_bytes(ops::gemma4_moe_workspace_bytes(t, t)));
                fork_.record(s);
                fork_.wait(side_.get());
                ops::gemma4_moe(x, expert_input, *layer.moe, eps, experts, *moe_scratch,
                                side_.get(), layer.moe_activations);
                join_.record(side_.get());
            }
            Tensor product = work_.alloc(DType::BF16, {c_.intermediate[i], t});
            if (ops::linear_geglu_supports(layer.gate_up.weight)) {
                // GeGLU in the projection's epilogue: no BF16 gate/up plane, no second kernel.
                auto geglu_scope  = work_.scope();
                const auto policy = activations == Activations::Verify ? ops::verify_linear_policy(layer.gate_up.policy)
                                                                        : layer.gate_up.policy;
                WorkspaceArena scratch(work_.alloc_bytes(
                    std::max<std::size_t>(ops::linear_geglu_workspace_capacity_bytes(
                                              layer.gate_up.weight.qtype, layer.gate_up.weight.n,
                                              layer.gate_up.weight.k, policy, t, t),
                                          256)));
                ops::linear_geglu(ffn_input, layer.gate_up.weight, product, policy, scratch, s);
            } else {
                Tensor gate_up = work_.alloc(DType::BF16, {layer.gate_up.weight.n, t});
                project(ffn_input, layer.gate_up, gate_up, work_, s, activations);
                ops::gelu_tanh_mul(gate_up, product, s);
            }
            project(product, layer.down, branch, work_, s, activations);
            if (layer.has_moe()) { join_.wait(s); }
            if (dump) {
                dump_tensor(branch, t, p0, i, "mlp");
                if (layer.has_moe()) { dump_tensor(experts, t, p0, i, "moe"); }
                if (layer.moe) {
                    // Every MoE route carves its selection first: ids I32 [8*cap], then the
                    // route weights FP32 [8*cap] at the next 256-byte boundary.
                    const std::int32_t cap = std::min<std::int32_t>(t, 4096);
                    const auto* base       = static_cast<const char*>(moe_scratch->base());
                    const std::size_t ids  = std::size_t(8) * cap * 4;
                    dump_columns(base, 32, t, p0, i, "sel");
                    dump_columns(base + (ids + 255) / 256 * 256, 32, t, p0, i, "rw");
                }
            }

            // The feed-forward branch: the dense MLP, joined on the MoE model by the experts.
            const ops::ResidualBranch joined =
                layer.has_moe() ? ops::ResidualBranch{branch, &layer.post_mlp_norm, &experts,
                                                      &layer.post_moe_norm}
                                : ops::ResidualBranch{branch};
            if (ple != nullptr) {
                // The per-layer embedding block reads the stream after the feed-forward update,
                // and the layer scalar applies after it:
                //   x += norm(output(gelu_tanh(gate(x)) * ple_i), post_ple_norm);  x *= scale.
                ops::gemma4_residual_norm(x, joined, layer.post_ffn_norm, eps, 1.0F, {}, s);
                const std::int32_t pw = c_.ple_width;
                Tensor gate           = work_.alloc(DType::BF16, {pw, t});
                Tensor gated          = work_.alloc(DType::BF16, {pw, t});
                project(x, layer.ple_gate, gate, work_, s, activations);
                ops::gelu_tanh_mul(gate, ple->slice(0, i * pw, pw), gated, s);
                project(gated, layer.ple_output, branch, work_, s, activations);
                ops::gemma4_residual_norm(x, {branch}, layer.post_ple_norm, eps, layer.output_scale,
                                          {{next_norm, h}}, s);
            } else {
                // A programmatic dependent of the down projection (and, through the join, of the
                // experts): its weights load under them; it waits before it reads either branch.
                ops::gemma4_residual_norm(x, joined, layer.post_ffn_norm, eps, layer.output_scale,
                                          {{next_norm, h}}, s, !dump);
            }
            if (dump) { dump_tensor(x, t, p0, i, "x"); }
            // x is now this layer's output (the layer scalar applied): a tapped layer copies it
            // into its row block of the sink.
            if (tap != nullptr && tap->slots[i] >= 0) {
                if (tap->project) {
                    throw std::logic_error("Gemma 4 taps features into the sink only "
                                           "(FeatureTap::project is the dense decoder's)");
                }
                const std::size_t row = bf16_bytes(c_.hidden_size);
                CUDA_CHECK(cudaMemcpy2DAsync(
                    static_cast<char*>(tap->sink.data) + tap->slots[i] * row, tap->sink.nb[1],
                    x.data, row, row, t, cudaMemcpyDeviceToDevice, s));
            }
        } catch (const std::exception&) {
            rethrow_with_context("text/layers/" + std::to_string(i) + " T=" + std::to_string(t));
        }
    }
}

void Model::embed(const Tensor& ids, Tensor& out, cudaStream_t stream) const {
    ops::embedding(ids, weights_.token_embedding, out, stream);
    if (config_.embedding_scale != 1.0F) { ops::scale_bf16(out, config_.embedding_scale, stream); }
}

Tensor TargetPass::forward(DeviceArena& work, const Tensor& ids, const Tensor& pos, KvBindings kv,
                           ops::Gemma4AttentionEnvelope envelope, Activations activations,
                           const FeatureTap* tap, const Tensor* embeddings) const {
    const std::int32_t t = ids.ne[0];
    const auto& w        = model_.weights();
    Tensor x             = work.alloc(DType::BF16, {c_.hidden_size, t});
    Tensor h             = work.alloc(DType::BF16, {c_.hidden_size, t});
    if (embeddings != nullptr) {
        // An image's soft tokens enter as they are: the reference scatters them over the scaled
        // text embeddings, unscaled.
        if (embeddings->dtype != DType::BF16 || !embeddings->is_contiguous() ||
            embeddings->ne[0] != c_.hidden_size || embeddings->ne[1] != t || c_.ple() ||
            kv.batch != 1) {
            throw std::invalid_argument("Gemma 4 forward: image embeddings must be BF16 [hidden, T] "
                                        "of one sequence on a model without per-layer inputs");
        }
        CUDA_CHECK(cudaMemcpyAsync(x.data, embeddings->data, x.bytes(), cudaMemcpyDeviceToDevice,
                                   device_.stream));
    } else {
        model_.embed(ids, x, device_.stream);
    }
    // Per-layer inputs: the table rows of each token plus the normed projection of its embedding,
    // one P-wide slice per layer, alive for the whole pass.
    Tensor ple;
    if (c_.ple()) {
        ple = work.alloc(DType::BF16, {c_.layers * c_.ple_width, t});
        project(x, w.ple_projection, ple, work, device_.stream, activations);
        auto scope  = work.scope();
        Tensor rows = work.alloc(DType::BF16, {c_.layers * c_.ple_width, t});
        ops::embedding(ids, w.ple_table, rows, device_.stream);
        ops::gemma4_ple_inputs(ple, rows, w.ple_norm, c_.rms_norm_eps, c_.ple_projection_scale,
                               c_.ple_embedding_scale, c_.ple_input_scale, ple, device_.stream);
    }
    ops::rmsnorm(x, w.layers.front().input_norm, c_.rms_norm_eps, false, h, device_.stream);
    layers(work, x, h, pos, kv, envelope, activations, tap, c_.ple() ? &ple : nullptr);
    return h;
}

void TargetPass::head(DeviceArena& work, const Tensor& normed, Tensor& out) const {
    const cudaStream_t s = device_.stream;
    project(normed, model_.weights().output_head, out, work, s);
    if (c_.softcap) { ops::logit_softcap(out, *c_.softcap, s); }
}

void TargetPass::uncapped_head(DeviceArena& work, const Tensor& normed, Tensor& out,
                               Activations activations) const {
    project(normed, model_.weights().output_head, out, work, device_.stream, activations);
}

Tensor Target::forward(const Tensor& ids, const Tensor& pos, ops::Gemma4AttentionEnvelope envelope,
                       Activations activations, const FeatureTap* tap) {
    return pass_.forward(work_, ids, pos, {.layers = views_}, envelope, activations, tap);
}

void Target::head(const Tensor& normed, Tensor& out) { pass_.head(work_, normed, out); }

void Target::uncapped_head(const Tensor& normed, Tensor& out, Activations activations) {
    pass_.uncapped_head(work_, normed, out, activations);
}

speculation::TargetShape Target::shape() const noexcept {
    const auto& c = model_.config();
    return {c.hidden_size, c.vocab_size, c.layers};
}

Tensor Target::draft_forward(const Tensor& ids, const Tensor& pos,
                             ops::Gemma4AttentionEnvelope envelope, bool verify,
                             const FeatureTap* tap) {
    return forward(ids, pos, envelope, verify ? Activations::Verify : Activations::Recorded, tap);
}

void Target::greedy_logits(const Tensor& normed, Tensor& out, bool verify) {
    uncapped_head(normed, out, verify ? Activations::Verify : Activations::Recorded);
}

void Target::draft_embedding(const Tensor& ids, Tensor& out) {
    // The table is stored pre-scaled by sqrt(H): exactly the drafter's input.
    ops::embedding(ids, model_.weights().token_embedding, out, device_.stream);
}

} // namespace ninferno::models::gemma4
