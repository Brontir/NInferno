#include "models/dense/target.h"

#include "core/nvtx.h"
#include "models/common/layer_dump.h"
#include "ninferno/ops/dense_fused.h"
#include "ninferno/ops/embedding.h"
#include "ninferno/ops/gemma4_fused.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/linear_add.h"
#include "ninferno/ops/linear_swiglu.h"
#include "ninferno/ops/logit_softcap.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/rmsnorm.h"
#include "ninferno/ops/sampling.h"
#include "ninferno/ops/segmented_projection.h"
#include "ninferno/ops/sigmoid_mul.h"
#include "ninferno/ops/silu_mul.h"
#include "ops/linear/linear_registry.h"

#include <algorithm>
#include <atomic>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::models::dense {
namespace {

// Debug aid: NINFERNO_DENSE_DUMP=<dir> writes, for every eager forward call (a captured graph is
// skipped) whose batch is 1, the token columns at absolute positions in
// NINFERNO_DENSE_DUMP_ROWS=lo:hi (default all) of each layer's intermediates to
// <dir>/c<call>_p<first position>_t<tokens>.L<layer>.<kind>.bin: q (queries after RoPE), attn
// (attention context), ao (output projection), mlp (feed-forward branch), x (residual after the
// layer); L-1.in is the embedding, L<layers>.h the final-norm output, and <prefix>.logits the head's
// logits over the whole vocabulary (the call's last column). It synchronizes the device:
// for localizing numerics between two paths (ninferno-dense and ninferno-serve), never for timing.
const LayerDump* dump_config() {
    static const std::optional<LayerDump> config = read_layer_dump("NINFERNO_DENSE_DUMP");
    return config ? &*config : nullptr;
}

bool dumping(cudaStream_t stream) { return dumps_on(dump_config(), stream); }

struct DumpCall {
    std::string prefix;
    long long p0 = 0;
    std::int32_t t = 0;
};

// The call prefix of the last forward that dumped on this thread ("" when it did not): the head
// that follows it dumps its logits there.
thread_local std::string last_dump_prefix;

void dump(const DumpCall& call, const Tensor& tensor, std::int32_t layer, const char* kind) {
    write_device_rows(*dump_config(), tensor.data,
                      tensor.bytes() / static_cast<std::size_t>(call.t), call.t, call.p0,
                      call.prefix + ".L" + std::to_string(layer) + "." + kind + ".bin", false);
}

std::size_t bf16_bytes(std::int64_t elements) { return static_cast<std::size_t>(elements) * 2; }

// A layer's separate gate and up as the [gate; up] rows of one segmented projection.
ops::SegmentedProjectionWeights split_swiglu(const Projection& gate, ops::LinearPolicy gate_policy,
                                             const Projection& up, ops::LinearPolicy up_policy) {
    ops::SegmentedProjectionWeights w;
    w.slices.push_back({gate.weight, 0, gate.weight.n, gate_policy});
    w.slices.push_back({up.weight, 0, up.weight.n, up_policy});
    return w;
}

ops::LinearPolicy policy_of(const Projection& p, Activations activations) {
    return activations == Activations::A16Only ? ops::verify_linear_policy(p.policy) : p.policy;
}

} // namespace

DensePass::DensePass(const Model& model, DeviceContext& device)
    : model_(model), c_(model.config()), device_(device) {}

Target::Target(const Model& model, DeviceContext& device, DeviceArena& work,
               std::int32_t max_context, std::int32_t prefill_chunk, bool kv_fp8)
    : model_(model), device_(device), work_(work), max_context_(max_context), pass_(model, device) {
    const Config& c = model.config();
    kv_.reserve(c.layers);
    views_.reserve(c.layers);
    for (const auto& spec : c.layer_specs) {
        // A sliding layer keeps a ring: a chunk of T queries reaches back window-1 keys before
        // its first query, so window + chunk slots hold every key any of them can see. A window
        // no shorter than the whole context excludes no key: that layer is a full one here.
        const std::int32_t window = spec.window < max_context ? spec.window : 0;
        const std::int32_t capacity = window > 0 ? window + prefill_chunk : max_context;
        const std::int64_t values = std::int64_t(c.head_dim) * capacity * c.kv_heads;
        const DType dtype         = kv_fp8 ? DType::FP8_E4M3FN : DType::BF16;
        const std::size_t bytes   = kv_fp8 ? static_cast<std::size_t>(values) : bf16_bytes(values);
        KvPlanes planes;
        planes.k = DeviceBuffer(bytes);
        planes.v = DeviceBuffer(bytes);
        ops::Gemma4KvLayer view;
        view.k      = Tensor(planes.k.p, dtype, {c.head_dim, capacity, c.kv_heads});
        view.v      = Tensor(planes.v.p, dtype, {c.head_dim, capacity, c.kv_heads});
        view.window = window;
        if (kv_fp8) {
            const std::size_t scale_bytes = sizeof(float) * std::size_t(capacity) * c.kv_heads;
            planes.k_scale = DeviceBuffer(scale_bytes);
            planes.v_scale = DeviceBuffer(scale_bytes);
            view.k_scale   = Tensor(planes.k_scale.p, DType::FP32, {capacity, c.kv_heads});
            view.v_scale   = Tensor(planes.v_scale.p, DType::FP32, {capacity, c.kv_heads});
        }
        kv_.push_back(std::move(planes));
        views_.push_back(view);
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
    return DensePass::workspace_bytes(model, max_context, prefill_chunk, head_columns);
}

std::size_t DensePass::workspace_bytes(const Model& model, std::int32_t max_context,
                                       std::int32_t prefill_chunk, std::int32_t head_columns) {
    const auto& c        = model.config();
    const std::int64_t t = prefill_chunk;
    const std::int64_t h = c.hidden_size;
    const std::int64_t q = std::int64_t(c.query_heads) * c.head_dim;
    const std::int64_t kv = std::int64_t(c.kv_heads) * c.head_dim;
    const std::int64_t inter = c.intermediate_size;
    std::size_t bytes = 0;
    const auto add    = [&](std::int64_t elements) { bytes += bf16_bytes(elements) + 256; };
    add(h * t);                  // x
    add(h * t);                  // normed input
    add((q + 2 * kv) * t);       // raw projections
    add(q * t);                  // prepared queries
    add(q * t);                  // attention context
    if (c.attention_output_gate) { add(q * t); } // output gate
    add(h * t);                  // branch
    add(h * t);                  // feed-forward input
    if (!c.moe) {
        add(2 * inter * t); // gate/up
        add(inter * t);     // product
    }
    add(std::int64_t(c.vocab_size) * std::max<std::int64_t>(head_columns, 1));
    std::size_t op = 0;
    for (const auto& layer : model.weights().layers) {
        if (layer.moe) {
            // Both route profiles and every T up to a chunk.
            op = std::max(op, ops::moe_workspace_bytes(*layer.moe, c.moe->op, 1, prefill_chunk));
            for (const Projection* p : {&layer.output}) {
                op = std::max(op, ops::linear_workspace_capacity_bytes(*p, 1, prefill_chunk));
                op = std::max(op, ops::linear_add_workspace_capacity_bytes(*p, 1, prefill_chunk));
            }
            if (layer.qkv) {
                op = std::max(op,
                              ops::linear_workspace_capacity_bytes(*layer.qkv, 1, prefill_chunk));
                if (layer.value.weight.n != 0) {
                    op = std::max(
                        op, ops::linear_workspace_capacity_bytes(layer.value, 1, prefill_chunk));
                }
            } else {
                for (const Projection* p : {&layer.query, &layer.key, &layer.value}) {
                    op = std::max(op, ops::linear_workspace_capacity_bytes(*p, 1, prefill_chunk));
                }
            }
            continue;
        }
        std::vector<const Projection*> all{&layer.output, &layer.down};
        if (layer.attention_gate) { all.push_back(&*layer.attention_gate); }
        if (layer.qkv) {
            all.push_back(&*layer.qkv);
            if (layer.value.weight.n != 0) { all.push_back(&layer.value); }
        } else {
            all.insert(all.end(), {&layer.query, &layer.key, &layer.value});
        }
        if (layer.gate_up) {
            if (c.activation == Activation::SiLU) {
                op = std::max(op, ops::linear_swiglu_workspace_capacity_bytes(
                                      layer.gate_up->weight.qtype, layer.gate_up->weight.n,
                                      layer.gate_up->weight.k, layer.gate_up->policy, 1,
                                      prefill_chunk));
            } else {
                all.push_back(&*layer.gate_up);
                if (ops::linear_geglu_supports(layer.gate_up->weight)) {
                    op = std::max(op, ops::linear_geglu_workspace_capacity_bytes(
                                          layer.gate_up->weight.qtype, layer.gate_up->weight.n,
                                          layer.gate_up->weight.k, layer.gate_up->policy, 1,
                                          prefill_chunk) +
                                          256);
                }
            }
        } else {
            all.insert(all.end(), {&layer.gate, &layer.up});
            if (c.activation == Activation::SiLU) {
                op = std::max(op, ops::segmented_swiglu_pair_workspace_capacity_bytes(
                                      split_swiglu(layer.gate, layer.gate.policy, layer.up,
                                                   layer.up.policy),
                                      1, prefill_chunk) +
                                      256);
            }
        }
        for (const Projection* p : all) {
            op = std::max(op, ops::linear_workspace_capacity_bytes(*p, 1, prefill_chunk));
        }
        for (const Projection* p : {&layer.output, &layer.down}) {
            op = std::max(op, ops::linear_add_workspace_capacity_bytes(*p, 1, prefill_chunk));
        }
    }
    op = std::max(op, ops::linear_workspace_capacity_bytes(model.weights().output_head, 1,
                                                           std::max(head_columns, 1)));
    for (const auto& spec : c.layer_specs) {
        const ops::Gemma4AttentionEnvelope envelope{static_cast<std::uint32_t>(max_context),
                                                    prefill_chunk};
        // A window no shorter than the context runs as a full layer (the constructor).
        for (const std::int32_t window : {spec.window, 0}) {
            op = std::max(op, ops::gemma4_attention_workspace_bytes(c.head_dim, c.query_heads,
                                                                    c.kv_heads, envelope, window));
        }
    }
    op = std::max(op, ops::sampling_workspace_capacity_bytes(c.vocab_size, 1, 1));
    return bytes + op + (std::size_t(64) << 20);
}

void DensePass::project(DeviceArena& work_, const Tensor& x, const Projection& p, Tensor& out,
                        Activations activations) const {
    auto scope = work_.scope();
    ops::linear(x, p.weight, out, policy_of(p, activations), work_, device_.stream);
}

// project with Call::release_next: true when the launch lets the next kernel start early, which
// must then wait before it reads `out` (the closing norm or the q/k/v preparation).
bool DensePass::project_release_next(DeviceArena& work_, const Tensor& x, const Projection& p,
                                     Tensor& out, Activations activations) const {
    auto scope = work_.scope();
    return ops::linear_release_next(x, p.weight, out, policy_of(p, activations), work_,
                                    device_.stream);
}

// Decode and verify widths: a layer's q/k/v and its attention output gate side by side
// (ops::project_side_by_side; Muse Glimmer 30B on an RTX 5090: 256 and 288 CTAs each, three fit
// per SM; NVFP4 decode +0.35%, its six Q8_G32 layers 79.62 -> 79.99 t/s).
bool DensePass::project_side_by_side(DeviceArena& work_, const Tensor& x,
                                     std::span<const Projection* const> projections,
                                     std::span<Tensor* const> outputs,
                                     Activations activations) const {
    if (projections.size() != outputs.size() ||
        projections.size() > ops::kSideBySideMaxProjections) {
        return false;
    }
    ops::SideBySideProjection requests[ops::kSideBySideMaxProjections];
    for (std::size_t i = 0; i < projections.size(); ++i) {
        requests[i] = {&projections[i]->weight, policy_of(*projections[i], activations),
                       outputs[i]};
    }
    return ops::project_side_by_side(x, std::span(requests, projections.size()), work_,
                                     device_.stream);
}

bool DensePass::dense_mlp(DeviceArena& work_, const Layer& layer, const Tensor& mlp_input,
                          Tensor& branch, std::int32_t t, Activations activations, bool add,
                          bool release) const {
    const cudaStream_t s = device_.stream;
    auto scope           = work_.scope();
    Tensor product       = work_.alloc(DType::BF16, {c_.intermediate_size, t});
    if (layer.gate_up && c_.activation == Activation::SiLU) {
        auto swiglu_scope = work_.scope();
        const auto& gu    = *layer.gate_up;
        const auto policy = policy_of(gu, activations);
        WorkspaceArena scratch(work_.alloc_bytes(std::max<std::size_t>(
            ops::linear_swiglu_workspace_capacity_bytes(gu.weight.qtype, gu.weight.n, gu.weight.k,
                                                        policy, t, t),
            256)));
        ops::linear_swiglu(mlp_input, gu.weight, product, policy, scratch, s);
    } else if (layer.gate_up && ops::linear_geglu_supports(layer.gate_up->weight)) {
        // GeGLU in the projection's epilogue (the generic path's formats).
        auto geglu_scope  = work_.scope();
        const auto& gu    = *layer.gate_up;
        const auto policy = policy_of(gu, activations);
        WorkspaceArena scratch(work_.alloc_bytes(
            std::max<std::size_t>(ops::linear_geglu_workspace_capacity_bytes(
                                      gu.weight.qtype, gu.weight.n, gu.weight.k, policy, t, t),
                                  256)));
        ops::linear_geglu(mlp_input, gu.weight, product, policy, scratch, s);
    } else if (layer.gate_up) {
        Tensor gate_up = work_.alloc(DType::BF16, {layer.gate_up->weight.n, t});
        project(work_, mlp_input, *layer.gate_up, gate_up, activations);
        ops::gelu_tanh_mul(gate_up, product, s);
    } else {
        if (c_.activation != Activation::SiLU) {
            throw std::invalid_argument("an unpacked gate/up needs the SiLU activation");
        }
        // A GGUF layer's gate and up (often of two formats) at decode and verify widths: one
        // quantization of x, the SiLU product in the vector kernel's epilogue.
        bool paired = false;
        {
            auto pair_scope  = work_.scope();
            const auto split = split_swiglu(layer.gate, policy_of(layer.gate, activations),
                                            layer.up, policy_of(layer.up, activations));
            const std::size_t bytes =
                ops::segmented_swiglu_pair_workspace_capacity_bytes(split, t, t);
            if (bytes != 0) {
                WorkspaceArena scratch(work_.alloc_bytes(bytes + 256));
                paired = ops::segmented_swiglu_pair(mlp_input, split, false, product, scratch, s);
            }
        }
        if (!paired) {
            Tensor gate = work_.alloc(DType::BF16, {c_.intermediate_size, t});
            Tensor up   = work_.alloc(DType::BF16, {c_.intermediate_size, t});
            project(work_, mlp_input, layer.gate, gate, activations);
            project(work_, mlp_input, layer.up, up, activations);
            ops::silu_mul(gate, up, product, s);
        }
    }
    if (add) {
        auto down_scope = work_.scope();
        ops::linear_add(product, layer.down.weight, branch, policy_of(layer.down, activations),
                        work_, s);
        return false;
    }
    if (release) return project_release_next(work_, product, layer.down, branch, activations);
    project(work_, product, layer.down, branch, activations);
    return false;
}

Tensor Target::forward(const Tensor& ids, const Tensor& pos, ops::Gemma4AttentionEnvelope envelope,
                       Activations activations, const FeatureTap* tap) {
    return pass_.forward(work_, ids, pos, {.layers = views_}, max_context_, envelope, activations,
                         tap);
}

speculation::TargetShape Target::shape() const noexcept {
    const auto& c = model_.config();
    return {c.hidden_size, c.vocab_size, c.layers};
}

Tensor Target::draft_forward(const Tensor& ids, const Tensor& pos,
                             ops::Gemma4AttentionEnvelope envelope, bool a16,
                             const FeatureTap* tap) {
    return forward(ids, pos, envelope, a16 ? Activations::A16Only : Activations::Recorded, tap);
}

void Target::greedy_logits(const Tensor& normed, Tensor& out, bool a16) {
    pass_.uncapped_head(work_, normed, out, a16 ? Activations::A16Only : Activations::Recorded);
}

void Target::draft_embedding(const Tensor& ids, Tensor& out) {
    // The table with any embedding multiplier folded in at conversion: what the model's
    // embedding module yields, before the embedding norm (Muse Glimmer) the forward applies.
    ops::embedding(ids, model_.weights().token_embedding, out, device_.stream);
}

Tensor DensePass::forward(DeviceArena& work_, const Tensor& ids, const Tensor& pos, KvBindings kv,
                          std::int32_t session_context, ops::Gemma4AttentionEnvelope envelope,
                          Activations activations, const FeatureTap* tap) const {
    const cudaStream_t s     = device_.stream;
    const std::int32_t t     = ids.ne[0];
    const std::int32_t batch = kv.batch;
    if (static_cast<std::int32_t>(kv.layers.size()) != c_.layers || batch < 1 || t % batch != 0) {
        throw std::invalid_argument("dense forward: KV bindings do not match the layers or T");
    }
    for (const auto& group : c_.rope_groups) {
        if (!group.long_inv_freq.empty() && session_context <= 0) {
            throw std::invalid_argument("rope group " + group.name +
                                        " is LongRoPE: its rotation mode is fixed by the "
                                        "session's context, which this call was not given");
        }
    }
    if (tap != nullptr && (static_cast<std::int32_t>(tap->slots.size()) != c_.layers ||
                           tap->sink.ne[1] != t)) {
        throw std::invalid_argument("dense forward: the feature tap does not match the layers or T");
    }
    // A batched call runs attention over [.., T/B, B]; every other step is per column.
    const std::int32_t row_tokens = t / batch;
    const Tensor attention_pos    = batch == 1 ? pos : pos.view({row_tokens, batch});
    const auto& w        = model_.weights();
    const auto eps       = c_.rms_norm_eps;
    const auto d         = c_.head_dim;
    const auto q_rows    = c_.query_heads * d;
    const auto kv_rows   = c_.kv_heads * d;
    const bool pre       = c_.pre_norms();
    Tensor x             = work_.alloc(DType::BF16, {c_.hidden_size, t});
    Tensor h             = work_.alloc(DType::BF16, {c_.hidden_size, t});
    ops::embedding(ids, w.token_embedding, x, s);
    if (c_.embedding_norm) {
        // RMS-normalized without weight (gain 1); h holds it, then the two swap roles.
        ops::rmsnorm(x, w.unit_gain_hidden, eps, false, h, s);
        std::swap(x, h);
    }
    const bool offset    = c_.norm_unit_offset;
    const float post_eps = c_.post_norm_eps;
    // A pre-norm layer without post-norms and with a unit residual multiplier adds its attention
    // and feed-forward outputs into x in the producing kernel's epilogue (the output projection's,
    // the down projection's, the MoE combine's), and the following norm reads x alone: the update
    // residual_norm makes, without the branch plane.
    if (pre) { ops::rmsnorm(x, w.layers.front().attention_norm, eps, offset, h, s); }
    std::optional<DumpCall> dump_call;
    last_dump_prefix.clear();
    if (batch == 1 && dumping(s)) {
        static std::atomic<int> calls{0};
        std::int32_t p0 = 0;
        // The positions arrive on s; the legacy-stream copy below does not order against it.
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaMemcpy(&p0, pos.data, sizeof(p0), cudaMemcpyDeviceToHost));
        dump_call = DumpCall{.prefix = dump_config()->dir + "/c" + std::to_string(calls++) + "_p" +
                                       std::to_string(p0) + "_t" + std::to_string(t),
                             .p0     = p0,
                             .t      = t};
        dump(*dump_call, x, -1, "in");
    }

    for (std::int32_t i = 0; i < c_.layers; ++i) {
        const auto& layer = w.layers[i];
        const auto& spec  = c_.layer_specs[i];
        auto scope        = work_.scope();
        try {
            nvtx::ScopedRange range(spec.window > 0 ? nvtx::Name::PrefillLayerGdn
                                                    : nvtx::Name::PrefillLayerFull,
                                    nvtx::Category::Attention, i);
            const Tensor& attention_input = pre ? h : x;
            ops::DenseQkvPrepare prep;
            // The attention output gate reads the attention input too: at decode widths it is
            // computed beside q/k/v (project_side_by_side) and applied after attention.
            Tensor gate = layer.attention_gate ? work_.alloc(DType::BF16, {q_rows, t}) : Tensor{};
            bool gate_done = false;
            // Whether the last q/k/v projection released the preparation (not under a dump, which
            // copies between them; a side-by-side launch does not).
            bool qkv_released             = false;
            const auto release_or_project = [&](const Projection& p, Tensor& out) {
                if (dump_call) {
                    project(work_, attention_input, p, out, activations);
                    return false;
                }
                return project_release_next(work_, attention_input, p, out, activations);
            };
            if (layer.gate_in_qkv) {
                // [q; k; gate; v], or [q; k; gate] with v apart (beside it in one launch where the
                // side-by-side route admits both).
                Tensor packed = work_.alloc(DType::BF16, {layer.qkv->weight.n, t});
                if (layer.value.weight.n != 0) {
                    prep.v_src            = work_.alloc(DType::BF16, {kv_rows, t});
                    const Projection* p[] = {&*layer.qkv, &layer.value};
                    Tensor* o[]           = {&packed, &prep.v_src};
                    if (!project_side_by_side(work_, attention_input, p, o, activations)) {
                        project(work_, attention_input, *layer.qkv, packed, activations);
                        project(work_, attention_input, layer.value, prep.v_src, activations);
                    }
                } else {
                    project(work_, attention_input, *layer.qkv, packed, activations);
                    prep.v_src = packed.slice(0, 2 * q_rows + kv_rows, kv_rows);
                }
                prep.q_src = packed.slice(0, 0, q_rows);
                prep.k_src = packed.slice(0, q_rows, kv_rows);
                gate       = packed.slice(0, q_rows + kv_rows, q_rows);
                gate_done  = true;
            } else if (layer.qkv && layer.value.weight.n != 0) {
                // [q; k] packed, v apart in its own type (beside it in one launch where the
                // side-by-side route admits both).
                Tensor packed         = work_.alloc(DType::BF16, {layer.qkv->weight.n, t});
                prep.v_src            = work_.alloc(DType::BF16, {kv_rows, t});
                const Projection* p[] = {&*layer.qkv, &layer.value,
                                         layer.attention_gate ? &*layer.attention_gate : nullptr};
                Tensor* o[]           = {&packed, &prep.v_src, &gate};
                const std::size_t n   = layer.attention_gate ? 3 : 2;
                const bool joined =
                    project_side_by_side(work_, attention_input, std::span(p).first(n),
                                         std::span(o).first(n), activations);
                gate_done = joined && layer.attention_gate;
                if (!joined) {
                    project(work_, attention_input, *layer.qkv, packed, activations);
                    project(work_, attention_input, layer.value, prep.v_src, activations);
                }
                prep.q_src = packed.slice(0, 0, q_rows);
                prep.k_src = packed.slice(0, q_rows, kv_rows);
            } else if (layer.qkv) {
                Tensor packed = work_.alloc(DType::BF16, {layer.qkv->weight.n, t});
                const Projection* p[] = {&*layer.qkv, layer.attention_gate ? &*layer.attention_gate
                                                                           : nullptr};
                Tensor* o[]           = {&packed, &gate};
                gate_done = layer.attention_gate &&
                            project_side_by_side(work_, attention_input, p, o, activations);
                if (!gate_done) qkv_released = release_or_project(*layer.qkv, packed);
                prep.q_src = packed.slice(0, 0, q_rows);
                prep.k_src = packed.slice(0, q_rows, kv_rows);
                prep.v_src = packed.slice(0, q_rows + kv_rows, kv_rows);
            } else {
                prep.q_src = work_.alloc(DType::BF16, {q_rows, t});
                prep.k_src = work_.alloc(DType::BF16, {kv_rows, t});
                prep.v_src = work_.alloc(DType::BF16, {kv_rows, t});
                const Projection* p[] = {&layer.query, &layer.key, &layer.value,
                                         layer.attention_gate ? &*layer.attention_gate : nullptr};
                Tensor* o[]           = {&prep.q_src, &prep.k_src, &prep.v_src, &gate};
                const std::size_t n   = layer.attention_gate ? 4 : 3;
                const bool joined     = project_side_by_side(
                    work_, attention_input, std::span(p).first(n), std::span(o).first(n),
                    activations);
                gate_done = joined && layer.attention_gate;
                if (!joined) {
                    project(work_, attention_input, layer.query, prep.q_src, activations);
                    project(work_, attention_input, layer.key, prep.k_src, activations);
                    qkv_released = release_or_project(layer.value, prep.v_src);
                }
            }
            prep.q_bias  = layer.query_bias;
            prep.k_bias  = layer.key_bias;
            prep.v_bias  = layer.value_bias;
            prep.qk_norm = c_.qk_norm;
            prep.q_norm  = c_.qk_norm_scaled ? layer.query_norm : w.unit_gain_head;
            prep.k_norm  = c_.qk_norm_scaled ? layer.key_norm : w.unit_gain_head;
            prep.norm_unit_offset = c_.qk_norm_scaled && offset;
            prep.eps     = eps;
            if (spec.rope >= 0) {
                const auto& group  = c_.rope_groups[spec.rope];
                prep.inv_freq      = group.long_mode(session_context) ? w.long_inv_freq[spec.rope]
                                                                      : w.inv_freq[spec.rope];
                prep.rotary_dim    = group.rotary_dim;
                prep.rotary_scale  = group.attention_scaling;
            }
            prep.query_beta    = c_.query_beta;
            prep.query_period  = c_.query_period;
            prep.join_previous = qkv_released;
            prep.positions    = pos;
            prep.q_out        = work_.alloc(DType::BF16, {d, c_.query_heads, t});
            prep.cache        = kv.layers[i];
            // The decode graph stays one chain: prefetching the output projection's weights into
            // L2 on a side branch forked here measured 0.4-1.1 % slower (RTX 5090, a 30B NVFP4
            // model), since any fork widens every kernel boundary of the layer and the GEMM gains
            // little from L2-resident weights.
            ops::dense_qkv_prepare(prep, s);
            if (dump_call) { dump(*dump_call, prep.q_out, i, "q"); }

            Tensor context = work_.alloc(DType::BF16, {d, c_.query_heads, t});
            {
                // Ops that own a workspace see all of it as theirs; give them a carved span.
                auto attention_scope = work_.scope();
                WorkspaceArena scratch(work_.alloc_bytes(std::max<std::size_t>(
                    ops::gemma4_attention_batch_workspace_bytes(work_, batch, d, c_.query_heads,
                                                                c_.kv_heads, envelope,
                                                                kv.layers[i].window),
                    256)));
                const Tensor* sinks = layer.sinks.data != nullptr ? &layer.sinks : nullptr;
                if (batch == 1) {
                    ops::gemma4_attention(prep.q_out, pos, kv.layers[i], c_.attention_scale,
                                          envelope, scratch, context, s,
                                          ops::Gemma4AttentionMask::Causal, sinks);
                } else {
                    const Tensor q_rows = prep.q_out.view({d, c_.query_heads, row_tokens, batch});
                    Tensor context_rows = context.view({d, c_.query_heads, row_tokens, batch});
                    ops::gemma4_attention(q_rows, attention_pos, kv.layers[i], c_.attention_scale,
                                          envelope, scratch, context_rows, s,
                                          ops::Gemma4AttentionMask::Causal, sinks);
                }
            }
            Tensor context_rows = context.view({q_rows, t});
            if (layer.attention_gate || layer.gate_in_qkv) {
                if (!gate_done) {
                    project(work_, attention_input, *layer.attention_gate, gate, activations);
                }
                ops::sigmoid_mul(gate, context_rows, s);
            }
            const bool fold =
                pre && !dump_call && !c_.post_norms() && c_.residual_multiplier == 1.0F;
            Tensor branch    = work_.alloc(DType::BF16, {c_.hidden_size, t});
            Tensor ffn_input = pre ? work_.alloc(DType::BF16, {c_.hidden_size, t}) : Tensor{};
            if (fold && layer.output_bias.data == nullptr) {
                {
                    auto add_scope = work_.scope();
                    ops::linear_add(context_rows, layer.output.weight, x,
                                    policy_of(layer.output, activations), work_, s);
                }
                ops::rmsnorm(x, layer.ffn_norm, eps, offset, ffn_input, s);
            } else {
                bool output_released = false;
                if (dump_call) {
                    project(work_, context_rows, layer.output, branch, activations);
                } else {
                    output_released = project_release_next(work_, context_rows, layer.output,
                                                           branch, activations);
                }
                if (dump_call) {
                    dump(*dump_call, context, i, "attn");
                    dump(*dump_call, branch, i, "ao");
                }
                ops::residual_norm(
                    {.x           = x,
                     .branch      = branch,
                     .bias        = layer.output_bias,
                     .post_weight = c_.post_norms() ? layer.attention_post_norm : Tensor{},
                     .scale       = c_.residual_multiplier,
                     .eps         = eps,
                     .post_eps    = post_eps,
                     .unit_offset = offset,
                     .out_weight  = pre ? layer.ffn_norm : Tensor{},
                     .out         = ffn_input,
                     .join_previous = output_released},
                    s);
            }
            const Tensor& mlp_input = pre ? ffn_input : x;
            bool added              = false; // the feed-forward added into x itself
            bool mlp_released       = false; // the down projection released the closing norm

            if (layer.moe) {
                // The routed feed-forward writes the branch; the closing update below adds it.
                auto moe_scope = work_.scope();
                WorkspaceArena scratch(work_.alloc_bytes(std::max<std::size_t>(
                    ops::moe_workspace_bytes(*layer.moe, c_.moe->op, t, t), 256)));
                added = fold;
                ops::moe(mlp_input, *layer.moe, c_.moe->op,
                         fold ? ops::MoeEpilogue::AddResidual : ops::MoeEpilogue::Store,
                         fold ? x : branch, scratch, s);
            } else {
                added = fold;
                mlp_released = dense_mlp(work_, layer, mlp_input, fold ? x : branch, t, activations,
                                         fold, !dump_call);
            }
            if (dump_call) { dump(*dump_call, branch, i, "mlp"); }
            // The closing update writes the next reader's normalized input: the next layer's
            // pre-attention norm, the final norm after the last layer, or nothing between
            // post-norm layers (they read x itself).
            const bool last        = i + 1 == c_.layers;
            const Tensor* next     = last  ? &w.final_norm
                                     : pre ? &w.layers[i + 1].attention_norm
                                           : nullptr;
            if (added) {
                if (next) {
                    ops::rmsnorm(x, *next, eps, last ? c_.final_norm_unit_offset : offset, h, s);
                }
            } else {
                ops::residual_norm({.x           = x,
                                    .branch      = branch,
                                    .post_weight = c_.post_norms() ? layer.ffn_post_norm : Tensor{},
                                    .scale       = c_.residual_multiplier,
                                    .eps         = eps,
                                    .post_eps    = post_eps,
                                    .unit_offset = offset,
                                    .out_weight  = next ? *next : Tensor{},
                                    .out         = next ? h : Tensor{},
                                    .out_unit_offset = last ? c_.final_norm_unit_offset : offset,
                                    .join_previous   = mlp_released},
                                   s);
            }
            if (dump_call) { dump(*dump_call, x, i, "x"); }
            // x is now this layer's output: a tapped layer copies it into its row block of the
            // sink, or hands it to the tap's projection once its own buffers are free again.
            if (tap != nullptr && tap->slots[i] >= 0) {
                if (tap->project) {
                    { const auto released = std::move(scope); }
                    tap->project(tap->slots[i], x);
                } else {
                    const std::size_t row = bf16_bytes(c_.hidden_size);
                    CUDA_CHECK(cudaMemcpy2DAsync(
                        static_cast<char*>(tap->sink.data) + tap->slots[i] * row,
                        tap->sink.nb[1], x.data, row, row, t, cudaMemcpyDeviceToDevice, s));
                }
            }
        } catch (const std::exception&) {
            rethrow_with_context("text/layers/" + std::to_string(i) + " T=" + std::to_string(t));
        }
    }
    if (dump_call) {
        dump(*dump_call, h, c_.layers, "h");
        // The head of this call dumps its logits only when its last column is in the window.
        const LayerDump& c   = *dump_config();
        const long long last = dump_call->p0 + dump_call->t - 1;
        if (last >= c.lo && last < c.hi) { last_dump_prefix = dump_call->prefix; }
    }
    return h;
}

void Target::head(const Tensor& normed, Tensor& out, Activations activations) {
    pass_.head(work_, normed, out, activations);
}

void DensePass::uncapped_head(DeviceArena& work, const Tensor& normed, Tensor& out,
                              Activations activations) const {
    project(work, normed, model_.weights().output_head, out, activations);
}

void DensePass::head(DeviceArena& work, const Tensor& normed, Tensor& out,
                     Activations activations) const {
    project(work, normed, model_.weights().output_head, out, activations);
    if (c_.softcap) {
        ops::logit_softcap(out, *c_.softcap, device_.stream, c_.logit_multiplier);
    }
    if (!last_dump_prefix.empty() && dumping(device_.stream)) {
        write_device_bytes(out.data, out.bytes(), last_dump_prefix + ".logits.bin", false);
        last_dump_prefix.clear();
    }
}

} // namespace ninferno::models::dense
