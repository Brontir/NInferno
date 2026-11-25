// The Inkling 2 forward (target.h): llama.cpp inkling2.cpp build_layer / build_output and
// inkling2-loop.cpp's schedule, op by op.
#include "models/inkling2/target.h"

#include "ninferno/ops/embedding.h"
#include "ninferno/ops/inkling2.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/logit_softcap.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/residual_add.h"
#include "ninferno/ops/rmsnorm.h"
#include "ninferno/ops/sigmoid_mul.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ninferno::models::inkling2 {
namespace {

constexpr std::size_t kWorkBytes = std::size_t(1)
                                   << 30; // per-call scratch (projections, MoE, attention)

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("inkling2: ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

void project(const Tensor& x, const Projection& p, Tensor& out, DeviceArena& work, cudaStream_t s) {
    auto scope = work.scope();
    ops::linear(x, p.weight, out, p.policy, work, s);
}

} // namespace

Target::Target(const Model& model, std::int32_t max_context, std::int32_t max_tokens)
    : model_(model), c_(model.config()), visits_(visit_schedule(model.config())),
      max_context_(max_context), max_tokens_(max_tokens), work_(kWorkBytes) {
    if (max_tokens < 1 || max_context < max_tokens) {
        throw std::invalid_argument("inkling2: max_tokens in [1, max_context]");
    }
    std::int32_t slots = 0;
    for (const auto& v : visits_) { slots = std::max(slots, v.slot + 1); }
    slots_.resize(static_cast<std::size_t>(slots));
    for (const auto& v : visits_) {
        auto& slot = slots_[static_cast<std::size_t>(v.slot)];
        if (slot.capacity) { continue; }
        const bool full = c_.kinds[static_cast<std::size_t>(v.layer)] == LayerKind::Full;
        slot.capacity =
            full ? max_context : std::min(max_context, c_.sliding_window + max_tokens - 1);
        const std::size_t plane = std::size_t(c_.head_dim) * slot.capacity * c_.kv_heads * 2;
        slot.k                  = DeviceBuffer(plane);
        slot.v                  = DeviceBuffer(plane);
        slot.conv               = DeviceBuffer(std::size_t(c_.conv_channels()) * 3 * sizeof(float));
    }
    ids_       = DeviceBuffer(std::size_t(max_tokens) * sizeof(std::int32_t));
    positions_ = DeviceBuffer(std::size_t(max_tokens) * sizeof(std::int32_t));
    logits_    = DeviceBuffer(std::size_t(c_.vocab_rows) * 2);
    reset();
}

void Target::reset() {
    for (auto& slot : slots_) {
        slot.conv.fill(0);
        slot.k.fill(0);
        slot.v.fill(0);
    }
    position_ = 0;
}

Tensor Target::state(std::int32_t slot, std::int32_t offset, std::int32_t channels) {
    auto& s = slots_[static_cast<std::size_t>(slot)];
    return Tensor(static_cast<float*>(s.conv.p) + std::size_t(offset) * 3, DType::FP32,
                  {3, channels});
}

Tensor Target::stream_buffer() {
    if (pool_next_ == pool_.size()) {
        pool_.emplace_back(std::size_t(c_.hidden_size) * max_tokens_ * 2);
    }
    return Tensor(pool_[pool_next_++].p, DType::BF16, {c_.hidden_size, max_tokens_});
}

std::vector<float> Target::forward(std::span<const std::int32_t> ids, cudaStream_t s) {
    const std::int32_t T = static_cast<std::int32_t>(ids.size());
    if (T < 1 || T > max_tokens_ || position_ + T > max_context_) {
        throw std::invalid_argument(
            "inkling2: a call takes 1..max_tokens tokens inside max_context");
    }
    const auto H = c_.hidden_size, D = c_.head_dim, Hq = c_.query_heads, Hkv = c_.kv_heads,
               Dr = c_.rel_width;
    const auto KV = D * Hkv;
    const auto& W = model_.weights();
    pool_next_    = 0;
    work_.reset();
    std::vector<std::int32_t> pos(static_cast<std::size_t>(T));
    for (std::int32_t t = 0; t < T; ++t) { pos[static_cast<std::size_t>(t)] = position_ + t; }
    check(cudaMemcpyAsync(ids_.p, ids.data(), ids.size_bytes(), cudaMemcpyHostToDevice, s),
          "ids upload");
    check(cudaMemcpyAsync(positions_.p, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice, s),
          "positions upload");
    const Tensor ids_t(ids_.p, DType::I32, {T});
    const Tensor positions(positions_.p, DType::I32, {T});
    const auto fresh = [&]() { // a [H, T] stream tensor that lives for the whole call
        Tensor b = stream_buffer();
        return Tensor(b.data, DType::BF16, {H, T});
    };
    const auto tmp = [&](std::int32_t rows) { return work_.alloc(DType::BF16, {rows, T}); };
    const auto mix = [&](const std::vector<Tensor>& states, const Tensor& cur, const Tensor& query,
                         std::span<Tensor> outs) {
        ops::Inkling2MixStates in;
        for (const auto& b : states) { in.data[static_cast<std::size_t>(in.count++)] = b.data; }
        in.data[static_cast<std::size_t>(in.count++)] = cur.data;
        ops::Inkling2MixOutputs out;
        for (auto& o : outs) { out.data[static_cast<std::size_t>(out.count++)] = o.data; }
        const Tensor q(query.data, DType::FP32, {H, static_cast<std::int32_t>(outs.size())});
        ops::inkling2_depth_mix(in, q, H, T, out, s);
    };
    const auto control = [&](const Tensor& cur, const Tensor& msg, const Controller& ctrl) {
        Tensor pair = tmp(2 * H);
        ops::inkling2_norm_pair(cur, msg, pair, s);
        Tensor z = tmp(c_.gate_rank), gain = tmp(H), forget = tmp(H), out = tmp(H);
        project(pair, ctrl.down, z, work_, s);
        project(z, ctrl.gain, gain, work_, s);
        project(z, ctrl.forget, forget, work_, s);
        ops::inkling2_control(cur, msg, gain, forget, out, s);
        return out;
    };
    const auto rms = [&](const Tensor& x, const Tensor& w) {
        Tensor out = tmp(H);
        ops::rmsnorm(x, w, c_.rms_norm_eps, false, out, s);
        return out;
    };

    // The first visit's intermediate tensors, under llama.cpp's names (llama-eval-callback), for a
    // layer-by-layer comparison.
    std::int32_t visit_index = 0;
    const auto see           = [&](const char* name, const Tensor& t) {
        if (observe && visit_index == 0) {
            check(cudaStreamSynchronize(s), "observe");
            observe(name, -1, t);
        }
    };

    // Token embeddings, normalized by tok_norm.
    Tensor cur = fresh();
    {
        Tensor raw = tmp(H);
        ops::embedding(ids_t, W.token_embedding.weight, raw, s);
        ops::rmsnorm(raw, W.embedding_norm, c_.rms_norm_eps, false, cur, s);
    }
    see("node_12", cur);

    std::vector<Tensor> blocks, prelude_blocks, bank;
    Tensor prelude, bank_out;
    std::int32_t pass = -1;
    for (const auto& v : visits_) {
        const Layer& L  = W.layers[static_cast<std::size_t>(v.layer)];
        const bool core = c_.looped() && v.layer >= c_.core_start && v.layer < c_.core_end;
        if (core && v.layer == c_.core_start) { // a pass starts: every pass starts from the prelude
            if (++pass == 0) {
                prelude        = cur;
                prelude_blocks = blocks;
            } else {
                bank.assign(blocks.begin() + std::ptrdiff_t(prelude_blocks.size()) + 1,
                            blocks.end());
                bank_out = cur;
                cur      = prelude;
                blocks   = prelude_blocks;
            }
        }
        const bool messages = core && pass > 0;
        auto scope          = work_.scope();
        const bool full     = L.kind == LayerKind::Full;

        // The loop's four messages (q, k, v, ffn) from the previous pass.
        Tensor msg[4];
        if (messages) {
            for (auto& m : msg) { m = tmp(H); }
            mix(bank, bank_out, L.message_query, std::span<Tensor>(msg, 4));
        }
        // Attention inputs.
        Tensor in[3] = {cur, cur, cur};
        if (L.attention_query.data) {
            const int m = full ? 3 : 1;
            Tensor mixed[3];
            for (int j = 0; j < m; ++j) { mixed[j] = tmp(H); }
            mix(blocks, cur, L.attention_query, std::span<Tensor>(mixed, std::size_t(m)));
            for (int j = 0; j < 3; ++j) { in[j] = mixed[full ? j : 0]; }
        }
        if (messages) {
            for (int j = 0; j < 3; ++j) {
                in[j] = control(in[j], msg[j], L.control[static_cast<std::size_t>(j)]);
            }
        }
        Tensor hq, hk, hv;
        if (full) {
            hq = rms(in[0], L.q_norm), hk = rms(in[1], L.k_norm), hv = rms(in[2], L.v_norm);
        } else if (messages) {
            hq = rms(in[0], L.input_norm), hk = rms(in[1], L.input_norm),
            hv = rms(in[2], L.input_norm);
        } else {
            hq = hk = hv = rms(in[0], L.input_norm);
        }

        // Self-attention.
        Tensor q = tmp(Hq * D), k = tmp(KV), vv = tmp(KV), r = tmp(Dr * Hq), sink = tmp(Hq),
               gate = tmp(Hq * D);
        project(hq, L.query, q, work_, s);
        project(hk, L.key, k, work_, s);
        project(hv, L.value, vv, work_, s);
        project(hq, L.relative, r, work_, s);
        project(hq, L.sink, sink, work_, s);
        project(hq, L.gate, gate, work_, s);
        see("node_14", hq);
        see("node_15", k);
        see("node_30", q);
        see("node_65", sink);
        Tensor ktaps, vtaps;
        if (full) {
            ktaps = tmp(Hkv * c_.conv_taps), vtaps = tmp(Hkv * c_.conv_taps);
            project(hk, L.key_taps, ktaps, work_, s);
            project(hv, L.value_taps, vtaps, work_, s);
        }
        {
            Tensor ks = state(v.slot, 0, KV), vs = state(v.slot, KV, KV);
            ops::inkling2_conv(k, L.key_conv, ktaps, Hkv, ks, k, s);
            ops::inkling2_conv(vv, L.value_conv, vtaps, Hkv, vs, vv, s);
        }
        Tensor q3(q.data, DType::BF16, {D, Hq, T}), k3(k.data, DType::BF16, {D, Hkv, T});
        Tensor v3(vv.data, DType::BF16, {D, Hkv, T}), r3(r.data, DType::BF16, {Dr, Hq, T});
        ops::Inkling2PrepareParams prep{c_.qk_clip,         c_.rel_clip,        c_.rms_norm_eps,
                                        c_.log_scale_alpha, c_.log_scale_floor, full};
        ops::inkling2_attention_prepare(q3, k3, v3, r3, L.query_clip, L.relative_clip, positions,
                                        prep, s);
        see("Qcur-0", q);
        see("Kcur-0", k);
        see("Vcur-0", vv);
        see("Rcur-0", r);
        auto& slot = slots_[static_cast<std::size_t>(v.slot)];
        ops::Gemma4KvLayer kv; // one sequence's slot storage (a ring on a sliding layer)
        kv.k      = Tensor(slot.k.p, DType::BF16, {D, slot.capacity, Hkv});
        kv.v      = Tensor(slot.v.p, DType::BF16, {D, slot.capacity, Hkv});
        kv.window = full ? 0 : c_.sliding_window;
        ops::gemma4_kv_append(k3, v3, positions, kv, s);
        Tensor context = tmp(Hq * D);
        Tensor c3(context.data, DType::BF16, {D, Hq, T});
        const Tensor sink3(sink.data, DType::BF16, {Hq, T, 1});
        ops::inkling2_attention(q3, r3, sink3, L.relative_bias, kv, positions.view({T, 1}),
                                c_.extent(v.layer), c3, s);
        see("kqv_out-0", context);
        ops::sigmoid_mul(gate, context, s);
        see("node_83", context);
        Tensor attn = fresh();
        project(context, L.output, attn, work_, s);
        see("node_84", attn);
        {
            Tensor os = state(v.slot, 2 * KV, H);
            ops::inkling2_conv(attn, L.output_conv, Tensor(), 1, os, attn, s);
        }
        see("ffn_inp-0", attn);

        // The first layer of each block starts a new residual stream; the earlier one joins the
        // blocks.
        Tensor x;
        if (v.layer % c_.block_size == 0) {
            blocks.push_back(cur);
            x = attn;
        } else {
            x = fresh();
            check(cudaMemcpyAsync(x.data, cur.data, std::size_t(H) * T * 2,
                                  cudaMemcpyDeviceToDevice, s),
                  "residual copy");
            ops::residual_add(attn, x, s);
        }

        // MoE.
        Tensor f = tmp(H);
        mix(blocks, x, L.ffn_query, std::span<Tensor>(&f, 1));
        if (messages) { f = control(f, msg[3], L.control[3]); }
        Tensor fn = rms(f, L.ffn_norm);
        see("node_108", fn);
        Tensor y = tmp(H);
        {
            auto moe_scope = work_.scope();
            WorkspaceArena scratch(work_.alloc_bytes(
                std::max<std::size_t>(ops::moe_workspace_bytes(L.moe, L.moe_config, T, T), 256)));
            ops::moe(fn, L.moe, L.moe_config, ops::MoeEpilogue::Store, y, scratch, s);
        }
        see("ffn_moe_out-0", y);
        {
            Tensor fs = state(v.slot, 2 * KV + H, H);
            ops::inkling2_conv(y, L.ffn_conv, Tensor(), 1, fs, y, s);
        }
        Tensor out = fresh();
        check(
            cudaMemcpyAsync(out.data, x.data, std::size_t(H) * T * 2, cudaMemcpyDeviceToDevice, s),
            "residual copy");
        ops::residual_add(y, out, s);
        cur = out;
        if (observe) {
            check(cudaStreamSynchronize(s), "observe");
            observe("l_out", visit_index, cur);
        }
        ++visit_index;
    }

    // Output: the depth mix of the blocks and the stream, the final norm, the head of the last
    // token.
    auto scope   = work_.scope();
    Tensor mixed = tmp(H);
    mix(blocks, cur, W.output_query, std::span<Tensor>(&mixed, 1));
    const Tensor last(static_cast<__nv_bfloat16*>(mixed.data) + std::size_t(H) * (T - 1),
                      DType::BF16, {H, 1});
    Tensor hidden = work_.alloc(DType::BF16, {H, 1});
    ops::rmsnorm(last, W.final_norm, c_.rms_norm_eps, false, hidden, s);
    if (observe) {
        check(cudaStreamSynchronize(s), "observe");
        observe("result_norm", -1, hidden);
    }
    Tensor logits(logits_.p, DType::BF16, {c_.vocab_rows, 1});
    project(hidden, W.output_head, logits, work_, s);
    ops::logit_softcap(logits, c_.softcap, s);
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(c_.vocab_size));
    check(cudaMemcpyAsync(bits.data(), logits_.p, bits.size() * 2, cudaMemcpyDeviceToHost, s),
          "logits download");
    check(cudaStreamSynchronize(s), "forward");
    position_ += T;
    std::vector<float> out(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        const std::uint32_t w = std::uint32_t(bits[i]) << 16;
        std::memcpy(&out[i], &w, 4);
    }
    return out;
}

} // namespace ninferno::models::inkling2
