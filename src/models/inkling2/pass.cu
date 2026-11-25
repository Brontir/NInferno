// The Inkling 2 forward over bindings (pass.h).
#include "models/inkling2/pass.h"

#include "ninferno/ops/embedding.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/logit_softcap.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/residual_add.h"
#include "ninferno/ops/rmsnorm.h"
#include "ninferno/ops/sigmoid_mul.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninferno::models::inkling2 {
namespace {

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

// The persistent stream planes: two for the stream (each visit writes the one the stream is not
// in), the residual input, the attention output, the loop's prelude and its last pass's output,
// the final hidden, then the blocks and the loop's bank.
enum Plane : std::int32_t { kCur0, kCur1, kX, kAttn, kPrelude, kBankOut, kHidden, kBlocks };

} // namespace

Pass::Pass(const Model& model)
    : model_(model), c_(model.config()), visits_(visit_schedule(model.config())) {
    for (const auto& v : visits_) { slots_ = std::max(slots_, v.slot + 1); }
    // Blocks: one a layer that starts one (layer % block_size == 0); the bank: the core's.
    for (std::int32_t l = 0; l < c_.layers; ++l) {
        if (l % c_.block_size == 0) {
            ++max_blocks_;
            if (c_.looped() && l >= c_.core_start && l < c_.core_end) { ++max_bank_; }
        }
    }
}

std::size_t Pass::work_bytes(const Model& model, std::int32_t columns) {
    const Config& c = model.config();
    std::size_t moe = 256;
    for (const auto& layer : model.weights().layers) {
        moe =
            std::max(moe, ops::moe_workspace_bytes(layer.moe, layer.moe_config, columns, columns));
    }
    // The stream planes (kBlocks + the blocks + the bank, below 16 H rows) and a visit's
    // temporaries at once (messages, mixes, the controller, the norms, the projections, the
    // context, the MoE's input and output): below 40 H BF16 rows a column; the projections' own
    // scratch (an A8 call's quantized activation) below 4 H bytes a column more.
    const std::size_t per_column = std::size_t(60) * c.hidden_size * 2;
    return per_column * columns + moe + (std::size_t(64) << 20);
}

Tensor Pass::forward(DeviceArena& work, const Tensor& tokens, const Tensor& positions,
                     const Bindings& bound, cudaStream_t s) {
    const std::int32_t n = tokens.ne[0], B = bound.batch;
    if (n < 1 || B < 1 || n % B != 0 || positions.ne[0] != n ||
        bound.kv.size() != static_cast<std::size_t>(slots_) ||
        bound.conv.size() != static_cast<std::size_t>(slots_)) {
        throw std::invalid_argument(
            "inkling2: a call takes T * B columns and a binding a visit slot");
    }
    const std::int32_t T = n / B;
    const auto H = c_.hidden_size, D = c_.head_dim, Hq = c_.query_heads, Hkv = c_.kv_heads,
               Dr = c_.rel_width;
    const auto KV = D * Hkv;
    const auto& W = model_.weights();
    // The call's stream planes, in `work` for the call (the caller's scope frees them).
    const Tensor planes = work.alloc(DType::BF16, {H, n * (kBlocks + max_blocks_ + max_bank_)});
    const auto P        = [&](std::int32_t index) {
        return Tensor(static_cast<__nv_bfloat16*>(planes.data) + std::size_t(index) * H * n,
                             DType::BF16, {H, n});
    };
    const auto tmp  = [&](std::int32_t rows) { return work.alloc(DType::BF16, {rows, n}); };
    const auto copy = [&](const Tensor& from, Tensor to) {
        check(cudaMemcpyAsync(to.data, from.data, std::size_t(H) * n * 2, cudaMemcpyDeviceToDevice,
                              s),
              "stream copy");
        return to;
    };
    const auto mix = [&](const std::vector<Tensor>& states, const Tensor& cur, const Tensor& query,
                         std::span<Tensor> outs) {
        ops::Inkling2MixStates in;
        for (const auto& b : states) { in.data[static_cast<std::size_t>(in.count++)] = b.data; }
        in.data[static_cast<std::size_t>(in.count++)] = cur.data;
        ops::Inkling2MixOutputs out;
        for (auto& o : outs) { out.data[static_cast<std::size_t>(out.count++)] = o.data; }
        const Tensor q(query.data, DType::FP32, {H, static_cast<std::int32_t>(outs.size())});
        auto scope           = work.scope();
        const Tensor scratch = work.alloc(
            DType::FP32,
            {static_cast<std::int32_t>(ops::inkling2_depth_mix_scratch_bytes(H, n) / 4)});
        ops::inkling2_depth_mix(in, q, H, n, out, scratch, s);
    };
    const auto control = [&](const Tensor& cur, const Tensor& msg, const Controller& ctrl) {
        Tensor pair = tmp(2 * H);
        ops::inkling2_norm_pair(cur, msg, pair, s);
        Tensor z = tmp(c_.gate_rank), gain = tmp(H), forget = tmp(H), out = tmp(H);
        project(pair, ctrl.down, z, work, s);
        project(z, ctrl.gain, gain, work, s);
        project(z, ctrl.forget, forget, work, s);
        ops::inkling2_control(cur, msg, gain, forget, out, s);
        return out;
    };
    const auto rms = [&](const Tensor& x, const Tensor& w) {
        Tensor out = tmp(H);
        ops::rmsnorm(x, w, c_.rms_norm_eps, false, out, s);
        return out;
    };
    // The slot's convolution state planes from channel `offset` on.
    const auto conv_at = [&](std::int32_t slot, std::int32_t offset) {
        ops::Inkling2ConvState st = bound.conv[static_cast<std::size_t>(slot)];
        st.data += std::size_t(offset) * 3;
        return st;
    };
    std::int32_t visit_index = 0;
    const auto see           = [&](const char* name, const Tensor& t) {
        if (observe && visit_index == 0) {
            check(cudaStreamSynchronize(s), "observe");
            observe(name, -1, t);
        }
    };

    // Token embeddings, normalized by tok_norm.
    std::int32_t stream = kCur0;
    Tensor cur          = P(stream);
    {
        auto scope = work.scope();
        Tensor raw = tmp(H);
        ops::embedding(tokens, W.token_embedding.weight, raw, s);
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
                prelude        = copy(cur, P(kPrelude));
                prelude_blocks = blocks;
            } else {
                // The bank: the previous pass's blocks after the prelude's and the block the
                // pass opened with (the prelude itself), copied out of the block planes this pass
                // reuses.
                bank.clear();
                for (std::size_t i = prelude_blocks.size() + 1; i < blocks.size(); ++i) {
                    bank.push_back(copy(blocks[i], P(kBlocks + max_blocks_ +
                                                     static_cast<std::int32_t>(bank.size()))));
                }
                bank_out = copy(cur, P(kBankOut));
                cur      = prelude;
                blocks   = prelude_blocks;
            }
        }
        const bool messages = core && pass > 0;
        auto scope          = work.scope();
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
        project(hq, L.query, q, work, s);
        project(hk, L.key, k, work, s);
        project(hv, L.value, vv, work, s);
        project(hq, L.relative, r, work, s);
        project(hq, L.sink, sink, work, s);
        project(hq, L.gate, gate, work, s);
        see("node_14", hq);
        see("node_15", k);
        see("node_30", q);
        see("node_65", sink);
        Tensor ktaps, vtaps;
        if (full) {
            ktaps = tmp(Hkv * c_.conv_taps), vtaps = tmp(Hkv * c_.conv_taps);
            project(hk, L.key_taps, ktaps, work, s);
            project(hv, L.value_taps, vtaps, work, s);
        }
        {
            Tensor k_rows = k.view({KV, T, B}), v_rows = vv.view({KV, T, B});
            const Tensor kt = full ? ktaps.view({Hkv * c_.conv_taps, T, B}) : Tensor();
            const Tensor vt = full ? vtaps.view({Hkv * c_.conv_taps, T, B}) : Tensor();
            ops::inkling2_conv(k_rows, L.key_conv, kt, Hkv, conv_at(v.slot, 0), k_rows, s);
            ops::inkling2_conv(v_rows, L.value_conv, vt, Hkv, conv_at(v.slot, KV), v_rows, s);
        }
        Tensor q3(q.data, DType::BF16, {D, Hq, n}), k3(k.data, DType::BF16, {D, Hkv, n});
        Tensor v3(vv.data, DType::BF16, {D, Hkv, n}), r3(r.data, DType::BF16, {Dr, Hq, n});
        const ops::Inkling2PrepareParams prep{
            c_.qk_clip, c_.rel_clip, c_.rms_norm_eps, c_.log_scale_alpha, c_.log_scale_floor, full};
        ops::inkling2_attention_prepare(q3, k3, v3, r3, L.query_clip, L.relative_clip, positions,
                                        prep, s);
        see("Qcur-0", q);
        see("Kcur-0", k);
        see("Vcur-0", vv);
        see("Rcur-0", r);
        const ops::Gemma4KvLayer& kv = bound.kv[static_cast<std::size_t>(v.slot)];
        const Tensor rows_pos        = positions.view({T, B});
        ops::gemma4_kv_append(k.view({D, Hkv, T, B}), vv.view({D, Hkv, T, B}), rows_pos, kv, s);
        Tensor context = tmp(Hq * D);
        Tensor c4      = context.view({D, Hq, T, B});
        ops::inkling2_attention(q.view({D, Hq, T, B}), r.view({Dr, Hq, T, B}),
                                sink.view({Hq, T, B}), L.relative_bias, kv, rows_pos,
                                c_.extent(v.layer), c4, s);
        see("kqv_out-0", context);
        ops::sigmoid_mul(gate, context, s);
        see("node_83", context);
        Tensor attn = P(kAttn);
        project(context, L.output, attn, work, s);
        see("node_84", attn);
        {
            Tensor a_rows = attn.view({H, T, B});
            ops::inkling2_conv(a_rows, L.output_conv, Tensor(), 1, conv_at(v.slot, 2 * KV), a_rows,
                               s);
        }
        see("ffn_inp-0", attn);

        // The first layer of each block starts a new residual stream; the earlier one joins the
        // blocks.
        Tensor x;
        if (v.layer % c_.block_size == 0) {
            blocks.push_back(copy(cur, P(kBlocks + static_cast<std::int32_t>(blocks.size()))));
            x = attn;
        } else {
            x = copy(cur, P(kX));
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
            auto moe_scope = work.scope();
            WorkspaceArena scratch(work.alloc_bytes(
                std::max<std::size_t>(ops::moe_workspace_bytes(L.moe, L.moe_config, n, n), 256)));
            ops::moe(fn, L.moe, L.moe_config, ops::MoeEpilogue::Store, y, scratch, s);
        }
        see("ffn_moe_out-0", y);
        {
            Tensor y_rows = y.view({H, T, B});
            ops::inkling2_conv(y_rows, L.ffn_conv, Tensor(), 1, conv_at(v.slot, 2 * KV + H), y_rows,
                               s);
        }
        // The stream's next plane: the one it is not in (the prelude, a block plane or the other).
        stream     = cur.data == P(kCur0).data ? kCur1 : kCur0;
        Tensor out = copy(x, P(stream));
        ops::residual_add(y, out, s);
        cur = out;
        if (observe) {
            check(cudaStreamSynchronize(s), "observe");
            observe("l_out", visit_index, cur);
        }
        ++visit_index;
    }

    // Output: the depth mix of the blocks and the stream, then the final norm, every column.
    auto scope    = work.scope();
    Tensor mixed  = tmp(H);
    Tensor hidden = P(kHidden);
    mix(blocks, cur, W.output_query, std::span<Tensor>(&mixed, 1));
    ops::rmsnorm(mixed, W.final_norm, c_.rms_norm_eps, false, hidden, s);
    return hidden;
}

void Pass::head(DeviceArena& work, const Tensor& hidden, Tensor& logits, cudaStream_t s) const {
    project(hidden, model_.weights().output_head, logits, work, s);
    ops::logit_softcap(logits, c_.softcap, s);
}

} // namespace ninferno::models::inkling2
