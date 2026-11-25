#pragma once

// The Inkling 2 family's own Ops (docs/maintainer/inkling2-model.md). Each states its mathematics
// per token column; the oracles in tests/ops/test_inkling2_*.cpp evaluate it in FP64 from the
// represented inputs. n(x) = x / sqrt(mean(x^2) + 1e-6) over a column.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>

namespace ninferno::ops {

/// The most states one depth mix reads: every residual block of a 24-layer decoder (4 blocks of
/// the prelude and core, 2 of the coda) plus the current stream.
inline constexpr std::int32_t kInkling2MaxMixStates = 12;
/// The most queries one depth mix answers: q, k, v of a full layer, or the loop's four messages.
inline constexpr std::int32_t kInkling2MaxMixQueries = 4;

/**
 * Op: depth mix (llama.cpp inkling2.cpp graph::mix), attention over the depth.
 *
 *   S_s      = n(states[s][:, t])                                for s < count
 *   w_{r,s}  = softmax_s( S_s . query[r, :] / sqrt(H) )
 *   out_r[:, t] = sum_s w_{r,s} S_s                              for r < queries
 *
 * states: `count` contiguous BF16 [H, T] tensors (1 <= count <= kInkling2MaxMixStates), the
 * current stream last by convention (the result does not depend on the order). query: contiguous
 * FP32 [queries, H]. out: `queries` contiguous BF16 [H, T] tensors, disjoint from every state.
 * H a multiple of 8, T >= 1. The softmax and both dot products accumulate in FP32; each output
 * value is rounded to BF16 once.
 */
struct Inkling2MixStates {
    std::array<const void*, kInkling2MaxMixStates> data{};
    std::int32_t count = 0;
};
struct Inkling2MixOutputs {
    std::array<void*, kInkling2MaxMixQueries> data{};
    std::int32_t count = 0;
};
void inkling2_depth_mix(const Inkling2MixStates& states, const Tensor& query, std::int32_t hidden,
                        std::int32_t tokens, const Inkling2MixOutputs& out, cudaStream_t stream);

/**
 * Op: short causal convolution with its state and a residual (graph::convolve).
 *
 * With u[c, -3..-1] = state[c, 0..2] (the three inputs before this call) and u[c, t] = x[c, t]:
 *
 *   y[c, t] = (x[c, t] + sum_{k<4} weight[c, k] u[c, t - 3 + k]
 *              + [taps] sum_{lag<4} taps[(c / (C / heads)) * 4 + lag, t] u[c, t - lag]) * scale
 *   scale   = 1 / sqrt(4 * (taps ? 2 : 1) + 1)
 *
 * and the state becomes the last three u of the call: state[c, j] = u[c, T - 3 + j].
 *
 * x, out: contiguous BF16 [C, T] (out may alias x exactly). weight: contiguous FP32 [C, 4].
 * taps: empty, or contiguous BF16 [heads * 4, T] (a dynamic tap per head and token; C % heads ==
 * 0). state: contiguous FP32 [C, 3], read and replaced. One sequence; T >= 1.
 */
void inkling2_conv(const Tensor& x, const Tensor& weight, const Tensor& taps, std::int32_t heads,
                   Tensor& state, Tensor& out, cudaStream_t stream);

/**
 * Op: loop controller update (graph::control), the elementwise half after the low-rank pair.
 *
 *   out[:, t] = cur[:, t] - tanh(forget[:, t]) * cur[:, t] + tanh(gain[:, t]) * message[:, t]
 *
 * gain and forget are the controller's two rank-r projections of [n(cur); n(message)]. Every
 * tensor is contiguous BF16 [H, T]; out may alias cur exactly.
 */
void inkling2_control(const Tensor& cur, const Tensor& message, const Tensor& gain,
                      const Tensor& forget, Tensor& out, cudaStream_t stream);

/**
 * Op: [n(a); n(b)] stacked column by column into one [2H, T] input (the controller's pair).
 * a, b contiguous BF16 [H, T]; out contiguous BF16 [2H, T].
 */
void inkling2_norm_pair(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream);

/**
 * Op: the attention inputs of one layer visit (the clips, the value norm, the log scaling).
 *
 *   clip(x, c) = c x / max(|x|, c) over each head's row
 *   q[:, h, t] = clip(q_clip[h, :] * q[:, h, t], qk_clip) * tau_t
 *   k[:, g, t] = clip(k[:, g, t], qk_clip)
 *   v[:, g, t] = v[:, g, t] / sqrt(mean_d v^2 + eps)
 *   r[:, h, t] = clip(rel_clip[h, :] * r[:, h, t], rel_clip_value) * tau_t
 *   tau_t      = full ? 1 + alpha ln(max(1, (p_t + 1) / floor)) : 1
 *
 * q: BF16 [D, Hq, T], k and v: BF16 [D, Hkv, T], r: BF16 [Dr, Hq, T], all updated in place;
 * q_clip FP32 [Hq, D], rel_clip FP32 [Hq, Dr]; positions device I32 [T].
 */
struct Inkling2PrepareParams {
    float qk_clip = 0, rel_clip = 0, eps = 0, alpha = 0, floor = 1;
    bool full = false;
};
void inkling2_attention_prepare(Tensor& q, Tensor& k, Tensor& v, Tensor& r, const Tensor& q_clip,
                                const Tensor& rel_clip, const Tensor& positions,
                                const Inkling2PrepareParams& params, cudaStream_t stream);

/**
 * Op: causal attention with a learned relative-position bias and a per-query sink.
 *
 *   s_ij   = (q_i . k_j + r_i . bias[min(p_i - p_j, extent), :]) / sqrt(D)   (bias row extent = 0)
 *   o_i    = sum_j exp(s_ij) v_j / (exp(sink_i / sqrt(D)) + sum_j exp(s_ij))
 *
 * (llama.cpp adds the bias and the negated sink to q.k and scales the sum, beside a zero sink:
 * build_attn_mha's soft_max_ext with kq_b and soft_max_add_sinks.)
 *
 * over the keys j with 0 <= p_i - p_j (and p_i - p_j < window on a sliding layer), for every head
 * h with its key head h / (Hq / Hkv). q: BF16 [D, Hq, T] (prepared), r: BF16 [Dr, Hq, T],
 * sink: BF16 [Hq, T], bias: FP32 [extent, Dr]; the cache planes k, v are BF16
 * [D, capacity, Hkv] with position p in slot p % capacity (capacity = max context on a full
 * layer, >= window + T - 1 on a sliding one) and already hold this call's keys.
 * positions: device I32 [T]. out: BF16 [D, Hq, T]. One sequence.
 */
struct Inkling2AttentionParams {
    std::int32_t extent   = 0;
    std::int32_t window   = 0; // 0: a full layer
    std::int32_t capacity = 0;
    std::int32_t max_keys = 0; // bounds p + 1 over the calls made under this launch
};
void inkling2_attention(const Tensor& q, const Tensor& r, const Tensor& sink, const Tensor& bias,
                        const Tensor& k_cache, const Tensor& v_cache, const Tensor& positions,
                        const Inkling2AttentionParams& params, Tensor& out, cudaStream_t stream);

/**
 * Op: append this call's K and V rows to a visit's cache planes:
 *   k_cache[:, p_t % capacity, g] = k[:, g, t], likewise v. k, v: BF16 [D, Hkv, T].
 */
void inkling2_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions, std::int32_t capacity,
                        Tensor& k_cache, Tensor& v_cache, cudaStream_t stream);

} // namespace ninferno::ops
