#pragma once

// The Inkling 2 family's own Ops (docs/maintainer/inkling2-model.md). Each states its mathematics
// per token column; the oracles in tests/ops/test_inkling2_*.cpp evaluate it in FP64 from the
// represented inputs. n(x) = x / sqrt(mean(x^2) + 1e-6) over a column.

#include "core/tensor.h"
#include "ninferno/ops/gemma4_attention.h"

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
 * x, out: contiguous BF16 [C, T, B] (out may alias x exactly): B sequences of T tokens each.
 * weight: contiguous FP32 [C, 4]. taps: empty, or contiguous BF16 [heads * 4, T, B] (a dynamic tap
 * per head and token; C % heads == 0). The state of batch row b is the FP32 [C, 3] plane at
 * state.data + slot_b * state.slot_stride, slot_b = state.slots[b] (device I32 [B]; null: b),
 * read and replaced; the rows' planes are distinct. T >= 1.
 */
struct Inkling2ConvState {
    float* data               = nullptr;
    std::int64_t slot_stride  = 0; // elements between two slots' planes
    const std::int32_t* slots = nullptr;
};

void inkling2_conv(const Tensor& x, const Tensor& weight, const Tensor& taps, std::int32_t heads,
                   const Inkling2ConvState& state, Tensor& out, cudaStream_t stream);

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
 *   s_ij   = q_i . k_j / sqrt(D) + r_i . bias[min(p_i - p_j, extent), :]   (bias row extent = 0)
 *   o_i    = sum_j exp(s_ij) v_j / (exp(sink_i) + sum_j exp(s_ij))
 *
 * (llama.cpp's build_attn_rel_pos folds the bias minus the sink into the mask, which the softmax
 * adds after the scale, beside a zero sink.)
 *
 * over the keys j with 0 <= p_i - p_j (and p_i - p_j < window on a sliding layer, cache.window),
 * for every head h with its key head h / (Hq / Hkv). q: BF16 [D, Hq, T, B] (prepared), r: BF16
 * [Dr, Hq, T, B], sink: BF16 [Hq, T, B], bias: FP32 [extent, Dr], positions: device I32 [T, B],
 * out: BF16 [D, Hq, T, B]. The keys and values are a Gemma4KvLayer's BF16 planes (one ring image
 * per batch row, or the paged pool through block tables; gemma4_attention.h), already holding this
 * call's rows (gemma4_kv_append): batch row b reads its binding cache.rows[b]. D <= 128, Dr <= 32.
 */
void inkling2_attention(const Tensor& q, const Tensor& r, const Tensor& sink, const Tensor& bias,
                        const Gemma4KvLayer& cache, const Tensor& positions, std::int32_t extent,
                        Tensor& out, cudaStream_t stream);

} // namespace ninferno::ops
