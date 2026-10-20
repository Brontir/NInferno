#pragma once

// ninferno::ops - elementwise Ops of the dense decoder family (src/models/dense): the attention
// input preparation of a Llama-lineage layer from a rotary frequency table, and the residual update
// that emits the next reader's normalized input. Every per-model choice (biases, the q/k norm and
// its scope, the rotary width and frequencies, NoPE, a position-dependent query temperature, a
// post-norm, a residual multiplier) is an argument, never a model name.

#include "core/tensor.h"
#include "ninferno/ops/gemma4_attention.h"

#include <cuda_runtime.h>

#include <optional>

#include <cstdint>

namespace ninferno::ops {

/// Scope of the RMSNorm a layer applies to its queries and keys before the rotation.
enum class QkNorm : std::uint8_t {
    None,       ///< Llama, Mistral, Granite, Seed-OSS
    Head,       ///< per head over D features, one [D] weight (Qwen3, Gemma)
    Projection, ///< over the whole projection of a token, one [heads*D] weight (OLMo 2/3)
};

/**
 * Arguments of dense_qkv_prepare. Tensors a configuration does not use stay empty (data null).
 *
 *   q_src, k_src, v_src   BF16 [Hq*D, T], [Hkv*D, T], [Hkv*D, T]; features contiguous, the token
 *                         stride nb[1] a multiple of 16 bytes (typically row slices of one packed
 *                         projection output)
 *   q_bias, k_bias, v_bias  optional contiguous BF16 [Hq*D] / [Hkv*D], added before anything else
 *   qk_norm, q_norm, k_norm the norm scope and its weights, contiguous BF16 ([D] for Head,
 *                         [Hq*D] / [Hkv*D] for Projection); eps > 0 when a norm is applied
 *   inv_freq              contiguous device FP32 [rotary_dim/2]: the rotary frequency of pair i;
 *                         empty for a layer without rotary position embedding (NoPE)
 *   rotary_dim            even, 0 < rotary_dim <= D with inv_freq: the rotated features
 *   rotary_scale          multiplies cos and sin (YaRN's attention factor; 1 for plain RoPE)
 *   query_beta, query_period  query temperature by position: with query_period > 0 every query
 *                         is multiplied by 1 + query_beta * ln(1 + floor(p / query_period))
 *                         (Llama 4 / Mistral 3); query_period 0 disables it
 *   positions             contiguous device I32 [T], nonnegative
 *   q_out                 contiguous BF16 [D, Hq, T]; empty (with q_src) for a call that
 *                         prepares keys and values only
 *   cache                 the layer's K/V planes (Gemma4KvLayer layout, BF16 or FP8 with row
 *                         scales); token t's rows land where gemma4_kv_append would put them:
 *                         slot positions[t] % capacity of slot storage, or, for a cache bound to
 *                         B batch rows (cache.rows), the T columns are B runs of T/B tokens and
 *                         run b's rows go to its binding (its ring image, or its block-table row
 *                         of a page pool)
 *   k_out, v_out          instead of a cache (its planes empty): contiguous BF16 [D, Hkv, T],
 *                         token t's key and value rows in column t (a DFlash drafter's block
 *                         keys and values, or its committed context before the prefix append)
 */
struct DenseQkvPrepare {
    Tensor q_src, k_src, v_src;
    // Launch as a programmatic dependent of the kernel before on the stream, which must be the
    // last projection of q/k/v and have released this one (ops::linear_release_next returned
    // true): the sources are read after the wait. The bytes are the same either way.
    bool join_previous = false;
    Tensor q_bias, k_bias, v_bias;
    QkNorm qk_norm = QkNorm::None;
    Tensor q_norm, k_norm;
    bool norm_unit_offset = false; // gains 1 + weight (Gemma's RMSNorm)
    float eps = 0;
    Tensor inv_freq;
    std::int32_t rotary_dim   = 0;
    float rotary_scale        = 1;
    float query_beta          = 0;
    std::int32_t query_period = 0;
    Tensor positions;
    Tensor q_out;
    Gemma4KvLayer cache;
    Tensor k_out, v_out;
};

/**
 * Op: dense_qkv_prepare
 *
 * Math / indexing, for token t at position p = positions[t], query head h, key/value head g,
 * D = head_dim, R = rotary_dim, and x_hat_S(x) = x / sqrt(mean_S(x^2) + eps) over the set S:
 *
 *   q0 = q_src[:,h,t] + q_bias[h]    k0 = k_src[:,g,t] + k_bias[g]    v0 = v_src[:,g,t] + v_bias[g]
 *   Head:       q1 = x_hat_D(q0) * g(q_norm),              k1 = x_hat_D(k0) * g(k_norm)
 *   Projection: q1 = x_hat_{Hq*D}(q0 over every head)[h] * g(q_norm)[h], k1 alike over Hkv*D
 *               with the gain g(w) = norm_unit_offset ? 1 + w : w
 *   None:       q1 = q0,                                   k1 = k0
 *   rotation of pair i < R/2 (partner i + R/2), phi = p * inv_freq[i], c = rotary_scale*cos(phi),
 *   s = rotary_scale*sin(phi):  y[i] = x[i] c - x[i+R/2] s,  y[i+R/2] = x[i+R/2] c + x[i] s;
 *   features >= R unchanged (partial rotary); no rotation without inv_freq (NoPE);
 *   q_out[:,h,t] = R_t(q1) * (1 + beta * ln(1 + floor(p / period)))   (factor 1 when period = 0)
 *   cache.k[:, p % C, g] = R_t(k1),  cache.v[:, p % C, g] = v0
 *   (without a cache: k_out[:, g, t] = R_t(k1),  v_out[:, g, t] = v0).
 *
 * Supported domain: D in {64, 128, 256}; Hq a positive multiple of Hkv, or Hq = 0 (no queries);
 * 1 <= T/B <= cache capacity.
 *
 * Numeric: the oracle evaluates the formula in FP64 from the represented inputs, angle included.
 * The kernel forms the angle in FP64 and everything else in FP32, rounding each output once; V is
 * a bit-exact copy when it has no bias. FP8 planes store the Gemma4KvLayer row codec of R_t(k1)
 * and v0.
 *
 * Effects: writes q_out and exactly the T addressed rows of each head in both cache planes
 * (each binding of a batched call distinct, so no two batch rows write the same bytes), or
 * k_out and v_out whole. No workspace. Stream-ordered and capture-safe; positions are read on the
 * device only.
 */
void dense_qkv_prepare(const DenseQkvPrepare& args, cudaStream_t stream);

/// The arguments of residual_norm. A tensor without data is absent.
struct ResidualNorm {
    Tensor x; // updated in place
    Tensor branch;
    Tensor bias;        // optional
    Tensor post_weight; // optional: a post-norm of the branch (OLMo, sandwich norms)
    float scale = 1;    // a residual multiplier
    float eps   = 0;
    std::optional<float> post_eps; // the post-norm's epsilon when it differs from eps
    bool unit_offset = false;      // gains 1 + weight (Gemma's RMSNorm)
    Tensor out_weight, out;        // optional: the normalized input of the next reader
    // The output norm's gain convention when it differs from the post-norm's (a final norm
    // without the unit offset after unit-offset layer norms).
    std::optional<bool> out_unit_offset;
    // Launch as a programmatic dependent of the kernel before on the stream (the branch's
    // projection, launched with ops::linear_release_next): x and the branch are read after the
    // wait. The bytes are the same either way.
    bool join_previous = false;
};

/**
 * Op: residual_norm - a residual update and the normalized input of whatever reads it next.
 *
 * Per column t of x [H,T]:
 *   b        = branch[:,t] + bias                               (bias optional, BF16 [H])
 *   update   = post_weight ? x_hat(b) * g(post_weight) : b      (x_hat with post_eps)
 *   x'[:,t]  = BF16( x[:,t] + scale * update )
 *   out[:,t] = BF16( x_hat(x'[:,t]) * g(out_weight) )           (only when out is given)
 * with the gain g(w) = unit_offset ? 1 + w : w.
 *
 * The output norm reads the stored BF16 x', as a separate RMSNorm would. x, branch and out are
 * contiguous BF16 [H,T], pairwise non-overlapping; weights contiguous BF16 [H]; H <= 8192. FP64
 * oracle from represented inputs, rounding x' and out once each. No workspace.
 */
void residual_norm(const ResidualNorm& args, cudaStream_t stream);

} // namespace ninferno::ops
