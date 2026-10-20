#pragma once

#include "core/tensor.h"
#include "ninferno/ops/gemma4_attention.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <initializer_list>
#include <span>

namespace ninferno::ops {

/**
 * Prepares one Gemma 4 attention call's Q/K/V from the raw projections, in one launch.
 *
 * For every token t, query head h and key/value head g, with D = head_dim, x_hat(x) =
 * x / sqrt(mean_d(x^2) + eps) and the pair partner of dimension i < D/2 being i + D/2:
 *
 *   qn[:,h,t]  = x_hat(q_src[:,h,t]) * q_weight
 *   kn[:,g,t]  = x_hat(k_src[:,g,t]) * k_weight
 *   q_out      = R_t(qn),  k_out = R_t(kn)
 *   v_out[:,g,t] = x_hat(v_src[:,g,t])                     (no learned scale, no rotation)
 *
 * R_t rotates pair (i, i+D/2) by phi = positions[t] * theta^(-2i/D) for i < rotary_pairs and
 * leaves every other dimension unchanged:
 *
 *   R[i]     = x[i] cos(phi) - x[i+D/2] sin(phi)
 *   R[i+D/2] = x[i+D/2] cos(phi) + x[i] sin(phi).
 *
 * rotary_pairs = D/2 is Gemma's `default` RoPE over the whole head (sliding layers);
 * rotary_pairs < D/2 is its `proportional` RoPE (full layers), whose frequency ladder is still
 * taken over D. There is no attention-scale factor here: Gemma folds it into q_weight.
 *
 * Layout: each source is a 2-D BF16 tensor [heads*D, T] whose features are contiguous and whose
 * token stride nb[1] may be padded (it is typically a row slice of one packed projection output).
 * v_src may alias k_src (Gemma's attention_k_eq_v) — both are read-only. q_weight/k_weight are
 * contiguous BF16 [D]. positions is contiguous device I32 [T]. Outputs are contiguous BF16
 * [D, heads, T] and must not overlap any input. D is 256 or 512; q_heads and kv_heads are
 * positive; theta positive and finite; 0 < rotary_pairs <= D/2.
 *
 * The oracle evaluates the formula in FP64 from the represented inputs, including the angle; the
 * BF16 outputs are compared directly with it. The kernel rounds once, at the output. No workspace.
 */
void gemma4_qkv_prepare(const Tensor& q_src, const Tensor& k_src, const Tensor& v_src,
                        std::int32_t head_dim, const Tensor& q_weight, const Tensor& k_weight,
                        float eps, const Tensor& positions, float theta, std::int32_t rotary_pairs,
                        Tensor& q_out, Tensor& k_out, Tensor& v_out, cudaStream_t stream);

/**
 * The same preparation with the keys and values written straight into one layer's cache: the
 * rows of token t land in slot positions[t] % capacity of `cache`, exactly what gemma4_kv_append
 * would store from k_out/v_out, which this form never materializes. The planes must hold T rows;
 * every other cache byte is unchanged. q_out as above.
 *
 * join_previous: the kernel before on the stream is the last projection of q/k/v and its launch
 * released this one (ops::linear_release_next returned true). On a device with programmatic
 * dependent launch the angle table is then built under the projection's tail and the sources are
 * read after the wait. The bytes are the same either way.
 */
void gemma4_qkv_prepare(const Tensor& q_src, const Tensor& k_src, const Tensor& v_src,
                        std::int32_t head_dim, const Tensor& q_weight, const Tensor& k_weight,
                        float eps, const Tensor& positions, float theta, std::int32_t rotary_pairs,
                        Tensor& q_out, const Gemma4KvLayer& cache, cudaStream_t stream,
                        bool join_previous = false);

/**
 * The query half of gemma4_qkv_prepare alone, for layers that own no keys or values (the MTP
 * drafter reads its target's cache): q_out = R_t(x_hat(q_src) * q_weight), same layout, domain,
 * oracle and rounding as above.
 */
void gemma4_q_prepare(const Tensor& q_src, std::int32_t head_dim, const Tensor& q_weight, float eps,
                      const Tensor& positions, float theta, std::int32_t rotary_pairs,
                      Tensor& q_out, cudaStream_t stream);

/**
 * Q/K/V preparation for the Qwen3-style layers of a Gemma 4 DFlash drafter (D = 128):
 *
 *   q_out = R_t(x_hat(q_src) * q_weight),  k_out = R_t(x_hat(k_src) * k_weight),  v_out = v_src
 *
 * with R_t rotating every pair (rotary_pairs = D/2 = 64) at theta as defined above, and V copied
 * unchanged into the [D, heads, T] layout. The query side is optional: pass all three of q_src,
 * q_weight and q_out, or none (a context-only call that prepares committed tokens' keys and
 * values). Layout, oracle and rounding as gemma4_qkv_prepare; v_out is a bit-exact copy.
 */
void gemma4_dflash_qkv_prepare(const Tensor* q_src, const Tensor& k_src, const Tensor& v_src,
                               const Tensor* q_weight, const Tensor& k_weight, float eps,
                               const Tensor& positions, float theta, Tensor* q_out, Tensor& k_out,
                               Tensor& v_out, cudaStream_t stream);

/// gemma4_dflash_qkv_prepare with the keys and values written straight into `cache`, as above.
void gemma4_dflash_qkv_prepare(const Tensor* q_src, const Tensor& k_src, const Tensor& v_src,
                               const Tensor* q_weight, const Tensor& k_weight, float eps,
                               const Tensor& positions, float theta, Tensor* q_out,
                               const Gemma4KvLayer& cache, cudaStream_t stream);

/**
 * GeGLU over a fused gate/up projection:
 *
 *   out[i,t] = gelu_tanh(gate_up[i,t]) * gate_up[I+i,t],  I = out.ne[0],
 *   gelu_tanh(z) = 0.5 z (1 + tanh(sqrt(2/pi) (z + 0.044715 z^3))).
 *
 * gate_up is contiguous BF16 [2I, T]; out is contiguous BF16 [I, T] and does not overlap it.
 * FP64 oracle from represented inputs; one output rounding. No workspace.
 */
void gelu_tanh_mul(const Tensor& gate_up, Tensor& out, cudaStream_t stream);

/**
 * The same GeGLU over separate planes: out[i,t] = gelu_tanh(gate[i,t]) * up[i,t]. gate and up
 * are BF16 [I, T] with contiguous features and a token stride of whole 4-byte words (up is
 * typically one layer's rows of the per-layer inputs); out is contiguous BF16 [I, T] and
 * overlaps neither. I is even. Oracle and rounding as above.
 */
void gelu_tanh_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream);

/**
 * In-place scale of BF16 values: x[i] = bf16(x[i] * scale), the product formed exactly in FP32
 * for a scale that is itself a BF16 value (transformers' embedding normalizer applied to the
 * gathered rows of a table kept as published). x is contiguous BF16 with an even element count.
 * No workspace.
 */
void scale_bf16(Tensor& x, float scale, cudaStream_t stream);

/**
 * Gemma 4 per-layer embedding inputs (PLE) of T tokens over L layers, P = weight.ne[0]:
 *
 *   a     = BF16(projection[lP:(l+1)P, t] * projection_scale)
 *   n     = BF16(x_hat(a) * weight)
 *   e     = BF16(table_rows[lP:(l+1)P, t] * embedding_scale)
 *   out   = BF16(BF16(n + e) * input_scale)
 *
 * which is transformers' Gemma 4 `project_per_layer_inputs` with its BF16 roundings: the
 * projection of the embedded token scaled by H^-1/2 and RMS-normalized per layer slice, plus the
 * per-layer table row scaled by sqrt(P), both scaled by 2^-1/2. projection, table_rows and out
 * are contiguous BF16 [L*P, T]; out may be `projection` itself and overlaps nothing else. weight
 * is contiguous BF16 [P], P even. No workspace.
 */
void gemma4_ple_inputs(const Tensor& projection, const Tensor& table_rows, const Tensor& weight,
                       float eps, float projection_scale, float embedding_scale,
                       float input_scale, Tensor& out, cudaStream_t stream);

/// The branch a residual update adds: `a` alone, or the join of two normalized branches,
/// x_hat(a) * a_weight + x_hat(b) * b_weight (the dense MLP and the expert bank of the MoE model).
struct ResidualBranch {
    const Tensor& a;
    const Tensor* a_weight = nullptr; // a_weight, b and b_weight: all three or none
    const Tensor* b        = nullptr;
    const Tensor* b_weight = nullptr;
};

/// One normalized output of a residual update: out = x_hat(x') * weight.
struct ResidualNormOutput {
    const Tensor& weight;
    Tensor& out;
};

/**
 * A residual update and the normalized inputs of whatever reads the updated stream, in one launch.
 * Per column t of x [H,T]:
 *
 *   x'[:,t]  = BF16( (x[:,t] + x_hat(branch[:,t]) * post_weight) * scale )
 *   out_i    = x_hat(x'[:,t]) * weight_i                  for each of up to three outputs
 *
 * The outputs are normalized from the *stored* BF16 x', exactly as a separate RMSNorm reading x
 * would see it, so fusing changes no rounding the unfused schedule had. The two Gemma 4 uses:
 *
 *   after attention:    branch = attention output, post = post_attention_norm, scale 1,
 *                       outputs = pre_ffn_norm (and pre_moe_norm on the MoE model, and the
 *                       router's norm, router_norm, where the generic MoE Op runs the experts);
 *   after feed-forward: branch = dense MLP, joined by the experts (MoE) with the post_mlp and
 *                       post_moe weights, post = post_ffn_norm, scale = layer scalar,
 *                       output = the next layer's input_norm, or final_norm after the last layer.
 *
 * No outputs asks for the update of x alone: the per-layer embedding block reads the
 * un-normalized stream after the feed-forward.
 *
 * x, the branch tensors and the outputs: contiguous BF16 [H,T], pairwise non-overlapping (the
 * branch read-only); weights contiguous BF16 [H]. H <= 8192. FP64 oracle from represented
 * inputs, rounding x' once and each output once. No workspace.
 *
 * join_previous: launch as a programmatic dependent of the kernel before on the stream (on a
 * device with programmatic dependent launch): the weights load before griddepcontrol.wait, x and
 * the branch after it. It pays when that kernel releases early (a projection launched with
 * ops::linear_release_next). The bytes are the same either way.
 */
void gemma4_residual_norm(Tensor& x, const ResidualBranch& branch, const Tensor& post_weight,
                          float eps, float scale, std::span<const ResidualNormOutput> outputs,
                          cudaStream_t stream, bool join_previous = false);

inline void gemma4_residual_norm(Tensor& x, const ResidualBranch& branch, const Tensor& post_weight,
                                 float eps, float scale,
                                 std::initializer_list<ResidualNormOutput> outputs,
                                 cudaStream_t stream, bool join_previous = false) {
    gemma4_residual_norm(x, branch, post_weight, eps, scale,
                         std::span<const ResidualNormOutput>(outputs.begin(), outputs.size()),
                         stream, join_previous);
}

} // namespace ninferno::ops
