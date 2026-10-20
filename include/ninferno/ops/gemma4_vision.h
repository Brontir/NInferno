#pragma once
// The two fused steps of Gemma 4's vision tower (models/gemma4/vision): the inputs of one
// attention call over an image's patches, and the image's soft tokens from its encoded patches.
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninferno::ops {

/**
 * Q/K/V of one vision attention call. With D = head_dim and x_hat(x) = x / sqrt(mean_d(x^2) + eps):
 *
 *   q_out[:,h,t] = q_scale * R_t(x_hat(q_src[:,h,t]) * q_weight)
 *   k_out[:,h,t] = R_t(x_hat(k_src[:,h,t]) * k_weight)
 *   v_out[:,h,t] = x_hat(v_src[:,h,t])                          (no learned scale, no rotation)
 *
 * R_t rotates pair (i, i+D/2), i < D/2, by phi = positions[t][a] * theta^(-4 j / D) with a = 0,
 * j = i for i < D/4 and a = 1, j = i - D/4 otherwise:
 *
 *   R[i]     = x[i] cos(phi) - x[i+D/2] sin(phi)
 *   R[i+D/2] = x[i+D/2] cos(phi) + x[i] sin(phi).
 *
 * That is the tower's axial 2-D rope (each axis rotates half of the head over D/4 frequencies)
 * with the head rows in the converter's order, which places each axis's rotated pairs at i and
 * i + D/2. q_scale folds an attention scale into the queries (1 for none).
 *
 * Layout: each source is BF16 [heads*D, T] with contiguous features and a token stride nb[1] that
 * may be padded (typically rows of one packed projection). positions is contiguous device I32
 * [2, T] read as positions[t][a] = element 2t + a (x then y). q_weight/k_weight are contiguous
 * BF16 [D]. Outputs are contiguous BF16 [D, heads, T] and overlap no input. D is a multiple of 4
 * in [4, 256]; heads and T are positive; eps and theta positive and finite; q_scale finite.
 *
 * The oracle evaluates the formula in FP64 from the represented inputs; the BF16 outputs are
 * compared directly with it. The kernel rounds once, at the output. No workspace.
 */
void gemma4_vision_qkv_prepare(const Tensor& q_src, const Tensor& k_src, const Tensor& v_src,
                               std::int32_t head_dim, const Tensor& positions,
                               const Tensor& q_weight, const Tensor& k_weight, float eps,
                               float theta, float q_scale, Tensor& q_out, Tensor& k_out,
                               Tensor& v_out, cudaStream_t stream);

/**
 * An image's soft tokens from its encoded patches: the k x k mean pool and sqrt(hidden) scale of
 * Gemma4VisionPooler, the tower's standardization, and the weightless RMS norm in front of the
 * text projection. For soft token n = ny * (W/k) + nx and feature d:
 *
 *   m[d,n]   = bf16((1/k^2) sum_{dy,dx < k} x[d, (k ny + dy) W + k nx + dx])
 *   s[d,n]   = bf16((m[d,n] * root - std_bias[d]) * std_scale[d])
 *   out[d,n] = s[d,n] / sqrt(mean_d(s[:,n]^2) + eps)
 *
 * The two inner roundings are the reference's own casts back to the working dtype. x is
 * contiguous BF16 [H, W * rows]: the patches in raster order (row by row, W per row); W and rows
 * are positive multiples of k. std_bias and std_scale are contiguous BF16 [H]. out is contiguous
 * BF16 [H, (W/k) (rows/k)] and does not overlap x. root and eps are positive and finite; k is
 * positive; H is at most 8192.
 *
 * The oracle evaluates the formula in FP64 from the represented inputs, with the two roundings;
 * the output is compared directly with it. No workspace.
 */
void gemma4_vision_pool(const Tensor& x, std::int32_t width, std::int32_t kernel,
                        const Tensor& std_bias, const Tensor& std_scale, float root, float eps,
                        Tensor& out, cudaStream_t stream);

} // namespace ninferno::ops
