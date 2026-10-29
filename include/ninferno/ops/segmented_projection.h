#pragma once

// ninferno::ops - projections of any geometry on the generic path.
//
// The fused input projections (attn_input_proj, gdn_input_proj and its convolution forms,
// gdn_norm_gating_proj) are registered for the shipped geometries only. A model of another size
// prepares its projection as SegmentedProjectionWeights (weight_input.h) and calls these Ops,
// which compose the generic projection kernels (every weight format, any [N,K], any T) with
// runtime-geometry convolution and gating kernels. Numerics follow the fused Ops' contracts:
// A16 routes accumulate in FP32 and round each output once; AllowA4 NVFP4 slices quantize the
// activation to NVFP4 from 17 columns on; a token's bytes do not depend on T (T >= 17) under the
// prefill route profile.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninferno/ops/weight_input.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninferno::ops {

// Rows [row_offset, row_offset + rows) of a contiguous BF16 tensor whose first dimension is its
// token stride ([C, T], or [C, W, B] read as [C, W*B]).
struct ProjectionOutput {
    Tensor* tensor          = nullptr;
    std::int32_t row_offset = 0;
    std::int32_t rows       = 0;
};

// Transient capacity for every T in [min_tokens, max_tokens] (the NVFP4 W4A4 activation plane of
// the widest slice under the active profile; zero for A16).
[[nodiscard]] std::size_t segmented_projection_workspace_capacity_bytes(
    const SegmentedProjectionWeights& weights, std::int32_t min_tokens, std::int32_t max_tokens);

// Writes the logical rows of the projection, in order, across `outputs` (their rows must sum to
// the projection's). x is BF16 [K, T] (a [K, W, B] block is read as T = W*B).
void segmented_projection(const Tensor& x, const SegmentedProjectionWeights& weights,
                          std::span<const ProjectionOutput> outputs, WorkspaceArena& workspace,
                          cudaStream_t stream);

// The SwiGLU (GeGLU: gelu) of a projection whose first half of rows is the gate and the second the
// up projection (a GGUF layer's gate and up of two tensors): out = act(gate . x) * (up . x), out
// BF16 [rows / 2, T]. Where each half is one weight's consecutive rows and both take ggml's vector
// kernel at this T, x is quantized once and the activation is the up product's epilogue
// (generic::ggml_vec_gated_pair); otherwise the projection and a separate activation kernel.
[[nodiscard]] std::size_t
segmented_swiglu_workspace_capacity_bytes(const SegmentedProjectionWeights& weights,
                                          std::int32_t min_tokens, std::int32_t max_tokens);
void segmented_swiglu(const Tensor& x, const SegmentedProjectionWeights& weights, bool gelu,
                      Tensor& out, WorkspaceArena& workspace, cudaStream_t stream);

// The vector kernel's gated pair of segmented_swiglu alone, for a caller that keeps its own
// separate path (a model whose projections go through the linear registry): true where it ran,
// false leaving out untouched. Its workspace over a token interval, 0 where it never runs.
[[nodiscard]] std::size_t
segmented_swiglu_pair_workspace_capacity_bytes(const SegmentedProjectionWeights& weights,
                                               std::int32_t min_tokens, std::int32_t max_tokens);
[[nodiscard]] bool segmented_swiglu_pair(const Tensor& x, const SegmentedProjectionWeights& weights,
                                         bool gelu, Tensor& out, WorkspaceArena& workspace,
                                         cudaStream_t stream);

// Projections of one x side by side: at decode and verify widths (T <= kSideBySideMaxTokens) a
// layer's q/k/v (and its attention output gate) are thin enough that each kernel alone leaves SMs
// idle; as one segmented projection each launch after the first runs alongside the one before it
// (generic Call::join_previous). Only where every weight admits it
// (detail::side_by_side_admits): every output then holds the bytes ops::linear gives that weight
// at this T. Returns false, with nothing launched, when they must run apart; the workspace comes
// from `work` and is released before the call returns.
struct SideBySideProjection {
    const Weight* weight = nullptr;
    LinearPolicy policy  = LinearPolicy::A16Only;
    Tensor* output       = nullptr; // BF16 [weight->n, T]
};

inline constexpr std::int32_t kSideBySideMaxTokens     = 16;
inline constexpr std::size_t kSideBySideMaxProjections = 4; // q, k, v and the gate
[[nodiscard]] bool project_side_by_side(const Tensor& x,
                                        std::span<const SideBySideProjection> projections,
                                        DeviceArena& work, cudaStream_t stream);

// gdn_input_proj_conv_snapshot / _record on a segmented q/k/v/z projection. Shapes and effects
// match the fused Ops (gdn_input_proj.h); query/key/value rows come from the outputs' extents.
[[nodiscard]] std::size_t segmented_gdn_conv_workspace_capacity_bytes(
    const SegmentedProjectionWeights& weights, std::int32_t channels, std::int32_t batch_size,
    std::int32_t min_width, std::int32_t max_width);

void segmented_gdn_input_proj_conv_snapshot(
    const Tensor& x, const SegmentedProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_state_slots,
    const Tensor& snapshot_base_slots, Tensor& query, Tensor& key, Tensor& value, Tensor& z,
    WorkspaceArena& workspace, cudaStream_t stream);

void segmented_gdn_input_proj_conv_record(const Tensor& x,
                                          const SegmentedProjectionWeights& weights,
                                          const Tensor& conv_weight, const Tensor& conv_states,
                                          const Tensor& valid_columns,
                                          const Tensor& initial_state_slots, Tensor& conv_record,
                                          Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                          WorkspaceArena& workspace, cudaStream_t stream);

// gdn_norm_gating_proj on a segmented a/b projection of any head count:
//   h = x * rsqrt(mean(x^2) + eps) * (1 + norm_weight)   (BF16 output)
//   a, b = projections of h;  g = -exp(A_log) * softplus(a + dt_bias);  beta = sigmoid(b)
// Composed (the norm, the generic projection, the gates), a and b are rounded to BF16 before the
// gate, as in gdn_gating. A call of at most kSegmentedFusedControlMaxTokens columns whose a and b
// are each one BF16 row block runs instead as one kernel, the 27B's fused kernel
// (gdn_norm_gating_proj) at one CTA width chosen from the hidden size alone: a and b come from the
// unrounded normalized input in FP32, and a token's bytes do not depend on the other columns of
// the call. Never under the invariant route profile.
inline constexpr std::int32_t kSegmentedFusedControlMaxTokens = 0;
[[nodiscard]] std::size_t segmented_gdn_norm_gating_workspace_capacity_bytes(
    const SegmentedProjectionWeights& weights, std::int32_t min_tokens, std::int32_t max_tokens);

void segmented_gdn_norm_gating_proj(const Tensor& x, const Tensor& norm_weight, float eps,
                                    const SegmentedProjectionWeights& weights, const Tensor& A_log,
                                    const Tensor& dt_bias, WorkspaceArena& workspace, Tensor& h,
                                    Tensor& g, Tensor& beta, cudaStream_t stream);

} // namespace ninferno::ops
