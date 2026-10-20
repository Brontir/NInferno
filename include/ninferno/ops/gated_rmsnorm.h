#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninferno::ops {

/**
 * Applies RMS normalization over ne[0] and an elementwise SiLU gate. For each logical row r:
 *
 *   inv_r    = 1 / sqrt((1/D) * sum_d x[d,r]^2 + eps)
 *   ideal[d,r] = x[d,r] * inv_r * weight[d] * SiLU(z[d,r]).
 *
 * `x`, `z`, and `out` are same-shaped contiguous BF16 tensors, `weight` is contiguous BF16 [D],
 * and eps is positive and finite. This form does not apply a unit offset to weight. Inputs and
 * output must not overlap. The oracle evaluates `ideal` naively in FP64 from the represented
 * inputs. The BF16 output is promoted and compared directly with that result; output storage
 * rounding belongs to the Op's numerical criterion, not the oracle. Kernel reduction, staging,
 * and accumulator precision are implementation choices. There is no workspace or persistent state
 * side effect. `execution` supplies the stream and the selected device's positive physical SM
 * count for launch selection.
 */
void gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, float eps, Tensor& out,
                   DeviceExecutionView execution);

// gated_rmsnorm of x [D, H, ...] written in llama.cpp's tiled head order, the input order of a
// GGUF linear-attention out_proj: head g * (H / groups) + j of a token lands at j * groups + g.
// The same bytes as gated_rmsnorm followed by regroup_bf16_heads, in one kernel. Where
// gated_rmsnorm_regroups (D 64..256 by 64, 4-byte aligned tensors, out apart from x and z);
// groups must divide H.
[[nodiscard]] bool gated_rmsnorm_regroups(const Tensor& x, const Tensor& weight, const Tensor& z,
                                          const Tensor& out);
void gated_rmsnorm_regrouped(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                             std::int32_t groups, Tensor& out, DeviceExecutionView execution);

} // namespace ninferno::ops
