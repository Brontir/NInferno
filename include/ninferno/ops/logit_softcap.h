#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninferno::ops {

/**
 * Bounds logits through a scaled hyperbolic tangent, in place. For z = logits[i]:
 *
 *   ideal[i] = cap * tanh(z / cap).
 *
 * This is a saturating transform, not a normalization: it maps the whole real line onto
 * (-cap, cap) while leaving small logits almost unchanged, so the ordering of candidates is
 * preserved and the gap between the leaders is compressed. Applying it twice is not the same as
 * applying it once, and applying it after any temperature or penalty is not the same as applying
 * it before; the caller owns that order.
 *
 * `logits` is an arbitrary-rank contiguous BF16 tensor and `cap` is positive and finite. The
 * oracle evaluates `ideal` in FP64 from the represented input. The updated BF16 values are
 * promoted and compared directly with that result; output storage rounding belongs to the Op's
 * numerical criterion, not the oracle. Private kernel arithmetic is implementation-defined —
 * in particular the identity tanh(z/c) = 1 - 2/(exp(2z/c)+1) may be used, and saturation for
 * large |z| is required to reach exactly +/-cap in BF16 rather than to overflow. The Op mutates
 * only `logits` and uses no workspace or persistent state.
 */
//
// `scale` multiplies z before the cap (a logit multiplier folded into the transform):
// ideal[i] = cap * tanh(scale * z / cap); scale 1 is the plain cap. `scale` is positive and finite.
void logit_softcap(Tensor& logits, float cap, cudaStream_t stream, float scale = 1.0F);

} // namespace ninferno::ops
