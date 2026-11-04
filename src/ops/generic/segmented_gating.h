#pragma once

// The GDN control's gate kernels (segmented_gating.cu), for segmented_gdn_norm_gating_proj.
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninferno::ops::detail::generic {

// One control projection's rows for the fused kernel: BF16 rows (scales null), or Q8_G32_FP16
// row-split codes with their FP16 scales, one per 32 values; the row stride is the call's K.
struct GdnControlRows {
    const void* rows   = nullptr;
    const void* scales = nullptr;
};

// The gates from the [a; b] plane of a composed control projection.
void launch_gdn_gating_runtime(const Tensor& ab, const Tensor& A_log, const Tensor& dt_bias,
                               Tensor& g, Tensor& beta, cudaStream_t stream);
// The norm, both control projections and the gates in one launch (a and b of one format).
void launch_gdn_norm_gating_simt(const Tensor& x, const Tensor& norm_weight, float eps,
                                 const GdnControlRows& a, const GdnControlRows& b,
                                 const Tensor& A_log, const Tensor& dt_bias, Tensor& h, Tensor& g,
                                 Tensor& beta, cudaStream_t stream);

} // namespace ninferno::ops::detail::generic
