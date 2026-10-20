// GDN gate preparation for any head count: the kernel of gdn_gating (kernel/gdn_gating.cuh)
// with the head count and the interleaved [a; b] plane read at run time.
#include "core/device.h"
#include "core/tensor.h"
#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

namespace ninferno::ops::detail::generic {
namespace {

__global__ void gdn_gating_runtime_kernel(const __nv_bfloat16* __restrict__ ab,
                                          const float* __restrict__ A_log,
                                          const float* __restrict__ dt_bias, float* __restrict__ g,
                                          float* __restrict__ beta, std::int32_t heads,
                                          std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const std::int64_t t = i / heads;
        const int h          = static_cast<int>(i - t * heads);
        const float av       = __bfloat162float(ab[t * 2 * heads + h]);
        const float bv       = __bfloat162float(ab[t * 2 * heads + heads + h]);
        const float sp       = softplus(av + dt_bias[h]);
        g[i]                 = -expf(A_log[h]) * sp;
        beta[i]              = sigmoid(bv);
    }
}

} // namespace

void launch_gdn_gating_runtime(const Tensor& ab, const Tensor& A_log, const Tensor& dt_bias,
                               Tensor& g, Tensor& beta, cudaStream_t stream) {
    const std::int32_t heads = g.ne[0];
    const std::int64_t n     = static_cast<std::int64_t>(heads) * g.ne[1];
    if (n == 0) return;
    constexpr int kThreads = 256;
    const int blocks       = static_cast<int>(std::min<std::int64_t>((n + kThreads - 1) / kThreads, 4096));
    gdn_gating_runtime_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(ab.data), static_cast<const float*>(A_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data), heads, n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninferno::ops::detail::generic
