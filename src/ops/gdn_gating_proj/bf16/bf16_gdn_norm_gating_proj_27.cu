#include "core/weight.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_norm_gating_simt.cuh"
#include "core/device.h"

#include <cuda_bf16.h>

namespace ninferno::ops::detail {

void bf16_gdn_norm_gating_proj_27_launch(const Tensor& x, const Tensor& norm_weight, float eps,
                                         Tensor& h, const Weight& a_weight, const Weight& b_weight,
                                         const Tensor& alog, const Tensor& bias, Tensor& g,
                                         Tensor& beta, cudaStream_t stream) {
    const auto launch = [&]<int T, int Threads>() {
        gdn_norm_gating_simt_kernel<T, Threads, 5120, 48>
            <<<dim3(48, (x.ne[1] + T - 1) / T), Threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const __nv_bfloat16*>(norm_weight.data),
                static_cast<const __nv_bfloat16*>(a_weight.qdata),
                static_cast<const __nv_bfloat16*>(b_weight.qdata),
                static_cast<const float*>(alog.data), static_cast<const float*>(bias.data),
                static_cast<__nv_bfloat16*>(h.data), static_cast<float*>(g.data),
                static_cast<float*>(beta.data), x.ne[1], eps, 5120, 48);
    };
    const int tokens = x.ne[1];
    if (tokens <= 2)
        launch.template operator()<1, 512>();
    else if (tokens <= 14)
        launch.template operator()<2, 512>();
    else if (tokens <= 28)
        launch.template operator()<2, 256>();
    else
        launch.template operator()<4, 256>();
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninferno::ops::detail
