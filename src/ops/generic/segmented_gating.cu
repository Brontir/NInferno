// GDN gate preparation for any head count: the kernel of gdn_gating (kernel/gdn_gating.cuh)
// with the head count and the interleaved [a; b] plane read at run time; and the fused norm and
// BF16 control projection of the 27B (bf16_gdn_norm_gating_simt.cuh) at a run-time geometry.
#include "core/device.h"
#include "core/device_class.h"
#include "core/tensor.h"
#include "ops/common/math.cuh"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_norm_gating_simt.cuh"

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

int gdn_norm_gating_simt_threads(std::int32_t d) noexcept {
    // One 16-byte load of each weight row per thread covers the row: the call is a single DRAM
    // round trip per CTA, wider CTAs only idle.
    for (const int threads : {128, 256}) {
        if (threads * 8 >= d) return threads;
    }
    return 512;
}

void launch_gdn_norm_gating_simt(const Tensor& x, const Tensor& norm_weight, float eps,
                                 const void* a_rows, const void* b_rows, const Tensor& A_log,
                                 const Tensor& dt_bias, Tensor& h, Tensor& g, Tensor& beta,
                                 cudaStream_t stream) {
    const std::int32_t d      = x.ne[0];
    const std::int32_t heads  = g.ne[0];
    const std::int32_t tokens = x.ne[1];
    const int threads         = gdn_norm_gating_simt_threads(d);
    // The token tile changes no byte: the fewest tokens per CTA whose grid the device holds at
    // once (its SMs times the CTAs of this size one SM keeps resident).
    const DeviceClass& device = device_class();
    const std::int64_t per_sm =
        std::max(1, std::min(device.max_blocks_per_sm, device.max_threads_per_sm / threads));
    const std::int64_t resident = device.wave(per_sm);
    int tile                    = 1;
    while (tile < 16 &&
           static_cast<std::int64_t>(heads) * ((tokens + tile - 1) / tile) > resident) {
        tile *= 2;
    }
    const auto launch = [&]<int Tile, int Threads>() {
        gdn_norm_gating_simt_kernel<Tile, Threads>
            <<<dim3(static_cast<unsigned>(heads),
                    static_cast<unsigned>((tokens + Tile - 1) / Tile)),
               Threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const __nv_bfloat16*>(norm_weight.data),
                static_cast<const __nv_bfloat16*>(a_rows),
                static_cast<const __nv_bfloat16*>(b_rows), static_cast<const float*>(A_log.data),
                static_cast<const float*>(dt_bias.data), static_cast<__nv_bfloat16*>(h.data),
                static_cast<float*>(g.data), static_cast<float*>(beta.data), tokens, eps, d, heads);
    };
    const auto with_threads = [&]<int Threads>() {
        switch (tile) {
        case 1:
            launch.template operator()<1, Threads>();
            break;
        case 2:
            launch.template operator()<2, Threads>();
            break;
        case 4:
            launch.template operator()<4, Threads>();
            break;
        case 8:
            launch.template operator()<8, Threads>();
            break;
        default:
            launch.template operator()<16, Threads>();
            break;
        }
    };
    switch (threads) {
    case 128:
        with_threads.template operator()<128>();
        break;
    case 256:
        with_threads.template operator()<256>();
        break;
    default:
        with_threads.template operator()<512>();
        break;
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninferno::ops::detail::generic
