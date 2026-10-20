// The decomposed routes of the dynamic grouped convolution Ops for projection weights in any
// format Linear computes: RMSNorm and Linear produce BF16 intermediates (n, d or z), and these
// element-wise kernels apply the convolution exactly as the Op contracts state it.
#include "ops/dynamic_grouped_conv/generic/generic_dynamic_grouped_conv.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninferno::ops::detail {
namespace {

constexpr int kHidden = 5120, kGroups = 320, kGroupWidth = 16;

__device__ __forceinline__ float f(__nv_bfloat16 v) { return __bfloat162float(v); }

// base_kernel [channel, tap, side]; d rows (side * 2 + tap) * G + g, one column per token.
__global__ void prepare_kernel(const __nv_bfloat16* __restrict__ n,
                               const __nv_bfloat16* __restrict__ d,
                               const __nv_bfloat16* __restrict__ base,
                               __nv_bfloat16* __restrict__ prepared,
                               __nv_bfloat16* __restrict__ finish_delta, int width, int columns) {
    const std::int64_t total = static_cast<std::int64_t>(kHidden) * columns;
    const std::int64_t delta = static_cast<std::int64_t>(kGroups) * 2 * columns;
    const std::int64_t index = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (index < total) {
        const int h      = static_cast<int>(index % kHidden);
        const int column = static_cast<int>(index / kHidden);
        const int i      = column % width;
        const int g      = h / kGroupWidth;
        const auto* dc   = d + static_cast<std::int64_t>(column) * (4 * kGroups);
        float value = (f(base[h]) + f(dc[g])) * f(n[index]);
        if (i > 0) value += (f(base[kHidden + h]) + f(dc[kGroups + g])) * f(n[index - kHidden]);
        prepared[index] = __float2bfloat16_rn(value);
    }
    if (index < delta) {
        const int g      = static_cast<int>(index % kGroups);
        const int t      = static_cast<int>((index / kGroups) % 2);
        const int column = static_cast<int>(index / (2 * kGroups));
        finish_delta[index] =
            d[static_cast<std::int64_t>(column) * (4 * kGroups) + (2 + t) * kGroups + g];
    }
}

__global__ void add_kernel(const __nv_bfloat16* __restrict__ z,
                           const __nv_bfloat16* __restrict__ base,
                           const __nv_bfloat16* __restrict__ finish_delta,
                           __nv_bfloat16* __restrict__ residual, int width, int columns) {
    const std::int64_t total = static_cast<std::int64_t>(kHidden) * columns;
    const std::int64_t index = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (index >= total) return;
    const int h      = static_cast<int>(index % kHidden);
    const int column = static_cast<int>(index / kHidden);
    const int i      = column % width;
    const int g      = h / kGroupWidth;
    const auto* fd   = finish_delta + static_cast<std::int64_t>(column) * (2 * kGroups);
    float value      = (f(base[2 * kHidden + h]) + f(fd[g])) * f(z[index]);
    if (i > 0) value += (f(base[3 * kHidden + h]) + f(fd[kGroups + g])) * f(z[index - kHidden]);
    residual[index] = __float2bfloat16_rn(f(residual[index]) + value);
}

} // namespace

void generic_dynamic_grouped_conv_prepare_launch(const Tensor& n, const Tensor& d,
                                                 const Tensor& base_kernel, Tensor& prepared,
                                                 Tensor& finish_delta, int width, int columns,
                                                 cudaStream_t stream) {
    const std::int64_t total = static_cast<std::int64_t>(kHidden) * columns;
    const int blocks         = static_cast<int>((total + 255) / 256);
    prepare_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(n.data), static_cast<const __nv_bfloat16*>(d.data),
        static_cast<const __nv_bfloat16*>(base_kernel.data),
        static_cast<__nv_bfloat16*>(prepared.data), static_cast<__nv_bfloat16*>(finish_delta.data),
        width, columns);
    CUDA_CHECK(cudaGetLastError());
}

void generic_linear_dynamic_grouped_conv_add_launch(const Tensor& z, const Tensor& base_kernel,
                                                    const Tensor& finish_delta, Tensor& residual,
                                                    int width, int columns, cudaStream_t stream) {
    const std::int64_t total = static_cast<std::int64_t>(kHidden) * columns;
    const int blocks         = static_cast<int>((total + 255) / 256);
    add_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(z.data),
        static_cast<const __nv_bfloat16*>(base_kernel.data),
        static_cast<const __nv_bfloat16*>(finish_delta.data),
        static_cast<__nv_bfloat16*>(residual.data), width, columns);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninferno::ops::detail
