// The grouped W4A8's activation quantizer (moe_w4a8.cuh): rows at a stride of K rounded up to
// whole 128-value tiles, the padding zero.
#include "core/device.h"
#include "ops/generic/generic_w4a8.cuh"
#include "ops/moe/moe_internal.h"
#include "ops/moe/moe_route.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninferno::ops::detail::moe {

namespace {

// One thread per (row, 32-value block of ld): quantize_x_task.
__global__ __launch_bounds__(256) void moe_w4a8_quantize_kernel(const __nv_bfloat16* __restrict__ x,
                                                                int ldx, int k, int ld, int rows,
                                                                std::int8_t* __restrict__ codes,
                                                                float* __restrict__ scales,
                                                                float* __restrict__ sums) {
    const int blocks = ld / 32;
    const std::int64_t task =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (task >= static_cast<std::int64_t>(rows) * blocks) return;
    quantize_x_task(x, ldx, k, ld, task, codes, scales, sums);
}

// One thread per (row, 32-value block of ld): E4M3 codes in value order and the block's scale.
__global__ __launch_bounds__(256) void moe_mx8_quantize_kernel(const __nv_bfloat16* __restrict__ x,
                                                               int ldx, int k, int ld, int rows,
                                                               std::int8_t* __restrict__ codes,
                                                               float* __restrict__ scales) {
    const int blocks = ld / 32;
    const std::int64_t task =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (task >= static_cast<std::int64_t>(rows) * blocks) return;
    const int row     = static_cast<int>(task / blocks);
    const int block   = static_cast<int>(task - static_cast<std::int64_t>(row) * blocks);
    unsigned words[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float scale       = 0.0F;
    if (block * 32 < k) {
        float v[32];
        generic::w4a8_load_block(x + static_cast<std::int64_t>(row) * ldx + block * 32, v);
        float amax = 0.0F;
#pragma unroll
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
        if (amax > 0.0F) {
            // 2^e >= amax / 448: the exponent of amax / 448, one up unless it is a power of two;
            // e in [1, 253] keeps 2^e and 2^-e normal.
            const unsigned bits = __float_as_uint(amax * (1.0F / 448.0F));
            int e               = static_cast<int>(bits >> 23) + ((bits & 0x7fffffU) != 0 ? 1 : 0);
            e                   = max(1, min(253, e));
            scale               = __uint_as_float(static_cast<unsigned>(e) << 23);
            const float inv     = __uint_as_float(static_cast<unsigned>(254 - e) << 23);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(
                    make_float2(v[4 * i] * inv, v[4 * i + 1] * inv), __NV_SATFINITE, __NV_E4M3);
                const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(
                    make_float2(v[4 * i + 2] * inv, v[4 * i + 3] * inv), __NV_SATFINITE, __NV_E4M3);
                words[i] = static_cast<unsigned>(lo) | (static_cast<unsigned>(hi) << 16);
            }
        }
    }
    auto* dst = reinterpret_cast<uint4*>(codes + static_cast<std::int64_t>(row) * ld + block * 32);
    dst[0]    = make_uint4(words[0], words[1], words[2], words[3]);
    dst[1]    = make_uint4(words[4], words[5], words[6], words[7]);
    scales[task] = scale;
}

} // namespace

void launch_moe_mx8_quantize(const void* x, int ldx, int k, int rows, const MoeW4a8Buffers& out,
                             cudaStream_t stream) {
    if (rows <= 0) return;
    const std::int64_t tasks = static_cast<std::int64_t>(rows) * (out.ld / 32);
    const auto grid          = static_cast<unsigned>((tasks + 255) / 256);
    moe_mx8_quantize_kernel<<<grid, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(x), ldx, k,
                                                      out.ld, rows, out.codes, out.scales);
    CUDA_CHECK(cudaGetLastError());
}

void launch_moe_w4a8_quantize(const void* x, int ldx, int k, int rows, const MoeW4a8Buffers& out,
                              cudaStream_t stream) {
    if (rows <= 0) return;
    const std::int64_t tasks = static_cast<std::int64_t>(rows) * (out.ld / 32);
    const auto grid          = static_cast<unsigned>((tasks + 255) / 256);
    moe_w4a8_quantize_kernel<<<grid, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(x), ldx, k,
                                                       out.ld, rows, out.codes, out.scales,
                                                       out.sums);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninferno::ops::detail::moe
