#include "ops/launcher/scatter.h"

#include "ops/kernel/scatter.cuh"
#include "ops/common/arch_limits.h"
#include "core/device.h"

#include <cstdint>
#include <algorithm>

namespace ninferno::ops::detail {

void scatter_launch(const Tensor& src, const Tensor& indices, Tensor& dst, cudaStream_t stream) {
    constexpr int block        = 256;
    constexpr int vector_block = 128;
    const int d                = src.ne[0];
    const int vision           = src.ne[1];
    const auto src_addr        = reinterpret_cast<std::uintptr_t>(src.data);
    const auto dst_addr        = reinterpret_cast<std::uintptr_t>(dst.data);
    if ((d % 8) == 0 && ((src_addr | dst_addr) & 0xfu) == 0) {
        scatter_bf16x8_kernel<<<vision, vector_block, 0, stream>>>(
            static_cast<const uint4*>(src.data), static_cast<const std::int32_t*>(indices.data),
            static_cast<uint4*>(dst.data), d / 8);
    } else if ((d & 1) == 0 && ((src_addr | dst_addr) & 0x3u) == 0) {
        scatter_bf16x2_kernel<<<vision, block, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(src.data),
            static_cast<const std::int32_t*>(indices.data), static_cast<__nv_bfloat162*>(dst.data),
            d / 2);
    } else {
        scatter_scalar_kernel<<<vision, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(src.data),
            static_cast<const std::int32_t*>(indices.data), static_cast<__nv_bfloat16*>(dst.data),
            d);
    }
    CUDA_CHECK(cudaGetLastError());
}

void scatter_bf16_batch_launch(const Tensor& source, const Tensor& lanes,
                               const Tensor& valid_columns, Tensor& destination,
                               cudaStream_t stream) {
    constexpr int block = kScatterBatchThreads;
    // Keep a grid-stride feature loop beyond CUDA's z-dimension limit.
    const int feature_tiles = std::min((source.ne[0] / 8 + block - 1) / block, kMaxGridYZ);
    const dim3 grid(source.ne[1], source.ne[2], feature_tiles);
    scatter_bf16_batch_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint4*>(source.data), static_cast<const std::int32_t*>(lanes.data),
        static_cast<const std::int32_t*>(valid_columns.data), static_cast<uint4*>(destination.data),
        source.ne[0] / 8, source.ne[1],
        destination.nb[1] / static_cast<std::int64_t>(sizeof(uint4)),
        destination.nb[2] / static_cast<std::int64_t>(sizeof(uint4)));
    CUDA_CHECK(cudaGetLastError());
}

namespace {

// Destination head hs = j * groups + g of token t is source head g * per + j; one thread moves
// eight BF16 values.
__global__ void regroup_bf16_heads_kernel(const uint4* __restrict__ source,
                                          uint4* __restrict__ destination, std::int64_t total,
                                          int chunks, int heads, int groups) {
    const std::int64_t i =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (i >= total) { return; }
    const std::int64_t row = i / chunks;
    const int c            = static_cast<int>(i - row * chunks);
    const std::int64_t t   = row / heads;
    const int hs           = static_cast<int>(row - t * heads);
    const int j            = hs / groups;
    const int g            = hs - j * groups;
    const int from         = g * (heads / groups) + j;
    destination[i]         = source[(t * heads + from) * chunks + c];
}

} // namespace

void regroup_bf16_heads_launch(const Tensor& source, std::int32_t groups, Tensor& destination,
                               cudaStream_t stream) {
    constexpr int block      = 256;
    const int chunks         = source.ne[0] / 8;
    const int heads          = source.ne[1];
    const std::int64_t total = static_cast<std::int64_t>(chunks) * heads * source.ne[2] * source.ne[3];
    regroup_bf16_heads_kernel<<<static_cast<unsigned>((total + block - 1) / block), block, 0,
                                stream>>>(static_cast<const uint4*>(source.data),
                                          static_cast<uint4*>(destination.data), total, chunks,
                                          heads, groups);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninferno::ops::detail
