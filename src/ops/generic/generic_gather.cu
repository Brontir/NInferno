// Act-order input gather: out[j, t] = x[order[j], t] for a weight stored with its K values in
// another order (QuantLayout::RowGroupZpOrdered).
#include "ops/generic/generic_gemm_internal.h"
#include "ops/common/arch_limits.h"

#include <algorithm>
#include <cstdint>

namespace ninferno::ops::detail::generic {
namespace {

__global__ void gather_k_kernel(const std::uint16_t* __restrict__ x, int ldx,
                                const std::int32_t* __restrict__ order, int k, int tokens,
                                std::uint16_t* __restrict__ out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    const int column = __ldg(order + j);
    for (int t = blockIdx.y; t < tokens; t += gridDim.y) {
        out[static_cast<std::size_t>(t) * k + j] = x[static_cast<std::size_t>(t) * ldx + column];
    }
}

} // namespace

void launch_gather_k(const void* x, int ldx, const void* order, int k, int tokens, void* out,
                     cudaStream_t stream) {
    constexpr int kThreads = 256;
    const dim3 grid((k + kThreads - 1) / kThreads, std::min(tokens, kMaxGridYZ));
    gather_k_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const std::uint16_t*>(x), ldx,
                                                   static_cast<const std::int32_t*>(order), k,
                                                   tokens, static_cast<std::uint16_t*>(out));
}

} // namespace ninferno::ops::detail::generic
