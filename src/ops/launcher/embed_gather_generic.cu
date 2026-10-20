// ninferno::ops::detail - embedding rows of a table in any format the generic projection path
// decodes (ggml blocks, groupwise Q3..Q8, zero-point int4, ...): one thread per output value, the
// generic scalar decode.
#include "core/device.h" // CUDA_CHECK
#include "core/weight.h"
#include "ops/generic/generic_gemm.cuh"
#include "ops/launcher/embed_gather.h"
#include "ops/common/arch_limits.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

__global__ void embed_gather_generic_kernel(const std::int32_t* ids, generic::GemmArgs a,
                                            generic::WeightCodec codec, __nv_bfloat16* out,
                                            std::int32_t T) {
    const std::int64_t n      = static_cast<std::int64_t>(a.k) * T;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += stride) {
        const auto t = static_cast<std::int32_t>(i / a.k);
        const auto k = static_cast<std::int32_t>(i - static_cast<std::int64_t>(t) * a.k);
        out[i]       = __float2bfloat16_rn(generic::decode_element(a, codec, ids[t], k));
    }
}

} // namespace

void embed_gather_generic_launch(const Tensor& ids, const Weight& table, Tensor& out,
                                 cudaStream_t stream) {
    if (!generic::supports(table) || table.k != out.ne[0]) {
        throw std::invalid_argument("embedding: table format, layout or width the generic decode "
                                    "cannot read");
    }
    const generic::GemmArgs a = generic::weight_args(table);
    const auto T              = static_cast<std::int32_t>(ids.ne[0]);
    const std::int64_t n      = static_cast<std::int64_t>(table.k) * T;
    // At most 65535 blocks; the kernel strides over the rest.
    constexpr std::int64_t kMaxBlocks = 65535;
    const int blocks = static_cast<int>(std::min<std::int64_t>((n + 255) / 256, kMaxBlocks));
    embed_gather_generic_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), a, generic::codec_of(table),
        static_cast<__nv_bfloat16*>(out.data), T);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninferno::ops::detail
