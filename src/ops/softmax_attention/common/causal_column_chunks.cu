#include "ops/softmax_attention/common/causal_column_chunks.h"

#include "core/device.h"

namespace ninferno::ops::detail {
namespace {

__global__ void chunk_valid_columns_kernel(const std::int32_t* valid, int offset, int width,
                                           std::int32_t* out) {
    out[0] = min(max(valid[0] - offset, 0), width);
}

} // namespace

void causal_chunk_valid_columns(const std::int32_t* valid, int offset, int width, std::int32_t* out,
                                cudaStream_t stream) {
    chunk_valid_columns_kernel<<<1, 1, 0, stream>>>(valid, offset, width, out);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninferno::ops::detail
