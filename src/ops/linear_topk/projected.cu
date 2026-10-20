// The projected route of linear_topk: vocabulary logits already computed by Linear (BF16 [N,U]),
// reduced to the stable top sixteen per column. Each producer CTA sorts one column's
// kLinearTopKProjectedRows rows by (score desc, id asc) and keeps its first sixteen keys; the
// shared merge tree does the rest.
#include "ops/linear_topk/linear_topk_launch.h"

#include "core/device.h"
#include "ops/common/score_id_order.cuh"

#include <cub/block/block_merge_sort.cuh>
#include <cuda_bf16.h>

#include <climits>
#include <cstdint>

namespace ninferno::ops::detail {
namespace {

inline constexpr int kProducerThreads = 256;
inline constexpr int kProducerItems   = kLinearTopKProjectedRows / kProducerThreads;

using ProducerSort = cub::BlockMergeSort<std::uint64_t, kProducerThreads, kProducerItems>;

__global__ __launch_bounds__(kProducerThreads) void linear_topk_projected_kernel(
    const __nv_bfloat16* __restrict__ logits, std::int32_t rows, std::int32_t valid_rows,
    std::uint64_t* __restrict__ partial_keys, std::int32_t producer_groups) {
    const std::int32_t group  = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.y);
    const int tid             = static_cast<int>(threadIdx.x);
    __shared__ typename ProducerSort::TempStorage storage;
    std::uint64_t keys[kProducerItems];
#pragma unroll
    for (int item = 0; item < kProducerItems; ++item) {
        const std::int32_t row = group * kLinearTopKProjectedRows + item * kProducerThreads + tid;
        keys[item]             = row < valid_rows
                                     ? score_id_order_key(__bfloat162float(
                                                  logits[static_cast<std::int64_t>(column) * rows + row]),
                                              row)
                                     : 0;
    }
    ProducerSort(storage).Sort(keys, ScoreIdOrderGreater{});
    // Blocked arrangement: thread t holds ranks [t * items, (t + 1) * items).
#pragma unroll
    for (int item = 0; item < kProducerItems; ++item) {
        const int rank = tid * kProducerItems + item;
        if (rank < kLinearTopK) {
            partial_keys[(static_cast<std::int64_t>(column) * producer_groups + group) *
                             kLinearTopK +
                         rank] = keys[item];
        }
    }
}

} // namespace

void linear_topk_projected_launch(const Tensor& logits, std::int32_t valid_rows,
                                  const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(workspace.producer_groups),
                    static_cast<unsigned>(workspace.columns), 1U);
    linear_topk_projected_kernel<<<grid, kProducerThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), logits.ne[0], valid_rows,
        static_cast<std::uint64_t*>(workspace.partial_keys.data), workspace.producer_groups);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninferno::ops::detail
