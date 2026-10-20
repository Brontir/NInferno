// Implements: include/ninferno/ops/token_distribution.h
// Match: wrapper-validated contiguous tensors and valid vocabulary rows.
// Algorithm assumptions: one independent CTA per column; no global workspace.
#include "ops/launcher/token_distribution.h"

#include "core/device.h"
#include "ops/kernel/token_distribution.cuh"

namespace ninferno::ops::detail {

void logprob_topk_launch(const Tensor& logits, std::int32_t valid_rows, Tensor& ids,
                         Tensor& logprobs, cudaStream_t stream) {
    const auto columns = static_cast<unsigned int>(logits.ne[1]);
    logprob_topk_kernel<kTokenDistributionBlock><<<columns, kTokenDistributionBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), valid_rows, logits.ne[0], ids.ne[0],
        static_cast<std::int32_t*>(ids.data), static_cast<float*>(logprobs.data));
    CUDA_CHECK(cudaGetLastError());
}

void gather_logprobs_launch(const Tensor& logits, std::int32_t valid_rows, const Tensor& ids,
                            Tensor& out, cudaStream_t stream) {
    const auto columns = static_cast<unsigned int>(logits.ne[1]);
    gather_logprobs_kernel<kTokenDistributionBlock>
        <<<columns, kTokenDistributionBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), valid_rows, logits.ne[0], ids.ne[0],
            static_cast<const std::int32_t*>(ids.data), static_cast<float*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninferno::ops::detail
