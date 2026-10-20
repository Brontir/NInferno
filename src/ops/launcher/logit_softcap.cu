#include "ops/launcher/logit_softcap.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kernel/logit_softcap.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ninferno::ops::detail {

void logit_softcap_launch(Tensor& logits, float cap, float scale, cudaStream_t stream) {
    constexpr int block  = 256;
    const std::int64_t n = logits.numel();
    // The reciprocal is computed once on the host: the kernel is memory bound, and a division
    // per element would be paid on every one of a quarter-million vocabulary rows per token.
    // A logit multiplier rides on the same factor: cap * tanh(z * (scale / cap)).
    const float inverse_cap = scale / cap;
    const auto address      = reinterpret_cast<std::uintptr_t>(logits.data);
    if ((address & (alignof(__nv_bfloat162) - 1)) == 0 && n >= 2) {
        const std::int64_t pairs = n / 2;
        const int grid           = static_cast<int>(std::min<std::int64_t>(
            div_up(pairs, static_cast<std::int64_t>(block)), std::numeric_limits<int>::max()));
        auto* paired             = static_cast<__nv_bfloat162*>(logits.data);
        auto* tail               = static_cast<__nv_bfloat16*>(logits.data) + pairs * 2;
        logit_softcap_bf16x2_kernel<<<grid, block, 0, stream>>>(paired, pairs, tail, (n & 1) != 0,
                                                                cap, inverse_cap);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const int grid = static_cast<int>(std::min<std::int64_t>(
        div_up(n, static_cast<std::int64_t>(block)), std::numeric_limits<int>::max()));
    logit_softcap_kernel<<<grid, block, 0, stream>>>(static_cast<__nv_bfloat16*>(logits.data), n,
                                                     cap, inverse_cap);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninferno::ops::detail
