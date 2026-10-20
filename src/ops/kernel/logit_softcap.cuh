#pragma once

// Implements: include/ninferno/ops/logit_softcap.h
// One saturating transform per element, vectorized two at a time when the storage allows it.

#include <cuda_bf16.h>

#include <cstdint>

namespace ninferno::ops {

__device__ __forceinline__ float softcap_value(float z, float cap, float inverse_cap) {
    // tanhf already saturates to +/-1 well before the argument overflows, so the product lands
    // exactly on +/-cap for large |z| rather than producing an infinity.
    return cap * tanhf(z * inverse_cap);
}

static __global__ void logit_softcap_kernel(__nv_bfloat16* logits, std::int64_t count, float cap,
                                            float inverse_cap) {
    const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        logits[index] = __float2bfloat16_rn(
            softcap_value(__bfloat162float(logits[index]), cap, inverse_cap));
    }
}

static __global__ void logit_softcap_bf16x2_kernel(__nv_bfloat162* logits, std::int64_t pairs,
                                                   __nv_bfloat16* tail, bool has_tail, float cap,
                                                   float inverse_cap) {
    const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < pairs; index += stride) {
        const float2 value = __bfloat1622float2(logits[index]);
        logits[index]      = __floats2bfloat162_rn(softcap_value(value.x, cap, inverse_cap),
                                                   softcap_value(value.y, cap, inverse_cap));
    }
    if (has_tail && blockIdx.x == 0 && threadIdx.x == 0) {
        *tail = __float2bfloat16_rn(softcap_value(__bfloat162float(*tail), cap, inverse_cap));
    }
}

} // namespace ninferno::ops
