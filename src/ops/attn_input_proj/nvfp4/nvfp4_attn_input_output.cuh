#pragma once
#include "ops/linear/common/output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninferno::ops::detail {

// The four planes of the Qwen3.5-27B attention input projection, by parent row: query [0, 6144),
// key [6144, 7168), gate [7168, 13312), value [13312, 14336); token-major within each plane.
// The exact route's register MMA stores through it (nvfp4_a16_exact.cuh).
struct Nvfp4AttentionInputPlanes {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float result) const {
        constexpr std::int32_t kQueryRows  = 6144;
        constexpr std::int32_t kKeyRows    = 1024;
        constexpr std::int32_t kGateRows   = 6144;
        constexpr std::int32_t kKeyBegin   = kQueryRows;
        constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
        constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;
        const __nv_bfloat16 result_bf16    = __float2bfloat16_rn(result);

        if (parent_row < kKeyBegin) {
            query[static_cast<std::int64_t>(token) * kQueryRows + parent_row] = result_bf16;
        } else if (parent_row < kGateBegin) {
            key[static_cast<std::int64_t>(token) * kKeyRows + parent_row - kKeyBegin] = result_bf16;
        } else if (parent_row < kValueBegin) {
            gate[static_cast<std::int64_t>(token) * kGateRows + parent_row - kGateBegin] =
                result_bf16;
        } else {
            value[static_cast<std::int64_t>(token) * kKeyRows + parent_row - kValueBegin] =
                result_bf16;
        }
    }
};

using Nvfp4AttentionInputOutput = LinearBf16SegmentedOutput<6144, 1024, 6144, 1024>;
} // namespace ninferno::ops::detail
