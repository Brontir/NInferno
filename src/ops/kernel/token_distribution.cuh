#pragma once

// Implements: include/ninferno/ops/token_distribution.h
// Match: contiguous BF16 [physical_rows,C], I32 / FP32 [K,C].
// Algorithm assumptions: one 256-thread CTA per column. The log-sum-exp is target_logprobs'
// single online pass. logprob_topk then narrows a threshold below the column's maximum until at
// most kTokenTopkCandidates rows reach it (counting passes, bisection over the BF16 values),
// collects those rows in shared memory and sorts them (bitonic, value descending, row ascending).

#include "ninferno/ops/token_distribution.h"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <climits>
#include <cstdint>

namespace ninferno::ops {

inline constexpr int kTokenDistributionBlock = 256;

// The column's maximum and log-sum-exp, in every thread.
// Maximum over a block of BlockSize threads, as block_sum: `scratch` holds one float per warp,
// every thread returns the maximum, and the caller may reuse `scratch` at once.
template <int BlockSize>
__device__ __forceinline__ float token_block_max(float value, float* scratch) {
    static_assert(BlockSize % kWarpSize == 0);
    constexpr int kWarps = BlockSize / kWarpSize;
    const int lane       = threadIdx.x & (kWarpSize - 1);
    const int warp       = threadIdx.x / kWarpSize;
    value                = warp_max(value);
    if (lane == 0) { scratch[warp] = value; }
    __syncthreads();
    float maximum = scratch[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) { maximum = fmaxf(maximum, scratch[w]); }
    __syncthreads();
    return maximum;
}

template <int BlockSize>
__device__ __forceinline__ void token_column_lse(const __nv_bfloat16* column,
                                                 std::int32_t valid_rows, float* scratch,
                                                 float& maximum, float& lse) {
    float local_max = -CUDART_INF_F;
    float local_sum = 0.0f;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        const float value   = __bfloat162float(column[row]);
        const float new_max = fmaxf(local_max, value);
        if (new_max > local_max) { local_sum *= expf(local_max - new_max); }
        local_max = new_max;
        local_sum += expf(value - local_max);
    }
    // The kernel's scratch, not target_logprobs_block_max's function-local shared memory: a
    // function's static __shared__ puts an -rdc -lineinfo object's shared variables in one
    // .nv_debug.shared section, and nvlink charged these kernels with another object's 48 KiB
    // (sm_89 beside fp8_attn_input_a8.cu: "uses too much shared data").
    maximum = token_block_max<BlockSize>(local_max, scratch);
    local_sum *= expf(local_max - maximum);
    lse = maximum + logf(block_sum<BlockSize>(local_sum, scratch));
}

// A BF16 value's bits as a key whose unsigned order is the values' order (NaN aside; -0 just
// below +0).
__device__ __forceinline__ unsigned bf16_order_key(__nv_bfloat16 value) {
    const unsigned bits = __bfloat16_as_ushort(value);
    return (bits & 0x8000U) != 0 ? (~bits & 0xffffU) : (bits | 0x8000U);
}

template <int BlockSize>
__launch_bounds__(BlockSize) __global__
    void logprob_topk_kernel(const __nv_bfloat16* logits, std::int32_t valid_rows,
                             std::int32_t physical_rows, std::int32_t k, std::int32_t* ids,
                             float* logprobs) {
    constexpr int kCap = kTokenTopkCandidates;
    static_assert((kCap & (kCap - 1)) == 0);
    __shared__ float s_value[kCap];
    __shared__ std::int32_t s_id[kCap];
    __shared__ float scratch[BlockSize / kWarpSize];
    __shared__ int s_count;
    const int tid               = static_cast<int>(threadIdx.x);
    const std::int64_t column   = blockIdx.x;
    const __nv_bfloat16* values = logits + column * physical_rows;

    float maximum = 0.0f, lse = 0.0f;
    token_column_lse<BlockSize>(values, valid_rows, scratch, maximum, lse);

    // Rows at or above maximum - delta.
    const auto count_at = [&](float delta) {
        const float threshold = maximum - delta;
        float count           = 0.0f;
        for (std::int32_t row = tid; row < valid_rows; row += BlockSize) {
            count += __bfloat162float(values[row]) >= threshold ? 1.0f : 0.0f;
        }
        return block_sum<BlockSize>(count, scratch);
    };
    // Rows whose order key (bf16_order_key) is at least `key`.
    const auto count_key = [&](unsigned key) {
        float count = 0.0f;
        for (std::int32_t row = tid; row < valid_rows; row += BlockSize) {
            count += bf16_order_key(values[row]) >= key ? 1.0f : 0.0f;
        }
        return block_sum<BlockSize>(count, scratch);
    };
    // The 30-logit window when its rows fit the buffer. Otherwise a bisection over the BF16 values
    // themselves (their order keys, 16 steps at most) for the first threshold whose rows fit and
    // number K or more; if none does (more than kCap rows tie at the K-th value), it ends on two
    // adjacent values: the rows above (fewer than K) go in first, then the tied rows of the value
    // below, truncated to the buffer, so only tied rows are dropped.
    bool keyed = false, truncating = false;
    unsigned key_hi = 0, key_lo = 0;
    if (count_at(kTokenTopkWindow) > static_cast<float>(kCap)) {
        keyed = true;
        // Every row reaches key 0, more than kCap of them; the maximum's key, at least one.
        unsigned lo = 0, hi = bf16_order_key(__float2bfloat16(maximum));
        float hi_count = count_key(hi);
        if (hi_count <= static_cast<float>(kCap)) { // else more than kCap rows tie at the maximum
            while (hi - lo > 1 && hi_count < static_cast<float>(k)) {
                const unsigned mid = (lo + hi) / 2;
                const float count  = count_key(mid);
                if (count > static_cast<float>(kCap)) {
                    lo = mid;
                } else {
                    hi       = mid;
                    hi_count = count;
                }
            }
            truncating = hi_count < static_cast<float>(k);
        }
        key_hi = hi;
        key_lo = lo;
    }

    const float threshold = maximum - kTokenTopkWindow;
    for (int i = tid; i < kCap; i += BlockSize) {
        s_value[i] = -CUDART_INF_F;
        s_id[i]    = INT_MAX;
    }
    if (tid == 0) s_count = 0;
    __syncthreads();
    if (truncating) { // uniform: the rows above the tie, fewer than K
        for (std::int32_t row = tid; row < valid_rows; row += BlockSize) {
            if (bf16_order_key(values[row]) >= key_hi) {
                const int slot = atomicAdd(&s_count, 1);
                s_value[slot]  = __bfloat162float(values[row]);
                s_id[slot]     = row;
            }
        }
        __syncthreads();
    }
    for (std::int32_t row = tid; row < valid_rows; row += BlockSize) {
        const float value  = __bfloat162float(values[row]);
        const unsigned key = bf16_order_key(values[row]);
        const bool take = truncating ? key == key_lo : keyed ? key >= key_hi : value >= threshold;
        if (take) {
            const int slot = atomicAdd(&s_count, 1);
            if (slot < kCap) {
                s_value[slot] = value;
                s_id[slot]    = row;
            }
        }
    }
    __syncthreads();

    // Bitonic sort: slot i before slot j when its value is larger, or equal with a lower row.
    const auto before = [&](int a, int b) {
        return s_value[a] > s_value[b] || (s_value[a] == s_value[b] && s_id[a] < s_id[b]);
    };
    for (int size = 2; size <= kCap; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < kCap; i += BlockSize) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool forward = (i & size) == 0;
                    if (forward ? before(j, i) : before(i, j)) {
                        const float v = s_value[i];
                        s_value[i]    = s_value[j];
                        s_value[j]    = v;
                        const int id  = s_id[i];
                        s_id[i]       = s_id[j];
                        s_id[j]       = id;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < k; i += BlockSize) {
        const bool listed        = s_id[i] != INT_MAX;
        ids[column * k + i]      = listed ? s_id[i] : -1;
        logprobs[column * k + i] = listed ? s_value[i] - lse : -CUDART_INF_F;
    }
}

template <int BlockSize>
__launch_bounds__(BlockSize) __global__
    void gather_logprobs_kernel(const __nv_bfloat16* logits, std::int32_t valid_rows,
                                std::int32_t physical_rows, std::int32_t k, const std::int32_t* ids,
                                float* out) {
    __shared__ float scratch[BlockSize / kWarpSize];
    const std::int64_t column   = blockIdx.x;
    const __nv_bfloat16* values = logits + column * physical_rows;
    float maximum = 0.0f, lse = 0.0f;
    token_column_lse<BlockSize>(values, valid_rows, scratch, maximum, lse);
    for (int i = static_cast<int>(threadIdx.x); i < k; i += BlockSize) {
        const std::int32_t id = ids[column * k + i];
        out[column * k + i] =
            id >= 0 && id < valid_rows ? __bfloat162float(values[id]) - lse : -CUDART_INF_F;
    }
}

} // namespace ninferno::ops
