#pragma once

// Implements: ops::token_logprobs (include/ninferno/ops/sampling.h)
// Match: contiguous BF16 logits [physical_rows,B], token domain <= 2^20.
// Algorithm assumptions: one block per row. Pass 1 takes the log-sum-exp online and the histogram
// of the top key digit; two more passes narrow the digit histogram under the chosen prefix; the
// key (ordered BF16 bits, inverted id) is unique, so the threshold key leaves exactly `top` keys at
// or above it, collected in a last pass and sorted by one thread.

#include "ninferno/ops/sampling.h"

#include <cub/block/block_scan.cuh>
#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninferno::ops {

inline constexpr int kLogprobBlock   = 1024;
inline constexpr int kLogprobIdBits  = 20;
inline constexpr int kLogprobBins    = 4096; // the widest digit, 12 bits
inline constexpr int kLogprobDigits  = 3;
// Digits of the 36-bit key, most significant first: bits [24,36), [12,24), [0,12).
__host__ __device__ constexpr int logprob_shift(int digit) { return (2 - digit) * 12; }
__host__ __device__ constexpr int logprob_width(int /*digit*/) { return 12; }

static_assert(kMaximumLogprobTokenDomain == (1 << kLogprobIdBits));
static_assert(kLogprobBins % kLogprobBlock == 0);

// NaN reads as -inf; +inf as the largest finite float, so a row holding it keeps a finite
// log-sum-exp (inf - inf would make every value NaN).
__device__ __forceinline__ float logprob_value(__nv_bfloat16 raw) {
    const float value = __bfloat162float(raw);
    return isnan(value) ? -CUDART_INF_F : fminf(value, 3.402823466e+38F);
}

// Higher logit => higher key; equal logits => the lower id has the higher key. NaN is -inf and
// -0 is +0.
__device__ __forceinline__ unsigned long long logprob_key(__nv_bfloat16 raw, int id) {
    unsigned int bits = __bfloat16_as_ushort(raw);
    if ((bits & 0x7fffu) > 0x7f80u) { bits = 0xff80u; }
    if ((bits & 0x7fffu) == 0u) { bits = 0u; }
    const unsigned int ordered = (bits & 0x8000u) ? ((~bits) & 0xffffu) : (bits ^ 0x8000u);
    return (static_cast<unsigned long long>(ordered) << kLogprobIdBits) |
           static_cast<unsigned long long>((1u << kLogprobIdBits) - 1u - static_cast<unsigned>(id));
}

__device__ __forceinline__ int logprob_key_id(unsigned long long key) {
    return static_cast<int>((1u << kLogprobIdBits) - 1u -
                            static_cast<unsigned>(key & ((1u << kLogprobIdBits) - 1u)));
}

// Merges two (max, sum of exp(x - max)) pairs.
__device__ __forceinline__ void logprob_merge(float& m, float& s, float other_m, float other_s) {
    const float top = fmaxf(m, other_m);
    if (top == -CUDART_INF_F) { return; }
    s = s * expf(m - top) + other_s * expf(other_m - top);
    m = top;
}

// Adds one to hist[bin] for every lane whose bin is >= 0, one atomic per distinct bin in the warp
// (logits cluster in few bins). Every lane of the warp must call.
__device__ __forceinline__ void logprob_count(int* hist, int bin) {
    const unsigned int peers = __match_any_sync(0xffffffffu, bin);
    if (bin >= 0 && static_cast<int>(threadIdx.x & 31u) == __ffs(static_cast<int>(peers)) - 1) {
        atomicAdd(&hist[bin], __popc(peers));
    }
}

__launch_bounds__(kLogprobBlock) __global__
    void token_logprobs_kernel(const __nv_bfloat16* logits, std::int32_t physical_rows,
                               std::int32_t token_domain, const std::int32_t* sampled,
                               const SamplingConfig* configs, TokenLogprobs* out) {
    const int row  = static_cast<int>(blockIdx.x);
    const int want = configs[row].logprobs;
    if (want < 0) { return; }
    const int top              = min(min(want, kMaximumTopLogprobs), token_domain);
    const __nv_bfloat16* x     = logits + static_cast<std::int64_t>(row) * physical_rows;
    const int lane             = static_cast<int>(threadIdx.x & 31u);
    const int warp             = static_cast<int>(threadIdx.x >> 5);
    constexpr int kBinsPerThread = kLogprobBins / kLogprobBlock;

    using Scan = cub::BlockScan<int, kLogprobBlock>;
    __shared__ typename Scan::TempStorage scan_storage;
    __shared__ int hist[kLogprobBins];
    __shared__ float warp_m[kLogprobBlock / 32];
    __shared__ float warp_s[kLogprobBlock / 32];
    __shared__ unsigned long long prefix;
    __shared__ int remaining;
    __shared__ int picked_count;
    __shared__ unsigned long long picked[kMaximumTopLogprobs];

    for (int i = static_cast<int>(threadIdx.x); i < kLogprobBins; i += kLogprobBlock) {
        hist[i] = 0;
    }
    if (threadIdx.x == 0) {
        prefix       = 0ull;
        remaining    = top;
        picked_count = 0;
    }
    __syncthreads();

    // Pass 1: the online log-sum-exp and the top digit's histogram.
    float m = -CUDART_INF_F;
    float s = 0.0f;
    for (int base = 0; base < token_domain; base += kLogprobBlock) {
        const int v = base + static_cast<int>(threadIdx.x);
        int bin     = -1;
        if (v < token_domain) {
            const __nv_bfloat16 raw = x[v];
            const float value       = logprob_value(raw);
            if (value > m) {
                s = s * expf(m - value) + 1.0f;
                m = value;
            } else if (value > -CUDART_INF_F) {
                s += expf(value - m);
            }
            if (top > 0) {
                bin = static_cast<int>(logprob_key(raw, v) >> logprob_shift(0)) &
                      ((1 << logprob_width(0)) - 1);
            }
        }
        logprob_count(hist, bin);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        logprob_merge(m, s, __shfl_down_sync(0xffffffffu, m, offset),
                      __shfl_down_sync(0xffffffffu, s, offset));
    }
    if (lane == 0) {
        warp_m[warp] = m;
        warp_s[warp] = s;
    }
    __syncthreads();
    if (warp == 0) {
        m = warp_m[lane];
        s = warp_s[lane];
        for (int offset = 16; offset > 0; offset >>= 1) {
            logprob_merge(m, s, __shfl_down_sync(0xffffffffu, m, offset),
                          __shfl_down_sync(0xffffffffu, s, offset));
        }
        if (lane == 0) { warp_m[0] = m + logf(s); }
    }
    __syncthreads();
    const float lse = warp_m[0];

    // Each digit: find the bin where the count from the top reaches `remaining`, fix it in the
    // prefix, and keep what is still to take inside it.
    unsigned long long prefix_mask = 0ull;
#pragma unroll
    for (int digit = 0; digit < kLogprobDigits; ++digit) {
        if (top == 0) { break; }
        const int shift = logprob_shift(digit);
        const int bins  = 1 << logprob_width(digit);
        if (digit > 0) {
            __syncthreads();
            for (int i = static_cast<int>(threadIdx.x); i < kLogprobBins; i += kLogprobBlock) {
                hist[i] = 0;
            }
            __syncthreads();
            const unsigned long long want_prefix = prefix;
            for (int base = 0; base < token_domain; base += kLogprobBlock) {
                const int v = base + static_cast<int>(threadIdx.x);
                int bin     = -1;
                if (v < token_domain) {
                    const unsigned long long key = logprob_key(x[v], v);
                    if ((key & prefix_mask) == want_prefix) {
                        bin = static_cast<int>(key >> shift) & (bins - 1);
                    }
                }
                logprob_count(hist, bin);
            }
            __syncthreads();
        }
        // Thread t owns kBinsPerThread consecutive bins counted from the top bin down.
        int counts[kBinsPerThread];
        int owned = 0;
#pragma unroll
        for (int j = 0; j < kBinsPerThread; ++j) {
            const int bin = bins - 1 - (static_cast<int>(threadIdx.x) * kBinsPerThread + j);
            counts[j]     = bin >= 0 ? hist[bin] : 0;
            owned += counts[j];
        }
        int above = 0;
        Scan(scan_storage).ExclusiveSum(owned, above);
        const int need = remaining;
        __syncthreads();
        if (above < need && above + owned >= need) {
#pragma unroll
            for (int j = 0; j < kBinsPerThread; ++j) {
                if (above < need && above + counts[j] >= need) {
                    const int bin = bins - 1 - (static_cast<int>(threadIdx.x) * kBinsPerThread + j);
                    prefix |= static_cast<unsigned long long>(bin) << shift;
                    remaining = need - above;
                }
                above += counts[j];
            }
        }
        prefix_mask |= static_cast<unsigned long long>(bins - 1) << shift;
        __syncthreads();
    }

    // The key is unique, so exactly `top` keys are at or above the threshold.
    if (top > 0) {
        const unsigned long long threshold = prefix;
        for (int base = 0; base < token_domain; base += kLogprobBlock) {
            const int v = base + static_cast<int>(threadIdx.x);
            if (v < token_domain) {
                const unsigned long long key = logprob_key(x[v], v);
                if (key >= threshold) {
                    const int slot = atomicAdd(&picked_count, 1);
                    if (slot < kMaximumTopLogprobs) { picked[slot] = key; }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x != 0) { return; }
    TokenLogprobs& result = out[row];
    const int count       = min(picked_count, top);
    for (int i = 1; i < count; ++i) {
        const unsigned long long key = picked[i];
        int j                        = i - 1;
        while (j >= 0 && picked[j] < key) {
            picked[j + 1] = picked[j];
            --j;
        }
        picked[j + 1] = key;
    }
    for (int i = 0; i < count; ++i) {
        const int id      = logprob_key_id(picked[i]);
        result.ids[i]     = id;
        result.values[i]  = logprob_value(x[id]) - lse;
    }
    result.count        = count;
    const int token     = sampled[row];
    result.sampled      = token >= 0 && token < token_domain ? logprob_value(x[token]) - lse
                                                             : -CUDART_INF_F;
}

} // namespace ninferno::ops
