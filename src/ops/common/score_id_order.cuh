#pragma once

// Total-order key for an FP32 score and non-negative int32 id. Larger keys mean higher scores;
// exact numeric ties mean lower ids. Numeric zero is canonicalized so +0 and -0 reach the id
// tie-break, and NaN ranks as -inf (rank_score). Key zero is reserved as a sentinel below every
// score/id pair.

#include <climits>
#include <cstdint>

#include <cuda_runtime.h>

namespace ninferno::ops {

struct ScoreIdOrderGreater {
    __device__ __forceinline__ bool operator()(std::uint64_t lhs, std::uint64_t rhs) const {
        return lhs > rhs;
    }
};

// The score a candidate ranks by: NaN ranks as -inf. A column of NaN logits (a numerically failed
// forward) then still selects an id of its domain, the lowest, where a reduction seeded with
// (-inf, INT_MAX) would keep its seed and hand an out-of-range id to the embedding and the token
// counts.
__device__ __forceinline__ float rank_score(float value) {
    return isnan(value) ? __int_as_float(0xff800000) : value;
}

__device__ __forceinline__ std::uint32_t ordered_score_bits(float value) {
    value = rank_score(value);
    if (value == 0.0f) { value = 0.0f; }
    const std::uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);
}

__device__ __forceinline__ std::uint64_t score_id_order_key(float value, std::int32_t id) {
    if (id == INT_MAX) { return 0; }
    return (static_cast<std::uint64_t>(ordered_score_bits(value)) << 32) |
           static_cast<std::uint32_t>(0xffffffffu - static_cast<std::uint32_t>(id));
}

__device__ __forceinline__ float score_from_order_key(std::uint64_t key) {
    const std::uint32_t ordered = static_cast<std::uint32_t>(key >> 32);
    const std::uint32_t bits    = (ordered & 0x80000000u) != 0 ? (ordered ^ 0x80000000u) : ~ordered;
    return __uint_as_float(bits);
}

__device__ __forceinline__ std::int32_t id_from_order_key(std::uint64_t key) {
    if (key == 0) { return INT_MAX; }
    return static_cast<std::int32_t>(0xffffffffu - static_cast<std::uint32_t>(key));
}

} // namespace ninferno::ops
