#pragma once

#include <cstdint>

namespace ninferno::ops {

__global__ void fill_i32_positions_kernel(std::int32_t* positions, std::int32_t count,
                                          std::int32_t start) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) { positions[i] = start + i; }
}

__global__ void offset_i32_positions_kernel(const std::int32_t* source, const std::int32_t* delta,
                                            std::int32_t* destination, std::int32_t count) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) { destination[i] = source[i] + delta[0]; }
}

__global__ void feed_sampled_tokens_kernel(const std::int32_t* sampled, std::int32_t* record,
                                           std::int32_t* tokens, std::int32_t* cache_positions,
                                           std::int32_t* rope_positions, std::int32_t count) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) {
        const std::int32_t token = sampled[i];
        record[i]                = token;
        tokens[i]                = token;
        cache_positions[i] += 1;
        rope_positions[i] += 1;
    }
}

} // namespace ninferno::ops
