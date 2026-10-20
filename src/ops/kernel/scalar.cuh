#pragma once

#include <cstdint>

namespace ninferno::ops {

__global__ void set_i32_scalar_kernel(std::int32_t* destination, std::int32_t value) {
    destination[0] = value;
}

__global__ void add_i32_scalars_kernel(const std::int32_t* lhs, const std::int32_t* rhs,
                                       std::int32_t* destination) {
    destination[0] = lhs[0] + rhs[0];
}

__global__ void increment_i32_scalar_kernel(std::int32_t* scalar) { ++scalar[0]; }

__global__ void increment_i64_scalar_kernel(std::int64_t* scalar) { ++scalar[0]; }

__global__ void fill_i32_kernel(std::int32_t* destination, std::int64_t count, std::int32_t value) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        destination[i] = value;
    }
}

// One CTA: 16-byte words when both ends allow them, then the remaining bytes.
__global__ void copy_bytes_kernel(unsigned char* destination, const unsigned char* source,
                                  std::int64_t bytes, bool wide) {
    std::int64_t done = 0;
    if (wide) {
        const std::int64_t words = bytes / 16;
        auto* d                  = reinterpret_cast<uint4*>(destination);
        const auto* s            = reinterpret_cast<const uint4*>(source);
        for (std::int64_t i = threadIdx.x; i < words; i += blockDim.x) { d[i] = s[i]; }
        done = words * 16;
    }
    for (std::int64_t i = done + threadIdx.x; i < bytes; i += blockDim.x) {
        destination[i] = source[i];
    }
    // A host polling a later copy's flag (DFlashDecodeEgress::drafts_epoch) sees these bytes first.
    __threadfence_system();
}

} // namespace ninferno::ops
