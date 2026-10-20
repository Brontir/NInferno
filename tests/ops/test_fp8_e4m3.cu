// E4M3 pair widening (ops/common/fp8_e4m3.cuh) for every pair of codes, bit for bit: the bit
// form and the dispatching decoder this build uses must each equal CUDA's own conversion for every
// finite code (signed zeros and subnormals included). On a GPU without the hardware conversion
// CUDA's result comes from its software fallback. The NaN codes (S.1111.111) are skipped: the bit
// form widens them to +-480, and no saturating writer stores them.

#include "ops/common/fp8_e4m3.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

__global__ void widen_all(std::uint32_t* bits, std::uint32_t* dispatched, std::uint32_t* cuda) {
    const std::uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    const __half2 a          = ninferno::ops::detail::e4m3x2_to_half2_bits(pair);
    const __half2 b          = ninferno::ops::detail::e4m3x2_to_half2(pair);
    const __half2 c =
        __half2(__nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair), __NV_E4M3));
    bits[pair]       = *reinterpret_cast<const std::uint32_t*>(&a);
    dispatched[pair] = *reinterpret_cast<const std::uint32_t*>(&b);
    cuda[pair]       = *reinterpret_cast<const std::uint32_t*>(&c);
}

bool nan_code(std::uint32_t code) { return (code & 0x7fU) == 0x7fU; }

} // namespace

int main() {
    if (!cuda_available()) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    constexpr std::uint32_t kPairs = 1U << 16;
    std::uint32_t* device          = nullptr;
    if (cudaMalloc(&device, 3 * kPairs * sizeof(std::uint32_t)) != cudaSuccess) return 1;
    widen_all<<<kPairs / 256, 256>>>(device, device + kPairs, device + 2 * kPairs);
    std::vector<std::uint32_t> host(3 * kPairs);
    if (cudaMemcpy(host.data(), device, host.size() * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::printf("FAIL: kernel or copy\n");
        return 1;
    }
    cudaFree(device);

    const char* kNames[2] = {"bit form", "dispatched"};
    int failures          = 0;
    std::uint32_t checked = 0;
    for (std::uint32_t pair = 0; pair < kPairs; ++pair) {
        if (nan_code(pair & 0xffU) || nan_code(pair >> 8)) continue;
        ++checked;
        const std::uint32_t want = host[2 * kPairs + pair];
        for (int path = 0; path < 2; ++path) {
            const std::uint32_t got = host[path * kPairs + pair];
            if (got != want && failures++ < 16)
                std::printf("FAIL %s pair 0x%04x: 0x%08x, want 0x%08x\n", kNames[path], pair, got,
                            want);
        }
    }
    if (failures != 0) {
        std::printf("FAIL: %d mismatches\n", failures);
        return 1;
    }
    std::printf("PASS: %u finite pairs, two paths\n", checked);
    return 0;
}
