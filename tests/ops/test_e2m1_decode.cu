// E2M1 pair decoding (ops/common/e2m1.cuh) for all 256 bytes, bit for bit: the bit form, the
// dispatching decoder this build uses, and CUDA's own conversion must each equal the E2M1 table
// (signed zero included). On a GPU without the hardware conversion CUDA's result comes from its
// software fallback, so the test covers both CUDA paths wherever it runs.

#include "ops/common/e2m1.cuh"

#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

__global__ void decode_all(std::uint32_t* bits, std::uint32_t* dispatched, std::uint32_t* cuda) {
    const std::uint32_t byte = threadIdx.x;
    const __half2 a          = ninferno::ops::detail::e2m1x2_to_half2_bits(byte);
    const __half2 b          = ninferno::ops::detail::e2m1x2_to_half2(byte);
    const __half2 c =
        __half2(__nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(byte), __NV_E2M1));
    bits[byte]       = *reinterpret_cast<const std::uint32_t*>(&a);
    dispatched[byte] = *reinterpret_cast<const std::uint32_t*>(&b);
    cuda[byte]       = *reinterpret_cast<const std::uint32_t*>(&c);
}

std::uint16_t expected_half(std::uint32_t code) {
    // The eight E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 as binary16 bits.
    static const std::uint16_t kMagnitude[8] = {0x0000, 0x3800, 0x3c00, 0x3e00,
                                                0x4000, 0x4200, 0x4400, 0x4600};
    return static_cast<std::uint16_t>(kMagnitude[code & 7U] | ((code & 8U) ? 0x8000U : 0U));
}

} // namespace

int main() {
    if (!cuda_available()) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    std::uint32_t* device = nullptr;
    cudaStream_t stream   = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return 1;
    if (cudaMalloc(&device, 3 * 256 * sizeof(std::uint32_t)) != cudaSuccess) return 1;
    decode_all<<<1, 256, 0, stream>>>(device, device + 256, device + 512);
    std::vector<std::uint32_t> host(3 * 256);
    if (cudaMemcpyAsync(host.data(), device, host.size() * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        std::printf("FAIL: kernel or copy\n");
        return 1;
    }
    cudaFree(device);
    cudaStreamDestroy(stream);

    const char* kNames[3] = {"bit form", "dispatched", "cuda"};
    int failures          = 0;
    for (std::uint32_t byte = 0; byte < 256; ++byte) {
        const std::uint32_t want = expected_half(byte & 0xfU) |
                                   (static_cast<std::uint32_t>(expected_half(byte >> 4)) << 16);
        for (int path = 0; path < 3; ++path) {
            const std::uint32_t got = host[path * 256 + byte];
            if (got != want && failures++ < 16)
                std::printf("FAIL %s byte 0x%02x: 0x%08x, want 0x%08x\n", kNames[path], byte, got,
                            want);
        }
    }
    if (failures != 0) {
        std::printf("FAIL: %d mismatches\n", failures);
        return 1;
    }
    std::printf("PASS: 256 bytes, three paths\n");
    return 0;
}
