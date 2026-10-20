// HostExperts' mailbox, the GPU side (handoff.h).
#include "ops/host_experts/handoff.h"

#include "core/device.h"

#include <algorithm>

namespace ninferno::ops::host {

namespace {

constexpr int kThreads = 256;

__device__ void copy_bytes(const void* from, void* to, std::size_t bytes, std::size_t begin,
                           std::size_t stride, bool uncached_source) {
    // 16-byte words while they last, then bytes. Every staging size here is a multiple of 16;
    // the byte tail keeps the copy exact for any size.
    const std::size_t words = bytes / 16;
    const auto* src         = static_cast<const uint4*>(from);
    auto* dst               = static_cast<uint4*>(to);
    for (std::size_t i = begin; i < words; i += stride) { dst[i] = uncached_source ? __ldcv(src + i) : src[i]; }
    const auto* src8 = static_cast<const unsigned char*>(from);
    auto* dst8       = static_cast<unsigned char*>(to);
    for (std::size_t i = words * 16 + begin; i < bytes; i += stride) {
        dst8[i] = uncached_source ? *static_cast<const volatile unsigned char*>(src8 + i) : src8[i];
    }
}

__global__ void __launch_bounds__(kThreads) post_kernel(PostArgs a) {
    copy_bytes(a.x, a.x_host, a.x_bytes, threadIdx.x, kThreads, false);
    for (int i = threadIdx.x; i < a.ids_count; i += kThreads) {
        a.ids_host[i] = a.ids[i];
        if (a.weights != nullptr) { a.weights_host[i] = a.weights[i]; }
    }
    // Every thread's inputs reach host memory before the call is published.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const std::uint64_t sequence = a.sequence[0] + 1;
        a.sequence[0]                = sequence;
        a.sequence[1]                = sequence;
        a.mailbox->call              = a.call;
        __threadfence_system();
        a.mailbox->request = sequence;
        __threadfence_system();
    }
}

__global__ void __launch_bounds__(kThreads) await_kernel(AwaitArgs a) {
    if (threadIdx.x == 0) {
        const std::uint64_t want = a.sequence[1];
        while (a.mailbox->done < want) { __nanosleep(256); }
        __threadfence_system();
    }
    __syncthreads();
    copy_bytes(a.out_host, a.out, a.out_bytes, blockIdx.x * kThreads + threadIdx.x,
               static_cast<std::size_t>(gridDim.x) * kThreads, true);
    if (a.slot_of != nullptr && blockIdx.x == 0) {
        for (int i = threadIdx.x; i < a.slots; i += kThreads) { a.slot_of[i] = i; }
    }
}

__global__ void reset_kernel(std::uint64_t* sequence) {
    sequence[0] = 0;
    sequence[1] = 0;
}

} // namespace

void launch_reset(std::uint64_t* sequence, cudaStream_t stream) {
    reset_kernel<<<1, 1, 0, stream>>>(sequence);
    CUDA_CHECK(cudaGetLastError());
}

void launch_post(const PostArgs& args, cudaStream_t stream) {
    post_kernel<<<1, kThreads, 0, stream>>>(args);
    CUDA_CHECK(cudaGetLastError());
}

void launch_await(const AwaitArgs& args, cudaStream_t stream) {
    // A few blocks share the read of the result over PCIe; each waits for the answer itself.
    const std::size_t words = (args.out_bytes + 15) / 16;
    const int blocks = static_cast<int>(std::clamp<std::size_t>((words + 4 * kThreads - 1) / (4 * kThreads), 1, 8));
    await_kernel<<<blocks, kThreads, 0, stream>>>(args);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninferno::ops::host
