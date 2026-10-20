#pragma once

// The GPU side of HostExperts' mailbox (host_experts.h): two kernels a decode call enqueues around
// the CPU's work. Both run in the stream's order and inside a captured CUDA graph.
//
//   post:  copies the call's inputs into mapped host memory, then (after a system-scope fence)
//          publishes the call: mailbox.call, then mailbox.request = ++sequence.
//   await: spins until mailbox.done reaches that sequence, then copies the CPU's result from
//          mapped host memory to the device (and, for the generic layer, writes slot_of as the
//          identity).

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninferno::ops::host {

// One cache line each for the words the two sides write, in mapped (cudaHostAllocMapped) memory.
struct alignas(128) Mailbox {
    volatile std::uint64_t request = 0; // written by the GPU: the sequence number of the call
    volatile std::uint32_t call    = 0; // written by the GPU: HostExperts' call index
    std::uint8_t pad0[116]{};
    volatile std::uint64_t done = 0; // written by the CPU: the last sequence number answered
    std::uint8_t pad1[120]{};
};

struct PostArgs {
    const void* x               = nullptr; // device, bytes x_bytes
    std::size_t x_bytes         = 0;
    const std::int32_t* ids     = nullptr; // device, count ids
    const float* weights        = nullptr; // device or null, count ids
    int ids_count               = 0;
    void* x_host                = nullptr; // device aliases of the mapped staging
    std::int32_t* ids_host      = nullptr;
    float* weights_host         = nullptr;
    Mailbox* mailbox            = nullptr; // device alias
    std::uint64_t* sequence     = nullptr; // device: [0] the counter, [1] the last posted
    std::uint32_t call          = 0;
};

struct AwaitArgs {
    const void* out_host       = nullptr; // device alias of the mapped result
    void* out                  = nullptr; // device
    std::size_t out_bytes      = 0;
    std::int32_t* slot_of      = nullptr; // device or null: set to the identity over slots
    int slots                  = 0;
    const Mailbox* mailbox     = nullptr; // device alias
    const std::uint64_t* sequence = nullptr;
};

// Zeroes the sequence words (before the first post).
void launch_reset(std::uint64_t* sequence, cudaStream_t stream);
void launch_post(const PostArgs& args, cudaStream_t stream);
void launch_await(const AwaitArgs& args, cudaStream_t stream);

} // namespace ninferno::ops::host
