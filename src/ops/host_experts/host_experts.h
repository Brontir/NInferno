#pragma once

// MoE expert offload's decode path, the CUDA side: a layer's routed experts computed on the CPU
// inside the stream's order (and inside a captured CUDA graph). On `stream`:
//
//   1. a kernel copies the call's inputs (x, the selected ids, and for Gemma 4 their route
//      weights) into mapped host memory and posts the call in a mailbox there (handoff.h),
//   2. this object's server thread, which watches the mailbox, computes the experts with
//      host::HostMoe from the stored planes in host memory, writes the result to mapped host
//      memory and answers,
//   3. a kernel waits for the answer and copies the result to the device.
//
// No host function node and no copy engine sit between the GPU and the CPU: a call's handoff costs
// two small kernels and a poll on each side. The server spins for a while after each call (decode
// calls come a layer apart), then polls at a coarser interval, so an idle server costs little.
//
// The router and the top-k selection stay on the GPU, so the experts are the ones the GPU route
// would select. The staging buffers and the mailbox are shared by every layer: calls on one
// HostExperts must be stream-ordered (one stream, or streams joined between calls). A call that
// fails on the CPU aborts the process with the reason: it cannot report through the stream, and
// the GPU would wait for its answer.

#include "ops/host_experts/handoff.h"
#include "ops/host_experts/host_moe.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace ninferno::ops {

class HostExperts {
public:
    // hidden, top_k: the largest over the layers this object serves (the staging sizes).
    HostExperts(std::int32_t hidden, std::int32_t top_k, int threads, host::Isa isa);
    ~HostExperts();
    HostExperts(const HostExperts&)            = delete;
    HostExperts& operator=(const HostExperts&) = delete;

    [[nodiscard]] int threads() const noexcept { return moe_.threads(); }
    [[nodiscard]] host::Isa isa() const noexcept { return moe_.isa(); }
    [[nodiscard]] static constexpr int max_tokens() noexcept { return host::kHostMaxTokens; }

    // x: BF16 [H, T] on the device; ids: I32 [T * top_k]; y: BF16 [T * top_k, H] (assignment
    // order); slot_of: I32 [T * top_k], set to the identity (y's slot of assignment a is a).
    void enqueue_generic(const host::GenericLayer& layer, int tokens, const void* x,
                         const std::int32_t* ids, void* y, std::int32_t* slot_of,
                         cudaStream_t stream);
    // x: BF16 [H, T]; ids, weights: [T * top_k] rank order; out: BF16 [H, T].
    void enqueue_gemma4(const host::Gemma4Layer& layer, int tokens, const void* x,
                        const std::int32_t* ids, const float* weights, void* out,
                        cudaStream_t stream);

    struct Call; // one layer at one token count (host_experts.cpp)

private:
    std::uint32_t call_for(int kind, const void* key, int tokens, const host::GenericLayer* generic,
                           const host::Gemma4Layer* gemma4);
    void serve();
    void execute(const Call& call);

    host::HostMoe moe_;
    std::int32_t hidden_ = 0, top_k_ = 0;
    // Mapped (cudaHostAllocMapped) staging and mailbox, with their device aliases.
    std::uint16_t* x_host_       = nullptr;
    std::int32_t* ids_host_      = nullptr;
    float* weights_host_         = nullptr;
    std::uint16_t* out_host_     = nullptr;
    host::Mailbox* mailbox_      = nullptr;
    void* x_alias_               = nullptr;
    std::int32_t* ids_alias_     = nullptr;
    float* weights_alias_        = nullptr;
    void* out_alias_             = nullptr;
    host::Mailbox* mailbox_alias_ = nullptr;
    std::uint64_t* sequence_     = nullptr; // device: the post counter and the last posted
    // Calls by index (the mailbox names one); the server reads them under the mutex.
    std::mutex calls_mutex_;
    // Behind a pointer: MSVC's deque wants Call complete where the member is declared.
    std::unique_ptr<std::deque<Call>> calls_;
    std::vector<Call*> by_index_;
    std::map<std::tuple<int, const void*, int>, std::uint32_t> index_;
    std::atomic<bool> stop_{false};
    std::thread server_;
};

} // namespace ninferno::ops
