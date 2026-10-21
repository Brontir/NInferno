#pragma once

#include "core/cuda_error.h"
#include "core/device_class.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninferno {

// The CUDA synchronization schedule (ninferno/option_enums.h, --cuda-sync), declared opaque: this
// header is also read by targets that see src/ alone (ops/generic/ggml_mmq), not include/.
enum class CudaSyncSchedule : std::uint8_t;

// Throws CudaError when err is not cudaSuccess (after printing the call and its place to stderr).
void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

// Called inside a catch block: rethrows the exception being handled with `context` in front of
// its message. A CudaError stays a CudaError, so the Engine still answers it as a device failure;
// any other std::exception becomes a std::runtime_error.
[[noreturn]] void rethrow_with_context(const std::string& context);

#define CUDA_CHECK(expr) ::ninferno::cuda_check((expr), #expr, __FILE__, __LINE__)

// Copies destination.bytes() bytes of host memory at `source` into the contiguous device tensor
// `destination` on `stream`; nothing for an empty tensor.
inline void copy_from_host(const void* source, const Tensor& destination, cudaStream_t stream) {
    if (destination.bytes() == 0) { return; }
    if (source == nullptr || destination.data == nullptr || !destination.is_contiguous()) {
        throw std::invalid_argument(
            "copy_from_host: needs host memory and a contiguous device tensor");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}

// Non-owning execution facts passed to Ops whose launch policy depends on physical device
// capacity. DeviceContext remains the owner and authoritative source of both values.
struct DeviceExecutionView {
    cudaStream_t stream               = nullptr;
    std::int32_t multiprocessor_count = 0;

    [[nodiscard]] constexpr DeviceExecutionView
    on_stream(cudaStream_t target_stream) const noexcept {
        return {target_stream, multiprocessor_count};
    }
};

struct DeviceContext {
    int device                   = 0;
    cudaStream_t stream          = nullptr;
    cudaStream_t transfer_stream = nullptr;
    cudaDeviceProp props{};

    // The device under the spin synchronization schedule, or under `sync`
    // (EngineOptions::pacing.cuda_sync).
    explicit DeviceContext(int device_id = 0);
    DeviceContext(int device_id, CudaSyncSchedule sync);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    void bind_to_current_thread() const;
    void bind_to_current_thread_noexcept() const noexcept;
    int compute_capability() const noexcept;
    int multiprocessor_count() const noexcept;
    // The cached class of this context's device (core/device_class.h).
    const DeviceClass& device_class() const;
    DeviceExecutionView execution_view() const noexcept;
    std::size_t total_vram() const noexcept;
    const char* sync_mode() const;
    void synchronize() const;
};

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    // `blocking_stop`: a thread that waits for the stop event blocks (cudaEventBlockingSync)
    // instead of following the device's synchronization schedule.
    CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream, bool blocking_stop = false);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();
    // Waits until the recorded stop event has completed.
    void wait_stop() const;

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
};

// Reusable non-timing event for worker-driven asynchronous control transactions. The owning
// component records it after enqueueing one transfer batch and polls it from later boundaries.
class CudaCompletionEvent {
public:
    explicit CudaCompletionEvent(const DeviceContext& ctx);
    ~CudaCompletionEvent();

    CudaCompletionEvent(const CudaCompletionEvent&)            = delete;
    CudaCompletionEvent& operator=(const CudaCompletionEvent&) = delete;
    CudaCompletionEvent(CudaCompletionEvent&& other) noexcept;
    CudaCompletionEvent& operator=(CudaCompletionEvent&& other) noexcept;

    void record(cudaStream_t stream);
    void wait(cudaStream_t stream) const;
    [[nodiscard]] bool ready() const;
    void synchronize() const;

private:
    int device_        = 0;
    cudaEvent_t event_ = nullptr;
};

// A non-blocking stream a component owns for a parallel branch of its own work (captured into a
// graph by fork/join events); destroyed with its owner.
class CudaStream {
public:
    explicit CudaStream(const DeviceContext& ctx);
    ~CudaStream();

    CudaStream(const CudaStream&)            = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

} // namespace ninferno
