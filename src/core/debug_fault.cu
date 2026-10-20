#include "core/debug_poison.h"

#include "core/cuda_error.h"
#include "core/sys.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

namespace ninferno {
namespace {

__global__ void debug_trap_kernel() { __trap(); }

} // namespace

void debug_device_fault_point() noexcept {
    static const std::string path = sys::env("NINFERNO_DEBUG_TRAP_FILE");
    if (path.empty()) { return; }
    static std::atomic<bool> fired{false};
    std::error_code error;
    if (fired.load() || !std::filesystem::exists(path, error)) { return; }
    fired.store(true);
    // A device-side trap: the context is poisoned, so the executor's next checked CUDA call fails
    // the way a kernel's device-side assert makes it fail.
    debug_trap_kernel<<<1, 1>>>();
    (void)cudaDeviceSynchronize();
}

void debug_transient_cuda_error_point() {
    static const std::string path = sys::env("NINFERNO_DEBUG_CUDA_ERROR_FILE");
    if (path.empty()) { return; }
    std::error_code error;
    if (!std::filesystem::remove(path, error)) { return; }
    throw CudaError(cudaErrorMemoryAllocation,
                    "CUDA call failed: cudaErrorMemoryAllocation (NINFERNO_DEBUG_CUDA_ERROR_FILE)");
}

} // namespace ninferno
