#pragma once

// CudaError alone, without core/device.h's CUDA_CHECK macro, for units that must not see it (the
// ggml archive defines its own).

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace ninferno {

// A failed CUDA call. Most such failures (a device-side fault, an illegal address, a lost device)
// leave the context unusable, so the Engine that catches one fails all its requests as Unavailable
// and the server exits for its supervisor to restart it.
class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    [[nodiscard]] cudaError_t code() const noexcept { return code_; }

private:
    cudaError_t code_;
};

// What to do about a driver older than the runtime this build links: the CUDA version the
// installed driver supports, the one the build needs, and the driver branch that brings it.
[[nodiscard]] inline std::string driver_update_hint() {
    int driver = 0;
    (void)cudaDriverGetVersion(&driver);
    const int major = CUDART_VERSION / 1000, minor = (CUDART_VERSION % 1000) / 10;
    std::string hint = "; the installed NVIDIA driver ";
    hint += driver > 0 ? "supports CUDA up to " + std::to_string(driver / 1000) + "." +
                             std::to_string((driver % 1000) / 10)
                       : std::string("was not found or did not answer");
    hint += ", this NInferno build uses CUDA " + std::to_string(major) + "." +
            std::to_string(minor) + ": install NVIDIA driver " + (major >= 13 ? "580" : "570") +
            " or newer (nvidia.com/drivers)";
    return hint;
}

// "<prefix>: <error name>: <error text>", and for a driver too old for this build what to install.
[[nodiscard]] inline std::string cuda_error_message(const char* prefix, cudaError_t err) {
    std::string message =
        std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
    if (err == cudaErrorInsufficientDriver || err == cudaErrorNoDevice ||
        err == cudaErrorCompatNotSupportedOnDevice) {
        message += driver_update_hint();
    }
    return message;
}

// Reports a failed cleanup call (a destructor's free or destroy) on stderr; it cannot throw.
inline void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

} // namespace ninferno
