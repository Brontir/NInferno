#include "core/device_traits.h"

#include <cuda_runtime.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

namespace ninferno {

bool paravirtualized_gpu(int device) {
#ifdef _WIN32
    int tcc = 0;
    return cudaDeviceGetAttribute(&tcc, cudaDevAttrTccDriver, device) != cudaSuccess || tcc == 0;
#else
    (void)device;
    static const bool present = [] {
        std::error_code error;
        return std::filesystem::exists("/dev/dxg", error);
    }();
    return present;
#endif
}

bool drives_display(int device) {
    int watchdog = 0;
    if (const cudaError_t status =
            cudaDeviceGetAttribute(&watchdog, cudaDevAttrKernelExecTimeout, device);
        status != cudaSuccess) {
        throw std::runtime_error("GPU " + std::to_string(device) +
                                 ": cudaDevAttrKernelExecTimeout: " + cudaGetErrorString(status));
    }
    return watchdog != 0;
}

} // namespace ninferno
