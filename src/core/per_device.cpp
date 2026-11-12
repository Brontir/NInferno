#include "core/per_device.h"

#include <stdexcept>

namespace ninferno {

int current_device_ordinal() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxDevices) {
        throw std::out_of_range("device ordinal past the supported 64 devices");
    }
    return device;
}

int current_device_ordinal_or_none() noexcept {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kMaxDevices) {
        (void)cudaGetLastError();
        return -1;
    }
    return device;
}

} // namespace ninferno
