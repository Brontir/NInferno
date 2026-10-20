// Whether a device can load the kernels this build compiled: an empty kernel compiled with the
// rest of the tree either has an image (SASS or PTX) for the device's architecture or not.
#include "core/device_class.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace ninferno {
namespace {
__global__ void device_image_probe_kernel() {}
} // namespace

bool kernel_image_loadable(const void* kernel, int device) {
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess) {
        throw std::runtime_error("device image probe: cudaGetDevice failed");
    }
    if (previous != device && cudaSetDevice(device) != cudaSuccess) {
        throw std::runtime_error("device image probe: cudaSetDevice failed");
    }
    cudaFuncAttributes attributes{};
    const cudaError_t image = cudaFuncGetAttributes(&attributes, kernel);
    if (image != cudaSuccess) { (void)cudaGetLastError(); }
    if (previous != device && cudaSetDevice(previous) != cudaSuccess) {
        throw std::runtime_error("device image probe: cudaSetDevice failed");
    }
    if (image != cudaSuccess && image != cudaErrorNoKernelImageForDevice &&
        image != cudaErrorInvalidDeviceFunction) {
        throw std::runtime_error(std::string("device image probe: ") + cudaGetErrorName(image));
    }
    return image == cudaSuccess;
}

bool probe_device_kernel_image(int device) {
    return kernel_image_loadable(reinterpret_cast<const void*>(device_image_probe_kernel), device);
}

} // namespace ninferno
