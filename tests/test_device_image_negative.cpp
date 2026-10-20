// A binary without an image for the device must be refused at startup, by name, not fail later
// with an illegal instruction or "no kernel image" on the first decode.
//
// tests/device_image_sm90.cu is compiled for sm_90 SASS only (tests/cmake/CoreTests.cmake): it
// stands for a library of the build compiled for another architecture. First the control: the
// build as it is starts (a DeviceContext constructs). Then that library registers its kernel as a
// probe; the probe must report no image on a device that is not sm_90, require_device_support must
// name the library, and constructing a DeviceContext - the startup path every entry point takes -
// must throw before any kernel of the library is launched.
#include "core/device.h"
#include "core/device_class.h"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace ninferno::test {
const void* sm90_only_kernel();
}

namespace {

int expect(bool ok, const std::string& what) {
    std::cout << (ok ? "ok   " : "FAIL ") << what << '\n';
    return ok ? 0 : 1;
}

constexpr const char* kLibrary = "the sm_90-only test library";

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    const auto& device = ninferno::device_class(0);
    if (device.major == 9 && device.minor == 0) {
        std::cout << "SKIP: the device is sm_90, the architecture the test library was built for\n";
        return 77;
    }
    int failures = 0;
    try {
        ninferno::DeviceContext context;
        failures += expect(true, "control: the build starts on sm_" + std::to_string(device.major) +
                                     std::to_string(device.minor));
    } catch (const std::exception& error) {
        failures += expect(false, std::string("control: the build starts: ") + error.what());
    }

    const void* kernel = ninferno::test::sm90_only_kernel();
    failures += expect(!ninferno::kernel_image_loadable(kernel, device.device),
                       "the sm_90-only kernel has no image for this device");
    ninferno::register_kernel_image_probe(kLibrary, kernel);
    std::string message;
    try {
        ninferno::require_device_support(device, {}, "NInferno");
    } catch (const std::runtime_error& error) { message = error.what(); }
    failures += expect(message.find(kLibrary) != std::string::npos,
                       "require_device_support refuses by name: " + message);
    message.clear();
    try {
        ninferno::DeviceContext context;
    } catch (const std::exception& error) { message = error.what(); }
    failures += expect(message.find(kLibrary) != std::string::npos,
                       "DeviceContext refuses at startup: " + message);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " device image negative\n";
    return failures == 0 ? 0 : 1;
}
