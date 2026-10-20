#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "support/cuda.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {

int expect_value(void* device, std::uint32_t expected, const char* label) {
    std::uint32_t actual  = 0;
    const cudaError_t err = cudaMemcpy(&actual, device, sizeof(actual), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " copy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    if (actual == expected) { return 0; }
    std::cerr << label << " expected 0x" << std::hex << expected << ", got 0x" << actual << std::dec
              << '\n';
    return 1;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    try {
        ninferno::DeviceContext device(0);
        ninferno::DeviceArena storage(sizeof(std::uint32_t));
        CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x33, sizeof(std::uint32_t), device.stream));
        device.synchronize();

        ninferno::DecodeGraphDefinition first;
        first.capture(device.stream, [&] {
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x11, sizeof(std::uint32_t), device.stream));
        });
        ninferno::DecodeGraphDefinition second;
        second.capture(device.stream, [&] {
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x22, sizeof(std::uint32_t), device.stream));
        });

        int failures = 0;
        ninferno::DecodeGraphExecutable executable;
        executable.instantiate(first);
        executable.upload(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x33333333U, "initial graph upload");

        executable.launch(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x11111111U, "first graph launch");

        executable.update(second);
        executable.upload(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x11111111U, "updated graph upload");

        executable.launch(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x22222222U, "updated graph launch");

        // Another topology (two nodes): CUDA rejects the update. try_update() says so and update()
        // throws, neither leaving the rejection as the pending CUDA error (a later launch's
        // CUDA_CHECK(cudaGetLastError()) would report it as that launch's device failure), and the
        // executable takes a matching definition again.
        ninferno::DecodeGraphDefinition other_topology;
        other_topology.capture(device.stream, [&] {
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x44, sizeof(std::uint32_t), device.stream));
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x55, sizeof(std::uint32_t), device.stream));
        });
        if (executable.try_update(other_topology)) {
            std::cerr << "try_update accepted a definition of another topology\n";
            ++failures;
        }
        cudaError_t pending = cudaGetLastError();
        if (pending != cudaSuccess) {
            std::cerr << "a rejected try_update left " << cudaGetErrorName(pending) << " pending\n";
            ++failures;
        }
        bool threw = false;
        try {
            executable.update(other_topology);
        } catch (const ninferno::CudaError&) {
            // A rejected definition is the caller's, not the device's: no 503 and restart for it.
            std::cerr << "a rejected update was reported as a device failure\n";
            ++failures;
            threw = true;
        } catch (const std::runtime_error&) { threw = true; }
        if (!threw) {
            std::cerr << "update accepted a definition of another topology\n";
            ++failures;
        }
        pending = cudaGetLastError();
        if (pending != cudaSuccess) {
            std::cerr << "a rejected update left " << cudaGetErrorName(pending) << " pending\n";
            ++failures;
        }
        if (!executable.try_update(first)) {
            std::cerr << "try_update rejected a definition of the executable's topology\n";
            ++failures;
        }
        executable.launch(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x11111111U, "graph launch after a rejection");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "decode graph test failed: " << error.what() << '\n';
        return 1;
    }
}
