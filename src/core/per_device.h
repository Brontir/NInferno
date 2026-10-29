#pragma once

// Per-device state for a process that drives several GPUs: a value computed once for each device
// ordinal, and a kernel attribute set once on each device. A kernel's attributes, a device's
// facts and a table measured for a GPU belong to that device, so a function-local `static` that
// caches one of them answers every GPU with whichever launched first (a second GPU then launches
// an over-48-KiB kernel without its opt-in and fails with an invalid argument).

#include "core/device.h"
#include "core/device_class.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace ninferno {

inline constexpr int kMaxDevices = 64;

// The calling thread's current CUDA device.
inline int current_device_ordinal() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxDevices) {
        throw std::out_of_range("device ordinal past the supported 64 devices");
    }
    return device;
}

// One T per device, made on first use by make(device). A make that throws leaves the slot unset,
// so the next call retries: a failure is never cached. While this thread's device_class() answers
// a substitute (ScopedDeviceClass), the value is the substitute's, kept per substitute content
// (device_class_key) and ordinal apart from the ordinal's own: what a cache read of one board
// never answers another. Without a current CUDA device (a host-only process), the substitute's
// ordinal is -1.
template <class T>
class PerDevice {
public:
    template <class Make>
    const T& get(Make&& make) {
        if (const std::string* substitute = substituted_device_class_key()) {
            int device = -1;
            if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kMaxDevices) {
                (void)cudaGetLastError();
                device = -1;
            }
            const std::string key = std::to_string(device) + '|' + *substitute;
            std::lock_guard lock(substituted_mutex_);
            auto it = substituted_.find(key);
            if (it == substituted_.end()) {
                it = substituted_.emplace(key, make(device < 0 ? 0 : device)).first;
            }
            return it->second;
        }
        Slot& slot = slots_[static_cast<std::size_t>(current_device_ordinal())];
        std::call_once(slot.once, [&] { slot.value = make(current_device_ordinal()); });
        return slot.value;
    }

private:
    struct Slot {
        std::once_flag once;
        T value{};
    };
    std::array<Slot, kMaxDevices> slots_;
    std::mutex substituted_mutex_;
    std::map<std::string, T> substituted_; // node-based: references stay valid
};

// The dynamic shared-memory opt-in of `kernel`, once per device. The caller owns the slot (one
// `static PerDevice<bool>` per launch site): kernels of one signature share a function type, so a
// slot keyed on the type would serve whichever instantiation launched first.
template <class Kernel>
void allow_dynamic_smem(PerDevice<bool>& slot, Kernel kernel, std::size_t bytes) {
    (void)slot.get([&](int) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(bytes)));
        return true;
    });
}

} // namespace ninferno
