#include "core/device_class.h"

#include <stdexcept>

// The device of a host-only build (#620 step 11): there is none. device_class() answers a
// substituted class (ScopedDeviceClass) and otherwise reaches this, which throws the same
// std::runtime_error a machine whose CUDA cannot report a device does, so the selector's CPU-only
// paths (they catch it) behave as they do on such a machine. ninferno_core defines the real one.
namespace ninferno {

const DeviceClass& current_device_class() {
    throw std::runtime_error("device class: this build has no CUDA runtime");
}

} // namespace ninferno
