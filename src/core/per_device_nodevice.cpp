#include "core/per_device.h"

#include <stdexcept>

// The current device of a host-only build (#620 step 11): there is none. PerDevice reads the ordinal
// only through these two calls; a substituted class (ScopedDeviceClass) is answered under ordinal -1,
// and an unsubstituted read throws, as a machine whose CUDA cannot report a device does. ninferno_core
// defines the real ones (per_device.cpp); a host-only test adds this file to its own sources, and
// nothing beside it links this.
namespace ninferno {

int current_device_ordinal() {
    throw std::runtime_error("per device: this build has no CUDA runtime");
}

int current_device_ordinal_or_none() noexcept { return -1; }

} // namespace ninferno
