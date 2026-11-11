// sass_arch(): the kernel-facts label of the device this thread runs on. The part that reads no
// device is sass_arch_of (kernel_facts.cpp, ninferno_ops_host); this file adds the device's class,
// current_device_class() of ninferno_core, so it is built into ninferno_ops and ninferno_ops_host
// keeps no reference into ninferno_core.

#include "core/device_class.h"
#include "ops/generic/kernel_facts.h"

#include <exception>

namespace ninferno::ops::detail::generic::kernel_facts {

std::optional<std::string> sass_arch() {
    if (const DeviceClass* s = substituted_device_class()) { return sass_arch_of(*s, true); }
    try {
        return sass_arch_of(device_class(), false);
    } catch (const std::exception&) {} // no current device: no kernels run
    return std::nullopt;
}

} // namespace ninferno::ops::detail::generic::kernel_facts
