// Link-order check for ninferno_ops: this executable links nothing but ninferno_ops (CMake adds the
// archives it depends on). Its only reference into the selector is kernel_facts::sass_arch(), which
// lives in ninferno_ops_host and calls the inline device_class(); the reference to
// current_device_class() (ninferno_core) therefore appears only while that host archive is scanned,
// and the link fails unless ninferno_core comes after it. Nothing here needs a GPU: where CUDA
// cannot report a device sass_arch() throws or returns nothing, and both are fine.

#include "ops/generic/kernel_facts.h"

#include <exception>

int main() {
    try {
        (void)ninferno::ops::detail::generic::kernel_facts::sass_arch();
    } catch (const std::exception&) {}
    return 0;
}
