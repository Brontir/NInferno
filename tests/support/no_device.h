#pragma once

// A host test that must read no GPU (#615 W4): every CUDA device hidden from this process before
// the runtime starts (CUDA_VISIBLE_DEVICES=-1: an invalid first index hides every device), so a
// device read that a substituted class (core/device_class.h, ScopedDeviceClass) does not answer
// throws, as it does on a machine with no GPU and in a decisions-only build
// (core/device_class_nodevice.cpp), instead of reading whatever card this machine has. Call it
// first in main, before any CUDA call: the runtime reads the variable once, when it starts.

#include "core/device_class.h"
#include "core/per_device.h"
#include "support/posix.h"

#include <exception>

// Whether the devices are hidden: no current device (PerDevice keeps a substitute's value under
// ordinal -1) and an unsubstituted device_class() throws. False where the CUDA runtime had already
// started with a device (a static initializer that called it): run the test with
// CUDA_VISIBLE_DEVICES=-1 then.
inline bool hide_cuda_devices() {
    setenv("CUDA_VISIBLE_DEVICES", "-1", 1);
    if (ninferno::current_device_ordinal_or_none() != -1) { return false; }
    try {
        (void)ninferno::device_class();
    } catch (const std::exception&) { return true; }
    return false;
}
