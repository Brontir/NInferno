#pragma once

// The CTAs of a kernel one SM holds at once, computed from facts (#612): the kernel's registers,
// static shared memory and threads, the launch's dynamic shared memory, and the device's limits
// (DeviceClass: threads, registers and shared memory per SM, the opt-in per block), by the CUDA
// occupancy calculator (cuda_occupancy.h, the toolkit's own algorithm, the one behind
// cudaOccupancyMaxActiveBlocksPerMultiprocessor). No GPU is asked: a substituted board answers as
// the real one would, and a host without a device answers at all. The kernel's facts come from
// the committed kernel-facts snapshot (ops/generic/kernel_facts.h), never from the host's kernels.

#include "core/device_class.h"

#include <cstddef>

namespace ninferno {

struct KernelShape {
    int registers                    = 0; // per thread
    std::size_t static_shared_bytes  = 0; // per CTA
    int threads                      = 0; // per CTA, the launch's
    std::size_t dynamic_shared_bytes = 0; // per CTA, the launch's
};

// 0 where the kernel does not fit one SM, a field is unknown (0), or the calculator does not know
// the architecture.
[[nodiscard]] int resident_ctas_per_sm(const DeviceClass& device, const KernelShape& kernel);

// The shared memory the driver reserves per resident CTA on the architecture (CUDA programming
// guide: 1 KiB from Ampere on).
[[nodiscard]] constexpr std::size_t reserved_shared_bytes_per_cta(int major) noexcept {
    return major >= 8 ? 1024 : 0;
}

} // namespace ninferno
