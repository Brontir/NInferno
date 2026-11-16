#pragma once

// A CTA's compiled resources and the serial chain of its K loop (#617 D), for the residency-round
// price of two schedules of one problem: a CTA's time on an SM it shares with the other resident
// CTAs is its own chain of K stages, so a grid takes ceil(ceil(CTAs / SMs) / resident) rounds of
// that chain. Counted, no constant: the resources are what cuobjdump --dump-resource-usage reports
// for the kernel in the sm_89 build (the most registers over its instances; other builds may
// differ by a few registers, #949), the chain is read off the K loop.

#include "core/device_class.h"
#include "core/kernel_occupancy.h"

#include <cmath>
#include <cstddef>
#include <optional>

namespace ninferno::ops::detail {

struct CtaFacts {
    int threads;      // a CTA's threads
    int registers;    // a thread's registers (the most of the kernel's instances)
    int shared_bytes; // a CTA's static shared memory
    int stage_k;      // K a CTA advances in one stage of its loop (one barrier step)
    int warp_k;       // K one warp multiplies in a stage (stage_k / the K-sliced warps)
    int warp_rows;    // weight rows of a warp's MMA tile
    int warp_tokens;  // token columns of a warp's MMA tile
    int sets;         // weights a warp multiplies in a stage (2: the fused K/V pair)
};

// One CTA's serial chain in warp MMA steps of 16 x 16 x 64: its K stages, each a warp's tile of
// every set over the warp's K of the stage.
[[nodiscard]] inline double cta_chain_steps(const CtaFacts& f, double k) {
    return std::ceil(k / f.stage_k) * (f.warp_rows / 16.0) * (f.warp_tokens / 16.0) *
           (f.warp_k / 64.0) * f.sets;
}

// The launch's price on `device`: residency rounds of the busiest SM times one CTA's chain. The
// CTAs an SM holds come from the CUDA occupancy calculator (core/kernel_occupancy.h) on the
// kernel's facts and the device's limits; nullopt where it cannot answer (a limit unknown, the
// kernel too large for an SM).
[[nodiscard]] inline std::optional<double> residency_chain(double ctas, const CtaFacts& f,
                                                           const DeviceClass& device, double k) {
    const int resident = resident_ctas_per_sm(
        device, KernelShape{f.registers, static_cast<std::size_t>(f.shared_bytes), f.threads, 0});
    if (resident <= 0 || device.sms <= 0) { return std::nullopt; }
    const double rounds = std::ceil(std::ceil(ctas / device.sms) / resident);
    return rounds * cta_chain_steps(f, k);
}

} // namespace ninferno::ops::detail
