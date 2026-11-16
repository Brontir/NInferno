#pragma once

// A CTA's compiled resources and the serial chain of its K loop (#617 D), for the residency-round
// price of two schedules of one problem: a CTA's time on an SM it shares with the other resident
// CTAs is its own chain of K stages, so a grid takes ceil(ceil(CTAs / SMs) / resident) rounds of
// that chain. Counted, no constant: the resources are what cuobjdump --dump-resource-usage reports
// for the kernel (the most registers over its instances), the chain is read off the K loop.

#include <algorithm>
#include <cmath>
#include <cstddef>

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

// An SM's residency limits as the device query reports them.
struct SmLimits {
    int sms;
    int registers;            // registers an SM holds
    std::size_t shared_bytes; // shared memory an SM holds
    int threads;              // threads an SM holds
    int ctas;                 // CTAs an SM holds
};

// CTAs of `f` one SM holds at once: registers (allocated a warp at a time in units of 256), shared
// memory (with the 1 KB the driver reserves a CTA from sm_80, kept everywhere: it never adds a
// CTA where it is not reserved), threads and the CTA limit; at least 1.
[[nodiscard]] inline int resident_ctas(const CtaFacts& f, const SmLimits& sm) {
    const int warps          = (f.threads + 31) / 32;
    const int warp_registers = ((f.registers * 32 + 255) / 256) * 256;
    const int by_registers   = warps * warp_registers > 0 ? sm.registers / (warps * warp_registers) : 1;
    const int by_shared      = static_cast<int>(sm.shared_bytes / (f.shared_bytes + 1024));
    const int by_threads     = f.threads > 0 ? sm.threads / f.threads : 1;
    return std::max(1, std::min({by_registers, by_shared, by_threads, std::max(sm.ctas, 1)}));
}

// One CTA's serial chain in warp MMA steps of 16 x 16 x 64: its K stages, each a warp's tile of
// every set over the warp's K of the stage.
[[nodiscard]] inline double cta_chain_steps(const CtaFacts& f, double k) {
    return std::ceil(k / f.stage_k) * (f.warp_rows / 16.0) * (f.warp_tokens / 16.0) *
           (f.warp_k / 64.0) * f.sets;
}

// The launch's price: residency rounds of the busiest SM times one CTA's chain.
[[nodiscard]] inline double residency_chain(double ctas, const CtaFacts& f, const SmLimits& sm,
                                            double k) {
    const double rounds = std::ceil(std::ceil(ctas / sm.sms) / resident_ctas(f, sm));
    return rounds * cta_chain_steps(f, k);
}

} // namespace ninferno::ops::detail
