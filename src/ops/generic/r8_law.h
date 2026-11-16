#pragma once

// The R8 decode schedule law (#613 PR-10, step 10b): the time of one r8_kernel call
// (generic_r8.cuh) under a schedule of W warps of R rows, counted from the kernel's facts (the r8
// section of the kernel-facts snapshot: registers, the dp4a loop's counts and K per iteration,
// the activation quantizer's loop, the pro_ columns) and the chip's published facts. No CUDA, no
// measurement, no fitted constant: the predictions it makes are frozen before the lab's grids are
// scored (#613, the PR-10 plan).
//
//   C      = ceil(N / (W x R))                     CTAs
//   c      = resident CTAs an SM holds (registers, the activation's shared bytes, threads)
//   q      = ceil(C / SMs), rounds = ceil(q / c)   on the busiest SM
//   dram   = the busiest SM's residency rounds of dram_terms(C, c, W, CTA bytes, a warp's weight
//            bytes in flight) (#690)
//   issue  = q x I_cta x cycles / (4 schedulers x clock),
//            I_cta = warp passes x prologue + W x (iterations x loop + the rest outside it)
//   chain  = rounds x (passes x prologue stall + iterations x loop stall + the rest
//            + the waits on global loads x their latency: the loop's at DRAM's L0, the
//            quantizer's and the rest's at an L2 hit) / clock
//   rank   = max(dram, issue, chain): what schedules are compared on
//   total  = max(bus, rank), bus = every CTA's bytes over the chip's bandwidth (the same for
//            every schedule)
// The quantizer loop (generic_r8.cuh: `for task = tid; task < NT x blocks; task += threads`) runs
// ceil(NT x K / 32 / 32) warp passes a CTA and ceil(NT x K / 32 / (32 W)) passes a thread before
// the CTA barrier. The dp4a loop takes `slots` groups an iteration (the two-slot ring; one slot
// on Volta, NINFERNO_R8_ONE_SLOT) and stops after the first slot of the last one, so a warp runs
// ceil(slots x K / k_iter) / slots iterations, k_iter the K values a whole iteration covers per
// row and token. The snapshot's outside counts hold the quantizer loop once; the law takes it out
// of them. A warp keeps one iteration's weight bytes in flight; its other loads (codebook
// lookups, re-read headers) hit the cache. A kernel that spills (stack or local bytes) has no
// term here: the law gives no answer for it.
// Where a fact is missing (no r8 row, no loop counts, no SM rate row) the law gives no answer, and
// the rules answer (rules_r8).

#include "core/device_class.h"
#include "core/isa_facts.h"
#include "core/kernel_occupancy.h"
#include "ops/generic/kernel_facts.h"
#include "ops/generic/kernel_roofline.h"
#include "ops/generic/rmma_counted_terms.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

namespace ninferno::ops::detail::generic::r8_law {

using kernel_roofline::ChipFacts;
using kernel_roofline::dram_terms;
using kernel_roofline::legacy_dram;

// One call as the law reads it.
struct Call {
    std::int64_t n          = 0;     // output rows (SwiGLU: the gate rows)
    std::int64_t k          = 0;     // K values
    int nt                  = 1;     // the kernel's token slots (1, 2, 4, 8)
    bool swiglu             = false; // two row sets a warp: gate and up
    bool sums               = false; // a codec with a min: 40 activation bytes a block, not 36
    std::int64_t grid_bytes = 0; // a codebook grid staged in shared memory (below sm_80), with pad
    double weight_bytes_per_value = 0; // DRAM bytes of the weight per row and K value
    int slots                     = 2; // the dp4a ring's slots: slots_of(device)
};

// The R8 kernel's ring slots: one on sm_70 (NINFERNO_R8_ONE_SLOT), two elsewhere.
[[nodiscard]] inline int slots_of(const DeviceClass& d) {
    return d.major == 7 && d.minor == 0 ? 1 : 2;
}

struct Schedule {
    int warps = 4;
    int rows  = 1;
};

struct Terms {
    double ctas           = 0;
    double occupancy      = 0; // c
    double waves          = 0; // q
    double rounds         = 0;
    double passes         = 0; // quantizer passes a thread runs before the barrier
    double warp_passes    = 0; // quantizer warp passes a CTA issues
    double iterations     = 0; // dp4a loop iterations a warp runs
    double cta_bytes      = 0;
    double bus            = 0; // seconds: every CTA's bytes over the chip's bandwidth
    double dram           = 0; // seconds: the busiest SM's residency rounds at its own rate
    double issue          = 0; // seconds: the busiest SM's issue, the INT32 and FP32 pipes
    double exposed_clocks = 0; // a warp's waits on global loads at their latency, SM clocks
    double chain          = 0; // seconds: one warp's static schedule and waits, round after round
    double rank           = 0; // max(dram, issue, chain): what schedules are compared on
    double total          = 0; // max(bus, rank)
};

// The K values one quantizer pass covers: one 32-value block a thread.
inline constexpr double kValuesPerTask = 32.0;

// The law's terms, or nullopt where a fact it reads is missing.
[[nodiscard]] inline std::optional<Terms> terms(const ChipFacts& chip, const DeviceClass& device,
                                                const kernel_facts::KernelFacts& f,
                                                const Call& call, const Schedule& s) {
    if (!f.loop || !f.prologue || f.loop->loops != 1 || f.loop->k_iter <= 0 || legacy_dram(chip) ||
        chip.sms <= 0 || chip.clock_hz <= 0 || call.n <= 0 || call.k <= 0 || s.warps <= 0 ||
        s.rows <= 0 || call.slots <= 0 || !(call.weight_bytes_per_value > 0) ||
        f.spill_bytes() > 0 || f.loop->local > 0) {
        return std::nullopt;
    }
    const kernel_facts::LoopCounts& l     = *f.loop;
    const kernel_facts::PrologueCounts& p = *f.prologue;
    const int threads                     = 32 * s.warps;
    KernelShape shape;
    shape.registers           = f.registers;
    shape.static_shared_bytes = f.static_shared_bytes;
    shape.threads             = threads;
    shape.dynamic_shared_bytes =
        static_cast<std::size_t>(call.nt * (call.k / 32) * (call.sums ? 40 : 36) + call.grid_bytes);
    const int resident = resident_ctas_per_sm(device, shape);
    if (resident <= 0) return std::nullopt;
    // The L2 hit's latency of one warp alone (the ISA table, in SM clocks).
    const auto isa = isa_row(device.major, device.minor);
    if (!isa || !isa->l2_latency.w1) return std::nullopt;
    const double l2_clocks = *isa->l2_latency.w1;

    Terms t;
    const double rows_per_cta = static_cast<double>(s.warps) * s.rows;
    const double sets         = call.swiglu ? 2.0 : 1.0;
    const double tasks        = call.nt * std::ceil(static_cast<double>(call.k) / kValuesPerTask);
    t.ctas                    = std::ceil(static_cast<double>(call.n) / rows_per_cta);
    t.occupancy               = resident;
    t.waves                   = std::ceil(t.ctas / chip.sms);
    t.rounds                  = std::ceil(t.waves / t.occupancy);
    t.passes                  = p.loops > 0 ? std::ceil(tasks / threads) : 0.0;
    t.warp_passes             = p.loops > 0 ? std::ceil(tasks / 32.0) : 0.0;
    t.iterations = std::ceil(call.slots * static_cast<double>(call.k) / l.k_iter) / call.slots;
    t.cta_bytes  = rows_per_cta * sets * static_cast<double>(call.k) * call.weight_bytes_per_value;
    // A warp's weight bytes in flight: one iteration's, at most what its loads request.
    const double iteration_bytes =
        static_cast<double>(l.k_iter) * s.rows * sets * call.weight_bytes_per_value;
    const double in_flight = std::min(static_cast<double>(l.ldg_bytes), iteration_bytes);
    const auto d = dram_terms(chip, t.ctas, t.occupancy, s.warps, t.cta_bytes, in_flight, true);
    t.bus        = d.bus;
    t.dram       = d.sm;

    // Issue: every warp instruction over the four schedulers, or the INT32 / FP32 pipe where it
    // takes longer (rmma_counted_terms.h's ISA facts). The outside counts hold the quantizer loop
    // once: the rest is outside minus it.
    const double quantizer = static_cast<double>(p.loops);
    const double outside =
        l.schedule ? std::max(0.0, l.schedule->outside_total - quantizer * p.total) : 0.0;
    const double cta_instructions =
        t.warp_passes * p.total + s.warps * (t.iterations * l.total + outside);
    const double cta_int = t.warp_passes * p.int_alu + s.warps * t.iterations * l.int_alu;
    const double cta_fp  = t.warp_passes * p.fp_alu + s.warps * t.iterations * l.fp_alu;
    const double cycles  = std::max(
        {cta_instructions, cta_int * rmma_counted::int32_cycles_per_warp_instruction(chip.major),
          cta_fp * rmma_counted::fp32_cycles_per_warp_instruction(chip.major)});
    t.issue = t.waves * cycles / (rmma_counted::kSchedulersPerSm * chip.clock_hz);

    // Chain: one warp alone issues its static schedule, and every wait on a global load exposes
    // that load's latency: the dp4a loop's (the weights) a DRAM fetch (L0), the quantizer's and
    // the code's outside the loops (the activation, which every CTA reads) an L2 hit. A round
    // lasts at least that long; the CTAs resident at once share it (rounds counts them).
    if (l.schedule) {
        const double outside_stall =
            std::max(0.0, l.schedule->outside_stall - quantizer * p.stall_sum);
        const double warp_clocks =
            t.passes * p.stall_sum + t.iterations * l.schedule->stall_sum + outside_stall;
        const double dram_clocks = *chip.dram_latency_s * chip.clock_hz;
        t.exposed_clocks         = t.iterations * p.loop_global_waits * dram_clocks +
                           (t.passes * p.global_waits + p.outside_global_waits) * l2_clocks;
        t.chain = t.rounds * (warp_clocks + t.exposed_clocks) / chip.clock_hz;
    }
    t.rank  = std::max({t.dram, t.issue, t.chain});
    t.total = std::max(t.bus, t.rank);
    return t;
}

} // namespace ninferno::ops::detail::generic::r8_law
