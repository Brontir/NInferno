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
//   dram   = dram_terms(C, c, W, CTA bytes, the warp's counted bytes in flight)   (#690)
//   issue  = q x I_cta x cycles / (4 schedulers x clock),
//            I_cta = W x (passes x prologue + iterations x loop + outside)
//   chain  = rounds x (passes x prologue stall + iterations x loop stall + outside stall) / clock
//   t      = max(dram, issue, chain)
// passes = ceil(NT x K / 32 / (32 W)) is how often each thread runs the quantizer loop before the
// CTA barrier; iterations = ceil(K / k_iter) the dp4a loop's, k_iter the K values one iteration
// covers per row and token. The terms every schedule shares (the launch, the first byte, the bus
// bound of the whole call) move no choice between schedules and are left to the caller.
// Where a fact is missing (no r8 row, no loop counts, no SM rate row) the law gives no answer, and
// the rules answer (rules_r8).

#include "core/device_class.h"
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
};

struct Schedule {
    int warps = 4;
    int rows  = 1;
};

struct Terms {
    double ctas       = 0;
    double occupancy  = 0; // c
    double waves      = 0; // q
    double rounds     = 0;
    double passes     = 0; // quantizer passes a thread runs before the barrier
    double iterations = 0; // dp4a loop iterations a warp runs
    double cta_bytes  = 0;
    double dram       = 0; // seconds: dram_terms' max(bus, busiest SM)
    double issue      = 0; // seconds: the busiest SM's issue, the INT32 and FP32 pipes
    double chain      = 0; // seconds: one warp's static schedule, round after round
    double total      = 0; // max(dram, issue, chain)
};

// The K values one quantizer pass covers: one 32-value block a thread.
inline constexpr double kValuesPerTask = 32.0;

// The law's terms, or nullopt where a fact it reads is missing.
[[nodiscard]] inline std::optional<Terms> terms(const ChipFacts& chip, const DeviceClass& device,
                                                const kernel_facts::KernelFacts& f,
                                                const Call& call, const Schedule& s) {
    if (!f.loop || !f.prologue || f.loop->loops != 1 || f.loop->k_iter <= 0 || legacy_dram(chip) ||
        chip.sms <= 0 || chip.clock_hz <= 0 || call.n <= 0 || call.k <= 0 || s.warps <= 0 ||
        s.rows <= 0 || !(call.weight_bytes_per_value > 0)) {
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

    Terms t;
    const double rows_per_cta = static_cast<double>(s.warps) * s.rows;
    const double sets         = call.swiglu ? 2.0 : 1.0;
    t.ctas                    = std::ceil(static_cast<double>(call.n) / rows_per_cta);
    t.occupancy               = resident;
    t.waves                   = std::ceil(t.ctas / chip.sms);
    t.rounds                  = std::ceil(t.waves / t.occupancy);
    t.passes     = p.loops > 0 ? std::ceil(call.nt * (call.k / kValuesPerTask) / threads) : 0.0;
    t.iterations = std::ceil(static_cast<double>(call.k) / l.k_iter);
    t.cta_bytes  = rows_per_cta * sets * static_cast<double>(call.k) * call.weight_bytes_per_value;
    t.dram = dram_terms(chip, t.ctas, t.occupancy, s.warps, t.cta_bytes, l.ldg_bytes, true).total;

    // Issue: every warp instruction over the four schedulers, or the INT32 / FP32 pipe where it
    // takes longer (rmma_counted_terms.h's ISA facts).
    const double outside           = l.schedule ? l.schedule->outside_total : 0.0;
    const double warp_instructions = t.passes * p.total + t.iterations * l.total + outside;
    const double warp_int          = t.passes * p.int_alu + t.iterations * l.int_alu;
    const double warp_fp           = t.passes * p.fp_alu + t.iterations * l.fp_alu;
    const double cycles            = std::max(
        {warp_instructions, warp_int * rmma_counted::int32_cycles_per_warp_instruction(chip.major),
                    warp_fp * rmma_counted::fp32_cycles_per_warp_instruction(chip.major)});
    t.issue = t.waves * s.warps * cycles / (rmma_counted::kSchedulersPerSm * chip.clock_hz);

    // Chain: one warp alone issues its static schedule; a round lasts at least that long.
    if (l.schedule) {
        const double warp_clocks = t.passes * p.stall_sum + t.iterations * l.schedule->stall_sum +
                                   l.schedule->outside_stall;
        t.chain = t.rounds * warp_clocks / chip.clock_hz;
    }
    t.total = std::max({t.dram, t.issue, t.chain});
    return t;
}

} // namespace ninferno::ops::detail::generic::r8_law
