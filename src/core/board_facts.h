#pragma once

// A board's per-SM quantities, derived from its published facts (#612 step 7): constexpr functions
// over the rows, never stored, so a neighbouring board of the same architecture interpolates by its
// own SMs, clocks, bus and L2. No CUDA; host and device code alike.
//
// Checked against the profiler's rated widths (ncu peak_sustained, #612's comments): the L2's
// slices x 64 B an L2 clock (arch_registry l2_bytes_per_l2_clock, six Ada and GB20x cards), and the
// DRAM's bus bytes a memory clock (dram__bytes.sum.peak_sustained: 96 B on the 384-bit RTX 4090 and
// RTX 3090, 128 B on the 512-bit RTX 5090, 48 B on the 192-bit RTX 3060: bus bits / 4 at the
// half-rate clock the table's data rate gives).

#include "core/arch_registry.h"

namespace ninferno::board_facts {

// DRAM bytes a board moves a second: its bus width times its data rate (Gbit/s per pin).
[[nodiscard]] constexpr double dram_bytes_per_s(int bus_bits, double gbps) noexcept {
    return bus_bits > 0 && gbps > 0 ? bus_bits / 8.0 * gbps * 1e9 : 0;
}

// The DRAM bytes one SM may draw an SM clock: the board's DRAM rate over its SMs at the clock.
[[nodiscard]] constexpr double dram_bytes_per_sm_clock(double dram_bytes_per_second, int sms,
                                                       double clock_hz) noexcept {
    return dram_bytes_per_second > 0 && sms > 0 && clock_hz > 0
               ? dram_bytes_per_second / (sms * clock_hz)
               : 0;
}

// The L2's rated bytes per SM an L2 clock (slices x 64 B over the SMs); 0 where the architecture's
// slice size is unknown. Per SM clock it is this times the L2 clock over the SM clock, which no
// table publishes (0.87-0.95 on the cards ncu read).
[[nodiscard]] constexpr double l2_bytes_per_sm_l2_clock(int cc, double l2_mib, int sms) noexcept {
    return sms > 0 ? arch_registry::l2_bytes_per_l2_clock(cc, l2_mib) / sms : 0;
}

// The waves a grid of `ctas` runs in on `sms` SMs holding `per_sm` CTAs each: ceil(ctas / slots).
[[nodiscard]] constexpr long long waves(long long ctas, int sms, int per_sm) noexcept {
    const long long slots = static_cast<long long>(sms) * per_sm;
    return ctas > 0 && slots > 0 ? (ctas + slots - 1) / slots : 0;
}

// The checks the comment above names, on the facts table's rows (data rate per pin, bus bits).
static_assert(dram_bytes_per_s(384, 21.0) == 1008e9, "RTX 4090: 1008 GB/s");
static_assert(dram_bytes_per_s(512, 28.0) == 1792e9, "RTX 5090: 1792 GB/s");
static_assert(l2_bytes_per_sm_l2_clock(120, 96, 128) == 24.0, "48 slices x 64 B over 128 SMs");
static_assert(waves(1024, 170, 6) == 2 && waves(1020, 170, 6) == 1 && waves(0, 170, 6) == 0);

} // namespace ninferno::board_facts
