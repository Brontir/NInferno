#pragma once

// A board's per-SM quantities, derived from its published facts (#612 step 7): constexpr functions
// over the rows, never stored, so a neighbouring board of the same architecture interpolates by its
// own SMs, clocks, bus and L2. No CUDA: includable from host code and from a .cu's host side.
//
// The L2's per-SM width is arch_registry's (l2_bytes_per_sm_l2_clock: it reads the architecture
// rows, which this header does not include, so machine_facts.h may). The DRAM rate is the facts
// table's: two transfers a memory clock across the bus (the clock half the data rate), the spelling
// MachineFacts::dram_bytes_per_s calls.


namespace ninferno::board_facts {

// DRAM bytes a board moves a second: two transfers a memory clock (kHz) across its bus, times the
// share of the bus the board's SMs reach (MachineFacts::share).
[[nodiscard]] constexpr double dram_bytes_per_s(int memory_clock_khz, int bus_bits,
                                                double share) noexcept {
    return 2.0 * memory_clock_khz * 1e3 * bus_bits / 8.0 * share;
}

// The DRAM bytes one SM may draw an SM clock: the board's DRAM rate over its SMs at the clock.
[[nodiscard]] constexpr double dram_bytes_per_sm_clock(double dram_bytes_per_second, int sms,
                                                       double clock_hz) noexcept {
    return dram_bytes_per_second > 0 && sms > 0 && clock_hz > 0
               ? dram_bytes_per_second / (sms * clock_hz)
               : 0;
}

// A board's tensor rate a second: its SMs at the clock, times the rate per SM clock (the
// generation row's, generation.h bf16/s8).
[[nodiscard]] constexpr double tensor_flops_per_s(int sms, double clock_hz,
                                                  double flops_per_sm_clock) noexcept {
    return sms * clock_hz * flops_per_sm_clock;
}

// The waves a grid of `ctas` runs in on `sms` SMs holding `per_sm` CTAs each: ceil(ctas / slots).
[[nodiscard]] constexpr long long waves(long long ctas, int sms, int per_sm) noexcept {
    const long long slots = static_cast<long long>(sms) * per_sm;
    return ctas > 0 && slots > 0 ? (ctas + slots - 1) / slots : 0;
}

// The facts table's rows (memory clock, bus bits) and the profiler's L2 width.
static_assert(dram_bytes_per_s(10500000, 384, 1.0) == 1008e9, "RTX 4090: 1008 GB/s");
static_assert(dram_bytes_per_s(14000000, 512, 1.0) == 1792e9, "RTX 5090: 1792 GB/s");
static_assert(waves(1024, 170, 6) == 2 && waves(1020, 170, 6) == 1 && waves(0, 170, 6) == 0);

} // namespace ninferno::board_facts
