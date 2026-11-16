#pragma once

// The architecture table: the SM limits of each compute capability, constexpr, for host and device
// code alike (it includes no CUDA and no other core header). The rows are generated from the
// "archs" array of core/data/gpu_facts.json at configure time (cmake/BoardRegistry.cmake), so a
// capability's shared memory, threads, blocks and registers per SM are written once; the build's
// compile-time limits (ops/common/arch_limits.h), the MMQ tile widths (core/mmq_columns.h) and the
// synthetic device class (core/device_class.cpp) read them here.
//
// A capability the table lacks gets its row 0, "unlisted compute capability": the limits of
// GA10x, Ada and consumer Blackwell.

#include <cstddef>

namespace ninferno::arch_registry {

// One row of the table, in the file's order.
struct ArchRow {
    int cc;                           // major * 10 + minor; 0 is the row of an unlisted capability
    int cc_to;                        // the row covers cc..cc_to
    std::size_t smem_per_block_optin; // bytes (cudaDeviceProp::sharedMemPerBlockOptin)
    std::size_t smem_per_sm;          // bytes
    // The two limits kernels are compiled against (ops/common/arch_limits.h): the device's, except
    // the rows of no chip at 7.1, 7.3 and 7.4, which compile as Volta.
    std::size_t build_smem_per_block_optin;
    std::size_t build_smem_per_sm;
    int smem_reserved_per_block; // bytes of the SM's shared memory the driver keeps per block
    int max_threads_per_sm;
    int max_blocks_per_sm;
    int registers_per_sm;
    // The L2 slice's size in KiB (0: no card of the capability was read), from the profiler's
    // rated L2 width (ncu lts__t_bytes.sum.peak_sustained: 64 B a slice an L2 clock) over the
    // board's L2 (#612 step 7). A board's slices are its L2 over this, so neighbouring boards of
    // the capability interpolate by their L2 size.
    int l2_slice_kib = 0;
};

// The bytes one L2 slice moves an L2 clock (the profiler's rated width, every card read so far).
inline constexpr int kL2SliceBytesPerClock = 64;

inline constexpr ArchRow kRows[] = {
#include "core/arch_registry_rows.inc"
};

// The row of a compute capability; the unlisted row (cc 0) when the table has none.
[[nodiscard]] constexpr const ArchRow& find(int cc) noexcept {
    for (const ArchRow& row : kRows) {
        if (cc != 0 && row.cc <= cc && cc <= row.cc_to) { return row; }
    }
    return kRows[0];
}

[[nodiscard]] constexpr const ArchRow& find(int major, int minor) noexcept {
    return find(major * 10 + minor);
}

// Whether the table lists the capability itself (find() would otherwise answer with row 0).
[[nodiscard]] constexpr bool listed(int cc) noexcept {
    for (const ArchRow& row : kRows) {
        if (cc != 0 && row.cc <= cc && cc <= row.cc_to) { return true; }
    }
    return false;
}

// The shared bytes the driver reserves per resident block: the row's, and for a capability the
// table lacks 1 KiB from sm_80 on, none below.
[[nodiscard]] constexpr int shared_reserved_per_block(int cc) noexcept {
    return listed(cc) ? find(cc).smem_reserved_per_block : (cc >= 80 ? 1024 : 0);
}

static_assert(kRows[0].cc == 0, "the first row is the unlisted capability");

// A board's L2 slices: its L2 (MiB, the facts table's l2_mib) over the capability's slice size; 0
// when the capability's slice size is unknown.
[[nodiscard]] constexpr int l2_slices(int cc, double l2_mib) noexcept {
    const int slice = find(cc).l2_slice_kib;
    return slice > 0 && l2_mib > 0 ? static_cast<int>(l2_mib * 1024.0 / slice + 0.5) : 0;
}

// A board's rated L2 width in bytes an L2 clock: its slices x 64 B; 0 when unknown. The ceiling of
// any L2 rate the cost model prices (per SM: over the board's SMs, at the L2 clock).
[[nodiscard]] constexpr double l2_bytes_per_l2_clock(int cc, double l2_mib) noexcept {
    return static_cast<double>(l2_slices(cc, l2_mib)) * kL2SliceBytesPerClock;
}

// The cards ncu read (#612 step 7): the rule reproduces each one's rated width.
static_assert(l2_bytes_per_l2_clock(120, 96) == 3072, "RTX 5090: 48 slices");
static_assert(l2_bytes_per_l2_clock(120, 48) == 1536, "RTX 5070 Ti: 24 slices");
static_assert(l2_bytes_per_l2_clock(89, 72) == 2304, "RTX 4090: 36 slices");
static_assert(l2_bytes_per_l2_clock(89, 32) == 1024, "RTX 4060 Ti 16GB: 16 slices");
static_assert(l2_bytes_per_l2_clock(86, 6) == 1536, "RTX 3090: 24 slices");
static_assert(l2_bytes_per_l2_clock(86, 2.25) == 576, "RTX 3060: 9 slices");
static_assert(l2_bytes_per_l2_clock(75, 5.5) == 1408, "RTX 2080 Ti: 22 slices");
static_assert(l2_bytes_per_l2_clock(70, 6) == 2048, "Quadro GV100: 32 slices");

// The L2's rated bytes per SM an L2 clock (slices x 64 B over the SMs); 0 where the architecture's
// slice size is unknown. Per SM clock it is this times the L2 clock over the SM clock, which no
// table publishes (0.87-0.95 on the cards ncu read).
[[nodiscard]] constexpr double l2_bytes_per_sm_l2_clock(int cc, double l2_mib, int sms) noexcept {
    return sms > 0 ? l2_bytes_per_l2_clock(cc, l2_mib) / sms : 0;
}

static_assert(l2_bytes_per_sm_l2_clock(89, 72, 128) == 18.0, "RTX 4090: 36 slices x 64 B, 128 SMs");

} // namespace ninferno::arch_registry
