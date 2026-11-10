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
    // The two limits kernels are compiled against (ops/common/arch_limits.h): the values they had
    // before this table, below the device's for sm_90..sm_119 until a measured change raises them.
    std::size_t build_smem_per_block_optin;
    std::size_t build_smem_per_sm;
    int smem_reserved_per_block; // bytes of the SM's shared memory the driver keeps per block
    int max_threads_per_sm;
    int max_blocks_per_sm;
    int registers_per_sm;
};

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

} // namespace ninferno::arch_registry
