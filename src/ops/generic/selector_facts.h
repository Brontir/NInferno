#pragma once

// The boards the selector goldens are kept for, and the maintainer's live check that the
// kernel-facts snapshot the register-MMA price reads (ops/generic/kernel_facts.h, #612) equals
// this build's kernels on a GPU that runs them (`ninferno-serve selector-facts`,
// tools/release/package.sh; ninferno_kernel_facts_live_test). One implementation for the test and
// the release, so what the release checks is what the goldens were made from.

#include "core/board_registry.h"
#include "core/device_class.h"
#include "ops/generic/kernel_facts.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninferno::ops::detail::generic {

// A board as the CUDA device query reports a stock one: compute capability, SMs, L2, memory size,
// boost SM clock, memory clock (half the data rate) and bus width.
struct FactsBoard {
    const char* name;
    std::uint32_t pci_device_id; // 0: an id the registry has no row of
    int major, minor, sms;
    double l2_mib;
    std::size_t memory_gib;
    int sm_clock_mhz, memory_clock_mhz, bus_bits;
};

// The golden board of a PCI id: its row of the board registry (core/board_registry.h, generated
// from core/data/gpu_facts.json), with the reference clock as the boost clock and half the row's
// data rate as the memory clock (machine_facts reads the row, not a driver's rounding such as
// 14001 MHz against 14000). An id the registry lacks is a compile error.
[[nodiscard]] constexpr FactsBoard registry_facts_board(const char* name,
                                                        std::uint32_t pci_device_id) {
    const board_registry::BoardRow& r = board_registry::row(pci_device_id);
    return {name,
            pci_device_id,
            r.major,
            r.minor,
            r.sms,
            r.l2_mib,
            static_cast<std::size_t>(r.vram_gib),
            r.boost_mhz,
            static_cast<int>(r.gbps * 500 + 0.5),
            r.bus_bits};
}

inline constexpr FactsBoard kFactsBoards[] = {
    registry_facts_board("rtx5090", 0x2B85),
    registry_facts_board("rtx5080", 0x2C02),
    registry_facts_board("rtx5070ti", 0x2C05),
    registry_facts_board("rtx5060ti", 0x2D04),
    registry_facts_board("rtx4090", 0x2684),
    registry_facts_board("rtx4060ti", 0x2805),
    registry_facts_board("l4", 0x27B8),
    registry_facts_board("rtx3090", 0x2204),
    registry_facts_board("a10", 0x2236),
    registry_facts_board("a100", 0x20B0),
    registry_facts_board("rtx2080ti", 0x1E07),
    registry_facts_board("v100", 0x1DB5),
    // Not a target, but the row where spellings of one generation fact used to disagree.
    registry_facts_board("h100-sxm", 0x2330),
    // Boards the registry has no row of, so their facts are declared here: the B300 and the GB10
    // (DGX Spark), which the table does not list, and a made-up sm_120 board of an unknown id. As
    // for any unknown id, machine_facts prices them from these driver values.
    {"b300", 0, 10, 3, 148, 126, 268, 2032, 3996, 8192},
    {"gb10-spark", 0, 12, 1, 48, 24, 128, 2418, 4266, 256},
    {"unknown-sm120-48sm", 0, 12, 0, 48, 32, 12, 2500, 14001, 192},
};

// The board as a substituted DeviceClass; the scales multiply the clocks the driver reports.
[[nodiscard]] DeviceClass board_device_class(const FactsBoard& board, double sm_scale = 1.0,
                                             double memory_scale = 1.0);

// Whether the board's register-MMA price reads compiled kernel facts (rmma_terms_measured).
[[nodiscard]] bool board_prices_from_facts(const FactsBoard& board);

// The register-MMA price of `board` over every format with sub-block scales
// (has_subblock_scales), a fixed set of shapes, epilogues and widths: the reads it makes of the
// kernel facts, as a count. Throws where it reads none (the price of a facts-priced board found
// no snapshot: a build whose kernel facts are not committed).
std::size_t selector_facts_probe(const FactsBoard& board);

// The probe under the live check: every kernel the price described, read from the snapshot and
// from the running GPU (kernel_facts::LiveRead); `mismatches` holds the reads that disagree.
struct LiveCheckResult {
    std::size_t checked = 0;
    std::vector<kernel_facts::LiveRead> mismatches;
};

[[nodiscard]] LiveCheckResult selector_facts_live_check(const FactsBoard& board);

} // namespace ninferno::ops::detail::generic

namespace ninferno::tools {

// `ninferno-serve selector-facts`: for each facts-priced golden board of this build's compute
// capability (NINFERNO_BUILD_ARCH), the live check of the kernels its price reads against the
// committed snapshot, on the first GPU that runs this build's kernels (the same major and an equal
// or higher minor; the same minor for an architecture-specific build). One line per board:
// "<board> ok <n> kernels", or the disagreeing reads. Exit 0 where every board agrees; 2 where this
// build prices no board from facts or no GPU runs its kernels (a release then reports the check as
// not run); 1 on a mismatch or any other error.
int selector_facts_main(int argc, char** argv);

} // namespace ninferno::tools
