#pragma once

// The board registry: the constexpr form of the PCI-ID reference table (core/data/gpu_facts.json),
// for the data keyed by a board at compile time. The rows are generated from that file at
// configure time (cmake/BoardRegistry.cmake), so a board's compute capability, SMs, L2, memory bus
// and rate and reference clock are written once. Any other table names a board by its PCI id and
// reads the facts here (core/measured_boards.h); it holds no copy of them.
//
// A PCI id the table lacks is a compile error where the registry is read in a constant expression.
// The runtime parse of the same file is core/machine_facts.h; tests/test_board_registry.cpp checks
// that the two agree row for row.

#include "core/device_class.h"

#include <cstdint>

namespace ninferno::board_registry {

// One row of the table, in the file's order.
struct BoardRow {
    std::uint32_t pci_device_id;
    int major, minor;
    int sms;
    double l2_mib; // 0: the row does not give it
    int bus_bits;
    double gbps; // memory data rate per pin
    int boost_mhz;
    bool mma_slow; // no tensor cores (TU116, TU117, GTX 16 boards)
};

inline constexpr BoardRow kRows[] = {
#include "core/board_registry_rows.inc"
};

// DRAM bandwidth in GB/s: the bus width times the data rate per pin, over 8 bits.
[[nodiscard]] constexpr double dram_gbps(const BoardRow& row) noexcept {
    return row.bus_bits * row.gbps / 8.0;
}

// The row of a PCI product id; null when the table has none.
[[nodiscard]] constexpr const BoardRow* find(std::uint32_t pci_device_id) noexcept {
    for (const BoardRow& row : kRows) {
        if (row.pci_device_id == pci_device_id) { return &row; }
    }
    return nullptr;
}

// The Board (core/device_class.h) of a PCI id, from its row. `dram_gbps_override` (0: none) names
// a bandwidth that the data keyed by this board was measured under and the row spells differently;
// each use is a recorded conflict between tables (core/measured_boards.h), not a free parameter.
[[nodiscard]] constexpr Board board(std::uint32_t pci_device_id, double dram_gbps_override = 0) {
    const BoardRow* row = find(pci_device_id);
    if (row == nullptr) { throw "board registry: unknown PCI device id"; }
    return Board{row->major,
                 row->minor,
                 row->sms,
                 row->l2_mib,
                 dram_gbps_override != 0 ? dram_gbps_override : dram_gbps(*row),
                 row->mma_slow};
}

} // namespace ninferno::board_registry
