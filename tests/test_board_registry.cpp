// The board registry (core/board_registry.h), no GPU: its constexpr rows are the reference table
// (core/data/gpu_facts.json) row for row, and each measured board (core/measured_boards.h) is the
// figures it was spelled with before the registry held them. A change of a row that moves a
// measured board's figures fails here, at compile time, before any distance or route moves.
#include "core/board_registry.h"
#include "core/device_class.h"
#include "core/machine_facts.h"
#include "core/measured_boards.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <string>

namespace {

using namespace ninferno;

constexpr bool same_board(const Board& a, const Board& b) {
    return a.major == b.major && a.minor == b.minor && a.sms == b.sms && a.l2_mib == b.l2_mib &&
           a.dram_gbps == b.dram_gbps && a.mma_slow == b.mma_slow;
}

// The literals the measured boards carried before the registry (major, minor, SMs, L2 MiB, DRAM
// GB/s): the value every derived distance and gate was computed from.
static_assert(same_board(measured::kRtx5090, Board{12, 0, 170, 96, 1792}));
static_assert(same_board(measured::kRtx5080, Board{12, 0, 84, 64, 960}));
static_assert(same_board(measured::kRtx5060Ti, Board{12, 0, 36, 32, 448}));
static_assert(same_board(measured::kRtx4090, Board{8, 9, 128, 72, 1008}));
static_assert(same_board(measured::kRtx4080Super, Board{8, 9, 80, 64, 736}));
static_assert(same_board(measured::kL4, Board{8, 9, 58, 48, 300}));
static_assert(same_board(measured::kRtx4060Ti, Board{8, 9, 34, 32, 288}));
static_assert(same_board(measured::kRtx3090, Board{8, 6, 82, 6, 936}));
static_assert(same_board(measured::kA10, Board{8, 6, 72, 6, 600}));
static_assert(same_board(measured::kA100Sxm40, Board{8, 0, 108, 40, 1555.2}));
static_assert(same_board(measured::kRtx2080Ti, Board{7, 5, 68, 5.5, 616}));
static_assert(same_board(measured::kV100Sxm2, Board{7, 0, 80, 6, 900}));

// The one conflict: the V100 SXM2 row's bandwidth is not the measured board's (898.048 against
// 900), so the board carries an override; every other measured board takes the row's own.
static_assert(board_registry::dram_gbps(*board_registry::find(0x1DB5)) == 898.048);
static_assert(board_registry::board(0x1DB5).dram_gbps == 898.048);
static_assert(board_registry::board(0x2B85).dram_gbps == 1792);

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

// The constexpr rows are the parsed table: same rows, same order, same figures.
void registry_matches_the_table() {
    const std::span<const GpuFactsRow> table = gpu_facts_table();
    expect(std::size(board_registry::kRows) == table.size(), "registry has one row per table row");
    const std::size_t n = std::min<std::size_t>(std::size(board_registry::kRows), table.size());
    for (std::size_t i = 0; i < n; ++i) {
        const board_registry::BoardRow& r = board_registry::kRows[i];
        const GpuFactsRow& t              = table[i];
        const std::string what            = "row " + std::to_string(i) + " " + t.name;
        expect(r.pci_device_id == t.pci_device_id, what + ": pci id");
        expect(r.major == t.major && r.minor == t.minor, what + ": compute capability");
        expect(r.sms == t.sms, what + ": SMs");
        expect(r.l2_mib == t.l2_mib, what + ": L2");
        expect(r.bus_bits == t.bus_bits, what + ": bus bits");
        expect(r.gbps == t.gbps, what + ": data rate");
        expect(r.boost_mhz == t.boost_mhz, what + ": boost clock");
        expect(board_registry::find(t.pci_device_id) != nullptr, what + ": found by id");
    }
    expect(board_registry::find(0) == nullptr, "an unknown id has no row");
}

} // namespace

int main() {
    try {
        registry_matches_the_table();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        ++failures;
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "board registry: ok\n";
    return 0;
}
