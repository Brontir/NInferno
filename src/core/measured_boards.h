#pragma once

// The boards NInferno's per-card data was measured on (route tables, R8 decode tables, small-T
// bands, measured gates, context-cost machines), by their published figures: compute capability,
// SMs, L2 MiB, DRAM GB/s.
// Data keyed by one of these carries to any other board through board_distance; a table names
// the board it ran on instead of a chip, whose boards differ (core/device_class.h: Board).

#include "core/board_registry.h"
#include "core/device_class.h"
#include "core/measured_disposition.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

namespace ninferno::measured {

// Each board is its PCI id in the registry (core/board_registry.h, generated from
// core/data/gpu_facts.json): the facts are not spelled here. The pins below hold the figures these
// boards were measured under and fail the build when a row drifts from them.
inline constexpr Board kRtx5090      = board_registry::board(0x2B85);
inline constexpr Board kRtx5080      = board_registry::board(0x2C02);
inline constexpr Board kRtx5060Ti    = board_registry::board(0x2D04);
inline constexpr Board kRtx4090      = board_registry::board(0x2684);
inline constexpr Board kRtx4080Super = board_registry::board(0x2702);
inline constexpr Board kL4           = board_registry::board(0x27B8);
inline constexpr Board kRtx4060Ti    = board_registry::board(0x2805);
inline constexpr Board kRtx3090      = board_registry::board(0x2204);
inline constexpr Board kA10          = board_registry::board(0x2236);
inline constexpr Board kA100Sxm40    = board_registry::board(0x20B0);
inline constexpr Board kRtx2080Ti    = board_registry::board(0x1E07);

// The recorded conflicts between a registry row and the bandwidth the data keyed by that board was
// measured under (board_registry::board's override), in one place: the measured constants below
// and every table that names its board by PCI id (board_of_pci) read them.
struct DramConflict {
    std::uint32_t pci_device_id;
    double dram_gbps;
};

inline constexpr DramConflict kDramConflicts[] = {
    // The row spells the V100 SXM2's 4096-bit HBM2 at 1.754 Gbps, 898.048 GB/s; the route and R8
    // tables and the gates were measured under, and board_distance has always read, the published
    // 900 GB/s. Kept at 900 so every distance is as before.
    {0x1DB5, 900},
};

// The bandwidth a conflict records for a PCI id; 0 (none) for every other id.
[[nodiscard]] constexpr double conflict_dram_gbps(std::uint32_t pci_device_id) noexcept {
    for (const DramConflict& c : kDramConflicts) {
        if (c.pci_device_id == pci_device_id) { return c.dram_gbps; }
    }
    return 0;
}

// The board data recorded by its PCI id was measured on (a route table's gpu.pci_id, an R8 decode
// table's row): the registry's row with its recorded conflict. An id the registry lacks is a
// compile error in a constant expression, as board_registry::board.
[[nodiscard]] constexpr Board board_of_pci(std::uint32_t pci_device_id) {
    return board_registry::board(pci_device_id, conflict_dram_gbps(pci_device_id));
}

// board_of_pci for an id read at run time (a route table's gpu.pci_id); nullopt where the registry
// has no row for it.
[[nodiscard]] constexpr std::optional<Board>
find_board_of_pci(std::uint32_t pci_device_id) noexcept {
    if (board_registry::find(pci_device_id) == nullptr) { return std::nullopt; }
    return board_of_pci(pci_device_id);
}

inline constexpr Board kV100Sxm2 = board_of_pci(0x1DB5);

// Pins, checked in every translation unit that includes this header: a change of a gpu_facts.json
// row that moves a measured board's figures fails every product build, not only the test.
constexpr bool same_board_facts(const Board& a, const Board& b) {
    return a.major == b.major && a.minor == b.minor && a.sms == b.sms && a.l2_mib == b.l2_mib &&
           a.dram_gbps == b.dram_gbps && a.mma_slow == b.mma_slow;
}

// A device's board against its id's registry row: where they agree, a route table names the board
// by its id (route_table.cpp write_route_table; the shipped-boards test asks the same). The DRAM
// bandwidth reaches the two sides by different arithmetic (board_of from the memory clock in kHz
// of half the data rate, the row from bus * rate / 8), which differs in the last bit on four rows (the A100 PCIe
// 80GB, the H100 NVL, the A30, the A800 PCIe 80GB), so it and the L2 are compared to a part in
// 10^4; a twin on other memory differs by percents (the RTX 4070 with GDDR6: 20 against 21 Gbps).
// A row that gives no L2 (0) agrees only with a device that reports none, so a real card of such a
// row (the H100 NVL, the A30) is written by its facts until its row gives the L2.
constexpr bool row_board_facts(const Board& a, const Board& b) {
    const auto close = [](double x, double y) {
        return (x > y ? x - y : y - x) <= 1e-4 * (x > y ? x : y);
    };
    return a.major == b.major && a.minor == b.minor && a.sms == b.sms &&
           close(a.l2_mib, b.l2_mib) && close(a.dram_gbps, b.dram_gbps) && a.mma_slow == b.mma_slow;
}

// The literals the measured boards carried before the registry (major, minor, SMs, L2 MiB, DRAM
// GB/s): the value every derived distance and gate was computed from.
static_assert(same_board_facts(measured::kRtx5090, Board{12, 0, 170, 96, 1792}));
static_assert(same_board_facts(measured::kRtx5080, Board{12, 0, 84, 64, 960}));
static_assert(same_board_facts(measured::kRtx5060Ti, Board{12, 0, 36, 32, 448}));
static_assert(same_board_facts(measured::kRtx4090, Board{8, 9, 128, 72, 1008}));
static_assert(same_board_facts(measured::kRtx4080Super, Board{8, 9, 80, 64, 736}));
static_assert(same_board_facts(measured::kL4, Board{8, 9, 58, 48, 300}));
static_assert(same_board_facts(measured::kRtx4060Ti, Board{8, 9, 34, 32, 288}));
static_assert(same_board_facts(measured::kRtx3090, Board{8, 6, 82, 6, 936}));
static_assert(same_board_facts(measured::kA10, Board{8, 6, 72, 6, 600}));
static_assert(same_board_facts(measured::kA100Sxm40, Board{8, 0, 108, 40, 1555.2}));
static_assert(same_board_facts(measured::kRtx2080Ti, Board{7, 5, 68, 5.5, 616}));
static_assert(same_board_facts(measured::kV100Sxm2, Board{7, 0, 80, 6, 900}));

// How far a measured choice carries (board_distance): to the variants of the board (a 4090 D or a
// 3090 Ti, 0.17 and 0.14 from the 4090 and the 3090) and to its near neighbours (an A40 0.43 from
// the A10, an RTX 5070 Ti 0.57 from the 5080), not to the next class of card: the RTX 4090's R8
// decode entries on the 4080 SUPER (1.2) won 41 and lost 30, to -22 %. A board farther from every
// measured one keeps the unmeasured default.
// The value is CHOSEN, not measured or derived: the examples above sit below it and the one
// measured loss (1.2) above it, and nothing measured places the 1.0 between them (route_rules.h
// lists it with Basis::Unknown).
inline constexpr double kCarryDistance = 1.0;

// A choice measured on one board: whether it paid there.
struct Gate {
    Board board;
    bool pays;
};

// What a measured gate's answer for a board rests on (#373): the donor (the nearest measured board
// within the gate's distance; null: none), how far it is, what the distance could not compare, the
// answer, and how the evidence applies (Disposition, core/measured_disposition.h: the one
// vocabulary of the gates and of the value tables below). The answer is the legacy rule's, but for
// one case: a gate bounded by a distance does not take an Incomplete donor's answer (the
// unmeasured default instead), since the quantity it could not compare may be what puts the donor
// out of reach. So a missing fact never promotes a transfer the full facts would refuse
// (conservatively: also when the known quantities alone would reach); a gate that carries at any
// distance (the MMQ gate's) admits regardless of the quantities, and keeps its answer.
struct GateAnswer {
    using Disposition = measured::Disposition;
    const Gate* donor = nullptr;
    double distance   = 0;
    BoardUnknowns unknown;
    bool pays               = false;
    Disposition disposition = Disposition::None;
};
[[nodiscard]] constexpr const char* gate_disposition_name(GateAnswer::Disposition d) noexcept {
    return disposition_name(d);
}
[[nodiscard]] inline GateAnswer gate_answer(std::span<const Gate> gates, const Board& board,
                                            double max_distance) noexcept {
    using D = GateAnswer::Disposition;
    GateAnswer out;
    out.donor = nearest_measured(gates, board, max_distance);
    if (out.donor == nullptr) { return out; }
    out.distance    = board_distance(board, out.donor->board);
    out.unknown     = board_unknowns(board, out.donor->board);
    out.pays        = out.donor->pays;
    out.disposition = out.unknown.any()                ? D::Incomplete
                      : out.distance == 0              ? D::Measured
                      : out.distance <= kCarryDistance ? D::Carried
                                                       : D::Transferred;
    if (out.disposition == D::Incomplete && max_distance < std::numeric_limits<double>::max()) {
        out.pays = false;
    }
    return out;
}

// The current device's answer from `gates`: gate_answer's for the device within `max_distance`
// (false for an Incomplete donor), else false (the unmeasured default), as on no device.
[[nodiscard]] inline bool measured_gate(std::span<const Gate> gates,
                                        double max_distance = kCarryDistance) noexcept {
    try {
        return gate_answer(gates, board_of(device_class()), max_distance).pays;
    } catch (...) {
        return false;
    }
}

// Whether ggml's matrix kernel takes a prefill of the ggml block formats from 17 tokens on
// (ops/generic generic_select.cpp mmq_prefill_wins, where the measurements stand): the nearest
// measured board's answer of its instruction set, at any distance.
inline constexpr Gate kMmqPrefillPays[] = {
    {kRtx5090, true},    {kRtx5060Ti, false}, {kRtx4090, true},
    {kL4, true},         {kRtx4060Ti, true},  {kA10, true},
    {kA100Sxm40, false}, {kRtx2080Ti, true}};

// kMmqPrefillPays for `board` by its rule (the nearest measured board of the instruction set, at
// any distance), with what the answer rests on: one query for the device's gate (ops/generic
// mmq_prefill_device) and the product's estimate of a board it is not running on.
[[nodiscard]] inline GateAnswer mmq_prefill_gate(const Board& board) noexcept {
    return gate_answer(std::span<const Gate>(kMmqPrefillPays), board,
                       std::numeric_limits<double>::max());
}
[[nodiscard]] inline bool mmq_prefill_pays(const Board& board) noexcept {
    return mmq_prefill_gate(board).pays;
}
// The current device's answer: mmq_prefill_gate of its board, else false (no device: the
// unmeasured default, as measured_gate gives).
[[nodiscard]] inline bool mmq_prefill_device_pays() noexcept {
    try {
        return mmq_prefill_gate(board_of(device_class())).pays;
    } catch (...) {
        return false;
    }
}

// What a measured VALUE table's answer for a board rests on (#616 step 3): the tables that carry a
// measured value (a schedule, a token frontier) by board distance, not a yes/no gate, in the gates'
// vocabulary (Disposition). Measured: the board's facts are a table row's (distance 0, every
// quantity known). Carried: a measured board within the table's distance answers. Incomplete: no
// entry with every quantity known is within the distance, but the nearest entry by plain distance
// is, with a quantity unknown on either side (L2 or bandwidth), so it may be near in the facts that
// were left out; the value is the default, as gate_answer's bounded gate takes it. None: no entry
// is within the distance, and the value is the table's declared default (a literal, its provenance
// named by the table's owner). The donor is chosen exactly as the lookups choose it
// (nearest_known_measured), so the disposition never disagrees with the answer. Where every entry
// knows its facts (both tables assert measured_boards_known), an unknown is the board's own, no
// entry is complete for it, and the answer is Incomplete or None. Only a table with an incomplete
// entry could give Measured or Carried from a complete entry while a nearer entry is Incomplete.
template <class Entry>
struct TableAnswer {
    const Entry* entry      = nullptr; // the donor's entry, null on Incomplete and None
    double distance         = 0;       // board_distance to the donor (0 without one)
    Disposition disposition = Disposition::None;
};

// `table`'s answer for `board` within `max_distance`; an entry needs a `board` member.
template <class Entry, std::size_t Extent>
[[nodiscard]] TableAnswer<Entry> table_answer(std::span<const Entry, Extent> table,
                                              const Board& board, double max_distance) noexcept {
    TableAnswer<Entry> out;
    out.entry = nearest_known_measured(table, board, max_distance);
    if (out.entry != nullptr) {
        out.distance    = board_distance(board, out.entry->board);
        out.disposition = out.distance == 0 ? Disposition::Measured : Disposition::Carried;
        return out;
    }
    const Entry* plain = nearest_measured(table, board, max_distance);
    if (plain != nullptr && board_unknowns(board, plain->board).any()) {
        out.disposition = Disposition::Incomplete;
    }
    return out;
}

} // namespace ninferno::measured
