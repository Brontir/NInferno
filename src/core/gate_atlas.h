#pragma once

// The MMQ prefill gate's decision atlas (#373, step 2 of its offline audit): its answer for every
// SKU of the reference table (core/data/gpu_facts.json), with what the answer rests on, computed by
// the gate's own query (measured::mmq_prefill_gate), never a second copy of its rule. A row says:
// - the target's facts and where each came from (the table: unknown ones stay unknown, never
//   taken from a neighbour);
// - the donor (the nearest measured board of the instruction set, named by the table when it
//   has the board), its distance, the quantities the distance could not compare, the answer and
//   its disposition;
// - the nearest measured board that answers the other way (the alternative donor, its distance):
//   where the two are close, the answer rests on which of them is nearer, not on a measurement of
//   the target;
// - fact withdrawal: the answer and disposition with the target's L2, then its bandwidth, taken
//   away. Withdrawing a fact must never raise the evidence's authority (gate_authority).
// The gate is board-wide: one answer per board, not per shape, and the atlas says so (scope).
// Pure: no device, no clock; the same table gives the same atlas.

#include "core/device_class.h"
#include "core/machine_facts.h"
#include "core/measured_boards.h"

#include <span>
#include <string>
#include <vector>

namespace ninferno::measured {

// How much authority an answer's evidence has, by disposition: Measured 4 (the board itself),
// Carried 3 (a near neighbour, every quantity compared), Transferred 2 (the gate's own rule past
// the carry distance), Incomplete 1 (a quantity unknown: the distance can understate), None 0.
[[nodiscard]] constexpr int gate_authority(GateAnswer::Disposition d) noexcept {
    switch (d) {
    case GateAnswer::Disposition::Measured: return 4;
    case GateAnswer::Disposition::Carried: return 3;
    case GateAnswer::Disposition::Transferred: return 2;
    case GateAnswer::Disposition::Incomplete: return 1;
    case GateAnswer::Disposition::None: return 0;
    }
    return 0;
}

struct GateAtlasRow {
    std::uint32_t pci_device_id = 0;
    std::string name;
    Board board;
    GateAnswer answer;
    std::string donor_name; // the table's name of the donor's board, "" when the table has none
    // The nearest measured board answering the other way: null when every donor agrees.
    const Gate* alternative = nullptr;
    std::string alternative_name;
    double alternative_distance = 0;
    GateAnswer without_l2;   // the answer with the target's L2 withdrawn
    GateAnswer without_dram; // ... its bandwidth withdrawn
};

// The table's board facts of a row (bandwidth = data rate x bus / 8; 0 where the row has none).
[[nodiscard]] Board table_board(const GpuFactsRow& row) noexcept;

// The atlas over `rows` (every target row of the table: identification-only rows are skipped).
[[nodiscard]] std::vector<GateAtlasRow> mmq_gate_atlas(std::span<const GpuFactsRow> rows);

// The atlas as JSON (one object per row), for a maintainer's report.
[[nodiscard]] std::string gate_atlas_json(std::span<const GateAtlasRow> atlas);

} // namespace ninferno::measured
