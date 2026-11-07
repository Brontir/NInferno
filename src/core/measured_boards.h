#pragma once

// The boards NInferno's per-card data was measured on (route tables, R8 decode tables, small-T
// bands, measured gates, context-cost machines), by their published figures: compute capability,
// SMs, L2 MiB, DRAM GB/s.
// Data keyed by one of these carries to any other board through board_distance; a table names
// the board it ran on instead of a chip, whose boards differ (core/device_class.h: Board).

#include "core/device_class.h"

#include <limits>
#include <span>

namespace ninferno::measured {

inline constexpr Board kRtx5090{12, 0, 170, 96, 1792};
inline constexpr Board kRtx5080{12, 0, 84, 64, 960};
inline constexpr Board kRtx5060Ti{12, 0, 36, 32, 448};
inline constexpr Board kRtx4090{8, 9, 128, 72, 1008};
inline constexpr Board kRtx4080Super{8, 9, 80, 64, 736};
inline constexpr Board kL4{8, 9, 58, 48, 300};
inline constexpr Board kRtx4060Ti{8, 9, 34, 32, 288};
inline constexpr Board kRtx3090{8, 6, 82, 6, 936};
inline constexpr Board kA10{8, 6, 72, 6, 600};
inline constexpr Board kA100Sxm40{8, 0, 108, 40, 1555.2};
inline constexpr Board kRtx2080Ti{7, 5, 68, 5.5, 616};
inline constexpr Board kV100Sxm2{7, 0, 80, 6, 900};

// How far a measured choice carries (board_distance): to the variants of the board (a 4090 D or a
// 3090 Ti, 0.17 and 0.14 from the 4090 and the 3090) and to its near neighbours (an A40 0.43 from
// the A10, an RTX 5070 Ti 0.57 from the 5080), not to the next class of card: the RTX 4090's R8
// decode entries on the 4080 SUPER (1.2) won 41 and lost 30, to -22 %. A board farther from every
// measured one keeps the unmeasured default.
inline constexpr double kCarryDistance = 1.0;

// A choice measured on one board: whether it paid there.
struct Gate {
    Board board;
    bool pays;
};

// What a measured gate's answer for a board rests on (#373): the donor (the nearest measured board
// within the gate's distance; null: none), how far it is, what the distance could not compare, the
// answer, and how the evidence applies. Measured: the board itself (distance 0, every quantity
// known: the board's facts are a measured board's). Carried: within kCarryDistance, every quantity compared. Transferred: farther, the
// gate's own rule carrying it (the MMQ gate's: at any distance). Incomplete: a quantity unknown on
// either side, so the distance can understate the difference. None: no donor (the unmeasured
// default). The answer is the legacy rule's, but for one case: a gate bounded by a distance does
// not take an Incomplete donor's answer (the unmeasured default instead), since the quantity it
// could not compare may be what puts the donor out of reach. A missing fact so never promotes a
// transfer the full facts would refuse; a gate that carries at any distance (the MMQ gate's)
// admits regardless of the quantities, and keeps its answer.
struct GateAnswer {
    enum class Disposition { Measured, Carried, Transferred, Incomplete, None };
    const Gate* donor = nullptr;
    double distance   = 0;
    BoardUnknowns unknown;
    bool pays               = false;
    Disposition disposition = Disposition::None;
};
[[nodiscard]] constexpr const char* gate_disposition_name(GateAnswer::Disposition d) noexcept {
    switch (d) {
    case GateAnswer::Disposition::Measured: return "measured";
    case GateAnswer::Disposition::Carried: return "carried";
    case GateAnswer::Disposition::Transferred: return "transferred";
    case GateAnswer::Disposition::Incomplete: return "incomplete";
    case GateAnswer::Disposition::None: return "none";
    }
    return "none";
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

// The current device's answer from `gates`: gate_answer's for the device within `max_distance`,
// else false (the unmeasured default), as on no device.
[[nodiscard]] inline bool measured_gate(std::span<const Gate> gates,
                                        double max_distance = kCarryDistance) noexcept {
    try {
        return gate_answer(gates, board_of(device_class()), max_distance).pays;
    } catch (...) {
        return false;
    }
}

// Whether ggml's matrix kernel takes a prefill of the ggml block formats from 17 tokens on
// (ops/generic generic_gemm.cpp mmq_prefill_wins, where the measurements stand): the nearest
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

} // namespace ninferno::measured
