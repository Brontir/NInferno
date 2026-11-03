#include "core/gate_atlas.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <limits>

namespace ninferno::measured {
namespace {

// The table's name of a measured board: the first target row with its facts.
std::string table_name_of(const Board& board, std::span<const GpuFactsRow> rows) {
    for (const GpuFactsRow& row : rows) {
        const Board b = table_board(row);
        if (row.target && b.major == board.major && b.minor == board.minor && b.sms == board.sms &&
            b.l2_mib == board.l2_mib && std::abs(b.dram_gbps - board.dram_gbps) < 0.5) {
            return row.name;
        }
    }
    return {};
}

nlohmann::ordered_json answer_json(const GateAnswer& a) {
    nlohmann::ordered_json out;
    out["pays"]        = a.pays;
    out["disposition"] = gate_disposition_name(a.disposition);
    out["distance"]    = a.donor ? nlohmann::json(a.distance) : nlohmann::json(nullptr);
    nlohmann::json unknown = nlohmann::json::array();
    if (a.unknown.l2) { unknown.push_back("l2"); }
    if (a.unknown.dram) { unknown.push_back("dram"); }
    out["unknown"] = std::move(unknown);
    return out;
}

nlohmann::ordered_json board_json(const Board& b) {
    nlohmann::ordered_json out;
    out["cc"]        = b.major * 10 + b.minor;
    out["sms"]       = b.sms;
    out["l2_mib"]    = b.l2_mib > 0 ? nlohmann::json(b.l2_mib) : nlohmann::json(nullptr);
    out["dram_gbps"] = b.dram_gbps > 0 ? nlohmann::json(b.dram_gbps) : nlohmann::json(nullptr);
    out["mma_slow"]  = b.mma_slow;
    return out;
}

} // namespace

Board table_board(const GpuFactsRow& row) noexcept {
    return {row.major, row.minor, row.sms, row.l2_mib,
            row.gbps > 0 && row.bus_bits > 0 ? row.gbps * row.bus_bits / 8.0 : 0.0, row.mma_slow};
}

std::vector<GateAtlasRow> mmq_gate_atlas(std::span<const GpuFactsRow> rows) {
    std::vector<GateAtlasRow> out;
    for (const GpuFactsRow& row : rows) {
        if (!row.target) { continue; }
        GateAtlasRow r;
        r.pci_device_id = row.pci_device_id;
        r.name          = row.name;
        r.board         = table_board(row);
        r.answer        = mmq_prefill_gate(r.board);
        if (r.answer.donor) { r.donor_name = table_name_of(r.answer.donor->board, rows); }
        double nearest = std::numeric_limits<double>::infinity();
        for (const Gate& g : kMmqPrefillPays) {
            if (!r.answer.donor || g.pays == r.answer.pays) { continue; }
            const double d = board_distance(r.board, g.board);
            if (std::isfinite(d) && d < nearest) {
                nearest       = d;
                r.alternative = &g;
            }
        }
        r.alternative_distance = r.alternative ? nearest : 0;
        if (r.alternative) { r.alternative_name = table_name_of(r.alternative->board, rows); }
        Board no_l2 = r.board;
        no_l2.l2_mib = 0;
        r.without_l2 = mmq_prefill_gate(no_l2);
        Board no_dram = r.board;
        no_dram.dram_gbps = 0;
        r.without_dram = mmq_prefill_gate(no_dram);
        out.push_back(std::move(r));
    }
    return out;
}

std::string gate_atlas_json(std::span<const GateAtlasRow> atlas) {
    nlohmann::ordered_json rows = nlohmann::json::array();
    for (const GateAtlasRow& r : atlas) {
        nlohmann::ordered_json row;
        char id[16];
        std::snprintf(id, sizeof id, "0x%04X", r.pci_device_id);
        row["id"]     = id;
        row["name"]   = r.name;
        row["board"]  = board_json(r.board);
        row["scope"]  = "board-wide: one answer for every format, shape and context";
        row["answer"] = answer_json(r.answer);
        row["donor"]  = r.answer.donor ? nlohmann::ordered_json{{"name", r.donor_name},
                                                                {"board", board_json(r.answer.donor->board)}}
                                       : nlohmann::ordered_json(nullptr);
        row["alternative"] =
            r.alternative ? nlohmann::ordered_json{{"name", r.alternative_name},
                                                   {"board", board_json(r.alternative->board)},
                                                   {"pays", r.alternative->pays},
                                                   {"distance", r.alternative_distance}}
                          : nlohmann::ordered_json(nullptr);
        row["without_l2"]   = answer_json(r.without_l2);
        row["without_dram"] = answer_json(r.without_dram);
        rows.push_back(std::move(row));
    }
    return rows.dump(1);
}

} // namespace ninferno::measured
