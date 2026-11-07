// The MMQ prefill gate's decision atlas over the reference table (#373, core/gate_atlas.h), no GPU:
// - every measured board answers for itself;
// - the issue's three transfers come with their donor and distance;
// - withdrawing a fact never raises the evidence's authority;
// - the alternative donor answers the other way and is no nearer;
// - the atlas is the same twice.
// `--json` prints the atlas (the maintainer's report).
#include "core/gate_atlas.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>

namespace {

namespace m = ninferno::measured;
int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok && ++failures <= 20) { std::fprintf(stderr, "FAIL: %s\n", what.c_str()); }
}

const m::GateAtlasRow* row_named(const std::vector<m::GateAtlasRow>& atlas, const std::string& name) {
    for (const auto& r : atlas) {
        if (r.name == name) { return &r; }
    }
    return nullptr;
}

} // namespace

int main(int argc, char** argv) {
    const auto table = ninferno::gpu_facts_table();
    const auto atlas = m::mmq_gate_atlas(table);
    if (argc > 1 && std::strcmp(argv[1], "--json") == 0) {
        std::printf("%s\n", m::gate_atlas_json(atlas).c_str());
        return 0;
    }
    expect(!atlas.empty(), "the atlas has rows");

    // Each measured board of the gate is a row of the table, and answers for itself.
    for (const m::Gate& g : m::kMmqPrefillPays) {
        bool found = false;
        for (const auto& r : atlas) {
            if (r.answer.donor == &g && r.answer.distance == 0) {
                found = true;
                expect(r.answer.pays == g.pays &&
                           r.answer.disposition == m::GateAnswer::Disposition::Measured,
                       r.name + ": a measured board answers for itself");
            }
        }
        expect(found, "a measured board of the gate is a row of the table (sms " +
                          std::to_string(g.board.sms) + ")");
    }
    for (const char* name : {"GeForce RTX 5090", "GeForce RTX 5060 Ti", "GeForce RTX 4090",
                             "GeForce RTX 4060 Ti", "GeForce RTX 2080 Ti"}) {
        const auto* r = row_named(atlas, name);
        expect(r && r->answer.disposition == m::GateAnswer::Disposition::Measured &&
                   r->donor_name == name,
               std::string(name) + ": measured, its own donor");
    }

    // The issue's transfers: donor, distance (board_distance: |log2| SMs + |log2| bandwidth +
    // half |log2| L2), answer.
    struct Transfer {
        const char* target;
        const char* donor;
        double distance;
        bool pays;
    };
    for (const Transfer t : {Transfer{"GeForce RTX 5080", "GeForce RTX 5090", 2.210019, true},
                             Transfer{"GeForce RTX 5070 Ti", "GeForce RTX 5060 Ti", 2.251839, false},
                             Transfer{"GeForce RTX 4070 Ti", "L4", 0.797371, true}}) {
        const auto* r = row_named(atlas, t.target);
        expect(r != nullptr, std::string(t.target) + " is in the table");
        if (!r) { continue; }
        expect(r->donor_name == t.donor && std::abs(r->answer.distance - t.distance) < 1e-5 &&
                   r->answer.pays == t.pays,
               std::string(t.target) + ": donor " + r->donor_name + " at " +
                   std::to_string(r->answer.distance));
        expect(r->answer.disposition == (t.distance <= m::kCarryDistance
                                             ? m::GateAnswer::Disposition::Carried
                                             : m::GateAnswer::Disposition::Transferred),
               std::string(t.target) + ": carried within the carry distance, else transferred");
    }
    // The RTX 5080's answer rests on the 5090 being nearer than the 5060 Ti, which answers the
    // other way.
    if (const auto* r = row_named(atlas, "GeForce RTX 5080")) {
        expect(r->alternative != nullptr && r->alternative_name == "GeForce RTX 5060 Ti" &&
                   !r->alternative->pays && r->alternative_distance >= r->answer.distance,
               "the RTX 5080's alternative donor: " + r->alternative_name);
    }

    // Withdrawing a fact never raises the authority, and leaves the fact unknown. (Whether a donor
    // exists rests on the instruction set alone, so the authority falls to Incomplete by
    // construction: the check keeps it so.)
    for (const auto& r : atlas) {
        const int authority = m::gate_authority(r.answer.disposition);
        expect(m::gate_authority(r.without_l2.disposition) <= authority &&
                   m::gate_authority(r.without_dram.disposition) <= authority,
               r.name + ": a withdrawn fact raised the authority");
        if (r.without_l2.donor) {
            expect(r.without_l2.unknown.l2 &&
                       r.without_l2.disposition == m::GateAnswer::Disposition::Incomplete,
                   r.name + ": without its L2 the answer is incomplete");
        }
        if (r.alternative) {
            expect(r.alternative->pays != r.answer.pays &&
                       r.alternative_distance >= r.answer.distance,
                   r.name + ": the alternative donor answers the other way, no nearer");
        }
    }

    // A missing fact never promotes a bounded transfer (#373's synthetic case): a 170-SM sm_120
    // board with 8 MiB of L2 is 1.79 from the RTX 5090, out of kCarryDistance; with its L2 unknown
    // the distance drops to 0, and the bounded gate still refuses the donor's answer. The MMQ gate,
    // which carries at any distance, keeps its answer.
    {
        const m::Gate gates[] = {{m::kRtx5090, true}};
        const ninferno::Board small_l2{12, 0, 170, 8, 1792};
        const ninferno::Board unknown_l2{12, 0, 170, 0, 1792};
        const auto known   = m::gate_answer(gates, small_l2, m::kCarryDistance);
        const auto missing = m::gate_answer(gates, unknown_l2, m::kCarryDistance);
        expect(known.donor == nullptr && !known.pays, "the full facts put the donor out of reach");
        expect(missing.disposition == m::GateAnswer::Disposition::Incomplete && !missing.pays,
               "a missing L2 promoted a bounded transfer");
        const auto mmq = m::mmq_prefill_gate(unknown_l2);
        expect(mmq.disposition == m::GateAnswer::Disposition::Incomplete && mmq.pays,
               "the MMQ gate lost its answer to a missing L2");
        // The bounded table lookups (R8 decode tables, A8 routes, Q4 frontiers, spread, context
        // cost machines) take nearest_known_measured: the same board with every fact finds its
        // entry; with its L2 or bandwidth unknown it finds none and keeps the default; an entry
        // with a fact unknown is never anyone's donor.
        const ninferno::Board own_5090 = m::kRtx5090;
        const ninferno::Board unknown_dram{12, 0, 170, 96, 0};
        const auto* own =
            ninferno::nearest_known_measured(std::span(gates), own_5090, m::kCarryDistance);
        expect(own == &gates[0], "a known board lost its own entry");
        expect(ninferno::nearest_known_measured(std::span(gates), unknown_l2, m::kCarryDistance) ==
                   nullptr,
               "a missing L2 promoted a bounded table lookup");
        expect(ninferno::nearest_known_measured(std::span(gates), unknown_dram,
                                                m::kCarryDistance) == nullptr,
               "a missing bandwidth promoted a bounded table lookup");
        const m::Gate partial[] = {{ninferno::Board{12, 0, 170, 0, 1792}, true}};
        expect(ninferno::nearest_known_measured(std::span(partial), own_5090, m::kCarryDistance) ==
                   nullptr,
               "an entry of unknown L2 was taken as a donor");
        // The unbounded lookup is unchanged: same instruction set, nearest by what is known.
        expect(ninferno::nearest_measured(std::span(gates), unknown_l2) == &gates[0],
               "the unbounded lookup changed");
    }

    // Pure: the same table gives the same atlas.
    expect(m::gate_atlas_json(atlas) == m::gate_atlas_json(m::mmq_gate_atlas(table)),
           "the atlas is deterministic");

    if (failures != 0) {
        std::fprintf(stderr, "gate atlas: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("gate atlas: %zu boards, measured ones answer for themselves, no withdrawn fact "
                "raises authority\n",
                atlas.size());
    return 0;
}
