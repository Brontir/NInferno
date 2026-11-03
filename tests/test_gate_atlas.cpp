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

    // Each measured board of the gate, as the table gives it, answers for itself.
    for (const m::Gate& g : m::kMmqPrefillPays) {
        bool found = false;
        for (const auto& r : atlas) {
            if (r.answer.donor == &g && r.answer.distance == 0) {
                found = true;
                // Measured where the table has every fact the measured board has; where its row
                // lacks one (the L4's and the A100's L2), incomplete, the fact named.
                const bool complete = (r.board.l2_mib > 0) == (g.board.l2_mib > 0) &&
                                      (r.board.dram_gbps > 0) == (g.board.dram_gbps > 0);
                expect(r.answer.pays == g.pays &&
                           (complete ? r.answer.disposition == m::GateAnswer::Disposition::Measured
                                     : r.answer.disposition ==
                                               m::GateAnswer::Disposition::Incomplete &&
                                           r.answer.unknown.any()),
                       r.name + ": a measured board answers for itself");
            }
        }
        (void)found; // a measured board the table's facts do not reproduce has no row of its own
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
                             Transfer{"GeForce RTX 4070 Ti", "NVIDIA L4", 0.797371, true}}) {
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

    // Withdrawing a fact never raises the authority, and leaves the fact unknown.
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
