// The counted MMQ unit (ops/generic/mmq_unit.h, #613 PR-5) equals the chain law's frozen
// predictions: tools/dev/mmq_chain_law.py's unit() for the RTX 4090 (gpu_facts.json: 128 SMs,
// 2520 MHz, 384 bits x 21 Gbps), arch 89's isa_probe facts (isa_probe 18 row) and an L2 floor
// of 15.2 B/clk/SM, on the launch law's grid (64 weight tiles on 128 CTAs). The expected seconds
// were printed by that script (python: m.unit(m.board("RTX 4090"), m.isa_facts("89"),
// m.waits("89"), 15.2, fmt, J)); they move only when it is run again. No GPU.

#include "core/format_facts.h"
#include "core/isa_facts.h"
#include "core/mmq_columns.h"
#include "ops/generic/mmq_unit.h"

#include <cmath>
#include <iostream>
#include <string>

namespace {

using namespace ninferno;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

struct Case {
    QType q;
    int j;
    double seconds;
};

constexpr Case kCases[] = {
    {QType::GGML_Q4_K, 16, 2.0962357142857145e-06}, {QType::GGML_Q4_K, 128, 4.272595530492899e-06},
    {QType::GGML_Q5_K, 16, 2.4575904761904763e-06}, {QType::GGML_Q5_K, 128, 4.453635087719299e-06},
    {QType::GGML_Q6_K, 16, 3.044779761904762e-06},  {QType::GGML_Q6_K, 128, 5.488620238095238e-06},
};

} // namespace

int main() {
    expect(mmq_unit::bytes_per_256(QType::GGML_Q4_K) == 144, "Q4_K: 144 bytes per 256 values");
    expect(mmq_unit::bytes_per_256(QType::GGML_Q5_K) == 176, "Q5_K: 176 bytes per 256 values");
    expect(mmq_unit::bytes_per_256(QType::GGML_Q6_K) == 210, "Q6_K: 210 bytes per 256 values");
    std::optional<IsaRow> isa = isa_row(8, 9);
    expect(isa.has_value(), "arch 89 has an isa_probe row");
    if (!isa) return 1;
    IsaRow row                = *isa;
    row.l2_bytes_per_sm_clock = 15.2;
    const mmq_unit::UnitFacts f{64, 128, 2520e6, 384 / 8.0 * 21e9};
    for (const Case& c : kCases) {
        const std::string what = std::string(format_facts(c.q).name) + " J " + std::to_string(c.j);
        const auto u           = mmq_unit::unit_seconds("89", c.q, c.j, row, f);
        expect(u.has_value(), what + ": priced");
        if (u) {
            expect(std::fabs(*u / c.seconds - 1) < 1e-12, what + ": " + std::to_string(*u) +
                                                              " s, the script's " +
                                                              std::to_string(c.seconds));
        }
    }
    const auto law = mmq_unit::law("89", QType::GGML_Q4_K, row, f);
    expect(law.has_value(), "the Q4_K law of arch 89");
    if (law) {
        for (int i = 1; i < 6; ++i) {
            expect(law->unit_clocks[i] > law->unit_clocks[i - 1],
                   "units grow with the column tile");
        }
    }
    // The launch law's grid as a launch: 64 row tiles x 2 column tiles on 128 SMs, a tile a CTA,
    // read 64 weight tiles on 128 CTAs; 32 row tiles x 4 column tiles, 32 of 128.
    const auto g = mmq_unit::unit_facts_of(64 * 128, 4096, 2 * 16, 16, 128, 2520e6, 1008e9);
    expect(g.weight_tiles == 64 && g.ctas == 128, "the law grid: 64 weight tiles on 128 CTAs");
    const auto four = mmq_unit::unit_facts_of(32 * 128, 4096, 4 * 64, 64, 128, 2520e6, 1008e9);
    expect(four.weight_tiles == 32 && four.ctas == 128, "four column tiles share each weight tile");
    // Stream-k: 32 tiles of 16 K steps on 128 SMs run 128 CTAs of 4 units, CTA b from unit 4 b:
    // every CTA on its own (row tile, K step), 128 weight tiles, all 9.4 MB of Q4_K weights read
    // over the 4 units (#613 PR-5 review: a tile a CTA would price 32 CTAs and a quarter of it).
    const auto split = mmq_unit::unit_facts_of(32 * 128, 4096, 64, 64, 128, 2520e6, 1008e9);
    expect(split.weight_tiles == 128 && split.ctas == 128, "stream-k: a weight tile a CTA");
    // Starts that do not fall on a tile's K step: 40 tiles of 16 steps on 128 CTAs of 5 units.
    // The 20 row tiles' 320 weight tiles over 5 units: 64 an iteration, not one a CTA (the
    // neighbouring column tile reads the same weights an iteration later, from the L2).
    const auto skew = mmq_unit::unit_facts_of(20 * 128, 4096, 32, 16, 128, 2520e6, 1008e9);
    expect(skew.weight_tiles == 64 && skew.ctas == 128, "stream-k: the weights once a launch");
    // 200 row tiles of one column tile: stream-k over 128 CTAs of 25 units: 3200 / 25 = 128.
    const auto one = mmq_unit::unit_facts_of(200 * 128, 4096, 16, 16, 128, 2520e6, 1008e9);
    expect(one.weight_tiles == 128 && one.ctas == 128, "one column tile: a weight tile a CTA");
    // A launch is its longest CTA's units (stream-k: 32 tiles on 128 SMs; a tile launch: 128
    // tiles, one full wave) at its own CTAs' unit, plus, where the CTAs do not
    // divide the tiles, the fixup's bytes at the DRAM rate and the fixup's graph node.
    {
        const double bw = 384 / 8.0 * 21e9, node = 3e-6;
        for (const int rows : {32 * 128, 128 * 128}) {
            const int k = 4096, columns = 64, j = 64;
            const mmq_columns::ColumnLaunch l = mmq_columns::launch_of(rows, k, columns, j, 128);
            const auto u                      = mmq_unit::unit_seconds(
                "89", QType::GGML_Q4_K, j, row,
                mmq_unit::unit_facts_of(rows, k, columns, j, 128, 2520e6, bw));
            const auto s = mmq_unit::launch_seconds("89", QType::GGML_Q4_K, row, rows, k, columns,
                                                    j, 128, 2520e6, bw, node);
            const std::string what = "launch of " + std::to_string(rows) + " rows";
            expect(u && s, what + ": priced");
            if (u && s) {
                const double want =
                    l.units * *u + (l.fixup ? mmq_columns::fixup_bytes(l) / bw + node : 0.0);
                expect(std::fabs(*s / want - 1) < 1e-12, what + ": units x unit + fixup");
            }
            expect(l.split == (rows == 32 * 128), what + ": stream-k only for 32 tiles");
        }
    }
    IsaRow old = *isa;
    old.l2_bytes_per_sm_clock.reset();
    expect(!mmq_unit::unit_seconds("89", QType::GGML_Q4_K, 16, old, f),
           "no unit where the row has no L2 floor (an older probe's)");
    expect(!mmq_unit::unit_seconds("89", QType::GGML_Q4_K, 24, row, f),
           "no unit at a width no kernel has");
    if (failures == 0) std::cout << "mmq_unit: ok\n";
    return failures == 0 ? 0 : 1;
}
