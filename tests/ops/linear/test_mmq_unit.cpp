// The counted MMQ unit (ops/generic/mmq_unit.h, #613 PR-5) equals the chain law's frozen
// predictions: tools/dev/mmq_chain_law.py's unit() for the RTX 4090 (gpu_facts.json: 128 SMs,
// 2520 MHz, 384 bits x 21 Gbps), arch 89's isa_probe facts (isa_probe 19 row) and an L2 floor
// of 15.2 B/clk/SM, on the launch law's grid (64 weight tiles on 128 CTAs). The expected seconds
// were printed by that script (python: m.unit(m.board("RTX 4090"), m.isa_facts("89"),
// m.waits("89"), 15.2, fmt, J)); they move only when it is run again. No GPU.

#include "core/format_facts.h"
#include "core/isa_facts.h"
#include "core/launch_facts.h"
#include "core/mmq_columns.h"
#include "ops/generic/mmq_unit.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <iostream>
#include <optional>
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
    {QType::GGML_Q4_K, 16, 2.095977380952381e-06},  {QType::GGML_Q4_K, 128, 4.272487990810359e-06},
    {QType::GGML_Q5_K, 16, 2.4573361111111112e-06}, {QType::GGML_Q5_K, 128, 4.4535235797827905e-06},
    {QType::GGML_Q6_K, 16, 3.0445472222222223e-06}, {QType::GGML_Q6_K, 128, 5.488613888888889e-06},
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
    // The units at the six work widths grow with the column tile (Q4_K, arch 89).
    for (std::size_t i = 1; i < std::size(mmq_columns::kTilesWork); ++i) {
        const auto a =
            mmq_unit::unit_seconds("89", QType::GGML_Q4_K, mmq_columns::kTilesWork[i - 1], row, f);
        const auto b =
            mmq_unit::unit_seconds("89", QType::GGML_Q4_K, mmq_columns::kTilesWork[i], row, f);
        expect(a && b && *b > *a, "units grow with the column tile");
    }
    // The board both clients price on (#973): the switch on, a scored arch with its isa row and
    // graph node; off, or an arch outside the scored three (sm_75), none.
    mmq_unit::counted_for_test(true);
    {
        const auto board = mmq_unit::counted_board(8, 9);
        expect(board && board->arch == "89" && board->isa.dram_latency.w1 == isa->dram_latency.w1,
               "the counted board of cc 8.9: arch 89 and its isa row");
        expect(board && board->node_seconds > 0 &&
                   board->node_seconds == launch_seconds(8, 9, LaunchKind::GraphNode).value_or(0.0),
               "the counted board of cc 8.9: its graph node");
        expect(!mmq_unit::counted_board(7, 5), "cc 7.5 is not scored: no counted board");
    }
    mmq_unit::counted_for_test(false);
    expect(!mmq_unit::counted_board(8, 9), "the switch off: no counted board");
    mmq_unit::counted_for_test(std::nullopt);
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
    // divide the tiles, the fixup's bytes at the L2 rate of its working CTAs (four a split tile,
    // at most the SMs) and the fixup's graph node. 16 tiles on 128 SMs: 8 CTAs a tile, 16 split
    // tiles, so 64 fixup CTAs, not the grid's 128 (#973).
    {
        const double bw = 384 / 8.0 * 21e9, node = 3e-6;
        for (const int rows : {16 * 128, 32 * 128, 48 * 128, 128 * 128}) {
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
                // The partial tiles move through the L2 at its rate over the fixing CTAs.
                const int fixers = std::min(l.split_tiles * mmq_columns::kFixupCtasPerTile, 128);
                if (rows == 16 * 128) {
                    expect(l.split_tiles == 16 && fixers == 64, what + ": 64 fixup CTAs");
                    // 8 CTAs a tile: the fixup walks back over 7 partials (#1095).
                    expect(l.rounds == 7, what + ": 7 fixup rounds, " + std::to_string(l.rounds));
                }
                // 48 tiles of 16 K steps on 128 CTAs of 6 units: every tile split, 192 fixup
                // CTAs, held to the 128 SMs.
                if (rows == 48 * 128) {
                    expect(l.split_tiles == 48 && fixers == 128, what + ": fixup CTAs at the SMs");
                    // 6 units a CTA over 16-step tiles: a tile meets 3 or 4 CTAs.
                    expect(l.rounds == 3, what + ": 3 fixup rounds, " + std::to_string(l.rounds));
                }
                const double l2_rate = *row.l2_bytes_per_sm_clock * fixers * 2520e6;
                // ...no faster than its walk over a tile's partials: rounds L2 trips (#1095).
                const double trip = row.l2_latency.w4.value_or(row.l2_latency.w1.value_or(0));
                const double walk = l.rounds * trip / 2520e6;
                const double want =
                    l.units * *u +
                    (l.fixup ? std::max(mmq_columns::fixup_bytes(l) / l2_rate, walk) + node : 0.0);
                expect(std::fabs(*s / want - 1) < 1e-12, what + ": units x unit + fixup");
            }
            expect(l.split == (rows != 128 * 128), what + ": stream-k below a full wave");
        }
    }
    // #1095's RTX 5090 call (iq4_xs 1024 x 4096 at T 16, J 16): 8 tiles on 170 SMs, 128 CTAs of one
    // unit each, every tile split 16 ways: the fixup walks back over 15 partials a tile.
    {
        const mmq_columns::ColumnLaunch l = mmq_columns::launch_of(1024, 4096, 16, 16, 170);
        expect(l.split && l.fixup && l.tiles == 8 && l.blocks == 128 && l.split_tiles == 8 &&
                   l.rounds == 15,
               "#1095: 8 tiles split 16 ways, 15 fixup rounds (" + std::to_string(l.rounds) + ")");
    }
    // On that split the walk, not the bytes, is the fixup's time (the RTX 4090's facts here, the
    // same launch on 128 SMs: 15 rounds of an L2 trip against 1.1 MB over 32 fixing CTAs).
    {
        const double bw = 384 / 8.0 * 21e9, node = 3e-6;
        const int rows = 1024, k = 4096, columns = 16, j = 16;
        const mmq_columns::ColumnLaunch l = mmq_columns::launch_of(rows, k, columns, j, 128);
        const auto u =
            mmq_unit::unit_seconds("89", QType::GGML_Q4_K, j, row,
                                   mmq_unit::unit_facts_of(rows, k, columns, j, 128, 2520e6, bw));
        const auto s = mmq_unit::launch_seconds("89", QType::GGML_Q4_K, row, rows, k, columns, j,
                                                128, 2520e6, bw, node);
        const double trip = row.l2_latency.w4.value_or(row.l2_latency.w1.value_or(0));
        const double walk = l.rounds * trip / 2520e6;
        const double bytes =
            mmq_columns::fixup_bytes(l) /
            (*row.l2_bytes_per_sm_clock * mmq_columns::fixup_ctas(l, 128) * 2520e6);
        expect(u && s && l.rounds == 15 && trip > 0 && walk > bytes,
               "#1095: a 16-way split's fixup is its walk (" + std::to_string(walk * 1e6) +
                   " us against " + std::to_string(bytes * 1e6) + " us of bytes)");
        if (u && s) {
            expect(std::fabs(*s / (l.units * *u + walk + node) - 1) < 1e-12,
                   "#1095: units x unit + the walk + the node");
        }
    }
    IsaRow old = *isa;
    old.l2_bytes_per_sm_clock.reset();
    expect(!mmq_unit::unit_seconds("89", QType::GGML_Q4_K, 16, old, f),
           "no unit where the row has no L2 floor (an older probe's)");
    expect(!mmq_unit::unit_seconds("89", QType::GGML_Q4_K, 24, row, f),
           "no unit at a width no kernel has");
    // The architectures the unit was scored on; Turing's prefill fixtures refuted it (scored_arch).
    expect(mmq_unit::scored_arch("86") && mmq_unit::scored_arch("89") &&
               mmq_unit::scored_arch("120a"),
           "the unit prices the scored architectures");
    expect(!mmq_unit::scored_arch("75") && !mmq_unit::scored_arch("70") &&
               !mmq_unit::scored_arch("80") && !mmq_unit::scored_arch("90") &&
               !mmq_unit::scored_arch("100"),
           "the fitted price where the unit is not scored");
    if (failures == 0) std::cout << "mmq_unit: ok\n";
    return failures == 0 ? 0 : 1;
}
