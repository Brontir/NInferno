// The counted MMQ unit (ops/generic/mmq_unit.h, #613 PR-5) equals the chain law's frozen
// predictions: tools/dev/mmq_chain_law.py's unit() for the RTX 4090 (gpu_facts.json: 128 SMs,
// 2520 MHz, 384 bits x 21 Gbps), arch 89's isa_probe facts and an L2 floor of 15.2 B/clk/SM, on
// the launch law's grid (64 weight tiles on 128 CTAs). The expected seconds were printed by that
// script (python: m.unit(m.board("RTX 4090"), m.isa_facts("89"), m.waits("89"), 15.2, fmt, J));
// they move only when it is run again. No GPU.

#include "core/format_facts.h"
#include "core/isa_facts.h"
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
    {QType::GGML_Q4_K, 16, 2.0961563492063494e-06}, {QType::GGML_Q4_K, 128, 4.272576482873851e-06},
    {QType::GGML_Q5_K, 16, 2.4575126984126985e-06}, {QType::GGML_Q5_K, 128, 4.4536144527986635e-06},
    {QType::GGML_Q6_K, 16, 3.0447107142857143e-06}, {QType::GGML_Q6_K, 128, 5.488641666666667e-06},
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
    IsaRow old = *isa;
    old.l2_bytes_per_sm_clock.reset();
    expect(!mmq_unit::unit_seconds("89", QType::GGML_Q4_K, 16, old, f),
           "no unit where the row has no L2 floor (an older probe's)");
    expect(!mmq_unit::unit_seconds("89", QType::GGML_Q4_K, 24, row, f),
           "no unit at a width no kernel has");
    if (failures == 0) std::cout << "mmq_unit: ok\n";
    return failures == 0 ? 0 : 1;
}
