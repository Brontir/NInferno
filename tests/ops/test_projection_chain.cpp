// Whole-chain pricing and selection of a projection chain's fusion alternatives (#310,
// ops/common/projection_chain.h): output writes added to every projection step, the pointwise
// read at the L2's or the DRAM's bandwidth, the lowest chain seconds, a tie (ln ratio within the
// speed estimate's resolution) to fewer launches, then to the canonical order, an optional price
// per launch; unpriced steps are not selectable. No GPU.
#include "ops/common/projection_chain.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

namespace chain = ninferno::ops::detail::chain;
using chain::Alternative;
using chain::Step;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "projection chain: %s\n", what.c_str());
        ++failures;
    }
}

bool near(double a, double b) {
    return std::abs(a - b) <= 1e-12 * std::max(std::abs(a), std::abs(b));
}

std::string chosen(const std::vector<Alternative>& a, double resolution = 0.0,
                   double launch_seconds = 0.0) {
    const auto i = chain::select(a, resolution, launch_seconds);
    return i ? a[*i].name : std::string("none");
}

} // namespace

int main() {
    // 1 TB/s DRAM, a 4 MiB L2 at 4 TB/s.
    const chain::Memory mem{1.0e12, 4194304.0, 4.0e12};
    // Price: seconds and launches add; an unpriced, negative or infinite step, a step with no
    // launch, or an empty chain has none.
    {
        const auto p = chain::price(Alternative{"x", {Step{1.0e-5, 1}, Step{2.0e-5, 2}}});
        expect(p && near(p->seconds, 3.0e-5) && p->launches == 3, "sum of steps");
        expect(!chain::price(Alternative{"x", {Step{1.0e-5, 1}, Step{std::nullopt, 1}}}),
               "unpriced step");
        expect(!chain::price(Alternative{"x", {}}), "empty chain");
        expect(!chain::price(Alternative{"x", {Step{-1.0, 1}}}), "negative seconds refused");
        expect(!chain::price(Alternative{"x", {Step{1.0e-5, 0}}}), "launches = 0 refused");
        expect(!chain::price(Alternative{"x", {Step{std::numeric_limits<double>::infinity(), 1}}}),
               "infinite seconds refused");
        std::vector<Alternative> a{
            Alternative{"inf", {Step{std::numeric_limits<double>::infinity(), 1}}},
            Alternative{"ok", {Step{1.0, 1}}}};
        expect(chosen(a) == "ok", "an infinite step is not selectable");
    }
    // Projection step: the output write over the DRAM bandwidth is added.
    {
        const auto s = chain::projection_step_seconds(1.0e-5, 1.0e6, mem);
        expect(s && near(*s, 1.0e-5 + 1.0e-6), "output write added");
        expect(!chain::projection_step_seconds(std::nullopt, 1.0e6, mem), "unpriced route");
        expect(!chain::projection_step_seconds(1.0e-5, 1.0e6, chain::Memory{}), "no bandwidth");
    }
    // Pointwise read: at the L2's bandwidth when the intermediate fits, else at the DRAM's.
    {
        const auto in_l2   = chain::pointwise_seconds(2.0e6, 1.0e6, 2.0e6, mem);
        const auto in_dram = chain::pointwise_seconds(2.0e6, 1.0e6, 8.0e6, mem);
        expect(in_l2 && near(*in_l2, 2.0e6 / 4.0e12 + 1.0e6 / 1.0e12), "read from the L2");
        expect(in_dram && near(*in_dram, 2.0e6 / 1.0e12 + 1.0e6 / 1.0e12), "read from the DRAM");
        // An unknown L2 falls back to the DRAM.
        const auto no_l2 =
            chain::pointwise_seconds(2.0e6, 1.0e6, 2.0e6, chain::Memory{1.0e12, 0, 0});
        expect(no_l2 && near(*no_l2, 3.0e6 / 1.0e12), "no L2 facts: the DRAM");
        expect(!chain::pointwise_seconds(1.0, 1.0, 1.0, chain::Memory{}), "no DRAM bandwidth");
    }
    // Pairs: both alternatives write the same outputs; the cheaper chain wins on either side.
    {
        chain::PairPrices p;
        p.fused        = 50.0e-6;
        p.first        = 30.0e-6;
        p.second       = 30.0e-6;
        p.output_bytes = 1.0e6;
        p.memory       = mem;
        expect(chosen(chain::pair_alternatives(p)) == "fused", "fused 50 < 60");
        const auto fused = chain::price(chain::pair_alternatives(p)[0]);
        const auto sep   = chain::price(chain::pair_alternatives(p)[1]);
        expect(fused && sep && near(fused->seconds, 50.0e-6 + 2.0e-6) &&
                   near(sep->seconds, 60.0e-6 + 2.0e-6),
               "each alternative pays the two outputs once");
        p.fused = 70.0e-6;
        expect(chosen(chain::pair_alternatives(p)) == "separate", "separate 60 < 70");
    }
    // Ties: equal seconds (within the relative band) go to fewer launches, then to the order.
    {
        chain::PairPrices p;
        p.fused  = 0.5;
        p.first  = 0.25;
        p.second = 0.25;
        p.memory = mem;
        expect(chosen(chain::pair_alternatives(p)) == "fused", "tie, fused has 1 launch against 2");
        p.fused_launches = 2;
        expect(chosen(chain::pair_alternatives(p)) == "fused", "tie, equal launches: order");
        p.fused_launches = 3;
        expect(chosen(chain::pair_alternatives(p)) == "separate", "tie, fused has more launches");
        // Resolution 0 compares exactly: any difference decides, launches break only equality.
        p.fused_launches = 1;
        p.fused          = 0.5 * (1.0 + 1.0e-11);
        expect(chosen(chain::pair_alternatives(p)) == "separate", "resolution 0: cheaper wins");
        p.fused = 0.5;
        expect(chosen(chain::pair_alternatives(p)) == "fused",
               "resolution 0: equal, fewer launches");
        // A resolution of 2 %: fused dearer by 1 % is a tie and wins on launches (1 against 2);
        // dearer by 3 % is outside the band.
        p.fused = 0.5 * 1.01;
        expect(chosen(chain::pair_alternatives(p), 0.02) == "fused",
               "inside the resolution, fewer launches win though dearer");
        expect(chosen(chain::pair_alternatives(p)) == "separate", "the same at resolution 0");
        p.fused = 0.5 * 1.03;
        expect(chosen(chain::pair_alternatives(p), 0.02) == "separate", "outside the resolution");
        // Inside the band with more launches on the cheaper side: fewer launches still win.
        p.fused          = 0.5 * 0.99;
        p.fused_launches = 3;
        expect(chosen(chain::pair_alternatives(p), 0.02) == "separate",
               "inside the resolution, the cheaper one has more launches: it loses");
        expect(chosen(chain::pair_alternatives(p)) == "fused", "the same at resolution 0");
        p.fused_launches = 2;
        expect(chosen(chain::pair_alternatives(p), 0.02) == "fused",
               "inside the resolution, equal launches: canonical order (fused is first)");
        // The band is in ln units: its edge is a tie, a hair beyond it is not.
        p.fused_launches = 1;
        p.fused          = 0.5 * std::exp(0.05);
        expect(chosen(chain::pair_alternatives(p), 0.05 * (1.0 + 1.0e-6)) == "fused",
               "at the edge of the band: a tie");
        expect(chosen(chain::pair_alternatives(p), 0.05 * (1.0 - 1.0e-6)) == "separate",
               "beyond the band: not a tie");
        // Invalid resolutions compare exactly.
        p.fused = 0.5 * 1.01;
        expect(chosen(chain::pair_alternatives(p), -1.0) == "separate", "negative resolution");
        expect(chosen(chain::pair_alternatives(p), std::nan("")) == "separate", "NaN resolution");
    }
    // Launch pricing: launch_seconds is added per launch. Fused is 1 launch at 0.50, separate 2
    // launches at 0.49.
    {
        chain::PairPrices p;
        p.fused      = 0.50;
        p.first      = 0.245;
        p.second     = 0.245;
        p.memory     = mem;
        const auto a = chain::pair_alternatives(p);
        expect(chosen(a) == "separate", "no launch price: the cheaper chain");
        expect(chosen(a, 0.0, 0.001) == "separate", "a cheap launch: 0.501 against 0.492");
        expect(chosen(a, 0.0, 0.02) == "fused", "a dear launch: 0.52 against 0.53");
        expect(chosen(a, 0.0, -1.0) == "separate", "a negative launch price is none");
        expect(chosen(a, 0.0, std::numeric_limits<double>::infinity()) == "separate",
               "a non-finite launch price is none");
        // Totals 0.501 and 0.492 are 1.8 % apart: a tie at 2 %, and one launch wins.
        expect(chosen(a, 0.02, 0.001) == "fused", "a launch price, then a tie by the resolution");
    }
    // The band does not drift: A 1.000 / 3 launches, B 1.015 / 2, C 1.030 / 1 at 2 %. B is inside
    // the band of the minimum A, C outside (3 %): B, in every order.
    {
        const Alternative A{"A", {Step{1.000, 3}}};
        const Alternative B{"B", {Step{1.015, 2}}};
        const Alternative C{"C", {Step{1.030, 1}}};
        const std::vector<std::vector<Alternative>> orders{{A, B, C}, {C, B, A}, {B, A, C},
                                                           {B, C, A}, {A, C, B}, {C, A, B}};
        for (const auto& order : orders) {
            expect(chosen(order, 0.02) == "B", "three alternatives: B, the order does not matter");
            expect(chosen(order, 0.0) == "A", "three alternatives, resolution 0: the minimum");
            expect(chosen(order, 0.04) == "C", "three alternatives, 4 %: C is in the band");
        }
    }
    // Unpriced alternatives are skipped; with none priced nothing is selected.
    {
        chain::PairPrices p;
        p.first  = 30.0e-6;
        p.second = 30.0e-6;
        p.memory = mem;
        expect(chosen(chain::pair_alternatives(p)) == "separate", "fused unpriced");
        p.first = std::nullopt;
        expect(chosen(chain::pair_alternatives(p)) == "none", "nothing priced");
        expect(!chain::select({}), "no alternatives");
    }
    // SwiGLU, intermediate in the L2 (inter 8192, 64 tokens: 2 MiB gate/up matrix, 4 MiB L2) and
    // out of it (256 tokens: 8 MiB).
    for (const int tokens : {64, 256}) {
        chain::SwigluPrices p;
        p.fused          = 100.0e-6;
        p.projection     = 100.0e-6;
        p.inter          = 8192;
        p.tokens         = tokens;
        p.memory         = mem;
        const double out = 8192.0 * tokens * 2.0;
        const bool fits  = 2.0 * (2.0 * out) <= mem.l2_bytes; // twice the 2 x inter x T matrix
        expect(fits == (tokens == 64), "fixture: the 64-token matrix fits, the 256-token one not");
        const auto alt = chain::swiglu_alternatives(p);
        const auto fp  = chain::price(alt[0]);
        const auto pp  = chain::price(alt[1]);
        const double read =
            2.0 * out / (fits ? mem.l2_bytes_per_second : mem.dram_bytes_per_second);
        expect(fp && near(fp->seconds, 100.0e-6 + out / 1.0e12) && fp->launches == 1,
               "fused epilogue: projection plus its inter x tokens write");
        expect(pp && pp->launches == 2 &&
                   near(pp->seconds, 100.0e-6 + 2.0 * out / 1.0e12 + read + out / 1.0e12),
               "projection, matrix write, read back, product write");
        expect(chosen(alt) == "fused_epilogue", "fused epilogue at equal projection time");
        // A fused kernel slower than the whole unfused chain loses.
        p.fused = 100.0e-6 + 3.0 * out / 1.0e12 + 2.0 * read;
        expect(chosen(chain::swiglu_alternatives(p)) == "projection_then_pointwise",
               "slow epilogue loses");
        // No bandwidth: nothing is priced.
        p.memory = chain::Memory{};
        expect(chosen(chain::swiglu_alternatives(p)) == "none", "no bandwidth: no price");
    }
    if (failures != 0) { return 1; }
    std::puts("projection chain: output writes, L2 or DRAM read, lowest seconds, launches, order");
    return 0;
}
