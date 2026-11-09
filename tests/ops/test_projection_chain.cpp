// Whole-chain pricing and selection of a projection chain's fusion alternatives (#310,
// ops/common/projection_chain.h): the lowest chain seconds, a tie to fewer launches, then to the
// canonical order; unpriced steps are not selectable. No GPU.
#include "ops/common/projection_chain.h"

#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace ninferno::ops::detail::chain;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "projection chain: %s\n", what.c_str());
        ++failures;
    }
}

std::string chosen(const std::vector<Alternative>& a) {
    const auto i = select(a);
    return i ? a[*i].name : std::string("none");
}

} // namespace

int main() {
    // Price: seconds and launches add; an unpriced step, a negative one or an empty chain has none.
    {
        const auto p = price(Alternative{"x", {Step{1.0e-5, 1}, Step{2.0e-5, 2}}});
        expect(p && p->seconds == 1.0e-5 + 2.0e-5 && p->launches == 3, "sum of steps");
        expect(!price(Alternative{"x", {Step{1.0e-5, 1}, Step{std::nullopt, 1}}}), "unpriced step");
        expect(!price(Alternative{"x", {}}), "empty chain");
        expect(!price(Alternative{"x", {Step{-1.0, 1}}}), "negative seconds refused");
    }
    // Pairs: the cheaper chain wins whichever side it is on.
    {
        PairPrices p{50.0e-6, 1, 30.0e-6, 30.0e-6};
        expect(chosen(pair_alternatives(p)) == "fused", "fused 50 < 60");
        p.fused = 70.0e-6;
        expect(chosen(pair_alternatives(p)) == "separate", "separate 60 < 70");
    }
    // Ties: equal seconds go to fewer launches, then to the canonical order.
    {
        PairPrices p{0.5, 1, 0.25, 0.25};
        expect(chosen(pair_alternatives(p)) == "fused", "tie, fused has 1 launch against 2");
        p.fused_launches = 2;
        expect(chosen(pair_alternatives(p)) == "fused", "tie, equal launches: canonical order");
        p.fused_launches = 3;
        expect(chosen(pair_alternatives(p)) == "separate", "tie, fused has more launches");
    }
    // Unpriced alternatives are skipped; with none priced nothing is selected.
    {
        PairPrices p{std::nullopt, 1, 30.0e-6, 30.0e-6};
        expect(chosen(pair_alternatives(p)) == "separate", "fused unpriced");
        p.first = std::nullopt;
        expect(chosen(pair_alternatives(p)) == "none", "nothing priced");
        expect(!select({}), "no alternatives");
    }
    // SwiGLU: the pointwise chain pays the 2 x inter x tokens read and the inter x tokens write
    // through the DRAM bandwidth; the fused epilogue wins at equal projection time.
    {
        SwigluPrices p;
        p.fused                 = 100.0e-6;
        p.projection            = 100.0e-6;
        p.inter                 = 8192;
        p.tokens                = 64;
        p.dram_bytes_per_second = 1.0e12;
        const auto alt          = swiglu_alternatives(p);
        const auto pr           = price(alt[1]);
        const double bytes      = 3.0 * 8192 * 64 * 2; // read 2x, write 1x of inter x tokens BF16
        expect(pr && pr->launches == 2 && pr->seconds == 100.0e-6 + bytes / 1.0e12,
               "pointwise price");
        expect(chosen(alt) == "fused_epilogue", "fused epilogue at equal projection time");
        // A fused kernel slower than the projection plus the pointwise kernel loses.
        p.fused = 100.0e-6 + 2.0 * bytes / 1.0e12;
        expect(chosen(swiglu_alternatives(p)) == "projection_then_pointwise",
               "slow epilogue loses");
        // No bandwidth: the pointwise side is unpriced, the fused epilogue is the only one.
        p.dram_bytes_per_second = 0.0;
        expect(chosen(swiglu_alternatives(p)) == "fused_epilogue", "no bandwidth keeps the fused");
    }
    if (failures != 0) { return 1; }
    std::puts("projection chain: lowest seconds, then fewer launches, then canonical order");
    return 0;
}
