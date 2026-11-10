// The cross-row ranking of the quantizations (product::rank_rows, order_bounds, bound_in_reach) is
// a pure function of plain values. Pinned on hand-made rows (a worth win, a KLD tie-break, the
// rule's order on a full tie, the band edge), and compared with the loop plan_quants ran inline
// before it was lifted out (kept here as the reference) over many generated tables, ties included.
// CPU, no network, no GPU.

#include "product/choice/choice_constants.h"
#include "product/plan/plan_quants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

namespace pr = ninferno::product;
namespace rt = ninferno::runtime;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

pr::RankedRow row(std::size_t index, double worth, double kld, const std::string& label, int bits,
                  std::uint64_t bytes) {
    pr::RankedRow r;
    r.index                     = index;
    r.context                   = 32768;
    r.worth                     = worth;
    r.kld                       = kld;
    r.candidate.label           = label;
    r.candidate.label_bits      = bits;
    r.candidate.bytes           = bytes;
    r.candidate.bits_per_weight = bits;
    return r;
}

// The selection as plan_quants wrote it before the lift (one `top`, then a forward pass).
std::optional<std::size_t> legacy_best(const std::vector<pr::RankedRow>& rows, double band) {
    double top = -INFINITY;
    for (const auto& r : rows) { top = std::max(top, r.worth); }
    std::optional<std::size_t> best;
    double best_worth = 0;
    double best_kld   = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& s = rows[i];
        if (s.worth < top - band) { continue; }
        const bool wins =
            !best || s.kld < best_kld - ninferno::product::choice::kWorthTie ||
            (std::abs(s.kld - best_kld) <= ninferno::product::choice::kWorthTie &&
             (s.worth > best_worth + ninferno::product::choice::kWorthTie ||
              (std::abs(s.worth - best_worth) <= ninferno::product::choice::kWorthTie &&
               rt::quant_ranks_above(s.candidate, rows[*best].candidate))));
        if (wins) {
            best       = i;
            best_worth = s.worth;
            best_kld   = s.kld;
        }
    }
    return best;
}

void test_pinned() {
    const double band = 0.02;
    {
        const auto r = pr::rank_rows({}, band);
        check(!r.best, "no rows: no winner");
    }
    {
        // Q8 is better by 0.5 of worth: Q4 is out of the band.
        const auto r = pr::rank_rows(
            {row(0, 1.0, 0.001, "Q8_0", 8, 8000), row(1, 0.5, 0.010, "Q4_K_M", 4, 4000)}, band);
        check(r.best == 0 && r.reason == pr::RankReason::Only && r.top == 1.0,
              "worth wins outside the band");
    }
    {
        // Inside the band the lower KLD wins, whatever the listing order.
        const auto a = pr::rank_rows(
            {row(0, 1.00, 0.010, "Q4_K_M", 4, 4000), row(1, 0.99, 0.002, "Q8_0", 8, 8000)}, band);
        const auto b = pr::rank_rows(
            {row(1, 0.99, 0.002, "Q8_0", 8, 8000), row(0, 1.00, 0.010, "Q4_K_M", 4, 4000)}, band);
        check(a.best == 1 && a.reason == pr::RankReason::LowerKld, "lower KLD inside the band");
        check(b.best == 0 && b.reason == pr::RankReason::LowerKld,
              "same pick, other listing order");
    }
    {
        // Same KLD: the higher worth.
        const auto r = pr::rank_rows(
            {row(0, 0.99, 0.004, "Q6_K", 6, 6000), row(1, 1.00, 0.004, "Q5_K_M", 5, 5000)}, band);
        check(r.best == 1 && r.reason == pr::RankReason::HigherWorth, "higher worth at equal KLD");
    }
    {
        // A full tie goes to the rule's order, from either side.
        const auto small     = row(0, 1.0, 0.004, "Q4_K_M", 4, 4000);
        const auto big       = row(1, 1.0, 0.004, "Q8_0", 8, 8000);
        const bool big_first = rt::quant_ranks_above(big.candidate, small.candidate);
        const auto a         = pr::rank_rows({small, big}, band);
        const auto b         = pr::rank_rows({big, small}, band);
        check(big_first ? (a.best == 1 && b.best == 0) : (a.best == 0 && b.best == 1),
              "full tie follows quant_ranks_above");
    }
    {
        // The band edge: exactly `band` below the top is inside.
        const auto r = pr::rank_rows(
            {row(0, 1.0, 0.010, "Q8_0", 8, 8000), row(1, 1.0 - band, 0.001, "Q4_K_M", 4, 4000)},
            band);
        check(r.best == 1, "a worth exactly one band below still counts");
    }
    {
        check(pr::bound_in_reach(0.98, 1.0, 0.02), "bound at the edge is in reach");
        check(!pr::bound_in_reach(0.97, 1.0, 0.02), "bound past the edge is out");
        check(pr::bound_in_reach(-5, -INFINITY, 0.02),
              "nothing served yet: every bound is in reach");
        const auto ordered = pr::order_bounds({{0, 1.0}, {1, 2.0}, {2, 1.0}, {3, 3.0}});
        check(ordered.size() == 4 && ordered[0].index == 3 && ordered[1].index == 1 &&
                  ordered[2].index == 0 && ordered[3].index == 2,
              "bounds descend by worth, equal ones keep their order");
    }
}

void test_equals_legacy() {
    // A small deterministic generator; worths and KLDs on a coarse grid so that ties and band
    // edges occur often.
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    const auto next     = [&](std::uint32_t n) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>((state >> 33) % n);
    };

    const struct {
        const char* label;
        int bits;
    } kinds[] = {{"Q2_K", 2}, {"Q3_K_M", 3}, {"Q4_K_M", 4}, {"Q5_K_M", 5},
                 {"Q6_K", 6}, {"Q8_0", 8},   {"BF16", 16}};

    int compared = 0;
    for (int round = 0; round < 4000; ++round) {
        std::vector<pr::RankedRow> rows;
        const std::uint32_t n = 1 + next(6);
        for (std::uint32_t i = 0; i < n; ++i) {
            const auto& k = kinds[next(7)];
            rows.push_back(row(i, 1.0 + 0.01 * next(5), 0.001 * next(4), k.label, k.bits,
                               1000ULL * k.bits + next(3)));
        }
        const double band = 0.01 * next(4);
        const auto got    = pr::rank_rows(rows, band);
        check(got.best == legacy_best(rows, band),
              "rank_rows equals the inline loop (round " + std::to_string(round) + ")");
        ++compared;
    }
    check(compared == 4000, "all rounds compared");
}

} // namespace

int main() {
    test_pinned();
    test_equals_legacy();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "rank rows OK\n";
    return 0;
}
