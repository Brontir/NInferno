// #617 D's replica: the busiest SM's MMA work (ops::detail::linear_pair_work) against the fused
// Q8 pair and the two linear() calls as measured on six boards. No GPU: the law reads the tile
// geometry and the board's SM count only.
//
// The cells: ninferno_linear_pair_bench --alternative pair|separate, K = 5120 (the 27B MTP K/V
// pair, 1024 rows), graph replay, cold L2, master 26902b47 (2026-11-16), the median of four runs
// a side (two ABBA rounds), microseconds. From 86 columns the fused schedule is one launch.
//
// What the law must do (#617 D):
//   - wherever it takes the two calls (fused work above theirs), the two calls are measured faster;
//     it never trades a measured fused win for the two calls;
//   - wherever its ratio is off a tie and the measured margin exceeds 3 %, it picks the measured
//     winner.
// Where the work ties (both grids full, the same products) the fused kernel keeps the call, as on
// master; there the two calls win on five of six boards by 2-59 %, the tile efficiency the work
// does not count (the fused DualMmaR32C64 holds one warp a scheduler: ncu, #947). Those cells are
// counted, printed and held to their number, so a law that names the term must change it.

#include "ops/linear/linear_registry.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

struct Cell {
    const char* card;
    int sms;
    std::int32_t tokens;
    double fused_us;
    double separate_us;
};

constexpr Cell kCells[] = {
    {"RTX 4090", 128, 86, 132.1, 45.1},      {"RTX 4090", 128, 96, 132.1, 47.1},
    {"RTX 4090", 128, 128, 126.5, 57.9},     {"RTX 4090", 128, 192, 127.0, 132.1},
    {"RTX 4090", 128, 256, 127.0, 123.4},    {"RTX 4090", 128, 512, 137.2, 128.5},
    {"RTX 4090", 128, 1024, 192.5, 183.3},   {"RTX 4090", 128, 2048, 376.8, 354.8},
    {"RTX 4070 Ti", 60, 86, 133.1, 77.3},    {"RTX 4070 Ti", 60, 96, 132.8, 80.9},
    {"RTX 4070 Ti", 60, 128, 123.9, 99.3},   {"RTX 4070 Ti", 60, 192, 133.1, 165.9},
    {"RTX 4070 Ti", 60, 256, 146.4, 165.4},  {"RTX 4070 Ti", 60, 512, 263.7, 264.2},
    {"RTX 4070 Ti", 60, 1024, 458.2, 413.2}, {"RTX 4070 Ti", 60, 2048, 793.7, 712.2},
    {"RTX 4060 Ti", 34, 86, 138.2, 129.0},   {"RTX 4060 Ti", 34, 96, 138.2, 129.0},
    {"RTX 4060 Ti", 34, 128, 133.1, 150.0},  {"RTX 4060 Ti", 34, 192, 145.8, 165.9},
    {"RTX 4060 Ti", 34, 256, 259.1, 163.3},  {"RTX 4060 Ti", 34, 512, 396.3, 321.5},
    {"RTX 4060 Ti", 34, 1024, 691.2, 627.7}, {"RTX 4060 Ti", 34, 2048, 1366.6, 1232.9},
    {"RTX 5070 Ti", 70, 86, 106.5, 67.6},    {"RTX 5070 Ti", 70, 96, 106.5, 69.6},
    {"RTX 5070 Ti", 70, 128, 102.4, 77.8},   {"RTX 5070 Ti", 70, 192, 104.4, 120.5},
    {"RTX 5070 Ti", 70, 256, 112.5, 110.6},  {"RTX 5070 Ti", 70, 512, 219.1, 155.6},
    {"RTX 5070 Ti", 70, 1024, 307.2, 306.9}, {"RTX 5070 Ti", 70, 2048, 602.1, 593.9},
    {"RTX 3090", 82, 86, 171.5, 89.1},       {"RTX 3090", 82, 96, 171.1, 89.1},
    {"RTX 3090", 82, 128, 165.9, 104.4},     {"RTX 3090", 82, 192, 175.6, 179.2},
    {"RTX 3090", 82, 256, 176.6, 166.9},     {"RTX 3090", 82, 512, 359.4, 237.1},
    {"RTX 3090", 82, 1024, 496.1, 464.9},    {"RTX 3090", 82, 2048, 896.0, 830.5},
    {"RTX 5090", 170, 86, 109.6, 41.0},      {"RTX 5090", 170, 96, 109.6, 41.0},
    {"RTX 5090", 170, 128, 104.4, 51.2},     {"RTX 5090", 170, 192, 104.4, 124.9},
    {"RTX 5090", 170, 256, 104.4, 112.6},    {"RTX 5090", 170, 512, 112.6, 116.7},
    {"RTX 5090", 170, 1024, 159.7, 163.8},   {"RTX 5090", 170, 2048, 311.3, 331.8},
};

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

} // namespace

int main() {
    namespace d   = ninferno::ops::detail;
    int ties_lost = 0;
    for (const Cell& c : kCells) {
        const std::string at = std::string(" ") + c.card + " T=" + std::to_string(c.tokens);
        const auto w         = d::linear_pair_work(1024, 5120, c.tokens, c.sms);
        check(w.fused.has_value() && w.separate.has_value(), "both sides priced" + at);
        if (!w.fused || !w.separate) continue;
        const bool takes_separate = *w.fused > *w.separate;
        const bool tie            = *w.fused == *w.separate;
        const double margin       = c.fused_us / c.separate_us - 1.0;
        if (takes_separate) {
            check(c.separate_us < c.fused_us,
                  "the two calls taken where the fused kernel wins" + at);
        } else if (!tie && std::abs(margin) > 0.03) {
            check(c.fused_us < c.separate_us, "the fused kernel kept where the two calls win" + at);
        } else if (tie && margin > 0.03) {
            ++ties_lost;
            std::cout << "open (tile efficiency):" << at << " fused " << c.fused_us
                      << " us, two calls " << c.separate_us << " us\n";
        }
    }
    // The open cells of the tie, held to their number (see the header).
    check(ties_lost == 16,
          "16 tied cells where the two calls win, not " + std::to_string(ties_lost));
    // The law needs a board: no SMs, nothing priced.
    const auto none = d::linear_pair_work(1024, 5120, 128, 0);
    check(!none.fused && !none.separate, "no SM count prices nothing");
    if (failures != 0) return EXIT_FAILURE;
    std::cout << "linear_pair work replica: ok\n";
    return EXIT_SUCCESS;
}
