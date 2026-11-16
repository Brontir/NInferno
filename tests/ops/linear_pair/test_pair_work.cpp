// #617 D's replica: the residency price (ops::detail::linear_pair_price, cta_facts.h) against the
// fused Q8 pair and the two linear() calls as measured on seven boards. No GPU: the law reads the
// kernels' compiled resources and K loops and the board's SM limits only. The separate side's
// tile comes from the shape's table (select_q8_a16_launch, which reads a route table's route_as
// band where the board has one: no table has a Q8 1024x5120 band, so the table's own tile).
//
// The cells: ninferno_linear_pair_bench --alternative pair|separate, K = 5120 (the 27B MTP K/V
// pair, 1024 rows), graph replay, cold L2, master 26902b47 (2026-11-16), the median of four runs
// a side (two ABBA rounds), microseconds. From 86 columns the fused schedule is one launch.
//
// What the law must do (#617 D):
//   - wherever it takes the two calls (the fused price above theirs), the two calls are measured
//     faster: it never trades a measured fused win for the two calls;
//   - wherever its ratio is off a tie and the measured margin exceeds 3 %, it picks the measured
//     winner.
// Where the price ties (the fused CTA's chain is the two calls' together, both grids in the same
// rounds) the fused kernel keeps the call, as on master; there the two calls win on five of seven
// boards by 5-11 % (the 5070 Ti's ties are within 2 %, the 5090 keeps the fused kernel ahead), the tile efficiency the chain does not count (the fused DualMma holds one warp
// a scheduler: ncu, #947). Those cells are counted, printed and held to their number, so a law
// that names the term must change it. (The busiest-SM MMA work, the first law tried, missed the
// RTX 3060 at T = 128: it took the two calls, measured 12 % slower.)

#include "core/device_class.h"
#include "ops/linear/linear_registry.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

struct Cell {
    const char* card;
    int major, minor, sms, max_ctas_per_sm; // the board's architecture, SMs and CTA limit
    std::int32_t tokens;
    double fused_us;
    double separate_us;
};

constexpr Cell kCells[] = {
    {"RTX 4090", 8, 9, 128, 24, 86, 132.1, 45.1},
    {"RTX 4090", 8, 9, 128, 24, 96, 132.1, 47.1},
    {"RTX 4090", 8, 9, 128, 24, 128, 126.5, 57.9},
    {"RTX 4090", 8, 9, 128, 24, 192, 127.0, 132.1},
    {"RTX 4090", 8, 9, 128, 24, 256, 127.0, 123.4},
    {"RTX 4090", 8, 9, 128, 24, 512, 137.2, 128.5},
    {"RTX 4090", 8, 9, 128, 24, 1024, 192.5, 183.3},
    {"RTX 4090", 8, 9, 128, 24, 2048, 376.8, 354.8},
    {"RTX 4070 Ti", 8, 9, 60, 24, 86, 133.1, 77.3},
    {"RTX 4070 Ti", 8, 9, 60, 24, 96, 132.8, 80.9},
    {"RTX 4070 Ti", 8, 9, 60, 24, 128, 123.9, 99.3},
    {"RTX 4070 Ti", 8, 9, 60, 24, 192, 133.1, 165.9},
    {"RTX 4070 Ti", 8, 9, 60, 24, 256, 146.4, 165.4},
    {"RTX 4070 Ti", 8, 9, 60, 24, 512, 263.7, 264.2},
    {"RTX 4070 Ti", 8, 9, 60, 24, 1024, 458.2, 413.2},
    {"RTX 4070 Ti", 8, 9, 60, 24, 2048, 793.7, 712.2},
    {"RTX 4060 Ti", 8, 9, 34, 24, 86, 138.2, 129.0},
    {"RTX 4060 Ti", 8, 9, 34, 24, 96, 138.2, 129.0},
    {"RTX 4060 Ti", 8, 9, 34, 24, 128, 133.1, 150.0},
    {"RTX 4060 Ti", 8, 9, 34, 24, 192, 145.8, 165.9},
    {"RTX 4060 Ti", 8, 9, 34, 24, 256, 259.1, 163.3},
    {"RTX 4060 Ti", 8, 9, 34, 24, 512, 396.3, 321.5},
    {"RTX 4060 Ti", 8, 9, 34, 24, 1024, 691.2, 627.7},
    {"RTX 4060 Ti", 8, 9, 34, 24, 2048, 1366.6, 1232.9},
    {"RTX 5070 Ti", 12, 0, 70, 32, 86, 106.5, 67.6},
    {"RTX 5070 Ti", 12, 0, 70, 32, 96, 106.5, 69.6},
    {"RTX 5070 Ti", 12, 0, 70, 32, 128, 102.4, 77.8},
    {"RTX 5070 Ti", 12, 0, 70, 32, 192, 104.4, 120.5},
    {"RTX 5070 Ti", 12, 0, 70, 32, 256, 112.5, 110.6},
    {"RTX 5070 Ti", 12, 0, 70, 32, 512, 219.1, 155.6},
    {"RTX 5070 Ti", 12, 0, 70, 32, 1024, 307.2, 306.9},
    {"RTX 5070 Ti", 12, 0, 70, 32, 2048, 602.1, 593.9},
    {"RTX 3090", 8, 6, 82, 16, 86, 171.5, 89.1},
    {"RTX 3090", 8, 6, 82, 16, 96, 171.1, 89.1},
    {"RTX 3090", 8, 6, 82, 16, 128, 165.9, 104.4},
    {"RTX 3090", 8, 6, 82, 16, 192, 175.6, 179.2},
    {"RTX 3090", 8, 6, 82, 16, 256, 176.6, 166.9},
    {"RTX 3090", 8, 6, 82, 16, 512, 359.4, 237.1},
    {"RTX 3090", 8, 6, 82, 16, 1024, 496.1, 464.9},
    {"RTX 3090", 8, 6, 82, 16, 2048, 896.0, 830.5},
    {"RTX 5090", 12, 0, 170, 32, 86, 109.6, 41.0},
    {"RTX 5090", 12, 0, 170, 32, 96, 109.6, 41.0},
    {"RTX 5090", 12, 0, 170, 32, 128, 104.4, 51.2},
    {"RTX 5090", 12, 0, 170, 32, 192, 104.4, 124.9},
    {"RTX 5090", 12, 0, 170, 32, 256, 104.4, 112.6},
    {"RTX 5090", 12, 0, 170, 32, 512, 112.6, 116.7},
    {"RTX 5090", 12, 0, 170, 32, 1024, 159.7, 163.8},
    {"RTX 5090", 12, 0, 170, 32, 2048, 311.3, 331.8},
    {"RTX 3060", 8, 6, 28, 16, 86, 225.3, 201.2},
    {"RTX 3060", 8, 6, 28, 16, 96, 225.3, 204.8},
    {"RTX 3060", 8, 6, 28, 16, 128, 210.9, 239.1},
    {"RTX 3060", 8, 6, 28, 16, 192, 361.6, 394.8},
    {"RTX 3060", 8, 6, 28, 16, 256, 384.5, 380.9},
    {"RTX 3060", 8, 6, 28, 16, 512, 757.8, 596.0},
    {"RTX 3060", 8, 6, 28, 16, 1024, 1256.0, 1132.0},
    {"RTX 3060", 8, 6, 28, 16, 2048, 2362.4, 2140.2},
};

// The board as the device query describes it: 64 Ki registers, 100 KiB of shared memory (99 KiB a
// block with the opt-in) and 1536 threads an SM on every architecture of the table.
ninferno::DeviceClass board(const Cell& c) {
    ninferno::DeviceClass d;
    d.major                = c.major;
    d.minor                = c.minor;
    d.sms                  = c.sms;
    d.registers_per_sm     = 64 * 1024;
    d.smem_per_sm          = 100 * 1024;
    d.smem_per_block_optin = 99 * 1024;
    d.max_threads_per_sm   = 1536;
    d.max_blocks_per_sm    = c.max_ctas_per_sm;
    return d;
}

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
        const auto w = d::linear_pair_price(1024, 5120, c.tokens, board(c));
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
    check(ties_lost == 12, "12 tied cells where the two calls win, not " + std::to_string(ties_lost));
    // The law needs a board: no limits, nothing priced.
    const auto none = d::linear_pair_price(1024, 5120, 128, ninferno::DeviceClass{});
    check(!none.fused && !none.separate, "no SM count prices nothing");
    if (failures != 0) return EXIT_FAILURE;
    std::cout << "linear_pair residency replica: ok\n";
    return EXIT_SUCCESS;
}
