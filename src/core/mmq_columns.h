#pragma once

// The column tiles of ggml's matrix kernel (ops/generic/ggml_mmq) that a prefill product may take,
// and what a tile of each costs: one place for the kernel's launch choice (select_launch) and the
// product's estimate of it (product/choice prefill_linears_seconds), so the estimate prices the
// tiles the kernel runs.
//
// A tile is 128 weight rows x J tokens over the whole K. Per K step its CTA decodes the weight
// rows (the same for every J) and runs one MMA per 16 rows x 8 tokens x 32 values, so a tile costs
// a fixed part plus a part in J (DECISIONS 69, 74).
//
// Two tables. Everywhere: J = 16, 32, 64, 128 and the fewest tiles, the narrower on a tie (the
// choice before DECISIONS 74). GA10x (sm_86: RTX 3060 / 3090, A10), where DECISIONS 74's grid
// measured every width: J = 16, 32, 64, 80, 96, 128 and the least work ceil(columns / J) x
// step_share(J). The fixed share of the step_share is the RTX 4060 Ti's Q4_K 64-column tile (0.43);
// the RTX 3060's own Q4_K grid agrees with the picks it makes. No other card or format has been
// run over the widths, so no other card takes them: a card or format that is measured joins by
// its own table here.

#include <cmath>
#include <span>

namespace ninferno::mmq_columns {

inline constexpr int kTile = 128; // the widest tile, the weight rows of every tile
inline constexpr int kTilesEverywhere[] = {16, 32, 64, 128};
inline constexpr int kTilesGa10x[]      = {16, 32, 64, 80, 96, 128};
inline constexpr double kTileFixed      = 0.43;

[[nodiscard]] constexpr bool uses_work_table(int major, int minor) noexcept {
    return major == 8 && minor == 6;
}
[[nodiscard]] constexpr std::span<const int> tiles(int major, int minor) noexcept {
    return uses_work_table(major, minor) ? std::span<const int>(kTilesGa10x)
                                         : std::span<const int>(kTilesEverywhere);
}

// A J-column tile's K step against a 128-column tile's.
[[nodiscard]] constexpr double step_share(int j) noexcept {
    return kTileFixed + (1 - kTileFixed) * j / kTile;
}

[[nodiscard]] constexpr int tile_count(int columns, int j) noexcept { return (columns + j - 1) / j; }

// The cost the launch choice minimises for covering `columns` tokens with tiles of width j on the
// card's table: the least work (GA10x), else the fewest tiles.
[[nodiscard]] constexpr double launch_cost(int columns, int j, bool work_table) noexcept {
    return work_table ? tile_count(columns, j) * step_share(j) : tile_count(columns, j);
}

// The width the card's table takes for `columns` tokens (every width available).
[[nodiscard]] constexpr int pick(int columns, int major, int minor) noexcept {
    const bool work = uses_work_table(major, minor);
    int best        = 0;
    double least    = 1e300;
    for (const int j : tiles(major, minor)) {
        if (const double c = launch_cost(columns, j, work); c < least) {
            least = c;
            best  = j;
        }
    }
    return best;
}

} // namespace ninferno::mmq_columns
