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
// choice before DECISIONS 74). Where the generation row says mmq_work_tiles (core/generation.h:
// GA10x on the RTX 3060 and 3090, Ada on the RTX 4070 Ti and 4090, GB20x on the RTX 5090: the grids
// of DECISIONS 74) and the weight format is one those grids ran (work_tiles_format): J = 16, 32,
// 64, 80, 96, 128 and the least work ceil(columns / J) x step_share(J, the row's mmq_tile_fixed).
// No other generation or format has been run over the widths: a generation joins by its row, a
// format by its grid.

#include "core/generation.h"
#include "core/weight.h"

#include <cmath>
#include <span>

namespace ninferno::mmq_columns {

inline constexpr int kTile              = 128; // the widest tile, the weight rows of every tile
inline constexpr int kTilesEverywhere[] = {16, 32, 64, 128};
inline constexpr int kTilesWork[]       = {16, 32, 64, 80, 96, 128};
inline constexpr double kTileFixed      = 0.43;

// The weight formats the grids ran (Qwen3-8B Q4_K_M: its Q4_K and Q6_K linears). Another decode
// has another fixed share (the Q4_0 and IQ ones, DECISIONS 74): it keeps the fewest tiles.
[[nodiscard]] constexpr bool work_tiles_format(QType qtype) noexcept {
    return qtype == QType::GGML_Q4_K || qtype == QType::GGML_Q6_K;
}

[[nodiscard]] constexpr bool uses_work_table(int major, int minor, bool format) noexcept {
    return format && mmq_work_tiles_measured(major, minor);
}

[[nodiscard]] constexpr std::span<const int> tiles(int major, int minor, bool format) noexcept {
    return uses_work_table(major, minor, format) ? std::span<const int>(kTilesWork)
                                                 : std::span<const int>(kTilesEverywhere);
}

// The fixed share a tile of that table is priced with: the row's where it takes the work table,
// else the DECISIONS 69 point.
[[nodiscard]] constexpr double tile_fixed(int major, int minor, bool format) noexcept {
    return uses_work_table(major, minor, format) ? mmq_tile_fixed_of(major, minor) : kTileFixed;
}

// A J-column tile's K step against a 128-column tile's.
[[nodiscard]] constexpr double step_share(int j, double fixed = kTileFixed) noexcept {
    return fixed + (1 - fixed) * j / kTile;
}

[[nodiscard]] constexpr int tile_count(int columns, int j) noexcept {
    return (columns + j - 1) / j;
}

// The cost the launch choice minimises for covering `columns` tokens with tiles of width j on the
// card's table: the least work (the work table), else the fewest tiles.
[[nodiscard]] constexpr double launch_cost(int columns, int j, bool work_table,
                                           double fixed = kTileFixed) noexcept {
    return work_table ? tile_count(columns, j) * step_share(j, fixed) : tile_count(columns, j);
}

// The width the card's table takes for `columns` tokens of a format (work_tiles_format; every
// width available).
[[nodiscard]] constexpr int pick(int columns, int major, int minor, bool format) noexcept {
    const bool work    = uses_work_table(major, minor, format);
    const double fixed = tile_fixed(major, minor, format);
    int best           = 0;
    double least       = 1e300;
    for (const int j : tiles(major, minor, format)) {
        if (const double c = launch_cost(columns, j, work, fixed); c < least) {
            least = c;
            best  = j;
        }
    }
    return best;
}

} // namespace ninferno::mmq_columns
