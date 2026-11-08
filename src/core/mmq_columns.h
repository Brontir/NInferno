#pragma once

// The column tiles of ggml's matrix kernel (ops/generic/ggml_mmq) that a prefill product may take,
// and what a tile of each costs: one place for the kernel's launch choice (select_launch) and the
// product's estimate of it (product/choice prefill_linears_seconds), so the estimate prices the
// tiles the kernel runs.
//
// A tile is 128 weight rows x J tokens over the whole K. Per K step its CTA decodes the weight
// rows (the same for every J) and runs one MMA per 16 rows x 8 tokens x 32 values, so a tile costs
// a fixed part plus a part in J: kTileFixed + (1 - kTileFixed) J / 128 of a 128-column tile
// (DECISIONS 69). A product of `columns` tokens takes ceil(columns / J) tiles of the width whose
// tiles cost least, the narrower on a tie; the widths are the ones of ggml's Ampere list
// (mmq-config-ampere.cuh) that save padding where the prefill's slices land (DECISIONS 74).

#include <cmath>

namespace ninferno::mmq_columns {

inline constexpr int kTile = 128; // the widest tile, the weight rows of every tile
inline constexpr int kTiles[] = {16, 32, 64, 80, 96, 112, 128};
inline constexpr double kTileFixed = 0.43;

// A J-column tile's K step against a 128-column tile's.
[[nodiscard]] constexpr double step_share(int j) noexcept {
    return kTileFixed + (1 - kTileFixed) * j / kTile;
}

// The work of covering `columns` tokens with tiles of width j, in 128-column tiles' K steps.
[[nodiscard]] inline double work(int columns, int j) noexcept {
    return std::ceil(static_cast<double>(columns) / j) * step_share(j);
}

} // namespace ninferno::mmq_columns
