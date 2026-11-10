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

#include "core/arch_registry.h"
#include "core/generation.h"
#include "core/weight.h"

#include <cmath>
#include <cstddef>
#include <optional>
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

// Shared memory a tile needs, as ggml's mmq_get_nbytes_shared computes it for the MMA data layout
// (Turing on, compute capability >= 7.5; third_party/ggml-quants mmq.cuh): the column ids (J ints),
// the weight tile (I x the format's sram stride, in 32-bit words) and the activation tile (J
// block_q8_1_mmq of 144 bytes, padded to a multiple of the CTA's threads x 4 bytes). The tables of
// every format run I = 128 rows and 256 threads. ggml_mmq::tile_shared_bytes reads the kernel's own
// number for the test that holds this one to it.
enum class SramLayout { Q8_0, Q8_1, Q2_K, Q3_K, Q6_K };

inline constexpr int kTileThreads   = 256;
inline constexpr int kActivationRow = 144; // sizeof(block_q8_1_mmq)
inline constexpr int kTileK         = 32;  // MMQ_TILE_NE_K

// ggml_cuda_mmq_get_sram_stride: 32-bit words per weight row of the tile.
[[nodiscard]] constexpr int sram_stride(SramLayout layout) noexcept {
    switch (layout) {
    case SramLayout::Q8_0:
        return 2 * kTileK + 2 * kTileK / 8 + 4;
    case SramLayout::Q8_1:
        return 2 * kTileK + 2 * kTileK / 8 + 4; // QI8_1 = 8: the same as Q8_0
    case SramLayout::Q2_K:
        return 2 * kTileK + kTileK + 4;
    case SramLayout::Q3_K:
        return 2 * kTileK + kTileK / 2 + 4;
    case SramLayout::Q6_K:
        return 2 * kTileK + kTileK / 32 + kTileK / 8 + 7; // QI6_K = 32
    }
    return 0;
}

// The sram layout of a weight format's tile in the table of the MMA data layout
// (mmq-config-ampere.cuh); none for a format the matrix kernel does not run.
[[nodiscard]] constexpr std::optional<SramLayout> sram_layout_of(QType qtype) noexcept {
    switch (qtype) {
    case QType::GGML_Q2_K:
        return SramLayout::Q2_K;
    case QType::GGML_Q3_K:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_S:
        return SramLayout::Q3_K;
    case QType::GGML_Q6_K:
        return SramLayout::Q6_K;
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_1:
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
        return SramLayout::Q8_1;
    case QType::GGML_Q4_0:
    case QType::GGML_Q5_0:
    case QType::GGML_IQ4_NL:
    case QType::GGML_IQ4_XS:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ1_S:
    case QType::GGML_Q1_0:
    case QType::GGML_Q2_0:
        return SramLayout::Q8_0;
    default:
        return std::nullopt;
    }
}

[[nodiscard]] constexpr std::size_t tile_shared_bytes(SramLayout layout, int j) noexcept {
    const std::size_t ids  = std::size_t(j) * sizeof(int);
    const std::size_t x    = std::size_t(kTile) * sram_stride(layout) * 4;
    const std::size_t y    = std::size_t(j) * kActivationRow;
    const std::size_t unit = std::size_t(kTileThreads) * sizeof(int);
    return ids + x + (y + unit - 1) / unit * unit;
}

// The most shared memory a block may take on compute capability major.minor (cudaDeviceProp::
// sharedMemPerBlockOptin; the CUDA programming guide's table): 96 KiB on sm_70, 64 KiB on sm_75,
// 163 KiB on sm_80 and sm_87, 227 KiB from sm_90 on, 99 KiB on sm_86, sm_89 and sm_12x.
[[nodiscard]] constexpr std::size_t shared_per_block(int major, int minor) noexcept {
    return arch_registry::find(major, minor).smem_per_block_optin; // the archs table
}

// The widest tile of the widths both tables hold (128, 64, 32, 16) that the device's shared memory
// holds for a format: 128 where it is not computed (below sm_75 ggml's dp4a layout, whose tiles are
// smaller than the MMA layout's and all fit; a format the kernel does not run). Turing's 64 KiB:
// Q2_K's 128-column tile is 70 144 bytes, so it tops out at 64.
[[nodiscard]] constexpr int widest_tile(QType qtype, int major, int minor) noexcept {
    const auto layout = sram_layout_of(qtype);
    if (!layout || major * 10 + minor < 75) { return kTile; }
    for (const int j : {128, 64, 32}) {
        if (tile_shared_bytes(*layout, j) <= shared_per_block(major, minor)) { return j; }
    }
    return 16;
}

// The width the card's table takes for `columns` tokens of a format (work_tiles_format), among the
// widths up to `widest` (widest_tile: a format whose wider tile the device's shared memory cannot
// hold, Turing's 64 KiB and Q2_K's 128-column tile, runs narrower ones).
[[nodiscard]] constexpr int pick(int columns, int major, int minor, bool format,
                                 int widest = 128) noexcept {
    const bool work    = uses_work_table(major, minor, format);
    const double fixed = tile_fixed(major, minor, format);
    int best           = 0;
    double least       = 1e300;
    for (const int j : tiles(major, minor, format)) {
        if (j > widest) { continue; }
        if (const double c = launch_cost(columns, j, work, fixed); c < least) {
            least = c;
            best  = j;
        }
    }
    return best;
}

} // namespace ninferno::mmq_columns
