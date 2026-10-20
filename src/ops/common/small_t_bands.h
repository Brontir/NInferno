#pragma once

#include "core/device_class.h"

#include <cstdint>
#include <limits>
#include <span>

namespace ninferno::ops::detail {

// A measured choice of small-T tile by call width: a call of at most `max_cols` columns (and more
// than the previous band's) takes `variant`, 0 being the op's own grouped tile.
struct SmallTBand {
    std::int32_t max_cols;
    int variant;
};

// The bands measured on one board (core/measured_boards.h).
struct SmallTChipBands {
    Board board;
    std::span<const SmallTBand> bands;
};

// Which entry of a table the last device a thread asked about resolved to (-1: the fallback): the
// entry of the nearest measured board of the device's instruction set (nearest_measured), so an
// RTX 5070 Ti takes the RTX 5080's bands and a card of an unmeasured architecture the fallback.
// A board none of whose measured boards shares its FP32-accumulate HMMA rate takes the nearest by
// memory (board_memory_distance): these widths stream weights. The A100 lost the RTX 3090's T = 1
// small-T MMA GDN projection when the rates were set apart (a7217e247) and fell to the fused-
// convolution GEMVs: 27B int8 C=1 decode 46.9 -> 41.5 tok/s.
struct SmallTBandCache {
    const DeviceClass* device = nullptr;
    int entry                 = -1;
};

inline int small_t_band_entry(std::span<const SmallTChipBands> measured, const DeviceClass& d) {
    const Board board              = board_of(d);
    const SmallTChipBands* nearest = nearest_measured(measured, board);
    if (nearest == nullptr) {
        double best = std::numeric_limits<double>::infinity();
        for (const SmallTChipBands& m : measured) {
            const double distance = board_memory_distance(board, m.board);
            if (distance < best) {
                best    = distance;
                nearest = &m;
            }
        }
    }
    return nearest == nullptr ? -1 : static_cast<int>(nearest - measured.data());
}

inline int small_t_band_variant(std::span<const SmallTChipBands> measured,
                                std::span<const SmallTBand> fallback, SmallTBandCache& cache,
                                std::int32_t cols) noexcept {
    std::span<const SmallTBand> bands = fallback;
    try {
        const DeviceClass& d = device_class();
        if (cache.device != &d) {
            cache.entry  = small_t_band_entry(measured, d);
            cache.device = &d;
        }
        if (cache.entry >= 0) bands = measured[static_cast<std::size_t>(cache.entry)].bands;
    } catch (...) {}
    for (const SmallTBand& band : bands) {
        if (cols <= band.max_cols) return band.variant;
    }
    return 0;
}

} // namespace ninferno::ops::detail
