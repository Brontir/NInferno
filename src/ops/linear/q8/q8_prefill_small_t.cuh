#pragma once

// Q8G32 MMA tiles for short prefill calls under the prefill route profile.
//
// The MMA contraction dequantizes each weight element to BF16 and runs one m16n8k16 MMA chain
// over K in order for every output element, so its result does not depend on the tile: these
// tiles give a column the same bytes as the wide MMA routes. A narrow tile gives a short call
// enough CTAs to cover the device, which the wide tiles leave idle. Each operator picks the tile
// per call size from its own sweep (RTX 5090, cold cache).

#include "ops/linear/q8/q8_schedule.cuh"

#include <cstdint>

namespace ninferno::ops::detail {

using Q8PrefillSmallR16C32 = Q8A16MmaSchedule<16, 32, 128, 16, 8, 2, 4>;
using Q8PrefillSmallR16C64 = Q8A16MmaSchedule<16, 64, 128, 16, 16, 2, 4>;
using Q8PrefillSmallR32C32 = Q8A16MmaSchedule<32, 32, 128, 32, 8, 2, 4>;

enum class Q8PrefillSmallTile : std::uint8_t {
    R16C32,
    R16C64,
    R32C32,
};

// Runs `launch.template operator()<Schedule>()` with the schedule of `tile`.
template <class Launch>
void q8_prefill_small_t_dispatch(Q8PrefillSmallTile tile, Launch&& launch) {
    switch (tile) {
    case Q8PrefillSmallTile::R16C32:
        launch.template operator()<Q8PrefillSmallR16C32>();
        return;
    case Q8PrefillSmallTile::R16C64:
        launch.template operator()<Q8PrefillSmallR16C64>();
        return;
    case Q8PrefillSmallTile::R32C32:
        launch.template operator()<Q8PrefillSmallR32C32>();
        return;
    }
}

} // namespace ninferno::ops::detail
