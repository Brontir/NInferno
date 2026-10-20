#pragma once

#include "ops/softmax_attention/dense/causal_cache/int8/schedule.cuh"

namespace ninferno::ops::detail {

template <class G, int Tokens>
struct Int8KvGroupedInstance {
    static_assert(Tokens > 0 && Tokens * G::GroupSize <= 64);
    static constexpr int kRowTiles  = (Tokens * G::GroupSize + 15) / 16;
    // 64-key tiles take a 64 KiB arena: under the limit of every build but Turing's.
    static constexpr bool kWideKeys = G::GroupSize == 8 && kRowTiles >= 3 &&
                                      4 * 64 * 256 < kMaxSharedBytesPerBlock;
    using Schedule                  = Int8KvGroupedMmaSchedule<Tokens,
                                              kWideKeys        ? 4 * kRowTiles
                                                               : kRowTiles <= 2 ? 8
                                                                                : 2 * kRowTiles,
                                              kWideKeys ? 64 : 32, kWideKeys ? 1 : 2>;
    using Merge                     = Int8KvMergeSchedule<G::KVHeads == 4 ? 256 : 64>;
    // Three 32-key row tiles in twelve warps, one CTA per SM: two QK producers per row tile and six
    // V loaders (ninfer-all's 12x1x32q). A100 T = 6 at a short window: 475 against 707 us a round.
    static constexpr bool kHasWide  = !kWideKeys && kRowTiles == 3;
    using WideSchedule              = Int8KvGroupedMmaSchedule<Tokens, 4 * kRowTiles, 32, 1>;
};

// 90 KiB tuned; Turing's 64 KiB takes 32-key tiles (54 KiB).
using Int8KvTiledInstance = fitting_schedule<Int8KvTiledMmaSchedule<64, 64, 128>,
                                             Int8KvTiledMmaSchedule<64, 32, 128>>;

} // namespace ninferno::ops::detail
