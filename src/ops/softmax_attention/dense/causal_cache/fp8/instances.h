#pragma once

#include "ops/softmax_attention/dense/causal_cache/fp8/schedule.cuh"

namespace ninferno::ops::detail {

template <class G, int Tokens>
struct Fp8KvGroupedInstance {
    static_assert(Tokens > 0 && Tokens * G::GroupSize <= 64);
    static constexpr int kRowTiles = (Tokens * G::GroupSize + 15) / 16;
    // 64-key tiles: the 4 * 64 * 256 B arena plus the static Q, P and scale tiles. Turing's
    // 64 KiB takes 32-key tiles, two blocks an SM.
    static constexpr bool kWideKeys =
        Tokens != 1 &&
        std::size_t{4 * 64 * 256} + std::size_t{kRowTiles} * 16 * (256 + 2 * 64 + 4) + 64 * 4 <=
            kMaxSharedBytesPerBlock;
    using Schedule                 = Fp8KvGroupedMmaSchedule<Tokens,
                                             kRowTiles == 4   ? 16
                                                             : kRowTiles == 3 ? 12
                                                                              : 8,
                                             kWideKeys ? 64 : 32, kWideKeys ? 1 : 2>;
    using Merge                    = Fp8KvMergeSchedule<G::KVHeads == 4 ? 256 : 64>;
};

// 128 x 64 tuned; Volta's 96 KiB and Turing's 64 KiB take the first smaller tile that fits (the
// split plan still counts 128-row tiles, which only changes how many KV splits it picks).
using Fp8KvTiledInstance =
    fitting_schedule<Fp8KvTiledMmaSchedule<kMxfp8TiledQueryRows, 64>,
                     fitting_schedule<Fp8KvTiledMmaSchedule<kMxfp8TiledQueryRows, 32>,
                                      Fp8KvTiledMmaSchedule<64, 32>>>;

} // namespace ninferno::ops::detail
