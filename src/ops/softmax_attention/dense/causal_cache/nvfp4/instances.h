#pragma once
#include "ops/softmax_attention/dense/causal_cache/nvfp4/schedule.cuh"

namespace ninferno::ops::detail {
template <class G, int Tokens>
struct Nvfp4KvGroupedInstance {
    static_assert(Tokens > 0 && Tokens * G::GroupSize <= 64);
    static constexpr int kRowTiles = (Tokens * G::GroupSize + 15) / 16;
    // 64-key tiles where they fit the build's shared memory (all but Turing's 64 KiB).
    static constexpr bool kWideKeys = kRowTiles > 2 &&
                                      kNvfp4KvGroupedSharedBytes<kRowTiles, 64, true> <=
                                          int(kMaxSharedBytesPerBlock);
    using Schedule = Nvfp4KvGroupedMmaSchedule<Tokens, kRowTiles <= 2 ? 8 : 4 * kRowTiles,
                                               kWideKeys ? 64 : 32, kRowTiles <= 2 ? 2 : 1, true>;
    using Merge    = Nvfp4KvMergeSchedule;
};

using Nvfp4KvTiledInstance = Nvfp4KvTiledMmaSchedule<64, 12>;
} // namespace ninferno::ops::detail
