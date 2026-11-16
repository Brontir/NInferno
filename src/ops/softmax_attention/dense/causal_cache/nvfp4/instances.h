#pragma once
#include "ops/softmax_attention/dense/causal_cache/nvfp4/schedule.cuh"

#include <type_traits>

namespace ninferno::ops::detail {
template <class G, int Tokens>
struct Nvfp4KvGroupedInstance {
    static_assert(Tokens > 0 && Tokens * G::GroupSize <= 64);
    static constexpr int kRowTiles = (Tokens * G::GroupSize + 15) / 16;
    // 64-key tiles where they fit the build's shared memory (all but Turing's 64 KiB).
    static constexpr bool kWideKeys = kRowTiles > 2 &&
                                      kNvfp4KvGroupedSharedBytes<kRowTiles, 64, true> <=
                                          int(kMaxSharedBytesPerBlock);
    using Grouped = Nvfp4KvGroupedMmaSchedule<Tokens, kRowTiles <= 2 ? 8 : 4 * kRowTiles,
                                              kWideKeys ? 64 : 32, kRowTiles <= 2 ? 2 : 1, true>;
    // The slab kernel where it serves the rows (slab_rule.h): one n8 tile of query rows on sm_80+.
    static constexpr bool kSlab = nvfp4_kv_slab_serves(Tokens * G::GroupSize);
    using Schedule =
        std::conditional_t<kSlab, Nvfp4KvSlabSchedule<Tokens, Tokens * G::GroupSize, Grouped>,
                           Grouped>;
    static_assert(Schedule::kMinBlocks == nvfp4_kv_grouped_min_blocks(Tokens * G::GroupSize));
    using Merge    = Nvfp4KvMergeSchedule;
};

using Nvfp4KvTiledInstance = Nvfp4KvTiledMmaSchedule<64, 12>;
} // namespace ninferno::ops::detail
