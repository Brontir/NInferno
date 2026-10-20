#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace ninferno::ops::detail {

// Capture reserves partials for the largest live row. Producer and merge use
// the same live count; a wider capture never changes a row's work partition.
struct CausalKvPartition {
    static constexpr int kMaxSplits = 256;
    // Aligned splits: at most kAlignedSplits over the call's KV row, each a power of two of at
    // least 64 keys from position 0. Every call of a session reads rows of one width (its
    // block-table rows), so the partition is the same for all of them: a 4K row splits every 64
    // keys, a 256K row every 4096, and the partials of a wide call stay within a few hundred MiB.
    static constexpr int kAlignedSplits      = 64;
    static constexpr int kAlignedMinKeyShift = 6;
    // The longest row an aligned split is sized for: no call sees more keys
    // (kCausalAttentionMaximumVisibleKeys: 1M keys under YaRN), so a longer row splits as this
    // one, every 16384 keys. A row of up to 256K keys keeps the shift it always had (6..12).
    static constexpr int kAlignedMaxRowKeys = 1 << 20;
    int capacity                            = 1;
    int target                              = 1;
    // log2 of the minimum KV keys per split: 64 keys on every geometry and width (the wave target
    // already bounds the split count, and a larger minimum starves short windows); an aligned
    // partition takes it from its row (aligned_key_shift).
    int key_shift = 6;
    // Aligned (the invariant route profile, prefill_routes.h): split s owns the keys
    // [s << key_shift, (s + 1) << key_shift), so a token's partial sums and their merge do not
    // depend on the window, width or batch of the call that carries it. Otherwise the window's
    // key tiles are spread evenly over the active splits.
    bool aligned = false;

    __host__ __device__ int active(int visible) const {
        const int count = (visible + (1 << key_shift) - 1) >> key_shift;
        return count < target ? count : target;
    }

    // The first of `tiles` key tiles of `tile_keys` keys that split `split` of `splits` owns
    // (aligned: from the split's own key range, whatever `splits`); split + 1 gives the end of
    // its range.
    __host__ __device__ int first_tile(int split, int tiles, int splits, int tile_keys) const {
        if (!aligned) return split * tiles / splits;
        const int tile = (split << key_shift) / tile_keys;
        return tile < tiles ? tile : tiles;
    }

    // log2 of the keys an aligned split owns on a KV row of row_keys keys: 6..14.
    __host__ __device__ static constexpr int aligned_key_shift(int row_keys) {
        const int row = row_keys < kAlignedMaxRowKeys ? row_keys : kAlignedMaxRowKeys;
        int shift     = kAlignedMinKeyShift;
        while ((kAlignedSplits << shift) < row) ++shift;
        return shift;
    }

    // The aligned splits a call seeing up to max_visible_keys keys runs on a row of row_keys
    // keys; row_keys 0 gives the bound over every row (a workspace query).
    static int aligned_capacity(int max_visible_keys, int row_keys) {
        const int shift = aligned_key_shift(row_keys);
        const int count = (max_visible_keys + (1 << shift) - 1) >> shift;
        return count < kAlignedSplits ? count : kAlignedSplits;
    }

    static CausalKvPartition aligned_to_row(int row_keys, int max_visible_keys) {
        CausalKvPartition partition{1, kMaxSplits, aligned_key_shift(row_keys), true};
        partition.capacity = aligned_capacity(max_visible_keys, row_keys);
        return partition;
    }
};

inline constexpr int causal_partition_target(std::int64_t cta_budget, int independent_tiles) {
    return static_cast<int>(
        std::clamp<std::int64_t>(cta_budget / independent_tiles, 1, CausalKvPartition::kMaxSplits));
}

} // namespace ninferno::ops::detail
