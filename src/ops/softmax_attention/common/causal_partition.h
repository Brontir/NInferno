#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace ninferno::ops::detail {

// One query row's own aligned splits (CausalKvPartition::aligned_row): split j owns the keys
// [j << shift, (j + 1) << shift) and the row runs `splits` of them (0: the row stores none).
struct CausalAlignedRow {
    int shift  = 6;
    int splits = 0;

    // A key tile starting at k0 (past the CTA's first) closes one of the row's splits when the
    // row splits finer than the CTA (2^tile_shift keys) and k0 is on the row's split boundary.
    __host__ __device__ bool closes(int k0, int tile_shift) const {
        return shift < tile_shift && (k0 & ((1 << shift) - 1)) == 0;
    }
    // The row's split holding `key`.
    __host__ __device__ int split_of(int key) const { return key >> shift; }
    __host__ __device__ bool stores(int split) const { return split < splits; }
};

// Capture reserves partials for the largest live row. Producer and merge use
// the same live count; a wider capture never changes a row's work partition.
struct CausalKvPartition {
    static constexpr int kMaxSplits = 256;
    // Aligned splits: a token seeing v keys splits them every 2^aligned_key_shift(v) keys from
    // position 0, at most kAlignedSplits of them, each a power of two of at least 64 keys. The
    // partition follows the token's own position only (never the KV row a session sized from free
    // memory, nor the call that carries it): up to 4096 keys a split holds 64 keys, up to 8192
    // 128, ..., 256K 4096, 1M 16384.
    static constexpr int kAlignedSplits      = 64;
    static constexpr int kAlignedMinKeyShift = 6;
    // The most keys an aligned split is sized for: no call sees more
    // (kCausalAttentionMaximumVisibleKeys: 1M keys under YaRN), so more keys split as these do,
    // every 16384 keys.
    static constexpr int kAlignedMaxVisibleKeys = 1 << 20;
    int capacity                                = 1;
    int target                                  = 1;
    // log2 of the minimum KV keys per split: 64 keys on every geometry and width (the wave target
    // already bounds the split count, and a larger minimum starves short windows). An aligned
    // partition takes its shift from the visible keys instead (shift_for).
    int key_shift = 6;
    // Aligned (the invariant route profile, prefill_routes.h): split s of a token seeing v keys
    // owns the keys [s << shift_for(v), (s + 1) << shift_for(v)), so a token's partial sums and
    // their merge do not depend on the window, width, batch or KV row of the call that carries
    // it. Otherwise the window's key tiles are spread evenly over the active splits.
    bool aligned = false;

    // log2 of the keys a split owns for `visible` keys.
    __host__ __device__ int shift_for(int visible) const {
        return aligned ? aligned_key_shift(visible) : key_shift;
    }

    __host__ __device__ int active(int visible) const {
        const int shift = shift_for(visible);
        const int count = (visible + (1 << shift) - 1) >> shift;
        return count < target ? count : target;
    }

    // The first of `tiles` key tiles of `tile_keys` keys that split `split` of `splits` owns over
    // a window of `window` keys (aligned: from the split's own key range, whatever `splits`);
    // split + 1 gives the end of its range.
    __host__ __device__ int first_tile(int split, int tiles, int splits, int tile_keys,
                                       int window) const {
        if (!aligned) return split * tiles / splits;
        const int tile = (split << shift_for(window)) / tile_keys;
        return tile < tiles ? tile : tiles;
    }

    // The aligned splits of the query row at `position` (negative: a row that stores none).
    __host__ __device__ CausalAlignedRow aligned_row(int position) const {
        if (position < 0) return {kAlignedMinKeyShift, 0};
        return {shift_for(position + 1), active(position + 1)};
    }

    // log2 of the keys an aligned split owns for `visible` keys: 6..14, 6 up to 4096 keys, then
    // one more each time the keys double.
    __host__ __device__ static constexpr int aligned_key_shift(int visible) {
        const int keys = visible < kAlignedMaxVisibleKeys ? visible : kAlignedMaxVisibleKeys;
        int shift      = kAlignedMinKeyShift;
        while ((kAlignedSplits << shift) < keys) ++shift;
        return shift;
    }

    // The most aligned splits a token seeing up to max_visible_keys keys runs: 64 at 4096 keys
    // and past them (every v > 4096 runs at most 64), fewer below.
    static constexpr int aligned_capacity(int max_visible_keys) {
        const int count = (max_visible_keys + (1 << kAlignedMinKeyShift) - 1) >> kAlignedMinKeyShift;
        return count < kAlignedSplits ? count : kAlignedSplits;
    }

    // The launch capacity a partition needs for calls seeing up to max_visible_keys keys.
    int capacity_for(int max_visible_keys) const {
        return aligned ? aligned_capacity(max_visible_keys) : active(max_visible_keys);
    }

    static CausalKvPartition aligned_to_positions(int max_visible_keys) {
        CausalKvPartition partition{1, kMaxSplits, kAlignedMinKeyShift, true};
        partition.capacity = aligned_capacity(max_visible_keys);
        return partition;
    }
};

inline constexpr int causal_partition_target(std::int64_t cta_budget, int independent_tiles) {
    return static_cast<int>(
        std::clamp<std::int64_t>(cta_budget / independent_tiles, 1, CausalKvPartition::kMaxSplits));
}

} // namespace ninferno::ops::detail
