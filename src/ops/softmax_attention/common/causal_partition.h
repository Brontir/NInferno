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
    // position 0, at most kAlignedSplits of them, each a power of two of at least 512 keys. The
    // partition follows the token's own position only (never the KV row a session sized from free
    // memory, the call that carries it, nor the card): up to 32768 keys a split holds 512 keys,
    // up to 65536 1024, ..., 256K 4096, 1M 16384.
    static constexpr int kAlignedSplits = 64;
    // 512 keys: a split's fixed traffic, its query tile in (BF16), its FP32 partial out and the
    // merge reading it back (32 rows x 256 dims x 10 bytes = 80 KiB for the grouped CTA), is at
    // most a sixth of the BF16 keys and values it streams (512 x 256 x 4 bytes = 512 KiB). The
    // 64-key minimum this replaced cost the fixed work 1.25x the streamed bytes and launched up to
    // 8x the CTAs and partials of a prefill chunk below 32K keys (A100 exact prefill -6 % against
    // the row-aligned splits of a 32K session, which split every 512 keys there). A decode
    // splits as a prefill does: from 32K keys on it runs the 32..64 splits it had before.
    static constexpr int kAlignedMinKeyShift = 9;
    // The most keys an aligned split is sized for: no call sees more
    // (kCausalAttentionMaximumVisibleKeys: 1M keys under YaRN), so more keys split as these do,
    // every 16384 keys.
    static constexpr int kAlignedMaxVisibleKeys = 1 << 20;
    int capacity                                = 1;
    int target                                  = 1;
    // log2 of the minimum KV keys per split: 64 keys on every geometry and width (the wave target
    // already bounds the split count, and a larger minimum starves short windows). An aligned
    // partition takes its shift from the visible keys instead (aligned_key_shift).
    int key_shift = 6;
    // Aligned (the invariant route profile, prefill_routes.h): split s of a token seeing v keys
    // owns the keys [s << aligned_key_shift(v), (s + 1) << aligned_key_shift(v)), so a token's
    // partial sums and their merge do not depend on the window, width, batch or KV row of the call
    // that carries it. Otherwise the window's key tiles are spread evenly over the active splits.
    // The kernels take the mode as a template argument (Aligned): the aligned_* members serve the
    // aligned instances, active and first_tile the others.
    bool aligned = false;

    // The splits of a window of `visible` keys (not aligned).
    __host__ __device__ int active(int visible) const {
        const int count = (visible + (1 << key_shift) - 1) >> key_shift;
        return count < target ? count : target;
    }

    // The first of `tiles` key tiles of `tile_keys` keys that split `split` of `splits` owns (not
    // aligned: every launch of an aligned partition takes an Aligned instance, which uses
    // aligned_first_tile, and the other instances refuse it on the host); split + 1 gives the end
    // of its range.
    __host__ __device__ int first_tile(int split, int tiles, int splits, int) const {
        return split * tiles / splits;
    }

    // log2 of the keys an aligned split owns for `visible` keys: 9..14, 9 up to 32768 keys, then
    // one more each time the keys double.
    __host__ __device__ static constexpr int aligned_key_shift(int visible) {
        const int keys = visible < kAlignedMaxVisibleKeys ? visible : kAlignedMaxVisibleKeys;
        int shift      = kAlignedMinKeyShift;
        while ((kAlignedSplits << shift) < keys) ++shift;
        return shift;
    }

    // The aligned splits a token seeing `visible` keys runs (at most kAlignedSplits).
    __host__ __device__ static constexpr int aligned_active(int visible) {
        const int shift = aligned_key_shift(visible);
        return (visible + (1 << shift) - 1) >> shift;
    }

    // The first of `tiles` key tiles of `tile_keys` keys that aligned split `split` owns over a
    // window of `window` keys; split + 1 gives the end of its range.
    __host__ __device__ static int aligned_first_tile(int split, int tiles, int tile_keys,
                                                      int window) {
        const int tile = (split << aligned_key_shift(window)) / tile_keys;
        return tile < tiles ? tile : tiles;
    }

    // The aligned splits of the query row at `position` (negative: a row that stores none).
    __host__ __device__ static CausalAlignedRow aligned_row(int position) {
        if (position < 0) return {kAlignedMinKeyShift, 0};
        return {aligned_key_shift(position + 1), aligned_active(position + 1)};
    }

    // The splits a window of `visible` keys runs, in either mode (the host's checks).
    __host__ __device__ int splits(int visible) const {
        return aligned ? aligned_active(visible) : active(visible);
    }

    // The most aligned splits a token seeing up to max_visible_keys keys runs: 64 at 32768 keys
    // and past them (every v > 32768 runs at most 64), fewer below.
    static constexpr int aligned_capacity(int max_visible_keys) {
        const int keys  = (1 << kAlignedMinKeyShift) - 1;
        const int count = (max_visible_keys + keys) >> kAlignedMinKeyShift;
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
