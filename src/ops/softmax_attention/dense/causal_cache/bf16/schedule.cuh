#pragma once

#include "ops/common/arch_limits.h"
#include "ops/softmax_attention/common/causal_geometry.h"
#include <cstdint>

namespace ninferno::ops::detail {

// Query rows pack token/head pairs. KV warps cooperate on disjoint key columns. D warps split the
// output columns of one query and key slice: each scores the slice's keys and keeps 1/DWarps of
// the accumulators (#856: at D=256 one warp's Q and O fragments took 192 registers a lane).
template <int QueryRows, int KeyRows = 32, int KVWarps = 1, int MinBlocks = 2, int FixedWidth = 0,
          int DWarps = 1>
struct Bf16KvGroupedMmaSchedule {
    static_assert(QueryRows == 16 || QueryRows == 32 || QueryRows == 64);
    static_assert(KeyRows == 32 || KeyRows == 64);
    static_assert(KVWarps == 1 || KVWarps == 2 || KVWarps == 4);
    static_assert(KeyRows % (16 * KVWarps) == 0);
    static_assert(QueryRows <= 2 * KeyRows);
    static_assert(FixedWidth == 0 || FixedWidth == 1);
    static_assert(DWarps == 1 || DWarps == 2);
    static constexpr int kFixedWidth = FixedWidth;
    static constexpr int kQueryRows  = QueryRows;
    static constexpr int kKeyRows    = KeyRows;
    static constexpr int kWarpsQ     = QueryRows / 16;
    static constexpr int kWarpsKV    = KVWarps;
    static constexpr int kWarpsD     = DWarps;
    static constexpr int kWarps      = kWarpsQ * kWarpsKV * kWarpsD;
    static_assert(kWarps <= 8);
    static constexpr int kThreads            = 32 * kWarps;
    static constexpr int kLaunchBoundThreads = kThreads < 128 ? 128 : kThreads;
    static constexpr int kMinBlocks          = MinBlocks;
};

// Existing single-buffer Q/K/V pipeline. Page-local staging admits key tiles
// dividing the 64-row physical page, not arbitrary multiples of the MMA shape.
template <int QueryTile = 64, int KeyTile = 64, int MinBlocks = 1>
struct Bf16KvTiledMmaSchedule {
    static_assert(QueryTile == 16 || QueryTile == 32 || QueryTile == 64 || QueryTile == 128);
    static_assert(KeyTile == 16 || KeyTile == 32 || KeyTile == 64);
    static_assert(MinBlocks > 0);
    static constexpr int kQueryRows = QueryTile;
    static constexpr int kKeyRows   = KeyTile;
    static constexpr int kWarps     = QueryTile / 16;
    static constexpr int kThreads   = kWarps * 32;
    static constexpr int kMinBlocks = MinBlocks;
    // Q and a K/V pair of key tiles at the causal head dim (bf16_kv_tiled_shared_bytes).
    static constexpr bool kFitsSharedLimit =
        static_cast<std::size_t>((QueryTile + 2 * KeyTile) * kCausalHeadDim * 2) <=
        kMaxSharedBytesPerBlock;
};

template <int DChunk, int Threads = 256>
struct Bf16KvMergeSchedule {
    static_assert(Threads == 128 || Threads == 256);
    static_assert(DChunk > 0 && DChunk <= Threads);
    static constexpr int kDChunk  = DChunk;
    static constexpr int kThreads = Threads;
    static constexpr int kWarps   = Threads / 32;
};

// Width-1 decode (the GroupedDecode instance) packs one token's G::GroupSize query heads into a
// single query warp's 16 MMA rows. With at most 8 (every listed geometry), the rows 8-15, a lane's
// second MMA row, never hold a query (#856).
template <class G, class S>
inline constexpr bool kBf16KvLowRowsOnly =
    S::kFixedWidth == 1 && S::kWarpsQ == 1 && G::GroupSize <= 8;

template <class G, class S>
struct alignas(16) Bf16KvGroupedStorage {
    static constexpr int kStateRows = S::kWarpsKV > 1 ? S::kWarpsKV* S::kQueryRows : 1;
    // kBf16KvLowRowsOnly: the query rows 0-7 in the MMA's A form after the K and V tiles, read
    // every key tile; the other instances keep Q in registers. 4 KiB at D=256: the CTAs an SM
    // holds stay those of the 32 KiB K/V tiles on every arch (2 on sm_86/89/120, 1 on sm_75).
    static constexpr int kQueryElements = kBf16KvLowRowsOnly<G, S> ? 8 * G::kHeadDim : 0;

    union {
        std::uint16_t qkv[2 * S::kKeyRows * G::kHeadDim + kQueryElements];
        float reduction[S::kWarpsKV > 1 ? kStateRows* G::kHeadDim : 1];
    };

    // A rolling page window, not a bound on the KV partition length.
    static constexpr int kPageWindow = 32;
    int pages[kPageWindow];
    float maximum[kStateRows];
    float sum[kStateRows];
};

template <class Geometry, class Schedule>
inline constexpr int bf16_kv_tiled_shared_bytes =
    (Schedule::kQueryRows + 2 * Schedule::kKeyRows) * Geometry::kHeadDim * 2;


} // namespace ninferno::ops::detail
