#pragma once
#include "ops/common/arch_limits.h"
#include "ops/softmax_attention/common/causal_geometry.h"

namespace ninferno::ops::detail {
// NINFERNO-SPECIAL(arch:sm75): the grouped kernel's P tile keeps 64 columns for a 32-key tile (its
// swizzle folds the row over eight 8-column chunks), which leaves 64 query rows with a 32-key stage
// at 65 KiB, over Turing's 64 KiB. A build under 99 KiB keeps 32 columns there and folds the row
// over the tile's own four chunks; the tuned layout stays everywhere else.
inline constexpr bool kNvfp4KvCompactP = kMaxSharedBytesPerBlock < 99 * 1024;

// The grouped kernel's shared memory: Q, the K/V code and FP16 stage, P, the row rescales and the
// group scales (the kernel asserts the same sum).
template <int RowTiles, int Keys, bool CompactKV>
inline constexpr int kNvfp4KvGroupedSharedBytes =
    RowTiles * 16 * 256 * 2 + (CompactKV ? 3 : 5) * Keys * 256 +
    RowTiles * 16 * (Keys == 32 && !kNvfp4KvCompactP ? 64 : Keys) * 2 + RowTiles * 16 * 4 +
    2 * Keys * 16;

template <int Tokens, int Warps, int Keys, int MinBlocks, bool Compact, bool Dynamic = true>
struct Nvfp4KvGroupedMmaSchedule {
    static_assert(Tokens > 0 && Warps > 0 && Warps <= 16 && MinBlocks > 0);
    static_assert(Keys == 32 || Keys == 64);
    static constexpr int kTokenTile = Tokens, kWarps = Warps, kThreads = Warps * 32;
    static constexpr int kKeyRows = Keys, kMinBlocks = MinBlocks;
    static constexpr bool kCompactKVStage = Compact, kDynamicArena = Dynamic;
    static constexpr int kArenaBytes = (Compact ? 3 : 5) * Keys * 256;
};

// Four consumer warps each own 16 query rows; producer warps decode K/V concurrently.
// Register redistribution requires this kernel's non-RDC translation unit.
template <int Keys = 64, int ProducerWarps = 12, int ProducerRegisters = 40,
          int ConsumerRegisters = 232>
struct Nvfp4KvTiledMmaSchedule {
    static_assert(Keys == 32 || Keys == 64);
    static_assert(ProducerWarps >= 4 && ProducerWarps <= 12 && ProducerWarps % 4 == 0);
    static_assert(ProducerRegisters >= 24 && ProducerRegisters % 8 == 0);
    static_assert(ConsumerRegisters >= 128 && ConsumerRegisters <= 256 &&
                  ConsumerRegisters % 8 == 0);
    static_assert(ProducerWarps * 32 * ProducerRegisters + 128 * ConsumerRegisters <= 65536);
    static constexpr int kQueryRows = 64, kKeyRows = Keys;
    static constexpr int kProducerWarps = ProducerWarps, kConsumerWarps = 4;
    static constexpr int kProducerThreads = ProducerWarps * 32, kConsumerThreads = 128;
    static constexpr int kThreads           = kProducerThreads + kConsumerThreads;
    static constexpr int kProducerRegisters = ProducerRegisters,
                         kConsumerRegisters = ConsumerRegisters;
    static constexpr int kSharedBytes       = (kQueryRows + 2 * Keys) * 256 * 2 + 32;
};

struct Nvfp4KvMergeSchedule {
    static constexpr int kDChunk = 256, kThreads = 256;
};
} // namespace ninferno::ops::detail
