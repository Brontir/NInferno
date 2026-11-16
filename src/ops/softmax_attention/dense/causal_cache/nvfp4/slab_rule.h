#pragma once

// Which grouped NVFP4 instances the slab kernel serves (slab_mma.cuh, #849), and the CTAs per SM
// each instance keeps resident: facts of the schedule that the plan's wave partition reads
// (plan.cpp) and the instances assert (instances.h). Host and device code alike.

#include "ops/common/arch_limits.h"

namespace ninferno::ops::detail {

// Eight warps, each owning a 16-key slab of every 128 keys, a two-stage cp.async ring per warp of
// the slab's K and V codes and group scales.
inline constexpr int kNvfp4KvSlabWarps      = 8;
inline constexpr int kNvfp4KvSlabKeys       = 16;
inline constexpr int kNvfp4KvSlabStages     = 2;
inline constexpr int kNvfp4KvSlabStageBytes = kNvfp4KvSlabKeys * 2 * (128 + 16);
inline constexpr int kNvfp4KvSlabArenaBytes =
    kNvfp4KvSlabStages * kNvfp4KvSlabWarps * kNvfp4KvSlabStageBytes;
// One n8 tile of query rows: the 4-8 rows of a GQA decode step (W = 1 on every geometry). Wider
// grouped calls keep the grouped kernel.
inline constexpr int kNvfp4KvSlabMaxRows = 8;

// The slab kernel's shared memory for `rows` query rows: the dynamic ring (which the append
// scratch and the final merge reuse), Q in FP16 and the per-warp row statistics.
constexpr int nvfp4_kv_slab_shared_bytes(int rows) {
    const int padded = (rows + 7) / 8 * 8;
    return kNvfp4KvSlabArenaBytes + padded * 256 * 2 + 2 * kNvfp4KvSlabWarps * padded * 4;
}

// NINFERNO-SPECIAL(arch:sm70,sm75): the slab kernel needs sm_80's cp.async (Turing has movmatrix
// only, Volta neither) and more shared memory than Turing's 64 KiB; sm_70 and sm_75 builds keep
// the grouped kernel until a variant for them (#849 plan design A) has its own law.
#if defined(NINFERNO_BUILD_ARCH) && NINFERNO_BUILD_ARCH >= 80
inline constexpr bool kNvfp4KvSlabArch = true;
#else
inline constexpr bool kNvfp4KvSlabArch = false;
#endif

constexpr bool nvfp4_kv_slab_serves(int rows) {
    return kNvfp4KvSlabArch && rows >= 1 && rows <= kNvfp4KvSlabMaxRows &&
           nvfp4_kv_slab_shared_bytes(rows) <= int(kMaxSharedBytesPerBlock);
}

// The CTAs per SM a grouped instance of `rows` query rows is built for (its schedule's
// kMinBlocks): one for the slab kernel, two for the grouped kernel up to 32 rows, else one.
constexpr int nvfp4_kv_grouped_min_blocks(int rows) {
    if (nvfp4_kv_slab_serves(rows)) return 1;
    return (rows + 15) / 16 <= 2 ? 2 : 1;
}

} // namespace ninferno::ops::detail
