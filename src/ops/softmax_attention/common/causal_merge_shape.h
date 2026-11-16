#pragma once

#include <cstdint>

namespace ninferno::ops::detail {

// The natural merges' column chunk (causal_merge.cuh, dense/causal_cache/bf16/merge.cuh). A CTA
// merges DChunk output columns of one query head and token over every split, so the grid is
// query heads x 256 / DChunk x tokens CTAs of 256 threads (kernel-proposals.md, entry O).
//
// The chunk is chosen by the law of entry O, Variant 3: the time of the most loaded SM at each
// chunk, from two of its terms.
// - Issue. Every CTA runs the same fixed work whatever its chunk: the prologue, the two-stage
//   fetch, the head's statistics, a stage's store and the epilogue. The SASS of the W = 1
//   instances counts it at kCausalMergeFixedWarpInstructions a warp, and 8 warps a CTA. Each
//   further stage adds kCausalMergeStageWarpInstructions a warp, and each split
//   kCausalMergeSplitInstructions a warp that owns columns (DChunk / 32 warps). An SM issues at
//   most 4 warp instructions a clock, so its k CTAs take at least k x instructions / 4 clocks.
// - Bytes. Its k CTAs stream k x splits x DChunk x 4 bytes at one SM's fill rate of 32 B/clk
//   (sm_rate_facts.json).
// k = ceil(CTAs / SMs). The narrow chunk divides the bytes and the chain by four, but every narrow
// CTA repeats the fixed work. So it pays only where each narrow CTA has an SM to itself, or where
// the wide chunk's bytes cost more than the fixed work repeated on a shared SM.
//
// The RTX 3060 refuted the bytes-only rule this replaced (#867): at a decode token it ran 64
// narrow CTAs on 28 SMs, three a SM, and was slower than at 256 columns. With the law, a decode
// token of 16 query heads narrows on the boards from 64 SMs up and keeps 256 columns below that.
// The L2 round trip that both chunks share is left out, so a tie within it can fall either way.
// A merge that stores whole rotated rows (whole_rows: the NVFP4 and K8V4 inverse rotation) keeps
// 256 columns. The chunk changes which CTA sums a column, never the order of its sum: the output
// is the same bits either way.
inline constexpr int kCausalMergeNarrowDChunk = 64;
// SASS counts, per warp, of the 256-thread merge. Their source (#867):
// - the instances: causal_natural_merge_kernel<CausalGeometry<256, 16, 4>, D, false, false,
//   false, false> at D = 64 and 256 (the decode-token form; both chunks give the same counts);
// - the commands: `nvcc -std=c++20 -O3 -cubin -arch=sm_XX` (CUDA 13.2) of a file that takes their
//   addresses, then `cuobjdump -sass` (CUDA 13.1);
// - fixed work: the static instructions from the entry to the end of the first stage's refetch
//   (the twelfth LDG.E.128), plus those from the last barrier to the final EXIT. That is
//   sm_75 500, sm_86 497, sm_89 497, sm_120a 391;
// - a further stage: from the barrier after the first chain to the sixteenth LDG.E.128. That is
//   sm_75 65, sm_86 74, sm_89 74, sm_120a 67, rounded to 70;
// - a split: the chain's x16 body is 60 instructions (weight, test, value, FFMA), 3.75 a split,
//   rounded up to 4.
// One F = 500 serves every architecture, the largest count, so a board is never priced as
// narrowing more cheaply than its SASS says. sm_120a's own 391 is deliberately not used: on the
// RTX 5090 and 5070 Ti it changes no decision at W 1-11 with the wave partition's splits. At
// W = 1 a board's decision moves only for F under 210 (RTX 4070 Ti; 119 on the 4060 Ti, 61 on the
// 3060), far below every count. These counts are not yet in the kernel-facts snapshots
// (tools/kernel_facts, ninferno_kernel_facts_test). That tool reads the register-MMA kernels only,
// and adding the merges needs its kernel set and a snapshot per architecture, which is not a small
// change. Until then the counts are pinned here by hand.
inline constexpr std::int64_t kCausalMergeFixedWarpInstructions = 500;
inline constexpr std::int64_t kCausalMergeStageWarpInstructions = 70;
inline constexpr std::int64_t kCausalMergeSplitInstructions     = 4;
// One stage: CausalMergeColumns::kStageFloats.
inline constexpr std::int64_t kCausalMergeStageFloats = 4096;

// The most loaded SM's clocks at `d_chunk` columns, by the issue and bytes terms above. The CTAs
// are counted at 256 columns (ctas_at_256 = query heads x columns), and `splits` is the
// partition's capacity.
[[nodiscard]] constexpr std::int64_t causal_merge_sm_clocks(int d_chunk, std::int64_t ctas_at_256,
                                                            int splits, int sms) {
    const std::int64_t ctas      = ctas_at_256 * (256 / d_chunk);
    const std::int64_t per_sm    = (ctas + sms - 1) / sms;
    const std::int64_t s         = splits < 1 ? 1 : splits;
    const std::int64_t per_stage = kCausalMergeStageFloats / d_chunk;
    const std::int64_t stages    = (s + per_stage - 1) / per_stage;
    const std::int64_t warp_instructions =
        8 * (kCausalMergeFixedWarpInstructions + kCausalMergeStageWarpInstructions * (stages - 1)) +
        (d_chunk / 32) * kCausalMergeSplitInstructions * s;
    const std::int64_t issue = per_sm * warp_instructions / 4;
    const std::int64_t bytes = per_sm * s * d_chunk * 4 / 32;
    return issue > bytes ? issue : bytes;
}

[[nodiscard]] constexpr int causal_merge_d_chunk(int schedule_d_chunk, bool whole_rows,
                                                 int query_heads, std::int64_t columns, int splits,
                                                 int sms) {
    if (whole_rows || schedule_d_chunk != 256 || sms < 1) return schedule_d_chunk;
    const std::int64_t ctas = static_cast<std::int64_t>(query_heads) * columns;
    return causal_merge_sm_clocks(kCausalMergeNarrowDChunk, ctas, splits, sms) <
                   causal_merge_sm_clocks(256, ctas, splits, sms)
               ? kCausalMergeNarrowDChunk
               : schedule_d_chunk;
}

} // namespace ninferno::ops::detail
