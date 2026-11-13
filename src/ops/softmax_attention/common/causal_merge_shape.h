#pragma once

#include <cstdint>

namespace ninferno::ops::detail {

// The natural merges' column chunk (causal_merge.cuh, dense/causal_cache/bf16/merge.cuh): a CTA
// merges DChunk output columns of one query head and token over every split, so the grid is
// query heads x 256 / DChunk x tokens CTAs, and each CTA streams splits x DChunk x 4 bytes of
// partials through its SM at one SM's fill rate (kernel-proposals.md, entry N). A 256-wide
// schedule whose grid leaves SMs idle (query heads x columns below the SM count: 16 CTAs for a
// decode token of 16 query heads, on every board from 28 SMs up) narrows to 64 columns: four
// times the CTAs, a quarter of the bytes on the most loaded SM. Once the 256-wide grid covers the
// SMs, the most loaded SM's bytes no longer fall (ceil(4C / SMs) x 64 against ceil(C / SMs) x 256)
// while every narrow CTA would repeat the head's statistics, so the schedule's chunk stays. A
// merge that stores whole rotated rows (whole_rows: the NVFP4 and K8V4 inverse rotation) keeps
// 256 columns. The chunk changes which CTA sums a column, never the order of its sum: the output
// is the same bits either way.
inline constexpr int kCausalMergeNarrowDChunk = 64;

[[nodiscard]] constexpr int causal_merge_d_chunk(int schedule_d_chunk, bool whole_rows,
                                                 int query_heads, std::int64_t columns, int sms) {
    if (whole_rows || schedule_d_chunk <= kCausalMergeNarrowDChunk) return schedule_d_chunk;
    return static_cast<std::int64_t>(query_heads) * columns < sms ? kCausalMergeNarrowDChunk
                                                                  : schedule_d_chunk;
}

} // namespace ninferno::ops::detail
