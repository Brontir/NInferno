#pragma once

#include <cstdint>

namespace ninferno::ops::detail {

// The natural merges' column chunk (causal_merge.cuh, dense/causal_cache/bf16/merge.cuh): a CTA
// merges DChunk output columns of one query head and token over every split, so the grid is
// query heads x 256 / DChunk x tokens CTAs, and each CTA streams splits x DChunk x 4 bytes of
// partials through its SM at one SM's fill rate (kernel-proposals.md, entry O). With n = query
// heads x columns CTAs at 256 columns, the most loaded SM streams ceil(n / SMs) x 256 columns a
// split, and at 64 columns ceil(4n / SMs) x 64. A 256-wide schedule narrows to 64 columns only
// where that is fewer bytes within one wave: 4n <= 3 SMs (ceil(4n / SMs) <= 3, so at most 192
// columns against 256). A decode token of 16 query heads narrows on every board from 22 SMs up:
// 64 CTAs, a quarter of the bytes on the most loaded SM from 64 SMs up. Between 3/4 of the SMs and
// a full wave the narrow grid streams the same 256 columns on its most loaded SM at four times the
// CTAs, each repeating the head's statistics. Past a wave the narrow grid can still stream fewer
// bytes (n = SMs + 1: 320 columns against 512), but the registers cap the merge at 4 CTAs a SM,
// so 4n narrow CTAs past 4 x SMs run the statistics again in a second wave: the schedule's
// chunk stays (kernel-proposals.md, entry O, Variant 3). A merge that stores whole rotated rows
// (whole_rows: the NVFP4 and K8V4 inverse rotation) keeps 256 columns. The chunk changes which CTA
// sums a column, never the order of its sum: the output is the same bits either way.
inline constexpr int kCausalMergeNarrowDChunk = 64;

[[nodiscard]] constexpr int causal_merge_d_chunk(int schedule_d_chunk, bool whole_rows,
                                                 int query_heads, std::int64_t columns, int sms) {
    if (whole_rows || schedule_d_chunk <= kCausalMergeNarrowDChunk) return schedule_d_chunk;
    const std::int64_t ctas = static_cast<std::int64_t>(query_heads) * columns;
    return 4 * ctas <= 3 * static_cast<std::int64_t>(sms) ? kCausalMergeNarrowDChunk
                                                          : schedule_d_chunk;
}

} // namespace ninferno::ops::detail
