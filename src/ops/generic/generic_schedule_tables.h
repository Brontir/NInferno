#pragma once

// The host tables of the generic path's schedules, from the lists the CUDA files instantiate
// (generic_schedules.h), lifted out of generic_gemm.cpp's anonymous namespace so the host part of
// the selector (generic_routes.cpp) reads the same copy (#620).

#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_schedules.h"

#include <array>
#include <cstddef>

namespace ninferno::ops::detail::generic {

// The host tables, from the lists the CUDA files instantiate (generic_schedules.h): one source,
// so a schedule cannot be listed here and missing there, or listed in another order (#307).
template <std::size_t N>
constexpr bool indexed_in_order(const int (&index)[N]) {
    for (std::size_t i = 0; i < N; ++i) {
        if (index[i] != static_cast<int>(i)) { return false; }
    }
    return true;
}

#define NINFERNO_SCHEDULE_INDEX(i, ...) i,
static_assert(indexed_in_order({NINFERNO_GENERIC_RMMA_SCHEDULES(NINFERNO_SCHEDULE_INDEX)}));
static_assert(indexed_in_order({NINFERNO_GENERIC_W4A4_SCHEDULES(NINFERNO_SCHEDULE_INDEX)}));
static_assert(indexed_in_order({NINFERNO_GENERIC_W4A8_SCHEDULES(NINFERNO_SCHEDULE_INDEX)}));
static_assert(indexed_in_order({NINFERNO_GENERIC_R8_SCHEDULES(NINFERNO_SCHEDULE_INDEX)}));
static_assert(indexed_in_order({NINFERNO_GENERIC_TILED_SCHEDULES(NINFERNO_SCHEDULE_INDEX)}));
static_assert(indexed_in_order({NINFERNO_GENERIC_STAGED_SCHEDULES(NINFERNO_SCHEDULE_INDEX)}));
#undef NINFERNO_SCHEDULE_INDEX

// warps_m, warps_k, m_tiles, n_tiles, cluster.
#define NINFERNO_RMMA_INFO(i, wm, wk, mt, nt, cl) RmmaScheduleInfo{wm, wk, mt, nt, cl},
inline constexpr std::array kRmmaSchedules{NINFERNO_GENERIC_RMMA_SCHEDULES(NINFERNO_RMMA_INFO)};
#undef NINFERNO_RMMA_INFO

// block_m, block_n.
#define NINFERNO_W4A4_INFO(i, bm, bn, wm, wn, st) W4a4ScheduleInfo{bm, bn},
inline constexpr std::array kW4a4Schedules{NINFERNO_GENERIC_W4A4_SCHEDULES(NINFERNO_W4A4_INFO)};
#undef NINFERNO_W4A4_INFO

// block_m (tokens), block_n (rows), producer warps (0: every warp loads and multiplies).
#define NINFERNO_W4A8_INFO(i, bm, bn, wm, wn, pr) W4a8ScheduleInfo{bm, bn, pr},
inline constexpr std::array kW4a8Schedules{NINFERNO_GENERIC_W4A8_SCHEDULES(NINFERNO_W4A8_INFO)};
#undef NINFERNO_W4A8_INFO

// Warps per CTA, rows per warp.
#define NINFERNO_R8_INFO(i, w, r) R8ScheduleInfo{w, r},
inline constexpr std::array kR8Schedules{NINFERNO_GENERIC_R8_SCHEDULES(NINFERNO_R8_INFO)};
#undef NINFERNO_R8_INFO

// block_m (tokens), block_n (rows).
#define NINFERNO_TILED_INFO(i, bm, bn, wm, wn, st) TiledScheduleInfo{bm, bn},
inline constexpr std::array kTiledSchedules{NINFERNO_GENERIC_TILED_SCHEDULES(NINFERNO_TILED_INFO)};
#undef NINFERNO_TILED_INFO

// Rows, tokens, warps_m, warps_n, stages. 6 (48 KiB) runs two CTAs per SM, for grids of fewer
// tiles than twice the SMs; 9 (64 KiB) two where an SM holds 128 KiB; 10-12 (144-160 KiB) only
// where one block holds them (GA100, Orin).
#define NINFERNO_STAGED_INFO(i, r, t, wm, wn, st) StagedScheduleInfo{r, t, wm, wn, st},
inline constexpr std::array kStagedSchedules{
    NINFERNO_GENERIC_STAGED_SCHEDULES(NINFERNO_STAGED_INFO)};
#undef NINFERNO_STAGED_INFO
static_assert(kRmmaSchedules.size() == 20 && kW4a4Schedules.size() == 4 &&
              kW4a8Schedules.size() == 6 && kR8Schedules.size() == 4 &&
              kTiledSchedules.size() == 3 && kStagedSchedules.size() == 13);

// The NVFP4 tiled schedules (generic_nvfp4_tiled.cu dispatches by classes, no list of its own):
// classes (the register MMA's WarpsK whose order it reproduces), block_m (tokens), block_n (rows).
inline constexpr std::array<Nvfp4TiledScheduleInfo, 3> kNvfp4TiledSchedules{{
    {1, 64, 128},
    {4, 64, 128},
    {8, 64, 128},
}};

} // namespace ninferno::ops::detail::generic
