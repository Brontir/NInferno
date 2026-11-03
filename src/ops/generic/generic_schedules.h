#pragma once

// The schedules of the generic GEMM families, one X-macro list each: X(index, ...). The CUDA
// files instantiate and dispatch from these lists, and generic_gemm.cpp builds its host tables
// (the schedule infos the selectors read) from the same ones, so the two cannot drift (#307).
// Plain macros only: included by host and device code alike.

// Register MMA (generic_gemm_launch.cuh): index, warps_m, warps_k, m_tiles, n_tiles, cluster (K
// split over a thread-block cluster of that many CTAs). 18: K over a CTA pair, for decode calls of
// at most one CTA per SM (heuristic_rmma); 19: K over sixteen warps of one CTA, the same calls
// where clusters do not exist (sm_86, sm_89).
#define NINFERNO_GENERIC_RMMA_SCHEDULES(X)                                                         \
    X(0, 1, 4, 1, 1, 1)                                                                            \
    X(1, 1, 8, 1, 1, 1)                                                                            \
    X(2, 2, 2, 1, 1, 1)                                                                            \
    X(3, 4, 1, 1, 1, 1)                                                                            \
    X(4, 4, 1, 2, 1, 1)                                                                            \
    X(5, 2, 2, 1, 2, 1)                                                                            \
    X(6, 4, 1, 1, 2, 1)                                                                            \
    X(7, 4, 1, 2, 2, 1)                                                                            \
    X(8, 4, 1, 1, 4, 1)                                                                            \
    X(9, 4, 1, 2, 4, 1)                                                                            \
    X(10, 4, 1, 1, 8, 1)                                                                           \
    X(11, 4, 1, 2, 8, 1)                                                                           \
    X(12, 2, 1, 2, 8, 1)                                                                           \
    X(13, 8, 1, 1, 8, 1)                                                                           \
    X(14, 1, 8, 1, 4, 1)                                                                           \
    X(15, 1, 4, 1, 4, 1)                                                                           \
    X(16, 1, 8, 1, 1, 8)                                                                           \
    X(17, 1, 4, 1, 2, 1)                                                                           \
    X(18, 1, 8, 1, 1, 2)                                                                           \
    X(19, 1, 16, 1, 1, 1)

// W4A4 (generic_w4a4.cu): index, block_m, block_n, warps_m, warps_n, stages.
#define NINFERNO_GENERIC_W4A4_SCHEDULES(X)                                                         \
    X(0, 32, 64, 2, 2, 4)                                                                          \
    X(1, 32, 128, 2, 4, 4)                                                                         \
    X(2, 64, 128, 4, 2, 4)                                                                         \
    X(3, 128, 128, 4, 2, 3)

// W4A8 (generic_w4a8.cuh): index, block_m (tokens), block_n (rows), warps_m, warps_n, producer
// warps (0: every warp loads and multiplies).
#define NINFERNO_GENERIC_W4A8_SCHEDULES(X)                                                         \
    X(0, 32, 128, 2, 4, 0)                                                                         \
    X(1, 64, 128, 2, 4, 0)                                                                         \
    X(2, 128, 128, 2, 4, 0)                                                                        \
    X(3, 128, 64, 4, 2, 0)                                                                         \
    X(4, 64, 64, 2, 2, 0)                                                                          \
    X(5, 128, 64, 4, 2, 4)

// R8 (generic_r8.cuh): index, warps per CTA, rows per warp.
#define NINFERNO_GENERIC_R8_SCHEDULES(X)                                                           \
    X(0, 4, 1)                                                                                     \
    X(1, 4, 2)                                                                                     \
    X(2, 8, 2)                                                                                     \
    X(3, 8, 1)

// Tiled MMA (generic_tiled.cu): index, block_m (tokens), block_n (rows), warps_m, warps_n, stages.
#define NINFERNO_GENERIC_TILED_SCHEDULES(X)                                                        \
    X(0, 128, 128, 2, 4, 3)                                                                        \
    X(1, 64, 128, 2, 4, 3)                                                                         \
    X(2, 128, 64, 2, 2, 3)

// Staged MMA (generic_staged.cu): index, rows, tokens, warps_m, warps_n, stages.
#define NINFERNO_GENERIC_STAGED_SCHEDULES(X)                                                       \
    X(0, 128, 128, 2, 4, 3)                                                                        \
    X(1, 128, 64, 2, 2, 4)                                                                         \
    X(2, 64, 128, 1, 4, 4)                                                                         \
    X(3, 128, 128, 2, 2, 3)                                                                        \
    X(4, 256, 128, 4, 2, 2)                                                                        \
    X(5, 64, 64, 2, 2, 4)                                                                          \
    X(6, 128, 64, 2, 2, 2)                                                                         \
    X(7, 64, 128, 1, 4, 2)                                                                         \
    X(8, 64, 64, 2, 2, 3)                                                                          \
    X(9, 128, 128, 2, 4, 2)                                                                        \
    X(10, 128, 128, 2, 4, 5)                                                                       \
    X(11, 256, 128, 4, 2, 3)                                                                       \
    X(12, 128, 256, 2, 4, 3)
