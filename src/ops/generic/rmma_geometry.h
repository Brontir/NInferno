#pragma once

// The register-MMA kernel's static shared memory as constexpr functions of its template
// parameters (#612 step 5): the one spelling the kernel's declarations are static_asserted against
// (generic_gemm.cuh) and the host test checks against the kernel-facts snapshots
// (tests/ops/linear/test_rmma_geometry.cpp), so a change of the kernel's shared arrays that the
// host rules do not follow cannot go unseen (the #594 class).
//
// Plain C++: no CUDA type, includable from host code and from .cu alike.
//
// What the kernel keeps in static shared memory (generic_gemm.cuh rmma_kernel):
//   partial       the warps along K hand their accumulators to warp_k 0: WK - 1 times WM, sets, MT,
//                 NT, 4 and 32 floats, when WK > 1;
//   rank_partial  a cluster's CTAs hand theirs over: WM x sets x MT x NT x 4 x 32 floats, in a
//                 build with the NVFP4 kernels (the cluster path does not exist below sm_120a);
//   act_stage     a K-quant SwiGLU step of one token tile copies its activations ahead with
//                 cp.async: 2 slots x WM x WK warps x J x 32 lanes x 8 bytes;
//   the decoder's own table (the IQ codebooks), a constant of the decoder (decoder_table_bytes).

#include <cstddef>

// A kernel's static_assert calls these from device code.
#ifdef __CUDACC__
#    define NINFERNO_GEOMETRY_HD __host__ __device__
#else
#    define NINFERNO_GEOMETRY_HD
#endif

namespace ninferno::ops::detail::generic::geometry {

inline constexpr std::size_t kLanes = 32;

// A CTA of warps_m x warps_k warps.
[[nodiscard]] NINFERNO_GEOMETRY_HD constexpr int rmma_threads(int warps_m, int warps_k) noexcept {
    return static_cast<int>(kLanes) * warps_m * warps_k;
}

// `sets`: gate rows then the matching up rows (the SwiGLU epilogue) are 2, else 1.
[[nodiscard]] NINFERNO_GEOMETRY_HD constexpr int rmma_sets(bool swiglu) noexcept {
    return swiglu ? 2 : 1;
}

// One float of the accumulator tile (4 elements a lane, 32 lanes) per (set, m tile, n tile) and
// warp that hands it over.
inline constexpr std::size_t kTileBytes = 4 * kLanes * sizeof(float);

[[nodiscard]] NINFERNO_GEOMETRY_HD constexpr std::size_t
rmma_partial_bytes(int warps_m, int warps_k, int sets, int m_tiles, int n_tiles) noexcept {
    return warps_k > 1 ? static_cast<std::size_t>(warps_k - 1) * warps_m * sets * m_tiles *
                             n_tiles * kTileBytes
                       : 0;
}

[[nodiscard]] NINFERNO_GEOMETRY_HD constexpr std::size_t
rmma_rank_partial_bytes(int warps_m, int sets, int m_tiles, int n_tiles, int cluster,
                        bool with_nvfp4) noexcept {
    return cluster > 1 && with_nvfp4
               ? static_cast<std::size_t>(warps_m) * sets * m_tiles * n_tiles * kTileBytes
               : 0;
}

// The staged activations: two slots, every warp's J chunks of 8 bytes a lane.
[[nodiscard]] NINFERNO_GEOMETRY_HD constexpr std::size_t
rmma_act_stage_bytes(int warps_m, int warps_k, int j) noexcept {
    return static_cast<std::size_t>(2) * warps_m * warps_k * j * kLanes * 8;
}

// The whole of it: `stage_j` is 0 where the kernel does not stage its activations.
[[nodiscard]] NINFERNO_GEOMETRY_HD constexpr std::size_t
rmma_static_shared_bytes(int warps_m, int warps_k, int sets, int m_tiles, int n_tiles, int cluster,
                         bool with_nvfp4, int stage_j, std::size_t decoder_table) noexcept {
    return rmma_partial_bytes(warps_m, warps_k, sets, m_tiles, n_tiles) +
           rmma_rank_partial_bytes(warps_m, sets, m_tiles, n_tiles, cluster, with_nvfp4) +
           (stage_j > 0 ? rmma_act_stage_bytes(warps_m, warps_k, stage_j) : 0) + decoder_table;
}

} // namespace ninferno::ops::detail::generic::geometry
