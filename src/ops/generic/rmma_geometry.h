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
//   the decoder's own table (the IQ codebooks): kTableWords words, a constant of the decoder.
//
// The decoder constants the geometry reads (kRun, kTableWords, kStageActs, kGroupTail) are rows of
// kDecoderGeometry, keyed by the kernel-facts spelling of the decoder; generic_gemm.cuh static_
// asserts every row against its decoder, and the kernel's activation staging is decided by
// rmma_stages_activations below, one predicate for the kernel and the host.

#include <cstddef>
#include <string_view>

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

// ---- The decoders' constants, as host-visible rows (static_asserted by generic_gemm.cuh).

struct DecoderGeometry {
    std::string_view name; // the kernel-facts spelling (kernel_facts.h decoder_name)
    int run;               // kRun: values a lane reads a step; J = run / 4 MMAs a K step
    int table_words;       // kTableWords: the codebook staged in shared memory (0: none)
    bool stage_acts;       // kStageActs: a SwiGLU step copies its activations ahead (cp.async)
    bool group_tail;       // kGroupTail: K of whole scale groups (never stages)
};

// The decoders with a table, staging or a group tail; every other decoder has none of the three.
inline constexpr DecoderGeometry kDecoderGeometry[] = {
    {"GgmlIq1MDecoder", 32, 2048, false, false},    {"GgmlIq1SDecoder", 32, 2048, false, false},
    {"GgmlIq2SDecoder", 32, 2048, false, false},    {"GgmlIq2XsDecoder", 32, 1024, false, false},
    {"GgmlIq2XxsDecoder", 32, 512, false, false},   {"GgmlIq3SDecoder", 32, 512, false, false},
    {"GgmlIq3XxsDecoder", 32, 256, false, false},   {"GgmlQ3KDecoder", 32, 0, true, false},
    {"GgmlQ6KDecoder", 32, 0, true, false},         {"GgmlQ6KWideDecoder", 32, 0, true, false},
    {"GgmlQ45KQuadDecoder<4>", 32, 0, true, false}, {"GgmlQ45KQuadDecoder<5>", 32, 0, true, false},
    {"Ggml32Decoder<0>", 32, 0, false, true},       {"Ggml32Decoder<1>", 32, 0, false, true},
    {"Ggml32Decoder<2>", 32, 0, false, true},       {"Ggml32Decoder<3>", 32, 0, false, true},
    {"Ggml32Decoder<4>", 32, 0, false, true},       {"Mxfp4SpreadDecoder", 32, 0, false, true},
};

constexpr DecoderGeometry decoder_geometry(std::string_view name) noexcept {
    for (const DecoderGeometry& row : kDecoderGeometry) {
        if (row.name == name) return row;
    }
    return {name, 0, 0, false, false};
}

// The activation staging of a K-quant SwiGLU step (rmma_kernel's kActStage): the decoder stages,
// the kernel is the latency variant of a SwiGLU epilogue over one column tile, at most eight warps.
[[nodiscard]] NINFERNO_GEOMETRY_HD constexpr bool
rmma_stages_activations(bool latency, bool decoder_stages, bool swiglu, int n_tiles, int warps,
                        bool group_tail) noexcept {
    return latency && decoder_stages && swiglu && n_tiles == 1 && warps <= 8 && !group_tail;
}

// What the build carries: cp.async is sm_80 and up (ops/common/memory.cuh NINFERNO_HAS_CP_ASYNC;
// the sm_70 and sm_75 builds copy synchronously, so nothing is staged), the NVFP4 kernels and the
// cluster hand-over with them are the full build's (CMakeLists.txt NINFERNO_FULL_CUDA_ARCHS, which
// the geometry test reads): the caller passes `with_nvfp4` from that list.
[[nodiscard]] constexpr bool build_has_cp_async(int compute_capability) noexcept {
    return compute_capability >= 80;
}

// The static shared memory of one register-MMA instance of the named decoder on a build of
// `compute_capability` (70, 75, 80, ...): the sum of the pieces above.
[[nodiscard]] constexpr std::size_t
rmma_decoder_static_shared_bytes(std::string_view decoder, int warps_m, int warps_k, int m_tiles,
                                 int n_tiles, int cluster, bool swiglu, bool latency,
                                 int compute_capability, bool with_nvfp4) noexcept {
    const DecoderGeometry g = decoder_geometry(decoder);
    const bool staged       = build_has_cp_async(compute_capability) &&
                        rmma_stages_activations(latency, g.stage_acts, swiglu, n_tiles,
                                                warps_m * warps_k, g.group_tail);
    return rmma_static_shared_bytes(warps_m, warps_k, rmma_sets(swiglu), m_tiles, n_tiles, cluster,
                                    with_nvfp4, staged ? g.run / 4 : 0,
                                    static_cast<std::size_t>(g.table_words) * sizeof(unsigned));
}

} // namespace ninferno::ops::detail::generic::geometry

#undef NINFERNO_GEOMETRY_HD
