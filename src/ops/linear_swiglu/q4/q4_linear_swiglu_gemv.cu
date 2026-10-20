#include "core/weight.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "core/device.h" // CUDA_CHECK
#include "core/device_class.h"
#include "core/measured_boards.h"
#include "ops/linear/q4/q4_ksplit_mma.cuh"
#include "ops/linear/q4/q4_sliced_k_launch.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <array>
#include <utility>

namespace ninferno::ops::detail {
namespace {

constexpr int kN                 = 34816;
constexpr int kK                 = 5120;
constexpr int kIntermediate      = kN / 2;
constexpr int kGroupK            = 64;
constexpr int kGroups            = kK / kGroupK;
constexpr int kBytesPerGroup     = 32;
constexpr int kVecBytes          = 16;
constexpr int kGroupsPerWarpTile = 16;
constexpr int kVecsPerWarpTile   = kGroupsPerWarpTile * kBytesPerGroup / kVecBytes;
constexpr int kWarpsPerBlock     = 4;
constexpr int kBlockThreads      = kWarpsPerBlock * 32;
constexpr int kPairsPerBlock     = kWarpsPerBlock;
constexpr int kXVecs             = kK / 8; // x as uint4 (8 bf16 each)
constexpr int kTiles             = kGroups / kGroupsPerWarpTile;
static_assert(kIntermediate % kPairsPerBlock == 0);
static_assert(kBytesPerGroup == 2 * kVecBytes);
static_assert(kGroups % kGroupsPerWarpTile == 0);
static_assert(kVecsPerWarpTile == 32);

// RowTiles 16-row MMA tiles per CTA, each 8 gate rows and their 8 up rows: one staged activation
// slab feeds all of them.
template <int RowTiles>
struct Q4SwiGluSmallTRows {
    static constexpr int kOutputRowsPerCta = 8 * RowTiles;

    __host__ __device__ int output_rows(int rows) const { return rows / 2; }

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row >> 4) * 8 + (local_row & 7) +
               ((local_row & 8) != 0 ? kIntermediate : 0);
    }

    __device__ __forceinline__ int fragment_row(int output_row0, int mma_row, int gid) const {
        return output_row0 + mma_row * 8 + gid;
    }
};

struct Q4SwiGluSmallTEpilogue {
    template <class Output>
    __device__ __forceinline__ void store_fragment(const Output& output, int row, int token,
                                                   float4 projected, int rows,
                                                   int token_end) const {
        if (row >= rows / 2) return;
        if (token < token_end) output.store(row, token, silu(projected.x) * projected.z);
        if (token + 1 < token_end) output.store(row, token + 1, silu(projected.y) * projected.w);
    }
};

using SmallTLauncher = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// The tile per width: R16 rows and eight K-warps through 8 columns, R32 rows from the 16-column
// tile on. Cold, 27B shape, us at T = 12 / 16 / 24 / 32, R16 against R32 (both eight K-warps, one
// stage): RTX 4090 123.9 / 132.1 / 171.1 / 225.3 against 119.8 / 121.9 / 154.6 / 177.2, RTX 5090
// 94.2 / 98.3 / 127.0 / 172.0 against 86.0 / 88.1 / 108.5 / 129.0; at T <= 8 R16 is ahead
// (0.91..1.00). R32 with four K-warps is better still on the 5090 from T = 20 (1.23..1.47) but
// loses to R16 on the 4090 at 20 and 24. Turing, whose MMA takes BF16 operands through FP16 (two k8
// MMAs a k16), wants R32 rows on four K-warps at every width: RTX 2080 Ti T = 4 / 16 / 32 at 933.7
// / 1043.1 / 2513.9 us on R16 W8, 669.7 / 769.1 / 1164.9 on R32 W4 (u89), where R32 W8 lost 10 % at
// T = 28. From 24 columns the k-split tile (variant 5) is ahead where BF16 MMA is native: serve
// Qwen3.8-27B DFlash2 C4 (24-column verify) RTX 3090 59.1 -> 62.2 t/s, RTX 5090 130.4 -> 137.5,
// RTX 4090 even; at 16 columns (MTP3 C4) it lost 2 % on the 3090 and at 8 (C1) 1.5..2 %.
// The boards where the k-split tile measured ahead from one column (below); an Ampere board
// takes it by the nearest of them.
constexpr measured::Gate kAmpereKSplit[] = {
    {measured::kA100Sxm40, true}, {measured::kRtx3090, true}, {measured::kA10, true}};

int small_t_variant(int active_cols) {
    if (!device_class().features.mma_bf16_native) return 4;
    if (active_cols >= 24) return 5;
    // A100 (ga100): the k-split tile from 1 column too - MTP3 C1 (T = 4) 97.5 us on R16 W8 against
    // 91.6 us for ninfer-all's k-split, same products and reduction order. GA102 (3090, A10) too:
    // 3090 MTP3 C1 +2.4 %, C4 +4.5 % with the Q8 W8 tile, A10 C1 +2-3 %.
    if (active_cols <= 8 && measured::measured_gate(kAmpereKSplit)) return 5;
    return active_cols >= 16 ? 1 : 0;
}

template <int ActiveCols, int BlockRows, int KWarps, int Stages, int MinBlocks>
void launch_small_t_tile(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    using Schedule =
        Q4A16SlicedKMmaSchedule<BlockRows, (ActiveCols + 7) / 8 * 8, KWarps, Stages, Cache::cg,
                                Cache::ca, MinBlocks, kK, ActiveCols, Q4SlicedKReduction::Pairwise>;
    launch_q4_a16_sliced_k_mma<Schedule>(
        q4_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kIntermediate},
        Q4SwiGluSmallTEpilogue{}, stream, Q4SwiGluSmallTRows<BlockRows / 16>{});
}

// The k-split tile (ops/linear/q4/q4_ksplit_mma.cuh, ninfer-all's SwiGLU small-T schedule): a
// 16-row tile is eight gate and the matching eight up rows, K split across the warps, and from 16
// columns two or four row tiles share each staged activation slab (two tiles per warp at 24, each
// B fragment feeding both).
struct Q4SwiGluKSplitRows {
    static constexpr int kOutputRowsPerTile = 8;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row & 7) + (local_row >= 8 ? kIntermediate : 0);
    }
};

struct Q4SwiGluKSplitEpilogue {
    __nv_bfloat16* out;
    int columns;

    template <int ActiveCols>
    __device__ __forceinline__ void store(int row, int col0, float4 projected) const {
        if (col0 < columns) {
            out[static_cast<std::int64_t>(col0) * kIntermediate + row] =
                __float2bfloat16_rn(silu(projected.x) * projected.z);
        }
        if (col0 + 1 < columns) {
            out[static_cast<std::int64_t>(col0 + 1) * kIntermediate + row] =
                __float2bfloat16_rn(silu(projected.y) * projected.w);
        }
    }
};

template <int TileCols>
struct Q4SwiGluKSplitTile {
    static constexpr int kKWarps       = TileCols <= 8 ? 8 : (TileCols <= 24 ? 4 : 2);
    static constexpr int kStages       = TileCols <= 16 ? 1 : 2;
    static constexpr int kTilesPerWarp = TileCols == 24 ? 2 : 1;
    static constexpr int kMinBlocks    = kKWarps < 8 ? 2 : (TileCols <= 16 ? 6 : 4);
};

template <int TileCols, bool Masked>
void launch_small_t_ksplit_cols(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    using Tile            = Q4SwiGluKSplitTile<TileCols>;
    using Layout          = SmallTLayout<Tile::kKWarps, Tile::kTilesPerWarp>;
    constexpr int kBlocks =
        kIntermediate / (Q4SwiGluKSplitRows::kOutputRowsPerTile * Layout::kRowTiles);
    const Q4SwiGluKSplitEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data), x.ne[1]};
    q4_ksplit_mma_launch<Q4LinearGeometry<kIntermediate, kK>, TileCols, TileCols,
                         Q4SwiGluKSplitEpilogue, Q4SwiGluKSplitRows, Masked, Tile::kKWarps,
                         Tile::kStages, Tile::kTilesPerWarp, Tile::kMinBlocks>(
        kBlocks, stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), epilogue, Q4SwiGluKSplitRows{}, x.ne[1]);
}

// A width that fills its tile drops the runtime column count (2-4 % at 16 and 24 in ninfer-all);
// eight columns keeps it, the exact instantiation spills under the six-block bound.
template <int TileCols>
void launch_small_t_ksplit(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if constexpr (TileCols > 8) {
        if (x.ne[1] == TileCols) return launch_small_t_ksplit_cols<TileCols, false>(x, w, out, stream);
    }
    launch_small_t_ksplit_cols<TileCols, true>(x, w, out, stream);
}

template <int ActiveCols>
void launch_small_t_active(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    // Wider tiles are shared-memory limited before six CTAs can be resident.
    // Match launch bounds to that limit instead of forcing register spills.
    constexpr int kMinBlocks = ActiveCols <= 8    ? 6
                               : ActiveCols <= 16 ? 4
                               : ActiveCols <= 24 ? 3
                                                  : 2;
    switch (small_t_variant(ActiveCols)) {
    case 1:
        return launch_small_t_tile<ActiveCols, 32, 8, 1, ActiveCols <= 16 ? 3 : 2>(x, w, out,
                                                                                   stream);
    case 2:
        return launch_small_t_tile<ActiveCols, 32, 4, 2, 4>(x, w, out, stream);
    case 3:
        return launch_small_t_tile<ActiveCols, 16, 4, 2, 4>(x, w, out, stream);
    case 4:
        return launch_small_t_tile<ActiveCols, 32, 4, 1, 4>(x, w, out, stream);
    case 5:
        return launch_small_t_ksplit<ActiveCols>(x, w, out, stream);
    default:
        return launch_small_t_tile<ActiveCols, 16, 8, 1, kMinBlocks>(x, w, out, stream);
    }
}

template <std::size_t... Offsets>
constexpr auto make_small_t_launchers(std::index_sequence<Offsets...>) {
    return std::array<SmallTLauncher, sizeof...(Offsets)>{
        &launch_small_t_active<8 * (1 + static_cast<int>(Offsets))>...};
}

constexpr auto kSmallTLaunchers = make_small_t_launchers(std::make_index_sequence<4>{});

__device__ __forceinline__ void q4_issue_pair_tile(uint4 (*__restrict__ s_code)[kVecsPerWarpTile],
                                                   uint4 (*__restrict__ s_scale)[2],
                                                   const std::uint8_t* __restrict__ gate_code_row,
                                                   const std::uint8_t* __restrict__ gate_scale_row,
                                                   const std::uint8_t* __restrict__ up_code_row,
                                                   const std::uint8_t* __restrict__ up_scale_row,
                                                   int tile, int lane) {
    const int g0 = tile * kGroupsPerWarpTile;
    pipe_copy<16>(&s_code[0][lane],
                  reinterpret_cast<const uint4*>(gate_code_row + g0 * kBytesPerGroup) + lane);
    pipe_copy<16>(&s_code[1][lane],
                  reinterpret_cast<const uint4*>(up_code_row + g0 * kBytesPerGroup) + lane);
    if (lane < 2) {
        pipe_copy<16>(&s_scale[0][lane],
                      reinterpret_cast<const uint4*>(gate_scale_row + g0 * 2) + lane);
        pipe_copy<16>(&s_scale[1][lane],
                      reinterpret_cast<const uint4*>(up_scale_row + g0 * 2) + lane);
    }
    pipe_commit();
}

__global__ void q4_linear_swiglu_gemv_pair_kernel(const __nv_bfloat16* __restrict__ x,
                                                  const std::uint8_t* __restrict__ codes,
                                                  const std::uint8_t* __restrict__ scales,
                                                  __nv_bfloat16* __restrict__ out) {
    constexpr int kStages   = 3;
    constexpr int kPrefetch = kStages - 1;
    __shared__ __align__(16) __nv_bfloat16 x_sh[kK];
    __shared__ uint4 code_tile[kWarpsPerBlock][kStages][2][kVecsPerWarpTile];
    __shared__ uint4 scale_tile[kWarpsPerBlock][kStages][2][2];

    auto* x_sh_v    = reinterpret_cast<uint4*>(x_sh);
    const auto* x_g = reinterpret_cast<const uint4*>(x);
    for (int i = static_cast<int>(threadIdx.x); i < kXVecs; i += static_cast<int>(blockDim.x)) {
        x_sh_v[i] = x_g[i];
    }
    __syncthreads();

    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const int warp    = static_cast<int>(threadIdx.x) >> 5;
    const int out_row = static_cast<int>(blockIdx.x) * kPairsPerBlock + warp;

    const std::uint8_t* gate_code_row =
        codes + static_cast<std::int64_t>(out_row) * kGroups * kBytesPerGroup;
    const std::uint8_t* gate_scale_row = scales + static_cast<std::int64_t>(out_row) * kGroups * 2;
    const std::uint8_t* up_code_row =
        codes + static_cast<std::int64_t>(out_row + kIntermediate) * kGroups * kBytesPerGroup;
    const std::uint8_t* up_scale_row =
        scales + static_cast<std::int64_t>(out_row + kIntermediate) * kGroups * 2;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x_sh);

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < kTiles) {
            q4_issue_pair_tile(code_tile[warp][p], scale_tile[warp][p], gate_code_row,
                               gate_scale_row, up_code_row, up_scale_row, p, lane);
        } else {
            pipe_commit();
        }
    }

#pragma unroll 1
    for (int tile = 0; tile < kTiles; ++tile) {
        const int fetch = tile + kPrefetch;
        if (fetch < kTiles) {
            const int buf = fetch % kStages;
            q4_issue_pair_tile(code_tile[warp][buf], scale_tile[warp][buf], gate_code_row,
                               gate_scale_row, up_code_row, up_scale_row, fetch, lane);
        } else {
            pipe_commit();
        }
        pipe_wait<kPrefetch>();
        __syncwarp();

        const int buf           = tile % kStages;
        const auto* gate_codes  = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][0]);
        const auto* up_codes    = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][1]);
        const auto* gate_scales = reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][0]);
        const auto* up_scales   = reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][1]);
        // Eight weights per lane per step, as in the T = 1 Q5 GEMV (q5_a16_gemv.cuh): one 4-byte gate
        // word, one 4-byte up word, two scales and one 16-byte activation load make sixteen weights,
        // against five loads for four in a byte-per-lane loop; the loop is bound by memory-pipe
        // issue, not DRAM. Eight lanes cover a group's 32 code bytes, so 32 lanes cover four groups
        // and the word index is step * 32 + lane (conflict-free). Code byte b holds weights 2b / 2b + 1,
        // so word `pos` of a group is weights 8 pos .. 8 pos + 7 and nibble pair `pair` (bits 4 pair
        // and 16 + 4 pair) is weights pair / pair + 4. The nibble with its sign bit flipped is OR-ed
        // into the mantissa of half 1024.0 and 1032.0 subtracted, which is exactly sign_extend<4>;
        // the group scale is applied once to each group's sum.
        const auto* gate_words = reinterpret_cast<const std::uint32_t*>(gate_codes);
        const auto* up_words   = reinterpret_cast<const std::uint32_t*>(up_codes);
        const int sub          = lane >> 3;
        const int pos          = lane & 7;
        const __half2 bias     = __half2half2(__ushort_as_half(0x6408)); // 1032.0
#pragma unroll
        for (int step = 0; step < kGroupsPerWarpTile / 4; ++step) {
            const int tile_group = step * 4 + sub;
            const float gate_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(gate_scales[tile_group])));
            const float up_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(up_scales[tile_group])));
            const std::uint32_t gate_word = gate_words[tile_group * 8 + pos] ^ 0x88888888u;
            const std::uint32_t up_word   = up_words[tile_group * 8 + pos] ^ 0x88888888u;
            const int k0   = (tile * kGroupsPerWarpTile + tile_group) * kGroupK + pos * 8;
            const uint4 xv = load_vec<uint4>(reinterpret_cast<const uint4*>(x2 + (k0 >> 1)));
            const float2 x0 = bf16x2_bits_to_float2(xv.x), x1 = bf16x2_bits_to_float2(xv.y);
            const float2 x2v = bf16x2_bits_to_float2(xv.z), x3 = bf16x2_bits_to_float2(xv.w);
            const float xs[8]{x0.x, x0.y, x1.x, x1.y, x2v.x, x2v.y, x3.x, x3.y};
            float gate_group = 0.0f;
            float up_group   = 0.0f;
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const float2 g = __half22float2(__hsub2(
                    half2_from_bits(((gate_word >> (4 * pair)) & 0x000f000fu) | 0x64006400u), bias));
                const float2 u = __half22float2(__hsub2(
                    half2_from_bits(((up_word >> (4 * pair)) & 0x000f000fu) | 0x64006400u), bias));
                gate_group = fmaf(g.x, xs[pair], gate_group);
                gate_group = fmaf(g.y, xs[pair + 4], gate_group);
                up_group   = fmaf(u.x, xs[pair], up_group);
                up_group   = fmaf(u.y, xs[pair + 4], up_group);
            }
            gate_acc = fmaf(gate_group, gate_scale, gate_acc);
            up_acc   = fmaf(up_group, up_scale, up_acc);
        }
        __syncwarp();
    }

    gate_acc = warp_reduce_sum(gate_acc);
    up_acc   = warp_reduce_sum(up_acc);
    if (lane == 0) { out[out_row] = __float2bfloat16(silu(gate_acc) * up_acc); }
}

} // namespace

void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream) {
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("q4 linear_swiglu GEMV requires weight [34816,5120]");
    }
    const int grid = kIntermediate / kPairsPerBlock;
    q4_linear_swiglu_gemv_pair_kernel<<<grid, kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void q4_linear_swiglu_small_t_tiled_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream) {
    if (x.ne[1] < 2 || x.ne[1] > 32) {
        throw std::invalid_argument("Q4 LinearSwiGLU exact small-T requires T=2..32");
    }
    kSmallTLaunchers[static_cast<std::size_t>((x.ne[1] - 1) / 8)](x, w, out, stream);
}

} // namespace ninferno::ops::detail
