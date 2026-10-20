#pragma once

// NVFP4 x BF16 on tensor cores for wide T (prefill chunks), in the summation order of the A16 MMA
// kernels that split K over Classes warps with Bytes code bytes per lane and K step:
// nvfp4_a16_mma.cuh (the Qwen3.5-27B family, Classes = 8, Bytes = 16) and the generic register MMA
// on NVFP4 (generic_gemm.cuh rmma_kernel, Bytes = 8, Classes = its WarpsK). A token's bytes then
// do not depend on which of the kernels served its call.
//
// That order: K is cut into steps of 8 * Bytes values; step s belongs to class s % Classes; a class
// sums its steps in ascending order into its own FP32 partial, starting from zero, and the partials
// meet as ((((P0 + P1) + P2) + ...) + P[Classes-1]). Inside a step, MMA j of lane quad c takes the
// codes and activations of physical k = 2 * Bytes * c + 4j + {0, 1} and + {2, 3} (the "own
// contiguous run" mapping). The register MMA kernels run the classes on warps at once; this kernel
// walks them one after the other with two accumulators (the running total and the current class's
// partial), which frees the warps to split tokens instead of K: a CTA serves a tile of rows by a
// tile of tokens and reads each weight once per token tile instead of once per 8 or 32 tokens.
//
// A CTA of WarpsM x WarpsN warps: a warp owns MTiles m16 tiles of rows (read and dequantized from
// its own code bytes, as in the register MMA kernels) and NTiles n8 tiles of tokens. The
// activation tile of each step (Tokens x 8 * Bytes values) is staged in shared memory with
// cp.async, Stages steps ahead, in a swizzled layout that keeps the lanes' 16-byte reads free of
// bank conflicts.
//
// The problem (K, activation stride, rows) is either a registered geometry, all constants
// (Nvfp4A16StaticProblem), or runtime values for the generic path (Nvfp4A16RuntimeProblem).

#include "core/per_device.h"
#include "core/device.h"
#include "core/weight.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/nvfp4/nvfp4_a16_reg_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/common/arch_limits.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail {

template <int WarpsM, int WarpsN, int MTiles, int NTiles, int Stages, int MinBlocksPerSm = 1,
          int Classes = 8, int Bytes = 16, int CodeStages = 2>
struct Nvfp4A16TiledSchedule {
    static_assert(WarpsM > 0 && WarpsN > 0 && WarpsM * WarpsN <= 16);
    static_assert(MTiles > 0 && MTiles <= 4 && NTiles > 0 && NTiles <= 16);
    static_assert(Stages >= 2 && Stages <= 4);
    static_assert(Classes >= 1 && Classes <= 16);
    static_assert(Bytes == 8 || Bytes == 16);
    static_assert(CodeStages >= 1 && CodeStages <= 8);
    static constexpr int kWarpsM         = WarpsM;
    static constexpr int kWarpsN         = WarpsN;
    static constexpr int kMTiles         = MTiles;
    static constexpr int kNTiles         = NTiles;
    static constexpr int kStages         = Stages;
    static constexpr int kMinBlocksPerSm = MinBlocksPerSm;
    static constexpr int kThreads        = WarpsM * WarpsN * 32;
    static constexpr int kRowsPerCta     = WarpsM * MTiles * 16;
    static constexpr int kTokens         = WarpsN * NTiles * 8;
    // The order reproduced: Classes K classes, Bytes code bytes per lane and K step.
    static constexpr int kClasses       = Classes;
    static constexpr int kBytes         = Bytes;
    static constexpr int kKStep         = kBytes * 8; // four lanes of Bytes code bytes, two per byte
    static constexpr int kMmaSteps      = kBytes / 2;
    static constexpr int kGroupsPerLane = kBytes / 8;
    static constexpr int kCodeStages    = CodeStages; // weight steps in flight in registers
    // One step of activations: Tokens rows of kKStep BF16 values, Bytes 16-byte chunks each.
    static constexpr int kChunks     = kBytes;
    static constexpr int kStageBytes = kTokens * kKStep * 2;
    static constexpr int kSmemBytes  = Stages * kStageBytes;
};

// The q-th step of the class-ordered walk over `steps` steps: class w holds steps w, w + C, ...
template <int Classes>
struct Nvfp4A16ClassWalk {
    int steps;
    __device__ __forceinline__ int class_size(int w) const {
        return w < steps ? (steps - w + Classes - 1) / Classes : 0;
    }
    // Step index of walk position q, and whether it is its class's last.
    __device__ __forceinline__ int step(int q, bool& last) const {
        int w = 0;
        int n = class_size(0);
        while (q >= n) {
            q -= n;
            ++w;
            n = class_size(w);
        }
        last = q == n - 1;
        return w + Classes * q;
    }
};

// 16-byte chunk c of token row t within a stage (Chunks per row), XOR-swizzled so that a quarter
// warp's reads (two tokens x four lane runs of one MMA pair) land in eight distinct bank groups.
template <int Chunks>
__device__ __forceinline__ int nvfp4_a16_tiled_chunk(int t, int c) {
    if constexpr (Chunks == 16) {
        return t * 16 + (c ^ (((c >> 3) & 1) << 1) ^ (t & 1));
    } else {
        static_assert(Chunks == 8);
        return t * 8 + (c ^ (t & 1));
    }
}

// A registered geometry: every quantity a constant. Pair != 0: a gated projection whose output
// row r pairs weight rows r and r + Pair (Nvfp4A16RegPairOffset).
template <class Geometry, int Pair>
struct Nvfp4A16StaticProblem {
    static constexpr bool kPaired = Pair != 0;
    static_assert(Pair == 0 || Pair * 2 == Geometry::kOutputRows,
                  "paired rows split the output rows in two halves");
    __host__ __device__ static constexpr int k() { return Geometry::kInputRows; }
    __host__ __device__ static constexpr int ldx() { return Geometry::kInputRows; }
    __host__ __device__ static constexpr int scale_tiles() { return Geometry::kScaleTilesPerRow; }
    // Output rows (paired: the gate rows).
    __host__ __device__ static constexpr int rows() {
        return kPaired ? Pair : Geometry::kOutputRows;
    }
    __host__ __device__ static constexpr int pair() { return Pair; }
    // Whether a CTA of `rows_per_cta` output rows can reach past the last row.
    __host__ __device__ static constexpr bool row_guard(int rows_per_cta) {
        return rows() % rows_per_cta != 0;
    }
    __device__ static constexpr int weight_row(int row, int up) { return row + up * Pair; }
};

// The generic path (generic_gemm.h GemmArgs): output row l reads weight row row_begin + l and,
// paired, its up row row_begin + l + up_offset.
template <bool Paired>
struct Nvfp4A16RuntimeProblem {
    static constexpr bool kPaired = Paired;
    int k_;
    int ldx_;
    int rows_;
    int row_begin;
    int up_offset;
    __host__ __device__ int k() const { return k_; }
    __host__ __device__ int ldx() const { return ldx_; }
    __host__ __device__ int scale_tiles() const { return k_ / 64; }
    __host__ __device__ int rows() const { return rows_; }
    __host__ __device__ int pair() const { return up_offset; }
    __host__ __device__ static constexpr bool row_guard(int) { return true; }
    __device__ int weight_row(int row, int up) const { return row_begin + row + up * up_offset; }
};

__device__ __forceinline__ std::int64_t nvfp4_scale_byte(int row, int group, int scale_tiles) {
    const int m_tile     = row / 128;
    const int row_inner  = row - m_tile * 128;
    const int scale_tile = group / 4;
    const int scale_lane = group & 3;
    return (static_cast<std::int64_t>(m_tile) * scale_tiles + scale_tile) * 512 +
           (row_inner & 31) * 16 + (row_inner >> 5) * 4 + scale_lane;
}

template <class Problem, class Schedule, class Epilogue = LinearIdentityEpilogue,
          class Output = LinearBf16Output>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void
    nvfp4_a16_tiled_kernel(const Problem problem, const __nv_bfloat16* __restrict__ x,
                           const std::uint8_t* __restrict__ codes,
                           const std::uint8_t* __restrict__ scales,
                           float inverse_weight_divisor, Epilogue epilogue, Output output,
                           int tokens) {
    constexpr int B        = Schedule::kBytes;
    constexpr int C        = Schedule::kClasses;
    constexpr int MT       = Schedule::kMTiles;
    constexpr int NT       = Schedule::kNTiles;
    constexpr int S        = Schedule::kStages;
    constexpr int SW       = Schedule::kCodeStages;
    constexpr int kChunks  = Schedule::kChunks;
    constexpr bool kPaired = Problem::kPaired;
    const int K            = problem.k();
    const int ldx          = problem.ldx();
    const int kSteps       = K / Schedule::kKStep; // the launcher requires K % kKStep == 0

    extern __shared__ __align__(128) uint4 nvfp4_a16_tiled_smem[];

    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int warp_m = warp / Schedule::kWarpsN;
    const int warp_n = warp - warp_m * Schedule::kWarpsN;
    const int quad   = lane & 3;
    const int group  = lane >> 2;
    const int token0 = static_cast<int>(blockIdx.y) * Schedule::kTokens;
    // Unpaired: tile m covers output rows row0 + m*16 + [0, 16). Paired: output rows
    // row0 + m*8 + [0, 8), the upper half of the tile reading their first weight rows and the
    // lower half their second (Problem::weight_row).
    constexpr int kTileRows = kPaired ? 8 : 16;
    const int row0 = static_cast<int>(blockIdx.x) * (Schedule::kRowsPerCta / (kPaired ? 2 : 1)) +
                     warp_m * MT * kTileRows;
    const auto out_row = [&](int m, int h) {
        return kPaired ? row0 + m * 8 : row0 + m * 16 + h * 8;
    };
    // Rows past the problem's (a partial last CTA) are computed on its last row and not stored.
    constexpr bool kRowGuard =
        Problem::row_guard(Schedule::kRowsPerCta / (kPaired ? 2 : 1));
    const auto weight_row = [&](int m, int h) {
        const int row = out_row(m, h) + group;
        return problem.weight_row(kRowGuard ? min(row, problem.rows() - 1) : row, kPaired ? h : 0);
    };
    const Nvfp4A16ClassWalk<C> walk{kSteps};

    // Activation staging: the step's Tokens x kChunks chunks, zero past the call's tokens.
    const auto stage_activations = [&](int slot, int step) {
        uint4* dst = nvfp4_a16_tiled_smem + slot * (Schedule::kStageBytes / 16);
        for (int i = static_cast<int>(threadIdx.x); i < Schedule::kTokens * kChunks;
             i += Schedule::kThreads) {
            const int t     = i / kChunks;
            const int c     = i % kChunks;
            const int token = token0 + t;
            const bool live = token < tokens;
            const __nv_bfloat16* src = x + static_cast<std::int64_t>(live ? token : 0) * ldx +
                                       step * Schedule::kKStep + c * 8;
            cp_async_zfill<16, Cache::cg>(dst + nvfp4_a16_tiled_chunk<kChunks>(t, c), src,
                                          live ? 16 : 0);
        }
    };

    const auto code_address = [&](int row, int step) {
        return codes + static_cast<std::int64_t>(row) * (K / 2) + step * (Schedule::kKStep / 2) +
               quad * B;
    };
    const auto scale_address = [&](int row, int step, int g) {
        const int grp = step * (Schedule::kKStep / 16) + quad * Schedule::kGroupsPerLane + g;
        return scales + nvfp4_scale_byte(row, grp, problem.scale_tiles());
    };
    Nvfp4A16RegCodeRun<B> code_ring[SW][MT][2];
    std::uint8_t scale_ring[SW][MT][2][Schedule::kGroupsPerLane];
    const auto fetch_codes = [&](int slot, int step) {
#pragma unroll
        for (int m = 0; m < MT; ++m) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = weight_row(m, h);
                if constexpr (B == 16) {
                    const uint4 v = __ldg(reinterpret_cast<const uint4*>(code_address(row, step)));
                    code_ring[slot][m][h].words[0] = v.x;
                    code_ring[slot][m][h].words[1] = v.y;
                    code_ring[slot][m][h].words[2] = v.z;
                    code_ring[slot][m][h].words[3] = v.w;
                } else {
                    const uint2 v = __ldg(reinterpret_cast<const uint2*>(code_address(row, step)));
                    code_ring[slot][m][h].words[0] = v.x;
                    code_ring[slot][m][h].words[1] = v.y;
                }
#pragma unroll
                for (int g = 0; g < Schedule::kGroupsPerLane; ++g) {
                    scale_ring[slot][m][h][g] = __ldg(scale_address(row, step, g));
                }
            }
        }
    };

    float total[MT][NT][4];
    float part[MT][NT][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) total[m][n][e] = part[m][n][e] = 0.0F;

    // Prologue: S - 1 activation steps and SW weight steps in flight.
    bool unused = false;
#pragma unroll
    for (int s = 0; s < S - 1; ++s) {
        if (s < kSteps) stage_activations(s, walk.step(s, unused));
        cp_commit();
    }
#pragma unroll
    for (int s = 0; s < SW; ++s) {
        if (s < kSteps) fetch_codes(s, walk.step(s, unused));
    }

    bool first_class = true;
    // Unrolled by the code stages, so each register ring slot is a compile-time index.
#pragma unroll 1
    for (int q0 = 0; q0 < kSteps; q0 += SW) {
#pragma unroll
        for (int cs = 0; cs < SW; ++cs) {
            const int q = q0 + cs;
            if (q >= kSteps) break;
            bool class_end = false;
            (void)walk.step(q, class_end);
            cp_wait<S - 2>();
            __syncthreads();
            // The slot the next staging overwrites was read in iteration q - 1, which every warp
            // has finished (the barrier above).
            if (q + S - 1 < kSteps) {
                stage_activations((q + S - 1) % S, walk.step(q + S - 1, unused));
            }
            cp_commit();

            const uint4* act = nvfp4_a16_tiled_smem + (q % S) * (Schedule::kStageBytes / 16);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                __half2 scale[2][Schedule::kGroupsPerLane];
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int g = 0; g < Schedule::kGroupsPerLane; ++g)
                        scale[h][g] = decode_nvfp4_scale_half2(scale_ring[cs][m][h][g]);
#pragma unroll
                for (int jj = 0; jj < Schedule::kMmaSteps / 2; ++jj) {
                    // B fragments of MMA steps 2jj and 2jj + 1 for every n8 tile: this lane's
                    // token, values 2 * B * quad + 8 * jj .. + 7 (one 16-byte chunk).
                    uint4 b[NT];
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        const int t = warp_n * NT * 8 + n * 8 + group;
                        b[n] = act[nvfp4_a16_tiled_chunk<kChunks>(t, quad * (B / 4) + jj)];
                    }
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        const int j        = 2 * jj + half;
                        const int g        = (4 * j) / 16;
                        const int word     = (2 * j) / 4;
                        const int shift_lo = ((2 * j) % 4) * 8;
                        const std::uint32_t r0 = code_ring[cs][m][0].words[word];
                        const std::uint32_t r1 = code_ring[cs][m][1].words[word];
                        const unsigned a0 = dequant_nvfp4_pair_bf16(r0 >> shift_lo, scale[0][g]);
                        const unsigned a1 = dequant_nvfp4_pair_bf16(r1 >> shift_lo, scale[1][g]);
                        const unsigned a2 =
                            dequant_nvfp4_pair_bf16(r0 >> (shift_lo + 8), scale[0][g]);
                        const unsigned a3 =
                            dequant_nvfp4_pair_bf16(r1 >> (shift_lo + 8), scale[1][g]);
#pragma unroll
                        for (int n = 0; n < NT; ++n) {
                            const unsigned b0 = half == 0 ? b[n].x : b[n].z;
                            const unsigned b1 = half == 0 ? b[n].y : b[n].w;
                            mma_bf16(part[m][n][0], part[m][n][1], part[m][n][2], part[m][n][3],
                                     a0, a1, a2, a3, b0, b1);
                        }
                    }
                }
            }
            if (q + SW < kSteps) fetch_codes(cs, walk.step(q + SW, unused));

            if (class_end) {
                // The register MMA kernels' meeting of partials: class 0's partial, then each
                // later class's partial added in class order.
#pragma unroll
                for (int m = 0; m < MT; ++m)
#pragma unroll
                    for (int n = 0; n < NT; ++n)
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            total[m][n][e] =
                                first_class ? part[m][n][e] : total[m][n][e] + part[m][n][e];
                            part[m][n][e] = 0.0F;
                        }
                first_class = false;
            }
        }
    }
    // Classes without a step (K below C steps) still meet with a zero partial there.
#pragma unroll 1
    for (int w = kSteps; w < C; ++w) {
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int n = 0; n < NT; ++n)
#pragma unroll
                for (int e = 0; e < 4; ++e) total[m][n][e] += 0.0F;
    }
    cp_wait<0>();

    const int tokens_base = token0 + warp_n * NT * 8;
    const auto row_live   = [&](int row) { return !kRowGuard || row < problem.rows(); };
    if constexpr (!kPaired) {
#pragma unroll
        for (int m = 0; m < MT; ++m) {
#pragma unroll
            for (int n = 0; n < NT; ++n) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int row   = out_row(m, e >= 2 ? 1 : 0) + group;
                    const int token = tokens_base + n * 8 + 2 * quad + (e & 1);
                    if (token < tokens && row_live(row)) {
                        output.store(row, token,
                                     epilogue.apply(row, token,
                                                    total[m][n][e] * inverse_weight_divisor));
                    }
                }
            }
        }
    } else {
#pragma unroll
        for (int m = 0; m < MT; ++m) {
#pragma unroll
            for (int n = 0; n < NT; ++n) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int row   = out_row(m, 0) + group;
                    const int token = tokens_base + n * 8 + 2 * quad + e;
                    if (token < tokens && row_live(row)) {
                        output.store_pair(
                            row, token,
                            epilogue.apply(row, token, total[m][n][e] * inverse_weight_divisor),
                            epilogue.apply(row + problem.pair(), token,
                                           total[m][n][e + 2] * inverse_weight_divisor));
                    }
                }
            }
        }
    }
}

// One launch of `tokens` columns of x on a problem (static or runtime).
template <class Problem, class Schedule, class Epilogue, class Output>
void launch_nvfp4_a16_tiled_problem(const Problem& problem, const __nv_bfloat16* x,
                                    const std::uint8_t* codes, const std::uint8_t* scales,
                                    float inverse_weight_divisor, Epilogue epilogue,
                                    Output output, std::int32_t tokens, cudaStream_t stream) {
    if (tokens <= 0 || problem.rows() <= 0) {
        throw std::invalid_argument("nvfp4 A16 tiled: empty call");
    }
    if (problem.k() % Schedule::kKStep != 0) {
        throw std::invalid_argument("nvfp4 A16 tiled: K must be a whole number of K steps");
    }
    auto* kernel = nvfp4_a16_tiled_kernel<Problem, Schedule, Epilogue, Output>;
    if constexpr (Schedule::kSmemBytes > kStaticSharedBytesPerBlock) {
        static ::ninferno::PerDevice<bool> configured;
        ::ninferno::allow_dynamic_smem(configured, kernel, Schedule::kSmemBytes);
    }
    constexpr int kRowsPerCta = Schedule::kRowsPerCta / (Problem::kPaired ? 2 : 1);
    const dim3 grid((problem.rows() + kRowsPerCta - 1) / kRowsPerCta,
                    (tokens + Schedule::kTokens - 1) / Schedule::kTokens);
    kernel<<<grid, Schedule::kThreads, Schedule::kSmemBytes, stream>>>(
        problem, x, codes, scales, inverse_weight_divisor, epilogue, output, tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Schedule, class Epilogue, class Output>
void launch_nvfp4_a16_tiled(const __nv_bfloat16* x, const Weight& weight, Epilogue epilogue,
                            Output output, std::int32_t tokens, cudaStream_t stream) {
    using Problem = Nvfp4A16StaticProblem<Geometry, Nvfp4A16RegPairOffset<Output>::value>;
    static_assert(Geometry::kInputRows % Schedule::kKStep == 0, "a K step must tile K");
    launch_nvfp4_a16_tiled_problem<Problem, Schedule>(
        Problem{}, x, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), 1.0F / weight.weight_scale_divisor,
        epilogue, output, tokens, stream);
}

} // namespace ninferno::ops::detail
