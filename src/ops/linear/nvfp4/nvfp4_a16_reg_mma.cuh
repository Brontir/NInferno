#pragma once

// NVFP4 x BF16 on tensor cores for small T, with A16 semantics.
//
// out[N,T] = W[N,K] * x[K,T]. Each code is dequantized with its e4m3 block scale into BF16 and
// multiplied against the BF16 activation by mma.sync m16n8k16 with FP32 accumulation; the
// activation is never quantized. The weight divisor is applied once, in the epilogue.
//
// The product of an e2m1 code and an e4m3 scale has at most six significant bits and lies between
// 2^-10 and 2688, so it is exact in FP16 (where the two are multiplied) and in BF16 (where the MMA
// reads it). What remains is FP32 accumulation, the same class of rounding the SIMT route has.
//
// No shared memory holds weights. A dot product does not care in which order its K terms are
// summed, so the kernel assigns K to the MMA's k slots in the order that lets every lane read its
// A fragments straight from its own contiguous code bytes: lane c (= lane % 4) of a quad owns the
// Bytes-byte run at c * Bytes of each row's K step, one scale group per eight bytes, and MMA step
// j takes bytes 2j and 2j+1 of that run as its two A registers per row. The activation fragment
// uses the same assignment - physical k = 2 * Bytes * c + 4j + {0,1} and + {2,3} - so each lane
// reads its B fragments from its own contiguous activation run too.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/common/arch_limits.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <type_traits>

namespace ninferno::ops::detail {

// An Output with `static constexpr std::int32_t kPairOffset = P` receives rows in pairs: each m16
// tile computes eight rows r (its upper half) and the eight rows r + P (its lower half), and the
// kernel hands every lane's two results for one token together to
// Output::store_pair(r, token, value_r, value_r_plus_P). A gated projection (SwiGLU: gate rows
// [0, P), up rows [P, 2P)) finishes in registers that way. Any other Output gets consecutive rows.
template <class Output, class = void>
struct Nvfp4A16RegPairOffset : std::integral_constant<int, 0> {};

template <class Output>
struct Nvfp4A16RegPairOffset<Output, std::void_t<decltype(Output::kPairOffset)>>
    : std::integral_constant<int, Output::kPairOffset> {};

// WarpsM warps split the CTA's rows, WarpsK warps split each row's K steps (partial sums meet in
// shared memory). A warp owns MTiles 16-row tiles and every token up to Tokens (a multiple of 8).
// A lane reads Bytes code bytes of each row per K step (8: K steps of 64, 16: K steps of 128) and
// keeps Stages steps of weights in flight in registers.
template <int WarpsM, int WarpsK, int MTiles, int Tokens, int Bytes, int Stages, int MinBlocksPerSm>
struct Nvfp4A16RegMmaSchedule {
    static_assert(WarpsM > 0 && WarpsK > 0 && WarpsM * WarpsK <= 16);
    static_assert(MTiles > 0 && MTiles <= 8);
    static_assert(Tokens >= 8 && Tokens <= 64 && (Tokens % 8) == 0);
    static_assert(Bytes == 8 || Bytes == 16);
    static_assert(Stages >= 1 && Stages <= 4);
    static constexpr int kWarpsM         = WarpsM;
    static constexpr int kWarpsK         = WarpsK;
    static constexpr int kMTiles         = MTiles;
    static constexpr int kTokens         = Tokens;
    static constexpr int kNTiles         = Tokens / 8;
    static constexpr int kBytes          = Bytes;
    static constexpr int kStages         = Stages;
    static constexpr int kMinBlocksPerSm = MinBlocksPerSm;
    static constexpr int kThreads        = WarpsM * WarpsK * 32;
    static constexpr int kRowsPerCta     = WarpsM * MTiles * 16;
    static constexpr int kKStep         = Bytes * 8; // four lanes of Bytes code bytes, two per byte
    static constexpr int kMmaSteps      = Bytes / 2; // one k16 MMA per two bytes of a lane
    static constexpr int kGroupsPerLane = Bytes / 8; // 16-value scale groups a lane covers
    // m tiles whose WarpsK partials meet at once: the most (a divisor of MTiles) whose buffer fits
    // the 48 KB of static shared memory.
    static constexpr int kReduceTiles = [] {
        for (int tiles = MTiles; tiles > 1; --tiles) {
            if (MTiles % tiles == 0 && (WarpsK - 1) * WarpsM * tiles * (Tokens / 8) * 4 * 32 * 4 <=
                                           kStaticSharedBytesPerBlock) {
                return tiles;
            }
        }
        return 1;
    }();
    static_assert((WarpsK - 1) * WarpsM * kReduceTiles * (Tokens / 8) * 4 * 32 * 4 <=
                      kStaticSharedBytesPerBlock,
                  "one m tile's partials must fit static shared memory");
};

// Weights are read once per call and never reused by this CTA: stream them past L1.
__device__ __forceinline__ uint4 ld_global_cs_v4(const std::uint8_t* p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint2 ld_global_cs_v2(const std::uint8_t* p) {
    uint2 v;
    asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

template <int Bytes>
struct Nvfp4A16RegCodeRun {
    std::uint32_t words[Bytes / 4];
};

template <int Bytes>
__device__ __forceinline__ Nvfp4A16RegCodeRun<Bytes> load_nvfp4_a16_codes(const std::uint8_t* p) {
    Nvfp4A16RegCodeRun<Bytes> run;
    if constexpr (Bytes == 16) {
        const uint4 v = ld_global_cs_v4(p);
        run.words[0]  = v.x;
        run.words[1]  = v.y;
        run.words[2]  = v.z;
        run.words[3]  = v.w;
    } else {
        const uint2 v = ld_global_cs_v2(p);
        run.words[0]  = v.x;
        run.words[1]  = v.y;
    }
    return run;
}

// Two codes of one byte times their block scale, as the BF16 pair an MMA register holds (low half
// the even k). Every step is exact: see the header.
__device__ __forceinline__ std::uint32_t dequant_nvfp4_pair_bf16(std::uint32_t byte,
                                                                 __half2 scale) {
    const __half2 product  = __hmul2(e2m1x2_to_half2(byte), scale);
    const __nv_bfloat162 w = __float22bfloat162_rn(__half22float2(product));
    return *reinterpret_cast<const std::uint32_t*>(&w);
}

__device__ __forceinline__ __half2 decode_nvfp4_scale_half2(std::uint8_t storage) {
    const __half_raw h =
        __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(storage), __NV_E4M3);
    const __half s(h);
    return __half2(s, s);
}

// A token's sums depend on WarpsK and Bytes only (the K steps a warp owns, the order within a step
// and the order the warps' partials meet in): schedules that share the two compute every token's
// bytes identically whatever their token capacity, row tiling or pipeline depth.
//
// The result of (row, token) passes through Epilogue::apply(row, token, value) and lands through
// Output::store(row, token, value), as in the other linear kernels (linear/common/output.cuh). Grid row y
// serves tokens [y * Tokens, (y + 1) * Tokens) of the call, so one launch covers any T.
template <class Geometry, class Schedule, class Epilogue = LinearIdentityEpilogue,
          class Output = LinearBf16Output>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void nvfp4_a16_reg_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse_weight_divisor, Epilogue epilogue,
    Output output, int tokens) {
    constexpr int K      = Geometry::kInputRows;
    constexpr int B      = Schedule::kBytes;
    constexpr int MT     = Schedule::kMTiles;
    constexpr int NT     = Schedule::kNTiles;
    constexpr int WK     = Schedule::kWarpsK;
    constexpr int S      = Schedule::kStages;
    constexpr int kSteps = K / Schedule::kKStep;
    static_assert(K % Schedule::kKStep == 0, "a K step must tile K");
    static_assert(Geometry::kOutputRows % Schedule::kRowsPerCta == 0);
    constexpr int kPair = Nvfp4A16RegPairOffset<Output>::value;
    static_assert(kPair == 0 || kPair * 2 == Geometry::kOutputRows,
                  "paired rows split the output rows in two halves");
    static_assert(128 % 16 == 0);

    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int warp_m = warp / WK;
    const int warp_k = warp - warp_m * WK;
    const int quad   = lane & 3;  // which Bytes-byte run of the step
    const int group  = lane >> 2; // row within an m16 half, token within an n8 tile
    const int token0 = static_cast<int>(blockIdx.y) * Schedule::kTokens;
    // Unpaired: tile m covers rows row0 + m*16 + [0, 16). Paired: it covers row0 + m*8 + [0, 8)
    // (upper half) and the same rows + kPair (lower half); a CTA still computes kRowsPerCta rows.
    constexpr int kTileRows = kPair == 0 ? 16 : 8;
    const int row0 =
        static_cast<int>(blockIdx.x) * (Schedule::kRowsPerCta / (kPair == 0 ? 1 : 2)) +
        warp_m * MT * kTileRows;
    const auto tile_row = [&](int m, int h) {
        return kPair == 0 ? row0 + m * 16 + h * 8 : row0 + m * 8 + h * kPair;
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < NT; ++n)
            acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.0F;

    // Per row: this lane's code run and its scale bytes.
    const auto code_address = [&](int row, int step) {
        return codes + static_cast<std::int64_t>(row) * (K / 2) + step * (Schedule::kKStep / 2) +
               quad * B;
    };
    const auto scale_address = [&](int row, int step, int g) {
        const int grp = step * (Schedule::kKStep / 16) + quad * Schedule::kGroupsPerLane + g;
        return scales + nvfp4_scale_byte<Geometry::kScaleTilesPerRow>(row, grp);
    };

    Nvfp4A16RegCodeRun<B> code_ring[S][MT][2];
    std::uint8_t scale_ring[S][MT][2][Schedule::kGroupsPerLane];
    const auto fetch = [&](int slot, int step) {
#pragma unroll
        for (int m = 0; m < MT; ++m) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row         = tile_row(m, h) + group;
                code_ring[slot][m][h] = load_nvfp4_a16_codes<B>(code_address(row, step));
#pragma unroll
                for (int g = 0; g < Schedule::kGroupsPerLane; ++g) {
                    scale_ring[slot][m][h][g] = *scale_address(row, step, g);
                }
            }
        }
    };

#pragma unroll
    for (int s = 0; s < S; ++s) {
        const int step = warp_k + s * WK;
        if (step < kSteps) fetch(s, step);
    }

    const int local_steps = (kSteps - warp_k + WK - 1) / WK;
#pragma unroll 1
    for (int i = 0; i < local_steps; i += S) {
#pragma unroll
        for (int s = 0; s < S; ++s) {
            const int it = i + s;
            if (it >= local_steps) break;
            const int step = warp_k + it * WK;

            // Activation runs: token n8 tile, this lane's token, 2*B values of the step.
            std::uint32_t act[NT][B];
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const int token        = token0 + n * 8 + group;
                const __nv_bfloat16* p = x + static_cast<std::int64_t>(token) * K +
                                         step * Schedule::kKStep + quad * 2 * B;
#pragma unroll
                for (int v = 0; v < B / 4; ++v) {
                    uint4 w = make_uint4(0, 0, 0, 0);
                    if (token < tokens) w = load_vec<uint4>(p + v * 8);
                    act[n][v * 4 + 0] = w.x;
                    act[n][v * 4 + 1] = w.y;
                    act[n][v * 4 + 2] = w.z;
                    act[n][v * 4 + 3] = w.w;
                }
            }

#pragma unroll
            for (int m = 0; m < MT; ++m) {
                __half2 scale[2][Schedule::kGroupsPerLane];
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int g = 0; g < Schedule::kGroupsPerLane; ++g)
                        scale[h][g] = decode_nvfp4_scale_half2(scale_ring[s][m][h][g]);
#pragma unroll
                for (int j = 0; j < Schedule::kMmaSteps; ++j) {
                    const int g        = (4 * j) / 16; // scale group of this step within the run
                    const int word     = (2 * j) / 4;
                    const int shift_lo = ((2 * j) % 4) * 8;
                    const std::uint32_t r0 = code_ring[s][m][0].words[word];
                    const std::uint32_t r1 = code_ring[s][m][1].words[word];
                    const unsigned a0      = dequant_nvfp4_pair_bf16(r0 >> shift_lo, scale[0][g]);
                    const unsigned a1      = dequant_nvfp4_pair_bf16(r1 >> shift_lo, scale[1][g]);
                    const unsigned a2 = dequant_nvfp4_pair_bf16(r0 >> (shift_lo + 8), scale[0][g]);
                    const unsigned a3 = dequant_nvfp4_pair_bf16(r1 >> (shift_lo + 8), scale[1][g]);
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        mma_bf16(acc[m][n][0], acc[m][n][1], acc[m][n][2], acc[m][n][3], a0, a1, a2,
                                 a3, act[n][2 * j], act[n][2 * j + 1]);
                    }
                }
            }

            const int next = step + S * WK;
            if (next < kSteps) fetch(s, next);
        }
    }

    if constexpr (WK > 1) {
        // The partials meet in chunks of kChunk m tiles (all MT when they fit static shared
        // memory); either way every value adds warps 1..WK-1 onto warp 0's partial in warp order.
        constexpr int kChunk = Schedule::kReduceTiles;
        __shared__ float partial[WK - 1][Schedule::kWarpsM][kChunk][NT][4][32];
#pragma unroll
        for (int m0 = 0; m0 < MT; m0 += kChunk) {
            if (m0 > 0) __syncthreads(); // warp 0 has read the previous chunk
            if (warp_k > 0) {
#pragma unroll
                for (int m = 0; m < kChunk; ++m)
#pragma unroll
                    for (int n = 0; n < NT; ++n)
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                            partial[warp_k - 1][warp_m][m][n][e][lane] = acc[m0 + m][n][e];
            }
            __syncthreads();
            if (warp_k == 0) {
#pragma unroll
                for (int w = 0; w < WK - 1; ++w)
#pragma unroll
                    for (int m = 0; m < kChunk; ++m)
#pragma unroll
                        for (int n = 0; n < NT; ++n)
#pragma unroll
                            for (int e = 0; e < 4; ++e)
                                acc[m0 + m][n][e] += partial[w][warp_m][m][n][e][lane];
            }
        }
        if (warp_k != 0) return;
    }

    // C fragment: rows group (e = 0, 1) and group + 8 (e = 2, 3) of each m16 tile, tokens 2*quad
    // (even e) and 2*quad + 1 (odd e).
    if constexpr (kPair == 0) {
#pragma unroll
        for (int m = 0; m < MT; ++m) {
#pragma unroll
            for (int n = 0; n < NT; ++n) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int row   = tile_row(m, e >= 2 ? 1 : 0) + group;
                    const int token = token0 + n * 8 + 2 * quad + (e & 1);
                    if (token < tokens) {
                        output.store(row, token,
                                     epilogue.apply(row, token,
                                                    acc[m][n][e] * inverse_weight_divisor));
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
                    const int row   = tile_row(m, 0) + group;
                    const int token = token0 + n * 8 + 2 * quad + e;
                    if (token < tokens) {
                        output.store_pair(
                            row, token,
                            epilogue.apply(row, token, acc[m][n][e] * inverse_weight_divisor),
                            epilogue.apply(row + kPair, token,
                                           acc[m][n][e + 2] * inverse_weight_divisor));
                    }
                }
            }
        }
    }
}

} // namespace ninferno::ops::detail
