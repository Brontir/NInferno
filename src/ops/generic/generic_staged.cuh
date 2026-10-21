#pragma once

// Generic staged register-MMA: BF16 weights against BF16 activations on BF16 tensor cores, with
// both operands staged through shared memory, for prefill-sized T.
//
// The arithmetic is the register MMA's (generic_gemm.cuh, rmma_kernel<Bf16Decoder> with WarpsK == 1)
// exactly: the weight is the MMA's A operand (M = weight rows), the activation its B operand
// (N = tokens), a K step is 32 values, the lane of quad q owns values 8q .. 8q + 7 of the step in
// every row and every token, and MMA j of the step reads values 8q + 4j .. 8q + 4j + 3 of them. Each
// accumulator takes the steps in order, MMA 0 before MMA 1, and nothing splits K. Every fragment
// register therefore holds the same values as in the register MMA, every MMA adds them to the same
// running sum, and the epilogue is the same: this family's outputs are bit-identical to any
// WarpsK == 1 register-MMA schedule's, so tables may choose between the two freely.
//
// What changes is where the operands come from: a CTA stages, per K tile of 64 values, a tile of
// weight rows and a tile of tokens with cp.async into shared memory (a ring of stages), and warps
// read their runs from there with 16-byte loads, so each weight value is read from global memory
// once per token tile instead of once per 8 tokens. A 64-value row is eight 16-byte chunks; chunk c
// of row r is stored at chunk c ^ 4 (r & 1), so the eight lanes of a quarter-warp (two rows, four
// chunks) read eight distinct bank groups.

#include "ops/common/arch_limits.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/generic/generic_gemm.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninferno::ops::detail::generic {

template <int Rows, int Tokens, int WarpsM, int WarpsN, int Stages>
struct StagedSchedule {
    static_assert(Rows % (32 * WarpsM) == 0 && Tokens % (8 * WarpsN) == 0);
    static constexpr int kRows      = Rows;   // weight rows staged per CTA (SwiGLU: gate + up)
    static constexpr int kTokens    = Tokens; // tokens per CTA
    static constexpr int kWarpsM    = WarpsM;
    static constexpr int kWarpsN    = WarpsN;
    static constexpr int kStages    = Stages;
    static constexpr int kThreads   = WarpsM * WarpsN * 32;
    static constexpr int kWarpRows  = Rows / WarpsM;
    static constexpr int kWarpToks  = Tokens / WarpsN;
    static constexpr int kMTiles    = kWarpRows / 16;
    static constexpr int kNTiles    = kWarpToks / 8;
};

template <class S, int Stages>
struct StagedShared {
    alignas(16) __nv_bfloat16 w[Stages][S::kRows * 64];
    alignas(16) __nv_bfloat16 x[Stages][S::kTokens * 64];
};

// The schedule's pipeline depth, lowered until the stages fit one block of this build.
template <class S, int Stages = S::kStages>
struct StagedStages {
    static constexpr int value =
        sizeof(StagedShared<S, Stages>) <= kMaxSharedBytesPerBlock || Stages == 2
            ? Stages
            : StagedStages<S, Stages - 1>::value;
};
template <class S>
struct StagedStages<S, 1> {
    static constexpr int value = 2;
};

template <class S>
constexpr int staged_shared_bytes() {
    return static_cast<int>(sizeof(StagedShared<S, StagedStages<S>::value>));
}

__device__ __forceinline__ int staged_offset(int row, int chunk) {
    return row * 64 + ((chunk ^ ((row & 1) << 2)) << 3);
}

template <class S, int Sets, int St>
__device__ __forceinline__ void staged_load(const GemmArgs& a, StagedShared<S, St>& sh, int stage,
                                            int k_tile, int row_block, int token_begin) {
    constexpr int kSetRows = S::kRows / Sets;
    const auto* w          = reinterpret_cast<const __nv_bfloat16*>(a.codes);
    const auto* x          = static_cast<const __nv_bfloat16*>(a.x);
#pragma unroll
    for (int task = static_cast<int>(threadIdx.x); task < S::kRows * 8; task += S::kThreads) {
        const int row   = task >> 3;
        const int chunk = task & 7;
        const int set   = row / kSetRows;
        const int local = row_block * kSetRows + (row - set * kSetRows);
        const bool ok   = local < a.rows;
        const int wrow  = a.row_begin + (ok ? local : 0) + (set == 1 ? a.up_offset : 0);
        cp_async_zfill<16, Cache::cg>(&sh.w[stage][staged_offset(row, chunk)],
                                      w + static_cast<std::int64_t>(wrow) * a.k + k_tile * 64 +
                                          chunk * 8,
                                      ok ? 16 : 0);
    }
#pragma unroll
    for (int task = static_cast<int>(threadIdx.x); task < S::kTokens * 8; task += S::kThreads) {
        const int row   = task >> 3;
        const int chunk = task & 7;
        const int token = token_begin + row;
        const bool ok   = token < a.tokens;
        cp_async_zfill<16, Cache::cg>(&sh.x[stage][staged_offset(row, chunk)],
                                      x + static_cast<std::int64_t>(ok ? token : 0) * a.ldx +
                                          k_tile * 64 + chunk * 8,
                                      ok ? 16 : 0);
    }
}

template <class S, Epilogue E>
__global__ __launch_bounds__(S::kThreads) void staged_kernel(GemmArgs a) {
    constexpr bool kSwiGlu = E == Epilogue::SwiGlu;
    constexpr int kSets    = kSwiGlu ? 2 : 1; // gate rows, then the matching up rows
    constexpr int kSetRows = S::kRows / kSets;
    constexpr int kWarpSet = S::kWarpRows / kSets; // a warp's rows of one set
    constexpr int MT       = kWarpSet / 16;
    constexpr int NT       = S::kNTiles;
    static_assert(MT >= 1 && kWarpSet % 16 == 0);
    constexpr int St = StagedStages<S>::value;
    using Shared     = StagedShared<S, St>;

    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& sh              = *reinterpret_cast<Shared*>(shared_bytes);
    const int row_block   = static_cast<int>(blockIdx.x);
    const int token_begin = static_cast<int>(blockIdx.y) * S::kTokens;
    const int k_tiles     = a.k / 64;

    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int warp_m = warp / S::kWarpsN;
    const int warp_n = warp - warp_m * S::kWarpsN;
    const int quad   = lane & 3;
    const int group  = lane >> 2;

    float acc[kSets][MT][NT][4];
#pragma unroll
    for (int s = 0; s < kSets; ++s)
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int n = 0; n < NT; ++n)
                acc[s][m][n][0] = acc[s][m][n][1] = acc[s][m][n][2] = acc[s][m][n][3] = 0.0F;

#pragma unroll
    for (int stage = 0; stage < St - 1; ++stage) {
        if (stage < k_tiles) staged_load<S, kSets, St>(a, sh, stage, stage, row_block, token_begin);
        cp_commit();
    }

#pragma unroll 1
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        cp_wait<St - 2>();
        __syncthreads();
        // The stage refilled here was read in the previous iteration, before the barrier.
        const int ahead = k_tile + St - 1;
        if (ahead < k_tiles) staged_load<S, kSets, St>(a, sh, ahead % St, ahead, row_block, token_begin);
        cp_commit();

        const __nv_bfloat16* ws = sh.w[k_tile % St];
        const __nv_bfloat16* xs = sh.x[k_tile % St];
#pragma unroll
        for (int step = 0; step < 2; ++step) {
            const int chunk = 4 * step + quad;
            uint4 wr[kSets][MT][2];
            uint4 xr[NT];
#pragma unroll
            for (int s = 0; s < kSets; ++s)
#pragma unroll
                for (int m = 0; m < MT; ++m)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int row = s * kSetRows + warp_m * kWarpSet + m * 16 + h * 8 + group;
                        wr[s][m][h]   = load_vec<uint4>(ws + staged_offset(row, chunk));
                    }
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const int row = warp_n * S::kWarpToks + n * 8 + group;
                xr[n]         = load_vec<uint4>(xs + staged_offset(row, chunk));
            }
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int s = 0; s < kSets; ++s)
#pragma unroll
                    for (int m = 0; m < MT; ++m) {
                        const uint4& r0 = wr[s][m][0];
                        const uint4& r1 = wr[s][m][1];
                        const unsigned a0 = j == 0 ? r0.x : r0.z;
                        const unsigned a2 = j == 0 ? r0.y : r0.w;
                        const unsigned a1 = j == 0 ? r1.x : r1.z;
                        const unsigned a3 = j == 0 ? r1.y : r1.w;
#pragma unroll
                        for (int n = 0; n < NT; ++n) {
                            auto& c = acc[s][m][n];
                            mma_bf16(c[0], c[1], c[2], c[3], a0, a1, a2, a3,
                                     j == 0 ? xr[n].x : xr[n].z, j == 0 ? xr[n].y : xr[n].w);
                        }
                    }
        }
    }
    cp_wait<0>();

    // C fragment: rows group and group + 8 of each m16 tile, tokens 2*quad and 2*quad + 1.
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int l = row_block * kSetRows + warp_m * kWarpSet + m * 16 + group + (e >= 2 ? 8 : 0);
            if (l >= a.rows) continue;
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const int token = token_begin + warp_n * S::kWarpToks + n * 8 + 2 * quad + (e & 1);
                if (token >= a.tokens) continue;
                float value = acc[0][m][n][e];
                if constexpr (kSwiGlu) value = generic_gate(a, value) * acc[kSets - 1][m][n][e];
                store_output<E>(a, l, token, value);
            }
        }
    }
}

} // namespace ninferno::ops::detail::generic
