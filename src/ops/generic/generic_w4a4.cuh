#pragma once

// Generic NVFP4 W4A4: NVFP4 weights of any [N,K] (K % 64 == 0, as the layout requires; rows past
// N are neither read nor stored) against activations quantized to NVFP4, on block-scaled FP4
// tensor cores.
//
// This is a runtime-geometry relative of nvfp4_a4_mma_kernel (ops/linear/nvfp4/nvfp4_a4_mma.cuh):
// the same staging, swizzle and fragment code, with K read at run time, K tiles of 64, and a row
// map that also serves SwiGLU (gate and up rows interleaved in groups of eight so that one thread
// holds both accumulators of an output) and segmented outputs. K tiles accumulate in order and no
// schedule splits K, so every schedule sums an output element in the same order: a token's bytes
// do not depend on the call's T.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/generic/generic_gemm.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninferno::ops::detail::generic {

template <int BlockM, int BlockN, int WarpsM, int WarpsN, int Stages>
struct W4a4Schedule {
    static_assert(BlockM % (16 * WarpsM) == 0 && BlockN % (16 * WarpsN) == 0);
    static constexpr int kBlockM       = BlockM;
    static constexpr int kBlockN       = BlockN;
    static constexpr int kWarpsM       = WarpsM;
    static constexpr int kWarpsN       = WarpsN;
    static constexpr int kStages       = Stages;
    static constexpr int kThreads      = WarpsM * WarpsN * 32;
    static constexpr int kWarpM        = BlockM / WarpsM;
    static constexpr int kWarpN        = BlockN / WarpsN;
    static constexpr int kMmaM         = kWarpM / 16;
    static constexpr int kMmaN         = kWarpN / 8;
    static constexpr int kCodeRowBytes = 32; // one K tile of 64 values
};

template <class S>
struct W4a4Shared {
    alignas(16) std::uint8_t a_codes[S::kStages][S::kBlockM * S::kCodeRowBytes];
    alignas(16) std::uint8_t b_codes[S::kStages][S::kBlockN * S::kCodeRowBytes];
    alignas(16) std::uint32_t a_scale4[S::kStages][S::kBlockM];
    alignas(16) std::uint32_t b_scale4[S::kStages][S::kBlockN];
};

// Two 16-byte segments per 32-byte row, swapped on odd rows so an ldmatrix of eight rows hits
// both bank halves.
__device__ __forceinline__ int w4a4_swizzled_byte(int row, int logical_byte) {
    return (((logical_byte >> 4) ^ (row & 1)) << 4) | (logical_byte & 15);
}

struct W4a4Activation {
    const std::uint8_t* codes;  // [T, K/2]
    const std::uint8_t* scales; // [T, K/16] row-major
    float input_divisor;
};

template <class S, bool SwiGlu>
__device__ __forceinline__ void w4a4_stage(const GemmArgs& a, W4a4Activation act, W4a4Shared<S>& sh,
                                           int stage, int k_tile, int token_begin, int block_row) {
    const int k_bytes = a.k / 2;
    for (int task = static_cast<int>(threadIdx.x); task < S::kBlockM * 2; task += S::kThreads) {
        const int row   = task >> 1;
        const int seg   = task & 1;
        const int token = token_begin + row;
        const bool ok   = token < a.tokens;
        const auto* src = act.codes + static_cast<std::int64_t>(ok ? token : 0) * k_bytes +
                          k_tile * 32 + seg * 16;
        cp_async_zfill<16, Cache::cg>(sh.a_codes[stage] + row * 32 + w4a4_swizzled_byte(row, seg * 16),
                                      src, ok ? 16 : 0);
    }
    for (int row = static_cast<int>(threadIdx.x); row < S::kBlockM; row += S::kThreads) {
        const int token = token_begin + row;
        const bool ok   = token < a.tokens;
        const auto* src =
            act.scales + static_cast<std::int64_t>(ok ? token : 0) * (a.k / 16) + k_tile * 4;
        cp_async_zfill<4>(&sh.a_scale4[stage][row], src, ok ? 4 : 0);
    }
    for (int task = static_cast<int>(threadIdx.x); task < S::kBlockN * 2; task += S::kThreads) {
        const int row   = task >> 1;
        const int seg   = task & 1;
        const int local = w4a4_local_row<SwiGlu, S::kBlockN>(block_row, row);
        const bool ok   = local < a.rows;
        const int wrow  = a.row_begin + (ok ? local : 0) +
                         (SwiGlu && ((row >> 3) & 1) ? a.up_offset : 0);
        const auto* src = a.codes + static_cast<std::int64_t>(wrow) * k_bytes + k_tile * 32 + seg * 16;
        cp_async_zfill<16, Cache::cg>(sh.b_codes[stage] + row * 32 + w4a4_swizzled_byte(row, seg * 16),
                                      src, ok ? 16 : 0);
    }
    for (int row = static_cast<int>(threadIdx.x); row < S::kBlockN; row += S::kThreads) {
        const int local = w4a4_local_row<SwiGlu, S::kBlockN>(block_row, row);
        const bool ok   = local < a.rows;
        const int wrow  = a.row_begin + (ok ? local : 0) +
                         (SwiGlu && ((row >> 3) & 1) ? a.up_offset : 0);
        const int r_in  = wrow & 127;
        const auto* src = a.scales +
                          (static_cast<std::int64_t>(wrow >> 7) * a.scale_tiles + k_tile) * 512 +
                          (r_in & 31) * 16 + (r_in >> 5) * 4;
        cp_async_zfill<4>(&sh.b_scale4[stage][row], src, ok ? 4 : 0);
    }
}

template <class S, Epilogue E>
__global__ __launch_bounds__(S::kThreads) void w4a4_kernel(GemmArgs a, W4a4Activation act) {
    constexpr bool kSwiGlu = E == Epilogue::SwiGlu;
    static_assert(!kSwiGlu || (S::kMmaN % 2) == 0);
    __shared__ W4a4Shared<S> sh;
    const int block_row   = static_cast<int>(blockIdx.x);
    const int token_begin = static_cast<int>(blockIdx.y) * S::kBlockM;
    const int k_tiles     = a.k / 64;

#pragma unroll
    for (int stage = 0; stage < S::kStages; ++stage) {
        if (stage < k_tiles) w4a4_stage<S, kSwiGlu>(a, act, sh, stage, stage, token_begin, block_row);
        cp_commit();
    }

    float acc[S::kMmaM][S::kMmaN][4] = {};
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int warp_m = warp / S::kWarpsN;
    const int warp_n = warp - warp_m * S::kWarpsN;
    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row       = lane >> 2;

#pragma unroll 1
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int stage = k_tile % S::kStages;
        cp_wait<S::kStages - 1>();
        __syncthreads();
        unsigned af[S::kMmaM][4];
        unsigned bf[S::kMmaN][2];
        unsigned as[S::kMmaM];
        unsigned bs[S::kMmaN];
#pragma unroll
        for (int m = 0; m < S::kMmaM; ++m) {
            const int row = warp_m * S::kWarpM + m * 16 + a_row_offset;
            ldmatrix_x4(af[m][0], af[m][1], af[m][2], af[m][3],
                        smem_addr(sh.a_codes[stage] + row * 32 + w4a4_swizzled_byte(row, a_column_byte)));
            as[m] = sh.a_scale4[stage][warp_m * S::kWarpM + m * 16 + sfa_row];
        }
#pragma unroll
        for (int n = 0; n < S::kMmaN; ++n) {
            const int row = warp_n * S::kWarpN + n * 8 + b_row_offset;
            ldmatrix_x2(bf[n][0], bf[n][1],
                        smem_addr(sh.b_codes[stage] + row * 32 + w4a4_swizzled_byte(row, b_column_byte)));
            bs[n] = sh.b_scale4[stage][warp_n * S::kWarpN + n * 8 + sfb_row];
        }
#pragma unroll
        for (int m = 0; m < S::kMmaM; ++m)
#pragma unroll
            for (int n = 0; n < S::kMmaN; ++n)
                mma_nvfp4_e4m3(acc[m][n][0], acc[m][n][1], acc[m][n][2], acc[m][n][3], af[m][0],
                               af[m][1], af[m][2], af[m][3], bf[n][0], bf[n][1], as[m], bs[n]);
        __syncthreads();
        const int next = k_tile + S::kStages;
        if (next < k_tiles) w4a4_stage<S, kSwiGlu>(a, act, sh, stage, next, token_begin, block_row);
        cp_commit();
    }

    const int acc_row = lane >> 2;
    const int acc_col = 2 * (lane & 3);
    const float inv_in = 1.0F / act.input_divisor;
#pragma unroll
    for (int m = 0; m < S::kMmaM; ++m) {
        const int t0 = token_begin + warp_m * S::kWarpM + m * 16 + acc_row;
#pragma unroll
        for (int n = 0; n < S::kMmaN; n += (kSwiGlu ? 2 : 1)) {
            const int tile_row = warp_n * S::kWarpN + n * 8 + acc_col;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int local = w4a4_local_row<kSwiGlu, S::kBlockN>(block_row, tile_row + c);
                if (local >= a.rows) continue;
                const int grow   = a.row_begin + local;
                const float s0   = Nvfp4Decoder::row_scale(a, grow) * inv_in;
                float s1         = 0.0F;
                if constexpr (kSwiGlu) s1 = Nvfp4Decoder::row_scale(a, grow + a.up_offset) * inv_in;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int token = t0 + h * 8;
                    if (token >= a.tokens) continue;
                    float value = acc[m][n][h * 2 + c] * s0;
                    if constexpr (kSwiGlu) {
                        value = generic_gate(a, value) * (acc[m][n + 1][h * 2 + c] * s1);
                    }
                    store_output<E>(a, local, token, value);
                }
            }
        }
    }
}

// Row-major NVFP4 activation quantization for any K % 16 == 0.
__global__ __launch_bounds__(256) void w4a4_quantize_kernel(const __nv_bfloat16* __restrict__ x,
                                                            int ldx, int k, int tokens,
                                                            float input_divisor,
                                                            std::uint8_t* __restrict__ codes,
                                                            std::uint8_t* __restrict__ scales) {
    const int groups = k / 16;
    const std::int64_t task =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (task >= static_cast<std::int64_t>(tokens) * groups) return;
    const int token = static_cast<int>(task / groups);
    const int group = static_cast<int>(task - static_cast<std::int64_t>(token) * groups);
    const Nvfp4QuantizedK16 q =
        quantize_nvfp4_k16(x + static_cast<std::int64_t>(token) * ldx + group * 16, input_divisor);
    store_vec(codes + static_cast<std::int64_t>(token) * (k / 2) + group * 8,
              make_uint2(q.codes_lo, q.codes_hi));
    scales[static_cast<std::int64_t>(token) * groups + group] = q.scale;
}

} // namespace ninferno::ops::detail::generic
