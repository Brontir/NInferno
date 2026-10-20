#pragma once
#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/common/math.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ops/linear/q8/q8_operands.h"

#include <cstdint>

namespace ninferno::ops::detail {
template <class Schedule, bool FullRows, class Output, class Epilogue>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void q8_a16_gemv_kernel(
    Q8LinearOperands operands, Output output, Epilogue epilogue) {
    constexpr int R = Schedule::kBlockRows, W = Schedule::kWarpsPerRow;
    constexpr int T = Schedule::kBlockTokens, V = Schedule::kValuesPerLane;
    constexpr int kPhaseValues = Schedule::kPhaseValues;
    const int k        = Schedule::kStaticK > 0 ? Schedule::kStaticK : operands.k;
    const int padded_k = Schedule::kStaticK > 0 ? Schedule::kStaticK : operands.padded_k;
    const int tokens   = T == 1 ? 1 : operands.tokens;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int local_row = warp / W, split = warp % W;
    const int row = blockIdx.x * R + local_row;
    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.0f;
    // A programmatic dependent launch (where the device has PDL): this row's first codes go to L2
    // (about 64 bytes per lane, the NVFP4 and FP8 GEMVs' ring depth) while the previous kernel
    // finishes; x and whatever the epilogue reads are its. After the wait the next kernel may
    // launch: it waits for this one before it reads what this one writes.
    if (FullRows || row < operands.rows) {
        const auto* code_row          = operands.codes + static_cast<std::int64_t>(row) * padded_k;
        constexpr int kPrefetchPhases = 64 / V;
#pragma unroll
        for (int p = 0; p < kPrefetchPhases; ++p) {
            const int kk = (split + p * W) * kPhaseValues + lane * V;
            if (kk < k) asm volatile("prefetch.global.L2 [%0];" ::"l"(code_row + kk));
        }
    }
    pdl::wait_for_dependencies();
    if (threadIdx.x == 0) pdl::trigger_dependents();
    if (FullRows || row < operands.rows) {
        const auto* code_row = operands.codes + static_cast<std::int64_t>(row) * padded_k;
        const auto* scale_row =
            operands.scales + static_cast<std::int64_t>(row) * (padded_k / 32) * 2;
#pragma unroll Schedule::kUnroll
        for (int phase = split; phase < (k + kPhaseValues - 1) / kPhaseValues; phase += W) {
            // A phase spans V G32 groups: lane g loads group g's scale, lane l reads it from the
            // lane that loaded the group its codes fall in.
            unsigned bits = 0;
            if (lane < V && (Schedule::kStaticK > 0 || phase * kPhaseValues + lane * 32 < k))
                bits = load_ldg<std::uint16_t>(scale_row + (phase * V + lane) * 2);
            bits              = __shfl_sync(0xffffffffu, bits, lane * V / 32);
            const float scale = __half2float(__ushort_as_half(bits));
            const int kk      = phase * kPhaseValues + lane * V;
            if (Schedule::kStaticK > 0 || kk < k) {
                std::uint32_t words[V / 4];
                if constexpr (V == 16) {
                    const uint4 codes = load_ldg<uint4>(code_row + kk);
                    words[0] = codes.x, words[1] = codes.y, words[2] = codes.z, words[3] = codes.w;
                } else {
                    const uint2 codes = load_ldg<uint2>(code_row + kk);
                    words[0] = codes.x, words[1] = codes.y;
                }
                float weight[V];
#pragma unroll
                for (int i = 0; i < V; ++i)
                    weight[i] =
                        static_cast<float>(static_cast<std::int8_t>(words[i / 4] >> ((i & 3) * 8))) *
                        scale;
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    if (T == 1 || t < tokens) {
                        const auto* xt = operands.x + static_cast<std::int64_t>(t) * k + kk;
#pragma unroll
                        for (int chunk = 0; chunk < V / 8; ++chunk) {
                            const uint4 values = load_ldg<uint4>(xt + chunk * 8);
                            const float2 xv[]{
                                bf16x2_bits_to_float2(values.x), bf16x2_bits_to_float2(values.y),
                                bf16x2_bits_to_float2(values.z), bf16x2_bits_to_float2(values.w)};
#pragma unroll
                            for (int i = 0; i < 8; ++i)
                                acc[t] = fmaf(weight[chunk * 8 + i],
                                              (i & 1) ? xv[i / 2].y : xv[i / 2].x, acc[t]);
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = warp_reduce_sum(acc[t]);
    const auto tile = linear_output_tile<R>(output, blockIdx.x * R);
    if constexpr (W > 1) {
        __shared__ float partial[R][T][W];
        if (lane == 0) {
#pragma unroll
            for (int t = 0; t < T; ++t) partial[local_row][t][split] = acc[t];
        }
        __syncthreads();
        // One thread per (row, token) adds the row's W partials in split order.
        for (int task = threadIdx.x; task < R * T; task += Schedule::kThreads) {
            const int r = task / T, t = task - r * T, out_row = blockIdx.x * R + r;
            if (t < tokens && (FullRows || out_row < operands.rows)) {
                float values[1]{partial[r][t][0]};
#pragma unroll
                for (int i = 1; i < W; ++i) values[0] += partial[r][t][i];
                linear_finish_row(tile, epilogue, out_row, t, values, 1);
            }
        }
    } else if (lane == 0 && (FullRows || row < operands.rows)) {
        linear_finish_row(tile, epilogue, row, 0, acc, tokens);
    }
}
} // namespace ninferno::ops::detail
