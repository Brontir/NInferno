#pragma once

// Reusable row-scaled FP8 T=1 CUDA-core mainloop. Each warp owns contiguous output rows and reads
// naturally aligned E4M3 code packs. The persistent codes are decoded exactly, multiplied by the
// represented BF16 activation with FP32 FMA, reduced within the warp, and scaled once by the
// owning BF16 row multiplier. Output owns the physical row mapping so fused projections reuse the
// same weight/decode mainloop without materializing a packed parent output.

#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/fp8/fp8_a16_codec.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/fp8/fp8_operands.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninferno::ops::detail {

template <int Values>
struct alignas(Values) Fp8CodePack {
    static_assert(Values == 8 || Values == 16 || Values == 32);
    std::uint32_t words[Values / 4];
};

static_assert(sizeof(Fp8CodePack<8>) == 8);
static_assert(sizeof(Fp8CodePack<16>) == 16);
static_assert(sizeof(Fp8CodePack<32>) == 32);

template <Fp8CodeCache Cache, int Values>
__device__ __forceinline__ Fp8CodePack<Values> load_fp8_codes(const std::uint8_t* pointer) {
    if constexpr (Values == 32) {
        Fp8CodePack<Values> result;
        const Fp8CodePack<16> low  = load_fp8_codes<Cache, 16>(pointer);
        const Fp8CodePack<16> high = load_fp8_codes<Cache, 16>(pointer + 16);
#pragma unroll
        for (int word = 0; word < 4; ++word) {
            result.words[word]     = low.words[word];
            result.words[word + 4] = high.words[word];
        }
        return result;
    } else if constexpr (Cache == Fp8CodeCache::Default) {
        Fp8CodePack<Values> result;
        if constexpr (Values == 8) {
            const uint2 v   = load_ldg<uint2>(pointer);
            result.words[0] = v.x;
            result.words[1] = v.y;
        } else {
            const uint4 v   = load_ldg<uint4>(pointer);
            result.words[0] = v.x;
            result.words[1] = v.y;
            result.words[2] = v.z;
            result.words[3] = v.w;
        }
        return result;
    } else if constexpr (Values == 8) {
        Fp8CodePack<Values> result;
        asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];\n"
                     : "=r"(result.words[0]), "=r"(result.words[1])
                     : "l"(pointer));
        return result;
    } else {
        Fp8CodePack<Values> result;
        asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(result.words[0]), "=r"(result.words[1]), "=r"(result.words[2]),
                       "=r"(result.words[3])
                     : "l"(pointer));
        return result;
    }
}

__device__ __forceinline__ float2 decode_fp8_e4m3x2(std::uint16_t storage) {
    return fp8_e4m3x2_to_float2(storage);
}

template <int Values, int Rows, int AccumulatorChains>
__device__ __forceinline__ void accumulate_rows(const Fp8CodePack<Values> (&codes)[Rows],
                                                const std::uint32_t* activation_pairs,
                                                float (&accumulators)[Rows][AccumulatorChains]) {
    constexpr int kChainMask = AccumulatorChains - 1;
#pragma unroll
    for (int pair = 0; pair < Values / 2; ++pair) {
        const float2 value = bf16x2_bits_to_float2(activation_pairs[pair]);
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            const std::uint32_t word   = codes[row].words[pair >> 1];
            const std::uint16_t packed = static_cast<std::uint16_t>(word >> ((pair & 1) * 16));
            const float2 weight        = decode_fp8_e4m3x2(packed);
            accumulators[row][(2 * pair) & kChainMask] =
                fmaf(weight.x, value.x, accumulators[row][(2 * pair) & kChainMask]);
            accumulators[row][(2 * pair + 1) & kChainMask] =
                fmaf(weight.y, value.y, accumulators[row][(2 * pair + 1) & kChainMask]);
        }
    }
}

template <class Schedule, class Output, class Epilogue, class RowPolicy>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_a16_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ weight_codes,
    const __nv_bfloat16* __restrict__ row_scales, Output output, Epilogue epilogue,
    RowPolicy row_policy, int rows, int input_rows) {
    constexpr bool PairRows       = RowPolicy::kPaired;
    const int K                   = Schedule::kStaticK ? Schedule::kStaticK : input_rows;
    constexpr int kValuesPerPhase = kWarpSize * Schedule::kValuesPerLane;
    static_assert(!PairRows || (Schedule::kRowsPerWarp % 2) == 0);
    const int kPhases = K / kValuesPerPhase;
    constexpr int kStoredRowsPerWarp =
        PairRows ? Schedule::kRowsPerWarp / 2 : Schedule::kRowsPerWarp;
    constexpr int kStoredRowsPerCta = Schedule::kWarpsPerCta * kStoredRowsPerWarp;

    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int row_begin =
        static_cast<int>(blockIdx.x) * kStoredRowsPerCta + warp * kStoredRowsPerWarp;
    float accumulators[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains] = {};

    // Codes stream from DRAM exactly once, so a warp keeps kCodeDepth phases of them in flight
    // (about 32 bytes per lane) in a ring whose slots are static: the phase loop advances by
    // whole rings. 64 bytes per lane took the decode linear_add from 40 to 77 registers (three
    // CTAs per SM instead of six) and cost Qwen3.8-27B 1.2 % in serve; 32 bytes keeps 52. Left to
    // itself the compiler issued the codes one or two phases ahead of the FMAs, which underfeeds
    // the memory system when the CTAs fit in a single wave.
    using CodePack                   = Fp8CodePack<Schedule::kValuesPerLane>;
    constexpr int kCodeBytesPerPhase = Schedule::kRowsPerWarp * Schedule::kValuesPerLane;
    constexpr int kCodeDepth         = ring_depth(32, kCodeBytesPerPhase, kPhases);
    CodePack ring[kCodeDepth][Schedule::kRowsPerWarp];
    const auto load_phase_codes = [&](int phase, CodePack(&slot)[Schedule::kRowsPerWarp]) {
        const int value_begin = phase * kValuesPerPhase + lane * Schedule::kValuesPerLane;
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            const int weight_row = row_policy.weight_row(row_begin, local_row, rows);
            const auto* source =
                weight_codes + static_cast<std::int64_t>(weight_row) * K + value_begin;
            if constexpr (Schedule::kCodeCache == Fp8CodeCache::Default &&
                          Schedule::kValuesPerLane <= 16)
                slot[local_row] = load_vec<CodePack>(source);
            else
                slot[local_row] =
                    load_fp8_codes<Schedule::kCodeCache, Schedule::kValuesPerLane>(source);
        }
    };
#pragma unroll
    for (int slot = 0; slot < kCodeDepth; ++slot) load_phase_codes(slot, ring[slot]);
    // A programmatic dependent launch (where the device has PDL): the codes in the ring are this
    // layer's weights; x, and whatever the epilogue reads, are the previous kernel's. After the
    // wait the next kernel may launch: it waits for this one before it reads what this one
    // writes, and a joined call reading x without waiting finds it complete.
    pdl::wait_for_dependencies();
    if (threadIdx.x == 0) pdl::trigger_dependents();

    for (int base = 0; base < kPhases; base += kCodeDepth) {
#pragma unroll
        for (int slot = 0; slot < kCodeDepth; ++slot) {
            const int phase = base + slot;
            if (phase < kPhases) {
                CodePack row_codes[Schedule::kRowsPerWarp];
#pragma unroll
                for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row)
                    row_codes[local_row] = ring[slot][local_row];
                if (phase + kCodeDepth < kPhases) load_phase_codes(phase + kCodeDepth, ring[slot]);
                // A lane's activations are contiguous and 16-byte aligned (K is a multiple of
                // the phase width).
                const auto activation = load_bf16_words<Schedule::kValuesPerLane>(
                    x + phase * kValuesPerPhase + lane * Schedule::kValuesPerLane);
                accumulate_rows(row_codes, activation.words, accumulators);
            }
        }
    }

    const auto destination = linear_output_tile<kStoredRowsPerCta>(
        output, static_cast<int>(blockIdx.x) * kStoredRowsPerCta);
    if constexpr (PairRows) {
        float totals[Schedule::kRowsPerWarp];
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            float total = 0.0F;
#pragma unroll
            for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
                total += accumulators[local_row][chain];
            }
            totals[local_row] = warp_reduce_sum(total);
        }
        if (lane == 0) {
#pragma unroll
            for (int local_row = 0; local_row < kStoredRowsPerWarp; ++local_row) {
                const int gate_row = row_policy.weight_row(row_begin, local_row, rows);
                const int up_row =
                    row_policy.weight_row(row_begin, kStoredRowsPerWarp + local_row, rows);
                const float gate =
                    totals[local_row] * __bfloat162float(__ldg(row_scales + gate_row));
                const float up = totals[kStoredRowsPerWarp + local_row] *
                                 __bfloat162float(__ldg(row_scales + up_row));
                epilogue.apply_pair(destination, row_begin + local_row, 0, gate, up);
            }
        }
    } else {
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            float total = 0.0F;
#pragma unroll
            for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
                total += accumulators[local_row][chain];
            }
            total = warp_reduce_sum(total);
            if (lane == 0) {
                const int parent_row = row_policy.weight_row(row_begin, local_row, rows);
                const float values[1]{total * __bfloat162float(__ldg(row_scales + parent_row))};
                linear_finish_row(destination, epilogue, parent_row, 0, values, 1);
            }
        }
    }
}

} // namespace ninferno::ops::detail
