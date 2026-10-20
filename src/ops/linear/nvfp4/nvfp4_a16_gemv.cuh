#pragma once

#include "core/pdl.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_operands.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/vector_output.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninferno::ops::detail {

template <int Values>
struct alignas(Values / 2) Nvfp4CodePack {
    static_assert(Values == 8 || Values == 16 || Values == 32);
    std::uint32_t words[Values / 8];
};

static_assert(sizeof(Nvfp4CodePack<8>) == 4);
static_assert(sizeof(Nvfp4CodePack<16>) == 8);
static_assert(sizeof(Nvfp4CodePack<32>) == 16);

template <Nvfp4CodeCache Cache, int Values>
__device__ __forceinline__ Nvfp4CodePack<Values> load_nvfp4_codes(const std::uint8_t* pointer) {
    if constexpr (Cache == Nvfp4CodeCache::Default) {
        return load_vec<Nvfp4CodePack<Values>>(pointer);
    } else if constexpr (Values == 8) {
        Nvfp4CodePack<Values> result;
        asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(result.words[0]) : "l"(pointer));
        return result;
    } else if constexpr (Values == 16) {
        Nvfp4CodePack<Values> result;
        asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];\n"
                     : "=r"(result.words[0]), "=r"(result.words[1])
                     : "l"(pointer));
        return result;
    } else {
        Nvfp4CodePack<Values> result;
        asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(result.words[0]), "=r"(result.words[1]), "=r"(result.words[2]),
                       "=r"(result.words[3])
                     : "l"(pointer));
        return result;
    }
}

template <class Geometry, class Schedule>
struct Nvfp4A16GemvSharedStorage {
    static constexpr int kRawScaleBytes = Schedule::kScaleAccess == Nvfp4ScaleAccess::StagedRaw
                                              ? Schedule::kBlockRows * Geometry::kGroupsPerRow
                                              : 16;
    alignas(16) std::uint8_t raw_scales[kRawScaleBytes];
};

template <class Geometry, class Schedule>
__device__ __forceinline__ void
stage_nvfp4_scales(const std::uint8_t* __restrict__ scales,
                   Nvfp4A16GemvSharedStorage<Geometry, Schedule>& shared, int m_tile,
                   int rmod_base) {
    if constexpr (Schedule::kScaleAccess == Nvfp4ScaleAccess::StagedRaw) {
        constexpr int kQuartetsPerCta = Schedule::kBlockRows / 4;
        constexpr int kTasks          = Geometry::kScaleTilesPerRow * kQuartetsPerCta;
        for (int task = static_cast<int>(threadIdx.x); task < kTasks; task += Schedule::kThreads) {
            const int scale_tile = task / kQuartetsPerCta;
            const int quartet    = task - scale_tile * kQuartetsPerCta;
            const std::int64_t source_offset =
                static_cast<std::int64_t>(m_tile * Geometry::kScaleTilesPerRow + scale_tile) * 512 +
                static_cast<std::int64_t>(rmod_base + quartet) * 16;
            const uint4 packed           = load_vec<uint4>(scales + source_offset);
            const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
            for (int quartile = 0; quartile < 4; ++quartile) {
                const int local_row = quartet * 4 + quartile;
                auto* destination   = reinterpret_cast<std::uint32_t*>(
                    shared.raw_scales + local_row * Geometry::kGroupsPerRow + scale_tile * 4);
                *destination = words[quartile];
            }
        }
        __syncthreads();
    }
}

template <class Geometry>
__device__ __forceinline__ std::int64_t nvfp4_scale_offset(int parent_row, int group) {
    return nvfp4_scale_byte<Geometry::kScaleTilesPerRow>(parent_row, group);
}

// A K that is not a whole number of lane phases ends in a ragged phase: the lanes whose values lie
// past K hold no codes, scales or activations. K is always a whole number of 64-value scale tiles
// and a lane covers at most 32 values, so a lane is either wholly inside K or wholly past it.
// Every K that divides into whole phases compiles the ragged checks away.
template <class Geometry, int ValuesPerPhase>
inline constexpr bool kNvfp4RaggedK = (Geometry::kInputRows % ValuesPerPhase) != 0;

template <class Geometry, int ValuesPerPhase>
inline constexpr int kNvfp4Phases = (Geometry::kInputRows + ValuesPerPhase - 1) / ValuesPerPhase;

template <class Geometry, class Schedule>
__device__ __forceinline__ std::uint32_t
load_staged_scale_word(const Nvfp4A16GemvSharedStorage<Geometry, Schedule>& shared, int local_row,
                       int phase, int lane) {
    constexpr int kSubgroupWidth  = 64 / Schedule::kValuesPerLane;
    constexpr int kGroupsPerPhase = (32 * Schedule::kValuesPerLane) / 16;
    const int subgroup_lane       = lane & (kSubgroupWidth - 1);
    const int group_base          = phase * kGroupsPerPhase + (lane / kSubgroupWidth) * 4;
    std::uint32_t word            = 0;
    // The word holder covers the same 64 values as its subgroup, so past K the whole subgroup is
    // dead; the shuffle below still runs on every lane.
    bool inside = true;
    if constexpr (kNvfp4RaggedK<Geometry, 32 * Schedule::kValuesPerLane>) {
        inside = group_base < Geometry::kGroupsPerRow;
    }
    if (subgroup_lane == 0 && inside) {
        word = *reinterpret_cast<const std::uint32_t*>(
            shared.raw_scales + local_row * Geometry::kGroupsPerRow + group_base);
    }
    return __shfl_sync(0xffffffffU, word, 0, kSubgroupWidth);
}

template <class Geometry, class Schedule>
__device__ __forceinline__ void load_nvfp4_coefficients(
    const std::uint8_t* __restrict__ scales,
    const Nvfp4A16GemvSharedStorage<Geometry, Schedule>& shared, int parent_row, int local_row,
    int phase, int lane, float inverse_weight_divisor,
    float (&coefficients)[Schedule::kValuesPerLane < 16 ? 1 : Schedule::kValuesPerLane / 16]) {
    constexpr int kGroupsPerLane =
        Schedule::kValuesPerLane < 16 ? 1 : Schedule::kValuesPerLane / 16;
    constexpr int kLanesPerGroup =
        Schedule::kValuesPerLane < 16 ? 16 / Schedule::kValuesPerLane : 1;
    const int value_begin = phase * 32 * Schedule::kValuesPerLane + lane * Schedule::kValuesPerLane;
    const int group_begin = value_begin / 16;

    if constexpr (Schedule::kScaleAccess == Nvfp4ScaleAccess::StagedRaw) {
        constexpr int kSubgroupWidth = 64 / Schedule::kValuesPerLane;
        const int subgroup_lane      = lane & (kSubgroupWidth - 1);
        const int first_byte         = (subgroup_lane / kLanesPerGroup) * kGroupsPerLane;
        const std::uint32_t word =
            load_staged_scale_word<Geometry, Schedule>(shared, local_row, phase, lane);
#pragma unroll
        for (int group = 0; group < kGroupsPerLane; ++group) {
            const auto scale    = static_cast<std::uint8_t>(word >> (8 * (first_byte + group)));
            coefficients[group] = decode_nvfp4_e4m3(scale) * inverse_weight_divisor;
        }
    } else {
#pragma unroll
        for (int group = 0; group < kGroupsPerLane; ++group) {
            bool inside = true;
            if constexpr (kNvfp4RaggedK<Geometry, 32 * Schedule::kValuesPerLane>) {
                inside = group_begin + group < Geometry::kGroupsPerRow;
            }
            const std::uint8_t scale =
                inside ? scales[nvfp4_scale_offset<Geometry>(parent_row, group_begin + group)]
                       : std::uint8_t{0};
            coefficients[group] = decode_nvfp4_e4m3(scale) * inverse_weight_divisor;
        }
    }
}

template <class Geometry, class Schedule>
__device__ __forceinline__ void
compute_nvfp4_rows(const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
                   const std::uint8_t* __restrict__ scales,
                   const Nvfp4A16GemvSharedStorage<Geometry, Schedule>& shared,
                   float inverse_weight_divisor, const int (&parent_rows)[Schedule::kRowsPerWarp],
                   int flat_row0, int lane,
                   float (&accumulators)[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains]) {
    constexpr int kValuesPerPhase = 32 * Schedule::kValuesPerLane;
    constexpr int kPhases         = kNvfp4Phases<Geometry, kValuesPerPhase>;
    constexpr int kGroupsPerLane =
        Schedule::kValuesPerLane < 16 ? 1 : Schedule::kValuesPerLane / 16;
    static_assert((Geometry::kInputRows % 64) == 0);
    using CodePack = Nvfp4CodePack<Schedule::kValuesPerLane>;
    // Codes stream from DRAM exactly once, so a warp keeps kCodeDepth phases of them in flight
    // (about 64 bytes per lane); left to itself the compiler issues them one or two phases ahead
    // of the FMAs, and a single wave of CTAs (a short output, such as an MLP down projection)
    // then leaves the memory system underfed. The ring is static: the phase loop is unrolled.
    constexpr int kCodeBytesPerPhase = Schedule::kRowsPerWarp * Schedule::kValuesPerLane / 2;
    constexpr int kCodeDepth         = ring_depth(64, kCodeBytesPerPhase, kPhases);
    CodePack ring[kCodeDepth][Schedule::kRowsPerWarp];
    const auto load_phase_codes = [&](int phase, CodePack(&slot)[Schedule::kRowsPerWarp]) {
        // Only the last phase of a ragged K has dead lanes; they skip their loads and FMAs.
        if constexpr (kNvfp4RaggedK<Geometry, kValuesPerPhase>) {
            if (phase * kValuesPerPhase + lane * Schedule::kValuesPerLane >= Geometry::kInputRows)
                return;
        }
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            const std::int64_t code_offset =
                static_cast<std::int64_t>(parent_rows[local_row]) * Geometry::kCodeBytesPerRow +
                phase * (kValuesPerPhase / 2) + lane * (Schedule::kValuesPerLane / 2);
            slot[local_row] = load_nvfp4_codes<Schedule::kCodeCache, Schedule::kValuesPerLane>(
                codes + code_offset);
        }
    };
#pragma unroll
    for (int phase = 0; phase < kCodeDepth; ++phase) load_phase_codes(phase, ring[phase]);
    // A programmatic dependent launch (where the device has PDL): the codes in the ring are this
    // layer's weights; x, and whatever the epilogue reads, are the previous kernel's. After the
    // wait the next kernel may launch: it waits for this one before it reads what this one
    // writes, and a joined call reading x without waiting finds it complete.
    pdl::wait_for_dependencies();
    if (threadIdx.x == 0) pdl::trigger_dependents();

#pragma unroll
    for (int phase = 0; phase < kPhases; ++phase) {
        bool live = true;
        if constexpr (kNvfp4RaggedK<Geometry, kValuesPerPhase>) {
            live = phase * kValuesPerPhase + lane * Schedule::kValuesPerLane < Geometry::kInputRows;
        }
        float coefficients[Schedule::kRowsPerWarp][kGroupsPerLane];
        CodePack row_codes[Schedule::kRowsPerWarp];
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            load_nvfp4_coefficients<Geometry, Schedule>(
                scales, shared, parent_rows[local_row], flat_row0 + local_row, phase, lane,
                inverse_weight_divisor, coefficients[local_row]);
            row_codes[local_row] = ring[phase % kCodeDepth][local_row];
        }
        if (phase + kCodeDepth < kPhases)
            load_phase_codes(phase + kCodeDepth, ring[phase % kCodeDepth]);
        if (!live) { continue; }

        // A lane's activations are contiguous (x is the start of an activation row, K is a
        // multiple of 64, so every lane slice is 16-byte aligned).
        const auto activations = load_bf16_words<Schedule::kValuesPerLane>(
            x + phase * kValuesPerPhase + lane * Schedule::kValuesPerLane);

#pragma unroll
        for (int pair = 0; pair < Schedule::kPairsPerLane; ++pair) {
            const float2 activation = bf16x2_bits_to_float2(activations.words[pair]);
            const int group         = ((lane * Schedule::kValuesPerLane & 15) + pair * 2) / 16;
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                const std::uint32_t word  = row_codes[local_row].words[pair / 4];
                const std::uint8_t packed = static_cast<std::uint8_t>(word >> (8 * (pair & 3)));
                const float2 code         = decode_nvfp4_e2m1x2(packed);
                const float coefficient   = coefficients[local_row][group];
                constexpr int kChainMask  = Schedule::kAccumulatorChains - 1;
                accumulators[local_row][(2 * pair) & kChainMask] =
                    fmaf(code.x * coefficient, activation.x,
                         accumulators[local_row][(2 * pair) & kChainMask]);
                accumulators[local_row][(2 * pair + 1) & kChainMask] =
                    fmaf(code.y * coefficient, activation.y,
                         accumulators[local_row][(2 * pair + 1) & kChainMask]);
            }
        }
    }
}

// SIMT rows follow the stored M128 scale permutation. A paired policy maps
// adjacent local rows to the two semantic branches owned by the same warp.
template <class Schedule, class Rows>
__device__ __forceinline__ int nvfp4_a16_parent_row(int block, int local, int rows, Rows policy) {
    constexpr int branches = Rows::kPaired ? 2 : 1;
    const int flat         = block * (Schedule::kBlockRows / branches) + local / branches;
    const int physical     = (flat / 128) * 128 + ((flat & 127) >> 2) + (flat & 3) * 32;
    if constexpr (Rows::kPaired)
        return policy.weight_row(physical, local & 1, rows);
    else
        return policy.weight_row(0, physical, rows);
}

template <class Geometry, class Schedule, class Rows>
__device__ __forceinline__ void
nvfp4_stage_a16_scales(const std::uint8_t* scales,
                       Nvfp4A16GemvSharedStorage<Geometry, Schedule>& shared, int block, int rows,
                       Rows policy) {
    if constexpr ([] {
                      if constexpr (requires { Rows::kContiguous; })
                          return Rows::kContiguous;
                      else
                          return false;
                  }()) {
        constexpr int ctas = 128 / Schedule::kBlockRows;
        const int tile = block / ctas, within = block % ctas;
        stage_nvfp4_scales<Geometry, Schedule>(scales, shared, tile,
                                               within * (Schedule::kBlockRows / 4));
    } else if constexpr (Schedule::kScaleAccess == Nvfp4ScaleAccess::StagedRaw) {
        for (int item = threadIdx.x; item < Schedule::kBlockRows * Geometry::kScaleTilesPerRow;
             item += Schedule::kThreads) {
            const int local  = item / Geometry::kScaleTilesPerRow,
                      group  = (item % Geometry::kScaleTilesPerRow) * 4;
            const int parent = nvfp4_a16_parent_row<Schedule>(block, local, rows, policy);
            cp_async<4>(shared.raw_scales + local * Geometry::kGroupsPerRow + group,
                        scales + nvfp4_scale_offset<Geometry>(parent, group));
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();
    }
}

template <class Schedule, class Output, class Epilogue, class Rows>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void nvfp4_a16_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float alpha, Output output, Epilogue epilogue,
    Rows policy, int rows) {
    using Geometry = Nvfp4Geometry<128, Schedule::kStaticK>;
    static_assert(Schedule::kBlockRows % 4 == 0 && 128 % Schedule::kBlockRows == 0);
    static_assert(!Rows::kPaired || Schedule::kRowsPerWarp % 2 == 0);
    __shared__ Nvfp4A16GemvSharedStorage<Geometry, Schedule> shared;
    const int block = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    nvfp4_stage_a16_scales<Geometry, Schedule>(scales, shared, block, rows, policy);
    const int local0 = warp * Schedule::kRowsPerWarp;
    int parent[Schedule::kRowsPerWarp];
#pragma unroll
    for (int r = 0; r < Schedule::kRowsPerWarp; ++r)
        parent[r] = nvfp4_a16_parent_row<Schedule>(block, local0 + r, rows, policy);
    float accumulators[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains] = {};
    compute_nvfp4_rows<Geometry, Schedule>(x, codes, scales, shared, alpha, parent, local0, lane,
                                           accumulators);
    float totals[Schedule::kRowsPerWarp];
#pragma unroll
    for (int r = 0; r < Schedule::kRowsPerWarp; ++r) {
        float sum = 0;
#pragma unroll
        for (int c = 0; c < Schedule::kAccumulatorChains; ++c) sum += accumulators[r][c];
        totals[r] = warp_reduce_sum(sum);
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < Schedule::kRowsPerWarp; r += Rows::kPaired ? 2 : 1) {
            if constexpr (Rows::kPaired)
                epilogue.apply_pair(output, parent[r], 0, totals[r], totals[r + 1]);
            else {
                const float value[1]{totals[r]};
                linear_finish_row(output, epilogue, parent[r], 0, value, 1);
            }
        }
    }
}
} // namespace ninferno::ops::detail
