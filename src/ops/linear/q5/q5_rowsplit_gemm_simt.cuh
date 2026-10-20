#pragma once

// ninfer's kernel, taken back from ninfer-all (Apache-2.0, see NOTICE), which kept it after ninfer
// replaced it; changed here as the git history records.

// Warp-per-row small-T row-split low-bit GEMM: out[N,T] = W[N,K] . x[K,T].
//
// This is Q5's format-local small-column path. The design target is the DRAM
// roofline: stream the Q5 payload once per column tile at copy-ceiling
// bandwidth, so the cost of T<=tile width is nearly flat versus T==1.
//
//   - One warp owns one output row. blockIdx.y selects a tile of kTt activation
//     columns; for T <= kTt the weights are streamed exactly once.
//   - K is processed in 1024-value slabs staged through shared memory with a
//     cp.async double buffer (same structure as the tuned T==1 Q5 core): all
//     weight-plane reads are coalesced 128-bit loads, so DRAM latency is hidden
//     without relying on occupancy alone. Deeper pipelines and launch-bounds
//     occupancy caps were both measured slower (spills / no gain, see report).
//   - Consume is phase-interleaved: in phase c (0..3), lane L owns the 8
//     consecutive K-values [256c + 8L, 256c + 8L + 8). The warp's 32 x loads per
//     (column, phase) are one 16-byte uint4 per lane covering 512 *consecutive*
//     bytes, i.e. minimal L1 wavefronts (a per-lane-contiguous layout would
//     stride lanes 64B apart and burn 4x the L1 throughput on the same bytes).
//     The weight planes index out conflict-free or broadcast under the same
//     ownership.
//   - Dequant uses the fp16 mantissa trick: nibbles are OR-ed into the mantissa
//     of 1024.0f16 (plus Q5 high bits at mantissa bit 4), the sign
//     fixup is a constant xor folded into the packed word, and one hsub2 yields
//     two exact signed values; the group scale is applied per 8-value chunk.
//     fp32 FMA into per-column accumulators; warp-shuffle reduction per column.
//   - kTt is 4 or 8 only. Larger column tiles blow past the register budget
//     (kTt=16 needs ~98 regs -> 2 blocks/SM -> latency-bound at ~20% of DRAM),
//     so T > 8 re-streams the weights once per 8-column tile instead; the mma
//     tensor-core GEMM takes over where that stops winning.
//
// Correctness for arbitrary shapes: full 1024-value slabs require k % 8 == 0
// (16-byte aligned x columns) and cover [0, k/1024*1024); every remaining group
// (tail of odd k, or all groups when k % 8 != 0) uses the scalar per-pair path
// reading global memory directly, masked at the k boundary. Weights in the
// padded region [k, padded_k) are never used.

#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninferno::ops::detail {

// Q5 slab traits + dequant. A slab is 1024 K-values:
//   nibble/code bytes : 32 B per group  -> kNibU4 uint4 per slab
//   high-plane bytes  : 0/8/16 B per group -> kHighU4 uint4 per slab
//   scales            : 2 B per group -> kScaleU32 4-byte words per slab
// (Scales are staged with 4-byte cp.async because a row's scale plane is only
// guaranteed 4-byte aligned for generic kg.)
//
// dequant_chunk(s_nib, s_hi, s_sc, c, lane, w): dequantize the lane's 8 values
// of phase c into w[0..7], scale applied. All shared reads are conflict-free or
// broadcast under the phase-interleaved ownership.

struct Q5RowSplitSimtSchedule {
    using Codec                             = Q5ScalarDecodeAtom;
    static constexpr int kNibU4             = 32;
    static constexpr int kHighU4            = 8;
    static constexpr int kScaleU32          = 8;
    static constexpr int kHighBytesPerGroup = 8;

    // u = lo | hi<<4, signed s = (u ^ 16) - 16 = lo + 16*(hi^1) - 16. Build
    // 1024 + lo + 16*(hi^1) in fp16 (hi^1 at mantissa bit 4), subtract 1040.
    __device__ static __forceinline__ void dequant_chunk(const uint4* s_nib, const uint4* s_hi,
                                                         const std::uint32_t* s_sc, int c, int lane,
                                                         float (&w)[8]) {
        const std::uint32_t word = reinterpret_cast<const std::uint32_t*>(s_nib)[c * 32 + lane];
        const std::uint32_t hc = reinterpret_cast<const std::uint8_t*>(s_hi)[c * 32 + lane] ^ 0xffu;
        const float scale      = __half2float(
            __ushort_as_half(reinterpret_cast<const std::uint16_t*>(s_sc)[c * 4 + (lane >> 3)]));
        const __half2 bias = __half2half2(__ushort_as_half(0x6410)); // 1040.0
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            std::uint32_t bits = ((word >> (4 * p)) & 0x000f000fu) | 0x64006400u;
            bits |= (((hc >> p) & 1u) << 4) | (((hc >> (p + 4)) & 1u) << 20);
            const __half2 h = __hsub2(half2_from_bits(bits), bias);
            const float2 f  = __half22float2(h);
            w[p]            = f.x * scale;
            w[p + 4]        = f.y * scale;
        }
    }
};

// The occupancy targets of the split kernels scale with the column tile because their accumulators
// do. At 16 split2 blocks per SM ptxas gets 64 registers and at 10 split4 blocks 51, which the wide
// tiles cannot hold: they spill, and on kernels this bandwidth-bound the spill costs more than the
// extra resident warps return. Narrow tiles fit and keep the warps.
template <int kTt, int kStride>
inline constexpr int kQ5Split2BlocksPerSm = kTt <= 6 ? 16 : 8;
template <int kTt>
inline constexpr int kQ5Split4BlocksPerSm = kTt <= 5 ? 10 : 8;

template <int kTt, int kFullSlabs, int kStride, bool AddResidual = false>
__launch_bounds__(64, (kQ5Split2BlocksPerSm<kTt, kStride>)) __global__
    void q5_rowsplit_gemm_simt_split2_kernel(const __nv_bfloat16* __restrict__ x,
                                             const std::uint8_t* __restrict__ codes,
                                             const std::uint8_t* __restrict__ high,
                                             const std::uint8_t* __restrict__ scales,
                                             __nv_bfloat16* __restrict__ out, std::int32_t n,
                                             std::int32_t padded_k) {
    static_assert(kTt > 0, "direct split2 requires a positive column tile");
    static_assert(kFullSlabs > 0 && kStride > 0, "direct split2 requires exact positive shape");

    __shared__ float s_part[2][kTt];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int part = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x);
    if (row >= n) { return; }

    const int kg_padded          = padded_k / Q5RowSplitStorage::kGroupK;
    const std::uint8_t* code_row = codes + static_cast<std::int64_t>(row) * kg_padded * 32;
    const std::uint8_t* high_row = high + static_cast<std::int64_t>(row) * kg_padded *
                                              Q5RowSplitSimtSchedule::kHighBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kg_padded * 2;

    float acc[kTt];
#pragma unroll
    for (int i = 0; i < kTt; ++i) { acc[i] = 0.0f; }

#pragma unroll
    for (int s = 0; s < kFullSlabs; ++s) {
#pragma unroll
        for (int local = 0; local < 2; ++local) {
            const int chunk = part * 2 + local;
            const std::uint8_t* code_phase =
                code_row + static_cast<std::int64_t>(s) * 512 + chunk * 128 + lane * 4;
            const std::uint8_t* high_phase =
                high_row + static_cast<std::int64_t>(s) * 128 + chunk * 32 + lane;
            const int group_in_slab  = chunk * 4 + (lane >> 3);
            std::uint32_t scale_bits = 0;
            if ((lane & 7) == 0) {
                scale_bits = *reinterpret_cast<const std::uint16_t*>(
                    scale_row + (static_cast<std::int64_t>(s) * 16 + group_in_slab) * 2);
            }
            scale_bits = __shfl_sync(0xffffffffu, scale_bits, lane & ~7);

            const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(code_phase);
            const std::uint32_t hc   = static_cast<std::uint32_t>(*high_phase) ^ 0xffu;
            const float scale        = __half2float(__ushort_as_half(scale_bits));
            const __half2 bias       = __half2half2(__ushort_as_half(0x6410)); // 1040.0
            float w[8];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                std::uint32_t bits = ((word >> (4 * p)) & 0x000f000fu) | 0x64006400u;
                bits |= (((hc >> p) & 1u) << 4) | (((hc >> (p + 4)) & 1u) << 20);
                const __half2 h = __hsub2(half2_from_bits(bits), bias);
                const float2 f  = __half22float2(h);
                w[p]            = f.x * scale;
                w[p + 4]        = f.y * scale;
            }

            const std::int64_t xoff = static_cast<std::int64_t>(s) * 1024 + chunk * 256 + lane * 8;
#pragma unroll
            for (int tt = 0; tt < kTt; ++tt) {
                const uint4 xv =
                    load_vec<uint4>(x + static_cast<std::int64_t>(tt) * kStride + xoff);
                const float2 f0 = bf16x2_bits_to_float2(xv.x);
                const float2 f1 = bf16x2_bits_to_float2(xv.y);
                const float2 f2 = bf16x2_bits_to_float2(xv.z);
                const float2 f3 = bf16x2_bits_to_float2(xv.w);
                acc[tt]         = fmaf(w[0], f0.x, acc[tt]);
                acc[tt]         = fmaf(w[1], f0.y, acc[tt]);
                acc[tt]         = fmaf(w[2], f1.x, acc[tt]);
                acc[tt]         = fmaf(w[3], f1.y, acc[tt]);
                acc[tt]         = fmaf(w[4], f2.x, acc[tt]);
                acc[tt]         = fmaf(w[5], f2.y, acc[tt]);
                acc[tt]         = fmaf(w[6], f3.x, acc[tt]);
                acc[tt]         = fmaf(w[7], f3.y, acc[tt]);
            }
        }
    }

#pragma unroll
    for (int tt = 0; tt < kTt; ++tt) {
        float a = acc[tt];
        a       = warp_reduce_sum(a);
        if (lane == 0) { s_part[part][tt] = a; }
    }

    __syncthreads();

    if (part == 0 && lane < kTt) {
        const std::int64_t index = static_cast<std::int64_t>(lane) * n + row;
        float sum                = s_part[0][lane] + s_part[1][lane];
        if constexpr (AddResidual) { sum += __bfloat162float(out[index]); }
        out[index] = __float2bfloat16(sum);
    }
}

struct Q5Split4StoreEpilogue {
    template <bool SplitOutput, int SplitRow, int Tokens>
    __device__ __forceinline__ void
    operator()(__nv_bfloat16* out, __nv_bfloat16* out_tail, std::int32_t n, std::int32_t out_ld,
               std::int32_t row, const float (&values)[Tokens]) const {
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            if constexpr (SplitOutput) {
                if (row < SplitRow) {
                    out[static_cast<std::int64_t>(token) * out_ld + row] =
                        __float2bfloat16(values[token]);
                } else {
                    out_tail[static_cast<std::int64_t>(token) * (n - SplitRow) + row - SplitRow] =
                        __float2bfloat16(values[token]);
                }
            } else {
                out[static_cast<std::int64_t>(token) * out_ld + row] =
                    __float2bfloat16(values[token]);
            }
        }
    }
};

template <int kTt, int kFullSlabs, int kStride, bool SplitOutput = false, int SplitRow = 0,
          class Epilogue = Q5Split4StoreEpilogue, bool TriggerPdl = false, bool JoinPdl = false>
__launch_bounds__(128, kQ5Split4BlocksPerSm<kTt>) __global__
    void q5_rowsplit_gemm_simt_split4_kernel(
        const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
        const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
        __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ out_tail, std::int32_t n,
        std::int32_t out_ld, std::int32_t padded_k, Epilogue epilogue = {}) {
    static_assert(kTt > 0, "direct split4 requires a positive column tile");
    static_assert(kFullSlabs > 0 && kStride > 0, "direct split4 requires exact positive shape");
    static_assert(!SplitOutput || SplitRow > 0,
                  "split-output Q5 split4 requires a positive compile-time seam");
    if constexpr (TriggerPdl) {
        if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    }
    __shared__ float s_part[4][kTt];

    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int chunk = static_cast<int>(threadIdx.x) >> 5;
    const int row   = static_cast<int>(blockIdx.x);
    if (row >= n) { return; }

    const int kg_padded          = padded_k / Q5RowSplitStorage::kGroupK;
    const std::uint8_t* code_row = codes + static_cast<std::int64_t>(row) * kg_padded * 32;
    const std::uint8_t* high_row = high + static_cast<std::int64_t>(row) * kg_padded *
                                              Q5RowSplitSimtSchedule::kHighBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kg_padded * 2;

    float acc[kTt];
#pragma unroll
    for (int i = 0; i < kTt; ++i) { acc[i] = 0.0f; }

#pragma unroll
    for (int s = 0; s < kFullSlabs; ++s) {
        const std::uint8_t* code_phase =
            code_row + static_cast<std::int64_t>(s) * 512 + chunk * 128 + lane * 4;
        const std::uint8_t* high_phase =
            high_row + static_cast<std::int64_t>(s) * 128 + chunk * 32 + lane;
        const int group_in_slab  = chunk * 4 + (lane >> 3);
        std::uint32_t scale_bits = 0;
        if ((lane & 7) == 0) {
            scale_bits = *reinterpret_cast<const std::uint16_t*>(
                scale_row + (static_cast<std::int64_t>(s) * 16 + group_in_slab) * 2);
        }
        scale_bits = __shfl_sync(0xffffffffu, scale_bits, lane & ~7);

        const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(code_phase);
        const std::uint32_t hc   = static_cast<std::uint32_t>(*high_phase) ^ 0xffu;
        const float scale        = __half2float(__ushort_as_half(scale_bits));
        const __half2 bias       = __half2half2(__ushort_as_half(0x6410)); // 1040.0
        float w[8];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            std::uint32_t bits = ((word >> (4 * p)) & 0x000f000fu) | 0x64006400u;
            bits |= (((hc >> p) & 1u) << 4) | (((hc >> (p + 4)) & 1u) << 20);
            const __half2 h = __hsub2(half2_from_bits(bits), bias);
            const float2 f  = __half22float2(h);
            w[p]            = f.x * scale;
            w[p + 4]        = f.y * scale;
        }

        const std::int64_t xoff = static_cast<std::int64_t>(s) * 1024 + chunk * 256 + lane * 8;
#pragma unroll
        for (int tt = 0; tt < kTt; ++tt) {
            const uint4 xv  = load_vec<uint4>(x + static_cast<std::int64_t>(tt) * kStride + xoff);
            const float2 f0 = bf16x2_bits_to_float2(xv.x);
            const float2 f1 = bf16x2_bits_to_float2(xv.y);
            const float2 f2 = bf16x2_bits_to_float2(xv.z);
            const float2 f3 = bf16x2_bits_to_float2(xv.w);
            acc[tt]         = fmaf(w[0], f0.x, acc[tt]);
            acc[tt]         = fmaf(w[1], f0.y, acc[tt]);
            acc[tt]         = fmaf(w[2], f1.x, acc[tt]);
            acc[tt]         = fmaf(w[3], f1.y, acc[tt]);
            acc[tt]         = fmaf(w[4], f2.x, acc[tt]);
            acc[tt]         = fmaf(w[5], f2.y, acc[tt]);
            acc[tt]         = fmaf(w[6], f3.x, acc[tt]);
            acc[tt]         = fmaf(w[7], f3.y, acc[tt]);
        }
    }

#pragma unroll
    for (int tt = 0; tt < kTt; ++tt) {
        float a = acc[tt];
        a       = warp_reduce_sum(a);
        if (lane == 0) { s_part[chunk][tt] = a; }
    }

    __syncthreads();

    if constexpr (std::is_same_v<Epilogue, Q5Split4StoreEpilogue>) {
        if (chunk == 0 && lane < kTt) {
            float sum = 0.0f;
#pragma unroll
            for (int p = 0; p < 4; ++p) { sum += s_part[p][lane]; }
            if constexpr (SplitOutput) {
                if (row < SplitRow) {
                    out[static_cast<std::int64_t>(lane) * out_ld + row] = __float2bfloat16(sum);
                } else {
                    out_tail[static_cast<std::int64_t>(lane) * (n - SplitRow) + row - SplitRow] =
                        __float2bfloat16(sum);
                }
            } else {
                out[static_cast<std::int64_t>(lane) * out_ld + row] = __float2bfloat16(sum);
            }
        }
    } else {
        if (chunk == 0 && lane < kTt) {
            float sum = 0.0f;
#pragma unroll
            for (int p = 0; p < 4; ++p) { sum += s_part[p][lane]; }
            s_part[0][lane] = sum;
        }
        __syncthreads();
        if (chunk == 0 && lane == 0) {
            epilogue.template operator()<SplitOutput, SplitRow>(out, out_tail, n, out_ld, row,
                                                                s_part[0]);
        }
    }
    if constexpr (JoinPdl) { pdl::wait_for_dependencies(); }
}

} // namespace ninferno::ops::detail
