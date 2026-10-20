#pragma once

// The grouped W4A8 kernel of the generic MoE Op: the expert stages (GateUp, Up, Down) on int8
// tensor cores. It is the generic projection path's W4A8 tile (ops/generic/generic_w4a8.cuh) with
// the MoE addressing of moe_gemm.cuh:
//   weights  the codec decodes the job's expert rows to int8 in shared memory once per 128-value
//            K tile (row_base + l * row_step, + up_offset for the up set), as the dense W4A8 does;
//   columns  blockIdx.y is (job, sub-tile): the job's slots, BlockM per CTA; GateUp / Up gather
//            each slot's token through packed_token from the quantized x, Down reads the slot's
//            row of the quantized intermediate;
//   epilogue the register MMA's (moe_activation on the gate and up sums, biases, BF16 out).
// The activation is quantized beforehand (w4a8_quantize_kernel's codes: int8 per value, a binary32
// scale and the block's sum per 32 values) into the Op's workspace. K tiles accumulate in order and
// nothing splits K, so a slot's bytes do not depend on the call's T or on its job's width.
//
// On the register MMA the experts decode to f16 / bf16 and run on the FP32-accumulate tier
// (RTX 5090 255, 4090 165, 3090 71 TFLOP/s); here they run on the int8 tier (1027 / 660 / 284).

#include "core/device.h"
#include "ops/generic/generic_w4a8.cuh"
#include "ops/moe/moe_gemm.cuh"
#include "ops/moe/moe_internal.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace ninferno::ops::detail::moe {

// The one tile: 64 slots (a whole prefill job, kJobColumns) x 64 weight rows (32 local rows of a
// gated stage), 2 x 2 warps: the dense W4A8's 64 x 64 schedule.
using MoeW4a8Schedule = generic::W4a8Schedule<kMoeW4a8BlockM, kMoeW4a8BlockN, 2, 2>;

// The CTA's weight tile of one K tile: W4a8Weights with the MoE row map. Tile row n of a gated
// stage is the gate (even 8-row group) or up (odd group) row of local row w4a4_local_row(n).
template <class C, class S, int Threads, bool Gated>
struct MoeW4a8Weights {
    static constexpr int kSpan   = generic::w4a8_span<C>();
    static constexpr int kStride = generic::w4a8_stride<C>();
    static constexpr bool kAux   = C::kMin || generic::w4a8_halves<C>();
    static_assert(4 % (kSpan * kStride) == 0);
    static constexpr int kLoads = 4 / kSpan;
    static constexpr int kTasks = S::kBlockN * kLoads;
    static constexpr int kPer   = (kTasks + Threads - 1) / Threads;
    typename C::Raw raw[kPer];
    bool ok[kPer];

    __device__ static __forceinline__ int first_block(int q) {
        return q % kStride + (q / kStride) * kSpan * kStride;
    }

    __device__ __forceinline__ void load(const GemmArgs& a, std::int64_t row_base, int block_row,
                                         int tid, int k_tile) {
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int task  = tid + i * Threads;
            const int n     = task / kLoads;
            const int local = generic::w4a4_local_row<Gated, S::kBlockN>(block_row, n);
            const int kb    = k_tile * 4 + first_block(task % kLoads);
            // A K of whole 32-value blocks only: the last tile's blocks past K stay zero (a span
            // codec is a superblock format, whose K is whole superblocks and tiles).
            ok[i] = task < kTasks && local < a.rows && kb * 32 < a.w.k;
            if (ok[i]) {
                const bool up = Gated && ((n >> 3) & 1);
                const int wrow =
                    static_cast<int>(row_base + static_cast<std::int64_t>(local) * a.row_step +
                                     (up ? a.up_offset : 0));
                raw[i] = C::load(a.w, wrow, kb);
            }
        }
    }

    template <class B>
    __device__ __forceinline__ void store(const B& buf, int tid,
                                          const std::uint32_t* lut = nullptr) const {
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int task = tid + i * Threads;
            if (task >= kTasks) continue;
            const int n = task / kLoads, j0 = first_block(task % kLoads);
            auto* row = buf.b_codes + n * 128;
#pragma unroll
            for (int sp = 0; sp < kSpan; ++sp) {
                const int j = j0 + sp * kStride;
                uint4 lo = make_uint4(0, 0, 0, 0), hi = lo;
                float scale = 0.0F, min = 0.0F, min_lo = 0.0F, min_hi = 0.0F;
                if (ok[i]) {
                    if constexpr (generic::w4a8_half_min<C>()) {
                        C::decode(raw[i], sp, lo, hi, scale, min, min_lo, min_hi);
                    } else if constexpr (kSpan == 1) {
                        C::decode(raw[i], lo, hi, scale, min);
                    } else if constexpr (generic::w4a8_lut_words<C>() > 0) {
                        C::decode(raw[i], sp, lut, lo, hi, scale, min);
                    } else {
                        C::decode(raw[i], sp, lo, hi, scale, min);
                    }
                }
                *reinterpret_cast<uint4*>(row + 16 * generic::w4a8_chunk(n, 2 * j))     = lo;
                *reinterpret_cast<uint4*>(row + 16 * generic::w4a8_chunk(n, 2 * j + 1)) = hi;
                buf.b_scale[n * 4 + j]                                                  = scale;
                if constexpr (kAux) buf.b_min[n * 4 + j] = min;
                if constexpr (generic::w4a8_half_min<C>()) {
                    *reinterpret_cast<float2*>(buf.b_hmin + n * 8 + 2 * j) =
                        make_float2(min_lo, min_hi);
                }
            }
        }
    }
};

// The activation tile of one K tile: row r of the tile is column col0 + r of the job (a slot),
// whose source row is its token (First: gathered through packed_token) or the slot itself.
template <class C, class S, int Threads, bool First, class B>
__device__ __forceinline__ void
moe_w4a8_load_activation(const GemmArgs& a, const MoeW4a8Activation& act, int col0, int count,
                         int tid, int k_tile, const B& buf) {
    static_assert((S::kBlockM * 8) % Threads == 0);
    const int k       = act.ld;
    const auto source = [&](int r) {
        if (r >= count) return 0;
        return First ? a.packed_token[col0 + r] : col0 + r;
    };
#pragma unroll
    for (int i = 0; i < S::kBlockM * 8 / Threads; ++i) {
        const int task = tid + i * Threads;
        const int r = task >> 3, c = task & 7;
        const bool ok = r < count;
        const auto* src =
            act.codes + static_cast<std::int64_t>(source(r)) * k + k_tile * 128 + c * 16;
        cp_async_zfill<16, Cache::cg>(buf.a_codes + r * 128 + 16 * generic::w4a8_chunk(r, c), src,
                                      ok ? 16 : 0);
    }
    constexpr int kPlanes = C::kMin ? 2 : 1;
    const int kb_row      = k / 32;
    for (int task = tid; task < S::kBlockM * kPlanes; task += Threads) {
        const int r = task / kPlanes, which = task % kPlanes;
        const bool ok    = r < count;
        const float* src = (which ? act.sums : act.scales) +
                           static_cast<std::int64_t>(source(r)) * kb_row + k_tile * 4;
        cp_async_zfill<16, Cache::cg>((which ? buf.a_sum : buf.a_scale) + r * 4, src, ok ? 16 : 0);
    }
}

// Grid: x = row blocks (kBlockN tile rows: kBlockN / 2 local rows of a gated stage), y = (job,
// sub-tile) with kBlockM slots per sub-tile.
//
// Ahead: the weights stream two K tiles ahead in registers (two register sets, the loop unrolled
// by parity so each set is indexed statically): tile k + 2 loads while tile k multiplies and is
// decoded into shared memory at tile k + 1's decode step. A job of few slots (a 1024-token prefill
// call: about 32 per expert) multiplies too little per tile to cover one tile's load, and the
// shared-memory buffers (2 CTAs per SM on sm_120) cannot grow; registers can.
template <class C, class S, Stage St, int Flavor = generic::w4a8_classic_flavor<S>(),
          bool Ahead = false>
__global__ __launch_bounds__(S::kThreads) void moe_w4a8_kernel(GemmArgs a, MoeW4a8Activation act) {
    static_assert(St != Stage::Router);
    constexpr bool kGated = St == Stage::GateUp;
    constexpr bool kFirst = St == Stage::GateUp || St == Stage::Up;
    using Buf             = generic::W4a8BufferOf<C, S>;
    extern __shared__ __align__(16) unsigned char moe_w4a8_shared[];
    const int tid       = static_cast<int>(threadIdx.x);
    const int block_row = static_cast<int>(blockIdx.x);

    const int job = static_cast<int>(blockIdx.y) / a.sub_tiles;
    const int sub = static_cast<int>(blockIdx.y) - job * a.sub_tiles;
    if (job >= *a.job_total) return;
    const int expert = a.job_expert[job];
    const int col0   = a.job_begin[job] + sub * S::kBlockM;
    const int count  = min(S::kBlockM, a.job_begin[job] + a.job_count[job] - col0);
    if (count <= 0) return;
    const std::int64_t row_base = static_cast<std::int64_t>(expert) * a.expert_rows;
    const int k_tiles           = act.ld / 128;

    MoeW4a8Weights<C, S, S::kThreads, kGated> w;
    w.load(a, row_base, block_row, tid, 0);
    w.store(Buf::at(moe_w4a8_shared, 0), tid);
    moe_w4a8_load_activation<C, S, S::kThreads, kFirst>(a, act, col0, count, tid, 0,
                                                        Buf::at(moe_w4a8_shared, 0));
    cp_commit();

    float acc[S::kMmaM][S::kMmaN][4] = {};
    const int lane                   = tid & 31;
    const int warp                   = tid >> 5;
    const int warp_m                 = warp / S::kWarpsN;
    const int warp_n                 = warp - warp_m * S::kWarpsN;

    // The dense classic tile's schedule choices (w4a8_classic_flavor) for the same tile shape.
    constexpr int kDecodeStep = (Flavor & generic::kW4a8DecodeMid) ? 1 : 3;
    if constexpr (!Ahead) {
#pragma unroll 1
        for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
            const auto cur  = Buf::at(moe_w4a8_shared, k_tile & 1);
            const auto next = Buf::at(moe_w4a8_shared, (k_tile + 1) & 1);
            const bool more = k_tile + 1 < k_tiles;
            if (more) {
                w.load(a, row_base, block_row, tid, k_tile + 1);
                moe_w4a8_load_activation<C, S, S::kThreads, kFirst>(a, act, col0, count, tid,
                                                                    k_tile + 1, next);
            }
            cp_commit();
            cp_wait<1>();
            __syncthreads();
            const auto decode_next = [&](int kk) {
                if (kk == kDecodeStep && more) w.store(next, tid);
            };
            if constexpr (generic::w4a8_block_scaled<C>()) {
#if NINFERNO_HAS_NVFP4_MMA
                generic::w4a8_mma_tile_mx<S>(acc, cur, warp_m, warp_n, lane, [](int) {},
                                             decode_next);
#endif
            } else {
                generic::w4a8_mma_tile<C, S, (Flavor & generic::kW4a8Scales4) != 0>(
                    acc, cur, warp_m, warp_n, lane, [](int) {}, decode_next);
            }
            __syncthreads();
        }
    } else {
        // The block-scaled tile is not wired here: its codecs (one block a load) never stream ahead.
        static_assert(!generic::w4a8_block_scaled<C>());
        // w holds tile 1 after the prologue's store of tile 0; w2 takes the odd-ahead tiles.
        MoeW4a8Weights<C, S, S::kThreads, kGated> w2;
        if (k_tiles > 1) w.load(a, row_base, block_row, tid, 1);
        const auto step = [&](auto& held, auto& spare, int k_tile, auto parity) {
            constexpr int P = decltype(parity)::value;
            const auto cur  = Buf::at(moe_w4a8_shared, P);
            const auto next = Buf::at(moe_w4a8_shared, P ^ 1);
            const bool more = k_tile + 1 < k_tiles;
            if (k_tile + 2 < k_tiles) spare.load(a, row_base, block_row, tid, k_tile + 2);
            if (more) {
                moe_w4a8_load_activation<C, S, S::kThreads, kFirst>(a, act, col0, count, tid,
                                                                    k_tile + 1, next);
            }
            cp_commit();
            cp_wait<1>();
            __syncthreads();
            generic::w4a8_mma_tile<C, S, (Flavor & generic::kW4a8Scales4) != 0>(
                acc, cur, warp_m, warp_n, lane, [](int) {},
                [&](int kk) {
                    if (kk == kDecodeStep && more) held.store(next, tid);
                });
            __syncthreads();
        };
#pragma unroll 1
        for (int k_tile = 0; k_tile < k_tiles; k_tile += 2) {
            step(w, w2, k_tile, std::integral_constant<int, 0>{});
            if (k_tile + 1 < k_tiles) step(w2, w, k_tile + 1, std::integral_constant<int, 1>{});
        }
    }

    // C fragment: slot rows g and g + 8 of each m16 tile, weight tile rows 2t and 2t + 1 of each
    // n8 tile. A gated stage's n-tile pairs (n even: gate rows, n + 1: the same local rows' up).
    const int g = lane >> 2;
    const int t = lane & 3;
#pragma unroll
    for (int m = 0; m < S::kMmaM; ++m) {
#pragma unroll
        for (int n = 0; n < S::kMmaN; n += (kGated ? 2 : 1)) {
            const int tile_row = warp_n * S::kWarpN + n * 8 + 2 * t;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int l = generic::w4a4_local_row<kGated, S::kBlockN>(block_row, tile_row + c);
                if (l >= a.rows) continue;
                const std::int64_t row0 = row_base + static_cast<std::int64_t>(l) * a.row_step;
                const float bias0       = moe_bias(a, row0);
                const float bias1       = kGated ? moe_bias(a, row0 + a.up_offset) : 0.0F;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int slot = warp_m * S::kWarpM + m * 16 + g + h * 8;
                    if (slot >= count) continue;
                    const float v0 = acc[m][n][h * 2 + c] + bias0;
                    float value    = v0;
                    if constexpr (kGated) {
                        value = moe_activation(a, v0, acc[m][n + 1][h * 2 + c] + bias1);
                    } else if constexpr (St == Stage::Up) {
                        value = moe_activation(a, v0, v0);
                    }
                    static_cast<__nv_bfloat16*>(
                        a.out_bf16)[static_cast<std::int64_t>(col0 + slot) * a.ldo + l] =
                        __float2bfloat16_rn(value);
                }
            }
        }
    }
}

template <class C, Stage St>
void moe_w4a8_launch_stage(const GemmArgs& args, const MoeW4a8Activation& act, dim3 grid,
                           cudaStream_t stream) {
    using S             = MoeW4a8Schedule;
    constexpr int kSmem = generic::W4a8BufferOf<C, S>::kSmem;
    if constexpr (kSmem > kMaxSharedBytesPerBlock) {
        throw std::invalid_argument("moe W4A8: the tile exceeds this build's shared memory");
    } else {
        // The superblock codecs (K quants: several blocks a load) gain, the 32-value block ones
        // (Q4_0, ggml32, MXFP4) do not (RTX 5090: Q4_K 1024-token call 682 -> 666 us,
        // Qwen3.6 35B prefill +1.0..1.6 %; Q4_0 788 -> 807 us, MXFP4 1176 -> 1196, Nemotron -0.5
        // %).
        constexpr bool kSuperblock = generic::w4a8_span<C>() > 1;
        const auto kernel =
            kSuperblock
                ? moe_w4a8_kernel<C, S, St, generic::w4a8_classic_flavor<S>(), kSuperblock>
                : moe_w4a8_kernel<C, S, St>;
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        kSmem));
        kernel<<<grid, S::kThreads, kSmem, stream>>>(args, act);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <class C>
void moe_w4a8_launch_codec(Stage stage, const GemmArgs& args, const MoeW4a8Activation& act,
                           dim3 grid, cudaStream_t stream) {
    switch (stage) {
    case Stage::GateUp:
        moe_w4a8_launch_stage<C, Stage::GateUp>(args, act, grid, stream);
        return;
    case Stage::Up:
        moe_w4a8_launch_stage<C, Stage::Up>(args, act, grid, stream);
        return;
    case Stage::Down:
        moe_w4a8_launch_stage<C, Stage::Down>(args, act, grid, stream);
        return;
    case Stage::Router:
        break;
    }
    throw std::invalid_argument("moe W4A8: the router stays on the register MMA");
}

} // namespace ninferno::ops::detail::moe
