// Decode (T=1) and small-T routes of the Gemma 4 sparse MoE. Both keep the activation represented
// (BF16 in, FP32 intermediate) and read only the weights of the experts the tokens selected, each
// of them across DRAM once.
//
// Two access patterns, each measured against the others on this card (the bench notes list the
// variants that lost):
//
// - gate/up streams an expert's plane front to back: a lane owns sixteen consecutive code bytes,
//   so a warp instruction reads 512 contiguous bytes and a warp walks whole rows. A row's scales
//   are 512 bytes apart, but the rows that share their sectors are read by neighbouring warps at
//   the same time, so L2 serves those reuses.
// - down, whose rows are 352 bytes, is read by sector set. The NVFP4 scale tile interleaves 128
//   rows so that one 32-byte sector holds the four block scales of a 64-column group for eight
//   rows, {2m, 2m+1} + {0, 32, 64, 96} of the tile; a warp owns those eight rows, four lanes a
//   row and one NVFP4 group a lane, so every sector it touches is used whole.
//
// With few experts selected, gate/up also warms the down planes in L2, which joins their DRAM
// traffic to its own stream instead of paying a second kernel's ramp.

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/gemma4_moe/gemma4_moe_common.cuh"
#include "ops/gemma4_moe/gemma4_moe_internal.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail::gemma4 {
namespace {

constexpr int kSectorSets = 16;            // values of m
constexpr int kDownTiles  = kHidden / 128; // 22

// Row, inside its 128-row tile, of the r-th row (0..7) of sector set m.
__device__ __forceinline__ int sector_row(int m, int r8) {
    return 2 * m + (r8 & 1) + (r8 >> 1) * 32;
}

// One group of sixteen values of one row: eight code bytes and the group's
// scale. `tile` is the row's 128-row tile in the whole stacked plane, `inner`
// its row inside that tile.
template <int ScaleTilesPerRow>
struct GroupLoad {
    uint2 codes;
    std::uint8_t scale;
};

template <int ScaleTilesPerRow>
__device__ __forceinline__ GroupLoad<ScaleTilesPerRow>
load_group(const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
           std::int64_t row, int tile, int inner, int group64, int chunk) {
    constexpr int kCodeBytesPerRow = ScaleTilesPerRow * 32;
    GroupLoad<ScaleTilesPerRow> out;
    out.codes = load_vec<uint2>(codes + row * kCodeBytesPerRow + group64 * 32 + chunk * 8);
    out.scale = scales[(static_cast<std::int64_t>(tile) * ScaleTilesPerRow + group64) * 512 +
                       (inner & 31) * 16 + (inner >> 5) * 4 + chunk];
    return out;
}

__device__ __forceinline__ float group_dot_f32(uint2 codes, const float* x) {
    float w[8];
    const float4 x0 = load_vec<float4>(x);
    const float4 x1 = load_vec<float4>(x + 4);
    const float4 x2 = load_vec<float4>(x + 8);
    const float4 x3 = load_vec<float4>(x + 12);
    const float v0[8]{x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
    const float v1[8]{x2.x, x2.y, x2.z, x2.w, x3.x, x3.y, x3.z, x3.w};
    decode_e2m1x8(codes.x, w);
    float sum = dot8(w, v0);
    decode_e2m1x8(codes.y, w);
    return sum + dot8(w, v1);
}

__device__ __forceinline__ float sum_over_row_lanes(float value) {
    value += __shfl_xor_sync(kFullWarpMask, value, 1);
    value += __shfl_xor_sync(kFullWarpMask, value, 2);
    return value;
}

// ---- router -------------------------------------------------------------------------------------
// One block per (expert row, run of kRouterBlockTokens tokens); each of the 352 threads owns eight
// hidden channels. The row times the router scale is formed once and the block's tokens are
// streamed through it. The token's mean square is accumulated from the same loads, so no block
// needs the norm beforehand. A token's logit is the same whichever block computes it, so a
// small-T call reproduces the decode route's logits.
constexpr int kRouterThreads     = kHidden / 8;
constexpr int kRouterWarps       = kRouterThreads / 32;
constexpr int kRouterTokenChunk  = 16;
constexpr int kRouterBlockTokens = 4;
static_assert(kRouterThreads % 32 == 0);

__global__ __launch_bounds__(kRouterThreads) void router_simt_kernel(
    const __nv_bfloat16* __restrict__ x, int tokens, const __nv_bfloat16* __restrict__ router,
    const float* __restrict__ router_scale, float* __restrict__ dot_out,
    float* __restrict__ square_out) {
    __shared__ float partial[kRouterWarps][kRouterTokenChunk][2];
    if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    const int expert = static_cast<int>(blockIdx.x);
    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int h      = tid * 8;
    const int first  = static_cast<int>(blockIdx.y) * kRouterBlockTokens;
    const int last   = min(tokens, first + kRouterBlockTokens);

    float weight[8];
    bf16x8_to_float(load_vec<uint4>(router + static_cast<std::int64_t>(expert) * kHidden + h),
                    weight);
    const float4 s0 = load_vec<float4>(router_scale + h);
    const float4 s1 = load_vec<float4>(router_scale + h + 4);
    weight[0] *= s0.x;
    weight[1] *= s0.y;
    weight[2] *= s0.z;
    weight[3] *= s0.w;
    weight[4] *= s1.x;
    weight[5] *= s1.y;
    weight[6] *= s1.z;
    weight[7] *= s1.w;

    for (int token0 = first; token0 < last; token0 += kRouterTokenChunk) {
        const int count = min(kRouterTokenChunk, last - token0);
#pragma unroll
        for (int item = 0; item < kRouterTokenChunk; ++item) {
            if (item < count) {
                float value[8];
                bf16x8_to_float(
                    load_vec<uint4>(x + static_cast<std::int64_t>(token0 + item) * kHidden + h),
                    value);
                const float dot    = warp_sum(dot8(weight, value));
                const float square = warp_sum(dot8(value, value));
                if (lane == 0) {
                    partial[warp][item][0] = dot;
                    partial[warp][item][1] = square;
                }
            }
        }
        __syncthreads();
        if (tid < count) {
            float dot    = 0.0F;
            float square = 0.0F;
#pragma unroll
            for (int source = 0; source < kRouterWarps; ++source) {
                dot += partial[source][tid][0];
                square += partial[source][tid][1];
            }
            dot_out[static_cast<std::int64_t>(token0 + tid) * kExperts + expert] = dot;
            if (expert == 0) { square_out[token0 + tid] = square; }
        }
        __syncthreads();
    }
}

// One K slice: RouterPartials{dot, square, 1, tokens}.
void launch_router_simt(const __nv_bfloat16* x, int tokens, const Gemma4MoeWeights& weights,
                        float* dot, float* square, cudaStream_t stream) {
    const dim3 grid(kExperts, (tokens + kRouterBlockTokens - 1) / kRouterBlockTokens);
    router_simt_kernel<<<grid, kRouterThreads, 0, stream>>>(
        x, tokens, static_cast<const __nv_bfloat16*>(weights.router.qdata),
        static_cast<const float*>(weights.router_scale.data), dot, square);
    CUDA_CHECK(cudaGetLastError());
}

// ---- decode -------------------------------------------------------------------------------------
// The decode kernels stream each selected expert's planes front to back instead
// of by sector set: a lane owns sixteen consecutive code bytes (32 values, two
// NVFP4 groups), so a warp instruction reads 512 contiguous bytes and a warp
// walks a contiguous run of whole rows. The scale bytes of a row are 512 bytes
// apart, but the neighbouring rows that share their sectors are read by the
// neighbouring warps at the same time, so L2 serves those reuses.

// Sixteen code bytes of one row: 32 values against 32 inputs, two block scales.
__device__ __forceinline__ float chunk_dot_bf16(uint4 codes, std::uint16_t scale_pair,
                                                const __nv_bfloat16* x) {
    float w[8], v[8];
    float first = 0.0F, second = 0.0F;
    decode_e2m1x8(codes.x, w);
    bf16x8_to_float(load_vec<uint4>(x), v);
    first += dot8(w, v);
    decode_e2m1x8(codes.y, w);
    bf16x8_to_float(load_vec<uint4>(x + 8), v);
    first += dot8(w, v);
    decode_e2m1x8(codes.z, w);
    bf16x8_to_float(load_vec<uint4>(x + 16), v);
    second += dot8(w, v);
    decode_e2m1x8(codes.w, w);
    bf16x8_to_float(load_vec<uint4>(x + 24), v);
    second += dot8(w, v);
    return decode_nvfp4_e4m3(static_cast<std::uint8_t>(scale_pair)) * first +
           decode_nvfp4_e4m3(static_cast<std::uint8_t>(scale_pair >> 8)) * second;
}

// The two block scales of a 32-value chunk: adjacent bytes of the row's k64
// quad.
template <int ScaleTilesPerRow>
__device__ __forceinline__ std::uint16_t load_scale_pair(const std::uint8_t* __restrict__ scales,
                                                         std::int64_t row, int chunk32) {
    const std::int64_t at = ((row >> 7) * ScaleTilesPerRow + (chunk32 >> 1)) * 512 +
                            (row & 31) * 16 + ((row & 127) >> 5) * 4 + (chunk32 & 1) * 2;
    return *reinterpret_cast<const std::uint16_t*>(scales + at);
}

// D2: block (16 intermediate rows j0.., rank k). Warps 0-3 stream the 16 gate
// rows, warps 4-7 the 16 up rows, four whole rows (5632 contiguous bytes) a
// warp; the block then holds 16 complete GeGLU pairs. Every block selects the
// top-8 itself from the 128 logits, so selection costs no launch; block (0,0)
// publishes it for D3.
constexpr int kGateRowsPerWarp   = 4;
constexpr int kGateChunksPerRow  = kHidden / 32;                              // 88
constexpr int kGateSteps         = kGateRowsPerWarp * kGateChunksPerRow / 32; // 11
constexpr int kGatePairsPerBlock = 4 * kGateRowsPerWarp;                      // 16
static_assert(kGateRowsPerWarp * kGateChunksPerRow % 32 == 0);

// The top-8 of one token's 128 logits, for a whole block at once. Each logit is ranked by counting
// the logits that beat it - by value, then by the lower id, the order the warp merge uses - with
// the 256 threads splitting the 128 comparisons in two. It trades the merge network's five
// dependent shuffle steps on one warp for one pass of independent shared-memory reads.
__device__ __forceinline__ void block_select_top8(const RouterPartials& router, float eps,
                                                  float* logit_shared, int* count_shared,
                                                  int* selected, float* selected_logit) {
    const int tid = static_cast<int>(threadIdx.x);
    if (tid < 32) {
        float logit[4];
        load_token_logits<1>(router, 0, eps, logit);
#pragma unroll
        for (int item = 0; item < 4; ++item) { logit_shared[tid + 32 * item] = logit[item]; }
    }
    __syncthreads();
    const int expert = tid & (kExperts - 1);
    const int half   = tid >> 7;
    // NaN ranks as -inf (rank_score): every expert gets a distinct rank, so all eight slots fill.
    const float mine = rank_score(logit_shared[expert]);
    int beaten_by    = 0;
#pragma unroll 8
    for (int other = half * 64; other < half * 64 + 64; ++other) {
        const float theirs = rank_score(logit_shared[other]);
        beaten_by += (theirs > mine || (theirs == mine && other < expert)) ? 1 : 0;
    }
    count_shared[tid] = beaten_by;
    __syncthreads();
    if (tid < kExperts) {
        const int rank = count_shared[tid] + count_shared[tid + kExperts];
        if (rank < kTopK) {
            selected[rank]       = tid;
            selected_logit[rank] = mine;
        }
    }
    __syncthreads();
}

// Three blocks an SM are 510 slots for the 352 blocks: the whole grid in one wave, which a variant
// at 84 registers (two blocks an SM after allocation granularity) would lose.
// NINFERNO-SPECIAL(chip:gb202:170sm): one wave holds the grid on 170 SMs (510 slots); on 84 SMs
// (252 slots) it takes two. Correct anywhere; the launch bound was chosen on the 5090.
template <bool RankSelect, bool LoadsFirst>
__global__ __launch_bounds__(256, 3) void decode_gate_up_kernel(
    const __nv_bfloat16* __restrict__ x, RouterPartials router, float eps,
    const float* __restrict__ expert_scale, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, const float* __restrict__ divisors,
    const std::uint8_t* __restrict__ down_codes, const std::uint8_t* __restrict__ down_scales,
    int* __restrict__ ids_out, float* __restrict__ weights_out, float* __restrict__ activation) {
    __shared__ __align__(16) __nv_bfloat16 x_shared[kHidden];
    __shared__ float pre[2 * kGatePairsPerBlock];
    __shared__ int selected[kTopK];
    __shared__ float selected_logit[kTopK];
    __shared__ float logit_shared[RankSelect ? kExperts : 1];
    __shared__ int count_shared[RankSelect ? 2 * kExperts : 1];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid == 0) { pdl::trigger_dependents(); }
    for (int vector = tid; vector < kHidden / 8; vector += 256) {
        store_vec(x_shared + vector * 8, load_vec<uint4>(x + vector * 8));
    }
    pdl::wait_for_dependencies();
    const int rank     = static_cast<int>(blockIdx.y);
    const bool publish = blockIdx.x == 0 && blockIdx.y == 0;
    if constexpr (RankSelect) {
        block_select_top8(router, eps, logit_shared, count_shared, selected, selected_logit);
        if (publish && tid < kTopK) {
            float denominator = 0.0F;
#pragma unroll
            for (int k = 0; k < kTopK; ++k) {
                denominator += expf(selected_logit[k] - selected_logit[0]);
            }
            ids_out[tid]     = selected[tid];
            weights_out[tid] = expf(selected_logit[tid] - selected_logit[0]) / denominator *
                               expert_scale[selected[tid]];
        }
    } else {
        if (warp == 0) {
            float logit[4];
            load_token_logits<1>(router, 0, eps, logit);
            const Top8 top = select_top8(logit);
            if (lane == 0) {
#pragma unroll
                for (int k = 0; k < kTopK; ++k) {
                    selected[k] = top.id[k];
                    if (publish) {
                        ids_out[k]     = top.id[k];
                        weights_out[k] = top8_weight(top, k, expert_scale);
                    }
                }
            }
        }
        __syncthreads();
    }

    const int expert = selected[rank];
    const int j0     = static_cast<int>(blockIdx.x) * kGatePairsPerBlock;
    const int half   = warp >> 2;
    // D3 streams this expert's down plane next. The 44 blocks of a rank each warm one 44th of it
    // in L2, so that its DRAM traffic joins this kernel's stream instead of paying a second ramp
    // of its own. A pure cache hint: no value depends on it. LoadsFirst issues it after this
    // block's own weight loads, so that the prefetch queues behind the data this kernel waits on.
    const auto prefetch_down = [&] {
        constexpr int kCodeLines  = kHidden * kInter / 2 / 128 / (kInter / kGatePairsPerBlock);
        constexpr int kScaleLines = kHidden * kInter / 16 / 128 / (kInter / kGatePairsPerBlock);
        static_assert(kCodeLines + kScaleLines <= 256);
        if (tid < kCodeLines) {
            const std::uint8_t* line =
                down_codes + static_cast<std::int64_t>(expert) * (kHidden * kInter / 2) +
                (static_cast<std::int64_t>(blockIdx.x) * kCodeLines + tid) * 128;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(line));
        } else if (tid < kCodeLines + kScaleLines) {
            const std::uint8_t* line =
                down_scales + static_cast<std::int64_t>(expert) * (kHidden * kInter / 16) +
                (static_cast<std::int64_t>(blockIdx.x) * kScaleLines + tid - kCodeLines) * 128;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(line));
        }
    };
    if constexpr (!LoadsFirst) { prefetch_down(); }
    const std::int64_t first_row = static_cast<std::int64_t>(expert) * kGateUpRows + half * kInter +
                                   j0 + (warp & 3) * kGateRowsPerWarp;
    const std::uint8_t* base = codes + first_row * (kHidden / 2);
    uint4 code[kGateSteps];
    std::uint16_t scale[kGateSteps];
#pragma unroll
    for (int step = 0; step < kGateSteps; ++step) {
        const int chunk = step * 32 + lane;
        code[step]      = load_vec<uint4>(base + chunk * 16);
        scale[step]     = load_scale_pair<kHiddenScaleTiles>(
            scales, first_row + chunk / kGateChunksPerRow, chunk % kGateChunksPerRow);
    }
    if constexpr (LoadsFirst) { prefetch_down(); }
    float acc[kGateRowsPerWarp] = {};
#pragma unroll
    for (int step = 0; step < kGateSteps; ++step) {
        const int chunk = step * 32 + lane;
        const int row   = chunk / kGateChunksPerRow;
        const float value =
            chunk_dot_bf16(code[step], scale[step], x_shared + (chunk % kGateChunksPerRow) * 32);
#pragma unroll
        for (int r = 0; r < kGateRowsPerWarp; ++r) { acc[r] += row == r ? value : 0.0F; }
    }
#pragma unroll
    for (int r = 0; r < kGateRowsPerWarp; ++r) {
        const float value = warp_sum(acc[r]);
        if (lane == 0) {
            pre[half * kGatePairsPerBlock + (warp & 3) * kGateRowsPerWarp + r] = value;
        }
    }
    __syncthreads();
    if (tid < kGatePairsPerBlock) {
        const float inverse = 1.0F / divisors[expert];
        activation[rank * kInter + j0 + tid] =
            gelu_tanh(pre[tid] * inverse) * (pre[kGatePairsPerBlock + tid] * inverse);
    }
}

// D3: block (tile t, sector set m), one warp per rank: the eight rows of a
// scale sector, four lanes a row, one NVFP4 group a lane. D2 has warmed the
// planes in L2. The block sums its eight hidden rows over the eight experts in
// rank order.
//
// Staged: after the wait the block has four L2 round trips in a chain - ids, then the weight
// groups they address, then the activations the dot products read, then the divisor and route
// weights the epilogue reads. Staged issues everything that does not depend on the ids at once
// (the eight activation rows into shared memory, the route weights) and the divisor with the
// weight groups, which leaves two.
template <bool Staged>
__global__ __launch_bounds__(kTopK * 32, 3) void decode_down_kernel(
    const int* ids, const float* route_weights, const float* activation,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const float* __restrict__ divisors, __nv_bfloat16* __restrict__ out) {
    __shared__ float partial[kTopK][8];
    __shared__ __align__(16) float act_shared[Staged ? kTopK : 1][Staged ? kInter : 4];
    __shared__ float weight_shared[kTopK];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    pdl::wait_for_dependencies();
    const int tile   = static_cast<int>(blockIdx.x) / kSectorSets;
    const int m      = static_cast<int>(blockIdx.x) % kSectorSets;
    const int expert = ids[warp];
    if constexpr (Staged) {
        if (tid < kTopK) { weight_shared[tid] = route_weights[tid]; }
        for (int vector = tid; vector < kTopK * kInter / 4; vector += kTopK * 32) {
            store_vec(&act_shared[0][0] + vector * 4, load_vec<float4>(activation + vector * 4));
        }
    }
    const int r8            = lane >> 2;
    const int chunk         = lane & 3;
    const int inner         = sector_row(m, r8);
    const std::int64_t grow = static_cast<std::int64_t>(expert) * kHidden + tile * 128 + inner;
    const int plane_tile    = expert * kDownTiles + tile;
    GroupLoad<kInterScaleTiles> loaded[kInterScaleTiles];
#pragma unroll
    for (int g = 0; g < kInterScaleTiles; ++g) {
        loaded[g] = load_group<kInterScaleTiles>(codes, scales, grow, plane_tile, inner, g, chunk);
    }
    const float divisor = divisors[expert];
    if constexpr (Staged) { __syncthreads(); }
    const float* act = Staged ? &act_shared[warp][0] : activation + warp * kInter;
    float acc        = 0.0F;
#pragma unroll
    for (int g = 0; g < kInterScaleTiles; ++g) {
        acc = fmaf(decode_nvfp4_e4m3(loaded[g].scale),
                   group_dot_f32(loaded[g].codes, act + g * 64 + chunk * 16), acc);
    }
    acc = sum_over_row_lanes(acc);
    if (chunk == 0) { partial[warp][r8] = acc / divisor; }
    __syncthreads();
    if (tid < 8) {
        float value = 0.0F;
#pragma unroll
        for (int k = 0; k < kTopK; ++k) {
            value = fmaf(Staged ? weight_shared[k] : route_weights[k], partial[k][tid], value);
        }
        out[tile * 128 + sector_row(m, tid)] = __float2bfloat16_rn(value);
    }
}

// ---- small-T ------------------------------------------------------------------------------------
// The small-T route computes every (token, expert) product exactly as the decode route does - the
// same router kernel, D2's chunk sums, D3's group sums, the rank-order sum - so each column of a
// small-T call is the decode route's result for that token, bit for bit (the suite checks it). A
// verify pass over T drafted tokens therefore returns what T single-token steps would.
//
// S2: one block selects every token, one warp a token, and groups the
// assignments by expert. The order of the assignments inside an expert's run
// carries no numerics - every (token, expert) product is computed on its own
// and read back through packed_index in rank order - so a shared-memory atomic
// may hand out the ranks.
constexpr int kGroupThreads = 1024;

__global__ __launch_bounds__(kGroupThreads) void small_t_group_kernel(
    RouterPartials router, float eps, const float* __restrict__ expert_scale, int tokens,
    int* __restrict__ ids, float* __restrict__ route_weights, int* __restrict__ meta,
    int* __restrict__ slot_assignment, int* __restrict__ packed_index) {
    __shared__ int count[kExperts];
    __shared__ int offsets[kExperts + 1];
    __shared__ int selected[kTopK * kSmallTCapacity];
    __shared__ int rank_in_expert[kTopK * kSmallTCapacity];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid == 0) { pdl::trigger_dependents(); }
    if (tid < kExperts) { count[tid] = 0; }
    pdl::wait_for_dependencies();
    __syncthreads();
    for (int token = warp; token < tokens; token += kGroupThreads / 32) {
        float logit[4];
        load_token_logits<1>(router, token, eps, logit);
        const Top8 top = select_top8(logit);
        if (lane == 0) {
#pragma unroll
            for (int k = 0; k < kTopK; ++k) {
                const int assignment       = token * kTopK + k;
                ids[assignment]            = top.id[k];
                route_weights[assignment]  = top8_weight(top, k, expert_scale);
                selected[assignment]       = top.id[k];
                rank_in_expert[assignment] = atomicAdd(&count[top.id[k]], 1);
            }
        }
    }
    __syncthreads();

    if (warp == 0) {
        int c[4];
        int local   = 0;
        int nonzero = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            c[i] = count[lane * 4 + i];
            local += c[i];
            nonzero += c[i] > 0 ? 1 : 0;
        }
        int inclusive         = local;
        int inclusive_nonzero = nonzero;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int carried         = __shfl_up_sync(kFullWarpMask, inclusive, offset);
            const int carried_nonzero = __shfl_up_sync(kFullWarpMask, inclusive_nonzero, offset);
            if (lane >= offset) {
                inclusive += carried;
                inclusive_nonzero += carried_nonzero;
            }
        }
        int running = inclusive - local;
        int unique  = inclusive_nonzero - nonzero;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int expert            = lane * 4 + i;
            offsets[expert]             = running;
            meta[kMetaOffsets + expert] = running;
            if (c[i] > 0) { meta[kMetaUnique + unique++] = expert; }
            running += c[i];
        }
        if (lane == 31) {
            offsets[kExperts]             = running;
            meta[kMetaOffsets + kExperts] = running;
            meta[kMetaCount]              = unique;
        }
    }
    __syncthreads();
    for (int assignment = tid; assignment < tokens * kTopK; assignment += kGroupThreads) {
        const int packed         = offsets[selected[assignment]] + rank_in_expert[assignment];
        slot_assignment[packed]  = assignment;
        packed_index[assignment] = packed;
    }
}

// S3 and S4 issue every weight word a block will use - the eleven sixteen-byte gate/up chunks of a
// lane, the eleven NVFP4 down groups - as soon as the block knows its expert, before it stages the
// tokens, so the staging's L2 round trips run under the DRAM latency. The words stay in registers
// while the tokens routed to the expert pass through them one at a time, so an expert's weights
// cross DRAM once whatever its token count.
//
// The code words pass through opaque() inside the token loop: their decode is loop-invariant, and
// without the barrier the compiler hoists it out of the loop - 352 decoded floats a lane, which
// it then keeps on the stack (2208 bytes a thread, the kernel twelve times slower).
__device__ __forceinline__ uint4 opaque(uint4 word) {
    asm volatile("" : "+r"(word.x), "+r"(word.y), "+r"(word.z), "+r"(word.w));
    return word;
}

__device__ __forceinline__ uint2 opaque(uint2 word) {
    asm volatile("" : "+r"(word.x), "+r"(word.y));
    return word;
}

// chunk_dot_bf16 over inputs already widened to FP32: the same values in the same order, so the
// same result, without the eight shifts and permutes a lane spends on every eight BF16 inputs.
__device__ __forceinline__ float chunk_dot_f32(uint4 codes, std::uint16_t scale_pair,
                                               const float* x) {
    float w[8];
    float first = 0.0F, second = 0.0F;
    const auto eight = [&](const float* at) {
        const float4 a = load_vec<float4>(at);
        const float4 b = load_vec<float4>(at + 4);
        const float v[8]{a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        return dot8(w, v);
    };
    decode_e2m1x8(codes.x, w);
    first += eight(x);
    decode_e2m1x8(codes.y, w);
    first += eight(x + 8);
    decode_e2m1x8(codes.z, w);
    second += eight(x + 16);
    decode_e2m1x8(codes.w, w);
    second += eight(x + 24);
    return decode_nvfp4_e4m3(static_cast<std::uint8_t>(scale_pair)) * first +
           decode_nvfp4_e4m3(static_cast<std::uint8_t>(scale_pair >> 8)) * second;
}

// S3: block (16 intermediate rows j0.., selected expert), laid out as D2: warps 0-3 stream 16 gate
// rows, warps 4-7 the 16 up rows, four whole rows a warp. The tokens are widened to FP32 in shared
// memory two at a time. A lane reads its 32 inputs as eight float4; unpadded, the lanes of a
// quarter warp would be 128 bytes apart and every read an eight-way bank conflict, so each 32-value
// chunk is stored 36 floats apart (25.3 KB for two tokens). When few experts are selected the
// block warms its share of the expert's down plane in L2 for S4, as D2 does; with many, the
// gate/up stream that follows would evict those lines before S4 reads them, and the prefetch would
// only double the down traffic, so the launch turns it off.
constexpr int kGateTokens   = 2;
constexpr int kChunkStride  = 36;
constexpr int kPaddedHidden = kGateChunksPerRow * kChunkStride;
// Selected experts up to which the down planes are warmed: 16 of them are 18 MB of L2.
constexpr int kPrefetchDownExperts = 16;

__global__ __launch_bounds__(256, 3) void small_t_gate_up_kernel(
    const __nv_bfloat16* __restrict__ x, const int* meta, const int* slot_assignment,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const float* __restrict__ divisors, const std::uint8_t* __restrict__ down_codes,
    const std::uint8_t* __restrict__ down_scales, int prefetch_down,
    float* __restrict__ activation) {
    __shared__ __align__(16) float x_shared[kGateTokens][kPaddedHidden];
    __shared__ float pre[kGateTokens][2 * kGatePairsPerBlock];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid == 0) { pdl::trigger_dependents(); }
    pdl::wait_for_dependencies();
    const int unique = static_cast<int>(blockIdx.y);
    if (unique >= meta[kMetaCount]) { return; }
    const int expert             = meta[kMetaUnique + unique];
    const int j0                 = static_cast<int>(blockIdx.x) * kGatePairsPerBlock;
    const int half               = warp >> 2;
    const std::int64_t first_row = static_cast<std::int64_t>(expert) * kGateUpRows + half * kInter +
                                   j0 + (warp & 3) * kGateRowsPerWarp;
    const std::uint8_t* base = codes + first_row * (kHidden / 2);
    uint4 code[kGateSteps];
    std::uint16_t scale[kGateSteps];
#pragma unroll
    for (int step = 0; step < kGateSteps; ++step) {
        const int chunk = step * 32 + lane;
        code[step]      = load_vec<uint4>(base + chunk * 16);
        scale[step]     = load_scale_pair<kHiddenScaleTiles>(
            scales, first_row + chunk / kGateChunksPerRow, chunk % kGateChunksPerRow);
    }
    if (prefetch_down != 0) {
        constexpr int kCodeLines  = kHidden * kInter / 2 / 128 / (kInter / kGatePairsPerBlock);
        constexpr int kScaleLines = kHidden * kInter / 16 / 128 / (kInter / kGatePairsPerBlock);
        if (tid < kCodeLines) {
            const std::uint8_t* line =
                down_codes + static_cast<std::int64_t>(expert) * (kHidden * kInter / 2) +
                (static_cast<std::int64_t>(blockIdx.x) * kCodeLines + tid) * 128;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(line));
        } else if (tid < kCodeLines + kScaleLines) {
            const std::uint8_t* line =
                down_scales + static_cast<std::int64_t>(expert) * (kHidden * kInter / 16) +
                (static_cast<std::int64_t>(blockIdx.x) * kScaleLines + tid - kCodeLines) * 128;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(line));
        }
    }
    const int begin = meta[kMetaOffsets + expert];
    const int count = meta[kMetaOffsets + expert + 1] - begin;
    for (int token0 = 0; token0 < count; token0 += kGateTokens) {
        const int items = min(kGateTokens, count - token0);
        __syncthreads();
        for (int vector = tid; vector < items * (kHidden / 8); vector += 256) {
            const int item  = vector / (kHidden / 8);
            const int at    = vector - item * (kHidden / 8);
            const int token = slot_assignment[begin + token0 + item] / kTopK;
            float v[8];
            bf16x8_to_float(
                load_vec<uint4>(x + static_cast<std::int64_t>(token) * kHidden + at * 8), v);
            float* to = &x_shared[item][at / 4 * kChunkStride + at % 4 * 8];
            store_vec(to, make_float4(v[0], v[1], v[2], v[3]));
            store_vec(to + 4, make_float4(v[4], v[5], v[6], v[7]));
        }
        __syncthreads();
#pragma unroll 1
        for (int item = 0; item < items; ++item) {
            float acc[kGateRowsPerWarp] = {};
#pragma unroll
            for (int step = 0; step < kGateSteps; ++step) {
                const int chunk = step * 32 + lane;
                const int row   = chunk / kGateChunksPerRow;
                const float value =
                    chunk_dot_f32(opaque(code[step]), scale[step],
                                  &x_shared[item][(chunk % kGateChunksPerRow) * kChunkStride]);
#pragma unroll
                for (int r = 0; r < kGateRowsPerWarp; ++r) { acc[r] += row == r ? value : 0.0F; }
            }
#pragma unroll
            for (int r = 0; r < kGateRowsPerWarp; ++r) {
                const float value = warp_sum(acc[r]);
                if (lane == 0) {
                    pre[item][half * kGatePairsPerBlock + (warp & 3) * kGateRowsPerWarp + r] =
                        value;
                }
            }
        }
        __syncthreads();
        if (tid < items * kGatePairsPerBlock) {
            const float inverse = 1.0F / divisors[expert];
            const int item      = tid / kGatePairsPerBlock;
            const int jj        = tid % kGatePairsPerBlock;
            activation[static_cast<std::int64_t>(begin + token0 + item) * kInter + j0 + jj] =
                gelu_tanh(pre[item][jj] * inverse) * (pre[item][kGatePairsPerBlock + jj] * inverse);
        }
    }
}

// S4: block (eight sector sets, selected expert), one sector set a warp, as D3: the expert's
// weight groups stay in registers while the activations of its tokens pass through shared memory,
// eight at a time. The result, divided by the expert's divisor as D3 divides it, lands in the
// token's packed slot.
constexpr int kDownSetsPerBlock = 8;
constexpr int kDownTokens       = 8;

__global__ __launch_bounds__(kDownSetsPerBlock * 32,
                             3) void small_t_down_kernel(const float* activation, const int* meta,
                                                         const std::uint8_t* __restrict__ codes,
                                                         const std::uint8_t* __restrict__ scales,
                                                         const float* __restrict__ divisors,
                                                         float* __restrict__ expert_out) {
    __shared__ __align__(16) float act[kDownTokens][kInter];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid == 0) { pdl::trigger_dependents(); }
    pdl::wait_for_dependencies();
    const int unique = static_cast<int>(blockIdx.y);
    if (unique >= meta[kMetaCount]) { return; }
    const int expert        = meta[kMetaUnique + unique];
    const int set           = static_cast<int>(blockIdx.x) * kDownSetsPerBlock + warp;
    const int tile          = set / kSectorSets;
    const int m             = set % kSectorSets;
    const int r8            = lane >> 2;
    const int chunk         = lane & 3;
    const int inner         = sector_row(m, r8);
    const int row           = tile * 128 + inner;
    const std::int64_t grow = static_cast<std::int64_t>(expert) * kHidden + row;
    const int plane_tile    = expert * kDownTiles + tile;
    GroupLoad<kInterScaleTiles> loaded[kInterScaleTiles];
#pragma unroll
    for (int g = 0; g < kInterScaleTiles; ++g) {
        loaded[g] = load_group<kInterScaleTiles>(codes, scales, grow, plane_tile, inner, g, chunk);
    }
    const float divisor = divisors[expert];
    const int begin     = meta[kMetaOffsets + expert];
    const int count     = meta[kMetaOffsets + expert + 1] - begin;
    for (int token0 = 0; token0 < count; token0 += kDownTokens) {
        const int items = min(kDownTokens, count - token0);
        __syncthreads();
        for (int vector = tid; vector < items * (kInter / 4); vector += kDownSetsPerBlock * 32) {
            const int item = vector / (kInter / 4);
            const int at   = vector - item * (kInter / 4);
            store_vec(&act[item][at * 4],
                      load_vec<float4>(activation +
                                       static_cast<std::int64_t>(begin + token0 + item) * kInter +
                                       at * 4));
        }
        __syncthreads();
#pragma unroll 1
        for (int item = 0; item < items; ++item) {
            float acc = 0.0F;
#pragma unroll
            for (int g = 0; g < kInterScaleTiles; ++g) {
                acc = fmaf(decode_nvfp4_e4m3(loaded[g].scale),
                           group_dot_f32(opaque(loaded[g].codes), &act[item][g * 64 + chunk * 16]),
                           acc);
            }
            acc = sum_over_row_lanes(acc);
            if (chunk == 0) {
                expert_out[static_cast<std::int64_t>(begin + token0 + item) * kHidden + row] =
                    acc / divisor;
            }
        }
    }
}

// S5: every token sums its eight expert results in rank order.
__global__ __launch_bounds__(kHidden /
                             8) void small_t_reduce_kernel(const float* expert_out,
                                                           const int* packed_index,
                                                           const float* route_weights,
                                                           __nv_bfloat16* __restrict__ out) {
    pdl::wait_for_dependencies();
    const int token = static_cast<int>(blockIdx.x);
    const int h     = static_cast<int>(threadIdx.x) * 8;
    float value[8]  = {};
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        const int assignment = token * kTopK + k;
        const float weight   = route_weights[assignment];
        const float* source =
            expert_out + static_cast<std::int64_t>(packed_index[assignment]) * kHidden + h;
        const float4 y0 = load_vec<float4>(source);
        const float4 y1 = load_vec<float4>(source + 4);
        value[0]        = fmaf(weight, y0.x, value[0]);
        value[1]        = fmaf(weight, y0.y, value[1]);
        value[2]        = fmaf(weight, y0.z, value[2]);
        value[3]        = fmaf(weight, y0.w, value[3]);
        value[4]        = fmaf(weight, y1.x, value[4]);
        value[5]        = fmaf(weight, y1.y, value[5]);
        value[6]        = fmaf(weight, y1.z, value[6]);
        value[7]        = fmaf(weight, y1.w, value[7]);
    }
    const uint4 packed{pack_bf16x2(value[0], value[1]), pack_bf16x2(value[2], value[3]),
                       pack_bf16x2(value[4], value[5]), pack_bf16x2(value[6], value[7])};
    store_vec(out + static_cast<std::int64_t>(token) * kHidden + h, packed);
}

const std::uint8_t* codes_of(const Weight& weight) {
    return static_cast<const std::uint8_t*>(weight.qdata);
}

const std::uint8_t* scales_of(const Weight& weight) {
    return static_cast<const std::uint8_t*>(weight.scales);
}

const float* divisors_of(const Weight& weight) {
    return static_cast<const float*>(weight.weight_divisors);
}

} // namespace

void launch_decode(const Tensor& router_input, const Tensor& expert_input,
                   const Gemma4MoeWeights& weights, float eps, Tensor& out,
                   const DecodeWorkspace& workspace, DecodeOptions options, cudaStream_t stream) {
    static_assert(router_kind(Route::Decode) == RouterKind::Simt);
    auto* router_dot    = static_cast<float*>(workspace.router_dot.data);
    auto* router_square = static_cast<float*>(workspace.router_square.data);
    launch_router_simt(static_cast<const __nv_bfloat16*>(router_input.data), 1, weights, router_dot,
                       router_square, stream);
    auto* ids           = static_cast<int*>(workspace.selection.ids.data);
    auto* route_weights = static_cast<float*>(workspace.selection.weights.data);
    auto* activation    = static_cast<float*>(workspace.activation.data);
    const auto gate_up  = [&](auto kernel) {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(kInter / kGatePairsPerBlock, kTopK), dim3(256), 0, stream}, kernel,
            static_cast<const __nv_bfloat16*>(expert_input.data),
            RouterPartials{router_dot, router_square, 1, 1}, eps,
            static_cast<const float*>(weights.expert_scale.data), codes_of(weights.gate_up),
            scales_of(weights.gate_up), divisors_of(weights.gate_up), codes_of(weights.down),
            scales_of(weights.down), ids, route_weights, activation));
    };
    if (options.rank_select) {
        options.loads_first ? gate_up(decode_gate_up_kernel<true, true>)
                            : gate_up(decode_gate_up_kernel<true, false>);
    } else {
        options.loads_first ? gate_up(decode_gate_up_kernel<false, true>)
                            : gate_up(decode_gate_up_kernel<false, false>);
    }
    const auto down = [&](auto kernel) {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(kDownTiles * kSectorSets), dim3(kTopK * 32), 0, stream}, kernel,
            static_cast<const int*>(ids), static_cast<const float*>(route_weights),
            static_cast<const float*>(activation), codes_of(weights.down), scales_of(weights.down),
            divisors_of(weights.down), static_cast<__nv_bfloat16*>(out.data)));
    };
    options.staged_down ? down(decode_down_kernel<true>) : down(decode_down_kernel<false>);
}

void launch_host_select(const Tensor& router_input, const Gemma4MoeWeights& weights, float eps,
                        const HostSelectWorkspace& workspace, cudaStream_t stream) {
    const int tokens = router_input.ne[1];
    if (tokens < 1 || tokens > kSmallTCapacity) {
        throw std::invalid_argument("gemma4_moe host select: unsupported token count");
    }
    static_assert(router_kind(Route::SmallT) == RouterKind::Simt);
    auto* router_dot    = static_cast<float*>(workspace.router_dot.data);
    auto* router_square = static_cast<float*>(workspace.router_square.data);
    launch_router_simt(static_cast<const __nv_bfloat16*>(router_input.data), tokens, weights,
                       router_dot, router_square, stream);
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(1), dim3(kGroupThreads), 0, stream}, small_t_group_kernel,
        RouterPartials{router_dot, router_square, 1, tokens}, eps,
        static_cast<const float*>(weights.expert_scale.data), tokens,
        static_cast<int*>(workspace.selection.ids.data),
        static_cast<float*>(workspace.selection.weights.data), static_cast<int*>(workspace.meta.data),
        static_cast<int*>(workspace.slot_assignment.data),
        static_cast<int*>(workspace.packed_index.data)));
}

void launch_small_t(const Tensor& router_input, const Tensor& expert_input,
                    const Gemma4MoeWeights& weights, float eps, Tensor& out,
                    const SmallTWorkspace& workspace, cudaStream_t stream) {
    const int tokens = router_input.ne[1];
    if (tokens < 1 || tokens > kSmallTCapacity) {
        throw std::invalid_argument("gemma4_moe small-T: unsupported token count");
    }
    static_assert(router_kind(Route::SmallT) == RouterKind::Simt);
    auto* router_dot    = static_cast<float*>(workspace.router_dot.data);
    auto* router_square = static_cast<float*>(workspace.router_square.data);
    launch_router_simt(static_cast<const __nv_bfloat16*>(router_input.data), tokens, weights,
                       router_dot, router_square, stream);
    auto* ids             = static_cast<int*>(workspace.selection.ids.data);
    auto* route_weights   = static_cast<float*>(workspace.selection.weights.data);
    auto* meta            = static_cast<int*>(workspace.meta.data);
    auto* slot_assignment = static_cast<int*>(workspace.slot_assignment.data);
    auto* packed_index    = static_cast<int*>(workspace.packed_index.data);
    auto* activation      = static_cast<float*>(workspace.activation.data);
    auto* expert_out      = static_cast<float*>(workspace.expert_out.data);
    CUDA_CHECK(pdl::launch_dependent({dim3(1), dim3(kGroupThreads), 0, stream},
                                     small_t_group_kernel,
                                     RouterPartials{router_dot, router_square, 1, tokens}, eps,
                                     static_cast<const float*>(weights.expert_scale.data), tokens,
                                     ids, route_weights, meta, slot_assignment, packed_index));
    const int experts_bound = std::min(kExperts, kTopK * tokens);
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(kInter / kGatePairsPerBlock, static_cast<unsigned>(experts_bound)), dim3(256), 0,
         stream},
        small_t_gate_up_kernel, static_cast<const __nv_bfloat16*>(expert_input.data),
        static_cast<const int*>(meta), static_cast<const int*>(slot_assignment),
        codes_of(weights.gate_up), scales_of(weights.gate_up), divisors_of(weights.gate_up),
        codes_of(weights.down), scales_of(weights.down),
        experts_bound <= kPrefetchDownExperts ? 1 : 0, activation));
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(kDownTiles * kSectorSets / kDownSetsPerBlock, static_cast<unsigned>(experts_bound)),
         dim3(kDownSetsPerBlock * 32), 0, stream},
        small_t_down_kernel, static_cast<const float*>(activation), static_cast<const int*>(meta),
        codes_of(weights.down), scales_of(weights.down), divisors_of(weights.down), expert_out));
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(tokens), dim3(kHidden / 8), 0, stream}, small_t_reduce_kernel,
        static_cast<const float*>(expert_out), static_cast<const int*>(packed_index),
        static_cast<const float*>(route_weights), static_cast<__nv_bfloat16*>(out.data)));
}

} // namespace ninferno::ops::detail::gemma4
