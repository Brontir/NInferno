// Prefill route of the Gemma 4 sparse MoE: a tensor-core router, grouping of
// the assignments by expert, grouped gate/up and down GEMMs over the packed
// columns, and a rank-ordered reduction. The grouped GEMMs come in two tiers:
// A4 (activations encoded to NVFP4, block-scaled e2m1 MMA) and A16 (BF16
// activations against weights decoded to BF16, which is exact for NVFP4).

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/gemma4_moe/gemma4_moe_common.cuh"
#include "ops/gemma4_moe/gemma4_moe_internal.h"
// The A4 tier (kind::mxf4nvf4) is Blackwell-only; the Ada and Ampere builds keep this TU for its
// A16 tier and must not even parse the W4A4 kernel header.
#ifndef NINFERNO_WITHOUT_NVFP4
#    include "ops/linear/nvfp4/nvfp4_a4_mma.cuh"
#    include "ops/linear/nvfp4/nvfp4_a4_quantize.cuh"
#    include "ops/linear/nvfp4/nvfp4_geometry.h"
#    include "ops/linear/nvfp4/nvfp4_layout.h"
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail::gemma4 {
namespace {

// ---- router -------------------------------------------------------------------------------------
// logits[t, e] = (R[e] . (x_t * scale)) * 2816^-0.5 / rms(x_t). The product
// x*scale is split into a BF16 head and a BF16 tail and both go through the
// MMA, so the logit carries about sixteen significant bits of the product
// rather than eight: selection then agrees with the FP32 router of the smaller
// routes except at genuine near-ties.
constexpr int kRouterBM      = kRouterTileTokens;  // tokens per block
constexpr int kRouterBN      = kRouterTileExperts; // experts per block
constexpr int kRouterBK      = 64;
constexpr int kRouterWarps   = 4; // each warp: 16 tokens x 16 experts
constexpr int kRouterThreads = kRouterWarps * 32;
static_assert(kRouterKTiles * kRouterBK == kHidden);

// blockIdx.z is a slice of K: the block writes its partial dot products and,
// from the first expert tile, its partial sums of squares; the consumer adds
// the slices (RouterPartials).
__global__ __launch_bounds__(kRouterThreads) void router_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ router,
    const float* __restrict__ router_scale, float* __restrict__ dot_out,
    float* __restrict__ square_out, int tokens) {
    __shared__ __align__(16) __nv_bfloat16 weight_tile[2][kRouterBN * kRouterBK];
    __shared__ __align__(16) __nv_bfloat16 head_tile[2][kRouterBM * kRouterBK];
    __shared__ __align__(16) __nv_bfloat16 tail_tile[2][kRouterBM * kRouterBK];

    if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    const int split   = static_cast<int>(blockIdx.z);
    const int k_count = kRouterKTiles / static_cast<int>(gridDim.z);
    const int k_first = split * k_count;
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int token0  = static_cast<int>(blockIdx.x) * kRouterBM;
    const int expert0 = static_cast<int>(blockIdx.y) * kRouterBN;
    // Staging of the activation: one sixteen-byte chunk a thread.
    const int stage_token = tid >> 3;
    const int stage_chunk = tid & 7;
    const bool token_live = token0 + stage_token < tokens;
    const __nv_bfloat16* token_row =
        x + static_cast<std::int64_t>(token_live ? token0 + stage_token : 0) * kHidden;

    auto stage_weight = [&](int slot, int k_tile) {
        for (int item = tid; item < kRouterBN * (kRouterBK / 8); item += kRouterThreads) {
            const int row   = item >> 3;
            const int chunk = item & 7;
            cp_async<16, Cache::cg>(
                &weight_tile[slot][row * kRouterBK + gemm_swz64(row, chunk * 8)],
                router + static_cast<std::int64_t>(expert0 + row) * kHidden + k_tile * kRouterBK +
                    chunk * 8);
        }
        cp_commit();
    };
    float square = 0.0F;
    uint4 pending_x{};
    float4 pending_s0{}, pending_s1{};
    auto load_x = [&](int k_tile) {
        const int h = k_tile * kRouterBK + stage_chunk * 8;
        pending_x   = token_live ? load_vec<uint4>(token_row + h) : make_uint4(0, 0, 0, 0);
        pending_s0  = load_vec<float4>(router_scale + h);
        pending_s1  = load_vec<float4>(router_scale + h + 4);
    };
    auto store_x = [&](int slot) {
        float value[8];
        bf16x8_to_float(pending_x, value);
        const float scale[8]{pending_s0.x, pending_s0.y, pending_s0.z, pending_s0.w,
                             pending_s1.x, pending_s1.y, pending_s1.z, pending_s1.w};
        std::uint32_t head[4], tail[4];
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float v0             = value[2 * pair] * scale[2 * pair];
            const float v1             = value[2 * pair + 1] * scale[2 * pair + 1];
            const __nv_bfloat162 top   = __floats2bfloat162_rn(v0, v1);
            const float2 top_f         = __bfloat1622float2(top);
            const __nv_bfloat162 below = __floats2bfloat162_rn(v0 - top_f.x, v1 - top_f.y);
            head[pair]                 = *reinterpret_cast<const std::uint32_t*>(&top);
            tail[pair]                 = *reinterpret_cast<const std::uint32_t*>(&below);
            square                     = fmaf(value[2 * pair], value[2 * pair], square);
            square                     = fmaf(value[2 * pair + 1], value[2 * pair + 1], square);
        }
        const int at = stage_token * kRouterBK + gemm_swz64(stage_token, stage_chunk * 8);
        store_vec(&head_tile[slot][at], make_uint4(head[0], head[1], head[2], head[3]));
        store_vec(&tail_tile[slot][at], make_uint4(tail[0], tail[1], tail[2], tail[3]));
    };

    stage_weight(0, k_first);
    load_x(k_first);
    store_x(0);

    float acc[2][4] = {};
    const int a_row = (lane & 7) + (((lane >> 3) & 1) << 3);
    const int a_col = (lane >> 4) << 3;
    const int b_row = lane & 7;
    const int b_col = ((lane >> 3) & 1) << 3;

#pragma unroll 1
    for (int step = 0; step < k_count; ++step) {
        const int slot = step & 1;
        cp_wait<0>();
        __syncthreads();
        const bool more = step + 1 < k_count;
        if (more) {
            stage_weight(slot ^ 1, k_first + step + 1);
            load_x(k_first + step + 1);
        }
#pragma unroll
        for (int k16 = 0; k16 < kRouterBK / 16; ++k16) {
            unsigned head[4], tail[4];
            const int col = k16 * 16 + a_col;
            ldmatrix_x4(head[0], head[1], head[2], head[3],
                        smem_addr(&head_tile[slot][a_row * kRouterBK + gemm_swz64(a_row, col)]));
            ldmatrix_x4(tail[0], tail[1], tail[2], tail[3],
                        smem_addr(&tail_tile[slot][a_row * kRouterBK + gemm_swz64(a_row, col)]));
#pragma unroll
            for (int n8 = 0; n8 < 2; ++n8) {
                const int row = warp * 16 + n8 * 8 + b_row;
                unsigned b[2];
                ldmatrix_x2(
                    b[0], b[1],
                    smem_addr(
                        &weight_tile[slot][row * kRouterBK + gemm_swz64(row, k16 * 16 + b_col)]));
                mma_bf16(acc[n8][0], acc[n8][1], acc[n8][2], acc[n8][3], head[0], head[1], head[2],
                         head[3], b[0], b[1]);
                mma_bf16(acc[n8][0], acc[n8][1], acc[n8][2], acc[n8][3], tail[0], tail[1], tail[2],
                         tail[3], b[0], b[1]);
            }
        }
        if (more) { store_x(slot ^ 1); }
    }

    square = warp_sum<8>(square);
    if (blockIdx.y == 0 && stage_chunk == 0 && token_live) {
        square_out[static_cast<std::int64_t>(split) * tokens + token0 + stage_token] = square;
    }
    const int gid = lane >> 2;
    const int lid = lane & 3;
#pragma unroll
    for (int n8 = 0; n8 < 2; ++n8) {
        const int expert = expert0 + warp * 16 + n8 * 8 + 2 * lid;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int token = token0 + gid + half * 8;
            if (token < tokens) {
                float* destination =
                    dot_out + (static_cast<std::int64_t>(split) * tokens + token) * kExperts +
                    expert;
                store_vec(destination, make_float2(acc[n8][2 * half], acc[n8][2 * half + 1]));
            }
        }
    }
}

// ---- grouping -----------------------------------------------------------------------------------
// Selection writes a rank local to its eight-token tile; the scan turns the
// tile counts into bases, and the index kernel publishes where every assignment
// landed. The order inside an expert's run carries no numerics: every packed
// column is computed on its own and read back through packed_index in rank
// order.
__global__ __launch_bounds__(kRouteTile * 32) void select_count_kernel(
    RouterPartials router, float eps, const float* __restrict__ expert_scale, int tokens,
    int* __restrict__ ids, float* __restrict__ route_weights, int* __restrict__ local_rank,
    int* __restrict__ tile_counts) {
    __shared__ int counts[kExperts];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid < kExperts) { counts[tid] = 0; }
    __syncthreads();
    const int token = static_cast<int>(blockIdx.x) * kRouteTile + warp;
    if (token < tokens) {
        float logit[4];
        load_token_logits(router, token, eps, logit);
        const Top8 top = select_top8(logit);
        if (lane == 0) {
#pragma unroll
            for (int rank = 0; rank < kTopK; ++rank) {
                const int assignment      = token * kTopK + rank;
                ids[assignment]           = top.id[rank];
                route_weights[assignment] = top8_weight(top, rank, expert_scale);
                local_rank[assignment]    = atomicAdd(&counts[top.id[rank]], 1);
            }
        }
    }
    __syncthreads();
    if (tid < kExperts) {
        tile_counts[static_cast<std::int64_t>(blockIdx.x) * kExperts + tid] = counts[tid];
    }
}

__device__ __forceinline__ int block128_exclusive_scan(int value, int* totals) {
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    int inclusive  = value;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int carried = __shfl_up_sync(kFullWarpMask, inclusive, offset);
        if (lane >= offset) { inclusive += carried; }
    }
    if (lane == 31) { totals[warp] = inclusive; }
    __syncthreads();
    int base = 0;
    for (int w = 0; w < warp; ++w) { base += totals[w]; }
    __syncthreads();
    return base + inclusive - value;
}

__global__ __launch_bounds__(kExperts) void scan_kernel(
    const int* __restrict__ tile_counts, int route_tiles, int* __restrict__ tile_bases,
    int* __restrict__ offsets, int* __restrict__ job_experts, int* __restrict__ job_columns,
    int* __restrict__ job_count) {
    __shared__ int totals[kExperts / 32];
    const int expert = static_cast<int>(threadIdx.x);
    int count        = 0;
    for (int tile = 0; tile < route_tiles; ++tile) {
        count += tile_counts[static_cast<std::int64_t>(tile) * kExperts + expert];
    }
    const int base  = block128_exclusive_scan(count, totals);
    offsets[expert] = base;
    if (expert == kExperts - 1) { offsets[kExperts] = base + count; }
    int cursor = base;
    for (int tile = 0; tile < route_tiles; ++tile) {
        const std::int64_t index = static_cast<std::int64_t>(tile) * kExperts + expert;
        tile_bases[index]        = cursor;
        cursor += tile_counts[index];
    }
    const int jobs      = (count + kJobColumns - 1) / kJobColumns;
    const int job_begin = block128_exclusive_scan(jobs, totals);
    for (int job = 0; job < jobs; ++job) {
        job_experts[job_begin + job] = expert;
        job_columns[job_begin + job] = job * kJobColumns;
    }
    const int touched = __syncthreads_count(count > 0);
    if (expert == kExperts - 1) {
        job_count[0] = job_begin + jobs;
        job_count[1] = touched;
    }
}

__global__ void index_kernel(const int* __restrict__ ids, const int* __restrict__ local_rank,
                             const int* __restrict__ tile_bases, int assignments,
                             int* __restrict__ packed_index, int* __restrict__ packed_token,
                             int* __restrict__ packed_expert) {
    const int assignment =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (assignment >= assignments) { return; }
    const int token  = assignment / kTopK;
    const int expert = ids[assignment];
    const int tile   = token / kRouteTile;
    const int packed =
        tile_bases[static_cast<std::int64_t>(tile) * kExperts + expert] + local_rank[assignment];
    packed_index[assignment] = packed;
    packed_token[packed]     = token;
    packed_expert[packed]    = expert;
}

// ---- jobs ---------------------------------------------------------------------------------------
// One grid row a job: an expert's slice of packed columns. Both tiers' grouped GEMMs read it.
struct Jobs {
    const int* __restrict__ experts;
    const int* __restrict__ columns;
    const int* __restrict__ count;

    __device__ __forceinline__ int expert() const { return experts[blockIdx.y]; }

    __device__ __forceinline__ int column_base() const { return columns[blockIdx.y]; }

    __device__ __forceinline__ bool live() const { return static_cast<int>(blockIdx.y) < count[0]; }
};

// The A4 tier (kind::mxf4nvf4) exists only in the build with the NVFP4 family; everything from
// here to the A16 GEMM is its encoding, schedules and epilogues.
#ifndef NINFERNO_WITHOUT_NVFP4

// ---- A4 input encoding --------------------------------------------------------------------------
// Per expert: every packed column is the hidden state of its token encoded
// against its expert's checkpoint divisor, sixteen values a thread.
__global__ __launch_bounds__(256) void quantize_per_expert_kernel(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ packed_token,
    const int* __restrict__ packed_expert, const float* __restrict__ divisors, int assignments,
    std::uint8_t* __restrict__ codes, std::uint8_t* __restrict__ scales) {
    constexpr int kGroups = kHidden / 16;
    const int task =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (task >= assignments * kGroups) { return; }
    const int packed                  = task / kGroups;
    const int group                   = task - packed * kGroups;
    const Nvfp4QuantizedK16 quantized = quantize_nvfp4_k16(
        x + static_cast<std::int64_t>(packed_token[packed]) * kHidden + group * 16,
        divisors[packed_expert[packed]]);
    store_vec(codes + static_cast<std::int64_t>(packed) * (kHidden / 2) + group * 8,
              make_uint2(quantized.codes_lo, quantized.codes_hi));
    scales[static_cast<std::int64_t>(packed) * kGroups + group] = quantized.scale;
}

// Per token: one encoding of each token, against a divisor taken from its own
// largest magnitude so that its largest block lands exactly on the top of the
// e4m3 range.
constexpr int kQuantizeTokenThreads = kHidden / 16; // 176

__global__ __launch_bounds__(kQuantizeTokenThreads) void quantize_per_token_kernel(
    const __nv_bfloat16* __restrict__ x, std::uint8_t* __restrict__ codes,
    std::uint8_t* __restrict__ scales, float* __restrict__ token_divisor) {
    __shared__ float warp_max_abs[(kQuantizeTokenThreads + 31) / 32];
    const int token             = static_cast<int>(blockIdx.x);
    const int group             = static_cast<int>(threadIdx.x);
    const int lane              = group & 31;
    const int warp              = group >> 5;
    const __nv_bfloat16* source = x + static_cast<std::int64_t>(token) * kHidden + group * 16;
    const uint4 packed0         = load_vec<uint4>(source);
    const uint4 packed1         = load_vec<uint4>(source + 8);
    const std::uint32_t represented[8]{packed0.x, packed0.y, packed0.z, packed0.w,
                                       packed1.x, packed1.y, packed1.z, packed1.w};
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const float2 value = bf16x2_bits_to_float2(represented[pair]);
        max_abs            = fmaxf(max_abs, fmaxf(fabsf(value.x), fabsf(value.y)));
    }
    max_abs = warp_max(max_abs);
    if (lane == 0) { warp_max_abs[warp] = max_abs; }
    __syncthreads();
    float token_max = 0.0F;
#pragma unroll
    for (int w = 0; w < (kQuantizeTokenThreads + 31) / 32; ++w) {
        token_max = fmaxf(token_max, warp_max_abs[w]);
    }
    const float divisor               = token_max > 0.0F ? 2688.0F / token_max : 1.0F;
    const Nvfp4QuantizedK16 quantized = quantize_nvfp4_k16_bits(represented, divisor);
    store_vec(codes + static_cast<std::int64_t>(token) * (kHidden / 2) + group * 8,
              make_uint2(quantized.codes_lo, quantized.codes_hi));
    scales[static_cast<std::int64_t>(token) * kQuantizeTokenThreads + group] = quantized.scale;
    if (group == 0) { token_divisor[token] = divisor; }
}

// ---- A4 grouped GEMMs: policies for the shared W4A4 MMA kernel ----------------------------------
using GateUpGeometry = Nvfp4Geometry<kGateUpTotal, kHidden>;
using DownGeometry   = Nvfp4Geometry<kDownTotal, kInter>;

using A4GateUpSchedule = Nvfp4A4MmaSchedule<64, 128, 128, 2, 4, 2, 3>;
using A4DownSchedule   = Nvfp4A4MmaSchedule<64, 128, 64, 2, 4, 4, 3>;
// Half the rows a CTA: a job is one expert's column tile, and with a few tokens an expert the
// weights are the whole cost, so twice the CTAs a job read the same bytes with twice the loads in
// flight. An option until it is timed against the wide tile (PrefillOptions::narrow).
using A4GateUpNarrowSchedule = Nvfp4A4MmaSchedule<64, 64, 128, 2, 2, 3, 3>;
using A4DownNarrowSchedule   = Nvfp4A4MmaSchedule<64, 64, 64, 2, 2, 4, 3>;

template <class Schedule>
struct ScheduleTag {
    using type = Schedule;
};

struct GateUpRaster {
    Jobs jobs;

    __device__ __forceinline__ void blocks(int& block_row, int& block_token) const {
        block_row   = static_cast<int>(blockIdx.x);
        block_token = jobs.column_base() / kJobColumns;
    }

    __device__ __forceinline__ bool live() const { return jobs.live(); }
};

template <int BlockN>
struct DownRaster {
    Jobs jobs;

    __device__ __forceinline__ void blocks(int& block_row, int& block_token) const {
        block_row   = jobs.expert() * (kHidden / BlockN) + static_cast<int>(blockIdx.x);
        block_token = jobs.column_base() / kJobColumns;
    }

    __device__ __forceinline__ bool live() const { return jobs.live(); }
};

template <int RowsPerBranch>
struct GateUpRows {
    static constexpr bool kContiguous   = false;
    static constexpr bool kPaired       = true;
    static constexpr int kRowsPerBranch = RowsPerBranch;
    Jobs jobs;

    __device__ __forceinline__ int weight_row(int row_begin, int local_row, int) const {
        const int within = row_begin + (local_row & (kRowsPerBranch - 1)) +
                           (local_row >= kRowsPerBranch ? kInter : 0);
        return jobs.expert() * kGateUpRows + within;
    }
};

// Columns already in packed order: an offset into the expert's run.
struct PackedTokens {
    const int* __restrict__ offsets;
    Jobs jobs;

    __device__ __forceinline__ int source_token(int column) const {
        return offsets[jobs.expert()] + column;
    }

    __device__ __forceinline__ int active_tokens(int) const {
        const int expert = jobs.expert();
        return offsets[expert + 1] - offsets[expert];
    }
};

// Columns gathered from a token-ordered plane through the route.
struct GatheredTokens {
    const int* __restrict__ packed_token;
    const int* __restrict__ offsets;
    Jobs jobs;

    __device__ __forceinline__ int source_token(int column) const {
        return packed_token[offsets[jobs.expert()] + column];
    }

    __device__ __forceinline__ int active_tokens(int) const {
        const int expert = jobs.expert();
        return offsets[expert + 1] - offsets[expert];
    }
};

// Both divisors of the expert: the weight's and the activation's. A per-token
// activation divisor is looked up through the column's token instead.
struct GateUpEpilogue {
    const float* __restrict__ weight_divisors;
    const float* __restrict__ input_divisors;
    const float* __restrict__ token_divisors;
    const int* __restrict__ packed_token;
    const int* __restrict__ offsets;
    Jobs jobs;

    __device__ __forceinline__ float apply(std::int32_t, std::int32_t column, float value) const {
        const int expert  = jobs.expert();
        const float input = token_divisors != nullptr
                                ? token_divisors[packed_token[offsets[expert] + column]]
                                : input_divisors[expert];
        return value / (weight_divisors[expert] * input);
    }
};

struct DownEpilogue {
    const float* __restrict__ weight_divisors;
    const float* __restrict__ input_divisors;
    Jobs jobs;

    __device__ __forceinline__ float apply(std::int32_t, std::int32_t, float value) const {
        const int expert = jobs.expert();
        return value / (weight_divisors[expert] * input_divisors[expert]);
    }
};

union Bf16Pair {
    unsigned bits;
    __nv_bfloat162 values;
};

__device__ __forceinline__ unsigned geglu_pair(unsigned gate_bits, unsigned up_bits) {
    const Bf16Pair gate{gate_bits};
    const Bf16Pair up{up_bits};
    const float2 g = __bfloat1622float2(gate.values);
    const float2 u = __bfloat1622float2(up.values);
    Bf16Pair result;
    result.values = __floats2bfloat162_rn(gelu_tanh(g.x) * u.x, gelu_tanh(g.y) * u.y);
    return result.bits;
}

// Half an NVFP4 group of the GeGLU product per lane; the neighbouring lane
// holds the other half and completes the group maximum. Encoded against the
// down projection's divisor of the expert.
__device__ __forceinline__ void geglu_store_half_group(std::uint8_t* codes, std::uint8_t* scales,
                                                       std::int64_t packed, std::int32_t row,
                                                       uint4 gate, uint4 up, float divisor) {
    const std::uint32_t bits[4] = {geglu_pair(gate.x, up.x), geglu_pair(gate.y, up.y),
                                   geglu_pair(gate.z, up.z), geglu_pair(gate.w, up.w)};
    float2 values[4];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        values[pair] = bf16x2_bits_to_float2(bits[pair]);
        max_abs      = fmaxf(max_abs, fmaxf(fabsf(values[pair].x), fabsf(values[pair].y)));
    }
    const unsigned pair_mask    = 3U << (threadIdx.x & 31U & ~1U);
    max_abs                     = fmaxf(max_abs, __shfl_xor_sync(pair_mask, max_abs, 1));
    const float scale_unencoded = __fdiv_rn(divisor * max_abs, 6.0F);
    const std::uint8_t scale    = __nv_cvt_float_to_fp8(scale_unencoded, __NV_SATFINITE, __NV_E4M3);
    std::uint32_t word          = 0;
    if (scale != 0) {
        const float reciprocal = __frcp_rn(decode_nvfp4_e4m3(scale));
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            values[pair].x = values[pair].x * divisor * reciprocal;
            values[pair].y = values[pair].y * divisor * reciprocal;
        }
        word = pack_nvfp4_e2m1x8(values);
    }
    const int half      = (row >> 3) & 1;
    const int group_row = row & ~15;
    store_vec(codes + packed * (kInter / 2) + (group_row >> 1) + half * 4, word);
    if (half == 0) { scales[packed * (kInter / 16) + (group_row >> 4)] = scale; }
}

struct GateUpA4Output {
    std::uint8_t* codes;
    std::uint8_t* scales;
    const int* __restrict__ offsets;
    const float* __restrict__ down_input_divisors;
    Jobs jobs;

    __device__ __forceinline__ void store_pair_vector(std::int32_t row, std::int32_t column,
                                                      uint4 gate, uint4 up) const {
        const int expert = jobs.expert();
        geglu_store_half_group(codes, scales, offsets[expert] + column, row, gate, up,
                               down_input_divisors[expert]);
    }
};

struct DownOutput {
    __nv_bfloat16* data;
    const int* __restrict__ offsets;
    Jobs jobs;

    __device__ __forceinline__ void store_vector(std::int32_t row, std::int32_t column,
                                                 uint4 values) const {
        const int expert          = jobs.expert();
        const std::int64_t packed = offsets[expert] + column;
        store_vec(data + packed * kHidden + (row - expert * kHidden), values);
    }
};

#endif // !NINFERNO_WITHOUT_NVFP4: the A4 tier

// ---- A16 grouped GEMM ---------------------------------------------------------------------------
// BF16 activations against NVFP4 weights decoded in shared memory. e2m1 times
// an e4m3 scale has at most five significant bits and stays inside the BF16
// exponent range, so the decoded tile is the stored weight exactly; the divisor
// is applied to the FP32 accumulator.
constexpr int kA16BM      = 64;  // packed columns
constexpr int kA16BN      = 128; // weight rows
constexpr int kA16BK      = 64;
constexpr int kA16Stages  = 2;
constexpr int kA16Threads = 256; // 2 (M) x 4 (N) warps, 32 x 32 each

enum class A16Mode { GateUp, Down };

template <A16Mode Mode>
struct A16Traits;

template <>
struct A16Traits<A16Mode::GateUp> {
    static constexpr int kK          = kHidden;
    static constexpr int kScaleTiles = kHidden / 64;
    static constexpr int kRowBlocks  = kInter / (kA16BN / 2);
};

template <>
struct A16Traits<A16Mode::Down> {
    static constexpr int kK          = kInter;
    static constexpr int kScaleTiles = kInter / 64;
    static constexpr int kRowBlocks  = kHidden / kA16BN;
};

struct A16Shared {
    __nv_bfloat16 activation[kA16Stages][kA16BM * kA16BK];
    std::uint8_t codes[kA16Stages][kA16BN * kA16BK / 2];
    std::uint8_t scales[kA16Stages][kA16BN * 4];
    __nv_bfloat16 weight[kA16BN * kA16BK];
};

template <A16Mode Mode>
__global__ __launch_bounds__(kA16Threads, 2) void a16_grouped_kernel(
    const __nv_bfloat16* __restrict__ activation, const int* __restrict__ packed_token,
    const int* __restrict__ offsets, Jobs jobs, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, const float* __restrict__ divisors,
    __nv_bfloat16* __restrict__ output) {
    using Traits          = A16Traits<Mode>;
    constexpr int kKTiles = Traits::kK / kA16BK;
    __shared__ __align__(16) A16Shared shared;
    if (!jobs.live()) { return; }
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    const int warp_m      = warp >> 2;
    const int warp_n      = warp & 3;
    const int expert      = jobs.expert();
    const int column_base = jobs.column_base();
    const int begin       = offsets[expert];
    const int active      = min(kA16BM, offsets[expert + 1] - begin - column_base);
    const int block_row   = static_cast<int>(blockIdx.x);

    // Weight row of a local row of the tile, in the stacked plane.
    auto weight_row = [&](int local) -> std::int64_t {
        if constexpr (Mode == A16Mode::GateUp) {
            const int within = block_row * (kA16BN / 2) + (local & (kA16BN / 2 - 1)) +
                               (local >= kA16BN / 2 ? kInter : 0);
            return static_cast<std::int64_t>(expert) * kGateUpRows + within;
        } else {
            return static_cast<std::int64_t>(expert) * kHidden + block_row * kA16BN + local;
        }
    };

    // Activation rows: two sixteen-byte chunks a thread, resolved once for the
    // whole job.
    const __nv_bfloat16* source_row[2];
    int source_bytes[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int item   = tid + i * kA16Threads;
        const int column = item >> 3;
        const bool live  = column < active;
        const int packed = begin + column_base + (live ? column : 0);
        if constexpr (Mode == A16Mode::GateUp) {
            source_row[i] = activation + static_cast<std::int64_t>(packed_token[packed]) * kHidden;
        } else {
            source_row[i] = activation + static_cast<std::int64_t>(packed) * kInter;
        }
        source_bytes[i] = live ? 16 : 0;
    }

    auto stage = [&](int slot, int k_tile) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int item   = tid + i * kA16Threads;
            const int column = item >> 3;
            const int chunk  = item & 7;
            cp_async_zfill<16, Cache::cg>(
                &shared.activation[slot][column * kA16BK + gemm_swz64(column, chunk * 8)],
                source_row[i] + k_tile * kA16BK + chunk * 8, source_bytes[i]);
        }
        // Codes: 128 rows x 32 bytes, two 16-byte chunks a row.
        {
            const int row  = tid >> 1;
            const int half = tid & 1;
            cp_async<16, Cache::cg>(&shared.codes[slot][row * 32 + half * 16],
                                    codes + weight_row(row) * (Traits::kK / 2) + k_tile * 32 +
                                        half * 16);
        }
        // Scales: the four block scales of a row in this 64-column tile are four
        // adjacent bytes.
        if (tid < kA16BN) {
            const std::int64_t row = weight_row(tid);
            const std::int64_t at  = ((row >> 7) * Traits::kScaleTiles + k_tile) * 512 +
                                    (row & 31) * 16 + ((row & 127) >> 5) * 4;
            cp_async<4>(&shared.scales[slot][tid * 4], scales + at);
        }
    };

    auto decode = [&](int slot) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int item  = tid + i * kA16Threads;
            const int row   = item >> 3;
            const int chunk = item & 7;
            const std::uint32_t packed =
                *reinterpret_cast<const std::uint32_t*>(&shared.codes[slot][row * 32 + chunk * 4]);
            const float scale = decode_nvfp4_e4m3(shared.scales[slot][row * 4 + (chunk >> 1)]);
            float value[8];
            decode_e2m1x8(packed, value);
            const uint4 bits{pack_bf16x2(value[0] * scale, value[1] * scale),
                             pack_bf16x2(value[2] * scale, value[3] * scale),
                             pack_bf16x2(value[4] * scale, value[5] * scale),
                             pack_bf16x2(value[6] * scale, value[7] * scale)};
            store_vec(&shared.weight[row * kA16BK + gemm_swz64(row, chunk * 8)], bits);
        }
    };

#pragma unroll
    for (int s = 0; s < kA16Stages; ++s) {
        stage(s, s);
        cp_commit();
    }

    float acc[2][4][4] = {};
    const int a_row    = (lane & 7) + (((lane >> 3) & 1) << 3);
    const int a_col    = (lane >> 4) << 3;
    const int b_row    = lane & 7;
    const int b_col    = ((lane >> 3) & 1) << 3;

#pragma unroll 1
    for (int k_tile = 0; k_tile < kKTiles; ++k_tile) {
        const int slot = k_tile % kA16Stages;
        cp_wait<kA16Stages - 1>();
        __syncthreads();
        decode(slot);
        __syncthreads();
#pragma unroll
        for (int k16 = 0; k16 < kA16BK / 16; ++k16) {
            unsigned a[2][4];
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                const int row = warp_m * 32 + m * 16 + a_row;
                ldmatrix_x4(a[m][0], a[m][1], a[m][2], a[m][3],
                            smem_addr(&shared.activation[slot][row * kA16BK +
                                                               gemm_swz64(row, k16 * 16 + a_col)]));
            }
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int row = warp_n * 32 + n * 8 + b_row;
                unsigned b[2];
                ldmatrix_x2(
                    b[0], b[1],
                    smem_addr(&shared.weight[row * kA16BK + gemm_swz64(row, k16 * 16 + b_col)]));
#pragma unroll
                for (int m = 0; m < 2; ++m) {
                    mma_bf16(acc[m][n][0], acc[m][n][1], acc[m][n][2], acc[m][n][3], a[m][0],
                             a[m][1], a[m][2], a[m][3], b[0], b[1]);
                }
            }
        }
        __syncthreads();
        if (k_tile + kA16Stages < kKTiles) { stage(slot, k_tile + kA16Stages); }
        cp_commit();
    }
    cp_wait<0>();
    __syncthreads();

    // Epilogue through shared memory: [64 columns][128 rows] FP32 after the
    // divisor.
    static_assert(sizeof(A16Shared) >= kA16BM * (kA16BN + 4) * sizeof(float));
    auto* tile          = reinterpret_cast<float*>(&shared);
    constexpr int kPad  = kA16BN + 4;
    const float inverse = 1.0F / divisors[expert];
    const int gid       = lane >> 2;
    const int lid       = lane & 3;
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int column                    = warp_m * 32 + m * 16 + gid;
            const int row                       = warp_n * 32 + n * 8 + 2 * lid;
            tile[column * kPad + row]           = acc[m][n][0] * inverse;
            tile[column * kPad + row + 1]       = acc[m][n][1] * inverse;
            tile[(column + 8) * kPad + row]     = acc[m][n][2] * inverse;
            tile[(column + 8) * kPad + row + 1] = acc[m][n][3] * inverse;
        }
    }
    __syncthreads();
    if constexpr (Mode == A16Mode::GateUp) {
        // 64 columns x 64 products, eight a thread-task.
        for (int task = tid; task < kA16BM * (kA16BN / 2) / 8; task += kA16Threads) {
            const int column = task >> 3;
            const int j8     = (task & 7) * 8;
            if (column < active) {
                float product[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    product[i] = gelu_tanh(tile[column * kPad + j8 + i]) *
                                 tile[column * kPad + kA16BN / 2 + j8 + i];
                }
                const uint4 bits{
                    pack_bf16x2(product[0], product[1]), pack_bf16x2(product[2], product[3]),
                    pack_bf16x2(product[4], product[5]), pack_bf16x2(product[6], product[7])};
                const std::int64_t packed = begin + column_base + column;
                store_vec(output + packed * kInter + block_row * (kA16BN / 2) + j8, bits);
            }
        }
    } else {
        for (int task = tid; task < kA16BM * kA16BN / 8; task += kA16Threads) {
            const int column = task >> 4;
            const int r8     = (task & 15) * 8;
            if (column < active) {
                const float* value = tile + column * kPad + r8;
                const uint4 bits{pack_bf16x2(value[0], value[1]), pack_bf16x2(value[2], value[3]),
                                 pack_bf16x2(value[4], value[5]), pack_bf16x2(value[6], value[7])};
                const std::int64_t packed = begin + column_base + column;
                store_vec(output + packed * kHidden + block_row * kA16BN + r8, bits);
            }
        }
    }
}

// ---- reduction ----------------------------------------------------------------------------------
__global__ __launch_bounds__(kHidden /
                             8) void reduce_kernel(const __nv_bfloat16* __restrict__ expert_out,
                                                   const int* __restrict__ packed_index,
                                                   const float* __restrict__ route_weights,
                                                   __nv_bfloat16* __restrict__ out) {
    const int token = static_cast<int>(blockIdx.x);
    const int h     = static_cast<int>(threadIdx.x) * 8;
    float value[8]  = {};
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        const int assignment = token * kTopK + rank;
        const float weight   = route_weights[assignment];
        float y[8];
        bf16x8_to_float(
            load_vec<uint4>(expert_out +
                            static_cast<std::int64_t>(packed_index[assignment]) * kHidden + h),
            y);
#pragma unroll
        for (int i = 0; i < 8; ++i) { value[i] = fmaf(weight, y[i], value[i]); }
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

void launch_slice(const __nv_bfloat16* router_input, const __nv_bfloat16* expert_input,
                  const Gemma4MoeWeights& weights, float eps, __nv_bfloat16* out, int tokens,
                  const PrefillWorkspace& ws, PrefillOptions options, cudaStream_t stream) {
    auto* router_dot         = static_cast<float*>(ws.router_dot.data);
    auto* router_square      = static_cast<float*>(ws.router_square.data);
    auto* ids                = static_cast<int*>(ws.selection.ids.data);
    auto* route_weights      = static_cast<float*>(ws.selection.weights.data);
    auto* local_rank         = static_cast<int*>(ws.local_rank.data);
    auto* tile_counts        = static_cast<int*>(ws.tile_counts.data);
    auto* tile_bases         = static_cast<int*>(ws.tile_bases.data);
    auto* offsets            = static_cast<int*>(ws.offsets.data);
    auto* job_experts        = static_cast<int*>(ws.job_experts.data);
    auto* job_columns        = static_cast<int*>(ws.job_columns.data);
    auto* job_count          = static_cast<int*>(ws.job_count.data);
    auto* packed_index       = static_cast<int*>(ws.packed_index.data);
    auto* packed_token       = static_cast<int*>(ws.packed_token.data);
    auto* packed_expert      = static_cast<int*>(ws.packed_expert.data);
    auto* expert_out         = static_cast<__nv_bfloat16*>(ws.expert_out.data);
    const auto* expert_scale = static_cast<const float*>(weights.expert_scale.data);

    const int assignments = tokens * kTopK;
    const int route_tiles = (tokens + kRouteTile - 1) / kRouteTile;
    const int jobs_bound  = max_route_jobs(assignments);

    const int splits = router_splits(RouterKind::Mma, tokens);
    launch_router_mma(router_input, tokens, weights, router_dot, router_square, splits, stream);
    select_count_kernel<<<route_tiles, kRouteTile * 32, 0, stream>>>(
        RouterPartials{router_dot, router_square, splits, tokens}, eps, expert_scale, tokens, ids,
        route_weights, local_rank, tile_counts);
    CUDA_CHECK(cudaGetLastError());
    scan_kernel<<<1, kExperts, 0, stream>>>(tile_counts, route_tiles, tile_bases, offsets,
                                            job_experts, job_columns, job_count);
    CUDA_CHECK(cudaGetLastError());
    index_kernel<<<(assignments + 255) / 256, 256, 0, stream>>>(
        ids, local_rank, tile_bases, assignments, packed_index, packed_token, packed_expert);
    CUDA_CHECK(cudaGetLastError());

    const Jobs jobs{job_experts, job_columns, job_count};
    if (options.route == Route::PrefillA16) {
        auto* middle = static_cast<__nv_bfloat16*>(ws.middle.data);
        a16_grouped_kernel<A16Mode::GateUp>
            <<<dim3(A16Traits<A16Mode::GateUp>::kRowBlocks, jobs_bound), kA16Threads, 0, stream>>>(
                expert_input, packed_token, offsets, jobs, codes_of(weights.gate_up),
                scales_of(weights.gate_up), divisors_of(weights.gate_up), middle);
        CUDA_CHECK(cudaGetLastError());
        a16_grouped_kernel<A16Mode::Down>
            <<<dim3(A16Traits<A16Mode::Down>::kRowBlocks, jobs_bound), kA16Threads, 0, stream>>>(
                middle, packed_token, offsets, jobs, codes_of(weights.down),
                scales_of(weights.down), divisors_of(weights.down), expert_out);
        CUDA_CHECK(cudaGetLastError());
    }
#ifdef NINFERNO_WITHOUT_NVFP4
    else {
        throw std::logic_error("gemma4_moe: the A4 prefill tier needs the NVFP4 MMA (sm_120a)");
    }
#else
    else {
        auto* input_codes          = static_cast<std::uint8_t*>(ws.input_codes.data);
        auto* input_scales         = static_cast<std::uint8_t*>(ws.input_scales.data);
        auto* middle_codes         = static_cast<std::uint8_t*>(ws.middle.data);
        auto* middle_scale         = static_cast<std::uint8_t*>(ws.middle_scales.data);
        const auto* gate_up_inputs = static_cast<const float*>(weights.gate_up_input_divisors.data);
        const auto* down_inputs    = static_cast<const float*>(weights.down_input_divisors.data);
        const bool per_token       = options.input == A4Input::PerToken;
        auto* token_divisor        = static_cast<float*>(ws.token_divisor.data);
        if (per_token) {
            quantize_per_token_kernel<<<tokens, kQuantizeTokenThreads, 0, stream>>>(
                expert_input, input_codes, input_scales, token_divisor);
        } else {
            const int tasks = assignments * (kHidden / 16);
            quantize_per_expert_kernel<<<(tasks + 255) / 256, 256, 0, stream>>>(
                expert_input, packed_token, packed_expert, gate_up_inputs, assignments, input_codes,
                input_scales);
        }
        CUDA_CHECK(cudaGetLastError());

        const Nvfp4A4Plane input{input_codes, input_scales};
        const GateUpEpilogue gate_epilogue{divisors_of(weights.gate_up),
                                           gate_up_inputs,
                                           per_token ? token_divisor : nullptr,
                                           packed_token,
                                           offsets,
                                           jobs};
        const GateUpA4Output gate_output{middle_codes, middle_scale, offsets, down_inputs, jobs};
        const auto gate_up = [&](auto tag) {
            using Schedule          = typename decltype(tag)::type;
            constexpr int kPairRows = Schedule::kBlockRows / 2;
            using Rows              = GateUpRows<kPairRows>;
            static_assert(Schedule::kBlockTokens == kJobColumns && kInter % kPairRows == 0);
            const dim3 grid(kInter / kPairRows, jobs_bound);
            if (per_token) {
                launch_nvfp4_a4_mma_geometry<Schedule, GateUpGeometry>(
                    grid, input, codes_of(weights.gate_up), scales_of(weights.gate_up), assignments,
                    1.0F, gate_epilogue, gate_output, Rows{jobs},
                    GatheredTokens{packed_token, offsets, jobs}, GateUpRaster{jobs}, stream);
            } else {
                launch_nvfp4_a4_mma_geometry<Schedule, GateUpGeometry>(
                    grid, input, codes_of(weights.gate_up), scales_of(weights.gate_up), assignments,
                    1.0F, gate_epilogue, gate_output, Rows{jobs}, PackedTokens{offsets, jobs},
                    GateUpRaster{jobs}, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        };
        const Nvfp4A4Plane middle{middle_codes, middle_scale};
        const auto down = [&](auto tag) {
            using Schedule = typename decltype(tag)::type;
            static_assert(Schedule::kBlockTokens == kJobColumns &&
                          kHidden % Schedule::kBlockRows == 0);
            const dim3 grid(kHidden / Schedule::kBlockRows, jobs_bound);
            launch_nvfp4_a4_mma_geometry<Schedule, DownGeometry>(
                grid, middle, codes_of(weights.down), scales_of(weights.down), assignments, 1.0F,
                DownEpilogue{divisors_of(weights.down), down_inputs, jobs},
                DownOutput{expert_out, offsets, jobs}, Nvfp4IdentityRows{},
                PackedTokens{offsets, jobs}, DownRaster<Schedule::kBlockRows>{jobs}, stream);
            CUDA_CHECK(cudaGetLastError());
        };
        if (options.narrow) {
            gate_up(ScheduleTag<A4GateUpNarrowSchedule>{});
            down(ScheduleTag<A4DownNarrowSchedule>{});
        } else {
            gate_up(ScheduleTag<A4GateUpSchedule>{});
            down(ScheduleTag<A4DownSchedule>{});
        }
    }
#endif

    reduce_kernel<<<tokens, kHidden / 8, 0, stream>>>(expert_out, packed_index, route_weights, out);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_router_mma(const void* x, std::int32_t tokens, const Gemma4MoeWeights& weights,
                       float* dot, float* square, std::int32_t splits, cudaStream_t stream) {
    if (splits < 1 || kRouterKTiles % splits != 0) {
        throw std::logic_error("gemma4_moe router: the K split must divide the k-tiles");
    }
    router_mma_kernel<<<dim3((tokens + kRouterBM - 1) / kRouterBM, kExperts / kRouterBN, splits),
                        kRouterThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(weights.router.qdata),
        static_cast<const float*>(weights.router_scale.data), dot, square, tokens);
    CUDA_CHECK(cudaGetLastError());
}

void launch_prefill(const Tensor& router_input, const Tensor& expert_input,
                    const Gemma4MoeWeights& weights, float eps, Tensor& out,
                    const PrefillWorkspace& workspace, std::int32_t capacity,
                    PrefillOptions options, cudaStream_t stream) {
    const std::int32_t tokens = router_input.ne[1];
    for (std::int32_t token0 = 0; token0 < tokens; token0 += capacity) {
        const std::int32_t slice = std::min(capacity, tokens - token0);
        launch_slice(static_cast<const __nv_bfloat16*>(router_input.data) +
                         static_cast<std::int64_t>(token0) * kHidden,
                     static_cast<const __nv_bfloat16*>(expert_input.data) +
                         static_cast<std::int64_t>(token0) * kHidden,
                     weights, eps,
                     static_cast<__nv_bfloat16*>(out.data) +
                         static_cast<std::int64_t>(token0) * kHidden,
                     slice, workspace, options, stream);
    }
}

} // namespace ninferno::ops::detail::gemma4
