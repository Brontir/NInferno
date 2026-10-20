#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/gemma4_moe/gemma4_moe_internal.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/sparse_moe/sparse_moe_route.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninferno::ops::detail::gemma4 {

inline constexpr float kInverseSqrtHidden = 0.018844459036110108F; // 2816^-0.5
inline constexpr int kHiddenScaleTiles    = kHidden / 64;          // 44
inline constexpr int kInterScaleTiles     = kInter / 64;           // 11

__device__ __forceinline__ float gelu_tanh(float z) {
    constexpr float kSqrt2OverPi = 0.7978845608028654F;
    const float inner            = kSqrt2OverPi * (z + 0.044715F * z * z * z);
    return 0.5F * z * (1.0F + tanhf(inner));
}

// Eight e2m1 codes of one 32-bit word, low nibble first.
__device__ __forceinline__ void decode_e2m1x8(std::uint32_t packed, float (&values)[8]) {
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const float2 code    = decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(packed >> (8 * pair)));
        values[2 * pair]     = code.x;
        values[2 * pair + 1] = code.y;
    }
}

__device__ __forceinline__ void bf16x8_to_float(uint4 bits, float (&values)[8]) {
    const float2 v0 = bf16x2_bits_to_float2(bits.x);
    const float2 v1 = bf16x2_bits_to_float2(bits.y);
    const float2 v2 = bf16x2_bits_to_float2(bits.z);
    const float2 v3 = bf16x2_bits_to_float2(bits.w);
    values[0]       = v0.x;
    values[1]       = v0.y;
    values[2]       = v1.x;
    values[3]       = v1.y;
    values[4]       = v2.x;
    values[5]       = v2.y;
    values[6]       = v3.x;
    values[7]       = v3.y;
}

__device__ __forceinline__ float dot8(const float (&a)[8], const float (&b)[8]) {
    float sum = 0.0F;
#pragma unroll
    for (int i = 0; i < 8; ++i) { sum = fmaf(a[i], b[i], sum); }
    return sum;
}

// Programmatic dependent launch. A consumer kernel starts while its producer still runs, and
// griddepcontrol.wait (pdl::wait_for_dependencies) is where the producer's writes become its to
// read. A load through a const __restrict__ pointer is an invariant load - the pointee may not
// change while the kernel runs - which the compiler is free to hoist above that wait, and did:
// small-T gate/up read its expert count before the grouping had written it. So every pointer a
// consumer reads producer output through, here and in the kernels, is a plain one; __restrict__
// stays on the weights and the call's inputs, which nothing in the chain writes.
//
// The router's output. A router block covers a slice of K, so a logit is the
// sum of `splits` partial dot products, and the token's mean square the sum of
// as many partial sums of squares:
//   logit[t][e] = (sum_s dot[s][t][e]) * 2816^-0.5 / sqrt(sum_s square[s][t] /
//   2816 + eps).
// The router scale is inside dot; the RMS normalisation is applied here, once
// per token.
struct RouterPartials {
    const float* dot;    // [splits][tokens][128], producer output: no __restrict__
    const float* square; // [splits][tokens]
    int splits;
    int tokens;
};

// Every lane's four logits: experts lane, lane+32, lane+64, lane+96 of token t. The partials of
// all slices are loaded before any of them is added, so the read is one L2 round trip instead of
// one a slice (at eleven slices the serial read cost a selection 6 us); the sums still run slice
// by slice, in order, so the logits are unchanged. MaxSplits bounds router.splits.
template <int MaxSplits = kMaxRouterSplits>
__device__ __forceinline__ void load_token_logits(const RouterPartials& router, int token,
                                                  float eps, float (&logit)[4]) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float square_part[MaxSplits];
    float dot_part[MaxSplits][4];
#pragma unroll
    for (int split = 0; split < MaxSplits; ++split) {
        if (split < router.splits) {
            square_part[split] = router.square[split * router.tokens + token];
            const float* row =
                router.dot + (static_cast<std::int64_t>(split) * router.tokens + token) * kExperts;
#pragma unroll
            for (int item = 0; item < 4; ++item) { dot_part[split][item] = row[lane + 32 * item]; }
        }
    }
    float square = 0.0F;
    float dot[4] = {};
#pragma unroll
    for (int split = 0; split < MaxSplits; ++split) {
        if (split < router.splits) {
            square += square_part[split];
#pragma unroll
            for (int item = 0; item < 4; ++item) { dot[item] += dot_part[split][item]; }
        }
    }
    const float factor = kInverseSqrtHidden / sqrtf(square / static_cast<float>(kHidden) + eps);
#pragma unroll
    for (int item = 0; item < 4; ++item) { logit[item] = dot[item] * factor; }
}

// The warp-wide top-8 of 128 logits held four a lane. Every lane ends with the
// same eight ids and logits in rank order; an exact tie goes to the lower
// expert id.
struct Top8 {
    int id[kTopK];
    float logit[kTopK];
};

__device__ __forceinline__ Top8 select_top8(const float (&logit)[4]) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    SparseMoeRankedValue run[kSparseMoeTopK];
#pragma unroll
    for (int item = 0; item < 4; ++item) { run[item] = {logit[item], lane + 32 * item}; }
    // A five-exchange network sorts the four real entries; the padding already
    // sorts last.
    sparse_moe_order_pair(run[0], run[1]);
    sparse_moe_order_pair(run[2], run[3]);
    sparse_moe_order_pair(run[0], run[2]);
    sparse_moe_order_pair(run[1], run[3]);
    sparse_moe_order_pair(run[1], run[2]);
#pragma unroll
    for (int item = 4; item < kSparseMoeTopK; ++item) { run[item] = {-INFINITY, 1024 + item}; }
    sparse_moe_merge_ranked_runs(run);
    Top8 out;
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        out.id[rank]    = run[rank].id;
        out.logit[rank] = run[rank].value;
    }
    return out;
}

// w_k = p_k / sum_top p * expert_scale[id_k]. The full softmax denominator
// cancels, so this is the softmax over the eight selected logits, times each
// expert's gain.
__device__ __forceinline__ float top8_weight(const Top8& top, int rank,
                                             const float* __restrict__ expert_scale) {
    float denominator = 0.0F;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) { denominator += expf(top.logit[k] - top.logit[0]); }
    return expf(top.logit[rank] - top.logit[0]) / denominator * expert_scale[top.id[rank]];
}

} // namespace ninferno::ops::detail::gemma4
