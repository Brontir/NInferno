// The routing device functions of the generic MoE Op, used by its routing kernels (moe_route.cu).
#pragma once

#include "ops/generic/generic_w4a8.cuh"
#include "ops/moe/moe_internal.h"

#include <math_constants.h>

#include <cstddef>
#include <cstdint>

namespace ninferno::ops::detail::moe {

inline constexpr int kPerLane = kMaxExperts / 32;

// One task of the grouped W4A8 activation quantizer: row `task / (ld / 32)`, 32-value block
// `task % (ld / 32)` of x (rows at stride ldx, K values), into codes at stride ld (blocks past K
// zero), scales and sums [row * ld / 32 + block]. The dense W4A8 quantizer's arithmetic
// (w4a8_quantize_block), so a row's codes are the dense path's and the R8 CTA's own.
__device__ __forceinline__ void quantize_x_task(const __nv_bfloat16* x, int ldx, int k, int ld,
                                                std::int64_t task, std::int8_t* codes,
                                                float* scales, float* sums) {
    const int blocks  = ld / 32;
    const int row     = static_cast<int>(task / blocks);
    const int block   = static_cast<int>(task - static_cast<std::int64_t>(row) * blocks);
    unsigned words[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float scale       = 0.0F;
    int qsum          = 0;
    if (block * 32 < k) {
        float v[32];
        generic::w4a8_load_block(x + static_cast<std::int64_t>(row) * ldx + block * 32, v);
        generic::w4a8_quantize_block(v, words, scale, qsum);
    }
    auto* dst = reinterpret_cast<uint4*>(codes + static_cast<std::int64_t>(row) * ld + block * 32);
    dst[0]    = make_uint4(words[0], words[1], words[2], words[3]);
    dst[1]    = make_uint4(words[4], words[5], words[6], words[7]);
    scales[task] = scale;
    sums[task]   = scale * static_cast<float>(qsum);
}

// The selection order as one unsigned key per expert: a rank's bits made monotonic, with -0
// folded into +0 (the two compare equal and tie by id); 0 is below every expert (an absent lane).
__device__ __forceinline__ unsigned rank_key(float rank) {
    const unsigned u = __float_as_uint(rank + 0.0F);
    return (u & 0x80000000U) ? ~u : (u | 0x80000000U);
}

__device__ __forceinline__ unsigned warp_max(unsigned v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_max_sync(0xffffffffU, v);
#else
    for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffU, v, o));
    return v;
#endif
}

// One warp per token; lane l holds experts l, l + 32, ... (PerLane of them at most). The k picks
// are k arg-max reductions over the experts not yet taken, by (rank, then the lower id): rank =
// score + selection bias, NaN ranked last. A pick is two warp max reductions, over the key and
// then over ~id among the lanes that hold it; the logits load independently. The picks, the
// softmax sum and the weights are those of the published order: each weight is its score over
// the warp's sum, the picked scores summed in pick order. Every load (logits, selection bias,
// expert scale) is issued up front: beside a streaming GEMV each dependent L2 round trip waits
// behind its loads. ids_shared (non-null): the ids also go to [token * k + rank] there.
template <int PerLane>
__device__ __forceinline__ void select_token_lanes(const SelectArgs& a, int token, int lane,
                                                   int* ids_shared) {
    const float* logits = a.logits + static_cast<std::int64_t>(token) * a.experts;
    const float NegInf  = -CUDART_INF_F;
    float l[PerLane], bias[PerLane], escale[PerLane];
#pragma unroll
    for (int i = 0; i < PerLane; ++i) {
        const int e = lane + 32 * i;
        const bool in = e < a.experts;
        l[i]          = in ? logits[e] : NegInf;
        bias[i]       = in && a.selection_bias != nullptr ? a.selection_bias[e] : 0.0F;
        escale[i]     = in && a.expert_scale != nullptr ? a.expert_scale[e] : 1.0F;
    }
    float m = NegInf;
    if (!a.sigmoid) {
#pragma unroll
        for (int i = 0; i < PerLane; ++i) {
            if (lane + 32 * i < a.experts) m = fmaxf(m, l[i]);
        }
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffU, m, o));
    }
    float score[PerLane];
    unsigned key[PerLane];
    float z = 0.0F;
#pragma unroll
    for (int i = 0; i < PerLane; ++i) {
        const int e = lane + 32 * i;
        score[i]    = 0.0F;
        key[i]      = 0U;
        if (e < a.experts) {
            score[i] = a.sigmoid ? 1.0F / (1.0F + expf(-l[i])) : expf(l[i] - m);
            z += a.sigmoid ? 0.0F : score[i];
            const float r = score[i] + bias[i];
            key[i]        = rank_key(r != r ? NegInf : r);
        }
    }
    if (!a.sigmoid) {
        for (int o = 16; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffffU, z, o);
    }

    // Lane k keeps pick k (no per-thread arrays of picks: under the 512-thread plan CTA they
    // spilled to local memory), and writes it.
    unsigned taken   = 0;
    float my_score   = 0.0F;
    float my_scale   = 1.0F;
    int my_id        = 0;
    float picked_sum = 0.0F;
    for (int k = 0; k < a.top_k; ++k) {
        // The lane's best untaken expert: the highest key, the lowest id among equal ones.
        unsigned best = 0U;
        int best_i    = -1;
#pragma unroll
        for (int i = 0; i < PerLane; ++i) {
            if (!((taken >> i) & 1U) && key[i] != 0U && (best_i < 0 || key[i] > best)) {
                best   = key[i];
                best_i = i;
            }
        }
        const unsigned id  = best_i < 0 ? 0xffffffffU : static_cast<unsigned>(lane + 32 * best_i);
        const unsigned top = warp_max(best);
        const int best_id  = static_cast<int>(~warp_max(best == top && best_i >= 0 ? ~id : 0U));
        // The winner's own score (not its selection rank) and expert scale.
        float s = 0.0F, sc = 1.0F;
        if ((best_id & 31) == lane) {
            const int i = best_id >> 5;
            taken |= 1U << i;
#pragma unroll
            for (int j = 0; j < PerLane; ++j) {
                if (j == i) {
                    s  = score[j];
                    sc = escale[j];
                }
            }
        }
        s  = __shfl_sync(0xffffffffU, s, best_id & 31);
        sc = __shfl_sync(0xffffffffU, sc, best_id & 31);
        if (!a.sigmoid) s /= z;
        if (lane == k) {
            my_score = s;
            my_scale = sc;
            my_id    = best_id;
        }
        picked_sum += s;
    }
    if (lane < a.top_k) {
        float w = a.renormalize ? my_score / picked_sum : my_score;
        w *= a.routed_scale;
        if (a.expert_scale != nullptr) w *= my_scale;
        a.ids[static_cast<std::int64_t>(token) * a.top_k + lane]     = my_id;
        a.weights[static_cast<std::int64_t>(token) * a.top_k + lane] = w;
        if (ids_shared != nullptr) ids_shared[token * a.top_k + lane] = my_id;
    }
}

// The lanes' arrays sized to the call's experts (a bound of 32 a lane kept them in registers but
// cost twice the time at 256 experts).
__device__ __forceinline__ void select_token(const SelectArgs& a, int token, int lane,
                                             int* ids_shared = nullptr) {
    if (a.experts <= 128) {
        select_token_lanes<4>(a, token, lane, ids_shared);
    } else if (a.experts <= 256) {
        select_token_lanes<8>(a, token, lane, ids_shared);
    } else {
        select_token_lanes<kPerLane>(a, token, lane, ids_shared);
    }
}

// The plan does not depend on the warp count: an expert's slots follow the assignment order
// whatever range each warp walks.
// One CTA of Warps warps. Warp w owns a contiguous range of assignments. (1) counts per (warp,
// expert); (2) each expert's slots start at offsets[e] and warp w's share of them at offsets[e] +
// base[w][e]; (3) every warp walks its range in order and hands out slots, so an expert's slots
// follow the assignment order.
template <int Warps>
__device__ __forceinline__ void plan_body(const PlanArgs& a, int* plan_smem) {
    const int E      = a.experts;
    int* base        = plan_smem;             // [Warps][E]
    int* offsets     = plan_smem + Warps * E; // [E]
    int* job_offsets = offsets + E;           // [E]
    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int A      = a.assignments;
    const int chunk  = (A + Warps - 1) / Warps;
    const int lo     = min(A, warp * chunk);
    const int hi     = min(A, lo + chunk);

    for (int i = tid; i < Warps * E; i += blockDim.x) base[i] = 0;
    __syncthreads();
    for (int x = lo + lane; x < hi; x += 32) atomicAdd(&base[warp * E + a.ids[x]], 1);
    __syncthreads();
    for (int e = tid; e < E; e += blockDim.x) {
        int running = 0;
        for (int w = 0; w < Warps; ++w) {
            const int c     = base[w * E + e];
            base[w * E + e] = running;
            running += c;
        }
        offsets[e]     = running;                                       // the total until the scan
        job_offsets[e] = (running + a.job_columns - 1) / a.job_columns; // jobs, likewise
    }
    __syncthreads();
    // Exclusive scans over the experts, by warp 0: lane l owns a contiguous run of experts.
    if (warp == 0) {
        const int per = (E + 31) / 32;
        const int b   = min(E, lane * per);
        const int end = min(E, b + per);
        int slots = 0, jobs = 0;
        for (int e = b; e < end; ++e) {
            slots += offsets[e];
            jobs += job_offsets[e];
        }
        int slot_prefix = slots, job_prefix = jobs;
        for (int o = 1; o < 32; o <<= 1) {
            const int s = __shfl_up_sync(0xffffffffU, slot_prefix, o);
            const int j = __shfl_up_sync(0xffffffffU, job_prefix, o);
            if (lane >= o) {
                slot_prefix += s;
                job_prefix += j;
            }
        }
        int slot_at = slot_prefix - slots;
        int job_at  = job_prefix - jobs;
        for (int e = b; e < end; ++e) {
            const int s = offsets[e], j = job_offsets[e];
            offsets[e]     = slot_at;
            job_offsets[e] = job_at;
            slot_at += s;
            job_at += j;
        }
        if (lane == 31) *a.job_total = job_prefix;
    }
    __syncthreads();
    // The job list: expert e's slots cut into runs of job_columns.
    const int width = a.job_columns;
    for (int e = tid; e < E; e += blockDim.x) {
        const int begin = offsets[e];
        const int total = (e + 1 < E ? offsets[e + 1] : A) - begin;
        for (int j = 0; j * width < total; ++j) {
            const int job     = job_offsets[e] + j;
            a.job_expert[job] = e;
            a.job_begin[job]  = begin + j * width;
            a.job_count[job]  = min(width, total - j * width);
        }
    }
    // Slots, in assignment order within every expert.
    const unsigned below = (1U << lane) - 1U;
    for (int x0 = lo; x0 < hi; x0 += 32) {
        const int x         = x0 + lane;
        const bool valid    = x < hi;
        const int e         = valid ? a.ids[x] : -1;
        const unsigned same = __match_any_sync(0xffffffffU, e);
        int slot            = 0;
        if (valid) slot = offsets[e] + base[warp * E + e] + __popc(same & below);
        __syncwarp();
        if (valid && (same & below) == 0) base[warp * E + e] += __popc(same);
        __syncwarp();
        if (valid) {
            a.packed_token[slot] = x / a.top_k;
            a.slot_of[x]         = slot;
        }
    }
}

// The same plan for at most 32 assignments (decode: up to four tokens of top-8), by one warp with
// assignment x in lane x: no per-expert counters to clear and scan, no barrier. The slot of x
// counts the assignments of lower experts and of its own expert before it, which is plan_body's
// expert-major, assignment order: one ballot per bit of the expert id, from the top, counts the
// lanes that first differ below x and narrows `same` to x's expert. An expert's first lane holds
// its jobs at its first slot in `jobs_at`; the exclusive scan over the 32 slots starts them after
// the jobs of all lower experts. The ids come from `ids` (the selection's shared copy).
// jobs_at is 32 ints of the caller's shared memory (no function-local __shared__ in a header
// many TUs include).
__device__ __forceinline__ void plan_warp(const PlanArgs& a, int lane, const int* ids,
                                          int* jobs_at) {
    const int A          = a.assignments;
    const bool valid     = lane < A;
    const int e          = valid ? ids[lane] : 0;
    const unsigned below = (1U << lane) - 1U;
    unsigned same        = __ballot_sync(0xffffffffU, valid);
    int lower            = 0;
    for (int b = 31 - __clz(max(a.experts - 1, 1)); b >= 0; --b) {
        const bool bit     = ((e >> b) & 1) != 0;
        const unsigned set = __ballot_sync(0xffffffffU, valid && bit);
        if (bit) lower += __popc(same & ~set);
        same &= bit ? set : ~set;
    }
    const int slot   = lower + __popc(same & below);
    const bool first = valid && (same & below) == 0;
    const int count  = __popc(same);
    const int width  = a.job_columns;
    const int jobs   = first ? (count + width - 1) / width : 0;
    jobs_at[lane]    = 0;
    __syncwarp();
    if (first) jobs_at[slot] = jobs;
    __syncwarp();
    const int mine = jobs_at[lane];
    int scan       = mine;
    for (int o = 1; o < 32; o <<= 1) {
        const int s = __shfl_up_sync(0xffffffffU, scan, o);
        if (lane >= o) scan += s;
    }
    const int job_total = __shfl_sync(0xffffffffU, scan, 31);
    __syncwarp();
    jobs_at[lane] = scan - mine;
    __syncwarp();
    const int job_at = first ? jobs_at[slot] : 0;
    if (valid) {
        a.packed_token[slot] = lane / a.top_k;
        a.slot_of[lane]      = slot;
    }
    for (int j = 0; j < jobs; ++j) {
        a.job_expert[job_at + j] = e;
        a.job_begin[job_at + j]  = slot + j * width;
        a.job_count[job_at + j]  = min(width, count - j * width);
    }
    if (lane == 0) *a.job_total = job_total;
}

} // namespace ninferno::ops::detail::moe
