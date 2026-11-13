#pragma once

#include "ops/softmax_attention/dense/causal_cache/bf16/epilogue.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/softmax.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/split_policy.h"
#include <math_constants.h>

namespace ninferno::ops::detail {

// Volta's width-1 BF16 KV decode (#877). sm_70 has no BF16 MMA and none of the m16n8k16 shapes the
// grouped kernel issues: there each of its products is 550 instructions of FP32 FMAs and operand
// shuffles for 16 query rows, of which a decode fills G (GroupSize). It read the cache at about
// 20 % of DRAM on GV100, against 77 % for the INT8 key-major kernel on the same card.
//
// This kernel keeps the grouped kernel's partition, partials and merge, and computes:
//   S[key][row] on CUDA cores, one key a lane: the lane reads its key's 512-byte BF16 row,
//     widens it to FP32 exactly (a shift) and takes G FMAs a dimension against Q in shared memory
//     (FP32, from BF16 exactly). No product crosses lanes, so there is no shuffle reduction.
//   O^T[d][row] += V^T[d][key] . P^T[key][row] on Volta's native mma.m8n8k4 (f16 operands, FP32
//     accumulation): V is stored as FP16 and P is rounded to FP16 as in the grouped kernel.
// Each of the four warps owns every fourth 32-key step of the split with its own softmax state;
// the CTA merges the four states as the grouped kernel merges its KV warps.
inline constexpr int kBf16KvVoltaStepKeys = 32;
inline constexpr int kBf16KvVoltaWarps    = 4;

template <class G>
inline constexpr bool kBf16KvVoltaKeyMajor = G::kHeadDim == 256 && G::GroupSize <= 8 &&
                                             G::GroupSize % 2 == 0;

#if NINFERNO_HMMA884
// D(32x8 f32) += A(32x4 f16, column-major) * B(4x8 f16, row-major): a lane holds A rows
// 8 * (lane >> 2) + 4 * half .. + 3 at k = lane % 4 (a0, a1) and B[lane % 4][4 * (lane >> 4) .. + 3]
// (b0, b1) (PTX ISA, mma.m8n8k4 .col.row; int8_kv_key_major_kt_kernel's PV product).
__device__ __forceinline__ void bf16_volta_mma_col_row(float (&d)[8], unsigned a0, unsigned a1,
                                                       unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
#endif

template <class G, bool MultiBatch, bool Masked, class Input>
__launch_bounds__(128, 2) __global__
    void bf16_kv_key_major_volta_kernel(const __nv_bfloat16* q, Input input, const int* positions,
                                        typename Bf16KvCacheView<Input::writes_cache>::Key* cache_k,
                                        typename Bf16KvCacheView<Input::writes_cache>::Value* cache_v,
                                        const int* tables, const int* validity,
                                        const int* table_rows, int table_stride, float scale,
                                        Bf16KvPartition partition, CausalPartialView partial) {
#if NINFERNO_HMMA884
    static_assert(kBf16KvVoltaKeyMajor<G>);
    constexpr int D        = G::kHeadDim;
    constexpr int R        = G::GroupSize; // query rows of the one token
    constexpr int Warps    = kBf16KvVoltaWarps;
    constexpr int StepKeys = kBf16KvVoltaStepKeys;
    constexpr int width    = 1;
    static_assert(Warps * 32 == 128 && StepKeys == 32 && kPagedKVPageSize % StepKeys == 0);

    __shared__ __align__(16) float q_s[R * D];
    // P^T[key][row] of each warp's step in FP16; rows R..7 stay zero.
    __shared__ __align__(16) __half p_s[Warps * StepKeys * 8];
    // The warps' numerators [warp][row][d], their log2 maxima and sums.
    __shared__ __align__(16) float red_s[Warps * 8 * D];
    __shared__ float m_s[Warps * 8];
    __shared__ float l_s[Warps * 8];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int head = blockIdx.x, split = blockIdx.y, batch = MultiBatch ? blockIdx.z : 0;
    const int live = Masked ? validity[batch] : width;
    q += static_cast<std::int64_t>(batch) * width * D * G::QHeads;
    positions += batch * width;
    if constexpr (Input::writes_cache) {
        input.k += static_cast<std::int64_t>(batch) * width * D * G::KVHeads;
        input.v += static_cast<std::int64_t>(batch) * width * D * G::KVHeads;
    }
    partial.acc += static_cast<std::int64_t>(batch) * width * G::QHeads * D * partition.capacity;
    partial.maximum += static_cast<std::int64_t>(batch) * width * G::QHeads * partition.capacity;
    partial.sum += static_cast<std::int64_t>(batch) * width * G::QHeads * partition.capacity;
    if (live <= 0) return; // the merge writes exact zero for a masked column
    const int first = positions[0], window = positions[live - 1] + 1;
    const auto work = partition.live(window);
    if (split >= work.splits) return;
    const int start     = split * work.keys_per_split;
    const int stop      = min(window, start + work.keys_per_split);
    const int table_row = table_rows ? table_rows[batch] : 0;
    const int* table    = tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (Input::writes_cache) {
        // The split that holds the new key appends it; every split reads it from the input.
        for (int i = tid; i < live * (D / 8); i += 128) {
            const int token = i / (D / 8), d = i % (D / 8) * 8, key = positions[token];
            if (key >= start && key < stop) {
                const auto src = causal_new_index<G>(head, d, token);
                const auto dst = bf16_kv_cache_index<G>(table[key >> kPagedKVPageShift], head, d,
                                                        key & kPagedKVPageMask);
                store_vec(cache_k + dst, load_vec<int4>(input.k + src));
                store_vec(cache_v + dst, bf16x8_bits_to_f16x8_bits(load_vec<int4>(input.v + src)));
            }
        }
    }
    const int end = min(stop, positions[0] + 1);
    if (start >= end) {
        for (int i = tid; i < R * (D / 2); i += 128)
            bf16_kv_store_pair<G, true>(partial, nullptr, head * R + i / (D / 2), 0,
                                        i % (D / 2) * 2, width, split, 0.0f, 0.0f, -CUDART_INF_F,
                                        0.0f);
        return;
    }

    // Q in FP32 (BF16 widened exactly) and the P tiles' zero rows.
    for (int i = tid; i < R * (D / 8); i += 128) {
        const int g = i / (D / 8), d = i % (D / 8) * 8;
        const uint4 v = load_vec<uint4>(q + causal_q_index<G>(head * R + g, d, 0));
        float* dst    = q_s + g * D + d;
        *reinterpret_cast<float4*>(dst) =
            make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                        __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));
        *reinterpret_cast<float4*>(dst + 4) =
            make_float4(__uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                        __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u));
    }
    for (int i = tid; i < Warps * StepKeys * 8 / 2; i += 128)
        reinterpret_cast<unsigned*>(p_s)[i] = 0u;
    __syncthreads();

    Bf16KvSoftmaxRow state[R];
    float acc[D / 64][2][8];
#    pragma unroll
    for (int db = 0; db < D / 64; ++db)
#    pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#    pragma unroll
            for (int e = 0; e < 8; ++e) acc[db][mt][e] = 0.0f;
    const float scale_log2 = scale * kLog2E;
    __half* p_tile         = p_s + warp * StepKeys * 8;
    const int steps        = div_up(end - start, StepKeys);

#    pragma unroll 1
    for (int step = warp; step < steps; step += Warps) {
        const int k0   = start + step * StepKeys;
        const int page = table[k0 >> kPagedKVPageShift];

        // S for this lane's key: D / 8 chunks of 8 dimensions, two FMA chains a row.
        const int key       = k0 + lane;
        const bool key_live = key < end;
        const __nv_bfloat16* k_row =
            cache_k + bf16_kv_cache_index<G>(page, head, 0, key & kPagedKVPageMask);
        if constexpr (Input::writes_cache) {
            if (key >= first) k_row = input.k + causal_new_index<G>(head, 0, key - first);
        }
        float s[R][2];
#    pragma unroll
        for (int g = 0; g < R; ++g) s[g][0] = s[g][1] = 0.0f;
        uint4 kb[2][8];
        const auto load_k = [&](uint4 (&dst)[8], int chunk) {
#    pragma unroll
            for (int j = 0; j < 8; ++j)
                dst[j] = key_live ? __ldg(reinterpret_cast<const uint4*>(k_row + chunk * 64 + j * 8))
                                  : make_uint4(0u, 0u, 0u, 0u);
        };
        load_k(kb[0], 0);
#    pragma unroll
        for (int chunk = 0; chunk < D / 64; ++chunk) {
            if (chunk + 1 < D / 64) load_k(kb[(chunk + 1) & 1], chunk + 1);
#    pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint4 w = kb[chunk & 1][j];
                const float kf[8] = {__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xffff0000u),
                                     __uint_as_float(w.y << 16), __uint_as_float(w.y & 0xffff0000u),
                                     __uint_as_float(w.z << 16), __uint_as_float(w.z & 0xffff0000u),
                                     __uint_as_float(w.w << 16), __uint_as_float(w.w & 0xffff0000u)};
                const int d       = chunk * 64 + j * 8;
#    pragma unroll
                for (int g = 0; g < R; ++g) {
                    const float4 qa = *reinterpret_cast<const float4*>(q_s + g * D + d);
                    const float4 qb = *reinterpret_cast<const float4*>(q_s + g * D + d + 4);
                    float& c        = s[g][j & 1];
                    c               = __fmaf_rn(qa.x, kf[0], c);
                    c               = __fmaf_rn(qa.y, kf[1], c);
                    c               = __fmaf_rn(qa.z, kf[2], c);
                    c               = __fmaf_rn(qa.w, kf[3], c);
                    c               = __fmaf_rn(qb.x, kf[4], c);
                    c               = __fmaf_rn(qb.y, kf[5], c);
                    c               = __fmaf_rn(qb.z, kf[6], c);
                    c               = __fmaf_rn(qb.w, kf[7], c);
                }
            }
        }

        // Softmax per row over the step's 32 keys (the state is the same in every lane; a lane
        // sums its own key's P and the sums meet once at the end).
        float alpha[R];
        bool rescale = false;
        __syncwarp(); // the previous step's P reads
#    pragma unroll
        for (int g = 0; g < R; g += 2) {
            float p[2];
#    pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float score = key_live ? s[g + h][0] + s[g + h][1] : -CUDART_INF_F;
                alpha[g + h]      = state[g + h].update(warp_max(score), scale_log2);
                p[h]              = state[g + h].probability(score, scale_log2);
                state[g + h].accumulate(alpha[g + h], p[h]);
                rescale |= alpha[g + h] != 1.0f;
            }
            *reinterpret_cast<__half2*>(&p_tile[lane * 8 + g]) = __floats2half2_rn(p[0], p[1]);
        }
        __syncwarp();
        if (rescale) {
            // A lane's accumulator e is row (lane & 2) + (e & 5).
            float row_alpha[8];
#    pragma unroll
            for (int r = 0; r < 8; ++r) row_alpha[r] = r < R ? alpha[r < R ? r : 0] : 1.0f;
            const bool high = (lane & 2) != 0;
#    pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float a = high ? row_alpha[2 + (e & 5)] : row_alpha[e & 5];
#    pragma unroll
                for (int db = 0; db < D / 64; ++db) {
                    acc[db][0][e] *= a;
                    acc[db][1][e] *= a;
                }
            }
        }

        // O^T += V^T P^T: per 64 dimensions, a lane's A is 8 FP16 dimensions of key
        // k0 + 4u + lane % 4, its B that key's P row.
#    pragma unroll
        for (int db = 0; db < D / 64; ++db) {
            const int d = db * 64 + 8 * (lane >> 2);
            uint4 vb[8];
#    pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int vkey = k0 + 4 * u + (lane & 3);
                vb[u] = make_uint4(0u, 0u, 0u, 0u);
                if (vkey >= end) continue;
                if constexpr (Input::writes_cache) {
                    if (vkey >= first) {
                        const int4 h = bf16x8_bits_to_f16x8_bits(
                            load_vec<int4>(input.v + causal_new_index<G>(head, d, vkey - first)));
                        vb[u] = *reinterpret_cast<const uint4*>(&h);
                        continue;
                    }
                }
                vb[u] = __ldg(reinterpret_cast<const uint4*>(
                    cache_v + bf16_kv_cache_index<G>(page, head, d, vkey & kPagedKVPageMask)));
            }
#    pragma unroll
            for (int u = 0; u < 8; ++u) {
                const uint2 b = *reinterpret_cast<const uint2*>(
                    &p_tile[(4 * u + (lane & 3)) * 8 + 4 * (lane >> 4)]);
                bf16_volta_mma_col_row(acc[db][0], vb[u].x, vb[u].y, b.x, b.y);
                bf16_volta_mma_col_row(acc[db][1], vb[u].z, vb[u].w, b.x, b.y);
            }
        }
    }

    // The warp's state: log2 maxima and lane sums met, numerators to shared memory.
#    pragma unroll
    for (int g = 0; g < 8; ++g) {
        if (g >= R) {
            if (lane == 0) {
                m_s[warp * 8 + g] = -CUDART_INF_F;
                l_s[warp * 8 + g] = 0.0f;
            }
            continue;
        }
        const float sum = warp_sum(state[g].sum);
        if (lane == 0) {
            m_s[warp * 8 + g] = state[g].maximum * scale_log2;
            l_s[warp * 8 + g] = sum;
        }
    }
    float* red = red_s + warp * 8 * D;
#    pragma unroll
    for (int db = 0; db < D / 64; ++db)
#    pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#    pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = (lane & 2) + (e & 5);
                const int m   = (e & 2) + (lane & ~2);
                red[row * D + db * 64 + 8 * (m >> 2) + 4 * mt + (m & 3)] = acc[db][mt][e];
            }
    __syncthreads();
    for (int i = tid; i < R * (D / 2); i += 128) {
        const int g = i / (D / 2), d = i % (D / 2) * 2;
        float maximum = -CUDART_INF_F;
#    pragma unroll
        for (int w = 0; w < Warps; ++w) maximum = fmaxf(maximum, m_s[w * 8 + g]);
        float a = 0.0f, b = 0.0f, sum = 0.0f;
#    pragma unroll
        for (int w = 0; w < Warps; ++w) {
            const float weight = bf16_kv_state_weight(m_s[w * 8 + g], l_s[w * 8 + g], maximum);
            const float2 x = *reinterpret_cast<const float2*>(&red_s[(w * 8 + g) * D + d]);
            sum += weight * l_s[w * 8 + g];
            a += weight * x.x;
            b += weight * x.y;
        }
        bf16_kv_store_pair<G, true>(partial, nullptr, head * R + g, 0, d, width, split, a, b,
                                    maximum, sum);
    }
#else
    __trap();
#endif
}

} // namespace ninferno::ops::detail
