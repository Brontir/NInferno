#pragma once
#include "ops/softmax_attention/dense/causal_cache/int8/key_major_hmma.cuh"

namespace ninferno::ops::detail {

// Volta's INT8 key-major kernel with the products transposed (after Flo5k5's
// small_t_i8_volta_kt): keys on the 32-wide M axis of mma.m8n8k4 for QK^T and head dimensions on it
// for PV, query rows on the 8-wide N axis. int8_kv_key_major_hmma_kernel put the rows on M, re-read
// Q from shared memory per k4 step, staged the per-key scales behind two barriers per 32 keys and
// ran the softmax row by row (a warp max, a shuffle sum, two exponentials and a shared-memory round
// trip per row and tile); at 24K keys it streamed the cache at 247 GB/s (27B MTP 4 verify, V100,
// 1092 us per token) against the transposed kernel's 489 (545 us).
//
//   S^T[key][row]  = K[key][d] . Q[row][d]      mma.m8n8k4.row.col, A = K codes, B = Q codes
//   O^T[d][row]   += V^T[d][key] . P^T[key][row] mma.m8n8k4.col.row, A = V codes, B = P
//
// Warp g owns the 64 dimensions of INT8 group g: its K A-fragment is its lane's key, 64 bytes of the
// group straight from global memory, its V A-fragment four dimensions of one key per lane, both
// prefetched one step ahead in registers. Q keeps the grouped kernel's codec (normalized H256, INT8
// per group, codes as exact FP16 integers), and a group's score term is (q_scale * k_scale) * dot,
// summed over the four groups in order: the scores are the grouped kernel's, bit for bit. The
// softmax keeps a stale reference maximum per row (raised only when a score passes it by
// kKtHeadroom in log2 units, so P stays below 2^kKtHeadroom in FP16) and sums l per lane, reduced
// once at the end; it writes the natural-log m the merge reads. Every decision is per row, so a
// query row's partials do not depend on the call's width. Up to 32 rows (four N tiles) per CTA:
// widths of at most 32 / GroupSize take it, wider ones the row-major kernel.
inline constexpr int kKtStepKeys   = 32;
inline constexpr float kKtHeadroom = 8.0F;

template <class Geometry, int Tokens>
inline constexpr bool kInt8KvKeyMajorKt = Tokens * Geometry::GroupSize <= 32;

#if defined(NINFERNO_HMMA884) && NINFERNO_HMMA884
// D(32x8 f32) += A(32x4 f16, column-major) * B(4x8 f16, row-major).
__device__ __forceinline__ void kt_mma_col_row(float (&d)[8], unsigned a0, unsigned a1, unsigned b0,
                                               unsigned b1) {
    asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// D(32x8 f32) += A(32x4 f16, row-major) * B(4x8 f16, column-major): lane L holds A[L][0..3] and
// B[0..3][4 * (L >> 4) + (L & 3)].
__device__ __forceinline__ void kt_mma_row_col(float (&d)[8], unsigned a0, unsigned a1, unsigned b0,
                                               unsigned b1) {
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float kt_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
#endif

template <class Geometry, int Tokens, bool MultiBatch, bool Masked>
__launch_bounds__(128, 2) __global__
    void int8_kv_key_major_kt_kernel(const __nv_bfloat16* q, const std::int32_t* pos,
                                     const std::int8_t* cache_k_i8, const std::int8_t* cache_v_i8,
                                     const __half* cache_k_scale, const __half* cache_v_scale,
                                     const std::int32_t* block_tables,
                                     const std::int32_t* valid_columns,
                                     const std::int32_t* table_rows, std::int32_t table_stride,
                                     std::int32_t full_width, std::int32_t logical_capacity,
                                     CausalKvPartition partition, float scale, float* partial_acc,
                                     float* partial_m, float* partial_l) {
#if defined(NINFERNO_HMMA884) && NINFERNO_HMMA884
    static_assert(kInt8KvKeyMajorKt<Geometry, Tokens>);
    constexpr int RowCount      = Tokens * Geometry::GroupSize;
    constexpr int NT            = (RowCount + 7) / 8;
    constexpr int Rows          = NT * 8;
    constexpr int Warps         = 4;
    constexpr int Threads       = Warps * 32;
    constexpr int D             = 256;
    constexpr int Groups        = kKVCacheInt8Groups;
    constexpr int StepKeys      = kKtStepKeys;
    constexpr bool Redundant    = NT == 1; // every warp sums the partials and runs the softmax
    constexpr int XBuffers      = Redundant ? 2 : 1;
    constexpr int QStride       = D + 8;   // 16-byte pad: the 8 fragment rows hit distinct banks
    constexpr int StageStride   = 64 + 4;
    constexpr int XFloats       = XBuffers * Warps * NT * 256 > Warps * 8 * StageStride
                                      ? XBuffers * Warps * NT * 256
                                      : Warps * 8 * StageStride;
    constexpr float Log2E       = kLog2E;
    constexpr unsigned FullMask = 0xffffffffu;
    static_assert(Groups == Warps && kKVCacheInt8Group == 64);
    static_assert(StepKeys * 2 == kPagedKVPageSize, "a step must sit inside one KV page");

    __shared__ __align__(16) __half q_s[Rows * QStride];
    __shared__ float q_scale_s[Rows * Groups];
    // Partial score tiles [buffer][source group][N tile][half][lane][4].
    __shared__ __align__(16) float x_s[XFloats];
    // P^T tiles [N tile][key][row] in FP16; one per warp when every warp computes its own.
    __shared__ __align__(16) __half p_s[(Redundant ? Warps : 1) * NT * StepKeys * 8];
    __shared__ float alpha_s[Redundant ? 1 : NT * 8];
    __shared__ int flag_s[Redundant ? 1 : NT];

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;

    int valid_tokens = Tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch];
        valid_tokens        = remaining <= 0 ? 0 : min(remaining, Tokens);
    }
    std::int64_t column_base = 0;
    if constexpr (MultiBatch) { column_base = static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(256) * Geometry::QHeads * column_base;
    const int last_pos = pos[column_base + full_width - 1];
    const int live_last_pos =
        Masked ? pos[column_base + max(valid_columns[batch], 1) - 1] : last_pos;
    pos += column_base;
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc +=
            static_cast<std::int64_t>(batch) * 256 * Geometry::QHeads * full_width * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * full_width * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * full_width * split_count;
    }

    if (valid_tokens == 0) return; // Merge writes exact zero for masked columns.
    if (pos[0] < 0 || last_pos < 0 || last_pos >= logical_capacity) return;
    const int window             = last_pos + 1;
    const int active_split_count = partition.active(window);
    if (split >= active_split_count) return;
    // The row-major kernel's split boundaries (32-key tiles), so the merge sees the same partials.
    const int logical_tiles    = div_up(window, StepKeys);
    const int first_owned_tile = split * logical_tiles / active_split_count;
    const int end_owned_tile   = (split + 1) * logical_tiles / active_split_count;
    const int split_start      = first_owned_tile * StepKeys;
    const int split_end        = min(end_owned_tile * StepKeys, window);
    int key_end                = split_end;
    if constexpr (Masked) { key_end = min(split_end, live_last_pos + 1); }
    const int steps = key_end > split_start ? div_up(key_end - split_start, StepKeys) : 0;

    // Q: the grouped kernel's codec, as the row-major kernel stores it.
    for (int row = warp; row < Rows; row += Warps) {
        int q_head = 0, token = 0;
        causal_row_to_qt<Geometry>(row, kv_head, q_head, token);
        const bool live = row < RowCount && token < valid_tokens;
        float q_values[8];
#    pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            q_values[r] =
                live ? __bfloat162float(q[causal_q_index<Geometry>(q_head, d, token)]) : 0.0f;
        }
        normalized_hadamard_d256_inplace(q_values, lane);
#    pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            const int d0                 = grp * kKVCacheInt8Group + lane;
            const float x0               = q_values[2 * grp];
            const float x1               = q_values[2 * grp + 1];
            const float amax             = warp_max(fmaxf(fabsf(x0), fabsf(x1)), FullMask);
            const float qs               = amax > 0.0f ? amax / 127.0f : 0.0f;
            const float inv              = qs > 0.0f ? 1.0f / qs : 0.0f;
            q_s[row * QStride + d0]      = __int2half_rn(kv_cache_int8_quant_code(x0, inv));
            q_s[row * QStride + d0 + 32] = __int2half_rn(kv_cache_int8_quant_code(x1, inv));
            if (lane == 0) { q_scale_s[row * Groups + grp] = live ? qs : 0.0f; }
        }
    }

    const int slice      = warp; // this warp's group: QK partial and PV dimensions
    const int slice0     = slice * 64;
    const int q_frag_n   = (lane & 3) + ((lane >> 4) << 2); // B-fragment column (row) of the lane
    const float qk_scale = scale * Log2E;
    // A lane's D values cover rows (lane & 2) + {0, 1, 4, 5} of an N tile: slot(e).
    const auto slot_of     = [](int e) { return (e & 1) | ((e >> 1) & 2); };
    const auto row_of_slot = [&](int nt, int s) { return nt * 8 + (lane & 2) + (s & 1) + ((s & 2) << 1); };

    // Codes of one step (zero past the split's live keys, so a masked key's P of zero meets finite
    // V), prefetched one step ahead.
    uint4 k_raw[4], k_next[4];
    uint2 v_raw[8], v_next[8];
    __half k_scale_lane = __ushort_as_half(0), k_scale_next = __ushort_as_half(0);
    __half v_scale_lane = __ushort_as_half(0), v_scale_next = __ushort_as_half(0);
    const auto load_step = [&](int step, uint4 (&kd)[4], uint2 (&vd)[8], __half& ks, __half& vs) {
        const int k0            = split_start + step * StepKeys;
        const int physical_page = __ldg(block_table + (k0 >> kPagedKVPageShift));
        const bool whole        = k0 + StepKeys <= key_end;
        {
            const int key = k0 + lane;
            if (whole || key < key_end) {
                const auto* src = reinterpret_cast<const uint4*>(
                    &cache_k_i8[kv_cache_int8_quant_code_index<Geometry>(
                        physical_page, kv_head, slice0, key & kPagedKVPageMask)]);
#    pragma unroll
                for (int j = 0; j < 4; ++j) kd[j] = __ldg(src + j);
                const std::int64_t so = kv_cache_int8_quant_scale_index<Geometry>(
                    physical_page, kv_head, slice, key & kPagedKVPageMask);
                ks = cache_k_scale[so];
                vs = cache_v_scale[so];
            } else {
#    pragma unroll
                for (int j = 0; j < 4; ++j) kd[j] = make_uint4(0, 0, 0, 0);
                ks = __ushort_as_half(0);
                vs = __ushort_as_half(0);
            }
        }
#    pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int key = k0 + 4 * u + (lane & 3);
            vd[u]         = whole || key < key_end
                                ? __ldg(reinterpret_cast<const uint2*>(
                                      &cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(
                                          physical_page, kv_head, slice0 + 8 * (lane >> 2),
                                          key & kPagedKVPageMask)]))
                                : make_uint2(0u, 0u);
        }
    };
    if (steps > 0) load_step(0, k_raw, v_raw, k_scale_lane, v_scale_lane);
    __syncthreads(); // Q codes and scales

    // Q B-fragments of this warp's group: resident in registers for one N tile, read from shared
    // memory per step for more (four tiles' 128 registers would spill), and the per-row Q scales
    // its score terms take.
    const auto load_q_frag = [&](int nt, int pair) {
        return *reinterpret_cast<const uint4*>(&q_s[(nt * 8 + q_frag_n) * QStride + slice0 + pair * 8]);
    };
    uint4 q_resident[Redundant ? 8 : 1];
    if constexpr (Redundant) {
#    pragma unroll
        for (int pair = 0; pair < 8; ++pair) q_resident[pair] = load_q_frag(0, pair);
    }
    float q_scale_slot[NT][4];
#    pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#    pragma unroll
        for (int s = 0; s < 4; ++s) q_scale_slot[nt][s] = q_scale_s[row_of_slot(nt, s) * Groups + slice];

    // The N tile this warp's softmax handles (every warp the one tile when Redundant) and its rows'
    // query positions (-1: a row past the call's).
    const int own_nt = Redundant ? 0 : warp;
    const bool owns  = own_nt < NT;
    int row_pos[4];
#    pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int row = row_of_slot(own_nt, s);
        row_pos[s]    = owns && row < Tokens * Geometry::GroupSize ? pos[row / Geometry::GroupSize] : -1;
    }

    float acc[2][NT][8];
#    pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#    pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#    pragma unroll
            for (int e = 0; e < 8; ++e) acc[mt][nt][e] = 0.0f;
    float m_ref[4], m_sub[4], m_thr[4], l_run[4];
#    pragma unroll
    for (int s = 0; s < 4; ++s) {
        m_ref[s] = -CUDART_INF_F;
        m_sub[s] = 0.0f;          // m_ref, or 0 while the row has no finite score
        m_thr[s] = -CUDART_INF_F; // headroom, or -inf while the row has no finite score
        l_run[s] = 0.0f;
    }

#    pragma unroll 1
    for (int step = 0; step < steps; ++step) {
        const int k0        = split_start + step * StepKeys;
        const bool has_next = step + 1 < steps;
        float* x_buf        = &x_s[(Redundant ? (step & 1) : 0) * Warps * NT * 256];
        if (has_next) load_step(step + 1, k_next, v_next, k_scale_next, v_scale_next);

        // S^T partial of this warp's group: A = K codes (M = key), B = Q codes (N = row); one
        // accumulator per N tile at every width.
        {
            float s_part[NT][8];
#    pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#    pragma unroll
                for (int e = 0; e < 8; ++e) s_part[nt][e] = 0.0f;
#    pragma unroll
            for (int pair = 0; pair < 8; ++pair) {
                const unsigned w0 = (pair & 1) == 0 ? k_raw[pair >> 1].x : k_raw[pair >> 1].z;
                const unsigned w1 = (pair & 1) == 0 ? k_raw[pair >> 1].y : k_raw[pair >> 1].w;
                unsigned a[4];
                int8x4_to_f16x2x2(w0, a[0], a[1]);
                int8x4_to_f16x2x2(w1, a[2], a[3]);
                uint4 qf[NT];
#    pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    if constexpr (Redundant) {
                        qf[nt] = q_resident[pair];
                    } else {
                        qf[nt] = load_q_frag(nt, pair);
                    }
                }
                // Consecutive mma target different accumulators (N tiles).
#    pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                    kt_mma_row_col(s_part[nt], a[0], a[1], qf[nt].x, qf[nt].y);
#    pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                    kt_mma_row_col(s_part[nt], a[2], a[3], qf[nt].z, qf[nt].w);
            }
            // The D rows (keys) this lane holds: (lane & ~2) + {0, 2}.
            const float ks_own = __half2float(k_scale_lane);
            const float ks_lo  = __shfl_sync(FullMask, ks_own, lane & ~2);
            const float ks_hi  = __shfl_sync(FullMask, ks_own, (lane & ~2) + 2);
#    pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                float v[8];
#    pragma unroll
                for (int e = 0; e < 8; ++e)
                    // (q_scale * k_scale) * dot, the grouped kernel's order.
                    v[e] = q_scale_slot[nt][slot_of(e)] * ((e & 2) == 0 ? ks_lo : ks_hi) * s_part[nt][e];
                float* dst = &x_buf[((slice * NT + nt) * 2) * 128 + lane * 4];
                *reinterpret_cast<float4*>(dst)       = make_float4(v[0], v[1], v[2], v[3]);
                *reinterpret_cast<float4*>(dst + 128) = make_float4(v[4], v[5], v[6], v[7]);
            }
        }
        __syncthreads();

        // Softmax over the summed scores of this warp's N tile.
        float alpha_own[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        bool rescaled      = false;
        if (owns) {
            const int nt = own_nt;
            float x[8];
            {
                float4 lo = *reinterpret_cast<const float4*>(&x_buf[(nt * 2) * 128 + lane * 4]);
                float4 hi = *reinterpret_cast<const float4*>(&x_buf[(nt * 2 + 1) * 128 + lane * 4]);
#    pragma unroll
                for (int src = 1; src < Warps; ++src) {
                    const float4 plo =
                        *reinterpret_cast<const float4*>(&x_buf[((src * NT + nt) * 2) * 128 + lane * 4]);
                    const float4 phi = *reinterpret_cast<const float4*>(
                        &x_buf[((src * NT + nt) * 2 + 1) * 128 + lane * 4]);
                    lo.x += plo.x;
                    lo.y += plo.y;
                    lo.z += plo.z;
                    lo.w += plo.w;
                    hi.x += phi.x;
                    hi.y += phi.y;
                    hi.z += phi.z;
                    hi.w += phi.w;
                }
                x[0] = lo.x;
                x[1] = lo.y;
                x[2] = lo.z;
                x[3] = lo.w;
                x[4] = hi.x;
                x[5] = hi.y;
                x[6] = hi.z;
                x[7] = hi.w;
            }
#    pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int key = k0 + (e & 2) + (lane & ~2);
                const bool ok = key < key_end && key <= row_pos[slot_of(e)];
                x[e]          = ok ? x[e] * qk_scale : -CUDART_INF_F;
            }
            // Stale reference, decided per row (lanes with equal lane & 2 hold the same rows).
            bool over_slot[4] = {false, false, false, false};
#    pragma unroll
            for (int e = 0; e < 8; ++e) over_slot[slot_of(e)] |= x[e] - m_sub[slot_of(e)] > m_thr[slot_of(e)];
            const unsigned same_rows = (lane & 2) != 0 ? 0xccccccccu : 0x33333333u;
            bool raise[4];
            bool any_raise = false;
#    pragma unroll
            for (int s = 0; s < 4; ++s) {
                raise[s] = (__ballot_sync(FullMask, over_slot[s]) & same_rows) != 0u;
                any_raise |= raise[s];
            }
            if (__any_sync(FullMask, any_raise)) {
#    pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const int l0   = (s & 1) | ((s & 2) << 1);
                    float tile_max = fmaxf(x[l0], x[l0 | 2]);
                    tile_max       = fmaxf(tile_max, __shfl_xor_sync(FullMask, tile_max, 1));
                    tile_max       = fmaxf(tile_max, __shfl_xor_sync(FullMask, tile_max, 4));
                    tile_max       = fmaxf(tile_max, __shfl_xor_sync(FullMask, tile_max, 8));
                    tile_max       = fmaxf(tile_max, __shfl_xor_sync(FullMask, tile_max, 16));
                    if (!raise[s]) continue;
                    const float new_m = fmaxf(m_ref[s], tile_max);
                    const float alpha = new_m == m_ref[s] ? 1.0f : kt_exp2(m_ref[s] - new_m);
                    alpha_own[s]      = alpha;
                    l_run[s] *= alpha;
                    m_ref[s] = new_m;
                    m_sub[s] = new_m == -CUDART_INF_F ? 0.0f : new_m;
                    m_thr[s] = new_m == -CUDART_INF_F ? -CUDART_INF_F : kKtHeadroom;
                }
                rescaled = true;
            }
            float p[8];
#    pragma unroll
            for (int e = 0; e < 8; ++e) {
                p[e] = kt_exp2(x[e] - m_sub[slot_of(e)]);
                l_run[slot_of(e)] += p[e];
            }
            // P^T[key][row]: rows j and j + 1 of one key are adjacent halves.
            __half* p_tile = &p_s[((Redundant ? warp * NT : 0) + nt) * StepKeys * 8];
            const int key0 = lane & ~2;
            const int row0 = lane & 2;
            if constexpr (Redundant) { __syncwarp(); }
            *reinterpret_cast<__half2*>(&p_tile[key0 * 8 + row0])           = __floats2half2_rn(p[0], p[1]);
            *reinterpret_cast<__half2*>(&p_tile[key0 * 8 + row0 + 4])       = __floats2half2_rn(p[4], p[5]);
            *reinterpret_cast<__half2*>(&p_tile[(key0 + 2) * 8 + row0])     = __floats2half2_rn(p[2], p[3]);
            *reinterpret_cast<__half2*>(&p_tile[(key0 + 2) * 8 + row0 + 4]) = __floats2half2_rn(p[6], p[7]);
            if constexpr (!Redundant) {
                if (rescaled && (lane & ~2) == 0) {
#    pragma unroll
                    for (int s = 0; s < 4; ++s) alpha_s[nt * 8 + row0 + (s & 1) + ((s & 2) << 1)] = alpha_own[s];
                }
                if (lane == 0) { flag_s[nt] = rescaled ? 1 : 0; }
            }
        }
        if constexpr (Redundant) {
            __syncwarp();
        } else {
            __syncthreads();
        }

        // Rescale, then O^T += V^T P^T over this warp's group and every N tile.
#    pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            if constexpr (Redundant) {
                if (rescaled) {
#    pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
#    pragma unroll
                        for (int e = 0; e < 8; ++e) acc[mt][nt][e] *= alpha_own[slot_of(e)];
                }
            } else if (flag_s[nt] != 0) {
                float alpha[4];
#    pragma unroll
                for (int s = 0; s < 4; ++s) alpha[s] = alpha_s[nt * 8 + (lane & 2) + (s & 1) + ((s & 2) << 1)];
#    pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#    pragma unroll
                    for (int e = 0; e < 8; ++e) acc[mt][nt][e] *= alpha[slot_of(e)];
            }
        }
        const __half* p_base = &p_s[(Redundant ? warp * NT : 0) * StepKeys * 8];
        const __half2 vs_own = __half2half2(v_scale_lane);
        const unsigned vs_w  = *reinterpret_cast<const unsigned*>(&vs_own);
#    pragma unroll
        for (int u = 0; u < 8; ++u) {
            const unsigned vs_bits = __shfl_sync(FullMask, vs_w, 4 * u + (lane & 3));
            const __half2 vs       = *reinterpret_cast<const __half2*>(&vs_bits);
            unsigned a0[2], a1[2];
            int8x4_to_f16x2x2(v_raw[u].x, a0[0], a0[1]);
            int8x4_to_f16x2x2(v_raw[u].y, a1[0], a1[1]);
#    pragma unroll
            for (int i = 0; i < 2; ++i) {
                __half2 h0 = __hmul2(*reinterpret_cast<const __half2*>(&a0[i]), vs);
                __half2 h1 = __hmul2(*reinterpret_cast<const __half2*>(&a1[i]), vs);
                a0[i]      = *reinterpret_cast<const unsigned*>(&h0);
                a1[i]      = *reinterpret_cast<const unsigned*>(&h1);
            }
#    pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const uint2 b = *reinterpret_cast<const uint2*>(
                    &p_base[nt * StepKeys * 8 + (4 * u + (lane & 3)) * 8 + 4 * (lane >> 4)]);
                kt_mma_col_row(acc[0][nt], a0[0], a0[1], b.x, b.y);
                kt_mma_col_row(acc[1][nt], a1[0], a1[1], b.x, b.y);
            }
        }
#    pragma unroll
        for (int j = 0; j < 4; ++j) k_raw[j] = k_next[j];
#    pragma unroll
        for (int u = 0; u < 8; ++u) v_raw[u] = v_next[u];
        k_scale_lane = k_scale_next;
        v_scale_lane = v_scale_next;
    }

    // Row statistics: reduce the per-lane sums over the key lanes; m in natural log.
    if (owns) {
#    pragma unroll
        for (int s = 0; s < 4; ++s) {
            float l = l_run[s];
            l += __shfl_xor_sync(FullMask, l, 1);
            l += __shfl_xor_sync(FullMask, l, 4);
            l += __shfl_xor_sync(FullMask, l, 8);
            l += __shfl_xor_sync(FullMask, l, 16);
            l_run[s] = l;
        }
        if (!(Redundant && warp != 0) && (lane & ~2) == 0) {
#    pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int row = row_of_slot(own_nt, s);
                if (row >= Tokens * Geometry::GroupSize) continue;
                int q_head = 0, token = 0;
                causal_row_to_qt<Geometry>(row, kv_head, q_head, token);
                const std::int64_t si = causal_stat_index<Geometry>(q_head, token, split, full_width);
                partial_m[si]         = m_ref[s] == -CUDART_INF_F ? -CUDART_INF_F : m_ref[s] / Log2E;
                partial_l[si]         = l_run[s];
            }
        }
    }
    // O^T leaves through shared memory as [row][dim] rows of 64 floats per warp.
    __syncthreads();
    float* stage = &x_s[warp * 8 * StageStride];
#    pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        __syncwarp();
#    pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int row = (lane & 2) + (e & 5);
            const int m   = (e & 2) + (lane & ~2);
#    pragma unroll
            for (int mt = 0; mt < 2; ++mt) stage[row * StageStride + 8 * (m >> 2) + 4 * mt + (m & 3)] = acc[mt][nt][e];
        }
        __syncwarp();
#    pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int local_row = 2 * r + (lane >> 4);
            const int row       = nt * 8 + local_row;
            if (row >= Tokens * Geometry::GroupSize) continue;
            int q_head = 0, token = 0;
            causal_row_to_qt<Geometry>(row, kv_head, q_head, token);
            const int d = 4 * (lane & 15);
            *reinterpret_cast<float4*>(&partial_acc[causal_partial_index<Geometry>(
                q_head, slice0 + d, token, split, full_width)]) =
                *reinterpret_cast<const float4*>(&stage[local_row * StageStride + d]);
        }
    }
#else
    __trap();
#endif
}

} // namespace ninferno::ops::detail
