#pragma once
#include "ops/softmax_attention/dense/causal_cache/int8/tile_io.cuh"

#include "ops/common/arch_limits.h"
#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/common/causal_epilogue.cuh"
#include "ops/softmax_attention/common/causal_softmax.cuh"

namespace ninferno::ops::detail {

// Volta has no mma.sync m16n8k16 / m16n8k32: the grouped kernel's QK and PV tiles run in FP32 FMAs
// over shuffles there (ops/common/mma.cuh) on one producer warp per row tile, and a decode step at
// depth waits on them. This kernel computes the same partials on Volta's own mma.m8n8k4.
//
// QK, S = Q K^T: warp g owns the 64-dimension group g of the INT8 codec. Its four quad-pairs take
// 8 keys each (32 per tile) against the same 8 query rows; K codes come from global memory straight
// into the col-major B fragment (a lane's key, four consecutive dimensions) and Q codes from shared
// memory, both as exact FP16 integers. The FP32 accumulator then holds the group's integer dot
// product exactly (|sum| < 2^21), as the INT32 MMA does, so a group's term (q_scale * k_scale) *
// dot and the in-order sum over the four groups reproduce the grouped kernel's scores bit for bit.
// PV, O = P V: warp g owns output dimensions [64g, 64g + 64), which share the V scale of group g;
// the row-major B fragment is four consecutive V codes of one key, scaled in FP16 as the grouped
// kernel dequantizes them.
#if defined(NINFERNO_BUILD_ARCH) && NINFERNO_BUILD_ARCH >= 70 && NINFERNO_BUILD_ARCH < 75
inline constexpr bool kInt8KvKeyMajorHmma = true;
#else
inline constexpr bool kInt8KvKeyMajorHmma = false;
#endif

template <class Geometry, int Tokens>
struct Int8KvKeyMajorShape {
    static_assert(Tokens > 0 && Tokens * Geometry::GroupSize <= 64);
    static constexpr int kTokenTile   = Tokens;
    static constexpr int kRows        = Tokens * Geometry::GroupSize;
    static constexpr int kRowTiles    = (kRows + 7) / 8;
    static constexpr int kPaddedRows  = kRowTiles * 8;
    static constexpr int kKeys        = 32;
    static constexpr int kThreads     = 128;
    static constexpr int kQStride     = 256 + 4; // halves; 130 words: 8 rows on distinct banks
    static constexpr int kPStride     = kKeys + 4;
    static constexpr int kQBytes      = kPaddedRows * kQStride * 2;
    static constexpr int kQScaleBytes = kPaddedRows * 4 * 4;
    // Score partials [group][row][key], rows kKeys + 1 floats apart: a QK store's lanes span rows
    // r, r+1, r+4, r+5 (32 and 128 floats apart at kKeys, one bank, 4-way conflicts), which the odd
    // stride spreads over 32 banks; a softmax read (one row, lane = key) stays conflict-free.
    static constexpr int kPartStride  = kKeys + 1;
    static constexpr int kPartBytes   = 4 * kPaddedRows * kPartStride * 4;
    static constexpr int kPBytes      = kPaddedRows * kPStride * 2;
    static constexpr int kRowBytes    = 3 * kPaddedRows * 4; // alpha, m, l
    static constexpr int kScaleBytes  = 2 * kKeys * 4 * 2;
    static constexpr int kSharedBytes =
        kQBytes + kQScaleBytes + kPartBytes + kPBytes + kRowBytes + kScaleBytes;
    static_assert(kSharedBytes <= 96 * 1024);
};

#if defined(NINFERNO_HMMA884) && NINFERNO_HMMA884
// Four signed INT8 codes to two f16x2 exactly: 0x6400 | (c ^ 0x80) is 1024 + c + 128.
__device__ __forceinline__ void int8x4_to_f16x2x2(unsigned codes, unsigned& lo, unsigned& hi) {
    const unsigned biased = codes ^ 0x80808080u;
    const unsigned l      = __byte_perm(biased, 0x64646464u, 0x4140);
    const unsigned h      = __byte_perm(biased, 0x64646464u, 0x4342);
    const __half2 bias    = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));
    const __half2 lv      = __hsub2(*reinterpret_cast<const __half2*>(&l), bias);
    const __half2 hv      = __hsub2(*reinterpret_cast<const __half2*>(&h), bias);
    lo                    = *reinterpret_cast<const unsigned*>(&lv);
    hi                    = *reinterpret_cast<const unsigned*>(&hv);
}

__device__ __forceinline__ void mma_m8n8k4_f16_rr(float (&c)[8], unsigned a0, unsigned a1,
                                                  unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]),
                   "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
#endif

template <class Geometry, int Tokens, bool MultiBatch, bool Masked, bool ParallelQueries>
__launch_bounds__(128) __global__
    void int8_kv_key_major_hmma_kernel(const __nv_bfloat16* q, const std::int32_t* pos,
                                       const std::int8_t* cache_k_i8, const std::int8_t* cache_v_i8,
                                       const __half* cache_k_scale, const __half* cache_v_scale,
                                       const std::int32_t* block_tables,
                                       const std::int32_t* valid_columns,
                                       const std::int32_t* table_rows, std::int32_t table_stride,
                                       std::int32_t full_width, std::int32_t logical_capacity,
                                       CausalKvPartition partition, float scale, float* partial_acc,
                                       float* partial_m, float* partial_l) {
#if defined(NINFERNO_HMMA884) && NINFERNO_HMMA884
    using S                     = Int8KvKeyMajorShape<Geometry, Tokens>;
    constexpr int TokenTile     = S::kTokenTile;
    constexpr int RowCount      = S::kRows;
    constexpr int RowTiles      = S::kRowTiles;
    constexpr int Rp            = S::kPaddedRows;
    constexpr int Bc            = S::kKeys;
    constexpr int PartS         = S::kPartStride;
    constexpr int Groups        = kKVCacheInt8Groups;
    constexpr int QS            = S::kQStride;
    constexpr int PS            = S::kPStride;
    constexpr float Log2E       = kLog2E;
    constexpr unsigned FullMask = 0xffffffffu;
    static_assert(Groups == 4 && kKVCacheInt8Group == 64);

    extern __shared__ __align__(16) unsigned char arena[];
    __half* q_h      = reinterpret_cast<__half*>(arena);
    float* q_scale_s = reinterpret_cast<float*>(arena + S::kQBytes);
    float* part_s    = reinterpret_cast<float*>(arena + S::kQBytes + S::kQScaleBytes);
    __half* p_s = reinterpret_cast<__half*>(arena + S::kQBytes + S::kQScaleBytes + S::kPartBytes);
    float* alpha_s =
        reinterpret_cast<float*>(arena + S::kQBytes + S::kQScaleBytes + S::kPartBytes + S::kPBytes);
    float* m_s        = alpha_s + Rp;
    float* l_s        = m_s + Rp;
    __half* k_scale_s = reinterpret_cast<__half*>(l_s + Rp);
    __half* v_scale_s = k_scale_s + Bc * Groups;

    const int kv_head      = ParallelQueries ? blockIdx.x % Geometry::KVHeads : blockIdx.x;
    const int column_begin = ParallelQueries ? (blockIdx.x / Geometry::KVHeads) * TokenTile : 0;
    const int tile_tokens = ParallelQueries ? min(TokenTile, full_width - column_begin) : TokenTile;
    const int partial_width = full_width;
    const int split         = static_cast<int>(blockIdx.y);
    const int batch         = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count   = static_cast<int>(gridDim.y);
    const int tid           = static_cast<int>(threadIdx.x);
    const int warp          = tid >> 5;
    const int lane          = tid & 31;

    int valid_tokens = tile_tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : min(remaining, tile_tokens);
    }
    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(256) * Geometry::QHeads * column_base;
    const int last_pos = pos[(MultiBatch ? batch * full_width : 0) + full_width - 1];
    const int live_last_pos =
        Masked ? pos[(MultiBatch ? batch * full_width : 0) + max(valid_columns[batch], 1) - 1]
               : last_pos;
    pos += column_base;
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc +=
            static_cast<std::int64_t>(batch) * 256 * Geometry::QHeads * partial_width * split_count;
        partial_m +=
            static_cast<std::int64_t>(batch) * Geometry::QHeads * partial_width * split_count;
        partial_l +=
            static_cast<std::int64_t>(batch) * Geometry::QHeads * partial_width * split_count;
    }

    if (valid_tokens == 0) return; // Merge writes exact zero for masked columns.
    if (pos[0] < 0 || last_pos < 0 || last_pos >= logical_capacity) return;
    const int window             = last_pos + 1;
    const int active_split_count = partition.active(window);
    if (split >= active_split_count) return;
    // Split boundaries in 32-key tiles. The merge takes whatever key ranges the splits own; only
    // the split count is shared with it (partition.active).
    const int logical_tiles    = div_up(window, Bc);
    const int first_owned_tile = split * logical_tiles / active_split_count;
    const int end_owned_tile   = (split + 1) * logical_tiles / active_split_count;
    const int split_start      = first_owned_tile * Bc;
    const int split_end        = min(end_owned_tile * Bc, window);
    // As in the grouped kernel: masked padding columns may carry positions past the last valid one;
    // keys stop at the last live column so unwritten KV bytes are never read.
    int key_end = split_end;
    if constexpr (Masked) { key_end = min(split_end, live_last_pos + 1); }
    const int key_blocks = key_end > split_start ? div_up(key_end - split_start, Bc) : 0;

    // Q: the grouped kernel's codec (normalized H256, then per-group INT8 codes), stored as FP16
    // integers for the A fragments.
    for (int row = warp; row < Rp; row += 4) {
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
            const int d0            = grp * kKVCacheInt8Group + lane;
            const float x0          = q_values[2 * grp];
            const float x1          = q_values[2 * grp + 1];
            const float amax        = warp_max(fmaxf(fabsf(x0), fabsf(x1)), FullMask);
            const float qs          = amax > 0.0f ? amax / 127.0f : 0.0f;
            const float inv         = qs > 0.0f ? 1.0f / qs : 0.0f;
            q_h[row * QS + d0]      = __int2half_rn(kv_cache_int8_quant_code(x0, inv));
            q_h[row * QS + d0 + 32] = __int2half_rn(kv_cache_int8_quant_code(x1, inv));
            if (lane == 0) { q_scale_s[row * Groups + grp] = live ? qs : 0.0f; }
        }
        if (lane == 0) {
            m_s[row] = -CUDART_INF_F;
            l_s[row] = 0.0f;
        }
    }

    // Fragment coordinates (PTX m8n8k4): quad-pair qp, the lane's A row / B column, and the
    // accumulator's row pair and column pair.
    const int qp      = (lane >> 2) & 3;
    const int frag    = (lane & 3) + ((lane >> 4) << 2);
    const int acc_row = (lane & 1) + ((lane >> 4) << 2);        // + (i & 2)
    const int acc_col = lane & 2;                               // + (i & 4) + (i & 1)
    const int grp     = warp;                                   // QK group and PV dimension block
    const int qk_key  = qp * 8 + frag;                          // B column = key of the tile
    const int pv_key  = lane & 3;                               // + 4 * k4: row-major B row
    // PV columns: quad-pair qp owns dims [64g + 16 qp, + 16), nb the four-dim halves of each
    // eight (nb = 0: n 0..3 at + 0, n 4..7 at + 8; nb = 1 at + 4 and + 12), so a lane's two V
    // words of a key are eight consecutive codes, one 8-byte load.
    const int pv_d    = grp * 64 + qp * 16 + ((lane >> 4) << 3); // + 4 * nb

    // Register double buffer of the tile's codes: K 64 bytes of the lane's key in group g, V four
    // codes per (k4, nb).
    uint4 k_next[4];
    unsigned v_next[2][8];
    uint2 scale_next[2];
    auto load_tile = [&](int tile_k0) {
        const int physical_page = block_table[tile_k0 >> kPagedKVPageShift];
        {
            const int key = tile_k0 + qk_key;
            if (key >= split_start && key < key_end) {
                const auto* src = reinterpret_cast<const uint4*>(
                    &cache_k_i8[kv_cache_int8_quant_code_index<Geometry>(
                        physical_page, kv_head, grp * 64, key & kPagedKVPageMask)]);
#    pragma unroll
                for (int i = 0; i < 4; ++i) k_next[i] = __ldg(src + i);
            } else {
#    pragma unroll
                for (int i = 0; i < 4; ++i) k_next[i] = make_uint4(0, 0, 0, 0);
            }
        }
#    pragma unroll
        for (int k4 = 0; k4 < Bc / 4; ++k4) {
            const int key  = tile_k0 + 4 * k4 + pv_key;
            const bool own = key >= split_start && key < key_end;
            const uint2 v = own ? __ldg(reinterpret_cast<const uint2*>(
                                      &cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(
                                          physical_page, kv_head, pv_d, key & kPagedKVPageMask)]))
                                : make_uint2(0u, 0u);
            v_next[0][k4] = v.x;
            v_next[1][k4] = v.y;
        }
        if (tid < Bc) {
            const int key = tile_k0 + tid;
            if (key >= split_start && key < key_end) {
                const std::int64_t off = kv_cache_int8_quant_scale_index<Geometry>(
                    physical_page, kv_head, 0, key & kPagedKVPageMask);
                scale_next[0] = __ldg(reinterpret_cast<const uint2*>(&cache_k_scale[off]));
                scale_next[1] = __ldg(reinterpret_cast<const uint2*>(&cache_v_scale[off]));
            } else {
                scale_next[0] = make_uint2(0, 0);
                scale_next[1] = make_uint2(0, 0);
            }
        }
    };

    float acc[RowTiles][2][8];
#    pragma unroll
    for (int rt = 0; rt < RowTiles; ++rt)
#    pragma unroll
        for (int nb = 0; nb < 2; ++nb)
#    pragma unroll
            for (int i = 0; i < 8; ++i) acc[rt][nb][i] = 0.0f;

    if (key_blocks > 0) load_tile(split_start);

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = split_start + kb * Bc;
        uint4 k_codes[4];
        unsigned v_codes[2][8];
#    pragma unroll
        for (int i = 0; i < 4; ++i) k_codes[i] = k_next[i];
#    pragma unroll
        for (int nb = 0; nb < 2; ++nb)
#    pragma unroll
            for (int i = 0; i < 8; ++i) v_codes[nb][i] = v_next[nb][i];
        // The previous tile's PV has read p_s and the scales; Q is written before the first.
        __syncthreads();
        if (tid < Bc) {
            *reinterpret_cast<uint2*>(&k_scale_s[tid * Groups]) = scale_next[0];
            *reinterpret_cast<uint2*>(&v_scale_s[tid * Groups]) = scale_next[1];
        }
        if (kb + 1 < key_blocks) load_tile(k0 + Bc);
        __syncthreads();

        // QK of group g: per row tile, sixteen k4 steps over the group's 64 dimensions.
        const unsigned* kw = reinterpret_cast<const unsigned*>(k_codes);
        float ks[4];
#    pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int key = qp * 8 + (i & 2) * 2 + acc_col + (i & 1); // cols (i&4)+(lane&2)+(i&1)
            ks[i]         = __half2float(k_scale_s[key * Groups + grp]);
        }
#    pragma unroll
        for (int rt = 0; rt < RowTiles; ++rt) {
            float s[8];
#    pragma unroll
            for (int i = 0; i < 8; ++i) s[i] = 0.0f;
            const __half* qrow = &q_h[(rt * 8 + frag) * QS + grp * 64];
#    pragma unroll
            for (int k4 = 0; k4 < 16; ++k4) {
                const uint2 a = *reinterpret_cast<const uint2*>(qrow + 4 * k4);
                unsigned b0, b1;
                int8x4_to_f16x2x2(kw[k4], b0, b1);
                mma_m8n8k4_f16(s, a.x, a.y, b0, b1);
            }
#    pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = rt * 8 + acc_row + (i & 2);
                const int key = qp * 8 + (i & 4) + acc_col + (i & 1);
                // (q_scale * k_scale) * dot, the grouped kernel's order.
                part_s[(grp * Rp + row) * PartS + key] =
                    q_scale_s[row * Groups + grp] * ks[((i & 4) >> 1) | (i & 1)] * s[i];
            }
        }
        __syncthreads();

        // Softmax: warp w takes rows w, w + 4, ...; lane = key.
        for (int row = warp; row < Rp; row += 4) {
            int q_head = 0, token = 0;
            causal_row_to_qt<Geometry>(row, kv_head, q_head, token);
            const bool live_row = row < tile_tokens * Geometry::GroupSize;
            const int qabs      = live_row ? pos[token] : -1;
            const int key       = k0 + lane;
            float score         = part_s[(0 * Rp + row) * PartS + lane];
            score += part_s[(1 * Rp + row) * PartS + lane];
            score += part_s[(2 * Rp + row) * PartS + lane];
            score += part_s[(3 * Rp + row) * PartS + lane];
            score             = (live_row && key >= split_start && key < key_end && key <= qabs)
                                    ? score * scale
                                    : -CUDART_INF_F;
            const float m_old = m_s[row];
            const float bm    = warp_max(score, FullMask);
            const float nm    = fmaxf(m_old, bm);
            const float alpha =
                m_old == -CUDART_INF_F ? 0.0f : causal_exp_difference(m_old, nm, Log2E);
            const float p = (nm > -CUDART_INF_F && score > -CUDART_INF_F)
                                ? causal_exp_difference(score, nm, Log2E)
                                : 0.0f;
            float bl      = p;
#    pragma unroll
            for (int o = 16; o > 0; o >>= 1) bl += __shfl_xor_sync(FullMask, bl, o);
            p_s[row * PS + lane] = __float2half_rn(p);
            __syncwarp();
            if (lane == 0) {
                alpha_s[row] = alpha;
                m_s[row]     = nm;
                l_s[row]     = l_s[row] * alpha + bl;
            }
        }
        __syncthreads();

        // PV: O[rows][64g .. 64g + 64) += P V, V scaled by group g's per-key scale in FP16.
#    pragma unroll
        for (int rt = 0; rt < RowTiles; ++rt) {
            const float a_lo = alpha_s[rt * 8 + acc_row];
            const float a_hi = alpha_s[rt * 8 + acc_row + 2];
#    pragma unroll
            for (int nb = 0; nb < 2; ++nb)
#    pragma unroll
                for (int i = 0; i < 8; ++i) acc[rt][nb][i] *= (i & 2) ? a_hi : a_lo;
        }
#    pragma unroll
        for (int k4 = 0; k4 < Bc / 4; ++k4) {
            const __half vs  = v_scale_s[(4 * k4 + pv_key) * Groups + grp];
            const __half2 v2 = __halves2half2(vs, vs);
            unsigned vb[2][2];
#    pragma unroll
            for (int nb = 0; nb < 2; ++nb) {
                unsigned lo, hi;
                int8x4_to_f16x2x2(v_codes[nb][k4], lo, hi);
                const __half2 l2 = __hmul2(*reinterpret_cast<const __half2*>(&lo), v2);
                const __half2 h2 = __hmul2(*reinterpret_cast<const __half2*>(&hi), v2);
                vb[nb][0]        = *reinterpret_cast<const unsigned*>(&l2);
                vb[nb][1]        = *reinterpret_cast<const unsigned*>(&h2);
            }
#    pragma unroll
            for (int rt = 0; rt < RowTiles; ++rt) {
                const uint2 a =
                    *reinterpret_cast<const uint2*>(&p_s[(rt * 8 + frag) * PS + 4 * k4]);
#    pragma unroll
                for (int nb = 0; nb < 2; ++nb)
                    mma_m8n8k4_f16_rr(acc[rt][nb], a.x, a.y, vb[nb][0], vb[nb][1]);
            }
        }
    }
    __syncthreads();

    for (int row = tid; row < Rp; row += 128) {
        if (row >= tile_tokens * Geometry::GroupSize) continue;
        int q_head = 0, token = 0;
        causal_row_to_qt<Geometry>(row, kv_head, q_head, token);
        const std::int64_t si =
            causal_stat_index<Geometry>(q_head, column_begin + token, split, partial_width);
        partial_m[si] = m_s[row];
        partial_l[si] = l_s[row];
    }
#    pragma unroll
    for (int rt = 0; rt < RowTiles; ++rt) {
#    pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = rt * 8 + acc_row + 2 * half;
            if (row >= tile_tokens * Geometry::GroupSize) continue;
            int q_head = 0, token = 0;
            causal_row_to_qt<Geometry>(row, kv_head, q_head, token);
#    pragma unroll
            for (int nb = 0; nb < 2; ++nb) {
#    pragma unroll
                for (int c4 = 0; c4 < 2; ++c4) {
                    const int d = grp * 64 + qp * 16 + 8 * c4 + 4 * nb + acc_col;
                    const int i = 4 * c4 + 2 * half;
                    causal_store_partial_pair(
                        &partial_acc[causal_partial_index<Geometry>(q_head, d, column_begin + token,
                                                                    split, partial_width)],
                        acc[rt][nb][i], acc[rt][nb][i + 1]);
                }
            }
        }
    }
#else
    __trap();
#endif
}

} // namespace ninferno::ops::detail
