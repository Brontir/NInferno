#pragma once
#include "ops/softmax_attention/dense/causal_cache/nvfp4/slab_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/tile_io.cuh"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/schedule.cuh"
#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/common/causal_epilogue.cuh"
#include "ops/softmax_attention/common/causal_softmax.cuh"

namespace ninferno::ops::detail {

// NVFP4 KV decode attention on warp-private key slabs (#849, docs/maintainer/kernel-proposals.md
// entry G). The grouped kernel (grouped_mma.cuh) decodes K and V to FP16 through shared memory
// between four CTA barriers per 32-key tile and runs QK on one warp; here every warp owns a
// 16-key slab of every 128 keys, decodes its own K and V straight into MMA fragments in
// registers and keeps its own online softmax, with no CTA barrier in the loop. The CTA merges its
// eight warp states once at the end.
//
// Orientation: S^T = K Q^T, keys as the MMA's M (16 keys of the slab) and the query rows as its N
// (one n8 tile: the 4-8 rows of a GQA decode step), so Q's B fragments stay in registers for the
// whole loop. A lane's ldmatrix of the raw code tile gives it, for chunk c (32 d values) of key g
// and g + 8, the word of d = 32c + 8t .. 32c + 8t + 7 (t = lane % 4); its four pairs
// (slab_codec.cuh) fill the A slots 2t, 2t + 1 and 2t + 8, 2t + 9 of two k16 MMAs, and Q's B
// fragment holds the same d in the same slots, so each MMA sums the products of the grouped
// kernel's FP16 values in another order. O^T = V^T P^T: V decodes in the same natural layout and
// movmatrix transposes each 8x8 block into the V^T A fragment; P^T's B fragment is the S^T
// accumulator transposed the same way. A lane's O^T rows are block-strided d values (`slab_d`),
// which only the final store reads.
//
// Numerics: Q in FP16, K and V the represented FP16 values times an exact power of two
// (nvfp4_slab_fold, restored in FP32), FP32 accumulation. The order of the sums differs from the
// grouped kernel's: the k permutation inside an MMA, an online softmax per warp over 16-key slabs,
// the 8-way merge. Aligned launches (the invariant route profile) keep the grouped kernel
// (template_launch.cuh), so no pinned arithmetic moves.
template <class Geometry, class Schedule, bool MultiBatch, bool Masked, class CacheInput,
          bool ParallelQueries = false>
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocks) __global__
    void nvfp4_kv_slab_mma_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* positions,
        typename Nvfp4KvCacheView<CacheInput::writes_cache>::Code* cache_k,
        typename Nvfp4KvCacheView<CacheInput::writes_cache>::Code* cache_v,
        typename Nvfp4KvCacheView<CacheInput::writes_cache>::KeyScale* cache_k_scale,
        typename Nvfp4KvCacheView<CacheInput::writes_cache>::ValueScale* cache_v_scale,
        const std::int32_t* block_tables, const std::int32_t* valid_columns,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
        std::int32_t logical_capacity, CausalKvPartition partition, float attention_scale,
        float* partial_acc, float* partial_m, float* partial_l) {
    constexpr int TokenTile     = Schedule::kTokenTile;
    constexpr int Warps         = Schedule::kWarps;
    constexpr int Threads       = Schedule::kThreads;
    constexpr int SlabKeys      = Schedule::kSlabKeys;
    constexpr int StageBytes    = Schedule::kStageBytes;
    constexpr int Stages        = Schedule::kStages;
    constexpr int D             = 256;
    constexpr int CodeRowBytes  = D / 2;
    constexpr int ScaleRowBytes = kKVCacheNvfp4Groups;
    constexpr int Chunks        = D / 32; // 16-byte code chunks of a key row
    constexpr int RowCount      = TokenTile * Geometry::GroupSize;
    constexpr int Rows          = 8; // one n8 tile of query rows
    constexpr int MTiles        = D / 16;
    constexpr float Log2E       = kLog2E;
    constexpr unsigned FullMask = 0xffffffffU;
    // Offsets of one stage: K codes, V codes (16 rows of 128 bytes, chunks swizzled), K and V
    // scales (16 rows of 16 bytes).
    constexpr int KCodes = 0, VCodes = SlabKeys * CodeRowBytes;
    constexpr int KScales = 2 * SlabKeys * CodeRowBytes,
                  VScales = KScales + SlabKeys * ScaleRowBytes;

    static_assert(Warps == 8 && SlabKeys == 16 && Stages == 2 && Threads == Warps * 32);
    static_assert(RowCount >= 1 && RowCount <= Rows);
    static_assert(VScales + SlabKeys * ScaleRowBytes == StageBytes);
    static_assert(!ParallelQueries || !CacheInput::writes_cache);
    // The append scratch and the final merge reuse the ring.
    static_assert(Warps * D * 4 <= Schedule::kArenaBytes);
    static_assert(Warps * 32 * MTiles * 4 * 4 <= Schedule::kArenaBytes);
    static_assert(Schedule::kRows == RowCount && Schedule::kFitsSharedLimit);
    static_assert(Schedule::kArenaBytes + Rows * D * 2 + 2 * Warps * Rows * 4 ==
                  Schedule::kSharedBytes);

    extern __shared__ __align__(16) std::uint8_t arena[];
    __shared__ __align__(16) __half q_s[Rows * D];
    __shared__ float stat_m[Warps][Rows];
    __shared__ float stat_l[Warps][Rows];

    const int kv_head      = ParallelQueries ? blockIdx.x % Geometry::KVHeads : blockIdx.x;
    const int column_begin = ParallelQueries ? (blockIdx.x / Geometry::KVHeads) * TokenTile : 0;
    const int tile_tokens = ParallelQueries ? min(TokenTile, full_width - column_begin) : TokenTile;
    const int live_rows   = tile_tokens * Geometry::GroupSize;
    const int partial_width = full_width, partial_begin = column_begin;
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    const int gid         = lane >> 2;
    const int lid         = lane & 3;

    int valid_tokens = tile_tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : min(remaining, tile_tokens);
    }
    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(D) * Geometry::QHeads * column_base;
    const int last_pos = positions[(MultiBatch ? batch * full_width : 0) + full_width - 1];
    const int live_last_pos =
        Masked ? positions[(MultiBatch ? batch * full_width : 0) + max(valid_columns[batch], 1) - 1]
               : last_pos;
    positions += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(D) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(D) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc +=
            static_cast<std::int64_t>(batch) * D * Geometry::QHeads * partial_width * split_count;
        partial_m +=
            static_cast<std::int64_t>(batch) * Geometry::QHeads * partial_width * split_count;
        partial_l +=
            static_cast<std::int64_t>(batch) * Geometry::QHeads * partial_width * split_count;
    }

    if (valid_tokens == 0) return;
    if (positions[0] < 0 || last_pos < 0 || last_pos >= logical_capacity) return;
    const int window             = last_pos + 1;
    const int active_split_count = partition.active(window);
    if (split >= active_split_count) return;
    // The splits cut the window at slab boundaries, so a slab never straddles two of them, nor
    // (64-key pages) two pages.
    const int logical_slabs = div_up(window, SlabKeys);
    const int split_start =
        partition.first_tile(split, logical_slabs, active_split_count, SlabKeys) * SlabKeys;
    const int split_end =
        min(partition.first_tile(split + 1, logical_slabs, active_split_count, SlabKeys) * SlabKeys,
            window);
    // A masked call's padding columns may sit past the last valid one (grouped_mma.cuh): loads and
    // scores stop at key_end, so no unwritten key is read.
    int key_end = split_end;
    if constexpr (Masked) { key_end = min(split_end, live_last_pos + 1); }
    const int slabs = key_end > split_start ? div_up(key_end - split_start, SlabKeys) : 0;

    if constexpr (CacheInput::writes_cache) {
        // The grouped kernel's fused append, unchanged: one warp per new token, the same G16
        // operation and rounding order as the standalone append.
        float* quant_scratch = reinterpret_cast<float*>(arena);
        for (int token = warp; token < valid_tokens; token += Warps) {
            const int position = positions[token];
            if (position < split_start || position >= split_end) continue;
            const int physical_page = block_table[position >> kPagedKVPageShift];
            const int page_offset   = position & kPagedKVPageMask;
            float* row_s            = quant_scratch + warp * D;
            // One row (K, then V): Hadamard, G16 quantization, codes and scale into the page.
            const auto append_row = [&](const __nv_bfloat16* source, auto* codes, auto* scales) {
                float values[8];
#pragma unroll
                for (int r = 0; r < 8; ++r)
                    values[r] = __bfloat162float(
                        source[kv_cache_nvfp4_src_index<Geometry>(kv_head, lane + 32 * r, token)]);
                normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
                for (int r = 0; r < 8; ++r) row_s[lane + 32 * r] = values[r];
                __syncwarp();
                if (lane < kKVCacheNvfp4Groups) {
                    const auto quantized =
                        kv_cache_nvfp4_quantize_group16(row_s + lane * kKVCacheNvfp4Group);
                    const std::int64_t code_offset = kv_cache_nvfp4_code_index<Geometry>(
                        physical_page, kv_head, lane * kKVCacheNvfp4Group, page_offset);
                    store_vec(codes + code_offset,
                              make_uint2(quantized.codes_lo, quantized.codes_hi));
                    scales[kv_cache_nvfp4_scale_index<Geometry>(physical_page, kv_head, lane,
                                                                page_offset)] = quantized.scale;
                }
                __syncwarp();
            };
            append_row(input.k, cache_k, cache_k_scale);
            append_row(input.v, cache_v, cache_v_scale);
        }
        __syncthreads();
    }

    // Q rows: BF16 -> Hadamard -> FP16, every row of the n8 tile written (zero past the live rows).
    for (int row = warp; row < Rows; row += Warps) {
        int q_head = 0, token = 0;
        causal_row_to_qt<Geometry>(row, kv_head, q_head, token);
        const bool live = row < RowCount && token < valid_tokens;
        float values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            values[r] =
                live ? __bfloat162float(q[causal_q_index<Geometry>(q_head, d, token)]) : 0.0F;
        }
        normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
        for (int r = 0; r < 8; ++r) q_s[row * D + lane + 32 * r] = __float2half_rn(values[r]);
    }
    __syncthreads();

    // Q's B fragments: row gid's d = 32c + 8t .. + 7, paired as the code words are.
    unsigned qf[Chunks][4];
#pragma unroll
    for (int c = 0; c < Chunks; ++c) {
        const uint4 h = load_vec<uint4>(&q_s[gid * D + 32 * c + 8 * lid]);
#if NINFERNO_NVFP4_SLAB_HARDWARE_E2M1
        qf[c][0] = h.x;
        qf[c][1] = h.y;
        qf[c][2] = h.z;
        qf[c][3] = h.w;
#else
        qf[c][0] = __byte_perm(h.x, h.z, 0x5410U); // values 0, 4
        qf[c][1] = __byte_perm(h.x, h.z, 0x7632U); // 1, 5
        qf[c][2] = __byte_perm(h.y, h.w, 0x5410U); // 2, 6
        qf[c][3] = __byte_perm(h.y, h.w, 0x7632U); // 3, 7
#endif
    }

    // The lane's two query rows (2t, 2t + 1) and their causal limits (-1: a row past the tile).
    int qabs[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int row = 2 * lid + j;
        int q_head = 0, token = 0;
        causal_row_to_qt<Geometry>(row, kv_head, q_head, token);
        qabs[j] = row < live_rows ? positions[token] : -1;
    }
    // The scores carry the fold of K's decode, exactly (a power of two).
    const float score_scale = attention_scale * nvfp4_slab_fold();
    // The scale selectors: chunk c's group is 2c + t / 2, byte 2 (c & 1) + t / 2 of word c / 2.
    const unsigned scale_sel_even = nvfp4_slab_scale_selector(static_cast<unsigned>(lid >> 1));
    const unsigned scale_sel_odd  = nvfp4_slab_scale_selector(2U + static_cast<unsigned>(lid >> 1));

    float acc[MTiles][4];
#pragma unroll
    for (int mt = 0; mt < MTiles; ++mt) acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.0F;
    float m[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float l[2] = {0.0F, 0.0F}; // this lane's keys only; summed over the row's lanes at the end

    // The warp's slabs: warp, warp + Warps, ... of the split's.
    const int warp_slabs     = slabs > warp ? (slabs - warp + Warps - 1) / Warps : 0;
    std::uint8_t* const ring = arena + warp * Stages * StageBytes;
    // ldmatrix row addresses: lane L names row L % 8 (+ 8 for L & 8) of chunk 2i + L / 16.
    const int ld_key   = lane & 15;
    const int ld_chunk = lane >> 4;

    const auto issue = [&](int index) {
        const int key0          = split_start + (warp + index * Warps) * SlabKeys;
        std::uint8_t* stage     = ring + (index & 1) * StageBytes;
        const int physical_page = block_table[key0 >> kPagedKVPageShift];
        const int page_offset   = key0 & kPagedKVPageMask;
#pragma unroll
        for (int j = 0; j < SlabKeys * Chunks / 32; ++j) {
            const int chunk = lane + 32 * j;
            const int key_l = chunk / Chunks;
            const int c     = chunk % Chunks;
            if (key0 + key_l < key_end) {
                const std::int64_t offset = kv_cache_nvfp4_code_index<Geometry>(
                    physical_page, kv_head, 32 * c, page_offset + key_l);
                const int dst = key_l * CodeRowBytes + nvfp4_kv_code_swizzle(key_l, 16 * c);
                cp_async<16, Cache::cg>(stage + KCodes + dst, cache_k + offset);
                cp_async<16, Cache::cg>(stage + VCodes + dst, cache_v + offset);
            }
        }
        {
            // A key past key_end gets zero scales: its stale (finite) codes then decode to zero,
            // and an unwritten V never meets a zero weight.
            const int key_l   = lane & 15;
            std::uint8_t* dst = stage + (lane < 16 ? KScales : VScales) + key_l * ScaleRowBytes;
            if (key0 + key_l < key_end) {
                const std::int64_t offset = kv_cache_nvfp4_scale_index<Geometry>(
                    physical_page, kv_head, 0, page_offset + key_l);
                cp_async<16>(dst, lane < 16 ? cache_k_scale + offset : cache_v_scale + offset);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
    };

    if (warp_slabs > 0) issue(0);
    ninferno::ops::cp_commit();
#pragma unroll 1
    for (int index = 0; index < warp_slabs; ++index) {
        if (index + 1 < warp_slabs) issue(index + 1);
        ninferno::ops::cp_commit();
        ninferno::ops::cp_wait<1>();
        __syncwarp();
        const std::uint8_t* stage = ring + (index & 1) * StageBytes;
        const unsigned stage_addr = smem_addr(stage);
        const int key0            = split_start + (warp + index * Warps) * SlabKeys;

        // S^T = K Q^T over the 16 keys: two accumulator chains (even and odd k16 steps).
        float s[2][4] = {{0.0F, 0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F, 0.0F}};
        {
            unsigned kw[Chunks][2];
#pragma unroll
            for (int i = 0; i < Chunks / 2; ++i) {
                const int c = 2 * i + ld_chunk;
                ldmatrix_x4(kw[2 * i][0], kw[2 * i][1], kw[2 * i + 1][0], kw[2 * i + 1][1],
                            stage_addr + KCodes + ld_key * CodeRowBytes +
                                nvfp4_kv_code_swizzle(ld_key, 16 * c));
            }
            const uint4 ks0         = load_vec<uint4>(stage + KScales + gid * ScaleRowBytes);
            const uint4 ks1         = load_vec<uint4>(stage + KScales + (gid + 8) * ScaleRowBytes);
            const unsigned ks[2][4] = {{ks0.x, ks0.y, ks0.z, ks0.w}, {ks1.x, ks1.y, ks1.z, ks1.w}};
#pragma unroll
            for (int c = 0; c < Chunks; ++c) {
                const unsigned sel = (c & 1) ? scale_sel_odd : scale_sel_even;
                const __half2 s0   = nvfp4_slab_scale2(ks[0][c >> 1], sel);
                const __half2 s1   = nvfp4_slab_scale2(ks[1][c >> 1], sel);
                mma_f16(s[0][0], s[0][1], s[0][2], s[0][3],
                        nvfp4_slab_scaled(nvfp4_slab_pair<0>(kw[c][0]), s0),
                        nvfp4_slab_scaled(nvfp4_slab_pair<0>(kw[c][1]), s1),
                        nvfp4_slab_scaled(nvfp4_slab_pair<1>(kw[c][0]), s0),
                        nvfp4_slab_scaled(nvfp4_slab_pair<1>(kw[c][1]), s1), qf[c][0], qf[c][1]);
                mma_f16(s[1][0], s[1][1], s[1][2], s[1][3],
                        nvfp4_slab_scaled(nvfp4_slab_pair<2>(kw[c][0]), s0),
                        nvfp4_slab_scaled(nvfp4_slab_pair<2>(kw[c][1]), s1),
                        nvfp4_slab_scaled(nvfp4_slab_pair<3>(kw[c][0]), s0),
                        nvfp4_slab_scaled(nvfp4_slab_pair<3>(kw[c][1]), s1), qf[c][2], qf[c][3]);
            }
        }

        // Online softmax per row. The lane holds keys gid (s[.][0..1]) and gid + 8 (s[.][2..3]) of
        // rows 2t and 2t + 1; a row's 16 keys span the eight lanes of one t.
        const int key_lo = key0 + gid, key_hi = key_lo + 8;
        float p[4];
        float alpha[2];
        // A row whose accumulators its rescale cannot change: a factor of exactly 1, or no key
        // seen yet (the accumulators are still zero; a row past the tile never sees one).
        bool unchanged = true;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float lo  = key_lo < key_end && key_lo <= qabs[j]
                                  ? (s[0][j] + s[1][j]) * score_scale
                                  : -CUDART_INF_F;
            const float hi  = key_hi < key_end && key_hi <= qabs[j]
                                  ? (s[0][2 + j] + s[1][2 + j]) * score_scale
                                  : -CUDART_INF_F;
            float block_max = fmaxf(lo, hi);
            block_max       = fmaxf(block_max, __shfl_xor_sync(FullMask, block_max, 4));
            block_max       = fmaxf(block_max, __shfl_xor_sync(FullMask, block_max, 8));
            block_max       = fmaxf(block_max, __shfl_xor_sync(FullMask, block_max, 16));
            const float nm  = fmaxf(m[j], block_max);
            alpha[j]        = m[j] == -CUDART_INF_F ? 0.0F : causal_exp_difference(m[j], nm, Log2E);
            unchanged       = unchanged && (m[j] == -CUDART_INF_F || alpha[j] == 1.0F);
            p[j] = nm > -CUDART_INF_F && lo > -CUDART_INF_F ? causal_exp_difference(lo, nm, Log2E)
                                                            : 0.0F;
            p[2 + j] = nm > -CUDART_INF_F && hi > -CUDART_INF_F
                           ? causal_exp_difference(hi, nm, Log2E)
                           : 0.0F;
            l[j]     = __fmaf_rn(l[j], alpha[j], p[j] + p[2 + j]);
            m[j]     = nm;
        }
        // P^T's B fragment: the probabilities of keys gid / gid + 8 for rows 2t, 2t + 1, as FP16,
        // transposed to keys 2t, 2t + 1 (and + 8) of row gid.
        const __half2 p_lo = __floats2half2_rn(p[0], p[1]);
        const __half2 p_hi = __floats2half2_rn(p[2], p[3]);
        const unsigned pb0 = movmatrix_trans(*reinterpret_cast<const unsigned*>(&p_lo));
        const unsigned pb1 = movmatrix_trans(*reinterpret_cast<const unsigned*>(&p_hi));
        // The warp skips the rescale when it changes no row (the common case once a row's
        // maximum settles): the result is the same bits.
        if (!__all_sync(FullMask, unchanged)) {
#pragma unroll
            for (int mt = 0; mt < MTiles; ++mt) {
                acc[mt][0] *= alpha[0];
                acc[mt][1] *= alpha[1];
                acc[mt][2] *= alpha[0];
                acc[mt][3] *= alpha[1];
            }
        }

        // O^T += V^T P^T: per chunk c, pairs (0, 1) and (2, 3) are one m16 tile each.
        {
            unsigned vw[Chunks][2];
#pragma unroll
            for (int i = 0; i < Chunks / 2; ++i) {
                const int c = 2 * i + ld_chunk;
                ldmatrix_x4(vw[2 * i][0], vw[2 * i][1], vw[2 * i + 1][0], vw[2 * i + 1][1],
                            stage_addr + VCodes + ld_key * CodeRowBytes +
                                nvfp4_kv_code_swizzle(ld_key, 16 * c));
            }
            const uint4 vs0         = load_vec<uint4>(stage + VScales + gid * ScaleRowBytes);
            const uint4 vs1         = load_vec<uint4>(stage + VScales + (gid + 8) * ScaleRowBytes);
            const unsigned vs[2][4] = {{vs0.x, vs0.y, vs0.z, vs0.w}, {vs1.x, vs1.y, vs1.z, vs1.w}};
#pragma unroll
            for (int c = 0; c < Chunks; ++c) {
                const unsigned sel = (c & 1) ? scale_sel_odd : scale_sel_even;
                const __half2 s0   = nvfp4_slab_scale2(vs[0][c >> 1], sel);
                const __half2 s1   = nvfp4_slab_scale2(vs[1][c >> 1], sel);
                mma_f16(acc[2 * c][0], acc[2 * c][1], acc[2 * c][2], acc[2 * c][3],
                        movmatrix_trans(nvfp4_slab_scaled(nvfp4_slab_pair<0>(vw[c][0]), s0)),
                        movmatrix_trans(nvfp4_slab_scaled(nvfp4_slab_pair<1>(vw[c][0]), s0)),
                        movmatrix_trans(nvfp4_slab_scaled(nvfp4_slab_pair<0>(vw[c][1]), s1)),
                        movmatrix_trans(nvfp4_slab_scaled(nvfp4_slab_pair<1>(vw[c][1]), s1)), pb0,
                        pb1);
                mma_f16(acc[2 * c + 1][0], acc[2 * c + 1][1], acc[2 * c + 1][2], acc[2 * c + 1][3],
                        movmatrix_trans(nvfp4_slab_scaled(nvfp4_slab_pair<2>(vw[c][0]), s0)),
                        movmatrix_trans(nvfp4_slab_scaled(nvfp4_slab_pair<3>(vw[c][0]), s0)),
                        movmatrix_trans(nvfp4_slab_scaled(nvfp4_slab_pair<2>(vw[c][1]), s1)),
                        movmatrix_trans(nvfp4_slab_scaled(nvfp4_slab_pair<3>(vw[c][1]), s1)), pb0,
                        pb1);
            }
        }
        __syncwarp();
    }
    ninferno::ops::cp_wait<0>();

    // The 8-way merge. Each warp's row state: m (already the same on a row's eight lanes), l
    // (summed over them here) and O^T, which every warp holds in the same lane layout.
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        l[j] += __shfl_xor_sync(FullMask, l[j], 4);
        l[j] += __shfl_xor_sync(FullMask, l[j], 8);
        l[j] += __shfl_xor_sync(FullMask, l[j], 16);
    }
    if (gid == 0) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            stat_m[warp][2 * lid + j] = m[j];
            stat_l[warp][2 * lid + j] = l[j];
        }
    }
    __syncthreads(); // also: every warp is done with its ring
    float factor[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        float row_max = -CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < Warps; ++w) row_max = fmaxf(row_max, stat_m[w][2 * lid + j]);
        factor[j] = m[j] == -CUDART_INF_F ? 0.0F : causal_exp_difference(m[j], row_max, Log2E);
    }
    // Lane L of every warp writes its 64 scaled values as 16 float4, XOR-swizzled by L % 16 so
    // that eight lanes of a quarter warp hit eight distinct 16-byte bank groups.
    float4* merge = reinterpret_cast<float4*>(arena);
#pragma unroll
    for (int mt = 0; mt < MTiles; ++mt) {
        merge[(warp * 32 + lane) * MTiles + (mt ^ (lane & 15))] =
            make_float4(acc[mt][0] * factor[0], acc[mt][1] * factor[1], acc[mt][2] * factor[0],
                        acc[mt][3] * factor[1]);
    }
    __syncthreads();

    // Warp w sums m tiles 2w, 2w + 1 of its own lane over the eight warps, in warp order, and
    // stores them: element i of m tile mt is row 2t + (i & 1) at d = slab_d(mt, i >> 1).
    const float fold = nvfp4_slab_fold(); // O^T carries V's decode fold
#pragma unroll
    for (int k = 0; k < MTiles / Warps; ++k) {
        const int mt = warp * (MTiles / Warps) + k;
        float4 sum   = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
#pragma unroll
        for (int w = 0; w < Warps; ++w) {
            const float4 v = merge[(w * 32 + lane) * MTiles + (mt ^ (lane & 15))];
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
        }
        const float values[4] = {sum.x * fold, sum.y * fold, sum.z * fold, sum.w * fold};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = 2 * lid + (i & 1);
            if (row >= live_rows) continue;
            // m tile mt: chunk mt / 2, pair 2 (mt & 1) + i / 2 (rows gid of the pair's 8x8 block,
            // after the transpose: value d0 + nibble(pair, gid & 1) of the word of t = gid / 2).
            const int pair = 2 * (mt & 1) + (i >> 1);
            const int d    = 32 * (mt >> 1) + 8 * (gid >> 1) + nvfp4_slab_nibble(pair, gid & 1);
            int q_head = 0, token = 0;
            causal_row_to_qt<Geometry>(row, kv_head, q_head, token);
            partial_acc[causal_partial_index<Geometry>(q_head, d, partial_begin + token, split,
                                                       partial_width)] = values[i];
        }
    }
    if (tid < Rows && tid < live_rows) {
        float row_max = -CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < Warps; ++w) row_max = fmaxf(row_max, stat_m[w][tid]);
        float row_sum = 0.0F;
#pragma unroll
        for (int w = 0; w < Warps; ++w) {
            const float wm = stat_m[w][tid];
            row_sum += wm == -CUDART_INF_F
                           ? 0.0F
                           : stat_l[w][tid] * causal_exp_difference(wm, row_max, Log2E);
        }
        int q_head = 0, token = 0;
        causal_row_to_qt<Geometry>(tid, kv_head, q_head, token);
        const auto stat =
            causal_stat_index<Geometry>(q_head, partial_begin + token, split, partial_width);
        partial_m[stat] = row_max;
        partial_l[stat] = row_sum;
    }
}

} // namespace ninferno::ops::detail
