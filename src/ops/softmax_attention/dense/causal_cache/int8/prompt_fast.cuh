#pragma once

// INT8-cache causal prompt attention for wide prefill calls (Int8KvFamily::Tiled), after the fast
// prompt kernel of ninfer-all (Ian Ranson, Apache-2.0, extended there by the fork's authors; see
// NOTICE).
//
//   * Each warp owns 16 query rows of one head for the whole key sweep (eight warps per CTA, or
//     four for launches too narrow to fill every SM with 128-row CTAs), so scores, probabilities
//     and the D256 output accumulator stay in registers and no warp waits for another between QK
//     and PV. The only CTA barrier publishes a key tile.
//   * Q is rotated and G64-encoded as int8_kv_tiled_mma_kernel does; QK runs m16n8k32.s8 with the
//     group scales applied in FP32.
//   * K and V codes are double-buffered raw INT8 pages (one 64-key page per tile), the next page's
//     copy overlapping this tile's math. V is decoded in registers: an ldmatrix.trans b16 lane
//     holds two keys of two adjacent dimensions, one FP16 B fragment for an even and an odd n8
//     tile; each code widens exactly and takes its FP16 group scale once.
//   * PV runs FP16 MMA with FP16 accumulation over a tile's 64 keys, promoted into the FP32
//     accumulator once per tile, as the tiled kernel does. A tile whose largest V scale could take
//     a partial past the FP16 range decodes with its scales divided by an exact power of two and
//     multiplies the partial back.
//   * CTAs issue longest-first, so a causal prompt's heaviest row blocks do not form the tail.
//   * A row's keys are swept in pieces of kInt8PromptSplitKeys from position 0, one CTA each (the
//     launch's third dimension). A row that sees at most one piece is stored directly; a longer
//     row leaves an FP32 partial per piece, folded in piece order by the merge kernel below. The
//     pieces follow the row's own position, so a row's bytes do not depend on the call's width,
//     and a narrow call over a long context still spreads over the SMs.

#include "ops/softmax_attention/dense/causal_cache/int8/prompt_split.h"
#include "ops/softmax_attention/dense/causal_cache/int8/tile_io.cuh"
#include "ops/softmax_attention/common/causal_operands.h"

#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninferno::ops::detail {

inline constexpr int kInt8PromptFastBc        = 64;
inline constexpr int kInt8PromptFastD         = 256;
inline constexpr int kInt8PromptFastGroups    = kInt8PromptFastD / kKVCacheInt8Group;
inline constexpr int kInt8PromptFastTileBytes = kInt8PromptFastBc * kInt8PromptFastD;
inline constexpr int kInt8PromptFastScaleBytes =
    kInt8PromptFastBc * kInt8PromptFastGroups * static_cast<int>(sizeof(__half));
// One stage: K codes, V codes, then the G64 key scales and value scales.
inline constexpr int kInt8PromptFastStageBytes =
    2 * kInt8PromptFastTileBytes + 2 * kInt8PromptFastScaleBytes;

template <int Warps>
struct Int8PromptFastShape {
    static_assert(Warps == 4 || Warps == 8);
    static constexpr int kThreads     = Warps * 32;
    static constexpr int kQueryRows   = Warps * 16;
    static constexpr int kQBytes      = kQueryRows * kInt8PromptFastD;
    static constexpr int kSharedBytes = kQBytes + 2 * kInt8PromptFastStageBytes;
};

// A 64-key FP16 partial is bounded by 64 * 127 * max_scale (every probability is at most one);
// a scale at or below 8 keeps it, with rounding slack, under 65504.
inline constexpr float kInt8PromptFastF16PartialScaleLimit = 8.0f;

// The key tiles of one piece (int8/prompt_split.h); a row seeing v keys runs
// (v - 1) / kInt8PromptSplitKeys + 1 pieces.
inline constexpr int kInt8PromptSplitTiles = kInt8PromptSplitKeys / kInt8PromptFastBc;
static_assert(kInt8PromptSplitKeys % kInt8PromptFastBc == 0);

static_assert(kInt8PromptFastBc == kPagedKVPageSize);
static_assert(kInt8PromptFastGroups == 4);
static_assert(Int8PromptFastShape<8>::kSharedBytes == 100352);

// One ldmatrix.trans b16 lane of an INT8 [key][d] tile holds {V[k][d], V[k][d+1], V[k+1][d],
// V[k+1][d+1]}. Returns the FP16 B-fragment halves {V[k][d], V[k+1][d]} and {V[k][d+1],
// V[k+1][d+1]}, each code widened exactly and multiplied once by its key's group scale.
__device__ __forceinline__ void int8_prompt_fast_decode_v_pair(unsigned codes, unsigned scales,
                                                               unsigned& even, unsigned& odd) {
    // code ^ 0x80 is code + 128; 0x6400 | byte is the FP16 value 1024 + byte.
    const unsigned biased = codes ^ 0x80808080u;
    unsigned e            = __byte_perm(biased, 0x64646464u, 0x4240);
    unsigned o            = __byte_perm(biased, 0x64646464u, 0x4341);
    const __half2 offset  = __float2half2_rn(1152.0f);
    const __half2 s2      = load_vec<__half2>(&scales);
    const __half2 ve      = __hmul2(__hsub2(load_vec<__half2>(&e), offset), s2);
    const __half2 vo      = __hmul2(__hsub2(load_vec<__half2>(&o), offset), s2);
    even                  = load_vec<unsigned>(&ve);
    odd                   = load_vec<unsigned>(&vo);
}

template <typename Geometry, typename Metadata, int Warps>
__global__
__launch_bounds__(Int8PromptFastShape<Warps>::kThreads, 1) void int8_kv_prompt_fast_kernel(
    const __nv_bfloat16* __restrict__ q, const std::int8_t* __restrict__ cache_k,
    const std::int8_t* __restrict__ cache_v, const __half* __restrict__ cache_k_scale,
    const __half* __restrict__ cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    std::int32_t width, std::int32_t narrow_slots, CausalPartialView partial) {
    constexpr int D             = kInt8PromptFastD;
    constexpr int DB16          = D / 2;
    using Shape                 = Int8PromptFastShape<Warps>;
    constexpr int Threads       = Shape::kThreads;
    constexpr int Br            = Shape::kQueryRows;
    constexpr int Bc            = kInt8PromptFastBc;
    constexpr int Groups        = kInt8PromptFastGroups;
    constexpr int GroupKc       = kKVCacheInt8Group / 32;
    constexpr int QKNt          = Bc / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr int DBlocks       = D / 16;
    constexpr int GroupDBlocks  = kKVCacheInt8Group / 16;
    constexpr int PassDBlocks   = 2;
    constexpr int KScaleOffset  = 2 * kInt8PromptFastTileBytes;
    constexpr float Log2E       = kLog2E;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(Geometry::kHeadDim == D);
    static_assert(GroupKc == 2);
    static_assert(GroupDBlocks == 4);
    // A group's int32 dot product stays below 2^22 (int_accumulator_to_float).
    static_assert(kKVCacheInt8Group <= 256);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::int8_t* q_i8    = reinterpret_cast<std::int8_t*>(smem_raw);
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    // Longest-first issue order: the first QHeads CTAs take the last row block of every head. The
    // first narrow_slots * Br / 2 rows run as half-height CTAs, issued last: they fill the final
    // wave of a launch whose full-height CTAs would leave most SMs idle there.
    const int slots      = static_cast<int>(gridDim.x);
    const int wide_slots = slots - narrow_slots;
    const int linear     = static_cast<int>(blockIdx.x + blockIdx.y * gridDim.x);
    const int q_head     = linear % Geometry::QHeads;
    const int slot       = linear / Geometry::QHeads;
    const int cta_rows   = slot < wide_slots ? Br : Br / 2;
    const int q0         = slot < wide_slots
                               ? narrow_slots * (Br / 2) + (wide_slots - 1 - slot) * Br
                               : (slots - 1 - slot) * (Br / 2);
    const int kv_head    = q_head / Geometry::GroupSize;
    const int tokens     = metadata.valid_tokens(width);

    // Rows past the valid token count are inactive columns and publish zeros.
    const auto store_row = [&](int row, int d0, float v0, float v1, float v2, float v3) {
        const int token = q0 + row;
        if (row >= cta_rows || token >= width) { return; }
        const bool valid   = token < tokens;
        const uint2 packed = make_uint2(pack_bf16x2(valid ? v0 : 0.0f, valid ? v1 : 0.0f),
                                        pack_bf16x2(valid ? v2 : 0.0f, valid ? v3 : 0.0f));
        store_vec(&out[causal_q_index<Geometry>(q_head, d0, token)], packed);
    };

    const int split = static_cast<int>(blockIdx.z);
    if (q0 >= tokens) {
        if (split == 0) {
            causal_zero_rows<Geometry>(out, q_head, q0, min(q0 + cta_rows, width), tid, Threads);
        }
        return;
    }

    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    const int rows                  = min(cta_rows, tokens - q0);
    const int max_query_abs         = base_pos + q0 + rows - 1;
    const int key_blocks            = max_query_abs / Bc + 1;
    // This CTA's piece of the keys; none of its rows reaches a later piece.
    const int first_block = split * kInt8PromptSplitTiles;
    if (first_block >= key_blocks) { return; }
    const int end_block = min(key_blocks, first_block + kInt8PromptSplitTiles);

    // Each warp rotates and encodes its own 16 rows, keeping the two rows each lane owns in the
    // MMA C layout as register scales.
    float q_scale_r[2][Groups];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int row    = warp * 16 + r;
        const int token  = q0 + row;
        const bool valid = row < rows;
        float q_values[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            q_values[k] = valid ? __bfloat162float(q[causal_q_index<Geometry>(q_head, lane + 32 * k,
                                                                              token)])
                                : 0.0f;
        }
        normalized_hadamard_d256_inplace(q_values, lane);
#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            const int d0    = grp * kKVCacheInt8Group + lane;
            const float x0  = q_values[2 * grp];
            const float x1  = q_values[2 * grp + 1];
            const float mx  = warp_max(fmaxf(fabsf(x0), fabsf(x1)), FullMask);
            const float qs  = mx > 0.0f ? mx / 127.0f : 0.0f;
            const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
            causal_store_query_code(q_i8, row, d0, kv_cache_int8_quant_code(x0, inv));
            causal_store_query_code(q_i8, row, d0 + 32, kv_cache_int8_quant_code(x1, inv));
            if (gid == (r & 7)) { q_scale_r[r >> 3][grp] = qs; }
        }
    }

    const auto stage_base = [&](int stage) {
        return smem_raw + Shape::kQBytes + stage * kInt8PromptFastStageBytes;
    };

    // One tile is one physical page of this KV head: 16 KiB of K codes, 16 KiB of V codes and
    // their scales, each contiguous in the cache. Keys past the CTA's last visible key are
    // zero-filled so masked columns stay finite.
    const auto issue_tile = [&](int kb) {
        unsigned char* base = stage_base(kb & 1);
        std::int8_t* k_s    = reinterpret_cast<std::int8_t*>(base);
        std::int8_t* v_s    = k_s + kInt8PromptFastTileBytes;
        __half* ks_s        = reinterpret_cast<__half*>(base + KScaleOffset);
        __half* vs_s        = ks_s + Bc * Groups;
        const int page      = block_table[kb];
        const int valid     = min(Bc, max_query_abs + 1 - kb * Bc);
        const std::int64_t code_base = kv_cache_int8_quant_code_index<Geometry>(page, kv_head, 0, 0);
#pragma unroll
        for (int i = 0; i < kInt8PromptFastTileBytes / 16 / Threads; ++i) {
            const int chunk        = tid + i * Threads;
            const int key          = chunk >> 4;
            const int c            = chunk & 15;
            const int dst          = key * D + ((c ^ (key & 7)) << 4);
            const int bytes        = key < valid ? 16 : 0;
            const std::int64_t src = code_base + key * D + c * 16;
            cp_async_zfill<16, Cache::cg>(k_s + dst, cache_k + src, bytes);
            cp_async_zfill<16, Cache::cg>(v_s + dst, cache_v + src, bytes);
        }
        if (tid < Bc) {
            cp_async_zfill<8>(ks_s + tid * Groups,
                              cache_k_scale +
                                  kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, 0, tid),
                              tid < valid ? 8 : 0);
        } else if (tid < 2 * Bc) {
            const int key = tid - Bc;
            cp_async_zfill<8>(vs_s + key * Groups,
                              cache_v_scale +
                                  kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, 0, key),
                              key < valid ? 8 : 0);
        }
        cp_commit();
    };

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_koff   = ((lane >> 3) & 1) << 3;
    const int row_base = warp * 16;

    // Rows this warp owns, for warp-uniform tile skipping and mask elision.
    const bool warp_active  = row_base < rows;
    const int warp_min_qabs = base_pos + q0 + row_base;
    const int warp_max_qabs = base_pos + q0 + min(row_base + 15, rows - 1);
    const int row0          = row_base + gid;
    const int row1          = row0 + 8;
    const int qabs0         = row0 < rows ? base_pos + q0 + row0 : -1;
    const int qabs1         = row1 < rows ? base_pos + q0 + row1 : -1;

    // [16-dimension block][even/odd n8 tile]: 128 floats per lane.
    float acc[DBlocks][2][4];
#pragma unroll
    for (int b = 0; b < DBlocks; ++b) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[b][p][i] = 0.0f; }
        }
    }
    float running_m0     = -CUDART_INF_F;
    float running_m1     = -CUDART_INF_F;
    float running_l0     = 0.0f;
    float running_l1     = 0.0f;
    const float scale_l2 = scale * Log2E;

    // What QK hands to PV for one tile: the probability A fragments, the row rescale factors, and
    // the exact power of two that keeps the tile's FP16 partials representable.
    unsigned pa[PVKs][4];
    float tile_alpha0 = 0.0f;
    float tile_alpha1 = 0.0f;
    int tile_shift    = 0;
    bool tile_live    = false;

    const auto qk_softmax = [&](int kb) {
        tile_live    = false;
        const int k0 = kb * Bc;
        if (!warp_active || k0 > warp_max_qabs) { return; }
        const unsigned char* base  = stage_base(kb & 1);
        const __nv_bfloat16* k_b16 = reinterpret_cast<const __nv_bfloat16*>(base);
        const __half* ks_s         = reinterpret_cast<const __half*>(base + KScaleOffset);
        const __half* vs_s         = ks_s + Bc * Groups;

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            unsigned af[GroupKc][4];
#pragma unroll
            for (int kk = 0; kk < GroupKc; ++kk) {
                const int acol = (grp * GroupKc + kk) * 16 + a_coloff;
                ldmatrix_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                            smem_addr(&q_b16[(row_base + a_rowoff) * DB16 +
                                             causal_swizzle(row_base + a_rowoff, acol)]));
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int brow = nt * 8 + a_rin;
                    const int bcol = (grp * GroupKc + kk) * 16 + b_koff;
                    unsigned bf[2];
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&k_b16[brow * DB16 + causal_swizzle(brow, bcol)]));
                    mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                           bf[1]);
                }
                const int keya  = nt * 8 + 2 * lid;
                const float ks0 = __half2float(ks_s[keya * Groups + grp]);
                const float ks1 = __half2float(ks_s[(keya + 1) * Groups + grp]);
                const float qs0 = q_scale_r[0][grp];
                const float qs1 = q_scale_r[1][grp];
                score[nt][0]    = __fmaf_rn(qs0 * ks0, static_cast<float>(c0), score[nt][0]);
                score[nt][1]    = __fmaf_rn(qs0 * ks1, static_cast<float>(c1), score[nt][1]);
                score[nt][2]    = __fmaf_rn(qs1 * ks0, static_cast<float>(c2), score[nt][2]);
                score[nt][3]    = __fmaf_rn(qs1 * ks1, static_cast<float>(c3), score[nt][3]);
            }
        }

        const bool full_tile = k0 + Bc - 1 <= warp_min_qabs;
        float bm0            = -CUDART_INF_F;
        float bm1            = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            if (!full_tile) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                score[nt][0]   = key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                score[nt][1]   = key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                score[nt][2]   = key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                score[nt][3]   = key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
            }
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0        = fmaxf(running_m0, bm0);
        const float nm1        = fmaxf(running_m1, bm1);
        const float nm0_scaled = nm0 == -CUDART_INF_F ? 0.0f : nm0 * scale_l2;
        const float nm1_scaled = nm1 == -CUDART_INF_F ? 0.0f : nm1 * scale_l2;
        tile_alpha0            = running_m0 == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx(__fmaf_rn(running_m0, scale_l2, -nm0_scaled));
        tile_alpha1            = running_m1 == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx(__fmaf_rn(running_m1, scale_l2, -nm1_scaled));
        running_m0             = nm0;
        running_m1             = nm1;

        // Probabilities become the PV A fragments directly: k-step j covers keys 16j..16j+15,
        // which are score tiles 2j (keys 2t, 2t+1) and 2j+1 (keys 8+2t, 9+2t).
        float bl0 = 0.0f;
        float bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
            const float p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
            const float p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
            const float p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            pa[nt >> 1][(nt & 1) * 2 + 0] = pack_f16x2(p00, p01);
            pa[nt >> 1][(nt & 1) * 2 + 1] = pack_f16x2(p10, p11);
        }
        // Row sums stay lane-partial; the quad is reduced once after the sweep.
        running_l0 = __fmaf_rn(running_l0, tile_alpha0, bl0);
        running_l1 = __fmaf_rn(running_l1, tile_alpha1, bl1);

        static_assert(Bc * Groups % (32 * 8) == 0);
        __half2 vmax2 = __float2half2_rn(0.0f);
#pragma unroll
        for (int part = 0; part < Bc * Groups / (32 * 8); ++part) {
            const uint4 v8 = load_vec<uint4>(&vs_s[8 * (lane + 32 * part)]);
            vmax2          = __hmax2(vmax2, __hmax2(__hmax2(__habs2(load_vec<__half2>(&v8.x)),
                                                            __habs2(load_vec<__half2>(&v8.y))),
                                                    __hmax2(__habs2(load_vec<__half2>(&v8.z)),
                                                            __habs2(load_vec<__half2>(&v8.w)))));
        }
        const float vmax = warp_max(fmaxf(__low2float(vmax2), __high2float(vmax2)), FullMask);
        tile_shift       = 0;
        if (vmax > kInt8PromptFastF16PartialScaleLimit) {
            while (ldexpf(vmax, -tile_shift) > kInt8PromptFastF16PartialScaleLimit) { ++tile_shift; }
        }
        tile_live = true;
    };

    const auto pv = [&](int kb) {
        if (!tile_live) { return; }
        const std::int8_t* v_s =
            reinterpret_cast<const std::int8_t*>(stage_base(kb & 1)) + kInt8PromptFastTileBytes;
        const __half* vs_s =
            reinterpret_cast<const __half*>(stage_base(kb & 1) + KScaleOffset) + Bc * Groups;
        const __half2 mul   = __float2half2_rn(ldexpf(1.0f, -tile_shift));
        const float unscale = ldexpf(1.0f, tile_shift);
#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            // Group scales of the keys each lane supplies: (2t, 2t+1) and (8+2t, 9+2t) per k-step.
            unsigned vsc[PVKs][2];
#pragma unroll
            for (int j = 0; j < PVKs; ++j) {
                const int key = j * 16 + 2 * lid;
                __half2 lo =
                    __halves2half2(vs_s[key * Groups + grp], vs_s[(key + 1) * Groups + grp]);
                __half2 hi =
                    __halves2half2(vs_s[(key + 8) * Groups + grp], vs_s[(key + 9) * Groups + grp]);
                if (tile_shift != 0) {
                    lo = __hmul2(lo, mul);
                    hi = __hmul2(hi, mul);
                }
                vsc[j][0] = load_vec<unsigned>(&lo);
                vsc[j][1] = load_vec<unsigned>(&hi);
            }
#pragma unroll
            for (int pass = 0; pass < GroupDBlocks / PassDBlocks; ++pass) {
                const int db0 = grp * GroupDBlocks + pass * PassDBlocks;
                unsigned h[PassDBlocks][2][2];
#pragma unroll
                for (int b = 0; b < PassDBlocks; ++b) {
#pragma unroll
                    for (int p = 0; p < 2; ++p) { h[b][p][0] = h[b][p][1] = 0u; }
                }
#pragma unroll
                for (int j = 0; j < PVKs; ++j) {
#pragma unroll
                    for (int q2 = 0; q2 < PassDBlocks / 2; ++q2) {
                        // Matrices: keys 16j+0..7 and 16j+8..15 of d-block db, then of db + 1.
                        const int db    = db0 + 2 * q2;
                        const int key   = j * 16 + ((a_mat & 1) << 3) + a_rin;
                        const int chunk = db + (a_mat >> 1);
                        unsigned r[4];
                        ldmatrix_x4_t(r[0], r[1], r[2], r[3],
                                      smem_addr(&v_s[key * D + ((chunk ^ (key & 7)) << 4)]));
#pragma unroll
                        for (int b = 0; b < 2; ++b) {
                            unsigned even_lo, odd_lo, even_hi, odd_hi;
                            int8_prompt_fast_decode_v_pair(r[2 * b], vsc[j][0], even_lo, odd_lo);
                            int8_prompt_fast_decode_v_pair(r[2 * b + 1], vsc[j][1], even_hi,
                                                           odd_hi);
                            unsigned (&he)[2] = h[2 * q2 + b][0];
                            unsigned (&ho)[2] = h[2 * q2 + b][1];
                            mma_f16_acc16(he[0], he[1], pa[j][0], pa[j][1], pa[j][2], pa[j][3],
                                          even_lo, even_hi);
                            mma_f16_acc16(ho[0], ho[1], pa[j][0], pa[j][1], pa[j][2], pa[j][3],
                                          odd_lo, odd_hi);
                        }
                    }
                }
#pragma unroll
                for (int b = 0; b < PassDBlocks; ++b) {
#pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        float (&a)[4] = acc[db0 + b][p];
                        float2 r0     = __half22float2(load_vec<__half2>(&h[b][p][0]));
                        float2 r1     = __half22float2(load_vec<__half2>(&h[b][p][1]));
                        if (tile_shift != 0) {
                            r0.x *= unscale;
                            r0.y *= unscale;
                            r1.x *= unscale;
                            r1.y *= unscale;
                        }
                        a[0] = __fmaf_rn(a[0], tile_alpha0, r0.x);
                        a[1] = __fmaf_rn(a[1], tile_alpha0, r0.y);
                        a[2] = __fmaf_rn(a[2], tile_alpha1, r1.x);
                        a[3] = __fmaf_rn(a[3], tile_alpha1, r1.y);
                    }
                }
            }
        }
    };

    // Every warp publishes one tile per barrier; the next tile's copy overlaps this tile's math.
    issue_tile(first_block);
#pragma unroll 1
    for (int kb = first_block; kb < end_block; ++kb) {
        cp_wait<0>();
        __syncthreads();
        if (kb + 1 < end_block) { issue_tile(kb + 1); }
        qk_softmax(kb);
        pv(kb);
    }

    running_l0 = warp_sum<4>(running_l0, FullMask);
    running_l1 = warp_sum<4>(running_l1, FullMask);
    // A row's last piece; an inactive row (past the valid tokens) is stored by piece 0 as zeros.
    const int last0     = qabs0 < 0 ? 0 : qabs0 >> kInt8PromptSplitShift;
    const int last1     = qabs1 < 0 ? 0 : qabs1 >> kInt8PromptSplitShift;
    const bool direct0  = last0 == 0;
    const bool direct1  = last1 == 0;
    const bool partial0 = !direct0 && split <= last0;
    const bool partial1 = !direct1 && split <= last1;
    // Even n8 tiles hold dimensions 16b + 2n and odd tiles 16b + 2n + 1, so lane (g, t) owns the
    // four contiguous dimensions 16b + 4t .. 16b + 4t + 3 of its two rows.
    if (split == 0 && (direct0 || direct1)) {
        const float inv_l0 = running_l0 > 0.0f ? __frcp_rn(running_l0) : 0.0f;
        const float inv_l1 = running_l1 > 0.0f ? __frcp_rn(running_l1) : 0.0f;
#pragma unroll
        for (int b = 0; b < DBlocks; ++b) {
            const int d0 = b * 16 + 4 * lid;
            if (direct0) {
                store_row(row0, d0, acc[b][0][0] * inv_l0, acc[b][1][0] * inv_l0,
                          acc[b][0][1] * inv_l0, acc[b][1][1] * inv_l0);
            }
            if (direct1) {
                store_row(row1, d0, acc[b][0][2] * inv_l1, acc[b][1][2] * inv_l1,
                          acc[b][0][3] * inv_l1, acc[b][1][3] * inv_l1);
            }
        }
    }
    // A longer row's piece: the unnormalized sums, the row's maximum score and its probability sum.
    const auto store_partial = [&](int row, int d0, float v0, float v1, float v2, float v3) {
        const int token = q0 + row;
        store_vec(&partial.acc[causal_partial_index<Geometry>(q_head, d0, token, split, width)],
                  make_float4(v0, v1, v2, v3));
    };
    if (partial0 || partial1) {
#pragma unroll
        for (int b = 0; b < DBlocks; ++b) {
            const int d0 = b * 16 + 4 * lid;
            if (partial0) {
                store_partial(row0, d0, acc[b][0][0], acc[b][1][0], acc[b][0][1], acc[b][1][1]);
            }
            if (partial1) {
                store_partial(row1, d0, acc[b][0][2], acc[b][1][2], acc[b][0][3], acc[b][1][3]);
            }
        }
        if (lid == 0) {
            if (partial0) {
                const auto stat = causal_stat_index<Geometry>(q_head, q0 + row0, split, width);
                partial.maximum[stat] = running_m0;
                partial.sum[stat]     = running_l0;
            }
            if (partial1) {
                const auto stat = causal_stat_index<Geometry>(q_head, q0 + row1, split, width);
                partial.maximum[stat] = running_m1;
                partial.sum[stat]     = running_l1;
            }
        }
    }
}

// Folds a longer row's pieces in piece order: their largest maximum M, then each piece's sums
// weighted by exp2((m_s - M) * scale * log2 e) as the sweep rescales a tile, and the row divided
// by the weighted probability sum. One warp per (token, head); rows of one piece were stored
// directly. Every launch of a row takes this fold, so its bytes follow its own position only.
template <typename Geometry, typename Metadata>
__global__ __launch_bounds__(256) void int8_prompt_fast_merge_kernel(
    CausalPartialView partial, Metadata metadata, const std::int32_t* __restrict__ positions,
    float scale, __nv_bfloat16* __restrict__ out, std::int32_t width) {
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int linear = static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5);
    const int head   = linear % Geometry::QHeads;
    const int token  = linear / Geometry::QHeads;
    if (token >= width || token >= metadata.valid_tokens(width)) { return; }
    const int last   = positions[0] + token;
    const int pieces = (last >> kInt8PromptSplitShift) + 1;
    if (pieces == 1) { return; }
    const float scale_l2 = scale * kLog2E;
    float maximum        = -CUDART_INF_F;
    for (int s = 0; s < pieces; ++s) {
        maximum =
            fmaxf(maximum, partial.maximum[causal_stat_index<Geometry>(head, token, s, width)]);
    }
    const float maximum_scaled = maximum * scale_l2;
    float sum                  = 0.0f;
    float values[8]{};
    for (int s = 0; s < pieces; ++s) {
        const auto stat = causal_stat_index<Geometry>(head, token, s, width);
        const float w   = exp2_approx(__fmaf_rn(partial.maximum[stat], scale_l2, -maximum_scaled));
        sum             = __fmaf_rn(partial.sum[stat], w, sum);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float a =
                partial.acc[causal_partial_index<Geometry>(head, lane + 32 * i, token, s, width)];
            values[i] = __fmaf_rn(a, w, values[i]);
        }
    }
    const float inverse = sum > 0.0f ? __frcp_rn(sum) : 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[causal_q_index<Geometry>(head, lane + 32 * i, token)] =
            __float2bfloat16(values[i] * inverse);
    }
}

} // namespace ninferno::ops::detail
