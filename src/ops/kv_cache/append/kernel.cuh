#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/append/geometry.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninferno::ops {

template <typename Geometry>
__device__ __forceinline__ void
kv_cache_append_full_fp8_row(const __nv_bfloat16* __restrict__ k,
                             const __nv_bfloat16* __restrict__ v,
                             std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
                             __half* __restrict__ scale_k, __half* __restrict__ scale_v, int token,
                             int kv_head, int physical_page, int page_off, int lane) {
    constexpr unsigned FullMask = 0xffffffffU;
    float values[8];
    float local_absmax = 0.0F;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        values[r]   = __bfloat162float(k[kv_cache_fp8_src_index<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (float value : values) { local_absmax = fmaxf(local_absmax, fabsf(value)); }
    const auto k_quant = kv_cache_fp8_quant_params(warp_max(local_absmax, FullMask));
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        cache_k[kv_cache_fp8_code_index<Geometry>(physical_page, kv_head, d, page_off)] =
            kv_cache_fp8_quant_code(values[r], k_quant.inverse_scale);
    }
    if (lane == 0) {
        scale_k[kv_cache_fp8_scale_index<Geometry>(physical_page, kv_head, page_off)] =
            k_quant.scale;
    }

    local_absmax = 0.0F;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d  = lane + 32 * r;
        values[r]    = __bfloat162float(v[kv_cache_fp8_src_index<Geometry>(kv_head, d, token)]);
        local_absmax = fmaxf(local_absmax, fabsf(values[r]));
    }
    const auto v_quant = kv_cache_fp8_quant_params(warp_max(local_absmax, FullMask));
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        cache_v[kv_cache_fp8_code_index<Geometry>(physical_page, kv_head, d, page_off)] =
            kv_cache_fp8_quant_code(values[r], v_quant.inverse_scale);
    }
    if (lane == 0) {
        scale_v[kv_cache_fp8_scale_index<Geometry>(physical_page, kv_head, page_off)] =
            v_quant.scale;
    }
}

template <typename Geometry, typename Metadata>
__global__ void kv_cache_append_full_bf16_kernel(const __nv_bfloat16* __restrict__ k,
                                                 const __nv_bfloat16* __restrict__ v,
                                                 const std::int32_t* __restrict__ positions,
                                                 Metadata metadata,
                                                 __nv_bfloat16* __restrict__ cache_k,
                                                 __half* __restrict__ cache_v, std::int32_t width) {
    constexpr int VecElems = 8;
    const int tokens       = metadata.valid_tokens(width);
    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t n   = static_cast<std::int64_t>(tokens) * Geometry::KVHeads *
                           (kKVCacheAppendFullHeadDim / VecElems);
    if (idx >= n) return;

    const int vec      = static_cast<int>(idx % (kKVCacheAppendFullHeadDim / VecElems));
    const int tmp      = static_cast<int>(idx / (kKVCacheAppendFullHeadDim / VecElems));
    const int kv_head  = tmp % Geometry::KVHeads;
    const int token    = tmp / Geometry::KVHeads;
    const int d        = vec * VecElems;
    const int position = positions[0] + token;
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    const std::int64_t src_off =
        static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kKVCacheAppendFullHeadDim) *
                                           (kv_head + Geometry::KVHeads * token);
    const int4 k_value = load_vec<int4>(&k[src_off]);
    const int4 v_value = bf16x8_bits_to_f16x8_bits(load_vec<int4>(&v[src_off]));
    physical_page      = __shfl_sync(0xffffffffu, physical_page, 0);
    const std::int64_t cache_off =
        paged_kv_element_offset<kKVCacheAppendFullHeadDim, Geometry::KVHeads>(
            physical_page, kv_head, position & kPagedKVPageMask, d);
    store_vec(&cache_k[cache_off], k_value);
    store_vec(&cache_v[cache_off], v_value);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_fp8_kernel(const __nv_bfloat16* __restrict__ k,
                                         const __nv_bfloat16* __restrict__ v,
                                         const std::int32_t* __restrict__ positions,
                                         Metadata metadata, std::uint8_t* __restrict__ cache_k,
                                         std::uint8_t* __restrict__ cache_v,
                                         __half* __restrict__ scale_k, __half* __restrict__ scale_v,
                                         std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads;
    if (unit >= units) return;

    const int kv_head               = unit % Geometry::KVHeads;
    const int token                 = unit / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    physical_page                   = __shfl_sync(FullMask, physical_page, 0);
    kv_cache_append_full_fp8_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token, kv_head,
                                           physical_page, position & kPagedKVPageMask, lane);
}

// Independent request rows share physical planes, but select separate input and table rows.
template <typename Geometry, bool Masked>
__launch_bounds__(256) __global__
    void kv_cache_append_full_fp8_batch_kernel(const __nv_bfloat16* k, const __nv_bfloat16* v,
                                               const std::int32_t* positions,
                                               PagedKVBatchMetadata<Masked> metadata,
                                               std::uint8_t* cache_k, std::uint8_t* cache_v,
                                               __half* scale_k, __half* scale_v,
                                               std::int32_t width) {
    const int batch = blockIdx.z;
    metadata.table_rows += batch;
    if constexpr (Masked) metadata.valid_columns += batch;
    const int tokens = metadata.valid_tokens(width);
    const int unit   = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (unit >= tokens * Geometry::KVHeads) return;
    const int token         = unit / Geometry::KVHeads;
    const int kv_head       = unit % Geometry::KVHeads;
    const int lane          = threadIdx.x & 31;
    const auto input_offset = static_cast<std::int64_t>(batch) * width * 256 * Geometry::KVHeads;
    const int position      = positions[batch * width] + token;
    const auto* table       = metadata.block_table();
    int page                = lane == 0 ? paged_kv_physical_page(table, position) : 0;
    page                    = __shfl_sync(0xffffffffU, page, 0);
    kv_cache_append_full_fp8_row<Geometry>(k + input_offset, v + input_offset, cache_k, cache_v,
                                           scale_k, scale_v, token, kv_head, page,
                                           position & kPagedKVPageMask, lane);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_fp8_page_kernel(const __nv_bfloat16* __restrict__ k,
                                              const __nv_bfloat16* __restrict__ v,
                                              const std::int32_t* __restrict__ positions,
                                              Metadata metadata, std::uint8_t* __restrict__ cache_k,
                                              std::uint8_t* __restrict__ cache_v,
                                              __half* __restrict__ scale_k,
                                              __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int kv_head           = static_cast<int>(blockIdx.y);
    const int tile_delta        = static_cast<int>(blockIdx.x);
    const int base_position     = positions[0];
    const int tile_position     = (base_position / TokensPerTile + tile_delta) * TokensPerTile;
    const int logical_page      = tile_position >> kPagedKVPageShift;
    const int token_begin       = max(0, tile_position - base_position);
    const int token_end         = min(tokens, tile_position + TokensPerTile - base_position);
    if (token_begin >= token_end) return;

    const int token = token_begin + warp;
    if (token >= token_end) return;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? block_table[logical_page] : 0;
    physical_page                   = __shfl_sync(FullMask, physical_page, 0);
    const int position              = base_position + token;
    kv_cache_append_full_fp8_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token, kv_head,
                                           physical_page, position & kPagedKVPageMask, lane);
}

template <typename Geometry, typename Metadata, bool MultiBatch = false>
__launch_bounds__(256) __global__
    void kv_cache_append_full_i8_kernel(const __nv_bfloat16* __restrict__ k,
                                        const __nv_bfloat16* __restrict__ v,
                                        const std::int32_t* __restrict__ positions,
                                        Metadata metadata, std::int8_t* __restrict__ cache_k,
                                        std::int8_t* __restrict__ cache_v,
                                        __half* __restrict__ scale_k, __half* __restrict__ scale_v,
                                        std::int32_t width) {
    if constexpr (MultiBatch) {
        const int batch = blockIdx.z;
        metadata.table_rows += batch;
        if (metadata.valid_columns) metadata.valid_columns += batch;
        const auto offset = static_cast<std::int64_t>(batch) * width * 256 * Geometry::KVHeads;
        k += offset;
        v += offset;
        positions += batch * width;
    }
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads;
    if (unit >= units) return;

    const int kv_head               = unit % Geometry::KVHeads;
    const int token                 = unit / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int page                        = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    const int page_off              = position & kPagedKVPageMask;
    page                            = __shfl_sync(FullMask, page, 0);

    float k_values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        k_values[r] =
            __bfloat162float(k[kv_cache_int8_quant_src_index<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(k_values, lane);

#pragma unroll
    for (int group = 0; group < kKVCacheInt8Groups; ++group) {
        const int d0                 = group * kKVCacheInt8Group + lane;
        const int d1                 = d0 + 32;
        const float k0               = k_values[2 * group];
        const float k1               = k_values[2 * group + 1];
        const std::int64_t src0      = kv_cache_int8_quant_src_index<Geometry>(kv_head, d0, token);
        const std::int64_t src1      = kv_cache_int8_quant_src_index<Geometry>(kv_head, d1, token);
        const float v0               = __bfloat162float(v[src0]);
        const float v1               = __bfloat162float(v[src1]);
        const float k_abs            = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
        const float v_abs            = warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMask);
        const auto k_quant           = kv_cache_int8_quant_params(k_abs);
        const auto v_quant           = kv_cache_int8_quant_params(v_abs);
        const std::int64_t code_base = kv_cache_int8_quant_code_index<Geometry>(
            page, kv_head, group * kKVCacheInt8Group, page_off);
        cache_k[code_base + lane]      = kv_cache_int8_quant_code(k0, k_quant.inverse_scale);
        cache_k[code_base + lane + 32] = kv_cache_int8_quant_code(k1, k_quant.inverse_scale);
        cache_v[code_base + lane]      = kv_cache_int8_quant_code(v0, v_quant.inverse_scale);
        cache_v[code_base + lane + 32] = kv_cache_int8_quant_code(v1, v_quant.inverse_scale);
        if (lane == 0) {
            const std::int64_t scale_off =
                kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, group, page_off);
            scale_k[scale_off] = k_quant.scale;
            scale_v[scale_off] = v_quant.scale;
        }
    }
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_i8_page_kernel(const __nv_bfloat16* __restrict__ k,
                                             const __nv_bfloat16* __restrict__ v,
                                             const std::int32_t* __restrict__ positions,
                                             Metadata metadata, std::int8_t* __restrict__ cache_k,
                                             std::int8_t* __restrict__ cache_v,
                                             __half* __restrict__ scale_k,
                                             __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int kv_head           = static_cast<int>(blockIdx.y);
    const int tile_delta        = static_cast<int>(blockIdx.x);
    const int base_position     = positions[0];
    const int tile_position     = (base_position / TokensPerTile + tile_delta) * TokensPerTile;
    const int logical_page      = tile_position >> kPagedKVPageShift;
    const int token_begin       = max(0, tile_position - base_position);
    const int token_end         = min(tokens, tile_position + TokensPerTile - base_position);
    if (token_begin >= token_end) return;

    const int token = token_begin + warp;
    if (token >= token_end) return;

    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? block_table[logical_page] : 0;
    physical_page                   = __shfl_sync(FullMask, physical_page, 0);

    const int position = base_position + token;
    const int page_off = position & kPagedKVPageMask;

    float k_values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        k_values[r] =
            __bfloat162float(k[kv_cache_int8_quant_src_index<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(k_values, lane);

#pragma unroll
    for (int group = 0; group < kKVCacheInt8Groups; ++group) {
        const int d0                 = group * kKVCacheInt8Group + lane;
        const int d1                 = d0 + 32;
        const float k0               = k_values[2 * group];
        const float k1               = k_values[2 * group + 1];
        const std::int64_t src0      = kv_cache_int8_quant_src_index<Geometry>(kv_head, d0, token);
        const std::int64_t src1      = kv_cache_int8_quant_src_index<Geometry>(kv_head, d1, token);
        const float v0               = __bfloat162float(v[src0]);
        const float v1               = __bfloat162float(v[src1]);
        const float k_abs            = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
        const float v_abs            = warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMask);
        const auto k_quant           = kv_cache_int8_quant_params(k_abs);
        const auto v_quant           = kv_cache_int8_quant_params(v_abs);
        const std::int64_t code_base = kv_cache_int8_quant_code_index<Geometry>(
            physical_page, kv_head, group * kKVCacheInt8Group, page_off);
        cache_k[code_base + lane]      = kv_cache_int8_quant_code(k0, k_quant.inverse_scale);
        cache_k[code_base + lane + 32] = kv_cache_int8_quant_code(k1, k_quant.inverse_scale);
        cache_v[code_base + lane]      = kv_cache_int8_quant_code(v0, v_quant.inverse_scale);
        cache_v[code_base + lane + 32] = kv_cache_int8_quant_code(v1, v_quant.inverse_scale);
        if (lane == 0) {
            const std::int64_t scale_offset =
                kv_cache_int8_quant_scale_index<Geometry>(physical_page, kv_head, group, page_off);
            scale_k[scale_offset] = k_quant.scale;
            scale_v[scale_offset] = v_quant.scale;
        }
    }
}

inline constexpr int kKVCacheAppendPrefixHeadDim = 128;
inline constexpr int kKVCacheAppendPrefixHeads   = 8;
inline constexpr int kKVCacheAppendPrefixPage    = 64;

__device__ __forceinline__ void kv_cache_append_prefix_copy_cyclic_unit(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ cache_k, __half* __restrict__ cache_v, int token, int unit_in_token,
    int slot, int padded_capacity) {
    constexpr int ElementsPerUnit = 8;
    constexpr int UnitsPerHead    = kKVCacheAppendPrefixHeadDim / ElementsPerUnit;
    const int kv_head             = unit_in_token / UnitsPerHead;
    const int d                   = (unit_in_token - kv_head * UnitsPerHead) * ElementsPerUnit;
    const std::int64_t src =
        static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kKVCacheAppendPrefixHeadDim) *
                                           (kv_head + kKVCacheAppendPrefixHeads * token);
    const std::int64_t dst = static_cast<std::int64_t>(d) +
                             static_cast<std::int64_t>(kKVCacheAppendPrefixHeadDim) *
                                 (slot + static_cast<std::int64_t>(padded_capacity) * kv_head);

    const int4 key   = *reinterpret_cast<const int4*>(&k[src]);
    const int4 value = bf16x8_bits_to_f16x8_bits(*reinterpret_cast<const int4*>(&v[src]));
    *reinterpret_cast<int4*>(&cache_k[dst]) = key;
    *reinterpret_cast<int4*>(&cache_v[dst]) = value;
}

template <int HeadDim>
__device__ __forceinline__ void kv_cache_append_prefix_copy_paged_unit(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ cache_k, __half* __restrict__ cache_v, int token, int unit_in_token,
    int page_offset, int physical_page, int physical_pages) {
    constexpr int ElementsPerUnit = 16;
    constexpr int UnitsPerHead    = HeadDim / ElementsPerUnit;
    const int kv_head             = unit_in_token / UnitsPerHead;
    const int d                   = (unit_in_token - kv_head * UnitsPerHead) * ElementsPerUnit;
    const std::int64_t src        = static_cast<std::int64_t>(d) +
                             static_cast<std::int64_t>(HeadDim) *
                                 (kv_head + kKVCacheAppendPrefixHeads * token);
    const std::int64_t dst =
        static_cast<std::int64_t>(d) +
        static_cast<std::int64_t>(HeadDim) *
            (page_offset + kKVCacheAppendPrefixPage * (physical_page + physical_pages * kv_head));

    const int4 k0 = *reinterpret_cast<const int4*>(&k[src]);
    const int4 v0 = bf16x8_bits_to_f16x8_bits(*reinterpret_cast<const int4*>(&v[src]));
    *reinterpret_cast<int4*>(&cache_k[dst]) = k0;
    *reinterpret_cast<int4*>(&cache_v[dst]) = v0;
    const int4 k1                           = *reinterpret_cast<const int4*>(&k[src + 8]);
    const int4 v1 = bf16x8_bits_to_f16x8_bits(*reinterpret_cast<const int4*>(&v[src + 8]));
    *reinterpret_cast<int4*>(&cache_k[dst + 8]) = k1;
    *reinterpret_cast<int4*>(&cache_v[dst + 8]) = v1;
}

template <int Capacity, int Threads>
__global__ void kv_cache_append_prefix_cyclic_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ counts,
    const std::int32_t* __restrict__ lanes, __nv_bfloat16* __restrict__ cache_k,
    __half* __restrict__ cache_v, int min_count, int max_count, int width, int padded_capacity) {
    static_assert(Capacity == 2048 || Capacity == 4096);
    static_assert((Capacity & (Capacity - 1)) == 0);
    constexpr int UnitsPerToken = kKVCacheAppendPrefixHeads * kKVCacheAppendPrefixHeadDim / 8;
    const int unit              = static_cast<int>(blockIdx.x) * Threads + threadIdx.x;
    const int token             = unit / UnitsPerToken;
    const int unit_in_token     = unit % UnitsPerToken;
    const int batch             = static_cast<int>(blockIdx.y);
    const int count             = counts[batch];
    if (count < min_count || count > max_count || token >= count) return;

    constexpr std::int64_t ElementsPerToken =
        kKVCacheAppendPrefixHeadDim * kKVCacheAppendPrefixHeads;
    const std::int64_t input_offset = ElementsPerToken * width * batch;
    const std::int64_t cache_offset =
        ElementsPerToken * static_cast<std::int64_t>(padded_capacity) * lanes[batch];
    k += input_offset;
    v += input_offset;
    positions += static_cast<std::int64_t>(width) * batch;
    cache_k += cache_offset;
    cache_v += cache_offset;

    const int position = positions[token];
    const int slot     = position & (Capacity - 1);
    kv_cache_append_prefix_copy_cyclic_unit(k, v, cache_k, cache_v, token, unit_in_token, slot,
                                            padded_capacity);
}

// Int8Group64 at D=128 (the DFlash drafters' full layers): a unit is 16 values of one head, four
// consecutive units are one G64 group, so the group's absmax is two xor shuffles; the stored scale
// is FP16-RNE(absmax / 127) and the codes use the reciprocal of that represented value, as the
// target's int8 codec does.
__device__ __forceinline__ void
kv_cache_append_prefix_i8_unit(const __nv_bfloat16* __restrict__ src, std::int8_t* __restrict__ dst,
                               __half* __restrict__ scale_dst, bool active, bool writes_scale) {
    float x[16];
    if (active) {
        const int4 a             = *reinterpret_cast<const int4*>(src);
        const int4 b             = *reinterpret_cast<const int4*>(src + 8);
        const __nv_bfloat162* ha = reinterpret_cast<const __nv_bfloat162*>(&a);
        const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 fa  = __bfloat1622float2(ha[i]);
            const float2 fb  = __bfloat1622float2(hb[i]);
            x[2 * i]         = fa.x;
            x[2 * i + 1]     = fa.y;
            x[8 + 2 * i]     = fb.x;
            x[8 + 2 * i + 1] = fb.y;
        }
    } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) x[i] = 0.0f;
    }
    float m = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) m = fmaxf(m, fabsf(x[i]));
    m                   = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    m                   = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    const __half scale  = __float2half_rn(m > 0.0f ? m / 127.0f : 0.0f);
    const float shown   = __half2float(scale);
    const float inverse = shown > 0.0f ? 1.0f / shown : 0.0f;
    if (!active) return;
    alignas(16) std::int8_t codes[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int q = inverse == 0.0f ? 0 : max(-127, min(127, __float2int_rn(x[i] * inverse)));
        codes[i]    = static_cast<std::int8_t>(q);
    }
    *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(codes);
    if (writes_scale) *scale_dst = scale;
}

__global__ void kv_cache_append_prefix_paged_i8_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ counts,
    const std::int32_t* __restrict__ table_rows, std::int8_t* __restrict__ cache_k,
    std::int8_t* __restrict__ cache_v, __half* __restrict__ scale_k, __half* __restrict__ scale_v,
    const std::int32_t* __restrict__ block_tables, int physical_pages, int logical_pages,
    int min_count, int max_count, int width) {
    constexpr int HeadDim        = 128;
    constexpr int Groups         = HeadDim / 64;
    constexpr int UnitsPerHead   = HeadDim / 16;
    constexpr int UnitsPerToken  = kKVCacheAppendPrefixHeads * UnitsPerHead;
    constexpr int TokensPerBlock = 256 / UnitsPerToken;
    const int batch              = static_cast<int>(blockIdx.y);
    const int count              = counts[batch];
    if (count < min_count || count > max_count) return;

    constexpr std::int64_t ElementsPerToken = HeadDim * kKVCacheAppendPrefixHeads;
    const std::int64_t input_offset         = ElementsPerToken * width * batch;
    k += input_offset;
    v += input_offset;
    positions += static_cast<std::int64_t>(width) * batch;
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(logical_pages) * table_rows[batch];

    const int local         = static_cast<int>(threadIdx.x);
    const int local_token   = local / UnitsPerToken;
    const int unit_in_token = local - local_token * UnitsPerToken;
    const int lane          = local & 31;
    const int token         = static_cast<int>(blockIdx.x) * TokensPerBlock + local_token;
    int position            = 0;
    int physical_page       = 0;
    if (lane == 0 && token < count) {
        position      = positions[token];
        physical_page = block_table[position >> 6];
    }
    position      = __shfl_sync(0xffffffffu, position, 0);
    physical_page = __shfl_sync(0xffffffffu, physical_page, 0);
    // Every lane takes part in the group shuffles; only live tokens store.
    const bool active  = token < count;
    const int kv_head  = unit_in_token / UnitsPerHead;
    const int d        = (unit_in_token - kv_head * UnitsPerHead) * 16;
    const int page_off = position & (kKVCacheAppendPrefixPage - 1);
    const std::int64_t src =
        static_cast<std::int64_t>(d) +
        static_cast<std::int64_t>(HeadDim) * (kv_head + kKVCacheAppendPrefixHeads * token);
    const std::int64_t slot =
        page_off + static_cast<std::int64_t>(kKVCacheAppendPrefixPage) *
                       (physical_page + static_cast<std::int64_t>(physical_pages) * kv_head);
    const std::int64_t dst       = d + HeadDim * slot;
    const std::int64_t scale_off = (d >> 6) + Groups * slot;
    const bool writes_scale      = (d & 63) == 0;
    kv_cache_append_prefix_i8_unit(k + (active ? src : 0), cache_k + dst, scale_k + scale_off,
                                   active, writes_scale);
    kv_cache_append_prefix_i8_unit(v + (active ? src : 0), cache_v + dst, scale_v + scale_off,
                                   active, writes_scale);
}

// Int8Group64 cyclic ring at D=128 (core/cyclic_kv_cache.h): per lane the codes [128] x padded x
// heads, then the FP16 group scales [2] x padded x heads. A token is 64 units of 16 values; four
// consecutive units (lanes) are one G64 group, as in the paged int8 append.
template <int Capacity>
__global__ void kv_cache_append_prefix_cyclic_i8_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ counts,
    const std::int32_t* __restrict__ lanes, std::int8_t* __restrict__ cache_k,
    std::int8_t* __restrict__ cache_v, int min_count, int max_count, int width,
    int padded_capacity) {
    static_assert(Capacity == 2048 || Capacity == 4096);
    constexpr int HeadDim        = kKVCacheAppendPrefixHeadDim;
    constexpr int Heads          = kKVCacheAppendPrefixHeads;
    constexpr int Groups         = HeadDim / 64;
    constexpr int UnitsPerHead   = HeadDim / 16;
    constexpr int UnitsPerToken  = Heads * UnitsPerHead;
    constexpr int TokensPerBlock = 256 / UnitsPerToken;
    const int batch              = static_cast<int>(blockIdx.y);
    const int count              = counts[batch];
    if (count < min_count || count > max_count) return;

    constexpr std::int64_t ElementsPerToken = HeadDim * Heads;
    const std::int64_t input_offset         = ElementsPerToken * width * batch;
    k += input_offset;
    v += input_offset;
    positions += static_cast<std::int64_t>(width) * batch;
    const std::int64_t rows       = static_cast<std::int64_t>(padded_capacity) * Heads;
    const std::int64_t lane_bytes = rows * (HeadDim + 2 * Groups);
    cache_k += lane_bytes * lanes[batch];
    cache_v += lane_bytes * lanes[batch];
    __half* scale_k = reinterpret_cast<__half*>(cache_k + rows * HeadDim);
    __half* scale_v = reinterpret_cast<__half*>(cache_v + rows * HeadDim);

    const int local         = static_cast<int>(threadIdx.x);
    const int local_token   = local / UnitsPerToken;
    const int unit_in_token = local - local_token * UnitsPerToken;
    const int token         = static_cast<int>(blockIdx.x) * TokensPerBlock + local_token;
    // Every lane takes part in the group shuffles; only live tokens store.
    const bool active      = token < count;
    const int slot         = active ? positions[token] & (Capacity - 1) : 0;
    const int kv_head      = unit_in_token / UnitsPerHead;
    const int d            = (unit_in_token - kv_head * UnitsPerHead) * 16;
    const std::int64_t src = static_cast<std::int64_t>(d) +
                             static_cast<std::int64_t>(HeadDim) * (kv_head + Heads * token);
    const std::int64_t row       = slot + static_cast<std::int64_t>(padded_capacity) * kv_head;
    const std::int64_t dst       = d + HeadDim * row;
    const std::int64_t scale_off = (d >> 6) + Groups * row;
    const bool writes_scale      = (d & 63) == 0;
    kv_cache_append_prefix_i8_unit(k + (active ? src : 0), cache_k + dst, scale_k + scale_off,
                                   active, writes_scale);
    kv_cache_append_prefix_i8_unit(v + (active ? src : 0), cache_v + dst, scale_v + scale_off,
                                   active, writes_scale);
}

// A token's units are whole warps (lane 0 reads its position and page): D=128 and D=64 at Hkv=8.
template <int HeadDim>
__global__ void kv_cache_append_prefix_paged_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ counts,
    const std::int32_t* __restrict__ table_rows, __nv_bfloat16* __restrict__ cache_k,
    __half* __restrict__ cache_v, const std::int32_t* __restrict__ block_tables, int physical_pages,
    int logical_pages, int min_count, int max_count, int width) {
    constexpr int UnitsPerToken  = kKVCacheAppendPrefixHeads * HeadDim / 16;
    constexpr int TokensPerBlock = 256 / UnitsPerToken;
    static_assert(TokensPerBlock * UnitsPerToken == 256 && UnitsPerToken % 32 == 0);
    const int batch = static_cast<int>(blockIdx.y);
    const int count = counts[batch];
    if (count < min_count || count > max_count) return;

    constexpr std::int64_t ElementsPerToken = HeadDim * kKVCacheAppendPrefixHeads;
    const std::int64_t input_offset         = ElementsPerToken * width * batch;
    k += input_offset;
    v += input_offset;
    positions += static_cast<std::int64_t>(width) * batch;
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(logical_pages) * table_rows[batch];

    const int local         = static_cast<int>(threadIdx.x);
    const int local_token   = local / UnitsPerToken;
    const int unit_in_token = local - local_token * UnitsPerToken;
    const int lane          = local & 31;
    const int token         = static_cast<int>(blockIdx.x) * TokensPerBlock + local_token;
    int position            = 0;
    int physical_page       = 0;
    if (lane == 0 && token < count) {
        position      = positions[token];
        physical_page = block_table[position >> 6];
    }
    position      = __shfl_sync(0xffffffffu, position, 0);
    physical_page = __shfl_sync(0xffffffffu, physical_page, 0);
    if (token < count) {
        kv_cache_append_prefix_copy_paged_unit<HeadDim>(k, v, cache_k, cache_v, token,
                                                        unit_in_token,
                                               position & (kKVCacheAppendPrefixPage - 1),
                                               physical_page, physical_pages);
    }
}

} // namespace ninferno::ops
