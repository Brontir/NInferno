// Volta (sm_70) prefill attention over the INT8-G64 cache on the vendored split-D flash kernel
// (third_party/sm70_flash_d256: 1Cat-vLLM's D256 kernel as adapted by sm70-attn, vendored through
// ninfer-v100-sm70). 64 query rows per CTA, FP16 tensor-core operands on mma.m8n8k4, FP32
// accumulators for Q.K^T and P.V. The launch follows that fork's split-D launcher
// (src/ops/launcher/gqa_attention_volta_splitd.cu; Apache-2.0, see NOTICE), for the int8 cache.
//
// The call's K/V are appended to the paged cache before this runs (int8_kv_append_attention).
// The visible keys are gathered once per call into contiguous FP16 [key][kv_head][256]
// (dequantized; K stays in the cache's rotated basis). Each block of 1024 queries stages Q to FP16
// in the kernel's [head][row][256] layout, rotated by the same normalized Hadamard transform as
// K, runs the kernel (a three-way key split with an FP32 merge for long prefixes), and converts
// the FP32 output to BF16.
//
// This translation unit alone sees the vendored CuTe headers; CMake scopes their include
// directory and FLASH_NAMESPACE to it, on the sm_70 tree (NINFERNO_VOLTA_SPLITD).

#include "ops/softmax_attention/dense/causal_cache/int8/volta_splitd.h"

#include "core/device_class.h"

#include <algorithm>
#include <stdexcept>

#ifdef NINFERNO_VOLTA_SPLITD

#    include "fattn-sm70-d256-kernel.cuh" // vendored

#    include "core/device.h"
#    include "ops/kernel/paged_kv_address.cuh"
#    include "ops/kv_cache/hadamard_d256.cuh"
#    include "ops/kv_cache/int8_g64_codec.cuh"

#    include <cuda_bf16.h>
#    include <cuda_fp16.h>

namespace ninferno::ops::detail {
namespace {

using SplitDTraits = FLASH_NAMESPACE::Sm70D256SplitDTraits;
using Element      = cutlass::half_t;

constexpr int kHeadDim         = 256;
constexpr int kBlockRows       = 64;   // query rows per CTA
constexpr int kQueryBlock      = 1024; // queries staged per kernel launch
constexpr int kKeyPad          = 256;  // the gathered keys, zero past the visible ones
constexpr int kKeySplitMinimum = 2048; // keys from which a block splits them three ways
constexpr float kLog2E         = 1.4426950408889634f;
static_assert(SplitDTraits::kHeadDim == kHeadDim && SplitDTraits::kBlockM == kBlockRows);

template <class T>
constexpr T round_up(T value, T step) {
    return (value + step - 1) / step * step;
}

// Paged INT8-G64 -> contiguous FP16 [key][kv_head][256], zero past the visible keys.
template <int KvHeads>
__global__ void
gather_kv_kernel(const std::int8_t* __restrict__ k_pages, const std::int8_t* __restrict__ v_pages,
                 const __half* __restrict__ k_scales, const __half* __restrict__ v_scales,
                 const std::int32_t* __restrict__ block_tables,
                 const std::int32_t* __restrict__ table_rows, int logical_pages,
                 __half* __restrict__ k_out, __half* __restrict__ v_out, int keys) {
    const int key = static_cast<int>(blockIdx.x);
    const int h   = static_cast<int>(blockIdx.y);
    const int d   = static_cast<int>(threadIdx.x);
    const std::int64_t dst =
        d + static_cast<std::int64_t>(kHeadDim) * (h + static_cast<std::int64_t>(KvHeads) * key);
    if (key >= keys) {
        k_out[dst] = __float2half(0.0f);
        v_out[dst] = __float2half(0.0f);
        return;
    }
    const std::int32_t* table =
        block_tables + static_cast<std::int64_t>(logical_pages) * table_rows[0];
    const std::int64_t code = paged_kv_element_offset<kHeadDim, KvHeads>(table, h, key, d);
    const std::int64_t scale =
        paged_kv_element_offset<kKVCacheInt8Groups, KvHeads>(table, h, key, d / kKVCacheInt8Group);
    k_out[dst] = __float2half(static_cast<float>(k_pages[code]) * __half2float(k_scales[scale]));
    v_out[dst] = __float2half(static_cast<float>(v_pages[code]) * __half2float(v_scales[scale]));
}

// BF16 q [256, QHeads, T] -> FP16 [QHeads][q_pad][256], rotated into the cache's K basis. One warp
// per (row, head); rows at or past `tokens` are zero (the kernel's Q tile copy is unguarded).
__launch_bounds__(256) __global__
    void stage_q_kernel(const __nv_bfloat16* __restrict__ q, __half* __restrict__ staged,
                        int q_heads, int tokens, int q_pad) {
    constexpr int kWarps = 8;
    const int unit = static_cast<int>(blockIdx.x) * kWarps + (static_cast<int>(threadIdx.x) >> 5);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (unit >= q_pad * q_heads) return;
    const int head      = unit / q_pad;
    const int row       = unit % q_pad;
    __half* destination = staged + (static_cast<std::int64_t>(head) * q_pad + row) * kHeadDim;
    if (row >= tokens) {
#    pragma unroll
        for (int item = 0; item < 8; ++item) destination[lane + 32 * item] = __float2half(0.0f);
        return;
    }
    const __nv_bfloat16* source = q + (static_cast<std::int64_t>(row) * q_heads + head) * kHeadDim;
    float values[8];
#    pragma unroll
    for (int item = 0; item < 8; ++item) values[item] = __bfloat162float(source[lane + 32 * item]);
    normalized_hadamard_d256_inplace(values, lane);
#    pragma unroll
    for (int item = 0; item < 8; ++item)
        destination[lane + 32 * item] = __float2half_rn(values[item]);
}

// FP32 [q_pad][QHeads][256] -> BF16 [T][QHeads][256]: the first `count` elements share the order.
__global__ void convert_out_kernel(const float* __restrict__ in, __nv_bfloat16* __restrict__ out,
                                   std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = __float2bfloat16_rn(in[i]);
}

const auto kDenseKernel =
    FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<Element, false, float, false, false, false>;
const auto kSplitKernel =
    FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<Element, false, float, true, false, false>;

struct Layout {
    std::size_t keys, query, output, partials;
};

Layout layout_for(int query_heads, int kv_heads, int width, std::uint32_t visible_keys) {
    const std::int64_t q_pad = round_up(std::min(width, kQueryBlock), kBlockRows);
    const std::int64_t rows  = q_pad * query_heads;
    const std::int64_t keys =
        round_up(static_cast<std::int64_t>(visible_keys), std::int64_t{kKeyPad});
    return {static_cast<std::size_t>(keys * kv_heads * kHeadDim) * sizeof(__half),
            static_cast<std::size_t>(rows * kHeadDim) * sizeof(__half),
            static_cast<std::size_t>(rows * kHeadDim) * sizeof(float),
            static_cast<std::size_t>(3 * rows * (kHeadDim + 2)) * sizeof(float)};
}

template <int KvHeads>
void gather(const PagedKVBatchLayerView& cache, const Tensor& rows, int visible, __half* k_out,
            __half* v_out, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(round_up(visible, kKeyPad)), KvHeads);
    gather_kv_kernel<KvHeads>
        <<<grid, kHeadDim, 0, stream>>>(static_cast<const std::int8_t*>(cache.k_pages.data),
                                        static_cast<const std::int8_t*>(cache.v_pages.data),
                                        static_cast<const __half*>(cache.k_scale_pages.data),
                                        static_cast<const __half*>(cache.v_scale_pages.data),
                                        static_cast<const std::int32_t*>(cache.block_tables.data),
                                        static_cast<const std::int32_t*>(rows.data),
                                        cache.block_tables.ne[0], k_out, v_out, visible);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool int8_volta_splitd_takes(int query_heads, int kv_heads, int width, int batch,
                             CausalAttentionExecutionEnvelope envelope) {
    if (batch != 1 || width <= 256 || (kv_heads != 2 && kv_heads != 4 && kv_heads != 8) ||
        query_heads % kv_heads != 0 ||
        envelope.min_visible_keys != envelope.max_visible_keys) {
        return false;
    }
    try {
        return !device_class().features.mma_tensor;
    } catch (const std::exception&) { return false; }
}

std::size_t int8_volta_splitd_workspace_bytes(int query_heads, int kv_heads, int max_width,
                                              std::uint32_t max_visible_keys) {
    const Layout l = layout_for(query_heads, kv_heads, max_width, max_visible_keys);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(l.keys, 256);
    (void)layout.alloc_bytes(l.keys, 256);
    (void)layout.alloc_bytes(l.query, 256);
    (void)layout.alloc_bytes(l.output, 256);
    (void)layout.alloc_bytes(l.partials, 256);
    return layout.peak_bytes(1);
}

void int8_volta_splitd(const Tensor& q, const Tensor& rows, float scale,
                       const PagedKVBatchLayerView& cache,
                       CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                       Tensor& out, cudaStream_t stream) {
    const int q_heads  = q.ne[1];
    const int width    = q.ne[2];
    const int kv_heads = cache.num_kv_heads;
    const int visible  = static_cast<int>(envelope.max_visible_keys);
    for (const auto kernel : {kDenseKernel, kSplitKernel}) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        SplitDTraits::kSmemBytes));
    }
    auto scope     = workspace.scope();
    const Layout l = layout_for(q_heads, kv_heads, width, envelope.max_visible_keys);
    auto* keys     = static_cast<__half*>(workspace.alloc_bytes(l.keys, 256).data);
    auto* values   = static_cast<__half*>(workspace.alloc_bytes(l.keys, 256).data);
    auto* q_staged = static_cast<__half*>(workspace.alloc_bytes(l.query, 256).data);
    auto* o_staged = static_cast<float*>(workspace.alloc_bytes(l.output, 256).data);
    auto* partials = static_cast<float*>(workspace.alloc_bytes(l.partials, 256).data);
    switch (kv_heads) {
    case 2:
        gather<2>(cache, rows, visible, keys, values, stream);
        break;
    case 4:
        gather<4>(cache, rows, visible, keys, values, stream);
        break;
    case 8:
        gather<8>(cache, rows, visible, keys, values, stream);
        break;
    default:
        throw std::invalid_argument("INT8 Volta split-D attention: unsupported KV heads");
    }

    const float scale_log2 = scale * kLog2E;
    // Token t of this call sits at position visible - width + t, so a block's rows see the keys
    // [0, base + begin + row].
    const int base = visible - width;
    for (int begin = 0; begin < width; begin += kQueryBlock) {
        const int tokens              = std::min(kQueryBlock, width - begin);
        const int q_pad               = round_up(tokens, kBlockRows);
        const int kv_len              = base + begin + tokens;
        const std::int64_t count_rows = static_cast<std::int64_t>(q_pad) * q_heads;
        const auto* q_begin           = static_cast<const __nv_bfloat16*>(q.data) +
                              static_cast<std::int64_t>(begin) * q_heads * kHeadDim;
        constexpr int kStageWarps = 8;
        stage_q_kernel<<<static_cast<int>((count_rows + kStageWarps - 1) / kStageWarps),
                         kStageWarps * 32, 0, stream>>>(q_begin, q_staged, q_heads, tokens, q_pad);
        CUDA_CHECK(cudaGetLastError());

        const bool split   = kv_len >= kKeySplitMinimum && kv_len > tokens;
        float* partial_out = partials;
        float* partial_max = partial_out + 3 * count_rows * kHeadDim;
        float* partial_sum = partial_max + 3 * count_rows;
        const dim3 grid(static_cast<unsigned>(q_pad / kBlockRows), split ? 3u : 1u,
                        static_cast<unsigned>(q_heads));
        (split ? kSplitKernel
               : kDenseKernel)<<<grid, SplitDTraits::kNThreads, SplitDTraits::kSmemBytes, stream>>>(
            reinterpret_cast<const Element*>(q_staged), reinterpret_cast<const Element*>(keys),
            reinterpret_cast<const Element*>(values), o_staged,
            /*q_batch_stride=*/static_cast<int>(count_rows * kHeadDim),
            /*q_row_stride=*/kHeadDim,
            /*q_head_stride=*/q_pad * kHeadDim,
            /*k_outer_stride=*/0,
            /*k_row_stride=*/kv_heads * kHeadDim,
            /*k_head_stride=*/kHeadDim,
            /*v_outer_stride=*/0,
            /*v_row_stride=*/kv_heads * kHeadDim,
            /*v_head_stride=*/kHeadDim,
            /*query_len=*/q_pad, kv_len, q_heads, kv_heads,
            /*kv_offset=*/kv_len - tokens, scale_log2,
            /*block_table=*/nullptr, /*page_size=*/0, /*block_table_batch_stride=*/0,
            split ? partial_out : nullptr, split ? partial_max : nullptr,
            split ? partial_sum : nullptr);
        CUDA_CHECK(cudaGetLastError());
        if (split) {
            FLASH_NAMESPACE::sm70_d256_splitkv3_merge_kernel<<<static_cast<unsigned>(count_rows),
                                                               kHeadDim, 0, stream>>>(
                partial_out, partial_max, partial_sum, o_staged, count_rows, scale_log2);
            CUDA_CHECK(cudaGetLastError());
        }
        const std::int64_t count = static_cast<std::int64_t>(tokens) * q_heads * kHeadDim;
        convert_out_kernel<<<static_cast<int>((count + 255) / 256), 256, 0, stream>>>(
            o_staged,
            static_cast<__nv_bfloat16*>(out.data) +
                static_cast<std::int64_t>(begin) * q_heads * kHeadDim,
            count);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninferno::ops::detail

#else // the kernel is compiled on the sm_70 tree alone

namespace ninferno::ops::detail {

bool int8_volta_splitd_takes(int, int, int, int, CausalAttentionExecutionEnvelope) {
    return false;
}

std::size_t int8_volta_splitd_workspace_bytes(int, int, int, std::uint32_t) { return 0; }

void int8_volta_splitd(const Tensor&, const Tensor&, float, const PagedKVBatchLayerView&,
                       CausalAttentionExecutionEnvelope, WorkspaceArena&, Tensor&, cudaStream_t) {
    throw std::logic_error("INT8 Volta split-D attention is compiled on the sm_70 tree alone");
}

} // namespace ninferno::ops::detail

#endif
