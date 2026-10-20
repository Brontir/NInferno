// Integer-activation routes for the 27B Q4/Q5 G64 input projections, after ninfer-all's
// (Apache-2.0) src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_a8.cu and
// src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_a8.cu. Each projection is two row-split parents
// over one input, so the activation is quantised once and every launch reads it:
//
//   attention  query_key  Q4 [7168, 5120]: rows [0, 6144) -> q,    [6144, 7168) -> k
//              gate_value Q5 [7168, 5120]: rows [0, 6144) -> gate, [6144, 7168) -> v
//   GDN        qk         Q4 [4096, 5120]: rows [0, 4096)  -> qkv rows [0, 4096)
//              value_z    Q5 [12288, 5120]: rows [0, 6144) -> qkv rows [4096, 10240),
//                                           rows [6144, 12288) -> z
//
// The generic int8 tier runs each parent as its own GEMM with its own quantiser; the Q4 parent is
// where this kernel wins (RTX 3090 24K prefill: 893 against 1015 ms), the Q5 parent is even.

#include "ops/linear_swiglu/q4a8/q4a8_linear_swiglu.h"

#include "core/device.h"
#include "core/per_device.h"
#include "ops/common/rowsplit_a8_mma.cuh"
#include "ops/common/arch_limits.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <array>
#include <mutex>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

namespace a8 = rowsplit_a8;

constexpr std::int32_t kHidden    = 5120;
constexpr std::int32_t kQRows     = 6144;
constexpr std::int32_t kKvRows    = 1024;
constexpr std::int32_t kQkRows    = 4096;
constexpr std::int32_t kValueRows = 6144;
constexpr std::int32_t kZRows     = 6144;
constexpr std::int32_t kQkvRows   = kQkRows + kValueRows;

// One m-tile per warp stages 64 rows, which every row count here divides.
using Rows = a8::ContiguousRows<1>;

constexpr bool is_row_split(QuantLayout layout) { return layout == QuantLayout::RowSplit; }

bool parent_fits(const Weight& w, QType qtype, std::int32_t rows) {
    return w.qtype == qtype && is_row_split(w.layout) && w.n == rows && w.k == kHidden &&
           w.padded_shape[1] == kHidden && w.group == a8::kGroup && w.qdata != nullptr &&
           (qtype != QType::Q5_G64_FP16 || w.qhigh != nullptr) && w.scales != nullptr;
}

// Rows [row_begin, row_begin + row_count) of the weight into dst rows from dst_row_offset.
template <class Codec, int NT>
void launch_range(const Weight& weight, const std::int8_t* codes, const __half* scales,
                  std::int32_t tokens, std::int32_t row_begin, std::int32_t row_count,
                  a8::StoreEpilogue epilogue, cudaStream_t stream) {
    constexpr int BN       = a8::kWarpsN * NT * 8;
    const std::size_t smem = a8::shared_bytes<Codec, NT, Rows>();
    const dim3 grid(row_count / Rows::kRowsPerBlock, tokens / BN);
    auto* kernel = a8::a8_mma_kernel<Codec, kHidden, 1, NT, Rows, a8::StoreEpilogue>;
    if (smem > kStaticSharedBytesPerBlock) {
        static PerDevice<bool> once;
        allow_dynamic_smem(once, kernel, smem);
    }
    kernel<<<grid, a8::kThreads, smem, stream>>>(
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh), static_cast<const __half*>(weight.scales),
        codes, scales, tokens, Rows{row_begin}, epilogue, 0);
    CUDA_CHECK(cudaGetLastError());
}

__nv_bfloat16* bf16(Tensor& t) { return static_cast<__nv_bfloat16*>(t.data); }

// The quantiser writes the activations in the tile's fragment order, so the token tile is chosen
// once per call and every launch of the call uses it.
template <int NT>
void attn_all(const Tensor& x, const Weight& query_key, const Weight& gate_value,
              std::int32_t tokens, std::int8_t* codes, __half* scales, Tensor& q, Tensor& gate,
              Tensor& k, Tensor& v, cudaStream_t stream) {
    constexpr int BN = a8::kWarpsN * NT * 8;
    a8::quantize_activations<kHidden, BN><<<tokens, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data), tokens, codes, scales);
    CUDA_CHECK(cudaGetLastError());
    launch_range<a8::Q4Codec, NT>(query_key, codes, scales, tokens, 0, kQRows,
                                  {bf16(q), kQRows, 0}, stream);
    launch_range<a8::Q4Codec, NT>(query_key, codes, scales, tokens, kQRows, kKvRows,
                                  {bf16(k), kKvRows, 0}, stream);
    launch_range<a8::Q5Codec, NT>(gate_value, codes, scales, tokens, 0, kQRows,
                                  {bf16(gate), kQRows, 0}, stream);
    launch_range<a8::Q5Codec, NT>(gate_value, codes, scales, tokens, kQRows, kKvRows,
                                  {bf16(v), kKvRows, 0}, stream);
}

template <int NT>
void gdn_all(const Tensor& x, const Weight& qk, const Weight& value_z, std::int32_t tokens,
             std::int8_t* codes, __half* scales, Tensor& qkv, Tensor& z, cudaStream_t stream) {
    constexpr int BN = a8::kWarpsN * NT * 8;
    a8::quantize_activations<kHidden, BN><<<tokens, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data), tokens, codes, scales);
    CUDA_CHECK(cudaGetLastError());
    launch_range<a8::Q4Codec, NT>(qk, codes, scales, tokens, 0, kQkRows, {bf16(qkv), kQkvRows, 0},
                                  stream);
    launch_range<a8::Q5Codec, NT>(value_z, codes, scales, tokens, 0, kValueRows,
                                  {bf16(qkv), kQkvRows, kQkRows}, stream);
    launch_range<a8::Q5Codec, NT>(value_z, codes, scales, tokens, kValueRows, kZRows,
                                  {bf16(z), kZRows, 0}, stream);
}

struct ActivationPlanes {
    std::int8_t* codes;
    __half* scales;
};

ActivationPlanes allocate_planes(WorkspaceArena& workspace, std::int32_t tokens) {
    const DeviceSpan codes = workspace.alloc_bytes(static_cast<std::size_t>(tokens) * kHidden);
    const DeviceSpan scales = workspace.alloc_bytes(
        static_cast<std::size_t>(tokens) * (kHidden / a8::kGroup) * sizeof(__half));
    return {reinterpret_cast<std::int8_t*>(codes.data), reinterpret_cast<__half*>(scales.data)};
}

template <class Launch>
void by_token_tile(std::int32_t tokens, Launch&& launch) {
    switch (a8::token_tile(tokens, 512)) {
    case 512:
        launch.template operator()<16>();
        return;
    case 256:
        launch.template operator()<8>();
        return;
    default:
        launch.template operator()<4>();
        return;
    }
}

} // namespace

std::size_t a8_input_workspace_capacity_bytes(std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("a8 input projection workspace: invalid token interval");
    }
    return a8::activation_workspace_bytes(kHidden, max_tokens);
}

bool a8_attn_input_supported(const Weight& query_key, const Weight& gate_value,
                             std::int32_t tokens) {
    return a8::tokens_supported(tokens) &&
           parent_fits(query_key, QType::Q4_G64_FP16, kQRows + kKvRows) &&
           parent_fits(gate_value, QType::Q5_G64_FP16, kQRows + kKvRows);
}

bool a8_gdn_input_supported(const Weight& qk, const Weight& value_z, std::int32_t tokens) {
    return a8::tokens_supported(tokens) && parent_fits(qk, QType::Q4_G64_FP16, kQkRows) &&
           parent_fits(value_z, QType::Q5_G64_FP16, kValueRows + kZRows);
}

void a8_attn_input_launch(const Tensor& x, const Weight& query_key, const Weight& gate_value,
                          Tensor& q, Tensor& gate, Tensor& k, Tensor& v, WorkspaceArena& workspace,
                          cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (!a8_attn_input_supported(query_key, gate_value, tokens)) {
        throw std::invalid_argument("a8 attention input: unsupported profile");
    }
    auto scope        = workspace.scope();
    const auto planes = allocate_planes(workspace, tokens);
    by_token_tile(tokens, [&]<int NT>() {
        attn_all<NT>(x, query_key, gate_value, tokens, planes.codes, planes.scales, q, gate, k, v,
                     stream);
    });
}

void a8_gdn_input_launch(const Tensor& x, const Weight& qk, const Weight& value_z, Tensor& qkv,
                         Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (!a8_gdn_input_supported(qk, value_z, tokens)) {
        throw std::invalid_argument("a8 GDN input: unsupported profile");
    }
    auto scope        = workspace.scope();
    const auto planes = allocate_planes(workspace, tokens);
    by_token_tile(tokens, [&]<int NT>() {
        gdn_all<NT>(x, qk, value_z, tokens, planes.codes, planes.scales, qkv, z, stream);
    });
}

} // namespace ninferno::ops::detail
