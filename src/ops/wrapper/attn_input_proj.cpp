#include "core/weight.h"
#include "ops/generic/generic_gemm.h"
#include "ops/common/native_nvfp4.h"
#include "ninferno/ops/attn_input_proj.h"
#include "ninferno/ops/prefill_routes.h"

#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"
#include "ops/attn_input_proj/q8/q8_attn_input_plan.h"
#include "ops/linear/fp8/fp8_geometry.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear_swiglu/q4a8/q4a8_linear_swiglu.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace ninferno::ops {
namespace {

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t cols, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != cols ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_rowsplit(const Weight& weight, QType qtype, std::int32_t rows, const char* label) {
    const bool q4_planes =
        qtype != QType::Q4_G64_FP16 || (weight.qhigh == nullptr && weight.high_plane_bytes == 0);
    const bool q5_planes =
        qtype != QType::Q5_G64_FP16 || (weight.qhigh != nullptr && weight.high_plane_bytes != 0);
    if (weight.qtype != qtype || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 64 || weight.group != 64 ||
        weight.ndim != 2 || weight.n != rows || weight.k != 5120 || weight.shape[0] != rows ||
        weight.shape[1] != 5120 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 5120 || !q4_planes || !q5_planes ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 4) ||
        (qtype == QType::Q5_G64_FP16 && !aligned_to(weight.qhigh, 16))) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_q8_rowsplit(const Weight& weight, std::int32_t rows, std::int32_t hidden,
                         const char* label) {
    if (weight.qtype != QType::Q8_G32_FP16 || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        weight.ndim != 2 || weight.n != rows || weight.k != hidden || weight.shape[0] != rows ||
        weight.shape[1] != hidden || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != hidden || weight.qhigh != nullptr ||
        weight.high_plane_bytes != 0 || !aligned_to(weight.qdata, 16) ||
        !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_bf16_contiguous(const Weight& weight, std::int32_t rows, std::int32_t hidden,
                             const char* label) {
    const std::uint64_t payload_bytes = static_cast<std::uint64_t>(rows) *
                                        static_cast<std::uint64_t>(hidden) * sizeof(std::uint16_t);
    if (weight.qtype != QType::BF16 || weight.layout != QuantLayout::Contiguous ||
        weight.payload_bytes < payload_bytes || weight.high_plane_bytes != 0 || weight.ndim != 2 ||
        weight.n != rows || weight.k != hidden || weight.shape[0] != rows ||
        weight.shape[1] != hidden || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != hidden || weight.qhigh != nullptr || weight.scales != nullptr ||
        weight.group_size != 0 || weight.group != 0 || !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("attn_input_proj: invalid compute policy");
}

void dispatch_single_parent(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                            cudaStream_t stream) {
    validate_policy(policy);
    if (weight.qtype == QType::BF16) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        require_bf16_contiguous(weight, kRows, kHidden, "query/key/gate/value weight");
        detail::bf16_attn_input_dispatch(x, weight, q, gate, k, v, stream);
        return;
    }

    if (weight.qtype == QType::NVFP4) {
#ifdef NINFERNO_WITHOUT_NVFP4
        detail::no_native_nvfp4("attn_input_proj");
#else
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        detail::validate_nvfp4_weight(weight, "nvfp4 attn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("nvfp4 attn_input_proj: unsupported weight shape");
        }
        detail::nvfp4_attn_input_dispatch(x, weight, q, gate, k, v, policy, workspace, stream);
        return;
#endif
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        detail::validate_fp8_weight(weight, "fp8 attn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("fp8 attn_input_proj: unsupported weight shape");
        }
        detail::fp8_attn_input_dispatch(x, weight, q, gate, k, v, policy, workspace, stream);
        return;
    }

    constexpr std::int32_t kHidden = 2048;
    constexpr std::int32_t kQRows  = 4096;
    constexpr std::int32_t kKvRows = 512;
    constexpr std::int32_t kRows   = 9216;
    const std::int32_t cols        = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(gate, kQRows, cols, "gate");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_q8_rowsplit(weight, kRows, kHidden, "query/key/gate/value weight");
    // owned_decode_r8: at decode the generic R8 writes the four parts (the parent's
    // row order: q, k, gate, v) in one call. So does the invariant route profile at every width
    // (the Q8 kernels split K by width; gdn_input_proj).
    if (invariant_routes_active() || detail::generic::owned_decode_r8(weight, policy, cols) ||
        detail::generic::q8_prefill_a8(weight, policy, cols) ||
        detail::generic::volta_tiled_route(weight, policy, cols)) {
        detail::generic::Call call;
        call.x                    = &x;
        call.weight               = &weight;
        call.rows                 = kRows;
        Tensor* outs[]            = {&q, &k, &gate, &v};
        const std::int32_t rows[] = {kQRows, kKvRows, kQRows, kKvRows};
        for (int s = 0; s < 4; ++s) {
            call.out[s]          = outs[s];
            call.segment_rows[s] = rows[s];
        }
        call.segments = 4;
        detail::generic::run(call, policy, workspace, stream);
        return;
    }
    detail::q8_attn_input_dispatch(x, weight, q, gate, k, v, stream);
}

// Int8 prefill activations (prefill_a8_policy, generic_gemm.h): inside a prefill a call of 17 columns or more of the
// Q4/Q5 pair takes the generic path's int8 tiers, one call per parent writing its two parts, as
// linear and the Q4/Q5 GDN input do. Off unless the option asks for it.
constexpr std::int32_t kPairHidden = 5120;
constexpr std::int32_t kPairQRows  = 6144;
constexpr std::int32_t kPairKvRows = 1024;

// The invariant route profile takes the same two generic calls at every width: the Q4/Q5 pair
// kernels pick their tile by width.
bool pair_prefill_a8(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     WorkspaceArena* workspace, cudaStream_t stream) {
    const bool invariant = invariant_routes_active();
    if (!invariant &&
        (workspace == nullptr ||
         !detail::generic::fixed_pair_prefill_a8(query_key_weight, gate_value_weight, x.ne[1]))) {
        return false;
    }
    if (!invariant && detail::a8_input_pays() &&
        detail::a8_attn_input_supported(query_key_weight, gate_value_weight, x.ne[1]) &&
        x.is_contiguous() && aligned_to(x.data, 16) && q.is_contiguous() && gate.is_contiguous() &&
        k.is_contiguous() && v.is_contiguous()) {
        detail::a8_attn_input_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                     *workspace, stream);
        return true;
    }
    const auto run_parent = [&](const Weight& weight, Tensor& first, Tensor& second) {
        detail::generic::Call call;
        call.x               = &x;
        call.weight          = &weight;
        call.rows            = kPairQRows + kPairKvRows;
        call.out[0]          = &first;
        call.segment_rows[0] = kPairQRows;
        call.out[1]          = &second;
        call.segment_rows[1] = kPairKvRows;
        call.segments        = 2;
        detail::generic::run(call,
                             invariant ? LinearPolicy::A16Only
                                       : detail::generic::fixed_parent_prefill_a8_policy(weight.qtype),
                             workspace, stream);
    };
    run_parent(query_key_weight, q, k);
    run_parent(gate_value_weight, gate, v);
    return true;
}

} // namespace

std::size_t attn_input_proj_pair_workspace_capacity_bytes(std::int32_t min_tokens,
                                                          std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("attn_input_proj workspace: invalid token interval");
    }
    // The Volta tiled route takes each parent through CUTLASS, which decodes it here.
    const std::size_t volta =
        std::max(detail::generic::volta_prefill_workspace_bytes(
                     QType::Q4_G64_FP16, kPairQRows + kPairKvRows, kPairHidden, max_tokens, false),
                 detail::generic::volta_prefill_workspace_bytes(
                     QType::Q5_G64_FP16, kPairQRows + kPairKvRows, kPairHidden, max_tokens, false));
    // The dedicated int8 route of the listed cards (a8_input_pays) quantises x once for both.
    const std::size_t dedicated =
        max_tokens >= 128 && detail::a8_input_pays()
            ? detail::a8_input_workspace_capacity_bytes(std::max(min_tokens, 128), max_tokens)
            : 0;
    return std::max({volta, dedicated,
                     detail::generic::fixed_parents_prefill_a8_workspace_bytes(
                         {{QType::Q4_G64_FP16, kPairQRows + kPairKvRows},
                          {QType::Q5_G64_FP16, kPairQRows + kPairKvRows}},
                         kPairHidden, min_tokens, max_tokens)});
}

std::size_t attn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                                     std::int32_t input_rows, LinearPolicy policy,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("attn_input_proj workspace: invalid token interval");
    }

    switch (parent_qtype) {
    case QType::BF16:
        if (parent_rows != 14336 || input_rows != 5120) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported BF16 profile");
        }
        return 0;
    case QType::NVFP4:
#ifdef NINFERNO_WITHOUT_NVFP4
        detail::no_native_nvfp4("attn_input_proj");
#else
        if (parent_rows != detail::Nvfp4N14336K5120::kOutputRows ||
            input_rows != detail::Nvfp4N14336K5120::kInputRows) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_attn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
#endif
    case QType::FP8_E4M3FN_ROW_BF16:
        if (parent_rows != detail::Fp8N14336K5120::kOutputRows ||
            input_rows != detail::Fp8N14336K5120::kInputRows) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported FP8 profile");
        }
        return detail::fp8_attn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    case QType::Q8_G32_FP16:
        if (parent_rows != 9216 || input_rows != 2048) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported Q8 profile");
        }
        (void)detail::q8_attn_input_resolve_plan(
            {input_rows, 4096, 512, parent_rows, input_rows, min_tokens});
        (void)detail::q8_attn_input_resolve_plan(
            {input_rows, 4096, 512, parent_rows, input_rows, max_tokens});
        if (allows_a8(policy) && max_tokens >= detail::generic::kQ8PrefillA8Tokens) {
            return detail::generic::workspace_capacity_bytes(
                {parent_qtype, policy, parent_rows, input_rows, detail::generic::Epilogue::Store},
                nullptr, min_tokens, max_tokens);
        }
        // A decode call may take the generic R8 (owned_decode_r8).
        if (allows_a8(policy) &&
            min_tokens <= detail::generic::kR8DecodeTokens) {
            return detail::generic::workspace_capacity_bytes(
                {parent_qtype, policy, parent_rows, input_rows, detail::generic::Epilogue::Store},
                nullptr, min_tokens, detail::generic::kR8DecodeTokens);
        }
        return 0;
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::FP32:
    case QType::INT32:
        break;
    }
    throw std::invalid_argument("attn_input_proj workspace: unsupported parent qtype");
}

// Without tensor mma.sync (Volta) a call of 17 tokens or more that the tiled tensor-core route
// takes (generic::volta_tiled_route) runs there: the shape plans' MMA tiles run on FP32 FMAs on
// that GPU. One generic call per parent weight, its rows written segment by segment.
static bool volta_tiled_parent(const Tensor& x, const Weight& weight,
                               std::initializer_list<Tensor*> outs,
                               std::initializer_list<std::int32_t> rows, WorkspaceArena* workspace,
                               cudaStream_t stream) {
    if (!detail::generic::volta_tiled_route(weight, LinearPolicy::A16Only, x.ne[1])) return false;
    detail::generic::Call call;
    call.x      = &x;
    call.weight = &weight;
    int s       = 0;
    for (Tensor* out : outs) call.out[s++] = out;
    s = 0;
    for (const std::int32_t r : rows) {
        call.segment_rows[s++] = r;
        call.rows += r;
    }
    call.segments = s;
    detail::generic::run(call, LinearPolicy::A16Only, workspace, stream);
    return true;
}

void attn_input_proj(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     cudaStream_t stream) {
    attn_input_proj(x, query_key_weight, gate_value_weight, q, gate, k, v, nullptr, stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     WorkspaceArena& workspace, cudaStream_t stream) {
    attn_input_proj(x, query_key_weight, gate_value_weight, q, gate, k, v, &workspace, stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     WorkspaceArena* workspace, cudaStream_t stream) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    const std::int32_t cols        = x.ne[1];
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(gate, kQRows, cols, "gate");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_rowsplit(query_key_weight, QType::Q4_G64_FP16, kQRows + kKvRows, "query/key weight");
    require_rowsplit(gate_value_weight, QType::Q5_G64_FP16, kQRows + kKvRows, "gate/value weight");

    if (pair_prefill_a8(x, query_key_weight, gate_value_weight, q, gate, k, v, workspace, stream)) {
        return;
    }
    if (detail::generic::volta_tiled_route(query_key_weight, LinearPolicy::A16Only, cols) &&
        detail::generic::volta_tiled_route(gate_value_weight, LinearPolicy::A16Only, cols)) {
        volta_tiled_parent(x, query_key_weight, {&q, &k}, {kQRows, kKvRows}, workspace, stream);
        volta_tiled_parent(x, gate_value_weight, {&gate, &v}, {kQRows, kKvRows}, workspace, stream);
        return;
    }
    detail::q4_q5_attn_input_dispatch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                      stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                     WorkspaceArena& workspace, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_gate_value_weight, q, gate, k, v, policy, &workspace,
                           stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_gate_value_weight, q, gate, k, v, LinearPolicy::A16Only,
                           nullptr, stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_value_weight, Tensor& q, Tensor& k,
                     Tensor& v, cudaStream_t stream) {
    constexpr std::int32_t kQRows  = 4096;
    constexpr std::int32_t kKvRows = 1024;
    constexpr std::int32_t kRows   = 6144;
    const std::int32_t hidden      = x.ne[0];
    const std::int32_t cols        = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
    if (hidden != 2048 && hidden != 5120) {
        throw std::invalid_argument("attn_input_proj: unsupported Q8 Q/K/V profile");
    }
    require_matrix(x, hidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_q8_rowsplit(query_key_value_weight, kRows, hidden, "query/key/value weight");

    // The invariant route profile takes the generic path at every width (the Q8 kernels pick
    // their K split by width).
    if (invariant_routes_active()) {
        detail::generic::Call call;
        call.x               = &x;
        call.weight          = &query_key_value_weight;
        call.rows            = kRows;
        call.out[0]          = &q;
        call.segment_rows[0] = kQRows;
        call.out[1]          = &k;
        call.segment_rows[1] = kKvRows;
        call.out[2]          = &v;
        call.segment_rows[2] = kKvRows;
        call.segments        = 3;
        detail::generic::run(call, LinearPolicy::A16Only, nullptr, stream);
        return;
    }
    if (volta_tiled_parent(x, query_key_value_weight, {&q, &k, &v}, {kQRows, kKvRows, kKvRows},
                           nullptr, stream)) {
        return;
    }
    detail::q8_attn_input_dispatch(x, query_key_value_weight, q, k, v, stream);
}

} // namespace ninferno::ops
