#include "models/qwen3_5/execution/mtp.h"

#include "core/layout.h"
#include "ninferno/ops/attn_input_proj.h"
#include "ninferno/ops/linear_pair.h"
#include "ninferno/ops/mtp_pack.h"

#include <algorithm>

namespace ninferno::models::qwen3_5::execution {

namespace {

std::size_t rows_linear_bytes(const LinearParameters& p, std::int32_t first, std::int32_t last) {
    const auto& w = p.weight;
    return ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first, last);
}

// Whether the fused Q/K/gate/V projection (attn_input_proj) reads the packed parent: its formats.
// A published GGUF's blocks (ggml K- and I-quants: an MoE model's MTP layer from unsloth's
// UD-IQ3_S) run the generic linear and the split instead.
bool fused(const MtpProjectionParameters& parameters) {
    if (parameters.rows) { return false; }
    switch (parameters.packed->weight.qtype) {
    case QType::BF16:
    case QType::NVFP4:
    case QType::FP8_E4M3FN_ROW_BF16:
    case QType::Q8_G32_FP16: return true;
    default: return false;
    }
}

} // namespace

std::size_t mtp_projection_workspace_bytes(const MtpProjectionParameters& parameters,
                                           std::int32_t first, std::int32_t last) {
    if (!parameters.packed) {
        // Separate parents: Q and gate one Linear each, K and V one pair (no workspace).
        const auto& rows = *parameters.rows;
        return std::max(rows_linear_bytes(rows[0], first, last),
                        rows_linear_bytes(rows[2], first, last));
    }
    const auto& p = *parameters.packed;
    const auto& w = p.weight;
    if (fused(parameters)) {
        return ops::attn_input_proj_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first,
                                                             last);
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {w.n, last});
    (void)layout.alloc_bytes(
        ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first, last));
    return layout.peak_bytes(1);
}

std::size_t mtp_kv_workspace_bytes(const MtpProjectionParameters& parameters,
                                   const AttentionConfig& config, std::int32_t first,
                                   std::int32_t last) {
    if (parameters.rows) {
        return ops::linear_pair_workspace_capacity_bytes((*parameters.rows)[1].weight,
                                                         (*parameters.rows)[3].weight, first, last);
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {dimension(config.query_width()), last});
    (void)layout.alloc(DType::BF16, {dimension(config.query_width()), last});
    (void)layout.alloc_bytes(mtp_projection_workspace_bytes(parameters, first, last));
    return layout.peak_bytes(1);
}

std::size_t mtp_query_gate_workspace_bytes(const MtpProjectionParameters& parameters,
                                           const AttentionConfig& config, std::int32_t first,
                                           std::int32_t last) {
    if (parameters.rows) {
        const auto bytes = [&](std::size_t index) {
            const auto& p = (*parameters.rows)[index];
            const auto& w = p.weight;
            return ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first, last);
        };
        return std::max(bytes(0), bytes(2));
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {dimension(config.key_width()), last});
    (void)layout.alloc(DType::BF16, {dimension(config.key_width()), last});
    (void)layout.alloc_bytes(mtp_projection_workspace_bytes(parameters, first, last));
    return layout.peak_bytes(1);
}

void mtp_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                    const AttentionConfig& config, Tensor& query, Tensor& gate, Tensor& key,
                    Tensor& value, WorkspaceArena& workspace, cudaStream_t stream) {
    if (!parameters.packed) {
        mtp_query_gate_projection(hidden, parameters, config, query, gate, workspace, stream);
        mtp_kv_projection(hidden, parameters, config, key, value, workspace, stream);
        return;
    }
    const auto& p = *parameters.packed;
    if (fused(parameters)) {
        ops::attn_input_proj(hidden, p.weight, query, gate, key, value, p.policy, workspace,
                             stream);
        return;
    }
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    Tensor packed      = workspace.alloc(DType::BF16, {p.weight.n, columns});
    ops::linear(hidden, p.weight, packed, p.policy, workspace, stream);
    Tensor q =
        query.view({dimension(config.head_dim), dimension(config.num_attention_heads), columns});
    Tensor k =
        key.view({dimension(config.head_dim), dimension(config.num_key_value_heads), columns});
    Tensor g =
        gate.view({dimension(config.head_dim), dimension(config.num_attention_heads), columns});
    Tensor v =
        value.view({dimension(config.head_dim), dimension(config.num_key_value_heads), columns});
    ops::mtp_split_attn_in(packed, q, k, g, v, stream);
}

void mtp_kv_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                       const AttentionConfig& config, Tensor& key, Tensor& value,
                       WorkspaceArena& workspace, cudaStream_t stream) {
    if (parameters.rows) {
        ops::linear_pair(hidden, (*parameters.rows)[1].weight, (*parameters.rows)[3].weight, key,
                         value, stream);
        return;
    }
    auto scope   = workspace.scope();
    Tensor query = workspace.alloc(DType::BF16, {dimension(config.query_width()), hidden.ne[1]});
    Tensor gate  = workspace.alloc(DType::BF16, {dimension(config.query_width()), hidden.ne[1]});
    mtp_projection(hidden, parameters, config, query, gate, key, value, workspace, stream);
}

void mtp_query_gate_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                               const AttentionConfig& config, Tensor& query, Tensor& gate,
                               WorkspaceArena& workspace, cudaStream_t stream) {
    if (parameters.rows) {
        const auto& q = (*parameters.rows)[0];
        const auto& g = (*parameters.rows)[2];
        {
            auto scope = workspace.scope();
            ops::linear(hidden, q.weight, query, q.policy, workspace, stream);
        }
        ops::linear(hidden, g.weight, gate, g.policy, workspace, stream);
        return;
    }
    auto scope   = workspace.scope();
    Tensor key   = workspace.alloc(DType::BF16, {dimension(config.key_width()), hidden.ne[1]});
    Tensor value = workspace.alloc(DType::BF16, {dimension(config.key_width()), hidden.ne[1]});
    mtp_projection(hidden, parameters, config, query, gate, key, value, workspace, stream);
}

} // namespace ninferno::models::qwen3_5::execution
