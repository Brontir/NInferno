#include "models/qwen3_5/execution/ffn.h"

#include "core/layout.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/linear_add.h"
#include "ninferno/ops/linear_swiglu.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/residual_add.h"
#include "ninferno/ops/segmented_projection.h"
#include "ninferno/ops/silu_mul.h"

#include <array>
#include <stdexcept>

namespace ninferno::models::qwen3_5::execution {

std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                std::int32_t last, bool mtp) {
    if (first <= 0 || last < first) { throw std::invalid_argument("FFN: invalid column interval"); }
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        return ops::sparse_moe_workspace_capacity_bytes(moe->routed_gate_up.qtype,
                                                        moe->routed_down.qtype, first, last);
    }
    if (const auto* moe = std::get_if<GenericMoeParameters>(&parameters)) {
        return ops::moe_workspace_bytes(moe->weights, moe->config, first, last);
    }
    const auto& p    = std::get<DenseParameters>(parameters);
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    WorkspaceLayoutBuilder layout;
    if (p.split_gate_up) {
        const auto rows = p.split_gate_up->rows();
        (void)layout.alloc(DType::BF16, {rows / 2, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(
                ops::segmented_swiglu_workspace_capacity_bytes(*p.split_gate_up, first, last));
        }
        if (mtp) { (void)layout.alloc(DType::BF16, {down.n, last}); }
        auto scope = layout.scope();
        (void)layout.alloc_bytes(
            mtp ? ops::linear_workspace_capacity_bytes(down.qtype, down.n, down.k, p.down.policy,
                                                       first, last)
                : ops::linear_add_workspace_capacity_bytes(down.qtype, down.n, down.k,
                                                           p.down.policy, first, last));
        return layout.peak_bytes(1);
    }
    if (mtp) {
        (void)layout.alloc(DType::BF16, {gu.n, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last));
        }
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        (void)layout.alloc(DType::BF16, {down.n, last});
        (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(down.qtype, down.n, down.k,
                                                                      p.down.policy, first, last));
    } else {
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last));
        }
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
                down.qtype, down.n, down.k, p.down.policy, first, last));
        }
    }
    return layout.peak_bytes(1);
}

void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, DeviceExecutionView execution,
         bool mtp) {
    const auto stream = execution.stream;
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        const auto storage =
            workspace.alloc_bytes(ffn_workspace_bytes(parameters, columns, columns));
        WorkspaceArena scratch(storage);
        ops::sparse_moe(hidden, *moe, ops::SparseMoeEpilogue::AddResidual, residual, hints, scratch,
                        execution);
        return;
    }
    if (const auto* moe = std::get_if<GenericMoeParameters>(&parameters)) {
        const auto storage =
            workspace.alloc_bytes(ffn_workspace_bytes(parameters, columns, columns));
        WorkspaceArena scratch(storage);
        ops::moe(hidden, moe->weights, moe->config, ops::MoeEpilogue::AddResidual, residual,
                 scratch, stream);
        return;
    }
    const auto& p    = std::get<DenseParameters>(parameters);
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    if (p.split_gate_up) {
        const auto inter  = p.split_gate_up->rows() / 2;
        Tensor activation = workspace.alloc(DType::BF16, {inter, columns});
        {
            auto call = workspace.scope();
            ops::segmented_swiglu(hidden, *p.split_gate_up, false, activation, workspace, stream);
        }
        if (mtp) {
            Tensor delta = workspace.alloc(DType::BF16, {down.n, columns});
            ops::linear(activation, down, delta, p.down.policy, workspace, stream);
            ops::residual_add(delta, residual, stream);
        } else {
            ops::linear_add(activation, down, residual, p.down.policy, workspace, stream);
        }
        return;
    }
    if (mtp) {
        Tensor gate_up = workspace.alloc(DType::BF16, {gu.n, columns});
        {
            auto call = workspace.scope();
            ops::linear(hidden, gu, gate_up, p.gate_up.policy, workspace, stream);
        }
        Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
        ops::silu_mul(gate_up.slice(0, 0, gu.n / 2), gate_up.slice(0, gu.n / 2, gu.n / 2),
                      activation, stream);
        Tensor delta = workspace.alloc(DType::BF16, {down.n, columns});
        ops::linear(activation, down, delta, p.down.policy, workspace, stream);
        ops::residual_add(delta, residual, stream);
        return;
    }
    Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
    {
        auto call = workspace.scope();
        ops::linear_swiglu(hidden, gu, activation, p.gate_up.policy, workspace, stream);
    }
    ops::linear_add(activation, down, residual, p.down.policy, workspace, stream);
}

} // namespace ninferno::models::qwen3_5::execution
