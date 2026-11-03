#include "models/gemma4/workspace.h"

#include "ninferno/ops/gemma4_attention.h"
#include "ninferno/ops/gemma4_moe.h"
#include "ninferno/ops/linear_swiglu.h"
#include "ninferno/ops/sampling.h"

#include <algorithm>

namespace ninferno::models::gemma4 {
namespace {

std::size_t bf16_row_bytes(std::int64_t elements) {
    return static_cast<std::size_t>(elements) * 2 + 256;
}

} // namespace

std::size_t target_activation_bytes(const Config& c, std::int32_t columns, std::int32_t head_columns) {
    const std::int64_t t = columns;
    const std::int64_t h = c.hidden_size;
    std::int64_t proj = 0, attn = 0;
    for (const auto* g : {&c.sliding, &c.full}) {
        proj = std::max<std::int64_t>(proj, std::int64_t(c.query_heads + 2 * g->kv_heads) * g->head_dim);
        attn = std::max<std::int64_t>(attn, std::int64_t(c.query_heads) * g->head_dim);
    }
    std::size_t bytes = 0;
    const auto add    = [&](std::int64_t elements) { bytes += bf16_row_bytes(elements); };
    add(h * t);                                     // x
    add(h * t);                                     // normed input
    add(proj * t);                                  // raw projection(s)
    add(proj * t);                                  // prepared q/k/v
    add(attn * t);                                  // attention context
    add(h * t);                                     // branch output
    add(2 * std::int64_t(c.intermediate_size) * t); // gate/up
    add(std::int64_t(c.intermediate_size) * t);     // GeGLU product
    add(h * t);                                     // dense branch
    add(h * t);                                     // expert input
    add(h * t);                                     // router input (generic MoE)
    add(h * t);                                     // expert output
    add(h * t);                                     // branch join
    if (c.ple()) {
        const std::int64_t rows = std::int64_t(c.layers) * c.ple_width;
        add(rows * t);                          // per-layer inputs (the projection, combined in place)
        add(rows * t);                          // gathered table rows
        add(2 * std::int64_t(c.ple_width) * t); // gate, product
    }
    add(std::int64_t(c.vocab_size) * std::max<std::int64_t>(head_columns, 1));
    return bytes;
}

std::size_t target_op_bytes(const Config& c, std::span<const WorkspaceLinear> linears,
                            const WorkspaceLinear& head, std::int32_t max_context,
                            std::int32_t columns, std::int32_t head_columns,
                            std::size_t routed_experts) {
    std::size_t op = routed_experts;
    for (const WorkspaceLinear& p : linears) {
        if (p.n == 0) { continue; }
        op = std::max(op, ops::linear_workspace_capacity_bytes(p.qtype, p.n, p.k, p.policy, 1, columns));
        if (p.geglu) {
            for (const auto policy : {p.policy, ops::LinearPolicy::A16Only}) {
                op = std::max(op, ops::linear_geglu_workspace_capacity_bytes(p.qtype, p.n, p.k, policy, 1,
                                                                             columns) +
                                      256);
            }
        }
    }
    for (const auto* g : {&c.sliding, &c.full}) {
        const ops::Gemma4AttentionEnvelope envelope{static_cast<std::uint32_t>(max_context), columns};
        const std::int32_t window = g == &c.sliding ? c.sliding_window : 0;
        op = std::max(op, ops::gemma4_attention_workspace_bytes(g->head_dim, c.query_heads, g->kv_heads,
                                                                envelope, window));
    }
    if (c.moe()) { op = std::max(op, ops::gemma4_moe_workspace_bytes(1, columns)); }
    // The output head runs on this arena too (TargetPass::head, uncapped_head on A16 for the exact
    // logits), over up to head_columns columns.
    for (const auto policy : {head.policy, ops::LinearPolicy::A16Only}) {
        op = std::max(op, ops::linear_workspace_capacity_bytes(head.qtype, head.n, head.k, policy, 1,
                                                               std::max(head_columns, 1)));
    }
    return std::max(op, ops::sampling_workspace_capacity_bytes(c.vocab_size, 1, 1));
}

} // namespace ninferno::models::gemma4
