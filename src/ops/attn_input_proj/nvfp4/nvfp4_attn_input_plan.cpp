#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "core/weight.h"

#include "ops/common/route_probe.h"
#include "ops/generic/generic_gemm.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

enum class Nvfp4AttnInputRoute : std::uint8_t {
    A16,
    A16Exact, // A16Only: tensor-core A16 invariant in T (nvfp4_a16_exact.cuh)
    A4,
    W4A8, // AllowA8 from 17 tokens: the generic path's int8 W4A8 (generic::owned_prefill_w4a8)
};

// A16Only is the exact contract (ActivationMode::Exact): one token-invariant A16 route at every
// width. AllowA8 has no 8-bit form among these kernels: it keeps the tuned A16 routes below 17
// tokens (decode and verify) and from 17 takes the generic W4A8, the four parts in one call.
Nvfp4AttnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 attn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Nvfp4AttnInputRoute::A16Exact; }
    if (policy == LinearPolicy::AllowA8) {
        return tokens >= 17 ? Nvfp4AttnInputRoute::W4A8 : Nvfp4AttnInputRoute::A16;
    }
    if (!allows_a4(policy)) {
        throw std::invalid_argument("nvfp4 attn_input_proj: unsupported policy");
    }
    return route_quantized_activations(tokens >= 4) ? Nvfp4AttnInputRoute::A4
                                                    : Nvfp4AttnInputRoute::A16;
}


} // namespace

std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 attn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    const Nvfp4AttnInputRoute route = resolve_route(policy, max_tokens);
    if (route == Nvfp4AttnInputRoute::W4A8) {
        return generic::workspace_capacity_bytes(
            {QType::NVFP4, policy, Nvfp4N14336K5120::kOutputRows, Nvfp4N14336K5120::kInputRows,
             generic::Epilogue::Store},
            nullptr, std::max(min_tokens, 17), max_tokens);
    }
    return route == Nvfp4AttnInputRoute::A4
               ? nvfp4_a4_workspace_capacity_bytes(max_tokens, Nvfp4N14336K5120::kInputRows)
               : 0;
}

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream) {
    Nvfp4AttnInputRoute route = resolve_route(policy, x.ne[1]);
    // A weight the W4A8 tiles cannot read (generic::owned_prefill_w4a8) keeps the A16 route.
    if (route == Nvfp4AttnInputRoute::W4A8 &&
        (workspace == nullptr || !generic::owned_prefill_w4a8(weight, policy, x.ne[1]))) {
        route = Nvfp4AttnInputRoute::A16;
    }
    if (route == Nvfp4AttnInputRoute::A16Exact) {
        nvfp4_attn_input_exact_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    if (route == Nvfp4AttnInputRoute::W4A8) {
        // The parent's row order (nvfp4_attn_input_output.cuh): query, key, gate, value.
        generic::Call call;
        call.x               = &x;
        call.weight          = &weight;
        call.rows            = Nvfp4N14336K5120::kOutputRows;
        Tensor* const outs[] = {&q, &k, &gate, &v};
        for (int s = 0; s < 4; ++s) {
            call.out[s]          = outs[s];
            call.segment_rows[s] = outs[s]->ne[0];
        }
        call.segments = 4;
        generic::run(call, policy, workspace, stream);
        return;
    }
    if (route == Nvfp4AttnInputRoute::A16) {
        nvfp4_attn_input_a16_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 A4 attn_input_proj requires caller workspace");
    }
    auto scope                     = workspace->scope();
    const Nvfp4A4Workspace scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_attn_input_a4_launch(x, weight, q, gate, k, v, scratch, stream);
}

} // namespace ninferno::ops::detail
