#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include "core/weight.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/linear/fp8/fp8_policy.h"

#include "ops/linear/fp8/fp8_geometry.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

enum class Fp8AttnInputRoute : std::uint8_t {
    A16,
    A8,
};

Fp8AttnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 attn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Fp8AttnInputRoute::A16; }
    if (!allows_a8(policy)) {
        throw std::invalid_argument("fp8 attn_input_proj: unsupported policy");
    }
    if (!fp8_a8_allowed(policy)) { return Fp8AttnInputRoute::A16; }
    return tokens >= 17 ? Fp8AttnInputRoute::A8 : Fp8AttnInputRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                Tensor& v, cudaStream_t stream) {
    // The invariant route profile (the exact contract) takes the generic path's register MMA at
    // every width (the parent's row order: q, k, gate, v): the decode, small and tiled A16
    // kernels each sum K their own way.
    if (invariant_routes_active()) {
        generic::Call call;
        call.x                    = &x;
        call.weight               = &weight;
        call.rows                 = weight.n;
        Tensor* outs[]            = {&q, &k, &gate, &v};
        for (int s = 0; s < 4; ++s) {
            call.out[s]          = outs[s];
            call.segment_rows[s] = static_cast<std::int32_t>(outs[s]->ne[0]);
        }
        call.segments = 4;
        generic::run(call, LinearPolicy::A16Only, nullptr, stream);
        return;
    }
    if (x.ne[1] == 1)
        fp8_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
    else if (x.ne[1] <= kFp8AttnInputLastSmallMmaT)
        fp8_attn_input_a16_small_mma_launch(x, weight, q, gate, k, v, stream);
    else
        fp8_attn_input_a16_gemm_launch(x, weight, q, gate, k, v, stream);
}

} // namespace

std::size_t fp8_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 attn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    return resolve_route(policy, max_tokens) == Fp8AttnInputRoute::A8
               ? fp8_a8_workspace_capacity_bytes(max_tokens, Fp8N14336K5120::kInputRows,
                                                 fp8_attn_input_partial_capacity_bytes(max_tokens))
               : 0;
}

void fp8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                             Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                             cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Fp8AttnInputRoute::A16) {
        launch_a16(x, weight, q, gate, k, v, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("fp8 A8 attn_input_proj requires caller workspace");
    }
    auto scope                   = workspace->scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(
        *workspace, x.ne[1], weight.k, fp8_attn_input_partial_capacity_bytes(x.ne[1]));
    fp8_attn_input_a8_launch(x, weight, q, gate, k, v, scratch, stream);
}

} // namespace ninferno::ops::detail
