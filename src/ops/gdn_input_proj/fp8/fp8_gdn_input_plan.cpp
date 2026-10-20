#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include "core/weight.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/linear/fp8/fp8_policy.h"

#include "ops/linear/fp8/fp8_geometry.h"

#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

enum class Fp8GdnInputRoute : std::uint8_t {
    A16,
    A8,
};

Fp8GdnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 gdn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Fp8GdnInputRoute::A16; }
    if (!allows_a8(policy)) {
        throw std::invalid_argument("fp8 gdn_input_proj: unsupported policy");
    }
    if (!fp8_a8_allowed(policy)) { return Fp8GdnInputRoute::A16; }
    return tokens >= 17 ? Fp8GdnInputRoute::A8 : Fp8GdnInputRoute::A16;
}

} // namespace

std::size_t fp8_gdn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 gdn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    return resolve_route(policy, max_tokens) == Fp8GdnInputRoute::A8
               ? fp8_a8_workspace_capacity_bytes(max_tokens, Fp8N16384K5120::kInputRows,
                                                 fp8_gdn_input_partial_capacity_bytes(max_tokens))
               : 0;
}

void fp8_gdn_input_a16_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream) {
    // The invariant route profile (the exact contract) takes the generic path's register MMA at
    // every width: the decode, small and matrix A16 kernels each sum K their own way, so a verify
    // launch of several rows gave a row other bytes than the row alone.
    if (invariant_routes_active()) {
        generic::Call call;
        call.x               = &x;
        call.weight          = &weight;
        call.rows            = weight.n;
        call.out[0]          = &qkv;
        call.segment_rows[0] = static_cast<std::int32_t>(qkv.ne[0]);
        call.out[1]          = &z;
        call.segment_rows[1] = static_cast<std::int32_t>(z.ne[0]);
        call.segments        = 2;
        generic::run(call, LinearPolicy::A16Only, nullptr, stream);
        return;
    }
    if (x.ne[1] == 1) {
        fp8_gdn_input_decode_launch(x, weight, qkv, z, stream);
    } else {
        fp8_gdn_input_matrix_launch(x, weight, qkv, z, stream);
    }
}

void fp8_gdn_input_a8_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                   = workspace.scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(
        workspace, x.ne[1], weight.k, fp8_gdn_input_partial_capacity_bytes(x.ne[1]));
    fp8_gdn_input_a8_launch(x, weight, qkv, z, scratch, stream);
}

void fp8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                            LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Fp8GdnInputRoute::A16) {
        fp8_gdn_input_a16_dispatch(x, weight, qkv, z, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("fp8 A8 gdn_input_proj requires caller workspace");
    }
    fp8_gdn_input_a8_dispatch(x, weight, qkv, z, *workspace, stream);
}

} // namespace ninferno::ops::detail
