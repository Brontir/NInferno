#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "core/weight.h"

#include "ops/common/route_probe.h"
#include "ops/generic/generic_gemm.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

enum class Nvfp4GdnInputRoute : std::uint8_t {
    A16,
    A16Exact, // A16Only: tensor-core A16 invariant in T (nvfp4_a16_exact.cuh)
    A4,
    W4A8, // AllowA8 from 17 tokens: the generic path's int8 W4A8 (generic::owned_prefill_w4a8)
};

// A16Only is the exact contract (ActivationMode::Exact): one token-invariant A16 route at every
// width. AllowA8 has no 8-bit form among these kernels: it keeps the tuned A16 routes below 17
// tokens and outside a prefill (decode, any verify), and a prefill call from 17 takes the generic
// W4A8, both parts in one call (the dispatch asks generic::owned_prefill_w4a8; the workspace
// query, which cannot know the profile, reserves its planes).
Nvfp4GdnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 gdn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Nvfp4GdnInputRoute::A16Exact; }
    if (policy == LinearPolicy::AllowA8) {
        return tokens >= 17 ? Nvfp4GdnInputRoute::W4A8 : Nvfp4GdnInputRoute::A16;
    }
    if (allows_a4(policy)) {
        return route_quantized_activations(true) ? Nvfp4GdnInputRoute::A4
                                                 : Nvfp4GdnInputRoute::A16;
    }
    throw std::invalid_argument("nvfp4 gdn_input_proj: unsupported policy");
}


} // namespace

std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 gdn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    const Nvfp4GdnInputRoute route = resolve_route(policy, max_tokens);
    if (route == Nvfp4GdnInputRoute::W4A8) {
        return generic::workspace_capacity_bytes(
            {QType::NVFP4, policy, Nvfp4N16384K5120::kOutputRows, Nvfp4N16384K5120::kInputRows,
             generic::Epilogue::Store},
            nullptr, std::max(min_tokens, 17), max_tokens);
    }
    return route == Nvfp4GdnInputRoute::A4
               ? nvfp4_a4_workspace_capacity_bytes(max_tokens, Nvfp4N16384K5120::kInputRows)
               : 0;
}

void nvfp4_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                              LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    Nvfp4GdnInputRoute route = resolve_route(policy, x.ne[1]);
    // A call outside a prefill, or of a weight the W4A8 tiles cannot read
    // (generic::owned_prefill_w4a8), keeps the A16 route.
    if (route == Nvfp4GdnInputRoute::W4A8 &&
        (workspace == nullptr || !generic::owned_prefill_w4a8(weight, policy, x.ne[1]))) {
        route = Nvfp4GdnInputRoute::A16;
    }
    if (route == Nvfp4GdnInputRoute::A16Exact) {
        nvfp4_gdn_input_exact_launch(x, weight, qkv, z, stream);
        return;
    }
    if (route == Nvfp4GdnInputRoute::W4A8) {
        generic::Call call;
        call.x               = &x;
        call.weight          = &weight;
        call.rows            = Nvfp4N16384K5120::kOutputRows;
        call.out[0]          = &qkv;
        call.segment_rows[0] = qkv.ne[0];
        call.out[1]          = &z;
        call.segment_rows[1] = z.ne[0];
        call.segments        = 2;
        generic::run(call, policy, workspace, stream);
        return;
    }
    if (route == Nvfp4GdnInputRoute::A16) {
        nvfp4_gdn_input_a16_launch(x, weight, qkv, z, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 A4 gdn_input_proj requires caller workspace");
    }
    auto scope                     = workspace->scope();
    const Nvfp4A4Workspace scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_gdn_input_a4_launch(x, weight, qkv, z, scratch, stream);
}

} // namespace ninferno::ops::detail
