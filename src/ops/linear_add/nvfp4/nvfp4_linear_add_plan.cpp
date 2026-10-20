#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "core/weight.h"

#include "ops/common/route_probe.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

enum class Nvfp4LinearAddRoute : std::uint8_t {
    A16,
    A16Exact, // A16Only: tensor-core A16 invariant in T (nvfp4_a16_exact.cuh)
    A4,
};

// A16Only is the exact contract (ActivationMode::Exact): one token-invariant A16 route at every
// width. AllowA8 has no 8-bit form on NVFP4 and keeps the tuned A16 routes.

// Speculative verify of 2..8 tokens takes the exact family's reg_mma route: RTX 5090, Qwen3.8-27B
// NVFP4 (an nsys census, MTP3 at 24K): SwiGLU 34816x5120 at T = 4 64.8 us against the small-T fused
// 69.2, down 5120x17408 34.1 against SIMT 41.9. Serve (two rounds each): MTP3 24K
// 174.9 -> 182.8 t/s, MTP3 C1 (T = 4) +5 %, DFlash 5 C1 (T = 6) +6 %; at T = 16 (MTP3 C4) it
// lost 5 %, so wider calls keep their routes.
// A route probe forcing the quantized route keeps it (fused activation routes test).
bool nvfp4_verify_exact(std::int32_t tokens) {
    return tokens >= 2 && tokens <= 8 && route_probe_state.quantized_act != 1;
}

Nvfp4LinearAddRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                                  LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0 || !nvfp4_linear_add_registered(output_rows, input_rows)) {
        throw std::invalid_argument("nvfp4 linear_add: unsupported shape");
    }
    if (policy == LinearPolicy::A16Only || nvfp4_verify_exact(tokens)) {
        return Nvfp4LinearAddRoute::A16Exact;
    }
    if (policy == LinearPolicy::AllowA8) { return Nvfp4LinearAddRoute::A16; }
    if (!allows_a4(policy)) { throw std::invalid_argument("nvfp4 linear_add: unsupported policy"); }
    const std::int32_t first_a4 = input_rows == 6144 ? 17 : 8;
    return route_quantized_activations(tokens >= first_a4) ? Nvfp4LinearAddRoute::A4
                                                           : Nvfp4LinearAddRoute::A16;
}


} // namespace

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                      std::int32_t input_rows, LinearPolicy policy,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_add workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    return resolve_route(output_rows, input_rows, policy, max_tokens) == Nvfp4LinearAddRoute::A4
               ? nvfp4_a4_workspace_capacity_bytes(max_tokens, input_rows)
               : 0;
}

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena& workspace,
                               cudaStream_t stream) {
    const Nvfp4LinearAddRoute route = resolve_route(weight.n, weight.k, policy, x.ne[1]);
    if (route == Nvfp4LinearAddRoute::A16Exact) {
        nvfp4_linear_add_exact_launch(x, weight, residual, stream);
        return;
    }
    if (route == Nvfp4LinearAddRoute::A16) {
        nvfp4_linear_add_a16_launch(x, weight, residual, stream);
        return;
    }
    auto scope                     = workspace.scope();
    const Nvfp4A4Workspace scratch = allocate_nvfp4_a4_workspace(workspace, x.ne[1], weight.k);
    nvfp4_linear_add_a4_launch(x, weight, residual, scratch, stream);
}

} // namespace ninferno::ops::detail
