#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "core/weight.h"

#include "core/layout.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/silu_mul.h"
#include "ops/common/route_probe.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma_launch.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

enum class Nvfp4LinearSwiGluRoute {
    DecodeFusedA16,
    SmallTFusedA16,
    FusedA4,
    TmaFusedA4,
    LinearA16Post, // A16 wider than the fused A16 routes: the Linear A16 projection, then SiLU
    ExactA16,      // A16Only: token-invariant A16 with SiLU in its epilogue (nvfp4_a16_exact.cuh)
};

// Speculative verify of 2..8 tokens takes the exact family's reg_mma route: RTX 5090, Qwen3.8-27B
// NVFP4 (an nsys census, MTP3 at 24K): SwiGLU 34816x5120 at T = 4 64.8 us against the small-T fused
// 69.2, down 5120x17408 34.1 against SIMT 41.9. Serve (two rounds each): MTP3 24K
// 174.9 -> 182.8 t/s, MTP3 C1 (T = 4) +5 %, DFlash 5 C1 (T = 6) +6 %; at T = 16 (MTP3 C4) it
// lost 5 %, so wider calls keep their routes.
// A route probe forcing the quantized route keeps it (fused activation routes test).
bool nvfp4_verify_exact(std::int32_t tokens) {
    return tokens >= 2 && tokens <= 8 && route_probe_state.quantized_act != 1;
}

Nvfp4LinearSwiGluRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 linear_swiglu: T must be positive"); }
    if (!valid_linear_policy(policy)) {
        throw std::invalid_argument("nvfp4 linear_swiglu: invalid compute policy");
    }
    // A16Only is the exact contract (ActivationMode::Exact): one token-invariant route at every
    // width. AllowA8 has no 8-bit form on NVFP4 and keeps the tuned A16 routes; so does a route
    // probe that forces A16.
    if (policy == LinearPolicy::A16Only || nvfp4_verify_exact(tokens)) {
        return Nvfp4LinearSwiGluRoute::ExactA16;
    }
    if (policy == LinearPolicy::AllowA8 || !route_quantized_activations(true)) {
        if (tokens == 1) { return Nvfp4LinearSwiGluRoute::DecodeFusedA16; }
        if (tokens <= 16) { return Nvfp4LinearSwiGluRoute::SmallTFusedA16; }
        return Nvfp4LinearSwiGluRoute::LinearA16Post;
    }
    // The fused A16 routes keep T <= 4 unless a route probe forces the quantized route.
    if (route_probe_state.quantized_act != 1) {
        if (tokens == 1) { return Nvfp4LinearSwiGluRoute::DecodeFusedA16; }
        if (tokens <= 4) { return Nvfp4LinearSwiGluRoute::SmallTFusedA16; }
    }
    // This route dispatches its own fused kernel rather than a Linear shape's, so it carries its
    // own condition; the call site below forces the matching scale layout.
    if (tokens >= 256) { return Nvfp4LinearSwiGluRoute::TmaFusedA4; }
    return Nvfp4LinearSwiGluRoute::FusedA4;
}

template <class Allocator>
Nvfp4A4Workspace allocate_fused_workspace(Allocator& allocator, std::int32_t tokens) {
    return allocate_nvfp4_a4_workspace(allocator, tokens, Nvfp4N34816K5120::kInputRows);
}

std::size_t fused_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_fused_workspace(layout, tokens);
    return layout.peak_bytes(1);
}

// LinearA16Post: the projected gate/up plane, then SiLU into the output.
Tensor allocate_projected(WorkspaceArena& arena, std::int32_t tokens) {
    return arena.alloc(DType::BF16, {Nvfp4N34816K5120::kOutputRows, tokens}, 256);
}

std::size_t projected_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(static_cast<std::size_t>(Nvfp4N34816K5120::kOutputRows) *
                                 static_cast<std::size_t>(tokens) * 2,
                             256);
    return layout.peak_bytes(1);
}

} // namespace

std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                         std::int32_t min_tokens,
                                                         std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_swiglu workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    (void)resolve_route(policy, max_tokens);
    if (policy == LinearPolicy::A16Only) { return 0; }
    if (policy == LinearPolicy::AllowA8 || !route_quantized_activations(true)) {
        return max_tokens > 16 ? projected_workspace_bytes(max_tokens) : 0;
    }
    // The verify widths run the exact route without a workspace: an interval up to 8 tokens needs
    // the fused one only for a T = 1 call a route probe sends to A4.
    if (nvfp4_verify_exact(max_tokens)) {
        return min_tokens == 1 && resolve_route(policy, 1) == Nvfp4LinearSwiGluRoute::FusedA4
                   ? fused_workspace_bytes(1)
                   : 0;
    }
    if (max_tokens <= 4 && route_probe_state.quantized_act != 1) { return 0; }

    return fused_workspace_bytes(max_tokens);
}

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, WorkspaceArena& workspace,
                                  cudaStream_t stream) {
    switch (resolve_route(policy, x.ne[1])) {
    case Nvfp4LinearSwiGluRoute::DecodeFusedA16:
        nvfp4_linear_swiglu_decode_launch(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::SmallTFusedA16:
        nvfp4_linear_swiglu_small_t_launch(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::FusedA4:
        nvfp4_linear_swiglu_a4_launch(x, weight, out, workspace, stream);
        return;
    case Nvfp4LinearSwiGluRoute::TmaFusedA4: {
        auto scope                     = workspace.scope();
        const Nvfp4A4Workspace scratch = allocate_fused_workspace(workspace, x.ne[1]);
        launch_nvfp4_a4_quantize(x, weight, scratch, Nvfp4ScaleLayout::Tiled256, stream);
        launch_nvfp4_linear_swiglu_a4_tma(
            nvfp4_a4_operands(weight, scratch, x.ne[1], Nvfp4ScaleLayout::Tiled256),
            static_cast<__nv_bfloat16*>(out.data), stream);
        return;
    }
    case Nvfp4LinearSwiGluRoute::ExactA16:
        nvfp4_linear_swiglu_exact_launch(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::LinearA16Post: {
        auto scope              = workspace.scope();
        const Tensor projected  = allocate_projected(workspace, x.ne[1]);
        WorkspaceArena no_linear_scratch(DeviceSpan{});
        Tensor destination      = projected;
        linear(x, weight, destination, LinearPolicy::A16Only, no_linear_scratch, stream);
        constexpr std::int32_t kIntermediate = Nvfp4N34816K5120::kOutputRows / 2;
        silu_mul(projected.slice(0, 0, kIntermediate),
                 projected.slice(0, kIntermediate, kIntermediate), out, stream);
        return;
    }
    }
    throw std::logic_error("unreachable NVFP4 SwiGLU route");
}
} // namespace ninferno::ops::detail
