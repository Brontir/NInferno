#include "ops/linear/nvfp4/nvfp4_dispatch.h"

#include "ops/common/shape_routes.h"
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/common/route_probe.h"
#include "ops/route_pins.h"
#include <array>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {
const std::array kShapes{&kNvfp4N14336K5120, &kNvfp4N16384K5120, &kNvfp4N34816K5120,
                         &kNvfp4N5120K6144,  &kNvfp4N5120K17408, &kNvfp4N8192K2816,
                         &kNvfp4N1024K2816,  &kNvfp4N2816K4096,  &kNvfp4N2816K8192,
                         &kNvfp4N2816K2112,  &kNvfp4N16384K5376, &kNvfp4N18432K5376,
                         &kNvfp4N2048K5376,  &kNvfp4N43008K5376, &kNvfp4N5376K8192,
                         &kNvfp4N5376K16384, &kNvfp4N5376K21504};

const Nvfp4LinearShape& resolve_shape(std::int32_t n, std::int32_t k, LinearPolicy policy) {
    if (!valid_linear_policy(policy))
        throw std::invalid_argument("nvfp4 linear: unsupported policy");
    if (const auto* shape = find_shape(kShapes, n, k)) { return *shape; }
    throw std::invalid_argument("nvfp4 linear: unsupported shape");
}
} // namespace

bool nvfp4_linear_registered(std::int32_t n, std::int32_t k) {
    return find_shape(kShapes, n, k) != nullptr;
}

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
                                                  LinearPolicy policy, std::int32_t min_tokens,
                                                  std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("nvfp4 linear workspace: invalid token interval");
    const auto& shape = resolve_shape(n, k, policy);
    // --prefill-route w4a4 (ops/route_pins.h) takes A4 at every prefill width it pins.
    const bool pinned_a4 = route_probe_state.quantized_act < 0 &&
                           max_tokens >= kPrefillPinMinTokens &&
                           route_pins().prefill == PrefillRoute::W4a4;
    return allows_a4(policy) &&
                   (route_quantized_activations(shape.uses_a4(min_tokens, max_tokens)) || pinned_a4)
               ? nvfp4_a4_workspace_capacity_bytes(max_tokens, k)
               : 0;
}

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream) {
    validate_nvfp4_weight(weight, "nvfp4 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("nvfp4 linear: T must be positive");
    const auto& shape = resolve_shape(weight.n, weight.k, policy);
    bool a4 = allows_a4(policy) && route_quantized_activations(shape.uses_a4(x.ne[1], x.ne[1]));
    // A route pin decides where no route probe does: a16 keeps BF16 activations, w4a4 takes A4.
    if (allows_a4(policy) && route_probe_state.quantized_act < 0) {
        if (const auto pinned = pinned_shape_quantized(x.ne[1], PrefillRoute::W4a4)) a4 = *pinned;
    }
    note_shape_activations(
        x.ne[1], allows_a4(policy) ? std::optional(PrefillRoute::W4a4) : std::nullopt, a4);
    if (!a4) return shape.a16(x, weight, out, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("nvfp4 A4 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], weight.k);
    shape.a4(x, weight, out, scratch, stream);
}
} // namespace ninferno::ops::detail
