#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "core/weight.h"
#include "ops/linear/fp8/fp8_policy.h"

#include "ninferno/ops/prefill_routes.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_geometry.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail {

bool fp8_linear_add_registered(std::int32_t rows, std::int32_t k) noexcept {
    return rows == Fp8N5120K6144::kOutputRows &&
           (k == Fp8N5120K6144::kInputRows || k == Fp8N5120K17408::kInputRows);
}

namespace {

// Under the prefill profile, A8 calls up to this size take the short-call tiles.
constexpr std::int32_t kFp8LinearAddPrefillSmallTMax = 128;

enum class Fp8LinearAddRoute : std::uint8_t {
    A16,
    A8,
};

Fp8LinearAddRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                                LinearPolicy policy, std::int32_t tokens, bool prefill) {
    if (tokens <= 0 || !fp8_linear_add_registered(output_rows, input_rows)) {
        throw std::invalid_argument("fp8 linear_add: unsupported shape");
    }
    if (policy == LinearPolicy::A16Only) { return Fp8LinearAddRoute::A16; }
    if (!allows_a8(policy)) { throw std::invalid_argument("fp8 linear_add: unsupported policy"); }
    if (!fp8_a8_allowed(policy)) { return Fp8LinearAddRoute::A16; }
    // Under the prefill route profile A8 starts at the first prefill-sized call (T=17): A16 keeps
    // the activation in BF16 and rounds differently, and a token's prefill bytes must not depend on
    // the size of the chunk that carries it. Otherwise the measured speed frontiers apply (T=17
    // for K=6144, T=20 for K=17408).
    const std::int32_t first_a8 =
        prefill ? 17 : (input_rows == Fp8N5120K6144::kInputRows ? 17 : 20);
    return tokens >= first_a8 ? Fp8LinearAddRoute::A8 : Fp8LinearAddRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    if (x.ne[1] == 1) return fp8_linear_add_decode_launch(x, weight, residual, stream);
    fp8_linear_add_matrix_launch(x, weight, residual, stream);
}

} // namespace

std::size_t fp8_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                    std::int32_t input_rows, LinearPolicy policy,
                                                    std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 linear_add workspace: invalid token interval");
    }
    // For the active route profile, like the dispatch.
    const bool prefill = prefill_routes_active();
    (void)resolve_route(output_rows, input_rows, policy, min_tokens, prefill);
    return resolve_route(output_rows, input_rows, policy, max_tokens, prefill) ==
                   Fp8LinearAddRoute::A8
               ? fp8_a8_workspace_capacity_bytes(
                     max_tokens, input_rows,
                     fp8_linear_add_partial_capacity_bytes(input_rows, max_tokens))
               : 0;
}

void fp8_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                             LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) {
    const bool prefill            = prefill_routes_active();
    const Fp8LinearAddRoute route = resolve_route(weight.n, weight.k, policy, x.ne[1], prefill);
    if (route == Fp8LinearAddRoute::A16) {
        launch_a16(x, weight, residual, stream);
        return;
    }
    fp8_linear_add_a8_launch(x, weight, residual, workspace, stream,
                             prefill && x.ne[1] <= kFp8LinearAddPrefillSmallTMax);
}

} // namespace ninferno::ops::detail
