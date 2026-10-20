#include "ninferno/ops/gemma4_moe.h"

#include "core/layout.h"
#include "core/weight.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/common/native_nvfp4.h"
#include "ops/gemma4_moe/gemma4_moe_internal.h"
#include "ops/host_experts/host_experts.h"
#include "ops/linear/nvfp4/nvfp4_format.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::ops {
namespace {

using namespace detail::gemma4;

struct AddressRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end   = 0;
    std::string name;
};

AddressRange address_range(const void* pointer, std::size_t bytes, std::string name) {
    if (pointer == nullptr || bytes == 0) {
        throw std::invalid_argument("gemma4_moe: " + name + " storage must be non-empty");
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::overflow_error("gemma4_moe: " + name + " address range overflows");
    }
    return {begin, begin + bytes, std::move(name)};
}

void require_disjoint(const std::vector<AddressRange>& ranges) {
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        for (std::size_t j = i + 1; j < ranges.size(); ++j) {
            if (ranges[i].begin < ranges[j].end && ranges[j].begin < ranges[i].end) {
                throw std::invalid_argument("gemma4_moe: " + ranges[i].name + " overlaps " +
                                            ranges[j].name);
            }
        }
    }
}

std::int32_t require_activation(const Tensor& tensor, const char* name) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != kHidden || tensor.ne[1] < 1 ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("gemma4_moe: invalid ") + name);
    }
    return tensor.ne[1];
}

void require_vector(const Tensor& tensor, std::int32_t length, const char* name) {
    if (tensor.dtype != DType::FP32 || tensor.ne[0] != length || tensor.ne[1] != 1 ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("gemma4_moe: invalid ") + name);
    }
}

void require_matrix_metadata(const Weight& weight, std::int32_t n, std::int32_t k,
                             const char* name) {
    if (weight.ndim != 2 || weight.n != n || weight.k != k || weight.shape[0] != n ||
        weight.shape[1] != k || weight.shape[2] != 1 || weight.shape[3] != 1 ||
        weight.padded_shape[0] != n || weight.padded_shape[1] != k || weight.padded_shape[2] != 1 ||
        weight.padded_shape[3] != 1) {
        throw std::invalid_argument(std::string("gemma4_moe: invalid shape for ") + name);
    }
}

void require_router(const Weight& weight, std::vector<AddressRange>& ranges) {
    require_matrix_metadata(weight, kExperts, kHidden, "router");
    const std::size_t bytes = static_cast<std::size_t>(kExperts) * kHidden * 2;
    if (weight.qtype != QType::BF16 || weight.layout != QuantLayout::Contiguous ||
        weight.qhigh != nullptr || weight.scales != nullptr || weight.payload_bytes < bytes ||
        !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument("gemma4_moe: router must be aligned contiguous BF16");
    }
    ranges.push_back(address_range(weight.qdata, bytes, "router"));
}

void require_expert_plane(const Weight& weight, std::int32_t rows, std::int32_t k,
                          std::int32_t rows_per_expert, const char* name,
                          std::vector<AddressRange>& ranges) {
    if (weight.qtype != QType::NVFP4) {
        throw std::invalid_argument(std::string("gemma4_moe: ") + name + " must be NVFP4");
    }
#ifdef NINFERNO_WITHOUT_NVFP4
    detail::no_native_nvfp4("gemma4_moe", name);
#else
    require_matrix_metadata(weight, rows, k, name);
    const auto geometry =
        detail::validate_nvfp4_weight(weight, (std::string("gemma4_moe ") + name).c_str(), true);
    // One divisor per expert: a plane assembled from a different number of
    // sources would be read with the wrong divisor for some expert, silently.
    if (weight.weight_divisor_rows != rows_per_expert) {
        throw std::invalid_argument(std::string("gemma4_moe: ") + name +
                                    " must carry one weight divisor per expert");
    }
    ranges.push_back(
        address_range(weight.qdata, geometry.code_plane_bytes, std::string(name) + " codes"));
    ranges.push_back(
        address_range(weight.scales, geometry.scale_plane_bytes, std::string(name) + " scales"));
    ranges.push_back(address_range(weight.weight_divisors, sizeof(float) * kExperts,
                                   std::string(name) + " divisors"));
#endif
}

struct Validated {
    std::int32_t tokens = 0;
    std::vector<AddressRange> ranges;
};

Validated validate(const Tensor& router_input, const Tensor& expert_input,
                   const Gemma4MoeWeights& weights, float eps, const Tensor& out) {
    Validated result;
    result.tokens = require_activation(router_input, "router_input");
    if (require_activation(expert_input, "expert_input") != result.tokens ||
        require_activation(out, "out") != result.tokens) {
        throw std::invalid_argument("gemma4_moe: inputs and out must have one token count");
    }
    if (!(eps >= 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument("gemma4_moe: eps must be finite and non-negative");
    }
    auto& ranges = result.ranges;
    ranges.reserve(16);
    ranges.push_back(address_range(router_input.data, router_input.bytes(), "router_input"));
    // The two inputs may be one tensor; any other overlap is an error.
    if (expert_input.data != router_input.data) {
        ranges.push_back(address_range(expert_input.data, expert_input.bytes(), "expert_input"));
    }
    ranges.push_back(address_range(out.data, out.bytes(), "out"));
    require_router(weights.router, ranges);
    require_vector(weights.router_scale, kHidden, "router_scale");
    require_vector(weights.expert_scale, kExperts, "expert_scale");
    require_vector(weights.gate_up_input_divisors, kExperts, "gate_up_input_divisors");
    require_vector(weights.down_input_divisors, kExperts, "down_input_divisors");
    ranges.push_back(
        address_range(weights.router_scale.data, weights.router_scale.bytes(), "router_scale"));
    ranges.push_back(
        address_range(weights.expert_scale.data, weights.expert_scale.bytes(), "expert_scale"));
    ranges.push_back(address_range(weights.gate_up_input_divisors.data,
                                   weights.gate_up_input_divisors.bytes(),
                                   "gate_up_input_divisors"));
    ranges.push_back(address_range(weights.down_input_divisors.data,
                                   weights.down_input_divisors.bytes(), "down_input_divisors"));
    require_expert_plane(weights.gate_up, kGateUpTotal, kHidden, kGateUpRows, "gate_up", ranges);
    require_expert_plane(weights.down, kDownTotal, kInter, kHidden, "down", ranges);
    return result;
}

template <class Arena>
void allocate_route(Arena& arena, Route route, std::int32_t tokens) {
    switch (route) {
    case Route::Decode:
        (void)allocate_decode_workspace(arena);
        return;
    case Route::SmallT:
        (void)allocate_small_t_workspace(arena, tokens);
        return;
    case Route::PrefillA4:
    case Route::PrefillA16:
        (void)allocate_prefill_workspace(arena, std::min(tokens, kSliceMax));
        return;
    }
    throw std::logic_error("gemma4_moe: unknown route");
}

void require_route_domain(Route route, std::int32_t tokens) {
    if (route == Route::Decode && tokens != 1) {
        throw std::invalid_argument("gemma4_moe: the decode route takes one token");
    }
    if (route == Route::SmallT && tokens > kSmallTCapacity) {
        throw std::invalid_argument("gemma4_moe: too many tokens for the small-T route");
    }
}

} // namespace

namespace detail::gemma4 {

std::size_t route_workspace_bytes(Route route, std::int32_t tokens) {
    if (tokens < 1) { throw std::invalid_argument("gemma4_moe workspace: invalid token count"); }
    require_route_domain(route, tokens);
    WorkspaceLayoutBuilder layout;
    allocate_route(layout, route, tokens);
    return layout.peak_bytes(1);
}

void gemma4_moe_with_route(const Tensor& router_input, const Tensor& expert_input,
                           const Gemma4MoeWeights& weights, float eps, Tensor& out,
                           WorkspaceArena& workspace, cudaStream_t stream, Route route,
                           A4Input input, bool narrow, DecodeOptions decode) {
    Validated checked         = validate(router_input, expert_input, weights, eps, out);
    const std::int32_t tokens = checked.tokens;
    require_route_domain(route, tokens);
    const std::size_t required = route_workspace_bytes(route, tokens);
    if (workspace.base() == nullptr || workspace.capacity() < required ||
        workspace.used() > workspace.capacity() - required) {
        throw std::invalid_argument("gemma4_moe: insufficient workspace capacity");
    }
    checked.ranges.push_back(address_range(workspace.base(), workspace.capacity(), "workspace"));
    require_disjoint(checked.ranges);

    auto scope = workspace.scope();
    switch (route) {
    case Route::Decode: {
        const DecodeWorkspace views = allocate_decode_workspace(workspace);
        launch_decode(router_input, expert_input, weights, eps, out, views, decode, stream);
        return;
    }
    case Route::SmallT: {
        const SmallTWorkspace views = allocate_small_t_workspace(workspace, tokens);
        launch_small_t(router_input, expert_input, weights, eps, out, views, stream);
        return;
    }
    case Route::PrefillA4:
    case Route::PrefillA16: {
        const std::int32_t capacity  = std::min(tokens, kSliceMax);
        const PrefillWorkspace views = allocate_prefill_workspace(workspace, capacity);
        launch_prefill(router_input, expert_input, weights, eps, out, views, capacity,
                       PrefillOptions{route, input, narrow}, stream);
        return;
    }
    }
    throw std::logic_error("gemma4_moe: unknown route");
}

} // namespace detail::gemma4

bool gemma4_moe_available() noexcept { return detail::kNativeNvfp4; }

std::size_t gemma4_moe_workspace_bytes(std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("gemma4_moe workspace: invalid token interval");
    }
    std::size_t required = 0;
    if (min_tokens == 1) { required = route_workspace_bytes(Route::Decode, 1); }
    const std::int32_t small_first = std::max(min_tokens, kSmallTMin);
    const std::int32_t small_last  = std::min(max_tokens, kSmallTMax);
    if (small_first <= small_last) {
        required = std::max(required, route_workspace_bytes(Route::SmallT, small_last));
    }
    // The A4 and A16 prefill tiers carve one layout; either sizes both. The invariant route profile
    // takes the A16 prefill route at every T, and may be chosen after the capacity is sized.
    required = std::max({required, route_workspace_bytes(Route::PrefillA4, max_tokens),
                         route_workspace_bytes(Route::PrefillA16, max_tokens)});
    return required;
}

void gemma4_moe(const Tensor& router_input, const Tensor& expert_input,
                const Gemma4MoeWeights& weights, float eps, Tensor& out, WorkspaceArena& workspace,
                cudaStream_t stream, Gemma4MoeActivations activations) {
    if (activations != Gemma4MoeActivations::Recorded &&
        activations != Gemma4MoeActivations::A16Only) {
        throw std::invalid_argument("gemma4_moe: invalid activation choice");
    }
    const std::int32_t tokens = router_input.ne[1];
    // The invariant route profile (--activations exact, prefill_routes.h): the A16 prefill route at
    // every T, a single token included. Its tensor-core router splits K the same way at every T and
    // its grouped GEMMs sum each column over the whole K in one order, so a token's bytes do not
    // depend on the call; decode and small-T round differently from it.
    const Route route = invariant_routes_active()
                            ? Route::PrefillA16
                            : default_route(tokens > 0 ? tokens : 1, activations);
    detail::gemma4::gemma4_moe_with_route(router_input, expert_input, weights, eps, out, workspace,
                                          stream, route, A4Input::PerExpert);
}

void gemma4_moe_host_experts(const Tensor& router_input, const Tensor& expert_input,
                             const Gemma4MoeWeights& weights, const Gemma4MoeWeights& host_weights,
                             float eps, Tensor& out, WorkspaceArena& workspace,
                             cudaStream_t stream, HostExperts& host) {
    Validated checked         = validate(router_input, expert_input, weights, eps, out);
    const std::int32_t tokens = checked.tokens;
    if (tokens > HostExperts::max_tokens()) {
        throw std::invalid_argument("gemma4_moe: the CPU expert path takes at most 16 tokens");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_host_select_workspace(layout, tokens);
    const std::size_t required = layout.peak_bytes(1);
    if (workspace.base() == nullptr || workspace.capacity() < required ||
        workspace.used() > workspace.capacity() - required) {
        throw std::invalid_argument("gemma4_moe: insufficient workspace capacity");
    }
    checked.ranges.push_back(address_range(workspace.base(), workspace.capacity(), "workspace"));
    require_disjoint(checked.ranges);

    // The CPU side is checked before the first launch.
    const auto same_plane = [](const Weight& a, const Weight& b) {
        return a.qtype == b.qtype && a.layout == b.layout && a.n == b.n && a.k == b.k;
    };
    if (!same_plane(weights.gate_up, host_weights.gate_up) ||
        !same_plane(weights.down, host_weights.down)) {
        throw std::invalid_argument(
            "gemma4_moe: host_weights must be the same layer's experts as weights");
    }
    host::Gemma4Layer layer;
    layer.gate_up = host::host_plane(host_weights.gate_up);
    layer.down    = host::host_plane(host_weights.down);

    auto scope                      = workspace.scope();
    const HostSelectWorkspace views = allocate_host_select_workspace(workspace, tokens);
    detail::gemma4::launch_host_select(router_input, weights, eps, views, stream);
    layer.experts = kExperts;
    layer.hidden  = kHidden;
    layer.inter   = kInter;
    layer.top_k   = kTopK;
    host.enqueue_gemma4(layer, tokens, expert_input.data,
                        static_cast<const std::int32_t*>(views.selection.ids.data),
                        static_cast<const float*>(views.selection.weights.data), out.data, stream);
}

} // namespace ninferno::ops
