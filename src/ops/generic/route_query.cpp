#include "ops/generic/route_query.h"

#include "core/weight_view.h"
#include "ops/generic/generic_ggml_internal.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>

namespace ninferno::ops::detail::generic {

namespace {

// The weight as the converter lays it out, its bytes never read: an aligned address stands in.
// Filled in place: its view and weight point into its parent.
struct DescribedWeight {
    WeightParent parent;
    WeightView view;
    Weight weight;

    DescribedWeight()                                  = default;
    DescribedWeight(const DescribedWeight&)            = delete;
    DescribedWeight& operator=(const DescribedWeight&) = delete;
};

void describe_weight(DescribedWeight& out, QType qtype, Epilogue epilogue, std::int32_t n,
                     std::int32_t k, std::int32_t parent_rows,
                     std::optional<QuantLayout> layout = std::nullopt) {
    const std::uint64_t own = epilogue == Epilogue::SwiGlu ? 2ULL * static_cast<std::uint64_t>(n)
                                                           : static_cast<std::uint64_t>(n);
    const bool nvfp4 = qtype == QType::NVFP4;
    out.parent.data                 = reinterpret_cast<const std::byte*>(std::uintptr_t{1} << 32);
    out.parent.weight_scale_divisor = nvfp4 ? 1.0F : 0.0F;
    // The stored matrix whole, as the engine passes a projection slice (ops/weight_input.cpp
    // projection_slice: native_weight of the complete parent, the call's rows in Problem::n).
    const std::uint64_t stored = std::max<std::uint64_t>(own, static_cast<std::uint64_t>(parent_rows));
    const std::uint64_t shape[2] = {stored, static_cast<std::uint64_t>(k)};
    out.parent.geometry = weight_geometry(qtype, layout.value_or(default_weight_layout(qtype)), shape);
    out.view.shape      = {stored, static_cast<std::uint64_t>(k)};
    out.view.parts      = {WeightRegion{&out.parent, 0, stored * static_cast<std::uint64_t>(k)}};
    out.weight          = native_weight(out.view, nvfp4 ? 1.0F : 0.0F);
}

// The engine's call as route_query describes it: `n` rows, K-wide contiguous x unless `ldx` says.
CallLayout engine_call(std::int32_t n, std::int32_t k, std::int32_t tokens, Epilogue epilogue,
                       std::optional<std::size_t> workspace_free, bool x_aligned16, int ldx,
                       int segments) {
    return CallLayout{.x_aligned16    = x_aligned16,
                      .ldx            = ldx != 0 ? ldx : k,
                      .tokens         = tokens,
                      .rows           = n,
                      .segments       = segments,
                      .epilogue       = epilogue,
                      .workspace_free = workspace_free};
}

} // namespace

std::optional<Route> route_query(const DeviceClass& device, QType qtype, LinearPolicy policy,
                                 Epilogue epilogue, std::int32_t n, std::int32_t k,
                                 std::int32_t tokens, bool prefill, std::int32_t parent_rows,
                                 std::optional<QuantLayout> layout) {
    if (n <= 0 || k <= 0 || tokens <= 0 || parent_rows < 0) return std::nullopt;
    const ScopedDeviceClass scoped(device);
    DescribedWeight described;
    describe_weight(described, qtype, epilogue, n, k, parent_rows, layout);
    if (!supports(described.weight)) return std::nullopt;
    const Problem problem{qtype, policy, n, k, epilogue};
    return select_route(problem, described.weight, tokens, prefill);
}

std::optional<ResolvedRoute>
resolved_route_query(const DeviceClass& device, QType qtype, LinearPolicy policy, Epilogue epilogue,
                     std::int32_t n, std::int32_t k, std::int32_t tokens,
                     std::optional<std::size_t> workspace_free, bool prefill,
                     std::int32_t parent_rows, bool x_aligned16, int ldx, int segments) {
    if (n <= 0 || k <= 0 || tokens <= 0 || parent_rows < 0) return std::nullopt;
    const ScopedDeviceClass scoped(device);
    DescribedWeight described;
    describe_weight(described, qtype, epilogue, n, k, parent_rows);
    if (!supports(described.weight)) return std::nullopt;
    const Problem problem{qtype, policy, n, k, epilogue};
    // As run() hands it on: a call without workspace takes A16 where its family needs some.
    const Route asked = workspace_free_route(select_route(problem, described.weight, tokens, prefill),
                                             problem, described.weight, tokens, prefill,
                                             workspace_free.has_value());
    return resolve_route(
        asked, described.weight,
        engine_call(n, k, tokens, epilogue, workspace_free, x_aligned16, ldx, segments));
}

namespace {

RouteDescription describe_resolved(const DescribedWeight& described, const Problem& problem,
                                   const ResolvedRoute& resolved, std::int32_t tokens,
                                   bool with_kernel) {
    const Weight& w = described.weight;
    const Route ran = resolved.executed;
    RouteDescription out;
    out.resolved = resolved;
    // ggml's matrix kernel holds its planes (both of a SwiGLU pair's) for one prefill slice, as
    // route_resources sizes it; every other family's are route_workspace_bytes'.
    out.workspace_bytes =
        ran.family == Family::GgmlMmq
            ? mmq_workspace_bytes(problem.k,
                                  problem.epilogue == Epilogue::SwiGlu ? 2 * problem.n : problem.n,
                                  std::min(tokens, kMmqPrefillSlice))
            : route_workspace_bytes(ran, problem.k, tokens);
    out.tile         = route_tile(ran, problem.epilogue, tokens);
    out.traffic      = route_traffic(ran, w, problem, tokens);
    out.prepared     = prepared_rule(resolved, w);
    out.cost_seconds = route_cost_seconds(problem, w, ran, tokens);
    if (with_kernel) {
        try {
            out.kernel = route_resources(ran, w, problem.epilogue, tokens);
        } catch (const std::exception&) {} // no CUDA device to read the binary from
    }
    return out;
}

const char* representation_name(PreparedRepresentation r) noexcept {
    switch (r) {
    case PreparedRepresentation::GgmlQ8_1:
        return "ggml_q8_1";
    case PreparedRepresentation::W4a8Blocks:
        return "w4a8_blocks";
    case PreparedRepresentation::I8Blocks64:
        return "i8_blocks64";
    case PreparedRepresentation::GgmlQ8_1Mmq:
        return "ggml_q8_1_mmq";
    case PreparedRepresentation::Nvfp4:
        return "nvfp4";
    }
    return "unknown";
}

} // namespace

std::optional<RouteDescription> describe_route(const DeviceClass& device, Route asked, QType qtype,
                                               LinearPolicy policy, Epilogue epilogue,
                                               std::int32_t n, std::int32_t k, std::int32_t tokens,
                                               std::optional<std::size_t> workspace_free,
                                               std::int32_t parent_rows, bool with_kernel) {
    if (n <= 0 || k <= 0 || tokens <= 0 || parent_rows < 0) return std::nullopt;
    const ScopedDeviceClass scoped(device);
    DescribedWeight described;
    describe_weight(described, qtype, epilogue, n, k, parent_rows);
    if (!supports(described.weight) || !route_admits(asked, described.weight, policy)) {
        return std::nullopt;
    }
    const Problem problem{qtype, policy, n, k, epilogue};
    const ResolvedRoute resolved = resolve_route(
        asked, described.weight, engine_call(n, k, tokens, epilogue, workspace_free, true, 0, 1));
    return describe_resolved(described, problem, resolved, tokens, with_kernel);
}

std::optional<RouteDescription> describe_call(const DeviceClass& device, QType qtype,
                                              LinearPolicy policy, Epilogue epilogue,
                                              std::int32_t n, std::int32_t k, std::int32_t tokens,
                                              std::optional<std::size_t> workspace_free,
                                              bool prefill, std::int32_t parent_rows,
                                              bool with_kernel) {
    if (n <= 0 || k <= 0 || tokens <= 0 || parent_rows < 0) return std::nullopt;
    const ScopedDeviceClass scoped(device);
    DescribedWeight described;
    describe_weight(described, qtype, epilogue, n, k, parent_rows);
    if (!supports(described.weight)) return std::nullopt;
    const Problem problem{qtype, policy, n, k, epilogue};
    const Route asked =
        workspace_free_route(select_route(problem, described.weight, tokens, prefill), problem,
                             described.weight, tokens, prefill, workspace_free.has_value());
    const ResolvedRoute resolved = resolve_route(
        asked, described.weight, engine_call(n, k, tokens, epilogue, workspace_free, true, 0, 1));
    return describe_resolved(described, problem, resolved, tokens, with_kernel);
}

std::string describe_text(const RouteDescription& d) {
    const auto whole = [](double v) { return std::to_string(static_cast<std::uint64_t>(v)); };
    std::string out =
        "asked=" + route_name(d.resolved.asked) + " runs=" + route_name(d.resolved.executed);
    if (d.resolved.reason != Demotion::None) {
        out += " demoted=" + std::string(demotion_name(d.resolved.reason));
    }
    if (d.resolved.r8_prequantized) out += " r8_prequantized";
    out += " workspace=" + std::to_string(d.workspace_bytes);
    if (d.tile) {
        out += " tile=" + std::to_string(d.tile->out_rows) + "x" + std::to_string(d.tile->tokens);
        if (d.tile->k_split > 1) out += " ksplit=" + std::to_string(d.tile->k_split);
    }
    if (d.traffic) {
        out += " weight_bytes=" + whole(d.traffic->weight_bytes) +
               " activation_bytes=" + whole(d.traffic->activation_bytes) +
               " output_bytes=" + whole(d.traffic->output_bytes);
    }
    if (d.prepared)
        out += std::string(" prepared=") + representation_name(d.prepared->representation);
    if (d.cost_seconds) out += " cost_us=" + std::to_string(*d.cost_seconds * 1e6);
    if (d.kernel) {
        const KernelResources& k = *d.kernel;
        out += " regs=" + std::to_string(k.registers) + " local=" + std::to_string(k.local_bytes) +
               " smem=" + std::to_string(k.static_smem) + "+" + std::to_string(k.dynamic_smem) +
               " threads=" + std::to_string(k.threads) + " grid=" + std::to_string(k.grid_x) + "x" +
               std::to_string(k.grid_y) + "x" + std::to_string(k.grid_z);
    }
    return out;
}

} // namespace ninferno::ops::detail::generic
