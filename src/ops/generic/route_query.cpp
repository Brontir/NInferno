#include "ops/generic/route_query.h"

#include "core/weight_view.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_select_internal.h"

#include <algorithm>
#include <charconv>
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
                                   std::optional<std::size_t> workspace_free, bool with_kernel) {
    const Weight& w = described.weight;
    const Route ran = resolved.executed;
    RouteDescription out;
    out.resolved        = resolved;
    out.workspace_bytes = executed_workspace_bytes(ran, resolved.r8_prequantized, problem.k,
                                                   problem.n, problem.epilogue, tokens);
    // Volta's prefill band runs a tiled call as CUTLASS over a decoded FP16 copy of the weight
    // rows (run_route, launch_volta_cutlass) when the call has workspace that holds a chunk: the
    // chunk that fits is the workspace it takes. Without room the tiled figure stands (a format
    // with no tile of its own then runs scalar).
    if (ran.family == Family::TiledMma && volta_cutlass_compiled() && workspace_free) {
        try {
            if (volta_cutlass_runs(volta_cutlass_only(codec_of(w)), tokens)) {
                const std::size_t chunk = volta_cutlass_chunk_bytes(
                    problem.n, problem.k, tokens, problem.epilogue == Epilogue::SwiGlu,
                    *workspace_free);
                if (chunk != 0) out.workspace_bytes = chunk;
            }
        } catch (const std::exception&) {
            // a format without a codec has no CUTLASS copy: the tiled figure stands
        }
    }
    out.tile = route_tile(ran, problem.epilogue, tokens);
    // The call's own rows of the weight: a slice of a fused parent reads its share of the payload
    // (route_traffic takes the payload over the call's n rows as bytes per weight). The share is
    // proportional to rows: close, not exact, where scales are swizzled or padded to row tiles.
    Weight call_weight         = w;
    const std::uint64_t own    = problem.epilogue == Epilogue::SwiGlu
                                     ? 2ULL * static_cast<std::uint64_t>(problem.n)
                                     : static_cast<std::uint64_t>(problem.n);
    const std::uint64_t stored = described.view.shape[0];
    if (stored > own) call_weight.payload_bytes = w.payload_bytes / stored * own;
    out.traffic      = route_traffic(ran, call_weight, problem, tokens);
    out.prepared     = prepared_rule(resolved, w);
    out.cost_seconds = route_cost_seconds(problem, call_weight, ran, tokens);
    if (with_kernel) {
        try {
            out.kernel = route_resources(ran, w, problem.epilogue, tokens, problem.n,
                                         ran.family == Family::R8Mma
                                             ? std::optional<bool>(resolved.r8_prequantized)
                                             : std::nullopt);
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
    return describe_resolved(described, problem, resolved, tokens, workspace_free, with_kernel);
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
    return describe_resolved(described, problem, resolved, tokens, workspace_free, with_kernel);
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
    if (d.cost_seconds) {
        // Locale-independent: std::to_chars, three decimals.
        char digits[64];
        const auto end = std::to_chars(digits, digits + sizeof(digits), *d.cost_seconds * 1e6,
                                       std::chars_format::fixed, 3);
        out += " cost_us=" + std::string(digits, end.ptr);
    }
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
