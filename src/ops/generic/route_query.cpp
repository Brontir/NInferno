#include "ops/generic/route_query.h"

#include "core/weight_view.h"
#include "ops/generic/generic_select_internal.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

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
    return resolve_route(asked, described.weight,
                         CallLayout{.x_aligned16    = x_aligned16,
                                    .ldx            = ldx != 0 ? ldx : k,
                                    .tokens         = tokens,
                                    .rows           = n,
                                    .segments       = segments,
                                    .epilogue       = epilogue,
                                    .workspace_free = workspace_free});
}

std::vector<PricedRouteQuery> priced_routes_query(const DeviceClass& device, QType qtype,
                                                  LinearPolicy policy, Epilogue epilogue,
                                                  std::int32_t n, std::int32_t k,
                                                  std::int32_t tokens) {
    std::vector<PricedRouteQuery> out;
    if (n <= 0 || k <= 0 || tokens <= 0) return out;
    const ScopedDeviceClass scoped(device);
    DescribedWeight described;
    describe_weight(described, qtype, epilogue, n, k, 0);
    if (!supports(described.weight)) return out;
    const Problem problem{qtype, policy, n, k, epilogue};
    price_routes(problem, described.weight, tokens,
                 [&](Route r, double seconds) { out.push_back({r, seconds}); });
    return out;
}

std::uint64_t described_weight_bytes(QType qtype, Epilogue epilogue, std::int32_t n, std::int32_t k,
                                     std::int32_t parent_rows) {
    DescribedWeight described;
    describe_weight(described, qtype, epilogue, n, k, parent_rows);
    return described.weight.payload_bytes;
}

} // namespace ninferno::ops::detail::generic
