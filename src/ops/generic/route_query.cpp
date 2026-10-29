#include "ops/generic/route_query.h"

#include "core/weight_view.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ninferno::ops::detail::generic {

std::optional<Route> route_query(const DeviceClass& device, QType qtype, LinearPolicy policy,
                                 Epilogue epilogue, std::int32_t n, std::int32_t k,
                                 std::int32_t tokens, bool prefill, std::int32_t parent_rows) {
    if (n <= 0 || k <= 0 || tokens <= 0 || parent_rows < 0) return std::nullopt;
    const ScopedDeviceClass scoped(device);
    // The weight as the converter lays it out, its bytes never read: an aligned address stands in.
    const std::uint64_t own = epilogue == Epilogue::SwiGlu ? 2ULL * static_cast<std::uint64_t>(n)
                                                           : static_cast<std::uint64_t>(n);
    const bool nvfp4 = qtype == QType::NVFP4;
    WeightParent parent;
    parent.data                 = reinterpret_cast<const std::byte*>(std::uintptr_t{1} << 32);
    parent.weight_scale_divisor = nvfp4 ? 1.0F : 0.0F;
    // The stored matrix whole, as the engine passes a projection slice (ops/weight_input.cpp
    // projection_slice: native_weight of the complete parent, the call's rows in Problem::n).
    const std::uint64_t stored = std::max<std::uint64_t>(own, static_cast<std::uint64_t>(parent_rows));
    const std::uint64_t shape[2] = {stored, static_cast<std::uint64_t>(k)};
    parent.geometry = weight_geometry(qtype, default_weight_layout(qtype), shape);
    WeightView view;
    view.shape = {stored, static_cast<std::uint64_t>(k)};
    view.parts = {WeightRegion{&parent, 0, stored * static_cast<std::uint64_t>(k)}};
    const Weight weight = native_weight(view, nvfp4 ? 1.0F : 0.0F);
    if (!supports(weight)) return std::nullopt;
    const Problem problem{qtype, policy, n, k, epilogue};
    return select_route(problem, weight, tokens, prefill);
}

} // namespace ninferno::ops::detail::generic
