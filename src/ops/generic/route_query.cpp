#include "ops/generic/route_query.h"

#include "core/weight_view.h"

#include <cstddef>
#include <cstdint>

namespace ninferno::ops::detail::generic {

const char* route_arithmetic_name(RouteArithmetic a) {
    switch (a) {
    case RouteArithmetic::Dp4a: return "dp4a";
    case RouteArithmetic::Bf16Mma: return "bf16_mma";
    case RouteArithmetic::Int8Mma: return "int8_mma";
    case RouteArithmetic::Fp4Mma: return "fp4_mma";
    case RouteArithmetic::Fma: return "fma";
    }
    return "?";
}

RouteShape shape_of_route(Route route) {
    RouteShape out;
    out.route = route;
    switch (route.family) {
    case Family::RegisterMma: {
        // m16n8k16: every n tile 8 token columns.
        const auto s   = rmma_schedule_info(route.schedule);
        out.arithmetic = RouteArithmetic::Bf16Mma;
        out.token_tile = 8 * s.n_tiles;
        break;
    }
    case Family::Scalar:
        out.arithmetic = RouteArithmetic::Fma;
        out.token_tile = 1;
        break;
    case Family::W4a4Mma:
        out.arithmetic = RouteArithmetic::Fp4Mma;
        out.token_tile = w4a4_schedule_info(route.schedule).block_m;
        break;
    case Family::TiledMma:
        out.arithmetic = RouteArithmetic::Bf16Mma;
        out.token_tile = tiled_schedule_info(route.schedule).block_m;
        break;
    case Family::W4a8Mma:
        out.arithmetic = RouteArithmetic::Int8Mma;
        out.token_tile = w4a8_schedule_info(route.schedule).block_m;
        break;
    case Family::StagedMma:
        out.arithmetic = RouteArithmetic::Bf16Mma;
        out.token_tile = staged_schedule_info(route.schedule).tokens;
        break;
    case Family::Nvfp4TiledMma:
        out.arithmetic = RouteArithmetic::Bf16Mma;
        out.token_tile = nvfp4_tiled_schedule_info(route.schedule).block_m;
        break;
    case Family::R8Mma:
        // m16n8k32 with the (at most 8) tokens as its 8 columns.
        out.arithmetic = RouteArithmetic::Int8Mma;
        out.token_tile = 8;
        break;
    case Family::GgmlMmq:
        // ggml's int8 tiles, at their narrowest 8 token columns (mmq_x) at verify widths.
        out.arithmetic = RouteArithmetic::Int8Mma;
        out.token_tile = 8;
        break;
    case Family::GgmlVec:
        out.arithmetic = RouteArithmetic::Dp4a;
        out.token_tile = 1;
        break;
    case Family::I8Mma:
        out.arithmetic = RouteArithmetic::Int8Mma;
        out.token_tile = i8_schedule_info(route.schedule).block_m;
        break;
    }
    if (out.token_tile < 1) out.token_tile = 1;
    return out;
}

std::optional<RouteShape> route_shape(const DeviceClass& device, QType qtype, LinearPolicy policy,
                                      Epilogue epilogue, std::int32_t n, std::int32_t k,
                                      std::int32_t tokens, bool prefill) {
    if (n <= 0 || k <= 0 || tokens <= 0) return std::nullopt;
    const ScopedDeviceClass scoped(device);
    // The weight as the converter lays it out, its bytes never read: an aligned address stands in.
    const std::uint64_t rows = epilogue == Epilogue::SwiGlu ? 2ULL * static_cast<std::uint64_t>(n)
                                                            : static_cast<std::uint64_t>(n);
    const std::uint64_t shape[2] = {rows, static_cast<std::uint64_t>(k)};
    WeightParent parent;
    parent.geometry = weight_geometry(qtype, default_weight_layout(qtype), shape);
    parent.data     = reinterpret_cast<const std::byte*>(std::uintptr_t{1} << 32);
    const bool nvfp4 = qtype == QType::NVFP4;
    parent.weight_scale_divisor = nvfp4 ? 1.0F : 0.0F;
    WeightView view;
    view.shape = {rows, static_cast<std::uint64_t>(k)};
    view.parts = {WeightRegion{&parent, 0, rows * static_cast<std::uint64_t>(k)}};
    const Weight weight = native_weight(view, nvfp4 ? 1.0F : 0.0F);
    if (!supports(weight)) return std::nullopt;
    const Problem problem{qtype, policy, n, k, epilogue};
    return shape_of_route(select_route(problem, weight, tokens, prefill));
}

} // namespace ninferno::ops::detail::generic
