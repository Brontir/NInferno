#include "ops/moe/moe_route_query.h"

#include "core/weight_view.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/prefill_routes.h"

#include <cstdint>
#include <exception>
#include <optional>

namespace ninferno::ops::detail::moe {

namespace {

// A bank as the converter lays it out, its bytes never read: an aligned address stands in (as
// route_query's descriptors). Filled in place: its view and weight point into its parent.
struct DescribedBank {
    WeightParent parent;
    WeightView view;
    Weight weight;

    DescribedBank()                                = default;
    DescribedBank(const DescribedBank&)            = delete;
    DescribedBank& operator=(const DescribedBank&) = delete;
};

void describe_bank(DescribedBank& out, QType qtype, QuantLayout layout, std::uint64_t rows,
                   std::uint64_t k) {
    const bool nvfp4                = qtype == QType::NVFP4;
    out.parent.data                 = reinterpret_cast<const std::byte*>(std::uintptr_t{1} << 32);
    out.parent.weight_scale_divisor = nvfp4 ? 1.0F : 0.0F;
    const std::uint64_t shape[2]    = {rows, k};
    out.parent.geometry             = weight_geometry(qtype, layout, shape);
    out.view.shape                  = {rows, k};
    out.view.parts                  = {WeightRegion{&out.parent, 0, rows * k}};
    out.weight                      = native_weight(out.view, nvfp4 ? 1.0F : 0.0F);
}

} // namespace

std::optional<MoeRouteAnswer> moe_route_query(const DeviceClass& device, const MoeLayerShape& layer,
                                              std::int32_t tokens, bool prefill, bool invariant) {
    if (tokens < 1 || layer.experts < 1 || layer.top_k < 1 || layer.hidden < 1 || layer.inter < 1 ||
        layer.gate_up_k < 0) {
        return std::nullopt;
    }
    const ScopedDeviceClass scoped(device);
    const ScopedPrefillRoutes prefill_routes(prefill);
    const ScopedInvariantRoutes invariant_routes(invariant);
    try {
        const auto experts       = static_cast<std::uint64_t>(layer.experts);
        const std::uint64_t sets = layer.relu2 ? 1 : 2;
        const std::uint64_t k    = layer.gate_up_k > 0 ? layer.gate_up_k : layer.hidden;
        DescribedBank gate_up, down;
        describe_bank(gate_up, layer.gate_up, layer.gate_up_layout,
                      experts * sets * static_cast<std::uint64_t>(layer.inter), k);
        describe_bank(down, layer.down, layer.down_layout,
                      experts * static_cast<std::uint64_t>(layer.hidden),
                      static_cast<std::uint64_t>(layer.inter));
        MoeWeights weights;
        // The BF16 router [E, H]: a contiguous descriptor.
        Weight& r       = weights.router;
        r.qtype         = QType::BF16;
        r.layout        = QuantLayout::Contiguous;
        r.qdata         = gate_up.parent.data;
        r.payload       = gate_up.parent.data;
        r.payload_bytes = experts * static_cast<std::size_t>(layer.hidden) * 2;
        r.n = r.shape[0] = r.padded_shape[0] = layer.experts;
        r.k = r.shape[1] = r.padded_shape[1] = layer.hidden;
        r.ndim                               = 2;
        weights.gate_up                      = gate_up.weight;
        weights.down                         = down.weight;
        weights.expert_policy                = layer.policy;
        MoeConfig config;
        config.top_k = layer.top_k;
        if (layer.relu2) { config.activation = MoeActivation::Relu2; }

        // moe()'s own split of the call (Volta's build), then the cascade on the call it runs.
        const std::int32_t split = moe_split_tokens(weights, config, tokens);
        const std::int32_t call  = split > 0 ? split : tokens;
        MoePlanQuery q = moe_plan_query(weights, config, call, split > 0 ? false : prefill);
        MoeRouteAnswer out;
        out.stages      = stage_routes(weights, q.plan);
        out.plan        = q.plan;
        out.authority   = q.authority;
        out.evidence    = std::move(q.evidence);
        out.call_tokens = call;
        return out;
    } catch (const std::exception&) {
        // The op refuses the layer (moe's validation, a geometry the layout cannot hold).
        return std::nullopt;
    }
}

} // namespace ninferno::ops::detail::moe
