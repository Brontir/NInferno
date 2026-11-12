#include "runtime/engine/plan_moe_routes.h"

#include "artifact/formats.h"
#include "core/device_class.h"
#include "ninferno/ops/linear.h"
#include "ops/moe/moe_route_query.h"
#include "runtime/engine/plan_dry_run.h"

#include <exception>
#include <optional>
#include <string>

namespace ninferno::runtime {
namespace {

namespace moe = ops::detail::moe;

PlanExpertKernel kernel_of(moe::MoeStageKernel kernel) {
    switch (kernel) {
    case moe::MoeStageKernel::R8:
        return PlanExpertKernel::R8;
    case moe::MoeStageKernel::RegisterMma:
        return PlanExpertKernel::RegisterMma;
    case moe::MoeStageKernel::W4a8:
        return PlanExpertKernel::W4a8;
    case moe::MoeStageKernel::VoltaHmma:
        return PlanExpertKernel::VoltaHmma;
    }
    return PlanExpertKernel::Unknown;
}

// The planned board's class, as gpu_key_of builds it (product/plan plan_speculation.cpp): the chip
// from the PCI id or the SM count, the board table's facts. A synthetic class (no L2 size, no
// driver clocks or bus width, the architecture's features not intersected with this build's):
// the answer is moe()'s on that class, which can differ from the real card's where a rule reads
// what the class leaves out. It only labels the step's basis (DECISIONS 79).
DeviceClass class_of(const PlanDeviceFacts& device) {
    DeviceClass d =
        synthetic_device_class(device.pci_device_id, device.compute_capability / 10,
                               device.compute_capability % 10, device.sms, 0, device.total_bytes);
    d.facts = machine_facts_of(d);
    return d;
}

} // namespace

void register_plan_moe_route_query() {
    set_plan_moe_route_query([](const PlanDeviceFacts& device, const PlanModelFacts& model,
                                const PlanExpertLayers& layers, std::uint32_t tokens,
                                bool exact) -> std::optional<PlanExpertRoute> {
        if (layers.gate_up_format.empty() || layers.down_format.empty() || model.experts == 0 ||
            model.experts_used == 0 || model.hidden == 0 || layers.down_k == 0 ||
            device.compute_capability == 0) {
            return std::nullopt;
        }
        try {
            moe::MoeLayerShape shape;
            shape.gate_up        = artifact::parse_format(layers.gate_up_format);
            shape.down           = artifact::parse_format(layers.down_format);
            shape.gate_up_layout = artifact::parse_layout(layers.gate_up_layout);
            shape.down_layout    = artifact::parse_layout(layers.down_layout);
            shape.experts        = static_cast<std::int32_t>(model.experts);
            shape.top_k          = static_cast<std::int32_t>(model.experts_used);
            shape.hidden         = static_cast<std::int32_t>(model.hidden);
            shape.gate_up_k      = static_cast<std::int32_t>(layers.gate_up_k);
            shape.inter          = static_cast<std::int32_t>(layers.down_k);
            shape.relu2          = !layers.gated;
            // The experts' policy as the start reads it (PlanExpertLayers::policy); not known (a
            // bank that records none): no answer, the step's route unknown.
            switch (layers.policy) {
            case PlanExpertPolicy::A16Only:
                shape.policy = ops::LinearPolicy::A16Only;
                break;
            case PlanExpertPolicy::AllowA8:
                shape.policy = ops::LinearPolicy::AllowA8;
                break;
            case PlanExpertPolicy::AllowA4:
                shape.policy = ops::LinearPolicy::AllowA4;
                break;
            case PlanExpertPolicy::Unknown:
                return std::nullopt;
            }
            const auto answer = moe::moe_route_query(
                class_of(device), shape, static_cast<std::int32_t>(tokens), false, exact);
            if (!answer) { return std::nullopt; }
            return PlanExpertRoute{kernel_of(answer->stages.gate_up.kernel),
                                   kernel_of(answer->stages.down.kernel),
                                   moe::authority_name(answer->authority)};
        } catch (const std::exception&) {
            // A format or layout name the artifact library does not parse.
            return std::nullopt;
        }
    });
}

} // namespace ninferno::runtime
