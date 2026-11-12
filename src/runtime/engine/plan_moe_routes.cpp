#include "runtime/engine/plan_moe_routes.h"

#include "artifact/formats.h"
#include "core/device_class.h"
#include "models/common/weight_use.h"
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

} // namespace

void register_plan_moe_route_query() {
    set_plan_moe_route_query([](const PlanDeviceFacts& device, const PlanModelFacts& model,
                                const PlanExpertLayers& layers, std::uint32_t tokens,
                                ActivationMode activations) -> std::optional<PlanExpertRoute> {
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
            // The experts' policy as the start runs them (models::effective_policy of the
            // recorded one, PlanExpertLayers::policy): Exact and Int8 whatever was recorded; Fast
            // the recorded one, and none where it is not known (a bank that records none).
            ops::LinearPolicy recorded = ops::LinearPolicy::A16Only;
            switch (layers.policy) {
            case PlanExpertPolicy::A16Only:
                recorded = ops::LinearPolicy::A16Only;
                break;
            case PlanExpertPolicy::AllowA8:
                recorded = ops::LinearPolicy::AllowA8;
                break;
            case PlanExpertPolicy::AllowA4:
                recorded = ops::LinearPolicy::AllowA4;
                break;
            case PlanExpertPolicy::Unknown:
                if (activations == ActivationMode::Fast) { return std::nullopt; }
                break;
            }
            shape.policy = models::effective_policy(recorded, activations);
            // On the planned board's class (plan_device_class, as gpu_key_of reads it): a
            // synthetic class, so the answer is moe()'s on that class, which can differ from the
            // real card's where a rule reads what the class leaves out. It only labels the step's
            // basis (DECISIONS 79).
            const auto answer = moe::moe_route_query(plan_device_class(device), shape,
                                                     static_cast<std::int32_t>(tokens), false,
                                                     activations == ActivationMode::Exact);
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
