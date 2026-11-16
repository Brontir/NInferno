#include "runtime/engine/plan_prefill_routes.h"

#include "core/device_class.h"
#include "core/format_facts.h"
#include "ninferno/ops/linear.h"
#include "ops/generic/route_query.h"
#include "runtime/engine/plan_dry_run.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <optional>
#include <string_view>

namespace ninferno::runtime {
namespace {

namespace generic = ops::detail::generic;

// The kernel kind a generic route runs on (PlanPrefillKernel): by its family alone, as the
// choice's laws price a family's tiles (a schedule's tile is the family's to give).
PlanPrefillKernel kernel_of(generic::Family family) {
    switch (family) {
    case generic::Family::GgmlMmq:
        return PlanPrefillKernel::Mmq;
    case generic::Family::W4a8Mma:
        return PlanPrefillKernel::W4a8;
    case generic::Family::I8Mma:
    case generic::Family::R8Mma:
    case generic::Family::GgmlVec:
    case generic::Family::W4a4Mma:
        return PlanPrefillKernel::OtherA8;
    case generic::Family::RegisterMma:
    case generic::Family::Scalar:
    case generic::Family::TiledMma:
    case generic::Family::StagedMma:
    case generic::Family::Nvfp4TiledMma:
        return PlanPrefillKernel::A16;
    }
    return PlanPrefillKernel::Unknown;
}

// The planned board's class, as the MoE route query builds it (plan_moe_routes.cpp class_of) and
// gpu_key_of (product/plan plan_speculation.cpp): the chip from the PCI id or the SM count, the
// board table's facts.
DeviceClass class_of(const PlanDeviceFacts& device) {
    DeviceClass d =
        synthetic_device_class(device.pci_device_id, device.compute_capability / 10,
                               device.compute_capability % 10, device.sms, 0, device.total_bytes);
    d.facts = machine_facts_of(d);
    return d;
}

} // namespace

void register_plan_prefill_route_query() {
    set_plan_prefill_route_query([](const PlanDeviceFacts& device, std::string_view format,
                                    double n, double k, bool gated, std::uint32_t tokens,
                                    ActivationMode activations) {
        const std::optional<QType> qtype = find_format(format);
        // The call's output rows: a gated pair's SwiGLU writes one of its two matrices' rows.
        const double rows = gated ? n / 2 : n;
        if (!qtype || device.compute_capability == 0 || tokens == 0 || rows < 1 || k < 1 ||
            rows != std::floor(rows) || k != std::floor(k) || rows > INT32_MAX || k > INT32_MAX) {
            return PlanPrefillKernel::Unknown;
        }
        const ops::LinearPolicy policy = activations == ActivationMode::Exact
                                             ? ops::LinearPolicy::A16Only
                                             : ops::LinearPolicy::AllowA8;
        try {
            const std::optional<generic::Route> route =
                generic::route_query(class_of(device), *qtype, policy,
                                     gated ? generic::Epilogue::SwiGlu : generic::Epilogue::Store,
                                     static_cast<std::int32_t>(rows), static_cast<std::int32_t>(k),
                                     static_cast<std::int32_t>(tokens), true);
            return route ? kernel_of(route->family) : PlanPrefillKernel::Unknown;
        } catch (const std::exception&) {
            // A shape the described layout cannot hold, or a refusal of the selector's.
            return PlanPrefillKernel::Unknown;
        }
    });
}

} // namespace ninferno::runtime
