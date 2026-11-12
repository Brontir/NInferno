#pragma once

// The generic MoE op's route as a pure query (#616 step 7): the plan a call of T tokens of a routed
// layer takes on a board given by its class (core/device_class.h) rather than the current device,
// and the kernel each grouped stage then runs. The answer is moe()'s own: the same cascade of
// authorities (moe_plan_query: the route tables, the class's A8 route, the R8 token limits, Volta's
// HMMA), the same build, and the split moe() makes of a Volta call wider than its R8 limit
// (moe_split_tokens), so a cost model outside the kernels (the configuration choice's decode step,
// product/choice) reads the op instead of restating its rules. The counterpart of the projection
// path's route_query (ops/generic/route_query.h).
//
// Nothing is read but `device`, the build and the route tables on disk: the layer's banks are
// descriptors in the layout given (their bytes never read; an aligned address stands in), the
// router a BF16 descriptor of [E, H]. The query substitutes the class for the call
// (ScopedDeviceClass), as the plan goldens do (tests/ops/test_moe_plan_golden.cpp, which checks
// that this answer is the one the goldens freeze for every board, layer and width). A load-time
// tuner's plan is the device's own (TuneMode::Off in the product): a substituted board has none.

#include "core/device_class.h"
#include "core/weight.h"
#include "ninferno/ops/linear.h"
#include "ops/moe/moe_internal.h"

#include <cstdint>
#include <optional>

namespace ninferno::ops::detail::moe {

// One routed layer as a plan reads it: the banks' formats and stored layouts, the experts, the
// top-k, the hidden width H (the router's K), the gate/up bank's K (H, or a zero-padded width above
// it: 0 reads as H), the expert width F, and the experts' activation policy (moe_policy of the two
// banks' recorded policies).
struct MoeLayerShape {
    QType gate_up = QType::BF16, down = QType::BF16;
    QuantLayout gate_up_layout = QuantLayout::RowSplit, down_layout = QuantLayout::RowSplit;
    std::int32_t experts = 0, top_k = 0, hidden = 0, gate_up_k = 0, inter = 0;
    bool relu2          = false; // a non-gated expert (Nemotron-H): one up set, no gate
    LinearPolicy policy = LinearPolicy::AllowA8;
};

struct MoeRouteAnswer {
    MoePlan plan;
    MoeStageRoutes stages;
    MoeAuthority authority = MoeAuthority::DecodeStreaming;
    Evidence evidence;
    // The width of the calls moe() runs the call as (moe_split_tokens: Volta's build splits a call
    // wider than its R8 limit into calls of that many tokens, each on its own plan; the plan and
    // stages above are the first call's); the call's own T where it runs whole.
    std::int32_t call_tokens = 0;
};

// The route a call of `tokens` tokens of `layer` takes on `device` under the speed profile, or the
// prefill profile (`prefill`) or the invariant one (`invariant`: --activations exact, every route
// token invariant). nullopt where the op refuses the layer (its validation throws: a format,
// layout, K or geometry no grouped kernel reads) or the shape is empty.
[[nodiscard]] std::optional<MoeRouteAnswer>
moe_route_query(const DeviceClass& device, const MoeLayerShape& layer, std::int32_t tokens,
                bool prefill = false, bool invariant = false);

} // namespace ninferno::ops::detail::moe
