#pragma once

// ninferno::ops - route probe: a tuning instrument that re-asks a route table at another width.
//
// The pure Linear shape tables were tuned on one GPU class; their crossovers move with the SM
// count, L2 and bandwidth of another. While a RouteProbe lives on the calling thread, every table
// selects the route it would take at `tokens_as` tokens (never at fewer tokens than the call
// has, so a capacity-bound route still holds the call), and `quantized_activations` overrides
// the A4/A8-versus-A16 decision of the NVFP4 and FP8 shapes and of the fused NVFP4 plans. The
// product chooses activations through the weight policy (ActivationMode, ninferno/types.h), never
// through a probe. The probe changes which compiled route runs, never what a route computes, so
// timing a call under each probe times every route a table holds on the running device
// (bench/ops/linear_route_sweep_bench.cu). The engine never constructs one; unprobed, the hooks
// below return their argument.

#include <cstdint>

namespace ninferno::ops::detail {

struct RouteProbeState {
    std::int32_t tokens_as    = 0;  // 0: no override
    std::int8_t quantized_act = -1; // -1: table decides, 0: force A16, 1: force A4/A8
};

inline thread_local RouteProbeState route_probe_state{};

// The width a route table is asked about for a call of `tokens` tokens.
[[nodiscard]] inline std::int32_t route_tokens(std::int32_t tokens) noexcept {
    const std::int32_t as = route_probe_state.tokens_as;
    return as > tokens ? as : tokens;
}

// The A4/A8 decision for a call the table planned as `planned`.
[[nodiscard]] inline bool route_quantized_activations(bool planned) noexcept {
    const std::int8_t forced = route_probe_state.quantized_act;
    return forced < 0 ? planned : forced != 0;
}

class RouteProbe {
public:
    RouteProbe(std::int32_t tokens_as, std::int8_t quantized_act = -1) noexcept
        : previous_(route_probe_state) {
        route_probe_state = {tokens_as, quantized_act};
    }
    ~RouteProbe() { route_probe_state = previous_; }
    RouteProbe(const RouteProbe&)            = delete;
    RouteProbe& operator=(const RouteProbe&) = delete;

private:
    RouteProbeState previous_;
};

} // namespace ninferno::ops::detail
