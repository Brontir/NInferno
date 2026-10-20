#include "ninferno/ops/prefill_routes.h"

namespace ninferno::ops {
namespace {

thread_local bool t_prefill_routes   = false;
thread_local bool t_invariant_routes = false;
thread_local int t_round_lanes       = 0;

} // namespace

bool prefill_routes_active() noexcept { return t_prefill_routes; }

ScopedPrefillRoutes::ScopedPrefillRoutes(bool active) noexcept : previous_(t_prefill_routes) {
    t_prefill_routes = active;
}

ScopedPrefillRoutes::~ScopedPrefillRoutes() { t_prefill_routes = previous_; }

bool invariant_routes_active() noexcept { return t_invariant_routes; }

ScopedInvariantRoutes::ScopedInvariantRoutes(bool active) noexcept
    : previous_(t_invariant_routes) {
    t_invariant_routes = active;
}

ScopedInvariantRoutes::~ScopedInvariantRoutes() { t_invariant_routes = previous_; }

int round_lanes() noexcept { return t_round_lanes; }

ScopedRoundLanes::ScopedRoundLanes(int lanes) noexcept : previous_(t_round_lanes) {
    t_round_lanes = lanes;
}

ScopedRoundLanes::~ScopedRoundLanes() { t_round_lanes = previous_; }

} // namespace ninferno::ops
