#pragma once

// The generic selector's route query for the plan without the weights (#613 PR-9): which kernel a
// trunk linear's prefill call runs on a planned board, asked of the selector itself (ops/generic
// route_query: its rules, its tables, its build) rather than assumed by the choice's tile laws.
// The engine library links the Ops; the plan's own library does not, so a program that plans
// registers it once (ninferno-serve, ninferno), as it registers the MoE op's route query
// (plan_moe_routes.h).

namespace ninferno::runtime {

void register_plan_prefill_route_query();

} // namespace ninferno::runtime
