#pragma once

// The MoE op's route query for the plan without the weights (#616 step 7): which kernel each
// grouped stage of a routed layer runs on a decode step of a planned board, asked of the op itself
// (ops/moe moe_route_query: its cascade, its tables, its build) rather than restated. The engine
// library links the Ops; the plan's own library does not, so a program that plans registers it
// once (ninferno-serve, ninferno), as it registers the workspace queries (plan_workspace.h).

namespace ninferno::runtime {

void register_plan_moe_route_query();

} // namespace ninferno::runtime
