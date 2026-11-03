#pragma once

// The families' workspace queries for the plan without the weights (#302): each sizes the arena
// with the same functions a start sizes it with, over the model's facts in place of the bound
// weights. The engine library links the Ops these read; the plan's own library does not, so a
// program that plans registers them once (ninferno-serve, ninferno).

namespace ninferno::runtime {

void register_plan_workspace_queries();

} // namespace ninferno::runtime
