#pragma once

namespace ninferno::runtime {

// Write-only telemetry of one planning call. The planner reports where its optional search starts,
// first improves on the initial incumbent and ends; the caller may time those points for the
// request log (planning_stopwatch.h). Nothing flows back: the planner reads no clock, and its
// search counts work units (materialization_budget.h), so a planning problem has one answer on
// every host.
class PlanningTelemetry {
public:
    virtual ~PlanningTelemetry() = default;

    virtual void search_started() noexcept {}

    virtual void first_improvement() noexcept {}

    virtual void search_ended() noexcept {}
};

} // namespace ninferno::runtime
