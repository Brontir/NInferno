#pragma once

#include "ninferno/types.h"
#include "runtime/engine/context_cache/planning_telemetry.h"

#include <chrono>
#include <cstdint>
#include <optional>

namespace ninferno::runtime {

// The monitor side of PlanningTelemetry: times the points a planning call reports and the whole
// call (from construction) for MaterializationDiagnostics. It is write-only for the planner, which
// reports into it and reads nothing back; only the request log reads the fields it fills.
// The clock the request log's planning times are read on; a test passes a fake one.
using DefaultPlanningClock = std::chrono::steady_clock;

template <class Clock = DefaultPlanningClock>
class BasicPlanningStopwatch final : public PlanningTelemetry {
public:
    BasicPlanningStopwatch() : planning_started_(Clock::now()) {}

    void search_started() noexcept override { search_started_ = Clock::now(); }

    void first_improvement() noexcept override {
        if (search_started_ && !first_improvement_) { first_improvement_ = Clock::now(); }
    }

    void search_ended() noexcept override { search_ended_ = Clock::now(); }

    // Fills the wall-time fields of `diagnostics`: planning since construction, and the search
    // (zero, and no first improvement, when the planner did not search).
    void stamp(MaterializationDiagnostics& diagnostics) const noexcept {
        diagnostics.planning_elapsed_ns = elapsed_ns(planning_started_, Clock::now());
        diagnostics.search_elapsed_ns =
            search_started_ && search_ended_ ? elapsed_ns(*search_started_, *search_ended_) : 0;
        diagnostics.first_improvement_ns =
            search_started_ && first_improvement_
                ? std::optional<std::uint64_t>(elapsed_ns(*search_started_, *first_improvement_))
                : std::nullopt;
    }

private:
    [[nodiscard]] static std::uint64_t elapsed_ns(typename Clock::time_point begin,
                                                  typename Clock::time_point end) noexcept {
        const auto count =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        return count > 0 ? static_cast<std::uint64_t>(count) : 0;
    }

    typename Clock::time_point planning_started_;
    std::optional<typename Clock::time_point> search_started_;
    std::optional<typename Clock::time_point> first_improvement_;
    std::optional<typename Clock::time_point> search_ended_;
};

using PlanningStopwatch = BasicPlanningStopwatch<>;

} // namespace ninferno::runtime
