#pragma once

#include "ninferno/types.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>

namespace ninferno::runtime {

// Optional admission planning is bounded by counted work, never by a clock, so the same catalog,
// prompt and boundary make the same plan on every machine and on a CPU
// (docs/maintainer/deterministic-decisions.md). Each Program operation of the search is charged
// in work units from discrete facts of the operation itself. One unit is the nominal host cost of
// one construction-option step (about 1 us). An assessment adds one unit per owner decision or
// checkpoint-recovery alternative it projected (its projection_work), an expansion one per target
// it committed, so a larger catalog costs more without a clock. The planner predicts the next
// operation's cost from the units the previous ones were charged.
//
// No clock bounds the search, so the units must bound its wall time on the slowest host. The
// seed estimates were 20 us an assessment or an expansion, but an assessment of a large catalog
// can take about 200 us of host time beyond what its projection count shows; charged 20 units, a
// busy boundary of 10 000 units could then run 500 assessments, 100 ms of every waiting lane's
// time. An assessment and an expansion are therefore charged at that worst cost, 200 units, so a
// boundary's worst case stays at its former wall size: 10 000 units, at most 50 assessments or
// about 10 ms, while other requests wait, 50 000 (about 50 ms) when none does. A construction
// option is charged its nominal 1 unit and assumed to cost at most kWorstOptionUs (1 us): a
// boundary of options alone stays within 10 ms only while that holds, which no lab host has
// measured yet. A typical
// assessment costs far less, so a typical search now does fewer assessments than the clock
// allowed. Every constant here is a declared estimate awaiting calibration on lab hosts against
// the request log's materialization telemetry (search_elapsed_ns against search_units,
// planning_elapsed_ns against planning_units); a wrong one changes how much optional planning a
// boundary does, never a request's output.
struct PlanningWork {
    // Per operation: a construction option; an exact target assessment, plus its projections;
    // a target expansion, plus its new targets.
    static constexpr std::uint64_t kOption     = 1;
    static constexpr std::uint64_t kAssessment = 200;
    static constexpr std::uint64_t kExpansion  = 200;
    // The worst host cost of one assessment, and the assumed worst of one construction option,
    // the boundary sizes are checked against.
    static constexpr std::uint64_t kWorstAssessmentUs = 200;
    static constexpr std::uint64_t kWorstOptionUs     = 1;
    // Units of one Engine admission boundary.
    static constexpr std::uint64_t kIdleBoundary = 50'000;
    static constexpr std::uint64_t kBusyBoundary = 10'000;
    // The first window of a search, and the most one discovery episode may add.
    static constexpr std::uint64_t kInitialGrant   = 5'000;
    static constexpr std::uint64_t kDiscoveryGrant = 5'000;
    // The economic rule: a search may spend a twentieth of the predicted saving, shared by the
    // affected requests; one unit is worth kPredictedNsPerUnit predicted device nanoseconds.
    static constexpr std::uint64_t kEconomicShare      = 20;
    static constexpr std::uint64_t kPredictedNsPerUnit = 1'000;
};

// A busy boundary at the worst assessment cost stays within its former 10 ms of wall time, an idle
// one within 50 ms.
static_assert(PlanningWork::kBusyBoundary / PlanningWork::kAssessment *
                      PlanningWork::kWorstAssessmentUs <=
                  10'000,
              "a busy boundary's worst case exceeds 10 ms");
static_assert(PlanningWork::kIdleBoundary / PlanningWork::kAssessment *
                      PlanningWork::kWorstAssessmentUs <=
                  50'000,
              "an idle boundary's worst case exceeds 50 ms");
static_assert(PlanningWork::kBusyBoundary / PlanningWork::kOption * PlanningWork::kWorstOptionUs <=
                  10'000,
              "a busy boundary of options exceeds 10 ms");

// The units spent at one Engine admission boundary: the head inspection and every backfill
// inspection charge the same ledger, so none of them resets the boundary's allowance.
struct PlanningLedger {
    std::uint64_t spent = 0;
};

// One Engine admission boundary. Correctness work and sealing cannot be cancelled by this
// allowance; the mandatory identity assessments are charged to it all the same, so they reduce the
// optional headroom left to the search.
struct PlanningAllowance {
    std::uint64_t limit                   = PlanningWork::kIdleBoundary;
    std::uint32_t affected_requests       = 1;
    const std::atomic<bool>* cancellation = nullptr;
    // Null: the allowance belongs to one planning call (the planner then keeps its own ledger).
    PlanningLedger* ledger = nullptr;

    [[nodiscard]] static PlanningAllowance boundary(std::uint32_t other_runnable,
                                                    PlanningLedger* ledger = nullptr) noexcept {
        return {.limit =
                    other_runnable == 0 ? PlanningWork::kIdleBoundary : PlanningWork::kBusyBoundary,
                .affected_requests = 1U + other_runnable,
                .ledger            = ledger};
    }

    [[nodiscard]] std::uint64_t spent() const noexcept { return ledger ? ledger->spent : 0; }

    void charge(std::uint64_t units) const noexcept {
        if (ledger == nullptr) { return; }
        ledger->spent = units > std::numeric_limits<std::uint64_t>::max() - ledger->spent
                            ? std::numeric_limits<std::uint64_t>::max()
                            : ledger->spent + units;
    }

    // Units left once `spent` of the boundary is used. A cancelled request (the client's own
    // action, not a timing) keeps no optional headroom.
    [[nodiscard]] std::uint64_t remaining(std::uint64_t spent) const noexcept {
        if (cancellation && cancellation->load(std::memory_order_relaxed)) { return 0; }
        return spent < limit ? limit - spent : 0;
    }

    [[nodiscard]] std::uint64_t remaining() const noexcept { return remaining(spent()); }
};

// The economic cap in units for a predicted saving of `gain_ns` shared by `affected` requests.
// A saturated (unknown) cost is no evidence of an unlimited gain.
[[nodiscard]] constexpr std::uint64_t economic_planning_units(std::uint64_t gain_ns,
                                                              std::uint32_t affected) noexcept {
    if (gain_ns == std::numeric_limits<std::uint64_t>::max()) { return 0; }
    return gain_ns / PlanningWork::kEconomicShare / PlanningWork::kPredictedNsPerUnit /
           std::max(1U, affected);
}

// A pure function of its inputs: `spent` is the boundary ledger's position (the planner passes it,
// never a clock), operation and completion costs are in work units, and gains are predicted device
// nanoseconds. Estimates control optional work only; they never certify feasibility or a search
// bound.
class MaterializationSearchBudget {
public:
    MaterializationSearchBudget(PlanningAllowance allowance, std::uint64_t started,
                                std::uint64_t initial_cost_ns) noexcept
        : allowance_(allowance), started_(started),
          granted_(std::min({PlanningWork::kInitialGrant, economic(initial_cost_ns),
                             allowance.remaining(started)})) {}

    [[nodiscard]] bool allow(std::uint64_t spent, std::uint64_t next_operation,
                             std::uint64_t completion, std::uint64_t gain_ns,
                             bool complete_prediction, std::uint64_t progress,
                             bool discovery_eligible = true) noexcept {
        boundary_limited_             = false;
        const std::uint64_t used      = spent > started_ ? spent - started_ : 0;
        const std::uint64_t remaining = allowance_.remaining(spent);
        next_operation                = std::max<std::uint64_t>(1, next_operation);
        completion                    = std::max(next_operation, completion);
        if (next_operation > remaining) {
            boundary_limited_ = true;
            reason_           = MaterializationStopReason::PlanningBudget;
            return false;
        }
        if (used < granted_ && next_operation <= granted_ - used) { return true; }
        if (completion > remaining || completion > economic(gain_ns)) {
            reason_ = MaterializationStopReason::InsufficientExpectedGain;
            return false;
        }
        // Unknown forecasts get one bounded discovery episode, not a fresh grant for every node.
        if (!complete_prediction && (!discovery_eligible || discovery_used_)) {
            reason_ = MaterializationStopReason::InsufficientExpectedGain;
            return false;
        }
        if (renewals_ != 0 && progress <= renewal_progress_) {
            reason_ = MaterializationStopReason::InsufficientExpectedGain;
            return false;
        }
        const auto hard   = used + remaining;
        const auto growth = std::max(granted_, next_operation);
        auto added        = std::min(growth, hard > granted_ ? hard - granted_ : 0);
        if (!complete_prediction) { added = std::min(added, PlanningWork::kDiscoveryGrant); }
        if (added == 0 || used + next_operation > granted_ + added) {
            reason_ = MaterializationStopReason::PlanningBudget;
            return false;
        }
        granted_ += added;
        ++renewals_;
        renewal_progress_ = progress;
        discovery_used_   = discovery_used_ || !complete_prediction;
        return true;
    }

    [[nodiscard]] std::uint64_t granted_units() const noexcept { return granted_; }

    [[nodiscard]] std::uint32_t renewals() const noexcept { return renewals_; }

    [[nodiscard]] bool boundary_limited() const noexcept { return boundary_limited_; }

    [[nodiscard]] bool discovery_used() const noexcept { return discovery_used_; }

    [[nodiscard]] MaterializationStopReason stop_reason() const noexcept { return reason_; }

private:
    [[nodiscard]] std::uint64_t economic(std::uint64_t gain_ns) const noexcept {
        return economic_planning_units(gain_ns, allowance_.affected_requests);
    }

    bool boundary_limited_ = false;
    PlanningAllowance allowance_;
    std::uint64_t started_            = 0;
    std::uint64_t granted_            = 0;
    std::uint64_t renewal_progress_   = 0;
    std::uint32_t renewals_           = 0;
    bool discovery_used_              = false;
    MaterializationStopReason reason_ = MaterializationStopReason::PlanningBudget;
};

} // namespace ninferno::runtime
