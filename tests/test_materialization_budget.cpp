// The admission search budget is counted work, not time: every decision is a pure function of the
// boundary ledger's position, the declared operation costs and the predicted gains, so the same
// inputs give the same answers on any machine and on a CPU.

#include "runtime/engine/context_cache/materialization_budget.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>

using namespace ninferno::runtime;

static void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

int main() {
    // Predicted device nanoseconds whose economic share is one work unit.
    constexpr std::uint64_t unit_gain =
        PlanningWork::kEconomicShare * PlanningWork::kPredictedNsPerUnit;
    try {
        require(PlanningAllowance::boundary(0).limit == 50'000 &&
                    PlanningAllowance::boundary(3).limit == 10'000 &&
                    PlanningAllowance::boundary(3).affected_requests == 4,
                "boundary allowances are not the declared unit counts");
        require(economic_planning_units(80'000 * unit_gain, 1) == 80'000 &&
                    economic_planning_units(80'000 * unit_gain, 4) == 20'000 &&
                    economic_planning_units(UINT64_MAX, 1) == 0,
                "economic share is not a twentieth of the predicted saving per unit");

        auto idle = PlanningAllowance::boundary(0);
        MaterializationSearchBudget expensive(idle, 0, 80'000 * unit_gain);
        require(expensive.granted_units() == PlanningWork::kInitialGrant,
                "the first window is not the declared grant");
        require(expensive.allow(4'000, 2'000, 3'000, 70'000 * unit_gain, true, 1),
                "valuable completion could not cross the initial window");
        require(expensive.granted_units() == 10'000 && expensive.renewals() == 1,
                "extension did not retain cumulative accounting");
        require(!expensive.allow(9'000, 2'000, 2'000, unit_gain, true, 2),
                "improved incumbent retained the expensive root's budget");
        require(expensive.stop_reason() ==
                    ninferno::MaterializationStopReason::InsufficientExpectedGain,
                "economic stopping was reported as budget exhaustion");

        MaterializationSearchBudget discovery(idle, 0, 80'000 * unit_gain);
        require(discovery.allow(5'000, 1'000, 4'000, 70'000 * unit_gain, false, 1),
                "unknown candidate could not receive bounded discovery");
        require(!discovery.allow(10'000, 1'000, 1'000, 70'000 * unit_gain, false, 2),
                "unknown candidate repeatedly renewed discovery");
        require(discovery.allow(10'000, 1'000, 1'000, 70'000 * unit_gain, true, 2),
                "complete prediction could not continue after discovery");

        // Head and backfill inspections share one boundary ledger.
        PlanningLedger ledger;
        auto busy    = PlanningAllowance::boundary(2, &ledger);
        ledger.spent = busy.limit - 4'000;
        MaterializationSearchBudget first(busy, ledger.spent, 80'000 * unit_gain);
        require(first.granted_units() == 4'000, "mandatory work did not consume boundary units");
        busy.charge(3'000);
        MaterializationSearchBudget backfill(busy, ledger.spent, 80'000 * unit_gain);
        require(backfill.granted_units() == 1'000, "backfill reset the boundary allowance");
        require(!backfill.allow(busy.limit, 1, 1, 70'000 * unit_gain, true, 1) &&
                    backfill.boundary_limited() &&
                    backfill.stop_reason() == ninferno::MaterializationStopReason::PlanningBudget,
                "search delayed runnable requests beyond their boundary");
        busy.charge(UINT64_MAX);
        require(ledger.spent == UINT64_MAX && busy.remaining() == 0, "the ledger did not saturate");

        // Mandatory setup must leave a busy boundary room for the first complete reuse
        // assessment: discarding the whole cache makes every later continuation cold.
        MaterializationSearchBudget restore_after_setup(PlanningAllowance::boundary(2), 3'500,
                                                        80'000 * unit_gain);
        require(restore_after_setup.allow(3'500, PlanningWork::kAssessment,
                                          PlanningWork::kAssessment, 70'000 * unit_gain, true, 1),
                "mandatory setup starved the first complete reuse assessment in a busy boundary");

        MaterializationSearchBudget cheap(idle, 0, 1'000 * unit_gain);
        require(cheap.granted_units() == 1'000, "cheap request received the full initial grant");
        require(!cheap.allow(1'000, 20'000, 20'000, 1'000 * unit_gain, false, 1),
                "discovery ignored the economic cap for a cheap request");
        MaterializationSearchBudget saturated(idle, 0, UINT64_MAX);
        require(saturated.granted_units() == 0 &&
                    !saturated.allow(0, 1'000, 1'000, UINT64_MAX, true, 1),
                "saturated cost was used as evidence of unlimited gain");
        MaterializationSearchBudget seeded(idle, 0, 80'000 * unit_gain);
        require(!seeded.allow(5'000, 1'000, 1'000, 70'000 * unit_gain, false, 1, false),
                "an already-seeded candidate renewed solely on an incomplete optimistic estimate");
        require(seeded.allow(5'000, 1'000, 1'000, 70'000 * unit_gain, true, 1, false),
                "a complete profitable refinement was denied after seeding");
        std::atomic<bool> cancelled{false};
        auto controlled         = PlanningAllowance::boundary(0);
        controlled.cancellation = &cancelled;
        require(controlled.remaining(49'000) == 1'000, "remaining units are not the limit's rest");
        cancelled.store(true);
        require(controlled.remaining(0) == 0, "cancelled request kept optional planning headroom");
        MaterializationSearchBudget stalled(idle, 0, 80'000 * unit_gain);
        require(stalled.allow(5'000, 1'000, 1'000, 70'000 * unit_gain, true, 7),
                "first forecast grant failed");
        require(!stalled.allow(10'000, 1'000, 1'000, 70'000 * unit_gain, true, 7),
                "stalled work renewed its allowance");

        // Purity: two budgets fed the same sequence answer identically, step for step.
        const auto replay = [&](std::uint32_t other_runnable) {
            MaterializationSearchBudget budget(PlanningAllowance::boundary(other_runnable), 0,
                                               80'000 * unit_gain);
            std::uint64_t spent = 0;
            std::uint64_t trace = 0;
            for (std::uint64_t step = 0; step < 4'000; ++step) {
                const std::uint64_t cost =
                    step % 3 == 0 ? PlanningWork::kAssessment : PlanningWork::kOption;
                const bool allowed = budget.allow(spent, cost, cost * 4,
                                                  (80'000 - step) * unit_gain, step % 5 != 0, step);
                trace = trace * 1'000'003U + (allowed ? 1U : 2U) + budget.granted_units();
                if (allowed) { spent += cost; }
            }
            return trace ^ spent;
        };
        require(replay(0) == replay(0) && replay(3) == replay(3) && replay(0) != replay(3),
                "the budget is not a pure function of its inputs");
        std::cout << "ok\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
