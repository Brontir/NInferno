#include "runtime/engine/admission_policy.h"
#include "runtime/engine/scheduler.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

struct SchedulerRequest {
    using SequenceHandle = std::uint64_t;

    struct Budget {
        std::uint32_t tokens = 0;

        [[nodiscard]] std::uint32_t remaining() const noexcept { return tokens; }
    };

    struct Output {
        std::uint32_t model_tokens = 0;
        std::vector<ninferno::TokenId> control;

        [[nodiscard]] std::uint32_t
        model_token_budget_remaining(std::uint32_t total) const noexcept {
            return std::min(total, model_tokens);
        }

        [[nodiscard]] std::span<const ninferno::TokenId> pending_control_tokens() const noexcept {
            return control;
        }
    };

    enum class State : std::uint8_t { Decode, Control };

    [[nodiscard]] bool is_decode_ready() const noexcept { return state == State::Decode; }

    [[nodiscard]] bool is_control_ready() const noexcept { return state == State::Control; }

    std::uint64_t id                              = 0;
    std::uint64_t remaining_service_work          = 1;
    std::uint64_t backfill_epoch                  = 0;
    ninferno::runtime::BackfillClass backfill_class = ninferno::runtime::BackfillClass::None;
    State state                                   = State::Decode;
    bool capture_pending                          = false;
    bool lease_parked                             = false;
    std::optional<Budget> budget;
    std::optional<SequenceHandle> sequence;
    Output output;
};

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

template <class Function>
bool throws_logic(Function&& function) {
    try {
        function();
    } catch (const std::logic_error&) { return true; }
    return false;
}

} // namespace

int main() {
    using ninferno::runtime::ActiveAdmissionSnapshot;
    using ninferno::runtime::BackfillClass;
    using ninferno::runtime::ProgramResourceRevision;
    using Scheduler     = ninferno::runtime::Scheduler<SchedulerRequest>;
    const auto revision = [](std::uint64_t value) {
        return ProgramResourceRevision{.value = value};
    };

    int failures = 0;
    const std::array<ActiveAdmissionSnapshot, 2> incumbents{
        ActiveAdmissionSnapshot{.request_id = 1},
        ActiveAdmissionSnapshot{.request_id = 2},
    };

    auto protection = ninferno::runtime::make_admission_protection(7, 10, revision(31), incumbents);
    failures +=
        check(protection.donor_count == 2 && protection.donor_ids[0] == 1 &&
                  protection.donor_ids[1] == 2 && protection.resource_revision == revision(31),
              "protection did not freeze every incumbent donor");
    failures += check(ninferno::runtime::protection_has_live_donor(protection, incumbents),
                      "live donor was not recognized");
    failures += check(ninferno::runtime::persistent_backfill_is_authorized(
                          protection, 11, incumbents, revision(31)),
                      "matching Program proof did not authorize persistent backfill");
    failures += check(!ninferno::runtime::persistent_backfill_is_authorized(protection, 11,
                                                                          incumbents, revision(30)),
                      "stale Program proof authorized persistent backfill");

    const std::array<ActiveAdmissionSnapshot, 3> with_borrower{
        incumbents[0],
        incumbents[1],
        ActiveAdmissionSnapshot{
            .request_id     = 11,
            .backfill_epoch = 7,
            .backfill_class = BackfillClass::Persistent,
        },
    };
    ninferno::runtime::rebind_admission_protection(protection, with_borrower, revision(32));
    failures += check(protection.donor_count == 2 && protection.resource_revision == revision(32) &&
                          ninferno::runtime::persistent_backfill_is_authorized(
                              protection, 12, with_borrower, revision(32)),
                      "revision rebind changed the frozen donor partition");

    const std::array<ActiveAdmissionSnapshot, 1> borrower_only{with_borrower[2]};
    ninferno::runtime::rebind_admission_protection(protection, borrower_only, revision(33));
    failures += check(!ninferno::runtime::protection_has_live_donor(protection, borrower_only) &&
                          !ninferno::runtime::persistent_backfill_is_authorized(
                              protection, 12, borrower_only, revision(33)),
                      "backfill remained open after every frozen donor completed");

    const std::array<ActiveAdmissionSnapshot, 1> unknown_active{
        ActiveAdmissionSnapshot{.request_id = 99},
    };
    failures += check(throws_logic([&] {
                          ninferno::runtime::rebind_admission_protection(protection, unknown_active,
                                                                       revision(34));
                      }),
                      "unclassified post-protection active request was accepted as a donor");

    Scheduler scheduler;
    scheduler.observe_fifo_head(10);
    failures += check(scheduler.protect_blocked_head(10, incumbents, revision(40)),
                      "blocked FIFO head did not open a donor epoch");
    auto stale = scheduler.qualify_backfill(11, 50, incumbents, revision(40));
    failures += check(stale && stale->backfill_class() == BackfillClass::Persistent &&
                          scheduler.validate_grant(*stale),
                      "Program-proved borrower did not receive a persistent grant");
    if (!stale) { return 1; }
    const std::uint64_t epoch = stale->protection_epoch();
    failures += check(scheduler.protect_blocked_head(10, incumbents, revision(41)) &&
                          !scheduler.validate_grant(*stale),
                      "resource revision did not invalidate an uncommitted grant");

    auto first = scheduler.qualify_backfill(11, 50, incumbents, revision(41));
    failures += check(first && first->protection_epoch() == epoch &&
                          first->resource_revision() == revision(41),
                      "revalidated borrower changed the logical protection epoch");
    if (!first) { return 1; }
    scheduler.commit_admission(std::move(*first));

    const std::array<ActiveAdmissionSnapshot, 3> active_after_first{
        incumbents[0],
        incumbents[1],
        ActiveAdmissionSnapshot{
            .request_id     = 11,
            .backfill_epoch = epoch,
            .backfill_class = BackfillClass::Persistent,
        },
    };
    failures += check(scheduler.protect_blocked_head(10, active_after_first, revision(42)),
                      "existing persistent borrower prevented revision revalidation");
    auto second = scheduler.qualify_backfill(12, 1, active_after_first, revision(42));
    failures += check(second && scheduler.validate_grant(*second),
                      "second Program-proved borrower was not cumulatively admitted");

    scheduler.on_waiting_removed(10);
    scheduler.observe_fifo_head(14);
    auto head_grant = scheduler.grant_head(14, 1);
    failures += check(scheduler.validate_grant(head_grant),
                      "new FIFO head inherited stale protection state");
    scheduler.commit_admission(std::move(head_grant));

    using ExecutionAction = Scheduler::ExecutionAction;
    failures += check(scheduler.should_attempt_admission(true, true, false, false, false) &&
                          !scheduler.should_attempt_admission(false, true, true, true, false) &&
                          !scheduler.should_attempt_admission(true, false, true, true, false) &&
                          !scheduler.should_attempt_admission(true, true, true, false, false) &&
                          scheduler.should_attempt_admission(true, true, true, true, false) &&
                          !scheduler.should_attempt_admission(true, true, false, false, true) &&
                          scheduler.choose_execution(true, false, false) == ExecutionAction::Decode,
                      "admission and GPU-unit fairness gates changed");
    scheduler.set_prefill_lane(0);
    failures +=
        check(!scheduler.should_attempt_admission(true, true, true, true, false) &&
                  scheduler.choose_execution(true, true, false) == ExecutionAction::Decode &&
                  scheduler.choose_execution(true, true, true) == ExecutionAction::Prefill,
              "prefill/decode alternation changed");
    scheduler.clear_prefill_lane(0);

    // --concurrent-prefill: admission stays open while lanes prefill; lanes are served in turn.
    scheduler.allow_concurrent_prefill(true);
    scheduler.set_prefill_lane(1);
    scheduler.set_prefill_lane(3);
    const auto every = [](std::uint32_t) { return true; };
    const auto lane_a = scheduler.next_prefill_lane(every);
    scheduler.note_prefill_served(lane_a.value_or(0));
    const auto lane_b = scheduler.next_prefill_lane(every);
    scheduler.note_prefill_served(lane_b.value_or(0));
    const auto lane_c = scheduler.next_prefill_lane(every);
    const auto skip  = scheduler.next_prefill_lane([](std::uint32_t lane) { return lane != 1; });
    failures += check(scheduler.should_attempt_admission(true, true, true, true, false) &&
                          lane_a == 1U && lane_b == 3U && lane_c == 1U && skip == 3U &&
                          scheduler.owns_prefill_lane(3) && !scheduler.owns_prefill_lane(2),
                      "concurrent prefill must keep admission open and serve lanes in turn");
    // Equal GPU share per request, counted in prompt tokens: decode owed after a chunk goes on
    // before the next chunk.
    failures +=
        check(scheduler.choose_execution(true, true, true, true) == ExecutionAction::Decode &&
                  scheduler.choose_execution(true, true, true, false) == ExecutionAction::Prefill &&
                  scheduler.fair_decode_credit(2048, 3, 2) == 3072 &&
                  scheduler.fair_decode_credit(2048, 2, 0) == 0 &&
                  scheduler.decode_round_prefill_tokens(1) == 32 &&
                  scheduler.decode_round_prefill_tokens(8) == 32 + 7 * 8,
              "fair decode credit beside prefill changed");
    {
        // The interleaving is a pure function of the counts: a 2048-token chunk of one prefilling
        // request owes one decoding lane 64 rounds, eight lanes 187 (the last one partial).
        const auto rounds_owed = [&](std::uint64_t chunk, std::uint32_t lanes) {
            std::uint64_t credit = scheduler.fair_decode_credit(chunk, lanes, 1);
            std::uint32_t rounds = 0;
            while (credit > 0) {
                credit -= std::min(credit, scheduler.decode_round_prefill_tokens(lanes));
                ++rounds;
            }
            return rounds;
        };
        failures += check(rounds_owed(2048, 1) == 64 && rounds_owed(2048, 8) == 187 &&
                              rounds_owed(0, 4) == 0,
                          "decode rounds owed per prefill chunk are not the declared count");
    }
    scheduler.clear_prefill_lane(1);
    scheduler.clear_prefill_lane(3);
    scheduler.allow_concurrent_prefill(false);
    scheduler.reset();

    std::array<std::shared_ptr<SchedulerRequest>, ninferno::kMaximumConcurrency> slots{};
    slots[0]                      = std::make_shared<SchedulerRequest>();
    slots[0]->id                  = 21;
    slots[0]->budget              = SchedulerRequest::Budget{.tokens = 11};
    slots[0]->sequence            = 23;
    slots[0]->output.model_tokens = 3;
    const auto model_membership   = scheduler.build_round_membership(slots, 1);
    failures += check(model_membership.size == 1 &&
                          model_membership.budgets[0].generated_tokens_remaining == 3,
                      "model-token licensing changed");
    // A request waiting on KV lease pages sits out the round and comes back once it may grow.
    slots[0]->lease_parked = true;
    failures += check(scheduler.build_round_membership(slots, 1).size == 0,
                      "a parked KV lease still joined the decode round");
    slots[0]->lease_parked = false;

    const auto active_set = scheduler.active_admission_set(slots, 1);
    failures += check(active_set.size == 1 && active_set.requests[0].request_id == 21,
                      "active logical admission snapshot lost request identity");

    slots[0]->state          = SchedulerRequest::State::Control;
    slots[0]->output.control = {7, 8, 9};
    const auto control       = scheduler.build_control_membership(slots, 1);
    failures += check(control.size == 1 && control.row_stride == 3 &&
                          control.tokens == slots[0]->output.control && control.sequences[0] == 23,
                      "ControlReady membership changed the exact control span");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
