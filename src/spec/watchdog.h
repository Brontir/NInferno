#pragma once

// Runtime acceptance watchdog. Greedy verification keeps speculative output lossless whatever the
// drafter proposes, so a wrong drafter (another tokenizer, a mismatched checkpoint, a workload it
// was never trained for) shows up only as speed: rounds that emit little more than one token cost
// a verify pass plus the draft phase each. The watchdog follows the rolling tokens emitted per
// round over the last `window_rounds` rounds and trips when it falls below a floor; the owner
// then speculates shallower (restart() with that depth's floor) or stops speculating for the rest
// of the request (latched: a tripped watchdog stays tripped until reset(), restart() or rearm()).
//
// The floor is the break-even point measured by spec-tune: the tokens per round at which a round
// costs exactly as much as plain decode of the tokens it emits (round time over plain-decode
// time). The verdict
// is final for the request while acceptance often recovers after a slow opening (a thinking
// preamble, a list header), so the watchdog trips only below `trip_margin` of the floor's drafted
// tokens: 1 + trip_margin * (floor - 1) tokens per round. Measured on gpt-oss, Nemotron and the
// Qwen families (5090, 4090, 3090, V100): a trip at the break-even itself stopped requests whose
// acceptance came back, 8..50 % slower than speculating on.
//
// A trip judges a minimum (the window that fell below the trip point), so an owner that sizes the
// next depth reads depth_tokens_per_round(), the rate over every round at the depth just judged.

#include <cstdint>
#include <deque>
#include <utility>

namespace ninferno::spec {

struct WatchdogPolicy {
    double floor_tokens_per_round = 0.0; // <= 1: disabled (a round always emits one token)
    std::int64_t window_rounds    = 32;  // rolling window
    std::int64_t min_rounds       = 16;  // no verdict before this many rounds
    double trip_margin            = 0.5; // trips below this share of the floor's accepted drafts

    // The rolling tokens per round below which the watchdog trips.
    [[nodiscard]] double trip_tokens_per_round() const noexcept {
        return 1.0 + trip_margin * (floor_tokens_per_round - 1.0);
    }

    [[nodiscard]] bool enabled() const noexcept { return floor_tokens_per_round > 1.0; }
};

class AcceptanceWatchdog {
public:
    AcceptanceWatchdog() { reset(); }
    explicit AcceptanceWatchdog(WatchdogPolicy policy) : policy_(policy) { reset(); }

    // A new request: forget the history and the verdict.
    void reset() { restart(policy_.floor_tokens_per_round, 0, 0); }

    // Another depth mid-request: judged against `floor` on the rounds after the request's counters
    // now (`rounds`, `accepted`), with a fresh min_rounds before any verdict.
    void restart(double floor, std::int64_t rounds, std::int64_t accepted) {
        policy_.floor_tokens_per_round = floor;
        history_.assign(1, {rounds, accepted});
        tripped_  = false;
        rolling_  = 0.0;
        observed_ = rounds;
        start_    = rounds;
        start_accepted_ = accepted;
        last_accepted_  = accepted;
    }

    // The request's cumulative counters after some rounds: rounds run and drafts accepted since
    // the request began. Returns true on the call that trips the watchdog.
    bool observe(std::int64_t rounds, std::int64_t accepted) {
        if (!policy_.enabled() || tripped_ || rounds <= history_.back().first) { return false; }
        history_.emplace_back(rounds, accepted);
        // Keep as the baseline the latest checkpoint at least a window before the newest one.
        while (history_.size() > 2 && rounds - history_[1].first >= policy_.window_rounds) {
            history_.pop_front();
        }
        const auto [r0, a0] = history_.front();
        const auto span     = rounds - r0;
        observed_           = rounds;
        last_accepted_      = accepted;
        rolling_            = double(span + (accepted - a0)) / double(span);
        if (rounds - start_ >= policy_.min_rounds && rolling_ < policy_.trip_tokens_per_round()) {
            tripped_ = true;
            return true;
        }
        return false;
    }

    // Undoes the verdict of the call that just tripped (the owner kept the request speculating):
    // the history stays, and the next observe() judges again.
    void rearm() noexcept { tripped_ = false; }

    [[nodiscard]] bool tripped() const noexcept { return tripped_; }
    [[nodiscard]] double rolling_tokens_per_round() const noexcept { return rolling_; }
    // Tokens per round over every round observed since the last reset() or restart(); 0 before any.
    [[nodiscard]] double depth_tokens_per_round() const noexcept {
        const auto span = observed_ - start_;
        return span > 0 ? double(span + (last_accepted_ - start_accepted_)) / double(span) : 0.0;
    }
    [[nodiscard]] std::int64_t rounds_observed() const noexcept { return observed_; }
    [[nodiscard]] const WatchdogPolicy& policy() const noexcept { return policy_; }

private:
    WatchdogPolicy policy_;
    std::deque<std::pair<std::int64_t, std::int64_t>> history_;
    bool tripped_          = false;
    double rolling_        = 0.0;
    std::int64_t observed_ = 0;
    std::int64_t start_    = 0; // the round the current verdict counts min_rounds from
    std::int64_t start_accepted_ = 0; // drafts accepted by then
    std::int64_t last_accepted_  = 0; // drafts accepted at the last observe()
};

} // namespace ninferno::spec
