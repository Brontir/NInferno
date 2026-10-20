#include "models/speculation/driver.h"

#include "models/speculation/ngram_round.h"

#include <algorithm>
#include <cstdio>

namespace ninferno::models::speculation {

void SequenceDriver::reset() noexcept {
    history_.clear();
    fresh_   = true;
    latched_ = false;
    plain_   = 0;
    misses_  = 0;
}

void SequenceDriver::record(std::span<const std::int32_t> tokens) {
    if (options_.ngram) { history_.insert(history_.end(), tokens.begin(), tokens.end()); }
}

void SequenceDriver::observe(Steps& steps) {
    if (options_.acceptance_floor <= 1.0 || latched_) { return; }
    const auto now = steps.stats();
    // A prompt-lookup step without a match is a one-token round: it pays a host round trip per
    // token that plain decode does not.
    if (watchdog_.observe(now.rounds - base_.rounds + misses_, now.accepted - base_.accepted)) {
        latched_ = true;
        if (options_.log) {
            char line[256];
            std::snprintf(line, sizeof(line),
                          "speculation watchdog: %.3f tokens/round over the last rounds "
                          "(floor %.3f) after %lld rounds; plain decode for the rest of "
                          "this sequence",
                          watchdog_.rolling_tokens_per_round(), options_.acceptance_floor,
                          static_cast<long long>(watchdog_.rounds_observed()));
            options_.log(line);
        }
    }
}

std::vector<std::int32_t> SequenceDriver::run(std::int32_t max_tokens,
                                              std::span<const std::int32_t> stops, Steps& steps) {
    const auto k     = options_.draft_tokens;
    const auto start = steps.position();
    std::vector<std::int32_t> out;
    const auto finish = [&] {
        steps.set_position(start + static_cast<std::int32_t>(out.size()));
        return out;
    };
    // Appends consumed tokens; true when the call is complete (budget or stop id).
    const auto take = [&](const std::vector<std::int32_t>& part) {
        for (const auto token : part) {
            out.push_back(token);
            if (static_cast<std::int32_t>(out.size()) == max_tokens ||
                std::find(stops.begin(), stops.end(), token) != stops.end()) {
                return true;
            }
        }
        return false;
    };
    const bool watched = options_.acceptance_floor > 1.0;
    if (!watched && !options_.ngram) {
        while (static_cast<std::int32_t>(out.size()) < max_tokens) {
            // Enough rounds for the missing tokens if every draft were accepted, within the
            // context.
            const auto missing = max_tokens - static_cast<std::int32_t>(out.size());
            const auto room    = (options_.max_context - steps.position() - 1) / (k + 1);
            const auto rounds  = std::min((missing + k) / (k + 1), room);
            if (rounds <= 0) { break; }
            if (take(steps.speculate(rounds))) { break; }
        }
        return finish();
    }

    // Watched or prompt lookup: the sequence's counters since its first call feed the watchdog.
    if (fresh_) {
        spec::WatchdogPolicy policy;
        policy.floor_tokens_per_round = options_.acceptance_floor;
        policy.window_rounds          = options_.watchdog_window;
        policy.min_rounds             = options_.watchdog_min_rounds;
        watchdog_                     = spec::AcceptanceWatchdog(policy);
        base_                         = steps.stats();
        misses_                       = 0;
        fresh_                        = false;
    }
    constexpr std::int32_t kSlice = 8; // rounds between watchdog checks
    bool done                     = false;
    if (options_.ngram) {
        std::int32_t pending = steps.pending_token();
        while (!done && !latched_ && static_cast<std::int32_t>(out.size()) < max_tokens &&
               steps.position() + k + 1 < options_.max_context) {
            std::vector<std::int32_t> drafts;
            history_.push_back(pending);
            const auto limit = std::min(k, max_tokens - static_cast<std::int32_t>(out.size()) - 1);
            if (limit > 0) {
                drafts = ngram_lookup(history_, options_.ngram_min, options_.ngram_max, limit);
            }
            history_.pop_back();
            std::vector<std::int32_t> part;
            if (!drafts.empty()) {
                part = steps.verify_drafts(drafts, pending);
                history_.insert(history_.end(), part.begin(), part.end());
                observe(steps);
            } else {
                part = steps.decode(1); // the session records it into the history
                ++plain_;
                ++misses_;
                observe(steps);
                pending = steps.pending_token();
            }
            done = take(part);
        }
    }
    while (!options_.ngram && !done && !latched_ &&
           static_cast<std::int32_t>(out.size()) < max_tokens) {
        const auto missing = max_tokens - static_cast<std::int32_t>(out.size());
        const auto room    = (options_.max_context - steps.position() - 1) / (k + 1);
        const auto rounds  = std::min({(missing + k) / (k + 1), room, kSlice});
        if (rounds <= 0) { break; }
        done = take(steps.speculate(rounds));
        observe(steps);
    }
    // After the watchdog tripped (or prompt lookup at the context's end): plain decode, a few
    // steps per call.
    while (!done && static_cast<std::int32_t>(out.size()) < max_tokens) {
        const auto missing = max_tokens - static_cast<std::int32_t>(out.size());
        const auto steps_n = std::min({missing, 16, options_.max_context - steps.position()});
        if (steps_n <= 0) { break; }
        plain_ += steps_n;
        done = take(steps.decode(steps_n));
    }
    return finish();
}

} // namespace ninferno::models::speculation
