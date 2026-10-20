#pragma once

// A sequence's speculative decode up to a budget or a stop token (a session's speculate_until):
// the rounds in slices, prompt lookup's host search, and the acceptance watchdog that hands a
// sequence whose drafts stopped paying over to plain decode. The session supplies its rounds and
// its plain decode through Steps; the driver keeps the per-sequence state.

#include "models/speculation/draft_target.h"
#include "spec/watchdog.h"

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ninferno::models::speculation {

struct DriverOptions {
    std::int32_t draft_tokens = 4; // the deepest round
    std::int32_t max_context  = 0;
    // Prompt lookup instead of a drafter: each round drafts what followed the most recent
    // earlier occurrence of the context's last ngram_max .. ngram_min tokens.
    bool ngram             = false;
    std::int32_t ngram_min = 2;
    std::int32_t ngram_max = 4;
    // The watchdog: once the tokens emitted per round over the last watchdog_window rounds fall
    // below acceptance_floor (1 or less: off), the sequence continues with plain decode until
    // the next prefill, and `log` is told once.
    double acceptance_floor          = 0;
    std::int32_t watchdog_window     = 32;
    std::int32_t watchdog_min_rounds = 16;
    std::function<void(const std::string&)> log;

    // The driver's fields of a session's options (every session's options carry them by name).
    template <class SessionOptions>
    [[nodiscard]] static DriverOptions of(const SessionOptions& o) {
        DriverOptions out;
        out.draft_tokens        = o.draft_tokens;
        out.max_context         = o.max_context;
        out.ngram               = o.ngram;
        out.ngram_min           = o.ngram_min;
        out.ngram_max           = o.ngram_max;
        out.acceptance_floor    = o.acceptance_floor;
        out.watchdog_window     = o.watchdog_window;
        out.watchdog_min_rounds = o.watchdog_min_rounds;
        out.log                 = o.log;
        return out;
    }
};

// What the driver runs on the session. Every call advances the session's position past the
// tokens it returns (the pending token is never among them until consumed).
class Steps {
public:
    virtual ~Steps() = default;
    // `rounds` speculative rounds; returns the tokens they consumed, the pending one first.
    virtual std::vector<std::int32_t> speculate(std::int32_t rounds) = 0;
    // `count` plain decode steps; returns the tokens they consumed.
    virtual std::vector<std::int32_t> decode(std::int32_t count) = 0;
    // One prompt-lookup round over the host's drafts; sets `pending` to the new pending token.
    virtual std::vector<std::int32_t> verify_drafts(std::span<const std::int32_t> drafts,
                                                    std::int32_t& pending) = 0;
    [[nodiscard]] virtual std::int32_t pending_token() = 0; // reads the device
    [[nodiscard]] virtual std::int32_t position() const = 0;
    virtual void set_position(std::int32_t position) = 0;
    [[nodiscard]] virtual SpeculationStats stats() const = 0;
};

class SequenceDriver {
public:
    explicit SequenceDriver(DriverOptions options) : options_(std::move(options)) {}

    // A new sequence: its history, watchdog and counters start over.
    void reset() noexcept;

    // Tokens the session consumed outside the driver (prefill, plain decode): prompt lookup's
    // history.
    void record(std::span<const std::int32_t> tokens);

    // At most `max_tokens` tokens, stopping after the first token in `stops` wherever it lands in
    // a round. Returns them in order, the pending token first; the session's position ends past
    // them (rounds that emitted beyond leave state the next prefill rewrites).
    [[nodiscard]] std::vector<std::int32_t> run(std::int32_t max_tokens,
                                                std::span<const std::int32_t> stops, Steps& steps);

    [[nodiscard]] bool latched() const noexcept { return latched_; }
    [[nodiscard]] std::int64_t plain_steps() const noexcept { return plain_; }

private:
    void observe(Steps& steps);

    DriverOptions options_;
    std::vector<std::int32_t> history_; // prompt lookup only
    spec::AcceptanceWatchdog watchdog_;
    SpeculationStats base_{};
    bool fresh_          = true;
    bool latched_        = false;
    std::int64_t plain_  = 0;
    std::int64_t misses_ = 0; // prompt-lookup steps without a match
};

} // namespace ninferno::models::speculation
