#pragma once

// x-ninferno-cache-hint: a trusted router's word on a thread's fate (#338), e.g.
//
//   x-ninferno-cache-hint: session=<hex>; retain=drop; ttl-ms=120000; next=0.93; turns=12.4
//
// Read only under --trust-cache-hints (any client could otherwise pin its own state). A malformed
// hint is none, never an error: it must not be able to fail a request. Unknown fields are ignored.
// What was granted goes back in x-ninferno-hint-applied: keep, normal, drop or none.
//
// drop: the request's cached state is Disposable. normal: nothing changes. keep (with a session and
// ttl-ms, under --cache-keep-tokens): the thread's cached state takes the live-session weight
// (CacheRetentionHint::Kept) while the kept threads together hold no more than the budget's
// context tokens (the oldest kept falls first), and until ttl-ms passes without a new turn, when
// the server releases it (Engine::release_kept_session). The serve side keeps the clock: the
// cache's decisions read none. Without a budget, a session or a ttl, keep is granted as normal.

#include "ninferno/types.h"

#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace ninferno::serve {

inline constexpr std::string_view kCacheHintHeader        = "x-ninferno-cache-hint";
inline constexpr std::string_view kCacheHintAppliedHeader = "x-ninferno-hint-applied";

enum class CacheHintRetain { Keep, Normal, Drop };

struct CacheHint {
    std::optional<std::string> session; // the thread's opaque name (letters, digits, - _ . :)
    CacheHintRetain retain = CacheHintRetain::Normal;
    std::optional<double> ttl_ms, next, turns, gap_ms, gap_p90_ms; // read, not yet weighed
};

// nullopt: malformed (a known field with a bad value, a field without '=' or without a name, a
// field named twice, an empty header).
[[nodiscard]] std::optional<CacheHint> parse_cache_hint(std::string_view header);

// A keep granted: the cache's session key it holds under, and how long without a new turn.
struct KeptGrant {
    std::string session_key;
    std::chrono::milliseconds ttl{0};
};

// The session key a router's thread is kept under when the request names none of its own.
inline constexpr std::string_view kKeptSessionPrefix = "router:";

// Applies `header` (nullptr: the request carried none) to `hints` when `trusted`; returns what was
// granted, as x-ninferno-hint-applied says it. A keep is granted only with `kept_tokens` (the
// --cache-keep-tokens budget) above 0, a session and a ttl-ms above 0; `kept` (when given) then
// receives the grant.
[[nodiscard]] std::string_view apply_cache_hint(const std::string* header, bool trusted,
                                                std::uint32_t kept_tokens, ContextCacheHints& hints,
                                                std::optional<KeptGrant>* kept = nullptr);
[[nodiscard]] inline std::string_view apply_cache_hint(const std::string* header, bool trusted,
                                                       ContextCacheHints& hints) {
    return apply_cache_hint(header, trusted, 0, hints);
}

// What a finished turn found, for the router (#338), under --trust-cache-hints: the server's
// instance (the request log's server_instance_id: a restart empties every cache), how its prompt
// was reused, the tokens reused and computed, its queue wait and prefill time.
struct RouterReport {
    std::string instance;
    std::string reuse_path;
    std::int64_t reused_prompt_tokens   = 0;
    std::int64_t computed_prompt_tokens = 0;
    double queue_wait_seconds           = 0;
    double prefill_seconds              = 0;
};
// A stream's last frame, an SSE comment the clients ignore and the router parses:
//   : ninfer {"placement":{"instance":..,"reuse_path":..,"reused_prompt_tokens":..,
//             "computed_prompt_tokens":..,"queue_wait_seconds":..,"prefill_seconds":..}}
[[nodiscard]] std::string router_report_frame(const RouterReport& report);
// A rendered answer's headers: x-ninfer-instance, x-ninfer-reuse-path, x-ninfer-reused-tokens,
// x-ninfer-computed-tokens, x-ninfer-queue-wait-ms, x-ninfer-prefill-ms.
[[nodiscard]] std::vector<std::pair<std::string, std::string>>
router_report_headers(const RouterReport& report);

// The kept threads' deadlines: a grant sets (or moves) its session's, `expired` hands back each
// session whose deadline passed, once, with its ttl (a release the cache answers InFlight is kept
// again for another ttl). A thread's ttl runs from the end of its last turn: begin_turn counts a
// turn from its grant (before it is prepared, queued or materialized, when the cache does not see
// it yet) and its guard, dropped when the request ends, counts it off and moves the deadline to a
// ttl after; a deadline that passes while a turn of its thread runs moves on another ttl.
class KeptSessions {
public:
    using Clock = std::chrono::steady_clock;
    void keep(const KeptGrant& grant, Clock::time_point now);
    [[nodiscard]] std::shared_ptr<void> begin_turn(const KeptGrant& grant, Clock::time_point now);
    [[nodiscard]] std::vector<KeptGrant> expired(Clock::time_point now);

private:
    void end_turn(const std::string& session_key, Clock::time_point now);
    struct Deadline {
        Clock::time_point until;
        std::chrono::milliseconds ttl;
        int turns = 0;
    };
    std::mutex mutex_;
    std::map<std::string, Deadline> until_;
};

} // namespace ninferno::serve
