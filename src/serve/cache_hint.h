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
#include <string_view>
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

// The kept threads' deadlines: a grant sets (or moves) its session's, `expired` hands back each
// session whose deadline passed, once.
class KeptSessions {
public:
    using Clock = std::chrono::steady_clock;
    void keep(const KeptGrant& grant, Clock::time_point now);
    [[nodiscard]] std::vector<std::string> expired(Clock::time_point now);

private:
    std::mutex mutex_;
    std::map<std::string, Clock::time_point> until_;
};

} // namespace ninferno::serve
