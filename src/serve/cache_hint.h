#pragma once

// x-ninferno-cache-hint: a trusted router's word on a thread's fate (#338), e.g.
//
//   x-ninferno-cache-hint: session=<hex>; retain=drop; ttl-ms=120000; next=0.93; turns=12.4
//
// Read only under --trust-cache-hints (any client could otherwise pin its own state). A malformed
// hint is none, never an error: it must not be able to fail a request. Unknown fields are ignored.
// What was granted goes back in x-ninferno-hint-applied: keep, normal, drop or none.
//
// Granted today: drop (the request's cached state is Disposable) and normal (nothing changes).
// keep is granted as normal until the cache can expire a kept thread (ttl-ms) and cap the bytes
// held as kept; without both, a thread that went silent would keep the highest weight forever.

#include "ninferno/types.h"

#include <optional>
#include <string>
#include <string_view>

namespace ninferno::serve {

inline constexpr std::string_view kCacheHintHeader        = "x-ninferno-cache-hint";
inline constexpr std::string_view kCacheHintAppliedHeader = "x-ninferno-hint-applied";

enum class CacheHintRetain { Keep, Normal, Drop };

struct CacheHint {
    std::optional<std::string> session; // the thread's opaque name (hex)
    CacheHintRetain retain = CacheHintRetain::Normal;
    std::optional<double> ttl_ms, next, turns, gap_ms, gap_p90_ms; // read, not yet weighed
};

// nullopt: malformed (a known field with a bad value, a field without '=', an empty header).
[[nodiscard]] std::optional<CacheHint> parse_cache_hint(std::string_view header);

// Applies `header` (nullptr: the request carried none) to `hints` when `trusted`; returns what was
// granted, as x-ninferno-hint-applied says it.
[[nodiscard]] std::string_view apply_cache_hint(const std::string* header, bool trusted,
                                                ContextCacheHints& hints);

} // namespace ninferno::serve
