#include "serve/cache_hint.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <vector>

namespace ninferno::serve {
namespace {

std::string_view trim(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())) != 0) { s.remove_prefix(1); }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())) != 0) { s.remove_suffix(1); }
    return s;
}

// A finite, non-negative number written in full.
std::optional<double> number(std::string_view text) {
    double value     = 0;
    const char* end  = text.data() + text.size();
    const auto result = std::from_chars(text.data(), end, value);
    if (result.ec != std::errc() || result.ptr != end || !std::isfinite(value) || value < 0) {
        return std::nullopt;
    }
    return value;
}

} // namespace

std::optional<CacheHint> parse_cache_hint(std::string_view header) {
    CacheHint hint;
    bool any = false;
    std::vector<std::string_view> seen;
    while (!header.empty()) {
        const std::size_t semi = header.find(';');
        const std::string_view field = trim(header.substr(0, semi));
        header = semi == std::string_view::npos ? std::string_view() : header.substr(semi + 1);
        if (field.empty()) { continue; }
        const std::size_t eq = field.find('=');
        if (eq == std::string_view::npos) { return std::nullopt; }
        const std::string_view key   = trim(field.substr(0, eq));
        const std::string_view value = trim(field.substr(eq + 1));
        // A field without a name, or one named twice, makes the hint ambiguous: none.
        if (key.empty() || std::find(seen.begin(), seen.end(), key) != seen.end()) { return std::nullopt; }
        seen.push_back(key);
        any = true;
        if (key == "session") {
            // An opaque token (hex, a UUID, ...), as long as the cache's session keys may be
            // (1..256 bytes).
            if (value.empty() || value.size() > 256 ||
                !std::all_of(value.begin(), value.end(), [](char c) {
                    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '-' || c == '_' ||
                           c == '.' || c == ':';
                })) {
                return std::nullopt;
            }
            hint.session = std::string(value);
        } else if (key == "retain") {
            if (value == "keep") {
                hint.retain = CacheHintRetain::Keep;
            } else if (value == "normal") {
                hint.retain = CacheHintRetain::Normal;
            } else if (value == "drop") {
                hint.retain = CacheHintRetain::Drop;
            } else {
                return std::nullopt;
            }
        } else if (key == "ttl-ms" || key == "next" || key == "turns" || key == "gap-ms" ||
                   key == "gap-p90-ms") {
            const auto v = number(value);
            if (!v || (key == "next" && *v > 1)) { return std::nullopt; }
            (key == "ttl-ms"  ? hint.ttl_ms
             : key == "next"  ? hint.next
             : key == "turns" ? hint.turns
             : key == "gap-ms" ? hint.gap_ms
                               : hint.gap_p90_ms) = *v;
        }
        // Any other field: a later version's, ignored.
    }
    if (!any) { return std::nullopt; }
    return hint;
}

std::string_view apply_cache_hint(const std::string* header, bool trusted, std::uint32_t kept_tokens,
                                  ContextCacheHints& hints, std::optional<KeptGrant>* kept) {
    if (header == nullptr || !trusted) { return "none"; }
    const auto hint = parse_cache_hint(*header);
    if (!hint) { return "none"; }
    switch (hint->retain) {
    case CacheHintRetain::Drop:
        hints.retention = CacheRetentionHint::Disposable;
        return "drop";
    case CacheHintRetain::Keep: {
        // Only a request that names no session of its own (a stored Response already has the
        // live-session weight, which a keep's end must not lower), advances the session index (a
        // release finds the entry there) and is not disposable; under the router's key.
        const std::string key =
            hint->session ? std::string(kKeptSessionPrefix) + *hint->session : std::string();
        // An hour at most, so a deadline in milliseconds cannot overflow the clock.
        constexpr double kLongestTtlMs = 3.6e6;
        if (kept_tokens == 0 || key.empty() || key.size() > 256 || !hint->ttl_ms ||
            !(*hint->ttl_ms > 0) || hints.session_key || !hints.update_session_index ||
            hints.retention == CacheRetentionHint::Disposable) {
            return "normal";
        }
        hints.session_key = key;
        hints.retention   = CacheRetentionHint::Kept;
        if (kept != nullptr) {
            kept->emplace(KeptGrant{
                key, std::chrono::milliseconds(static_cast<std::int64_t>(
                         std::ceil(std::min(*hint->ttl_ms, kLongestTtlMs))))});
        }
        return "keep";
    }
    case CacheHintRetain::Normal: // a session_key would raise Default to LiveSession: not set
        return "normal";
    }
    return "none";
}

void KeptSessions::keep(const KeptGrant& grant, Clock::time_point now) {
    std::lock_guard lock(mutex_);
    Deadline& d = until_[grant.session_key];
    d.until     = now + grant.ttl;
    d.ttl       = grant.ttl;
}

std::shared_ptr<void> KeptSessions::begin_turn(const KeptGrant& grant, Clock::time_point now) {
    {
        std::lock_guard lock(mutex_);
        Deadline& d = until_[grant.session_key];
        d.until     = now + grant.ttl;
        d.ttl       = grant.ttl;
        ++d.turns;
    }
    return std::shared_ptr<void>(nullptr, [this, key = grant.session_key](void*) {
        end_turn(key, Clock::now());
    });
}

void KeptSessions::end_turn(const std::string& session_key, Clock::time_point now) {
    std::lock_guard lock(mutex_);
    const auto it = until_.find(session_key);
    if (it == until_.end()) { return; }
    if (it->second.turns > 0) { --it->second.turns; }
    it->second.until = now + it->second.ttl;
}

std::vector<KeptGrant> KeptSessions::expired(Clock::time_point now) {
    std::vector<KeptGrant> out;
    std::lock_guard lock(mutex_);
    for (auto it = until_.begin(); it != until_.end();) {
        if (it->second.until <= now && it->second.turns > 0) {
            it->second.until = now + it->second.ttl; // its thread's turn still runs
            ++it;
        } else if (it->second.until <= now) {
            out.push_back({it->first, it->second.ttl});
            it = until_.erase(it);
        } else {
            ++it;
        }
    }
    return out;
}

} // namespace ninferno::serve
