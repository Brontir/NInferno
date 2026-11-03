#include "serve/cache_hint.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>

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
    while (!header.empty()) {
        const std::size_t semi = header.find(';');
        const std::string_view field = trim(header.substr(0, semi));
        header = semi == std::string_view::npos ? std::string_view() : header.substr(semi + 1);
        if (field.empty()) { continue; }
        const std::size_t eq = field.find('=');
        if (eq == std::string_view::npos) { return std::nullopt; }
        const std::string_view key   = trim(field.substr(0, eq));
        const std::string_view value = trim(field.substr(eq + 1));
        any = true;
        if (key == "session") {
            // An opaque hex name, as long as the cache's session keys may be (1..256 bytes).
            if (value.empty() || value.size() > 256 ||
                !std::all_of(value.begin(), value.end(),
                             [](char c) { return std::isxdigit(static_cast<unsigned char>(c)) != 0; })) {
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

std::string_view apply_cache_hint(const std::string* header, bool trusted, ContextCacheHints& hints) {
    if (header == nullptr || !trusted) { return "none"; }
    const auto hint = parse_cache_hint(*header);
    if (!hint) { return "none"; }
    switch (hint->retain) {
    case CacheHintRetain::Drop:
        hints.retention = CacheRetentionHint::Disposable;
        return "drop";
    case CacheHintRetain::Keep:   // granted as normal (cache_hint.h): no expiry, no budget yet
    case CacheHintRetain::Normal: // a session_key would raise Default to LiveSession: not set
        return "normal";
    }
    return "none";
}

} // namespace ninferno::serve
