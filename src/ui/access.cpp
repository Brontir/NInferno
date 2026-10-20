#include "ui/access.h"
#include "core/ascii.h"

namespace ninferno::ui {

bool is_loopback_host(std::string_view host) noexcept {
    if (host.size() >= 2 && host.front() == '[' && host.back() == ']') {
        host = host.substr(1, host.size() - 2);
    }
    if (host == "localhost" || host == "::1") { return true; }
    // 127.x.y.z: four dot-separated decimal parts, the first 127.
    if (host.substr(0, 4) != "127.") { return false; }
    int parts = 1, digits = 0;
    for (const char c : host.substr(4)) {
        if (c == '.') {
            if (digits == 0) { return false; }
            ++parts;
            digits = 0;
        } else if (c >= '0' && c <= '9') {
            if (++digits > 3) { return false; }
        } else {
            return false;
        }
    }
    return parts == 3 && digits > 0;
}

std::string host_of_authority(std::string_view authority) {
    // Strip any userinfo (never sent by browsers, but cheap to ignore).
    if (const auto at = authority.rfind('@'); at != std::string_view::npos) {
        authority = authority.substr(at + 1);
    }
    if (!authority.empty() && authority.front() == '[') {
        const auto close = authority.find(']');
        if (close == std::string_view::npos) { return ascii_lower(authority); }
        return ascii_lower(authority.substr(1, close - 1));
    }
    if (const auto colon = authority.find(':'); colon != std::string_view::npos) {
        authority = authority.substr(0, colon);
    }
    return ascii_lower(authority);
}

bool host_allowed(const AccessPolicy& policy, std::string_view host_header) {
    if (host_header.empty()) { return false; }
    const std::string host = host_of_authority(host_header);
    if (is_loopback_host(host)) { return true; }
    // A listener the user opened to the network on purpose answers any name it is reached by;
    // the loopback default answers only loopback names.
    return !is_loopback_host(host_of_authority(policy.listen));
}

bool origin_allowed(std::string_view origin, std::string_view host_header) {
    if (origin.empty()) { return true; }
    if (origin == "null") { return false; }
    constexpr std::string_view http = "http://", https = "https://";
    std::string_view authority;
    if (origin.substr(0, http.size()) == http) {
        authority = origin.substr(http.size());
    } else if (origin.substr(0, https.size()) == https) {
        authority = origin.substr(https.size());
    } else {
        return false;
    }
    if (const auto slash = authority.find('/'); slash != std::string_view::npos) {
        authority = authority.substr(0, slash);
    }
    return ascii_lower(authority) == ascii_lower(host_header);
}

} // namespace ninferno::ui
