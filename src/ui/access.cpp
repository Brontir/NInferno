#include "ui/access.h"
#include "core/ascii.h"
#include "core/sys.h"

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

namespace {

bool same_secret(std::string_view a, std::string_view b) noexcept {
    unsigned char diff = a.size() != b.size() ? 1 : 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        diff |= static_cast<unsigned char>(a[i] ^ b[i]);
    }
    return diff == 0;
}

} // namespace

std::string cookie_value(std::string_view cookie_header, std::string_view name) {
    while (!cookie_header.empty()) {
        const auto end        = cookie_header.find(';');
        std::string_view pair = cookie_header.substr(0, end);
        cookie_header = end == std::string_view::npos ? std::string_view() : cookie_header.substr(end + 1);
        while (!pair.empty() && pair.front() == ' ') { pair.remove_prefix(1); }
        if (pair.size() > name.size() && pair.substr(0, name.size()) == name &&
            pair[name.size()] == '=') {
            return std::string(pair.substr(name.size() + 1));
        }
    }
    return {};
}

KeyCheck check_key(const AccessPolicy& policy, std::string_view remote_addr,
                   std::string_view host_header, bool proxied, std::string_view cookie_header,
                   std::string_view authorization, std::string_view query_key) {
    if (policy.key.empty()) { return KeyCheck::pass; }
    if (remote_addr.substr(0, 7) == "::ffff:") { remote_addr.remove_prefix(7); }
    // A page whose name was rebound to 127.0.0.1 keeps its own Host; a proxy here marks itself.
    if (!proxied && !remote_addr.empty() && is_loopback_host(remote_addr) &&
        is_loopback_host(host_of_authority(host_header))) {
        return KeyCheck::pass;
    }
    if (same_secret(cookie_value(cookie_header, kKeyCookie), policy.key)) { return KeyCheck::pass; }
    constexpr std::string_view bearer = "bearer "; // the scheme's name is case-insensitive
    if (ascii_lower(authorization.substr(0, bearer.size())) == bearer) {
        auto credential = authorization.substr(bearer.size());
        while (!credential.empty() && credential.front() == ' ') { credential.remove_prefix(1); }
        if (same_secret(credential, policy.key)) { return KeyCheck::pass; }
    }
    if (same_secret(query_key, policy.key)) { return KeyCheck::pass_set_cookie; }
    return KeyCheck::refused;
}

bool usable_key(std::string_view key) noexcept {
    if (key.size() < 16) { return false; }
    for (const char c : key) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                        c == '.' || c == '_' || c == '~' || c == '-';
        if (!ok) { return false; }
    }
    return true;
}

bool forwarded_https(std::string_view x_forwarded_proto, std::string_view forwarded) {
    if (ascii_lower(x_forwarded_proto.substr(0, x_forwarded_proto.find(','))) == "https") {
        return true;
    }
    return ascii_lower(forwarded).find("proto=https") != std::string::npos;
}

std::string new_access_key() {
    unsigned char bytes[16];
    if (!sys::random_bytes(bytes, sizeof(bytes))) { return {}; }
    constexpr char digits[] = "0123456789abcdef";
    std::string key;
    for (const unsigned char b : bytes) {
        key += digits[b >> 4];
        key += digits[b & 15];
    }
    return key;
}

} // namespace ninferno::ui
