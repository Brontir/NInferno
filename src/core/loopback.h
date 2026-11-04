#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace ninferno {

namespace loopback_detail {

// A dotted quad's four parts (one to three decimal digits, at most 255 each), else false.
constexpr bool dotted_quad(std::string_view s, std::array<int, 4>& out) noexcept {
    int parts = 0, digits = 0, value = 0;
    for (const char c : s) {
        if (c == '.') {
            if (digits == 0 || parts == 3) { return false; }
            out[parts++] = value;
            digits = 0;
            value  = 0;
        } else if (c >= '0' && c <= '9') {
            value = value * 10 + (c - '0');
            if (++digits > 3 || value > 255) { return false; }
        } else {
            return false;
        }
    }
    if (parts != 3 || digits == 0) { return false; }
    out[3] = value;
    return true;
}

constexpr int hex_digit(char c) noexcept {
    if (c >= '0' && c <= '9') { return c - '0'; }
    if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
    if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
    return -1;
}

// An IPv6 address's eight 16-bit groups: hex groups of one to four digits, at most one "::", and
// an optional dotted quad as the last two groups. False for anything else (a zone id included).
constexpr bool ipv6_groups(std::string_view s, std::array<std::uint32_t, 8>& out) noexcept {
    std::array<std::uint32_t, 8> head{}, tail{};
    int heads = 0, tails = 0;
    bool gap  = false;
    if (s.substr(0, 2) == "::") {
        gap = true;
        s   = s.substr(2);
    } else if (!s.empty() && s.front() == ':') {
        return false;
    }
    while (!s.empty()) {
        auto& groups = gap ? tail : head;
        int& count   = gap ? tails : heads;
        const auto colon = s.find(':');
        const auto part  = s.substr(0, colon);
        if (part.find('.') != std::string_view::npos) {
            std::array<int, 4> quad{};
            if (colon != std::string_view::npos || !dotted_quad(part, quad) || count > 6) {
                return false;
            }
            groups[count++] = static_cast<std::uint32_t>(quad[0] << 8 | quad[1]);
            groups[count++] = static_cast<std::uint32_t>(quad[2] << 8 | quad[3]);
            s = {};
            break;
        }
        if (part.empty() || part.size() > 4 || count == 8) { return false; }
        std::uint32_t value = 0;
        for (const char c : part) {
            const int d = hex_digit(c);
            if (d < 0) { return false; }
            value = value * 16 + static_cast<std::uint32_t>(d);
        }
        groups[count++] = value;
        if (colon == std::string_view::npos) {
            s = {};
        } else if (s.substr(colon, 2) == "::") {
            if (gap) { return false; }
            gap = true;
            s   = s.substr(colon + 2);
        } else {
            s = s.substr(colon + 1);
            if (s.empty()) { return false; }
        }
    }
    if (gap ? heads + tails > 7 : heads != 8) { return false; }
    out = {};
    for (int i = 0; i < heads; ++i) { out[i] = head[i]; }
    for (int i = 0; i < tails; ++i) { out[8 - tails + i] = tail[i]; }
    return true;
}

} // namespace loopback_detail

// Whether a host name or address stays on this machine: "localhost" (any case), 127.0.0.0/8 as a
// dotted quad, and IPv6 ::1 or an IPv4-mapped 127.0.0.0/8 in any textual form (::1,
// 0:0:0:0:0:0:0:1, ::ffff:127.0.0.1; with or without brackets). Anything else, 0.0.0.0 and ::
// included, is reachable from the network. The one test the server, the local page and the
// connector share.
[[nodiscard]] constexpr bool is_loopback_host(std::string_view host) noexcept {
    if (host.size() >= 2 && host.front() == '[' && host.back() == ']') {
        host = host.substr(1, host.size() - 2);
    }
    if (host.size() == 9) {
        constexpr std::string_view kLocalhost = "localhost";
        bool same = true;
        for (std::size_t i = 0; i < host.size(); ++i) {
            const char c = host[i] >= 'A' && host[i] <= 'Z' ? static_cast<char>(host[i] - 'A' + 'a')
                                                             : host[i];
            same = same && c == kLocalhost[i];
        }
        if (same) { return true; }
    }
    // 127.x.y.z (127.999.0.1 is a name some resolver may send anywhere, not an address).
    if (std::array<int, 4> quad{}; loopback_detail::dotted_quad(host, quad)) {
        return quad[0] == 127;
    }
    std::array<std::uint32_t, 8> g{};
    if (host.find(':') == std::string_view::npos || !loopback_detail::ipv6_groups(host, g)) {
        return false;
    }
    bool zero5 = true;
    for (int i = 0; i < 5; ++i) { zero5 = zero5 && g[i] == 0; }
    if (!zero5) { return false; }
    if (g[5] == 0 && g[6] == 0 && g[7] == 1) { return true; }  // ::1
    return g[5] == 0xffff && (g[6] >> 8) == 127;                // ::ffff:127.x.y.z
}

} // namespace ninferno
