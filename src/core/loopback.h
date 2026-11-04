#pragma once

#include <string_view>

namespace ninferno {

// Whether a host name or address stays on this machine: "localhost" (any case), 127.0.0.0/8 as a
// dotted quad, and ::1 (with or without brackets). Anything else, 0.0.0.0 and :: included, is
// reachable from the network. The one test the server, the local page and the connector share.
[[nodiscard]] constexpr bool is_loopback_host(std::string_view host) noexcept {
    if (host.size() >= 2 && host.front() == '[' && host.back() == ']') {
        host = host.substr(1, host.size() - 2);
    }
    if (host == "::1") { return true; }
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
    // 127.x.y.z: four dot-separated decimal parts of one to three digits, the first 127.
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

} // namespace ninferno
