#pragma once

// Who may talk to ninferno-ui. It starts processes, downloads models and spends the GPUs, so by
// default it answers only this machine.
//
//   listen    127.0.0.1 unless --listen names another address;
//   key       on another address, a request from another machine carries the access key: the
//             ninferno_ui_key cookie, "Authorization: Bearer KEY", or ?key=KEY (which sets the
//             cookie); requests from this machine need none;
//   Host      on a loopback listener, only localhost, 127.0.0.1 or [::1] (any port): a page on
//             another site whose name resolves to 127.0.0.1 (DNS rebinding) is refused;
//   Origin    a request that changes something (POST) must come from the UI's own page: an Origin
//             header, when the browser sends one, names the same host and port the request went
//             to, so another site open in the same browser cannot start or stop anything
//             (cross-site request forgery). A request without Origin (curl, scripts) is allowed.

#include <string>
#include <string_view>

namespace ninferno::ui {

struct AccessPolicy {
    std::string listen = "127.0.0.1"; // the address the server binds
    int port           = 8090;
    std::string key;                  // the access key; empty: none asked
};

// 127.0.0.0/8, ::1 and "localhost" (with or without brackets).
[[nodiscard]] bool is_loopback_host(std::string_view host) noexcept;

// The host part of a Host header or of an origin's authority: "[::1]:8090" -> "::1",
// "localhost:8090" -> "localhost", "example.com" -> "example.com". Lower-cased.
[[nodiscard]] std::string host_of_authority(std::string_view authority);

// Whether a request with this Host header is served at all.
[[nodiscard]] bool host_allowed(const AccessPolicy& policy, std::string_view host_header);

// Whether a state-changing request with this Origin header (empty when absent) is accepted.
[[nodiscard]] bool origin_allowed(std::string_view origin, std::string_view host_header);

inline constexpr std::string_view kKeyCookie = "ninferno_ui_key";

enum class KeyCheck { pass, pass_set_cookie, refused };

// Whether a request from `remote_addr` carries the policy's key in its Cookie header, its
// Authorization header or its ?key= value (pass_set_cookie: only there, so the reply sets the
// cookie). Compared in constant time.
[[nodiscard]] KeyCheck check_key(const AccessPolicy& policy, std::string_view remote_addr,
                                 std::string_view cookie_header, std::string_view authorization,
                                 std::string_view query_key);

// The value of cookie `name` in a Cookie header; empty when absent.
[[nodiscard]] std::string cookie_value(std::string_view cookie_header, std::string_view name);

// A new key: 32 hex digits from the operating system's random source; empty if it gives none.
[[nodiscard]] std::string new_access_key();

} // namespace ninferno::ui
