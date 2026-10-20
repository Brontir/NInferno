#pragma once

// Who may talk to ninferno-ui. The UI has no password: it starts processes, downloads models and
// spends the GPUs, so by default it answers only this machine.
//
//   listen    127.0.0.1 unless --listen names another address (the log then warns that anyone who
//             reaches it controls the GPUs);
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

} // namespace ninferno::ui
