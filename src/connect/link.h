#pragma once

#include "core/loopback.h"

// The worker's side of a reversed connection to a router (#339, docs/connect.md): the worker dials
// the router, asks it to switch the connection to HTTP/2 with the roles reversed, and from then on
// the router sends requests over it as an HTTP/2 client and the worker answers them as the server.
//
// RouterUrl: where the router listens for its workers, https://host[:port][/path] (TLS, the
// router's certificate checked against the system's or a given CA and the host's name) or
// http://... (plain TCP: only to a loopback address unless allow_plaintext says otherwise).
// Link: the dialed byte stream, non-blocking once the handshake is done. handshake(): the HTTP/1.1
// Upgrade request carrying the worker's key and its manifest (the engine's /props), and the
// router's answer: 101 switches, 401 and 403 refuse the key (the worker stops dialing that router),
// anything else is refused this time.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace ninferno::connect {

// The protocol token of the Upgrade, and the default path the router takes workers on (never /v1).
inline constexpr std::string_view kUpgradeToken = "ninfer-reverse-h2";
inline constexpr std::string_view kDefaultConnectPath = "/ninfer/connect";

struct RouterUrl {
    bool tls = true;
    std::string host;
    int port = 443;
    std::string path{kDefaultConnectPath};

    [[nodiscard]] std::string display() const;
};

// nullopt with *why on a URL that is not https:// or http://, has no host, or a bad port.
[[nodiscard]] std::optional<RouterUrl> parse_router_url(std::string_view url, std::string* why);
// localhost, 127.0.0.0/8 and ::1: where plain TCP stays on the machine (core/loopback.h).
using ninferno::is_loopback_host;

struct TlsOptions {
    std::string ca_file; // empty: the system's default trust store
};

class Link {
public:
    virtual ~Link() = default;
    [[nodiscard]] virtual int fd() const noexcept = 0;
    // Bytes read (> 0), 0 when nothing is available now, -1 when the link is closed or failed.
    [[nodiscard]] virtual long read(std::uint8_t* data, std::size_t size) = 0;
    // Bytes written (> 0), 0 when the link cannot take any now, -1 when it is closed or failed.
    [[nodiscard]] virtual long write(const std::uint8_t* data, std::size_t size) = 0;
    // Whether the last 0 from read() or write() waits for the socket to become writable (TLS can
    // need to write to read): the loop then polls for POLLOUT too.
    [[nodiscard]] virtual bool wants_writable() const noexcept = 0;
    // Whether a write that took nothing waits for the socket to become readable (TLS: SSL_write
    // wanting to read): the loop then does not poll for POLLOUT for it, which would return at
    // once while the socket is writable.
    [[nodiscard]] virtual bool write_waits_readable() const noexcept = 0;
    // Switches the socket to non-blocking (after the blocking handshake).
    virtual void set_nonblocking() = 0;
    // Bytes the link holds already read from the socket (TLS: a record's rest): read them without
    // waiting for the socket.
    [[nodiscard]] virtual bool pending() const noexcept = 0;
};

// A TCP connection (and the TLS handshake for https) within timeout_ms; nullptr with *why.
[[nodiscard]] std::unique_ptr<Link> dial(const RouterUrl& url, const TlsOptions& tls,
                                         int timeout_ms, std::string* why);
// A link over a connected socket the caller owns no more (plain TCP): for tests.
[[nodiscard]] std::unique_ptr<Link> plain_link(int fd);

enum class HandshakeResult : std::uint8_t { Switched, KeyRefused, Refused, Failed };

struct Handshake {
    HandshakeResult result = HandshakeResult::Failed;
    int status = 0;
    std::string why;
    // Bytes the router sent after its 101 answer: the start of its HTTP/2 connection preface.
    std::string leftover;
};

// The Upgrade request and the router's answer, blocking (the link is still blocking); the body
// is the registration (JSON), the key goes as `Authorization: Bearer <key>`.
[[nodiscard]] Handshake handshake(Link& link, const RouterUrl& url, std::string_view key,
                                  std::string_view registration_json);

} // namespace ninferno::connect
