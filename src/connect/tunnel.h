#pragma once

// One reversed connection's HTTP/2 session (#339, docs/connect.md): on a link the worker dialed
// and the router switched, the worker is the HTTP/2 server. Each stream the router opens is one
// HTTP request, forwarded to the local engine as plain HTTP/1.1 on a connection of its own, and the
// engine's answer streams back as that stream's response:
// - many requests at once: a stream each, a forwarding thread each;
// - a slow reader stalls only its stream: HTTP/2 flow control, and a stream's buffer is bounded so
//   the engine's connection waits rather than the worker's memory growing;
// - a router's RST_STREAM closes that request's engine connection, which cancels the generation as
//   a client that went away does (also during prefill, before any byte of the answer);
// - liveness: a PING every ping interval; two left unanswered end the session;
// - the router's GOAWAY: no new streams come, the ones running finish, and on_goaway lets the
//   caller dial a replacement meanwhile;
// - a lost link fails the streams it carried, as an engine killed mid-answer does.

#include "connect/link.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>

namespace ninferno::connect {

struct EngineAddress {
    std::string host = "127.0.0.1";
    int port         = 8080;
};

struct TunnelOptions {
    EngineAddress engine;
    std::chrono::milliseconds ping_interval{15000};
    int max_concurrent_streams     = 64;
    std::size_t max_request_bytes  = 64u << 20; // a request body past this is answered 413
    std::size_t stream_buffer_bytes = 1u << 20;  // a stream's answer buffered ahead of the router
};

enum class TunnelEnd : std::uint8_t {
    Closed,       // the router closed the link (or it failed)
    PingTimeout,  // two PINGs went unanswered
    Drained,      // after the router's GOAWAY, every stream finished
    Stopped,      // the caller's stop: GOAWAY sent and the streams finished
    ProtocolError // the router broke HTTP/2
};
[[nodiscard]] const char* tunnel_end_name(TunnelEnd end) noexcept;

struct TunnelEvents {
    std::function<void()> on_goaway; // the router's GOAWAY arrived (the session drains)
};

// Runs the session to its end. `leftover`: bytes that came with the router's 101 answer. `stop`
// asks for a graceful end (GOAWAY, then the running streams finish).
[[nodiscard]] TunnelEnd serve_tunnel(Link& link, std::string leftover, const TunnelOptions& options,
                                     const TunnelEvents& events, const std::atomic<bool>& stop);

} // namespace ninferno::connect
