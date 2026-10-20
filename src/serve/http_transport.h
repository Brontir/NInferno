#pragma once

#include "serve/generation_service.h"
#include "serve/request_json.h"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninferno::serve {

class ClientDisconnected final : public std::exception {
public:
    [[nodiscard]] const char* what() const noexcept override { return "client disconnected"; }
};

class ResponseRenderFailure final : public std::runtime_error {
public:
    explicit ResponseRenderFailure(const std::string& message) : std::runtime_error(message) {}
};

struct HttpGenerationStream {
    explicit HttpGenerationStream(PreparedRequest request) : prepared(std::move(request)) {}

    PreparedRequest prepared;
    std::atomic<bool> cancelled{false};
    std::atomic<bool> started{false};
};

class SseTransport final {
public:
    using Clock = std::chrono::steady_clock;

    static constexpr std::chrono::seconds kHeartbeatInterval{5};
    static constexpr std::string_view kHeartbeatComment = ": keep-alive\n\n";
    // Anthropic Messages keeps a quiet stream live with a protocol event, as Anthropic does:
    // clients that time out on events rather than bytes ignore SSE comments.
    static constexpr std::string_view kAnthropicPing = "event: ping\ndata: {\"type\":\"ping\"}\n\n";

    explicit SseTransport(httplib::DataSink& sink, std::atomic<bool>& cancelled,
                          Clock::duration heartbeat_interval = kHeartbeatInterval,
                          Clock::time_point now              = Clock::now(),
                          std::string_view heartbeat         = kHeartbeatComment);

    void write(std::string_view item, Clock::time_point now = Clock::now());
    void write(const std::vector<std::string>& items, Clock::time_point now = Clock::now());

    // Called by the Engine wait loop. Besides observing an already-closed socket, a quiet stream
    // periodically writes its heartbeat so TCP_USER_TIMEOUT has traffic with which to detect an
    // unacknowledged peer. A failed probe marks the request for cancellation.
    [[nodiscard]] bool poll(Clock::time_point now = Clock::now());

private:
    bool mark_cancelled() noexcept;

    httplib::DataSink& sink_;
    std::atomic<bool>& cancelled_;
    Clock::duration heartbeat_interval_;
    Clock::time_point last_write_;
    std::string_view heartbeat_;
};

template <class Render>
void render_and_write(SseTransport& transport, Render&& render) {
    try {
        auto payload = std::forward<Render>(render)();
        transport.write(payload);
    } catch (const ClientDisconnected&) { throw; } catch (const ResponseRenderFailure&) {
        throw;
    } catch (const std::exception& exception) { throw ResponseRenderFailure(exception.what()); }
}

RequestJson parse_json_body(const httplib::Request& request);
[[nodiscard]] bool client_disconnected(const httplib::Request& request);

// The HTTP worker pool in front of the Engine. Admission belongs to the Engine alone: it runs
// max_concurrency requests, holds max_pending_requests more, and answers the rest with an honest
// 429/503. The pool therefore must never close an accepted connection without a response. It keeps
// a worker for every request the Engine may hold, headroom for requests the Engine rejects at once
// and for idle keep-alive connections (each holds a worker until its keep-alive timeout), and it
// queues accepted sockets beyond that instead of dropping them.
struct HttpWorkerPlan {
    std::size_t threads    = 0;
    std::size_t max_queued = 0; // 0: unbounded; an accepted socket waits for a free worker
};

[[nodiscard]] HttpWorkerPlan plan_http_workers(std::uint32_t max_concurrency,
                                               std::uint32_t max_pending_requests) noexcept;

// Executor liveness for /health: the Engine is stalled when it holds work
// (running, waiting or materializing requests) and none of its progress counters (prefill and
// decode tokens, decode rounds, completed or aborted captures, state transfers) has moved for
// `threshold`. An idle Engine is never stalled; progress clears the state at once.
class ExecutorStallMonitor {
public:
    using Clock = std::chrono::steady_clock;

    explicit ExecutorStallMonitor(std::chrono::milliseconds threshold) noexcept
        : threshold_(threshold) {}

    // Feeds one observation; returns whether the Engine is stalled as of `now`.
    bool update(const ninferno::RuntimeStats& stats, Clock::time_point now) noexcept;
    [[nodiscard]] bool stalled() const noexcept { return stalled_; }

private:
    std::chrono::milliseconds threshold_;
    std::uint64_t progress_ = 0;
    bool seen_              = false;
    bool stalled_           = false;
    Clock::time_point last_progress_{};
};

void prepare_sse_response(httplib::Response& response);
void configure_http_server_socket(socket_t socket) noexcept;
void set_owned_json_content(httplib::Response& response, std::string body,
                            std::shared_ptr<RequestLifetime> lifetime);

} // namespace ninferno::serve
