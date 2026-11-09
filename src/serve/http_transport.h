#pragma once

#include "serve/generation_service.h"
#include "serve/request.h"
#include "serve/request_json.h"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
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

// The request body as JSON. Refused with a 400 when it is not JSON, nests deeper than
// kMaximumRequestJsonDepth, or holds more than limits.max_json_values values: both are counted
// while the document is built and stop the parse at the limit.
RequestJson parse_json_body(const httplib::Request& request, const RequestLimits& limits);
[[nodiscard]] bool client_disconnected(const httplib::Request& request);

// A worker thread that parsed a large body has built and freed a document in its own malloc arena;
// glibc keeps such fragmented memory resident, and every worker keeps its own share (about half of
// the peak, measured in #527), so the process grows by one share per worker that ever served a big
// body. This hands it back (malloc_trim walks every arena) if the calling thread parsed one since
// the last call.
void release_request_memory() noexcept;

// The pool's task queue: httplib's thread pool, running release_request_memory() after each
// connection a worker served, when its request, its parsed document and its connection are gone
// and the response has long been sent. A keep-alive connection is one such connection: it trims
// when it ends (the keep-alive timeout is seconds), and the thread's next big body reuses the
// memory.
//
// The bound (max_queued > 0) counts the connections in hand, served or waiting: one more than
// threads + max_queued is refused. httplib's own bound counts its job list alone, which a burst of
// accepts fills before idle workers have woken to take from it, so it refused connections while
// workers were free; the inner pool is therefore unbounded.
class WorkerQueue final : public httplib::TaskQueue {
public:
    explicit WorkerQueue(std::size_t threads, std::size_t max_queued)
        : pool_(threads, threads, 0), limit_(max_queued == 0 ? 0 : threads + max_queued) {}

    bool enqueue(std::function<void()> connection) override;

    void shutdown() override { pool_.shutdown(); }

    void on_idle() override { pool_.on_idle(); }

    // Connections enqueued and not yet finished (served or waiting for a worker).
    [[nodiscard]] std::size_t in_hand() const noexcept {
        return in_hand_.load(std::memory_order_relaxed);
    }

private:
    httplib::ThreadPool pool_;
    std::size_t limit_; // 0: unbounded
    std::atomic<std::size_t> in_hand_{0};
};

// The HTTP worker pool in front of the Engine. Admission belongs to the Engine alone: it runs
// max_concurrency requests, holds max_pending_requests more, and answers the rest with an honest
// 429/503. The pool therefore must never close an accepted connection without a response. It keeps
// a worker for every request the Engine may hold, headroom for requests the Engine rejects at once
// and for idle keep-alive connections (each holds a worker until its keep-alive timeout), and it
// queues some accepted sockets beyond that (Linux; elsewhere all of them). A socket past the queue
// is answered 503 by answer_overloaded_connection, never dropped, and never left waiting without
// bound behind workers that slow senders hold.
struct HttpWorkerPlan {
    std::size_t threads    = 0;
    std::size_t max_queued = 0; // accepted sockets that wait for a free worker; more are refused
};

// The window over which a request body must keep arriving at
// ServeOptions::request_body_min_kib_per_sec. The library bounds the request line and headers as a
// whole and each wait for body bytes, but not a sender that delivers a few bytes every few seconds;
// without this such a sender holds a worker for as long as it likes.
inline constexpr std::chrono::milliseconds kRequestBodyRateWindow{10000};

// What /health answers: the status code, the JSON body, and whether the connection may be kept.
struct HealthReply {
    int status = 200;
    std::string body;
    bool keeps_connection = false;
};

// The answer to a connection the worker pool refused because every worker is busy and its queue is
// full (the pool runs on the accepting thread and must not block): a `GET /health` is answered
// with `health()`, so a liveness probe is not failed by a pool that slow senders have filled, and
// anything else gets HTTP 503 with Retry-After, server_busy, and the connection closes. The
// request line is peeked at without waiting (the listening socket defers accept until bytes have
// arrived) and what has arrived is then dropped, so closing does not reset the connection under the
// answer of a small request. A connection with no bytes at all gets no answer: it is closed, as an
// idle pooled connection of a proxy is, and the proxy retries on another. /health here answers for
// the Engine (liveness), not for the pool: a saturated pool is not an unhealthy server. Linux only;
// elsewhere the queue stays unbounded and this is never called.
void answer_overloaded_connection(socket_t socket, const std::function<HealthReply()>& health);

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
