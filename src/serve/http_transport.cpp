#include "serve/http_transport.h"

#include "serve/request_validation.h"
#include "text/json_depth.h"

#if defined(__linux__)
#    include <netinet/tcp.h>
#    include <sys/socket.h>
#endif

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninferno::serve {
namespace {

#if defined(__linux__)
constexpr int kKeepAliveIdleSeconds                = 10;
constexpr int kKeepAliveIntervalSeconds            = 3;
constexpr int kKeepAliveProbeCount                 = 3;
constexpr unsigned int kTcpUserTimeoutMilliseconds = 15000;

template <class T>
void set_socket_option(socket_t socket, int level, int option, const T& value) noexcept {
    (void)::setsockopt(socket, level, option, &value, sizeof(value));
}
#endif

} // namespace

RequestJson parse_json_body(const httplib::Request& request) {
    RequestJson body;
    try {
        body = RequestJson::parse(request.body);
    } catch (const std::exception&) { bad_request("request body is not valid JSON"); }
    if (text::json_nests_deeper_than(body, kMaximumRequestJsonDepth)) {
        bad_request("request JSON nests deeper than " + std::to_string(kMaximumRequestJsonDepth) +
                    " levels");
    }
    return body;
}

bool client_disconnected(const httplib::Request& request) { return request.is_connection_closed(); }

void prepare_sse_response(httplib::Response& response) {
    response.set_header("Cache-Control", "no-cache");
    response.set_header("X-Accel-Buffering", "no");
}

SseTransport::SseTransport(httplib::DataSink& sink, std::atomic<bool>& cancelled,
                           Clock::duration heartbeat_interval, Clock::time_point now,
                           std::string_view heartbeat)
    : sink_(sink), cancelled_(cancelled), heartbeat_interval_(heartbeat_interval), last_write_(now),
      heartbeat_(heartbeat) {
    if (heartbeat_interval_ <= Clock::duration::zero()) {
        throw std::invalid_argument("SSE heartbeat interval must be positive");
    }
}

bool SseTransport::mark_cancelled() noexcept {
    cancelled_.store(true, std::memory_order_release);
    return true;
}

void SseTransport::write(std::string_view item, Clock::time_point now) {
    if (cancelled_.load(std::memory_order_acquire) || !sink_.write(item.data(), item.size())) {
        mark_cancelled();
        throw ClientDisconnected();
    }
    last_write_ = now;
}

void SseTransport::write(const std::vector<std::string>& items, Clock::time_point now) {
    for (const std::string& item : items) { write(item, now); }
}

bool SseTransport::poll(Clock::time_point now) {
    if (cancelled_.load(std::memory_order_acquire)) { return true; }
    if (sink_.is_writable && !sink_.is_writable()) { return mark_cancelled(); }
    if (now - last_write_ < heartbeat_interval_) { return false; }
    if (!sink_.write(heartbeat_.data(), heartbeat_.size())) { return mark_cancelled(); }
    last_write_ = now;
    return false;
}

HttpWorkerPlan plan_http_workers(std::uint32_t max_concurrency,
                                 std::uint32_t max_pending_requests) noexcept {
    // Rejections in flight and idle keep-alive sockets; threads cost only their stacks.
    constexpr std::size_t kMinimumHeadroom = 32;
    const std::size_t engine_held =
        static_cast<std::size_t>(max_concurrency) + static_cast<std::size_t>(max_pending_requests);
    return HttpWorkerPlan{.threads    = engine_held + std::max(kMinimumHeadroom, engine_held),
                          .max_queued = 0};
}

bool ExecutorStallMonitor::update(const ninferno::RuntimeStats& stats,
                                  Clock::time_point now) noexcept {
    // Any counter that moves when the executor completes work; a sum only has to change.
    const std::uint64_t progress =
        stats.computed_prefill_tokens + stats.committed_decode_tokens + stats.decode_rounds +
        stats.active_captures_completed + stats.active_captures_aborted + stats.state_moves +
        stats.state_forks + stats.state_restores + stats.root_selections +
        stats.private_endpoint_selections + stats.private_turn_closure_selections +
        stats.private_response_replay_selections + stats.private_long_anchor_selections +
        stats.shared_stable_prefix_selections;
    const bool busy = stats.running_requests != 0 || stats.waiting_requests != 0 ||
                      stats.materializing_requests != 0;
    if (!seen_ || progress != progress_ || !busy) {
        seen_          = true;
        progress_      = progress;
        last_progress_ = now;
        stalled_       = false;
        return false;
    }
    stalled_ = threshold_.count() > 0 && now - last_progress_ >= threshold_;
    return stalled_;
}

void configure_http_server_socket(socket_t socket) noexcept {
    httplib::default_socket_options(socket);
#if defined(__linux__)
    const int enabled = 1;
    // httplib sets only SO_REUSEPORT. Linux allows binding over a port's
    // TIME_WAIT connections only when the old and new sockets both set
    // SO_REUSEADDR, or both set SO_REUSEPORT under one uid. Without this, a
    // server using SO_REUSEADDR cannot take the port for about a minute after
    // NInferno stops, and NInferno cannot bind for as long after such a server stops.
    set_socket_option(socket, SOL_SOCKET, SO_REUSEADDR, enabled);
    set_socket_option(socket, SOL_SOCKET, SO_KEEPALIVE, enabled);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPIDLE, kKeepAliveIdleSeconds);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPINTVL, kKeepAliveIntervalSeconds);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPCNT, kKeepAliveProbeCount);
    set_socket_option(socket, IPPROTO_TCP, TCP_USER_TIMEOUT, kTcpUserTimeoutMilliseconds);
#endif
}

void set_owned_json_content(httplib::Response& response, std::string body,
                            std::shared_ptr<RequestLifetime> lifetime) {
    response.set_content(std::move(body), "application/json");
    response.user_data.set("ninferno.request_lifetime", std::move(lifetime));
}

} // namespace ninferno::serve
