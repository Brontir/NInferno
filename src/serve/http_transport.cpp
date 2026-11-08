#include "serve/http_transport.h"

#include "serve/request_validation.h"
#include "text/json_depth.h"

#if defined(__GLIBC__)
#    include <malloc.h>
#endif
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

// The document builder RequestJson::parse uses, which also counts the values it builds and stops
// the parse (returns false from the SAX event) at the limit, before the next one is allocated. A
// callback parser would do the same, but nlohmann's takes quadratic time on a long array. The base
// class is nlohmann's internal one; a single pass costs no more than parse().
template <class InputAdapter>
class BoundedDomBuilder final
    : public nlohmann::detail::json_sax_dom_parser<RequestJson, InputAdapter> {
    using Base = nlohmann::detail::json_sax_dom_parser<RequestJson, InputAdapter>;

public:
    BoundedDomBuilder(RequestJson& root, std::size_t limit) : Base(root), limit_(limit) {}

    [[nodiscard]] bool exceeded() const noexcept { return exceeded_; }

    bool null() { return admit() && Base::null(); }

    bool boolean(bool value) { return admit() && Base::boolean(value); }

    bool number_integer(RequestJson::number_integer_t value) {
        return admit() && Base::number_integer(value);
    }

    bool number_unsigned(RequestJson::number_unsigned_t value) {
        return admit() && Base::number_unsigned(value);
    }

    bool number_float(RequestJson::number_float_t value, const std::string& text) {
        return admit() && Base::number_float(value, text);
    }

    bool string(std::string& value) { return admit() && Base::string(value); }

    bool start_object(std::size_t elements) { return admit() && Base::start_object(elements); }

    bool start_array(std::size_t elements) { return admit() && Base::start_array(elements); }

    bool key(std::string& name) { return admit() && Base::key(name); }

private:
    // One more value (an object key counts: it is a string of its own).
    bool admit() noexcept {
        exceeded_ = exceeded_ || ++values_ > limit_;
        return !exceeded_;
    }

    std::size_t limit_;
    std::size_t values_ = 0;
    bool exceeded_      = false;
};

} // namespace

RequestJson parse_json_body(const httplib::Request& request, const RequestLimits& limits) {
    RequestJson body;
    BoundedDomBuilder<decltype(nlohmann::detail::input_adapter(request.body))> builder(
        body, limits.max_json_values);
    bool parsed = false;
    try {
        parsed = RequestJson::sax_parse(request.body, &builder);
    } catch (const std::exception&) { bad_request("request body is not valid JSON"); }
    if (builder.exceeded()) {
        bad_request("request JSON holds more than " + std::to_string(limits.max_json_values) +
                        " values (--max-request-json-values)",
                    {}, "request_json_too_large");
    }
    if (!parsed) { bad_request("request body is not valid JSON"); }
    if (text::json_nests_deeper_than(body, kMaximumRequestJsonDepth)) {
        bad_request("request JSON nests deeper than " + std::to_string(kMaximumRequestJsonDepth) +
                    " levels");
    }
    return body;
}

void release_request_memory(const httplib::Request& request) noexcept {
    // Smaller bodies leave too little behind to be worth walking every arena.
    constexpr std::size_t kMinimumBodyBytes = 1u << 20;
#if defined(__GLIBC__)
    if (request.body.size() >= kMinimumBodyBytes) { (void)::malloc_trim(0); }
#else
    (void)request;
    (void)kMinimumBodyBytes;
#endif
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
        stats.shared_stable_prefix_selections + stats.private_truncated_endpoint_selections;
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
