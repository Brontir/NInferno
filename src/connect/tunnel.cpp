#include "connect/tunnel.h"

#include <httplib.h>
#include <nghttp2/nghttp2.h>

#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace ninferno::connect {
namespace {

using Clock = std::chrono::steady_clock;

// The headers HTTP/2 does not carry (RFC 9113 8.2.2) and the ones the engine's hop sets itself.
bool hop_by_hop(std::string_view name) {
    static constexpr std::array<std::string_view, 8> kNames = {
        "connection", "keep-alive", "proxy-connection", "transfer-encoding",
        "upgrade",    "te",         "host",             "content-length"};
    return std::find(kNames.begin(), kNames.end(), name) != kNames.end();
}

std::string lower(std::string_view s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

// Wakes the session's poll from a forwarding thread.
class Waker {
public:
    Waker() {
        if (::pipe2(fds_, O_NONBLOCK | O_CLOEXEC) != 0) { fds_[0] = fds_[1] = -1; }
    }
    ~Waker() {
        for (const int fd : fds_) {
            if (fd >= 0) { ::close(fd); }
        }
    }
    Waker(const Waker&)            = delete;
    Waker& operator=(const Waker&) = delete;
    [[nodiscard]] int fd() const noexcept { return fds_[0]; }
    void wake() const noexcept {
        const char one = 1;
        (void)!::write(fds_[1], &one, 1);
    }
    void drain() const noexcept {
        char buffer[256];
        while (::read(fds_[0], buffer, sizeof buffer) > 0) {}
    }

private:
    int fds_[2];
};

// One request: what the router sent (owned by the session thread until the forwarding thread
// starts), and the engine's answer (shared, under mu).
struct Stream {
    std::int32_t id = 0;
    std::string method;
    std::string path;
    std::string authority;
    httplib::Headers headers;
    std::string body;
    bool too_large = false;

    std::mutex mu;
    std::condition_variable room; // the buffer drained, or the stream was cancelled
    bool head_ready = false;
    int status      = 0;
    std::vector<std::pair<std::string, std::string>> response_headers;
    std::string out;
    std::size_t out_off   = 0;
    bool upstream_done    = false;
    bool upstream_failed  = false; // the engine's answer broke off after its head
    bool cancelled        = false;
    std::atomic<bool> cancel_flag{false}; // `cancelled`, read where the mutex is not held
    std::shared_ptr<httplib::Client> client;

    // Session-thread state.
    bool head_submitted = false;
    bool deferred       = false;
    bool closed         = false;
    std::atomic<bool> worker_finished{false};
    std::thread worker;

    [[nodiscard]] std::size_t buffered() const noexcept { return out.size() - out_off; }
};

// An answer the worker gives itself (the engine unreachable, a body too large), as the engine's
// own error would come.
void local_answer(Stream& s, int status, std::string_view message) {
    std::lock_guard lock(s.mu);
    s.status = status;
    s.response_headers = {{"content-type", "application/json"}};
    s.out = R"({"error":{"message":")" + std::string(message) + R"(","type":"ninferno_connect"}})";
    s.out_off       = 0;
    s.head_ready    = true;
    s.upstream_done = true;
}

void forward(const std::shared_ptr<Stream>& s, const EngineAddress engine,
             const std::size_t buffer_cap, const std::shared_ptr<Waker>& waker) {
    auto client = std::make_shared<httplib::Client>(engine.host, engine.port);
    client->set_connection_timeout(std::chrono::seconds(3)); // the engine is on this machine
    client->set_read_timeout(std::chrono::hours(24)); // a prefill of a long prompt is not a stall
    client->set_write_timeout(std::chrono::minutes(5));
    client->set_decompress(false); // the engine's bytes as they are
    client->set_keep_alive(false);
    // A reset that comes before the engine's socket exists finds nothing to stop: the socket is
    // shut as it is made instead, so the request fails at once.
    client->set_socket_options([s](auto sock) {
        if (s->cancel_flag.load()) { ::shutdown(sock, SHUT_RDWR); }
    });
    {
        std::lock_guard lock(s->mu);
        if (s->cancelled) {
            s->upstream_done = true;
            s->worker_finished.store(true);
            waker->wake();
            return;
        }
        s->client = client;
    }
    httplib::Request request;
    request.method  = s->method;
    request.path    = s->path;
    request.headers = std::move(s->headers);
    request.body    = std::move(s->body);
    if (!s->authority.empty()) { request.set_header("X-Forwarded-Host", s->authority); }
    // The answer's head (under s->mu).
    // A bodyless answer's Content-Length (HEAD, 304) is the representation's size, an end-to-end
    // field: it passes. Elsewhere HTTP/2's DATA frames carry the length.
    const bool bodyless_length = s->method == "HEAD";
    const auto take_head = [&](const httplib::Response& response) {
        s->status = response.status;
        for (const auto& [name, value] : response.headers) {
            std::string key = lower(name);
            const bool length_kept =
                key == "content-length" && (bodyless_length || response.status == 304);
            if (!hop_by_hop(key) || length_kept) {
                s->response_headers.emplace_back(std::move(key), value);
            }
        }
        s->head_ready = true;
    };
    request.response_handler = [&](const httplib::Response& response) {
        std::lock_guard lock(s->mu);
        take_head(response);
        waker->wake();
        return !s->cancelled;
    };
    request.content_receiver = [&](const char* data, std::size_t size, std::uint64_t,
                                   std::uint64_t) {
        std::unique_lock lock(s->mu);
        s->room.wait(lock, [&] { return s->cancelled || s->buffered() < buffer_cap; });
        if (s->cancelled) { return false; }
        s->out.append(data, size);
        waker->wake();
        return true;
    };
    const httplib::Result result = client->send(request);
    bool answered = false;
    {
        std::lock_guard lock(s->mu);
        s->client = nullptr;
        // An answer with no body (204, a HEAD's, an OPTIONS preflight's) reaches no response
        // handler: its head comes with the result.
        if (!s->head_ready && !s->cancelled && result) { take_head(*result); }
        answered = s->head_ready || s->cancelled;
    }
    if (!answered) {
        local_answer(*s, 502,
                     "the engine did not answer: " + httplib::to_string(result.error()));
    } else {
        std::lock_guard lock(s->mu);
        // An answer cut off after its head (the engine killed, a read error) is no whole answer:
        // its stream is reset once the bytes it had are out, never ended as if complete.
        s->upstream_failed = !s->cancelled && result.error() != httplib::Error::Success;
        s->upstream_done   = true;
    }
    s->worker_finished.store(true);
    waker->wake();
}

class Session {
public:
    Session(Link& link, const TunnelOptions& options, const TunnelEvents& events)
        : link_(link), options_(options), events_(events), waker_(std::make_shared<Waker>()) {
        nghttp2_session_callbacks* callbacks = nullptr;
        nghttp2_session_callbacks_new(&callbacks);
        nghttp2_session_callbacks_set_on_begin_headers_callback(callbacks, &Session::on_begin_headers);
        nghttp2_session_callbacks_set_on_header_callback(callbacks, &Session::on_header);
        nghttp2_session_callbacks_set_on_data_chunk_recv_callback(callbacks, &Session::on_data);
        nghttp2_session_callbacks_set_on_frame_recv_callback(callbacks, &Session::on_frame);
        nghttp2_session_callbacks_set_on_stream_close_callback(callbacks, &Session::on_close);
        nghttp2_session_server_new(&session_, callbacks, this);
        nghttp2_session_callbacks_del(callbacks);
        const nghttp2_settings_entry settings[] = {
            {NGHTTP2_SETTINGS_MAX_CONCURRENT_STREAMS,
             static_cast<std::uint32_t>(std::max(1, options_.max_concurrent_streams))},
            {NGHTTP2_SETTINGS_ENABLE_PUSH, 0},
        };
        nghttp2_submit_settings(session_, NGHTTP2_FLAG_NONE, settings, 2);
    }
    ~Session() {
        for (auto& [id, s] : streams_) { cancel(*s); }
        for (auto& s : retiring_) { cancel(*s); }
        for (auto& [id, s] : streams_) { join(*s); }
        for (auto& s : retiring_) { join(*s); }
        nghttp2_session_del(session_);
    }
    Session(const Session&)            = delete;
    Session& operator=(const Session&) = delete;

    TunnelEnd run(std::string leftover, const std::atomic<bool>& stop) {
        link_.set_nonblocking();
        if (!leftover.empty() && !receive(reinterpret_cast<const std::uint8_t*>(leftover.data()),
                                          leftover.size())) {
            return TunnelEnd::ProtocolError;
        }
        auto next_ping = Clock::now() + options_.ping_interval;
        int unanswered = 0;
        bool stopping  = false;
        std::array<std::uint8_t, 16384> buffer{};
        for (;;) {
            if (protocol_error_) { return TunnelEnd::ProtocolError; }
            pump_streams();
            if (!send()) { return TunnelEnd::Closed; }
            if (stop.load() && !stopping) {
                stopping = true;
                nghttp2_submit_goaway(session_, NGHTTP2_FLAG_NONE,
                                      nghttp2_session_get_last_proc_stream_id(session_),
                                      NGHTTP2_NO_ERROR, nullptr, 0);
                continue;
            }
            if ((goaway_received_ || stopping) && streams_.empty() && out_.empty()) {
                return stopping ? TunnelEnd::Stopped : TunnelEnd::Drained;
            }
            if (!nghttp2_session_want_read(session_) && !nghttp2_session_want_write(session_) &&
                out_.empty()) {
                return goaway_received_ ? TunnelEnd::Drained : TunnelEnd::Closed;
            }
            const auto now = Clock::now();
            if (now >= next_ping) {
                if (ping_outstanding_) {
                    if (++unanswered >= 2) { return TunnelEnd::PingTimeout; }
                } else {
                    unanswered = 0;
                }
                nghttp2_submit_ping(session_, NGHTTP2_FLAG_NONE, nullptr);
                ping_outstanding_ = true;
                next_ping         = now + options_.ping_interval;
                continue;
            }
            const auto wait = std::chrono::duration_cast<std::chrono::milliseconds>(next_ping - now);
            pollfd fds[2] = {{link_.fd(), POLLIN, 0}, {waker_->fd(), POLLIN, 0}};
            const bool read_wants_write = link_.wants_writable();
            if ((!out_.empty() && !link_.write_waits_readable()) || read_wants_write) {
                fds[0].events |= POLLOUT;
            }
            // Bytes the link holds already are read without waiting for the socket.
            const bool held   = link_.pending();
            const int timeout = held ? 0 : static_cast<int>(std::clamp<long long>(wait.count(), 1, 250));
            if (::poll(fds, 2, timeout) < 0 && errno != EINTR) { return TunnelEnd::Closed; }
            if (fds[1].revents & POLLIN) { waker_->drain(); }
            if (held || (fds[0].revents & (POLLIN | POLLHUP | POLLERR)) ||
                (read_wants_write && (fds[0].revents & POLLOUT))) {
                for (;;) {
                    const long n = link_.read(buffer.data(), buffer.size());
                    if (n < 0) { return TunnelEnd::Closed; }
                    if (n == 0) { break; }
                    if (!receive(buffer.data(), static_cast<std::size_t>(n))) {
                        return TunnelEnd::ProtocolError;
                    }
                }
            }
            reap();
        }
    }

private:
    bool receive(const std::uint8_t* data, std::size_t size) {
        const nghttp2_ssize n = nghttp2_session_mem_recv2(session_, data, size);
        return n >= 0 && static_cast<std::size_t>(n) == size;
    }

    // nghttp2's output into out_, and out_ onto the link as far as it takes it now.
    bool send() {
        while (out_.size() < (1u << 20)) {
            const std::uint8_t* data = nullptr;
            const nghttp2_ssize n    = nghttp2_session_mem_send2(session_, &data);
            if (n < 0) {
                protocol_error_ = true;
                return true;
            }
            if (n == 0) { break; }
            out_.append(reinterpret_cast<const char*>(data), static_cast<std::size_t>(n));
        }
        while (!out_.empty()) {
            // A write that waited (TLS: WANT_WRITE) is retried with the same length: OpenSSL takes
            // a moved buffer but not other bytes; what was appended since goes in the next write.
            const std::size_t length = retry_length_ != 0 ? retry_length_ : out_.size();
            const long n = link_.write(reinterpret_cast<const std::uint8_t*>(out_.data()), length);
            if (n < 0) { return false; }
            if (n == 0) {
                retry_length_ = length;
                break;
            }
            retry_length_ = 0;
            out_.erase(0, static_cast<std::size_t>(n));
        }
        return true;
    }

    // Submits the answers whose heads arrived and resumes the streams whose bytes did.
    void pump_streams() {
        for (auto& [id, s] : streams_) {
            bool submit = false;
            bool resume = false;
            {
                std::lock_guard lock(s->mu);
                if (s->closed) { continue; }
                submit = s->head_ready && !s->head_submitted;
                resume = s->head_submitted && s->deferred && (s->buffered() > 0 || s->upstream_done);
            }
            if (submit) {
                submit_head(*s);
            } else if (resume) {
                s->deferred = false;
                nghttp2_session_resume_data(session_, id);
            }
        }
    }

    void submit_head(Stream& s) {
        std::string status;
        std::vector<std::pair<std::string, std::string>> headers;
        {
            std::lock_guard lock(s.mu);
            status  = std::to_string(s.status);
            headers = s.response_headers;
        }
        std::vector<nghttp2_nv> nva;
        nva.reserve(headers.size() + 1);
        const auto nv = [](const std::string& name, const std::string& value) {
            return nghttp2_nv{reinterpret_cast<std::uint8_t*>(const_cast<char*>(name.data())),
                              reinterpret_cast<std::uint8_t*>(const_cast<char*>(value.data())),
                              name.size(), value.size(), NGHTTP2_NV_FLAG_NONE};
        };
        static const std::string kStatus = ":status";
        nva.push_back(nv(kStatus, status));
        for (const auto& [name, value] : headers) { nva.push_back(nv(name, value)); }
        s.head_submitted = true;
        // A HEAD answer, a 1xx, 204 or 304 has no body: it ends on its HEADERS frame (RFC 9113
        // 8.1), with no DATA frame after it.
        const int code = std::atoi(status.c_str());
        if (s.method == "HEAD" || (code >= 100 && code < 200) || code == 204 || code == 304) {
            nghttp2_submit_response2(session_, s.id, nva.data(), nva.size(), nullptr);
            return;
        }
        nghttp2_data_provider2 provider{};
        provider.source.ptr     = &s;
        provider.read_callback  = &Session::read_body;
        nghttp2_submit_response2(session_, s.id, nva.data(), nva.size(), &provider);
    }

    static nghttp2_ssize read_body(nghttp2_session*, std::int32_t, std::uint8_t* buf,
                                   std::size_t length, std::uint32_t* flags,
                                   nghttp2_data_source* source, void*) {
        auto& s = *static_cast<Stream*>(source->ptr);
        std::lock_guard lock(s.mu);
        const std::size_t available = s.buffered();
        if (available == 0) {
            if (s.upstream_done && s.upstream_failed) { return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE; }
            if (s.upstream_done) {
                *flags |= NGHTTP2_DATA_FLAG_EOF;
                return 0;
            }
            s.deferred = true;
            return NGHTTP2_ERR_DEFERRED;
        }
        const std::size_t n = std::min(length, available);
        std::memcpy(buf, s.out.data() + s.out_off, n);
        s.out_off += n;
        if (s.out_off == s.out.size()) {
            s.out.clear();
            s.out_off = 0;
            if (s.upstream_done && !s.upstream_failed) { *flags |= NGHTTP2_DATA_FLAG_EOF; }
        }
        s.room.notify_all();
        return static_cast<nghttp2_ssize>(n);
    }

    void start(const std::shared_ptr<Stream>& s) {
        if (s->too_large) {
            local_answer(*s, 413, "the request body is larger than this worker takes");
            s->worker_finished.store(true);
            return;
        }
        if (s->method.empty() || s->path.empty()) {
            nghttp2_submit_rst_stream(session_, NGHTTP2_FLAG_NONE, s->id, NGHTTP2_PROTOCOL_ERROR);
            s->worker_finished.store(true);
            return;
        }
        s->worker = std::thread(forward, s, options_.engine, options_.stream_buffer_bytes, waker_);
    }

    // Cancels a stream's request: its engine connection is closed by a thread of its own, since
    // httplib's stop() can wait for a connect in progress, and the session must not wait.
    static void cancel(Stream& s) {
        std::shared_ptr<httplib::Client> client;
        {
            std::lock_guard lock(s.mu);
            s.cancelled = true;
            s.cancel_flag.store(true);
            client = s.client;
            s.room.notify_all();
        }
        if (client) { std::thread([client] { client->stop(); }).detach(); }
    }
    static void join(Stream& s) {
        if (s.worker.joinable()) { s.worker.join(); }
    }

    // Joins the forwarding threads of closed streams that finished.
    void reap() {
        std::erase_if(retiring_, [](const std::shared_ptr<Stream>& s) {
            if (!s->worker_finished.load() && s->worker.joinable()) { return false; }
            join(*s);
            return true;
        });
    }

    static Session& self(void* user_data) { return *static_cast<Session*>(user_data); }

    static int on_begin_headers(nghttp2_session* session, const nghttp2_frame* frame,
                                void* user_data) {
        if (frame->hd.type != NGHTTP2_HEADERS || frame->headers.cat != NGHTTP2_HCAT_REQUEST) {
            return 0;
        }
        auto s = std::make_shared<Stream>();
        s->id  = frame->hd.stream_id;
        self(user_data).streams_.emplace(s->id, s);
        nghttp2_session_set_stream_user_data(session, s->id, s.get());
        return 0;
    }

    static int on_header(nghttp2_session* session, const nghttp2_frame* frame,
                         const std::uint8_t* name, std::size_t namelen, const std::uint8_t* value,
                         std::size_t valuelen, std::uint8_t, void*) {
        if (frame->hd.type != NGHTTP2_HEADERS || frame->headers.cat != NGHTTP2_HCAT_REQUEST) {
            return 0;
        }
        auto* s = static_cast<Stream*>(nghttp2_session_get_stream_user_data(session, frame->hd.stream_id));
        if (!s) { return 0; }
        const std::string_view key(reinterpret_cast<const char*>(name), namelen);
        std::string val(reinterpret_cast<const char*>(value), valuelen);
        if (key == ":method") {
            s->method = std::move(val);
        } else if (key == ":path") {
            s->path = std::move(val);
        } else if (key == ":authority") {
            s->authority = std::move(val);
        } else if (!key.starts_with(":") && !hop_by_hop(key)) {
            s->headers.emplace(std::string(key), std::move(val));
        }
        return 0;
    }

    static int on_data(nghttp2_session* session, std::uint8_t, std::int32_t stream_id,
                       const std::uint8_t* data, std::size_t len, void* user_data) {
        auto* s = static_cast<Stream*>(nghttp2_session_get_stream_user_data(session, stream_id));
        if (!s || s->too_large) { return 0; }
        if (s->body.size() + len > self(user_data).options_.max_request_bytes) {
            s->too_large = true;
            s->body.clear();
            s->body.shrink_to_fit();
            return 0;
        }
        s->body.append(reinterpret_cast<const char*>(data), len);
        return 0;
    }

    static int on_frame(nghttp2_session*, const nghttp2_frame* frame, void* user_data) {
        Session& session = self(user_data);
        switch (frame->hd.type) {
        case NGHTTP2_HEADERS:
        case NGHTTP2_DATA:
            if (frame->hd.flags & NGHTTP2_FLAG_END_STREAM) {
                if (const auto it = session.streams_.find(frame->hd.stream_id);
                    it != session.streams_.end() && !it->second->worker.joinable() &&
                    !it->second->worker_finished.load()) {
                    session.start(it->second);
                }
            }
            break;
        case NGHTTP2_PING:
            if (frame->hd.flags & NGHTTP2_FLAG_ACK) { session.ping_outstanding_ = false; }
            break;
        case NGHTTP2_GOAWAY:
            if (!session.goaway_received_) {
                session.goaway_received_ = true;
                if (session.events_.on_goaway) { session.events_.on_goaway(); }
            }
            break;
        default: break;
        }
        return 0;
    }

    static int on_close(nghttp2_session*, std::int32_t stream_id, std::uint32_t, void* user_data) {
        Session& session = self(user_data);
        const auto it    = session.streams_.find(stream_id);
        if (it == session.streams_.end()) { return 0; }
        std::shared_ptr<Stream> s = std::move(it->second);
        session.streams_.erase(it);
        s->closed = true;
        cancel(*s); // a stream reset (or ended) closes the engine's connection
        session.retiring_.push_back(std::move(s));
        return 0;
    }

    Link& link_;
    const TunnelOptions& options_;
    const TunnelEvents& events_;
    std::shared_ptr<Waker> waker_;
    nghttp2_session* session_ = nullptr;
    std::map<std::int32_t, std::shared_ptr<Stream>> streams_;
    std::vector<std::shared_ptr<Stream>> retiring_;
    std::string out_;
    std::size_t retry_length_ = 0; // the length of a write that waited, retried as it was
    bool ping_outstanding_ = false;
    bool goaway_received_  = false;
    bool protocol_error_   = false;
};

} // namespace

const char* tunnel_end_name(TunnelEnd end) noexcept {
    switch (end) {
    case TunnelEnd::Closed: return "closed";
    case TunnelEnd::PingTimeout: return "ping timeout";
    case TunnelEnd::Drained: return "drained after the router's GOAWAY";
    case TunnelEnd::Stopped: return "stopped";
    case TunnelEnd::ProtocolError: return "HTTP/2 protocol error";
    }
    return "";
}

TunnelEnd serve_tunnel(Link& link, std::string leftover, const TunnelOptions& options,
                       const TunnelEvents& events, const std::atomic<bool>& stop) {
    Session session(link, options, events);
    return session.run(std::move(leftover), stop);
}

} // namespace ninferno::connect
