#include "serve/http_transport.h"
#include "text/jinja.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#    include <arpa/inet.h>
#    include <netinet/in.h>
#    include <netinet/tcp.h>
#    include <poll.h>
#    include <sys/socket.h>
#    include <unistd.h>
#endif

namespace {

using ninferno::serve::ClientDisconnected;
using ninferno::serve::SseTransport;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

int test_sse_transport() {
    using namespace std::chrono_literals;
    int failures = 0;

    std::vector<std::string> writes;
    bool writable = true;
    bool write_ok = true;
    httplib::DataSink sink;
    sink.write = [&](const char* data, std::size_t size) {
        if (!write_ok) { return false; }
        writes.emplace_back(data, size);
        return true;
    };
    sink.is_writable = [&] { return writable; };

    std::atomic<bool> cancelled{false};
    const SseTransport::Clock::time_point start{100s};
    SseTransport transport(sink, cancelled, 5s, start);

    failures += check(!transport.poll(start + 4999ms) && writes.empty() && !cancelled,
                      "SSE transport emitted a heartbeat before the quiet interval");
    failures +=
        check(!transport.poll(start + 5s) &&
                  writes == std::vector<std::string>{std::string(SseTransport::kHeartbeatComment)},
              "SSE transport did not emit the standard comment heartbeat on time");
    failures += check(!transport.poll(start + 9999ms) && writes.size() == 1,
                      "SSE heartbeat did not reset the quiet interval");

    transport.write("data: token\n\n", start + 10s);
    failures += check(!transport.poll(start + 14999ms) && writes.size() == 2,
                      "normal SSE output did not reset the heartbeat interval");
    failures += check(!transport.poll(start + 15s) && writes.size() == 3 &&
                          writes.back() == SseTransport::kHeartbeatComment,
                      "quiet SSE output did not resume heartbeats after a data event");

    // Anthropic Messages streams heartbeat with the protocol's ping event instead of a comment.
    std::vector<std::string> ping_writes;
    httplib::DataSink ping_sink;
    ping_sink.write = [&](const char* data, std::size_t size) {
        ping_writes.emplace_back(data, size);
        return true;
    };
    ping_sink.is_writable = [] { return true; };
    std::atomic<bool> ping_cancelled{false};
    SseTransport ping_transport(ping_sink, ping_cancelled, 5s, start, SseTransport::kAnthropicPing);
    failures += check(
        !ping_transport.poll(start + 5s) &&
            ping_writes == std::vector<std::string>{"event: ping\ndata: {\"type\":\"ping\"}\n\n"},
        "Anthropic SSE transport did not heartbeat with a ping event");

    transport.write(std::vector<std::string>{"data: one\n\n", "data: two\n\n"}, start + 16s);
    failures +=
        check(writes.size() == 5 && writes[3] == "data: one\n\n" && writes[4] == "data: two\n\n",
              "SSE transport changed ordered multi-event output");

    writable = false;
    failures += check(transport.poll(start + 16s) && cancelled,
                      "unwritable SSE transport did not cancel its request");
    bool cancelled_write_threw = false;
    try {
        transport.write("data: unreachable\n\n", start + 16s);
    } catch (const ClientDisconnected&) { cancelled_write_threw = true; }
    failures += check(cancelled_write_threw && writes.size() == 5,
                      "cancelled SSE transport accepted another event");

    httplib::DataSink failed_sink;
    failed_sink.write       = [](const char*, std::size_t) { return false; };
    failed_sink.is_writable = [] { return true; };
    std::atomic<bool> heartbeat_cancelled{false};
    SseTransport failed_heartbeat(failed_sink, heartbeat_cancelled, 5s, start);
    failures += check(failed_heartbeat.poll(start + 5s) && heartbeat_cancelled,
                      "failed SSE heartbeat did not cancel its request");

    write_ok = false;
    writable = true;
    std::atomic<bool> event_cancelled{false};
    SseTransport failed_event(sink, event_cancelled, 5s, start);
    bool failed_event_threw = false;
    try {
        failed_event.write("data: failed\n\n", start);
    } catch (const ClientDisconnected&) { failed_event_threw = true; }
    failures += check(failed_event_threw && event_cancelled,
                      "failed SSE event write did not cancel its request");

    std::atomic<bool> releaser_cancelled{true};
    SseTransport released(sink, releaser_cancelled, 5s, start);
    failures += check(released.poll(start),
                      "response resource cancellation was not observed by SSE transport");
    return failures;
}

// A body nested past the limit is a 400, not a walk that overflows the stack; the chat template
// refuses deep input of its own (a tool call's arguments in history arrive as a string). JSON a
// field carries as text (a schema, a structural tag) has the same limit (test_openai_schema).
int test_json_nesting_limit() {
    const auto nested = [](std::size_t levels) {
        return std::string(levels, '[') + std::string(levels, ']');
    };
    const auto parses = [](std::string body) {
        httplib::Request request;
        request.body = std::move(body);
        try {
            (void)ninferno::serve::parse_json_body(request);
            return true;
        } catch (const std::exception&) { return false; }
    };
    int failures = 0;
    failures += check(parses(nested(128)), "a body nested 128 levels was refused");
    failures += check(!parses(nested(129)), "a body nested 129 levels was accepted");
    failures += check(!parses(nested(200000)), "a body nested 200,000 levels was accepted");

    const ninferno::text::JinjaTemplate jinja("{{ x | tojson }}", "depth");
    nlohmann::ordered_json deep = nullptr;
    for (int level = 0; level < 300; ++level) { deep = nlohmann::ordered_json::array({deep}); }
    bool refused = false;
    try {
        (void)jinja.render(nlohmann::ordered_json{{"x", deep}}, {});
    } catch (const std::invalid_argument&) { refused = true; }
    failures += check(refused, "a template context nested 300 levels was rendered");
    bool rendered = false;
    try {
        rendered = !jinja.render(nlohmann::ordered_json{{"x", {{"a", {1, 2}}}}}, {}).text.empty();
    } catch (const std::exception&) {}
    failures += check(rendered, "a shallow template context was refused");
    return failures;
}

int test_sse_response_headers() {
    httplib::Response response;
    ninferno::serve::prepare_sse_response(response);
    return check(response.get_header_value("Cache-Control") == "no-cache" &&
                     response.get_header_value("X-Accel-Buffering") == "no",
                 "SSE response lost its no-cache or anti-buffering header");
}

int test_prompt_json_member_order() {
    httplib::Request request;
    request.body =
        R"({"model":"claude-local","tools":[{"name":"probe","input_schema":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"object","properties":{"yankee":{"type":"integer"},"bravo":{"type":"boolean"}}}}}}],"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"probe","input":{"zeta":"last","alpha":{"yankee":2,"bravo":true}}}]}],"max_tokens":32})";

    const auto parsed = ninferno::serve::parse_json_body(request);
    return check(
        parsed.at("tools").at(0).at("input_schema").at("properties").dump() ==
                R"({"zeta":{"type":"string"},"alpha":{"type":"object","properties":{"yankee":{"type":"integer"},"bravo":{"type":"boolean"}}}})" &&
            parsed.at("messages").at(0).at("content").at(0).at("input").dump() ==
                R"({"zeta":"last","alpha":{"yankee":2,"bravo":true}})",
        "HTTP request JSON changed prompt-bearing object member order");
}

#if defined(__linux__)
class Socket final {
public:
    explicit Socket(int descriptor = -1) : descriptor_(descriptor) {}

    ~Socket() {
        if (descriptor_ >= 0) { ::close(descriptor_); }
    }

    Socket(const Socket&)            = delete;
    Socket& operator=(const Socket&) = delete;

    [[nodiscard]] int get() const noexcept { return descriptor_; }

private:
    int descriptor_ = -1;
};

template <class T>
bool socket_option_equals(int socket, int level, int option, T expected) {
    T actual{};
    socklen_t size = sizeof(actual);
    return ::getsockopt(socket, level, option, &actual, &size) == 0 && size == sizeof(actual) &&
           actual == expected;
}

int test_inherited_socket_liveness() {
    int failures = 0;
    Socket listener(::socket(AF_INET, SOCK_STREAM, 0));
    if (listener.get() < 0) { return check(false, "failed to create HTTP listener test socket"); }
    ninferno::serve::configure_http_server_socket(listener.get());

    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port        = 0;
    if (::bind(listener.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(listener.get(), 1) != 0) {
        return check(false, "failed to bind HTTP listener test socket");
    }
    socklen_t address_size = sizeof(address);
    if (::getsockname(listener.get(), reinterpret_cast<sockaddr*>(&address), &address_size) != 0) {
        return check(false, "failed to inspect HTTP listener test address");
    }

    Socket client(::socket(AF_INET, SOCK_STREAM, 0));
    if (client.get() < 0 || ::connect(client.get(), reinterpret_cast<const sockaddr*>(&address),
                                      sizeof(address)) != 0) {
        return check(false, "failed to connect HTTP liveness test socket");
    }
    Socket accepted(::accept(listener.get(), nullptr, nullptr));
    if (accepted.get() < 0) { return check(false, "failed to accept HTTP liveness test socket"); }

    failures += check(socket_option_equals(accepted.get(), SOL_SOCKET, SO_KEEPALIVE, 1),
                      "accepted HTTP socket did not inherit SO_KEEPALIVE");
    failures += check(socket_option_equals(accepted.get(), IPPROTO_TCP, TCP_KEEPIDLE, 10),
                      "accepted HTTP socket did not inherit TCP_KEEPIDLE");
    failures += check(socket_option_equals(accepted.get(), IPPROTO_TCP, TCP_KEEPINTVL, 3),
                      "accepted HTTP socket did not inherit TCP_KEEPINTVL");
    failures += check(socket_option_equals(accepted.get(), IPPROTO_TCP, TCP_KEEPCNT, 3),
                      "accepted HTTP socket did not inherit TCP_KEEPCNT");
    failures += check(
        socket_option_equals<unsigned int>(accepted.get(), IPPROTO_TCP, TCP_USER_TIMEOUT, 15000U),
        "accepted HTTP socket did not inherit TCP_USER_TIMEOUT");
    return failures;
}

using SocketSetup = void (*)(int);

void configure_http_listener(int socket) { ninferno::serve::configure_http_server_socket(socket); }

// The listener setup of servers such as llama.cpp and uvicorn.
void reuse_address_only(int socket) {
    const int enabled = 1;
    (void)::setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
}

// Leaves a server-closed connection in TIME_WAIT on a loopback port whose
// listener was prepared by `setup`, and closes that listener. Returns the
// port's address, or a zero port on failure.
sockaddr_in leave_server_time_wait(SocketSetup setup) {
    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    Socket listener(::socket(AF_INET, SOCK_STREAM, 0));
    if (listener.get() < 0) { return {}; }
    setup(listener.get());
    socklen_t address_size = sizeof(address);
    if (::bind(listener.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(listener.get(), 1) != 0 ||
        ::getsockname(listener.get(), reinterpret_cast<sockaddr*>(&address), &address_size) != 0) {
        return {};
    }
    Socket client(::socket(AF_INET, SOCK_STREAM, 0));
    if (client.get() < 0 || ::connect(client.get(), reinterpret_cast<const sockaddr*>(&address),
                                      sizeof(address)) != 0) {
        return {};
    }
    {
        // The server closes first, so its side of the connection enters TIME_WAIT.
        Socket accepted(::accept(listener.get(), nullptr, nullptr));
        if (accepted.get() < 0) { return {}; }
    }
    char byte = 0;
    if (::recv(client.get(), &byte, 1, 0) != 0) { return {}; }
    return address;
}

bool successor_binds(const sockaddr_in& address, SocketSetup setup) {
    Socket successor(::socket(AF_INET, SOCK_STREAM, 0));
    if (successor.get() < 0) { return false; }
    setup(successor.get());
    return ::bind(successor.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) ==
           0;
}

int test_successor_binds_over_time_wait() {
    struct Case {
        SocketSetup previous;
        SocketSetup next;
        const char* message;
    };

    const Case cases[] = {
        {configure_http_listener, reuse_address_only,
         "a SO_REUSEADDR server could not bind over the HTTP listener's TIME_WAIT connection"},
        {reuse_address_only, configure_http_listener,
         "the HTTP listener could not bind over a SO_REUSEADDR server's TIME_WAIT connection"},
        {configure_http_listener, configure_http_listener,
         "the HTTP listener could not bind over its own TIME_WAIT connection"},
    };
    int failures = 0;
    for (const Case& test_case : cases) {
        const sockaddr_in address = leave_server_time_wait(test_case.previous);
        if (address.sin_port == 0) {
            failures += check(false, "failed to leave a TIME_WAIT test connection");
            continue;
        }
        failures += check(successor_binds(address, test_case.next), test_case.message);
    }
    return failures;
}
#endif

} // namespace

#if defined(__linux__)
// Overload is an honest 429/503 from the Engine's admission, never a connection
// the HTTP layer closes without a response. Found by the chaos harness: 64 simultaneous requests
// against max_concurrency 4 + max_pending 8 left 22 clients with "Remote end closed connection
// without response", because the worker pool (C+P+1 threads, C+P queued) closed every accepted
// socket beyond that. Idle keep-alive sockets hold workers the same way. Here idle connections
// occupy every worker of the planned pool and more; one real request must still be answered.
int test_worker_plan_answers_every_connection() {
    const ninferno::serve::HttpWorkerPlan plan = ninferno::serve::plan_http_workers(1, 1);
    int failures                               = check(plan.threads >= 2 && plan.max_queued >= 8,
                                                       "HTTP worker plan does not queue the connections it was sized for");

    httplib::Server server;
    server.new_task_queue = [plan] {
        return new httplib::ThreadPool(plan.threads, plan.threads, plan.max_queued);
    };
    server.set_read_timeout(1, 0);
    server.set_keep_alive_timeout(1);
    server.Get("/ping", [](const httplib::Request&, httplib::Response& response) {
        response.set_content("pong", "text/plain");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) { return failures + check(false, "test HTTP server could not bind"); }
    std::thread listener([&] { server.listen_after_bind(); });
    server.wait_until_ready();

    std::vector<int> idle;
    for (std::size_t i = 0; i < plan.threads + 8; ++i) {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in address{};
        address.sin_family      = AF_INET;
        address.sin_port        = htons(static_cast<std::uint16_t>(port));
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (fd >= 0 &&
            ::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0) {
            idle.push_back(fd);
        } else if (fd >= 0) {
            ::close(fd);
        }
    }
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(10, 0);
    client.set_read_timeout(10, 0);
    const httplib::Result result = client.Get("/ping");
    failures += check(result && result->status == 200 && result->body == "pong",
                      "an HTTP request behind saturated workers was dropped without a response");
    for (const int fd : idle) { ::close(fd); }
    server.stop();
    listener.join();
    return failures;
}

// An error response the application marks reusable keeps the connection for the caller's next
// request; one that sends `Connection: close` also closes the socket, so a client waiting for EOF
// is not held until the keep-alive timeout (#319).
int test_error_connection_reuse() {
    httplib::Server server;
    server.set_read_timeout(5, 0);
    server.set_keep_alive_timeout(30);
    server.Get("/refuse", [](const httplib::Request&, httplib::Response& response) {
        response.status                 = 401;
        response.keep_alive_after_error = true;
        response.set_content("no", "text/plain");
    });
    server.Get("/close", [](const httplib::Request&, httplib::Response& response) {
        response.status = 503;
        response.set_content("bye", "text/plain");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) { return check(false, "test HTTP server could not bind"); }
    std::thread listener([&] { server.listen_after_bind(); });
    server.wait_until_ready();

    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_port        = htons(static_cast<std::uint16_t>(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    timeval timeout{3, 0};
    int failures = 0;
    if (fd < 0 || ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0 ||
        ::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        failures += check(false, "test HTTP client could not connect");
    } else {
        // Reads one response (headers and a Content-Length body); empty on EOF or timeout.
        std::string pending;
        const auto read_response = [&]() -> std::string {
            char buffer[512];
            for (;;) {
                const std::size_t end = pending.find("\r\n\r\n");
                if (end != std::string::npos) {
                    const std::size_t at = pending.find("Content-Length: ");
                    const std::size_t length =
                        at == std::string::npos ? 0 : std::stoul(pending.substr(at + 16));
                    if (pending.size() >= end + 4 + length) {
                        std::string response = pending.substr(0, end + 4 + length);
                        pending.erase(0, end + 4 + length);
                        return response;
                    }
                }
                const ssize_t got = ::recv(fd, buffer, sizeof(buffer), 0);
                if (got <= 0) { return {}; }
                pending.append(buffer, static_cast<std::size_t>(got));
            }
        };
        const auto send_request = [&](const char* path) {
            const std::string request =
                std::string("GET ") + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n";
            return ::send(fd, request.data(), request.size(), 0) ==
                   static_cast<ssize_t>(request.size());
        };
        const bool refused = send_request("/refuse");
        const std::string first = read_response();
        failures += check(refused && first.find(" 401 ") != std::string::npos &&
                              first.find("Connection: close") == std::string::npos,
                          "a reusable HTTP refusal closed the connection");
        const bool closing = send_request("/close");
        const std::string second = read_response();
        failures += check(closing && second.find(" 503 ") != std::string::npos &&
                              second.find("Connection: close") != std::string::npos,
                          "an HTTP error was not answered on the reused connection with close");
        char byte = 0;
        const auto start  = std::chrono::steady_clock::now();
        const ssize_t eof = ::recv(fd, &byte, 1, 0);
        failures += check(eof == 0 && std::chrono::steady_clock::now() - start <
                                          std::chrono::seconds(2),
                          "an HTTP response with Connection: close left the socket open");
    }
    if (fd >= 0) { ::close(fd); }
    server.stop();
    listener.join();
    return failures;
}

// Request smuggling: a request whose body framing is ambiguous (two Content-Length fields, a list
// value "5, 5", a Content-Length beside Transfer-Encoding) is refused with 400 and its connection
// closed, so its body and the bytes after it are never read as further requests. A regression
// guard across cpp-httplib updates: v0.54.1 and v0.65.0 both refuse all three.
int test_ambiguous_framing_is_refused(const std::string& framing, const std::string& body,
                                      const char* label) {
    httplib::Server server;
    server.set_read_timeout(3, 0);
    std::atomic<bool> smuggled{false};
    server.Post("/echo", [](const httplib::Request& request, httplib::Response& response) {
        response.set_content(request.body, "text/plain");
    });
    server.Get("/smuggled", [&](const httplib::Request&, httplib::Response& response) {
        smuggled = true;
        response.set_content("smuggled", "text/plain");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) { return check(false, "test HTTP server could not bind"); }
    std::thread listener([&] { server.listen_after_bind(); });
    server.wait_until_ready();

    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_port        = htons(static_cast<std::uint16_t>(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    timeval timeout{3, 0};
    int failures = 0;
    if (fd < 0 || ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0 ||
        ::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        failures += check(false, "test HTTP client could not connect");
    } else {
        const std::string request = "POST /echo HTTP/1.1\r\nHost: 127.0.0.1\r\n" + framing +
                                    "\r\n" + body +
                                    "GET /smuggled HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n";
        const bool sent = ::send(fd, request.data(), request.size(), 0) ==
                          static_cast<ssize_t>(request.size());
        std::string received;
        char buffer[512];
        for (;;) {
            const ssize_t got = ::recv(fd, buffer, sizeof(buffer), 0);
            if (got <= 0) { break; }
            received.append(buffer, static_cast<std::size_t>(got));
        }
        const std::string first = received.substr(0, received.find("\r\n"));
        if (!sent || first.find(" 400 ") == std::string::npos) {
            std::cerr << label << ": first response \"" << first << "\"\n";
            failures += check(false, "ambiguous request framing was not refused with 400");
        }
        failures += check(!smuggled && received.find("smuggled") == std::string::npos,
                          "the bytes after a refused request were served as a second request");
    }
    if (fd >= 0) { ::close(fd); }
    server.stop();
    listener.join();
    return failures;
}

// A sender that trickles a request body must not hold its worker (#526): the library bounds the
// headers as a whole and every wait for a body byte, but a peer delivering a byte every few seconds
// passed both. Here the body must grow by 1000 B/s over each 400 ms window. A trickle of one byte
// per 50 ms is answered 408 and its connection closed within about a second, while a steady sender
// that takes much longer than a window in all, and one that pauses briefly after a fast start, are
// served.
int test_slow_request_body_is_cut_off() {
    using namespace std::chrono_literals;
    httplib::Server server;
    server.set_read_timeout(5, 0);
    server.set_request_body_min_rate(1000, 400);
    server.Post("/echo", [](const httplib::Request& request, httplib::Response& response) {
        response.set_content(std::to_string(request.body.size()), "text/plain");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) { return check(false, "test HTTP server could not bind"); }
    std::thread listener([&] { server.listen_after_bind(); });
    server.wait_until_ready();

    // Sends the headers for `length` body bytes, then `pieces` (bytes, pause after) in order.
    // Returns the status line and whether the server closed the connection, and the elapsed time.
    struct Outcome {
        std::string status;
        bool closed = false;
        std::chrono::milliseconds elapsed{0};
    };

    const auto exchange =
        [&](const std::string& framing, const std::string& prefix, char fill,
            const std::vector<std::pair<std::size_t, std::chrono::milliseconds>>& pieces) {
            Outcome outcome;
            const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            sockaddr_in address{};
            address.sin_family      = AF_INET;
            address.sin_port        = htons(static_cast<std::uint16_t>(port));
            address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            timeval timeout{5, 0};
            if (fd < 0 ||
                ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0 ||
                ::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
                if (fd >= 0) { ::close(fd); }
                return outcome;
            }
            const auto start = std::chrono::steady_clock::now();
            const std::string head =
                "POST /echo HTTP/1.1\r\nHost: 127.0.0.1\r\n" + framing + "\r\n\r\n" + prefix;
            (void)::send(fd, head.data(), head.size(), MSG_NOSIGNAL);
            for (const auto& [bytes, pause] : pieces) {
                const std::string piece(bytes, fill);
                if (::send(fd, piece.data(), piece.size(), MSG_NOSIGNAL) < 0) { break; }
                // The pause ends early when the server has already answered.
                pollfd readable{fd, POLLIN, 0};
                if (::poll(&readable, 1, static_cast<int>(pause.count())) > 0) { break; }
            }
            std::string received;
            char buffer[512];
            for (;;) {
                const ssize_t got = ::recv(fd, buffer, sizeof(buffer), 0);
                if (got <= 0) {
                    outcome.closed = got == 0;
                    break;
                }
                received.append(buffer, static_cast<std::size_t>(got));
            }
            outcome.elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start);
            outcome.status = received.substr(0, received.find("\r\n"));
            ::close(fd);
            return outcome;
        };

    int failures = 0;
    std::vector<std::pair<std::size_t, std::chrono::milliseconds>> trickle(100, {1, 50ms});
    const Outcome slow = exchange("Content-Length: 100", "", 'x', trickle);
    failures += check(slow.status.find(" 408 ") != std::string::npos && slow.closed,
                      "a body trickled one byte at a time was not answered 408 and closed");
    failures += check(slow.elapsed < 2500ms, "a trickled body held its worker past the window");

    std::vector<std::pair<std::size_t, std::chrono::milliseconds>> steady(12, {200, 100ms});
    const Outcome fast = exchange("Content-Length: 2400", "", 'x', steady);
    failures += check(fast.status.find(" 200 ") != std::string::npos,
                      "a body arriving above the minimum rate was cut off");

    const Outcome paused = exchange("Content-Length: 4100", "", 'x', {{4000, 600ms}, {100, 0ms}});
    failures += check(paused.status.find(" 200 ") != std::string::npos,
                      "a body that paused briefly after a fast start was cut off");

    // The framing counts too: a chunk-size line with an extension sent a byte at a time never
    // reaches the body, and used to hold the worker for the whole length the line may have.
    std::vector<std::pair<std::size_t, std::chrono::milliseconds>> extension(100, {1, 50ms});
    const Outcome framing = exchange("Transfer-Encoding: chunked", "1;", 'a', extension);
    failures += check(framing.status.find(" 408 ") != std::string::npos && framing.closed &&
                          framing.elapsed < 2500ms,
                      "a chunk-size line trickled one byte at a time held its worker");

    server.stop();
    listener.join();
    return failures;
}

// Slow senders that fill the worker pool and its queue must not make the server unreachable
// (#526): a connection past the queue is answered at once, /health with the health answer and the
// rest with 503 and Retry-After. Two workers and one queue slot are held by stalled requests.
int test_overloaded_pool_still_answers() {
    using namespace std::chrono_literals;
    httplib::Server server;
    server.new_task_queue = [] { return new httplib::ThreadPool(2, 2, 1); };
    server.set_read_timeout(5, 0);
    server.set_socket_options(ninferno::serve::configure_http_server_socket);
    server.set_overload_handler([](socket_t socket) {
        ninferno::serve::answer_overloaded_connection(socket, [] {
            return ninferno::serve::HealthReply{.status = 200, .body = R"({"status":"ok"})"};
        });
    });
    server.Get("/health", [](const httplib::Request&, httplib::Response& response) {
        response.set_content("from a worker", "text/plain");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) { return check(false, "test HTTP server could not bind"); }
    std::thread listener([&] { server.listen_after_bind(); });
    server.wait_until_ready();

    const auto connect_to = [&] {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in address{};
        address.sin_family      = AF_INET;
        address.sin_port        = htons(static_cast<std::uint16_t>(port));
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        timeval timeout{3, 0};
        (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
            ::close(fd);
            return -1;
        }
        return fd;
    };
    const auto get = [&](const std::string& path) {
        const int fd = connect_to();
        if (fd < 0) { return std::string(); }
        const std::string request = "GET " + path + " HTTP/1.1\r\nHost: x\r\n\r\n";
        (void)::send(fd, request.data(), request.size(), MSG_NOSIGNAL);
        std::string received;
        char buffer[512];
        for (ssize_t got; (got = ::recv(fd, buffer, sizeof(buffer), 0)) > 0;) {
            received.append(buffer, static_cast<std::size_t>(got));
        }
        ::close(fd);
        return received;
    };

    std::vector<int> silent;
    for (int i = 0; i < 3; ++i) {
        // A request that stops in its headers: it holds its worker until the read timeout. (A
        // connection that sent nothing is not even accepted, see TCP_DEFER_ACCEPT.)
        const int fd              = connect_to();
        const std::string partial = "GET /slow HTTP/1.1\r\nHo";
        (void)::send(fd, partial.data(), partial.size(), MSG_NOSIGNAL);
        silent.push_back(fd);
        std::this_thread::sleep_for(100ms);
    }
    int failures             = 0;
    const auto start         = std::chrono::steady_clock::now();
    const std::string health = get("/health");
    failures += check(health.find(" 200 ") != std::string::npos &&
                          health.find(R"({"status":"ok"})") != std::string::npos,
                      "/health was not answered while every worker and the queue were held");
    const std::string other = get("/v1/models");
    failures += check(other.find(" 503 ") != std::string::npos &&
                          other.find("Retry-After: 5") != std::string::npos &&
                          other.find("server_busy") != std::string::npos,
                      "a connection past a full queue was not answered 503 with Retry-After");
    failures += check(std::chrono::steady_clock::now() - start < 2s,
                      "answers for an overloaded pool were not immediate");
    for (const int fd : silent) {
        if (fd >= 0) { ::close(fd); }
    }
    server.stop();
    listener.join();
    return failures;
}
#endif

// /health reports a stalled executor. Busy without progress for the threshold is a stall;
// progress, or having no work, clears it; a zero threshold disables the check.
int test_executor_stall_monitor() {
    using namespace std::chrono_literals;
    using Monitor = ninferno::serve::ExecutorStallMonitor;
    const Monitor::Clock::time_point t0{};
    int failures = 0;
    ninferno::RuntimeStats busy;
    busy.running_requests = 2;
    Monitor monitor(3000ms);
    failures += check(!monitor.update(busy, t0), "first observation is not a stall");
    failures += check(!monitor.update(busy, t0 + 2999ms), "busy below the threshold");
    failures += check(monitor.update(busy, t0 + 3000ms), "busy without progress for 3 s stalls");
    ninferno::RuntimeStats moved       = busy;
    moved.committed_decode_tokens    = 5;
    failures += check(!monitor.update(moved, t0 + 3100ms), "progress clears the stall at once");
    failures += check(!monitor.update(moved, t0 + 6000ms), "the clock restarts at progress");
    failures += check(monitor.update(moved, t0 + 6100ms), "and a stall is reported again after it");
    ninferno::RuntimeStats idle = moved;
    idle.running_requests     = 0;
    failures += check(!monitor.update(idle, t0 + 20000ms), "an idle engine is never stalled");
    ninferno::RuntimeStats queued = idle;
    queued.waiting_requests     = 1;
    failures += check(!monitor.update(queued, t0 + 20001ms), "queued work starts a new window");
    failures += check(monitor.update(queued, t0 + 23001ms), "queued work that is never admitted stalls");
    Monitor off(0ms);
    failures += check(!off.update(busy, t0) && !off.update(busy, t0 + 3600s),
                      "a zero threshold disables the check");
    return failures;
}

int main() {
    int failures = test_sse_transport() + test_sse_response_headers() +
                   test_prompt_json_member_order() + test_executor_stall_monitor() +
                   test_json_nesting_limit();
#if defined(__linux__)
    failures += test_inherited_socket_liveness();
    failures += test_worker_plan_answers_every_connection();
    failures += test_successor_binds_over_time_wait();
    failures += test_error_connection_reuse();
    failures += test_slow_request_body_is_cut_off();
    failures += test_overloaded_pool_still_answers();
    failures += test_ambiguous_framing_is_refused("Content-Length: 5\r\nContent-Length: 6\r\n",
                                                  "hello", "two Content-Length fields");
    failures += test_ambiguous_framing_is_refused("Content-Length: 5, 5\r\n", "hello",
                                                  "a Content-Length list");
    failures += test_ambiguous_framing_is_refused(
        "Content-Length: 5\r\nTransfer-Encoding: chunked\r\n", "0\r\n\r\n",
        "Content-Length beside Transfer-Encoding");
#endif
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
