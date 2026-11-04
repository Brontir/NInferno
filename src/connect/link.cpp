#include "connect/link.h"

#include <openssl/err.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <utility>

namespace ninferno::connect {
namespace {

std::string lower(std::string_view s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

std::string openssl_error() {
    const unsigned long code = ERR_get_error();
    if (code == 0) { return "unknown TLS error"; }
    char buffer[256];
    ERR_error_string_n(code, buffer, sizeof buffer);
    ERR_clear_error();
    return buffer;
}

void set_timeouts(int fd, int timeout_ms) {
    timeval tv{};
    tv.tv_sec  = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
}

void make_nonblocking(int fd) {
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0) { (void)fcntl(fd, F_SETFL, flags | O_NONBLOCK); }
    // The handshake's timeouts would otherwise bound nothing now, but clear them for clarity.
    set_timeouts(fd, 0);
}

class PlainLink final : public Link {
public:
    explicit PlainLink(int fd) : fd_(fd) {}
    ~PlainLink() override {
        if (fd_ >= 0) { ::close(fd_); }
    }
    PlainLink(const PlainLink&)            = delete;
    PlainLink& operator=(const PlainLink&) = delete;

    int fd() const noexcept override { return fd_; }
    long read(std::uint8_t* data, std::size_t size) override {
        const ssize_t n = ::recv(fd_, data, size, 0);
        if (n > 0) { return static_cast<long>(n); }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) { return 0; }
        return -1;
    }
    long write(const std::uint8_t* data, std::size_t size) override {
        const ssize_t n = ::send(fd_, data, size, MSG_NOSIGNAL);
        if (n >= 0) { return static_cast<long>(n); }
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) { return 0; }
        return -1;
    }
    bool wants_writable() const noexcept override { return false; }
    bool write_waits_readable() const noexcept override { return false; }
    void set_nonblocking() override { make_nonblocking(fd_); }
    bool pending() const noexcept override { return false; }

private:
    int fd_;
};

class TlsLink final : public Link {
public:
    TlsLink(int fd, SSL_CTX* ctx, SSL* ssl) : fd_(fd), ctx_(ctx), ssl_(ssl) {}
    ~TlsLink() override {
        if (ssl_) {
            (void)SSL_shutdown(ssl_);
            SSL_free(ssl_);
        }
        if (ctx_) { SSL_CTX_free(ctx_); }
        if (fd_ >= 0) { ::close(fd_); }
    }
    TlsLink(const TlsLink&)            = delete;
    TlsLink& operator=(const TlsLink&) = delete;

    int fd() const noexcept override { return fd_; }
    // An operation that waits for the other direction (SSL_read wanting to write, as a TLS 1.3
    // KeyUpdate's answer does; SSL_write wanting to read) is retried before the other operation is
    // issued: until then the other reports nothing to do, and the loop's next pass retries it.
    long read(std::uint8_t* data, std::size_t size) override {
        if (blocked_ == Blocked::WriteWantsRead) { return 0; }
        wants_writable_ = false;
        const int n     = SSL_read(ssl_, data, static_cast<int>(std::min<std::size_t>(size, 1 << 30)));
        if (n > 0) {
            blocked_ = Blocked::None;
            return n;
        }
        return settle(n, false);
    }
    long write(const std::uint8_t* data, std::size_t size) override {
        if (blocked_ == Blocked::ReadWantsWrite) {
            wants_writable_ = true;
            return 0;
        }
        wants_writable_ = false;
        const int n = SSL_write(ssl_, data, static_cast<int>(std::min<std::size_t>(size, 1 << 30)));
        if (n > 0) {
            blocked_ = Blocked::None;
            return n;
        }
        return settle(n, true);
    }
    bool wants_writable() const noexcept override { return wants_writable_; }
    bool write_waits_readable() const noexcept override {
        return blocked_ == Blocked::WriteWantsRead;
    }
    void set_nonblocking() override { make_nonblocking(fd_); }
    bool pending() const noexcept override { return SSL_pending(ssl_) > 0; }

private:
    enum class Blocked { None, ReadWantsWrite, WriteWantsRead };

    long settle(int n, bool writing) {
        switch (SSL_get_error(ssl_, n)) {
        case SSL_ERROR_WANT_READ:
            blocked_ = writing ? Blocked::WriteWantsRead : Blocked::None;
            return 0;
        case SSL_ERROR_WANT_WRITE:
            blocked_        = writing ? Blocked::None : Blocked::ReadWantsWrite;
            wants_writable_ = true;
            return 0;
        default: ERR_clear_error(); return -1;
        }
    }

    int fd_;
    SSL_CTX* ctx_;
    SSL* ssl_;
    bool wants_writable_ = false;
    Blocked blocked_     = Blocked::None;
};

int tcp_connect(const RouterUrl& url, int timeout_ms, std::string* why) {
    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* found   = nullptr;
    const std::string port = std::to_string(url.port);
    if (const int rc = getaddrinfo(url.host.c_str(), port.c_str(), &hints, &found); rc != 0) {
        *why = "cannot resolve " + url.host + ": " + gai_strerror(rc);
        return -1;
    }
    int fd = -1;
    std::string last = "no address";
    for (addrinfo* a = found; a; a = a->ai_next) {
        fd = ::socket(a->ai_family, a->ai_socktype | SOCK_CLOEXEC, a->ai_protocol);
        if (fd < 0) { continue; }
        set_timeouts(fd, timeout_ms); // a blocking connect() honours SO_SNDTIMEO on Linux
        if (::connect(fd, a->ai_addr, a->ai_addrlen) == 0) { break; }
        last = std::strerror(errno);
        ::close(fd);
        fd = -1;
    }
    freeaddrinfo(found);
    if (fd < 0) {
        *why = "cannot connect to " + url.display() + ": " + last;
        return -1;
    }
    const int one = 1;
    (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    (void)setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof one);
    return fd;
}

// Reads one HTTP/1.1 response head (to the blank line) from a blocking link; the bytes past it
// stay in *rest.
bool read_head(Link& link, std::string* head, std::string* rest) {
    std::string buffer;
    std::uint8_t chunk[4096];
    while (buffer.size() < 64 * 1024) {
        const long n = link.read(chunk, sizeof chunk);
        if (n <= 0) { return false; }
        buffer.append(reinterpret_cast<const char*>(chunk), static_cast<std::size_t>(n));
        if (const auto end = buffer.find("\r\n\r\n"); end != std::string::npos) {
            *head = buffer.substr(0, end + 2);
            *rest = buffer.substr(end + 4);
            // What the link read already past the head (a TLS record's rest) goes with it.
            while (link.pending()) {
                const long more = link.read(chunk, sizeof chunk);
                if (more <= 0) { break; }
                rest->append(reinterpret_cast<const char*>(chunk), static_cast<std::size_t>(more));
            }
            return true;
        }
    }
    return false;
}

bool write_all(Link& link, std::string_view data) {
    while (!data.empty()) {
        const long n = link.write(reinterpret_cast<const std::uint8_t*>(data.data()), data.size());
        if (n <= 0) { return false; }
        data.remove_prefix(static_cast<std::size_t>(n));
    }
    return true;
}

} // namespace

std::string RouterUrl::display() const {
    const bool v6 = host.find(':') != std::string::npos;
    return std::string(tls ? "https://" : "http://") + (v6 ? "[" + host + "]" : host) + ":" +
           std::to_string(port) + path;
}

std::optional<RouterUrl> parse_router_url(std::string_view url, std::string* why) {
    RouterUrl out;
    if (url.starts_with("https://")) {
        url.remove_prefix(8);
    } else if (url.starts_with("http://")) {
        out.tls  = false;
        out.port = 80;
        url.remove_prefix(7);
    } else {
        *why = "a router URL starts with https:// or http://";
        return std::nullopt;
    }
    std::string_view authority = url;
    if (const auto slash = url.find('/'); slash != std::string_view::npos) {
        authority = url.substr(0, slash);
        out.path  = std::string(url.substr(slash));
    }
    std::string_view port;
    if (authority.starts_with("[")) {
        const auto close = authority.find(']');
        if (close == std::string_view::npos) {
            *why = "an IPv6 router address needs its closing ]";
            return std::nullopt;
        }
        out.host = std::string(authority.substr(1, close - 1));
        const std::string_view after = authority.substr(close + 1);
        if (after.starts_with(":")) {
            port = after.substr(1);
        } else if (!after.empty()) {
            *why = "unexpected text after the IPv6 router address";
            return std::nullopt;
        }
    } else if (const auto colon = authority.rfind(':'); colon != std::string_view::npos) {
        out.host = std::string(authority.substr(0, colon));
        port     = authority.substr(colon + 1);
    } else {
        out.host = std::string(authority);
    }
    if (out.host.empty()) {
        *why = "a router URL needs a host";
        return std::nullopt;
    }
    if (!port.empty() || authority.ends_with(":")) {
        int value     = 0;
        const auto rc = std::from_chars(port.data(), port.data() + port.size(), value);
        if (rc.ec != std::errc{} || rc.ptr != port.data() + port.size() || value < 1 ||
            value > 65535) {
            *why = "a router port is 1..65535";
            return std::nullopt;
        }
        out.port = value;
    }
    return out;
}

std::unique_ptr<Link> dial(const RouterUrl& url, const TlsOptions& tls, int timeout_ms,
                           std::string* why) {
    const int fd = tcp_connect(url, timeout_ms, why);
    if (fd < 0) { return nullptr; }
    if (!url.tls) { return std::make_unique<PlainLink>(fd); }

    SSL_CTX* ctx = SSL_CTX_new(TLS_client_method());
    if (!ctx) {
        *why = "TLS: " + openssl_error();
        ::close(fd);
        return nullptr;
    }
    (void)SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
    // A write retried after WANT_WRITE comes from the session's output buffer, which may have grown
    // (moved) since; and a write may take part of what it is given.
    SSL_CTX_set_mode(ctx, SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER | SSL_MODE_ENABLE_PARTIAL_WRITE);
    SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);
    const bool trusted = tls.ca_file.empty()
                             ? SSL_CTX_set_default_verify_paths(ctx) == 1
                             : SSL_CTX_load_verify_locations(ctx, tls.ca_file.c_str(), nullptr) == 1;
    if (!trusted) {
        *why = "TLS: cannot load the trust store" +
               (tls.ca_file.empty() ? std::string() : " " + tls.ca_file) + ": " + openssl_error();
        SSL_CTX_free(ctx);
        ::close(fd);
        return nullptr;
    }
    SSL* ssl = SSL_new(ctx);
    // What the certificate must carry: the host's DNS name, or for an address literal that address
    // (an IP name of its own); the name the router is asked for (SNI) only for a DNS name.
    in6_addr any{};
    const bool literal = inet_pton(AF_INET, url.host.c_str(), &any) == 1 ||
                         inet_pton(AF_INET6, url.host.c_str(), &any) == 1;
    if (literal) {
        (void)X509_VERIFY_PARAM_set1_ip_asc(SSL_get0_param(ssl), url.host.c_str());
    } else {
        (void)SSL_set1_host(ssl, url.host.c_str());
        (void)SSL_set_tlsext_host_name(ssl, url.host.c_str());
    }
    static constexpr unsigned char kAlpn[] = {8, 'h', 't', 't', 'p', '/', '1', '.', '1'};
    (void)SSL_set_alpn_protos(ssl, kAlpn, sizeof kAlpn);
    SSL_set_fd(ssl, fd);
    if (SSL_connect(ssl) != 1) {
        const long verify = SSL_get_verify_result(ssl);
        *why = "TLS handshake with " + url.display() + " failed: " +
               (verify != X509_V_OK ? std::string(X509_verify_cert_error_string(verify))
                                    : openssl_error());
        SSL_free(ssl);
        SSL_CTX_free(ctx);
        ::close(fd);
        return nullptr;
    }
    return std::make_unique<TlsLink>(fd, ctx, ssl);
}

std::unique_ptr<Link> plain_link(int fd) { return std::make_unique<PlainLink>(fd); }

Handshake handshake(Link& link, const RouterUrl& url, std::string_view key,
                    std::string_view registration_json) {
    Handshake out;
    const bool v6       = url.host.find(':') != std::string::npos;
    const std::string host = (v6 ? "[" + url.host + "]" : url.host) +
                             ((url.port == (url.tls ? 443 : 80)) ? "" : ":" + std::to_string(url.port));
    std::string request = "POST " + url.path + " HTTP/1.1\r\n";
    request += "Host: " + host + "\r\n";
    request += "Connection: Upgrade\r\n";
    request += "Upgrade: " + std::string(kUpgradeToken) + "\r\n";
    request += "Authorization: Bearer " + std::string(key) + "\r\n";
    request += "Content-Type: application/json\r\n";
    request += "Content-Length: " + std::to_string(registration_json.size()) + "\r\n\r\n";
    request += registration_json;
    if (!write_all(link, request)) {
        out.why = "the router closed the connection before the Upgrade was sent";
        return out;
    }
    std::string head;
    if (!read_head(link, &head, &out.leftover)) {
        out.why = "the router sent no answer to the Upgrade";
        return out;
    }
    // "HTTP/1.1 101 Switching Protocols"
    const auto space = head.find(' ');
    if (!head.starts_with("HTTP/1.") || space == std::string::npos || head.size() < space + 4) {
        out.why = "the router's answer is not HTTP/1.1";
        return out;
    }
    (void)std::from_chars(head.data() + space + 1, head.data() + space + 4, out.status);
    const std::string headers = lower(head);
    const auto line_end       = head.find("\r\n");
    const std::string reason  = head.substr(space + 1, line_end - space - 1);
    if (out.status == 101) {
        const auto at = headers.find("\r\nupgrade:");
        std::string token;
        if (at != std::string::npos) {
            const auto start = at + 10;
            token = headers.substr(start, headers.find("\r\n", start) - start);
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t") + 1);
        }
        if (token != kUpgradeToken) {
            out.result = HandshakeResult::Failed;
            out.why    = "the router switched to \"" + token + "\", not " + std::string(kUpgradeToken);
            return out;
        }
        out.result = HandshakeResult::Switched;
        return out;
    }
    out.result = (out.status == 401 || out.status == 403) ? HandshakeResult::KeyRefused
                                                          : HandshakeResult::Refused;
    out.why    = "the router answered " + reason;
    return out;
}

} // namespace ninferno::connect
