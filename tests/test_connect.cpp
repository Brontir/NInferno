// ninferno-connect (#339, docs/connect.md), no GPU: a stand-in router that the worker dials over
// loopback (the Upgrade, then an HTTP/2 client on the switched connection) and a stand-in engine
// (httplib on loopback).
// - The registration carries the key and the engine's /props.
// - Streams run at once.
// - A body past the limit is answered 413.
// - A reset closes the engine's connection, both while the answer streams and before its first
//   byte.
// - Unanswered PINGs end the session, and the worker dials again.
// - After a GOAWAY a new connection comes while the old one finishes its answer.
// - A refused key stops the worker.
// - An answer the engine breaks off is reset, not ended as whole.
// - Over TLS: a router the worker trusts serves, 8 MiB whole to a router that reads late; an
//   untrusted certificate, or one for another name, is refused before anything is sent.
#include "connect/connector.h"
#include "connect/link.h"

#include <httplib.h>
#include <nghttp2/nghttp2.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <string>
#include <thread>
#include <tuple>

namespace {

namespace nc = ninferno::connect;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

constexpr std::size_t kBigBytes = 8u << 20;

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

// --- The stand-in engine ---------------------------------------------------------------------

struct Engine {
    httplib::Server server;
    int port = 0;
    std::thread thread;
    std::atomic<bool> b_arrived{false};
    std::atomic<bool> stream_saw_close{false};
    std::atomic<bool> hold_saw_close{false};

    Engine() {
        server.Get("/props", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(R"({"model":"m","max_context":4096})", "application/json");
        });
        server.Post("/echo", [](const httplib::Request& req, httplib::Response& res) {
            res.set_content(req.body, "text/plain");
        });
        // /a answers only once /b arrived (or after 5 s): both must be in flight together.
        server.Get("/a", [this](const httplib::Request&, httplib::Response& res) {
            const auto until = Clock::now() + 5s;
            while (!b_arrived.load() && Clock::now() < until) { std::this_thread::sleep_for(5ms); }
            res.set_content(b_arrived.load() ? "a:together" : "a:alone", "text/plain");
        });
        server.Get("/b", [this](const httplib::Request&, httplib::Response& res) {
            b_arrived.store(true);
            res.set_content("b", "text/plain");
        });
        // A tick every 20 ms until the client goes away.
        server.Get("/stream", [this](const httplib::Request&, httplib::Response& res) {
            res.set_chunked_content_provider("text/event-stream", [this](std::size_t,
                                                                         httplib::DataSink& sink) {
                std::this_thread::sleep_for(20ms);
                if (!sink.is_writable() || !sink.write("tick\n", 5)) {
                    stream_saw_close.store(true);
                    return false;
                }
                return true;
            });
        });
        // Five chunks 50 ms apart, then the end.
        server.Get("/finite", [](const httplib::Request&, httplib::Response& res) {
            auto sent = std::make_shared<int>(0);
            res.set_chunked_content_provider("text/plain", [sent](std::size_t,
                                                                   httplib::DataSink& sink) {
                if (*sent == 5) {
                    sink.done();
                    return true;
                }
                std::this_thread::sleep_for(50ms);
                const std::string chunk = "c" + std::to_string((*sent)++);
                return sink.write(chunk.data(), chunk.size());
            });
        });
        // A CORS preflight: 204, no body.
        server.Options("/pre", [](const httplib::Request&, httplib::Response& res) {
            res.status = 204;
            res.set_header("Access-Control-Allow-Origin", "*");
        });
        // 8 MiB in 64 KiB chunks, each byte its offset mod 251.
        server.Get("/big", [](const httplib::Request&, httplib::Response& res) {
            auto sent = std::make_shared<std::size_t>(0);
            res.set_chunked_content_provider("application/octet-stream", [sent](std::size_t,
                                                                                httplib::DataSink& sink) {
                if (*sent == kBigBytes) {
                    sink.done();
                    return true;
                }
                std::string chunk(64 * 1024, '\0');
                for (std::size_t i = 0; i < chunk.size(); ++i) {
                    chunk[i] = static_cast<char>((*sent + i) % 251);
                }
                *sent += chunk.size();
                return sink.write(chunk.data(), chunk.size());
            });
        });
        // Two chunks, then the engine "dies": the provider fails, the answer is cut off.
        server.Get("/die", [](const httplib::Request&, httplib::Response& res) {
            auto sent = std::make_shared<int>(0);
            res.set_chunked_content_provider("text/plain", [sent](std::size_t, httplib::DataSink& sink) {
                if ((*sent)++ == 2) { return false; }
                return sink.write("part", 4);
            });
        });
        // A "prefill": nothing is answered for up to 5 s, the connection watched meanwhile.
        server.Get("/hold", [this](const httplib::Request& req, httplib::Response& res) {
            const auto until = Clock::now() + 5s;
            while (Clock::now() < until) {
                if (req.is_connection_closed()) {
                    hold_saw_close.store(true);
                    return;
                }
                std::this_thread::sleep_for(10ms);
            }
            res.set_content("held", "text/plain");
        });
        port   = server.bind_to_any_port("127.0.0.1");
        thread = std::thread([this] { server.listen_after_bind(); });
        server.wait_until_ready();
    }
    ~Engine() {
        server.stop();
        thread.join();
    }
};

// --- The stand-in router ---------------------------------------------------------------------

struct Listener {
    int fd   = -1;
    int port = 0;
    Listener() {
        fd = ::socket(AF_INET, SOCK_STREAM, 0);
        const int one = 1;
        (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        (void)::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr);
        (void)::listen(fd, 16);
        socklen_t len = sizeof addr;
        (void)getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
        port = ntohs(addr.sin_port);
    }
    ~Listener() { ::close(fd); }
    // A dialed connection within `timeout`, or -1.
    [[nodiscard]] int accept(std::chrono::milliseconds timeout) const {
        pollfd p{fd, POLLIN, 0};
        if (::poll(&p, 1, static_cast<int>(timeout.count())) <= 0) { return -1; }
        return ::accept(fd, nullptr, nullptr);
    }
};

// The router's end of one dialed connection: plain TCP, or TLS over it.
struct Conn {
    int fd   = -1;
    SSL* ssl = nullptr;
    Conn(int f, SSL* s = nullptr) : fd(f), ssl(s) {} // NOLINT: a plain fd is a plain connection
    [[nodiscard]] bool wait(int ms) const {
        if (ssl && SSL_pending(ssl) > 0) { return true; }
        pollfd p{fd, POLLIN, 0};
        return ::poll(&p, 1, ms) > 0;
    }
    ssize_t recv(char* data, std::size_t size) const {
        return ssl ? SSL_read(ssl, data, static_cast<int>(size)) : ::recv(fd, data, size, 0);
    }
    ssize_t send(const char* data, std::size_t size) const {
        return ssl ? SSL_write(ssl, data, static_cast<int>(size))
                   : ::send(fd, data, size, MSG_NOSIGNAL);
    }
    void close() const {
        if (ssl) {
            (void)SSL_shutdown(ssl);
            SSL_free(ssl);
        }
        ::close(fd);
    }
};

struct Upgrade {
    std::string head; // lower-cased
    std::string body;
};

// The worker's Upgrade request: its head (to the blank line) and its Content-Length body.
bool read_upgrade(Conn conn, Upgrade* out) {
    std::string buffer;
    char chunk[4096];
    std::size_t end = std::string::npos;
    while ((end = buffer.find("\r\n\r\n")) == std::string::npos) {
        if (!conn.wait(3000)) { return false; }
        const ssize_t n = conn.recv(chunk, sizeof chunk);
        if (n <= 0) { return false; }
        buffer.append(chunk, static_cast<std::size_t>(n));
    }
    out->head = buffer.substr(0, end + 2);
    for (char& c : out->head) { c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }
    std::size_t length = 0;
    if (const auto at = out->head.find("content-length:"); at != std::string::npos) {
        length = std::stoul(out->head.substr(at + 15));
    }
    out->body = buffer.substr(end + 4);
    while (out->body.size() < length) {
        if (!conn.wait(3000)) { return false; }
        const ssize_t n = conn.recv(chunk, sizeof chunk);
        if (n <= 0) { return false; }
        out->body.append(chunk, static_cast<std::size_t>(n));
    }
    return true;
}

bool send_all(Conn conn, std::string_view data) {
    while (!data.empty()) {
        const ssize_t n = conn.send(data.data(), data.size());
        if (n <= 0) { return false; }
        data.remove_prefix(static_cast<std::size_t>(n));
    }
    return true;
}

bool switch_protocols(Conn conn) {
    return send_all(conn, "HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade\r\nUpgrade: "
                        "ninfer-reverse-h2\r\n\r\n");
}

struct H2Stream {
    int status = 0;
    std::string data;
    bool closed         = false;
    std::uint32_t error = 0; // the code it closed with (0: ended)
};

// The router's HTTP/2 client on a switched connection.
class H2Client {
public:
    explicit H2Client(Conn conn) : conn_(conn) {
        nghttp2_session_callbacks* cb = nullptr;
        nghttp2_session_callbacks_new(&cb);
        nghttp2_session_callbacks_set_on_header_callback(cb, &H2Client::on_header);
        nghttp2_session_callbacks_set_on_data_chunk_recv_callback(cb, &H2Client::on_data);
        nghttp2_session_callbacks_set_on_stream_close_callback(cb, &H2Client::on_close);
        nghttp2_session_client_new(&session_, cb, this);
        nghttp2_session_callbacks_del(cb);
        nghttp2_submit_settings(session_, NGHTTP2_FLAG_NONE, nullptr, 0);
    }
    ~H2Client() {
        nghttp2_session_del(session_);
        conn_.close();
    }
    H2Client(const H2Client&)            = delete;
    H2Client& operator=(const H2Client&) = delete;

    std::int32_t request(const std::string& method, const std::string& path,
                         const std::string& body = {}) {
        const auto nv = [](const char* n, const std::string& v) {
            return nghttp2_nv{reinterpret_cast<std::uint8_t*>(const_cast<char*>(n)),
                              reinterpret_cast<std::uint8_t*>(const_cast<char*>(v.data())),
                              std::strlen(n), v.size(), NGHTTP2_NV_FLAG_NONE};
        };
        static const std::string kScheme = "http", kAuthority = "worker";
        const nghttp2_nv nva[] = {nv(":method", method), nv(":scheme", kScheme),
                                  nv(":authority", kAuthority), nv(":path", path)};
        if (body.empty()) {
            const std::int32_t id = nghttp2_submit_request2(session_, nullptr, nva, 4, nullptr, nullptr);
            streams_[id];
            return id;
        }
        auto source = std::make_shared<std::pair<std::string, std::size_t>>(body, 0);
        nghttp2_data_provider2 provider{};
        provider.source.ptr    = source.get();
        provider.read_callback = [](nghttp2_session*, std::int32_t, std::uint8_t* buf,
                                    std::size_t length, std::uint32_t* flags,
                                    nghttp2_data_source* src, void*) -> nghttp2_ssize {
            auto& [text, off] = *static_cast<std::pair<std::string, std::size_t>*>(src->ptr);
            const std::size_t n = std::min(length, text.size() - off);
            std::memcpy(buf, text.data() + off, n);
            off += n;
            if (off == text.size()) { *flags |= NGHTTP2_DATA_FLAG_EOF; }
            return static_cast<nghttp2_ssize>(n);
        };
        const std::int32_t id = nghttp2_submit_request2(session_, nullptr, nva, 4, &provider, nullptr);
        bodies_[id] = source;
        streams_[id];
        return id;
    }
    void reset(std::int32_t id) { nghttp2_submit_rst_stream(session_, NGHTTP2_FLAG_NONE, id, NGHTTP2_CANCEL); }
    void goaway() {
        nghttp2_submit_goaway(session_, NGHTTP2_FLAG_NONE, 0, NGHTTP2_NO_ERROR, nullptr, 0);
    }
    // Exchanges frames until `done` or `timeout`; whether `done` holds.
    bool pump(std::chrono::milliseconds timeout, const std::function<bool()>& done = [] { return false; }) {
        const auto until = Clock::now() + timeout;
        char buffer[16384];
        for (;;) {
            const std::uint8_t* data = nullptr;
            nghttp2_ssize n          = 0;
            while ((n = nghttp2_session_mem_send2(session_, &data)) > 0) {
                if (!send_all(conn_, std::string_view(reinterpret_cast<const char*>(data),
                                                     static_cast<std::size_t>(n)))) {
                    return done();
                }
            }
            if (done()) { return true; }
            if (Clock::now() >= until) { return false; }
            if (conn_.wait(10)) {
                const ssize_t got = conn_.recv(buffer, sizeof buffer);
                if (got <= 0) { return done(); }
                if (nghttp2_session_mem_recv2(session_, reinterpret_cast<const std::uint8_t*>(buffer),
                                              static_cast<std::size_t>(got)) < 0) {
                    return done();
                }
            }
        }
    }
    [[nodiscard]] const H2Stream& stream(std::int32_t id) { return streams_[id]; }

private:
    static H2Client& self(void* p) { return *static_cast<H2Client*>(p); }
    static int on_header(nghttp2_session*, const nghttp2_frame* frame, const std::uint8_t* name,
                         std::size_t namelen, const std::uint8_t* value, std::size_t valuelen,
                         std::uint8_t, void* user) {
        if (frame->hd.type == NGHTTP2_HEADERS &&
            std::string_view(reinterpret_cast<const char*>(name), namelen) == ":status") {
            self(user).streams_[frame->hd.stream_id].status =
                std::stoi(std::string(reinterpret_cast<const char*>(value), valuelen));
        }
        return 0;
    }
    static int on_data(nghttp2_session*, std::uint8_t, std::int32_t id, const std::uint8_t* data,
                       std::size_t len, void* user) {
        self(user).streams_[id].data.append(reinterpret_cast<const char*>(data), len);
        return 0;
    }
    static int on_close(nghttp2_session*, std::int32_t id, std::uint32_t error, void* user) {
        self(user).streams_[id].closed = true;
        self(user).streams_[id].error  = error;
        return 0;
    }

    Conn conn_;
    nghttp2_session* session_ = nullptr;
    std::map<std::int32_t, H2Stream> streams_;
    std::map<std::int32_t, std::shared_ptr<std::pair<std::string, std::size_t>>> bodies_;
};

// --- The worker under test -------------------------------------------------------------------

struct Worker {
    std::atomic<bool> stop{false};
    std::future<int> rc;
    Worker(const Listener& router, const Engine& engine, std::chrono::milliseconds ping,
           std::size_t max_request_bytes = 64u << 20, const std::string& tls_ca_file = {},
           int connections = 1) {
        nc::ConnectorOptions o;
        std::string why;
        o.routers = {*nc::parse_router_url(
            (tls_ca_file.empty() ? "http://127.0.0.1:" : "https://localhost:") +
                std::to_string(router.port) + "/ninfer/connect",
            &why)};
        o.tls.ca_file               = tls_ca_file;
        o.key                       = "k-1";
        o.name                      = "box-1";
        o.connections               = connections;
        o.tunnel.engine             = {"127.0.0.1", engine.port};
        o.tunnel.ping_interval      = ping;
        o.tunnel.max_request_bytes  = max_request_bytes;
        o.min_backoff               = 50ms;
        o.max_backoff               = 200ms;
        o.reconnect_jitter          = 50ms;
        rc = std::async(std::launch::async, [this, o] {
            return nc::run_connector(o, "test", {}, stop);
        });
    }
    ~Worker() {
        stop.store(true);
        if (rc.valid() && rc.wait_for(10s) != std::future_status::ready) {
            std::fprintf(stderr, "FAIL: the worker did not stop within 10 s\n");
            std::_Exit(1); // the future's destructor would wait for it forever
        }
    }
};

// Accepts the worker's dial, checks nothing, switches: the router's side of one connection.
std::unique_ptr<H2Client> accept_switched(const Listener& router, Upgrade* upgrade = nullptr) {
    const int fd = router.accept(5s);
    if (fd < 0) { return nullptr; }
    Upgrade local;
    if (!read_upgrade(fd, upgrade ? upgrade : &local) || !switch_protocols(fd)) {
        ::close(fd);
        return nullptr;
    }
    return std::make_unique<H2Client>(fd);
}

// A self-signed P-256 certificate for `name` (its subject and its one DNS name), PEM in two files
// under the temporary directory; the certificate is also the worker's CA file.
struct Certificate {
    std::string cert_path;
    std::string key_path;
    explicit Certificate(const std::string& name) {
        const std::string base = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
                                 "/ninferno_connect_test_" + std::to_string(::getpid()) + "_" + name;
        cert_path = base + ".crt";
        key_path  = base + ".key";
        EVP_PKEY* key = EVP_EC_gen("P-256");
        X509* x509    = X509_new();
        ASN1_INTEGER_set(X509_get_serialNumber(x509), 1);
        X509_gmtime_adj(X509_getm_notBefore(x509), -60);
        X509_gmtime_adj(X509_getm_notAfter(x509), 3600);
        X509_set_pubkey(x509, key);
        X509_NAME* subject = X509_get_subject_name(x509);
        X509_NAME_add_entry_by_txt(subject, "CN", MBSTRING_ASC,
                                   reinterpret_cast<const unsigned char*>(name.c_str()), -1, -1, 0);
        X509_set_issuer_name(x509, subject);
        X509V3_CTX ctx;
        X509V3_set_ctx_nodb(&ctx);
        X509V3_set_ctx(&ctx, x509, x509, nullptr, nullptr, 0);
        const std::string san = "DNS:" + name;
        for (const auto& [nid, value] :
             {std::pair{NID_subject_alt_name, san.c_str()}, std::pair{NID_basic_constraints, "critical,CA:TRUE"}}) {
            X509_EXTENSION* ext = X509V3_EXT_conf_nid(nullptr, &ctx, nid, value);
            X509_add_ext(x509, ext, -1);
            X509_EXTENSION_free(ext);
        }
        X509_sign(x509, key, EVP_sha256());
        FILE* c = std::fopen(cert_path.c_str(), "w");
        PEM_write_X509(c, x509);
        std::fclose(c);
        FILE* k = std::fopen(key_path.c_str(), "w");
        PEM_write_PrivateKey(k, key, nullptr, nullptr, 0, nullptr, nullptr);
        std::fclose(k);
        X509_free(x509);
        EVP_PKEY_free(key);
    }
    ~Certificate() {
        std::remove(cert_path.c_str());
        std::remove(key_path.c_str());
    }
};

// The router's TLS accept of one dial (nullptr when the worker refused its certificate).
SSL* tls_accept(SSL_CTX* ctx, int fd) {
    SSL* ssl = SSL_new(ctx);
    SSL_set_fd(ssl, fd);
    if (SSL_accept(ssl) != 1) {
        SSL_free(ssl);
        return nullptr;
    }
    return ssl;
}

void test_tls() {
    const Certificate certificate("localhost");
    SSL_CTX* server = SSL_CTX_new(TLS_server_method());
    expect(SSL_CTX_use_certificate_file(server, certificate.cert_path.c_str(), SSL_FILETYPE_PEM) == 1 &&
               SSL_CTX_use_PrivateKey_file(server, certificate.key_path.c_str(), SSL_FILETYPE_PEM) == 1,
           "the stand-in router's certificate");
    Engine engine;
    {
        // Trusted, named: the worker dials over TLS and serves.
        Listener router;
        Worker worker(router, engine, 1000ms, 64u << 20, certificate.cert_path);
        const int fd = router.accept(5s);
        SSL* ssl     = fd >= 0 ? tls_accept(server, fd) : nullptr;
        expect(ssl != nullptr, "the worker's TLS handshake with a router it trusts");
        Upgrade up;
        if (ssl && read_upgrade(Conn{fd, ssl}, &up) && switch_protocols(Conn{fd, ssl})) {
            expect(up.head.find("authorization: bearer k-1") != std::string::npos,
                   "the key goes inside TLS");
            H2Client c(Conn{fd, ssl});
            const auto e = c.request("POST", "/echo", "over tls");
            expect(c.pump(5s, [&] { return c.stream(e).closed; }) && c.stream(e).data == "over tls",
                   "a request over the TLS tunnel");
            const auto big = c.request("GET", "/big");
            (void)c.pump(50ms);
            std::this_thread::sleep_for(500ms); // the router stops reading: the socket fills
            const bool whole = c.pump(20s, [&] { return c.stream(big).closed; });
            const std::string& got = c.stream(big).data;
            bool pattern = got.size() == kBigBytes;
            for (std::size_t i = 0; pattern && i < got.size(); i += 4093) {
                pattern = static_cast<unsigned char>(got[i]) == i % 251;
            }
            expect(whole && c.stream(big).error == 0 && pattern,
                   "8 MiB over TLS to a router that reads late, whole (" +
                       std::to_string(got.size()) + " bytes)");
        } else if (fd >= 0) {
            Conn{fd, ssl}.close();
        }
    }
    // A router whose certificate the worker does not trust, or that names another host: no dial.
    for (const auto& [host, ca, what] :
         {std::tuple{std::string("localhost"), std::string(), "a certificate no trusted CA signed"},
          std::tuple{std::string("127.0.0.1"), certificate.cert_path,
                     "a certificate for another name"}}) {
        Listener router;
        auto accepted = std::async(std::launch::async, [&] {
            const int fd = router.accept(5s);
            if (fd < 0) { return false; }
            SSL* ssl = tls_accept(server, fd);
            const bool ok = ssl != nullptr;
            Conn{fd, ssl}.close();
            return ok;
        });
        std::string why;
        auto url  = nc::parse_router_url("https://" + host + ":" + std::to_string(router.port), &why);
        auto link = nc::dial(*url, {ca}, 3000, &why);
        expect(link == nullptr && why.find("TLS handshake") != std::string::npos,
               std::string(what) + ": refused (" + why + ")");
        link.reset();
        expect(!accepted.get(), std::string(what) + ": the router saw no handshake complete");
    }
    SSL_CTX_free(server);
}

void test_urls() {
    std::string why;
    const auto a = nc::parse_router_url("https://router.example:7443/workers", &why);
    expect(a && a->tls && a->host == "router.example" && a->port == 7443 && a->path == "/workers",
           "https URL with port and path");
    const auto b = nc::parse_router_url("http://[::1]", &why);
    expect(b && !b->tls && b->host == "::1" && b->port == 80 && b->path == "/ninfer/connect",
           "http URL, IPv6, defaults");
    expect(!nc::parse_router_url("ftp://x", &why), "ftp refused");
    expect(!nc::parse_router_url("https://x:0", &why), "port 0 refused");
    expect(!nc::parse_router_url("https://:443", &why), "no host refused");
    expect(nc::is_loopback_host("127.0.0.2") && nc::is_loopback_host("localhost") &&
               !nc::is_loopback_host("10.0.0.1") && !nc::is_loopback_host("router.example"),
           "loopback hosts");
}

void test_registration_and_streams() {
    Engine engine;
    Listener router;
    Worker worker(router, engine, 1000ms, 1024);
    Upgrade upgrade;
    auto c = accept_switched(router, &upgrade);
    expect(c != nullptr, "the worker dialed and was switched");
    if (!c) { return; }
    expect(upgrade.head.starts_with("post /ninfer/connect http/1.1"), "the Upgrade's request line");
    expect(upgrade.head.find("authorization: bearer k-1") != std::string::npos, "the key");
    expect(upgrade.head.find("upgrade: ninfer-reverse-h2") != std::string::npos, "the token");
    const auto reg = nlohmann::json::parse(upgrade.body, nullptr, false);
    expect(!reg.is_discarded() && reg["props"]["model"] == "m" && reg["version"] == "test" &&
               reg["name"] == "box-1",
           "the registration carries the engine's /props: " + upgrade.body);

    const auto a = c->request("GET", "/a");
    const auto b = c->request("GET", "/b");
    const auto e = c->request("POST", "/echo", "hello");
    const auto big = c->request("POST", "/echo", std::string(2048, 'x'));
    const bool done = c->pump(8s, [&] {
        return c->stream(a).closed && c->stream(b).closed && c->stream(e).closed &&
               c->stream(big).closed;
    });
    expect(done, "four streams finished");
    expect(c->stream(a).status == 200 && c->stream(a).data == "a:together",
           "two requests in flight at once: " + c->stream(a).data);
    expect(c->stream(e).status == 200 && c->stream(e).data == "hello", "a POST body forwarded");
    expect(c->stream(big).status == 413, "a body past the limit: 413, got " +
                                             std::to_string(c->stream(big).status));
    const auto pre = c->request("OPTIONS", "/pre");
    expect(c->pump(5s, [&] { return c->stream(pre).closed; }) && c->stream(pre).status == 204 &&
               c->stream(pre).error == 0,
           "a body-less answer (a 204 preflight) passes as it is, got " +
               std::to_string(c->stream(pre).status));
    const auto cut = c->request("GET", "/die");
    expect(c->pump(5s, [&] { return c->stream(cut).closed; }) && c->stream(cut).error != 0 &&
               c->stream(cut).data == "partpart",
           "an answer the engine broke off is reset, not ended as whole (error " +
               std::to_string(c->stream(cut).error) + ", " + c->stream(cut).data + ")");
}

void test_reset_closes_the_engine_connection() {
    Engine engine;
    Listener router;
    Worker worker(router, engine, 1000ms);
    auto c = accept_switched(router);
    expect(c != nullptr, "dialed (reset)");
    if (!c) { return; }
    const auto s = c->request("GET", "/stream");
    expect(c->pump(5s, [&] { return !c->stream(s).data.empty(); }), "the stream's first ticks");
    c->reset(s);
    expect(c->pump(3s, [&] { return engine.stream_saw_close.load(); }),
           "a reset while the answer streams closes the engine's connection");
    const auto h = c->request("GET", "/hold");
    (void)c->pump(200ms);
    c->reset(h);
    expect(c->pump(3s, [&] { return engine.hold_saw_close.load(); }),
           "a reset before the answer's first byte closes the engine's connection");
}

void test_ping_timeout_dials_again() {
    Engine engine;
    Listener router;
    Worker worker(router, engine, 100ms);
    auto c = accept_switched(router);
    expect(c != nullptr, "dialed (ping)");
    if (!c) { return; }
    (void)c->pump(50ms); // settings exchanged; from now on nothing answers its PINGs
    const auto start = Clock::now();
    auto again       = accept_switched(router);
    const auto took  = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start);
    expect(again != nullptr && took < 3s,
           "two unanswered PINGs end the session and the worker dials again (" +
               std::to_string(took.count()) + " ms)");
}

void test_goaway_replacement_while_draining() {
    Engine engine;
    Listener router;
    Worker worker(router, engine, 1000ms);
    auto c = accept_switched(router);
    expect(c != nullptr, "dialed (goaway)");
    if (!c) { return; }
    const auto s = c->request("GET", "/finite");
    expect(c->pump(3s, [&] { return !c->stream(s).data.empty(); }), "the answer started");
    c->goaway();
    std::unique_ptr<H2Client> next;
    const auto until = Clock::now() + 3s;
    while (!next && Clock::now() < until) {
        (void)c->pump(20ms);
        const int fd = router.accept(10ms);
        if (fd >= 0) {
            Upgrade up;
            if (read_upgrade(fd, &up) && switch_protocols(fd)) {
                next = std::make_unique<H2Client>(fd);
            } else {
                ::close(fd);
            }
        }
    }
    expect(next != nullptr, "a new connection after the GOAWAY");
    expect(c->pump(3s, [&] { return c->stream(s).closed; }) && c->stream(s).status == 200 &&
               c->stream(s).data == "c0c1c2c3c4",
           "the old connection's answer finished whole: " + c->stream(s).data);
    if (next) {
        const auto e = next->request("POST", "/echo", "again");
        expect(next->pump(3s, [&] { return next->stream(e).closed; }) &&
                   next->stream(e).data == "again",
               "the new connection serves");
    }
}

void test_refused_key_stops() {
    Engine engine;
    Listener router;
    Worker worker(router, engine, 1000ms);
    const int fd = router.accept(5s);
    expect(fd >= 0, "dialed (refused key)");
    if (fd < 0) { return; }
    Upgrade up;
    (void)read_upgrade(fd, &up);
    (void)send_all(fd, "HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n");
    ::close(fd);
    expect(worker.rc.wait_for(3s) == std::future_status::ready && worker.rc.get() == 3,
           "a refused key ends the run with 3");
    expect(router.accept(300ms) < 0, "no dial after the refusal");
}

// A revoked key stops every connection to that router: one slot's live tunnel too, not only the
// slot the 401 reached.
void test_refused_key_stops_every_slot() {
    Engine engine;
    Listener router;
    Worker worker(router, engine, 1000ms, 64u << 20, {}, 2);
    auto live = accept_switched(router);
    expect(live != nullptr, "the first slot's tunnel");
    const int fd = router.accept(5s);
    expect(fd >= 0, "the second slot dialed");
    if (!live || fd < 0) { return; }
    Upgrade up;
    (void)read_upgrade(fd, &up);
    (void)send_all(fd, "HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n");
    ::close(fd);
    // The live tunnel drains (GOAWAY) and closes while the router keeps answering it.
    // (The worker closes the link before its run returns: the pump ends there, the run follows.)
    (void)live->pump(5s, [&] { return worker.rc.wait_for(0s) == std::future_status::ready; });
    const bool ended = worker.rc.wait_for(3s) == std::future_status::ready;
    expect(ended && worker.rc.get() == 3, "a refused key ends every slot of the router, the run with 3");
}

} // namespace

int main() {
    test_urls();
    test_registration_and_streams();
    test_reset_closes_the_engine_connection();
    test_ping_timeout_dials_again();
    test_goaway_replacement_while_draining();
    test_refused_key_stops();
    test_refused_key_stops_every_slot();
    test_tls();
    if (failures != 0) {
        std::fprintf(stderr, "connect: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("connect: registration, streams, resets, ping, GOAWAY, a refused key and TLS\n");
    return 0;
}
