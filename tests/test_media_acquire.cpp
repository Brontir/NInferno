// A media URL's failures are classified for the HTTP status the request gets: the URL's own
// server answering 4xx is the request's input (RemoteRejected, 400), one answering 5xx is an
// upstream failure (RemoteUnavailable, 502); a 200 answer returns its bytes. A media path stays
// under --media-root.

#include "product/media_acquire/acquire.h"

#include <httplib.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#endif

#include "support/posix.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <thread>

namespace {

namespace acquire = ninferno::product::media_acquire;

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

std::optional<acquire::ErrorKind> failure(const std::string& url) {
    acquire::Policy policy;
    policy.allow_private_network = true;
    policy.deadline              = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    try {
        (void)acquire::acquire_bytes(
            acquire::Source{.kind = acquire::SourceKind::Url, .value = url}, policy);
    } catch (const acquire::Error& error) { return error.kind(); }
    return std::nullopt;
}

bool private_address(const char* text) {
    sockaddr_in6 v6{};
    if (inet_pton(AF_INET6, text, &v6.sin6_addr) == 1) {
        v6.sin6_family = AF_INET6;
        return acquire::private_network_address(reinterpret_cast<const sockaddr*>(&v6));
    }
    sockaddr_in v4{};
    if (inet_pton(AF_INET, text, &v4.sin_addr) != 1) { return false; }
    v4.sin_family = AF_INET;
    return acquire::private_network_address(reinterpret_cast<const sockaddr*>(&v4));
}

} // namespace

// --media-root: a file under the root is read, a name that only starts with ".." included; a path
// that leaves the root, through ".." or absolute, is refused.
int media_root_paths() {
    namespace fs = std::filesystem;
    const fs::path base =
        fs::temp_directory_path() / ("ninferno_media_root_" + std::to_string(::getpid()));
    const fs::path root = base / "root";
    fs::create_directories(root);
    std::ofstream(root / "..notes", std::ios::binary) << "in";
    std::ofstream(base / "outside.bin", std::ios::binary) << "out";

    acquire::Policy policy;
    policy.media_root = root;
    policy.deadline   = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    const auto read   = [&](const fs::path& path) {
        return acquire::acquire_bytes(
            acquire::Source{.kind = acquire::SourceKind::Path, .value = path.string()}, policy);
    };
    const auto outside = [&](const fs::path& path) {
        try {
            (void)read(path);
        } catch (const std::invalid_argument& error) {
            return std::string(error.what()).find("outside configured media root") !=
                   std::string::npos;
        }
        return false;
    };
    int failures     = 0;
    const auto bytes = read(root / "..notes");
    failures += check(std::string(bytes.begin(), bytes.end()) == "in",
                      "a file named ..notes under the media root was not read");
    failures += check(outside(root / ".." / "outside.bin"), "a ../ path left the media root");
    failures += check(outside(fs::absolute(base / "outside.bin")),
                      "an absolute path outside the media root was read");
    fs::remove_all(base);
    return failures;
}

int main() {
    httplib::Server server;
    server.Get("/missing",
               [](const httplib::Request&, httplib::Response& res) { res.status = 404; });
    server.Get("/broken",
               [](const httplib::Request&, httplib::Response& res) { res.status = 503; });
    server.Get("/nowhere",
               [](const httplib::Request&, httplib::Response& res) { res.status = 302; });
    server.Get("/ok", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("abc", "image/png");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    std::thread thread([&] { server.listen_after_bind(); });
    server.wait_until_ready();
    const std::string base = "http://127.0.0.1:" + std::to_string(port);

    int failures = media_root_paths();
    failures += check(failure(base + "/missing") == acquire::ErrorKind::RemoteRejected,
                      "an HTTP 404 media URL is the request's input");
    failures += check(failure(base + "/nowhere") == acquire::ErrorKind::RemoteRejected,
                      "a redirect without a location is the request's input");
    failures += check(failure(base + "/broken") == acquire::ErrorKind::RemoteUnavailable,
                      "an HTTP 503 media URL is an upstream failure");
    acquire::Policy policy;
    policy.allow_private_network = true;
    policy.deadline              = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    const auto bytes             = acquire::acquire_bytes(
        acquire::Source{.kind = acquire::SourceKind::Url, .value = base + "/ok"}, policy);
    failures += check(std::string(bytes.begin(), bytes.end()) == "abc",
                      "an HTTP 200 media URL returns its bytes");

    // The guard refuses every form that reaches a private IPv4 address and nothing public.
    for (const char* address : {"127.0.0.1",
                                "10.1.2.3",
                                "172.16.0.1",
                                "192.168.1.1",
                                "169.254.169.254",
                                "100.64.0.1",
                                "192.0.0.8",
                                "192.0.2.1",
                                "198.51.100.7",
                                "203.0.113.9",
                                "192.88.99.1",
                                "::1",
                                "fe80::1",
                                "fec0::1",
                                "fd00::1",
                                "::ffff:127.0.0.1",
                                "::127.0.0.1",
                                "64:ff9b::7f00:1",
                                "64:ff9b::a9fe:a9fe",
                                "64:ff9b:1::1",
                                "2002:7f00:1::",
                                "2002:a9fe:a9fe::1",
                                "2001:0:4136:e378::1"}) {
        failures += check(private_address(address), std::string(address) + " is refused");
    }
    for (const char* address : {"8.8.8.8", "1.1.1.1", "2606:4700:4700::1111", "64:ff9b::808:808",
                                "2002:808:808::1", "2a00:1450:4001::1"}) {
        failures += check(!private_address(address), std::string(address) + " is allowed");
    }

    server.stop();
    thread.join();
    if (failures == 0) { std::cout << "media acquire: URL failures classified\n"; }
    return failures == 0 ? 0 : 1;
}
