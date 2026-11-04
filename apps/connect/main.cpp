// ninferno-connect: lets an engine with no public address serve a router by dialing out (#339,
// docs/connect.md). It runs beside ninferno-serve, which keeps listening on localhost: the worker
// dials each router, the router switches the connection to HTTP/2 with the roles reversed and sends
// its requests over it, and each request is forwarded to the engine as plain HTTP. A crash or an
// update of this program does not touch the engine and its loaded model.
//
//   ninferno-connect --router https://router.example:7443 --key-file worker.key
//                    [--engine http://127.0.0.1:8080] [--connections 2] [--ca-file ca.pem]
//                    [--name gpu-box-3] [--ping-seconds 15] [--max-streams 64]
//                    [--allow-plaintext]
//
// SIGINT or SIGTERM: each session tells its router GOAWAY and its running answers finish.

#include "connect/connector.h"
#include "core/command_line.h"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>

#ifndef NINFERNO_VERSION_STRING
#define NINFERNO_VERSION_STRING "dev"
#endif

namespace {

std::atomic<bool> g_stop{false};
void handle_signal(int) { g_stop.store(true); }

constexpr const char* kUsage =
    "usage: ninferno-connect --router URL [--router URL ...] --key-file PATH [options]\n"
    "\n"
    "Dials each router and serves its requests over the connection, forwarding them to the local\n"
    "engine (ninferno-serve). For a worker no router can reach: behind NAT, or on a machine with\n"
    "no inbound ports.\n"
    "\n"
    "  --router URL       https://host[:port][/path] of a router's worker port (repeatable)\n"
    "  --key-file PATH    this worker's key, sent as Authorization: Bearer <key>\n"
    "  --engine URL       the engine, http://127.0.0.1:8080 by default\n"
    "  --connections N    connections to each router, 2 by default\n"
    "  --ca-file PATH     the CA the routers' certificates are checked against (default: the\n"
    "                     system's trust store)\n"
    "  --name NAME        this worker's name in its registration\n"
    "  --ping-seconds S   a PING every S seconds, 15 by default; two unanswered end the session\n"
    "  --max-streams N    requests one connection carries at once, 64 by default\n"
    "  --allow-plaintext  allow an http:// router that is not on this machine (the key and the\n"
    "                     prompts then cross the network unencrypted)\n";

int fail(const std::string& message) {
    std::cerr << "ninferno-connect: " << message << "\n";
    return 2;
}

bool parse_int(std::string_view text, int min, int max, int* out) {
    try {
        std::size_t used = 0;
        const int value  = std::stoi(std::string(text), &used);
        if (used != text.size() || value < min || value > max) { return false; }
        *out = value;
        return true;
    } catch (const std::exception&) { return false; }
}

} // namespace

int main(int argc, char** argv) {
    namespace nc = ninferno::connect;
    if (ninferno::version_requested(argc, argv)) {
        std::cout << "ninferno-connect " << NINFERNO_VERSION_STRING << "\n";
        return 0;
    }
    if (ninferno::help_requested(argc, argv)) {
        std::cout << kUsage;
        return 0;
    }
    nc::ConnectorOptions options;
    std::string key_file;
    std::string engine_url = "http://127.0.0.1:8080";
    bool allow_plaintext   = false;
    int ping_seconds       = 15;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto value = [&]() -> const char* {
            if (i + 1 >= argc) { return nullptr; }
            return argv[++i];
        };
        if (arg == "--allow-plaintext") {
            allow_plaintext = true;
            continue;
        }
        const char* v = value();
        if (!v) { return fail(std::string(arg) + " needs a value"); }
        std::string why;
        if (arg == "--router") {
            auto url = nc::parse_router_url(v, &why);
            if (!url) { return fail(std::string("--router ") + v + ": " + why); }
            options.routers.push_back(std::move(*url));
        } else if (arg == "--key-file") {
            key_file = v;
        } else if (arg == "--engine") {
            engine_url = v;
        } else if (arg == "--connections") {
            if (!parse_int(v, 1, 64, &options.connections)) { return fail("--connections is 1..64"); }
        } else if (arg == "--ca-file") {
            options.tls.ca_file = v;
        } else if (arg == "--name") {
            options.name = v;
        } else if (arg == "--ping-seconds") {
            if (!parse_int(v, 1, 3600, &ping_seconds)) { return fail("--ping-seconds is 1..3600"); }
        } else if (arg == "--max-streams") {
            if (!parse_int(v, 1, 1024, &options.tunnel.max_concurrent_streams)) {
                return fail("--max-streams is 1..1024");
            }
        } else {
            return fail("unknown option " + std::string(arg) + " (see --help)");
        }
    }
    if (options.routers.empty()) { return fail("at least one --router is needed (see --help)"); }
    if (key_file.empty()) { return fail("--key-file is needed"); }
    {
        std::ifstream in(key_file);
        if (!in) { return fail("cannot read --key-file " + key_file); }
        std::stringstream text;
        text << in.rdbuf();
        options.key = text.str();
        options.key.erase(options.key.find_last_not_of(" \t\r\n") + 1);
        options.key.erase(0, options.key.find_first_not_of(" \t\r\n"));
        if (options.key.empty()) { return fail("--key-file " + key_file + " is empty"); }
    }
    for (const auto& router : options.routers) {
        if (!router.tls && !allow_plaintext && !nc::is_loopback_host(router.host)) {
            return fail(router.display() +
                        " is plain HTTP off this machine: use https://, or --allow-plaintext");
        }
    }
    {
        std::string why;
        const auto engine = nc::parse_router_url(engine_url, &why);
        if (!engine || engine->tls) {
            return fail("--engine is http://host:port (the engine on this machine or its network)");
        }
        options.tunnel.engine = {engine->host, engine->port};
    }
    options.tunnel.ping_interval = std::chrono::seconds(ping_seconds);

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    std::signal(SIGPIPE, SIG_IGN);
    std::mutex log_mu;
    nc::ConnectorLog log{[&](const std::string& line) {
        std::lock_guard lock(log_mu);
        std::cerr << "ninferno-connect: " << line << std::endl;
    }};
    const int rc = nc::run_connector(options, NINFERNO_VERSION_STRING, log, g_stop);
    if (rc == 3) { std::cerr << "ninferno-connect: every router refused the key\n"; }
    return rc;
}
