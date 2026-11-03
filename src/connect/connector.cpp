#include "connect/connector.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <condition_variable>
#include <list>
#include <optional>
#include <memory>
#include <mutex>
#include <random>
#include <thread>

namespace ninferno::connect {
namespace {

using Clock = std::chrono::steady_clock;
// How long a session lives before its end no longer counts as a failed dial.
constexpr std::chrono::seconds kSettled{10};

// Sleeps up to `duration`, waking early on stop. False when stopped.
bool sleep_unless_stopped(std::chrono::milliseconds duration, const std::atomic<bool>& stop) {
    const auto until = Clock::now() + duration;
    while (Clock::now() < until) {
        if (stop.load()) { return false; }
        std::this_thread::sleep_for(std::min<Clock::duration>(std::chrono::milliseconds(50),
                                                              until - Clock::now()));
    }
    return !stop.load();
}

std::chrono::milliseconds jittered(std::chrono::milliseconds base, std::chrono::milliseconds jitter,
                                   std::mt19937_64& rng) {
    if (jitter.count() <= 0) { return base; }
    std::uniform_int_distribution<long long> extra(0, jitter.count());
    return base + std::chrono::milliseconds(extra(rng));
}

// The engine's manifest, or nullopt when it does not answer.
std::optional<std::string> fetch_props(const EngineAddress& engine) {
    httplib::Client client(engine.host, engine.port);
    client.set_connection_timeout(std::chrono::seconds(5));
    client.set_read_timeout(std::chrono::seconds(30));
    const auto result = client.Get("/props");
    if (!result || result->status != 200) { return std::nullopt; }
    return result->body;
}

// One session's end, and the GOAWAY that lets the slot dial its replacement before it.
struct SessionSignal {
    std::mutex mu;
    std::condition_variable changed;
    bool goaway = false;
    bool ended  = false;
    TunnelEnd end = TunnelEnd::Closed;
};

enum class SlotEnd : std::uint8_t { Stopped, KeyRefused };

SlotEnd run_slot(const ConnectorOptions& options, const RouterUrl& router, int index,
                 const std::string& version, const ConnectorLog& log,
                 const std::atomic<bool>& stop) {
    const auto say = [&](const std::string& line) {
        if (log.line) { log.line(router.display() + " #" + std::to_string(index) + ": " + line); }
    };
    std::mt19937_64 rng(std::random_device{}() ^ (static_cast<std::uint64_t>(index) << 32));
    std::list<std::pair<std::thread, std::shared_ptr<SessionSignal>>> draining;
    const auto reap = [&](bool all) {
        for (auto it = draining.begin(); it != draining.end();) {
            bool ended = false;
            {
                std::lock_guard lock(it->second->mu);
                ended = it->second->ended;
            }
            if (ended || all) {
                it->first.join();
                it = draining.erase(it);
            } else {
                ++it;
            }
        }
    };
    auto backoff = options.min_backoff;
    SlotEnd result = SlotEnd::Stopped;
    bool props_missing_said = false;
    while (!stop.load()) {
        reap(false);
        const auto props = fetch_props(options.tunnel.engine);
        if (!props) {
            if (!props_missing_said) {
                say("the engine at " + options.tunnel.engine.host + ":" +
                    std::to_string(options.tunnel.engine.port) + " does not answer /props; waiting");
                props_missing_said = true;
            }
            if (!sleep_unless_stopped(options.min_backoff, stop)) { break; }
            continue;
        }
        props_missing_said = false;
        std::string why;
        auto link = dial(router, options.tls, static_cast<int>(options.dial_timeout.count()), &why);
        Handshake hs;
        if (link) {
            hs = handshake(*link, router, options.key, registration_json(*props, options.name, version));
            why = hs.why;
        }
        if (!link || hs.result != HandshakeResult::Switched) {
            if (link && hs.result == HandshakeResult::KeyRefused) {
                say("the router refused the key (" + why + "); not dialing it again");
                result = SlotEnd::KeyRefused;
                break;
            }
            const auto wait = jittered(backoff, backoff / 2, rng);
            say(why + "; again in " + std::to_string(wait.count()) + " ms");
            backoff = std::min(options.max_backoff, backoff * 2);
            if (!sleep_unless_stopped(wait, stop)) { break; }
            continue;
        }
        say("connected");
        const auto connected_at = Clock::now();
        auto signal = std::make_shared<SessionSignal>();
        std::thread session([&options, &stop, signal, link = std::move(link),
                             leftover = std::move(hs.leftover)]() mutable {
            TunnelEvents events;
            events.on_goaway = [signal] {
                std::lock_guard lock(signal->mu);
                signal->goaway = true;
                signal->changed.notify_all();
            };
            const TunnelEnd end = serve_tunnel(*link, std::move(leftover), options.tunnel, events, stop);
            link.reset();
            std::lock_guard lock(signal->mu);
            signal->ended = true;
            signal->end   = end;
            signal->changed.notify_all();
        });
        bool goaway = false;
        TunnelEnd end = TunnelEnd::Closed;
        {
            std::unique_lock lock(signal->mu);
            signal->changed.wait(lock, [&] { return signal->goaway || signal->ended; });
            goaway = signal->goaway && !signal->ended;
            end    = signal->end;
        }
        if (goaway) {
            backoff = options.min_backoff;
            // The old session drains in its thread; the replacement comes after a jitter.
            draining.emplace_back(std::move(session), signal);
            const auto wait = jittered(std::chrono::milliseconds(0), options.reconnect_jitter, rng);
            say("the router sent GOAWAY; draining, a new connection in " + std::to_string(wait.count()) + " ms");
            if (!sleep_unless_stopped(wait, stop)) { break; }
            continue;
        }
        session.join();
        if (stop.load()) { break; }
        // A session that lived (kSettled) dials again at once, with jitter; one that died young (a
        // router that switches and then closes, or breaks HTTP/2) backs off as a refused dial does.
        std::chrono::milliseconds wait{0};
        if (Clock::now() - connected_at >= kSettled) {
            backoff = options.min_backoff;
            wait    = jittered(std::chrono::milliseconds(0), options.reconnect_jitter / 5, rng);
        } else {
            wait    = jittered(backoff, backoff / 2, rng);
            backoff = std::min(options.max_backoff, backoff * 2);
        }
        say(std::string("session ended: ") + tunnel_end_name(end) + "; again in " +
            std::to_string(wait.count()) + " ms");
        if (!sleep_unless_stopped(wait, stop)) { break; }
    }
    reap(true);
    return result;
}

} // namespace

std::string registration_json(const std::string& props_json, const std::string& name,
                              const std::string& version) {
    nlohmann::ordered_json body;
    body["props"] = nlohmann::json::parse(props_json, nullptr, false);
    if (body["props"].is_discarded()) { body["props"] = nullptr; }
    body["version"] = version;
    if (!name.empty()) { body["name"] = name; }
    return body.dump();
}

int run_connector(const ConnectorOptions& options, const std::string& version,
                  const ConnectorLog& log, const std::atomic<bool>& stop) {
    std::vector<std::thread> slots;
    std::vector<int> refused(options.routers.size(), 0);
    std::mutex mu;
    for (std::size_t r = 0; r < options.routers.size(); ++r) {
        for (int i = 0; i < std::max(1, options.connections); ++i) {
            slots.emplace_back([&, r, i] {
                if (run_slot(options, options.routers[r], i, version, log, stop) ==
                    SlotEnd::KeyRefused) {
                    std::lock_guard lock(mu);
                    ++refused[r];
                }
            });
        }
    }
    for (auto& slot : slots) { slot.join(); }
    const bool all_refused =
        !options.routers.empty() &&
        std::all_of(refused.begin(), refused.end(), [](int n) { return n > 0; });
    return all_refused ? 3 : 0;
}

} // namespace ninferno::connect
