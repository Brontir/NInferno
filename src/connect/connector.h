#pragma once

// ninferno-connect's whole run (#339, docs/connect.md): `connections` reversed connections to each
// router, each one kept up:
// - before each dial the engine's /props is read (the worker's manifest: the router learns the
//   model and its limits from it, as from a worker it calls), and a worker whose engine does not
//   answer does not dial;
// - a refused dial or Upgrade is tried again after a backoff that doubles to max_backoff, with
//   jitter so that many workers do not dial together;
// - a key the router refuses (401, 403) stops that router's connections: the key was revoked;
// - a lost link or a ping timeout dials again at once (with jitter);
// - the router's GOAWAY dials the replacement after a jitter while the old session drains, so a
//   router rollout neither drops the answers running nor brings every worker back at once.

#include "connect/link.h"
#include "connect/tunnel.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace ninferno::connect {

struct ConnectorOptions {
    std::vector<RouterUrl> routers;
    std::string key;
    std::string name; // the worker's name in its registration (empty: none)
    TlsOptions tls;
    TunnelOptions tunnel;
    int connections = 2; // to each router
    std::chrono::milliseconds dial_timeout{10000};
    std::chrono::milliseconds min_backoff{1000};
    std::chrono::milliseconds max_backoff{60000};
    std::chrono::milliseconds reconnect_jitter{5000}; // after a GOAWAY or a lost link
};

// What a run says as it goes (the binary logs it; the tests watch it).
struct ConnectorLog {
    std::function<void(const std::string&)> line;
};

// The registration body: {"props": <the engine's /props>, "version": .., "name": ..}.
[[nodiscard]] std::string registration_json(const std::string& props_json, const std::string& name,
                                            const std::string& version);

// Runs until `stop` (each session drains first) or until every router refused the key: 0 after a
// stop, 3 when every router refused the key.
[[nodiscard]] int run_connector(const ConnectorOptions& options, const std::string& version,
                                const ConnectorLog& log, const std::atomic<bool>& stop);

} // namespace ninferno::connect
