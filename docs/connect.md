# Serving a router from behind NAT: ninferno-connect

`ninferno-connect` lets an engine with no public address serve a router. This covers a worker
behind NAT, or on a rented machine with no inbound ports. Instead of the router calling the
worker, the worker dials the router. The router then sends its requests back over that same
connection.

A VPN such as WireGuard is still the first answer where it works. The worker joins a hub, gets a
stable private address, and the router sees an ordinary URL; no code is needed on either side.
`ninferno-connect` is for the places where a VPN does not work: UDP is closed, nobody administers
the keys, or the worker has to join on its own.

```text
router <--HTTP/2 over the worker's connection, roles reversed--> ninferno-connect <--HTTP on localhost--> ninferno-serve
```

The engine does not change. `ninferno-serve` keeps listening on localhost, and `ninferno-connect`
runs as a separate program beside it:

- if the tunnel crashes, the model stays loaded;
- updating the tunnel does not unload the model;
- the tunnel also works in front of an older engine.

The cost is one extra localhost hop per request.

## Running it

```bash
ninferno-serve model.ninferno --host 127.0.0.1 --port 8080
ninferno-connect --router https://router.example:7443 --key-file worker.key
```

| Option | Meaning |
|---|---|
| `--router URL` | `https://host[:port][/path]` of a router's worker port. Repeat it for several routers; the worker keeps connections to each. |
| `--key-file PATH` | this worker's key, sent as `Authorization: Bearer <key>` |
| `--engine URL` | the engine, `http://127.0.0.1:8080` by default |
| `--connections N` | connections to each router, 2 by default. With several, one lost TCP packet stalls only one connection's streams. |
| `--ca-file PATH` | the CA that the routers' certificates are checked against. The default is the system's trust store. The certificate must name the router's host. |
| `--name NAME` | this worker's name in its registration |
| `--ping-seconds S` | a PING every S seconds, 15 by default. When two go unanswered the worker drops the connection and dials again. |
| `--max-streams N` | requests one connection carries at once, 64 by default |
| `--allow-plaintext` | allow an `http://` router that is not on this machine. Without it, plain HTTP is accepted only to a loopback address. |

SIGINT or SIGTERM sends each router a GOAWAY. The running answers then finish and the program
exits. It exits with 3 when every router refused the key.

`ninferno-connect` is built where pkg-config finds libnghttp2 1.60 or later and CMake finds
OpenSSL 3, which on Debian and Ubuntu means `libnghttp2-dev` and `libssl-dev`. It is not built on
Windows. Without those libraries the tree builds as before, just without this program.

## The protocol (for a router)

### 1. The worker dials and asks to switch

The worker opens a TLS connection to the router's worker port. This must be a port of its own,
with its own keys and limits, and never the router's `/v1`. The worker then sends one HTTP/1.1
request:

```http
POST /ninfer/connect HTTP/1.1
Host: router.example:7443
Connection: Upgrade
Upgrade: ninfer-reverse-h2
Authorization: Bearer <the worker's key>
Content-Type: application/json
Content-Length: <n>

{"props": <the engine's GET /props>, "version": "<ninferno-connect's version>", "name": "<--name>"}
```

The body is the registration. `props` is what the engine's `/props` serves, so a worker can appear
without any change to the router's configuration. Revoking the key removes the worker.

The router answers in one of three ways:

- **`HTTP/1.1 101 Switching Protocols` with `Upgrade: ninfer-reverse-h2`:** the connection is
  switched. The router is now the HTTP/2 **client**: it sends the client connection preface and its
  SETTINGS. The worker is the HTTP/2 **server**.
- **401 or 403:** the key is refused, and the worker does not dial that router again.
- **Anything else:** the worker tries again after a backoff of 1 s, doubling up to 60 s, with
  jitter.

### 2. Requests on the switched connection

Each request is one HTTP/2 stream. Inside it are the engine's ordinary endpoints (`/v1/...`,
`/props`, `/health`), so cache hints (#338) are just headers. The worker forwards each stream to the
engine as its own HTTP/1.1 request, and streams the engine's answer back on that stream.

| Need | How |
|---|---|
| many requests at once | one stream per request; the worker announces `SETTINGS_MAX_CONCURRENT_STREAMS` (`--max-streams`) |
| a slow client does not stall the others | HTTP/2 flow control per stream. Each stream buffers at most 1 MiB ahead of the router, after which the engine's connection waits. |
| the client went away | the router sends `RST_STREAM`. The worker closes that request's engine connection, which cancels the generation the same way a closed connection does, during prefill too. |
| a dead link is noticed | the worker sends a `PING` every `--ping-seconds`. After two unanswered it drops the connection and dials again. The router's own PINGs are answered. |
| a router rollout without a reconnect storm | the router sends `GOAWAY`. The worker takes no new streams on that connection, lets the running answers finish, and meanwhile dials a replacement after a random 0–5 s. |
| the link is lost mid-answer | the answers it carried fail, as when the engine is killed |
| a request body that is too large | answered 413 by the worker; the limit is 64 MiB |
| the engine does not answer | answered 502 by the worker. While the engine's `/props` does not answer, the worker does not dial at all. |

### What it does not solve

- **A remote worker sees the prompts and can misreport tokens.** This is a question of trust, not
  of transport. A router needs a trust class per worker and spot checks of `usage`.
- **Narrow links.** A 2 MB prompt at 20 Mbit/s takes 0.8 s before prefill starts, so a router
  should add the transfer time to its estimate.
