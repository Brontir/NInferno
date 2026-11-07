# cpp-httplib in NInferno

`httplib.h` is yhirose/cpp-httplib v0.65.0 (MIT, `LICENSE`), the HTTP server under the whole API
(`src/serve/http_server.cpp`, `http_transport.cpp`).

Local change, marked "NInferno patch": `Response::keep_alive_after_error` and its use where the
server decides `Connection: close`. An error response the application rendered after the request
was read completely keeps the connection, so a refused caller's next request reuses it
(`src/serve/http_server.cpp` sets it); every other error response still closes it.

Since v0.65.0 upstream closes the connection after any response carrying `Connection: close`
(RFC 9112 §9.6), which v0.54.1 needed a second local change for.

To update: replace `httplib.h` and `LICENSE` from the new tag, re-apply the marked change, and run
`ninferno_http_server_test` and the serve tests.
