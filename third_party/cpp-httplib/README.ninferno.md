# cpp-httplib in NInferno

`httplib.h` is yhirose/cpp-httplib v0.65.0 (MIT, `LICENSE`), the HTTP server under the whole API
(`src/serve/http_server.cpp`, `http_transport.cpp`).

Local change, marked "NInferno patch": `Response::keep_alive_after_error` and its use where the
server decides `Connection: close`. An error response the application rendered after the request
was read completely keeps the connection, so a refused caller's next request reuses it
(`src/serve/http_server.cpp` sets it); every other error response still closes it.

Local change, marked "NInferno patch": `Server::set_request_body_min_rate`. The request body is read
in windows, and one that grew by less than the minimum rate over a window stops the read and is
answered 408 (`--request-body-min-kib-per-sec`). Upstream bounds the headers as a whole and each wait
for a body byte, but not a body that trickles in. The bytes counted are those the reader takes,
chunk-size lines and trailers included.

Local change, marked "NInferno patch": `Server::set_overload_handler`. It is called on the accepting
thread with a connection the task queue refused (it was full), before the socket is closed, so the
server can answer it instead of dropping it silently (`answer_overloaded_connection`).

Both patches are in httplib's own formatting; CI's clang-format check skips `third_party/`.

Since v0.65.0 upstream closes the connection after any response carrying `Connection: close`
(RFC 9112 §9.6), which v0.54.1 needed a second local change for.

To update: replace `httplib.h` and `LICENSE` from the new tag, re-apply the marked changes, and run
`ninferno_http_server_test` and the serve tests.
