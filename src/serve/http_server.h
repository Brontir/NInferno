#pragma once

#include "serve/generation_service.h"
#include "serve/operational_log.h"
#include "serve/openai_responses_store.h"
#include "serve/request_log.h"
#include "serve/serve_options.h"

#include <httplib.h>

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

namespace ninferno::serve {

void write_openai_error(httplib::Response& response, const ApiError& error);
void write_anthropic_error(httplib::Response& response, const ApiError& error,
                           const std::string& request_id);

// The --cors preflight answer (204): it allows the request headers the browser asks for, as browser
// SDKs add their own (OpenAI's and Anthropic's x-stainless-*, Anthropic's
// anthropic-dangerous-direct-browser-access), and a public page's Private Network Access request.
void answer_cors_preflight(const httplib::Request& request, httplib::Response& response);

// cpp-httplib invokes the error handler for every application response with status >= 400. Only
// an empty 413 is its own pre-routing payload-limit rejection; application-authored errors must be
// left untouched.
httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response);

[[nodiscard]] bool matches_bearer_credential(std::string_view authorization,
                                             std::string_view api_key) noexcept;
// The presented value against the configured key, in time that does not depend on how much of it
// matches. An empty key matches nothing.
[[nodiscard]] bool matches_api_key(std::string_view presented, std::string_view api_key) noexcept;

// Anthropic SDKs append /v1/<endpoint> to their base URL, so a client given the OpenAI-style base
// URL that already ends in /v1 requests /v1/v1/<endpoint>. Every API route also answers there:
// api_route_pattern("/messages") matches /v1/messages and /v1/v1/messages without adding a capture
// group, and canonical_api_path maps the doubled form back to the /v1/<endpoint> it names.
[[nodiscard]] std::string api_route_pattern(std::string_view endpoint);
[[nodiscard]] std::string_view canonical_api_path(std::string_view path) noexcept;
// The endpoint without its version prefix (/v1/chat/completions, /v1/v1/... and /chat/completions
// are one endpoint: /chat/completions).
[[nodiscard]] std::string_view api_endpoint(std::string_view path) noexcept;

// A browser names the page's origin (Origin) on every cross-site request and on a DNS-rebound one.
// Without --cors no browser page is meant to call the server, yet a "simple request" (a text/plain
// or form body, which needs no preflight) would run a generation for any page its user visits
// such requests are refused. /health stays open.
[[nodiscard]] bool refuses_browser_origin(const httplib::Request& request, bool cors_enabled);

class HttpServer {
public:
    HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger);

    // Reserves the configured address before model loading. The service is attached only after its
    // Engine is ready, then listen() enters the blocking accept loop on the already-bound socket.
    bool bind();
    void attach(GenerationService& service);
    bool listen();
    void stop();
    // Async-signal-safe (SIGTERM/SIGINT). The first call starts a drain: /health and new requests
    // answer 503 while requests in flight finish, for at most ServeOptions::drain_timeout_ms, and
    // then the server stops; a second call stops at once.
    void request_drain() noexcept;

    // An Engine-wide failure is permanent: the server then drains and stops on its own, and the
    // process exits with a failure status so that its supervisor starts a new one.
    [[nodiscard]] bool engine_failed() const noexcept { return engine_failed_.load(); }

    [[nodiscard]] const std::string& public_model_id() const noexcept { return public_model_id_; }

private:
    class RequestLifecycle {
    public:
        RequestLifecycle(HttpServer& owner, RequestLogContext context);
        ~RequestLifecycle();
        RequestLifecycle(const RequestLifecycle&)            = delete;
        RequestLifecycle& operator=(const RequestLifecycle&) = delete;

        void done(const GenerationOutcome& outcome);
        void failure(const RequestFailure& failure);
        void response_failure(const RequestFailure& failure);
        // The answer went out unstored: it alone is more than the response store's capacity.
        void response_not_stored(std::size_t store_capacity_bytes);

        [[nodiscard]] std::uint64_t request_id() const noexcept { return context_.id; }

    private:
        enum class State : std::uint8_t {
            Pending,
            Done,
            Error,
        };

        [[nodiscard]] bool claim(State terminal) noexcept;

        HttpServer* owner_ = nullptr;
        RequestLogContext context_;
        std::atomic<State> state_{State::Pending};
    };

    // A generation's lifecycle, its log lines naming the response's request id.
    [[nodiscard]] std::shared_ptr<RequestLifecycle>
    begin_request(RequestLogContext context, const httplib::Response& response);
    // A client error (status below 500, or any ApiError the HTTP layer raises): logged under the
    // response's request id, then written in the endpoint's shape.
    void refuse(const httplib::Request& request, httplib::Response& response,
                const ApiError& error) const;

    void register_routes();
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_messages(const httplib::Request& req, httplib::Response& res);
    void handle_count_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_responses(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_response_get(const httplib::Request& req, httplib::Response& res);
    void handle_response_delete(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_items(const httplib::Request& req, httplib::Response& res);
    void handle_response_cancel(const httplib::Request& req, httplib::Response& res);
    void handle_response_compact(const httplib::Request& req, httplib::Response& res);
    void handle_models(const httplib::Request& req, httplib::Response& res) const;
    void handle_model(const httplib::Request& req, httplib::Response& res) const;

    void record_request_start(const RequestLogContext& context);
    void record_request_rejected(const RequestRejectionLogContext& context);
    void record_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void record_request_failure(const RequestLogContext& context, const RequestFailure& failure);
    void record_response_failure(std::uint64_t request_id, const RequestFailure& failure);
    void record_throughput(const ThroughputReport& report);
    void run_stats_reporter();
    void stop_stats_reporter();
    // Watches executor liveness for /health and carries out a requested drain.
    void run_supervisor();
    void stop_supervisor();
    [[nodiscard]] bool reject_while_draining(const httplib::Request& req, httplib::Response& res);
    // The server-side limits every request parser applies, from the serve options.
    [[nodiscard]] RequestLimits request_limits() const noexcept;

    GenerationService* service_ = nullptr;
    ServeOptions options_;
    std::string public_model_id_;
    std::int64_t model_created_ = 0;       // the model object's `created`: when the model loaded
    std::string sampling_defaults_header_; // x-ninferno-sampling-defaults
    ninferno::ModelMetadata model_metadata_;
    OpenAIResponsesStore openai_responses_store_;
    OperationalLog operational_log_;
    JsonlRequestLog request_jsonl_;
    httplib::Server server_;
    std::atomic<std::uint64_t> request_seq_{0};
    std::mutex stats_mutex_;
    std::condition_variable stats_cv_;
    std::thread stats_thread_;
    bool stats_stopping_ = false;
    // Generation requests from parsing to their last byte (a RequestLifecycle's lifetime, which a
    // stream's content provider holds until the stream ends).
    std::atomic<std::uint32_t> active_requests_{0};
    std::atomic<bool> drain_requested_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> draining_{false};
    std::atomic<bool> stalled_{false};
    std::atomic<bool> engine_failed_{false};
    std::atomic<bool> supervisor_stopping_{false};
    std::thread supervisor_thread_;
};

} // namespace ninferno::serve
