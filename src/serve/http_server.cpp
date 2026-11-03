#include "serve/http_server.h"
#include "serve/cache_hint.h"

#include "serve/anthropic_messages.h"
#include "serve/http_transport.h"
#include "serve/openai_common.h"
#include "serve/request_log.h"
#include "serve/request_validation.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <utility>

namespace ninferno::serve {
namespace {

bool is_anthropic_path(std::string_view path) {
    return api_endpoint(path).starts_with("/messages");
}

// The x-ninferno-sampling-defaults value: the source of the model's presets, then what the server's
// own flags override (request fields override both, as the client knows).
std::string sampling_defaults_header(const ninferno::ModelSamplingDefaults& defaults,
                                     const ServeOptions& options) {
    std::string out = defaults.source.empty() ? std::string("unspecified") : defaults.source;
    const auto& overrides = options.sampling_overrides;
    std::string named;
    const auto add = [&](bool present, const char* name) {
        if (present) { named += (named.empty() ? "" : ", ") + std::string(name); }
    };
    add(overrides.temperature.has_value(), "temperature");
    add(overrides.top_k.has_value(), "top_k");
    add(overrides.top_p.has_value(), "top_p");
    add(overrides.min_p.has_value(), "min_p");
    add(overrides.presence_penalty.has_value(), "presence_penalty");
    add(overrides.frequency_penalty.has_value(), "frequency_penalty");
    add(overrides.seed.has_value(), "seed");
    if (!named.empty()) { out += "; server flags: " + named; }
    if (options.greedy) { out += "; server --greedy: temperature 0"; }
    return out;
}

bool is_generation_path(std::string_view request_path) {
    const std::string_view path = api_endpoint(request_path);
    return path == "/chat/completions" || path == "/responses" || path == "/messages";
}

// The OpenAI-compatible routes, with or without their /v1 prefix (api_route_pattern serves both).
bool is_openai_path(std::string_view path) {
    if (is_anthropic_path(path)) { return false; }
    if (path.starts_with("/v1/")) { return true; }
    const std::string_view endpoint = api_endpoint(path);
    return endpoint.starts_with("/chat/") || endpoint.starts_with("/models") ||
           endpoint.starts_with("/responses");
}

// A client's own X-Request-Id names the request when it is a plain token (as vLLM takes it), so
// its logs and ours share it; else the server mints one.
bool usable_client_request_id(std::string_view id) noexcept {
    if (id.empty() || id.size() > 128) { return false; }
    return std::all_of(id.begin(), id.end(), [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '-' || c == '_' ||
               c == '.' || c == ':';
    });
}

void ensure_openai_request_id(const httplib::Request& request, httplib::Response& response) {
    if (!is_openai_path(request.path) || response.has_header("x-request-id")) { return; }
    const std::string client = request.get_header_value("x-request-id");
    response.set_header("x-request-id",
                        usable_client_request_id(client) ? client : new_openai_request_id());
}

// The id of every response (x-request-id on the OpenAI paths, request-id on Anthropic's), set once
// so that the log lines and the error a client reads name the same request.
void ensure_request_id(const httplib::Request& request, httplib::Response& response) {
    ensure_openai_request_id(request, response);
    if (is_anthropic_path(request.path) && !response.has_header("request-id")) {
        response.set_header("request-id", new_anthropic_request_id());
    }
}

ThroughputReport make_throughput_report(const ninferno::RuntimeStats& previous,
                                        const ninferno::RuntimeStats& current,
                                        double interval_seconds) {
    return ThroughputReport{
        .interval_seconds = interval_seconds,
        .computed_prefill_tokens =
            current.computed_prefill_tokens - previous.computed_prefill_tokens,
        .committed_decode_tokens =
            current.committed_decode_tokens - previous.committed_decode_tokens,
        .decode_rounds     = current.decode_rounds - previous.decode_rounds,
        .decode_row_rounds = current.decode_row_rounds - previous.decode_row_rounds,
        .previous          = previous,
        .current           = current,
    };
}

bool report_has_activity(const ThroughputReport& report) {
    return report.computed_prefill_tokens != 0 || report.committed_decode_tokens != 0 ||
           report.decode_rounds != 0 || report.current.running_requests != 0 ||
           report.current.waiting_requests != 0 || report.current.materializing_requests != 0 ||
           report.current.capture_pending_requests != 0 ||
           report.current.terminal_pending_requests != 0 ||
           report.current.active_captures_completed != report.previous.active_captures_completed ||
           report.current.active_captures_aborted != report.previous.active_captures_aborted ||
           report.current.root_selections != report.previous.root_selections ||
           report.current.private_endpoint_selections !=
               report.previous.private_endpoint_selections ||
           report.current.private_turn_closure_selections !=
               report.previous.private_turn_closure_selections ||
           report.current.private_response_replay_selections !=
               report.previous.private_response_replay_selections ||
           report.current.private_long_anchor_selections !=
               report.previous.private_long_anchor_selections ||
           report.current.shared_stable_prefix_selections !=
               report.previous.shared_stable_prefix_selections ||
           report.current.state_moves != report.previous.state_moves ||
           report.current.state_forks != report.previous.state_forks ||
           report.current.state_restores != report.previous.state_restores ||
           report.current.state_d2h_count != report.previous.state_d2h_count ||
           report.current.state_h2d_count != report.previous.state_h2d_count ||
           report.current.state_d2d_count != report.previous.state_d2d_count ||
           report.current.main_kv_d2h_pages != report.previous.main_kv_d2h_pages ||
           report.current.main_kv_h2d_pages != report.previous.main_kv_h2d_pages ||
           report.current.main_kv_d2d_pages != report.previous.main_kv_d2d_pages ||
           report.current.backend_kv_d2h_pages != report.previous.backend_kv_d2h_pages ||
           report.current.backend_kv_h2d_pages != report.previous.backend_kv_h2d_pages ||
           report.current.backend_kv_d2d_pages != report.previous.backend_kv_d2d_pages ||
           report.current.pressure_spill_pages != report.previous.pressure_spill_pages ||
           report.current.partial_tail_cow_pages != report.previous.partial_tail_cow_pages ||
           report.current.pressure_private_owners_degraded !=
               report.previous.pressure_private_owners_degraded ||
           report.current.pressure_private_owners_evicted !=
               report.previous.pressure_private_owners_evicted ||
           report.current.pressure_shared_owners_degraded !=
               report.previous.pressure_shared_owners_degraded ||
           report.current.pressure_shared_owners_evicted !=
               report.previous.pressure_shared_owners_evicted ||
           report.current.pressure_checkpoints_dropped !=
               report.previous.pressure_checkpoints_dropped ||
           report.current.pressure_searches != report.previous.pressure_searches ||
           report.current.pressure_search_budget_exhaustions !=
               report.previous.pressure_search_budget_exhaustions ||
           report.current.pressure_maximal_fallback_selections !=
               report.previous.pressure_maximal_fallback_selections ||
           report.current.historical_fork_hits != report.previous.historical_fork_hits ||
           report.current.device_state_occupied_slots !=
               report.previous.device_state_occupied_slots ||
           report.current.host_state_occupied_slots != report.previous.host_state_occupied_slots ||
           report.current.device_main_kv_occupied_pages !=
               report.previous.device_main_kv_occupied_pages ||
           report.current.device_backend_kv_occupied_pages !=
               report.previous.device_backend_kv_occupied_pages ||
           report.current.host_kv_occupied_bytes != report.previous.host_kv_occupied_bytes ||
           report.current.shared_active_references != report.previous.shared_active_references ||
           report.current.host_work.engine_boundary_ns !=
               report.previous.host_work.engine_boundary_ns ||
           report.current.host_work.program_submit_ns !=
               report.previous.host_work.program_submit_ns ||
           report.current.host_work.program_post_ns != report.previous.host_work.program_post_ns ||
           report.current.host_work.engine_commit_output_ns !=
               report.previous.host_work.engine_commit_output_ns ||
           report.current.host_work.engine_maintenance_ns !=
               report.previous.host_work.engine_maintenance_ns ||
           report.current.host_work.device_wait_ns != report.previous.host_work.device_wait_ns;
}

const char* endpoint_name(std::string_view request_path) noexcept {
    const std::string_view path = canonical_api_path(request_path);
    if (path == "/v1/chat/completions") { return "openai_chat_completions"; }
    if (path == "/v1/responses") { return "openai_responses"; }
    if (path == "/v1/responses/input_tokens") { return "openai_responses_input_tokens"; }
    if (path == "/v1/messages") { return "anthropic_messages"; }
    if (path == "/v1/messages/count_tokens") { return "anthropic_count_tokens"; }
    return "http_route";
}

std::string response_request_id(const httplib::Response& response) {
    if (response.has_header("x-request-id")) { return response.get_header_value("x-request-id"); }
    if (response.has_header("request-id")) { return response.get_header_value("request-id"); }
    return {};
}

// A full queue (429, Anthropic 529) or an admission that timed out (503) says when to retry, as
// /health and draining do: SDK clients back off by it instead of guessing.
// A full queue clears in about a second; an unavailable server (an admission timeout, /health
// while starting or failed, draining) in a few.
constexpr const char* kRetryAfterOverloadSeconds    = "1";
constexpr const char* kRetryAfterUnavailableSeconds = "5";

void set_retry_after(httplib::Response& response, int status) {
    if (response.has_header("Retry-After")) { return; }
    if (status == 429 || status == 529) {
        response.set_header("Retry-After", kRetryAfterOverloadSeconds);
    } else if (status == 503 || status == 504) { // 504: the request timed out in the queue
        response.set_header("Retry-After", kRetryAfterUnavailableSeconds);
    }
}

} // namespace

// A handler's refusal comes after the request was read whole: the caller's next request may reuse
// the connection, unless the server is draining (it closes every socket). Every refusal before
// routing sets it again after writing (a request that may carry an unread body closes, #305).
void write_openai_error(httplib::Response& response, const ApiError& error) {
    response.keep_alive_after_error = error.code != "server_draining";
    response.status                 = error.status;
    set_retry_after(response, error.status);
    response.set_content(make_error_body(error), "application/json");
}

void write_anthropic_error(httplib::Response& response, const ApiError& api_error,
                           const std::string& request_id) {
    const ApiError error            = normalize_anthropic_error(api_error);
    response.keep_alive_after_error = api_error.code != "server_draining";
    response.status                 = error.status;
    set_retry_after(response, error.status);
    response.headers.erase("request-id");
    response.set_header("request-id", request_id);
    response.set_content(make_anthropic_error_body(error, request_id), "application/json");
}

namespace {

// Whether the request may have a body the server has not read: a refusal before routing (or the
// library's own, such as 413) leaves it unread, and its bytes would be read as the next request,
// so the connection closes after such a refusal. A request without a body keeps it (#305).
bool may_carry_body(const httplib::Request& req) {
    if (req.has_header("Transfer-Encoding")) { return true; }
    const std::string length = req.get_header_value("Content-Length");
    // A malformed length counts as a body (the library refuses it and nothing is read).
    return !length.empty() && (length.find_first_not_of('0') != std::string::npos);
}

// An error in the shape the request's endpoint speaks, under the id the response already carries:
// a handler that set its request-id (and logged under it) keeps it, as the exception handler's
// log line does (ensure_request_id).
void write_error_for(const httplib::Request& request, httplib::Response& response,
                     const ApiError& error) {
    ensure_request_id(request, response);
    if (is_anthropic_path(request.path)) {
        write_anthropic_error(response, error, response.get_header_value("request-id"));
    } else {
        write_openai_error(response, error);
    }
}

} // namespace

httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response) {
    ensure_openai_request_id(request, response);
    if (!response.body.empty()) { return httplib::Server::HandlerResponse::Unhandled; }

    ApiError error;
    if (response.status == 415) {
        // The HTTP library refuses a Content-Encoding it cannot decode, before any handler.
        error.status  = 415;
        error.type    = "invalid_request_error";
        error.code    = "unsupported_content_encoding";
        error.message = "request body Content-Encoding '" +
                        ascii_preview(request.get_header_value("Content-Encoding")) +
                        "' is not supported; send the body uncompressed";
    } else if (response.status == 413) {
        error.status  = 413;
        error.type    = "invalid_request_error";
        error.code    = "request_too_large";
        error.message = "request body exceeds the configured payload limit of " +
                        std::to_string(options.max_request_bytes) + " bytes";
    } else if (response.status == 404 && is_anthropic_path(request.path)) {
        error.status  = 404;
        error.code    = "not_found";
        error.message = "requested Anthropic resource was not found";
    } else if (response.status == 404) {
        // A JSON body for every unknown path: clients parse the error, an empty 404 reads as a
        // broken server.
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "not_found";
        error.message = "no endpoint " + ascii_preview(request.method) + " " +
                        ascii_preview(request.path) + " (OpenAI /v1/chat/completions, " +
                        "/v1/responses, /v1/models; Anthropic /v1/messages)";
    } else {
        return httplib::Server::HandlerResponse::Unhandled;
    }
    // The library's own error carries no Anthropic request-id of this request: one already on
    // the response is stale, and write_error_for names a fresh one.
    response.headers.erase("request-id");
    write_error_for(request, response, error);
    response.keep_alive_after_error = !may_carry_body(request);
    return httplib::Server::HandlerResponse::Handled;
}

std::string api_route_pattern(std::string_view endpoint) {
    return "(?:/v1){0,2}" + std::string(endpoint);
}

std::string_view canonical_api_path(std::string_view path) noexcept {
    constexpr std::string_view kVersion = "/v1";
    if (path.starts_with("/v1/v1/")) { path.remove_prefix(kVersion.size()); }
    return path;
}

// The endpoint without its version prefix: /v1/chat/completions, /v1/v1/chat/completions and
// /chat/completions are one endpoint.
std::string_view api_endpoint(std::string_view path) noexcept {
    path = canonical_api_path(path);
    if (path.starts_with("/v1/")) { path.remove_prefix(3); }
    return path;
}

bool refuses_browser_origin(const httplib::Request& request, bool cors_enabled) {
    return !cors_enabled && request.path != "/health" && request.has_header("Origin");
}

bool matches_bearer_credential(std::string_view authorization, std::string_view api_key) noexcept {
    if (api_key.empty()) { return false; }
    const auto is_whitespace = [](char value) { return value == ' ' || value == '\t'; };
    const auto ascii_equal   = [](char lhs, char rhs) {
        if (lhs >= 'A' && lhs <= 'Z') { lhs = static_cast<char>(lhs - 'A' + 'a'); }
        if (rhs >= 'A' && rhs <= 'Z') { rhs = static_cast<char>(rhs - 'A' + 'a'); }
        return lhs == rhs;
    };

    std::size_t position = 0;
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    constexpr std::string_view scheme = "Bearer";
    if (authorization.size() - position < scheme.size()) { return false; }
    for (std::size_t index = 0; index < scheme.size(); ++index) {
        if (!ascii_equal(authorization[position + index], scheme[index])) { return false; }
    }
    position += scheme.size();
    if (position == authorization.size() || !is_whitespace(authorization[position])) {
        return false;
    }
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    std::size_t end = authorization.size();
    while (end > position && is_whitespace(authorization[end - 1])) { --end; }
    return matches_api_key(authorization.substr(position, end - position), api_key);
}

bool matches_api_key(std::string_view presented, std::string_view api_key) noexcept {
    if (api_key.empty()) { return false; }
    // Every byte of the key is compared, whatever was presented, so the time a guess takes does
    // not tell how long a prefix of it was right.
    unsigned difference = presented.size() == api_key.size() ? 0U : 1U;
    for (std::size_t index = 0; index < api_key.size(); ++index) {
        const auto presented_byte =
            index < presented.size() ? static_cast<unsigned char>(presented[index]) : 0U;
        difference |= presented_byte ^ static_cast<unsigned char>(api_key[index]);
    }
    return difference == 0;
}

HttpServer::HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger)
    : options_(std::move(options)), openai_responses_store_(options_.response_store_max_records,
                                                            options_.response_store_max_bytes),
      operational_log_(logger),
      request_jsonl_(options_.request_log_jsonl, options_.artifact_path, std::move(logger)) {
    const HttpWorkerPlan workers =
        plan_http_workers(options_.max_concurrency, options_.max_pending_requests);
    server_.new_task_queue = [workers] {
        return new httplib::ThreadPool(workers.threads, workers.threads, workers.max_queued);
    };
    // Streamed tokens leave as small SSE writes, one per decode round: without TCP_NODELAY Nagle holds
    // a write behind the previous one's ACK (delayed up to 40 ms) on a busy loopback or LAN client
    // (ninfer-all sets it too).
    server_.set_tcp_nodelay(true);
    server_.set_socket_options(configure_http_server_socket);
    server_.set_payload_max_length(options_.max_request_bytes);
    register_routes();
}

HttpServer::RequestLifecycle::RequestLifecycle(HttpServer& owner, RequestLogContext context)
    : owner_(&owner), context_(std::move(context)) {
    // Counted once the start is logged: a throwing log runs no destructor to take it back.
    owner_->record_request_start(context_);
    owner_->active_requests_.fetch_add(1, std::memory_order_acq_rel);
}

HttpServer::RequestLifecycle::~RequestLifecycle() {
    owner_->active_requests_.fetch_sub(1, std::memory_order_acq_rel);
}

bool HttpServer::RequestLifecycle::claim(State terminal) noexcept {
    State expected = State::Pending;
    return state_.compare_exchange_strong(expected, terminal, std::memory_order_acq_rel);
}

void HttpServer::RequestLifecycle::done(const GenerationOutcome& outcome) {
    if (claim(State::Done)) { owner_->record_request_done(context_, outcome); }
}

void HttpServer::RequestLifecycle::failure(const RequestFailure& failure) {
    if (claim(State::Error)) { owner_->record_request_failure(context_, failure); }
}

void HttpServer::RequestLifecycle::response_failure(const RequestFailure& failure) {
    owner_->record_response_failure(context_.id, failure);
}

void HttpServer::RequestLifecycle::response_not_stored(std::size_t store_capacity_bytes) {
    owner_->operational_log_.response_not_stored(context_.id, store_capacity_bytes);
}

bool HttpServer::stream_admitted(
    PreparedRequest& prepared, const httplib::Request& request, httplib::Response& response,
    RequestLifecycle& lifecycle,
    const std::function<void(httplib::Response&, const ApiError&)>& write_error) {
    const auto disconnected = [&request] { return client_disconnected(request); };
    try {
        if (service_->await_admission(prepared, disconnected)) { return true; }
        // It ended before it ran: its wait reports why (a queue timeout throws). One that returns
        // was cancelled: by a client that went away (no one to answer), or by the server giving
        // in-flight work up (a drain), answered as an unstreamed request is.
        StreamSink discarded;
        (void)service_->run(prepared, &discarded, disconnected);
        if (client_disconnected(request)) {
            lifecycle.failure(make_client_disconnected_failure(RequestFailurePhase::Transport));
        } else {
            const ApiError error = service_->aborting()
                                       ? server_draining_error()
                                       : internal_error("the request ended before it ran");
            lifecycle.failure(make_request_failure(RequestFailurePhase::Generation, error));
            write_error(response, error);
        }
    } catch (const ApiException& exception) {
        lifecycle.failure(make_generation_request_failure(exception.error()));
        write_error(response, exception.error());
    } catch (const std::exception& exception) {
        lifecycle.failure(
            make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
        write_error(response, internal_error(exception.what()));
    }
    return false;
}

std::shared_ptr<HttpServer::RequestLifecycle>
HttpServer::begin_request(RequestLogContext context, const httplib::Response& response) {
    context.http_request_id = response_request_id(response);
    return std::make_shared<RequestLifecycle>(*this, std::move(context));
}

void HttpServer::refuse(const httplib::Request& request, httplib::Response& response,
                        const ApiError& error) const {
    ensure_request_id(request, response);
    operational_log_.http_failure(endpoint_name(request.path),
                                  make_request_failure(RequestFailurePhase::Http, error),
                                  response_request_id(response));
    write_error_for(request, response, error);
}

void HttpServer::record_request_start(const RequestLogContext& context) {
    request_jsonl_.write_request_start(context);
    operational_log_.request_start(context);
}

void HttpServer::record_request_rejected(const RequestRejectionLogContext& context) {
    request_jsonl_.write_request_rejected(context);
    operational_log_.request_rejected(context);
}

void HttpServer::record_request_done(const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
    request_jsonl_.write_request_done(context, outcome);
    operational_log_.request_done(context, outcome);
}

void HttpServer::record_request_failure(const RequestLogContext& context,
                                        const RequestFailure& failure) {
    request_jsonl_.write_request_error(context, failure.machine_message);
    operational_log_.request_failure(context, failure);
}

void HttpServer::record_response_failure(std::uint64_t request_id, const RequestFailure& failure) {
    operational_log_.response_failure(request_id, failure);
}

void HttpServer::record_throughput(const ThroughputReport& report) {
    request_jsonl_.write_throughput(report);
    operational_log_.throughput(report);
}

void HttpServer::run_stats_reporter() {
    using Clock                     = std::chrono::steady_clock;
    ninferno::RuntimeStats previous   = service_->runtime_stats();
    Clock::time_point previous_time = Clock::now();
    const auto interval             = std::chrono::milliseconds(options_.log_stats_interval_ms);
    Clock::time_point next_deadline = previous_time + interval;

    for (;;) {
        {
            std::unique_lock lock(stats_mutex_);
            if (stats_cv_.wait_until(lock, next_deadline, [this] { return stats_stopping_; })) {
                break;
            }
        }

        const ninferno::RuntimeStats current = service_->runtime_stats();
        const Clock::time_point now        = Clock::now();
        const ThroughputReport report      = make_throughput_report(
            previous, current, std::chrono::duration<double>(now - previous_time).count());
        if (report_has_activity(report)) { record_throughput(report); }
        previous      = current;
        previous_time = now;
        next_deadline += interval;
        const Clock::time_point after_write = Clock::now();
        if (next_deadline <= after_write) { next_deadline = after_write + interval; }
    }

    const ninferno::RuntimeStats current = service_->runtime_stats();
    const Clock::time_point now        = Clock::now();
    const ThroughputReport tail        = make_throughput_report(
        previous, current, std::chrono::duration<double>(now - previous_time).count());
    // The exact partial interval remains useful to measurement consumers. Pretty throughput is a
    // fixed-cadence operational record and deliberately has no irregular shutdown tail.
    if (report_has_activity(tail)) { request_jsonl_.write_throughput(tail); }
}

void HttpServer::stop_stats_reporter() {
    if (!stats_thread_.joinable()) { return; }
    {
        std::lock_guard lock(stats_mutex_);
        stats_stopping_ = true;
    }
    stats_cv_.notify_one();
    stats_thread_.join();
}

void answer_cors_preflight(const httplib::Request& request, httplib::Response& response) {
    const std::string requested = request.get_header_value("Access-Control-Request-Headers");
    if (!requested.empty()) {
        response.headers.erase("Access-Control-Allow-Headers");
        response.set_header("Access-Control-Allow-Headers", requested);
    }
    if (request.get_header_value("Access-Control-Request-Private-Network") == "true") {
        response.set_header("Access-Control-Allow-Private-Network", "true");
    }
    response.set_header("Access-Control-Max-Age", "600");
    response.status = 204;
}

void HttpServer::register_routes() {
    server_.set_error_handler([this](const httplib::Request& request, httplib::Response& response) {
        const auto handled = handle_unrendered_http_error(options_, request, response);
        // Every error status passes here before the library decides the connection: while the
        // server drains, an error of a permitted GET (a missing stored response) closes too.
        if (draining_.load()) { response.keep_alive_after_error = false; }
        return handled;
    });
    if (options_.enable_cors) {
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Expose-Headers",
              "x-request-id, request-id, x-ninferno-sampling-defaults, x-ninferno-hint-applied"},
             {"Access-Control-Allow-Headers",
              "Authorization, Content-Type, X-API-Key, anthropic-version, anthropic-beta, "
              "anthropic-user-profile-id, x-ninferno-cache-hint"},
             {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"}});
        // CORS preflight: browsers send OPTIONS with no credentials before the real
        // request; answer it without auth so the actual GET/POST can carry the key.
        server_.Options(R"(.*)", answer_cors_preflight);
    }

    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        ensure_openai_request_id(req, res);
        // Where the sampling a request does not name comes from (the model's generation config,
        // transformers' defaults, the family's presets), on every generation response.
        if (service_ != nullptr && req.method == "POST" && is_generation_path(req.path)) {
            res.set_header("x-ninferno-sampling-defaults", sampling_defaults_header_);
        }
        if (reject_while_draining(req, res)) {
            res.keep_alive_after_error = false; // the server stops: no connection is kept
            return httplib::Server::HandlerResponse::Handled;
        }
        if (refuses_browser_origin(req, options_.enable_cors)) {
            ApiError error;
            error.status  = 403;
            error.type    = "invalid_request_error";
            error.code    = "cross_origin_request";
            error.message = "a browser request from origin '" +
                            ascii_preview(req.get_header_value("Origin")) +
                            "' is refused; start the server with --cors to serve browser pages";
            refuse(req, res, error);
            res.keep_alive_after_error = !may_carry_body(req);
            return httplib::Server::HandlerResponse::Handled;
        }
        if (options_.api_key.empty() || req.path == "/health" || req.method == "OPTIONS") {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        // Accept both the OpenAI-style bearer token and the Anthropic-style
        // x-api-key header so OpenAI clients and Claude Code (ANTHROPIC_API_KEY
        // -> x-api-key, ANTHROPIC_AUTH_TOKEN -> Authorization: Bearer) both work.
        const bool bearer_ok =
            matches_bearer_credential(req.get_header_value("Authorization"), options_.api_key);
        const bool x_api_key_ok =
            matches_api_key(req.get_header_value("x-api-key"), options_.api_key);
        if (!bearer_ok && !x_api_key_ok) {
            ApiError error;
            error.status  = 401;
            error.type    = "invalid_request_error";
            error.code    = "invalid_api_key";
            error.message = "missing or invalid API key";
            refuse(req, res, error); // in the shape the target endpoint speaks
            res.keep_alive_after_error = !may_carry_body(req);
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    server_.set_exception_handler(
        [this](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
            ensure_request_id(req, res);
            try {
                std::rethrow_exception(ep);
            } catch (const ApiException& e) {
                refuse(req, res, e.error());
            } catch (const std::exception& e) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, e.what()),
                    response_request_id(res));
                write_error_for(req, res, internal_error(e));
                // A failure inside the library (an allocation while the body was being read) may
                // leave the body half read: a request that could carry one closes the connection.
                res.keep_alive_after_error = !may_carry_body(req);
            } catch (...) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, "unknown error"),
                    response_request_id(res));
                ApiError error = internal_error("unknown error");
                write_error_for(req, res, error);
                res.keep_alive_after_error = !may_carry_body(req);
            }
        });

    // /health is truthful: 503 when the Engine failed or is stopping, while the
    // server drains after SIGTERM, and while the executor holds work without progress; the body
    // carries the load gauges.
    server_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        const bool available = service_ != nullptr && service_->is_available();
        const char* status   = !available                 ? "unavailable"
                               : draining_.load()         ? "draining"
                               : stalled_.load()          ? "stalled"
                                                          : "ok";
        res.status           = std::string_view(status) == "ok" ? 200 : 503;
        nlohmann::json body{{"status", status},
                            {"active_requests", active_requests_.load()}};
        if (service_ != nullptr) {
            const ninferno::RuntimeStats stats = service_->runtime_stats();
            body["running_requests"]         = stats.running_requests;
            body["waiting_requests"]         = stats.waiting_requests;
            body["materializing_requests"]   = stats.materializing_requests;
        }
        if (res.status != 200) { res.set_header("Retry-After", kRetryAfterUnavailableSeconds); }
        // A stalled executor recovers in this process: a load balancer's probe keeps its
        // connection. A failed or draining server goes away.
        res.keep_alive_after_error = std::string_view(status) == "stalled";
        res.set_content(body.dump(), "application/json");
    });
    server_.Get("/props", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(make_props(public_model_id_, options_.max_context,
                                   options_.max_concurrency, model_metadata_),
                        "application/json");
    });
    server_.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
        const ninferno::RuntimeStats stats =
            service_ != nullptr ? service_->runtime_stats() : ninferno::RuntimeStats{};
        res.set_content(make_prometheus_metrics(stats, active_requests_.load(), model_metadata_),
                        "text/plain; version=0.0.4");
    });
    server_.Get(
        api_route_pattern("/models"),
        [this](const httplib::Request& req, httplib::Response& res) { handle_models(req, res); });
    server_.Get(
        api_route_pattern(R"(/models/(.+))"),
        [this](const httplib::Request& req, httplib::Response& res) { handle_model(req, res); });
    server_.Post(api_route_pattern("/chat/completions"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_chat_completions(req, res);
                 });
    server_.Post(api_route_pattern("/responses"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_responses(req, res);
                 });
    server_.Post(api_route_pattern("/responses/input_tokens"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_input_tokens(req, res);
                 });
    server_.Post(api_route_pattern("/responses/compact"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_compact(req, res);
                 });
    server_.Post(api_route_pattern(R"(/responses/([^/]+)/cancel)"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_cancel(req, res);
                 });
    server_.Get(api_route_pattern(R"(/responses/([^/]+)/input_items)"),
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_input_items(req, res);
                });
    server_.Get(api_route_pattern(R"(/responses/([^/]+))"),
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_get(req, res);
                });
    server_.Delete(api_route_pattern(R"(/responses/([^/]+))"),
                   [this](const httplib::Request& req, httplib::Response& res) {
                       handle_response_delete(req, res);
                   });
    server_.Post(api_route_pattern("/messages/count_tokens"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_count_tokens(req, res);
                 });
    server_.Post(
        api_route_pattern("/messages"),
        [this](const httplib::Request& req, httplib::Response& res) { handle_messages(req, res); });
}

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_models_list(public_model_id_, model_created_, options_.max_context,
                                     model_metadata_, &startup_speculation_),
                    "application/json");
}

void HttpServer::handle_model(const httplib::Request& req, httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    if (id != public_model_id_) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + ascii_preview(id) + "' not found";
        write_openai_error(res, error);
        return;
    }
    res.set_content(make_model_object(public_model_id_, model_created_, options_.max_context,
                                      model_metadata_, &startup_speculation_),
                    "application/json");
}

bool HttpServer::bind() { return server_.bind_to_port(options_.host, options_.port); }

void HttpServer::attach(GenerationService& service) {
    if (service_ != nullptr) {
        throw std::logic_error("HTTP generation service is already attached");
    }
    const ninferno::LoadSummary load = service.load_summary();
    public_model_id_               = resolve_public_model_id(options_, load.model_name);
    model_created_                   = unix_time_now();
    sampling_defaults_header_ = sampling_defaults_header(service.sampling_defaults(), options_);
    model_metadata_                = service.model_metadata();
    startup_speculation_           = service.startup_speculation();
    service_                       = &service;
    // An automatic max_context is what the Engine chose (/v1/models reports it).
    options_.max_context           = service.engine_options().max_context;
    request_jsonl_.write_server_start(options_, service.engine_options(),
                                      service.sampling_defaults(), public_model_id_, load,
                                      service.memory_summary());
}

bool HttpServer::listen() {
    if (service_ == nullptr) { throw std::logic_error("HTTP generation service is not attached"); }
    if (public_model_id_.empty()) {
        throw std::logic_error("HTTP public model id is not resolved");
    }
    // Inside the try: a thread that fails to start must not leave the other one joinable.
    try {
        if (options_.log_stats_interval_ms != 0) {
            stats_stopping_ = false;
            stats_thread_   = std::thread([this] { run_stats_reporter(); });
        }
        supervisor_stopping_ = false;
        supervisor_thread_   = std::thread([this] { run_supervisor(); });
        const bool result = server_.listen_after_bind();
        stop_supervisor();
        stop_stats_reporter();
        return result;
    } catch (...) {
        stop_supervisor();
        stop_stats_reporter();
        throw;
    }
}

void HttpServer::stop() { server_.stop(); }

void HttpServer::request_drain() noexcept {
    if (drain_requested_.exchange(true)) { stop_requested_.store(true); }
}

ContextCacheHints HttpServer::request_cache_hints(const httplib::Request& req, httplib::Response& res,
                                                 std::shared_ptr<void>* kept_turn,
                                                 ContextCacheHints base) const {
    if (!req.has_header(std::string(kCacheHintHeader).c_str())) { return base; }
    const std::string header = req.get_header_value(std::string(kCacheHintHeader).c_str());
    std::optional<KeptGrant> kept;
    const std::string_view applied =
        apply_cache_hint(&header, options_.trust_cache_hints,
                         options_.context_cache.enabled ? options_.context_cache.kept_tokens : 0,
                         base, &kept);
    if (kept) { *kept_turn = kept_sessions_.begin_turn(*kept, std::chrono::steady_clock::now()); }
    res.set_header(std::string(kCacheHintAppliedHeader).c_str(), std::string(applied));
    return base;
}

RequestLimits HttpServer::request_limits() const noexcept {
    return RequestLimits{.default_max_tokens = options_.default_max_tokens,
                         .max_messages       = options_.max_request_messages};
}

bool HttpServer::reject_while_draining(const httplib::Request& req, httplib::Response& res) {
    if (!draining_.load() || req.method == "GET" || req.method == "OPTIONS") { return false; }
    res.set_header("Retry-After", kRetryAfterUnavailableSeconds);
    write_error_for(req, res, server_draining_error());
    return true;
}

void HttpServer::run_supervisor() {
    using Clock = std::chrono::steady_clock;
    ExecutorStallMonitor monitor{std::chrono::milliseconds(options_.health_stall_ms)};
    std::optional<Clock::time_point> drain_started;
    while (!supervisor_stopping_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const Clock::time_point now      = Clock::now();
        const ninferno::RuntimeStats stats = service_->runtime_stats();
        const bool stalled               = monitor.update(stats, now);
        // A kept thread silent past its ttl-ms falls to the private weight (cache_hint.h).
        // A turn still running under the keep publishes its entry kept: asked again after a ttl.
        for (const KeptGrant& grant : kept_sessions_.expired(now)) {
            if (service_->release_kept_session(grant.session_key) ==
                ninferno::KeptSessionRelease::InFlight) {
                kept_sessions_.keep(grant, now);
            }
        }
        if (stalled != stalled_.exchange(stalled)) { operational_log_.executor_stall(stalled); }
        // Checked while draining too: a device that fails then still ends the process with 1.
        if (!service_->is_available() && !engine_failed_.exchange(true)) {
            operational_log_.engine_failed();
            drain_requested_.store(true);
        }
        if (drain_requested_.load() && !drain_started) {
            drain_started = now;
            draining_.store(true);
            operational_log_.drain_started(active_requests_.load());
        }
        if (!drain_started) { continue; }
        const bool idle = active_requests_.load() == 0 && stats.running_requests == 0 &&
                          stats.waiting_requests == 0 && stats.materializing_requests == 0;
        const bool timed_out =
            now - *drain_started >= std::chrono::milliseconds(options_.drain_timeout_ms);
        if (idle || timed_out || stop_requested_.load()) {
            operational_log_.drain_finished(
                idle ? "requests in flight finished"
                     : (stop_requested_.load() ? "second signal" : "drain timeout"),
                std::chrono::duration<double>(now - *drain_started).count());
            // The listener's workers finish their requests before listen() returns: the ones
            // still running end now, as a disconnected client's.
            if (!idle) { service_->abort_in_flight(); }
            server_.stop();
            return;
        }
    }
}

void HttpServer::stop_supervisor() {
    supervisor_stopping_.store(true);
    if (supervisor_thread_.joinable()) { supervisor_thread_.join(); }
}

} // namespace ninferno::serve
