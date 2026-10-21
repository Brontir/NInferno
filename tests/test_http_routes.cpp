#include "serve/http_server.h"
#include "serve/openai_common.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <regex>
#include <string>

namespace {

using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    using ninferno::serve::api_route_pattern;
    using ninferno::serve::canonical_api_path;

    // An Anthropic SDK given a base URL ending in /v1 adds one more /v1; only that one is removed.
    failures += check(canonical_api_path("/v1/v1/messages") == "/v1/messages" &&
                          canonical_api_path("/v1/v1/models/qwen") == "/v1/models/qwen",
                      "a doubled /v1 prefix must name the /v1 endpoint");
    failures += check(canonical_api_path("/v1/messages") == "/v1/messages" &&
                          canonical_api_path("/v1/v1/v1/messages") == "/v1/v1/messages" &&
                          canonical_api_path("/v1/v1") == "/v1/v1" &&
                          canonical_api_path("/health") == "/health",
                      "only a single doubled /v1 prefix may be removed");

    // Route patterns match under either prefix, keep the endpoint's own captures and match
    // nothing else.
    const std::regex cancel(api_route_pattern(R"(/responses/([^/]+)/cancel)"));
    std::smatch match;
    const std::string doubled = "/v1/v1/responses/resp_1/cancel";
    const std::string single  = "/v1/responses/resp_2/cancel";
    failures +=
        check(std::regex_match(doubled, match, cancel) && match.size() == 2 && match[1] == "resp_1",
              "a doubled route must keep its capture group");
    failures += check(std::regex_match(single, match, cancel) && match[1] == "resp_2",
                      "the single-prefix route must still match");
    const std::regex messages(api_route_pattern("/messages"));
    failures += check(!std::regex_match(std::string("/v1/v1/v1/messages"), messages) &&
                          !std::regex_match(std::string("/v2/messages"), messages),
                      "a route must not match other prefixes");
    failures += check(std::regex_match(std::string("/messages"), messages) &&
                          ninferno::serve::api_endpoint("/chat/completions") == "/chat/completions" &&
                          ninferno::serve::api_endpoint("/v1/v1/chat/completions") ==
                              "/chat/completions",
                      "the endpoints also answer without /v1 (llama.cpp-style base URLs)");
    {
        httplib::Request unknown;
        unknown.method = "GET";
        unknown.path   = "/v1/nothing";
        httplib::Response unknown_response;
        unknown_response.status = 404;
        const auto handled = ninferno::serve::handle_unrendered_http_error(
            ninferno::serve::ServeOptions{}, unknown, unknown_response);
        failures += check(handled == httplib::Server::HandlerResponse::Handled &&
                              Json::parse(unknown_response.body).at("error").at("code") ==
                                  "not_found",
                          "an unknown path answers a JSON 404");
    }

    // Errors on a doubled path keep the envelope and request-id header of the endpoint it names.
    ninferno::serve::ServeOptions options;
    httplib::Request missing_messages;
    missing_messages.path = "/v1/v1/messages/missing";
    httplib::Response missing_response;
    missing_response.status = 404;
    const auto missing_result =
        ninferno::serve::handle_unrendered_http_error(options, missing_messages, missing_response);
    failures += check(missing_result == httplib::Server::HandlerResponse::Handled &&
                          !missing_response.body.empty() &&
                          Json::parse(missing_response.body).at("type") == "error" &&
                          missing_response.has_header("request-id") &&
                          !missing_response.has_header("x-request-id"),
                      "a doubled Anthropic path must keep the Anthropic error envelope");
    httplib::Request oversized_responses;
    oversized_responses.path = "/v1/v1/responses";
    httplib::Response oversized_response;
    oversized_response.status   = 413;
    const auto oversized_result = ninferno::serve::handle_unrendered_http_error(
        options, oversized_responses, oversized_response);
    failures += check(oversized_result == httplib::Server::HandlerResponse::Handled &&
                          Json::parse(oversized_response.body).at("error").at("code") ==
                              "request_too_large" &&
                          oversized_response.has_header("x-request-id"),
                      "a doubled OpenAI path must keep the OpenAI error envelope");

    // /metrics speaks Prometheus text under llama.cpp's names; /props carries context and slots.
    ninferno::RuntimeStats stats;
    stats.computed_prefill_tokens = 7;
    stats.committed_decode_tokens = 11;
    stats.decode_rounds           = 4;
    stats.decode_row_rounds       = 6;
    stats.waiting_requests        = 2;
    stats.materializing_requests  = 5;
    ninferno::ModelMetadata fingerprinted;
    fingerprinted.decision_fingerprint = "nf1-0123456789abcdef";
    fingerprinted.decision_manifest    = "nm1-fedcba9876543210";
    const std::string metrics = ninferno::serve::make_prometheus_metrics(stats, 3, fingerprinted);
    failures += check(
        metrics.find("# TYPE ninferno_decision_fingerprint_info gauge\n"
                     "ninferno_decision_fingerprint_info{fingerprint=\"nf1-0123456789abcdef\","
                     "manifest=\"nm1-fedcba9876543210\",schema=\"ninferno-fingerprint/1\"} 1\n") !=
                std::string::npos &&
            ninferno::serve::make_prometheus_metrics(stats, 3, ninferno::ModelMetadata{})
                    .find("ninferno_decision_fingerprint_info") == std::string::npos,
        "/metrics must carry the decision fingerprint as an info gauge when there is one");
    const Json models = Json::parse(ninferno::serve::make_models_list("m", 0, 4096, fingerprinted));
    failures += check(
        models.at("data").at(0).at("meta").at("decision_fingerprint") == "nf1-0123456789abcdef" &&
            models.at("data").at(0).at("meta").at("decision_manifest") == "nm1-fedcba9876543210",
        "/v1/models meta must carry the decision fingerprint and manifest");
    failures +=
        check(metrics.find("# TYPE llamacpp:prompt_tokens_total counter\n"
                           "llamacpp:prompt_tokens_total 7\n") != std::string::npos &&
                  metrics.find("\nllamacpp:tokens_predicted_total 11\n") != std::string::npos &&
                  metrics.find("\nllamacpp:n_busy_slots_per_decode 1.5\n") != std::string::npos &&
                  metrics.find("\nllamacpp:requests_deferred 2\n") != std::string::npos &&
                  metrics.find("\nninferno_http_active_requests 3\n") != std::string::npos &&
                  metrics.find("\nninferno_requests_materializing 5\n") != std::string::npos,
              "/metrics must expose the llama.cpp counters in Prometheus text");
    const Json props =
        Json::parse(ninferno::serve::make_props("m", 4096, 2, ninferno::ModelMetadata{}));
    failures += check(props.at("default_generation_settings").at("n_ctx") == 4096 &&
                          props.at("total_slots") == 2,
                      "/props must carry n_ctx and total_slots");

    if (failures == 0) { std::cout << "http routes: ok\n"; }
    return failures == 0 ? 0 : 1;
}
