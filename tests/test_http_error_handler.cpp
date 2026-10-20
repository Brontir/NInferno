#include "serve/http_server.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <utility>

namespace {

using Json = nlohmann::json;
using ninferno::serve::ServeOptions;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    ServeOptions options;
    options.max_request_bytes = 1234;

    // A browser SDK's preflight names its own headers; the permissive answer allows them.
    httplib::Request preflight;
    preflight.method = "OPTIONS";
    preflight.set_header("Access-Control-Request-Headers",
                         "authorization,content-type,x-stainless-lang,x-stainless-os,"
                         "anthropic-dangerous-direct-browser-access");
    preflight.set_header("Access-Control-Request-Private-Network", "true");
    httplib::Response preflight_response;
    preflight_response.set_header("Access-Control-Allow-Headers", "Authorization, Content-Type");
    ninferno::serve::answer_cors_preflight(preflight, preflight_response);
    failures += check(
        preflight_response.status == 204 &&
            preflight_response.get_header_value_count("Access-Control-Allow-Headers") == 1 &&
            preflight_response.get_header_value("Access-Control-Allow-Headers") ==
                preflight.get_header_value("Access-Control-Request-Headers") &&
            preflight_response.get_header_value("Access-Control-Allow-Private-Network") == "true",
        "a CORS preflight did not allow the headers and private network access it asked for");
    httplib::Request plain_preflight;
    plain_preflight.method = "OPTIONS";
    httplib::Response plain_response;
    plain_response.set_header("Access-Control-Allow-Headers", "Authorization, Content-Type");
    ninferno::serve::answer_cors_preflight(plain_preflight, plain_response);
    failures += check(plain_response.status == 204 &&
                          plain_response.get_header_value("Access-Control-Allow-Headers") ==
                              "Authorization, Content-Type" &&
                          !plain_response.has_header("Access-Control-Allow-Private-Network"),
                      "a preflight that asks for nothing changed the default CORS headers");

    const ninferno::serve::ApiError media_budget = ninferno::serve::request_error_to_api_error(
        ninferno::RequestError(ninferno::RequestErrorKind::MediaBudgetExceeded,
                             "vision tokens exceed processor budget"));
    failures += check(media_budget.status == 400 && media_budget.code == "media_budget_exceeded",
                      "media resource rejection did not map to HTTP 400");
    const ninferno::serve::ApiError invalid_media = ninferno::serve::request_error_to_api_error(
        ninferno::RequestError(ninferno::RequestErrorKind::InvalidMedia, "failed to open media"));
    failures += check(invalid_media.status == 400 && invalid_media.code == "invalid_media" &&
                          invalid_media.param == "messages",
                      "invalid media did not retain its client-input classification");
    const ninferno::serve::ApiError numerical = ninferno::serve::request_error_to_api_error(
        ninferno::RequestError(ninferno::RequestErrorKind::NumericalFailure,
                             "the model computed a NaN or an infinity in its final hidden state"));
    failures += check(numerical.status == 500 && numerical.code == "numerical_failure" &&
                          numerical.type == "server_error" && numerical.param.empty(),
                      "a numerical failure did not map to a server error");
    const ninferno::serve::ApiError context_limit = ninferno::serve::request_error_to_api_error(
        ninferno::RequestError(ninferno::RequestErrorKind::ContextLengthExceeded,
                             "prepared prompt has 200 tokens, exceeding Engine max_context 128"));
    failures += check(
        context_limit.status == 400 && context_limit.code == "context_length_exceeded" &&
            context_limit.message.find("200 tokens") != std::string::npos &&
            context_limit.message.find("128") != std::string::npos &&
            context_limit.message.starts_with("prompt is too long") &&
            context_limit.message.find("exceeds the available context size") != std::string::npos,
        "context rejection lost its HTTP classification or capacity details");
    const ninferno::serve::ApiError thinking_capacity = ninferno::serve::request_error_to_api_error(
        ninferno::RequestError(ninferno::RequestErrorKind::ThinkingBudgetCapacityInsufficient,
                             "thinking control suffix does not fit"));
    failures += check(thinking_capacity.status == 400 &&
                          thinking_capacity.code == "thinking_budget_capacity_insufficient" &&
                          thinking_capacity.param.empty(),
                      "thinking budget capacity error mapping mismatch");
    const ninferno::serve::ApiError cancelled =
        ninferno::serve::request_error_to_api_error(ninferno::RequestError(
            ninferno::RequestErrorKind::Cancelled, "request cancelled during preparation"));
    failures += check(cancelled.status == 499 && cancelled.code == "client_disconnected" &&
                          cancelled.param.empty(),
                      "preparation cancellation did not retain its HTTP classification");

    failures +=
        check(ninferno::serve::matches_bearer_credential("Bearer secret", "secret") &&
                  ninferno::serve::matches_bearer_credential("bearer secret", "secret") &&
                  ninferno::serve::matches_bearer_credential("\tBEARER   secret\t", "secret"),
              "valid Bearer credentials were rejected because of scheme case or whitespace");
    failures += check(!ninferno::serve::matches_bearer_credential("Basic secret", "secret") &&
                          !ninferno::serve::matches_bearer_credential("Bearer wrong", "secret") &&
                          !ninferno::serve::matches_bearer_credential("Bearersecret", "secret") &&
                          !ninferno::serve::matches_bearer_credential("Bearer secret", ""),
                      "invalid Bearer credentials were accepted");
    failures += check(ninferno::serve::matches_api_key("secret", "secret") &&
                          !ninferno::serve::matches_api_key("secreT", "secret") &&
                          !ninferno::serve::matches_api_key("secret ", "secret") &&
                          !ninferno::serve::matches_api_key("secre", "secret") &&
                          !ninferno::serve::matches_api_key("", "secret") &&
                          !ninferno::serve::matches_api_key("", ""),
                      "x-api-key comparison accepted a different value or an empty key");

    // A page's cross-site "simple request" carries Origin; curl and the SDKs send none.
    {
        httplib::Request browser;
        browser.path = "/v1/chat/completions";
        browser.set_header("Origin", "http://evil.example");
        httplib::Request sdk;
        sdk.path = "/v1/chat/completions";
        httplib::Request probe;
        probe.path = "/health";
        probe.set_header("Origin", "http://evil.example");
        failures += check(ninferno::serve::refuses_browser_origin(browser, false) &&
                              !ninferno::serve::refuses_browser_origin(browser, true) &&
                              !ninferno::serve::refuses_browser_origin(sdk, false) &&
                              !ninferno::serve::refuses_browser_origin(probe, false),
                          "a browser origin was not refused without --cors, or a client was");
    }

    httplib::Request messages_request;
    messages_request.path = "/v1/messages";
    httplib::Response messages_response;
    messages_response.status = 413;
    const auto messages_result =
        ninferno::serve::handle_unrendered_http_error(options, messages_request, messages_response);
    const Json messages_body = Json::parse(messages_response.body);
    failures += check(messages_result == httplib::Server::HandlerResponse::Handled &&
                          messages_body.at("type") == "error" &&
                          messages_body.at("error").at("type") == "request_too_large" &&
                          messages_body.at("request_id").get<std::string>().starts_with("req_") &&
                          messages_response.get_header_value("request-id") ==
                              messages_body.at("request_id").get<std::string>() &&
                          !messages_response.has_header("x-request-id") &&
                          messages_body.at("error").at("message").get<std::string>().find(
                              "1234 bytes") != std::string::npos,
                      "empty Anthropic 413 did not become a payload-limit error");

    httplib::Request openai_request;
    openai_request.path = "/v1/responses";
    httplib::Response openai_response;
    openai_response.status = 413;
    const auto openai_result =
        ninferno::serve::handle_unrendered_http_error(options, openai_request, openai_response);
    const Json openai_body = Json::parse(openai_response.body);
    failures += check(openai_result == httplib::Server::HandlerResponse::Handled &&
                          openai_body.at("error").at("code") == "request_too_large" &&
                          openai_response.get_header_value_count("x-request-id") == 1 &&
                          openai_response.get_header_value("x-request-id").starts_with("req_") &&
                          openai_body.at("error").at("message").get<std::string>().find(
                              "1234 bytes") != std::string::npos,
                      "empty OpenAI 413 did not become a payload-limit error");

    httplib::Request missing_messages_request;
    missing_messages_request.path = "/v1/messages/missing";
    httplib::Response missing_messages_response;
    missing_messages_response.status = 404;
    missing_messages_response.set_header("request-id", "req_stale");
    const auto missing_messages_result = ninferno::serve::handle_unrendered_http_error(
        options, missing_messages_request, missing_messages_response);
    const Json missing_messages_body = Json::parse(missing_messages_response.body);
    failures +=
        check(missing_messages_result == httplib::Server::HandlerResponse::Handled &&
                  missing_messages_response.status == 404 &&
                  missing_messages_body.at("error").at("type") == "not_found_error" &&
                  missing_messages_body.at("request_id").get<std::string>().starts_with("req_") &&
                  missing_messages_body.at("request_id") != "req_stale" &&
                  missing_messages_response.get_header_value_count("request-id") == 1 &&
                  missing_messages_response.get_header_value("request-id") ==
                      missing_messages_body.at("request_id").get<std::string>(),
              "missing Anthropic resource did not use the protocol error envelope");

    httplib::Response authored_response;
    authored_response.status = 413;
    authored_response.set_header("x-request-id", "req_existing");
    authored_response.set_content(R"({"error":{"code":"application_error"}})", "application/json");
    const std::string authored_body = authored_response.body;
    const auto authored_result =
        ninferno::serve::handle_unrendered_http_error(options, openai_request, authored_response);
    failures += check(authored_result == httplib::Server::HandlerResponse::Unhandled &&
                          authored_response.body == authored_body &&
                          authored_response.get_header_value_count("x-request-id") == 1 &&
                          authored_response.get_header_value("x-request-id") == "req_existing",
                      "application-authored 413 or its request ID was overwritten");

    httplib::Response other_response;
    other_response.status = 400;
    const auto other_result =
        ninferno::serve::handle_unrendered_http_error(options, openai_request, other_response);
    failures += check(other_result == httplib::Server::HandlerResponse::Unhandled &&
                          other_response.body.empty(),
                      "non-413 response was changed by the payload-limit handler");

    // A compressed body: the library's empty 415 gets an error body naming it.
    httplib::Request encoded_request;
    encoded_request.path = "/v1/chat/completions";
    encoded_request.set_header("Content-Encoding", "gzip");
    httplib::Response encoded_response;
    encoded_response.status = 415;
    const auto encoded_result =
        ninferno::serve::handle_unrendered_http_error(options, encoded_request, encoded_response);
    const Json encoded_body = Json::parse(encoded_response.body);
    failures += check(encoded_result == httplib::Server::HandlerResponse::Handled &&
                          encoded_response.status == 415 &&
                          encoded_body.at("error").at("code") == "unsupported_content_encoding" &&
                          encoded_body.at("error").at("message").get<std::string>().find(
                              "'gzip'") != std::string::npos,
                      "an unsupported Content-Encoding answered without an error body");

    // Overload and admission timeout say when to retry; a client error does not.
    for (const auto& [status, retry] :
         {std::pair{429, "1"}, std::pair{503, "5"}, std::pair{504, "5"}, std::pair{400, ""},
          std::pair{500, ""}}) {
        ninferno::serve::ApiError error;
        error.status  = status;
        error.message = "x";
        httplib::Response openai;
        ninferno::serve::write_openai_error(openai, error);
        httplib::Response anthropic;
        ninferno::serve::write_anthropic_error(anthropic, error, "req_1");
        failures += check(openai.get_header_value("Retry-After") == std::string(retry) &&
                              anthropic.has_header("Retry-After") == (std::string(retry) != ""),
                          "an error response's Retry-After does not match its status");
    }

    // A request refused or given up while the server shuts down, in both dialects.
    httplib::Response draining_openai;
    ninferno::serve::write_openai_error(draining_openai, ninferno::serve::server_draining_error());
    httplib::Response draining_anthropic;
    ninferno::serve::write_anthropic_error(draining_anthropic,
                                           ninferno::serve::server_draining_error(), "req_2");
    failures +=
        check(draining_openai.status == 503 && draining_anthropic.status == 503 &&
                  Json::parse(draining_openai.body).at("error").at("code") == "server_draining" &&
                  draining_openai.get_header_value("Retry-After") == "5" &&
                  draining_anthropic.get_header_value("Retry-After") == "5",
              "a draining refusal is not 503 server_draining with Retry-After 5");
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
