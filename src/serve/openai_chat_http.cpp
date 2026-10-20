#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"
#include "serve/request_validation.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ninferno::serve {
namespace {

// An error after the stream started: its error object, then [DONE] as every Chat stream ends (as
// vLLM ends one), so a client reading to the sentinel does not wait for the close.
std::string sse_error_event(const ApiError& error) {
    return "data: " + make_error_body(error) + "\n\ndata: [DONE]\n\n";
}

} // namespace

void HttpServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    OpenAIChatRequest request;
    try {
        const RequestLimits limits = request_limits();
        request                    = parse_chat_completion_request(parse_json_body(req), limits,
                                                                   options_.auto_system_shared_prefix);
    } catch (const ApiException& exception) {
        refuse(req, res, exception.error());
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    const RequestLogMetadata metadata{.model                  = request.model,
                                      .stream                 = request.stream,
                                      .output_tokens_explicit = request.output_tokens_explicit};
    PreparedRequest prepared;
    try {
        const ninferno::GenerationObservationOptions observation{
            .phase_timings   = true,
            .live_timings    = request.stream && request.timings_per_token,
            .prompt_progress = request.stream && request.return_progress,
        };
        prepared = service_->prepare(request.generation,
                                     request.stream ? GenerationConsumerMode::Streaming
                                                    : GenerationConsumerMode::Aggregate,
                                     observation, [&req] { return client_disconnected(req); });
    } catch (const ApiException& exception) {
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        ApiError error = internal_error(exception.what());
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, error));
        write_openai_error(res, error);
        return;
    }

    OpenAIChatResponseIdentity identity = make_openai_chat_response_identity(request.model);
    for (const ToolDefinition& tool : request.generation.tools) {
        if (tool.custom) { identity.custom_tools.push_back(tool.name); }
    }
    auto lifecycle = begin_request(make_request_log_context(req_id, "openai_chat_completions",
                                                            request.generation, metadata, prepared),
                                   res);
    if (const std::string warning = notes_header_value(request.generation); !warning.empty()) {
        res.set_header("X-NInferno-Warning", warning);
    }

    if (!request.stream) {
        GenerationOutcome outcome;
        try {
            outcome = service_->run(prepared, nullptr, [&req] { return client_disconnected(req); });
        } catch (const ApiException& exception) {
            lifecycle->failure(make_generation_request_failure(exception.error()));
            write_openai_error(res, exception.error());
            return;
        } catch (const std::exception& exception) {
            const RequestFailure failure =
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what());
            lifecycle->failure(failure);
            ApiError error = internal_error(exception.what());
            write_openai_error(res, error);
            return;
        }
        lifecycle->done(outcome);
        // A client that went away cancelled the run: there is no connection left to answer (the
        // streamed path's rule).
        if (outcome.finish_reason == ninferno::FinishReason::Cancelled &&
            client_disconnected(req)) {
            lifecycle->response_failure(
                make_client_disconnected_failure(RequestFailurePhase::Transport));
            return;
        }
        // The server gave the run up (drain timeout, second signal): the text is cut short, so
        // the client is told to retry rather than handed it as a finished answer.
        if (outcome.finish_reason == ninferno::FinishReason::Cancelled && service_->aborting()) {
            const ApiError error = server_draining_error();
            lifecycle->response_failure(
                make_request_failure(RequestFailurePhase::Generation, error));
            write_openai_error(res, error);
            return;
        }
        try {
            set_owned_json_content(res, make_chat_completion_response(identity, outcome),
                                   prepared.lifetime);
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            ApiError error = internal_error(exception.what());
            write_openai_error(res, error);
        }
        return;
    }

    try {
        const bool return_progress   = request.return_progress;
        const bool timings_per_token = request.timings_per_token;
        auto stream                  = std::make_shared<HttpGenerationStream>(std::move(prepared));
        auto encoder =
            std::make_shared<OpenAIChatStream>(identity, request.include_usage, timings_per_token,
                                               return_progress, options_.usage_chunk_choice);

        prepare_sse_response(res);
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, stream, encoder, lifecycle, return_progress,
             timings_per_token](std::size_t, httplib::DataSink& sink) -> bool {
                if (stream->started.exchange(true, std::memory_order_acq_rel)) {
                    sink.done();
                    return true;
                }
                SseTransport transport(sink, stream->cancelled);
                const auto send_error = [&](const ApiError& error) {
                    try {
                        render_and_write(transport, [&] { return sse_error_event(error); });
                        sink.done();
                        return true;
                    } catch (const ClientDisconnected&) {
                        lifecycle->response_failure(
                            make_client_disconnected_failure(RequestFailurePhase::Transport));
                        return false;
                    } catch (const ResponseRenderFailure& exception) {
                        lifecycle->response_failure(make_internal_request_failure(
                            RequestFailurePhase::ResponseRender, exception.what()));
                        return false;
                    }
                };
                try {
                    render_and_write(transport, [&] { return encoder->start(); });
                } catch (const ClientDisconnected&) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error = internal_error(exception.what());
                    return send_error(error);
                }

                GenerationOutcome outcome;
                try {
                    StreamSink output;
                    output.on_start = [&](const ninferno::GenerationStart& start) {
                        encoder->note_start(start);
                        if (return_progress) {
                            render_and_write(transport,
                                             [&] { return encoder->initial_prompt_progress(); });
                        }
                    };
                    if (return_progress) {
                        output.on_progress = [&](const ninferno::PromptProgress& progress) {
                            render_and_write(transport,
                                             [&] { return encoder->prompt_progress(progress); });
                        };
                    }
                    if (timings_per_token) {
                        output.on_timing =
                            [&](const ninferno::GenerationTimingObservation& timing) {
                                encoder->note_timing(timing);
                            };
                    }
                    output.on_content = [&](const std::string& text) {
                        render_and_write(transport, [&] { return encoder->content_delta(text); });
                    };
                    output.on_reasoning = [&](const std::string& text) {
                        render_and_write(transport, [&] { return encoder->reasoning_delta(text); });
                    };
                    output.is_cancelled = [&] { return transport.poll(); };

                    outcome = service_->run(stream->prepared, &output);
                } catch (const ClientDisconnected&) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error = internal_error(exception.what());
                    return send_error(error);
                } catch (const ApiException& exception) {
                    lifecycle->failure(make_generation_request_failure(exception.error()));
                    return send_error(exception.error());
                } catch (const std::exception& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::Generation, exception.what()));
                    ApiError error = internal_error(exception.what());
                    return send_error(error);
                }

                lifecycle->done(outcome);
                // A client that went away cancelled the run, before its first token as often as
                // not: there is nothing to render its end to, and no stream it was started on.
                if (outcome.finish_reason == ninferno::FinishReason::Cancelled &&
                    transport.poll()) {
                    lifecycle->response_failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                }
                std::vector<std::string> terminal;
                try {
                    terminal = encoder->finish(outcome);
                } catch (const std::exception& exception) {
                    lifecycle->response_failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error = internal_error(exception.what());
                    return send_error(error);
                }
                try {
                    transport.write(terminal);
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) {
                    lifecycle->response_failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                }
            },
            [stream, lifecycle](bool successful) {
                stream->cancelled.store(true, std::memory_order_release);
                if (!successful || !stream->started.load(std::memory_order_acquire)) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                }
            });
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::ResponseRender, exception.what()));
        ApiError error = internal_error(exception.what());
        write_openai_error(res, error);
    }
}

} // namespace ninferno::serve
