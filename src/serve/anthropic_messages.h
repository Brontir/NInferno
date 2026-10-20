#pragma once

// Anthropic Messages wire adapter. Request parsing lowers executable semantics into the common
// GenerationRequest; response construction owns Anthropic aggregate, SSE, usage, and error shapes.
// Visible Thinking text owns prompt semantics. Its wire signature is response metadata and never
// enters common request or Engine state, except the signature of hidden Thinking: with
// display "omitted" the reasoning travels base64url-encoded in the signature, and an echoed empty
// block restores exactly that text, so the next prompt is the one a visible-Thinking client sends.

#include "serve/request.h"
#include "serve/request_json.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::serve {

struct GenerationOutcome;

enum class AnthropicThinkingDisplay { Visible, Omitted };

struct AnthropicMessagesRequest {
    std::string model;
    GenerationRequest generation;
    bool stream                 = false;
    bool output_tokens_explicit = false;
    AnthropicThinkingDisplay thinking_display = AnthropicThinkingDisplay::Visible;
};

struct AnthropicCountTokensRequest {
    std::string model;
    GenerationRequest generation;
};

AnthropicMessagesRequest parse_anthropic_messages_request(const RequestJson& body,
                                                          const RequestLimits& limits);
AnthropicCountTokensRequest parse_anthropic_count_tokens_request(const RequestJson& body,
                                                                 const RequestLimits& limits = {});

struct AnthropicResponseIdentity {
    std::string request_id;
    std::string message_id;
    std::string model;
};

std::string new_anthropic_request_id();
AnthropicResponseIdentity make_anthropic_response_identity(std::string request_id,
                                                           std::string model);

ApiError normalize_anthropic_error(ApiError error);
std::string make_anthropic_error_body(const ApiError& error, const std::string& request_id);
std::string make_anthropic_sse_error(const ApiError& error, const std::string& request_id);

std::string make_anthropic_messages_response(
    const AnthropicResponseIdentity& identity, const GenerationOutcome& outcome,
    AnthropicThinkingDisplay display = AnthropicThinkingDisplay::Visible);

// The signature of a hidden Thinking block, and the reasoning an echoed one carries (none for a
// signature NInferno did not write).
std::string encode_hidden_thinking_signature(std::string_view reasoning);
std::optional<std::string> decode_hidden_thinking_signature(std::string_view signature);
std::string make_anthropic_count_tokens_response(int input_tokens);

class AnthropicMessagesStream {
public:
    AnthropicMessagesStream(AnthropicResponseIdentity identity, int input_tokens,
                            AnthropicThinkingDisplay display = AnthropicThinkingDisplay::Visible);

    // The Engine start event is exact for normal streams. The no-argument form is reserved for an
    // error raised before admission, so an Anthropic error event still has a valid stream prefix.
    std::string start();
    std::string start(const ninferno::GenerationStart& generation);

    [[nodiscard]] bool started() const noexcept { return started_; }

    std::vector<std::string> reasoning_delta(const std::string& text);
    std::vector<std::string> content_delta(const std::string& text);
    std::vector<std::string> finish(const GenerationOutcome& outcome);
    std::string error(const ApiError& api_error) const;

private:
    std::string start_with_cache(std::optional<int> cache_read_input_tokens);
    std::vector<std::string> close_thinking();
    std::vector<std::string> close_text();

    AnthropicResponseIdentity identity_;
    AnthropicThinkingDisplay display_ = AnthropicThinkingDisplay::Visible;
    std::string reasoning_;
    std::string content_;
    int input_tokens_   = 0;
    int next_index_     = 0;
    int thinking_index_ = -1;
    int text_index_     = -1;
    bool started_       = false;
    bool finished_      = false;
    bool thinking_open_ = false;
    bool text_open_     = false;
};

} // namespace ninferno::serve
