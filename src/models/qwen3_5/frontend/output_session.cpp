#include "models/qwen3_5/frontend/output_session.h"
#include "chat/conventions.h"
#include "chat/tool_calls.h"
#include "text/token_codec.h"
#include "text/unicode.h"
#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninferno::models::qwen3_5 {
namespace {
constexpr std::string_view kUtf8Replacement = "\xef\xbf\xbd";

std::size_t channel_index(OutputChannel channel) noexcept {
    return channel == OutputChannel::Reasoning ? 0 : 1;
}

void append_delta(PublishedOutput& output, OutputChannel channel, std::string text) {
    if (text.empty()) { return; }
    if (!output.empty() && output.back().channel == channel) {
        output.back().text += text;
    } else {
        output.push_back(OutputDelta{.channel = channel, .text = std::move(text)});
    }
}

std::string consume_generated_utf8(std::string& pending) {
    std::string decoded;
    decoded.reserve(pending.size());
    std::size_t offset = 0;
    while (offset < pending.size()) {
        const auto lead    = static_cast<unsigned char>(pending[offset]);
        std::size_t length = 0;
        if (lead <= 0x7fU) {
            decoded.push_back(pending[offset]);
            ++offset;
            continue;
        } else if (lead >= 0xc2U && lead <= 0xdfU) {
            length = 2;
        } else if (lead >= 0xe0U && lead <= 0xefU) {
            length = 3;
        } else if (lead >= 0xf0U && lead <= 0xf4U) {
            length = 4;
        } else {
            decoded.append(kUtf8Replacement);
            ++offset;
            continue;
        }

        bool malformed = false;
        for (std::size_t index = 1; index < length; ++index) {
            if (offset + index >= pending.size()) {
                pending.erase(0, offset);
                return decoded;
            }
            const auto byte      = static_cast<unsigned char>(pending[offset + index]);
            unsigned int minimum = 0x80U;
            unsigned int maximum = 0xbfU;
            if (index == 1) {
                if (lead == 0xe0U) {
                    minimum = 0xa0U;
                } else if (lead == 0xedU) {
                    maximum = 0x9fU;
                } else if (lead == 0xf0U) {
                    minimum = 0x90U;
                } else if (lead == 0xf4U) {
                    maximum = 0x8fU;
                }
            }
            if (byte < minimum || byte > maximum) {
                // Replace one maximal subpart. The first byte that cannot continue this sequence
                // is deliberately left for the next iteration, so valid following text is kept.
                decoded.append(kUtf8Replacement);
                offset += index;
                malformed = true;
                break;
            }
        }
        if (malformed) { continue; }

        decoded.append(pending, offset, length);
        offset += length;
    }
    pending.clear();
    return decoded;
}

std::size_t longest_suffix_prefix(std::string_view text, std::string_view marker,
                                  bool allow_complete = false) {
    const std::size_t maximum = std::min(text.size(), marker.size());
    for (std::size_t size = maximum; size != 0; --size) {
        if (!allow_complete && size == marker.size()) { continue; }
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

constexpr bool is_quote_byte(char byte) noexcept {
    return byte == '"' || byte == '\'' || byte == '`';
}

// The first reasoning close in `text` (whose preceding byte is `before`) that ends the reasoning.
// With quoted_is_text, a close right after a quote mark or backtick and followed by anything but a
// line break is text: models quote the marker while thinking about code that contains it
// (7 times in 4 of 6 agent runs), and the end of the thinking is followed by a line
// break. Such a close at the end of `text` is `undecided` until its next byte arrives.
struct ReasoningClose {
    std::size_t position = std::string_view::npos;
    bool undecided       = false;
};

ReasoningClose find_reasoning_close(std::string_view text, char before, std::string_view close,
                                    bool quoted_is_text) noexcept {
    for (std::size_t at = text.find(close); at != std::string_view::npos;
         at             = text.find(close, at + 1)) {
        const char prior = at == 0 ? before : text[at - 1];
        if (!quoted_is_text || !is_quote_byte(prior)) { return {.position = at}; }
        const std::size_t next = at + close.size();
        if (next == text.size()) { return {.position = at, .undecided = true}; }
        if (text[next] == '\n') { return {.position = at}; }
    }
    return {};
}

// Tracks the canonical reasoning close inside the output. The pattern and its failure table are
// shared (OutputConventions); the tracker itself is two words, copied every preview.
struct PrefixExecutionTracker {
    // Returns the byte offset immediately after the first completed boundary in this token.
    [[nodiscard]] std::optional<std::size_t> feed(std::string_view bytes,
                                                  const OutputConventions& conventions) noexcept {
        const std::string_view boundary = conventions.canonical_close;
        if (!tracking || boundary.empty()) { return std::nullopt; }
        const auto& failure = conventions.canonical_close_failure;
        for (std::size_t offset = 0; offset < bytes.size(); ++offset) {
            const char byte = bytes[offset];
            while (matched != 0 && byte != boundary[matched]) { matched = failure[matched - 1U]; }
            if (byte == boundary[matched]) { ++matched; }
            if (matched != boundary.size()) { continue; }
            tracking = false;
            matched  = 0;
            return offset + 1U;
        }
        return std::nullopt;
    }

    std::size_t matched = 0;
    bool tracking       = false;
};

// Whether the newest tokens repeat one short period (StopPolicy::repetition_window): a model caught
// in a loop writes one token or one sentence again and again until its output limit. runs[p] counts
// the trailing tokens each equal to the one p before it, so the last runs[p] + p tokens have period
// p; a window of them with at least kMinimumRepeats periods is a loop. 64 compares a token.
struct RepetitionTracker {
    static constexpr std::uint32_t kMaximumPeriod  = kRepetitionMaximumPeriod;
    static constexpr std::uint32_t kMinimumRepeats = kRepetitionMinimumRepeats;
    std::array<TokenId, kMaximumPeriod> recent{};
    std::array<std::uint32_t, kMaximumPeriod + 1U> runs{};
    std::uint64_t seen = 0;

    [[nodiscard]] bool feed(TokenId token, std::uint32_t window) noexcept {
        bool loop = false;
        for (std::uint32_t period = 1; period <= kMaximumPeriod; ++period) {
            const bool same = seen >= period && recent[(seen - period) % kMaximumPeriod] == token;
            runs[period]    = same ? runs[period] + 1U : 0U;
            loop = loop || (period * kMinimumRepeats <= window && runs[period] + period >= window);
        }
        recent[seen % kMaximumPeriod] = token;
        ++seen;
        return loop;
    }
};

struct DecoderState {
    std::string utf8_pending;
    std::string think_marker_pending;
    // Output start where the model may open reasoning: bytes held while they match the opener.
    std::string open_pending;
    bool awaiting_open = false;
    // The opener's canonical suffix, dropped when it follows a model-opened marker.
    std::string reasoning_strip;
    std::array<std::string, 2> stop_pending;
    // Content start where the answer may begin with a header (OutputConventions::content_open):
    // bytes held while they match one.
    std::string content_header_pending;
    bool awaiting_content_header   = false;
    bool in_reasoning              = false;
    bool quoted_close_is_text      = false;
    // The last reasoning byte published before think_marker_pending.
    char reasoning_last            = '\n';
    bool strip_content_leading     = false;
    bool terminal                  = false;
    std::uint64_t decoded_bytes    = 0;
    std::uint32_t reasoning_tokens = 0;
    std::optional<std::uint32_t> matched_stop_order;
    RepetitionTracker repetition;
};

struct SemanticThinkingState {
    std::optional<std::uint32_t> budget;
    std::string close_pending;
    // Output start where the model may open its reasoning itself (Gemma 4 writes
    // <|channel>thought): bytes held while they match the opener.
    std::string open_pending;
    bool awaiting_open                  = false;
    std::uint32_t model_thinking_tokens = 0;
    std::uint32_t injected_tokens       = 0;
    bool in_reasoning                   = false;
    bool control_pending                = false;
    bool applied                        = false;
    bool loop_concluded                 = false;
    bool quoted_close_is_text           = false;
    // The byte before close_pending.
    char last = '\n';
};

void feed_semantic_thinking(SemanticThinkingState& state, std::string_view bytes,
                            std::string_view close) {
    if (!state.in_reasoning || bytes.empty()) { return; }
    state.close_pending.append(bytes);
    const ReasoningClose found =
        find_reasoning_close(state.close_pending, state.last, close, state.quoted_close_is_text);
    if (found.position != std::string_view::npos && !found.undecided) {
        state.close_pending.clear();
        state.in_reasoning    = false;
        state.control_pending = false;
        return;
    }
    const std::size_t drop =
        found.undecided
            ? found.position
            : state.close_pending.size() - longest_suffix_prefix(state.close_pending, close, true);
    if (drop != 0) { state.last = state.close_pending[drop - 1]; }
    state.close_pending.erase(0, drop);
}

// Feeds the output start while it may still be the model's own reasoning opener. Returns the
// bytes after the opener once the output has written it (the tracker is then in reasoning), and
// nothing while the opener is incomplete or once the output began otherwise.
std::optional<std::string> feed_semantic_open(SemanticThinkingState& state, std::string_view bytes,
                                              std::string_view open) {
    state.open_pending.append(bytes);
    if (state.open_pending.size() < open.size() && open.starts_with(state.open_pending)) {
        return std::nullopt;
    }
    state.awaiting_open = false;
    std::string pending = std::move(state.open_pending);
    state.open_pending.clear();
    if (!pending.starts_with(open)) { return std::nullopt; }
    state.in_reasoning = true;
    return pending.substr(open.size());
}

struct StopMatch {
    bool found                      = false;
    std::uint32_t committed_tokens  = 0;
    std::uint64_t byte_cut          = 0;
    std::uint32_t declaration_order = 0;
    PublishedOutput output;
};

bool stop_match_precedes(std::uint32_t committed_tokens, std::uint64_t byte_cut,
                         std::uint32_t declaration_order, const StopMatch& current) noexcept {
    if (!current.found) { return true; }
    if (committed_tokens != current.committed_tokens) {
        return committed_tokens < current.committed_tokens;
    }
    if (byte_cut != current.byte_cut) { return byte_cut < current.byte_cut; }
    return declaration_order < current.declaration_order;
}

std::size_t stop_hold_size(std::string_view text, OutputChannel channel, const StopPolicy& policy) {
    std::size_t hold = 0;
    for (const StopString& stop : policy.strings) {
        if (stop.channel != channel) { continue; }
        hold = std::max(hold, longest_suffix_prefix(text, stop.text));
    }
    return hold;
}

void feed_channel(DecoderState& state, OutputChannel channel, std::string_view text,
                  const StopPolicy& policy, PublishedOutput& emitted,
                  std::uint32_t committed_tokens, StopMatch* best_match) {
    if (text.empty()) { return; }
    std::string combined          = state.stop_pending[channel_index(channel)];
    const std::size_t old_pending = combined.size();
    combined.append(text);
    const std::uint64_t combined_start = state.decoded_bytes - old_pending;

    if (best_match != nullptr) {
        for (std::size_t declaration = 0; declaration < policy.strings.size(); ++declaration) {
            const StopString& stop = policy.strings[declaration];
            if (stop.channel != channel) { continue; }
            const std::size_t found = combined.find(stop.text);
            if (found == std::string::npos) { continue; }
            const std::uint64_t byte_cut = combined_start + found;
            const auto order             = static_cast<std::uint32_t>(declaration);
            if (!stop_match_precedes(committed_tokens, byte_cut, order, *best_match)) { continue; }

            PublishedOutput candidate = emitted;
            append_delta(candidate, channel, combined.substr(0, found));
            if (stop.include_in_output) { append_delta(candidate, channel, stop.text); }
            *best_match = StopMatch{.found             = true,
                                    .committed_tokens  = committed_tokens,
                                    .byte_cut          = byte_cut,
                                    .declaration_order = order,
                                    .output            = std::move(candidate)};
        }
    }

    const std::size_t hold = stop_hold_size(combined, channel, policy);
    append_delta(emitted, channel, combined.substr(0, combined.size() - hold));
    state.stop_pending[channel_index(channel)] = combined.substr(combined.size() - hold);
    state.decoded_bytes += text.size();
}

void close_channel(DecoderState& state, OutputChannel channel, PublishedOutput& emitted) {
    std::string& pending = state.stop_pending[channel_index(channel)];
    append_delta(emitted, channel, std::move(pending));
    pending.clear();
}

// Drops the answer's header from the start of the content: returns the text to publish, or
// nothing while the held bytes may still become a header.
std::optional<std::string> strip_content_header(DecoderState& state, std::string text,
                                                std::span<const std::string> headers) {
    if (!state.awaiting_content_header) { return text; }
    state.content_header_pending.append(text);
    const std::string& pending = state.content_header_pending;
    for (const std::string& header : headers) {
        if (pending.starts_with(header)) {
            state.awaiting_content_header = false;
            std::string rest              = pending.substr(header.size());
            state.content_header_pending.clear();
            return rest;
        }
    }
    for (const std::string& header : headers) {
        if (header.starts_with(pending)) { return std::nullopt; }
    }
    state.awaiting_content_header = false;
    std::string held              = std::move(state.content_header_pending);
    state.content_header_pending.clear();
    return held;
}

void feed_content(DecoderState& state, std::string text, const StopPolicy& policy,
                  PublishedOutput& emitted, std::uint32_t committed_tokens, StopMatch* best_match,
                  std::span<const std::string> headers = {}) {
    std::optional<std::string> stripped = strip_content_header(state, std::move(text), headers);
    if (!stripped) { return; }
    text = std::move(*stripped);
    if (state.strip_content_leading) {
        std::size_t begin = 0;
        while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
            ++begin;
        }
        text.erase(0, begin);
        if (!text.empty()) { state.strip_content_leading = false; }
    }
    feed_channel(state, OutputChannel::Content, text, policy, emitted, committed_tokens,
                 best_match);
}

void feed_decoded_text(DecoderState& state, std::string_view text, const StopPolicy& policy,
                       const OutputConventions& conventions, PublishedOutput& emitted,
                       std::uint32_t committed_tokens, StopMatch* best_match) {
    std::string opened;
    if (state.awaiting_open) {
        const std::string_view open = conventions.reasoning_open;
        state.open_pending.append(text);
        if (state.open_pending.size() < open.size() && open.starts_with(state.open_pending)) {
            return;
        }
        state.awaiting_open = false;
        if (!state.open_pending.starts_with(open)) {
            const std::string content = std::move(state.open_pending);
            state.open_pending.clear();
            feed_content(state, content, policy, emitted, committed_tokens, best_match,
                         conventions.content_open);
            return;
        }
        opened = state.open_pending.substr(open.size());
        state.open_pending.clear();
        state.in_reasoning            = true;
        state.awaiting_content_header = false;
        state.reasoning_strip         = conventions.reasoning_open_suffix;
        text                  = opened;
    }
    if (!state.in_reasoning) {
        feed_content(state, std::string(text), policy, emitted, committed_tokens, best_match,
                     conventions.content_open);
        return;
    }
    while (!state.reasoning_strip.empty() && !text.empty()) {
        if (text.front() != state.reasoning_strip.front()) {
            state.reasoning_strip.clear();
            break;
        }
        state.reasoning_strip.erase(0, 1);
        text.remove_prefix(1);
    }

    const std::string_view think_close = conventions.reasoning_close;
    state.think_marker_pending.append(text);
    const ReasoningClose found = find_reasoning_close(
        state.think_marker_pending, state.reasoning_last, think_close, state.quoted_close_is_text);
    const std::size_t marker = found.undecided ? std::string::npos : found.position;
    if (marker != std::string::npos) {
        feed_channel(state, OutputChannel::Reasoning,
                     std::string_view(state.think_marker_pending).substr(0, marker), policy,
                     emitted, committed_tokens, best_match);
        close_channel(state, OutputChannel::Reasoning, emitted);
        std::string content = state.think_marker_pending.substr(marker + think_close.size());
        state.think_marker_pending.clear();
        state.in_reasoning            = false;
        state.strip_content_leading   = true;
        state.awaiting_content_header = !conventions.content_open.empty();
        feed_content(state, std::move(content), policy, emitted, committed_tokens, best_match,
                     conventions.content_open);
        return;
    }

    // A quoted close is held until its next byte tells whether it ends the reasoning.
    const std::size_t safe =
        found.undecided ? found.position
                        : state.think_marker_pending.size() -
                              longest_suffix_prefix(state.think_marker_pending, think_close, true);
    feed_channel(state, OutputChannel::Reasoning,
                 std::string_view(state.think_marker_pending).substr(0, safe), policy, emitted,
                 committed_tokens, best_match);
    if (safe != 0) { state.reasoning_last = state.think_marker_pending[safe - 1]; }
    state.think_marker_pending.erase(0, safe);
}

void feed_token_bytes(DecoderState& state, std::string_view bytes, const StopPolicy& policy,
                      const OutputConventions& conventions, PublishedOutput& emitted,
                      std::uint32_t committed_tokens, StopMatch* best_match) {
    state.utf8_pending.append(bytes);
    const std::string text = consume_generated_utf8(state.utf8_pending);
    feed_decoded_text(state, text, policy, conventions, emitted, committed_tokens, best_match);
}

void terminalize(DecoderState& state, const StopPolicy& policy,
                 const OutputConventions& conventions, PublishedOutput& emitted,
                 std::uint32_t committed_tokens) {
    if (!state.utf8_pending.empty()) {
        // A token budget can end between byte-level tokens of one code point.
        // Publish the standard replacement character rather than an invalid
        // UTF-8 suffix; the logical token prefix remains exact.
        state.utf8_pending.clear();
        feed_decoded_text(state, kUtf8Replacement, policy, conventions, emitted, committed_tokens,
                          nullptr);
    }
    if (state.awaiting_open) {
        state.awaiting_open = false;
        feed_content(state, std::move(state.open_pending), policy, emitted, committed_tokens,
                     nullptr, conventions.content_open);
        state.open_pending.clear();
    }
    if (state.awaiting_content_header) {
        // The output ended on a prefix of a header: those bytes are content after all, unless
        // they are header markup the budget cut short (chat::is_cut_content_header).
        state.awaiting_content_header = false;
        std::string held              = std::move(state.content_header_pending);
        state.content_header_pending.clear();
        if (!chat::is_cut_content_header(held, conventions.content_open)) {
            feed_content(state, std::move(held), policy, emitted, committed_tokens, nullptr);
        }
    }
    if (state.in_reasoning) {
        // A quoted close held for its next byte that ends the output ended the reasoning.
        std::string_view held = state.think_marker_pending;
        if (state.quoted_close_is_text && held.ends_with(conventions.reasoning_close) &&
            !conventions.reasoning_close.empty()) {
            held.remove_suffix(conventions.reasoning_close.size());
        }
        feed_channel(state, OutputChannel::Reasoning, std::string(held), policy, emitted,
                     committed_tokens, nullptr);
        state.think_marker_pending.clear();
        close_channel(state, OutputChannel::Reasoning, emitted);
    } else {
        close_channel(state, OutputChannel::Content, emitted);
    }
    state.stop_pending = {};
    state.terminal     = true;
}

DecoderState terminal_state(DecoderState state) {
    state.utf8_pending.clear();
    state.think_marker_pending.clear();
    state.open_pending.clear();
    state.awaiting_open = false;
    state.content_header_pending.clear();
    state.awaiting_content_header = false;
    state.stop_pending = {};
    state.terminal     = true;
    return state;
}

} // namespace

class OutputSession::Impl {
public:
    Impl(std::shared_ptr<const text::TokenCodec> tokenizer_, StopPolicy policy_,
         OutputOptions output, OutputStart start, ThinkingControlOptions thinking,
         std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens_,
         std::shared_ptr<const chat::ToolCallOutputContract> tool_call_output_,
         std::shared_ptr<const OutputConventions> conventions_)
        : tokenizer(std::move(tokenizer_)), policy(std::move(policy_)),
          thinking_control_tokens(std::move(thinking_control_tokens_)),
          conventions(std::move(conventions_)),
          preserve_special(output.raw || output.preserve_special_tokens),
          split_reasoning(start.in_reasoning && !output.raw),
          tool_contract(output.raw ? nullptr : std::move(tool_call_output_)),
          tool_name_max_length(output.tool_name_max_length),
          tool_call_output(tool_contract, output.tool_name_max_length) {
        if (thinking.budget && *thinking.budget == 0) {
            throw std::invalid_argument("thinking budget must be positive");
        }
        if (thinking.budget && (!thinking_control_tokens || thinking_control_tokens->empty())) {
            throw std::invalid_argument(
                "this model's chat template defines no thinking-budget control");
        }
        const bool starts_in_reasoning = start.in_reasoning;
        state.in_reasoning             = split_reasoning;
        state.awaiting_open = !output.raw && !starts_in_reasoning && start.reasoning_open_allowed &&
                              !conventions->reasoning_open.empty();
        // An answer the model begins without reasoning may carry its header too.
        state.awaiting_content_header = !output.raw && !starts_in_reasoning &&
                                        !conventions->content_open.empty();
        state.quoted_close_is_text    = start.quoted_close_is_text;
        semantic.quoted_close_is_text = start.quoted_close_is_text;
        prefix_execution.tracking     = starts_in_reasoning;
        semantic.budget               = thinking.budget;
        configured_thinking_budget    = thinking.budget;
        answer_room_budget            = thinking.budget && thinking.answer_room;
        // The semantic tracker counts model-origin thinking tokens whenever the response starts
        // in thinking or the model opens it at the output start, capped or not: the count is
        // reported per request, and it must not depend on the presentation decoder, which raw
        // output disables. A budget additionally caps it. Its per-token work is one bounded scan
        // for the close marker.
        semantic.in_reasoning  = starts_in_reasoning;
        semantic.awaiting_open = !starts_in_reasoning && start.reasoning_open_allowed &&
                                 !conventions->reasoning_open.empty();
    }

    std::shared_ptr<const text::TokenCodec> tokenizer;
    StopPolicy policy;
    std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens;
    std::shared_ptr<const OutputConventions> conventions;
    bool preserve_special = false;
    bool split_reasoning  = false;
    DecoderState state;
    DecoderState preview_state;
    SemanticThinkingState semantic;
    SemanticThinkingState preview_semantic;
    std::optional<std::uint32_t> configured_thinking_budget;
    bool answer_room_budget = false;
    PrefixExecutionTracker prefix_execution;
    PrefixExecutionTracker preview_prefix_execution;
    std::optional<std::uint32_t> preview_execution_split_after;
    PublishedOutput preview_output;
    std::shared_ptr<const chat::ToolCallOutputContract> tool_contract;
    std::size_t tool_name_max_length = 0;
    chat::ToolCallOutputDecoder tool_call_output;
    // The published reasoning, kept only while tools are declared: a call the model emits inside
    // a thinking span it never closes is recovered from it at the terminal.
    std::string reasoning_text;
    FinishReason finish_reason         = FinishReason::None;
    FinishReason preview_finish_reason = FinishReason::None;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics tool_call_parse;
    bool preview_ready = false;
};

PublishedOutput::PublishedOutput(PublishedOutput&& other) noexcept
    : values_(std::move(other.values_)), size_(std::exchange(other.size_, 0)) {}

PublishedOutput& PublishedOutput::operator=(PublishedOutput&& other) noexcept {
    if (this != &other) {
        values_ = std::move(other.values_);
        size_   = std::exchange(other.size_, 0);
    }
    return *this;
}

void PublishedOutput::clear() noexcept {
    for (std::size_t index = 0; index < size_; ++index) { values_[index] = {}; }
    size_ = 0;
}

void PublishedOutput::push_back(OutputDelta value) {
    if (size_ == values_.size()) {
        throw std::logic_error("output decoder produced more than two channel transitions");
    }
    values_[size_++] = std::move(value);
}

OutputSession::OutputSession() noexcept                           = default;
OutputSession::~OutputSession()                                   = default;
OutputSession::OutputSession(OutputSession&&) noexcept            = default;
OutputSession& OutputSession::operator=(OutputSession&&) noexcept = default;

bool OutputConventions::structural(TokenId token) const noexcept {
    return std::binary_search(structural_tokens.begin(), structural_tokens.end(), token);
}

OutputSession::OutputSession(std::shared_ptr<const text::TokenCodec> tokenizer, StopPolicy policy,
                             OutputOptions output, OutputStart start,
                             ThinkingControlOptions thinking,
                             std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens,
                             std::shared_ptr<const chat::ToolCallOutputContract> tool_call_output,
                             std::shared_ptr<const OutputConventions> conventions)
    : impl_(std::make_unique<Impl>(std::move(tokenizer), std::move(policy), output, start,
                                   thinking, std::move(thinking_control_tokens),
                                   std::move(tool_call_output), std::move(conventions))) {}

runtime::OutputDecision OutputSession::preview_model(std::span<const TokenId> tokens,
                                                     std::uint32_t total_budget_remaining,
                                                     FinishReason limit_reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (impl_->semantic.control_pending) {
        throw std::logic_error("model output cannot advance while thinking control is pending");
    }
    if (tokens.empty()) {
        throw std::invalid_argument("cannot preview an empty generated-token round");
    }
    if (tokens.size() > total_budget_remaining) {
        throw std::invalid_argument("generated-token round exceeds the remaining budget");
    }
    if (limit_reason != FinishReason::OutputLimit &&
        limit_reason != FinishReason::ContextCapacity) {
        throw std::invalid_argument("generated-token budget has an invalid limit reason");
    }

    impl_->preview_state            = impl_->state;
    impl_->preview_semantic         = impl_->semantic;
    impl_->preview_prefix_execution = impl_->prefix_execution;
    impl_->preview_execution_split_after.reset();
    impl_->preview_output.clear();

    const auto complete = [&](std::uint32_t count, FinishReason reason,
                              runtime::ContinuationAction continuation =
                                  runtime::ContinuationAction::Decode) {
        if (reason != FinishReason::None) { impl_->preview_semantic.control_pending = false; }
        if (impl_->preview_execution_split_after && *impl_->preview_execution_split_after > count) {
            throw std::logic_error("prefix execution split exceeds the accepted token prefix");
        }
        impl_->preview_finish_reason = reason;
        impl_->preview_ready         = true;
        return runtime::OutputDecision{
            .accepted_tokens              = count,
            .finish_reason                = reason,
            .continuation                 = continuation,
            .prefix_execution_split_after = impl_->preview_execution_split_after,
        };
    };

    bool conclude_thinking = false;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const std::uint32_t count          = static_cast<std::uint32_t>(index + 1);
        const TokenId token                = tokens[index];
        const text::DecodedTokenView decoded = impl_->tokenizer->decoded_token(token);
        const OutputConventions& conventions = *impl_->conventions;

        if (const auto boundary = impl_->preview_prefix_execution.feed(decoded.bytes, conventions);
            boundary && *boundary == decoded.bytes.size()) {
            impl_->preview_execution_split_after = count;
        }

        if (impl_->preview_state.in_reasoning) { ++impl_->preview_state.reasoning_tokens; }
        SemanticThinkingState& semantic = impl_->preview_semantic;
        std::string_view thinking_bytes = decoded.bytes;
        std::optional<std::string> after_open;
        if (semantic.awaiting_open) {
            after_open = feed_semantic_open(semantic, decoded.bytes, conventions.reasoning_open);
            // The opener's own tokens are not thinking; a token that carries thinking past it is.
            thinking_bytes = after_open ? std::string_view(*after_open) : std::string_view{};
            const std::uint64_t room = total_budget_remaining - count;
            if (after_open && semantic.budget && room > *semantic.budget &&
                room - *semantic.budget < impl_->thinking_control_tokens->size() + 1U) {
                // The control suffix and a model token after it no longer fit past the budget, a
                // refusal at submit for a prompt that starts in thinking; here the model chose to
                // think, so the output limit ends the thinking instead.
                semantic.budget.reset();
            }
        }
        if (semantic.in_reasoning && !thinking_bytes.empty()) {
            ++semantic.model_thinking_tokens;
            if (semantic.budget && semantic.model_thinking_tokens > *semantic.budget) {
                throw std::logic_error("model output exceeded the licensed thinking budget");
            }
            feed_semantic_thinking(semantic, thinking_bytes, conventions.reasoning_close);
        }

        const bool stop_token =
            std::find(impl_->policy.token_ids.begin(), impl_->policy.token_ids.end(), token) !=
            impl_->policy.token_ids.end();
        DecoderState before_state;
        PublishedOutput before_output;
        if (stop_token && !impl_->policy.publish_stop_token) {
            before_state  = impl_->preview_state;
            before_output = impl_->preview_output;
        }

        StopMatch match;
        const bool hidden = !impl_->preserve_special && decoded.special &&
                            !conventions.structural(token);
        const std::string_view bytes = hidden ? std::string_view{} : decoded.bytes;
        feed_token_bytes(impl_->preview_state, bytes, impl_->policy, conventions,
                         impl_->preview_output, count, &match);

        if (match.found) {
            impl_->preview_state = terminal_state(std::move(impl_->preview_state));
            impl_->preview_state.matched_stop_order = match.declaration_order;
            impl_->preview_output                   = std::move(match.output);
            return complete(match.committed_tokens, FinishReason::StopString);
        }

        if (stop_token) {
            if (!impl_->policy.publish_stop_token) {
                impl_->preview_state  = std::move(before_state);
                impl_->preview_output = std::move(before_output);
            }
            terminalize(impl_->preview_state, impl_->policy, conventions, impl_->preview_output,
                        count);
            return complete(count, FinishReason::StopToken);
        }

        if (impl_->policy.repetition_window != 0 &&
            impl_->preview_state.repetition.feed(token, impl_->policy.repetition_window)) {
            // A loop inside the thinking ends the thinking, not the output: the round is read to
            // its end (a non-terminal round accepts every token) and the control suffix follows.
            const std::size_t suffix =
                impl_->thinking_control_tokens ? impl_->thinking_control_tokens->size() : 0U;
            if (impl_->policy.repetition_concludes_thinking && suffix != 0 &&
                impl_->preview_semantic.in_reasoning && !impl_->preview_semantic.applied &&
                static_cast<std::uint64_t>(total_budget_remaining) >=
                    static_cast<std::uint64_t>(tokens.size()) + suffix + 1U) {
                conclude_thinking                 = true;
                impl_->preview_state.repetition = RepetitionTracker{};
                continue;
            }
            terminalize(impl_->preview_state, impl_->policy, conventions, impl_->preview_output,
                        count);
            return complete(count, FinishReason::Repetition);
        }
    }

    const auto count = static_cast<std::uint32_t>(tokens.size());
    if (tokens.size() == total_budget_remaining) {
        terminalize(impl_->preview_state, impl_->policy, *impl_->conventions,
                    impl_->preview_output, count);
        return complete(count, limit_reason);
    }
    if (conclude_thinking && impl_->preview_semantic.in_reasoning) {
        impl_->preview_semantic.control_pending = true;
        impl_->preview_semantic.loop_concluded  = true;
        return complete(count, FinishReason::None, runtime::ContinuationAction::ApplyTargetControl);
    }
    if (impl_->preview_semantic.in_reasoning && impl_->preview_semantic.budget &&
        impl_->preview_semantic.model_thinking_tokens == *impl_->preview_semantic.budget) {
        impl_->preview_semantic.control_pending = true;
        return complete(count, FinishReason::None, runtime::ContinuationAction::ApplyTargetControl);
    }
    return complete(count, FinishReason::None);
}

std::uint32_t
OutputSession::model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept {
    if (impl_ == nullptr || !impl_->semantic.budget || impl_->semantic.applied) {
        return total_budget_remaining;
    }
    // Before the model has written its opener, a round of at most the budget cannot carry more
    // thinking tokens than the budget past it.
    if (impl_->semantic.awaiting_open) {
        return std::min(total_budget_remaining, *impl_->semantic.budget);
    }
    if (!impl_->semantic.in_reasoning) { return total_budget_remaining; }
    if (impl_->semantic.control_pending ||
        impl_->semantic.model_thinking_tokens >= *impl_->semantic.budget) {
        return 0;
    }
    return std::min(total_budget_remaining,
                    *impl_->semantic.budget - impl_->semantic.model_thinking_tokens);
}

std::span<const TokenId> OutputSession::pending_control_tokens() const noexcept {
    if (impl_ == nullptr || !impl_->semantic.control_pending || !impl_->thinking_control_tokens) {
        return {};
    }
    return *impl_->thinking_control_tokens;
}

runtime::OutputDecision OutputSession::preview_control(std::span<const TokenId> tokens,
                                                       std::uint32_t total_budget_remaining) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    const std::span<const TokenId> expected = pending_control_tokens();
    if (expected.empty() || tokens.size() != expected.size() ||
        !std::equal(tokens.begin(), tokens.end(), expected.begin())) {
        throw std::invalid_argument("thinking control preview requires the exact pending span");
    }
    if (tokens.size() > total_budget_remaining) {
        throw std::invalid_argument("thinking control span exceeds the remaining output budget");
    }

    impl_->preview_state            = impl_->state;
    impl_->preview_semantic         = impl_->semantic;
    impl_->preview_prefix_execution = impl_->prefix_execution;
    impl_->preview_execution_split_after.reset();
    impl_->preview_output.clear();
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const TokenId token                = tokens[index];
        const text::DecodedTokenView decoded = impl_->tokenizer->decoded_token(token);
        const OutputConventions& conventions = *impl_->conventions;
        if (const auto boundary = impl_->preview_prefix_execution.feed(decoded.bytes, conventions);
            boundary && *boundary == decoded.bytes.size()) {
            impl_->preview_execution_split_after = static_cast<std::uint32_t>(index + 1U);
        }
        if (impl_->preview_state.in_reasoning) { ++impl_->preview_state.reasoning_tokens; }
        feed_semantic_thinking(impl_->preview_semantic, decoded.bytes, conventions.reasoning_close);
        const bool hidden = !impl_->preserve_special && decoded.special &&
                            !conventions.structural(token);
        const std::string_view presentation_bytes = hidden ? std::string_view{} : decoded.bytes;
        feed_token_bytes(impl_->preview_state, presentation_bytes, impl_->policy, conventions,
                         impl_->preview_output, static_cast<std::uint32_t>(index + 1), nullptr);
    }
    if (impl_->preview_semantic.in_reasoning) {
        throw std::logic_error("canonical thinking control did not close the thinking phase");
    }
    if (impl_->split_reasoning && impl_->preview_state.in_reasoning) {
        throw std::logic_error("canonical thinking control did not close the reasoning channel");
    }
    impl_->preview_semantic.control_pending = false;
    impl_->preview_semantic.applied         = true;
    impl_->preview_semantic.injected_tokens = static_cast<std::uint32_t>(tokens.size());
    impl_->preview_finish_reason            = FinishReason::None;
    impl_->preview_ready                    = true;
    return runtime::OutputDecision{
        .accepted_tokens              = static_cast<std::uint32_t>(tokens.size()),
        .prefix_execution_split_after = impl_->preview_execution_split_after,
    };
}

void OutputSession::validate_generation_capacity(std::uint32_t effective_output_tokens) const {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (!impl_->semantic.budget || !impl_->semantic.in_reasoning ||
        effective_output_tokens <= *impl_->semantic.budget) {
        return;
    }
    const std::uint64_t remaining =
        static_cast<std::uint64_t>(effective_output_tokens) - *impl_->semantic.budget;
    const std::uint64_t required =
        static_cast<std::uint64_t>(impl_->thinking_control_tokens->size()) + 1U;
    if (remaining < required) {
        throw std::invalid_argument(
            "effective output capacity after the thinking budget must fit the complete control "
            "suffix and one post-close model token");
    }
}

runtime::OutputDecision OutputSession::preview_terminal(FinishReason reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (reason == FinishReason::None || reason == FinishReason::StopString ||
        reason == FinishReason::StopToken || reason == FinishReason::Repetition) {
        throw std::invalid_argument("invalid between-round terminal decoder reason");
    }
    impl_->preview_state            = impl_->state;
    impl_->preview_semantic         = impl_->semantic;
    impl_->preview_prefix_execution = impl_->prefix_execution;
    impl_->preview_execution_split_after.reset();
    impl_->preview_semantic.control_pending = false;
    impl_->preview_output.clear();
    terminalize(impl_->preview_state, impl_->policy, *impl_->conventions, impl_->preview_output,
                0);
    impl_->preview_finish_reason = reason;
    impl_->preview_ready         = true;
    return runtime::OutputDecision{.accepted_tokens = 0, .finish_reason = reason};
}

PublishedOutput OutputSession::commit_preview() {
    if (impl_ == nullptr || !impl_->preview_ready) { std::terminate(); }
    using std::swap;
    swap(impl_->state, impl_->preview_state);
    swap(impl_->semantic, impl_->preview_semantic);
    swap(impl_->prefix_execution, impl_->preview_prefix_execution);
    PublishedOutput output = std::move(impl_->preview_output);
    impl_->preview_output.clear();
    impl_->preview_ready = false;
    impl_->finish_reason = impl_->preview_finish_reason;

    for (OutputDelta& delta : output) {
        if (delta.channel == OutputChannel::Content) {
            delta.text = impl_->tool_call_output.feed(delta.text);
        } else if (impl_->tool_contract) {
            impl_->reasoning_text.append(delta.text);
        }
    }
    if (impl_->state.terminal) {
        chat::ToolCallOutputDecoder::Terminal terminal =
            impl_->tool_call_output.finish(impl_->finish_reason == FinishReason::OutputLimit ||
                                           impl_->finish_reason == FinishReason::ContextCapacity);
        impl_->tool_calls                              = std::move(terminal.tool_calls);
        impl_->tool_call_parse                         = terminal.diagnostics;
        // The model ended the turn with its stop token inside a thinking span it never closed, so
        // the content channel is empty. A complete call ending that reasoning is still the turn's
        // call: the same strict decoder must accept the reasoning's terminal call
        // region whole (no text after the calls, no bare or unclosed call), so a call merely quoted
        // before more reasoning is never taken. The reasoning is already published as reasoning and
        // stays there.
        if (impl_->tool_calls.empty() && impl_->tool_contract && impl_->state.in_reasoning &&
            impl_->finish_reason == FinishReason::StopToken && !impl_->reasoning_text.empty()) {
            chat::ToolCallOutputDecoder probe(impl_->tool_contract, impl_->tool_name_max_length);
            (void)probe.feed(impl_->reasoning_text);
            chat::ToolCallOutputDecoder::Terminal recovered = probe.finish();
            const ToolCallParseDiagnostics& strict          = recovered.diagnostics;
            if (!recovered.tool_calls.empty() && strict.trailing_content_bytes == 0 &&
                strict.unwrapped_calls == 0 && strict.closed_calls == 0) {
                impl_->tool_calls                               = std::move(recovered.tool_calls);
                impl_->tool_call_parse                          = recovered.diagnostics;
                impl_->tool_call_parse.recovered_from_reasoning = true;
            }
        }
        impl_->tool_call_parse.ended_inside_reasoning =
            impl_->state.in_reasoning && impl_->finish_reason == FinishReason::StopToken &&
            impl_->tool_calls.empty() && terminal.content.empty();
        impl_->reasoning_text.clear();
        impl_->reasoning_text.shrink_to_fit();
        if (!terminal.content.empty()) {
            OutputDelta* content = nullptr;
            for (OutputDelta& delta : output) {
                if (delta.channel == OutputChannel::Content) { content = &delta; }
            }
            if (content != nullptr) {
                content->text += terminal.content;
            } else {
                output.push_back(OutputDelta{.channel = OutputChannel::Content,
                                             .text    = std::move(terminal.content)});
            }
        }
    }
    return output;
}

std::vector<GeneratedToolCall> OutputSession::take_tool_calls() noexcept {
    return impl_ != nullptr ? std::move(impl_->tool_calls) : std::vector<GeneratedToolCall>{};
}

ToolCallParseDiagnostics OutputSession::tool_call_parse_diagnostics() const noexcept {
    return impl_ != nullptr ? impl_->tool_call_parse : ToolCallParseDiagnostics{};
}

std::uint32_t OutputSession::reasoning_tokens() const noexcept {
    return impl_ != nullptr ? impl_->state.reasoning_tokens : 0;
}

ThinkingBudgetStats OutputSession::thinking_stats() const noexcept {
    if (impl_ == nullptr) { return {}; }
    return ThinkingBudgetStats{
        .configured_budget     = impl_->configured_thinking_budget,
        .answer_room           = impl_->answer_room_budget,
        .model_thinking_tokens = impl_->semantic.model_thinking_tokens,
        .injected_tokens       = impl_->semantic.injected_tokens,
        .applied               = impl_->semantic.applied,
        .loop_concluded        = impl_->semantic.loop_concluded,
    };
}

std::optional<std::string> OutputSession::matched_stop_string() const {
    if (impl_ == nullptr || !impl_->state.matched_stop_order) { return std::nullopt; }
    const std::size_t index = *impl_->state.matched_stop_order;
    if (index >= impl_->policy.strings.size()) {
        throw std::logic_error("matched stop declaration is outside the stop policy");
    }
    return impl_->policy.strings[index].text;
}

} // namespace ninferno::models::qwen3_5
