#pragma once
#include "ninferno/types.h"
#include "runtime/contract/request.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninferno::text {
class TokenCodec;
} // namespace ninferno::text

namespace ninferno::chat {
struct ToolCallOutputContract;
} // namespace ninferno::chat

namespace ninferno::models::qwen3_5 {
class Frontend;

// The chat conventions the output presentation reads, resolved once per Frontend.
struct OutputConventions {
    // Reasoning section markers. An empty open marker means the model never opens one itself.
    std::string reasoning_open;
    std::string reasoning_open_suffix;
    std::string reasoning_close;
    // Bytes whose completion inside reasoning is a prefix-execution boundary (empty: none) and
    // the failure table of its prefix matcher.
    std::string canonical_close;
    std::vector<std::size_t> canonical_close_failure;
    // Headers the answer may begin with (after reasoning, or at the output start); the first
    // that matches is dropped from the content.
    std::vector<std::string> content_open;
    // Special tokens that the reasoning and tool-call parsers read; they reach the presentation.
    std::vector<TokenId> structural_tokens;  // sorted

    [[nodiscard]] bool structural(TokenId token) const noexcept;
};

// Where the prompt leaves the output channel: already inside reasoning, or at a fresh assistant
// body where the model may open reasoning itself.
struct OutputStart {
    bool in_reasoning           = false;
    bool reasoning_open_allowed = false;
    // A reasoning close the model writes right after a quote mark or backtick, and follows with
    // anything but a line break, is it quoting the marker (code or prose about the format), not
    // ending its reasoning. Off under an output constraint, whose grammar opens at the first close.
    bool quoted_close_is_text = false;
};

class PublishedOutput {
public:
    using iterator       = std::array<OutputDelta, 2>::iterator;
    using const_iterator = std::array<OutputDelta, 2>::const_iterator;

    PublishedOutput()                                  = default;
    PublishedOutput(const PublishedOutput&)            = default;
    PublishedOutput& operator=(const PublishedOutput&) = default;
    PublishedOutput(PublishedOutput&& other) noexcept;
    PublishedOutput& operator=(PublishedOutput&& other) noexcept;

    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    [[nodiscard]] iterator begin() noexcept { return values_.begin(); }

    [[nodiscard]] const_iterator begin() const noexcept { return values_.begin(); }

    [[nodiscard]] iterator end() noexcept { return values_.begin() + size_; }

    [[nodiscard]] const_iterator end() const noexcept { return values_.begin() + size_; }

    [[nodiscard]] OutputDelta& back() noexcept { return values_[size_ - 1]; }

    [[nodiscard]] const OutputDelta& back() const noexcept { return values_[size_ - 1]; }

    void clear() noexcept;
    void push_back(OutputDelta value);

private:
    std::array<OutputDelta, 2> values_{};
    std::size_t size_ = 0;
};

class OutputSession {
public:
    OutputSession() noexcept;
    ~OutputSession();
    OutputSession(OutputSession&&) noexcept;
    OutputSession& operator=(OutputSession&&) noexcept;

    OutputSession(const OutputSession&)            = delete;
    OutputSession& operator=(const OutputSession&) = delete;

    [[nodiscard]] runtime::OutputDecision preview_model(std::span<const TokenId> tokens,
                                                        std::uint32_t total_budget_remaining,
                                                        FinishReason limit_reason);
    [[nodiscard]] std::uint32_t
    model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept;
    [[nodiscard]] std::span<const TokenId> pending_control_tokens() const noexcept;
    [[nodiscard]] runtime::OutputDecision preview_control(std::span<const TokenId> tokens,
                                                          std::uint32_t total_budget_remaining);
    void validate_generation_capacity(std::uint32_t effective_output_tokens) const;
    [[nodiscard]] runtime::OutputDecision preview_terminal(FinishReason reason);
    [[nodiscard]] PublishedOutput commit_preview();
    [[nodiscard]] std::vector<GeneratedToolCall> take_tool_calls() noexcept;
    [[nodiscard]] ToolCallParseDiagnostics tool_call_parse_diagnostics() const noexcept;
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept;
    [[nodiscard]] ThinkingBudgetStats thinking_stats() const noexcept;
    [[nodiscard]] std::optional<std::string> matched_stop_string() const;

private:
    class Impl;
    OutputSession(std::shared_ptr<const text::TokenCodec> tokenizer, StopPolicy policy,
                  OutputOptions output, OutputStart start, ThinkingControlOptions thinking,
                  std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens,
                  std::shared_ptr<const chat::ToolCallOutputContract> tool_call_output,
                  std::shared_ptr<const OutputConventions> conventions);
    std::unique_ptr<Impl> impl_;

    friend class Frontend;
};

} // namespace ninferno::models::qwen3_5
