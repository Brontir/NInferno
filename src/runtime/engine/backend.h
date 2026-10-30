#pragma once

// The Engine's view of one loaded model family. ninferno::Engine owns the device and exactly one
// Backend; everything above it (the public Engine, serve, the protocols) is written once against
// this interface, and everything below it is instantiated per model contract (contract_backend.h).
// Which Backend a given artifact gets is data: its text component's model_type, looked up in the
// family table (family_registry.h).

#include "ninferno/score_detail.h"
#include "ninferno/types.h"
#include "runtime/contract/request.h"
#include "structured/vocabulary.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::runtime {

// A family's prepared prompt, owned by the Backend that prepared it. The public PreparedPrompt
// carries it opaquely; only that Backend reads it back.
class PromptBox {
public:
    virtual ~PromptBox() = default;
};

struct PreparedBox {
    std::unique_ptr<PromptBox> prompt;
    PromptSummary summary;
    PromptPreparationStats preparation;
};

// One submitted request as its consumer sees it: wait() streams or aggregates its output once.
class SubmissionBox {
public:
    virtual ~SubmissionBox() = default;
    virtual GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) = 0;
};

class Backend {
public:
    virtual ~Backend() = default;

    // The prompt side: the family's tokenizer, chat template and conventions.
    [[nodiscard]] virtual PreparedBox prepare(PromptInput input,
                                              const PreparationControl& control) const = 0;
    [[nodiscard]] virtual PreparedBox prepare_tokens(std::vector<TokenId> token_ids,
                                                     bool allow_prefix_identity) const = 0;
    [[nodiscard]] virtual std::vector<TokenId> tokenize_text(std::string_view text) const = 0;
    [[nodiscard]] virtual std::optional<TokenId> document_start_token() const = 0;
    [[nodiscard]] virtual std::uint32_t count_tokens(PromptInput input,
                                                     const PreparationControl& control) const = 0;
    [[nodiscard]] virtual ModelSamplingDefaults sampling_defaults() const = 0;
    [[nodiscard]] virtual MediaCacheSummary media_cache_summary() const = 0;
    // The longest prompt the loaded model admits.
    [[nodiscard]] virtual std::uint32_t prompt_capacity() const = 0;

    // Constrained decoding: every token of the model's public domain as the grammar sees it, and
    // the structural tag of the model's own tool-call markup.
    [[nodiscard]] virtual structured::GrammarVocabulary grammar_vocabulary() const = 0;
    [[nodiscard]] virtual std::string
    tool_call_structural_tag(std::span<const ToolCallConstraintTool> tools,
                             ToolCallRequirement requirement, std::string_view named_tool,
                             bool parallel_tool_calls) const = 0;

    [[nodiscard]] virtual LoadSummary load_summary() const = 0;

    // Execution. `prompt` must come from this Backend's prepare or prepare_tokens.
    [[nodiscard]] virtual std::unique_ptr<SubmissionBox>
    submit(std::unique_ptr<PromptBox> prompt, PromptSummary summary, double prepare_seconds,
           ResolvedRequestOptions options, OutputConsumerMode consumer_mode,
           GenerationObservationOptions observation,
           std::chrono::steady_clock::time_point pending_deadline) = 0;
    // `detail` receives what `request` asks for (Engine::score_tokens).
    [[nodiscard]] virtual std::vector<float> score(std::unique_ptr<PromptBox> prompt,
                                                   std::uint32_t first_target,
                                                   const ScoreDetailRequest& request,
                                                   ScoreDetail& detail) = 0;

    [[nodiscard]] virtual MemorySummary memory_summary() const = 0;
    [[nodiscard]] virtual RuntimeStats runtime_stats() const = 0;
    [[nodiscard]] virtual bool is_available() const = 0;
    virtual void reset_memory_peaks() noexcept = 0;
};

} // namespace ninferno::runtime
