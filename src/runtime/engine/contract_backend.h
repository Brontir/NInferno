#pragma once

// The one Backend implementation, instantiated per model contract. It adapts a family's loaded
// Instance (its Frontend, its Program and its capacity) to the erased Backend by running the
// family-agnostic request controllers over it: EngineCore for generation, CausalScoreCore for
// scoring. A family provides an Instance; it writes none of this.
//
// Instance requirements:
//   typename Instance::ModelContract            the Program-side contract (runtime_types.h)
//   Instance::frontend                          prepare / prepare_tokens / tokenize_text /
//                                               count_tokens / sampling_defaults /
//                                               media_cache_summary / grammar_vocabulary /
//                                               tool_call_structural_tag / make_output_session
//   Instance::capacity                          the longest admitted prompt
//   Instance::token_domain()                    public token count (grammar domain)
//   Instance::program, Instance::kv_capacity_resolution

#include "core/device.h"
#include "core/startup.h"
#include "runtime/engine/backend.h"
#include "runtime/engine/causal_score_core.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/engine_core.h"

#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>

namespace ninferno::runtime {

template <class Instance>
class ContractBackend final : public Backend {
public:
    using ModelContract  = typename Instance::ModelContract;
    using FamilyPrompt   = typename ModelContract::PreparedPrompt;
    using GenerationCore = EngineCore<Instance>;
    using ScoringCore    = CausalScoreCore<Instance>;

    ContractBackend(std::unique_ptr<Instance> instance, LoadSummary load,
                    ContextMachineCostModel context_cost, const EngineOptions& options,
                    DeviceContext& device)
        : instance_(std::move(instance)), load_(std::move(load)) {
        StartupPhaseScope finalize(options.startup_observer, StartupPhase::EngineFinalize);
        if (options.purpose == EnginePurpose::CausalScoring) {
            core_ = std::make_unique<ScoringCore>(*instance_, device);
        } else {
            core_ = std::make_unique<GenerationCore>(*instance_, device, options,
                                                     std::move(context_cost));
        }
        finalize.complete();
    }

    // The cores own worker threads that read the instance: they stop first.
    ~ContractBackend() override { core_.template emplace<std::monostate>(); }

    ContractBackend(const ContractBackend&)            = delete;
    ContractBackend& operator=(const ContractBackend&) = delete;

    [[nodiscard]] PreparedBox prepare(PromptInput input,
                                      const PreparationControl& control) const override {
        return box(instance_->frontend.prepare(std::move(input), control));
    }

    [[nodiscard]] PreparedBox prepare_tokens(std::vector<TokenId> token_ids,
                                             bool allow_prefix_identity) const override {
        return box(instance_->frontend.prepare_tokens(std::move(token_ids), allow_prefix_identity));
    }

    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const override {
        return instance_->frontend.tokenize_text(text);
    }

    [[nodiscard]] std::optional<TokenId> document_start_token() const override {
        return instance_->frontend.document_start_token();
    }

    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control) const override {
        return instance_->frontend.count_tokens(std::move(input), control);
    }

    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const override {
        return instance_->frontend.sampling_defaults();
    }

    [[nodiscard]] MediaCacheSummary media_cache_summary() const override {
        return instance_->frontend.media_cache_summary();
    }

    [[nodiscard]] std::uint32_t prompt_capacity() const override { return instance_->capacity; }

    [[nodiscard]] structured::GrammarVocabulary grammar_vocabulary() const override {
        return instance_->frontend.grammar_vocabulary(instance_->token_domain());
    }

    [[nodiscard]] std::string
    tool_call_structural_tag(std::span<const ToolCallConstraintTool> tools,
                             ToolCallRequirement requirement, std::string_view named_tool,
                             bool parallel_tool_calls) const override {
        return instance_->frontend.tool_call_structural_tag(tools, requirement, named_tool,
                                                            parallel_tool_calls);
    }

    [[nodiscard]] LoadSummary load_summary() const override { return load_; }

    [[nodiscard]] std::unique_ptr<SubmissionBox>
    submit(std::unique_ptr<PromptBox> prompt, PromptSummary summary, double prepare_seconds,
           ResolvedRequestOptions options, OutputConsumerMode consumer_mode,
           GenerationObservationOptions observation,
           std::chrono::steady_clock::time_point pending_deadline) override {
        auto* core = std::get_if<std::unique_ptr<GenerationCore>>(&core_);
        if (core == nullptr || *core == nullptr) {
            throw std::logic_error("Engine generation core is unavailable");
        }
        auto submission = (*core)->submit(unbox(std::move(prompt)), summary, prepare_seconds,
                                          std::move(options), consumer_mode, observation,
                                          pending_deadline);
        return std::make_unique<Submitted>(std::move(submission));
    }

    [[nodiscard]] std::vector<float> score(std::unique_ptr<PromptBox> prompt,
                                           std::uint32_t first_target,
                                           const ScoreDetailRequest& request,
                                           ScoreDetail& detail) override {
        auto* core = std::get_if<std::unique_ptr<ScoringCore>>(&core_);
        if (core == nullptr || *core == nullptr) {
            throw std::logic_error("Engine scoring core is unavailable");
        }
        return (*core)->score(unbox(std::move(prompt)), first_target, request, detail);
    }

    [[nodiscard]] MemorySummary memory_summary() const override {
        return visit_core<MemorySummary>([](const auto& core) { return core->memory_summary(); });
    }

    [[nodiscard]] RuntimeStats runtime_stats() const override {
        return visit_core<RuntimeStats>([](const auto& core) { return core->runtime_stats(); });
    }

    [[nodiscard]] bool is_available() const override {
        return std::visit(
            [](const auto& core) {
                if constexpr (std::is_same_v<std::remove_cvref_t<decltype(core)>, std::monostate>) {
                    return false;
                } else {
                    return core != nullptr && core->is_available();
                }
            },
            core_);
    }

    void reset_memory_peaks() noexcept override {
        std::visit(
            [](auto& core) {
                if constexpr (!std::is_same_v<std::remove_cvref_t<decltype(core)>,
                                              std::monostate>) {
                    if (core != nullptr) { core->reset_memory_peaks(); }
                }
            },
            core_);
    }

private:
    struct TypedPrompt final : PromptBox {
        explicit TypedPrompt(FamilyPrompt prompt) : value(std::move(prompt)) {}
        FamilyPrompt value;
    };

    class Submitted final : public SubmissionBox {
    public:
        explicit Submitted(typename GenerationCore::Submission submission)
            : submission_(std::move(submission)) {}

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
            return submission_.wait(sink, cancellation);
        }

    private:
        typename GenerationCore::Submission submission_;
    };

    [[nodiscard]] static PreparedBox box(FamilyPrompt prepared) {
        PreparedBox out;
        out.summary     = prepared.summary();
        out.preparation = prepared.preparation_stats();
        out.prompt      = std::make_unique<TypedPrompt>(std::move(prepared));
        return out;
    }

    [[nodiscard]] static FamilyPrompt unbox(std::unique_ptr<PromptBox> prompt) {
        auto* typed = dynamic_cast<TypedPrompt*>(prompt.get());
        if (typed == nullptr) {
            throw std::invalid_argument("PreparedPrompt was not prepared by this Engine's model");
        }
        return std::move(typed->value);
    }

    template <class Result, class Fn>
    [[nodiscard]] Result visit_core(Fn&& fn) const {
        return std::visit(
            [&](const auto& core) -> Result {
                if constexpr (std::is_same_v<std::remove_cvref_t<decltype(core)>, std::monostate>) {
                    throw std::logic_error("Engine core is unavailable");
                } else {
                    return fn(core);
                }
            },
            core_);
    }

    std::unique_ptr<Instance> instance_;
    LoadSummary load_;
    std::variant<std::monostate, std::unique_ptr<GenerationCore>, std::unique_ptr<ScoringCore>>
        core_;
};

} // namespace ninferno::runtime
