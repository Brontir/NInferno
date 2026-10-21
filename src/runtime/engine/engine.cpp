#include "ninferno/engine.h"

#include "core/device.h"
#include "core/device_class.h"
#include "core/host_memory.h"
#include "core/nvtx.h"
#include "core/sha256.h"
#include "core/startup.h"
#include "runtime/contract/sampling.h"
#include "runtime/contract/request.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/route_table.h"
#include "ops/generic/generic_gemm.h"
#include "runtime/engine/backend.h"
#include "runtime/engine/decision_fingerprint.h"
#include "runtime/engine/decision_sources.h"
#include "runtime/engine/family_registry.h"
#include "runtime/engine/model_instance.h"
#include "structured/constraint.h"
#include "structured/output_constraint.h"
#include "structured/schema_support.h"

#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace ninferno {
namespace {

DeviceContext initialize_device(const EngineOptions& options) {
    StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaInitialize);
    DeviceContext device(options.device);
    phase.complete();
    return device;
}

runtime::ResolvedRequestOptions resolve_request_options(const ModelSamplingDefaults& defaults,
                                                        SamplingMode mode, RequestOptions options) {
    if (options.execution.thinking.budget && *options.execution.thinking.budget == 0) {
        throw std::invalid_argument("thinking budget must be positive");
    }
    if (const std::uint32_t window = options.stop.repetition_window;
        window != 0 && (window < kRepetitionMinimumWindow || window > kRepetitionMaximumWindow)) {
        throw std::invalid_argument("repetition_window must be 0 or in [" +
                                    std::to_string(kRepetitionMinimumWindow) + ", " +
                                    std::to_string(kRepetitionMaximumWindow) + "]");
    }
    runtime::ResolvedRequestOptions resolved;
    resolved.execution.sampling =
        runtime::resolve_sampling(defaults, mode, options.execution.sampling);
    resolved.execution.requested_output_tokens = options.execution.requested_output_tokens;
    resolved.execution.allow_prefix_reuse      = options.execution.allow_prefix_reuse;
    resolved.execution.thinking                = options.execution.thinking;
    resolved.execution.speculation             = options.execution.speculation;
    resolved.stop                              = std::move(options.stop);
    resolved.output                            = options.output;
    resolved.constraint                        = std::move(options.constraint);
    return resolved;
}

runtime::HostFacts host_facts() {
    runtime::HostFacts host;
    host.logical_cores = std::thread::hardware_concurrency();
    try {
        if (const auto memory = read_host_memory()) { host.ram_bytes = memory->total_bytes; }
    } catch (const std::exception&) {}
    return host;
}

// The decision fingerprint of a constructed Engine (runtime/engine/decision_fingerprint.h): the
// facts, sources and environment are collected here, where CUDA and files may be read, and handed
// to the pure collectors. `requested`: the options as the caller passed them; `resolved`:
// Engine::options() after the plan.
void fingerprint_decisions_or_throw(LoadSummary& load, const EngineOptions& requested,
                                    const EngineOptions& resolved, const MemorySummary& memory) {
    runtime::DecisionInputs in;
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) != cudaSuccess) {
        (void)cudaGetLastError();
        driver_version = 0;
    }
    in.facts   = runtime::facts_of(device_class(resolved.device), build_features(), driver_version,
                                   host_facts());
    in.build   = runtime::build_of();
    in.options = runtime::options_of(
        requested, load.artifact_id.empty() ? std::string("unreadable") : load.artifact_id,
        load.prefill_signature, runtime::drafter_identity(requested.speculative.drafter));
    for (const auto& [key, value] : requested.speculative.decision_sources) {
        in.sources[key] = value;
    }
    const ops::RouteTableStatus routes = ops::route_table_status();
    std::string route_table;
    if (routes.loaded) {
        in.sources["route_table"] =
            utf8_file_name(std::filesystem::path(routes.path)) + " sha256:" + routes.content_sha256;
        in.sources["route_table.class"] = routes.gpu_class;
        route_table                     = routes.content_sha256;
        if (!routes.lender_sha256.empty()) {
            in.sources["route_table.lenders"] = routes.lender_sha256;
            route_table += " + " + routes.lender_sha256;
        }
    } else {
        in.sources["route_table"] = "none";
    }
    for (auto& [key, value] : ops::detail::generic::dense_tune_sources()) {
        in.sources[key] = std::move(value);
    }
    for (auto& [key, value] : ops::moe_tune_sources()) { in.sources[key] = std::move(value); }
    in.sources["context_cost.presets"] =
        runtime::context_cost_presets_source(requested.context_cost.preset_path);
    in.env   = runtime::env_of([](const char* name) -> std::optional<std::string> {
        const char* value = std::getenv(name);
        return value != nullptr ? std::optional<std::string>(value) : std::nullopt;
    });
    in.plan  = runtime::plan_of(resolved, memory, ops::route_table_breakpoints(17), route_table);
    in.taint    = runtime::taint_of(in);
    in.observed = runtime::observed_of(memory);
    in.observed.merge(runtime::observed_of(device_class(resolved.device)));
    const runtime::DecisionFingerprint fingerprint = runtime::decision_fingerprint(in);
    load.decision_fingerprint                      = fingerprint.value;
    load.decision_manifest                         = fingerprint.manifest;
    load.decision_canonical                        = fingerprint.canonical;
    load.decision_observed                         = fingerprint.observed;
    load.metadata.decision_fingerprint             = fingerprint.value;
    load.metadata.decision_manifest                = fingerprint.manifest;
    load.decision_taint.clear();
    for (const auto& [what, detail] : in.taint) {
        load.decision_taint.push_back(what + "=" + detail);
    }
}

// Fingerprinting never stops a start: an input that cannot be read (a file, a name the platform
// cannot convert) leaves the fingerprint "unreadable" and says why in the taint.
void fingerprint_decisions(LoadSummary& load, const EngineOptions& requested,
                           const EngineOptions& resolved, const MemorySummary& memory) {
    std::string why;
    try {
        fingerprint_decisions_or_throw(load, requested, resolved, memory);
        return;
    } catch (const std::exception& error) { why = error.what(); } catch (...) {
        why = "unknown error";
    }
    load.decision_fingerprint          = "unreadable";
    load.decision_manifest             = "unreadable";
    load.metadata.decision_fingerprint = "unreadable";
    load.metadata.decision_manifest    = "unreadable";
    load.decision_canonical.clear();
    load.decision_observed.clear();
    load.decision_taint = {"fingerprint=unreadable: " + why};
}

std::string context_capacity_error(std::size_t prompt_tokens, std::uint32_t max_context) {
    return "prepared prompt has " + std::to_string(prompt_tokens) +
           " tokens, exceeding Engine max_context " + std::to_string(max_context);
}

} // namespace

class PreparedPrompt::Impl {
public:
    Impl(PromptSummary prompt_summary, PromptPreparationStats preparation, SamplingMode mode,
         std::unique_ptr<runtime::PromptBox> prepared)
        : summary(std::move(prompt_summary)), prepare(std::move(preparation)), sampling_mode(mode),
          value(std::move(prepared)) {}

    PromptSummary summary;
    PromptPreparationStats prepare;
    SamplingMode sampling_mode = SamplingMode::Thinking;
    std::unique_ptr<runtime::PromptBox> value;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PromptPreparationStats& PreparedPrompt::preparation_stats() const noexcept {
    static const PromptPreparationStats empty;
    return impl_ != nullptr ? impl_->prepare : empty;
}

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }

class GenerationHandle::Impl {
public:
    Impl(std::shared_ptr<void> keep_alive, std::unique_ptr<runtime::SubmissionBox> submission,
         ResolvedSamplingParameters sampling, OutputConsumerMode consumer_mode)
        : keep_alive_(std::move(keep_alive)), submission_(std::move(submission)),
          sampling_(sampling), consumer_mode_(consumer_mode) {}

    // A sink for a streamed request, none for an aggregated one: checked before the handle is
    // consumed, so a wrong call leaves the request running and the handle waitable.
    void check_sink(const OutputSink* sink) const {
        if ((consumer_mode_ == OutputConsumerMode::Streaming) != (sink != nullptr)) {
            throw std::invalid_argument(
                "GenerationHandle wait sink does not match its submitted consumer mode");
        }
    }

    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return submission_->wait(sink, cancellation);
    }

    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept {
        return sampling_;
    }

private:
    // Declared first, destroyed last: the Engine outlives the submission that reads its core.
    std::shared_ptr<void> keep_alive_;
    std::unique_ptr<runtime::SubmissionBox> submission_;
    ResolvedSamplingParameters sampling_;
    OutputConsumerMode consumer_mode_ = OutputConsumerMode::Aggregate;
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return impl_ != nullptr ? impl_->resolved_sampling() : empty;
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    impl_->check_sink(sink);
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(sink, cancellation);
}

class Engine::Impl {
public:
    explicit Impl(EngineOptions engine_options)
        : Impl(engine_options, runtime::normalize_engine_options(engine_options)) {}

    // `requested`: the options as the caller passed them, which the decision fingerprint records.
    Impl(const EngineOptions& requested, EngineOptions normalized)
        : options(std::move(normalized)), device(initialize_device(options)) {
        nvtx::ScopedRange load_range(nvtx::Name::EngineLoad, nvtx::Category::Runtime);
        backend             = runtime::make_backend(options, device);
        // Engine::options() reports the effective values: the resolved cached state count, an
        // automatic max_context, the prefill chunk and an automatic Host tier too.
        const MemorySummary memory               = backend->memory_summary();
        options.max_context                      = memory.max_context;
        options.prefill_chunk                    = memory.prefill_chunk;
        options.context_cache.device_state_slots = memory.cached_device_states;
        if (memory.host_tier_automatic) {
            options.context_cache.host_state_slots       = memory.host_state_capacity_slots;
            options.context_cache.host_kv_capacity_bytes = memory.host_kv_capacity_bytes;
        }
        load                = backend->load_summary();
        load.cuda_sync_mode = device.sync_mode();
        sampling_defaults   = backend->sampling_defaults();
        fingerprint_decisions(load, requested, options, memory);
    }

    ~Impl() noexcept {
        device.bind_to_current_thread_noexcept();
        backend.reset();
        try {
            device.synchronize();
        } catch (...) {}
    }

    // Built on first use: the grammar vocabulary covers the whole token domain, so an Engine that
    // never serves a constrained request pays nothing for it.
    [[nodiscard]] structured::ConstraintCompiler& constraint_compiler() const {
        std::lock_guard lock(constraint_mutex);
        if (!compiler) {
            compiler =
                std::make_unique<structured::ConstraintCompiler>(backend->grammar_vocabulary());
        }
        return *compiler;
    }

    EngineOptions options;
    DeviceContext device;
    std::unique_ptr<runtime::Backend> backend;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    mutable std::mutex constraint_mutex;
    mutable std::unique_ptr<structured::ConstraintCompiler> compiler;
};

Engine::Engine(EngineOptions options) {
    StartupObserver startup_observer = options.startup_observer;
    StartupPhaseScope startup_phase(startup_observer, StartupPhase::EngineStartup);
    impl_ = std::make_shared<Impl>(std::move(options));
    startup_phase.complete();
}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input, const PreparationControl& control) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime);
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    runtime::PreparedBox prepared = impl_->backend->prepare(std::move(input), control);
    const PromptSummary& info     = prepared.summary;
    const SamplingMode sampling_mode =
        info.thinking() ? SamplingMode::Thinking : SamplingMode::NonThinking;
    if (info.prompt_tokens > impl_->backend->prompt_capacity()) {
        throw std::logic_error("target Frontend admitted a prompt beyond Engine capacity");
    }
    return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
        info, prepared.preparation, sampling_mode, std::move(prepared.prompt)));
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime,
                                    static_cast<std::uint64_t>(token_ids.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    const std::uint32_t capacity = impl_->backend->prompt_capacity();
    if (token_ids.size() > capacity) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           context_capacity_error(token_ids.size(), capacity));
    }
    runtime::PreparedBox prepared =
        impl_->backend->prepare_tokens(std::move(token_ids), allow_prefix_identity);
    if (prepared.summary.prompt_tokens > capacity) {
        throw std::logic_error("target Frontend admitted prompt tokens beyond capacity");
    }
    return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(prepared.summary,
                                                                 prepared.preparation,
                                                                 SamplingMode::Thinking,
                                                                 std::move(prepared.prompt)));
}

std::vector<TokenId> Engine::tokenize_text(std::string_view text) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->backend->tokenize_text(text);
}

std::vector<float> Engine::score_tokens(std::vector<TokenId> tokens, std::uint32_t first_target) {
    ScoreDetail none;
    return score_tokens(std::move(tokens), first_target, ScoreDetailRequest{}, none);
}

std::vector<float> Engine::score_tokens(std::vector<TokenId> tokens, std::uint32_t first_target,
                                        const ScoreDetailRequest& request, ScoreDetail& detail) {
    nvtx::ScopedRange score_range(nvtx::Name::Score, nvtx::Category::Scoring,
                                  static_cast<std::uint64_t>(tokens.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::CausalScoring) {
        throw std::logic_error("score_tokens requires a CausalScoring Engine");
    }
    if (tokens.size() < 2 || tokens.size() > impl_->options.max_context) {
        throw std::invalid_argument("score_tokens token count must be in [2,max_context]");
    }
    if (first_target == 0 || first_target >= tokens.size()) {
        throw std::invalid_argument("score_tokens first_target must be in [1,token_count-1]");
    }
    if (request.top_k > 64 || request.given_k > 64) {
        throw std::invalid_argument("score_tokens: top_k and given_k must be at most 64");
    }
    const std::size_t targets = tokens.size() - first_target;
    if (request.given_ids.size() != targets * request.given_k) {
        throw std::invalid_argument("score_tokens: given_ids must hold given_k ids per target");
    }
    PreparedPrompt prompt      = prepare_tokens(std::move(tokens), false);
    const std::size_t expected = prompt.summary().prompt_tokens - first_target;
    detail                     = {};
    std::vector<float> result =
        impl_->backend->score(std::move(prompt.impl_->value), first_target, request, detail);
    if (result.size() != expected || detail.top_ids.size() != expected * request.top_k ||
        detail.top_logprobs.size() != expected * request.top_k ||
        detail.given_logprobs.size() != expected * request.given_k) {
        throw std::logic_error("target Program returned an invalid causal score count");
    }
    return result;
}

std::uint32_t Engine::count_tokens(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->backend->count_tokens(std::move(input), control);
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->sampling_defaults;
}

std::shared_ptr<const OutputConstraint>
Engine::compile_output_constraint(const OutputConstraintSpec& spec) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::Generation) {
        throw std::logic_error("compile_output_constraint requires a Generation Engine");
    }
    structured::ConstraintCompiler& compiler = impl_->constraint_compiler();
    if (spec.kind != OutputConstraintKind::ToolCalls) {
        return std::make_shared<const OutputConstraint>(compiler.compile(spec), &compiler);
    }
    // Tool calls are model markup: the Frontend renders it as a structural tag after every strict
    // argument schema passed the same support policy as a response_format schema.
    std::vector<ToolCallConstraintTool> tools = spec.tools;
    for (ToolCallConstraintTool& tool : tools) {
        if (!tool.strict) { continue; }
        const nlohmann::ordered_json schema =
            nlohmann::ordered_json::parse(tool.parameters_json, nullptr, false);
        if (schema.is_discarded() || !schema.is_object()) {
            throw OutputConstraintError("invalid_schema", "tool '" + tool.name +
                                                              "' parameters are not a JSON object");
        }
        try {
            tool.parameters_json = structured::normalize_tool_parameters_schema(schema).dump();
        } catch (const OutputConstraintError& error) {
            throw OutputConstraintError(error.code(),
                                        "tool '" + tool.name + "' parameters: " + error.what());
        }
    }
    OutputConstraintSpec rendered = spec;
    rendered.kind                 = OutputConstraintKind::StructuralTag;
    rendered.tools.clear();
    // With an answer schema the text answer is that JSON: the markup alternative is calls only.
    const ToolCallRequirement requirement =
        !spec.answer_schema.empty() && spec.tool_requirement == ToolCallRequirement::Optional
            ? ToolCallRequirement::Required
            : spec.tool_requirement;
    try {
        rendered.definition = impl_->backend->tool_call_structural_tag(
            tools, requirement, spec.named_tool, spec.parallel_tool_calls);
    } catch (const std::invalid_argument& error) {
        throw OutputConstraintError("invalid_tool_constraint", error.what());
    }
    return std::make_shared<const OutputConstraint>(compiler.compile(rendered), &compiler);
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                OutputConsumerMode consumer_mode,
                                GenerationObservationOptions observation,
                                std::chrono::steady_clock::time_point pending_deadline) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::Generation) {
        throw std::logic_error("submit requires a Generation Engine");
    }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }
    if (observation.live_timings) { observation.phase_timings = true; }
    if (consumer_mode != OutputConsumerMode::Streaming &&
        (observation.live_timings || observation.prompt_progress)) {
        throw std::invalid_argument("live generation observations require a Streaming consumer");
    }

    if (const auto& constraint = options.constraint.compiled) {
        {
            std::lock_guard lock(impl_->constraint_mutex);
            if (constraint->compiled() == nullptr || constraint->owner() != impl_->compiler.get()) {
                throw std::invalid_argument("output constraint was not compiled by this Engine");
            }
        }
        const bool visible = options.output.raw || options.output.preserve_special_tokens;
        if (structured::vocabulary_view(*constraint->compiled()).special_tokens_visible != visible) {
            throw std::invalid_argument(
                "output constraint special-token visibility does not match OutputOptions");
        }
        // A constrained output ends where its grammar does: the grammar admits the checkpoint's
        // stop tokens only once it is complete, so they stay stops under ignore_eos too. Without
        // them the accepted stop token did not end the output, and the next round's mask failed
        // the request (HTTP 500 structured_output_violation).
        options.stop.include_model_defaults = true;
    }
    runtime::ResolvedRequestOptions resolved_options = resolve_request_options(
        impl_->sampling_defaults, prompt.impl_->sampling_mode, std::move(options));
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;

    const PromptSummary prompt_summary = prompt.impl_->summary;
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            context_capacity_error(prompt_summary.prompt_tokens, impl_->options.max_context));
    }
    const double prepare_seconds = prompt.impl_->prepare.seconds;
    if (resolved_options.execution.requested_output_tokens == 0) {
        struct ImmediateSubmission final : runtime::SubmissionBox {
            GenerationResult result;

            GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
                // The sink matches the consumer mode (GenerationHandle::Impl::check_sink).
                if (cancellation.requested()) { result.finish_reason = FinishReason::Cancelled; }
                // A stream gets its one GenerationStart (engine.h) even with nothing to emit.
                if (sink != nullptr) { sink->start(GenerationStart{.prompt = result.prompt}); }
                return std::move(result);
            }
        };

        auto immediate = std::make_unique<ImmediateSubmission>();

        immediate->result.prompt                     = prompt_summary;
        immediate->result.finish_reason              = FinishReason::OutputLimit;
        immediate->result.thinking.configured_budget = resolved_options.execution.thinking.budget;
        immediate->result.timings.prepare_seconds    = prepare_seconds;
        immediate->result.timings.total_seconds      = prepare_seconds;
        prompt.impl_.reset();
        return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
            impl_, std::move(immediate), resolved_sampling, consumer_mode));
    }

    auto submission = impl_->backend->submit(std::move(prompt.impl_->value), prompt_summary,
                                             prepare_seconds, std::move(resolved_options),
                                             consumer_mode, observation, pending_deadline);
    return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
        impl_, std::move(submission), resolved_sampling, consumer_mode));
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    const OutputConsumerMode consumer_mode =
        sink != nullptr ? OutputConsumerMode::Streaming : OutputConsumerMode::Aggregate;
    return submit(std::move(prompt), std::move(options), consumer_mode, {})
        .wait(sink, cancellation);
}

const EngineOptions& Engine::options() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->options;
}

LoadSummary Engine::load_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->load;
}

ModelMetadata Engine::model_metadata() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->load.metadata;
}

MemorySummary Engine::memory_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->backend->memory_summary();
}

MediaCacheSummary Engine::media_cache_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->backend->media_cache_summary();
}

RuntimeStats Engine::runtime_stats() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->backend->runtime_stats();
}

bool Engine::is_available() const {
    return impl_ != nullptr && impl_->backend != nullptr && impl_->backend->is_available();
}

void Engine::reset_memory_peaks() noexcept {
    if (impl_ == nullptr || impl_->backend == nullptr) { return; }
    impl_->backend->reset_memory_peaks();
}

} // namespace ninferno
