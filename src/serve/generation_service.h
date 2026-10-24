#pragma once

// Product-side adapter from one protocol-neutral generation request to the public Engine. Wire
// adapters normalize before this layer and render IDs, usage, and response events after it.

#include "ninferno/engine.h"
#include "product/speculation/speculation.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::serve {

struct RequestLifetime;
struct RequestCapacity;

struct GenerationMetrics {
    double prepare_seconds         = 0.0;
    double ttft_seconds            = 0.0;
    double vision_seconds          = 0.0;
    double prefill_seconds         = 0.0;
    double decode_seconds          = 0.0;
    double prompt_wall_seconds     = 0.0;
    double generation_wall_seconds = 0.0;
    double total_seconds           = 0.0;
    ninferno::GenerationEngineTiming engine_timing;

    SpeculativeBackend speculative_backend    = SpeculativeBackend::None;
    std::uint32_t speculative_draft_window    = 0;
    std::uint64_t speculative_rounds          = 0;
    std::uint64_t speculative_draft_tokens    = 0;
    std::uint64_t speculative_accepted_tokens = 0;
    std::uint64_t speculative_fallback_steps  = 0;
    // Of those, the DFlash rounds a constrained request verified without drafts because its
    // step's declared mask time filled the masks before the launch (a client's EBNF grammar past
    // its reasoning).
    std::uint64_t speculative_constraint_fallback_steps = 0;
    std::vector<std::uint64_t> speculative_accepted_per_position;
    // The speculation this request ran: its configuration ("plain" when none), how that was chosen
    // ("fixed", "thinking" by --spec auto's profile; "explicit" by --spec; "startup" when --spec
    // auto runs nothing), why, and what the profile would have chosen when the engine could not.
    // Every request carries them (StartupSpeculation when the server does not choose).
    std::string speculative_config;
    std::string speculative_selection;
    std::string speculative_reason;
    std::string speculative_note;
    double speculative_floor                     = 0.0;
    std::uint64_t speculative_plain_steps    = 0;
    std::uint64_t speculative_wide_rounds    = 0;
    std::uint64_t speculative_ungraphed      = 0; // rounds run without their CUDA graph
    std::uint64_t speculative_capped_rounds  = 0; // verified below its depth by the load bound
    std::uint64_t speculative_copy_rounds    = 0; // verified a copy proposal (--ngram-draft-tokens)
    std::uint64_t speculative_copy_accepted  = 0; // the drafts those accepted
    std::uint64_t speculative_trips_held     = 0; // watchdog verdicts a shared launch held
    bool speculative_load_plain              = false; // a load bound of 0 turned it plain
    bool speculative_context_plain           = false; // its prompt passed the context bound
    // The fallback its speculation took (speculation_fallback()).
    std::string speculative_fallback;
    bool speculative_watchdog_tripped        = false;
    std::uint64_t speculative_watchdog_round = 0;
    std::uint32_t speculative_step_downs     = 0; // watchdog trips that went shallower
    std::uint32_t speculative_final_draft_window = 0;
    // Constrained decoding: mask rows the host computed and the host time spent computing them.
    std::uint64_t constraint_masked_positions = 0;
    double constraint_mask_host_seconds       = 0.0;
    std::uint32_t prefix_cache_hit_tokens     = 0;
    ninferno::PrefixReusePath prefix_reuse_path = ninferno::PrefixReusePath::Root;
    ninferno::MaterializationDiagnostics materialization;
};

// The fallback a request's speculation took, the first that applies: "context_plain" (its prompt
// passed the context bound, SpeculativeOptions::context_limit: plain from its first token),
// "load_plain" (a load bound of 0), "watchdog_plain" (the acceptance watchdog stopped its
// drafting), "step_down" (the watchdog moved it shallower), else the server's own since startup
// (`startup`, StartupSpeculation::fallback), else "none".
[[nodiscard]] std::string speculation_fallback(const GenerationMetrics& metrics,
                                               std::string_view startup);

// The speculation the server runs since startup, as /v1/models (meta.speculation) and the
// startup log show it before any request: its configuration ("plain" when none; the menu's
// primary under --spec auto), how it was chosen ("fixed" or "thinking" by the profile,
// "explicit" by --spec, "startup" when --spec auto runs nothing), why, and the fallback startup
// took: "fit_fallback" (the profile's choice did not fit and another runs), "startup_plain"
// (nothing that drafts fit: plain decode), else "none".
struct StartupSpeculation {
    std::string config = "plain";
    std::string selection;
    std::string reason;
    std::string fallback = "none";
    double floor        = 0.0; // the watchdog floor the engine runs without a served menu
};

struct GenerationOutcome {
    std::string text;
    std::string reasoning;
    std::vector<ninferno::GeneratedToolCall> tool_calls;
    ninferno::ToolCallParseDiagnostics tool_call_parse;
    int prompt_tokens     = 0;
    int completion_tokens = 0;
    int reasoning_tokens  = 0;
    ninferno::ThinkingBudgetStats thinking;
    ninferno::FinishReason finish_reason = ninferno::FinishReason::OutputLimit;
    std::optional<std::string> matched_stop_string;
    GenerationMetrics metrics;
};

struct StreamSink {
    std::function<void(const ninferno::GenerationStart& start)> on_start;
    std::function<void(const ninferno::PromptProgress& progress)> on_progress;
    std::function<void(const ninferno::GenerationTimingObservation& timing)> on_timing;
    std::function<void(const std::string& delta_text)> on_content;
    std::function<void(const std::string& delta_text)> on_reasoning;
    std::function<bool()> is_cancelled;
};

enum class GenerationConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

// Translate Engine request failures into the shared protocol-neutral HTTP error contract.
// The start of every context-overflow message: phrases agent harnesses match to compact and retry.
inline constexpr std::string_view kContextOverflowPhrase =
    "prompt is too long: the request exceeds the available context size; ";

ApiError request_error_to_api_error(const ninferno::RequestError& exception);

// Preparation ends by synchronously submitting the owning prompt to the Engine FIFO. The returned
// request keeps its ingress/response lifetime reservation until the HTTP response is released and
// is consumed exactly once by run().
struct PreparedRequest {
    ninferno::GenerationHandle generation;
    ninferno::ResolvedSamplingParameters sampling;
    double prepare_seconds     = 0.0;
    double acquisition_seconds = 0.0;
    PromptPreparationStats preparation;
    int prompt_tokens    = 0;
    bool enable_thinking = true;
    std::optional<std::uint32_t> thinking_budget;
    std::optional<ninferno::ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::shared_ptr<RequestLifetime> lifetime;
    // The speculation chosen for the request (GenerationMetrics' speculative_* fields).
    std::string speculation_config;
    std::string speculation_selection;
    std::string speculation_reason;
    std::string speculation_note;
    double speculation_floor = 0.0;
    std::string speculation_fallback = "none"; // the server's (StartupSpeculation::fallback)
    std::uint32_t plan_level         = 0;      // the plan ladder's rung (PlanSummary::rung)
};

// Why --spec auto decodes plainly: `refused` (the choice's label) did not fit, the last refusal
// `why`, and with an explicit --prefill-chunk what an automatic one would have tried first.
[[nodiscard]] std::string auto_plain_reason(const std::string& refused, const std::string& why,
                                            bool prefill_chunk_explicit);
// The refusal when --spec auto fell back to plain decode and plain decode (`plain`, its refusal)
// does not fit either.
[[nodiscard]] std::string auto_then_plain_refusal(const std::string& refused,
                                                  const std::string& plain);

// The Engine options a server's options give before any speculation is chosen (the speculative
// settings as given: --spec auto replaces them with the profile's choice): what a start and the
// plan without the weights (serve/plan_offline.h) both build from.
[[nodiscard]] ninferno::EngineOptions engine_options_of(const ServeOptions& options);

// The --plan dry run of a server's options: the plan ladder of the Engine it would build first
// (with the speculation --spec auto chooses, or as asked) and the rung the free memory picks now.
struct ServePlan {
    ninferno::PlanSummary plan;
    std::string speculation; // the speculation planned ("plain" without)
    // --spec auto when no rung fits beside the drafter: plain decode's ladder, which the server
    // falls back to after its shallower drafts.
    std::optional<ninferno::PlanSummary> plain;
};

class GenerationService {
public:
    explicit GenerationService(ServeOptions options, StartupObserver startup_observer = {});

    // The --plan dry run (ServePlan): no Engine is kept, nothing is served.
    [[nodiscard]] static ServePlan plan(ServeOptions options);

    [[nodiscard]] const ServeOptions& options() const noexcept { return options_; }

    // Engine owns the once-normalized startup configuration. Serving diagnostics must use this
    // value instead of reinterpreting optional defaults from ServeOptions.
    [[nodiscard]] const ninferno::EngineOptions& engine_options() const {
        return engine_->options();
    }

    [[nodiscard]] ninferno::LoadSummary load_summary() const { return engine_->load_summary(); }

    [[nodiscard]] ninferno::ModelMetadata model_metadata() const {
        return engine_->model_metadata();
    }

    [[nodiscard]] ninferno::MemorySummary memory_summary() const {
        return engine_->memory_summary();
    }

    [[nodiscard]] ninferno::RuntimeStats runtime_stats() const { return engine_->runtime_stats(); }

    [[nodiscard]] bool is_available() const { return engine_->is_available(); }

    // Cancels every request in preparation or generation, and every one prepared later, as a
    // disconnected client's: the server's drain timeout or second signal (HttpServer).
    void abort_in_flight() noexcept { aborting_.store(true); }

    // abort_in_flight() was called: a Cancelled outcome is the server's, not the client's.
    [[nodiscard]] bool aborting() const noexcept { return aborting_.load(); }

    [[nodiscard]] ninferno::MediaCacheSummary media_cache_summary() const {
        return engine_->media_cache_summary();
    }

    // --spec auto: the resolved profile laid onto this engine (nullopt without one), where it came
    // from, and why other sources or choices were passed over.
    [[nodiscard]] const std::optional<product::ServedSpeculation>& served_speculation() const {
        return served_speculation_;
    }
    // Set when --spec auto's speculation did not fit on this device (nor its fit fallback) and the
    // server decodes plainly instead: why. Logged as a warning.
    [[nodiscard]] const std::string& speculation_plain_reason() const {
        return speculation_plain_reason_;
    }
    // Set when --spec auto found no profile for the model on this GPU: why each source was passed
    // over (no manifest beside the artifact, no profile for this GPU class, ...).
    [[nodiscard]] const std::vector<std::string>& speculation_absent_notes() const {
        return speculation_absent_notes_;
    }

    [[nodiscard]] const StartupSpeculation& startup_speculation() const noexcept {
        return startup_speculation_;
    }

    // Set when an explicit --spec came without --draft-tokens: where its depth came from.
    [[nodiscard]] const std::string& draft_depth_source() const { return draft_depth_source_; }

    [[nodiscard]] ninferno::ModelSamplingDefaults sampling_defaults() const {
        return engine_->sampling_defaults();
    }

    [[nodiscard]] PreparedRequest prepare(const GenerationRequest& req,
                                          GenerationConsumerMode consumer_mode,
                                          ninferno::GenerationObservationOptions observation = {},
                                          std::function<bool()> is_cancelled               = {},
                                          ContextCacheHints context_cache = {}) const;
    [[nodiscard]] int count_prompt_tokens(const GenerationRequest& req,
                                          std::function<bool()> is_cancelled = {}) const;

    // Consumes prepared.generation. A PreparedRequest is single-use.
    GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink,
                          std::function<bool()> is_cancelled = {});

    void warmup();

private:
    enum class CacheParticipation : std::uint8_t {
        Disabled,
        ReadWrite,
    };

    enum class DeadlinePolicy : std::uint8_t {
        ClientPendingTimeout,
        UnboundedStartup,
    };

    [[nodiscard]] PreparedRequest
    prepare_impl(const GenerationRequest& req, GenerationConsumerMode consumer_mode,
                 ninferno::GenerationObservationOptions observation,
                 std::function<bool()> is_cancelled, ContextCacheHints context_cache,
                 CacheParticipation cache_participation, DeadlinePolicy deadline_policy) const;
    [[nodiscard]] std::shared_ptr<RequestLifetime>
    acquire_request_lifetime(DeadlinePolicy deadline_policy) const;
    [[nodiscard]] std::shared_ptr<RequestLifetime>
    acquire_lifetime(const std::shared_ptr<RequestCapacity>& capacity,
                     DeadlinePolicy deadline_policy, const char* full_message) const;
    [[nodiscard]] std::shared_ptr<const ninferno::OutputConstraint>
    compile_output_constraint(const OutputConstraintRequest& constraint, bool follows_reasoning,
                              bool special_tokens_visible) const;

    // Chooses the request's speculation (--spec auto) into request_options and prepared.
    void choose_speculation(bool thinking, ninferno::RequestOptions& request_options,
                            PreparedRequest& prepared) const;

    ServeOptions options_;
    std::optional<ServePlan> plan_; // ServeOptions::plan_only: the dry run's result, no Engine
    std::uint32_t plan_level_ = 0;  // the served plan ladder's rung (PlanSummary::rung)
    std::optional<product::ServedSpeculation> served_speculation_;
    std::string speculation_plain_reason_;
    std::vector<std::string> speculation_absent_notes_;
    StartupSpeculation startup_speculation_;
    std::string draft_depth_source_;
    std::unique_ptr<ninferno::Engine> engine_;
    std::shared_ptr<RequestCapacity> request_capacity_;
    // Token counting runs the whole preparation path on handler threads, so it has its own bound:
    // a flood of counts is rejected before it can occupy the threads generation prepares on.
    std::shared_ptr<RequestCapacity> count_capacity_;
    std::atomic<bool> aborting_{false}; // abort_in_flight()

    // A request's cancellation, which abort_in_flight() also sets.
    [[nodiscard]] std::function<bool()> abortable(std::function<bool()> is_cancelled) const;
};

} // namespace ninferno::serve
