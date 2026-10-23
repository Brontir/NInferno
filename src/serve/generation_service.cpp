#include "serve/generation_service.h"

#include "product/media_acquire/acquire.h"
#include "product/speculative_options.h"
#include "serve/translate.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninferno::serve {

struct RequestCapacity {
    explicit RequestCapacity(std::size_t limit) : maximum(limit) {}

    std::mutex mutex;
    std::size_t active = 0;
    const std::size_t maximum;
};

struct RequestLifetime {
    RequestLifetime(std::shared_ptr<RequestCapacity> owner,
                    std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point limit)
        : capacity(std::move(owner)), started(begin), deadline(limit) {}

    ~RequestLifetime() {
        std::lock_guard lock(capacity->mutex);
        --capacity->active;
    }

    std::shared_ptr<RequestCapacity> capacity;
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point deadline;
};

ApiError request_error_to_api_error(const ninferno::RequestError& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninferno::RequestErrorKind::ContextLengthExceeded:
        error.status = 400;
        error.code   = "context_length_exceeded";
        // Agent harnesses compact and retry only on an overflow they recognise, by its text:
        // Anthropic's "prompt is too long" (Claude Code) and llama.cpp's "exceeds the available
        // context size" (Pi and others). The capacity details follow.
        error.message = std::string(kContextOverflowPhrase) + exception.what();
        break;
    case ninferno::RequestErrorKind::ThinkingBudgetCapacityInsufficient:
        error.param.clear();
        error.status = 400;
        error.code   = "thinking_budget_capacity_insufficient";
        break;
    case ninferno::RequestErrorKind::MediaBudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case ninferno::RequestErrorKind::InvalidMedia:
        error.status = 400;
        error.code   = "invalid_media";
        break;
    case ninferno::RequestErrorKind::Overloaded:
        error.param.clear();
        error.status = 429;
        error.type   = "rate_limit_error";
        error.code   = "server_overloaded";
        break;
    case ninferno::RequestErrorKind::QueueTimeout:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninferno::RequestErrorKind::Cancelled:
        error.param.clear();
        error.status = 499;
        error.type   = "request_cancelled";
        error.code   = "client_disconnected";
        break;
    case ninferno::RequestErrorKind::Unavailable:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "service_unavailable";
        break;
    case ninferno::RequestErrorKind::NumericalFailure:
        error.param.clear();
        error.status = 500;
        error.type   = "server_error";
        error.code   = "numerical_failure";
        break;
    }
    return error;
}

namespace {

using Clock = std::chrono::steady_clock;

// What a numerical failure's message adds: what ran, so the report says where to look (the
// kernels of these weight formats on this GPU class, or the artifact itself).
std::string numerical_failure_context(const ninferno::LoadSummary& load) {
    std::string formats;
    for (const std::string& format : load.weight_formats) {
        formats += (formats.empty() ? "" : ",") + format;
    }
    return " (model " + load.model_name + ", " + load.architecture + ", weights " +
           (formats.empty() ? std::string("unknown") : formats) + ", GPU class " + load.gpu_class +
           "). Other requests are unaffected. A repeat on the same prompt points at a kernel route "
           "of these weight formats on this GPU class or at a damaged artifact: reconvert the "
           "model, or report the line above with the request log.";
}

[[noreturn]] void throw_preparation_cancelled();

[[noreturn]] void throw_media_error(const ninferno::product::media_acquire::Error& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninferno::product::media_acquire::ErrorKind::BudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case ninferno::product::media_acquire::ErrorKind::RemoteRejected:
        error.status = 400;
        error.code   = "invalid_media_url";
        break;
    case ninferno::product::media_acquire::ErrorKind::RemoteUnavailable:
        error.status = 502;
        error.type   = "server_error";
        error.code   = "media_fetch_failed";
        break;
    case ninferno::product::media_acquire::ErrorKind::RemoteTimeout:
        error.status = 504;
        error.type   = "server_error";
        error.code   = "media_fetch_timeout";
        break;
    case ninferno::product::media_acquire::ErrorKind::DeadlineExceeded:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninferno::product::media_acquire::ErrorKind::Cancelled:
        throw_preparation_cancelled();
    }
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_invalid_input(const std::exception& exception, const char* code) {
    ApiError error;
    error.status  = 400;
    error.param   = "messages";
    error.code    = code;
    error.message = exception.what();
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_preparation_cancelled() {
    ApiError error;
    error.status  = 499;
    error.type    = "request_cancelled";
    error.code    = "client_disconnected";
    error.message = "client disconnected during media preparation";
    throw ApiException(std::move(error));
}

ninferno::OwnedMedia acquire_media(const ContentPart& part, Clock::time_point deadline,
                                 const std::function<bool()>& is_cancelled,
                                 std::size_t& remaining_bytes) {
    if (remaining_bytes == 0) {
        throw_media_error(ninferno::product::media_acquire::Error(
            ninferno::product::media_acquire::ErrorKind::BudgetExceeded,
            "request media exceeds aggregate byte limit"));
    }
    ninferno::product::media_acquire::Policy policy;
    policy.max_bytes    = std::min(policy.max_bytes, remaining_bytes);
    policy.deadline     = deadline;
    policy.is_cancelled = is_cancelled;
    std::vector<std::uint8_t> source_bytes;
    try {
        source_bytes = ninferno::product::media_acquire::acquire_bytes(part.source, policy);
    } catch (const ninferno::product::media_acquire::Error& exception) {
        throw_media_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_media");
    }

    remaining_bytes -= source_bytes.size();
    ninferno::OwnedMedia media;
    media.kind =
        part.kind == ContentKind::Image ? ninferno::MediaKind::Image : ninferno::MediaKind::Video;
    media.media_type = part.source.media_type;
    switch (part.source.kind) {
    case ninferno::product::media_acquire::SourceKind::Path:
    case ninferno::product::media_acquire::SourceKind::Url:
        media.source_name = part.source.value;
        break;
    case ninferno::product::media_acquire::SourceKind::Data:
        media.source_name = "inline-data";
        break;
    case ninferno::product::media_acquire::SourceKind::Bytes:
        media.source_name = "inline-bytes";
        break;
    }
    media.bytes               = std::move(source_bytes);
    media.image_resize_policy = part.image_resize_policy;
    return media;
}

[[noreturn]] void throw_request_error(const ninferno::RequestError& exception) {
    throw ApiException(request_error_to_api_error(exception));
}

void check_preparation_control(Clock::time_point deadline,
                               const std::function<bool()>& is_cancelled) {
    if (is_cancelled && is_cancelled()) { throw_preparation_cancelled(); }
    if (Clock::now() >= deadline) {
        throw_request_error(ninferno::RequestError(RequestErrorKind::QueueTimeout,
                                                 "inference request expired during preparation"));
    }
}

class ServiceOutputSink final : public ninferno::OutputSink {
public:
    explicit ServiceOutputSink(const StreamSink& sink) : sink_(&sink) {}

    void start(ninferno::GenerationStart start) override {
        if (sink_->on_start) { sink_->on_start(start); }
    }

    void progress(ninferno::PromptProgress progress) override {
        if (sink_->on_progress) { sink_->on_progress(progress); }
    }

    void timing(ninferno::GenerationTimingObservation timing) override {
        if (sink_->on_timing) { sink_->on_timing(timing); }
    }

    void publish(ninferno::OutputDelta delta) override {
        if (delta.text.empty()) { return; }
        if (delta.channel == ninferno::OutputChannel::Reasoning) {
            if (sink_->on_reasoning) { sink_->on_reasoning(delta.text); }
        } else {
            if (sink_->on_content) { sink_->on_content(delta.text); }
        }
    }

private:
    const StreamSink* sink_ = nullptr;
};

// A same-session follow-up consumes the previous turn's continuation, and its own turn closure
// replaces the previous one, so while it runs no checkpoint is left where the conversation
// branched: an edited or replaced last message re-read the whole conversation
// (ValerioDolci/capy-flash#3). A private long-anchor opportunity at the boundary before the
// final message makes the prefill capture that branch point; the Engine keeps and replaces it
// as an explicit anchor, within its per-continuation anchor limit. Client markers keep their
// slots: the anchor takes a free one, and none where a marker already sits.
void add_turn_anchor(ninferno::PromptInput& input) {
    if (input.messages.size() < 2 ||
        input.context_cache.markers.size() >= ninferno::kMaximumExplicitPromptCacheMarkers) {
        return;
    }
    const auto boundary = static_cast<std::uint32_t>(input.messages.size() - 1U);
    for (const PromptCacheMarker& marker : input.context_cache.markers) {
        if (marker.location == ninferno::PromptCacheMarkerLocation::MessageBoundary &&
            marker.after_message_count == boundary) {
            return;
        }
    }
    input.context_cache.markers.push_back(PromptCacheMarker{
        .after_message_count = boundary,
        .kind                = ninferno::PromptCacheMarkerKind::PrivateLongAnchor,
        .evidence            = ninferno::SharedCandidateEvidence::None,
        .location            = ninferno::PromptCacheMarkerLocation::MessageBoundary,
    });
}

// A --spec auto fallback that a capacity refusal forced, for the decision fingerprint: the free
// memory chose it, so it is taint (sources and taint "spec_fit_fallback"; successive fallbacks
// join with "+").
void mark_fit_fallback(SpeculativeOptions& speculative, std::string_view label) {
    std::string& value = speculative.decision_sources["spec_fit_fallback"];
    value += (value.empty() ? "" : "+") + std::string(label);
}

} // namespace

std::string speculation_fallback(const GenerationMetrics& metrics, std::string_view startup) {
    if (metrics.speculative_context_plain) { return "context_plain"; }
    if (metrics.speculative_load_plain) { return "load_plain"; }
    if (metrics.speculative_watchdog_tripped) { return "watchdog_plain"; }
    if (metrics.speculative_step_downs != 0) { return "step_down"; }
    return startup.empty() ? std::string("none") : std::string(startup);
}

std::string auto_plain_reason(const std::string& refused, const std::string& why,
                              bool prefill_chunk_explicit) {
    return "--spec auto (" + refused +
           ") does not fit on this device, so the server decodes "
           "plainly (" +
           why + ")" +
           (prefill_chunk_explicit
                ? "; with --prefill-chunk left automatic the server also narrows the chunk "
                  "before it decodes plainly"
                : "");
}

std::string auto_then_plain_refusal(const std::string& refused, const std::string& plain) {
    return "--spec auto (" + refused +
           ") does not fit on this device, and plain decode does not fit either: " + plain;
}

// The full contexts an automatic KV pool holds (0 when the pool is explicit or the context
// automatic: the pool was then sized to the context, not the other way round).
std::uint32_t pool_contexts(const ninferno::MemorySummary& memory) {
    if (memory.kv_capacity_mode != ninferno::KvCapacityMode::Automatic ||
        memory.max_context_automatic || memory.max_context == 0) {
        return 0;
    }
    return memory.kv_capacity / memory.max_context;
}

GenerationService::GenerationService(ServeOptions options, StartupObserver startup_observer)
    : options_(std::move(options)) {
    ninferno::EngineOptions engine_options;
    engine_options.artifact_path            = options_.artifact_path;
    engine_options.chat_template_path       = options_.chat_template_path;
    engine_options.device                   = options_.device;
    engine_options.max_context              = options_.max_context;
    engine_options.max_context_automatic    = options_.max_context == 0;
    engine_options.kv_capacity              = options_.kv_capacity;
    engine_options.max_concurrency          = options_.max_concurrency;
    engine_options.max_pending_requests     = options_.max_pending_requests;
    engine_options.pending_timeout_ms       = options_.pending_timeout_ms;
    engine_options.concurrent_prefill       = options_.concurrent_prefill;
    engine_options.prefill_chunk            = options_.prefill_chunk;
    engine_options.prefill_chunk_automatic  = !options_.prefill_chunk_explicit;
    engine_options.kv_lease_window          = options_.kv_lease_window;
    engine_options.kv_cache                 = options_.kv_cache.value_or(KvCacheStorage::BFloat16);
    engine_options.kv_cache_automatic       = !options_.kv_cache.has_value();
    engine_options.sliding_kv_cache         = options_.sliding_kv_cache;
    engine_options.recurrent_state =
        options_.recurrent_state.value_or(RecurrentStateStorage::FP32);
    engine_options.recurrent_state_automatic = !options_.recurrent_state.has_value();
    engine_options.activations              = options_.activations;
    engine_options.rope_yarn_factor         = options_.rope_yarn_factor;
    engine_options.enable_vision            = options_.enable_vision;
    engine_options.use_cuda_graph           = options_.use_cuda_graph;
    engine_options.pacing                    = options_.pacing;
    engine_options.maintainer                = options_.maintainer;
    engine_options.speculative              = options_.speculative;
    // A tune named by --spec-profile serves or the server does not start (whatever --spec says).
    product::require_named_tune(options_.artifact_path, options_.device, options_.spec_profile);
    // An explicit --spec without --drafter drafts with the drafter that came with the model.
    if (engine_options.speculative.drafter.empty()) {
        engine_options.speculative.drafter =
            product::model_drafter(options_.artifact_path, engine_options.speculative.backend);
    }
    // An explicit --spec without --draft-tokens drafts at the depth the model's profile runs that
    // backend with on this GPU (its default for the server's thinking flag), else at the default.
    if (options_.draft_tokens_from_model) {
        std::string source;
        std::vector<std::string> notes;
        const auto profiled =
            product::profile_speculation(options_.artifact_path, options_.device,
                                         options_.enable_thinking.value_or(true), &source, &notes,
                                         options_.spec_profile);
        const SpeculativeBackend backend = engine_options.speculative.backend;
        if (profiled && profiled->backend == backend && profiled->draft_tokens != 0) {
            engine_options.speculative.draft_tokens = profiled->draft_tokens;
            draft_depth_source_                     = "the model's profile (" + source + ")";
        } else {
            engine_options.speculative.draft_tokens = product::default_draft_tokens(backend);
            draft_depth_source_ = profiled ? "the default (the model's profile drafts with "
                                             "another backend)"
                                           : "the default (the model has no profile for this GPU)";
        }
        options_.speculative.draft_tokens = engine_options.speculative.draft_tokens;
    }
    if (options_.speculation_from_profile) {
        served_speculation_ = product::served_speculation(
            options_.artifact_path, options_.device, options_.enable_thinking.value_or(true),
            options_.per_request_speculation, options_.spec_profile);
        if (served_speculation_) {
            engine_options.speculative =
                product::with_given_settings(served_speculation_->engine, options_.speculative);
        } else {
            speculation_absent_notes_ =
                product::speculation_notes(options_.artifact_path, options_.device,
                                           options_.spec_profile);
        }
    }
    // --spec-load replaces the profile's bounds under load (or bounds an explicit --spec), and
    // --draft-width the profile's widths. Under --spec auto a forced or bound width applies only
    // where the served choice drafts with DFlash and the model's Engine family runs the width;
    // otherwise it is dropped with a note (product::fit_draft_widths), so a width never keeps a
    // server from starting (an explicit --spec keeps it as given, for the Engine to refuse).
    if (options_.speculation_from_profile) {
        engine_options.speculative.draft_width = DFlashDraftWidth::Default;
    }
    const auto bound_load = [&](SpeculativeOptions& speculative) {
        if (served_speculation_ && options_.speculative.draft_width != DFlashDraftWidth::Default) {
            speculative.draft_width = options_.speculative.draft_width;
        }
        if (options_.speculation_load) {
            speculative.load = product::engine_load(*options_.speculation_load,
                                                    product::speculative_window(speculative));
        }
        if (!served_speculation_) { return; }
        for (std::string& note : product::fit_draft_widths(
                 speculative, product::engine_support(options_.artifact_path))) {
            served_speculation_->notes.push_back(std::move(note));
        }
        served_speculation_->engine.load        = speculative.load;
        served_speculation_->engine.draft_width = speculative.draft_width;
    };
    bound_load(engine_options.speculative);
    engine_options.context_cache            = options_.context_cache;
    engine_options.context_cost.preset_path = options_.context_cost_presets;
    engine_options.media_cache_bytes        = options_.media_cache_bytes;
    engine_options.media_live_bytes         = options_.media_live_bytes;
    engine_options.media_preprocess_threads = options_.media_preprocess_threads;
    engine_options.startup_observer         = std::move(startup_observer);
    // The --plan dry run stops here: the ladder of the first Engine the server would build.
    if (options_.plan_only) {
        ServePlan dry;
        dry.plan        = ninferno::plan_engine(engine_options);
        dry.speculation = engine_options.speculative.backend == SpeculativeBackend::None
                              ? std::string("plain")
                              : product::speculation_label(engine_options.speculative);
        if (dry.plan.rung == 0 && served_speculation_) {
            ninferno::EngineOptions plain = engine_options;
            plain.speculative             = SpeculativeOptions{};
            dry.plain                     = ninferno::plan_engine(plain);
        }
        plan_ = std::move(dry);
        return;
    }
    // A load bound at a depth the menu does not capture needs graphs of its own. When the Engine
    // refuses the plan and moving such bounds to captured depths (product::snap_load_to_captured)
    // changes it, the moved plan is tried once; if it is refused as well, the first refusal stands
    // (a context too long for the device is not a graph shortage).
    const auto build = [&] {
        try {
            engine_ = std::make_unique<ninferno::Engine>(engine_options);
        } catch (const ninferno::CapacityError& refused) {
            const SpeculativeOptions asked = engine_options.speculative;
            if (!product::snap_load_to_captured(engine_options.speculative,
                                                engine_options.max_concurrency)) {
                throw;
            }
            mark_fit_fallback(engine_options.speculative, "captured_load");
            try {
                engine_ = std::make_unique<ninferno::Engine>(engine_options);
            } catch (const ninferno::CapacityError&) {
                engine_options.speculative = asked;
                throw refused;
            }
            if (served_speculation_) {
                served_speculation_->engine.load = engine_options.speculative.load;
                const std::string why =
                    "load " + product::speculation_load_label(asked.load) +
                    " does not fit with graphs of its own on this device: bounds at captured "
                    "depths instead";
                served_speculation_->notes.push_back(why);
                // A fit fallback like the profile's own (startup and request fallback reports).
                if (served_speculation_->fit_fallback.empty()) {
                    served_speculation_->fit_fallback = why;
                }
            }
        }
    };
    try {
        build();
    } catch (const ninferno::CapacityError& refused) {
        // The profile's choice does not fit the requested context beside its drafter: the
        // profile's fit fallback runs instead (spec::Profile::fit_fallback), and the startup log
        // says so. When that does not fit either, or the profile has none, --spec auto decodes
        // plainly rather than keep a server from starting that plain decode lets start. An
        // explicit --spec keeps the refusal.
        if (!served_speculation_) { throw; }
        const std::string refused_label = product::speculation_label(engine_options.speculative);
        std::string why = refused.what();
        bool built      = false;
        if (auto fallback = product::served_fit_fallback(options_.artifact_path, options_.device,
                                                         options_.per_request_speculation,
                                                         options_.spec_profile)) {
            fallback->fit_fallback =
                product::fit_refusal(refused_label, engine_options.max_context, why);
            engine_options.speculative =
                product::with_given_settings(fallback->engine, options_.speculative);
            mark_fit_fallback(engine_options.speculative, "profile_fit_fallback");
            served_speculation_ = std::move(fallback);
            bound_load(engine_options.speculative);
            try {
                build();
                built = true;
            } catch (const ninferno::CapacityError& again) {
                why = again.what();
            }
        }
        // Then shallower drafts with the same drafter: the deepest that fits runs (a few builds,
        // product::deepest_fitting_shallower; the Engine's plan ladder has already narrowed an
        // automatic prefill chunk: 1024 halved down to 128). An explicit --prefill-chunk stays as
        // given and the shallower drafts are tried beside it: --spec auto asks for what fits, and
        // a depth step costs far less than plain decode (RTX 4080 SUPER, Qwen3.8-27B IQ3 dflash2
        // K=7 missed 26 MB at max_context 32768 with --prefill-chunk 2048 and decoded a 24K
        // prompt plainly at 50.9 tok/s against 99.9 speculating).
        if (!built) {
            const auto shallower = [&](const spec::SpecConfig& config, double floor) {
                return product::served_shallower(options_.artifact_path, options_.device,
                                                 options_.per_request_speculation, config, floor,
                                                 options_.spec_profile);
            };
            const std::string first_refusal = refused.what();
            // A candidate replaces the served speculation and builds the Engine (dropping one an
            // earlier candidate built, so each is planned on the whole device): true when it fits.
            const auto fits = [&](const product::ServedSpeculation& candidate,
                                  std::string_view fallback) {
                engine_.reset();
                auto served         = candidate;
                served.fit_fallback =
                    product::fit_refusal(refused_label, engine_options.max_context, first_refusal);
                engine_options.speculative =
                    product::with_given_settings(served.engine, options_.speculative);
                mark_fit_fallback(engine_options.speculative, fallback);
                served_speculation_ = std::move(served);
                bound_load(engine_options.speculative);
                try {
                    build();
                    return true;
                } catch (const ninferno::CapacityError& again) {
                    why = again.what();
                    return false;
                }
            };
            built = product::deepest_fitting_shallower(
                        served_speculation_->menu.primary, shallower,
                        [&](const product::ServedSpeculation& candidate) {
                            return fits(candidate, "shallower");
                        })
                        .has_value();
            // Then the model's own MTP head, before plain decode.
            if (!built) {
                if (auto head = product::served_head_fallback(
                        options_.artifact_path, options_.device,
                        options_.per_request_speculation, options_.spec_profile)) {
                    built = fits(*head, "head");
                }
            }
        }
        if (!built) {
            speculation_plain_reason_ =
                auto_plain_reason(refused_label, why, options_.prefill_chunk_explicit);
            served_speculation_.reset();
            engine_options.speculative = SpeculativeOptions{};
            mark_fit_fallback(engine_options.speculative, "plain");
            // When plain decode does not fit either, the refusal is plain decode's, and it says
            // that --spec auto was refused first.
            try {
                engine_ = std::make_unique<ninferno::Engine>(engine_options);
            } catch (const ninferno::CapacityError& plain) {
                throw ninferno::CapacityError(auto_then_plain_refusal(refused_label, plain.what()));
            }
        }
    }
    // --spec auto with an automatic KV pool: when the drafter (its weights, KV and graphs) leaves
    // the pool fewer full contexts than --max-concurrency and plain decode holds more, the server
    // decodes plainly, since requests waiting for pages cost more than drafting saves (RTX
    // 4090, gpt-oss-20b 4 x 129K: auto pool 262K of 516K tokens, 34.9 tok/s aggregate; plain 506K,
    // 239.4). One context always fits, so one request at a time never loses its drafter, and an
    // explicit --spec keeps it.
    if (served_speculation_ && options_.speculation_from_profile) {
        const ninferno::MemorySummary drafting = engine_->memory_summary();
        const std::uint32_t with_drafter     = pool_contexts(drafting);
        if (with_drafter > 0 && with_drafter < options_.max_concurrency) {
            const std::string drafted_label =
                product::speculation_label(engine_options.speculative);
            const SpeculativeOptions speculative = engine_options.speculative;
            engine_.reset();
            engine_options.speculative = SpeculativeOptions{};
            mark_fit_fallback(engine_options.speculative, "pool_plain");
            std::uint32_t plain        = 0;
            std::uint32_t plain_tokens = 0;
            try {
                engine_ = std::make_unique<ninferno::Engine>(engine_options);
                const ninferno::MemorySummary memory = engine_->memory_summary();
                plain                              = pool_contexts(memory);
                plain_tokens                       = memory.kv_capacity;
            } catch (const ninferno::CapacityError&) { engine_.reset(); }
            // The model's own MTP head before plain decode: it weighs far less than a separate
            // drafter, so it can leave the pool every context plain decode does, and drafting
            // still pays at these loads (RTX 4090, Qwen3.8 27B at --max-context 32768 and
            // --max-concurrency 4: dflash2 K=7 left 3 of 4 contexts and the server decoded plainly
            // at 48.9 tok/s a request; MTP 3 88.8, at 8: 46.2 against 69.4). It runs
            // when it holds at least four fifths of the contexts plain decode does (or of every one
            // served): one context fewer costs only when every request runs a full one, while plain
            // decode costs every request (RTX 3090, the same at 8: plain 38.3 tok/s a request in 5
            // contexts, MTP 3 51.1 in 4).
            bool head_kept = false;
            if (plain > with_drafter) {
                if (auto head = product::served_head_fallback(
                        options_.artifact_path, options_.device, options_.per_request_speculation,
                        options_.spec_profile)) {
                    engine_.reset();
                    head->fit_fallback = drafted_label + " leaves the KV pool " +
                                         std::to_string(with_drafter) + " of " +
                                         std::to_string(options_.max_concurrency) +
                                         " full contexts beside its drafter";
                    engine_options.speculative =
                        product::with_given_settings(head->engine, options_.speculative);
                    mark_fit_fallback(engine_options.speculative, "pool_head");
                    served_speculation_ = std::move(head);
                    bound_load(engine_options.speculative);
                    try {
                        build();
                        head_kept = 5 * pool_contexts(engine_->memory_summary()) >=
                                    4 * std::min(plain, options_.max_concurrency);
                    } catch (const ninferno::CapacityError&) {}
                    if (!head_kept) {
                        engine_.reset();
                        served_speculation_.reset();
                        engine_options.speculative = SpeculativeOptions{};
                        mark_fit_fallback(engine_options.speculative, "pool_plain");
                        engine_ = std::make_unique<ninferno::Engine>(engine_options);
                    }
                }
            }
            if (!head_kept && plain > with_drafter) {
                speculation_plain_reason_ =
                    "--spec auto (" + drafted_label + ") leaves the KV pool " +
                    std::to_string(with_drafter) + " of " +
                    std::to_string(options_.max_concurrency) + " full contexts (" +
                    std::to_string(drafting.kv_capacity) +
                    " tokens) beside its drafter; plain "
                    "decode holds " +
                    std::to_string(plain) + " (" + std::to_string(plain_tokens) +
                    "), so the server decodes plainly (an explicit --spec keeps the drafter)";
                served_speculation_.reset();
            } else if (!head_kept) {
                engine_.reset();
                engine_options.speculative = speculative;
                build();
            }
        }
    }
    // The prefill chunk an automatic fit narrowed to fit the served context. Whether the drafter
    // is why is not said: the loaded components (the target with its drafter) cannot be planned
    // again without the drafter to compare.
    if (served_speculation_ && !options_.prefill_chunk_explicit &&
        engine_->options().prefill_chunk < options_.prefill_chunk) {
        served_speculation_->notes.push_back(
            "prefill chunk " + std::to_string(engine_->options().prefill_chunk) + " (of " +
            std::to_string(options_.prefill_chunk) + ") to fit the served context");
    }
    // What runs since startup, for /v1/models, the startup log, and every request when the
    // server does not choose per request (choose_speculation).
    if (served_speculation_) {
        const spec::MenuChoice& primary = served_speculation_->menu.primary;
        const bool fell_back            = !served_speculation_->fit_fallback.empty();
        startup_speculation_            = {
                       .config    = primary.config.enabled() ? primary.config.label() : "plain",
                       .selection = options_.per_request_speculation ? "thinking" : "fixed",
                       .reason    = fell_back ? served_speculation_->fit_fallback
                                              : "profile " + served_speculation_->source,
                       .fallback  = fell_back ? "fit_fallback" : "none",
        };
    } else if (!speculation_plain_reason_.empty()) {
        startup_speculation_ = {.selection = "startup",
                                .reason    = speculation_plain_reason_,
                                .fallback  = "startup_plain"};
    } else if (options_.speculation_from_profile) {
        std::string why = "no speculation profile for this model on this GPU";
        for (const std::string& note : speculation_absent_notes_) {
            why += (&note == &speculation_absent_notes_.front() ? " (" : "; ") + note;
        }
        if (!speculation_absent_notes_.empty()) { why += ")"; }
        startup_speculation_ = {.selection = "startup", .reason = std::move(why)};
    } else {
        const SpeculativeOptions& given = engine_->options().speculative;
        const bool drafts               = given.backend != SpeculativeBackend::None;
        startup_speculation_ = {.config    = drafts ? product::speculation_label(given) : "plain",
                                .selection = "explicit",
                                .reason    = "--spec"};
    }
    // The plan ladder's rung the free memory picked, for every request record.
    plan_level_ = engine_->memory_summary().plan.rung;
    // An explicit --spec runs the engine's own watchdog floor.
    if (!served_speculation_ && engine_->options().speculative.backend != SpeculativeBackend::None) {
        startup_speculation_.floor = engine_->options().speculative.acceptance_floor;
    }
    request_capacity_ = std::make_shared<RequestCapacity>(
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests);
    count_capacity_ = std::make_shared<RequestCapacity>(
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests);
}

ServePlan GenerationService::plan(ServeOptions options) {
    options.plan_only = true;
    GenerationService service(std::move(options));
    return std::move(*service.plan_);
}

std::shared_ptr<RequestLifetime>
GenerationService::acquire_request_lifetime(DeadlinePolicy deadline_policy) const {
    return acquire_lifetime(request_capacity_, deadline_policy, "inference request queue is full");
}

std::shared_ptr<RequestLifetime>
GenerationService::acquire_lifetime(const std::shared_ptr<RequestCapacity>& capacity,
                                    DeadlinePolicy deadline_policy,
                                    const char* full_message) const {
    const auto started = Clock::now();
    {
        std::lock_guard lock(capacity->mutex);
        if (capacity->active >= capacity->maximum) {
            throw_request_error(ninferno::RequestError(RequestErrorKind::Overloaded, full_message));
        }
        ++capacity->active;
    }
    try {
        const Clock::time_point deadline =
            deadline_policy == DeadlinePolicy::UnboundedStartup
                ? Clock::time_point::max()
                : started + std::chrono::milliseconds(options_.pending_timeout_ms);
        return std::make_shared<RequestLifetime>(capacity, started, deadline);
    } catch (...) {
        std::lock_guard lock(capacity->mutex);
        --capacity->active;
        throw;
    }
}

PreparedRequest GenerationService::prepare(const GenerationRequest& request,
                                           GenerationConsumerMode consumer_mode,
                                           ninferno::GenerationObservationOptions observation,
                                           std::function<bool()> is_cancelled,
                                           ContextCacheHints context_cache) const {
    return prepare_impl(
        request, consumer_mode, observation, std::move(is_cancelled), std::move(context_cache),
        options_.allow_prefix_reuse ? CacheParticipation::ReadWrite : CacheParticipation::Disabled,
        DeadlinePolicy::ClientPendingTimeout);
}

PreparedRequest GenerationService::prepare_impl(const GenerationRequest& request,
                                                GenerationConsumerMode consumer_mode,
                                                ninferno::GenerationObservationOptions observation,
                                                std::function<bool()> is_cancelled,
                                                ContextCacheHints context_cache,
                                                CacheParticipation cache_participation,
                                                DeadlinePolicy deadline_policy) const {
    is_cancelled = abortable(std::move(is_cancelled));
    PreparedRequest prepared;
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    ninferno::RequestOptions request_options  = to_request_options(
        request, options_, semantics, cache_participation == CacheParticipation::ReadWrite);
    prepared.thinking_budget     = request_options.execution.thinking.budget;
    prepared.reasoning_effort    = semantics.reasoning_effort;
    prepared.preserve_thinking   = semantics.preserve_thinking;
    const bool request_has_media = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    prepared.lifetime = acquire_request_lifetime(deadline_policy);

    try {
        const auto acquisition_started = Clock::now();
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, ninferno::kMaximumPromptMediaBytes);
        ninferno::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, prepared.lifetime->deadline, is_cancelled,
                                     remaining_media_bytes);
            });
        std::vector<PromptCacheMarker> protocol_markers = std::move(input.context_cache.markers);
        const bool protocol_allows_engine_automatic =
            input.context_cache.allow_engine_automatic_shared_prefixes;
        input.context_cache = std::move(context_cache);
        input.context_cache.markers.insert(input.context_cache.markers.end(),
                                           std::make_move_iterator(protocol_markers.begin()),
                                           std::make_move_iterator(protocol_markers.end()));
        input.context_cache.allow_engine_automatic_shared_prefixes =
            input.context_cache.allow_engine_automatic_shared_prefixes &&
            protocol_allows_engine_automatic;
        if (cache_participation == CacheParticipation::ReadWrite) { add_turn_anchor(input); }
        prepared.acquisition_seconds =
            std::chrono::duration<double>(Clock::now() - acquisition_started).count();
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = prepared.lifetime->deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        ninferno::PreparedPrompt prompt = engine_->prepare(std::move(input), control);
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        if (request.output_constraint) {
            request_options.constraint.compiled = compile_output_constraint(
                *request.output_constraint, prompt.summary().starts_in_reasoning,
                request_options.output.raw || request_options.output.preserve_special_tokens);
            check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        }
        prepared.enable_thinking = prompt.summary().thinking();
        // A budget applies where the output thinks: from the prompt's start, or from the opener
        // the model writes itself (Gemma 4). --default-thinking-budget applies where the template
        // can end thinking early; a budget the request itself gives on a template that cannot is
        // refused as before.
        if (!prepared.enable_thinking ||
            (!request.thinking_budget && !prompt.summary().thinking_budget_control)) {
            request_options.execution.thinking.budget.reset();
            prepared.thinking_budget.reset();
        }
        choose_speculation(prepared.enable_thinking, request_options, prepared);
        prepared.prompt_tokens = static_cast<int>(prompt.summary().prompt_tokens);
        prepared.preparation   = prompt.preparation_stats();
        prepared.prepare_seconds =
            std::chrono::duration<double>(Clock::now() - prepared.lifetime->started).count();
        prepared.generation = engine_->submit(std::move(prompt), std::move(request_options),
                                              consumer_mode == GenerationConsumerMode::Streaming
                                                  ? ninferno::OutputConsumerMode::Streaming
                                                  : ninferno::OutputConsumerMode::Aggregate,
                                              observation, prepared.lifetime->deadline);
        prepared.sampling   = prepared.generation.resolved_sampling();
    } catch (const ApiException&) { throw; } catch (const ninferno::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
    return prepared;
}

void GenerationService::choose_speculation(bool thinking, ninferno::RequestOptions& request_options,
                                           PreparedRequest& prepared) const {
    prepared.speculation_fallback = startup_speculation_.fallback;
    prepared.plan_level           = plan_level_;
    if (!served_speculation_) {
        prepared.speculation_config    = startup_speculation_.config;
        prepared.speculation_selection = startup_speculation_.selection;
        prepared.speculation_reason    = startup_speculation_.reason;
        prepared.speculation_floor     = startup_speculation_.floor;
        return;
    }
    const spec::SpeculationMenu& menu = served_speculation_->menu;
    const spec::MenuChoice* choice    = &menu.primary;
    if (!options_.per_request_speculation) {
        prepared.speculation_selection = "fixed";
        prepared.speculation_reason    = "--spec-select fixed";
    } else {
        choice                         = &menu.pick(thinking);
        prepared.speculation_selection = "thinking";
        prepared.speculation_reason    = thinking ? "thinking request" : "non-thinking request";
        request_options.execution.speculation = product::request_speculation(menu, *choice);
    }
    prepared.speculation_config = choice->config.enabled() ? choice->config.label() : "plain";
    prepared.speculation_note   = choice->note;
    // The fit fallback the server runs in place of the profile's choice: why.
    if (!served_speculation_->fit_fallback.empty()) {
        prepared.speculation_note +=
            (prepared.speculation_note.empty() ? "fit fallback: " : "; fit fallback: ") +
            served_speculation_->fit_fallback;
    }
    prepared.speculation_floor =
        choice->config.enabled() ? product::watchdog_floor(choice->floor) : 0.0;
}

std::shared_ptr<const ninferno::OutputConstraint>
GenerationService::compile_output_constraint(const OutputConstraintRequest& constraint,
                                             bool follows_reasoning,
                                             bool special_tokens_visible) const {
    ninferno::OutputConstraintSpec spec;
    spec.kind                   = constraint.kind;
    spec.definition             = constraint.definition;
    spec.choices                = constraint.choices;
    spec.compact_json           = constraint.compact_json;
    spec.tools                  = constraint.tools;
    spec.tool_requirement       = constraint.tool_requirement;
    spec.named_tool             = constraint.named_tool;
    spec.parallel_tool_calls    = constraint.parallel_tool_calls;
    spec.answer_schema          = constraint.answer_schema;
    spec.follows_reasoning      = follows_reasoning;
    spec.special_tokens_visible = special_tokens_visible;
    try {
        return engine_->compile_output_constraint(spec);
    } catch (const ninferno::OutputConstraintError& exception) {
        ApiError error;
        error.param   = constraint.param;
        error.code    = exception.code();
        error.message = exception.what();
        if (exception.code() == "constraint_compiler_busy") {
            error.status = 503;
            error.type   = "server_error";
        }
        throw ApiException(std::move(error));
    }
}

std::function<bool()> GenerationService::abortable(std::function<bool()> is_cancelled) const {
    return [this, external = std::move(is_cancelled)] {
        return aborting_.load() || (external && external());
    };
}

int GenerationService::count_prompt_tokens(const GenerationRequest& request,
                                           std::function<bool()> is_cancelled) const {
    is_cancelled                 = abortable(std::move(is_cancelled));
    const bool request_has_media = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    const std::shared_ptr<RequestLifetime> lifetime = acquire_lifetime(
        count_capacity_, DeadlinePolicy::ClientPendingTimeout, "token count queue is full");
    const Clock::time_point deadline        = lifetime->deadline;
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    try {
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, ninferno::kMaximumPromptMediaBytes);
        ninferno::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, deadline, is_cancelled, remaining_media_bytes);
            });
        check_preparation_control(deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        const int prompt_tokens =
            static_cast<int>(engine_->count_tokens(std::move(input), control));
        check_preparation_control(deadline, is_cancelled);
        return prompt_tokens;
    } catch (const ApiException&) { throw; } catch (const ninferno::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
}

GenerationOutcome GenerationService::run(PreparedRequest& prepared, const StreamSink* sink,
                                         std::function<bool()> is_cancelled) {
    std::unique_ptr<ServiceOutputSink> output_sink;
    if (sink != nullptr) { output_sink = std::make_unique<ServiceOutputSink>(*sink); }
    ninferno::OutputSink* public_sink = output_sink.get();
    const ninferno::CancellationView cancellation(
        [external = abortable(std::move(is_cancelled)), sink]() {
            return external() || (sink != nullptr && sink->is_cancelled && sink->is_cancelled());
        });

    ninferno::GenerationResult result;
    try {
        result = prepared.generation.wait(public_sink, cancellation);
    } catch (const ninferno::RequestError& exception) {
        if (exception.kind() != ninferno::RequestErrorKind::NumericalFailure) {
            throw_request_error(exception);
        }
        ApiError error = request_error_to_api_error(exception);
        error.message += numerical_failure_context(load_summary());
        throw ApiException(std::move(error));
    }
    if (result.constraint.violated) {
        // Never present unenforced output as conforming.
        ApiError error;
        error.status  = 500;
        error.type    = "server_error";
        error.code    = "structured_output_violation";
        error.message = "the requested output constraint could not be enforced: " +
                        result.constraint.violation;
        throw ApiException(std::move(error));
    }
    GenerationOutcome outcome;
    outcome.text                = std::move(result.content);
    outcome.reasoning           = std::move(result.reasoning);
    outcome.prompt_tokens       = static_cast<int>(result.prompt.prompt_tokens);
    outcome.completion_tokens   = static_cast<int>(result.generated_token_ids.size());
    outcome.reasoning_tokens    = static_cast<int>(result.reasoning_tokens);
    outcome.thinking            = result.thinking;
    outcome.finish_reason       = result.finish_reason;
    outcome.matched_stop_string = std::move(result.matched_stop_string);

    outcome.metrics.prepare_seconds = prepared.prepare_seconds;
    outcome.metrics.ttft_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.first_token_seconds - result.timings.prepare_seconds);
    outcome.metrics.vision_seconds          = result.timings.vision_seconds;
    outcome.metrics.prefill_seconds         = result.timings.prefill_seconds;
    outcome.metrics.decode_seconds          = result.timings.decode_seconds;
    outcome.metrics.prompt_wall_seconds     = result.timings.prompt_wall_seconds;
    outcome.metrics.generation_wall_seconds = result.timings.generation_wall_seconds;
    outcome.metrics.total_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.total_seconds - result.timings.prepare_seconds);
    outcome.metrics.engine_timing               = result.engine_timing;
    outcome.metrics.prefix_cache_hit_tokens     = result.reused_prompt_tokens;
    outcome.metrics.prefix_reuse_path           = result.prefix_reuse_path;
    outcome.metrics.materialization             = result.materialization;
    outcome.metrics.speculative_backend         = result.speculative.backend;
    outcome.metrics.speculative_draft_window    = result.speculative.draft_window;
    outcome.metrics.speculative_rounds          = result.speculative.rounds;
    outcome.metrics.speculative_draft_tokens    = result.speculative.drafted_tokens;
    outcome.metrics.speculative_accepted_tokens = result.speculative.accepted_tokens;
    outcome.metrics.speculative_fallback_steps  = result.speculative.fallback_steps;
    outcome.metrics.speculative_constraint_fallback_steps =
        result.speculative.constraint_fallback_steps;
    outcome.metrics.speculative_accepted_per_position =
        std::move(result.speculative.accepted_per_position);
    outcome.metrics.speculative_config                = prepared.speculation_config;
    outcome.metrics.speculative_selection             = prepared.speculation_selection;
    outcome.metrics.speculative_reason                = prepared.speculation_reason;
    outcome.metrics.speculative_note                  = prepared.speculation_note;
    outcome.metrics.speculative_floor                 = prepared.speculation_floor;
    outcome.metrics.speculative_plain_steps           = result.speculative.plain_steps;
    outcome.metrics.speculative_wide_rounds           = result.speculative.wide_rounds;
    outcome.metrics.speculative_ungraphed             = result.speculative.ungraphed_rounds;
    outcome.metrics.speculative_capped_rounds         = result.speculative.capped_rounds;
    outcome.metrics.speculative_copy_rounds           = result.speculative.copy_rounds;
    outcome.metrics.speculative_copy_accepted         = result.speculative.copy_accepted_tokens;
    outcome.metrics.speculative_trips_held            = result.speculative.shared_trip_deferrals;
    outcome.metrics.speculative_load_plain            = result.speculative.load_plain;
    outcome.metrics.speculative_context_plain         = result.speculative.context_plain;
    outcome.metrics.speculative_watchdog_tripped      = result.speculative.watchdog_tripped;
    outcome.metrics.speculative_watchdog_round        = result.speculative.watchdog_round;
    outcome.metrics.speculative_step_downs            = result.speculative.step_downs;
    outcome.metrics.speculative_final_draft_window    = result.speculative.final_draft_window;
    outcome.metrics.speculative_fallback =
        speculation_fallback(outcome.metrics, prepared.speculation_fallback);
    outcome.metrics.constraint_masked_positions = result.constraint.masked_positions;
    outcome.metrics.constraint_mask_host_seconds = result.constraint.mask_host_seconds;

    outcome.tool_calls      = std::move(result.tool_calls);
    outcome.tool_call_parse = result.tool_call_parse;
    return outcome;
}

void GenerationService::warmup() {
    GenerationRequest request;
    ChatTurn turn;
    turn.role = ChatRole::User;
    ContentPart content;
    content.kind     = ContentKind::Text;
    content.text     = "hi";
    content.type_raw = "text";
    turn.content.push_back(std::move(content));
    request.messages.push_back(std::move(turn));
    request.max_tokens = 4;
    PreparedRequest prepared =
        prepare_impl(request, GenerationConsumerMode::Aggregate, {}, {}, {},
                     CacheParticipation::Disabled, DeadlinePolicy::UnboundedStartup);
    run(prepared, nullptr);

    // Build the grammar vocabulary and exercise the device mask path once, so a platform that
    // cannot map host memory fails at startup rather than on the first structured request.
    request.output_constraint = OutputConstraintRequest{
        .kind = ninferno::OutputConstraintKind::JsonObject, .param = "warmup"};
    request.enable_thinking   = false;
    PreparedRequest constrained =
        prepare_impl(request, GenerationConsumerMode::Aggregate, {}, {}, {},
                     CacheParticipation::Disabled, DeadlinePolicy::UnboundedStartup);
    run(constrained, nullptr);
}

} // namespace ninferno::serve
