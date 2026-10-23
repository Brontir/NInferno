#include "serve/operational_log.h"
#include "serve/plan_report.h"
#include "serve/generation_service.h"

#include "product/logging/pretty_format.h"
#include "product/activation_options.h"
#include "product/speculative_options.h"

#include <spdlog/fmt/fmt.h>
#include <spdlog/logger.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace ninferno::serve {
namespace {

const char* phase_name(RequestFailurePhase phase) noexcept {
    switch (phase) {
    case RequestFailurePhase::Prepare:
        return "prepare";
    case RequestFailurePhase::Generation:
        return "generation";
    case RequestFailurePhase::ResponseRender:
        return "response rendering";
    case RequestFailurePhase::ResponseStore:
        return "response storage";
    case RequestFailurePhase::Transport:
        return "transport";
    case RequestFailurePhase::Http:
        return "http";
    }
    return "unknown";
}

const char* classification_name(RequestFailureClass classification) noexcept {
    switch (classification) {
    case RequestFailureClass::ClientInput:
        return "client input";
    case RequestFailureClass::ClientDisconnected:
        return "client disconnected";
    case RequestFailureClass::Overload:
        return "overload";
    case RequestFailureClass::Timeout:
        return "timeout";
    case RequestFailureClass::Unavailable:
        return "unavailable";
    case RequestFailureClass::Upstream:
        return "upstream error";
    case RequestFailureClass::Internal:
        return "internal error";
    }
    return "unknown";
}

OperationalSeverity failure_severity(RequestFailureClass classification) noexcept {
    switch (classification) {
    case RequestFailureClass::ClientInput:
    case RequestFailureClass::ClientDisconnected:
        return OperationalSeverity::Info;
    case RequestFailureClass::Overload:
    case RequestFailureClass::Timeout:
    case RequestFailureClass::Unavailable:
    case RequestFailureClass::Upstream:
        return OperationalSeverity::Warning;
    case RequestFailureClass::Internal:
        return OperationalSeverity::Error;
    }
    return OperationalSeverity::Error;
}

const char* prefix_reuse_path_name(ninferno::PrefixReusePath path) noexcept {
    switch (path) {
    case ninferno::PrefixReusePath::Root:
        return "root";
    case ninferno::PrefixReusePath::PrivateEndpoint:
        return "private endpoint";
    case ninferno::PrefixReusePath::PrivateTurnClosure:
        return "turn closure";
    case ninferno::PrefixReusePath::PrivateResponseReplay:
        return "response replay";
    case ninferno::PrefixReusePath::PrivateLongAnchor:
        return "long anchor";
    case ninferno::PrefixReusePath::SharedStablePrefix:
        return "shared prefix";
    }
    return "unknown";
}

std::string_view requested_effort_name(const RequestLogContext& context) noexcept {
    if (context.requested_reasoning_effort)
        return ninferno::reasoning_effort_name(*context.requested_reasoning_effort);
    return "template default";
}

const char* protocol_name(std::string_view protocol) noexcept {
    if (protocol == "openai_chat_completions") { return "openai-chat"; }
    if (protocol == "openai_responses") { return "openai-responses"; }
    if (protocol == "anthropic_messages") { return "anthropic"; }
    if (protocol == "openai_responses_input_tokens") { return "openai-input-tokens"; }
    if (protocol == "anthropic_count_tokens") { return "anthropic-count-tokens"; }
    return "http";
}

void append_clause(std::ostringstream& out, std::string_view clause) { out << " | " << clause; }

void append_counted_clause(std::ostringstream& out, std::string_view label, std::uint64_t count) {
    out << " | " << label << ' ' << product::format_pretty_count(count);
}

// The kernel warm-up's widths in one clause: "1..48 + 142 class widths (190 of 1024)", or
// "1..1024 (1024 of 1024)" when every width ran. The server-start JSON lists them all.
std::string warm_up_widths_summary(const std::vector<std::uint32_t>& run, std::uint32_t range) {
    if (run.empty()) { return "1.." + std::to_string(range); }
    std::size_t exact = 0;
    while (exact < run.size() && run[exact] == exact + 1) { ++exact; }
    const std::string total = fmt::format("({} of {})", run.size(), range);
    if (exact == run.size()) { return fmt::format("1..{} {}", exact, total); }
    return fmt::format("1..{} + {} class widths {}", exact, run.size() - exact, total);
}

std::string pretty_code(std::string_view code) {
    std::string result;
    result.reserve(code.size());
    for (const char ch : code) { result.push_back(ch == '_' ? ' ' : ch); }
    return result;
}

template <class T>
T monotonic_delta(T previous, T current) noexcept {
    return current >= previous ? current - previous : T{};
}

std::uint64_t host_active_ns(const ThroughputReport& report) noexcept {
    const ninferno::RuntimeHostWorkStats& previous = report.previous.host_work;
    const ninferno::RuntimeHostWorkStats& current  = report.current.host_work;
    return monotonic_delta(previous.engine_boundary_ns, current.engine_boundary_ns) +
           monotonic_delta(previous.program_submit_ns, current.program_submit_ns) +
           monotonic_delta(previous.program_post_ns, current.program_post_ns) +
           monotonic_delta(previous.engine_commit_output_ns, current.engine_commit_output_ns) +
           monotonic_delta(previous.engine_maintenance_ns, current.engine_maintenance_ns);
}

void append_failure_fields(std::ostringstream& out, const RequestFailure& failure) {
    if (failure.http_status != 0) { out << " | HTTP " << failure.http_status; }
    if (!failure.error_code.empty()) {
        append_clause(out, pretty_code(failure.error_code));
    } else {
        append_clause(out, classification_name(failure.classification));
    }
}

} // namespace

OperationalRecord render_request_start(const RequestLogContext& context) {
    std::ostringstream out;
    out << "req#" << context.id << " started | " << protocol_name(context.protocol) << ' '
        << (context.stream ? "stream" : "non-stream") << " | ";
    if (!context.http_request_id.empty()) {
        out << "request " << product::format_pretty_text(context.http_request_id) << " | ";
    }
    out << product::format_pretty_count(context.message_count)
        << (context.message_count == 1 ? " message" : " messages") << " | max output "
        << product::format_pretty_count(
               static_cast<std::uint64_t>(std::max(context.requested_output_tokens, 0)));
    out << " | thinking ";
    if (context.enable_thinking) {
        out << requested_effort_name(context);
        if (context.thinking_budget) {
            out << ", budget " << product::format_pretty_count(*context.thinking_budget);
        }
    } else {
        out << "off";
    }
    if (context.media_item_count != 0) {
        append_counted_clause(out, "media", context.media_item_count);
        if (context.preparation.seconds > 0.0) {
            out << ", prepared " << product::format_pretty_duration(context.preparation.seconds);
        }
    }
    if (context.tool_count != 0) { append_counted_clause(out, "tools", context.tool_count); }
    if (context.preserve_thinking == true) { append_clause(out, "preserve thinking"); }
    for (const std::string& note : context.notes) { append_clause(out, note); }
    return {.severity =
                context.notes.empty() ? OperationalSeverity::Info : OperationalSeverity::Warning,
            .message = out.str()};
}

OperationalRecord render_request_rejected(const RequestRejectionLogContext& context) {
    const RequestFailure failure =
        make_request_failure(RequestFailurePhase::Prepare, context.error);
    std::ostringstream out;
    const char* status = "failed";
    if (failure.classification == RequestFailureClass::ClientDisconnected) {
        status = "cancelled";
    } else if (failure.classification == RequestFailureClass::ClientInput ||
               failure.classification == RequestFailureClass::Overload) {
        status = "rejected";
    }
    out << "req#" << context.id << ' ' << status << " during prepare | "
        << protocol_name(context.protocol) << ' ' << (context.stream ? "stream" : "non-stream");
    append_failure_fields(out, failure);
    append_counted_clause(out, "messages", context.message_count);
    if (context.media_item_count != 0) {
        append_counted_clause(out, "media", context.media_item_count);
    }
    if (context.tool_count != 0) { append_counted_clause(out, "tools", context.tool_count); }
    return {.severity = failure_severity(failure.classification), .message = out.str()};
}

OperationalRecord render_request_done(const RequestLogContext& context,
                                      const GenerationOutcome& outcome) {
    const GenerationMetrics& metrics     = outcome.metrics;
    const double computed_prefill_tokens = static_cast<double>(
        std::max(0, outcome.prompt_tokens - static_cast<int>(metrics.prefix_cache_hit_tokens)));
    const double decode_tokens =
        outcome.completion_tokens > 0 ? static_cast<double>(outcome.completion_tokens - 1) : 0.0;
    std::ostringstream out;
    out << "req#" << context.id << " done | " << protocol_name(context.protocol) << " | ";
    if (outcome.tool_calls.empty()) {
        out << pretty_code(ninferno::finish_reason_name(outcome.finish_reason));
    } else {
        out << "tool calls " << product::format_pretty_count(outcome.tool_calls.size());
    }
    out << " | prompt "
        << product::format_pretty_count(
               static_cast<std::uint64_t>(std::max(outcome.prompt_tokens, 0)))
        << " | output "
        << product::format_pretty_count(
               static_cast<std::uint64_t>(std::max(outcome.completion_tokens, 0)));
    const double cache_ratio = outcome.prompt_tokens > 0
                                   ? static_cast<double>(metrics.prefix_cache_hit_tokens) /
                                         static_cast<double>(outcome.prompt_tokens)
                                   : 0.0;
    out << " | cache " << product::format_pretty_count(metrics.prefix_cache_hit_tokens) << " ("
        << product::format_pretty_percent(cache_ratio);
    if (metrics.prefix_reuse_path != ninferno::PrefixReusePath::Root) {
        out << ", " << prefix_reuse_path_name(metrics.prefix_reuse_path);
    }
    out << ") | TTFT " << product::format_pretty_duration(metrics.ttft_seconds) << " | total "
        << product::format_pretty_duration(metrics.total_seconds);
    if (metrics.engine_timing.queue_wait_seconds >= 0.01) {
        out << " | queue "
            << product::format_pretty_duration(metrics.engine_timing.queue_wait_seconds);
    }
    if (metrics.prefill_seconds > 0.0) {
        out << " | prefill "
            << product::format_pretty_rate(computed_prefill_tokens / metrics.prefill_seconds,
                                           "tok");
    }
    if (metrics.decode_seconds > 0.0) {
        out << " | decode "
            << product::format_pretty_rate(decode_tokens / metrics.decode_seconds, "tok");
    }
    if (!metrics.speculative_config.empty()) {
        out << " | " << metrics.speculative_config << " (" << metrics.speculative_selection << ')';
    }
    if (metrics.speculative_draft_tokens != 0) {
        const double acceptance = static_cast<double>(metrics.speculative_accepted_tokens) /
                                  static_cast<double>(metrics.speculative_draft_tokens);
        out << " | " << product::speculative_backend_name(metrics.speculative_backend)
            << " accepted " << product::format_pretty_count(metrics.speculative_accepted_tokens)
            << '/' << product::format_pretty_count(metrics.speculative_draft_tokens) << " ("
            << product::format_pretty_percent(acceptance) << ')';
    }
    if (metrics.speculative_step_downs != 0) {
        out << " | stepped down " << metrics.speculative_step_downs << "x to depth "
            << metrics.speculative_final_draft_window;
    }
    if (metrics.speculative_watchdog_tripped) {
        out << " | watchdog at round " << metrics.speculative_watchdog_round << ", plain "
            << product::format_pretty_count(metrics.speculative_plain_steps);
    }
    if (metrics.speculative_trips_held != 0) {
        out << " | watchdog held by the batch "
            << product::format_pretty_count(metrics.speculative_trips_held) << "x";
    }
    if (metrics.speculative_capped_rounds != 0) {
        out << " | load-capped " << product::format_pretty_count(metrics.speculative_capped_rounds)
            << " rounds";
    }
    if (metrics.speculative_copy_rounds != 0) {
        out << " | copies " << product::format_pretty_count(metrics.speculative_copy_rounds)
            << " rounds, " << product::format_pretty_count(metrics.speculative_copy_accepted)
            << " accepted";
    }
    if (metrics.speculative_load_plain) {
        out << " | load plain " << product::format_pretty_count(metrics.speculative_plain_steps);
    }
    if (!metrics.speculative_fallback.empty() && metrics.speculative_fallback != "none") {
        out << " | fallback " << metrics.speculative_fallback;
    }
    if (outcome.thinking.configured_budget) {
        out << " | thinking "
            << product::format_pretty_count(outcome.thinking.model_thinking_tokens) << '/'
            << product::format_pretty_count(*outcome.thinking.configured_budget);
        if (outcome.thinking.injected_tokens != 0) {
            out << ", control " << product::format_pretty_count(outcome.thinking.injected_tokens);
        }
    }
    return {.severity = OperationalSeverity::Info, .message = out.str()};
}

std::optional<OperationalRecord> render_tool_call_fallback(const RequestLogContext& context,
                                                           const GenerationOutcome& outcome) {
    const ninferno::ToolCallParseFallbackReason reason = outcome.tool_call_parse.fallback_reason;
    if (!outcome.tool_call_parse.marker_seen ||
        reason == ninferno::ToolCallParseFallbackReason::None) {
        return std::nullopt;
    }
    return OperationalRecord{
        .severity = OperationalSeverity::Warning,
        .message  = "req#" + std::to_string(context.id) + " tool markup returned as text | " +
                   pretty_code(ninferno::tool_call_parse_fallback_reason_name(reason)),
    };
}

OperationalRecord render_request_failure(const RequestLogContext& context,
                                         const RequestFailure& failure) {
    std::ostringstream out;
    out << "req#" << context.id
        << (failure.classification == RequestFailureClass::ClientDisconnected ? " cancelled during "
                                                                              : " failed during ")
        << phase_name(failure.phase) << " | " << protocol_name(context.protocol);
    append_failure_fields(out, failure);
    return {.severity = failure_severity(failure.classification), .message = out.str()};
}

OperationalRecord render_response_failure(std::uint64_t request_id, const RequestFailure& failure) {
    std::ostringstream out;
    out << "req#" << request_id << " response failed during " << phase_name(failure.phase);
    append_failure_fields(out, failure);
    return {.severity = failure_severity(failure.classification), .message = out.str()};
}

std::optional<OperationalRecord> render_kv_shortfall(const ninferno::MemorySummary& memory,
                                                     std::uint32_t max_concurrency) {
    if (memory.kv_capacity_mode != ninferno::KvCapacityMode::Automatic ||
        memory.max_context_automatic || memory.kv_capacity_page_groups == 0 ||
        memory.kv_capacity_page_groups >= memory.kv_capacity_max_page_groups) {
        return std::nullopt;
    }
    // The pool is planned in whole pages: every slot's context rounded up to its pages, so the
    // pool that did not fit (and one page group short of it) can hold more tokens than C x ctx.
    const std::uint64_t page_tokens = memory.kv_capacity / memory.kv_capacity_page_groups;
    const std::uint64_t wanted      = page_tokens * memory.kv_capacity_max_page_groups;
    const std::uint64_t contexts    = std::uint64_t{max_concurrency} * memory.max_context;
    return OperationalRecord{
        .severity = OperationalSeverity::Info,
        .message  = fmt::format(
            "capacity | KV {} of {} tokens (auto: --max-concurrency {} x --max-context {}{} does "
            "not fit the {} free after weights; the most that does, at least one context): a "
            "request waits while the pool is full",
            product::format_pretty_count(memory.kv_capacity),
            product::format_pretty_count(wanted), max_concurrency,
            product::format_pretty_count(memory.max_context),
            wanted != contexts ? fmt::format(" in whole pages of {}", page_tokens) : std::string(),
            product::format_pretty_bytes(memory.available_after_weights_bytes))};
}

OperationalRecord render_throughput(const ThroughputReport& report) {
    const double prefill_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.computed_prefill_tokens) / report.interval_seconds
            : 0.0;
    const double decode_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.committed_decode_tokens) / report.interval_seconds
            : 0.0;
    std::ostringstream out;
    out << "throughput | " << product::format_pretty_duration(report.interval_seconds);
    if (report.computed_prefill_tokens != 0) {
        out << " | prefill " << product::format_pretty_rate(prefill_rate, "tok") << " ("
            << product::format_pretty_count(report.computed_prefill_tokens) << " tok)";
    }
    if (report.committed_decode_tokens != 0) {
        out << " | decode " << product::format_pretty_rate(decode_rate, "tok") << " ("
            << product::format_pretty_count(report.committed_decode_tokens) << " tok)";
    }
    out << " | running " << report.current.running_requests;
    if (report.current.prefilling_requests != 0 || report.current.decode_ready_requests != 0) {
        out << " (";
        bool separator = false;
        if (report.current.prefilling_requests != 0) {
            out << "prefill " << report.current.prefilling_requests;
            separator = true;
        }
        if (report.current.decode_ready_requests != 0) {
            if (separator) { out << ", "; }
            out << "decode-ready " << report.current.decode_ready_requests;
        }
        out << ')';
    }
    if (report.current.waiting_requests != 0) {
        out << " | waiting " << report.current.waiting_requests;
    }
    if (report.current.materializing_requests != 0) {
        out << " | materializing " << report.current.materializing_requests;
    }
    if (report.current.capture_pending_requests != 0) {
        out << " | capture-pending " << report.current.capture_pending_requests;
    }
    if (report.current.terminal_pending_requests != 0) {
        out << " | terminal-pending " << report.current.terminal_pending_requests;
    }
    if (report.decode_rounds != 0) {
        out << " | batch " << std::fixed << std::setprecision(2)
            << static_cast<double>(report.decode_row_rounds) /
                   static_cast<double>(report.decode_rounds);
    }
    const double host_seconds = static_cast<double>(host_active_ns(report)) * 1.0e-9;
    const double host_ratio =
        report.interval_seconds > 0.0 ? host_seconds / report.interval_seconds : 0.0;
    out << " | host " << product::format_pretty_percent(host_ratio) << " ("
        << product::format_pretty_duration(host_seconds) << ')';
    return {.severity = OperationalSeverity::Info, .message = out.str()};
}

OperationalLog::OperationalLog(std::shared_ptr<spdlog::logger> logger)
    : logger_(std::move(logger)) {}

void OperationalLog::write(OperationalRecord record) const {
    switch (record.severity) {
    case OperationalSeverity::Info:
        logger_->info("{}", record.message);
        return;
    case OperationalSeverity::Warning:
        logger_->warn("{}", record.message);
        return;
    case OperationalSeverity::Error:
        logger_->error("{}", record.message);
        return;
    }
}

void OperationalLog::request_start(const RequestLogContext& context) const {
    write(render_request_start(context));
}

void OperationalLog::request_rejected(const RequestRejectionLogContext& context) const {
    write(render_request_rejected(context));
}

void OperationalLog::request_done(const RequestLogContext& context,
                                  const GenerationOutcome& outcome) const {
    write(render_request_done(context, outcome));
    if (std::optional<OperationalRecord> fallback = render_tool_call_fallback(context, outcome)) {
        write(std::move(*fallback));
    }
}

void OperationalLog::request_failure(const RequestLogContext& context,
                                     const RequestFailure& failure) const {
    write(render_request_failure(context, failure));
}

void OperationalLog::response_failure(std::uint64_t request_id,
                                      const RequestFailure& failure) const {
    write(render_response_failure(request_id, failure));
}

void OperationalLog::response_not_stored(std::uint64_t request_id,
                                         std::size_t store_capacity_bytes) const {
    logger_->warn(
        "request {} | response not stored: it is larger than the response store ({} MiB) "
        "| answered with store:false; raise --response-store-max-mib to keep such answers",
        request_id, store_capacity_bytes >> 20);
}

void OperationalLog::throughput(const ThroughputReport& report) const {
    write(render_throughput(report));
}

void OperationalLog::http_failure(std::string_view endpoint, const RequestFailure& failure,
                                  std::string_view request_id) const {
    std::ostringstream out;
    out << "HTTP request failed during " << phase_name(failure.phase) << " | "
        << protocol_name(endpoint);
    if (!request_id.empty()) { out << " | request " << product::format_pretty_text(request_id); }
    append_failure_fields(out, failure);
    write({.severity = failure_severity(failure.classification), .message = out.str()});
}

void OperationalLog::engine_capacity(const GenerationService& service) const {
    const ninferno::MemorySummary memory            = service.memory_summary();
    const ninferno::EngineOptions& engine           = service.engine_options();
    const ninferno::ContextCacheOptions& cache      = engine.context_cache;
    const ninferno::ContextCostSummary context_cost = service.load_summary().context_cost;

    {
        const std::string native = product::format_pretty_count(memory.max_context_native);
        std::string origin;
        if (!memory.max_context_automatic) {
            origin = "--max-context; the model's own is " + native;
        } else if (memory.max_context == memory.max_context_native) {
            origin = "auto: the model's own";
        } else if (engine.kv_capacity.mode == ninferno::KvCapacityMode::Explicit) {
            origin = "auto: the model's own " + native + ", bounded by --kv-capacity";
        } else {
            origin = "auto: the model's own " + native + " does not fit the " +
                     product::format_pretty_bytes(memory.available_after_weights_bytes) +
                     " free after weights; the plan ladder's first rung that does";
        }
        logger_->info("context | {} tokens per request ({})",
                      product::format_pretty_count(memory.max_context), origin);
        for (OperationalRecord& record :
             render_plan(memory, !memory.max_context_automatic, engine.max_concurrency)) {
            write(std::move(record));
        }
        if (!memory.lower_concurrency_fits.empty()) {
            std::string fits;
            for (const ninferno::ConcurrencyFit& fit : memory.lower_concurrency_fits) {
                fits += (fits.empty() ? "" : "; ") + std::string("--max-concurrency ") +
                        std::to_string(fit.max_concurrency) + " fits " +
                        product::format_pretty_count(fit.max_context);
            }
            logger_->warn("context | {} tokens at --max-concurrency {} is below the plan "
                          "ladder's context floor (each slot keeps its own StateImage): {}",
                          product::format_pretty_count(memory.max_context), engine.max_concurrency,
                          fits);
        }
        if (engine.prefill_chunk != service.options().prefill_chunk) {
            logger_->info("prefill | chunk {} tokens (auto: the plan rung's; with {} no rung at "
                          "this context fits, and a narrower chunk takes less workspace and "
                          "StateImage)",
                          product::format_pretty_count(engine.prefill_chunk),
                          product::format_pretty_count(service.options().prefill_chunk));
        }
    }

    logger_->info("capacity | KV {} tokens, {}{}, {} | activations {}{} | pages {}/{} | runtime {} | "
                  "free {}",
                  product::format_pretty_count(memory.kv_capacity),
                  ninferno::kv_cache_storage_name(memory.kv_cache),
                  engine.sliding_kv_cache == ninferno::KvCacheStorage::BFloat16
                      ? ""
                      : std::string(", sliding rings ") +
                            std::string(ninferno::kv_cache_storage_name(engine.sliding_kv_cache)),
                  ninferno::kv_capacity_mode_name(memory.kv_capacity_mode),
                  product::activation_mode_name(engine.activations),
                  engine.recurrent_state == ninferno::RecurrentStateStorage::FP32
                      ? std::string()
                      : std::string(", state ") + std::string(ninferno::recurrent_state_storage_name(
                                                      engine.recurrent_state)),
                  product::format_pretty_count(memory.kv_capacity_page_groups),
                  product::format_pretty_count(memory.kv_capacity_max_page_groups),
                  product::format_pretty_bytes(memory.runtime_reservation_bytes),
                  product::format_pretty_bytes(memory.available_after_startup_bytes));
    // What the checkpoint's publisher validated its kept matrices with, against the mode in use.
    if (const auto& model = service.load_summary().metadata; !model.published_activations.empty()) {
        std::string widths;
        for (auto it = model.published_activations.rbegin(); it != model.published_activations.rend();
             ++it) {
            widths += (widths.empty() ? "" : ", ") +
                      (it->first == 0 ? std::string("unnamed") : std::to_string(it->first) + "-bit") +
                      " on " + std::to_string(it->second);
        }
        const auto below = engine.activations == ninferno::ActivationMode::Int8
                               ? model.below_published_activations_int8
                           : engine.activations == ninferno::ActivationMode::Fast
                               ? model.below_published_activations
                               : 0;
        if (below > 0) {
            logger_->warn("activations | the publisher validated {} matrices; {} allows {} of them "
                          "fewer bits where this card has the route (--activations exact keeps the "
                          "published width)",
                          widths, product::activation_mode_name(engine.activations), below);
        } else {
            logger_->info("activations | the publisher validated {} matrices", widths);
        }
    }
    // An explicit context with an automatic KV capacity (several slots, no --kv-capacity): every
    // slot's full context when it fits, else the most that does.
    if (const auto shortfall = render_kv_shortfall(memory, engine.max_concurrency)) {
        write(*shortfall);
    }
    if (memory.kernel_warm_up) {
        logger_->info("kernel warm-up | prefill widths 1..{} | ran {} | decode graphs {} {} | "
                      "{:.1f}s | kept {} (kernel code, stack, graphs) | stack {} B per thread "
                      "(driver default {} B{}) | Host tier pinning takes {} | graphs given back {}",
                      memory.kernel_warm_up_prefill_widths,
                      warm_up_widths_summary(memory.kernel_warm_up_widths_run,
                                             memory.kernel_warm_up_prefill_widths),
                      memory.kernel_warm_up_graph_passes,
                      memory.kernel_warm_up_graph_passes == 1 ? "pass" : "passes",
                      memory.kernel_warm_up_seconds,
                      product::format_pretty_bytes(memory.kernel_warm_up_lasting_bytes),
                      memory.stack_limit_bytes, memory.stack_limit_before_bytes,
                      memory.stack_limit_source.empty()
                          ? std::string()
                          : "; raised last by " + memory.stack_limit_source,
                      product::format_pretty_bytes(memory.host_pin_device_bytes),
                      product::format_pretty_bytes(memory.graph_returned_bytes));
    }

    {
        const ninferno::ModelSamplingDefaults sampling = service.sampling_defaults();
        const auto preset = [](const ninferno::SamplingPreset& p) {
            return fmt::format("temperature {:g}, top_k {}, top_p {:g}, min_p {:g}, presence {:g}, "
                               "frequency {:g}",
                               p.temperature, p.top_k, p.top_p, p.min_p, p.presence_penalty,
                               p.frequency_penalty);
        };
        logger_->info("sampling | defaults: {} | thinking: {} | non-thinking: {}{}",
                      sampling.source.empty() ? std::string("unspecified") : sampling.source,
                      preset(sampling.thinking), preset(sampling.non_thinking),
                      service.options().greedy ? " | --greedy: temperature 0" : "");
        for (const auto& note : sampling.notes) { logger_->warn("sampling | {}", note); }
    }

    if (!service.draft_depth_source().empty()) {
        logger_->info("speculation | --draft-tokens {} from {}",
                      service.engine_options().speculative.draft_tokens,
                      service.draft_depth_source());
    }
    if (const auto& served = service.served_speculation()) {
        const auto label = [](const spec::MenuChoice& c) {
            return c.config.enabled() ? c.config.label() : std::string("plain");
        };
        std::string depths;
        for (const std::int32_t depth : served->menu.depths()) {
            depths += (depths.empty() ? "" : ",") + std::to_string(depth);
        }
        logger_->info("speculation | {} | default {} | depths {} | {} | load {}", served->source,
                      label(served->menu.primary), depths.empty() ? "none" : depths,
                      service.options().per_request_speculation ? "by thinking flag" : "fixed",
                      product::speculation_load_label(served->engine.load));
        for (const auto& note : served->notes) { logger_->info("speculation | {}", note); }
        for (std::size_t i = 0; i < spec::kContextClasses; ++i) {
            const auto& c = served->classes[i];
            logger_->info("speculation | class {} | depth {} | {} | {}",
                          spec::class_name(static_cast<spec::ContextClass>(i)), c.depth,
                          spec::class_basis_label(c), c.reason);
        }
        if (!memory.speculative_graph_windows.empty()) {
            // An explicit plan reserves an allowance per window; an automatic one reserves none,
            // since the kernel warm-up measured the graphs' memory before the fit.
            std::string windows;
            bool reserved = false;
            for (const auto& window : memory.speculative_graph_windows) {
                if (window.first_batch == 0) { continue; } // no batch captures it
                windows += fmt::format("{}depth {} batches {}-{}", windows.empty() ? "" : ", ",
                                       window.draft_tokens, window.first_batch, window.last_batch);
                if (window.allowance_bytes > 0) {
                    windows += " " + product::format_pretty_bytes(window.allowance_bytes);
                    reserved = true;
                }
            }
            logger_->info("speculation | graphs {}{}", windows,
                          reserved ? "" : " (memory measured by the kernel warm-up)");
        }
        if (!served->fit_fallback.empty()) {
            logger_->warn("speculation | fit fallback: {}; running {}", served->fit_fallback,
                          label(served->menu.primary));
        }
    } else if (!service.speculation_plain_reason().empty()) {
        logger_->warn("speculation | {}", service.speculation_plain_reason());
    } else if (service.options().speculation_from_profile) {
        std::string why;
        for (const auto& note : service.speculation_absent_notes()) {
            why += (why.empty() ? " (" : "; ") + note;
        }
        const std::string line =
            fmt::format("speculation | the model came with no profile for this GPU: plain decode{}",
                        why.empty() ? std::string() : why + ")");
        // Asked for by --spec auto: a warning; by default: what a model without one does.
        if (service.options().speculation_by_default) {
            logger_->info("{}", line);
        } else {
            logger_->warn("{}", line);
        }
    }
    // One line with what runs since startup, whatever chose it (/v1/models meta.speculation).
    {
        const StartupSpeculation& startup = service.startup_speculation();
        logger_->info("speculation | serves {} ({}) | fallback {} | {}", startup.config,
                      startup.selection, startup.fallback, startup.reason);
    }
    // --spec-class-depth, --spec auto or an explicit --spec: the bounds the started Engine runs,
    // clamped to its captured depths (product::clamp_class_override).
    if (const auto& asked = service.options().class_depth) {
        logger_->info(
            "speculation | class bounds {} (--spec-class-depth {}, clamped to captured depths)",
            spec::class_depths_label(service.engine_options().speculative.class_depth),
            spec::class_depths_label(*asked));
    }

    if (cache.enabled) {
        std::string cached = std::to_string(*cache.device_state_slots);
        if (memory.cached_device_states_automatic) {
            // With an automatic KV capacity fewer states are kept for the pool (kv_capacity.h).
            const std::string fewer =
                memory.kv_capacity_mode == ninferno::KvCapacityMode::Automatic
                    ? " would leave the KV pool fewer whole slot contexts)"
                    : " does not fit in free memory)";
            cached += memory.cached_device_states < memory.cached_device_states_default
                          ? " (auto: the default " +
                                std::to_string(memory.cached_device_states_default) + fewer
                          : " (auto)";
        }
        std::string host = std::to_string(cache.host_state_slots) + " states, " +
                           product::format_pretty_bytes(cache.host_kv_capacity_bytes) + " KV";
        if (memory.host_tier_automatic) {
            host += memory.host_tier_reduced
                        ? " (auto: the default " + std::to_string(memory.host_state_slots_default) +
                              " states, " +
                              product::format_pretty_bytes(memory.host_kv_capacity_default_bytes) +
                              " KV does not fit the " +
                              product::format_pretty_bytes(memory.host_tier_budget_bytes) +
                              " this host can pin for it)"
                        : " (auto)";
        }
        logger_->info(
            "context cache | {} active + {} cached device states | host {} | "
            "private {} | shared {} | anchors {}",
            engine.max_concurrency, cached, host, *cache.max_private_continuations,
            *cache.max_shared_prefixes, *cache.max_long_anchors_per_continuation);
    } else {
        logger_->info("context cache | root only");
    }

    if (service.options().enable_vision) {
        const ninferno::MediaCacheSummary media = service.media_cache_summary();
        logger_->info("media | {} preprocess workers | cache {} | live {}",
                      media.preprocess_threads, product::format_pretty_bytes(media.capacity_bytes),
                      product::format_pretty_bytes(media.live_capacity_bytes));
    }

    // What startup took beyond the planned reservation (graphs, driver and module state): the
    // planned slack plus the graph allowance, less what is still free. Above the allowance plus
    // the headroom, the plan under-counted.
    const auto unplanned = static_cast<std::int64_t>(memory.planned_slack_bytes) +
                           static_cast<std::int64_t>(memory.cuda_graph_allowance_bytes) -
                           static_cast<std::int64_t>(memory.available_after_startup_bytes);
    logger_->debug("memory ledger | after weights {} | after startup {} | headroom {} | slack {} | "
                   "CUDA graphs {} planned, {}{} taken with the driver's",
                   product::format_pretty_bytes(memory.available_after_weights_bytes),
                   product::format_pretty_bytes(memory.available_after_startup_bytes),
                   product::format_pretty_bytes(memory.kv_capacity_headroom_bytes),
                   product::format_pretty_bytes(memory.planned_slack_bytes),
                   product::format_pretty_bytes(memory.cuda_graph_allowance_bytes),
                   unplanned < 0 ? "-" : "",
                   product::format_pretty_bytes(static_cast<std::size_t>(
                       unplanned < 0 ? -unplanned : unplanned)));
    // Under WSL 2 Windows shares the card and WDDM pages Device memory out to Host memory when it
    // fills: a pool left this close to full prefilled 4x slower (runtime/engine/kv_capacity.h).
    if (std::error_code error; std::filesystem::exists("/dev/dxg", error) &&
                               memory.available_after_startup_bytes < (256ULL << 20)) {
        logger_->warn("memory | {} left on a GPU Windows shares (WSL 2): when the card fills, "
                      "Windows pages memory out and prefill slows several times; a smaller "
                      "--max-context or --kv-capacity leaves room",
                      product::format_pretty_bytes(memory.available_after_startup_bytes));
    }
    logger_->debug("context cost | transfer {} | prefill {} | hardware {} | prefill signature {}",
                   ninferno::context_cost_preset_source_name(context_cost.transfer_source),
                   ninferno::context_cost_preset_source_name(context_cost.prefill_source),
                   product::format_pretty_text(context_cost.hardware_class),
                   product::format_pretty_text(context_cost.prefill_signature));
}

void OperationalLog::warmup_started() const { logger_->debug("warming up"); }

void OperationalLog::warmup_complete(double seconds) const {
    if (seconds >= 0.25) {
        logger_->info("warmup complete | {}", product::format_pretty_duration(seconds));
    } else {
        logger_->debug("warmup complete | {}", product::format_pretty_duration(seconds));
    }
}

void OperationalLog::warmup_failure(double seconds, std::string_view detail) const {
    logger_->critical("warmup failed | {} | {}", product::format_pretty_duration(seconds),
                      product::format_pretty_text(detail));
}

void OperationalLog::bind_failure(std::string_view host, int port) const {
    logger_->error("cannot bind {}:{}", product::format_pretty_text(host), port);
}

void OperationalLog::listen_failure(std::string_view host, int port) const {
    logger_->error("server listen failed | {}:{}", product::format_pretty_text(host), port);
}

void OperationalLog::server_ready(std::string_view host, int port, std::string_view model_id,
                                  bool auth_enabled) const {
    logger_->info("listening on http://{}:{} | model {} | auth {}",
                  product::format_pretty_text(host), port, product::format_pretty_text(model_id),
                  auth_enabled ? "bearer" : "disabled");
}

void OperationalLog::server_stopped() const { logger_->info("server stopped"); }

void OperationalLog::drain_started(std::uint32_t in_flight) const {
    logger_->info("draining | {} requests in flight | /health and new requests answer 503",
                  in_flight);
}

void OperationalLog::drain_finished(std::string_view reason, double seconds) const {
    logger_->info("drain finished | {} | {}", reason, product::format_pretty_duration(seconds));
}

void OperationalLog::executor_stall(bool stalled) const {
    if (stalled) {
        logger_->error("engine stalled | work is held and no progress was made | /health 503");
    } else {
        logger_->warn("engine progressing again | /health 200");
    }
}

void OperationalLog::engine_failed() const {
    logger_->critical("engine failed | it does not recover | draining, then exiting with status 1 "
                      "for the supervisor to restart the server");
}

void OperationalLog::server_failure(bool serving, std::string_view detail) const {
    logger_->critical("server failed during {} | {}", serving ? "serving" : "startup",
                      product::format_pretty_text(detail));
}

} // namespace ninferno::serve
