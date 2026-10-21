#include "serve/request_log.h"
#include "product/logging/pretty_format.h"
#include "product/activation_options.h"
#include "product/speculative_options.h"

#include <spdlog/logger.h>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include "core/sys.h"

namespace ninferno::serve {
namespace {

using Json = nlohmann::json;

template <class T>
T monotonic_delta(T previous, T current) noexcept {
    return current >= previous ? current - previous : T{};
}

std::uint64_t unix_time_ms() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

std::string new_server_instance_id() {
    const auto now    = std::chrono::system_clock::now().time_since_epoch();
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
    return "serve-" + std::to_string(sys::pid()) + '-' +
           std::to_string(micros);
}

std::filesystem::path normalized_absolute_path(const std::string& value) {
    std::error_code error;
    std::filesystem::path path = std::filesystem::weakly_canonical(value, error);
    if (!error) { return path; }
    error.clear();
    path = std::filesystem::absolute(value, error);
    return error ? std::filesystem::path(value).lexically_normal() : path.lexically_normal();
}

std::string cuda_version_string(int version) {
    if (version <= 0) { return {}; }
    return std::to_string(version / 1000) + '.' + std::to_string((version % 1000) / 10);
}

std::string cuda_uuid_string(const cudaUUID_t& uuid) {
    std::ostringstream out;
    out << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) { out << '-'; }
        out << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(uuid.bytes[i]));
    }
    return out.str();
}

Json tool_call_parse_json(const ninferno::ToolCallParseDiagnostics& diagnostics) {
    return Json{{"marker_seen", diagnostics.marker_seen},
                {"structured_call_count", diagnostics.structured_call_count},
                {"empty_arguments_omitted", diagnostics.empty_arguments_omitted},
                {"schema_mismatch_arguments", diagnostics.schema_mismatch_arguments},
                {"repeated_parameters", diagnostics.repeated_parameters},
                {"recovered_from_reasoning", diagnostics.recovered_from_reasoning},
                {"ended_inside_reasoning", diagnostics.ended_inside_reasoning},
                {"cut_calls_dropped", diagnostics.cut_calls_dropped},
                {"undeclared_calls", diagnostics.undeclared_calls},
                {"trailing_content_bytes", diagnostics.trailing_content_bytes},
                {"unwrapped_calls", diagnostics.unwrapped_calls},
                {"closed_calls", diagnostics.closed_calls},
                {"fallback_reason",
                 ninferno::tool_call_parse_fallback_reason_name(diagnostics.fallback_reason)}};
}

std::string tool_choice_name(const ToolChoice& choice) {
    switch (choice.mode) {
    case ToolChoiceMode::Auto:
        return "auto";
    case ToolChoiceMode::None:
        return "none";
    }
    return "unknown";
}

Json requested_reasoning_effort_json(const std::optional<ninferno::ReasoningEffort>& requested) {
    return requested ? Json(std::string(ninferno::reasoning_effort_name(*requested)))
                     : Json(nullptr);
}

const char* proposal_head_name(ninferno::ProposalHead proposal) {
    return proposal == ninferno::ProposalHead::Optimized ? "optimized" : "full";
}

const char* prefix_reuse_path_name(ninferno::PrefixReusePath path) {
    switch (path) {
    case ninferno::PrefixReusePath::Root:
        return "root";
    case ninferno::PrefixReusePath::PrivateEndpoint:
        return "private_endpoint";
    case ninferno::PrefixReusePath::PrivateTurnClosure:
        return "private_turn_closure";
    case ninferno::PrefixReusePath::PrivateResponseReplay:
        return "private_response_replay";
    case ninferno::PrefixReusePath::PrivateLongAnchor:
        return "private_long_anchor";
    case ninferno::PrefixReusePath::SharedStablePrefix:
        return "shared_stable_prefix";
    }
    return "unknown";
}

Json event_base(const std::string& server_instance_id, std::uint64_t timestamp, const char* event) {
    return Json{{"artifact_type", kRequestLogArtifactType},
                {"schema_version", kRequestLogSchemaVersion},
                {"event", event},
                {"timestamp_unix_ms", timestamp},
                {"server_instance_id", server_instance_id}};
}

Json sampler_json(const ninferno::ResolvedSamplingParameters& sampling) {
    return Json{{"temperature", sampling.temperature},
                {"top_p", sampling.top_p},
                {"top_k", sampling.top_k},
                {"min_p", sampling.min_p},
                {"presence_penalty", sampling.presence_penalty},
                {"frequency_penalty", sampling.frequency_penalty},
                {"seed", sampling.seed}};
}

Json preset_json(const ninferno::SamplingPreset& preset) {
    return Json{{"temperature", preset.temperature},
                {"top_p", preset.top_p},
                {"top_k", preset.top_k},
                {"min_p", preset.min_p},
                {"presence_penalty", preset.presence_penalty},
                {"frequency_penalty", preset.frequency_penalty}};
}

Json overrides_json(const ninferno::SamplingOverrides& overrides) {
    Json result{{"temperature", nullptr},
                {"top_p", nullptr},
                {"top_k", nullptr},
                {"min_p", nullptr},
                {"presence_penalty", nullptr},
                {"frequency_penalty", nullptr},
                {"seed", nullptr}};
    if (overrides.temperature) { result["temperature"] = *overrides.temperature; }
    if (overrides.top_p) { result["top_p"] = *overrides.top_p; }
    if (overrides.top_k) { result["top_k"] = *overrides.top_k; }
    if (overrides.min_p) { result["min_p"] = *overrides.min_p; }
    if (overrides.presence_penalty) { result["presence_penalty"] = *overrides.presence_penalty; }
    if (overrides.frequency_penalty) { result["frequency_penalty"] = *overrides.frequency_penalty; }
    if (overrides.seed) { result["seed"] = *overrides.seed; }
    return result;
}

Json request_json(const RequestLogContext& context) {
    Json thinking_budget = nullptr;
    if (context.thinking_budget) { thinking_budget = *context.thinking_budget; }
    return Json{{"request_id", context.id},
                {"protocol", context.protocol},
                {"model", context.model},
                {"stream", context.stream},
                {"message_count", context.message_count},
                {"media_item_count", context.media_item_count},
                {"requested_output_tokens", context.requested_output_tokens},
                {"requested_output_tokens_source",
                 context.requested_output_tokens_client_set ? "client" : "server_default"},
                {"tool_count", context.tool_count},
                {"tool_choice", tool_choice_name(context.tool_choice)},
                {"has_tool_history", context.has_tool_history},
                {"enable_thinking", context.enable_thinking},
                {"thinking_budget", std::move(thinking_budget)},
                {"requested_reasoning_effort",
                 requested_reasoning_effort_json(context.requested_reasoning_effort)},
                {"preserve_thinking",
                 context.preserve_thinking ? Json(*context.preserve_thinking) : Json(nullptr)},
                {"preserve_thinking_semantic_change", context.preserve_thinking_semantic_change},
                {"sampling", sampler_json(context.sampling)},
                {"notes", context.notes}};
}

Json preparation_json(const RequestLogContext& context) {
    const PromptPreparationStats& stats = context.preparation;
    return Json{{"total", stats.seconds},
                {"acquisition", context.acquisition_seconds},
                {"media_preprocess", stats.media_preprocess_seconds},
                {"media_preprocess_work", stats.media_preprocess_work_seconds},
                {"tokenize", stats.tokenize_seconds},
                {"media_items", stats.media_items},
                {"media_bytes", stats.media_bytes},
                {"raw_patches", stats.raw_patches},
                {"vision_tokens", stats.vision_tokens},
                {"patch_bytes", stats.patch_bytes},
                {"cache_hits", stats.media_cache_hits},
                {"cache_misses", stats.media_cache_misses},
                {"singleflight_waits", stats.media_singleflight_waits},
                {"built_patch_bytes", stats.built_patch_bytes},
                {"reused_patch_bytes", stats.reused_patch_bytes}};
}

Json rejected_request_json(const RequestRejectionLogContext& context) {
    return Json{{"request_id", context.id},
                {"protocol", context.protocol},
                {"model", context.model},
                {"stream", context.stream},
                {"message_count", context.message_count},
                {"media_item_count", context.media_item_count},
                {"requested_output_tokens", context.requested_output_tokens},
                {"requested_output_tokens_source",
                 context.requested_output_tokens_client_set ? "client" : "server_default"},
                {"tool_count", context.tool_count},
                {"tool_choice", tool_choice_name(context.tool_choice)},
                {"has_tool_history", context.has_tool_history},
                {"requested_reasoning_effort",
                 requested_reasoning_effort_json(context.requested_reasoning_effort)}};
}

Json error_json(const ApiError& error) {
    Json code  = error.code.empty() ? Json(nullptr) : Json(error.code);
    Json param = error.param.empty() ? Json(nullptr) : Json(error.param);
    return Json{{"status", error.status},
                {"type", error.type},
                {"code", std::move(code)},
                {"param", std::move(param)},
                {"message", error.message}};
}

Json arena_json(const ninferno::ArenaMemorySummary& arena) {
    return Json{{"capacity_bytes", arena.capacity_bytes},
                {"used_bytes", arena.used_bytes},
                {"peak_used_bytes", arena.peak_used_bytes}};
}

Json vision_workspace_json(const std::optional<ninferno::VisionWorkspaceMemorySummary>& vision) {
    if (!vision) { return nullptr; }
    return Json{{"aggregate_prompt_tokens", vision->aggregate_prompt_tokens},
                {"max_item_tokens", vision->max_item_tokens},
                {"general_capacity_bytes", vision->general_capacity_bytes},
                {"encode_peak_bytes", vision->encode_peak_bytes},
                {"handoff_offset_bytes", vision->handoff_offset_bytes},
                {"handoff_capacity_bytes", vision->handoff_capacity_bytes},
                {"handoff_active_bytes", vision->handoff_active_bytes},
                {"handoff_peak_bytes", vision->handoff_peak_bytes}};
}

Json speculative_json(const GenerationMetrics& metrics) {
    Json out{{"backend", product::speculative_backend_name(metrics.speculative_backend)},
             {"draft_window", metrics.speculative_draft_window},
             {"rounds", metrics.speculative_rounds},
             {"drafted_tokens", metrics.speculative_draft_tokens},
             {"accepted_tokens", metrics.speculative_accepted_tokens},
             {"fallback_steps", metrics.speculative_fallback_steps},
             {"plain_steps", metrics.speculative_plain_steps},
             {"wide_rounds", metrics.speculative_wide_rounds},
             {"ungraphed_rounds", metrics.speculative_ungraphed},
             {"capped_rounds", metrics.speculative_capped_rounds},
             {"copy_rounds", metrics.speculative_copy_rounds},
             {"copy_accepted_tokens", metrics.speculative_copy_accepted},
             {"shared_trip_deferrals", metrics.speculative_trips_held},
             {"load_plain", metrics.speculative_load_plain},
             {"context_plain", metrics.speculative_context_plain},
             {"fallback", metrics.speculative_fallback.empty() ? std::string("none")
                                                               : metrics.speculative_fallback},
             {"watchdog_tripped", metrics.speculative_watchdog_tripped},
             {"watchdog_round", metrics.speculative_watchdog_round},
             {"step_downs", metrics.speculative_step_downs},
             {"final_draft_window", metrics.speculative_final_draft_window},
             {"accepted_per_position", metrics.speculative_accepted_per_position}};
    if (!metrics.speculative_config.empty()) {
        out["config"]    = metrics.speculative_config;
        out["selection"] = metrics.speculative_selection;
        out["reason"]    = metrics.speculative_reason;
        out["floor"]     = metrics.speculative_floor;
        if (!metrics.speculative_note.empty()) { out["note"] = metrics.speculative_note; }
    }
    return out;
}

Json materialization_json(const ninferno::MaterializationDiagnostics& diagnostics) {
    return Json{
        {"predicted_now_ns", diagnostics.predicted_now_ns},
        {"predicted_future_loss_ns", diagnostics.predicted_future_loss_ns},
        {"predicted_total_ns", diagnostics.predicted_total_ns},
        {"targets_evaluated", diagnostics.targets_evaluated},
        {"projection_work", diagnostics.projection_work},
        {"planning_elapsed_ns", diagnostics.planning_elapsed_ns},
        {"search_elapsed_ns", diagnostics.search_elapsed_ns},
        {"stop_reason", ninferno::materialization_stop_reason_name(diagnostics.stop_reason)},
        {"budget_exhausted", diagnostics.budget_exhausted},
        {"selected_degradation_units", diagnostics.selected_degradation_units},
        {"selected_maximal_fallback", diagnostics.selected_maximal_fallback},
        {"initial_predicted_total_ns", diagnostics.initial_predicted_total_ns},
        {"first_improvement_ns", diagnostics.first_improvement_ns
                                     ? Json(*diagnostics.first_improvement_ns)
                                     : Json(nullptr)},
        {"incumbent_improvements", diagnostics.incumbent_improvements},
        {"search_work", diagnostics.search_work},
        {"search_granted_units", diagnostics.search_granted_units},
        {"search_units", diagnostics.search_units},
        {"planning_units", diagnostics.planning_units},
        {"search_renewals", diagnostics.search_renewals},
        {"search_discovery_used", diagnostics.search_discovery_used},
        {"search_stop_phase",
         ninferno::materialization_search_phase_name(diagnostics.search_stop_phase)},
        {"search_boundary_limited", diagnostics.search_boundary_limited},
    };
}

double nanoseconds_to_seconds(std::uint64_t value) noexcept {
    return static_cast<double>(value) * 1.0e-9;
}

double nanoseconds_to_microseconds(std::uint64_t value) noexcept {
    return static_cast<double>(value) * 1.0e-3;
}

double request_host_exposed_seconds(const ninferno::GenerationEngineTiming& timing) noexcept {
    return timing.engine_boundary_exposed_seconds + timing.program_submit_exposed_seconds +
           timing.program_post_exposed_seconds + timing.engine_commit_output_exposed_seconds +
           timing.engine_maintenance_exposed_seconds;
}

Json request_engine_timing_json(const ninferno::GenerationEngineTiming& timing) {
    return Json{
        {"queue_wait_seconds", timing.queue_wait_seconds},
        {"host_exposed_seconds",
         Json{{"engine_boundary", timing.engine_boundary_exposed_seconds},
              {"program_submit", timing.program_submit_exposed_seconds},
              {"program_post", timing.program_post_exposed_seconds},
              {"engine_commit_output", timing.engine_commit_output_exposed_seconds},
              {"engine_maintenance", timing.engine_maintenance_exposed_seconds},
              {"total", request_host_exposed_seconds(timing)}}},
        {"device_wait_exposed_seconds", timing.device_wait_exposed_seconds},
        {"decode", Json{{"host_exposed_seconds", timing.decode_host_exposed_seconds},
                        {"device_wait_exposed_seconds", timing.decode_device_wait_exposed_seconds},
                        {"rounds", timing.decode_rounds}}},
        {"units", Json{{"prefill", timing.prefill_units}, {"control", timing.control_units}}},
    };
}

ninferno::RuntimeHostWorkStats host_work_delta(const ninferno::RuntimeHostWorkStats& previous,
                                             const ninferno::RuntimeHostWorkStats& current) {
    return ninferno::RuntimeHostWorkStats{
        .engine_boundary_ns =
            monotonic_delta(previous.engine_boundary_ns, current.engine_boundary_ns),
        .program_submit_ns = monotonic_delta(previous.program_submit_ns, current.program_submit_ns),
        .program_post_ns   = monotonic_delta(previous.program_post_ns, current.program_post_ns),
        .engine_commit_output_ns =
            monotonic_delta(previous.engine_commit_output_ns, current.engine_commit_output_ns),
        .engine_maintenance_ns =
            monotonic_delta(previous.engine_maintenance_ns, current.engine_maintenance_ns),
        .device_wait_ns = monotonic_delta(previous.device_wait_ns, current.device_wait_ns),
        .decode_host_ns = monotonic_delta(previous.decode_host_ns, current.decode_host_ns),
        .decode_device_wait_ns =
            monotonic_delta(previous.decode_device_wait_ns, current.decode_device_wait_ns),
        .prefill_host_ns = monotonic_delta(previous.prefill_host_ns, current.prefill_host_ns),
        .prefill_device_wait_ns =
            monotonic_delta(previous.prefill_device_wait_ns, current.prefill_device_wait_ns),
        .control_host_ns = monotonic_delta(previous.control_host_ns, current.control_host_ns),
        .control_device_wait_ns =
            monotonic_delta(previous.control_device_wait_ns, current.control_device_wait_ns),
        .prefill_units = monotonic_delta(previous.prefill_units, current.prefill_units),
        .control_units = monotonic_delta(previous.control_units, current.control_units),
        .admission_policy_ns =
            monotonic_delta(previous.admission_policy_ns, current.admission_policy_ns),
        .context_progress_ns =
            monotonic_delta(previous.context_progress_ns, current.context_progress_ns),
        .stats_publication_ns =
            monotonic_delta(previous.stats_publication_ns, current.stats_publication_ns),
        .admission_policy_invocations  = monotonic_delta(previous.admission_policy_invocations,
                                                         current.admission_policy_invocations),
        .context_progress_invocations  = monotonic_delta(previous.context_progress_invocations,
                                                         current.context_progress_invocations),
        .stats_publication_invocations = monotonic_delta(previous.stats_publication_invocations,
                                                         current.stats_publication_invocations),
    };
}

std::uint64_t host_active_ns(const ninferno::RuntimeHostWorkStats& timing) noexcept {
    return timing.engine_boundary_ns + timing.program_submit_ns + timing.program_post_ns +
           timing.engine_commit_output_ns + timing.engine_maintenance_ns;
}

Json microseconds_per(std::uint64_t nanoseconds, std::uint64_t count) {
    if (count == 0) { return nullptr; }
    return nanoseconds_to_microseconds(nanoseconds) / static_cast<double>(count);
}

} // namespace

std::string format_server_start_json(
    const std::string& server_instance_id, std::uint64_t timestamp, const ServeOptions& options,
    const ninferno::EngineOptions& engine_options,
    const ninferno::ModelSamplingDefaults& sampling_defaults, const std::string& public_model_id,
    const ninferno::LoadSummary& load, const ninferno::MemorySummary& memory,
    const ServerLogEnvironment& environment, std::optional<std::uint64_t> artifact_size_bytes) {
    Json record = event_base(server_instance_id, timestamp, "server_start");

    Json artifact_size = nullptr;
    if (artifact_size_bytes.has_value()) { artifact_size = *artifact_size_bytes; }
    Json default_thinking_budget = nullptr;
    if (options.default_thinking_budget) {
        default_thinking_budget = *options.default_thinking_budget;
    }

    record["server"] =
        Json{{"host", options.host},
             {"port", options.port},
             {"public_model_id", public_model_id},
             {"api_key_configured", !options.api_key.empty()},
             {"cors_enabled", options.enable_cors},
             {"max_request_bytes", options.max_request_bytes},
             {"media_cache_bytes", options.media_cache_bytes},
             {"media_live_bytes", options.media_live_bytes},
             {"media_preprocess_threads", options.media_preprocess_threads},
             {"request_log_jsonl", options.request_log_jsonl},
             {"default_output_tokens", options.default_max_tokens},
             {"default_thinking",
              options.enable_thinking ? Json(*options.enable_thinking) : Json(nullptr)},
             {"default_thinking_budget", std::move(default_thinking_budget)},
             {"repetition_stop", options.repetition_stop},
             {"default_preserve_thinking",
              options.preserve_thinking ? Json(*options.preserve_thinking) : Json(nullptr)}};
    record["artifact"]                             = Json{{"path", options.artifact_path},
                                                          {"size_bytes", std::move(artifact_size)},
                                                          {"architecture", load.architecture},
                                                          {"name", load.model_name},
                                                          {"formats", load.weight_formats},
                                                          {"prefill_signature", load.prefill_signature},
                                                          {"bytes_read", load.artifact_bytes_read},
                                                          {"host_to_device_bytes", load.host_to_device_bytes},
                                                          {"peak_staging_bytes", load.peak_staging_bytes},
                                                          {"device_object_count", load.device_object_count},
                                                          {"host_object_count", load.host_object_count},
                                                          {"load_seconds", load.load_seconds},
                                                          {"upload_seconds", load.upload_seconds}};
    record["decision"] =
        Json{{"schema", "ninferno-fingerprint/1"},   {"fingerprint", load.decision_fingerprint},
             {"manifest", load.decision_manifest},   {"taint", load.decision_taint},
             {"canonical", load.decision_canonical}, {"observed", load.decision_observed}};
    const ninferno::ContextCacheOptions& cache       = engine_options.context_cache;
    const ninferno::ContextCostSummary& context_cost = load.context_cost;
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(engine_options.max_concurrency) +
        cache.device_state_slots.value();
    record["engine"] =
        Json{{"device", engine_options.device},
             {"max_context", engine_options.max_context},
             {"max_context_automatic", memory.max_context_automatic},
             {"max_context_native", memory.max_context_native},
             {"lower_concurrency_fits",
              [&memory] {
                  Json fits = Json::array();
                  for (const ninferno::ConcurrencyFit& fit : memory.lower_concurrency_fits) {
                      fits.push_back(Json{{"max_concurrency", fit.max_concurrency},
                                          {"max_context", fit.max_context}});
                  }
                  return fits;
              }()},
             {"kv_capacity_mode",
              std::string(ninferno::kv_capacity_mode_name(memory.kv_capacity_mode))},
             {"kv_capacity", memory.kv_capacity},
             {"kv_capacity_page_groups", memory.kv_capacity_page_groups},
             {"kv_capacity_max_page_groups", memory.kv_capacity_max_page_groups},
             {"max_concurrency", engine_options.max_concurrency},
             {"max_pending_requests", engine_options.max_pending_requests},
             {"pending_timeout_ms", engine_options.pending_timeout_ms},
             {"prefill_chunk", engine_options.prefill_chunk},
             {"log_stats_interval_ms", options.log_stats_interval_ms},
             {"kv_cache", std::string(ninferno::kv_cache_storage_layout(engine_options.kv_cache))},
             {"sliding_kv_cache",
              std::string(ninferno::kv_cache_storage_layout(engine_options.sliding_kv_cache))},
             {"activations", product::activation_mode_name(engine_options.activations)},
             {"vision", engine_options.enable_vision},
             {"cuda_graph", engine_options.use_cuda_graph},
             {"prefix_reuse", options.allow_prefix_reuse},
             {"speculative_backend",
              product::speculative_backend_name(engine_options.speculative.backend)},
             {"speculative_draft_window", engine_options.speculative.draft_tokens},
             {"speculative_request_depths", engine_options.speculative.request_draft_tokens},
             {"speculative_load", product::speculation_load_label(engine_options.speculative.load)},
             {"proposal_head", proposal_head_name(engine_options.speculative.proposal_head)},
             {"context_cost", Json{{"transfer_source", ninferno::context_cost_preset_source_name(
                                                           context_cost.transfer_source)},
                                   {"prefill_source", ninferno::context_cost_preset_source_name(
                                                          context_cost.prefill_source)},
                                   {"hardware_class", context_cost.hardware_class},
                                   {"prefill_signature", context_cost.prefill_signature},
                                   {"preset_path", context_cost.preset_path.string()},
                                   {"similar_to", context_cost.similar_to}}},
             {"context_cache",
              Json{{"enabled", cache.enabled},
                   {"device_state_slots", cache.device_state_slots.value()},
                   {"device_state_slots_automatic", memory.cached_device_states_automatic},
                   {"device_state_slots_default", memory.cached_device_states_default},
                   {"total_device_state_slots", total_device_state_slots},
                   {"host_state_slots", cache.host_state_slots},
                   {"host_kv_capacity_bytes", cache.host_kv_capacity_bytes},
                   {"host_tier_automatic", memory.host_tier_automatic},
                   {"host_tier_reduced", memory.host_tier_reduced},
                   {"host_state_slots_default", memory.host_state_slots_default},
                   {"host_kv_capacity_default_bytes", memory.host_kv_capacity_default_bytes},
                   {"host_tier_budget_bytes", memory.host_tier_budget_bytes},
                   {"max_private_continuations", cache.max_private_continuations.value()},
                   {"max_shared_prefixes", cache.max_shared_prefixes.value()},
                   {"max_long_anchors_per_continuation",
                    cache.max_long_anchors_per_continuation.value()}}}};
    record["sampling_defaults"] =
        Json{{"source", sampling_defaults.source},
             {"notes", sampling_defaults.notes},
             {"thinking", preset_json(sampling_defaults.thinking)},
             {"non_thinking", preset_json(sampling_defaults.non_thinking)},
             {"server_overrides", overrides_json(options.sampling_overrides)},
             {"omitted_seed", "random"},
             {"greedy", options.greedy}};
    record["memory"] =
        Json{{"weights", arena_json(memory.weights)},
             {"sequence", arena_json(memory.sequence)},
             {"workspace", arena_json(memory.workspace)},
             {"vision_workspace", vision_workspace_json(memory.vision_workspace)},
             {"minimum_runtime_reservation_bytes", memory.minimum_runtime_reservation_bytes},
             {"kv_capacity_increment_bytes", memory.kv_capacity_increment_bytes},
             {"runtime_reservation_bytes", memory.runtime_reservation_bytes},
             {"available_after_weights_bytes", memory.available_after_weights_bytes},
             {"available_after_startup_bytes", memory.available_after_startup_bytes},
             {"kv_capacity_headroom_bytes", memory.kv_capacity_headroom_bytes},
             {"planned_slack_bytes", memory.planned_slack_bytes},
             {"cuda_graph_allowance_bytes", memory.cuda_graph_allowance_bytes},
             {"kv_payload_bytes", memory.kv_payload_bytes},
             {"host_state_capacity_slots", memory.host_state_capacity_slots},
             {"host_state_occupied_slots", memory.host_state_occupied_slots},
             {"host_kv_capacity_bytes", memory.host_kv_capacity_bytes},
             {"host_kv_occupied_bytes", memory.host_kv_occupied_bytes},
             {"kernel_warm_up", Json{{"ran", memory.kernel_warm_up},
                                     {"prefill_widths", memory.kernel_warm_up_prefill_widths},
                                     {"widths_run", memory.kernel_warm_up_widths_run},
                                     {"graph_passes", memory.kernel_warm_up_graph_passes},
                                     {"seconds", memory.kernel_warm_up_seconds},
                                     {"lasting_bytes", memory.kernel_warm_up_lasting_bytes},
                                     {"stack_limit_before_bytes", memory.stack_limit_before_bytes},
                                     {"stack_limit_bytes", memory.stack_limit_bytes},
                                     {"stack_limit_source", memory.stack_limit_source},
                                     {"host_pin_device_bytes", memory.host_pin_device_bytes},
                                     {"graph_returned_bytes", memory.graph_returned_bytes}}}};
    record["environment"] =
        Json{{"device", environment.device},
             {"gpu_name", environment.gpu_name},
             {"gpu_uuid", environment.gpu_uuid},
             {"total_device_memory_bytes", environment.total_device_memory_bytes},
             {"compute_capability_major", environment.compute_capability_major},
             {"compute_capability_minor", environment.compute_capability_minor},
             {"cuda_compile_version", environment.cuda_compile_version},
             {"cuda_runtime_version", environment.cuda_runtime_version},
             {"cuda_driver_version", environment.cuda_driver_version}};
    record["argv"] = options.startup_argv;
    return record.dump();
}

std::string format_request_start_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp, const RequestLogContext& context) {
    Json record                   = event_base(server_instance_id, timestamp, "request_start");
    record["request"]             = request_json(context);
    record["preparation_seconds"] = preparation_json(context);
    return record.dump();
}

std::string format_request_rejected_json(const std::string& server_instance_id,
                                         std::uint64_t timestamp,
                                         const RequestRejectionLogContext& context) {
    Json record       = event_base(server_instance_id, timestamp, "request_rejected");
    record["phase"]   = "prepare";
    record["request"] = rejected_request_json(context);
    record["error"]   = error_json(context.error);
    return record.dump();
}

std::string format_request_done_json(const std::string& server_instance_id, std::uint64_t timestamp,
                                     const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
    Json record       = event_base(server_instance_id, timestamp, "request_done");
    record["request"] = request_json(context);
    record["result"] =
        Json{{"finish_reason", std::string(ninferno::finish_reason_name(outcome.finish_reason))},
             {"prompt_tokens", outcome.prompt_tokens},
             {"completion_tokens", outcome.completion_tokens},
             {"computed_prefill_tokens",
              std::max(0, outcome.prompt_tokens -
                              static_cast<int>(outcome.metrics.prefix_cache_hit_tokens))},
             {"prefix_cache_hit_tokens", outcome.metrics.prefix_cache_hit_tokens},
             {"prefix_reuse_path", prefix_reuse_path_name(outcome.metrics.prefix_reuse_path)},
             {"thinking_budget", outcome.thinking.configured_budget
                                     ? Json(*outcome.thinking.configured_budget)
                                     : Json(nullptr)},
             {"model_thinking_tokens", outcome.thinking.model_thinking_tokens},
             {"thinking_control_tokens", outcome.thinking.injected_tokens},
             {"thinking_control_applied", outcome.thinking.applied},
             {"tool_call_count", outcome.tool_calls.size()},
             {"tool_call_parse", tool_call_parse_json(outcome.tool_call_parse)}};
    record["timings_seconds"] = Json{
        {"prepare", outcome.metrics.prepare_seconds}, {"ttft", outcome.metrics.ttft_seconds},
        {"vision", outcome.metrics.vision_seconds},   {"prefill", outcome.metrics.prefill_seconds},
        {"decode", outcome.metrics.decode_seconds},   {"total", outcome.metrics.total_seconds}};
    record["engine_timing"]   = request_engine_timing_json(outcome.metrics.engine_timing);
    record["speculative"]     = speculative_json(outcome.metrics);
    record["materialization"] = materialization_json(outcome.metrics.materialization);
    return record.dump();
}

std::string format_request_error_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp, const RequestLogContext& context,
                                      const std::string& message) {
    Json record       = event_base(server_instance_id, timestamp, "request_error");
    record["request"] = request_json(context);
    record["error"]   = Json{{"message", message}};
    return record.dump();
}

std::string format_throughput_json(const std::string& server_instance_id, std::uint64_t timestamp,
                                   const ThroughputReport& report) {
    Json record                          = event_base(server_instance_id, timestamp, "throughput");
    const ninferno::RuntimeStats& previous = report.previous;
    const ninferno::RuntimeStats& current  = report.current;
    const double prefill_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.computed_prefill_tokens) / report.interval_seconds
            : 0.0;
    const double decode_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.committed_decode_tokens) / report.interval_seconds
            : 0.0;
    Json average_batch = nullptr;
    if (report.decode_rounds != 0) {
        average_batch = static_cast<double>(report.decode_row_rounds) /
                        static_cast<double>(report.decode_rounds);
    }
    const ninferno::RuntimeHostWorkStats host =
        host_work_delta(previous.host_work, current.host_work);
    const std::uint64_t active_host = host_active_ns(host);
    record["interval_seconds"]      = report.interval_seconds;
    record["tokens"]                = Json{{"computed_prefill", report.computed_prefill_tokens},
                                           {"committed_decode", report.committed_decode_tokens}};
    record["throughput_tokens_per_second"] =
        Json{{"prefill", prefill_rate}, {"decode", decode_rate}};
    record["scheduler"]    = Json{{"running", current.running_requests},
                                  {"prefilling", current.prefilling_requests},
                                  {"decode_ready", current.decode_ready_requests},
                                  {"waiting", current.waiting_requests},
                                  {"materializing", current.materializing_requests},
                                  {"capture_pending", current.capture_pending_requests},
                                  {"terminal_pending", current.terminal_pending_requests}};
    record["decode_batch"] = Json{{"rounds", report.decode_rounds},
                                  {"row_rounds", report.decode_row_rounds},
                                  {"average_size", std::move(average_batch)}};
    record["host_work"]    = Json{
           {"elapsed_seconds",
            Json{{"engine_boundary", nanoseconds_to_seconds(host.engine_boundary_ns)},
                 {"program_submit", nanoseconds_to_seconds(host.program_submit_ns)},
                 {"program_post", nanoseconds_to_seconds(host.program_post_ns)},
                 {"engine_commit_output", nanoseconds_to_seconds(host.engine_commit_output_ns)},
                 {"engine_maintenance", nanoseconds_to_seconds(host.engine_maintenance_ns)},
                 {"total", nanoseconds_to_seconds(active_host)}}},
           {"device_wait_seconds", nanoseconds_to_seconds(host.device_wait_ns)},
           {"work_class_seconds",
            Json{{"decode_host", nanoseconds_to_seconds(host.decode_host_ns)},
                 {"decode_device_wait", nanoseconds_to_seconds(host.decode_device_wait_ns)},
                 {"prefill_host", nanoseconds_to_seconds(host.prefill_host_ns)},
                 {"prefill_device_wait", nanoseconds_to_seconds(host.prefill_device_wait_ns)},
                 {"control_host", nanoseconds_to_seconds(host.control_host_ns)},
                 {"control_device_wait", nanoseconds_to_seconds(host.control_device_wait_ns)}}},
           {"detail_subset_seconds",
            Json{{"admission_policy", nanoseconds_to_seconds(host.admission_policy_ns)},
                 {"context_progress", nanoseconds_to_seconds(host.context_progress_ns)},
                 {"stats_publication", nanoseconds_to_seconds(host.stats_publication_ns)}}},
           {"detail_invocations", Json{{"admission_policy", host.admission_policy_invocations},
                                       {"context_progress", host.context_progress_invocations},
                                       {"stats_publication", host.stats_publication_invocations}}},
           {"units", Json{{"prefill", host.prefill_units}, {"control", host.control_units}}},
           {"decode_host_microseconds_per_round",
            microseconds_per(host.decode_host_ns, report.decode_rounds)},
           {"decode_host_microseconds_per_row_round",
            microseconds_per(host.decode_host_ns, report.decode_row_rounds)},
           {"decode_device_wait_microseconds_per_round",
            microseconds_per(host.decode_device_wait_ns, report.decode_rounds)},
           {"detail_microseconds_per_invocation",
            Json{{"admission_policy",
                  microseconds_per(host.admission_policy_ns, host.admission_policy_invocations)},
                 {"context_progress",
                  microseconds_per(host.context_progress_ns, host.context_progress_invocations)},
                 {"stats_publication",
                  microseconds_per(host.stats_publication_ns, host.stats_publication_invocations)}}},
    };
    record["context_cache"] = Json{
        {"captures", Json{{"completed", monotonic_delta(previous.active_captures_completed,
                                                        current.active_captures_completed)},
                          {"aborted", monotonic_delta(previous.active_captures_aborted,
                                                      current.active_captures_aborted)}}},
        {"selections",
         Json{{"root", monotonic_delta(previous.root_selections, current.root_selections)},
              {"private_endpoint", monotonic_delta(previous.private_endpoint_selections,
                                                   current.private_endpoint_selections)},
              {"private_turn_closure", monotonic_delta(previous.private_turn_closure_selections,
                                                       current.private_turn_closure_selections)},
              {"private_response_replay",
               monotonic_delta(previous.private_response_replay_selections,
                               current.private_response_replay_selections)},
              {"private_long_anchor", monotonic_delta(previous.private_long_anchor_selections,
                                                      current.private_long_anchor_selections)},
              {"shared_stable_prefix", monotonic_delta(previous.shared_stable_prefix_selections,
                                                       current.shared_stable_prefix_selections)},
              {"reused_prompt_tokens",
               monotonic_delta(previous.reused_prompt_tokens, current.reused_prompt_tokens)}}},
        {"last_selection", Json{{"frontier_tokens", current.last_selected_frontier_tokens}}},
        {"state_operations",
         Json{{"moves", monotonic_delta(previous.state_moves, current.state_moves)},
              {"forks", monotonic_delta(previous.state_forks, current.state_forks)},
              {"restores", monotonic_delta(previous.state_restores, current.state_restores)}}},
        {"state_transfers",
         Json{{"d2h",
               Json{{"count", monotonic_delta(previous.state_d2h_count, current.state_d2h_count)},
                    {"bytes", monotonic_delta(previous.state_d2h_bytes, current.state_d2h_bytes)},
                    {"seconds",
                     monotonic_delta(previous.state_d2h_seconds, current.state_d2h_seconds)}}},
              {"h2d",
               Json{{"count", monotonic_delta(previous.state_h2d_count, current.state_h2d_count)},
                    {"bytes", monotonic_delta(previous.state_h2d_bytes, current.state_h2d_bytes)},
                    {"seconds",
                     monotonic_delta(previous.state_h2d_seconds, current.state_h2d_seconds)}}},
              {"d2d",
               Json{{"count", monotonic_delta(previous.state_d2d_count, current.state_d2d_count)},
                    {"bytes", monotonic_delta(previous.state_d2d_bytes, current.state_d2d_bytes)},
                    {"seconds",
                     monotonic_delta(previous.state_d2d_seconds, current.state_d2d_seconds)}}}}},
        {"main_kv_transfers",
         Json{
             {"d2h",
              Json{
                  {"pages", monotonic_delta(previous.main_kv_d2h_pages, current.main_kv_d2h_pages)},
                  {"bytes", monotonic_delta(previous.main_kv_d2h_bytes, current.main_kv_d2h_bytes)},
                  {"seconds",
                   monotonic_delta(previous.main_kv_d2h_seconds, current.main_kv_d2h_seconds)}}},
             {"h2d",
              Json{
                  {"pages", monotonic_delta(previous.main_kv_h2d_pages, current.main_kv_h2d_pages)},
                  {"bytes", monotonic_delta(previous.main_kv_h2d_bytes, current.main_kv_h2d_bytes)},
                  {"seconds",
                   monotonic_delta(previous.main_kv_h2d_seconds, current.main_kv_h2d_seconds)}}},
             {"d2d",
              Json{
                  {"pages", monotonic_delta(previous.main_kv_d2d_pages, current.main_kv_d2d_pages)},
                  {"bytes", monotonic_delta(previous.main_kv_d2d_bytes, current.main_kv_d2d_bytes)},
                  {"seconds",
                   monotonic_delta(previous.main_kv_d2d_seconds, current.main_kv_d2d_seconds)}}}}},
        {"backend_kv_transfers",
         Json{{"d2h", Json{{"pages", monotonic_delta(previous.backend_kv_d2h_pages,
                                                     current.backend_kv_d2h_pages)},
                           {"bytes", monotonic_delta(previous.backend_kv_d2h_bytes,
                                                     current.backend_kv_d2h_bytes)},
                           {"seconds", monotonic_delta(previous.backend_kv_d2h_seconds,
                                                       current.backend_kv_d2h_seconds)}}},
              {"h2d", Json{{"pages", monotonic_delta(previous.backend_kv_h2d_pages,
                                                     current.backend_kv_h2d_pages)},
                           {"bytes", monotonic_delta(previous.backend_kv_h2d_bytes,
                                                     current.backend_kv_h2d_bytes)},
                           {"seconds", monotonic_delta(previous.backend_kv_h2d_seconds,
                                                       current.backend_kv_h2d_seconds)}}},
              {"d2d", Json{{"pages", monotonic_delta(previous.backend_kv_d2d_pages,
                                                     current.backend_kv_d2d_pages)},
                           {"bytes", monotonic_delta(previous.backend_kv_d2d_bytes,
                                                     current.backend_kv_d2d_bytes)},
                           {"seconds", monotonic_delta(previous.backend_kv_d2d_seconds,
                                                       current.backend_kv_d2d_seconds)}}}}},
        {"pressure",
         Json{
             {"spill_pages",
              monotonic_delta(previous.pressure_spill_pages, current.pressure_spill_pages)},
             {"partial_tail_cow_pages",
              monotonic_delta(previous.partial_tail_cow_pages, current.partial_tail_cow_pages)},
             {"private_owners_degraded", monotonic_delta(previous.pressure_private_owners_degraded,
                                                         current.pressure_private_owners_degraded)},
             {"private_owners_evicted", monotonic_delta(previous.pressure_private_owners_evicted,
                                                        current.pressure_private_owners_evicted)},
             {"shared_owners_degraded", monotonic_delta(previous.pressure_shared_owners_degraded,
                                                        current.pressure_shared_owners_degraded)},
             {"shared_owners_evicted", monotonic_delta(previous.pressure_shared_owners_evicted,
                                                       current.pressure_shared_owners_evicted)},
             {"checkpoints_dropped", monotonic_delta(previous.pressure_checkpoints_dropped,
                                                     current.pressure_checkpoints_dropped)},
             {"searches", monotonic_delta(previous.pressure_searches, current.pressure_searches)},
             {"search_budget_exhaustions",
              monotonic_delta(previous.pressure_search_budget_exhaustions,
                              current.pressure_search_budget_exhaustions)},
             {"maximal_fallback_selections",
              monotonic_delta(previous.pressure_maximal_fallback_selections,
                              current.pressure_maximal_fallback_selections)},
             {"historical_fork_hits",
              monotonic_delta(previous.historical_fork_hits, current.historical_fork_hits)}}},
        {"occupancy", Json{{"device_state_slots", current.device_state_occupied_slots},
                           {"host_state_slots", current.host_state_occupied_slots},
                           {"device_main_kv_pages", current.device_main_kv_occupied_pages},
                           {"device_backend_kv_pages", current.device_backend_kv_occupied_pages},
                           {"host_kv_bytes", current.host_kv_occupied_bytes},
                           {"shared_active_references", current.shared_active_references}}},
        {"actual_transfer_seconds", monotonic_delta(previous.actual_context_transfer_seconds,
                                                    current.actual_context_transfer_seconds)}};
    return record.dump();
}

ServerLogEnvironment query_server_log_environment(int device) {
    ServerLogEnvironment environment;
    environment.device               = device;
    environment.cuda_compile_version = cuda_version_string(CUDART_VERSION);

    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        environment.cuda_runtime_version = cuda_version_string(runtime_version);
    }
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
        environment.cuda_driver_version = cuda_version_string(driver_version);
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
        environment.gpu_name                  = properties.name;
        environment.gpu_uuid                  = cuda_uuid_string(properties.uuid);
        environment.total_device_memory_bytes = properties.totalGlobalMem;
        environment.compute_capability_major  = properties.major;
        environment.compute_capability_minor  = properties.minor;
    }
    return environment;
}

JsonlRequestLog::JsonlRequestLog(const std::string& path,
                                 const std::string& protected_artifact_path,
                                 std::shared_ptr<spdlog::logger> logger)
    : path_(path), logger_(std::move(logger)) {
    if (path_.empty()) { return; }
    if (!protected_artifact_path.empty() &&
        normalized_absolute_path(path_) == normalized_absolute_path(protected_artifact_path)) {
        throw std::invalid_argument("request JSONL log must not overwrite the model artifact");
    }
    server_instance_id_ = new_server_instance_id();
    output_.open(path_, std::ios::out | std::ios::app);
    if (!output_) {
        throw std::runtime_error("failed to open request JSONL log for append: " + path_);
    }
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(path_, error);
    bytes_                    = error ? 0 : size;
}

void JsonlRequestLog::write_server_start(const ServeOptions& options,
                                         const ninferno::EngineOptions& engine_options,
                                         const ninferno::ModelSamplingDefaults& sampling_defaults,
                                         const std::string& public_model_id,
                                         const ninferno::LoadSummary& load,
                                         const ninferno::MemorySummary& memory) {
    if (!enabled()) { return; }
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(options.artifact_path, error);
    const std::optional<std::uint64_t> artifact_size =
        error ? std::nullopt : std::optional<std::uint64_t>(size);
    append(format_server_start_json(server_instance_id_, unix_time_ms(), options, engine_options,
                                    sampling_defaults, public_model_id, load, memory,
                                    query_server_log_environment(options.device), artifact_size));
}

void JsonlRequestLog::write_request_start(const RequestLogContext& context) {
    if (!enabled()) { return; }
    append(format_request_start_json(server_instance_id_, unix_time_ms(), context));
}

void JsonlRequestLog::write_request_rejected(const RequestRejectionLogContext& context) {
    if (!enabled()) { return; }
    append(format_request_rejected_json(server_instance_id_, unix_time_ms(), context));
}

void JsonlRequestLog::write_request_done(const RequestLogContext& context,
                                         const GenerationOutcome& outcome) {
    if (!enabled()) { return; }
    append(format_request_done_json(server_instance_id_, unix_time_ms(), context, outcome));
}

void JsonlRequestLog::write_request_error(const RequestLogContext& context,
                                          const std::string& message) {
    if (!enabled()) { return; }
    append(format_request_error_json(server_instance_id_, unix_time_ms(), context, message));
}

void JsonlRequestLog::write_throughput(const ThroughputReport& report) {
    if (!enabled()) { return; }
    append(format_throughput_json(server_instance_id_, unix_time_ms(), report));
}

void JsonlRequestLog::append(std::string record) {
    bool report_failure = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failed_) { return; }
        if (bytes_ >= kRotateBytes) {
            output_.close();
            std::error_code error;
            std::filesystem::rename(path_, path_ + ".1", error);
            bytes_ = 0;
            output_.open(path_, std::ios::out | std::ios::app);
        }
        output_ << record << '\n';
        output_.flush();
        bytes_ += record.size() + 1;
        if (!output_) {
            failed_        = true;
            report_failure = true;
        }
    }
    if (report_failure && logger_ != nullptr) {
        logger_->error("request log disabled | write failed | {}",
                       product::format_pretty_text(path_));
    }
}

} // namespace ninferno::serve
