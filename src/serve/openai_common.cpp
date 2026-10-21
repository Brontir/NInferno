#include "serve/openai_common.h"
#include "serve/request_validation.h"

#include "ninferno/types.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <vector>
#include <string_view>
#include <utility>

namespace ninferno::serve {

namespace {

// Responses API identifiers: `prefix`_ and 32 hex digits.
std::string responses_identifier(std::string_view prefix) {
    return random_identifier(std::string(prefix) + "_", 2);
}

} // namespace

using Json = nlohmann::json;

namespace {

// OpenAI model object carrying the llama.cpp-compatible `meta` field. `id` is the public OpenAI
// alias; the `meta` facts describe the registered artifact behind it. n_ctx is the configured
// per-request context ceiling (equal to max_model_len); n_ctx_train is the model's native
// training context. ftype is the registered weights profile (the NInferno quantization name).
Json make_model_object_json(const std::string& model_id, std::int64_t created,
                            std::uint32_t max_model_len, const ninferno::ModelMetadata& metadata) {
    Json meta{{"n_vocab", metadata.vocab_size},
              {"n_ctx", max_model_len},
              {"n_ctx_train", metadata.native_context},
              {"n_embd", metadata.embedding_size},
              {"n_params", metadata.parameters},
              {"size", metadata.weight_bytes},
              {"ftype", metadata.weights_id}};
    if (!metadata.published_activations.empty()) {
        Json published = Json::object();
        for (const auto& [bits, count] : metadata.published_activations) {
            published[bits == 0 ? std::string("unknown") : std::to_string(bits)] = count;
        }
        meta["published_activations"] = published;
    }
    return Json{{"id", model_id},
                {"object", "model"},
                {"created", created},
                {"owned_by", "ninferno"},
                {"max_model_len", max_model_len},
                {"meta", std::move(meta)}};
}

} // namespace

bool parse_openai_prompt_cache_breakpoint(const RequestJson& value, std::string_view param) {
    if (!value.contains("prompt_cache_breakpoint") ||
        value.at("prompt_cache_breakpoint").is_null()) {
        return false;
    }
    const RequestJson& breakpoint = value.at("prompt_cache_breakpoint");
    if (!breakpoint.is_object() || !breakpoint.contains("mode") ||
        !breakpoint.at("mode").is_string() ||
        breakpoint.at("mode").get<std::string>() != "explicit") {
        bad_request("prompt_cache_breakpoint must be {mode:'explicit'}", std::string(param),
                    "invalid_cache_breakpoint");
    }
    return true;
}

OpenAIPromptCachePolicy parse_openai_prompt_cache_policy(const RequestJson& body) {
    auto require_string_hint = [&](const char* field, std::optional<std::size_t> maximum = {}) {
        if (!body.contains(field) || body.at(field).is_null()) { return; }
        if (!body.at(field).is_string()) {
            bad_request(std::string(field) + " must be a string", field);
        }
        if (maximum && body.at(field).get_ref<const std::string&>().size() > *maximum) {
            bad_request(std::string(field) + " must be at most " + std::to_string(*maximum) +
                            " characters",
                        field);
        }
    };
    // Neutral hints: a cache key may be as long as a context-cache session key, and OpenAI allows
    // 128 characters of safety identifier (gateways hash user ids into them).
    require_string_hint("prompt_cache_key", kMaximumContextCacheSessionKeyBytes);
    require_string_hint("safety_identifier", 128U);
    require_string_hint("user");

    if (body.contains("prompt_cache_retention") && !body.at("prompt_cache_retention").is_null()) {
        if (!body.at("prompt_cache_retention").is_string()) {
            bad_request("prompt_cache_retention must be a string", "prompt_cache_retention");
        }
        const std::string value = body.at("prompt_cache_retention").get<std::string>();
        if (value != "in_memory" && value != "24h") {
            bad_request("prompt_cache_retention must be 'in_memory' or '24h'",
                        "prompt_cache_retention");
        }
    }

    OpenAIPromptCachePolicy policy;
    if (!body.contains("prompt_cache_options") || body.at("prompt_cache_options").is_null()) {
        return policy;
    }
    const RequestJson& options = body.at("prompt_cache_options");
    if (!options.is_object()) {
        bad_request("prompt_cache_options must be an object", "prompt_cache_options");
    }
    policy.automatic = OpenAIPromptCacheAutomatic::Requested;
    if (options.contains("mode") && !options.at("mode").is_null()) {
        if (!options.at("mode").is_string()) {
            bad_request("prompt_cache_options.mode must be a string", "prompt_cache_options");
        }
        const std::string mode = options.at("mode").get<std::string>();
        if (mode == "explicit") {
            policy.automatic = OpenAIPromptCacheAutomatic::Disabled;
        } else if (mode != "implicit") {
            bad_request("prompt_cache_options.mode must be 'implicit' or 'explicit'",
                        "prompt_cache_options");
        }
    }
    if (options.contains("ttl") && !options.at("ttl").is_null() &&
        (!options.at("ttl").is_string() || options.at("ttl").get<std::string>() != "30m")) {
        bad_request("prompt_cache_options.ttl must be '30m'", "prompt_cache_options");
    }
    return policy;
}

void apply_openai_prompt_cache_policy(GenerationRequest& request, OpenAIPromptCachePolicy policy) {
    std::vector<std::optional<CacheBoundary>*> explicit_boundaries;
    for (ToolDefinition& tool : request.tools) {
        if (tool.cache_boundary_after) {
            explicit_boundaries.push_back(&tool.cache_boundary_after);
        }
    }
    for (ChatTurn& turn : request.messages) {
        for (ContentPart& part : turn.content) {
            if (part.cache_boundary_after) {
                explicit_boundaries.push_back(&part.cache_boundary_after);
            }
        }
        if (turn.cache_boundary_after) {
            explicit_boundaries.push_back(&turn.cache_boundary_after);
        }
    }

    std::optional<CacheBoundary>* automatic_target = nullptr;
    for (auto turn = request.messages.rbegin(); turn != request.messages.rend(); ++turn) {
        if (!turn->tool_calls.empty()) {
            automatic_target = &turn->cache_boundary_after;
            break;
        }
        if (!turn->content.empty()) {
            automatic_target = &turn->content.back().cache_boundary_after;
            break;
        }
    }
    if (automatic_target == nullptr && !request.tools.empty()) {
        automatic_target = &request.tools.back().cache_boundary_after;
    }

    const bool automatic_enabled =
        policy.automatic != OpenAIPromptCacheAutomatic::Disabled && automatic_target != nullptr;
    const bool automatic_merges_explicit   = automatic_enabled && automatic_target->has_value();
    // The automatic candidate at the end of the leading system/developer run (agent clients send
    // no breakpoints, and sibling sessions share that head): a distinct marker when the run is
    // unmarked, so it takes one of the frontend's marker slots.
    std::size_t leading_instruction = static_cast<std::size_t>(-1);
    if (policy.auto_system_shared_prefix) {
        for (std::size_t index = 0; index < request.messages.size(); ++index) {
            const ChatRole role = request.messages[index].role;
            if (role != ChatRole::System && role != ChatRole::Developer) { break; }
            leading_instruction = index;
        }
    }
    const bool system_candidate =
        policy.auto_system_shared_prefix && automatic_enabled &&
        leading_instruction != static_cast<std::size_t>(-1) &&
        !request.messages[leading_instruction].cache_boundary_after.has_value();
    const std::size_t base_slots           = !automatic_enabled || automatic_merges_explicit
                                                 ? kMaximumExplicitPromptCacheMarkers
                                                 : kMaximumExplicitPromptCacheMarkers - 1U;
    const std::size_t explicit_write_slots = base_slots - (system_candidate ? 1U : 0U);
    const std::size_t first_selected       = explicit_boundaries.size() > explicit_write_slots
                                                 ? explicit_boundaries.size() - explicit_write_slots
                                                 : 0U;
    for (std::size_t index = 0; index < explicit_boundaries.size(); ++index) {
        if (index < first_selected) {
            explicit_boundaries[index]->reset();
        } else {
            explicit_boundaries[index]->value().kind =
                ninferno::PromptCacheMarkerKind::SharedStablePrefix;
            explicit_boundaries[index]->value().evidence =
                ninferno::SharedCandidateEvidence::ExplicitBoundary;
        }
    }

    if (automatic_enabled) {
        const ninferno::SharedCandidateEvidence evidence =
            policy.automatic == OpenAIPromptCacheAutomatic::Default
                ? ninferno::SharedCandidateEvidence::DefaultAutomatic
                : ninferno::SharedCandidateEvidence::RequestedAutomatic;
        if (*automatic_target) {
            automatic_target->value().evidence |= evidence;
        } else {
            *automatic_target = CacheBoundary{.evidence = evidence};
        }
        if (system_candidate) {
            request.messages[leading_instruction].cache_boundary_after =
                CacheBoundary{.evidence = ninferno::SharedCandidateEvidence::DefaultAutomatic};
        }
    }
    // OpenAI already defines the automatic/explicit write policy for every request. Existing
    // exact shared residents are still considered by the Engine independently of this switch.
    request.allow_engine_automatic_shared_prefixes = false;
}

std::string make_models_list(const std::string& model_id, std::int64_t created,
                             std::uint32_t max_model_len, const ninferno::ModelMetadata& metadata) {
    // vLLM/llama.cpp-compatible discovery metadata for the configured per-request context limit.
    // Codex also decodes a required top-level `models` array of its own backend's ModelInfo
    // (slug, shell_type, truncation_policy, ...). Nothing here can truthfully fill one, so it
    // is empty: the catalog refresh decodes and Codex keeps its fallback metadata.
    const Json payload = {
        {"object", "list"},
        {"data", Json::array({make_model_object_json(model_id, created, max_model_len, metadata)})},
        {"models", Json::array()}};
    return payload.dump();
}

std::string make_props(const std::string& model_id, std::uint32_t max_context,
                       std::uint32_t slots, const ninferno::ModelMetadata& metadata) {
    const Json payload{
        {"default_generation_settings", Json{{"n_ctx", max_context}, {"model", model_id}}},
        {"total_slots", slots},
        {"model_alias", model_id},
        {"n_ctx", max_context},
        {"n_ctx_train", metadata.native_context},
        {"n_vocab", metadata.vocab_size},
        {"build_info", "ninferno"}};
    return payload.dump();
}

std::string make_prometheus_metrics(const ninferno::RuntimeStats& stats,
                                    std::uint32_t http_active_requests) {
    std::string out;
    const auto metric = [&out](const char* name, const char* type, const char* help,
                               double value) {
        char number[64];
        std::snprintf(number, sizeof(number), "%.17g", value);
        out += std::string("# HELP ") + name + ' ' + help + "\n# TYPE " + name + ' ' + type + '\n' +
               name + ' ' + number + '\n';
    };
    const double busy = stats.decode_rounds == 0
                            ? 0.0
                            : static_cast<double>(stats.decode_row_rounds) /
                                  static_cast<double>(stats.decode_rounds);
    metric("llamacpp:prompt_tokens_total", "counter", "Number of prompt tokens processed.",
           static_cast<double>(stats.computed_prefill_tokens));
    metric("llamacpp:tokens_predicted_total", "counter", "Number of generation tokens processed.",
           static_cast<double>(stats.committed_decode_tokens));
    metric("llamacpp:n_decode_total", "counter", "Total number of decode rounds.",
           static_cast<double>(stats.decode_rounds));
    metric("llamacpp:n_busy_slots_per_decode", "gauge",
           "Average number of busy slots per decode round.", busy);
    metric("llamacpp:requests_processing", "gauge", "Number of requests processing.",
           static_cast<double>(stats.running_requests));
    metric("llamacpp:requests_deferred", "gauge", "Number of requests deferred.",
           static_cast<double>(stats.waiting_requests));
    metric("ninferno_reused_prompt_tokens_total", "counter",
           "Prompt tokens served from the context cache instead of prefill.",
           static_cast<double>(stats.reused_prompt_tokens));
    metric("ninferno_requests_prefilling", "gauge", "Requests in prefill.",
           static_cast<double>(stats.prefilling_requests));
    metric("ninferno_requests_materializing", "gauge",
           "Requests restoring cached context before they run (in neither llama.cpp gauge).",
           static_cast<double>(stats.materializing_requests));
    metric("ninferno_http_active_requests", "gauge", "HTTP requests being served.",
           static_cast<double>(http_active_requests));
    return out;
}

std::string make_model_object(const std::string& model_id, std::int64_t created,
                              std::uint32_t max_model_len,
                              const ninferno::ModelMetadata& metadata) {
    return make_model_object_json(model_id, created, max_model_len, metadata).dump();
}

std::string make_error_body(const ApiError& error) {
    Json rendered     = {{"message", error.message}, {"type", error.type}};
    rendered["param"] = error.param.empty() ? Json(nullptr) : Json(error.param);
    rendered["code"]  = error.code.empty() ? Json(nullptr) : Json(error.code);
    // A message can echo client bytes (an id from the path, a field value): never let a byte that
    // is not UTF-8 turn the error into a failure of its own.
    return Json{{"error", std::move(rendered)}}.dump(-1, ' ', false,
                                                     Json::error_handler_t::replace);
}

std::int64_t unix_time_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string new_openai_chat_completion_id() { return random_identifier("chatcmpl-"); }

std::string new_openai_chat_tool_call_id() { return random_identifier("call_"); }

std::string new_openai_request_id() { return responses_identifier("req"); }

std::string new_openai_response_id() { return responses_identifier("resp"); }

std::string new_openai_response_item_id(std::string_view prefix) {
    return responses_identifier(prefix);
}

std::string custom_tool_input_schema(const RequestJson& custom) {
    std::string description = "The complete custom tool input.";
    if (custom.contains("format") && custom.at("format").is_object()) {
        const RequestJson& format = custom.at("format");
        if (format.contains("type") && format.at("type").is_string()) {
            description += " Declared format: " + format.at("type").get<std::string>() + ".";
        }
        const RequestJson* grammar = &format;
        if (format.contains("grammar") && format.at("grammar").is_object()) {
            grammar = &format.at("grammar");
        }
        if (grammar->contains("syntax") && grammar->at("syntax").is_string()) {
            description += " Grammar syntax: " + grammar->at("syntax").get<std::string>() + ".";
        }
        if (grammar->contains("definition") && grammar->at("definition").is_string()) {
            description += " Grammar: " + grammar->at("definition").get<std::string>();
        } else if (format.contains("grammar") && format.at("grammar").is_string()) {
            description += " Grammar: " + format.at("grammar").get<std::string>();
        }
    }
    return RequestJson{
        {"type", "object"},
        {"properties",
         RequestJson{
             {"input", RequestJson{{"type", "string"}, {"description", std::move(description)}}}}},
        {"required", RequestJson::array({"input"})},
        {"additionalProperties", false}}
        .dump();
}

std::string custom_tool_call_input(std::string_view arguments_json) {
    const RequestJson arguments =
        RequestJson::parse(arguments_json.begin(), arguments_json.end(), nullptr, false);
    if (!arguments.is_discarded() && arguments.is_object() && arguments.contains("input") &&
        arguments.at("input").is_string()) {
        return arguments.at("input").get<std::string>();
    }
    return std::string(arguments_json);
}

} // namespace ninferno::serve
