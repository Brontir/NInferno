#include "models/qwen3_5/frontend/frontend.h"

#include "core/sha256.h"
#include "core/units.h"

#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"

#include "models/qwen3_5/frontend/chat_template.h"
#include "models/qwen3_5/frontend/media_cache.h"
#include "models/qwen3_5/frontend/processor.h"
#include "models/qwen3_5/frontend/test_access.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "chat/tool_calls.h"
#include "text/hf_token_codec.h"
#include "text/unicode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninferno::models::qwen3_5 {
namespace {

using Json   = nlohmann::json;
using Clock  = std::chrono::steady_clock;
namespace fi = frontend;

constexpr double kRescaleFactor = 1.0 / 255.0;
constexpr double kVideoFps      = 2.0;
constexpr int kVideoMinFrames   = 4;
constexpr int kVideoMaxFrames   = 768;

Json parse_resource_json(std::string_view bytes, std::string_view name) {
    try {
        Json result = Json::parse(bytes);
        if (!result.is_object()) {
            throw std::invalid_argument(std::string(name) + " must contain a JSON object");
        }
        return result;
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument("malformed " + std::string(name) + ": " + error.what());
    }
}

std::int64_t require_integer(const Json& object, std::string_view field,
                             std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_number_integer()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be an integer");
    }
    return object.at(key).get<std::int64_t>();
}

double require_number(const Json& object, std::string_view field, std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_number()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be a number");
    }
    return object.at(key).get<double>();
}

double number_or_default(const Json& object, std::string_view field, std::string_view resource,
                         double default_value) {
    const std::string key(field);
    return object.contains(key) ? require_number(object, field, resource) : default_value;
}

std::int64_t integer_or_default(const Json& object, std::string_view field,
                                std::string_view resource, std::int64_t default_value) {
    const std::string key(field);
    return object.contains(key) ? require_integer(object, field, resource) : default_value;
}

const Json& require_object(const Json& object, std::string_view field, std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_object()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be an object");
    }
    return object.at(key);
}

std::uint64_t positive_u64(std::int64_t value, std::string_view field) {
    if (value <= 0) { throw std::invalid_argument(std::string(field) + " must be positive"); }
    return static_cast<std::uint64_t>(value);
}

void validate_pixel_pipeline(const Json& config, std::string_view resource) {
    for (const char* flag : {"do_resize", "do_rescale", "do_normalize", "do_convert_rgb"}) {
        if (config.contains(flag) && (!config[flag].is_boolean() || !config[flag].get<bool>())) {
            throw std::invalid_argument(std::string(resource) + "." + flag +
                                        " must be enabled for the compiled pixel pipeline");
        }
    }
    if (config.contains("resample") &&
        (!config["resample"].is_number_integer() || config["resample"].get<int>() != 3)) {
        throw std::invalid_argument(std::string(resource) +
                                    ".resample must select bicubic interpolation");
    }
    if (require_integer(config, "patch_size", resource) != 16 ||
        require_integer(config, "temporal_patch_size", resource) != 2 ||
        require_integer(config, "merge_size", resource) != 2) {
        throw std::invalid_argument(std::string(resource) +
                                    " does not match the compiled Vision patch geometry");
    }
    const auto require_half_triplet = [&](std::string_view field) {
        const std::string key(field);
        if (!config.contains(key) || !config.at(key).is_array() || config.at(key).size() != 3) {
            throw std::invalid_argument(std::string(resource) + "." + key +
                                        " must contain three values");
        }
        for (const Json& value : config.at(key)) {
            if (!value.is_number() || value.get<double>() != 0.5) {
                throw std::invalid_argument(std::string(resource) + "." + key +
                                            " does not match the compiled normalization");
            }
        }
    };
    require_half_triplet("image_mean");
    require_half_triplet("image_std");
    if (number_or_default(config, "rescale_factor", resource, kRescaleFactor) != kRescaleFactor) {
        throw std::invalid_argument(std::string(resource) +
                                    ".rescale_factor does not match the compiled normalization");
    }
}

fi::ProcessorOptions processor_options(const FrontendResources& resources) {
    const Json image =
        parse_resource_json(resources.preprocessor_config_json, "preprocessor_config.json");
    const Json video = parse_resource_json(resources.video_preprocessor_config_json,
                                           "video_preprocessor_config.json");
    validate_pixel_pipeline(image, "preprocessor_config.json");
    validate_pixel_pipeline(video, "video_preprocessor_config.json");

    const Json& image_size = require_object(image, "size", "preprocessor_config.json");
    const Json& video_size = require_object(video, "size", "video_preprocessor_config.json");

    fi::ProcessorOptions options;
    options.image_min_pixels =
        positive_u64(require_integer(image_size, "shortest_edge", "preprocessor_config.json.size"),
                     "image shortest_edge");
    options.image_max_pixels =
        positive_u64(require_integer(image_size, "longest_edge", "preprocessor_config.json.size"),
                     "image longest_edge");
    options.video_min_pixels = positive_u64(
        require_integer(video_size, "shortest_edge", "video_preprocessor_config.json.size"),
        "video shortest_edge");
    options.video_max_pixels = positive_u64(
        require_integer(video_size, "longest_edge", "video_preprocessor_config.json.size"),
        "video longest_edge");
    options.video_fps =
        number_or_default(video, "fps", "video_preprocessor_config.json", kVideoFps);
    options.video_min_frames = static_cast<int>(
        integer_or_default(video, "min_frames", "video_preprocessor_config.json", kVideoMinFrames));
    options.video_max_frames = static_cast<int>(
        integer_or_default(video, "max_frames", "video_preprocessor_config.json", kVideoMaxFrames));
    if (options.video_fps != kVideoFps || options.video_min_frames != kVideoMinFrames ||
        options.video_max_frames != kVideoMaxFrames) {
        throw std::invalid_argument(
            "video_preprocessor_config.json does not match registered sampling defaults");
    }

    return options;
}

void validate_tokenizer_config(const FrontendResources& resources, TokenizerConfigPolicy policy) {
    const Json tokenizer_config =
        parse_resource_json(resources.tokenizer_config_json, "tokenizer_config.json");
    if (policy == TokenizerConfigPolicy::ChatTemplateOwnsSpecialTokens) { return; }
    // A transformers-5 tokenizer_config.json may omit add_bos_token; a tokenizer without a BOS
    // token has none to add (transformers 5 then reports add_bos_token false), so an absent flag
    // means "adds BOS" only when a BOS token is named.
    // It may omit add_prefix_space too: the tokenizer.json ByteLevel step, which the tokenizer
    // requires without a prefix space, then decides.
    const bool has_bos = tokenizer_config.contains("bos_token") &&
                         !tokenizer_config.at("bos_token").is_null();
    if (tokenizer_config.value("add_bos_token", has_bos) ||
        tokenizer_config.value("add_prefix_space", false)) {
        throw std::invalid_argument(
            "tokenizer_config.json does not match Qwen3.5 tokenizer prefix semantics");
    }
    // The pad token is the one the model publishes (Qwen's own checkpoints name <|endoftext|>,
    // NVIDIA's NVFP4 Qwen3.8 <|im_end|>): generation never emits or reads padding, so any pad is
    // accepted that is a token of this vocabulary. One the tokenizer cannot resolve would make the
    // published resources inconsistent, and is refused.
    const auto pad = tokenizer_config.find("pad_token");
    if (pad == tokenizer_config.end() || pad->is_null()) { return; }
    std::string name;
    if (pad->is_string()) {
        name = pad->get<std::string>();
    } else if (pad->is_object() && pad->contains("content") && pad->at("content").is_string()) {
        name = pad->at("content").get<std::string>();
    } else {
        throw std::invalid_argument(
            "tokenizer_config.json pad_token must be a token string or an object with content");
    }
    if (resources.tokenizer && !resources.tokenizer->token_id(name)) {
        throw std::invalid_argument("tokenizer_config.json pad_token " + name +
                                    " is not a token of tokenizer.json");
    }
}

// `source_name` receives what the decision fingerprint calls the template: "artifact", or the
// override file named by the bytes this call read and compiled.
fi::CompiledChatTemplate compile_chat_template(const FrontendResources& resources,
                                               const std::filesystem::path& override_path,
                                               TokenizerConfigPolicy policy,
                                               std::string& source_name) {
    validate_tokenizer_config(resources, policy);
    nlohmann::ordered_json tokens = nlohmann::ordered_json::object();
    const auto config             = nlohmann::ordered_json::parse(resources.tokenizer_config_json);
    for (const char* key : {"bos_token", "eos_token", "pad_token", "unk_token", "sep_token",
                            "cls_token", "mask_token"}) {
        if (!config.contains(key) || config[key].is_null()) continue;
        const auto& value = config[key];
        if (value.is_string())
            tokens[key] = value;
        else if (value.is_object() && value.contains("content"))
            tokens[key] = value.at("content");
    }
    // transformers 5 renamed the list extra_special_tokens (its dict form names model-specific
    // tokens instead); templates still read additional_special_tokens.
    if (config.contains("additional_special_tokens"))
        tokens["additional_special_tokens"] = config["additional_special_tokens"];
    else if (config.contains("extra_special_tokens") && config["extra_special_tokens"].is_array())
        tokens["additional_special_tokens"] = config["extra_special_tokens"];
    std::string source(resources.chat_template_jinja);
    std::string name = "artifact:chat_template.jinja";
    if (!override_path.empty()) {
        name = override_path.string();
        std::ifstream file(override_path, std::ios::binary | std::ios::ate);
        if (!file) throw std::invalid_argument("cannot read chat template: " + name);
        const auto size = file.tellg();
        if (size <= 0 || size > std::streamoff(16 * kMiB)) {
            throw std::invalid_argument(
                "chat template must be a nonempty UTF-8 source file up to 16 MiB: " + name);
        }
        source.resize(static_cast<std::size_t>(size));
        file.seekg(0);
        if (!file.read(source.data(), static_cast<std::streamsize>(source.size()))) {
            throw std::invalid_argument("cannot read chat template: " + name);
        }
    }
    source_name =
        override_path.empty() ? std::string("artifact") : named_bytes_sha256(override_path, source);
    return fi::CompiledChatTemplate::resolve(source, std::move(name), std::move(tokens));
}

[[noreturn]] void throw_processor_error(const fi::ProcessorError& error) {
    switch (error.kind()) {
    case fi::ProcessorErrorKind::BudgetExceeded:
        throw RequestError(RequestErrorKind::MediaBudgetExceeded, error.what());
    case fi::ProcessorErrorKind::ContextLengthExceeded:
        throw RequestError(RequestErrorKind::ContextLengthExceeded, error.what());
    case fi::ProcessorErrorKind::InvalidMedia:
        throw RequestError(RequestErrorKind::InvalidMedia, error.what());
    }
    throw std::logic_error("unknown Qwen3.5 processor error kind");
}

[[noreturn]] void throw_context_length_exceeded(std::uint32_t max_context) {
    throw RequestError(RequestErrorKind::ContextLengthExceeded,
                       "prepared prompt exceeds Engine max_context " + std::to_string(max_context));
}

std::vector<fi::ChatMessage> convert_messages(std::vector<ChatMessage> messages) {
    std::vector<fi::ChatMessage> result;
    result.reserve(messages.size());
    for (ChatMessage& source : messages) {
        fi::ChatMessage target;
        target.role              = source.role;
        target.reasoning_content = std::move(source.reasoning_content);
        target.tool_call_id      = std::move(source.tool_call_id);
        target.tool_calls.reserve(source.tool_calls.size());
        for (ToolCall& call : source.tool_calls) {
            target.tool_calls.push_back(
                fi::ToolCall{.id             = std::move(call.id),
                             .name           = std::move(call.name),
                             .arguments_json = std::move(call.arguments_json)});
        }
        target.parts.reserve(source.parts.size());
        for (MessagePart& part : source.parts) {
            switch (part.kind) {
            case MessagePartKind::Text:
                target.parts.push_back(fi::ChatPart::text_part(std::move(part.text)));
                break;
            case MessagePartKind::Media: {
                if (part.media.bytes.empty()) {
                    throw std::invalid_argument("frontend media input contains no owning bytes");
                }
                fi::MediaData media;
                media.source_name         = std::move(part.media.source_name);
                media.media_type          = std::move(part.media.media_type);
                media.bytes               = std::move(part.media.bytes);
                media.image_resize_policy = part.media.image_resize_policy;
                switch (part.media.kind) {
                case MediaKind::Image:
                    target.parts.push_back(fi::ChatPart::image(std::move(media)));
                    break;
                case MediaKind::Video:
                    target.parts.push_back(fi::ChatPart::video(std::move(media)));
                    break;
                default:
                    throw std::invalid_argument("frontend media kind is invalid");
                }
                break;
            }
            default:
                throw std::invalid_argument("frontend message-part kind is invalid");
            }
        }
        result.push_back(std::move(target));
    }
    return result;
}

fi::ChatRenderOptions render_options(const PromptOptions& options,
                                     std::span<const PromptCacheMarker> cache_markers = {}) {
    fi::ChatRenderOptions rendered{.continuation              = options.continuation,
                                   .enable_thinking           = options.enable_thinking,
                                   .reasoning_effort          = options.reasoning_effort,
                                   .preserve_thinking         = options.preserve_thinking,
                                   .chat_template_kwargs_json = options.chat_template_kwargs_json,
                                   .add_vision_id             = options.add_vision_id,
                                   .tool_jsons                = options.tool_jsons};
    rendered.cache_markers.assign(cache_markers.begin(), cache_markers.end());
    return rendered;
}

std::uint32_t checked_token_count(std::size_t count) {
    if (count == 0) {
        throw std::invalid_argument("prepared prompt must contain at least one token");
    }
    if (count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("prepared prompt token count exceeds the target domain");
    }
    return static_cast<std::uint32_t>(count);
}

void assign_text_positions(PreparedPromptData& prompt) {
    const std::size_t count = prompt.token_ids.size();
    prompt.token_types.assign(count, 0);
    prompt.positions.resize(count * 3);
    for (std::size_t axis = 0; axis < 3; ++axis) {
        for (std::size_t index = 0; index < count; ++index) {
            if (index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
                throw std::invalid_argument("prepared prompt position exceeds int32 range");
            }
            prompt.positions[axis * count + index] = static_cast<std::int32_t>(index);
        }
    }
    prompt.rope_delta = 0;
}

VisionItem convert_vision_item(fi::VisionItem item) {
    VisionItem result;
    result.modality =
        item.modality == fi::Modality::Image ? PromptModality::Image : PromptModality::Video;
    result.grid = VisionGrid{.temporal = item.grid.t, .height = item.grid.h, .width = item.grid.w};
    result.patch_begin    = item.patch_begin;
    result.patch_count    = item.patch_count;
    result.patch_features = fi::vision_patch_features(item.style);
    result.content_digest = item.content_digest;
    result.timestamps     = std::move(item.timestamps);
    result.token_spans.reserve(item.token_spans.size());
    for (const fi::TokenSpan span : item.token_spans) {
        result.token_spans.push_back(TokenSpan{.begin = span.begin, .count = span.count});
    }
    return result;
}

StopPolicy merge_stop_policy(const text::TokenCodec& tokenizer,
                             std::span<const TokenId> model_stops, const StopPolicy& caller) {
    StopPolicy result;
    result.publish_stop_token = caller.publish_stop_token;
    result.repetition_window  = caller.repetition_window;
    result.repetition_concludes_thinking = caller.repetition_concludes_thinking;
    const auto append_token   = [&](TokenId token) {
        if (!tokenizer.is_valid_token(token)) {
            throw std::invalid_argument("stop token id is outside the checkpoint vocabulary: " +
                                          std::to_string(token));
        }
        if (std::find(result.token_ids.begin(), result.token_ids.end(), token) ==
            result.token_ids.end()) {
            result.token_ids.push_back(token);
        }
    };
    if (caller.include_model_defaults) {
        for (const TokenId token : model_stops) { append_token(token); }
    }
    for (const TokenId token : caller.token_ids) { append_token(token); }

    result.strings.reserve(caller.strings.size());
    for (const StopString& stop : caller.strings) {
        if (stop.text.empty()) { throw std::invalid_argument("stop string must not be empty"); }
        (void)ninferno::text::unicode_internal::utf8_codepoints(stop.text, "stop string");
        const auto duplicate = std::find_if(
            result.strings.begin(), result.strings.end(), [&](const StopString& existing) {
                return existing.text == stop.text && existing.channel == stop.channel &&
                       existing.include_in_output == stop.include_in_output;
            });
        if (duplicate == result.strings.end()) { result.strings.push_back(stop); }
    }
    return result;
}

std::optional<std::uint32_t> leading_instruction_boundary(std::span<const ChatRole> roles) {
    std::uint32_t leading = 0;
    while (leading < roles.size() &&
           (roles[leading] == ChatRole::System || roles[leading] == ChatRole::Developer)) {
        ++leading;
    }
    return leading != 0 ? std::optional<std::uint32_t>(leading) : std::nullopt;
}

bool exact_vision_frontier(std::uint32_t frontier, std::span<const VisionItem> items) {
    for (const VisionItem& item : items) {
        if (item.token_spans.empty()) { return false; }
        const TokenSpan& first = item.token_spans.front();
        const TokenSpan& last  = item.token_spans.back();
        if (last.count > std::numeric_limits<std::size_t>::max() - last.begin) { return false; }
        const std::size_t end = last.begin + last.count;
        if (first.begin < frontier && frontier < end) { return false; }
    }
    return true;
}

PreparedContextCache prepare_context_cache(
    ContextCacheHints hints, std::size_t message_count, std::span<const std::size_t> message_parts,
    std::span<const std::optional<std::uint32_t>> message_boundaries,
    std::span<const PromptCacheMarker> rendered_markers,
    std::span<const std::optional<std::uint32_t>> cache_boundaries,
    std::span<const VisionItem> vision_items, std::optional<std::size_t> engine_tool_marker_index,
    std::optional<std::uint32_t> leading_boundary, std::uint32_t full_prompt_frontier) {
    if (hints.markers.size() > kMaximumExplicitPromptCacheMarkers) {
        throw std::invalid_argument("PromptInput supports at most four explicit cache markers");
    }
    if (cache_boundaries.size() != rendered_markers.size()) {
        throw std::logic_error("rendered cache marker count changed during preparation");
    }

    PreparedContextCache out;
    if (hints.session_key) {
        if (hints.session_key->empty() || hints.session_key->size() > kPreparedSessionKeyCapacity) {
            throw std::invalid_argument("context cache session_key must contain 1 to 256 bytes");
        }
        PreparedSessionKey key;
        key.size = static_cast<std::uint16_t>(hints.session_key->size());
        std::copy(hints.session_key->begin(), hints.session_key->end(), key.bytes.begin());
        out.session_key = key;
    }
    switch (hints.retention) {
    case CacheRetentionHint::Default:
        out.retention = out.session_key ? runtime::RetentionClass::LiveSession
                                        : runtime::RetentionClass::RecentPrivate;
        break;
    case CacheRetentionHint::LiveSession:
        if (!out.session_key) {
            throw std::invalid_argument("LiveSession retention requires a session_key");
        }
        out.retention = runtime::RetentionClass::LiveSession;
        break;
    case CacheRetentionHint::Disposable:
        out.retention = runtime::RetentionClass::Disposable;
        break;
    default:
        throw std::invalid_argument("context cache retention hint is invalid");
    }
    out.update_session_index = hints.update_session_index;

    for (const PromptCacheMarker marker : hints.markers) {
        switch (marker.kind) {
        case PromptCacheMarkerKind::SharedStablePrefix:
        case PromptCacheMarkerKind::PrivateLongAnchor:
            break;
        default:
            throw std::invalid_argument("context cache marker kind is invalid");
        }
        switch (marker.location) {
        case PromptCacheMarkerLocation::MessageBoundary:
            if (marker.after_message_count > message_count ||
                marker.leading_instruction_bytes != 0 || marker.after_tool_count != 0 ||
                marker.after_message_part_count != 0) {
                throw std::invalid_argument("context cache message marker is invalid");
            }
            break;
        case PromptCacheMarkerLocation::MessagePartBoundary:
            if (marker.after_message_count == 0 || marker.after_message_count > message_count ||
                marker.after_message_part_count == 0 || marker.leading_instruction_bytes != 0 ||
                marker.after_tool_count != 0) {
                throw std::invalid_argument("context cache message-part marker is invalid");
            }
            break;
        case PromptCacheMarkerLocation::LeadingInstructionBoundary:
            if (marker.after_message_count != 0 || marker.leading_instruction_bytes == 0 ||
                marker.after_tool_count != 0 || marker.after_message_part_count != 0) {
                throw std::invalid_argument("context cache leading-instruction marker is invalid");
            }
            break;
        case PromptCacheMarkerLocation::ToolBoundary:
            if (marker.after_message_count != 0 || marker.leading_instruction_bytes != 0 ||
                marker.after_tool_count == 0 || marker.after_message_part_count != 0) {
                throw std::invalid_argument("context cache tool marker is invalid");
            }
            break;
        default:
            throw std::invalid_argument("context cache marker location is invalid");
        }
        if (marker.kind == PromptCacheMarkerKind::SharedStablePrefix &&
            marker.evidence == SharedCandidateEvidence::None) {
            throw std::invalid_argument("shared context cache marker evidence is empty");
        }
    }

    out.opportunities.reserve(kMaximumPreparedPromptCacheCandidatesPerRequest);
    const auto add_opportunity = [&](PromptCacheMarkerKind kind, SharedCandidateEvidence evidence,
                                     std::uint32_t frontier, std::uint32_t input_order) {
        if (frontier == 0 || !exact_vision_frontier(frontier, vision_items)) { return; }
        const auto duplicate = std::find_if(
            out.opportunities.begin(), out.opportunities.end(), [&](const auto& existing) {
                return existing.kind == kind && existing.frontier == frontier;
            });
        if (duplicate == out.opportunities.end()) {
            out.opportunities.push_back(PreparedCacheOpportunity{.kind        = kind,
                                                                 .evidence    = evidence,
                                                                 .frontier    = frontier,
                                                                 .input_order = input_order});
        } else if (kind == PromptCacheMarkerKind::SharedStablePrefix) {
            duplicate->evidence |= evidence;
        }
    };

    for (std::size_t index = 0; index < hints.markers.size(); ++index) {
        const PromptCacheMarker marker = hints.markers[index];
        std::optional<std::uint32_t> resolved;
        // A boundary after a message's last part (OpenAI's automatic one) is taken at the closed
        // turn's end: the bytes between are the template's, the same in every rendering that
        // continues the conversation, and the turn's end is an execution frontier where a next
        // turn opens (prompt_layout.cpp), so the capture needs no prefill split of its own. At
        // the content's end the program moves it down to the 64-token grid, and the 17 to 80
        // tokens up to the message's end cost a whole-model call of their own (DECISIONS 63).
        const bool last_part =
            marker.location == PromptCacheMarkerLocation::MessagePartBoundary &&
            marker.after_message_count != 0 && marker.after_message_count <= message_parts.size() &&
            marker.after_message_part_count == message_parts[marker.after_message_count - 1U];
        if (marker.location == PromptCacheMarkerLocation::MessageBoundary || last_part) {
            if (marker.after_message_count < message_boundaries.size()) {
                resolved = message_boundaries[marker.after_message_count];
            }
        }
        if (!resolved && marker.location != PromptCacheMarkerLocation::MessageBoundary &&
            index < cache_boundaries.size()) {
            resolved = cache_boundaries[index];
        }
        if (!resolved) { continue; }
        add_opportunity(marker.kind, marker.evidence, *resolved, static_cast<std::uint32_t>(index));
    }

    std::uint32_t engine_order = static_cast<std::uint32_t>(hints.markers.size());
    if (hints.allow_engine_automatic_shared_prefixes) {
        if (engine_tool_marker_index && *engine_tool_marker_index < cache_boundaries.size() &&
            cache_boundaries[*engine_tool_marker_index]) {
            add_opportunity(PromptCacheMarkerKind::SharedStablePrefix,
                            SharedCandidateEvidence::EngineStructural,
                            *cache_boundaries[*engine_tool_marker_index], engine_order++);
        }
        if (leading_boundary && *leading_boundary < message_boundaries.size() &&
            message_boundaries[*leading_boundary]) {
            add_opportunity(PromptCacheMarkerKind::SharedStablePrefix,
                            SharedCandidateEvidence::EngineStructural,
                            *message_boundaries[*leading_boundary], engine_order++);
        }
        add_opportunity(PromptCacheMarkerKind::SharedStablePrefix,
                        SharedCandidateEvidence::EngineObserved, full_prompt_frontier,
                        engine_order);
    }
    return out;
}

// Generation must stop at the token that closes an assistant turn in this model's own
// chat template, even when generation_config.json lists only other end tokens (fine-tunes often
// keep just <|endoftext|>). The terminator is read from the template: render one closed assistant
// turn and take the special token that immediately follows its content. A template that cannot
// render the probe, or closes turns with ordinary text, contributes nothing.
std::optional<TokenId> assistant_turn_terminator(const fi::CompiledChatTemplate& chat_template,
                                                 const text::TokenCodec& tokenizer) {
    constexpr std::string_view kProbe = "NINFERNO_TURN_PROBE_7f3a";
    try {
        fi::ChatMessage user;
        user.role = ChatRole::User;
        user.parts.push_back(fi::ChatPart::text_part("x"));
        fi::ChatMessage assistant;
        assistant.role = ChatRole::Assistant;
        assistant.parts.push_back(fi::ChatPart::text_part(std::string(kProbe)));
        fi::ChatRenderOptions options;
        options.add_generation_prompt = false;
        options.enable_thinking       = false;
        const fi::RenderedChat rendered = chat_template.render({user, assistant}, options);
        const std::size_t probe         = rendered.text.rfind(kProbe);
        if (probe == std::string::npos) { return std::nullopt; }
        const std::vector<int> after =
            tokenizer.encode(std::string_view(rendered.text).substr(probe + kProbe.size()));
        if (after.empty() || !tokenizer.decoded_token(after.front()).special) {
            return std::nullopt;
        }
        return static_cast<TokenId>(after.front());
    } catch (const std::exception&) { return std::nullopt; }
}

std::vector<std::size_t> prefix_failure_table(std::string_view pattern) {
    std::vector<std::size_t> failure(pattern.size());
    for (std::size_t index = 1; index < pattern.size(); ++index) {
        std::size_t matched = failure[index - 1U];
        while (matched != 0 && pattern[index] != pattern[matched]) { matched = failure[matched - 1U]; }
        if (pattern[index] == pattern[matched]) { ++matched; }
        failure[index] = matched;
    }
    return failure;
}

std::shared_ptr<const OutputConventions> output_conventions(const chat::ChatConventions& chat,
                                                            const text::TokenCodec& tokenizer) {
    auto result = std::make_shared<OutputConventions>();
    if (chat.reasoning) {
        result->reasoning_open          = chat.reasoning->open;
        result->reasoning_open_suffix   = chat.reasoning->open_suffix;
        result->reasoning_close         = chat.reasoning->close;
        result->canonical_close         = chat.reasoning->canonical_close;
        result->canonical_close_failure = prefix_failure_table(result->canonical_close);
        result->content_open            = chat.reasoning->content_open;
    }
    // Every special token a marker is spelled with is structural, including one that begins a
    // longer marker: Gemma 4 opens reasoning with the special token <|channel> followed by the
    // plain text "thought", and a hidden <|channel> would leave the open marker unmatched.
    for (const std::string& marker : chat.structural_markers()) {
        for (const int id : tokenizer.encode(marker)) {
            if (tokenizer.is_special_token(id)) { result->structural_tokens.push_back(id); }
        }
    }
    std::sort(result->structural_tokens.begin(), result->structural_tokens.end());
    result->structural_tokens.erase(
        std::unique(result->structural_tokens.begin(), result->structural_tokens.end()),
        result->structural_tokens.end());
    return result;
}

// A completed assistant turn with no text, no media and no calls (only reasoning, or nothing) is
// left out of the prompt: with such turns in its history the model
// imitates them and ends later turns without the call it was asked for (opencode: 0-6 of 20
// calls made, 20 of 20 without them). The final message stays: an assistant continuation or a
// client's own placeholder is the request's to make. Message-indexed cache markers move with the
// messages they follow.
void drop_empty_assistant_turns(std::vector<ChatMessage>& messages,
                                std::vector<PromptCacheMarker>& markers) {
    const auto empty = [](const ChatMessage& message) {
        if (message.role != ChatRole::Assistant || !message.tool_calls.empty()) { return false; }
        return std::all_of(message.parts.begin(), message.parts.end(), [](const MessagePart& part) {
            return part.kind == MessagePartKind::Text &&
                   part.text.find_first_not_of(" \t\r\n") == std::string::npos;
        });
    };
    std::vector<std::uint32_t> dropped_before(messages.size() + 1, 0);
    std::vector<ChatMessage> kept;
    kept.reserve(messages.size());
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const bool drop       = i + 1 < messages.size() && empty(messages[i]);
        dropped_before[i + 1] = dropped_before[i] + (drop ? 1U : 0U);
        if (!drop) { kept.push_back(std::move(messages[i])); }
    }
    if (dropped_before.back() == 0) {
        messages = std::move(kept);
        return;
    }
    for (PromptCacheMarker& marker : markers) {
        if ((marker.location == PromptCacheMarkerLocation::MessageBoundary ||
             marker.location == PromptCacheMarkerLocation::MessagePartBoundary) &&
            marker.after_message_count < dropped_before.size()) {
            marker.after_message_count -= dropped_before[marker.after_message_count];
        }
    }
    messages = std::move(kept);
}

} // namespace

ModelSamplingDefaults default_sampling(Architecture architecture) {
    ModelSamplingDefaults sampling;
    sampling.thinking     = {.temperature      = 1.0F,
                             .top_k            = 20,
                             .top_p            = 0.95F,
                             .min_p            = 0.0F,
                             .presence_penalty = architecture == Architecture::Qwen3_5Moe ? 1.5F : 0.0F,
                             .frequency_penalty = 0.0F};
    sampling.non_thinking = {.temperature       = 0.7F,
                             .top_k             = 20,
                             .top_p             = 0.80F,
                             .min_p             = 0.0F,
                             .presence_penalty  = 1.5F,
                             .frequency_penalty = 0.0F};
    sampling.source = "Qwen3.5 model card presets";
    return sampling;
}

std::optional<ModelSamplingDefaults> generation_sampling(std::string_view generation_config_json) {
    const auto config = nlohmann::json::parse(generation_config_json, nullptr, false);
    if (!config.is_object()) { return std::nullopt; }
    SamplingPreset preset{.temperature = 1.0F, .top_k = 0, .top_p = 1.0F};
    bool named      = false;
    const auto read = [&](const char* key, auto& field) {
        if (const auto it = config.find(key); it != config.end() && it->is_number()) {
            field = it->get<std::remove_reference_t<decltype(field)>>();
            named = true;
        }
    };
    read("temperature", preset.temperature);
    read("top_k", preset.top_k);
    read("top_p", preset.top_p);
    read("min_p", preset.min_p);
    if (!named) { return std::nullopt; }
    ModelSamplingDefaults out;
    out.source = "generation_config.json";
    // The sampler draws from an exact candidate domain of kMaximumTopK (ops::sample), and a request
    // may name top_k only in [0, kMaximumTopK]. A recommended top_k above it keeps the whole
    // domain, the same as top_k 0; the rest of the recommendation (temperature, top_p, min_p) is
    // kept as is. Gemma 4's 64 fits it.
    if (preset.top_k > kMaximumTopK) {
        out.notes.push_back("top_k " + std::to_string(preset.top_k) +
                            " is wider than the sampler's exact candidate domain: " +
                            std::to_string(kMaximumTopK));
        preset.top_k = kMaximumTopK;
    }
    if (const auto it = config.find("repetition_penalty");
        it != config.end() && it->is_number() && it->get<double>() != 1.0) {
        std::ostringstream value;
        value << it->get<double>();
        out.notes.push_back("repetition_penalty " + value.str() +
                            " is not applied: NInferno has no repetition-penalty sampler");
    }
    if (const auto it = config.find("do_sample"); it != config.end() && it->is_boolean() &&
                                                   !it->get<bool>()) {
        out.notes.push_back("do_sample is false (transformers would decode greedily); the named "
                            "sampling is applied, as vLLM does: ask for temperature 0 for greedy");
    }
    out.thinking     = preset;
    out.non_thinking = preset;
    return out;
}

ModelSamplingDefaults checkpoint_sampling(std::string_view generation_config_json) {
    if (auto recommended = generation_sampling(generation_config_json)) { return *recommended; }
    const SamplingPreset transformers{.temperature = 1.0F, .top_k = 50, .top_p = 1.0F};
    return ModelSamplingDefaults{.thinking     = transformers,
                                 .non_thinking = transformers,
                                 .source       = "transformers defaults"};
}

// The one token a document starts with, as transformers' tokenizer encodes a text with its special
// tokens: tokenizer_config.json add_bos_token when it says; absent, the class's own default
// (LlamaTokenizer, CodeLlamaTokenizer and GemmaTokenizer, fast or not, add their bos_token:
// Gemma 4 names GemmaTokenizer and no add_bos_token), else what the tokenizer.json
// TemplateProcessing puts in front (none for gpt-oss: a ByteLevel post_processor and the
// generic PreTrainedTokenizerFast). nullopt when that is nothing, or not one token of this
// vocabulary.
namespace {

std::optional<TokenId> adds_document_start(std::string_view tokenizer_config_json,
                                           const text::TokenCodec& tokenizer) {
    const Json config = parse_resource_json(tokenizer_config_json, "tokenizer_config.json");
    const auto bos_id = [&]() -> std::optional<TokenId> {
        const auto bos = config.find("bos_token");
        if (bos == config.end() || bos->is_null()) { return std::nullopt; }
        const std::string content = bos->is_string() ? bos->get<std::string>()
                                    : bos->is_object() && bos->contains("content")
                                        ? bos->at("content").get<std::string>()
                                        : std::string();
        if (content.empty()) { return std::nullopt; }
        const std::optional<int> id = tokenizer.token_id(content);
        if (!id) { return std::nullopt; }
        return static_cast<TokenId>(*id);
    };
    if (const auto add = config.find("add_bos_token"); add != config.end() && add->is_boolean()) {
        return add->get<bool>() ? bos_id() : std::nullopt;
    }
    static constexpr std::string_view kAddsBos[] = {"LlamaTokenizer", "LlamaTokenizerFast",
                                                    "CodeLlamaTokenizer", "CodeLlamaTokenizerFast",
                                                    "GemmaTokenizer", "GemmaTokenizerFast"};
    const std::string cls = config.value("tokenizer_class", std::string());
    if (std::find(std::begin(kAddsBos), std::end(kAddsBos), cls) != std::end(kAddsBos)) {
        return bos_id();
    }
    const std::vector<int> prefix = tokenizer.template_prefix();
    if (prefix.size() == 1) { return static_cast<TokenId>(prefix.front()); }
    return std::nullopt;
}

} // namespace

class Frontend::Impl {
public:
    Impl(const FrontendResources& resources, FrontendOptions options)
        : chat_template(compile_chat_template(resources, options.chat_template_path,
                                              options.tokenizer_config, chat_template_source)),
          tokenizer(resources.tokenizer),
          processor(!options.vision_enabled                       ? fi::ProcessorOptions{}
                    : options.vision_style == VisionStyle::Gemma4 ? fi::ProcessorOptions{}
                                                                  : processor_options(resources)),
          vision_enabled(options.vision_enabled), vision_component(options.vision_component),
          max_context(options.max_context) {
        if (options.max_context == 0) {
            throw std::invalid_argument("frontend max_context must be nonzero");
        }
        processor.style = options.vision_style;
        const std::uint64_t merge = fi::vision_merge(processor.style);
        const std::uint64_t vision_tokens =
            std::min<std::uint64_t>(options.max_context, kMaximumPromptVisionTokens);
        processor.max_vision_tokens = vision_tokens;
        processor.max_raw_patches   = vision_tokens * merge * merge;
        if (vision_enabled) {
            const std::uint64_t minimum_live = processor.max_raw_patches *
                                               fi::vision_patch_features(processor.style) *
                                               sizeof(std::uint16_t);
            if (minimum_live > options.media_live_bytes) {
                throw std::invalid_argument(
                    "media live-byte capacity cannot hold the maximum supported Vision prompt");
            }
            media_cache = std::make_shared<fi::MediaPreprocessCache>(
                options.media_cache_bytes, options.media_live_bytes,
                options.media_preprocess_threads, static_cast<std::size_t>(minimum_live));
        }
        if (!tokenizer || resources.public_token_count != tokenizer->vocab_size()) {
            throw std::invalid_argument(
                "Frontend requires the parsed model tokenizer and public token domain");
        }
        sampling = options.sampling ? *options.sampling : default_sampling(options.architecture);
        document_start = adds_document_start(resources.tokenizer_config_json, *tokenizer);
        for (const int token : tokenizer->default_stop_token_ids()) {
            if (!tokenizer->is_valid_token(token)) {
                throw std::invalid_argument(
                    "generation_config.json contains a stop token outside the vocabulary");
            }
            defaults.token_ids.push_back(token);
        }
        const chat::ChatConventions& chat = chat_template.conventions();
        const auto add_stop               = [&](int id) {
            if (std::find(defaults.token_ids.begin(), defaults.token_ids.end(), id) ==
                defaults.token_ids.end()) {
                defaults.token_ids.push_back(id);
            }
        };
        // The stops of the template's chat conventions (src/chat/data/conventions.json: Gemma 4
        // <turn|>, <eos> and <|tool_response>, the tool's turn; harmony <|return|> and <|call|>),
        // each one whole token. A configuration rebuilt from a GGUF or copied from another
        // publisher may leave one out (the model then writes the tool's answer after its call, or
        // <eos> into its answer); a tokenizer rebuilt from a GGUF keeps some of them user-defined
        // tokens, not special ones.
        for (const std::string& stop : chat.generation_stops()) {
            if (const std::optional<int> id = tokenizer->token_id(stop);
                id && tokenizer->decoded_token(*id).bytes == stop) {
                add_stop(*id);
            }
        }
        // The template's own assistant-turn terminator and the tokenizer's end-of-sequence token
        // (tokenizer_config.json eos_token, a special token) end the output too, unless the
        // conventions place them inside a reply.
        if (const std::optional<TokenId> terminator =
                assistant_turn_terminator(chat_template, *tokenizer)) {
            add_stop(*terminator);
        }
        if (const auto config =
                nlohmann::json::parse(resources.tokenizer_config_json, nullptr, false);
            config.is_object() && config.contains("eos_token")) {
            const nlohmann::json& value = config["eos_token"];
            const nlohmann::json* name =
                value.is_object() && value.contains("content") ? &value["content"] : &value;
            const auto& reasoning = chat_template.conventions().reasoning;
            // A configuration naming a reasoning marker its eos would end the output there.
            if (name->is_string() &&
                !(reasoning && (name->get<std::string>() == reasoning->close ||
                                name->get<std::string>() == reasoning->open))) {
                if (const std::optional<int> eos = tokenizer->token_id(name->get<std::string>());
                    eos && tokenizer->decoded_token(*eos).special) {
                    add_stop(*eos);
                }
            }
        }
        // No stop may end a reply where the template continues it: a reasoning marker, or a close
        // that ends one message of a reply (harmony's <|end|> after the analysis channel, which
        // unsloth's gpt-oss-20b-GGUF template writes as its turn terminator and a GGUF's rebuilt
        // configuration may name its eos: every content was empty).
        std::erase_if(defaults.token_ids, [&](int id) {
            return chat.within_reply(tokenizer->decoded_token(id).bytes);
        });
        output = output_conventions(chat, *tokenizer);
        if (chat.reasoning && !chat.reasoning->budget_control.empty()) {
            // The budget control closes reasoning in the model's own markup: it must round-trip
            // exactly, present the close marker and not end generation.
            const std::string_view control = chat.reasoning->budget_control;
            std::vector<TokenId> encoded   = tokenizer->encode(control);
            if (encoded.empty()) {
                throw std::invalid_argument(
                    "tokenizer cannot encode the chat template's thinking control suffix");
            }
            const std::string exact =
                tokenizer->decode(encoded, fi::DecodeOptions{.skip_special_tokens = false});
            if (exact != control || exact.find(chat.reasoning->close) == std::string::npos) {
                throw std::invalid_argument(
                    "tokenizer cannot present the chat template's thinking control suffix");
            }
            for (const TokenId token : encoded) {
                if (std::find(defaults.token_ids.begin(), defaults.token_ids.end(), token) !=
                    defaults.token_ids.end()) {
                    throw std::invalid_argument(
                        "canonical thinking control suffix contains a default terminal token");
                }
            }
            thinking_control_tokens =
                std::make_shared<const std::vector<TokenId>>(std::move(encoded));
        }
    }

    // Declared first: compiling chat_template names its source here (compile_chat_template).
    std::string chat_template_source;
    fi::CompiledChatTemplate chat_template;
    std::shared_ptr<const text::TokenCodec> tokenizer;
    std::optional<TokenId> document_start; // document_start_token()
    std::shared_ptr<const OutputConventions> output;
    fi::ProcessorOptions processor;
    std::shared_ptr<fi::MediaPreprocessCache> media_cache;
    StopPolicy defaults;
    ModelSamplingDefaults sampling;
    std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens;
    bool vision_enabled       = true;
    bool vision_component     = true;
    std::uint32_t max_context = 0;

    [[noreturn]] void refuse_media() const {
        throw std::invalid_argument(vision_component
                                        ? "Vision is disabled for this Engine"
                                        : "no vision component: this artifact is text-only");
    }
};

std::span<const std::int32_t> PreparedPromptData::position_axis(int axis) const {
    if (axis < 0 || axis >= 3 || positions.size() != token_ids.size() * 3) {
        throw std::out_of_range("invalid prepared-prompt position axis");
    }
    return std::span<const std::int32_t>(positions).subspan(
        static_cast<std::size_t>(axis) * token_ids.size(), token_ids.size());
}

PreparedPrompt::PreparedPrompt() noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept
    : data_(std::move(data)) {}

PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PromptSummary PreparedPrompt::summary() const {
    if (data_ == nullptr) { throw std::logic_error("prepared prompt is empty"); }
    return PromptSummary{.starts_in_reasoning = data_->starts_in_reasoning,
                         // The constraint gate's view: a request that turned reasoning off
                         // keeps its grammar from the first token (gemma-4-E2B-it under a
                         // json_schema opened a thought channel and reasoned to max_tokens).
                         .reasoning_open_allowed =
                             data_->reasoning_open_allowed && !data_->reasoning_declined,
                         .prompt_tokens = checked_token_count(data_->token_ids.size()),
                         .has_media     = data_->has_media(),
                         .thinking_budget_control = data_->thinking_budget_control};
}

PromptPreparationStats PreparedPrompt::preparation_stats() const noexcept {
    if (data_ == nullptr) { return {}; }
    const PrepareStats& stats = data_->prepare;
    return PromptPreparationStats{
        .seconds                       = stats.seconds,
        .media_preprocess_seconds      = stats.media_preprocess_seconds,
        .media_preprocess_work_seconds = stats.media_preprocess_work_seconds,
        .tokenize_seconds              = stats.tokenize_seconds,
        .media_items                   = stats.media_items,
        .media_bytes                   = stats.media_bytes,
        .raw_patches                   = stats.raw_patches,
        .vision_tokens                 = stats.vision_tokens,
        .patch_bytes                   = stats.patch_bytes,
        .media_cache_hits              = stats.media_cache_hits,
        .media_cache_misses            = stats.media_cache_misses,
        .media_singleflight_waits      = stats.media_singleflight_waits,
        .built_patch_bytes             = stats.built_patch_bytes,
        .reused_patch_bytes            = stats.reused_patch_bytes,
    };
}

PreparedPrompt::operator bool() const noexcept { return data_ != nullptr; }

Frontend::Frontend(std::shared_ptr<const Impl> impl) noexcept : impl_(std::move(impl)) {}

Frontend::Frontend(const Frontend&)                = default;
Frontend& Frontend::operator=(const Frontend&)     = default;
Frontend::Frontend(Frontend&&) noexcept            = default;
Frontend& Frontend::operator=(Frontend&&) noexcept = default;
Frontend::~Frontend()                              = default;

const ModelSamplingDefaults& Frontend::sampling_defaults() const noexcept {
    return impl_->sampling;
}

const std::string& Frontend::chat_template_source() const noexcept {
    return impl_->chat_template_source;
}

Frontend make_frontend(const FrontendResources& resources, FrontendOptions options) {
    return Frontend(std::make_shared<const Frontend::Impl>(resources, options));
}

void attach_hf_text_tokenizer(FrontendResources& resources) {
    auto tokenizer = std::make_shared<const text::HfTokenCodec>(
        resources.tokenizer_json, resources.tokenizer_config_json,
        resources.generation_config_json);
    resources.public_token_count = static_cast<std::uint32_t>(tokenizer->vocab_size());
    resources.tokenizer          = std::move(tokenizer);
}

void apply_hf_text_options(FrontendOptions& options) {
    options.tokenizer_config = TokenizerConfigPolicy::ChatTemplateOwnsSpecialTokens;
    options.vision_enabled   = false;
}

const PreparedPromptData& PreparedPromptAccess::view(const PreparedPrompt& prompt) {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    return *prompt.data_;
}

PreparedPromptData PreparedPromptAccess::take(PreparedPrompt&& prompt) {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    auto data = std::move(prompt.data_);
    return std::move(*data);
}

const PreparedPromptData& FrontendTestAccess::inspect(const PreparedPrompt& prompt) {
    return PreparedPromptAccess::view(prompt);
}

PreparedPrompt Frontend::prepare(PromptInput input, const PreparationControl& control) const {
    fi::check_preparation_control(control);
    const auto start              = Clock::now();
    const PromptOptions options   = input.options;
    ContextCacheHints cache_hints = std::move(input.context_cache);
    if (cache_hints.markers.size() > kMaximumExplicitPromptCacheMarkers) {
        throw std::invalid_argument("PromptInput supports at most four explicit cache markers");
    }
    drop_empty_assistant_turns(input.messages, cache_hints.markers);
    std::vector<ChatRole> message_roles;
    message_roles.reserve(input.messages.size());
    for (const ChatMessage& message : input.messages) { message_roles.push_back(message.role); }
    if (options.tool_calls_enabled && !options.tool_jsons.empty() &&
        impl_->chat_template.conventions().tools.format == chat::ToolCallFormat::None) {
        // The template would drop the declarations, and nothing could read a call back.
        throw std::invalid_argument(
            "this model's chat template declares no tool-call syntax, so it cannot take tools; "
            "serve it with a chat template that does (--chat-template)");
    }
    const auto tool_call_output = chat::build_tool_call_output_contract(
        options.tool_jsons, options.tool_calls_enabled && !options.tool_jsons.empty(),
        impl_->chat_template.conventions().tools.format);
    const std::optional<std::uint32_t> leading_boundary =
        leading_instruction_boundary(message_roles);
    std::vector<PromptCacheMarker> rendered_markers = cache_hints.markers;
    std::optional<std::size_t> engine_tool_marker_index;
    if (cache_hints.allow_engine_automatic_shared_prefixes && !options.tool_jsons.empty()) {
        engine_tool_marker_index = rendered_markers.size();
        rendered_markers.push_back(PromptCacheMarker{
            .kind             = PromptCacheMarkerKind::SharedStablePrefix,
            .evidence         = SharedCandidateEvidence::EngineStructural,
            .location         = PromptCacheMarkerLocation::ToolBoundary,
            .after_tool_count = static_cast<std::uint32_t>(options.tool_jsons.size()),
        });
    }
    const std::size_t message_count = input.messages.size();
    std::vector<std::size_t> message_parts;
    message_parts.reserve(message_count);
    for (const ChatMessage& message : input.messages) {
        // A message whose turn carries tool calls after its parts keeps its part boundaries.
        message_parts.push_back(message.tool_calls.empty() ? message.parts.size() : 0U);
    }
    std::vector<fi::ChatMessage> messages = convert_messages(std::move(input.messages));
    const bool has_media =
        std::any_of(messages.begin(), messages.end(),
                    [](const fi::ChatMessage& message) { return message.has_media(); });
    if (has_media && !impl_->vision_enabled) { impl_->refuse_media(); }

    auto prepared              = std::make_unique<PreparedPromptData>();
    PreparedPromptData& result = *prepared;
    result.tool_call_output    = tool_call_output;
    std::vector<std::optional<std::uint32_t>> message_boundaries;
    std::vector<std::optional<std::uint32_t>> cache_boundaries;
    if (has_media) {
        fi::Processor processor(*impl_->tokenizer, impl_->chat_template, impl_->processor,
                                impl_->media_cache);
        fi::ProcessedInput processed;
        try {
            processed =
                processor.process(std::move(messages), render_options(options, rendered_markers),
                                  control, impl_->max_context);
        } catch (const fi::ProcessorError& error) { throw_processor_error(error); }
        result.token_ids.assign(processed.input_ids.begin(), processed.input_ids.end());
        result.starts_in_reasoning    = processed.starts_in_reasoning;
        result.reasoning_open_allowed = processed.reasoning_open_allowed;
        result.reasoning_declined     = processed.reasoning_declined;
        result.thinking_budget_control =
            impl_->thinking_control_tokens && !impl_->thinking_control_tokens->empty();
        result.token_types         = std::move(processed.token_types);
        result.positions           = std::move(processed.positions);
        result.rope_delta          = processed.rope_delta;
        result.media_payloads      = std::move(processed.media_payloads);
        result.vision_items.reserve(processed.vision_items.size());
        for (fi::VisionItem& item : processed.vision_items) {
            result.vision_items.push_back(convert_vision_item(std::move(item)));
        }
        result.prepare.media_items              = processed.stats.media_items;
        result.prepare.media_bytes              = processed.stats.media_bytes;
        result.prepare.raw_patches              = processed.stats.raw_patches;
        result.prepare.vision_tokens            = processed.stats.vision_tokens;
        result.prepare.attention_pairs          = processed.stats.attention_pairs;
        result.prepare.patch_bytes              = processed.stats.patch_bytes;
        result.prepare.media_cache_hits         = processed.stats.media_cache_hits;
        result.prepare.media_cache_misses       = processed.stats.media_cache_misses;
        result.prepare.media_singleflight_waits = processed.stats.media_singleflight_waits;
        result.prepare.built_patch_bytes        = processed.stats.built_patch_bytes;
        result.prepare.reused_patch_bytes       = processed.stats.reused_patch_bytes;
        result.prepare.media_preprocess_seconds = processed.stats.media_preprocess_seconds;
        result.prepare.media_preprocess_work_seconds =
            processed.stats.media_preprocess_work_seconds;
        result.prepare.tokenize_seconds    = processed.stats.tokenize_seconds;
        result.identity.rewrite_checkpoint = processed.rewrite_checkpoint;
        result.identity.rewrite_execution_frontiers =
            std::move(processed.rewrite_execution_frontiers);
        message_boundaries = std::move(processed.message_boundaries);
        cache_boundaries   = std::move(processed.cache_boundaries);
    } else {
        const fi::RenderedChat rendered = impl_->chat_template.render(
            messages, render_options(options, rendered_markers), control);
        result.starts_in_reasoning    = rendered.starts_in_reasoning;
        result.reasoning_open_allowed = rendered.reasoning_open_allowed;
        result.reasoning_declined     = rendered.reasoning_declined;
        result.thinking_budget_control =
            impl_->thinking_control_tokens && !impl_->thinking_control_tokens->empty();
        const auto tokenize_started = Clock::now();
        fi::EncodedChat encoded     = fi::encode_rendered_chat(
            *impl_->tokenizer, rendered, static_cast<std::size_t>(impl_->max_context) + 1U);
        result.prepare.tokenize_seconds =
            std::chrono::duration<double>(Clock::now() - tokenize_started).count();
        fi::check_preparation_control(control, "tokenization");
        if (encoded.input_ids.size() > impl_->max_context) {
            throw_context_length_exceeded(impl_->max_context);
        }
        result.token_ids                   = std::move(encoded.input_ids);
        result.identity.rewrite_checkpoint = encoded.rewrite_checkpoint;
        result.identity.rewrite_execution_frontiers =
            std::move(encoded.rewrite_execution_frontiers);
        message_boundaries = std::move(encoded.message_boundaries);
        cache_boundaries   = std::move(encoded.cache_boundaries);
        assign_text_positions(result);
    }
    (void)checked_token_count(result.token_ids.size());
    result.identity.reusable = true;
    result.context_cache     = prepare_context_cache(
        std::move(cache_hints), message_count, message_parts, message_boundaries, rendered_markers,
        cache_boundaries, result.vision_items, engine_tool_marker_index, leading_boundary,
        checked_token_count(result.token_ids.size()));
    result.prepare.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return PreparedPrompt(std::move(prepared));
}

std::uint32_t Frontend::count_tokens(PromptInput input, const PreparationControl& control) const {
    fi::check_preparation_control(control);
    const PromptOptions options           = input.options;
    std::vector<fi::ChatMessage> messages = convert_messages(std::move(input.messages));
    const bool has_media =
        std::any_of(messages.begin(), messages.end(),
                    [](const fi::ChatMessage& message) { return message.has_media(); });
    if (has_media && !impl_->vision_enabled) { impl_->refuse_media(); }
    if (!has_media) {
        const fi::RenderedChat rendered =
            impl_->chat_template.render(messages, render_options(options), control);
        const std::uint32_t count = checked_token_count(
            fi::encode_rendered_chat(*impl_->tokenizer, rendered).input_ids.size());
        fi::check_preparation_control(control, "tokenization");
        return count;
    }

    fi::Processor processor(*impl_->tokenizer, impl_->chat_template, impl_->processor,
                            impl_->media_cache);
    try {
        return checked_token_count(
            processor.count_tokens(std::move(messages), render_options(options), control));
    } catch (const fi::ProcessorError& error) { throw_processor_error(error); }
}

MediaCacheSummary Frontend::media_cache_summary() const {
    if (impl_ == nullptr || !impl_->media_cache) { return {}; }
    const fi::MediaCacheStats stats = impl_->media_cache->stats();
    return MediaCacheSummary{
        .capacity_bytes      = stats.capacity_bytes,
        .live_capacity_bytes = stats.live_capacity_bytes,
        .retained_bytes      = stats.retained_bytes,
        .live_bytes          = stats.live_bytes,
        .entries             = stats.entries,
        .inflight            = stats.inflight,
        .queued_tasks        = stats.queued_tasks,
        .active_tasks        = stats.active_tasks,
        .preprocess_threads  = stats.preprocess_threads,
        .hits                = stats.hits,
        .misses              = stats.misses,
        .singleflight_waits  = stats.singleflight_waits,
        .evictions           = stats.evictions,
        .oversize_bypasses   = stats.oversize_bypasses,
    };
}

bool Frontend::resumable(const PreparedPrompt& prompt) {
    return !PreparedPromptAccess::view(prompt).has_media();
}

PreparedPrompt Frontend::copy(const PreparedPrompt& prompt) {
    return PreparedPrompt(
        std::make_unique<PreparedPromptData>(PreparedPromptAccess::view(prompt)));
}

PreparedPrompt Frontend::continued(const PreparedPrompt& prompt,
                                   std::span<const TokenId> generated) {
    const PreparedPromptData& source = PreparedPromptAccess::view(prompt);
    const std::size_t count          = source.token_ids.size();
    const std::size_t total          = count + generated.size();
    if (source.positions.size() != count * 3 || source.token_types.size() != count ||
        total > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("prompt cannot be continued by its generated tokens");
    }
    auto data = std::make_unique<PreparedPromptData>(source);
    data->token_ids.insert(data->token_ids.end(), generated.begin(), generated.end());
    data->token_types.resize(total, 0);
    // Generated text follows the prompt on every axis at its logical index plus the rope delta,
    // the positions decode gave it.
    data->positions.assign(total * 3, 0);
    for (std::size_t axis = 0; axis < 3; ++axis) {
        std::copy_n(source.positions.begin() + static_cast<std::ptrdiff_t>(axis * count), count,
                    data->positions.begin() + static_cast<std::ptrdiff_t>(axis * total));
        for (std::size_t index = count; index < total; ++index) {
            data->positions[axis * total + index] =
                static_cast<std::int32_t>(index) + source.rope_delta;
        }
    }
    return PreparedPrompt(std::move(data));
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> token_ids,
                                        bool allow_prefix_identity) const {
    const auto start = Clock::now();
    if (token_ids.size() > impl_->max_context) {
        throw_context_length_exceeded(impl_->max_context);
    }
    (void)checked_token_count(token_ids.size());
    for (const TokenId token : token_ids) {
        if (!impl_->tokenizer->is_valid_token(token)) {
            throw std::out_of_range("prompt token is outside the checkpoint vocabulary: " +
                                    std::to_string(token));
        }
    }
    auto prepared              = std::make_unique<PreparedPromptData>();
    PreparedPromptData& result = *prepared;
    result.token_ids           = std::move(token_ids);
    assign_text_positions(result);
    result.identity.reusable                  = allow_prefix_identity;
    result.context_cache.retention            = runtime::RetentionClass::RecentPrivate;
    result.context_cache.update_session_index = false;
    result.prepare.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return PreparedPrompt(std::move(prepared));
}

std::optional<TokenId> Frontend::document_start_token() const {
    if (impl_ == nullptr) { throw std::logic_error("frontend is empty"); }
    return impl_->document_start;
}

std::vector<TokenId> Frontend::tokenize_text(std::string_view text) const {
    if (impl_ == nullptr) { throw std::logic_error("frontend is empty"); }
    return impl_->tokenizer->encode(text);
}

OutputSession Frontend::make_output_session(const PreparedPrompt& prompt,
                                            const StopPolicy& caller_stop,
                                            const OutputOptions& output,
                                            const ThinkingControlOptions& thinking,
                                            bool quoted_reasoning_close_is_text) const {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    StopPolicy policy =
        merge_stop_policy(*impl_->tokenizer, impl_->defaults.token_ids, caller_stop);
    if (output.raw) { policy.publish_stop_token = true; }
    return OutputSession(impl_->tokenizer, std::move(policy), output,
                         OutputStart{.in_reasoning           = prompt.data_->starts_in_reasoning,
                                     .reasoning_open_allowed = prompt.data_->reasoning_open_allowed,
                                     .quoted_close_is_text   = quoted_reasoning_close_is_text},
                         thinking, impl_->thinking_control_tokens, prompt.data_->tool_call_output,
                         impl_->output);
}

const StopPolicy& Frontend::default_stop_policy() const noexcept { return impl_->defaults; }

structured::GrammarVocabulary Frontend::grammar_vocabulary(std::uint32_t token_domain) const {
    if (impl_ == nullptr) { throw std::logic_error("frontend is empty"); }
    const text::TokenCodec& tokenizer = *impl_->tokenizer;
    if (token_domain == 0 || token_domain > tokenizer.vocab_size()) {
        throw std::invalid_argument("grammar token domain exceeds the tokenizer vocabulary");
    }
    structured::GrammarVocabulary vocabulary;
    vocabulary.token_domain = token_domain;
    vocabulary.token_bytes.resize(token_domain);
    vocabulary.special.assign(token_domain, false);
    for (std::uint32_t token = 0; token < token_domain; ++token) {
        const int id = static_cast<int>(token);
        if (!tokenizer.is_valid_token(id)) {
            // Undefined ids are never produced by a constraint.
            vocabulary.special[token] = true;
            continue;
        }
        const fi::DecodedTokenView decoded = tokenizer.decoded_token(id);
        vocabulary.token_bytes[token]      = std::string(decoded.bytes);
        vocabulary.special[token] = decoded.special && !impl_->output->structural(token);
    }
    for (const TokenId token : impl_->defaults.token_ids) {
        if (token >= 0 && static_cast<std::uint32_t>(token) < token_domain) {
            vocabulary.stop_token_ids.push_back(token);
        }
    }
    // The output presentation splits reasoning from content at these bytes (OutputSession).
    vocabulary.reasoning_close = impl_->output->reasoning_close;
    vocabulary.reasoning_open  = impl_->output->reasoning_open;
    return vocabulary;
}

std::string Frontend::tool_call_structural_tag(std::span<const ToolCallConstraintTool> tools,
                                               ToolCallRequirement requirement,
                                               std::string_view named_tool,
                                               bool parallel_tool_calls) const {
    return chat::tool_call_syntax(impl_->chat_template.conventions().tools.format)
        .structural_tag(tools, requirement, named_tool, parallel_tool_calls);
}

} // namespace ninferno::models::qwen3_5
