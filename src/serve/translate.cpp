#include "serve/translate.h"
#include "runtime/contract/sampling.h"
#include "serve/request_json.h"
#include "serve/request_validation.h"
#include "media/pdf/pdf_text.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninferno::serve {
namespace {

[[noreturn]] void document_error(std::string message, std::string code) {
    ApiError error;
    error.message = std::move(message);
    error.param   = "messages";
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

bool is_pdf(const ninferno::OwnedMedia& file) {
    static constexpr char kMagic[] = {'%', 'P', 'D', 'F', '-'};
    // The header may follow up to 1024 bytes of junk (PDF 32000-1, 7.5.2 as readers apply it).
    const std::size_t scan = std::min<std::size_t>(file.bytes.size(), 1024 + sizeof kMagic);
    for (std::size_t i = 0; i + sizeof kMagic <= scan; ++i) {
        if (std::equal(std::begin(kMagic), std::end(kMagic), file.bytes.begin() + i)) return true;
    }
    return false;
}

// A PDF as the model reads it: a <document> with its title and context, then each page's text
// layer. With vision, a page without one (a scan, a figure) is its rendered image instead, between
// the text parts; without vision such a page is marked as unread.
void append_pdf_document(std::vector<ninferno::MessagePart>& parts, const ContentPart& part,
                         const ninferno::OwnedMedia& file, bool vision) {
    if (!is_pdf(file)) {
        document_error("the document is not a PDF (media type '" + file.media_type +
                           "'); send other files as text documents",
                       "documents_not_supported");
    }
    ninferno::media::pdf::Document doc;
    try {
        ninferno::media::pdf::Limits limits;
        limits.render_textless = vision;
        doc                    = ninferno::media::pdf::read(file.bytes, limits);
    } catch (const ninferno::media::pdf::Error& error) {
        using ninferno::media::pdf::ErrorKind;
        switch (error.kind()) {
        case ErrorKind::Unavailable: document_error(error.what(), "documents_not_supported");
        case ErrorKind::TooLarge: document_error(error.what(), "document_too_large");
        case ErrorKind::Encrypted:
        case ErrorKind::Malformed: document_error(error.what(), "invalid_document");
        }
        throw;
    }
    bool readable = false;
    for (const auto& page : doc.pages) {
        readable = readable || !page.image_png.empty() ||
                   page.text.find_first_not_of(" \t\n") != std::string::npos;
    }
    if (!readable) {
        document_error(vision ? "the PDF document has neither text nor pages that render"
                              : "the PDF document has no text layer (a scanned document?); a "
                                "server with vision reads its pages as images",
                       "documents_not_supported");
    }
    // The document's own text is escaped as XML text: a page holding "</page></document>" cannot
    // end the markup around it.
    const auto escaped = [](std::string_view raw) {
        std::string out;
        out.reserve(raw.size());
        for (const char c : raw) {
            if (c == '<') {
                out += "&lt;";
            } else if (c == '&') {
                out += "&amp;";
            } else {
                out += c;
            }
        }
        return out;
    };
    std::string text = "<document>\n";
    if (!part.document_title.empty()) {
        text += "<title>" + escaped(part.document_title) + "</title>\n";
    }
    if (!part.document_context.empty()) {
        text += "<context>" + escaped(part.document_context) + "</context>\n";
    }
    for (std::size_t p = 0; p < doc.pages.size(); ++p) {
        auto& page = doc.pages[p];
        while (!page.text.empty() && (page.text.back() == '\n' || page.text.back() == ' ')) {
            page.text.pop_back();
        }
        text += "<page number=\"" + std::to_string(p + 1) + "\">\n";
        if (!page.image_png.empty()) {
            ninferno::MessagePart words;
            words.text = std::move(text);
            parts.push_back(std::move(words));
            ninferno::MessagePart image;
            image.kind              = ninferno::MessagePartKind::Media;
            image.media.kind        = ninferno::MediaKind::Image;
            image.media.media_type  = "image/png";
            image.media.source_name = "pdf page " + std::to_string(p + 1);
            image.media.bytes       = std::move(page.image_png);
            parts.push_back(std::move(image));
            text = "\n</page>\n";
            continue;
        }
        text += page.text.empty() && vision ? "[this page did not render]" : escaped(page.text);
        if (page.text.empty() && !vision) text += "[no text on this page]";
        text += "\n</page>\n";
    }
    if (doc.textless_not_rendered != 0) {
        text += "[" + std::to_string(doc.textless_not_rendered) +
                " more pages without text were not rendered]\n";
    }
    text += "</document>";
    ninferno::MessagePart words;
    words.text = std::move(text);
    parts.push_back(std::move(words));
}

std::uint64_t random_seed() {
    return unpredictable_engine()();
}

[[noreturn]] void invalid_sampling(std::string message, std::string param) {
    ApiError error;
    error.message = std::move(message);
    error.param   = std::move(param);
    throw ApiException(std::move(error));
}

[[noreturn]] void invalid_prompt_option(std::string message, std::string param, std::string code) {
    ApiError error;
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

ninferno::SamplingOverrides resolve_sampling_overrides(const SamplingParams& request,
                                                     const ServeOptions& server) {
    ninferno::SamplingOverrides sampling = server.sampling_overrides;
    if (request.temperature) { sampling.temperature = static_cast<float>(*request.temperature); }
    if (request.top_p) { sampling.top_p = static_cast<float>(*request.top_p); }
    if (request.min_p) { sampling.min_p = static_cast<float>(*request.min_p); }
    if (request.top_k) { sampling.top_k = static_cast<std::int32_t>(*request.top_k); }
    if (request.presence_penalty) {
        sampling.presence_penalty = static_cast<float>(*request.presence_penalty);
    }
    if (request.frequency_penalty) {
        sampling.frequency_penalty = static_cast<float>(*request.frequency_penalty);
    }
    if (request.repetition_penalty) {
        sampling.repetition_penalty = static_cast<float>(*request.repetition_penalty);
    }
    sampling.logit_bias = request.logit_bias;
    sampling.logprobs   = request.logprobs;
    if (request.seed) {
        sampling.seed = *request.seed;
    } else if (server.sampling_overrides.seed) {
        sampling.seed = *server.sampling_overrides.seed;
    } else {
        sampling.seed = random_seed();
    }

    const auto finite = [](const std::optional<float>& value) {
        return !value || std::isfinite(*value);
    };
    if (!finite(sampling.temperature) || !finite(sampling.top_p) || !finite(sampling.min_p) ||
        !finite(sampling.presence_penalty) || !finite(sampling.frequency_penalty) ||
        !finite(sampling.repetition_penalty)) {
        invalid_sampling("sampling parameters must be finite", "sampling");
    }
    const auto check = [](const std::optional<float>& value, SamplingRange range,
                          const char* field) {
        if (value && !range.contains(*value)) {
            invalid_sampling(runtime::range_message(field, range), field);
        }
    };
    check(sampling.temperature, kTemperatureRange, "temperature");
    check(sampling.top_p, kProbabilityRange, "top_p");
    if (sampling.top_k && (*sampling.top_k < 0 || *sampling.top_k > kMaximumTopK)) {
        invalid_sampling(top_k_limit_message("top_k", "send"), "top_k");
    }
    check(sampling.min_p, kProbabilityRange, "min_p");
    check(sampling.presence_penalty, kPenaltyRange, "presence_penalty");
    check(sampling.frequency_penalty, kPenaltyRange, "frequency_penalty");
    if (sampling.repetition_penalty && !repetition_penalty_valid(*sampling.repetition_penalty)) {
        invalid_sampling("repetition_penalty must be in (0,2]", "repetition_penalty");
    }
    if (server.greedy) { sampling.temperature = 0.0F; }
    return sampling;
}

// --tool-order sorted: clients that send the same tools in the order their MCP servers finished
// registering, which can differ from one request (or client restart) to the next, would move the
// shared prefix back thousands of tokens on every swap, as the template renders the tool list
// before the system prompt. Sorted by name, every order of one set renders the same prompt, on any
// process and after a restart; nothing is remembered between requests. A request with an explicit
// cache boundary after a tool or a duplicate name keeps the client's order, as does the default
// (the order apply_chat_template renders).
void sort_tools(std::vector<const ToolDefinition*>& tools) {
    if (tools.size() < 2) { return; }
    if (std::any_of(tools.begin(), tools.end(),
                    [](const ToolDefinition* tool) { return tool->cache_boundary_after; })) {
        return;
    }
    std::vector<const ToolDefinition*> sorted = tools;
    std::stable_sort(
        sorted.begin(), sorted.end(),
        [](const ToolDefinition* a, const ToolDefinition* b) { return a->name < b->name; });
    const auto same_name = [](const ToolDefinition* a, const ToolDefinition* b) {
        return a->name == b->name;
    };
    if (std::adjacent_find(sorted.begin(), sorted.end(), same_name) != sorted.end()) { return; }
    tools = std::move(sorted);
}

// tool_choice none keeps the declarations (vLLM's default too): dropping them changed the prompt
// from the first byte after the system turn, and a compaction or summary turn of an agent lost the
// whole cached conversation. Calls are not read back then.
std::vector<const ToolDefinition*> effective_tools(const GenerationRequest& request, bool sorted) {
    std::vector<const ToolDefinition*> tools;
    tools.reserve(request.tools.size());
    for (const ToolDefinition& tool : request.tools) { tools.push_back(&tool); }
    if (sorted) { sort_tools(tools); }
    return tools;
}

// The tool object the chat template receives: the fields the client declared, in its order, and
// nothing the client did not send (the prompt equals what transformers'
// apply_chat_template renders for the request's own tools). Without a declaration (a tool built
// by the server) the canonical order is name, description, parameters.
std::string render_tool_definition(const ToolDefinition& tool) {
    using Json = RequestJson;
    static const std::vector<std::string> kCanonical = {"name", "description", "parameters",
                                                        "input_examples"};
    const bool declared = !tool.declared_fields.empty();
    Json function       = Json::object();
    for (const std::string& field : declared ? tool.declared_fields : kCanonical) {
        if (field == "name") {
            function["name"] = tool.name;
        } else if (field == "description") {
            if (declared || !tool.description.empty()) { function["description"] = tool.description; }
        } else if (field == "parameters") {
            function["parameters"] = Json::parse(tool.input_schema_json);
        } else if (field == "strict") {
            function["strict"] = tool.strict;
        } else if (field == "input_examples" && tool.input_examples_json) {
            function["input_examples"] = Json::parse(*tool.input_examples_json);
        }
    }
    if (!function.contains("name")) { function["name"] = tool.name; }
    return Json{{"type", "function"}, {"function", std::move(function)}}.dump();
}

} // namespace

ResolvedPromptSemantics resolve_prompt_semantics(const GenerationRequest& request,
                                                 const ServeOptions& server) {
    using Json  = RequestJson;
    Json kwargs = request.chat_template_kwargs_json.empty()
                      ? Json::object()
                      : Json::parse(request.chat_template_kwargs_json);
    if (!kwargs.is_object())
        invalid_prompt_option("chat_template_kwargs must be an object", "chat_template_kwargs",
                              "invalid_template_option");
    auto merge_boolean = [&](const char* key, std::optional<bool> typed) {
        if (!kwargs.contains(key) || kwargs[key].is_null()) {
            kwargs.erase(key);
            return typed;
        }
        if (!kwargs[key].is_boolean())
            invalid_prompt_option(std::string(key) + " must be a boolean", key,
                                  "invalid_template_option");
        const bool nested = kwargs[key].get<bool>();
        if (typed && *typed != nested)
            invalid_prompt_option(std::string("conflicting ") + key + " values", key,
                                  "conflicting_template_option");
        kwargs.erase(key);
        return std::optional<bool>(nested);
    };
    auto thinking = merge_boolean("enable_thinking", request.enable_thinking);
    auto preserve = merge_boolean("preserve_thinking", request.preserve_thinking);
    auto effort   = request.reasoning_effort;
    if (kwargs.contains("reasoning_effort") && !kwargs["reasoning_effort"].is_null()) {
        if (!kwargs["reasoning_effort"].is_string())
            invalid_prompt_option("reasoning_effort must be a string", "reasoning_effort",
                                  "invalid_template_option");
        const auto nested =
            ninferno::parse_reasoning_effort(kwargs["reasoning_effort"].get<std::string>());
        if (!nested)
            invalid_prompt_option("invalid reasoning_effort", "reasoning_effort",
                                  "invalid_template_option");
        if (effort && effort != nested)
            invalid_prompt_option("conflicting reasoning_effort values", "reasoning_effort",
                                  "conflicting_template_option");
        effort = nested;
    }
    kwargs.erase("reasoning_effort");
    ResolvedPromptSemantics result{
        .enable_thinking           = thinking ? thinking : server.enable_thinking,
        .preserve_thinking         = preserve ? preserve : server.preserve_thinking,
        .chat_template_kwargs_json = kwargs.dump(),
    };
    if (effort) {
        const bool enables = *effort != ninferno::ReasoningEffort::None;
        if (thinking && *thinking != enables)
            invalid_prompt_option("reasoning effort conflicts with enable_thinking",
                                  "reasoning_effort", "conflicting_template_option");
        // A request effort overrides the server's thinking default.
        result.enable_thinking = enables;
        result.reasoning_effort = *effort;
    }
    if (request.continuation == ninferno::PromptContinuationMode::ContinueFinalAssistant &&
        result.enable_thinking == true) {
        invalid_prompt_option("assistant prefill cannot be combined with enabled thinking",
                              "messages", "assistant_prefill_not_supported");
    }
    // Thinking off, preservation unset by the request and the server, no reasoning in the history:
    // preserved. A template that drops the think blocks of the assistant turns before the last user
    // message (Qwen3.6) then writes the same empty block into every assistant turn, as it already
    // does after that message, so a new user message no longer re-renders an agent's earlier tool
    // turns and their cached prefix stays (#463). With reasoning in the history (its own field, or a
    // closed think block in an assistant turn's text, which the template splits out), preserving
    // would put that text back into the prompt: left to the template.
    const auto reasoned = [](const ChatTurn& turn) {
        if (!turn.reasoning_content.empty()) { return true; }
        return turn.role == ChatRole::Assistant &&
               std::any_of(turn.content.begin(), turn.content.end(), [](const ContentPart& part) {
                   return part.kind == ContentKind::Text &&
                          part.text.find("</think>") != std::string::npos;
               });
    };
    result.preserve_for_reuse =
        !result.preserve_thinking && result.enable_thinking == false &&
        std::none_of(request.messages.begin(), request.messages.end(), reasoned);
    result.vision     = server.enable_vision;
    result.sort_tools = server.sort_tools;
    return result;
}

ninferno::PromptInput to_prompt_input(const GenerationRequest& request,
                                    const ResolvedPromptSemantics& semantics,
                                    const MediaAcquirer& acquire_media) {
    ninferno::PromptInput input;
    input.messages.reserve(request.messages.size());
    for (std::size_t turn_index = 0; turn_index < request.messages.size(); ++turn_index) {
        const ChatTurn& turn = request.messages[turn_index];
        ninferno::ChatMessage message;
        message.role              = turn.role;
        message.reasoning_content = turn.reasoning_content;
        message.tool_call_id      = turn.tool_call_id;
        message.tool_calls.reserve(turn.tool_calls.size());
        for (const ToolCall& call : turn.tool_calls) {
            message.tool_calls.push_back(
                ninferno::ToolCall{call.id, call.name, call.arguments_json});
        }

        if (turn.role == ChatRole::Tool && turn.tool_result_is_error) {
            ninferno::MessagePart error;
            error.text = "[tool_error]\n";
            message.parts.push_back(std::move(error));
        }

        std::uint64_t text_bytes = 0;
        for (std::size_t part_index = 0; part_index < turn.content.size(); ++part_index) {
            const ContentPart& part = turn.content[part_index];
            if (part.kind == ContentKind::Text) {
                ninferno::MessagePart text;
                text.text = part.text;
                message.parts.push_back(std::move(text));
                if (part.text.size() > std::numeric_limits<std::uint32_t>::max() - text_bytes) {
                    throw std::invalid_argument("cacheable instruction text exceeds uint32");
                }
                text_bytes += part.text.size();
            } else if (part.kind == ContentKind::Document) {
                if (!acquire_media) {
                    throw std::logic_error("media acquisition callback is not configured");
                }
                const std::size_t first = message.parts.size();
                append_pdf_document(message.parts, part, acquire_media(part), semantics.vision);
                for (std::size_t i = first; i < message.parts.size(); ++i) {
                    const std::size_t size = message.parts[i].text.size();
                    if (size > std::numeric_limits<std::uint32_t>::max() - text_bytes) {
                        throw std::invalid_argument("cacheable instruction text exceeds uint32");
                    }
                    text_bytes += size;
                }
            } else if (part.kind == ContentKind::Image || part.kind == ContentKind::Video) {
                if (!acquire_media) {
                    throw std::logic_error("media acquisition callback is not configured");
                }
                ninferno::MessagePart media;
                media.kind  = ninferno::MessagePartKind::Media;
                media.media = acquire_media(part);
                message.parts.push_back(std::move(media));
            } else {
                ApiError error;
                error.message = "content type '" + part.type_raw + "' is not supported";
                error.param   = "messages";
                error.code    = "modality_not_supported";
                throw ApiException(std::move(error));
            }
            if (part.cache_boundary_after) {
                if (turn_index >= std::numeric_limits<std::uint32_t>::max() ||
                    part_index >= std::numeric_limits<std::uint32_t>::max()) {
                    throw std::overflow_error("conversation cache boundary exceeds uint32");
                }
                const bool leading_instruction =
                    turn_index == 0 &&
                    (turn.role == ChatRole::System || turn.role == ChatRole::Developer) &&
                    part.kind == ContentKind::Text;
                if (leading_instruction) {
                    input.context_cache.markers.push_back(ninferno::PromptCacheMarker{
                        .kind     = part.cache_boundary_after->kind,
                        .evidence = part.cache_boundary_after->evidence,
                        .location = ninferno::PromptCacheMarkerLocation::LeadingInstructionBoundary,
                        .leading_instruction_bytes = static_cast<std::uint32_t>(text_bytes),
                    });
                } else {
                    input.context_cache.markers.push_back(ninferno::PromptCacheMarker{
                        .after_message_count = static_cast<std::uint32_t>(turn_index + 1U),
                        .kind                = part.cache_boundary_after->kind,
                        .evidence            = part.cache_boundary_after->evidence,
                        .location = ninferno::PromptCacheMarkerLocation::MessagePartBoundary,
                        .after_message_part_count =
                            static_cast<std::uint32_t>(message.parts.size()),
                    });
                }
            }
        }
        input.messages.push_back(std::move(message));
        if (turn.cache_boundary_after) {
            if (input.messages.size() > std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("conversation cache boundary exceeds uint32");
            }
            input.context_cache.markers.push_back(ninferno::PromptCacheMarker{
                .after_message_count = static_cast<std::uint32_t>(input.messages.size()),
                .kind                = turn.cache_boundary_after->kind,
                .evidence            = turn.cache_boundary_after->evidence,
                .location            = ninferno::PromptCacheMarkerLocation::MessageBoundary,
            });
        }
    }

    input.options.continuation                     = request.continuation;
    input.options.enable_thinking                  = semantics.enable_thinking;
    input.options.reasoning_effort                 = semantics.reasoning_effort;
    input.options.preserve_thinking =
        semantics.preserve_for_reuse ? std::optional<bool>(true) : semantics.preserve_thinking;
    input.options.chat_template_kwargs_json        = semantics.chat_template_kwargs_json;
    input.options.add_vision_id                    = false;
    const std::vector<const ToolDefinition*> tools = effective_tools(request, semantics.sort_tools);
    input.options.tool_calls_enabled               = request.uses_tools();
    // tool_choice none: a call written anyway is cut out of the answer, not shown as markup.
    input.options.tool_calls_discarded = !tools.empty() && !request.uses_tools();
    input.options.tool_jsons.reserve(tools.size());
    for (std::size_t index = 0; index < tools.size(); ++index) {
        input.options.tool_jsons.push_back(render_tool_definition(*tools[index]));
        if (tools[index]->cache_boundary_after) {
            input.context_cache.markers.push_back(ninferno::PromptCacheMarker{
                .kind             = tools[index]->cache_boundary_after->kind,
                .evidence         = tools[index]->cache_boundary_after->evidence,
                .location         = ninferno::PromptCacheMarkerLocation::ToolBoundary,
                .after_tool_count = static_cast<std::uint32_t>(index + 1U),
            });
        }
    }
    input.context_cache.allow_engine_automatic_shared_prefixes =
        request.allow_engine_automatic_shared_prefixes;
    return input;
}

// StopPolicy::repetition_window of a tool_choice none request, when the server sets none.
constexpr std::uint32_t kNoneRepetitionWindow = 48;

ninferno::RequestOptions to_request_options(const GenerationRequest& request,
                                          const ServeOptions& server,
                                          const ResolvedPromptSemantics& semantics,
                                          bool allow_prefix_reuse) {
    ninferno::RequestOptions options;
    options.execution.requested_output_tokens = static_cast<std::uint32_t>(request.max_tokens);
    options.execution.allow_prefix_reuse      = allow_prefix_reuse;
    if (semantics.enable_thinking != false) {
        options.execution.thinking.budget =
            request.thinking_budget ? request.thinking_budget : server.default_thinking_budget;
        options.execution.thinking.answer_room = server.answer_room;
    }
    options.execution.sampling             = resolve_sampling_overrides(request.sampling, server);
    options.output.raw                     = false;
    options.output.preserve_special_tokens = request.uses_tools() || request.has_tool_history();
    options.output.tool_name_max_length = static_cast<std::uint32_t>(request.tool_name_max_length);
    options.stop.include_model_defaults = !request.ignore_eos;
    options.stop.repetition_window      = server.repetition_stop;
    if (options.stop.repetition_window == 0 && !effective_tools(request).empty() &&
        !request.uses_tools()) {
        // tool_choice none: a model kept from its call can fall into a filler loop that nothing
        // else ends before max_tokens. An output that repeats a period of up to 12 tokens four
        // times in a row ends as a repetition stop does (the server's own setting wins).
        options.stop.repetition_window = kNoneRepetitionWindow;
    }
    options.stop.repetition_concludes_thinking = server.repetition_concludes_thinking;
    options.stop.strings.reserve(request.stop_strings.size());
    for (const std::string& stop : request.stop_strings) {
        if (!stop.empty()) {
            options.stop.strings.push_back(
                ninferno::StopString{.text              = stop,
                                     .channel           = ninferno::OutputChannel::Content,
                                     .include_in_output = false});
        }
    }
    return options;
}

} // namespace ninferno::serve
