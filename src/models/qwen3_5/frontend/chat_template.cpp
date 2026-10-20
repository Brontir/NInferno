#include "models/qwen3_5/frontend/chat_template.h"

#include "models/qwen3_5/frontend/media_cache.h"
#include "models/qwen3_5/frontend/prompt_layout.h"
#include "text/unicode.h"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace ninferno::models::qwen3_5::frontend {
namespace {

using Json = nlohmann::ordered_json;

// The role a template sees. A developer message (OpenAI's instruction role: Codex sends one, and
// the Responses API's `instructions` become one) renders as a system message: chat templates know
// system, user, assistant and tool, and the official Qwen templates raise on any other role
// (map, do not refuse).
std::string_view role_name(ChatRole role) {
    switch (role) {
    case ChatRole::System:
    case ChatRole::Developer:
        return "system";
    case ChatRole::User:
        return "user";
    case ChatRole::Assistant:
        return "assistant";
    case ChatRole::Tool:
        return "tool";
    }
    throw std::invalid_argument("invalid chat role");
}

bool instruction(ChatRole role) { return role == ChatRole::System || role == ChatRole::Developer; }

struct ContentSource {
    std::uint32_t tag = 0;
    std::vector<std::size_t> part_ends;
    std::vector<std::uint32_t> part_tags;
    std::vector<std::optional<std::size_t>> part_media;
    std::uint32_t reasoning_tag = 0;
};

void merge_option(Json& context, std::string_view key, const Json& value) {
    const std::string name(key);
    if (context.contains(name) && !context[name].is_null() && context[name] != value) {
        throw std::invalid_argument("conflicting chat template parameter: " + name);
    }
    context[name] = value;
}

Json template_parameters(const ChatRenderOptions& options, const Json& special_tokens) {
    Json context = options.chat_template_kwargs_json.empty()
                       ? Json::object()
                       : Json::parse(options.chat_template_kwargs_json);
    if (!context.is_object()) throw std::invalid_argument("chat_template_kwargs must be an object");
    for (const auto& item : context.items()) {
        const auto& key = item.key();
        if (key == "messages" || key == "tools" || key == "add_generation_prompt" ||
            key == "continue_final_message" || key == "bos_token" || key == "eos_token" ||
            key == "pad_token" || key == "unk_token" || key == "sep_token" || key == "cls_token" ||
            key == "mask_token" || key == "additional_special_tokens" ||
            special_tokens.contains(key)) {
            throw std::invalid_argument("chat_template_kwargs cannot override " + key);
        }
    }
    for (const char* name :
         {"enable_thinking", "preserve_thinking", "reasoning_effort", "add_vision_id"}) {
        if (context.contains(name) && context[name].is_null()) context.erase(name);
    }
    if (options.enable_thinking) merge_option(context, "enable_thinking", *options.enable_thinking);
    if (options.preserve_thinking)
        merge_option(context, "preserve_thinking", *options.preserve_thinking);
    if (options.reasoning_effort) {
        const auto name = reasoning_effort_name(*options.reasoning_effort);
        if (name.empty()) throw std::invalid_argument("invalid reasoning effort");
        merge_option(context, "reasoning_effort", name);
    }
    if (options.add_vision_id) merge_option(context, "add_vision_id", true);
    for (const char* name : {"enable_thinking", "preserve_thinking", "add_vision_id"}) {
        if (context.contains(name) && !context[name].is_boolean()) {
            throw std::invalid_argument(std::string(name) + " must be a boolean");
        }
    }
    if (context.contains("reasoning_effort")) {
        const auto& effort = context["reasoning_effort"];
        if (!effort.is_string() ||
            (effort != "none" && effort != "minimal" && effort != "low" && effort != "medium" &&
             effort != "high" && effort != "xhigh" && effort != "max")) {
            throw std::invalid_argument("invalid reasoning_effort template parameter");
        }
        const bool thinking = effort != "none";
        if (context.contains("enable_thinking") && context["enable_thinking"] != thinking) {
            throw std::invalid_argument("reasoning_effort conflicts with enable_thinking");
        }
        // The protocol's 'none' is an explicit disable. Other efforts retain template defaults.
        if (!thinking) context["enable_thinking"] = false;
    }
    context.update(special_tokens);
    return context;
}

std::optional<std::size_t> containing_message(const PromptLayout& layout, text::ByteSpan region) {
    for (std::size_t i = 0; i < layout.messages.size(); ++i) {
        const auto& message = layout.messages[i];
        if (region.begin >= message.content_begin && region.end <= message.content_end) return i;
    }
    return std::nullopt;
}

// Equal text alone does not prove equal control-token interpretation.
bool same_prefix(const text::TemplateOutput& lhs, const text::TemplateOutput& rhs,
                 std::size_t size) {
    if (lhs.text.size() < size || rhs.text.size() < size ||
        std::string_view(lhs.text).substr(0, size) != std::string_view(rhs.text).substr(0, size))
        return false;
    for (std::size_t i = 0;; ++i) {
        const bool left  = i < lhs.literal_spans.size() && lhs.literal_spans[i].begin < size;
        const bool right = i < rhs.literal_spans.size() && rhs.literal_spans[i].begin < size;
        if (!left || !right) return left == right;
        const auto a = lhs.literal_spans[i];
        const auto b = rhs.literal_spans[i];
        if (a.begin != b.begin || std::min(a.end, size) != std::min(b.end, size)) return false;
    }
}

// Text without its leading and trailing Unicode whitespace. Only the ends are decoded: a message
// may be many megabytes (decoding all of it took tens of seconds and gigabytes for 96 MiB).
std::string_view trim_unicode_whitespace(std::string_view text) {
    namespace uni     = text::unicode_internal;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const uni::CodepointSpan codepoint = uni::utf8_codepoint_at(text, begin, "message content");
        if (!uni::is_whitespace(codepoint.value)) break;
        begin += codepoint.length;
    }
    std::size_t end = text.size();
    while (end > begin) {
        std::size_t start = end - 1;
        while (start > begin && (static_cast<unsigned char>(text[start]) & 0xC0U) == 0x80U) --start;
        const uni::CodepointSpan codepoint = uni::utf8_codepoint_at(text, start, "message content");
        if (start + codepoint.length != end || !uni::is_whitespace(codepoint.value)) break;
        end = start;
    }
    return text.substr(begin, end - begin);
}

bool real_user(const ChatMessage& message, const chat::ToolMarkup& tools) {
    if (message.role != ChatRole::User) return false;
    if (message.has_media()) return true;
    if (tools.response_open.empty() || tools.response_close.empty()) return true;
    std::string content;
    for (const auto& part : message.parts) content += part.text;
    const std::string_view body = trim_unicode_whitespace(content);
    return !(body.starts_with(tools.response_open) && body.ends_with(tools.response_close));
}

} // namespace

bool ChatMessage::has_media() const noexcept {
    return std::any_of(parts.begin(), parts.end(),
                       [](const ChatPart& part) { return part.kind != ChatPartKind::Text; });
}

CompiledChatTemplate CompiledChatTemplate::resolve(std::string_view source, std::string source_name,
                                                   Json special_tokens) {
    CompiledChatTemplate result(text::JinjaTemplate(std::string(source), std::move(source_name)),
                                std::move(special_tokens), chat::resolve_conventions(source));
    // Whether enable_thinking switches reasoning, and which way an unset value goes: Gemma 4
    // defaults it to false and then opens no thought channel in the prompt, so only the request
    // tells a model that may open one itself not to (Qwen closes an empty <think> instead).
    try {
        std::vector<ChatMessage> probe(1);
        probe.front().parts.push_back(ChatPart::text_part("x"));
        ChatRenderOptions on, off;
        on.enable_thinking           = true;
        off.enable_thinking          = false;
        const std::string unset_text = result.render(probe, {}).text;
        const std::string off_text   = result.render(probe, off).text;
        result.thinking_switch_      = result.render(probe, on).text != off_text;
        result.thinking_default_     = unset_text != off_text;
    } catch (const std::exception&) {
        // A template that cannot render the probe has no switch the gate can rely on.
    }
    return result;
}

// A system or developer message after the conversation began (agent hosts such as Claude Code
// append one) is refused by the stock Qwen templates many artifacts embed ("System message must be
// at the beginning."). When the template refuses a conversation that has one, the
// conversation renders again with those messages as user messages: the prefix before them, and so
// the cache, is unchanged, and a template that accepts them keeps their role.
RenderedChat CompiledChatTemplate::render(const std::vector<ChatMessage>& messages,
                                          ChatRenderOptions options,
                                          const PreparationControl& control) const {
    // An instruction message's own rules hold before any retry may give it the user role.
    for (const auto& message : messages) {
        if (instruction(message.role) &&
            (message.has_media() || !message.reasoning_content.empty() ||
             !message.tool_calls.empty() || !message.tool_call_id.empty())) {
            throw std::invalid_argument("system and developer messages may contain only text");
        }
    }
    try {
        return render_once(messages, options, control);
    } catch (const std::invalid_argument&) {
        // Only the first message keeps an instruction role: the stock templates also refuse a
        // second one that opens the conversation (two system messages, a Responses `instructions`
        // with a developer message: Codex). The message count stays, so cache markers hold.
        const std::size_t leading = instruction(messages.front().role) ? 1 : 0;
        std::vector<ChatMessage> demoted;
        for (std::size_t i = leading; i < messages.size(); ++i) {
            if (!instruction(messages[i].role)) continue;
            if (demoted.empty()) demoted = messages;
            demoted[i].role = ChatRole::User;
        }
        // A tool the client declared without a description renders without one, exactly as
        // declared; a template that reads the description unconditionally ("// " +
        // tool.description in gpt-oss, `fn.description | tojson`) refuses it. Such a template sees
        // an empty description instead.
        std::optional<ChatRenderOptions> described;
        for (std::size_t i = 0; i < options.tool_jsons.size(); ++i) {
            Json tool = Json::parse(options.tool_jsons[i], nullptr, false);
            if (tool.is_discarded() || !tool.is_object()) continue;
            Json& function =
                tool.contains("function") && tool["function"].is_object() ? tool["function"] : tool;
            if (!function.contains("name") || function.contains("description")) continue;
            function["description"] = "";
            if (!described) described = options;
            described->tool_jsons[i] = tool.dump();
        }
        const std::vector<ChatMessage>& retried = demoted.empty() ? messages : demoted;
        if (!demoted.empty()) {
            try {
                return render_once(demoted, options, control);
            } catch (const std::invalid_argument&) {}
        }
        if (described) {
            try {
                return render_once(retried, *described, control);
            } catch (const std::invalid_argument&) {}
        }
        // A protocol reasoning effort the template does not take (Qwen 3.8 takes low, medium and
        // xhigh; OpenAI clients also send minimal and high) renders at the next stronger level the
        // template takes, never a weaker one. An effort given in chat_template_kwargs is the
        // client's own template input and stays as given.
        static_assert(static_cast<int>(ReasoningEffort::Minimal) == 1 &&
                          static_cast<int>(ReasoningEffort::Low) == 2 &&
                          static_cast<int>(ReasoningEffort::Medium) == 3 &&
                          static_cast<int>(ReasoningEffort::High) == 4 &&
                          static_cast<int>(ReasoningEffort::XHigh) == 5 &&
                          static_cast<int>(ReasoningEffort::Max) == 6,
                      "the effort levels are walked weakest to strongest");
        if (options.reasoning_effort && *options.reasoning_effort != ReasoningEffort::None &&
            *options.reasoning_effort != ReasoningEffort::Max) {
            ChatRenderOptions stronger = described.value_or(options);
            for (auto level = static_cast<int>(*options.reasoning_effort) + 1;
                 level <= static_cast<int>(ReasoningEffort::Max); ++level) {
                stronger.reasoning_effort = static_cast<ReasoningEffort>(level);
                try {
                    return render_once(retried, stronger, control);
                } catch (const std::invalid_argument&) {}
            }
        }
        throw;
    }
}

RenderedChat CompiledChatTemplate::render_once(const std::vector<ChatMessage>& messages,
                                               const ChatRenderOptions& options,
                                               const PreparationControl& control) const {
    if (messages.empty()) throw std::invalid_argument("chat requires at least one message");
    check_preparation_control(control, "chat template");
    const bool continuation =
        options.continuation == PromptContinuationMode::ContinueFinalAssistant;
    Json context = template_parameters(options, special_tokens_);
    if (continuation) {
        const auto& final = messages.back();
        if (final.role != ChatRole::Assistant || final.has_media() ||
            !final.reasoning_content.empty() || !final.tool_calls.empty() ||
            context.value("enable_thinking", false)) {
            throw std::invalid_argument("assistant continuation requires a final text-only "
                                        "assistant message and disabled thinking");
        }
    }
    context["continue_final_message"] = continuation;
    context["messages"]               = Json::array();
    context["add_generation_prompt"]  = !continuation && options.add_generation_prompt;
    std::vector<text::TemplateInputRegion> regions;
    std::uint32_t next_tag = 1;
    const auto tag         = [&](std::string pointer) {
        const auto id = next_tag++;
        regions.push_back({std::move(pointer), id});
        return id;
    };
    // A role is one of a closed set the request parser checked, not request text: it renders as
    // template text, so a template that wraps it in control-token markup ('<|' + role + '|>',
    // Phi-3/Phi-4) writes the control token, as transformers does.
    std::vector<std::string> control_pointers;
    std::vector<ContentSource> sources(messages.size());
    std::vector<Modality> media;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const auto& message = messages[i];
        if (instruction(message.role) &&
            (message.has_media() || !message.reasoning_content.empty() ||
             !message.tool_calls.empty() || !message.tool_call_id.empty())) {
            throw std::invalid_argument("system and developer messages may contain only text");
        }
        Json value{{"role", role_name(message.role)}};
        const auto pointer = "/messages/" + std::to_string(i);
        control_pointers.push_back(pointer + "/role");
        auto& origin       = sources[i];
        if (!message.has_media()) {
            std::string content;
            for (const auto& part : message.parts) {
                content += part.text;
                origin.part_ends.push_back(content.size());
            }
            value["content"] = std::move(content);
            origin.tag       = tag(pointer + "/content");
        } else {
            value["content"] = Json::array();
            for (std::size_t j = 0; j < message.parts.size(); ++j) {
                const auto& part = message.parts[j];
                const auto item  = pointer + "/content/" + std::to_string(j);
                if (part.kind == ChatPartKind::Text) {
                    value["content"].push_back({{"type", "text"}, {"text", part.text}});
                    origin.part_tags.push_back(tag(item + "/text"));
                    origin.part_media.push_back(std::nullopt);
                } else {
                    const bool image = part.kind == ChatPartKind::Image;
                    value["content"].push_back({{"type", image ? "image" : "video"}});
                    origin.part_media.push_back(media.size());
                    media.push_back(image ? Modality::Image : Modality::Video);
                    origin.part_tags.push_back(0);
                }
            }
        }
        if (!message.reasoning_content.empty()) {
            value["reasoning_content"] = message.reasoning_content;
            origin.reasoning_tag       = tag(pointer + "/reasoning_content");
        }
        if (!message.tool_calls.empty()) {
            value["tool_calls"] = Json::array();
            for (const auto& call : message.tool_calls) {
                // History replays what a model (ours or another backend's) emitted, and a model
                // can emit arguments that are not a JSON object. Refusing them would fail every
                // later request of the session, so the raw text is kept under "raw": a mapping
                // every template can render.
                Json arguments = Json::object();
                if (!call.arguments_json.empty()) {
                    arguments = Json::parse(call.arguments_json, nullptr, false);
                    if (arguments.is_discarded() || !arguments.is_object()) {
                        arguments = Json{{"raw", call.arguments_json}};
                    }
                }
                value["tool_calls"].push_back(
                    {{"id", call.id},
                     {"type", "function"},
                     {"function", {{"name", call.name}, {"arguments", std::move(arguments)}}}});
            }
        }
        if (!message.tool_call_id.empty()) value["tool_call_id"] = message.tool_call_id;
        context["messages"].push_back(std::move(value));
    }
    std::vector<std::uint32_t> tool_tags;
    if (!options.tool_jsons.empty()) {
        context["tools"] = Json::array();
        for (std::size_t i = 0; i < options.tool_jsons.size(); ++i) {
            context["tools"].push_back(Json::parse(options.tool_jsons[i]));
            tool_tags.push_back(tag("/tools/" + std::to_string(i)));
        }
    } else {
        // transformers' apply_chat_template always passes tools, None without any: a template
        // that tests `tools is none` (OLMo 3) must see none, not an undefined variable.
        context["tools"] = nullptr;
    }
    std::vector<std::string> control_variables;
    for (const auto& item : special_tokens_.items()) control_variables.push_back(item.key());
    text::TemplateRenderOptions execution{
        .checkpoint        = [&] { check_preparation_control(control, "chat template"); },
        .regions           = regions,
        .control_variables = control_variables,
        .control_pointers  = control_pointers};
    auto output = compiled_.render(context, execution);
    // A template may drop a system or developer message after the conversation began instead of
    // refusing it (a Qwen 3.6 template renders only the leading ones): its text would be lost
    // without an error. Such a render is refused like one the template raised on, and render()
    // gives the message the user role. The leading run is the template's to merge.
    std::size_t leading = 0;
    while (leading < messages.size() && instruction(messages[leading].role)) ++leading;
    for (std::size_t i = leading; i < messages.size(); ++i) {
        if (!instruction(messages[i].role)) continue;
        std::string text;
        for (const auto& part : messages[i].parts) text += part.text;
        const auto first = text.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        const auto last = text.find_last_not_of(" \t\r\n");
        const bool rendered =
            std::any_of(output.regions.begin(), output.regions.end(),
                        [&](const auto& region) { return region.tag == sources[i].tag; }) ||
            output.text.find(std::string_view(text).substr(first, last + 1 - first)) !=
                std::string::npos;
        if (!rendered) {
            throw std::invalid_argument("the chat template drops a " +
                                        std::string(role_name(messages[i].role)) +
                                        " message after the conversation began");
        }
    }
    auto layout = inspect_prompt_layout(output, media, conventions_);
    if (continuation) {
        const auto content = unique_output_region(output, sources.back().tag);
        if (!content || layout.messages.empty())
            throw std::invalid_argument(
                "chat template does not expose the final assistant content for continuation");
        const auto& final = layout.messages.back();
        if (final.role != ChatRole::Assistant || content->begin < final.content_begin ||
            content->end > final.content_end) {
            throw std::invalid_argument(
                "chat template cannot continue the final assistant content unambiguously");
        }
        output.text.resize(content->end);
        std::erase_if(output.regions,
                      [&](const auto& region) { return region.begin > output.text.size(); });
        for (auto& region : output.regions) region.end = std::min(region.end, output.text.size());
        std::erase_if(output.literal_spans,
                      [&](auto span) { return span.begin >= output.text.size(); });
        if (!output.literal_spans.empty())
            output.literal_spans.back().end =
                std::min(output.literal_spans.back().end, output.text.size());
        layout = inspect_prompt_layout(output, media, conventions_);
    }
    RenderedChat result;
    result.starts_in_reasoning          = layout.starts_in_reasoning;
    result.reasoning_open_allowed       = !continuation && layout.reasoning_open_allowed;
    const bool thinking =
        context.contains("enable_thinking") && context["enable_thinking"].is_boolean()
            ? context["enable_thinking"].get<bool>()
            : thinking_default_;
    result.reasoning_declined           = thinking_switch_ && !thinking;
    result.media_placeholders           = layout.media_placeholders;
    result.rewrite_execution_boundaries = layout.execution_boundaries;
    result.message_boundaries.resize(messages.size() + 1);
    result.cache_boundaries.resize(options.cache_markers.size());

    // Only requested/structural boundaries need proof, independent of history length.
    std::map<std::size_t, std::optional<std::size_t>> prefix_cache;
    auto prefix = [&](std::size_t count) -> std::optional<std::size_t> {
        if (count > messages.size()) return std::nullopt;
        if (const auto it = prefix_cache.find(count); it != prefix_cache.end()) return it->second;
        std::optional<std::size_t> boundary;
        Json probe = context;
        probe["messages"].erase(probe["messages"].begin() + static_cast<std::ptrdiff_t>(count),
                                probe["messages"].end());
        probe["add_generation_prompt"] = false;
        try {
            const auto rendered = compiled_.render(probe, execution);
            if (same_prefix(output, rendered, rendered.text.size()))
                boundary = rendered.text.size();
        } catch (const RequestError&) {
            throw;
        } catch (const std::invalid_argument&) { /* This subset has no independent serialization. */
        }
        prefix_cache.emplace(count, boundary);
        result.message_boundaries[count] = boundary;
        return boundary;
    };
    // Associate actual input regions with complete ChatML blocks. Several tool results may
    // share one user block; their individual closing tags provide the intermediate boundaries.
    std::vector<std::optional<std::size_t>> message_blocks(messages.size());
    std::vector<std::optional<text::ByteSpan>> content_regions(messages.size());
    std::vector<std::size_t> block_users(layout.messages.size());
    for (std::size_t i = 0; i < sources.size(); ++i) {
        const auto& origin = sources[i];
        std::optional<text::ByteSpan> extent;
        const auto include = [&](text::ByteSpan bytes) {
            if (!extent)
                extent = bytes;
            else {
                extent->begin = std::min(extent->begin, bytes.begin);
                extent->end   = std::max(extent->end, bytes.end);
            }
        };
        for (const auto& region : output.regions) {
            if (region.tag == origin.tag || region.tag == origin.reasoning_tag ||
                std::find(origin.part_tags.begin(), origin.part_tags.end(), region.tag) !=
                    origin.part_tags.end()) {
                include({region.begin, region.end});
            }
        }
        for (const auto index : origin.part_media) {
            if (index) include(layout.media_placeholders[*index].bytes);
        }
        content_regions[i] = extent;
        if (extent) message_blocks[i] = containing_message(layout, *extent);
        if (message_blocks[i]) ++block_users[*message_blocks[i]];
    }
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (!message_blocks[i]) continue;
        const auto& block = layout.messages[*message_blocks[i]];
        if (block.closed && block_users[*message_blocks[i]] == 1)
            result.message_boundaries[i + 1] = block.end;
        const std::string_view close = conventions_.tools.response_boundary;
        if (messages[i].role == ChatRole::Tool && content_regions[i] && !close.empty()) {
            const auto pos                   = output.text.find(close, content_regions[i]->end);
            if (pos != std::string::npos && pos + close.size() <= block.content_end) {
                const auto end = pos + close.size();
                const bool sourced =
                    text::overlaps(output.literal_spans, content_regions[i]->end, end);
                if (!sourced)
                    result.message_boundaries[i + 1] =
                        end == block.content_end && block.closed ? block.end : end;
            }
        }
    }
    if (instruction(messages.front().role) && !result.message_boundaries[1]) (void)prefix(1);

    const auto generation_begin = continuation
                                      ? std::optional<std::size_t>(layout.messages.back().begin)
                                  : options.add_generation_prompt ? prefix(messages.size())
                                                                  : std::nullopt;
    if (!options.add_generation_prompt && !continuation)
        result.message_boundaries.back() = output.text.size();
    std::optional<std::size_t> first_tail_assistant;
    if (!continuation) {
        std::optional<std::size_t> query_message;
        for (std::size_t i = messages.size(); i > 0; --i) {
            if (real_user(messages[i - 1], conventions_.tools)) {
                query_message = i - 1;
                break;
            }
            // Imported history may begin with a tool result after its user turn was removed.
            if (!query_message && messages[i - 1].role == ChatRole::Tool) query_message = i - 1;
        }
        if (query_message && message_blocks[*query_message]) {
            for (std::size_t j = *message_blocks[*query_message] + 1; j < layout.messages.size();
                 ++j) {
                if (layout.messages[j].role == ChatRole::Assistant) {
                    first_tail_assistant = j;
                    break;
                }
            }
        }
    }
    // Select a recovery point from the template's observed serialization. A single ordinary
    // next-turn probe distinguishes retained history from a rewritten open turn, including
    // templates whose defaults or aliases differ from the typed request hint. This only chooses
    // the checkpoint to retain: subsequent requests still require exact token/state identity.
    // A request ending with tool results is inside a tool loop: its next request appends the
    // model's next step to the same turn rather than a user turn, so that is the continuation to
    // probe. A template that rewrites the turn only once the user speaks again (Qwen dropping
    // in-loop reasoning, Gemma closing a model turn left open across tool results) then keeps the
    // loop's whole prefix instead of restarting every step at the loop's first assistant turn;
    // that earlier boundary was already published by the loop's first request.
    bool retain_open_turn = context.value("preserve_thinking", false);
    const bool tool_loop  = generation_begin && messages.back().role == ChatRole::Tool;
    if (first_tail_assistant && (layout.messages[*first_tail_assistant].closed || tool_loop)) {
        Json probe = context;
        probe["messages"].push_back(tool_loop ? Json{{"role", "assistant"}, {"content", "next"}}
                                              : Json{{"role", "user"}, {"content", ""}});
        probe["add_generation_prompt"] = false;
        auto probe_options             = execution;
        probe_options.regions          = {};
        std::vector<std::string> probe_pointers = control_pointers;
        probe_pointers.push_back("/messages/" + std::to_string(messages.size()) + "/role");
        probe_options.control_pointers = probe_pointers;
        try {
            const auto next_turn     = compiled_.render(probe, probe_options);
            const auto history_bytes = generation_begin.value_or(output.text.size());
            retain_open_turn         = same_prefix(next_turn, output, history_bytes);
        } catch (const RequestError&) { throw; } catch (const std::invalid_argument&) {
            retain_open_turn = false;
        }
    }
    if (generation_begin && *generation_begin > 0) {
        result.rewrite_checkpoint = RewriteCheckpointByteSpec{
            .kind   = continuation || retain_open_turn ? RewriteCheckpointKind::ResponseReplay
                                                       : RewriteCheckpointKind::TurnClosure,
            .offset = *generation_begin};
    }
    if (!continuation && !retain_open_turn && first_tail_assistant) {
        const auto begin = layout.messages[*first_tail_assistant].begin;
        if (begin > 0)
            result.rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::TurnClosure, .offset = begin};
    }
    for (std::size_t i = 0; i < options.cache_markers.size(); ++i) {
        const auto& marker = options.cache_markers[i];
        switch (marker.location) {
        case PromptCacheMarkerLocation::MessageBoundary:
            if (marker.after_message_count < result.message_boundaries.size()) {
                auto& boundary = result.message_boundaries[marker.after_message_count];
                if (!boundary) boundary = prefix(marker.after_message_count);
                result.cache_boundaries[i] = boundary;
            }
            break;
        case PromptCacheMarkerLocation::LeadingInstructionBoundary:
            if (instruction(messages.front().role))
                result.cache_boundaries[i] =
                    source_boundary(output, sources.front().tag, marker.leading_instruction_bytes);
            break;
        case PromptCacheMarkerLocation::ToolBoundary:
            if (marker.after_tool_count && marker.after_tool_count <= tool_tags.size()) {
                if (const auto region =
                        unique_output_region(output, tool_tags[marker.after_tool_count - 1]))
                    result.cache_boundaries[i] = region->end;
            }
            break;
        case PromptCacheMarkerLocation::MessagePartBoundary:
            if (marker.after_message_count && marker.after_message_count <= sources.size() &&
                marker.after_message_part_count) {
                const auto& origin = sources[marker.after_message_count - 1];
                const auto part    = marker.after_message_part_count - 1;
                if (part < origin.part_ends.size())
                    result.cache_boundaries[i] =
                        source_boundary(output, origin.tag, origin.part_ends[part]);
                else if (part < origin.part_tags.size()) {
                    if (origin.part_tags[part]) {
                        if (const auto region =
                                unique_output_region(output, origin.part_tags[part]))
                            result.cache_boundaries[i] = region->end;
                    } else if (origin.part_media[part]) {
                        result.cache_boundaries[i] =
                            layout.media_placeholders[*origin.part_media[part]].bytes.end +
                            std::string_view("<|vision_end|>").size();
                    }
                }
            }
            break;
        }
    }
    result.text          = std::move(output.text);
    result.literal_spans = std::move(output.literal_spans);
    return result;
}

} // namespace ninferno::models::qwen3_5::frontend
