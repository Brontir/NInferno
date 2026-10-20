#include "chat/conventions.h"

#include "chat/embedded_conventions.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>

namespace ninferno::chat {
namespace {

using Json = nlohmann::json;

const Json& table() {
    static const Json parsed = [] {
        Json value = Json::parse(embedded::kConventionsJson);
        if (!value.is_object()) { throw std::logic_error("conventions table must be an object"); }
        return value;
    }();
    return parsed;
}

bool detected(const Json& entry, std::string_view source) {
    for (const Json& marker : entry.at("detect")) {
        if (source.find(marker.get<std::string>()) == std::string_view::npos) { return false; }
    }
    return true;
}

const Json* first_match(const char* category, std::string_view source) {
    for (const Json& entry : table().at(category)) {
        if (detected(entry, source)) { return &entry; }
    }
    return nullptr;
}

ChatRole parse_role(const std::string& name) {
    if (name == "system") { return ChatRole::System; }
    if (name == "developer") { return ChatRole::Developer; }
    if (name == "user") { return ChatRole::User; }
    if (name == "assistant") { return ChatRole::Assistant; }
    if (name == "tool") { return ChatRole::Tool; }
    throw std::logic_error("conventions table names an unknown role: " + name);
}

ToolCallFormat parse_format(const std::string& name) {
    if (name == "qwen_xml") { return ToolCallFormat::QwenXml; }
    if (name == "hermes_json") { return ToolCallFormat::HermesJson; }
    if (name == "gemma4") { return ToolCallFormat::Gemma4; }
    if (name == "mistral_json") { return ToolCallFormat::MistralJson; }
    if (name == "llama3_json") { return ToolCallFormat::Llama3Json; }
    if (name == "harmony") { return ToolCallFormat::Harmony; }
    if (name == "pythonic") { return ToolCallFormat::Pythonic; }
    if (name == "granite_json") { return ToolCallFormat::GraniteJson; }
    if (name == "atem") { return ToolCallFormat::Atem; }
    throw std::logic_error("conventions table names an unknown tool-call format: " + name);
}

} // namespace

std::optional<ChatRole> TurnMarkup::role(std::string_view header) const noexcept {
    for (const auto& [text, role] : roles) {
        if (text == header) { return role; }
    }
    return std::nullopt;
}

std::vector<std::string> ChatConventions::structural_markers() const {
    std::vector<std::string> markers;
    if (reasoning) {
        markers.push_back(reasoning->open);
        markers.push_back(reasoning->close);
        markers.insert(markers.end(), reasoning->content_open.begin(),
                       reasoning->content_open.end());
    }
    switch (tools.format) {
    case ToolCallFormat::None:
        break;
    case ToolCallFormat::QwenXml:
    case ToolCallFormat::HermesJson:
        markers.insert(markers.end(), {"<tool_call>", "</tool_call>"});
        break;
    case ToolCallFormat::Gemma4:
        markers.insert(markers.end(), {"<|tool_call>", "<tool_call|>", "<|\"|>"});
        break;
    case ToolCallFormat::MistralJson:
        markers.insert(markers.end(), {"[TOOL_CALLS]", "[ARGS]"});
        break;
    case ToolCallFormat::Llama3Json:
        // Plain JSON text: no marker is a special token.
        break;
    case ToolCallFormat::Harmony:
        markers.insert(markers.end(),
                       {"<|start|>", "<|channel|>", "<|message|>", "<|constrain|>", "<|call|>"});
        break;
    case ToolCallFormat::Pythonic:
        markers.insert(markers.end(), {"<function_calls>", "</function_calls>"});
        break;
    case ToolCallFormat::GraniteJson:
        markers.push_back("<|tool_call|>");
        break;
    case ToolCallFormat::Atem:
        // The call's message header (special tokens in its tokenizer) and the block's tags.
        markers.insert(markers.end(), {"<|start|>", "<|message|>", "<|eom|>",
                                       "<atem:function_calls>", "</atem:function_calls>"});
        break;
    }
    return markers;
}

std::vector<std::string> ChatConventions::generation_stops() const {
    return turns ? turns->stops : std::vector<std::string>{};
}

bool ChatConventions::within_reply(std::string_view token) const {
    if (reasoning && (token == reasoning->open || token == reasoning->close)) { return true; }
    return turns && token == turns->close &&
           std::find(turns->stops.begin(), turns->stops.end(), token) == turns->stops.end();
}

std::vector<std::string> generic_generation_stops() {
    return table().at("generic_stops").get<std::vector<std::string>>();
}

ChatConventions resolve_conventions(std::string_view source) {
    ChatConventions result;
    if (const Json* entry = first_match("turns", source)) {
        TurnMarkup turns;
        turns.name        = entry->at("name").get<std::string>();
        turns.open        = entry->at("open").get<std::string>();
        turns.header_end  = entry->at("header_end").get<std::string>();
        turns.close       = entry->at("close").get<std::string>();
        turns.after_close = entry->at("after_close").get<std::string>();
        turns.stops       = entry->at("stops").get<std::vector<std::string>>();
        for (const auto& [header, role] : entry->at("roles").items()) {
            turns.roles.emplace_back(header, parse_role(role.get<std::string>()));
        }
        result.turns = std::move(turns);
    }
    if (const Json* entry = first_match("reasoning", source)) {
        result.reasoning = ReasoningMarkup{
            .name            = entry->at("name").get<std::string>(),
            .open            = entry->at("open").get<std::string>(),
            .open_suffix     = entry->at("open_suffix").get<std::string>(),
            .close           = entry->at("close").get<std::string>(),
            .canonical_close = entry->at("canonical_close").get<std::string>(),
            .budget_control  = entry->at("budget_control").get<std::string>(),
            .content_open    = entry->value("content_open", std::vector<std::string>{}),
        };
    }
    if (const Json* entry = first_match("tools", source)) {
        result.tools = ToolMarkup{
            .format            = parse_format(entry->at("format").get<std::string>()),
            .response_open     = entry->at("response_open").get<std::string>(),
            .response_close    = entry->at("response_close").get<std::string>(),
            .response_boundary = entry->at("response_boundary").get<std::string>(),
        };
    }
    return result;
}

bool is_cut_content_header(std::string_view text, std::span<const std::string> headers) {
    const bool markup =
        text.find("<|") != std::string_view::npos || text.find('=') != std::string_view::npos;
    if (!markup) { return false; }
    for (const std::string& header : headers) {
        if (text.size() < header.size() && std::string_view(header).starts_with(text)) {
            return true;
        }
    }
    return false;
}

namespace {

// The answer without the header it may begin with (ReasoningMarkup::content_open), as the serve
// presentation drops it.
std::string without_content_header(std::string_view content, const ReasoningMarkup& markup) {
    for (const std::string& header : markup.content_open) {
        if (content.starts_with(header)) { return std::string(content.substr(header.size())); }
    }
    if (is_cut_content_header(content, markup.content_open)) { return {}; }
    return std::string(content);
}

} // namespace

ReasoningSplit split_reasoning(std::string_view output, const ReasoningMarkup& markup) {
    ReasoningSplit split;
    std::size_t start = 0;
    while (start < output.size() && (output[start] == ' ' || output[start] == '\n')) { ++start; }
    // An opener that itself begins with a space (" to=self<|message|>") matches in place.
    std::size_t open_at = std::string_view::npos;
    if (!markup.open.empty()) {
        if (output.substr(0, markup.open.size()) == markup.open) {
            open_at = 0;
        } else if (output.substr(start, markup.open.size()) == markup.open) {
            open_at = start;
        }
    }
    if (open_at == std::string_view::npos) {
        split.content = without_content_header(output, markup);
        return split;
    }
    split.opened     = true;
    std::size_t body = open_at + markup.open.size();
    if (!markup.open_suffix.empty() &&
        output.substr(body, markup.open_suffix.size()) == markup.open_suffix) {
        body += markup.open_suffix.size();
    }
    const std::size_t close = output.find(markup.close, body);
    if (close == std::string_view::npos) {
        split.reasoning = std::string(output.substr(body));
        return split;
    }
    split.closed    = true;
    split.reasoning = std::string(output.substr(body, close - body));
    split.content   = without_content_header(output.substr(close + markup.close.size()), markup);
    return split;
}

std::string_view tool_call_format_name(ToolCallFormat format) noexcept {
    switch (format) {
    case ToolCallFormat::None:
        return "none";
    case ToolCallFormat::QwenXml:
        return "qwen_xml";
    case ToolCallFormat::HermesJson:
        return "hermes_json";
    case ToolCallFormat::Gemma4:
        return "gemma4";
    case ToolCallFormat::MistralJson:
        return "mistral_json";
    case ToolCallFormat::Llama3Json:
        return "llama3_json";
    case ToolCallFormat::Harmony:
        return "harmony";
    case ToolCallFormat::Pythonic:
        return "pythonic";
    case ToolCallFormat::GraniteJson:
        return "granite_json";
    case ToolCallFormat::Atem:
        return "atem";
    }
    return "unknown";
}

} // namespace ninferno::chat
