#include "runtime/contract/sampling.h"
#include "serve/anthropic_messages.h"
#include "serve/output_constraint_request.h"
#include "serve/request_validation.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace ninferno::serve {
namespace {

using Json = RequestJson;

// Tool names share the protocol-wide limit (kMaximumToolNameLength, serve/request.h). Anthropic's
// own boundary is narrower, but agent hosts send longer names — VS Code Copilot wraps MCP tools as
// "activate_fallback_mcp_<server>_<tool>", past 64 bytes.

enum class ParsePurpose {
    Messages,
    CountTokens,
};

[[noreturn]] void invalid_tool_history(std::string message) {
    bad_request(std::move(message), "messages", "invalid_tool_history");
}

const Json& require_object(const Json& body) {
    if (!body.is_object()) { bad_request("request body must be a JSON object"); }
    return body;
}

std::string require_string(const Json& object, const char* key, const char* param,
                           const char* description) {
    if (!object.contains(key) || !object.at(key).is_string()) {
        bad_request(std::string(description) + " must contain a string '" + key + "'", param);
    }
    return object.at(key).get<std::string>();
}

std::string require_block_type(const Json& block, const char* param) {
    if (!block.is_object() || !block.contains("type") || !block.at("type").is_string()) {
        bad_request("content blocks must be objects with a string 'type'", param);
    }
    return block.at("type").get<std::string>();
}

std::optional<CacheBoundary::Ttl> cache_boundary(const Json& value, const char* param) {
    if (!value.is_object() || !value.contains("cache_control") ||
        value.at("cache_control").is_null()) {
        return std::nullopt;
    }
    const Json& control = value.at("cache_control");
    if (!control.is_object()) {
        bad_request("cache_control must be an object", param, "invalid_cache_control");
    }
    if (!control.contains("type") || !control.at("type").is_string() ||
        control.at("type").get<std::string>() != "ephemeral") {
        bad_request("cache_control.type must be 'ephemeral'", param, "invalid_cache_control");
    }
    if (!control.contains("ttl") || control.at("ttl").is_null()) {
        return CacheBoundary::Ttl::FiveMinutes;
    }
    if (!control.at("ttl").is_string()) {
        bad_request("cache_control.ttl must be '5m' or '1h'", param, "invalid_cache_control");
    }
    const std::string ttl = control.at("ttl").get<std::string>();
    if (ttl == "5m") { return CacheBoundary::Ttl::FiveMinutes; }
    if (ttl == "1h") { return CacheBoundary::Ttl::OneHour; }
    bad_request("cache_control.ttl must be '5m' or '1h'", param, "invalid_cache_control");
}

CacheBoundary explicit_cache_boundary(CacheBoundary::Ttl ttl) {
    return CacheBoundary{.kind     = ninferno::PromptCacheMarkerKind::SharedStablePrefix,
                         .evidence = ninferno::SharedCandidateEvidence::ExplicitBoundary,
                         .ttl      = ttl};
}

ninferno::product::media_acquire::Source parse_image_source(const Json& block) {
    if (!block.contains("source") || !block.at("source").is_object()) {
        bad_request("image block must contain a source object", "messages");
    }
    const Json& source     = block.at("source");
    const std::string type = require_string(source, "type", "messages", "image source");
    ninferno::product::media_acquire::Source result;
    if (type == "base64") {
        result.media_type = require_string(source, "media_type", "messages", "base64 image source");
        const std::string data = require_string(source, "data", "messages", "base64 image source");
        if (data.empty()) { bad_request("base64 image data must not be empty", "messages"); }
        result.kind  = ninferno::product::media_acquire::SourceKind::Data;
        result.value = "data:" + result.media_type + ";base64," + data;
        return result;
    }
    if (type == "url") {
        return media_source(require_string(source, "url", "messages", "URL image source"),
                            "image URL", "messages", false);
    }
    if (type == "file") {
        bad_request("image file sources require an Anthropic Files API, which NInferno does not "
                    "provide",
                    "messages", "files_not_supported");
    }
    bad_request("unsupported image source type: " + ascii_preview(type), "messages");
}

ContentPart parse_image(const Json& block) {
    ContentPart result;
    result.kind     = ContentKind::Image;
    result.type_raw = "image";
    result.source   = parse_image_source(block);
    if (const auto ttl = cache_boundary(block, "messages")) {
        result.cache_boundary_after = explicit_cache_boundary(*ttl);
    }
    if (!block.contains("transformations") || block.at("transformations").is_null()) {
        return result;
    }
    const Json& transformations = block.at("transformations");
    if (!transformations.is_object()) {
        bad_request("image transformations must be an object", "messages");
    }
    if (!transformations.contains("oversized_image") ||
        transformations.at("oversized_image").is_null()) {
        return result;
    }
    if (!transformations.at("oversized_image").is_string()) {
        bad_request("transformations.oversized_image must be 'downsize' or 'error'", "messages");
    }
    const std::string policy = transformations.at("oversized_image").get<std::string>();
    if (policy == "downsize") { return result; }
    if (policy == "error") {
        result.image_resize_policy = ninferno::ImageResizePolicy::RejectOversized;
        return result;
    }
    bad_request("transformations.oversized_image must be 'downsize' or 'error'", "messages");
}

ContentPart text_part(std::string value,
                      std::optional<CacheBoundary::Ttl> boundary = std::nullopt) {
    ContentPart part;
    part.kind     = ContentKind::Text;
    part.type_raw = "text";
    part.text     = std::move(value);
    if (boundary) { part.cache_boundary_after = explicit_cache_boundary(*boundary); }
    return part;
}

// A document block as the text the model reads: its title and context, then its text in a
// <document> element. Plain-text sources (source.type "text") and content documents (source.type
// "content": text blocks, or a string) are read as they are; a PDF (source.type "base64" or "url")
// is read by its text layer once acquired (ContentKind::Document). Citations are not produced: the
// answer is ordinary text.
std::string document_text(const Json& block);

std::string optional_document_string(const Json& block, const char* field) {
    if (!block.contains(field) || block.at(field).is_null()) return {};
    if (!block.at(field).is_string()) {
        bad_request(std::string("document ") + field + " must be a string", "messages");
    }
    return block.at(field).get<std::string>();
}

ContentPart document_part(const Json& block) {
    const auto boundary = cache_boundary(block, "messages");
    if (block.contains("source") && block.at("source").is_object()) {
        const Json& source     = block.at("source");
        const std::string kind = source.contains("type") && source.at("type").is_string()
                                     ? source.at("type").get<std::string>()
                                     : std::string();
        if (kind == "base64" || kind == "url") {
            ContentPart part;
            part.kind     = ContentKind::Document;
            part.type_raw = "document";
            if (kind == "base64") {
                part.source.media_type =
                    require_string(source, "media_type", "messages", "base64 document source");
                if (part.source.media_type != "application/pdf") {
                    bad_request("a base64 document must be application/pdf; send text as a text "
                                "source",
                                "messages", "documents_not_supported");
                }
                const std::string data =
                    require_string(source, "data", "messages", "base64 document source");
                if (data.empty()) bad_request("base64 document data must not be empty", "messages");
                part.source.kind  = ninferno::product::media_acquire::SourceKind::Data;
                part.source.value = "data:application/pdf;base64," + data;
            } else {
                part.source =
                    media_source(require_string(source, "url", "messages", "URL document source"),
                                 "document URL", "messages", false);
                part.source.media_type = "application/pdf";
            }
            part.document_title   = optional_document_string(block, "title");
            part.document_context = optional_document_string(block, "context");
            if (boundary) part.cache_boundary_after = explicit_cache_boundary(*boundary);
            return part;
        }
        if (kind == "file") {
            bad_request("document file sources require an Anthropic Files API, which NInferno "
                        "does not provide",
                        "messages", "files_not_supported");
        }
    }
    return text_part(document_text(block), boundary);
}

std::string document_text(const Json& block) {
    if (!block.contains("source") || !block.at("source").is_object()) {
        bad_request("a document block needs a source object", "messages");
    }
    const Json& source = block.at("source");
    const std::string kind =
        source.contains("type") && source.at("type").is_string() ? source.at("type").get<std::string>()
                                                                 : std::string();
    std::string body;
    if (kind == "text") {
        body = require_string(source, "data", "messages", "text document source");
    } else if (kind == "content") {
        if (!source.contains("content")) {
            bad_request("a content document needs source.content", "messages");
        }
        const Json& content = source.at("content");
        if (content.is_string()) {
            body = content.get<std::string>();
        } else if (content.is_array()) {
            for (const Json& part : content) {
                if (require_block_type(part, "messages") != "text") {
                    bad_request("a content document holds text blocks only here", "messages",
                                "documents_not_supported");
                }
                if (!body.empty()) body += "\n\n";
                body += require_string(part, "text", "messages", "document text block");
            }
        } else {
            bad_request("source.content must be a string or an array of text blocks", "messages");
        }
    } else {
        bad_request("document source type '" + kind +
                        "' is not one NInferno reads; send text (source.type \"text\"), content "
                        "blocks, or a PDF (\"base64\" or \"url\")",
                    "messages", "documents_not_supported");
    }
    std::string out = "<document>\n";
    if (block.contains("title") && block.at("title").is_string()) {
        out += "<title>" + block.at("title").get<std::string>() + "</title>\n";
    }
    if (block.contains("context") && block.at("context").is_string()) {
        out += "<context>" + block.at("context").get<std::string>() + "</context>\n";
    }
    out += body;
    out += "\n</document>";
    return out;
}

bool is_server_tool_block(std::string_view type) {
    return type == "server_tool_use" || type == "mcp_tool_use" || type == "mcp_tool_result" ||
           type == "container_upload" || type == "tool_reference" ||
           (type.ends_with("_tool_result") && type != "tool_result");
}

void append_turn(ChatTurn& target, ChatTurn turn) {
    if (target.cache_boundary_after && !target.content.empty()) {
        target.content.back().cache_boundary_after = target.cache_boundary_after;
        target.cache_boundary_after.reset();
    }
    target.content.insert(target.content.end(), std::make_move_iterator(turn.content.begin()),
                          std::make_move_iterator(turn.content.end()));
    target.tool_calls.insert(target.tool_calls.end(),
                             std::make_move_iterator(turn.tool_calls.begin()),
                             std::make_move_iterator(turn.tool_calls.end()));
    target.reasoning_content += turn.reasoning_content;
    target.cache_boundary_after = turn.cache_boundary_after;
}

void merge_turn(GenerationRequest& request, ChatTurn turn) {
    if (request.messages.empty() ||
        (turn.role != ChatRole::User && turn.role != ChatRole::Assistant) ||
        request.messages.back().role != turn.role) {
        request.messages.push_back(std::move(turn));
        return;
    }
    append_turn(request.messages.back(), std::move(turn));
}

struct ParsedToolResult {
    std::string tool_use_id;
    std::vector<ContentPart> content;
    bool is_error = false;
    std::optional<CacheBoundary::Ttl> cache_boundary_after;
};

using ParsedUserBlock = std::variant<ContentPart, ParsedToolResult>;

struct ParsedMessage {
    ChatRole role = ChatRole::User;
    ChatTurn turn;
    std::vector<ParsedUserBlock> user_blocks;
};

std::string require_tool_history_id(const Json& object, const char* key,
                                    std::string_view block_name) {
    if (!object.contains(key) || !object.at(key).is_string() ||
        object.at(key).get_ref<const std::string&>().empty()) {
        invalid_tool_history(std::string(block_name) + " " + key + " must be a non-empty string");
    }
    return object.at(key).get<std::string>();
}

std::vector<ContentPart> parse_tool_result_content(const Json& block) {
    if (!block.contains("content") || block.at("content").is_null()) { return {text_part({})}; }
    const Json& content = block.at("content");
    if (content.is_string()) { return {text_part(content.get<std::string>())}; }
    if (!content.is_array()) {
        bad_request("tool_result content must be a string or array", "messages");
    }
    std::vector<ContentPart> result;
    for (const Json& part : content) {
        const std::string type = require_block_type(part, "messages");
        if (type == "text") {
            result.push_back(text_part(require_string(part, "text", "messages", "text block"),
                                       cache_boundary(part, "messages")));
        } else if (type == "image") {
            result.push_back(parse_image(part));
        } else if (type == "document") {
            result.push_back(document_part(part));
        } else {
            bad_request("unsupported tool_result content block: " + type, "messages",
                        "content_block_not_supported");
        }
    }
    if (result.empty()) { result.push_back(text_part({})); }
    return result;
}

ToolCall parse_tool_use(const Json& block) {
    ToolCall call;
    call.id   = require_tool_history_id(block, "id", "tool_use block");
    call.name = require_tool_name(block, "messages", "tool name");
    if (!block.contains("input") || !block.at("input").is_object()) {
        bad_request("tool_use blocks must contain an input object", "messages");
    }
    call.arguments_json = block.at("input").dump();
    return call;
}

ParsedToolResult parse_tool_result(const Json& block) {
    ParsedToolResult result;
    result.tool_use_id = require_tool_history_id(block, "tool_use_id", "tool_result block");
    result.content     = parse_tool_result_content(block);
    if (block.contains("is_error") && !block.at("is_error").is_null()) {
        if (!block.at("is_error").is_boolean()) {
            bad_request("tool_result.is_error must be a boolean", "messages");
        }
        result.is_error = block.at("is_error").get<bool>();
    }
    result.cache_boundary_after = cache_boundary(block, "messages");
    return result;
}

std::vector<ParsedUserBlock> parse_user_blocks(const Json& content) {
    std::vector<ParsedUserBlock> result;
    result.reserve(content.size());
    for (const Json& block : content) {
        const std::string type = require_block_type(block, "messages");
        if (type == "text") {
            result.emplace_back(text_part(require_string(block, "text", "messages", "text block"),
                                          cache_boundary(block, "messages")));
        } else if (type == "image") {
            result.emplace_back(parse_image(block));
        } else if (type == "tool_result") {
            result.emplace_back(parse_tool_result(block));
        } else if (type == "document") {
            result.emplace_back(document_part(block));
        } else if (type == "search_result") {
            bad_request("search_result blocks require source and citation semantics that NInferno "
                        "does not provide",
                        "messages", "search_results_not_supported");
        } else if (is_server_tool_block(type)) {
            bad_request("server tool result blocks require an executor that NInferno does not "
                        "provide",
                        "messages", "server_tools_not_supported");
        } else {
            bad_request("unsupported user content block: " + type, "messages",
                        "content_block_not_supported");
        }
    }
    return result;
}

ChatTurn parse_assistant_blocks(const Json& content) {
    ChatTurn assistant;
    assistant.role = ChatRole::Assistant;
    for (std::size_t index = 0; index < content.size(); ++index) {
        const Json& block      = content[index];
        const std::string type = require_block_type(block, "messages");
        if (type == "text") {
            assistant.content.push_back(
                text_part(require_string(block, "text", "messages", "text block"),
                          cache_boundary(block, "messages")));
        } else if (type == "thinking") {
            if (cache_boundary(block, "messages")) {
                bad_request("cache_control is not valid on thinking blocks", "messages",
                            "invalid_cache_control");
            }
            const std::string thinking =
                require_string(block, "thinking", "messages", "thinking block");
            // Visible Thinking owns prompt semantics and its signature stays outside the lowered
            // request. An empty block whose signature NInferno wrote for hidden Thinking (display
            // "omitted") restores that reasoning, so the prompt matches the visible form.
            if (thinking.empty() && block.contains("signature") &&
                block.at("signature").is_string()) {
                if (const std::optional<std::string> hidden = decode_hidden_thinking_signature(
                        block.at("signature").get<std::string>())) {
                    assistant.reasoning_content += *hidden;
                    continue;
                }
            }
            assistant.reasoning_content += thinking;
        } else if (type == "redacted_thinking") {
            if (cache_boundary(block, "messages")) {
                bad_request("cache_control is not valid on redacted_thinking blocks", "messages",
                            "invalid_cache_control");
            }
            // Claude-encrypted reasoning is intentionally opaque to the local model.
        } else if (type == "tool_use") {
            assistant.tool_calls.push_back(parse_tool_use(block));
            if (const auto ttl = cache_boundary(block, "messages")) {
                if (index + 1U == content.size()) {
                    assistant.cache_boundary_after = explicit_cache_boundary(*ttl);
                }
                // A valid non-terminal tool_use breakpoint cannot be represented by the Qwen
                // flattened assistant turn. It is advisory, so execution continues without it.
            }
        } else if (is_server_tool_block(type)) {
            bad_request("server tool content blocks require an executor that NInferno does not "
                        "provide",
                        "messages", "server_tools_not_supported");
        } else {
            bad_request("unsupported assistant content block: " + type, "messages",
                        "content_block_not_supported");
        }
    }
    return assistant;
}

ChatTurn parse_system_value(const Json& value, const char* param, std::size_t first_block = 0) {
    ChatTurn system;
    system.role = ChatRole::System;
    if (value.is_string()) {
        if (first_block != 0) { throw std::logic_error("string system value has a block offset"); }
        system.content.push_back(text_part(value.get<std::string>()));
        return system;
    }
    if (!value.is_array()) {
        bad_request("system content must be a string or an array of text blocks", param);
    }
    if (first_block > value.size()) { throw std::logic_error("system block offset is invalid"); }
    for (std::size_t index = first_block; index < value.size(); ++index) {
        const Json& block = value[index];
        if (require_block_type(block, param) != "text") {
            bad_request("only text system blocks are supported", param,
                        "content_block_not_supported");
        }
        system.content.push_back(
            text_part(require_string(block, "text", param, "system text block"),
                      cache_boundary(block, param)));
    }
    return system;
}

void parse_system(const Json& body, GenerationRequest& request) {
    if (!body.contains("system") || body.at("system").is_null()) { return; }
    const Json& value                             = body.at("system");
    constexpr std::string_view kAttributionPrefix = "x-anthropic-billing-header:";
    std::size_t first_block                       = 0;
    if (value.is_array() && !value.empty()) {
        const Json& first = value.front();
        if (first.is_object() && first.contains("type") && first.at("type").is_string() &&
            first.at("type").get_ref<const std::string&>() == "text" && first.contains("text") &&
            first.at("text").is_string() &&
            first.at("text").get_ref<const std::string&>().starts_with(kAttributionPrefix)) {
            first_block = 1;
        }
    }
    ChatTurn system = parse_system_value(value, "system", first_block);
    if (!system.content.empty()) { request.messages.push_back(std::move(system)); }
}

std::vector<ParsedMessage> normalize_same_role_messages(std::vector<ParsedMessage> messages) {
    std::vector<ParsedMessage> normalized;
    normalized.reserve(messages.size());
    for (ParsedMessage& message : messages) {
        if (normalized.empty() || message.role == ChatRole::System ||
            normalized.back().role != message.role) {
            normalized.push_back(std::move(message));
            continue;
        }
        ParsedMessage& target = normalized.back();
        if (message.role == ChatRole::User) {
            target.user_blocks.insert(target.user_blocks.end(),
                                      std::make_move_iterator(message.user_blocks.begin()),
                                      std::make_move_iterator(message.user_blocks.end()));
        } else {
            append_turn(target.turn, std::move(message.turn));
        }
    }
    return normalized;
}

std::size_t validate_user_result_prefix(const ParsedMessage& message) {
    bool saw_user_content    = false;
    std::size_t result_count = 0;
    std::unordered_set<std::string> result_ids;
    for (const ParsedUserBlock& block : message.user_blocks) {
        if (std::holds_alternative<ContentPart>(block)) {
            saw_user_content = true;
            continue;
        }
        const ParsedToolResult& result = std::get<ParsedToolResult>(block);
        if (saw_user_content) {
            invalid_tool_history(
                "tool_result blocks must precede all text and image blocks in a user message");
        }
        if (!result_ids.insert(result.tool_use_id).second) {
            invalid_tool_history("duplicate tool_result for tool_use id '" + result.tool_use_id +
                                 "'");
        }
        ++result_count;
    }
    return result_count;
}

void validate_assistant_tool_ids(const ParsedMessage& message) {
    std::unordered_set<std::string> ids;
    for (const ToolCall& call : message.turn.tool_calls) {
        if (!ids.insert(call.id).second) {
            invalid_tool_history("duplicate tool_use id '" + call.id +
                                 "' in one assistant message");
        }
    }
}

void normalize_tool_history(std::vector<ParsedMessage>& messages) {
    std::vector<std::size_t> result_counts(messages.size(), 0);
    std::vector<bool> paired_result_turn(messages.size(), false);
    for (std::size_t index = 0; index < messages.size(); ++index) {
        if (messages[index].role == ChatRole::User) {
            result_counts[index] = validate_user_result_prefix(messages[index]);
        } else if (messages[index].role == ChatRole::Assistant) {
            validate_assistant_tool_ids(messages[index]);
        }
    }

    for (std::size_t index = 0; index < messages.size(); ++index) {
        ParsedMessage& assistant = messages[index];
        if (assistant.role != ChatRole::Assistant || assistant.turn.tool_calls.empty()) {
            continue;
        }
        if (index + 1U >= messages.size() || messages[index + 1U].role != ChatRole::User) {
            invalid_tool_history("tool_result for visible tool_use id '" +
                                 assistant.turn.tool_calls.front().id +
                                 "' must immediately follow its assistant message");
        }

        ParsedMessage& user            = messages[index + 1U];
        const std::size_t result_count = result_counts[index + 1U];
        std::unordered_map<std::string, std::size_t> result_by_id;
        result_by_id.reserve(result_count);
        for (std::size_t result_index = 0; result_index < result_count; ++result_index) {
            const ParsedToolResult& result =
                std::get<ParsedToolResult>(user.user_blocks[result_index]);
            result_by_id.emplace(result.tool_use_id, result_index);
        }

        std::unordered_set<std::string> call_ids;
        call_ids.reserve(assistant.turn.tool_calls.size());
        for (const ToolCall& call : assistant.turn.tool_calls) { call_ids.insert(call.id); }
        for (std::size_t result_index = 0; result_index < result_count; ++result_index) {
            const ParsedToolResult& result =
                std::get<ParsedToolResult>(user.user_blocks[result_index]);
            if (!call_ids.contains(result.tool_use_id)) {
                invalid_tool_history("unknown tool_result id '" + result.tool_use_id +
                                     "' after visible assistant tool_use");
            }
        }
        for (const ToolCall& call : assistant.turn.tool_calls) {
            if (!result_by_id.contains(call.id)) {
                invalid_tool_history("missing tool_result for visible tool_use id '" + call.id +
                                     "'");
            }
        }

        std::vector<ParsedUserBlock> ordered;
        ordered.reserve(user.user_blocks.size());
        for (const ToolCall& call : assistant.turn.tool_calls) {
            ordered.push_back(std::move(user.user_blocks[result_by_id.at(call.id)]));
        }
        for (std::size_t block_index = result_count; block_index < user.user_blocks.size();
             ++block_index) {
            ordered.push_back(std::move(user.user_blocks[block_index]));
        }
        user.user_blocks               = std::move(ordered);
        paired_result_turn[index + 1U] = true;
    }

    for (std::size_t index = 0; index < messages.size(); ++index) {
        if (result_counts[index] == 0 || paired_result_turn[index]) { continue; }
        if (index == 0 && messages[index].role == ChatRole::User) { continue; }
        const ParsedToolResult& result =
            std::get<ParsedToolResult>(messages[index].user_blocks.front());
        invalid_tool_history("tool_result id '" + result.tool_use_id +
                             "' does not immediately follow a visible assistant tool_use");
    }
}

void lower_messages(std::vector<ParsedMessage> messages, GenerationRequest& request) {
    for (ParsedMessage& message : messages) {
        if (message.role != ChatRole::User) {
            merge_turn(request, std::move(message.turn));
            continue;
        }

        ChatTurn user;
        user.role = ChatRole::User;
        for (ParsedUserBlock& block : message.user_blocks) {
            if (std::holds_alternative<ContentPart>(block)) {
                user.content.push_back(std::move(std::get<ContentPart>(block)));
                continue;
            }
            ParsedToolResult& result = std::get<ParsedToolResult>(block);
            ChatTurn tool;
            tool.role                 = ChatRole::Tool;
            tool.tool_call_id         = std::move(result.tool_use_id);
            tool.content              = std::move(result.content);
            tool.tool_result_is_error = result.is_error;
            if (result.cache_boundary_after) {
                tool.cache_boundary_after = explicit_cache_boundary(*result.cache_boundary_after);
            }
            request.messages.push_back(std::move(tool));
        }
        if (!user.content.empty() || message.user_blocks.empty()) {
            merge_turn(request, std::move(user));
        }
    }
}

// System messages are accepted anywhere in `messages` (proxies such as LiteLLM send them first or
// between user turns): the first ones open the system prompt, and the frontend keeps later ones
// in place, or renders them as user turns for a template that refuses them. A run between an
// assistant's tool_use and its tool_result turn moves after that turn (LiteLLM moves it so too),
// since the results must follow their calls.
void place_system_messages(std::vector<ParsedMessage>& messages) {
    for (std::size_t index = 0; index < messages.size(); ++index) {
        if (messages[index].role != ChatRole::Assistant ||
            messages[index].turn.tool_calls.empty()) {
            continue;
        }
        std::size_t end = index + 1U;
        while (end < messages.size() && messages[end].role == ChatRole::System) { ++end; }
        if (end == index + 1U || end == messages.size() || messages[end].role != ChatRole::User) {
            continue;
        }
        const auto first = messages.begin() + static_cast<std::ptrdiff_t>(index + 1U);
        const auto user  = messages.begin() + static_cast<std::ptrdiff_t>(end);
        std::rotate(first, user, user + 1);
    }
}

void parse_messages(const Json& body, const RequestLimits& limits, GenerationRequest& request) {
    if (!body.contains("messages")) { bad_request("missing required field: messages", "messages"); }
    const Json& messages = body.at("messages");
    if (!messages.is_array() || messages.empty()) {
        bad_request("messages must be a non-empty array", "messages");
    }
    check_message_count(messages.size(), limits, "messages");

    std::vector<ChatRole> roles;
    roles.reserve(messages.size());
    for (std::size_t index = 0; index < messages.size(); ++index) {
        const Json& item = messages[index];
        if (!item.is_object() || !item.contains("role") || !item.at("role").is_string()) {
            bad_request("each message must be an object with a string role", "messages");
        }
        const std::string role = item.at("role").get<std::string>();
        if (role == "user") {
            roles.push_back(ChatRole::User);
        } else if (role == "assistant") {
            roles.push_back(ChatRole::Assistant);
        } else if (role == "system") {
            roles.push_back(ChatRole::System);
        } else {
            bad_request("message role must be 'user', 'assistant', or 'system'", "messages",
                        "unsupported_role");
        }
        if (!item.contains("content") || item.at("content").is_null()) {
            bad_request("each message must contain content", "messages");
        }
    }

    std::vector<ParsedMessage> parsed;
    parsed.reserve(messages.size());
    for (std::size_t index = 0; index < messages.size(); ++index) {
        const Json& content = messages[index].at("content");
        const ChatRole role = roles[index];
        ParsedMessage message;
        message.role      = role;
        message.turn.role = role;
        if (role == ChatRole::System) {
            message.turn = parse_system_value(content, "messages");
            parsed.push_back(std::move(message));
            continue;
        }
        if (content.is_string()) {
            if (role == ChatRole::Assistant) {
                message.turn.content.push_back(text_part(content.get<std::string>()));
            } else {
                message.user_blocks.emplace_back(text_part(content.get<std::string>()));
            }
            parsed.push_back(std::move(message));
            continue;
        }
        if (!content.is_array()) {
            bad_request("message content must be a string or an array", "messages");
        }
        if (role == ChatRole::Assistant) {
            message.turn = parse_assistant_blocks(content);
        } else {
            message.user_blocks = parse_user_blocks(content);
        }
        parsed.push_back(std::move(message));
    }

    place_system_messages(parsed);
    parsed = normalize_same_role_messages(std::move(parsed));
    normalize_tool_history(parsed);
    lower_messages(std::move(parsed), request);

    if (!request.messages.empty() && request.messages.back().role == ChatRole::Assistant) {
        const ChatTurn& final = request.messages.back();
        if (final.content.empty() || !final.reasoning_content.empty() ||
            !final.tool_calls.empty() ||
            std::any_of(final.content.begin(), final.content.end(),
                        [](const ContentPart& part) { return part.kind != ContentKind::Text; })) {
            bad_request("a final assistant prefill must contain only text", "messages",
                        "assistant_prefill_not_supported");
        }
        request.continuation = ninferno::PromptContinuationMode::ContinueFinalAssistant;
    }
}

enum class ToolSelectionKind {
    Auto,
    None,
    Any,
    Named,
};

struct ToolSelection {
    ToolSelectionKind kind = ToolSelectionKind::Auto;
    std::string name;
    bool disable_parallel = false;
};

ToolSelection parse_tool_choice(const Json& body) {
    ToolSelection result;
    if (!body.contains("tool_choice") || body.at("tool_choice").is_null()) { return result; }
    const Json& choice = body.at("tool_choice");
    if (!choice.is_object() || !choice.contains("type") || !choice.at("type").is_string()) {
        bad_request("tool_choice must be an object with a string type", "tool_choice");
    }
    const std::string type = choice.at("type").get<std::string>();
    if (type == "auto") {
        result.kind = ToolSelectionKind::Auto;
    } else if (type == "none") {
        result.kind = ToolSelectionKind::None;
    } else if (type == "any") {
        result.kind = ToolSelectionKind::Any;
    } else if (type == "tool") {
        result.kind = ToolSelectionKind::Named;
        result.name = require_tool_name(choice, "tool_choice", "tool name");
    } else {
        bad_request("unsupported tool_choice type: " + ascii_preview(type), "tool_choice");
    }
    result.disable_parallel = optional_bool(choice, "disable_parallel_tool_use", false);
    return result;
}

enum class ToolSource {
    UserDefined,
    AnthropicProvided,
    Toolset,
};

struct ParsedTool {
    ToolDefinition definition;
    ToolSource source = ToolSource::UserDefined;
    std::string source_type;
    bool strict        = false;
    bool defer_loading = false;
    std::optional<std::vector<std::string>> allowed_callers;
};

std::vector<ParsedTool> parse_tool_definitions(const Json& body) {
    std::vector<ParsedTool> result;
    if (!body.contains("tools") || body.at("tools").is_null()) { return result; }
    if (!body.at("tools").is_array()) { bad_request("tools must be an array", "tools"); }
    std::unordered_set<std::string> names;
    for (const Json& item : body.at("tools")) {
        if (!item.is_object()) { bad_request("tools entries must be objects", "tools"); }
        ParsedTool parsed;
        if (item.contains("type") && !item.at("type").is_null()) {
            if (!item.at("type").is_string()) {
                bad_request("tool type must be a string or null", "tools");
            }
            parsed.source_type = item.at("type").get<std::string>();
            if (parsed.source_type == "custom") {
                parsed.source = ToolSource::UserDefined;
            } else if (parsed.source_type == "toolset") {
                parsed.source = ToolSource::Toolset;
            } else {
                parsed.source = ToolSource::AnthropicProvided;
            }
        }

        if (parsed.source == ToolSource::UserDefined) {
            parsed.definition.name = require_tool_name(item, "tools", "tool name");
            if (!names.insert(parsed.definition.name).second) {
                bad_request("duplicate tool name: " + parsed.definition.name, "tools");
            }
            if (item.contains("description") && !item.at("description").is_null()) {
                if (!item.at("description").is_string()) {
                    bad_request("tool description must be a string", "tools");
                }
                parsed.definition.description = item.at("description").get<std::string>();
            }
            if (!item.contains("input_schema") || !item.at("input_schema").is_object()) {
                bad_request("tool input_schema must be a JSON object", "tools");
            }
            parsed.definition.input_schema_json = item.at("input_schema").dump();
            if (item.contains("input_examples") && !item.at("input_examples").is_null()) {
                if (!item.at("input_examples").is_array()) {
                    bad_request("tool input_examples must be an array", "tools");
                }
                parsed.definition.input_examples_json = item.at("input_examples").dump();
            }
            if (const auto ttl = cache_boundary(item, "tools")) {
                parsed.definition.cache_boundary_after = explicit_cache_boundary(*ttl);
            }
            for (auto field = item.begin(); field != item.end(); ++field) {
                const std::string& key = field.key();
                if (field.value().is_null()) { continue; }
                if (key == "input_schema") {
                    parsed.definition.declared_fields.emplace_back("parameters");
                } else if (key == "name" || key == "description" || key == "strict" ||
                           key == "input_examples") {
                    parsed.definition.declared_fields.push_back(key);
                }
            }
        } else if (item.contains("name") && item.at("name").is_string()) {
            parsed.definition.name = item.at("name").get<std::string>();
        }

        if (item.contains("strict") && !item.at("strict").is_null()) {
            if (!item.at("strict").is_boolean()) {
                bad_request("tool strict must be a boolean", "tools");
            }
            parsed.strict = item.at("strict").get<bool>();
        }
        if (item.contains("defer_loading") && !item.at("defer_loading").is_null()) {
            if (!item.at("defer_loading").is_boolean()) {
                bad_request("tool defer_loading must be a boolean", "tools");
            }
            parsed.defer_loading = item.at("defer_loading").get<bool>();
        }
        if (item.contains("eager_input_streaming") && !item.at("eager_input_streaming").is_null() &&
            !item.at("eager_input_streaming").is_boolean()) {
            bad_request("tool eager_input_streaming must be a boolean", "tools");
        }
        if (item.contains("allowed_callers") && !item.at("allowed_callers").is_null()) {
            if (!item.at("allowed_callers").is_array()) {
                bad_request("tool allowed_callers must be an array", "tools");
            }
            parsed.allowed_callers.emplace();
            for (const Json& caller : item.at("allowed_callers")) {
                if (!caller.is_string()) {
                    bad_request("tool allowed_callers entries must be strings", "tools");
                }
                parsed.allowed_callers->push_back(caller.get<std::string>());
            }
        }
        result.push_back(std::move(parsed));
    }
    return result;
}

void lower_tools(const Json& body, GenerationRequest& request) {
    const ToolSelection selection       = parse_tool_choice(body);
    std::vector<ParsedTool> definitions = parse_tool_definitions(body);
    const auto named                    = [&](const ParsedTool& tool) {
        return tool.definition.name == selection.name;
    };

    // Named and any are enforced by a tool-call constraint (attach_tool_call_constraint).
    if (selection.kind == ToolSelectionKind::Named) {
        if (std::none_of(definitions.begin(), definitions.end(), named)) {
            bad_request("tool_choice references unknown tool: " + selection.name, "tool_choice");
        }
        request.tool_choice.requirement = ninferno::ToolCallRequirement::Named;
        request.tool_choice.named       = selection.name;
    }
    if (selection.kind == ToolSelectionKind::Any) {
        if (definitions.empty()) { bad_request("tool_choice requires tools", "tool_choice"); }
        request.tool_choice.requirement = ninferno::ToolCallRequirement::Required;
    }

    request.tool_choice.mode =
        selection.kind == ToolSelectionKind::None ? ToolChoiceMode::None : ToolChoiceMode::Auto;
    if (selection.kind == ToolSelectionKind::None) {
        // The declarations stay in the prompt (tool_choice none renders them without reading
        // calls back): exactly the tools an auto turn renders, so the prompts agree. Deferred
        // tools and tools the model may not call directly are not among them.
        for (ParsedTool& tool : definitions) {
            const bool direct =
                !tool.allowed_callers ||
                std::find(tool.allowed_callers->begin(), tool.allowed_callers->end(), "direct") !=
                    tool.allowed_callers->end();
            if (tool.source == ToolSource::UserDefined && !tool.defer_loading && direct) {
                tool.definition.strict = tool.strict;
                request.tools.push_back(std::move(tool.definition));
            }
        }
        return;
    }

    for (ParsedTool& tool : definitions) {
        if (tool.source == ToolSource::Toolset) {
            bad_request("Anthropic toolsets require a tool loader that NInferno does not provide",
                        "tools", "toolsets_not_supported");
        }
        if (tool.source == ToolSource::AnthropicProvided) {
            bad_request("Anthropic-provided tool type '" + ascii_preview(tool.source_type) +
                            "' requires its predefined prompt schema or server executor, which "
                            "NInferno does not provide",
                        "tools", "anthropic_tools_not_supported");
        }
        tool.definition.strict = tool.strict;
        if (tool.defer_loading) {
            bad_request("defer_loading=true requires a deferred tool loader that NInferno does not "
                        "provide",
                        "tools", "deferred_tools_not_supported");
        }
        if (tool.allowed_callers &&
            std::find(tool.allowed_callers->begin(), tool.allowed_callers->end(), "direct") ==
                tool.allowed_callers->end()) {
            bad_request(
                "tool allowed_callers excludes direct model calls, and NInferno provides no "
                "alternate caller",
                "tools", "tool_caller_not_supported");
        }
        request.tools.push_back(std::move(tool.definition));
    }
    if (selection.disable_parallel && !request.tools.empty()) {
        request.parallel_tool_calls = false;
    }
}

AnthropicThinkingDisplay parse_thinking(const Json& body, GenerationRequest& request) {
    if (!body.contains("thinking") || body.at("thinking").is_null()) {
        return AnthropicThinkingDisplay::Visible;
    }
    const Json& thinking = body.at("thinking");
    if (!thinking.is_object() || !thinking.contains("type") || !thinking.at("type").is_string()) {
        bad_request("thinking must be an object with a string type", "thinking");
    }
    const std::string type = thinking.at("type").get<std::string>();
    if (type == "disabled") {
        request.enable_thinking = false;
    } else if (type == "adaptive") {
        request.enable_thinking = true;
    } else if (type == "enabled") {
        request.enable_thinking         = true;
        const std::optional<int> budget = optional_int(thinking, "budget_tokens");
        if (!budget || *budget < 1024) {
            bad_request("thinking.budget_tokens must be an integer of at least 1024", "thinking");
        }
        // Unlike the Anthropic API, a budget at or above max_tokens is accepted: the output limit
        // is reached before the budget, so it never takes effect. Clients such as Qwen Code send
        // a fixed budget while shrinking max_tokens to the context window left.
        request.thinking_budget = static_cast<std::uint32_t>(*budget);
    } else {
        bad_request("thinking.type must be 'disabled', 'adaptive', or 'enabled'", "thinking");
    }

    if (thinking.contains("display") && !thinking.at("display").is_null()) {
        if (!thinking.at("display").is_string()) {
            bad_request("thinking.display must be 'summarized' or 'omitted'", "thinking");
        }
        const std::string display = thinking.at("display").get<std::string>();
        if (display != "summarized" && display != "omitted") {
            bad_request("thinking.display must be 'summarized' or 'omitted'", "thinking");
        }
        if (type == "disabled") {
            bad_request("thinking.display is valid only when thinking is adaptive or enabled",
                        "thinking");
        }
        if (display == "omitted") { return AnthropicThinkingDisplay::Omitted; }
    }
    return AnthropicThinkingDisplay::Visible;
}

void parse_effort(const Json& body, GenerationRequest& request, ParsePurpose purpose) {
    if (!body.contains("output_config") || body.at("output_config").is_null()) { return; }
    const Json& config = body.at("output_config");
    if (!config.is_object()) { bad_request("output_config must be an object", "output_config"); }
    if (purpose == ParsePurpose::Messages && config.contains("format") &&
        !config.at("format").is_null()) {
        request.output_constraint =
            parse_anthropic_output_format(config.at("format"), "output_config.format");
    }
    if (!config.contains("effort") || config.at("effort").is_null()) { return; }
    if (!config.at("effort").is_string()) {
        bad_request("output_config.effort must be a string", "output_config.effort");
    }
    const std::string value = config.at("effort").get<std::string>();
    const auto effort       = ninferno::parse_reasoning_effort(value);
    if (!effort) {
        bad_request("output_config.effort is not a recognized effort value",
                    "output_config.effort");
    }
    request.reasoning_effort = *effort;
}

void parse_generation_fields(const Json& body, GenerationRequest& request) {
    if (body.contains("stop_sequences") && !body.at("stop_sequences").is_null()) {
        append_stop_strings(body.at("stop_sequences"), "stop_sequences", false,
                            request.stop_strings);
    }

    request.sampling.temperature = optional_number(body, "temperature");
    request.sampling.top_p       = optional_number(body, "top_p");
    request.sampling.top_k       = optional_int(body, "top_k");
    // Anthropic's temperature range is [0,1], narrower than OpenAI's.
    constexpr SamplingRange kAnthropicTemperatureRange{0.0F, 1.0F};
    if (request.sampling.temperature &&
        !kAnthropicTemperatureRange.contains(*request.sampling.temperature)) {
        bad_request(runtime::range_message("temperature", kAnthropicTemperatureRange),
                    "temperature");
    }
    if (request.sampling.top_p && !kProbabilityRange.contains(*request.sampling.top_p)) {
        bad_request(runtime::range_message("top_p", kProbabilityRange), "top_p");
    }
    if (request.sampling.top_k &&
        (*request.sampling.top_k < 0 || *request.sampling.top_k > kMaximumTopK)) {
        bad_request(top_k_limit_message("top_k", "send"), "top_k");
    }
}

std::string parse_model(const Json& body) {
    if (!body.contains("model") || !body.at("model").is_string() ||
        body.at("model").get<std::string>().empty()) {
        bad_request("missing required field: model", "model");
    }
    return body.at("model").get<std::string>();
}

void apply_anthropic_prompt_cache_policy(const Json& body, GenerationRequest& request) {
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
    if (explicit_boundaries.size() > kMaximumExplicitPromptCacheMarkers) {
        bad_request("at most four block-level cache_control breakpoints are supported per request",
                    "cache_control", "too_many_cache_breakpoints");
    }

    const std::optional<CacheBoundary::Ttl> automatic_ttl = cache_boundary(body, "cache_control");
    if (!automatic_ttl) { return; }
    request.allow_engine_automatic_shared_prefixes = false;

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
    if (automatic_target == nullptr) { return; }

    if (*automatic_target) {
        if (automatic_target->value().ttl != *automatic_ttl) {
            bad_request("request-level cache_control and the final block-level cache_control "
                        "target the same boundary with different ttl values",
                        "cache_control", "conflicting_cache_ttl");
        }
        automatic_target->value().evidence |= ninferno::SharedCandidateEvidence::RequestedAutomatic;
        return;
    }
    if (explicit_boundaries.size() == kMaximumExplicitPromptCacheMarkers) {
        bad_request("request-level cache_control needs a fifth distinct cache write after four "
                    "block-level breakpoints",
                    "cache_control", "too_many_cache_breakpoints");
    }
    *automatic_target =
        CacheBoundary{.kind     = ninferno::PromptCacheMarkerKind::SharedStablePrefix,
                      .evidence = ninferno::SharedCandidateEvidence::RequestedAutomatic,
                      .ttl      = *automatic_ttl};
}

AnthropicThinkingDisplay parse_common_prompt(const Json& body, const RequestLimits& limits,
                                             GenerationRequest& request, ParsePurpose purpose) {
    lower_tools(body, request);
    parse_system(body, request);
    parse_messages(body, limits, request);
    const AnthropicThinkingDisplay display = parse_thinking(body, request);
    parse_effort(body, request, purpose);
    attach_tool_call_constraint(request);
    apply_anthropic_prompt_cache_policy(body, request);
    if (body.contains("container") && !body.at("container").is_null()) {
        bad_request("container requires an external execution environment that NInferno does not "
                    "provide",
                    "container", "container_not_supported");
    }
    if (body.contains("chat_template_kwargs") && !body.at("chat_template_kwargs").is_null()) {
        if (!body.at("chat_template_kwargs").is_object())
            bad_request("chat_template_kwargs must be an object", "chat_template_kwargs");
        request.chat_template_kwargs_json = body.at("chat_template_kwargs").dump();
    }
    if (body.contains("preserve_thinking") && !body.at("preserve_thinking").is_null()) {
        if (!body.at("preserve_thinking").is_boolean()) {
            bad_request("preserve_thinking must be a boolean", "preserve_thinking");
        }
        request.preserve_thinking = body.at("preserve_thinking").get<bool>();
    }
    return display;
}

} // namespace

AnthropicMessagesRequest parse_anthropic_messages_request(const Json& body,
                                                          const RequestLimits& limits) {
    require_object(body);
    AnthropicMessagesRequest result;
    result.model                           = parse_model(body);
    result.generation.tool_name_max_length = kMaximumToolNameLength;
    result.stream                          = optional_bool(body, "stream", false);

    const std::optional<int> max_tokens = optional_int(body, "max_tokens");
    if (max_tokens) {
        result.output_tokens_explicit = true;
        if (*max_tokens == 0) {
            bad_request("max_tokens=0 requires a completed cache prewarm lifecycle that NInferno "
                        "does not provide",
                        "max_tokens", "cache_prewarm_not_supported");
        }
        if (*max_tokens < 0) { bad_request("max_tokens must be positive", "max_tokens"); }
        result.generation.max_tokens = *max_tokens;
    } else {
        result.generation.max_tokens = limits.default_max_tokens;
    }

    result.thinking_display =
        parse_common_prompt(body, limits, result.generation, ParsePurpose::Messages);
    parse_generation_fields(body, result.generation);
    result.generation.draft_tokens = parse_ninferno_extension(body).draft_tokens;
    // What the parser reads, plus standard fields with no local effect (metadata, service_tier).
    static constexpr std::string_view kKnown[] = {"cache_control", "chat_template_kwargs",
                                                  "container",     "max_tokens",
                                                  "messages",      "metadata",
                                                  "model",         "ninferno",
                                                  "output_config", "preserve_thinking",
                                                  "service_tier",  "stop_sequences",
                                                  "stream",        "system",
                                                  "temperature",   "thinking",
                                                  "tool_choice",   "tools",
                                                  "top_k",         "top_p"};
    note_unknown_fields(body, kKnown, result.generation);
    return result;
}

AnthropicCountTokensRequest parse_anthropic_count_tokens_request(const Json& body,
                                                                 const RequestLimits& limits) {
    require_object(body);
    AnthropicCountTokensRequest result;
    result.model                           = parse_model(body);
    result.generation.tool_name_max_length = kMaximumToolNameLength;
    parse_common_prompt(body, limits, result.generation, ParsePurpose::CountTokens);
    return result;
}

} // namespace ninferno::serve
