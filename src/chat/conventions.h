#pragma once

#include <ninferno/types.h>

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninferno::chat {

// How one chat template marks its turns, its reasoning section and its tool calls. Every family
// shares one frontend; what differs between models is this record, read from the data table
// src/chat/data/conventions.json by the markers the template itself contains. Nothing here is
// selected by model name.

struct TurnMarkup {
    std::string name;
    std::string open;         // precedes the role name: "<|im_start|>"
    std::string header_end;   // ends the header line: "\n"
    std::string close;        // closes the turn: "<|im_end|>"
    std::string after_close;  // optional byte run the template writes after close: "\n"
    // The tokens that end a reply in this markup, as the family's generation config lists them:
    // the close when it ends the reply ("<|im_end|>"), or the reply's own terminator when the close
    // only ends a message within it (harmony closes its analysis message with "<|end|>" and the
    // reply with "<|return|>"), and the token that hands a call to the tool (Gemma 4's
    // "<|tool_response>", harmony's "<|call|>"), whether or not the template offers tools.
    std::vector<std::string> stops;
    std::vector<std::pair<std::string, ChatRole>> roles;  // header text -> role

    [[nodiscard]] std::optional<ChatRole> role(std::string_view header) const noexcept;
};

struct ReasoningMarkup {
    std::string name;
    std::string open;          // "<think>"
    std::string open_suffix;   // follows open in a canonical opener: "\n"
    std::string close;         // "</think>"
    // The bytes that close a reasoning section in the template's own serialization of history;
    // a prompt boundary after them can be resumed exactly. Empty when the template has none.
    std::string canonical_close;
    // Text that ends reasoning when a thinking budget runs out. Its tokens end with
    // canonical_close when that is set. Empty: this template supports no thinking budget.
    std::string budget_control;
    // Headers the model writes before its answer, after reasoning or in place of it (a harmony
    // "final" channel); the presentation drops the one the answer begins with. Empty: none.
    std::vector<std::string> content_open;
};

enum class ToolCallFormat : std::uint8_t {
    None,
    QwenXml,     // <tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>
    HermesJson,  // <tool_call>\n{"name": NAME, "arguments": {...}}\n</tool_call>
    Gemma4,      // <|tool_call>call:NAME{k:<|"|>v<|"|>,...}<tool_call|>
    MistralJson, // [TOOL_CALLS]NAME[ARGS]{...}
    Llama3Json,  // {"name": NAME, "parameters": {...}} as the reply itself (Llama 3.1+)
    Harmony,     // <|start|>assistant<|channel|>commentary to=functions.NAME ...<|message|>{...}
    Pythonic,    // <function_calls>NAME(key=value, ...)</function_calls> (OLMo 3)
    GraniteJson, // <|tool_call|>[{"name": NAME, "arguments": {...}}, ...] (Granite 3.x)
    Atem,        // [<|start|>assistant] to=NAME<|message|><atem:function_calls>
                 // <atem:invoke name="NAME"><atem:parameter name="K">V</atem:parameter>...
                 // </atem:invoke></atem:function_calls>, one message per call (Muse Glimmer)
};

struct ToolMarkup {
    ToolCallFormat format = ToolCallFormat::None;
    // A user-role message whose content is one tool result wrapped in these markers is not a
    // real user query (the template folds tool results into user turns).
    std::string response_open;
    std::string response_close;
    // The template ends each serialized tool result with this; the byte after it is a message
    // boundary. Empty: no such boundary.
    std::string response_boundary;
};

struct ChatConventions {
    std::optional<TurnMarkup> turns;
    std::optional<ReasoningMarkup> reasoning;
    ToolMarkup tools;

    // Control strings the output parsers read. Their tokens pass through the output presentation
    // even when the tokenizer marks them special.
    [[nodiscard]] std::vector<std::string> structural_markers() const;

    // The tokens that end a generated reply: the turn markup's stops. Empty when no turn markup
    // matched the template.
    [[nodiscard]] std::vector<std::string> generation_stops() const;

    // A token the template writes inside a reply, never at its end, which no configuration may
    // make a stop: the reasoning markers, and the turn close when it is not one of the turn's
    // stops (harmony's "<|end|>" between the analysis and the final message).
    [[nodiscard]] bool within_reply(std::string_view token) const;
};

// The output ended inside an answer header (ReasoningMarkup::content_open) written in markup, with
// "<|" or "=" in it: a token budget cut "<|start|>assistant to=user" (Muse Glimmer UD-IQ3_XXS at
// max_tokens 64) short of "<|message|>". No answer began; those bytes are not content. A plain-text
// prefix (" to") still is.
[[nodiscard]] bool is_cut_content_header(std::string_view text,
                                         std::span<const std::string> headers);

// A generated answer split by its template's reasoning markup: the output opens with
// reasoning.open (in place, or after leading spaces and newlines), the reasoning runs to
// reasoning.close and the answer follows it, without a header of reasoning.content_open it
// begins with. `closed` is false when the output never closed its
// reasoning (a budget ran out): everything after the opener is reasoning and the answer is empty.
// Output that does not open with the marker is all answer.
struct ReasoningSplit {
    std::string reasoning;
    std::string content;
    bool opened = false;
    bool closed = false;
};
[[nodiscard]] ReasoningSplit split_reasoning(std::string_view output,
                                             const ReasoningMarkup& markup);

// Resolves the conventions of a chat template source from the data table. Every category picks
// the first entry whose markers all occur in the source; a category without a match stays empty.
[[nodiscard]] ChatConventions resolve_conventions(std::string_view chat_template_source);

[[nodiscard]] std::string_view tool_call_format_name(ToolCallFormat format) noexcept;

// The end-of-generation tokens for a template the table does not know (its generic_stops: the
// texts llama.cpp stops on, llama_vocab::load's special_eog_ids).
[[nodiscard]] std::vector<std::string> generic_generation_stops();

} // namespace ninferno::chat
