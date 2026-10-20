#pragma once

#include "chat/conventions.h"
#include "ninferno/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::chat {

// Tool calls in model output, one parser per ToolCallFormat behind one decoder.
//
// Qwen's tool syntax carries each argument as untyped text. This terminal contract records only
// the supported top-level JSON Schema types needed to normalize that text. A type mismatch remains
// a structured call for consumer validation; recursive validation is outside this non-strict
// contract.
struct ToolCallOutputContract {
    enum class SchemaType : std::uint8_t {
        Null    = 1U << 0U,
        Boolean = 1U << 1U,
        Integer = 1U << 2U,
        Number  = 1U << 3U,
        String  = 1U << 4U,
        Object  = 1U << 5U,
        Array   = 1U << 6U,
    };

    struct TypeSet {
        std::uint8_t bits = 0;
    };

    enum class NormalizationPolicy : std::uint8_t {
        Legacy,
        DeclaredTypes,
    };

    struct Parameter {
        std::string name;
        NormalizationPolicy policy = NormalizationPolicy::Legacy;
        TypeSet types;
    };

    struct Tool {
        std::string name;
        std::vector<Parameter> parameters;
        bool unambiguous = true;
    };

    std::vector<Tool> tools;
    bool enforce_declared_names = false;
    ToolCallFormat format       = ToolCallFormat::QwenXml;

    [[nodiscard]] bool declares(std::string_view name) const noexcept;
};

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics diagnostics;
};

[[nodiscard]] std::shared_ptr<const ToolCallOutputContract>
build_tool_call_output_contract(std::span<const std::string> tool_jsons, bool enabled,
                                ToolCallFormat format = ToolCallFormat::QwenXml);

// Whether held output bytes begin a tool-call region: not at all, possibly (a prefix of one), or
// for certain (the region starts at the held bytes).
enum class ToolCallStart : std::uint8_t { No, Partial, Complete };

// One tool-call language: the marker that opens its region of the output, the parser of that
// region and the XGrammar structural tag that constrains a model to it.
class ToolCallSyntax {
public:
    virtual ~ToolCallSyntax() = default;
    [[nodiscard]] virtual std::string_view open() const noexcept = 0;
    // A marker the model may write right before the open marker (whitespace between them
    // allowed) that belongs to the call, not to the content: Llama 3.1's <|python_tag|>.
    // Empty for syntaxes that have none.
    [[nodiscard]] virtual std::string_view lead() const noexcept { return {}; }

    // The marker that ends one call; empty for a syntax that has none (then a call the output
    // limit cut off cannot be told from a malformed one, and ToolCallOutputDecoder::finish keeps
    // its all-or-nothing reading).
    [[nodiscard]] virtual std::string_view close() const noexcept { return {}; }
    // Whether `held` (bytes the streaming decoder has not published) begins this syntax's call
    // region. The default: whitespace, an optional lead() and whitespace, then open(). A syntax
    // whose region opens with a header that names the function overrides it.
    [[nodiscard]] virtual ToolCallStart match_start(std::string_view held) const;
    // Text before the first open marker is content; the region from it must be calls only.
    [[nodiscard]] virtual ParsedToolCallOutput
    parse(const std::string& text, std::size_t max_tool_name_length,
          const ToolCallOutputContract& contract) const = 0;
    [[nodiscard]] virtual std::string
    structural_tag(std::span<const ToolCallConstraintTool> tools, ToolCallRequirement requirement,
                   std::string_view named_tool, bool parallel_tool_calls) const = 0;
};

// Throws std::invalid_argument for ToolCallFormat::None.
[[nodiscard]] const ToolCallSyntax& tool_call_syntax(ToolCallFormat format);

// A tool name's bytes, in a request and in generated calls alike: ASCII letters, digits, and
// `_ - . : /`, so MCP and namespaced names ("github.create_issue", "fs:read") are names. None of
// them is markup in a supported call syntax.
inline constexpr std::string_view kToolNameBytes = "[A-Za-z0-9_.:/-]";
[[nodiscard]] bool valid_tool_name(std::string_view name, std::size_t max_length) noexcept;

[[nodiscard]] ParsedToolCallOutput
parse_qwen_tool_call_output(const std::string& text, std::size_t max_tool_name_length,
                            const ToolCallOutputContract& contract);

// XGrammar structural tag for Qwen XML tool calls, accepting exactly the language
// parse_qwen_tool_call_output turns into structured calls: optional text without a tool-call
// marker, then calls separated by one newline and nothing after them. A strict tool's arguments
// follow its JSON Schema (already normalized by the structured-output schema policy); other tools'
// arguments are free text. Named emits only the named tool's call.
[[nodiscard]] std::string
build_tool_call_structural_tag(std::span<const ToolCallConstraintTool> tools,
                               ToolCallRequirement requirement, std::string_view named_tool,
                               bool parallel_tool_calls);

// Incrementally publishes bytes that are provably outside a possible terminal tool-call suffix
// (in the contract's format). At terminal time, valid calls are retained structurally; malformed output is restored
// verbatim.
class ToolCallOutputDecoder {
public:
    struct Terminal {
        std::string content;
        std::vector<GeneratedToolCall> tool_calls;
        ToolCallParseDiagnostics diagnostics;
    };

    ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                          std::size_t max_tool_name_length);

    [[nodiscard]] std::string feed(std::string_view text);
    // `cut_off`: the output limit ended the output. When the call region does not parse, the
    // complete calls before its last open marker are then the turn's calls, and that last,
    // unfinished call's markup is dropped (diagnostics.cut_calls_dropped).
    [[nodiscard]] Terminal finish(bool cut_off = false);

private:
    std::shared_ptr<const ToolCallOutputContract> contract_;
    const ToolCallSyntax* syntax_ = nullptr;  // the contract format's syntax
    // Bytes that may still begin the call region (ToolCallSyntax::match_start).
    std::string held_;
    std::string tool_region_;
    std::size_t max_tool_name_length_ = 0;
    bool saw_tool_marker_             = false;
    bool finished_                    = false;
};

} // namespace ninferno::chat
