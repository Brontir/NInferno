#pragma once

// Wire parsing for constrained decoding. Every accepted form is enforced by the Engine; every
// unsupported form is a 400 naming its parameter.

#include "serve/request.h"
#include "serve/request_json.h"

#include <optional>

namespace ninferno::serve {

// OpenAI response format object: Chat Completions response_format (json_schema nested under
// "json_schema") or Responses text.format (flat_json_schema: name/schema/strict at top level).
// {"type":"text"} returns nullopt.
[[nodiscard]] std::optional<OutputConstraintRequest>
parse_openai_response_format(const RequestJson& format, const char* param, bool flat_json_schema);

// Anthropic Messages output_config.format: {"type":"json_schema","schema":{...}}.
[[nodiscard]] OutputConstraintRequest parse_anthropic_output_format(const RequestJson& format,
                                                                    const char* param);

// vLLM structured_outputs and its legacy guided_* spellings, and llama.cpp's top-level grammar.
// Returns nullopt when none is present; rejects more than one source.
[[nodiscard]] std::optional<OutputConstraintRequest>
parse_structured_output_extensions(const RequestJson& body);

// Call after tools, tool choice, parallel_tool_calls and any response format are parsed. With
// callable tools, a response format is rejected (it would forbid every call). A strict tool, a
// required or named call, or disabled parallel calls attach a ToolCalls constraint on "tools".
void attach_tool_call_constraint(GenerationRequest& request);

} // namespace ninferno::serve
