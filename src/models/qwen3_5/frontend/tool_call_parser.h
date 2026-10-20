#pragma once

// The tool-call parsers are shared by every family (src/chat/tool_calls.h); Qwen's XML syntax is
// one of them. These names keep the Qwen frontend's spelling.
#include "chat/tool_calls.h"

namespace ninferno::models::qwen3_5::frontend {

using chat::build_tool_call_output_contract;
using chat::build_tool_call_structural_tag;
using chat::parse_qwen_tool_call_output;
using chat::ParsedToolCallOutput;
using chat::ToolCallOutputContract;
using chat::ToolCallOutputDecoder;

} // namespace ninferno::models::qwen3_5::frontend
