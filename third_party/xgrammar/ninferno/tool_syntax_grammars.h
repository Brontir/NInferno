#pragma once

// NInferno-owned file (not upstream). Grammars for a strict tool's arguments in tool-call syntaxes
// that XGrammar has no JSON-schema style for. Each returns an EBNF grammar (root rule "root") that
// admits exactly the arguments the schema allows, written the way the syntax's parser in
// src/chat/tool_calls.cpp reads them, with the same object key-order rule as JSON (see
// README.ninferno.md). Whitespace runs between tokens are bounded by max_whitespace.
//
// A schema the syntax cannot carry throws std::invalid_argument; an invalid schema throws
// XGrammar's error.

#include <string>

namespace ninferno_xgrammar {

// Gemma 4: {key:value,...} with bare keys (or <|"|>-quoted ones), strings wrapped in <|"|>, and
// numbers, booleans and null bare. The grammar covers the object including its braces.
[[nodiscard]] std::string gemma4_arguments_ebnf(const std::string& schema, int max_whitespace);

// Pythonic (OLMo 3): key=value, ... inside NAME(...), every value JSON. Top-level names must be
// Python identifiers. The grammar covers the text between the parentheses.
[[nodiscard]] std::string pythonic_arguments_ebnf(const std::string& schema, int max_whitespace);

// ATEM: <atem:parameter name="K">V</atem:parameter> per argument, a string value as raw text and
// any other value as JSON. The grammar covers the parameters between the invoke tags.
[[nodiscard]] std::string atem_arguments_ebnf(const std::string& schema, int max_whitespace);

} // namespace ninferno_xgrammar
