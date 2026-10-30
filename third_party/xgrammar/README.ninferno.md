# Vendored XGrammar

NInferno vendors the C++ core of XGrammar for constrained decoding (structured output):

- upstream: <https://github.com/mlc-ai/xgrammar>
- tag: `v0.2.7`
- commit: `82505d0d987c36a4209fb3d8571cf6b0f28b5acd`
- license: Apache-2.0; see `LICENSE` and `NOTICE`

Bundled third-party headers, unchanged:

- `3rdparty/picojson/picojson.h`: picojson (as carried by XGrammar v0.2.7), BSD 2-Clause; the
  license text is at the top of the header.
- `3rdparty/dlpack/include/dlpack/dlpack.h`: DLPack commit
  `bbd2f4d32427e548797929af08cfe2a9cbb3cf12` (the XGrammar v0.2.7 submodule pin), Apache-2.0;
  see `3rdparty/dlpack/LICENSE`.

The committed `include/` and `cpp/` files (including `cpp/support/`) are upstream files, unchanged
but for two local changes (below).
Python bindings (`python/`, `cpp/tvm_ffi/`), the web and Swift packages, documentation, tests,
googletest and cpptrace are intentionally omitted. The local `CMakeLists.txt` builds the same
static library source set as upstream's `xgrammar` target, without network access at configure
time.

The only integration point is upstream's own `XGRAMMAR_LOG_CUSTOMIZE` hook: the NInferno-owned
`ninferno/log_hook.{h,cc}` defines `LogFatalImpl` and `LogMessageImpl`. Fatal diagnostics still
throw `xgrammar::LogFatalError`; warnings raised while a request grammar is compiled are collected
and turned into a request error, because every such warning in the JSON Schema, regex and EBNF
converters reports a constraint that XGrammar would otherwise weaken silently.

Local change: `cpp/json_schema_converter.cc`, `JSONSchemaConverter::GetPartialRuleForProperties`,
one block marked "NInferno local change". Upstream compiles the properties of an object in schema
order with each optional one skippable, so once a model writes a later optional key, no earlier
optional key can be written any more: the output stays valid JSON and fields silently vanish. The
block lets an object's optional properties and additional keys come in any order, an optional
property at most once (tracked up to 256 rule states per object, none with patternProperties; past
that, the remaining optional properties may repeat), while its required properties keep their
schema order. An object with minProperties/maxProperties counts its entries in the state when all
its optional properties are tracked and it admits no other key (a repeatable key would count twice
for one property); any other counted object keeps upstream's rules, as does an object with
only required properties and no other keys. `any_order` (non-strict Qwen XML tool arguments) is
untouched. When updating, re-apply the block and run `ninferno_structured_output_test` and
`ninferno_chat_test` (their key-order contracts fail without it).

Local change: `JSONSchemaToGrammarWithConverter` (declared in `cpp/json_schema_converter.h`, defined
after `JSONSchemaToGrammar` in `cpp/json_schema_converter.cc`, both marked "NInferno local change")
parses a schema as `JSONSchemaToGrammar` does and converts it with a converter the caller makes.
The NInferno-owned `ninferno/tool_syntax_grammars.{h,cc}` uses it for the tool-call argument formats
XGrammar has no style for (Gemma 4, Pythonic, ATEM), with converters derived from
`JSONSchemaConverter` and `XMLToolCallingConverter`; they rely on the converter's protected hooks.
When updating, re-apply the function and run `ninferno_chat_test` and, with a tokenizer,
`ninferno_tool_grammar_budget_test`.

Local change: `cpp/support/container.h`, `List::Erase`, marked "NInferno (xgrammar #1003)": the
erased value is reset before its slot returns to the free list, so a token mask evicted from the
rule-level cache frees its memory then (upstream kept it alive until the slot was reused, past the
cache's byte budget). Upstream fixed it after v0.2.8 (mlc-ai/xgrammar#1003); drop the line when
updating past that fix.

To update: replace `include/`, `cpp/`, the two bundled headers and the license files from the new
tag, keep this file and `CMakeLists.txt`, and re-run the structured-output unit tests
(`ninferno_structured_output_test`).
