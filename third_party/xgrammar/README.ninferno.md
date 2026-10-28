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
but for one local change (below).
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
block lets an object's optional properties come in any order, each at most once (tracked up to
1024 rule states per object; past that, the remaining optional properties may repeat), while its
required properties keep their schema order. An object with only required properties, or with
minProperties/maxProperties, keeps upstream's rules. `any_order` (non-strict Qwen XML tool
arguments) is untouched. When updating, re-apply the block and run `ninferno_structured_output_test`
and `ninferno_chat_test` (their key-order contracts fail without it).

To update: replace `include/`, `cpp/`, the two bundled headers and the license files from the new
tag, keep this file and `CMakeLists.txt`, and re-run the structured-output unit tests
(`ninferno_structured_output_test`).
