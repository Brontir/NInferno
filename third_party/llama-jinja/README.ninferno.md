# NInferno Jinja source base

This maintained implementation originates from [llama.cpp](https://github.com/ggml-org/llama.cpp),
commit [`7609846557c50f9d984719a9e1e8c5f3d02f807b`](https://github.com/ggml-org/llama.cpp/commit/7609846557c50f9d984719a9e1e8c5f3d02f807b),
`common/jinja/`, under the MIT license; see [LICENSE](LICENSE).

NInferno maintains this fork for chat-template rendering and adopts upstream fixes selectively.

Unicode case data is generated from Unicode 17.0.0 `SpecialCasing.txt` and
`DerivedCoreProperties.txt`, under [UNICODE-LICENSE](UNICODE-LICENSE):

```bash
python3 tools/generate_jinja_unicode.py /path/to/unicode-17.0.0
```

Adopted and local fixes, each pinned by `tests/text/test_jinja.cpp`:

- llama.cpp #20782 (GHSA-q9j6-4hhc-rq9p, GHSA-2q4c-9gq5-5vfp) and #20913 are part of the base.
- Local: `lstrip_blocks` no longer drops a template's first character when only blanks separate
  it from the first block tag (`lexer.cpp`; upstream master still has the off-by-one).
- Local: `raise_exception()` throws `template_raised_exception`, which `statement::execute` passes
  through untraced, and `src/text/jinja.cpp` hands its message on without the source name, so a
  template's refusal ("No user query found in messages.") reaches the client as written; genuine
  template errors keep the source name and the source-location trace.
