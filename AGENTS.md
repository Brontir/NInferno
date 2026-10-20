# AGENTS.md

Guide for AI coding agents working in this repository. Human contributors: see
[`CONTRIBUTING.md`](CONTRIBUTING.md). These rules apply to the whole tree.

## Working rules

- Finish the requested deliverable within the product and external contracts. A necessary redesign
  is in scope; unrelated cleanup, hardening and benchmark campaigns are not.
- Analysis or a proposal does not authorize applying it. Implement only what the request covers.
- Commit only when asked. Use Conventional Commit subjects (`feat`, `fix`, `perf`, `bench`, `test`,
  `build`, `refactor`, `docs`, `chore`).
- Report favorable and unfavorable findings together: regressions, costs, failed or skipped checks,
  unverified claims. Never narrow the scope or change the acceptance criteria after seeing
  unfavorable results. State the baseline, workload, GPU and method behind any performance claim.
- Keep stable requirements in their existing page. Do not add parallel `final`/`v2`/`new` documents
  or permanent planning files.

## Product

NInferno is a C++/CUDA inference engine for one NVIDIA GPU, Volta (`sm_70`) to Blackwell
(`sm_120`). It runs Qwen3.5/3.6/3.8 (dense and MoE), Gemma 4, gpt-oss, Nemotron-H and other dense
and MoE decoders from `.ninferno` artifacts converted from original checkpoints, with speculative
decoding (MTP, DFlash, DFlash2, Gemma 4 assistant), prefix reuse, and OpenAI/Anthropic-compatible
HTTP serving. Capabilities and commands: [`README.md`](README.md), [`docs/cli.md`](docs/cli.md),
[`docs/serving.md`](docs/serving.md), and each executable's `--help` (the exact source for option
spelling).

Fixed limits: one GPU, one resident model per process, startup-fixed concurrency of 1 to 8, a
bounded FIFO queue; an active request is preempted only by the KV lease (`--kv-lease-window`: the
request admitted last leaves its lane and resumes from its text). New math architectures,
platforms, multi-GPU or other preemptive batching need an explicit product decision.

## Layout

| Path | Contents |
|---|---|
| `include/ninferno/` | public C++ interface (`engine.h`, `types.h`, `score_detail.h`, `ops/`) |
| `src/core`, `src/artifact` | physical primitives and raw transfers; generic artifact framing |
| `src/ops` | closed mathematical and state-transition implementations, route tables |
| `src/models` | architectures: `qwen3_5`, `gemma4`, `dense`, `nemotron_h`, speculation |
| `src/runtime`, `src/spec` | execution contracts and Engine; speculation profiles, drafter registry |
| `src/serve`, `src/product`, `src/media` | HTTP protocols, product input handling, media decoding |
| `src/convert` | native converter and hub fetching |
| `apps/` | `ninferno` (CLI), `ninferno-serve`, `ninferno-ui`, `ninferno-convert`, `ninferno-perplexity`, `ninferno-dispatch`, `ninferno-dense`, `ninferno-gemma4`, `ninferno-gemma4-inspect` |
| `routes/` | per-GPU-class route tables measured on real cards |
| `tools/` | Python: conversion recipes, artifact inspection, smoke checks, docs generators |
| `tests/`, `bench/`, `eval/` | CTest and Python suites; benchmarks; capability evaluation |
| `docs/` | user guides, `maintainer/` references, generated `performance/` pages |

[`docs/README.md`](docs/README.md) is the documentation map. Read the reference that owns a boundary
before changing it, starting with
[`docs/maintainer/engine-architecture.md`](docs/maintainer/engine-architecture.md).

## Architecture invariants

- The v3 `.ninferno` artifact is the only C++ model artifact. The CLI, the server and inference
  benchmarks use the public Engine. There is no Python inference route and no installed C++ SDK.
- Core owns physical primitives; artifact owns framing and materialization; Ops own closed
  mathematical implementations; Models own fixed mathematics, config interpretation, parameter
  binding and execution composition; Programs own mutable state, workspace and CUDA Graphs and share
  nothing; the converter chooses formats, and the loader validates, uploads and binds them as
  stored.
- GPU memory is planned once at startup. No device allocation or weight repacking happens while
  requests run.
- Runtime owns execution contracts; serving and product code own input acquisition and protocol
  translation. Model code does not acquire media or own transport.
- Prefer explicit implementations over generic model graphs, family base classes, plugin discovery,
  string-driven execution or placeholders for hypothetical targets.
- Defaults should be the measured optimum for the model and GPU; a flag exists to turn a default
  off, not to ask the user to tune.
- Project-owned APIs, CLIs, tools, fixtures and formats keep no backward compatibility: when
  replacing a behavior, remove the old aliases, fallbacks and their tests. The advertised OpenAI and
  Anthropic behavior is an external contract: change schema tests and
  [`docs/serving.md`](docs/serving.md) together.

## Build

Requirements are in the [README](README.md#build-it-yourself) (CUDA 12.8, 13.2 or 13.4, CMake 3.28+,
Ninja, C++20, FFmpeg, libcurl). Select the GPU with `CMAKE_CUDA_ARCHITECTURES` (default `120a`; the
others are listed at the top of `CMakeLists.txt` and in [building](docs/building.md)).

```bash
cmake --preset dev -DCMAKE_CUDA_ARCHITECTURES=89   # products + tests + benchmarks
cmake --build --preset dev -j
```

`--preset release` builds products only. Options: `NINFERNO_BUILD_APPS`, `BUILD_TESTING`,
`NINFERNO_BUILD_BENCHMARKS` ([build system](docs/maintainer/build-system.md)). Lower the `-j`
level only if memory pressure causes failures. Use Python 3.11 for the Python tools.

## Tests

```bash
ctest --preset dev                      # all CTest suites
ctest --test-dir build -R <name> --output-on-failure
python3 -m pytest tests/artifact tests/convert tests/test_serve_corpus.py
```

Tests needing a converted model skip unless `NINFERNO_TEST_ARTIFACT` is set; run GPU tests
serially. Skipped cases still pass: `tools/audit/ctest_skips.sh build` counts them. Commands for
opt-in integration tests, the serving smoke checks and the evaluation suite are in
[`tests/README.md`](tests/README.md).

Choose evidence to match the change:

| Change | Typical evidence |
|---|---|
| Documentation | links (`python3 tools/docs/check_anchors.py .`), `git diff --check` |
| C++ runtime or API | affected build targets and behavioral tests |
| Python tooling | `py_compile` and the affected pytest suites |
| Artifact, binding, conversion | contract tests; a real artifact when semantics require it |
| CUDA mathematics | independent oracle at real model shapes and route boundaries |
| Memory or lifetime | run the affected path; sanitizer for a concrete lifetime question |
| Performance | measurement at the claimed scope; profile only to resolve attribution |
| Serving | schema tests and observable request/stream behavior |

Numerical rules (details in [Op development](docs/maintainer/op-development.md)):

- Floating-point Ops are checked against a naive FP32/FP64 oracle; exact transforms and codecs
  against an exact oracle; packed inputs are decoded independently with their stored scales.
- Qualify the production route against the oracle, not against another kernel's output.
- An Op microbenchmark proves an Op result, not an end-to-end gain.
- Tests protect observable behavior, math or state semantics, and realistic regressions. Do not add
  tests that mirror implementation or freeze private file organization.

## Style

- Format C++/CUDA with the repository `.clang-format` (100 columns, 4-space indent). The Codex hook
  `.codex/hooks/clang_format.py` formats edited files; otherwise run `clang-format -i`.
- Follow the surrounding code; do not introduce a new style. English only. LF line endings.
- Only scripts that start with `#!` may be committed executable
  (`tools/check_file_modes.sh` enforces it).

## Documentation and generated files

- Update the page that owns a behavior in the same change.
- `docs/performance/` pages and the table in the top-level `README.md` between `perf:begin` and
  `perf:end` are generated from `docs/performance/data/`. Edit the data, then run
  `python3 tools/docs/gen_performance.py`; verify with `--check`.
- Select model artifacts by explicit path, never by glob order or modification time. Download or
  regenerate large checkpoints only when the task requires it.
