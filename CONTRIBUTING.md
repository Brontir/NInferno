# Contributing to NInferno

Thank you for helping. The most useful contributions are precise bug reports, reproducible
performance evidence, and small, well-verified pull requests. NInferno is derived from
[ninfer](https://github.com/Neroued/ninfer) by Neroued (Apache-2.0; see [`NOTICE`](NOTICE)).

## Scope

NInferno is an inference engine for one NVIDIA GPU, from Volta (`sm_70`) to Blackwell (`sm_120`).
The supported product is described in [`README.md`](README.md); the documentation map is
[`docs/README.md`](docs/README.md). Maintainers decide scope, architecture and long-term
maintenance. A good idea may still be implemented by a maintainer inside the current architecture,
and the report will be credited.

## Start with an issue

Open an issue before writing code for a bug, a performance opportunity, a feature or a protocol
change. Wait for a maintainer to confirm the scope and direction; a pull request without a
confirmed issue may be closed unread. Typos, broken links and obvious one-line fixes need no issue.

**Bug report.** Include the NInferno commit, GPU model, driver and CUDA version, the source
checkpoint and conversion command, the full command or request that fails (smallest reproduction),
expected versus observed behavior, logs as text, and how often it happens.

**Performance report.** Say what level you measured (operator, request phase or end to end) and give
baseline and candidate under the same conditions: workload, commands, GPU, toolchain, warmup and
repetitions. Note costs such as memory, numerical quality and effects on other GPUs. An end-to-end
number does not isolate an operator change; an operator proposal needs an operator benchmark and a
correctness check.

## Build

Requirements are in the [README](README.md#build-it-yourself). Build for your GPU architecture:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

For development use the `dev` preset, which also builds tests and benchmarks (add
`-DCMAKE_CUDA_ARCHITECTURES=<arch>` when the default `120a` is not your GPU):

```bash
cmake --preset dev
cmake --build --preset dev -j
```

Build options are listed in [`docs/maintainer/build-system.md`](docs/maintainer/build-system.md).

## Test before you open a PR

```bash
ctest --preset dev                      # or: ctest --test-dir build --output-on-failure
python3 -m pytest tests/artifact tests/convert tests/test_serve_corpus.py
```

Use Python 3.11. Run GPU tests serially. A case your GPU or build cannot run prints `SKIP` and still
passes; `tools/audit/ctest_skips.sh build` counts skipped cases. Tests that need a converted model
are skipped unless `NINFERNO_TEST_ARTIFACT` points at one. For a local change, build and run just
the affected target (`ctest --test-dir build -R <name>`). Test layout and the opt-in model tests
are in [`tests/README.md`](tests/README.md); benchmarks in [`bench/README.md`](bench/README.md).

What to verify depends on the change:

- exact transforms and formats: exact comparison;
- floating-point kernels: an independent naive FP32/FP64 oracle, as in
  [Op development](docs/maintainer/op-development.md);
- stateful behavior: the whole affected transition;
- CUDA changes: the relevant shapes and routes, and say which GPU classes you tested;
- performance: measure at the level you claim;
- protocol changes: the OpenAI/Anthropic schema tests and [`docs/serving.md`](docs/serving.md).

Say plainly which checks you could not run. Do not claim validation you did not do.

## Code style

C++20 and CUDA, formatted with the repository's [`.clang-format`](.clang-format) (LLVM-based,
4-space indent, 100 columns, left-aligned pointers, `#include` order left as written):

```bash
clang-format -i path/to/changed_file.cpp
```

Follow the surrounding code. Python tools target Python 3.11 and run from the repository root.
Write comments and docs in English. Do not leave dead code, unused options
or compatibility aliases for names you replaced.

## Documentation

Documentation lives in [`docs/`](docs/README.md): user guides ([serving](docs/serving.md),
[CLI](docs/cli.md), [conversion](docs/weight-conversion.md)) and maintainer references under
[`docs/maintainer/`](docs/maintainer/). Update the page that owns a behavior in the same PR as the
change. CI checks documentation (`.github/workflows/docs.yml`); run both checks locally:

```bash
python3 tools/docs/gen_performance.py --check   # performance pages match their data
python3 tools/docs/check_anchors.py .           # relative links and their #anchors resolve
```

**Performance pages** are generated; do not edit them by hand. Measurements are lines in
[`docs/performance/data/bench.jsonl`](docs/performance/data/bench.jsonl), with GPUs and models in
`gpus.json` and `models.json` beside it. After changing the data, regenerate:

```bash
python3 tools/docs/gen_performance.py
```

This rewrites `docs/performance/README.md`, the per-GPU, per-model and per-pair
(`runs/<gpu>/<model>.md`) pages, and the blocks between the `perf-summary` and `perf` markers in
the top-level `README.md`. Commit data and pages
together. Measurement rules are in
[`docs/performance/methodology.md`](docs/performance/methodology.md).

## Pull requests

- Target the `master` branch and link the confirmed issue.
- One PR is one coherent change: implementation, tests, tools and docs that belong together.
  Keep unrelated fixes, refactors and optimizations in separate PRs, and avoid drive-by cleanup.
- Use Conventional Commit subjects: `feat`, `fix`, `perf`, `bench`, `test`, `build`, `refactor`,
  `docs` or `chore`, for example `fix: reject empty tool arguments`.
- Fill in the [PR template](.github/pull_request_template.md): problem and scope, design, and the
  exact verification commands with summarized results (GPU model, toolchain, workload for any
  performance claim).
- Run `git diff --check` and keep line endings LF.

| Change | Evidence to include |
|---|---|
| Bug fix | The trigger, expected behavior, and verification of the fix |
| Conversion recipe | Source, target format, command, and which stages you verified (convert, load, run) |
| New Op or shape | Supported inputs and an independent numerical check ([Op development](docs/maintainer/op-development.md)) |
| Linear tuning | Correctness evidence and a report in the [existing format](docs/maintainer/linear-tuning.md#4-report-format) |
| Docs, build, tooling | The checks relevant to the change: links, affected targets, tool runs |

## Your responsibility

You are responsible for every change you submit, whichever tools produced it, including AI
assistants. Understand the whole diff, be able to explain and defend the design, and answer review
comments yourself. Review time is limited: a PR may be closed without detailed review if it lacks a
confirmed issue, mixes independent changes, falls outside the supported product, or lacks evidence
for its claims. Closing a PR does not reject the underlying report.

## Security

Report vulnerabilities privately; see [`SECURITY.md`](SECURITY.md).
