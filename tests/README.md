# Tests

The retained tests protect `.ninferno` artifact framing and conversion, numerical Op contracts, model
frontends and Programs, runtime transactions, serving protocols and benchmark reports. They run
under CTest (C++ and Python) and are grouped by observable risk, not by mirroring source files.
Repository verification principles are in [AGENTS.md](../AGENTS.md); Op contract and CUDA guidance is
in [op-development.md](../docs/maintainer/op-development.md); performance checks belong to
[bench/README.md](../bench/README.md), not here.

## Quick start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`cmake --preset dev` enables products, tests and benchmarks together and `ctest --preset dev` runs
the suite (see [Build system](../docs/maintainer/build-system.md)). Run GPU integration tests
serially. A focused run:

```bash
cmake --build build --parallel --target ninferno_sampling_test
ctest --test-dir build -R ninferno_sampling_test --output-on-failure
```

The checks that need no GPU carry the label `cpu`: the selector's properties and digests over
substituted boards, its golden files, route classes, the source scan of the selection rules and the
digests' self-consistency. The `selection` workflow builds them for the full (120a) and base
(sm_89) route sets in a CUDA container and runs `ctest --test-dir build -L cpu`; the sm_70 and
sm_75 sets are built on their stands. `ninferno_selector_properties_test --only <board>` runs the
boards whose name contains `<board>`.

### Updating the selector goldens and digests

A change that means to move routes regenerates, for each of the four route sets, the golden files
(`ops/linear/selector_golden/<set>/`) and the digests (`ops/linear/selector_digest/<set>.txt`). The
set is the build's: `full` from `CMAKE_CUDA_ARCHITECTURES=120a` (or 121a / 120f), `base` from 89
(or 86, 80), `sm75` from 75 and `sm70` from 70. In each of the four trees, with the GPU hidden:

```bash
export CUDA_VISIBLE_DEVICES=
build/tests/ninferno_selector_golden_test --update
build/tests/ninferno_selector_properties_test --update
ctest --test-dir build -L cpu --output-on-failure   # must now pass
```

The 120a and 89 trees build on any CUDA host (a workstation or the CI container). The 75 and 70
trees are built on the RTX 2080 Ti and V100 stands (or any host whose nvcc targets them): their
gate job builds the cpu-labelled targets, runs the two `--update` commands and packs
`selector_golden/` and `selector_digest/` for the change to take. Check the result with
`python3 tools/audit/selector_digest_check.py .`, which verifies that every set's digests are its
golden files' hashes and that the four sets name the same boards. A property violation that the
change accepts goes into `ops/linear/selector_properties_known.txt` with its gap, route sets
(`@set`) and boards (`%board`, a `group`, `%known-pci` or `%unknown-pci`).

## Build options

| Option | Default | Meaning |
|---|---|---|
| `BUILD_TESTING` | `OFF` | Build and register the tests. |
| `Python3_EXECUTABLE` | CMake's Python 3 | Interpreter for interop and frontend tests (Python 3.11 is the maintained version; the chat-template reference test uses Jinja2). |
| `NINFERNO_CONVERT_PYTHON` | `Python3_EXECUTABLE` | Interpreter with `torch` (and `transformers` for the oracle) for the converter-parity and Hugging Face oracle tests; they skip without it. |
| `NINFERNO_SHARED_OP_TESTS` | `OFF` | Link the Op tests against one shared `ninferno_ops` instead of each statically linking it. For test builds only. |

## Layout

| Path | Covers |
|---|---|
| `artifact/` | v3 framing, directory and binding records, codecs, sharding, selected-object materialization, Python-writer / C++-reader interoperability, poisoned-arena upload. |
| `convert/` | Python converter: source interpretation, Qwen and Gemma 4 logical mapping, recipe overrides and sharing, optional components, resources, proposals, numerical methods. |
| `convert_native/` | `ninferno-convert` (native): GGUF and Hugging Face inputs against Python twins, quantization numerics, format and recipe detection, fetch, drafter attachment. |
| `models/qwen3_5/` | Config and binding, frontend, state and context stores, workspace, MTP alignment, opt-in real Engine integration. |
| `models/gemma4/`, `models/dense/` | Loading, speculation, exact-output, window and CLI checks of those families; most need real artifacts. |
| `models/speculation/` | Draft depth, DFlash split, n-gram lookup (no GPU). |
| `ops/` | Semantic Op qualification against independent mathematical or state-transition oracles; Linear, LinearAdd, LinearSwiGLU and LinearPair suites are split by weight and activation path. |
| `chat/`, `text/` | Chat templates and tokenizer/Jinja behavior, including parity with `transformers`. |
| `oracle/` | Differential check of synthetic checkpoints against Hugging Face in FP32. |
| root `test_*.cpp` | Core storage, admission and resource policy, public API, serving protocols (OpenAI, Anthropic, HTTP), logging, request log, benchmark support, causal scoring. |
| `test_serve_corpus.py`, `test_registry_add.py` | Serving request-log schema versus its measurement consumer; the drafter-registry tool. |

`CMakeLists.txt` includes manifests from `cmake/`, `artifact/`, `convert_native/`, `models/*/`,
`oracle/` and `ops/`; registration helpers live in `cmake/NInfernoTests.cmake` and executables and
CTest working directories stay under `build/tests/`. `ops/op_tester.h` and `ops/op_check.h` own only
reusable device, guard and comparison mechanics; numerical tolerances are named by each semantic Op
suite (no cross-Op presets). `ops/quantized_weight.h` is the shared packed-weight fixture
(Q4/Q5/Q6/Q8, FP8, NVFP4): deterministic payloads, device `Weight` views, row views and independent
logical decoding.

A test that cannot run one case on this build or GPU (for example NVFP4 below the `120a` build, FP8
MMA below `sm_89`) prints `SKIP <case>: <reason>` and still passes; a test skipped as a whole exits
77, which CTest reports as skipped. A full pass on an older card is therefore not full coverage:
`tools/audit/ctest_skips.sh build` counts the last run's skipped cases by reason and by test.

## Opt-in tests

Tests that need model artifacts or reference data skip (exit 77) when their variable is unset.
Artifacts are `.ninferno` files converted from Hugging Face checkpoints
(see the [conversion guide](../docs/weight-conversion.md)).

| Variable | Used by | Meaning |
|---|---|---|
| `NINFERNO_TEST_ARTIFACT` | `ninferno_qwen3_5_prefix_real_test`, `_score_real_test`, `_moe_real_test`, `_dflash_real_test`, `_dflash2_real_test`, `_dflash_prefill_real_test`, `_speculative_invariants_real_test`, `_determinism_real_test`, `_vision_workspace_real_test` | Artifact to run the real Engine against. |
| `NINFERNO_PREFIX_REAL_SCENARIO` | `_prefix_real_test` | Scenario; default `all`. Others: `attention`, `vision`, `concurrent`, `host-restore`, `last-alias-eviction`, `rewrite-branch`, `host-tier-wave`, `light-load-capture`, `anthropic-prefix-regression`, `shared-rewrite-materialization`, `pressure-resume`, `private-checkpoint-pressure`, `source-pressure-protection`, `shared-replacement`, `private-long-anchor`, `rewrite-checkpoint-shared`, `rewrite-checkpoint`, `stream-observations`. Needs an artifact with MTP. |
| `NINFERNO_TEST_KV_DTYPE` | `attention` scenario | `bf16` (default), `int8`, `fp8`, `nvfp4`, `k8v4`. |
| `NINFERNO_TEST_SPECULATIVE` | `attention` scenario | `none`, `mtp` (default), `dflash`, `dflash2`; the artifact must carry the component. |
| `NINFERNO_TEST_BATCH` | `attention` scenario | Concurrent rows; default 2. |
| `NINFERNO_TEST_DRAFT_TOKENS` | `attention` scenario | Draft window; default 3 for MTP, 7 otherwise. |
| `NINFERNO_TEST_RUN_AHEAD_ARTIFACTS` | `_ordinary_run_ahead_real_test` | Artifacts for the decode run-ahead equivalence test. |
| `NINFERNO_TEST_SPEC_WAIT_ARTIFACT` (+ `_BACKEND`, `_K`, `_DRAFTER`) | `_speculative_wait_real_test` | Artifact (DFlash K=3 unless overridden) for the blocking-versus-spinning round-end equivalence test. |
| `NINFERNO_TEST_GEMMA4_DIR` | `ninferno_gemma4_loading_real_test`, `_speculation_real_test`, `_prompt_lookup_real_test` | Directory holding `gemma4_26b_a4b_nvfp4.ninferno`, `gemma4_26b_assistant_q8_sl98304r2.ninferno`, `gemma4_26b_dflash_q8.ninferno` and `gemma4_31b_nvfp4.ninferno` (config only); the loading test checks that both drafters refuse the 31B target. |
| `NINFERNO_TEST_GEMMA4_EXACT_ARTIFACT`, `NINFERNO_TEST_NARROW_ARTIFACT`, `NINFERNO_TEST_NARROW_DRAFTER` | `ninferno_gemma4_engine_exact_real_test`, `_narrowed_chunk_real_test` | Artifacts of the exact-output and narrowed-chunk tests. |
| `NINFERNO_TEST_GEMMA4_EVERY` | `ninferno_gemma4_loading_real_test` | When set, also loads every `.ninferno` found in `NINFERNO_TEST_GEMMA4_DIR` (each target and each drafter against the target of its size), one at a time. |
| `NINFERNO_TEST_GEMMA4_TEXT` (+ `_TARGET`, `_MTP`, `_DFLASH`, `_PROFILE`, `_DIAG`) | `ninferno_gemma4_window_real_test` | Token ids of ordinary text (with `NINFERNO_TEST_GEMMA4_DIR`), the target and drafters, the activation profile; listed at the head of `tests/models/gemma4/test_window_real.cpp`. |
| `NINFERNO_TEST_GEMMA4_WINDOW_STRICT=1` | `ninferno_gemma4_window_real_test` | Assert the decode-against-verify checks of a known-open window discrepancy; without it they are reported (KNOWN-OPEN), not asserted. |
| `NINFERNO_TEST_GEMMA4_REF` | `ninferno_gemma4_*_score_real_test` | Reference scores the Gemma 4 E2B/E4B score tests compare against (with `NINFERNO_TEST_GEMMA4_DIR`). |
| `NINFERNO_TEST_TOKENIZER_FIXTURES` | `ninferno_tokenizer_real_test` | Directory `tools/text/tokenizer_fixtures.py` writes: real tokenizers and the Hugging Face library's ids. |
| `NINFERNO_TEST_DENSE_ARTIFACTS`, `_TOKENS`, `_EXACT_ARTIFACT`, `_REUSE_ARTIFACT`, `_LONGROPE_ARTIFACT`, `_LONGROPE_TOKENS`, `_DFLASH_ARTIFACT`, `_DFLASH_DRAFTER`, `_DFLASH_DEPTH` | `ninferno_dense_*_real_test` | Artifacts and token files of the dense-model real tests. |
| `NINFERNO_TOKENIZER_PARITY_DIR` | `ninferno_qwen3_5_tokenizer_parity_test` | Directory written by `hf_tokenizer_parity.py` with a checkpoint's tokenizer files and expected ids. |
| `NINFERNO_CHAT_TEMPLATE_PARITY_DIR`, `_ROOT` | chat-template parity tests | `_DIR`: the directory `tests/models/qwen3_5/hf_chat_template_parity.py` writes (Qwen 3.5 test); `_ROOT`: the one `tests/chat/hf_chat_template_parity.py` writes (every template). |
| `NINFERNO_DET_*` (`_PHASES`, `_KV`, `_SPEC`, `_K`, `_GRAPHS`, `_HEAD`, `_ACTIVATIONS`, `_DRAFTER`, `_DUMP`, `_COMPARE`, ...) | `ninferno_qwen3_5_determinism_real_test` | Phases, configuration matrix and references of the determinism contracts; each is listed at the head of `tests/models/qwen3_5/test_engine_determinism_real.cpp`. |
| `NINFERNO_TEST_FORMATS=iq3xxs,q6k,...` | generic Linear test | Run only the listed formats (quick check of what a change touches); unset runs all; fails if nothing matches. |
| `NINFERNO_GGUF_CASES=frag,kquant,...` | `synthetic_gguf.py` | Run only those case groups. |
| `NINFERNO_OP_REPORT_STATS=1` | Op suites | Print one `OP_ERROR_STATS` record per comparison (case label, actual error, active limit, ratio). Reporting only; the verdict is unchanged. |

Examples:

```bash
NINFERNO_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninferno \
  ctest --test-dir build -R ninferno_qwen3_5_score_real_test --output-on-failure

NINFERNO_TEST_ARTIFACT=$PWD/out/qwen3_6_27b.ninferno \
NINFERNO_PREFIX_REAL_SCENARIO=attention NINFERNO_TEST_KV_DTYPE=fp8 \
NINFERNO_TEST_SPECULATIVE=mtp NINFERNO_TEST_BATCH=2 \
  ./build/tests/ninferno_qwen3_5_prefix_real_test

NINFERNO_TEST_GEMMA4_DIR=/path/to/gemma4/artifacts \
  ctest --test-dir build -R ninferno_gemma4_loading_real_test --output-on-failure

NINFERNO_OP_REPORT_STATS=1 \
  ctest --test-dir build -V -R '^ninferno_(rmsnorm|softmax_attention)_test$'
```

The real Engine tests check behavior and state accounting, not another numerical path's generated
tokens as a golden. `ninferno_qwen3_5_score_real_test` checks a full 1,024-column score tile,
overlapping target suffixes and repeated-window State/KV isolation.

### Real loading test

`ninferno_qwen3_5_loading_real_test` takes explicit arguments instead of an environment variable:

| Option | Meaning |
|---|---|
| `--artifact PATH` | Artifact; without it the test skips. |
| `--vision` | Load the Vision component. |
| `--speculative B` | `none`, `mtp`, `dflash` or `dflash2`. |
| `--proposal H` | `full` or `optimized`. |
| `--scoring` | Causal-scoring purpose. |
| `--host-only` | Check semantic binding without uploading weights (does not build a Program). |

```bash
./build/tests/ninferno_qwen3_5_loading_real_test \
  --artifact out/qwen3_6_27b.ninferno --vision --speculative mtp --proposal optimized
```

### DFlash and DFlash2 Engine integration

`ninferno_qwen3_5_dflash_prefill_real_test [dflash|dflash2]` (default `dflash2`) checks KV contents
after a StateImage fork and a conflicting decode binding, including shortened chunks and oversized
local/full KV appends, with `NINFERNO_TEST_ARTIFACT` pointing at an artifact that carries the draft
component:

```bash
NINFERNO_TEST_ARTIFACT=out/qwen3_8_27b_nvfp4.ninferno \
  build/tests/ninferno_qwen3_5_dflash_prefill_real_test dflash2
```

`ninferno_qwen3_5_dflash2_real_test` takes positional arguments:

| Position | Default | Meaning |
|---|---|---|
| 1 `K` | 15 | Draft tokens. |
| 2 | 1 | CUDA Graph enabled. |
| 3 | 1 | Optimized head enabled. |
| 4 | 8 | Maximum batch `B`. |
| 5 | `bf16` | Target KV dtype (any of `bf16`, `int8`, `fp8`, `nvfp4`, `k8v4`). |
| 6 | 0 | Vision enabled. |
| 7 | 3 | Extra Device StateImage slots (0 exercises Host snapshot/restore). |

```bash
NINFERNO_TEST_ARTIFACT=out/qwen3_8_27b.ninferno \
  build/tests/ninferno_qwen3_5_dflash2_real_test 7 1 0 2 bf16 1 0
```

It checks output budgets, speculative activity, forced thinking-control append, penalty-enabled
sampling, compact batches with unequal budgets, same-route same-seed replay, retained and fresh
prefix behavior and the absence of a full backend KV pool. A fixture shared with DFlash starts
decode at token 63, verifies across the page boundary, stops after one target column at token 64 and
checks the retained frontier and subsequent generation with and without reuse. `K>=7` also
exercises a stop inside a licensed block; `K=15` adds oversized prefill, local ring wrap and the
logical capacity tail. Exact floating-point route parity is not asserted; the Op suites are the
numerical oracle.

## Numerical suites

**Softmax attention.** `ninferno_softmax_attention_test` runs under CTest as separate tests: the base
test (`--other-only`: plain, packed and context attention), one test per causal KV dtype
(`ninferno_softmax_attention_{bf16,int8,fp8,nvfp4,k8v4}_test`, `--kv-dtype T`) and
`ninferno_softmax_attention_unwritten_test` (`--unwritten-only`). Run the executable directly with
`--causal-only` (all KV types) or `--kv-dtype bf16|int8|fp8|nvfp4|k8v4|all`. The causal suite covers
prefill, decode and speculative widths, batched prefixes, cache effects, and Graph replay with
changing live lengths, with small, unit-RMS and RMS1.8 Q/K inputs. The FP64 oracle keeps internal Q
quantization error, which the INT8 and FP8 compute budgets account for. Large reference calculations
use at most eight CPU workers while preserving each output's FP64 accumulation order; the runner
prints elapsed time per KV type.

**Linear.** Suites are independent per weight and activation path:

```bash
cmake --build build --parallel --target \
  ninferno_linear_q4_a16_test ninferno_linear_q5_a16_test \
  ninferno_linear_q6_a16_test ninferno_linear_q8_a16_test
ctest --test-dir build -R '^ninferno_linear_(q4|q5|q6|q8)_a16_test$' --output-on-failure
```

All Linear files use `ops/linear/linear_test_common.{h,cpp}` and the shared weight fixture, which
produces the complete packed GPU payload and exact-decodes the logical rows for the single
`cpu_linear_gemm_fp64()` reference. That reference accumulates naively in double and never
reproduces a route's activation quantization, staging, reduction tree or BF16 rounding. Each
activation compute path has one centrally defined tolerance for its whole suite, and test files call
only public `linear()`. The Linear, LinearAdd and LinearSwiGLU common sources compile once into a
support library; libraries and Op test executables receive `-fno-fast-math` and
`-ffp-contract=off` on GNU/Clang.

## Python suites

```bash
python3 -m pytest tests/artifact tests/convert tests/test_serve_corpus.py
```

They exercise conversion and encoded output without running inference (Python 3.11 with the
suites' dependencies); C++ binding and Engine tests cover consumption of the result. The evaluation
coordinator has its own environment and tests ([eval/README.md](../eval/README.md#verification)).

## Serving smoke checks

These need a running server and a real artifact, so they are not CTest tests. Start a server, then
run the clients from [tools/README.md](../tools/README.md#serving-smoke-checks):

```bash
./build/apps/ninferno-serve out/qwen3_6_27b.ninferno --host 127.0.0.1 --port 18080
python3 -m tools.smoke.serve_contract --base-url http://127.0.0.1:18080 --model qwen3.6-27b
```

`serve_thinking_preservation.py` starts and stops its own server, submits a fixed two-step tool
history (messages in
[`fixtures/serve/qwen3_6_thinking_preservation.json`](fixtures/serve/qwen3_6_thinking_preservation.json)),
compares stripped and preserved closed-turn prompt lengths and verifies compatible prefix reuse,
speculative execution, frontier bounds and Responses inheritance:

```bash
python3 tools/smoke/serve_thinking_preservation.py --artifact out/qwen3_6_27b.ninferno --backend mtp
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_35b_a3b.ninferno --backend dflash
```

## What belongs here

A permanent test protects one current risk: exact artifact bytes, geometry, binding or conversion
transforms; a numerical Op contract with an independent oracle; model frontend or Program behavior
(frontier, prefix, MTP, multimodal); token commit, stop and cancel consistency; observable public
benchmark or OpenAI/Anthropic behavior; or a reproduced supported bug. Performance-only assertions
belong in benchmarks. Source scans, implementation-shape assertions, trivial getters, retired
command surfaces and broad additions without a concrete regression risk do not.
