# NInferno tools

`tools/` holds the Python and shell tooling around the engine: the Python weight converter and
artifact inspector, benchmark orchestration, serving smoke clients, drafter-registry helpers,
documentation generators and repository audits. Start with the [project README](../README.md) to
build and run NInferno. Models come only from converting original Hugging Face checkpoints; the
usual route is the native `ninferno-convert` (or letting `ninferno-serve org/name` convert
automatically), described in the [weight conversion guide](../docs/weight-conversion.md). The Python
tools here are independent of CMake and run from the repository root with a Python 3.11 environment
that has the dependencies of the chosen tool.

## Quick start

```bash
# Convert a Hugging Face checkpoint and inspect the artifact
ninferno-convert --fetch org/name --out models/qwen3_6_27b.ninferno
python3 -m tools.artifact.inspect models/qwen3_6_27b.ninferno --objects

# Smoke-test a running server
python3 -m tools.smoke.serve_contract --base-url http://127.0.0.1:8080 --model qwen3.6-27b
```

## Task index

| Task | Tool |
|---|---|
| Convert weights with a custom or official Python recipe | `python3 -m tools.convert` ([below](#python-converter)); [guide](../docs/weight-conversion.md) |
| Inspect artifact metadata and objects | `python3 -m tools.artifact.inspect` |
| Run benchmark matrices and serving benchmarks | [`bench/`](bench/README.md) |
| Measure external Serve TTFT | [`bench/ttft/`](bench/ttft/README.md) |
| Exercise a running server | [`smoke/`](#serving-smoke-checks) |
| Compare an MoE decoder (gpt-oss, Qwen3-MoE) with Hugging Face transformers | [`moe_quality/`](#moe-quality-comparison) |
| Measure the physical device-memory bandwidth ceiling | [`hbm_bandwidth_probe.cu`](#hbm-bandwidth-probe) |
| Generate route and R8 decode tables from benchmark sweeps | [`route_as_table.py`, `moe_tune_to_table.py`, `route_table_merge.py`](#maintainer-tools), [`bench/r8_decode_table.py`](bench/README.md#r8-decode-table) |
| Maintain the drafter registry and speculation corpus | [`spec/`](#maintainer-tools) |
| Generate or check documentation | [`docs/`](#maintainer-tools) |
| Audit the source tree | [`audit/`, `check_file_modes.sh`, `specializations.sh`, `pdl_sass_audit.py`](#maintainer-tools) |
| Build and publish a release (archives, installer, WSL root file system, update manifest) | `release/`: `package.sh`, `install.sh`, `ninferno-update`, `ninferno-start`, `pdfium.sh`, `manifest.py`, `recommend.py`, `windows/` (`build.sh`, `package-native.ps1`, `make_rootfs.sh`), `server/` ([README](release/server/README.md)) |
| Talk to a running server from Python | `ninferno_serve/` (public-protocol clients), `streaming_http/` (timestamped HTTP and SSE) |
| Regenerate the dense RoPE test fixture from transformers | `convert/dense_rope_fixture.py <fixture.json> <name> <config.json>` |

## Python converter

`python3 -m tools.convert` converts explicitly selected local sources into a v3 `.ninferno`
artifact plus a `.conversion.json` report (it needs PyTorch and NumPy). Use it for custom recipes,
overrides or mixed sources; recipes, sources, custom methods, resources and sharding are described in the
[conversion guide](../docs/weight-conversion.md), and numeric formats and framing in the references
linked from the [documentation map](../docs/README.md).

| Option | Default | Meaning |
|---|---|---|
| `--model DIR` | required | Primary checkpoint or config; default resources come from it. |
| `--recipe R` | required | Official recipe name, or `file.py[:function]` (a file defaults to `configure`). |
| `--override FILE` | none | Python configuration applied after the recipe and proposal. |
| `--source NAME=PATH` | none | Named extra source such as `quantized`, `dflash`, `dflash2`; repeatable. |
| `--components LIST` | `text` | Comma-separated subset of `text,vision,mtp,dflash,dflash2`; no repeats. |
| `--resource ROLE=PATH` | none | Override a final frontend resource such as `chat_template.jinja`; repeatable. |
| `--proposal` | off | Include the indexed proposal head. |
| `--proposal-rows N` | 131072 | Rows of the proposal head. |
| `--ranking PATH` | built-in ranking | Token ranking for the proposal head. |
| `--name NAME` | none | Public instance name stored in the metadata. |
| `--out PATH` | required | Output `.ninferno`. |
| `--device D` | `cuda` | Device for quantization. |
| `--rows-per-chunk N` | 512 | Rows processed per chunk. |
| `--max-file-bytes N` | 32000000000 | Shard size limit. |

Official recipes used by the maintained Qwen models (the full list is in
[`convert/official_recipes.py`](convert/official_recipes.py)):

| Recipe | Typical `--components` | Extra source |
|---|---|---|
| `qwen3_6_27b` | `text,vision,mtp` | none |
| `qwen3_8_27b` | `text,vision,mtp,dflash2` | `dflash2=<DFlash2 checkpoint>` |
| `qwen3_6_35b_a3b` | `text,vision,mtp,dflash` | `dflash=<DFlash checkpoint>` |

```bash
python3 -m tools.convert --model /path/to/Qwen3.8-27B \
  --recipe qwen3_8_27b --components text,vision,mtp,dflash2 --proposal \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --name qwen3.8-27b --out out/qwen3_8_27b.ninferno
```

`tools/chat_templates/` carries the maintained `qwen3_6.jinja` and `qwen3_8.jinja` templates.

### Artifact inspector

```bash
python3 -m tools.artifact.inspect out/qwen3_6_27b.ninferno --objects
```

| Option | Meaning |
|---|---|
| `artifact` (positional) | `.ninferno` file. |
| `--objects` | Include physical object records. |
| `--bindings` | Include logical bindings and uses. |
| `--json` | Emit one JSON object. |

It needs no numerical libraries.

## Serving smoke checks

Clients for a running `ninferno-serve` (OpenAI Chat and Responses, Anthropic, streaming, usage,
multimodal, tool calls). They are not CTest tests because they need a real artifact, a supported GPU
and a live server.

```bash
build/apps/ninferno-serve out/qwen3_6_27b.ninferno --host 127.0.0.1 --port 18080   # terminal 1
python3 -m tools.smoke.serve_contract --base-url http://127.0.0.1:18080 --model qwen3.6-27b
```

| Script | Options | Purpose |
|---|---|---|
| `serve_contract.py` | `--model` (required), `--base-url` (`http://127.0.0.1:18080`), `--health-timeout` (300 s) | Product HTTP contract with the standard library. |
| `serve_conformance.py` | `--model` (required), `--base-url` (same default) | Paired API conformance: stream equals non-stream, SSE event shapes, tool-choice modes, structured output, for OpenAI chat and Anthropic messages. |
| `serve_sdk_conformance.py` | `--model` (required), `--base-url` (server root, no `/v1`), `--any-version` | The official OpenAI and Anthropic Python SDKs against the server (`tools/smoke/sdk_requirements.txt`); `--any-version` runs other SDK versions and records instead of failing. |
| `serve_image_cache_key.py` | `--base-url`, `--model` (both required) | Image cache-key probe against a `--vision` server: a different image of the same size must not reuse another image's blocks. Images are generated locally. |
| `serve_soak.py` | `soak` or `chaos`; `--model`, `--out` (required), `--base-url` (`http://127.0.0.1:18931`), `--pid`, `--seconds` (900), `--workers` (12), `--seed` (1), `--canary-seconds` (180), `--sample-seconds` (10), `--repeats` (40), `--saturation` (64), `--storms` (`disconnect,prefill,saturation`), `--iterations` (1), `--shared-requests` (500), `--pending-rounds` (5), `--request-log` | Mixed-traffic soak with periodic greedy canaries, and chaos storms (disconnects, long prefill, saturation, pending queue). `--pid` lets it sample the server process; `--request-log` is the server's `--request-log-jsonl` file. |
| `serve_thinking_preservation.py` | `--artifact` (required), `--backend mtp\|dflash` (required), `--server-bin` (`build/apps/ninferno-serve`), `--fixture` (`tests/fixtures/serve/qwen3_6_thinking_preservation.json`), `--startup-timeout` (600), `--port` (0 = free port) | Starts its own server and checks typed rewrite-checkpoint and thinking-history behavior. |

```bash
python3 tools/smoke/serve_thinking_preservation.py --artifact out/qwen3_6_27b.ninferno --backend mtp
```

## MoE quality comparison

`moe_quality/hf_reference.py` computes a Hugging Face transformers reference (CPU, FP32 by default)
for a checkpoint and token files; `moe_quality/compare.py` compares it with the engine's output:
delta NLL, top-1 agreement and greedy divergence.

```bash
python3 tools/moe_quality/hf_reference.py <checkpoint dir> <token dir> <tag> <out dir> \
  [--dtype fp32|bf16] [--decode N] [--nvfp4-dequant [--a4 MODE[,MODE...]]]
python3 tools/moe_quality/compare.py <ref dir> <engine dir> <tag> [names...]
```

Token files are `<tag>_<name>.i32`; `<tag>_chat` is a generation prompt, the others are scored. The
reference writes `<tag>_<name>.ref` (FP32 log-probabilities, I32 argmax, FP32 top-1 minus top-2
gap) and a greedy continuation for the chat prompt; the engine side is produced with
`ninferno-dense <artifact> score --tokens <tag>_<name>.i32 --window 2048 --dump ...` and
`ninferno-dense <artifact> generate --tokens <tag>_chat.i32 --decode 64 --out <tag>_chat.gen`.

## HBM bandwidth probe

`hbm_bandwidth_probe.cu` measures sustained device-memory read, write and copy bandwidth; the "bus
GB/s" column counts logical traffic (N bytes for read or write, 2N for copy) and is comparable to the
advertised figure. It is a standalone CUDA program, independent of the CMake benchmark targets:

```bash
mkdir -p build
nvcc -O3 -std=c++17 -arch=native tools/hbm_bandwidth_probe.cu -o build/hbm_bandwidth_probe
./build/hbm_bandwidth_probe
```

| Option | Default | Meaning |
|---|---|---|
| `--size-gib N` | min(4, 20% of free memory) | Bytes per large buffer (at least 4x L2). |
| `--seconds N` | 0.25 | Target duration of each timed trial. |
| `--trials N` | 5 | Timed trials per method. |
| `--peak-gbps N` | memory clock x bus width | Advertised bandwidth for the percentage column. |

## Maintainer tools

| Tool | Purpose |
|---|---|
| `route_as_table.py --gpu-class C --pass DIR [--pass DIR ...] --out routes/C.json` | Builds `route_as` entries for a GPU class from `ninferno_linear_route_sweep_bench` CSVs measured with `--random-weight --no-class-table`. Options: `--gpu-name`, `--min-gain` (5), `--min-us` (2.1), `--tie` (3), `--min-t` (2), `--exclude-format` (repeatable), `--merge FILE`, `--report FILE`. |
| `moe_tune_to_table.py --cache moe_tune.json --table routes/C.json [--class C] [--gpu-name NAME] [--tuned-on MODEL] [--replace]` | Writes the startup MoE tuner's cached plans (`ninferno-moe-tune/2`) as the decode bands of the table's `moe` entries, so a release runs them without timing at startup. |
| `route_table_merge.py --into routes/C.json FILE [FILE ...] [--replace]` | Merges route tables of one GPU class (per-model `ninferno_route_tune` outputs, `moe_tune_to_table.py` output): keys the table lacks are added, the ones it has stay unless `--replace`. |
| `spec/propose_candidates.py REPO [--endpoint URL] [--limit N] [--json]` | Proposes drafter candidates for a registry entry from the Hugging Face API (metadata only, nothing downloaded). Not part of the user path: the product attaches only drafters a model bundle declares or the official registry names. |
| `spec/registry_add.py MODEL_ID FILE [--registry PATH] [--requires a,b] [--fallback] [--load] [--create] [--restamp]` | Puts a `spec-tune --publish` result (or, with `--load`, a load table) into the drafter registry. |
| `spec/tokenize_corpus.py --tokenizer DIR [--corpus PATH] [--stops 1,106,50] [--label TEXT] [--check]` | Adds a model family's rendered prompt ids to the built-in speculation corpus (keyed by vocabulary fingerprint); `--check` only prints the fingerprint. Needs `transformers`. |
| `text/tokenizer_fixtures.py --root DIR [--download NAME=REPO ...] [--synthetic]` | Writes reference tokenizer fixtures (`fixture.json`) from Hugging Face `tokenizers` for `tests/text`; `HF_TOKEN` is used when set. |
| `generate_jinja_unicode.py` | Regenerates Jinja's Unicode full-case exceptions and contextual-case properties from `SpecialCasing.txt` and `DerivedCoreProperties.txt`. |
| `docs/gen_performance.py [--check]` | Generates the [performance pages](../docs/performance/README.md) from `docs/performance/data/`; `--check` fails when a page differs from the data. |
| `docs/check_anchors.py ROOT` | Fails when a relative Markdown link names a missing file, or an `#anchor` no heading of its target (GitHub slugs); links in code are skipped. |
| `audit/ctest_skips.sh BUILD_DIR` | Counts cases skipped inside passing CTest tests, by reason and by test. |
| `audit/legacy_stream_calls.sh SOURCE_ROOT` | Audits synchronous CUDA copies and fills on the legacy default stream (registered as `ninferno_legacy_stream_audit_test`). |
| `check_file_modes.sh` | Fails when a tracked non-script file is committed executable (exit 77 outside a git checkout). |
| `specializations.sh [axis]` | Lists `NINFERNO-SPECIAL(axis:value)` tags (chip, arch, mem, model, format, workaround) with file, line and reason. |
| `pdl_sass_audit.py --build BUILD --source SOURCE [--allow FILE] [--list]` | SASS audit of programmatic dependent launch: global loads a consumer kernel issues before its wait. `--objects OBJ... --allow FILE` audits given objects. |
