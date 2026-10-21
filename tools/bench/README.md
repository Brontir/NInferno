# tools/bench

Python orchestration for performance measurement: the `ninferno_bench` run matrix, the serving
corpus and concurrency runners, the external Serve TTFT client and its campaign controller, the
benchmark corpus baker and the R8 decode-table generator. Use these tools to produce comparable
reports; the executables they drive are documented in [bench/README.md](../../bench/README.md),
recorded results in [docs/performance](../../docs/performance/README.md), and workload definitions,
metric boundaries and publication format in the
[serving methodology](../../docs/performance/methodology.md). Correctness lives in the test suites
([tests/README.md](../../tests/README.md)).

All commands run from the repository root. Install the corpus-baker dependencies with
`pip install -r tools/bench/requirements.txt` (`transformers>=4.40`, `tokenizers>=0.19`); the other
tools use only the standard library. Reports are written below `profiles/bench/` (not tracked).

## Quick start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNINFERNO_BUILD_BENCHMARKS=ON
python3 tools/bench/run_ninferno_bench_matrix.py --preset core --dry-run   # show commands
python3 tools/bench/run_ninferno_bench_matrix.py --preset core             # run the matrix
```

| Tool | Purpose |
|---|---|
| `run_ninferno_bench_matrix.py` | Layered `ninferno_bench` matrix (prefill, decode, MTP sweep). |
| `run_serve_corpus.py` | Serial fixed-corpus run against `ninferno-serve`. |
| `run_serve_concurrency.py` | Decode saturation and corpus makespan at several concurrencies. |
| `run_serve_ttft_campaign.py`, `run_serve_ttft.py`, `summarize_serve_ttft.py` | External Serve TTFT benchmark ([ttft/README.md](ttft/README.md)). |
| `make_bench_corpus.py` | Bake the token-id corpus used by `ninferno_bench`. |
| `build_text_examples.py` | Build the long-context and long-decode CLI examples ([examples/cli](../../examples/cli/README.md)) from recent public-domain text and new problems. |
| `r8_decode_table.py` | Generate `src/ops/generic/r8_decode_table.h` from R8 schedule sweeps. |
| `quality_rows.py` | Perplexity and KL divergence of each `--activations` mode against `exact` on the fixed corpus, appended as `docs/performance/` rows. |

## Published performance numbers

[`perf_matrix.py`](perf_matrix.py) measures one server setting exactly as the
[performance pages](../../docs/performance/README.md) do: six ~24K-token PG-19 prompts (prefill and
decode, medians) and a wave of twenty short chats at C=1, 4 and 8, greedy, after a warm-up request.
With `--jsonl` it appends rows to `docs/performance/data/bench.jsonl`; `tools/docs/gen_performance.py`
then rewrites the pages. The protocol and a full example are in the
[methodology](../../docs/performance/methodology.md#reproducing-a-row).

```bash
python3 tools/bench/perf_matrix.py --port 8080                       # print the numbers
python3 tools/bench/perf_matrix.py --port 8080 --skip-long --concurrency 4
```

## NInferno performance matrix

`run_ninferno_bench_matrix.py` builds (unless `--no-build`) and runs `ninferno_bench` over a layered
grid. MTP `k=3` with the optimized proposal head is the primary path; `k=0` and `k=5` are controls
and `k=0..5` is swept on representative context-decode cases. Decode-bearing cases cover CUDA Graph
and eager execution; prefill-only cases vary prompt length and chunk.

| Option | Default | Meaning |
|---|---|---|
| `--preset P` | `core` | `smoke` (three one-shot cases), `core`, or `full` (adds 32K/64K prompts and context-decode points). |
| `--bench PATH` | `build/bench/ninferno_bench` | Benchmark binary. |
| `--weights PATH` | `out/qwen3_6_27b.ninferno` | `.ninferno` artifact passed to the bench. |
| `--corpus PATH` | `bench/fixtures/bench_corpus.ids` | Token-id corpus; must cover the longest prompt of the selection. |
| `--output-dir DIR` | `profiles/bench/ninferno-<preset>-<UTC stamp>` | Report directory. |
| `--device N` | 0 | CUDA device. |
| `--suite NAME` | all | Run only this suite; repeatable. Suites: `prefill_length`, `prefill_chunk`, `pure_decode`, `context_decode`, `mtp_sweep`, `tail_stress` (not in `smoke`). |
| `--limit N` | none | Run only the first N selected cases. |
| `--repetitions N`, `--warmup N` | per case | Override every case. |
| `--dry-run` | off | Write `commands.sh` and the manifest without running. |
| `--resume` | off | Skip cases with a valid JSON report in `--output-dir`. |
| `--no-build` | off | Do not build `ninferno_bench` first. |

```text
profiles/bench/ninferno-<preset>-<timestamp>/
  commands.sh  manifest.json  summary.csv  summary.json
  json/<suite>/<case>.json
  logs/<suite>.<case>.stderr.txt
```

Each raw report must be an `ninferno_bench_report` of `schema_version` 15. The flattened summary and
the matrix manifest (`ninferno_bench_matrix_run`, schema 4) carry native facts from the reports:
architecture, public name, formats, prefill signature, artifact, load/staging values, Engine memory
arenas, planned and allocator-observed workspace peaks, KV capacity, proposal head and graph mode,
phase timings and throughput, and speculative rounds, drafts, acceptance and fallbacks. The manifest
is descriptive: it records commands and selected inputs and does not make repository state part of
report validity.

## Serving corpus benchmark

`run_serve_corpus.py` starts `ninferno-serve` for each artifact and speculative mode, replays the
fixed corpus serially and writes results. Workloads and metric boundaries are defined in the
[serving methodology](../../docs/performance/methodology.md).

```bash
python3 tools/bench/run_serve_corpus.py \
  --artifact qwen3_6_27b=models/qwen3_6_27b.ninferno --kv-dtype int8 \
  --mode mtp0 --mode mtp3 --output profiles/bench/serve-corpus
```

| Option | Default | Meaning |
|---|---|---|
| `--serve PATH` | `build/apps/ninferno-serve` | Server executable. |
| `--artifact LABEL=PATH` | required | Report label (also the server model alias) and artifact; repeatable. |
| `--mode M` | `mtp0` and `mtp3` | `mtp0` (no speculation), `mtp3`, `dflash7`, `dflash2_7`; repeatable. DFlash modes need the companion weights in the artifact. |
| `--sampling S` | `stochastic` | `stochastic` or `greedy` (exact argmax). The stochastic profile pins temperature, top-p, top-k, min-p and penalties explicitly. |
| `--kv-dtype T` | `int8` | `bf16`, `int8`, `fp8`, `nvfp4` or `k8v4`. The runner verifies the loaded KV representation and records it; a run cannot resume records collected with another dtype. State it explicitly for new campaigns. |
| `--output DIR` | required | Campaign directory. |
| `--port N` | 8080 | Loopback port. |
| `--device N` | 0 | CUDA device. |
| `--prefill-route F`, `--decode-route F`, `--moe-path P` | `auto` | Maintainer route pins passed to `ninferno-serve` ([route pins](../../docs/maintainer/environment.md#route-pins)), for an A/B of the automatic kernel family against one family. The runner checks them in the `server_start` record and binds a campaign directory to its pins (`route_pins.json`): measure each pin into its own `--output`. |

Output: `run.jsonl` (completed requests and responses), `summary.csv`, `summary.md` (fixture and
category summaries) and per-server logs under `server/`. Results (schema 8) retain the KV dtype,
the actual prefill signature, request Host exposure and decode Host/Device-wait time per round from
the serving records. Request exposure is a latency distribution value and is never summed across
concurrent requests.

## Concurrent serving benchmark

`run_serve_concurrency.py` runs the `decode-saturation` and `corpus-makespan` suites; each
concurrency point starts a fresh server. Their time boundaries and dispatch are defined in the
methodology (workloads). A point is not resumable: combining fragments from separate processes would
not preserve a steady interval or one continuous makespan.

```bash
python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=models/qwen3_6_27b.ninferno --kv-dtype int8 \
  --mode mtp3 --suite decode-saturation --concurrency 1 --concurrency 2 --concurrency 4 \
  --decode-tokens 8192 --output profiles/bench/concurrent-decode
```

| Option | Default | Meaning |
|---|---|---|
| `--serve`, `--artifact`, `--mode`, `--sampling`, `--kv-dtype`, `--port`, `--device`, `--output`, `--prefill-route`, `--decode-route`, `--moe-path` | as for the corpus runner | `--artifact` and `--output` are required. |
| `--suite S` | required | `decode-saturation` or `corpus-makespan`; repeat for both. |
| `--concurrency N` | required | Server `--max-concurrency` and client worker count, 1..8; repeat to sweep. |
| `--decode-tokens N` | 8192 | Per-request output budget for decode saturation. |
| `--max-context N` | 262144 | Server context. |
| `--kv-capacity N\|auto` | 262144 | Shared Main KV capacity; must be at least `--max-context`. Use `auto` when the corpus needs more than the default pool. |
| `--prefill-chunk N` | 1024 | Positive multiple of 128. |
| `--dry-run` | off | Print point commands and request counts only. |

Output (schema 4): `points/*.json` (actual Engine configuration, automatic KV capacity, shuffle
seed where applicable, dispatch method, per-request positions), `server/*.jsonl`, and combined
`summary.json`, `summary.csv`, `summary.md`. C=1 corpus runs also write complete responses and
per-request phase summaries under `corpus/<point>/`.

## External Serve TTFT

[ttft/README.md](ttft/README.md) defines the black-box latency benchmark: a client that sends public
HTTP requests to a running `ninferno-serve`, plus a controller that gives every sample a fresh
server process.

```bash
python3 tools/bench/run_serve_ttft_campaign.py --campaign resource --samples 5
```

## Corpus baker

`ninferno_bench` benchmarks prefill at an exact length by slicing the first `P` token ids of the
corpus, so it must be real in-distribution text (not random tokens) at least as long as the largest
prefill you run. `make_bench_corpus.py` bakes it offline with a local Hugging Face Qwen3.6
tokenizer, encoded without chat template or special tokens. Built-in multi-domain prose
(Chinese, English, code, math) is tiled with rotating paragraphs and truncated to exactly `--tokens`;
repetition only fills length, since prefill and decode throughput are token-count and bandwidth
bound. Pass `--source-text` for genuinely diverse long content.

| Option | Default | Meaning |
|---|---|---|
| `--tokenizer-path DIR` | `NINFERNO_TOKENIZER_PATH` | Local tokenizer directory; never downloaded. Required unless `--check`. |
| `--out PATH` | `bench/fixtures/bench_corpus.ids` | Whitespace-separated decimal token ids (exactly `--tokens`). |
| `--manifest PATH` | `--out` with suffix `.manifest.json` | Provenance: tokenizer id, token count, sha256, source. |
| `--tokens N` | 65536 | Exact corpus size and ceiling on prefill length. |
| `--source-text FILE` | built-in bank | Text to tokenize instead; repeatable. |
| `--check` | off | Verify `.ids` against the manifest hash and token count; needs no tokenizer. |

```bash
python3 tools/bench/make_bench_corpus.py --tokenizer-path /path/to/Qwen3.6-27B --tokens 65536
python3 tools/bench/make_bench_corpus.py --tokenizer-path /path/to/Qwen3.6-27B \
  --tokens 131072 --source-text book.txt
python3 tools/bench/make_bench_corpus.py --check
```

## R8 decode table

`r8_decode_table.py` writes `src/ops/generic/r8_decode_table.h` from repeated
`ninferno_r8_schedule_bench` sweeps. Each positional argument is `GPU=FILE[,FILE...]` where `GPU` is
the class id printed on the bench's `device` line (for example `gb202:170sm`) and every file is a
complete sweep (its `done` line). A problem gets an entry when the fastest schedule differs from the
shape-class rule, beats it by at least `--min-gain` in the median and in every sweep, and all
schedules produced identical bytes. Each table records the board its sweeps ran on (the bench's
`board` line: compute capability, SMs, L2, DRAM bandwidth; `--merge` keeps a header's), and a
device takes the table of the nearest board within a variant's distance of it (`board_distance`,
`kR8CarryDistance` in `generic_gemm.cpp`).

| Option | Default | Meaning |
|---|---|---|
| `--out FILE` | required | Header to write. |
| `--min-gain F` | 0.005 | Minimum relative gain over the rules' choice. |
| `--merge HEADER` | none | Keep this header's entries for problems the given sweeps did not measure (another GPU or format family). |

```bash
python3 tools/bench/r8_decode_table.py --out src/ops/generic/r8_decode_table.h \
  ad102:128sm=a1.txt,a2.txt,a3.txt gb202:170sm=b1.txt,b2.txt,b3.txt
```
