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
| `hard_set.py` | One serving configuration through the hard question set ([eval/corpora/hard50](../../eval/corpora/hard50/README.md)): answers graded against the references, decode rate and, at a fixed speculative depth, tokens per round. |
| `selection_regret.py` | Offline regret of recorded production choices against qualified, matched alternatives on held-out boards or models ([below](#offline-selection-regret)). |
| `regret_bands.py` | Regret weighted by T band, per board, format and route family, from regret documents or plain route_tune CSVs ([below](#regret-by-t-band)). |
| `regret_collect.py` | The lab collector for `selection_regret.py`: `ninferno_route_tune --selected-out` (the route the production path runs per problem and width) and one `--routes-out` sweep per pass into its input ([below](#offline-selection-regret)). |
| `quality_rows.py` | Perplexity and KL divergence of each `--activations` mode against `exact` on the fixed corpus, appended as `docs/performance/` rows. |

## Offline selection regret

`selection_regret.py` evaluates **recorded production choices** against qualified alternatives on
lab data. It does not choose a route, fit a model, run measurements or feed runtime decisions.
Use it to evaluate a frozen selector on whole boards or model checkpoints withheld from fitting.
It complements `choice_check.py` (prediction error of one selected configuration): a good speed
prediction does not necessarily mean a good choice among alternatives.

```bash
python3 tools/bench/selection_regret.py profiles/bench/selection.json --max-regret 0.05 > profiles/bench/regret.json
python3 -m unittest discover -s tests -p test_selection_regret.py
```

Input is one `ninferno-selection-regret/1` JSON document, with a frozen `selector_revision`, a
`build_id` identifying the kernels/compiler/options/route tables, `holdout` (`board`, `model` or
`both`), explicit `fit_boards` and `fit_models` arrays, and `cases`. Each case supplies:

- `id`, `board`, `model`: stable case and board/checkpoint identities (a case `id` unique in the
  document; no outer whitespace in any identity). Group all shapes and quant
  variants of one checkpoint under the same model identity; aliases must be normalized upstream.
- `workload`: an unambiguous workload/protocol key including geometry, input/corpus, quant/layout,
  T, context/concurrency, numerical policy, memory budget and eager/graph/cache mode as applicable.
- `numerical_contract`: the common qualification criterion for this case's candidates. Compare
  alternatives satisfying the same required quality/state contract, not arbitrary precision modes.
- `selected`: the implementation/configuration actually resolved by the frozen production path,
  including any demotion. Record this before inspecting evaluation timings.
- `candidates`: the predeclared inventory. Each has a unique `id` and `status`: `ok` (admitted and
  passed independent correctness qualification), `oracle_failed` (wrong result), `ineligible`
  (outside build/device/policy/resource domain), or `unmeasured` (admitted but timing missing,
  interrupted or failed). Non-ok entries need a `reason` and must not supply `samples_ms`.
  Ok entries supply positive finite `samples_ms`, keyed by matched pass IDs.

For example, **synthetic arithmetic only**, not a GPU measurement:

```json
{
  "schema": "ninferno-selection-regret/1",
  "selector_revision": "synthetic-selector",
  "build_id": "synthetic-build",
  "holdout": "both",
  "fit_boards": ["training-board"],
  "fit_models": ["training-checkpoint"],
  "cases": [{
    "id": "case-1", "board": "held-out-board", "model": "held-out-checkpoint",
    "workload": "synthetic fixed workload", "numerical_contract": "same-qualified-contract",
    "selected": "a",
    "candidates": [
      {"id": "a", "status": "ok", "samples_ms": {"p1": 12, "p2": 12, "p3": 12}},
      {"id": "b", "status": "ok", "samples_ms": {"p1": 10, "p2": 10, "p3": 10}}
    ]
  }]
}
```

This reports regret 0.2: **20% extra time**, not 20% less throughput. Best observed means the
qualified candidate with lowest median time across matched passes. Per-pass ranges compare the
selected candidate with that one fixed winner, never a different hindsight winner each pass.
Summaries report equal-case-weight geometric mean slowdown, median, nearest-rank p95 and worst
regret, plus separate board/model summaries. They are descriptive, not confidence intervals or
uncertainty estimates. Keep raw pass timings; collect interleaved matched runs with the same
workload and report variation before drawing performance conclusions. Single-pass inputs are
accepted and reported as such (`passes` per case, `single_pass_cases` in each summary), not
treated as evidence of repeatability.

Any missing admitted alternative makes the case `incomplete`, an unqualified selected candidate
is `invalid_selection`, and fewer than two qualified measured candidates gives `no_comparison`.
These cases remain in coverage counts, have no numeric regret and fail the report. Invalid input
(including duplicate identities, NaN, mismatched pass sets or declared holdout leakage) exits 2;
incomplete/invalid selections, no-comparison cases or regret above `--max-regret` exit 1; otherwise
exit 0. Without a threshold, exit 0 means complete comparisons only, not acceptable performance.

The tool verifies disjointness against the **declared** fit inventory, not the truth of its history.
Include all boards/models used for fitting or manually choosing thresholds, including inherited
evidence. Empty fit lists assert no such data, not an unknown history. After inspecting a holdout
and tuning on it, it is no longer held out. Freeze the candidate inventory and threshold before
evaluation. Missing candidates cannot be inferred by this consumer; never silently omit failed
measurements. An oracle failure excludes an incorrect alternative, but must not be used to hide a
timeout (which is `unmeasured`). Regret is relative to the supplied inventory, not a global optimum.

There is no automatic import from route-sweep CSVs or product choice JSON: neither alone carries
all qualification, fit-inventory and matched-run facts required here. `regret_collect.py` combines
them explicitly for the linear route axis. Each band point of a `ninferno_route_tune` run becomes a
case; its candidates are the routes timed there, and the selected route is the one recorded by
`--selected-out`, which is timed too. The fit inventory, holdout and identities are passed by hand.
A candidate is `ok` only with `--qualified`, the correctness tests that passed on that build and
board; the statement is kept in the document. A route missing from a pass is `unmeasured`; a route
timed twice in one pass, or passes whose selections differ, are refused. Every row of both CSVs
names its run (`run`) and board (`device`: its device class and UUID), and a sweep row how it was timed (`protocol`: `copies`,
consecutive calls over `calls` weight copies at a decode width under `--rotate 1`, `flushed`,
one call after the L2 flush, or `warm`, one call where the timer had no flush buffer; `samples`). A pass's two files must come from one run, all passes from
one board, and a case's passes from one protocol, or the collector refuses them; each case's
`workload` states its protocol. Files written before these columns are refused, and so is one run given as two passes.

route_tune selects with its problems' policy (an `a16` problem is `A16Only`, the exact mode's
linears, not the default `AllowA8` path of a published model's projections) and the speed route
profile (`prefill_routes` inactive, so `T >= 17` is a batched decode call, not a prefill chunk);
each case's workload and numerical contract say so.

```bash
for p in 1 2 3; do ninferno_route_tune --shapes shapes.txt --out /dev/null --routes-out p$p.csv --selected-out sel_p$p.csv; done
python3 tools/bench/regret_collect.py --selected sel_p1.csv sel_p2.csv sel_p3.csv --routes p1.csv p2.csv p3.csv --board rtx3060 --model qwen3.5-9b --selector-revision <commit> --build-id <build> --holdout board --fit-boards rtx4090,rtx5090 --qualified "ninferno_linear_generic_test, ninferno_route_classes_test" > selection.json
```

The report preserves selector/build identity and a SHA-256
of the exact input bytes. This is an offline evaluation aid for #301, not runtime autotuning.

## Regret by T band

One geomean over all cases hides where a selector loses: a change can cost 10% at T 17-32 and win
at T 48+. `regret_bands.py` reports the extra time of the recorded picks over the best observed
route, **summed over the cases of a band** (time-weighted: a long GEMM counts for what it costs, a
tiny one barely), for the bands decode (T 1-8), verify (9-16), narrow (17-32), wide (33-128) and
batch129 (129+), by board, format and route family (the part of the route name before the first
dot). route_tune times the speed profile's selection up to T 512, so the widest band is batched
decode and wide verify, not the prefill profile's 2048-token chunks: this says nothing about
prefill.

There is no default cross-band mean: no committed mix gives band weights the sweeps can carry.
`--band-weights '{"decode":3,"verify":1}'` adds a `weighted` mean per group; a group with no case
in a band of nonzero weight gets none, rather than a mean of the bands it has. A case is skipped and
counted, never scored as zero regret, when its pick was not timed, when a pass did not time one of
its candidates, or (`--pick model`, T >= 17 only) when the model prices none of them.

```bash
python3 tools/bench/regret_bands.py --doc selection.json --doc other_board.json --md bands.md --json bands.json
python3 tools/bench/regret_bands.py --routes p1.csv p2.csv --selected sel_p1.csv --board rtx5090
python3 tools/bench/regret_bands.py --routes p1.csv p2.csv --pick model --board rtx5090   # the model price's own pick, needs model_us
python3 -m unittest discover -s tests -p test_regret_bands.py
```

`--doc` takes the `ninferno-selection-regret/1` documents `regret_collect.py` writes. `--routes`
takes plain `route_tune` CSVs (medians over the passes given), because the existing sweeps predate
the run/device columns `regret_collect.py` requires; the board is named by hand, and nothing then
checks that the passes come from one build and board.

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
