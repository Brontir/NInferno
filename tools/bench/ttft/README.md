# Serve TTFT benchmark

A black-box time-to-first-token (TTFT) benchmark for `ninferno-serve`. The measurement runner is a
client for an already-running server: it builds a fixed request graph from frozen text and media,
sends only public HTTP requests and measures TTFT. It never starts or configures the server, calls
`Engine`, reads request logs, or labels a request as a cache hit, transfer or eviction. A separate
campaign controller owns the repetitive process lifecycle around the runner. Use it to check
request-arrival latency under scheduling, cache and media scenarios; for throughput use
[tools/bench](../README.md) and [bench](../../../bench/README.md), and for published numbers see
[docs/performance](../../../docs/performance/README.md).

The standard campaign serves one resident `qwen3.8-27b/nvfp4` artifact with FP8 KV on one GPU and
one to eight active requests. Artifact, KV dtype, speculative backend and model are not benchmark
axes. The runner discovers the public model id through `/v1/models`; the controller applies the
fixed Serve profile and starts a fresh process for every measured sample.

## Quick start

The controller expects `build/apps/ninferno-serve` and `out/qwen3_8_27b_nvfp4.ninferno` (convert it
from the Hugging Face checkpoint, see the [conversion guide](../../../docs/weight-conversion.md)) and
a free `127.0.0.1:18080`:

```bash
python3 tools/bench/run_serve_ttft_campaign.py --campaign resource --samples 5
python3 tools/bench/run_serve_ttft_campaign.py --campaign smoke
```

## Options

### Campaign controller: `run_serve_ttft_campaign.py`

| Option | Default | Meaning |
|---|---|---|
| `--campaign C` | `resource` | `smoke` (the `cold-short` baseline), `resource` (12 pressure, Host-rotation and State working-set cases), `full` (every audited case). Mutually exclusive with `--case`. |
| `--case NAME` | none | Run one named case; repeat for a focused campaign (no duplicates). |
| `--samples N` | 1 | Fresh-process samples per case. |
| `--output-dir DIR` | `profiles/bench/ttft/qwen3_8_27b_nvfp4-fp8/<stamp>-<campaign>` | Must be empty or absent. |
| `--startup-timeout-seconds S` | 300 | Wait for the server to become ready. |
| `--request-timeout-seconds S` | 600 | Per-request timeout passed to the runner. |

The server binary, artifact, host `127.0.0.1` and port `18080` are fixed in the script. Each sample
starts `ninferno-serve` with `--host`, `--port`, `--request-log-jsonl` and the profile arguments
below. The `resource` campaign contains `shared-state-working-set-shift`,
`private-state-working-set-shift`, `shared-state-hot-prefix`, the six
`resume-after-interference-*` cases, `session-alternating-64k-host-swap`,
`session-rotation-55k-host` and `session-rotation-55k-two-cohort-stream`.

### Runner: `run_serve_ttft.py`

Use it when a separately managed server is intentional. Exit status is zero only for a constructed
case.

| Option | Default | Meaning |
|---|---|---|
| `--base-url URL` | required | Server root, for example `http://127.0.0.1:18080`. |
| `--case NAME` | required | One of the audited cases below. |
| `--profile-label NAME` | required | Profile label recorded in the result; validated, but the runner cannot verify that the process really used those arguments. |
| `--output PATH` | required | Raw JSON result. |
| `--timeout-seconds S` | 600 | Per-request timeout. |
| `--api-key-env VAR` | `NINFERNO_API_KEY` | Environment variable holding the API key, if the server needs one. |

```bash
python3 tools/bench/run_serve_ttft.py --base-url http://127.0.0.1:18080 \
  --case shared-fanout --profile-label shared-prefix --output profiles/bench/ttft/shared-fanout-01.json
```

The runner prints live progress to `stderr` (profile validation, corpus loading, model discovery,
barriers, each request's preparation, start, upload, acceptance, first output, terminal outcome,
cancellation and errors, plus a five-second heartbeat naming the last stage and active requests).
Only the final JSON goes to `stdout`.

### Summarizer: `summarize_serve_ttft.py`

```bash
python3 tools/bench/summarize_serve_ttft.py profiles/bench/ttft/qwen3_8_27b_nvfp4-fp8/<campaign> \
  [--baseline <other-campaign>]
```

Re-aggregates a complete or partial campaign; `--baseline` matches observations against an explicit
prior campaign. It reads only runs declared in the manifest, reports missing, invalid and
unconstructed samples and links each issue to its raw, progress, Serve and request diagnostics.

## Measurement contract

Request JSON and data URLs are prepared before timing. Every request owns a preconnected HTTP/1.1
connection.

```text
t0 = immediately before the first request bytes are written
tb = request body write completed
ta = first protocol metadata emitted after Serve accepted the request
t1 = first non-empty model text, reasoning or tool-argument delta
TTFT = t1 - t0
```

Connection setup and JSON serialization are outside TTFT. Body upload, server request preparation,
media acquisition and preprocessing, admission, prefill and the first model output are inside it.
HTTP headers, assistant-role chunks, `response.created`, `message_start`, usage, empty deltas and
finish metadata are not model output.

A rejection, timeout, cancellation, protocol failure or disconnect before `t1` has no TTFT; a
rejection's status and code are reported separately. A cancellation case is constructed only when
the client transport ends without a protocol terminal event within five seconds of the cancellation.
A clean request records only its first accepted, first-output and terminal events.

`constructed=true` means the requested arrival graph and all required external event inequalities
occurred; it does not mean an expected cache or scheduling optimization worked. A constructed case
with unexpectedly high TTFT is a valid regression signal, and the client deliberately does not use
internal diagnostics to reinterpret it.

Layers:

| Layer | Role |
|---|---|
| `bench/fixtures/ttft` | Frozen text, media, hashes and token facts (built offline by `tools/bench/ttft/build_fixtures.py`). |
| `tools/streaming_http` | Byte transport and incremental SSE timestamps. |
| `tools/ninferno_serve` | Public Chat, Responses and Anthropic adapters. |
| `tools/bench/ttft` | Request graphs, event coordination, reports. |
| `run_serve_ttft_campaign.py` | Profiles, fresh server lifecycle, artifacts, summary. |

The protocol is chosen by the case: Chat for ordinary scheduling and media, Responses for
`previous_response_id` sessions, Anthropic Messages for explicit `cache_control` boundaries.

## Resource formulas

For a request with expanded prompt tokens `p` and output limit `o`, the Main KV reservation used by
the cases is `g = o - 1` and `e = 64 * ceil((p + g) / 64)`; the first generated token is produced at
the prompt frontier, so only `o-1` tokens grow KV.

| Shape | `p` | `o` | `e` |
|---|---:|---:|---:|
| short | 30 | 32 | 64 |
| long source | 7680 | 16 | 7744 |
| independent long prefill | 7680 | 32 | 7744 |
| rotation source | 55000 | 32 | 55040 |
| interferer | 127 | 256 | 384 |
| decode holder | 127 | 4096 | 4224 |
| unsafe borrower | 30 | 3000 | 3072 |
| exact context | 8129 | 64 | 8192 |
| 256K input | 260096 | 32 | 260160 |

The resource-pressure graph is identical for all six pressure profiles; only the legal placement of
A changes.

```text
A: stored Responses long source completes
B and C: two disposable interferers start together and both reach first output
A2: previous_response_id=A resumes after B and C complete
e(A) + e(B) = 8128 <= 8192        e(A) + e(B) + e(C) = 8512 > 8192
```

The six-session rotation uses six byte-distinct, early-divergent 55000-token Responses roots
(`e=55040`). Four roots fit the 240000-token Device KV capacity (220160) and five do not (275200).
After all six complete, the graph warms root 0 once and branches from each original root in three
sequential round-robin passes; paired round-2 versus round-1 TTFT comparisons are declared for every
rotation position. The protected-head profile uses `e(holder)=4224`, `e(head)=7744`, `e(short)=64`,
`e(unsafe)=3072`, `K=7808`: a safe borrower fits beside the head, the unsafe one does not.

## Serve profiles

`profiles.py` is the executable catalog. Every sample uses a fresh process, and Serve's internal
warmup request has context-cache participation disabled, so each sample starts with an empty logical
context cache. Before the first process the controller stages the standard artifact once in
`/dev/shm/ninferno-artifacts/` under a source-identity name and every process reads that copy;
later campaigns reuse it while the source device, inode, size and mtime are unchanged (there is no
disk fallback or user option).

Common arguments for every profile: `--kv-dtype fp8 --no-thinking --greedy
--log-stats-interval-ms 0`. Capacities are tokens; `-mib` options are MiB. Every profile that sets
`--max-private-continuations` also passes `--max-long-anchors-per-continuation 0`.

| Profile | `--max-context` | `--kv-capacity` | `--max-concurrency` | Other arguments |
|---|---:|---:|---:|---|
| `text-cold-8k` | 8192 | 8192 | 1 | `--no-prefix-reuse` |
| `text-cold-64k` | 65536 | 65536 | 1 | `--no-prefix-reuse` |
| `text-cold-256k` | 262144 | 262144 | 1 | `--no-prefix-reuse` |
| `cache-hot` | 8192 | 8192 | 1 | `--device-state-slots 2 --host-state-slots 0 --host-kv-mib 0 --max-private-continuations 2 --max-shared-prefixes 0` |
| `cache-state-working-set` | 32768 | 32768 | 1 | `--prefill-chunk 1024 --spec dflash2 --draft-tokens 7 --lm-head-draft --device-state-slots 8 --host-state-slots 8 --host-kv-mib 0 --max-private-continuations 8 --max-shared-prefixes 8` |
| `cache-private-working-set` | 32768 | 32768 | 1 | `--prefill-chunk 1024 --spec dflash2 --draft-tokens 7 --lm-head-draft --device-state-slots 2 --host-state-slots 2 --host-kv-mib 0 --max-private-continuations 8 --max-shared-prefixes 0` |
| `cache-pressure-device` | 8192 | 16384 | 2 | `--device-state-slots 2 --host-state-slots 0 --host-kv-mib 0 --max-private-continuations 4 --max-shared-prefixes 0` |
| `cache-pressure-state-host` | 8192 | 16384 | 2 | `--device-state-slots 0 --host-state-slots 4 --host-kv-mib 0 --max-private-continuations 4 --max-shared-prefixes 0` |
| `cache-pressure-kv-host` | 8192 | 8192 | 2 | `--device-state-slots 2 --host-state-slots 0 --host-kv-mib 8192 --max-private-continuations 4 --max-shared-prefixes 0` |
| `cache-swap-64k-host` | 65536 | 65536 | 2 | `--device-state-slots 4 --host-state-slots 0 --host-kv-mib 4608 --max-private-continuations 4 --max-shared-prefixes 0` |
| `cache-rotation-55k-host` | 240000 | 240000 | 4 | `--max-pending-requests 32 --pending-timeout-ms 120000 --spec mtp --draft-tokens 3 --lm-head-draft --device-state-slots 2 --host-state-slots 24 --host-kv-mib 49152 --max-private-continuations 24 --max-shared-prefixes 24` |
| `cache-pressure-both-host` | 8192 | 8192 | 2 | `--device-state-slots 0 --host-state-slots 4 --host-kv-mib 8192 --max-private-continuations 4 --max-shared-prefixes 0` |
| `cache-pressure-evict` | 8192 | 8192 | 2 | `--device-state-slots 1 --host-state-slots 0 --host-kv-mib 0 --max-private-continuations 4 --max-shared-prefixes 0` |
| `cache-pressure-catalog` | 8192 | 16384 | 2 | `--device-state-slots 2 --host-state-slots 0 --host-kv-mib 0 --max-private-continuations 2 --max-shared-prefixes 0` |
| `cache-off` | 8192 | 8192 | 1 | `--no-prefix-reuse` |
| `shared-prefix` | 8192 | 16384 | 2 | `--device-state-slots 2 --host-state-slots 0 --host-kv-mib 0 --max-private-continuations 2 --max-shared-prefixes 1` |
| `shared-value` | 16384 | 16384 | 1 | `--device-state-slots 3 --host-state-slots 0 --host-kv-mib 0 --max-private-continuations 1 --max-shared-prefixes 1` |
| `session-order` | 8192 | 16384 | 2 | `--device-state-slots 8 --host-state-slots 0 --host-kv-mib 0 --max-private-continuations 4 --max-shared-prefixes 0` |
| `scheduler-overlap` | 8192 | 16384 | 2 | `--prefill-chunk 1024 --no-prefix-reuse` |
| `scheduler-prefill-128` | 8192 | 16384 | 2 | `--prefill-chunk 128 --no-prefix-reuse` |
| `scheduler-prefill-4096` | 8192 | 16384 | 2 | `--prefill-chunk 4096 --no-prefix-reuse` |
| `scheduler-backfill` | 7744 | 7808 | 2 | `--pending-timeout-ms 120000 --no-prefix-reuse` |
| `lane-limit-8` | 4224 | 33792 | 8 | `--max-pending-requests 1 --pending-timeout-ms 120000 --no-prefix-reuse` |
| `pending-timeout` | 4224 | 4224 | 1 | `--max-pending-requests 1 --pending-timeout-ms 100 --no-prefix-reuse` |
| `context-boundary` | 8192 | 8192 | 1 | `--no-prefix-reuse` |
| `vision-cache` | 32768 | 32768 | 1 | `--device-state-slots 2 --host-state-slots 0 --host-kv-mib 0 --max-private-continuations 2 --max-shared-prefixes 0 --vision --media-cache-mib 512 --media-live-mib 512` |
| `vision-thread-1` | 32768 | 32768 | 1 | `--device-state-slots 2 --host-state-slots 0 --host-kv-mib 0 --max-private-continuations 2 --max-shared-prefixes 0 --vision --media-cache-mib 512 --media-live-mib 512 --media-preprocess-threads 1` |
| `vision-concurrent` | 32768 | 65536 | 2 | `--vision --media-cache-mib 512 --media-live-mib 1024 --no-prefix-reuse` |
| `media-cache-tight` | 8192 | 8192 | 1 | `--vision --media-cache-mib 16 --media-live-mib 128 --no-prefix-reuse` |
| `vision-boundary` | 65536 | 65536 | 1 | `--vision --no-prefix-reuse` |
| `mixed-four` | 8192 | 32768 | 4 | `--vision` |

## Audited cases

Baseline and cache cases:

| Case | Profile | Request graph and covered behavior |
|---|---|---|
| `cold-short` | `text-cold-8k` | One 30-token prompt; short baseline. |
| `cold-long-8k` | `text-cold-8k` | One 7680-token prefill. |
| `cold-long-64k` | `text-cold-64k` | One 64512-token prefill. |
| `cold-long-256k` | `text-cold-256k` | One 260096-token extreme input. |
| `mixed-four-ordered` | `mixed-four` | Continuation, independent cold long, short and image requests, each submitted after the previous is accepted. The two long prompts diverge at the first system-content token. |
| `mixed-four-concurrent` | `mixed-four` | The same four requests released through one barrier. |
| `shared-state-working-set-shift` | `cache-state-working-set` | A/B/C then D/E/F, three repeated probe rounds and a final A probe in one process. |
| `private-state-working-set-shift` | `cache-private-working-set` | Shared disabled: complete-history A/B conversations, then four C/D rounds. |
| `shared-state-hot-prefix` | `cache-state-working-set` | Repeated A probes interleaved with one-shot B-F conversations. |
| `anonymous-hot-continuation` | `cache-hot` | Chat source, then exact full-history continuation. |
| `session-hot-continuation` | `cache-hot` | Stored Responses source, then `previous_response_id` continuation. |
| `session-alternating` | `cache-pressure-device` | `A1, B1, A2, B2` across two stored Responses lineages. |
| `session-alternating-64k-host-swap` | `cache-swap-64k-host` | Two early-divergent 64512-token sessions as `A1, B1, A2, B2`; one fits Device, the pair needs two Host KV covers. |
| `session-rotation-55k-host` | `cache-rotation-55k-host` | Six 55000-token Responses roots, one warm branch from root 0, three sequential six-root rounds. |
| `session-rotation-55k-two-cohort-stream` | `cache-rotation-55k-host` | The full rotation, then two 900-token store-free streams kept active while a second cohort of six roots is created and resumed three times (process history, concurrency and Host State descriptor pressure). |
| `unmarked-common-prefix-miss` | `cache-hot` | Two standalone user messages share over 4096 tokens but no legal marker. |
| `resume-after-interference-device` | `cache-pressure-device` | A/B/C/A2 with source placement available on Device. |
| `resume-after-interference-state-host` | `cache-pressure-state-host` | Checkpoint State available only on Host. |
| `resume-after-interference-kv-host` | `cache-pressure-kv-host` | KV pressure with Host KV available. |
| `resume-after-interference-both-host` | `cache-pressure-both-host` | Host State and Host KV capacity. |
| `resume-after-interference-evicted` | `cache-pressure-evict` | Source publishes initially, then pressure has no Host fallback. |
| `resume-after-interference-catalog` | `cache-pressure-catalog` | Logical catalog fills while physical capacity remains. |
| `continuation-cache-off` | `cache-off` | Stored Responses control with prefix reuse disabled. |
| `shared-sequential` | `shared-prefix` | Anthropic-marked system prefix, then a second suffix. |
| `shared-openai-explicit` | `shared-prefix` | OpenAI explicit system breakpoint, then a different suffix. |
| `shared-openai-implicit` | `shared-value` | Default full-prompt candidate, private displacement, then the identical prompt. |
| `shared-observed-promotion` | `shared-value` | Two unmarked Anthropic requests establish demand; a filler removes the private endpoint before a third identical request. |
| `shared-fanout` | `shared-prefix` | Seed, then two simultaneous branches from a 4149-token non-page-aligned frontier. |
| `shared-slot-retention` | `shared-value` | Equal-size marked A and B compete for one shared slot; a filler removes A's private fallback before A returns. |
| `shared-value-replacement` | `shared-value` | A 4149-token system owner and a 5074-token tool candidate compete for one slot. |
| `shared-tools-sequential` | `shared-prefix` | Two requests with the same marked 32-tool prefix. |
| `shared-tools-changed` | `shared-prefix` | The first tool identity changes, invalidating the marked tool prefix. |

Scheduling and boundary cases:

| Case | Profile | Required external relation or boundary |
|---|---|---|
| `short-during-prefill-128` | `scheduler-prefill-128` | `long.accepted < short.sent < long.first`. |
| `short-during-prefill-1024` | `scheduler-overlap` | Same arrival graph with the standard chunk. |
| `short-during-prefill-4096` | `scheduler-prefill-4096` | Same graph with the largest covered chunk. |
| `short-during-decode` | `scheduler-overlap` | `holder.first < short.sent < holder.completed`. |
| `protected-head-backfill` | `scheduler-backfill` | A safe short borrower passes a resource-blocked FIFO head. |
| `protected-head-no-backfill` | `scheduler-backfill` | A 3072-token borrower stays behind the head because it would keep blocking it. |
| `active-lanes-full-8` | `lane-limit-8` | Eight holders reach first output before a ninth probe (lane ceiling). |
| `session-publication-order` | `session-order` | Older child starts first but finishes after the newer child; the later continuation follows the newer. |
| `cancel-before-first` | `scheduler-overlap` | Cancel after acceptance; transport ends without first output or terminal event; then a probe. |
| `cancel-after-first` | `scheduler-overlap` | Cancel an active decode after first output; no terminal event; then a probe. |
| `pending-overflow` | `lane-limit-8` | Eight active plus one pending; the next request is 429 `server_overloaded`. |
| `pending-timeout` | `pending-timeout` | A blocked non-streaming request is 503 `request_queue_timeout`. |
| `context-exact` | `context-boundary` | `p + o - 1 = 8192` succeeds. |
| `context-over` | `context-boundary` | An 8193-token prompt is 400 `context_length_exceeded`. |

Media cases:

| Case | Profile | Request graph and covered behavior |
|---|---|---|
| `media-cold-image` | `vision-cache` | One image through acquisition, preprocess, Vision and text prefill. |
| `media-cold-image-video` | `vision-cache` | Interleaved image and video input. |
| `media-prefix-continuation` | `vision-cache` | Same media and full history followed by text. |
| `media-prefix-append` | `vision-cache` | Reused media history with one new suffix image. |
| `media-prefix-changed` | `vision-cache` | A byte-distinct replacement changes an earlier media identity. |
| `media-preprocess-warm` | `media-cache-tight` | The identical standalone image twice with context reuse disabled. |
| `media-cache-thrash` | `media-cache-tight` | `A, B, C, A` against a 16 MiB preprocess cache. |
| `many-image-28` | `vision-cache` | Legal 28-image request, 28752 expanded prompt tokens. |
| `many-image-28-thread-1` | `vision-thread-1` | Same request with one host preprocessing worker. |
| `text-during-media-prepare` | `vision-concurrent` | Media body completes, then short text arrives before media acceptance and must finish first. |
| `media-during-text-decode` | `vision-concurrent` | Heavy media request arrives while a text holder decodes. |
| `two-heavy-media-arrivals` | `vision-concurrent` | Two byte-distinct 28-image requests start through one barrier. |
| `vision-disabled` | `text-cold-8k` | An image request is 400 `vision_disabled`. |
| `vision-envelope-over` | `vision-boundary` | 33 images exceed the 32768 raw-patch envelope: `media_budget_exceeded`. |

Not covered, because public HTTP cannot force or prove them: private long anchors, multiple
independent cache markers, an invalid "KV without State" checkpoint, allocator-fragment geometry,
exact transaction or cancellation instruction boundaries, partial page residency. Those are Engine
tests. Large-scale concurrency, QoS, preemption, multi-GPU, multi-model serving, artifact
comparisons and speculative-backend sweeps are outside this benchmark.

## Results

```text
profiles/bench/ttft/qwen3_8_27b_nvfp4-fp8/<timestamp>-<campaign>/
  manifest.json                              source artifact and the tmpfs path Serve used
  raw/<profile>/<case>/sample-NNN.json       external TTFT measurement (authority)
  progress/<profile>/<case>/sample-NNN.log   runner progress
  serve/<profile>/<case>/sample-NNN.log      Serve text log
  request-log/<profile>/<case>/sample-NNN.jsonl   structured Serve timings (diagnosis only)
  summary.md  summary.json  summary.csv
  failures.json                              only when a run fails
```

Complete, failed and interrupted campaigns all produce the summary bundle. `summary.md` is the human
report, `summary.json` the full machine-readable aggregate, `summary.csv` the typed TTFT,
comparison, symmetric-order, rejection and optional cross-campaign rows. Raw results preserve
external timestamps, request body bytes, outcome, HTTP status and code, a compact event trace,
first-output and completion order, the public model id and failed construction conditions.

Successful constructed TTFT is grouped by public model, architecture category, case, profile and
request role; each group reports raw samples, min, median, max, median absolute deviation and
relative span. Declared comparisons use `subject - baseline` (positive means the subject is slower);
roles inside one case are paired per run, cross-case comparisons use independent group medians, and
symmetric barrier roles are ranked inside each run. The report does not publish QPS, throughput, P95
or inferred internal cache actions.
