# HTTP serving reference

Full reference for `build/apps/ninferno-serve`. The short guide is [serving.md](../serving.md).
For the exact option contract, run `./build/apps/ninferno-serve --help`. Per-GPU and per-model speeds
are in the [performance matrix](../performance/README.md).

`ninferno-serve` loads one model, a `.ninferno` file produced by conversion from the original
Hugging Face checkpoint. It exposes OpenAI- and Anthropic-compatible HTTP endpoints over one
resident NInferno Engine on one NVIDIA GPU (Volta `sm_70` through Blackwell `sm_120`; NVFP4 weights
run natively on `sm_120` and on slower generic kernels elsewhere). NInferno publishes no prebuilt model files: every model is converted
from its original checkpoint, which the server does on first start.

Contents:

1. [Model sources and first start](#1-model-sources-and-first-start)
2. [Startup configuration](#2-startup-configuration)
3. [Speculative decoding](#3-speculative-decoding)
4. [Endpoints and transport](#4-endpoints-and-transport)
5. [Models endpoint](#5-models-endpoint)
6. [OpenAI Chat Completions](#6-openai-chat-completions)
7. [OpenAI prompt caching](#7-openai-prompt-caching)
8. [OpenAI Responses Core](#8-openai-responses-core)
9. [Anthropic Messages](#9-anthropic-messages)
10. [Authentication, CORS and exposure](#10-authentication-cors-and-exposure)
11. [Server options](#11-server-options)
12. [Logging and metrics](#12-logging-and-metrics)
13. [Execution behavior](#13-execution-behavior)

## 1. Model sources and first start

```bash
./build/apps/ninferno-serve Qwen/Qwen3-1.7B            # or org/name@revision
```

The model argument is one of:

| Argument | What happens |
|---|---|
| a `.ninferno` file | served as is |
| a Hugging Face checkpoint directory or a GGUF file | converted once into `<dir>/.ninferno-cache/` |
| a hub id (`org/name[@revision]`) | fetched, converted, cached (below) |

A hub id is fetched into the Hugging Face cache (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else
`$XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub`). It is converted with the official recipe (dense and MoE decoders as
published; `--recipe NAME`, or `--quant LABEL` for a GGUF, names another) into
`<cache>/models--org--name/ninferno/<commit>/`. The drafters that the model or the official registry
declares are converted with it. Later starts reuse the result. Nothing else on the machine is
searched.

- `$HF_TOKEN` authorizes gated and private models.
- `HF_HUB_OFFLINE=1` uses the cache only.
- The served model id is the hub id.
- `--prepare-only` fetches and converts, then exits (no GPU).
- `--with-drafters` attaches the drafters of a GGUF hub model too (a separate download). A checkpoint's
  drafters are attached without it.

Refusals come before any weight is downloaded. Each says what to do instead:

- an architecture no recipe converts (with the supported list);
- a gated repository without an accepted license and token;
- a repository that does not exist or is private;
- a model whose weights as published exceed the GPU's free memory (with the quantizing recipes that fit);
- too little disk for the download.

### Automatic choices and the startup log

- **Context.** Without `--max-context` the server serves the model's own context
  (`max_position_embeddings`) when it fits the free GPU memory. Otherwise it serves the longest
  whole number of 64-token KV pages (at least 1024 tokens) that fits. The KV capacity then takes the
  rest (`auto`). A Qwen3.5 checkpoint whose `config.json` carries YaRN (`rope_type` `yarn`, as Qwen
  documents it for long contexts) keeps it through conversion: the model's context is then
  `original_max_position_embeddings` times `factor` (at most 1,048,576), and `--rope-yarn-factor`
  overrides the factor. Attention reads at most 1,048,576 keys, so a longer `--max-context` is
  refused at startup.
- **Host tier.** Without `--host-state-slots` and `--host-kv-mib`, the Host tier defaults are scaled
  down to half of what the host can pin; the state count is `max(8, 2 * private continuations)` (an endpoint
  and an edit point for each conversation).
- **Log lines.** The startup log states each choice (`context | ...`, `capacity | ...`,
  `context cache | ...`).
- **Sampling source.** Every generation response names where its default sampling comes from in
  `x-ninferno-sampling-defaults`: `generation_config.json`, `transformers defaults`, or the family's
  presets, plus server flags such as `--greedy`.

### Kernel warm-up

Before an automatic choice measures the free GPU memory, the server runs the kernels its plan can
choose for this model (`kernel warm-up | ...`). CUDA loads a kernel's code into GPU memory at its
first launch, and a kernel with a larger stack frame grows the local memory the driver keeps for
every resident thread. Neither is given back.

- A probe engine with the plan's concurrency, speculation, KV formats and prefill chunk takes one
  request of every prefill width up to the chunk. It captures every decode graph class at the
  fitted context. The fit is measured after it.
- The warm-up starts the CUDA stack limit from zero. It ends at the largest stack frame among the
  kernels the plan launched, instead of the driver's 1 KiB default for every resident thread.
- The log line names the memory this keeps: kernel code, the stack, and the memory of the probe's
  CUDA graphs. The driver keeps the graphs and the served graphs take them again, so the plan
  reserves no separate graph allowance. The line also gives the limit, the driver's default, and the
  warm-up step that raised it last.
- Pinning the context cache's Host tier takes GPU memory too (the driver's page tables for the
  pinned bytes, about 2 MiB per GiB). The warm-up pins buffers of the Host tier's sizes once,
  measures the cost, and the fit reserves it (`Host tier pinning takes ...`).
- The warm-up takes longer on larger models (one prefill of every width). Only an automatic choice
  runs it: with `--max-context`, `--kv-capacity`, `--prefill-chunk` and `--device-state-slots` all
  given, nothing depends on the free memory.

## 2. Startup configuration

See [CUDA synchronization](../cli.md#cuda-synchronization) for the shared `NINFERNO_CUDA_SYNC`
setting.

```bash
./build/apps/ninferno-serve models/qwen3_8_27b_nvfp4.ninferno \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

This example uses Qwen3.8-27B NVFP4 with a long-context FP8/MTP3 profile.

- Each request has a 240,000-token logical ceiling. A shared 240,000-token Main Text KV pool serves
  admitted requests. Either request may use the full capacity when running alone. Two requests run
  concurrently when their complete reservations fit.
- With `C=2` and two extra Device checkpoint slots, the process owns two active StateImage
  guarantees plus a global pool of two Device-resident checkpoints. Eight pinned Host State slots
  and 8 GiB of pinned Host KV retain inactive continuations under Device pressure. Active request
  capacity is two.
- Other converted models use the same command shape with their own path. For 35B-A3B DFlash,
  replace the MTP selection with `--spec dflash --draft-tokens 5 --lm-head-draft`.
- Qwen3.8-27B artifacts with DFlash2 companion weights also support
  `--spec dflash2 --draft-tokens N`, with `--lm-head-draft` optional. DFlash2 accepts draft counts
  1..15 and supports the same sampling, concurrency, prefix reuse, and image/video request
  surfaces. It may stay combined with `--vision`.

The MTP3 profile above is a conservative example, not the fastest setting. A greedy
single-request sweep over a mixed corpus found these fastest:

| Model | Fastest |
|---|---|
| 35B-A3B | DFlash K=5 |
| Qwen3.8-27B with DFlash2 | DFlash2 K=11-15 |
| Qwen3.8-27B without DFlash2 | MTP K=5 |

The optimum per task and its limits are in the
[CLI guide](../cli.md#choosing-the-backend-and-draft-count). Under concurrency `--spec auto` bounds
these depths per decode step ([section 3](#3-speculative-decoding)).

### Model id

When `--model-id` is omitted, the server advertises and accepts the artifact's `metadata.name`,
falling back to its file name when no name is stored (a model served from Hugging Face goes by its
hub id). An explicit `--model-id` is a public
HTTP alias override. It does not select or alter model execution.

### Vision

Vision is disabled by default. Its weights and Vision-specific unified-workspace extent are not
allocated, and media requests and token-count requests that carry media fail with HTTP 400
`vision_disabled`. Add `--vision` when the server must accept image or video input. A later request
cannot enable a capability omitted at startup. The artifact need only contain the Text backbone and
the optional components selected for this process.

DFlash and DFlash2 on a Qwen model can be combined with `--vision`. Each accelerates
generated-text decode after multimodal prefill. Vision encode and prefill stay outside speculative
acceleration.

### Explicit speculation flags

Speculative residency is frozen by `--spec mtp|dflash|dflash2` and its depth. Omitting `--spec`
runs the model's profile, as `--spec auto` ([section 3](#3-speculative-decoding)). `--spec none`
loads no speculative backend.

Without `--draft-tokens`, an explicit `--spec` drafts at the depth the model's profile runs that
backend with on this GPU, else at 3 for MTP and 7 for DFlash and DFlash2. The startup log names the
depth and where it came from. `--lm-head-draft` additionally loads the optimized proposal head.
`--drafter DRAFTER.ninferno` names a drafter artifact.

### Sampling defaults

The Engine selects sampling defaults from the loaded architecture and the request's resolved
thinking mode.

| Model | Mode | temperature / top-p / top-k / min-p / presence penalty |
|---|---|---|
| Qwen3.6-27B, Qwen3.8-27B | thinking | `1.0 / 0.95 / 20 / 0 / 0` |
| Qwen3.6-27B, Qwen3.8-27B | non-thinking | `0.7 / 0.80 / 20 / 0 / 1.5` |
| Qwen3.6-35B-A3B | thinking | as above, but presence penalty `1.5` |
| Qwen3.6-35B-A3B | non-thinking | as the 27B models |

Frequency penalty is `0` for all registered presets. Precedence, lowest to highest: registered
values, process flags, request fields. `--greedy` finally forces temperature `0`.

### Context cache capacity

For `C=--max-concurrency` and `H=--device-state-slots`, total Device StateImage capacity is `C+H`.
`C` slots guarantee active requests. `H` is a global checkpoint pool.

- Omitted, `H` is automatic, at most `2C` (three states a request, so an agent's tool loop keeps its
  turn closure and its endpoint while other requests run).
- Beside an explicit `--kv-capacity`, `H` is the most that fit.
- With the automatic KV capacity at a given `--max-context`, the pool comes first, a whole slot
  context at a time. `H` is the most that leave the pool as many whole contexts as the fewest
  states would (one, or none when the Host tier keeps states). That is all of them when the pool
  holds every slot's context.
- An automatic context (no `--max-context`) is chosen before that with one cached state per slot,
  fewer only below 8,192 tokens.
- Host State and Host KV are independent startup-fixed pinned-memory capacities. Host KV is shared
  by Main and the selected Backend pool and is consumed in physical page extents.
- `--no-prefix-reuse` selects root-only Engine mode. It cannot be combined with any of the six
  explicit context-cache capacity flags (`--device-state-slots`, `--host-state-slots`,
  `--host-kv-mib`, `--max-private-continuations`, `--max-shared-prefixes`,
  `--max-long-anchors-per-continuation`), including zero-valued ones.

### Context-cost presets

Context-cost coefficients resolve once at startup from generic defaults, matching compiled values,
and optional transfer or prefill entries from `--context-cost-presets FILE`. Prefill entries match
the hardware and a signature derived from the actual Text/Vision configuration, bindings and Uses.
A new representation without a matching measurement uses generic prefill coefficients. A malformed
file aborts startup. The operational context-cost record and JSONL `server_start` identify the
selected source.

## 3. Speculative decoding

### `--spec auto` (the default)

`--spec auto` runs the speculation the model came with for this GPU: the tuned profile that
`ninferno-convert` attached (see the [CLI guide](../cli.md)).

- A GPU class nobody tuned runs the `*` profile, taken from one tuned class. Its portable choices
  are the shallowest configuration within 2% of the best over the tuning corpus (a deep draft's win
  depends most on the GPU's cost of verifying it).
- Every choice is judged over the whole corpus mix with equal weight per section, never per prompt
  type.
- The server loads the drafter of the profile's default for its thinking default, with that
  choice's proposal head. It captures CUDA graphs for every depth the profile's choices use with
  that drafter.
- `--spec-select fixed` (the default) runs that configuration for every request.
  `--spec-select thinking` runs per request the profile's thinking or non-thinking default, by the
  request's own thinking flag.
- Requests of different depths share batches. A step verifies at the deepest request's depth, each
  request drafting at that launch's width.
- A choice that needs another drafter runs the default configuration, and the startup log says so.
- The `"ninferno"` request extension object has no members; any member is HTTP 400.

### Break-even floor

A request whose acceptance falls below its configuration's break-even floor stops drafting while
the others keep speculating. It verifies without drafts in their launches (one launch per step
instead of a second, plain one beside it), while the stopped requests' verify columns (their count
times the launch's width) stay within 16. It decodes plainly beyond that, or once no request of a
step drafts.

The floor is a break-even against plain decode of one request. While another request of its step
keeps drafting, the step's speculative launch runs anyway. A request below its floor therefore keeps
drafting as long as it emits at least 1.4 tokens per round (`--spec-shared-floor`;
`shared_trip_deferrals` counts such verdicts). It stops once it falls below that or drafts alone.

### Bounds under load

The C=1 choices are too deep for a busy server. A step's verify cost grows with its rows times the
depth plus one, while plain decode of several rows already reads the weights once for all of them.
The profile therefore bounds speculation under load, in its `load` table.

- **Measuring.** The table is measured per GPU class with
  `ninferno spec-tune <artifact> --load 2,4,8`, which loads the model once per candidate bound. It
  is measured again for a concurrency the widest engine does not hold. A concurrency it cannot
  decide leaves the table marked `load_partial`.
- **Local tunes.** A tune without `--load` writes no table. The server then takes the published
  table for this GPU and the tuned drafters, when its default drafts with the same kind, and names
  it at startup ("load bounds ... the local tune measured none").
- **Rule.** A decode step of at least `lanes` rows verifies at most `depth` drafts per row. At depth
  0 its rows finish their requests with plain decode. A row cannot resume speculating: its drafter
  state stops with its last round, so a later request reuses its checkpoints from before the
  switch (the turn closure at the prompt's end and long anchors), not its endpoint.
- **Joiners.** A request that joins such a row decodes plainly beside it rather than drafting in a
  second launch. On the RTX 5080 with Muse Glimmer 30B + DFlash and `3:0` at C=3, the last plain
  request beside a newcomer that drafted ran at 35 tok/s, against 50 alone. A joiner does not pass
  this on: once the rows the bound itself turned plain have finished, new requests speculate again
  even beside a joiner that still decodes plainly.
- **Limits.** A step of one row is never bounded, so a lone request runs exactly as the choices
  say. Table entries start at 2 lanes, and no speculating entry follows a 0. Load-capped rounds do
  not count towards a request's watchdog.
- **Fallback tables.** A profile measured without a table takes one whose default drafts with the
  same kind: another artifact's on the same GPU class first, else the `*` profile's for the same
  drafters.
- **Override.** `--spec-load LANES:DEPTH,...` replaces the table (`none`: no bound). It also bounds
  an explicit `--spec`, and needs one.
- **Draft width in a bound.** A speculating bound may also name a DFlash draft width,
  `LANES:DEPTH:block` or `LANES:DEPTH:drafts` (a table entry's `draft_width`). The width resolves
  from a launch's rows: a launch of at least that bound's lanes (and fewer than the next bound's)
  drafts at it, unless `--draft-width` forces one width for every launch. A launch no bound names
  drafts at the Engine's default: `block` for one row and `drafts` from two. `block` computes every
  column the drafter was trained on; `drafts` only the anchor and the round's drafts. Qwen 3.5's
  DFlash drafters run only `drafts`. Under `--spec auto`, a width the served speculation does not
  run (an MTP pick or fit fallback, `block` on Qwen 3.5) is dropped with a note in the startup log
  rather than refused.

### Graphs and workspace

The startup log's `load` field shows the bounds in force. `speculation | graphs` lists each captured
depth with the speculative batch sizes its CUDA graphs cover. For an explicit plan it also gives the
memory reserved for them (an automatic plan's kernel warm-up measures it instead).

- From a bound of 0's lanes on, no speculative graph or round workspace is planned.
- A depth only a bound asks is captured from that bound's lanes. A smaller speculative batch runs
  the next wider depth.
- A depth deeper than every bound a batch of that many rows can meet is not captured for it. With
  `2:3` at C=8, depths 5, 7 and 11 are captured for one row only: about 0.6 instead of 1.3 GiB of
  planned graph allowance for Qwen3.6 DFlash on a 16 GB card.
- An MTP round's workspace follows the same batches. A DFlash row that ran a wider round alone
  appends its features in its first round beside others. DFlash therefore keeps the full round
  workspace and runs that round without a graph (`ungraphed_rounds`).
- When such a bound's graphs do not fit, each moves to the deepest captured depth within it, and the
  startup log says so.

### When `--spec auto` does not fit

When the choice does not fit on the device at all, nor its fit fallback, the server proceeds in
this order:

1. It narrows an automatic prefill chunk (1024 halved down to 128). Whenever speculation is served
   on a narrowed automatic chunk, a note says which chunk it kept.
2. It tries shallower drafts with the same drafter: one draft shallower, then depth 1, then halving
   between them (a few Engine builds from any depth). It runs the deepest that fits (for a user's
   `--max-context` longer than the tune's, or a card nobody tuned).
3. It tries the model's own MTP head (when the choice drafts with another drafter and the model
   has one), the built-in MTP default with its floors calibrated on this GPU (a measurement being replaced by a model, [deterministic decisions](../maintainer/deterministic-decisions.md)).
4. Only then it decodes plainly and warns, rather than failing to start.

An explicit `--prefill-chunk` stays as given: the shallower drafts are tried beside it, and the
warning says that an automatic chunk would also have been narrowed. When plain decode does not
fit either, the refusal is plain decode's and says `--spec auto` was refused first.

### Per-request record

The configuration each request ran is in its request-log record (`speculative.config`, `family`,
`selection`, `reason`, `floor`, `plain_steps`, `wide_rounds`, `capped_rounds`, `ungraphed_rounds`,
`load_plain`, `watchdog_tripped`, `shared_trip_deferrals`) and in the OpenAI Chat
`timings.speculation` object.

### Measured throughput under load

Aggregate decode throughput (tok/s) for 20 requests of 512 tokens at temperature 1.0 with thinking
on. Each cell is plain decode / `--spec auto` without load bounds / `--spec auto` with the profile's
table (in parentheses in the Artifact column).

| GPU | Artifact (table) | C=1 | C=2 | C=4 | C=8 |
|---|---|---|---|---|---|
| RTX 5090 | Qwen3.6-35B-A3B NVFP4 (2:3) | 384 / 508 / 523 | 622 / 622 / 853 | 958 / 807 / 1129 | 1190 / 1069 / 1363 |
| RTX 5090 | Qwen3.8-27B NVFP4 with DFlash2 (2:7,5:5) | 75 / 174 / 178 | 140 / 280 / 309 | 231 / 427 / 451 | 357 / 565 / 633 |
| RTX 5090 | Gemma 4 26B-A4B NVFP4 (2:2,5:3) | 419 / 435 / 435 | 578 / 626 / 728 | 784 / 892 / 1023 | 953 / 1104 / 1297 |
| RTX 5090 | Gemma 4 31B NVFP4 (2:7,3:3,5:5; C=8 ran 5:7) | 79 / 182 / 178 | 147 / 228 / 350 | 220 / 317 / 469 | 378 / 345 / 590 |
| RTX 5090 | Nemotron 3.5 Lightning UD-Q4_K_XL, MTP Q4_0 (3:1,5:2) | 312 / 401 / 402 | 437 / 596 / 582 | 640 / 737 / 758 | 785 / 913 / 918 |
| RTX 5080 | Qwen3.6-35B-A3B UD-IQ3_XXS, DFlash (2:3) | 192 / 289 / 304 | 314 / 400 / 442 | 486 / 442 / 587 | 606 / 494 / 627 |
| RTX 5080 | Gemma 4 26B-A4B QAT UD-Q4_K_XL (2:3) | 246 / 397 / 400 | 375 / 538 / 590 | 517 / 596 / 772 | does not start |
| RTX 4090 | Nemotron 3.5 Lightning Q4_0, MTP Q4_0 (none) | 239 / 330 / 326 | 363 / 491 / 481 | 537 / 644 / 637 | 659 / 777 / 772 |

Notes:

- Without bounds, every step verifies at its deepest row's depth. This loses to plain decode from
  C=4 on Qwen3.6 and at C=8 on Gemma 4 31B.
- On the RTX 5080 at C=8 the Qwen3.6 bound's own graphs do not fit, so the server runs the captured
  depth 2 (measured with `--spec-load 2:2`).
- Gemma 4 26B does not start at C=8 on the RTX 5080, with or without a drafter.
- The Qwen3.6 RTX 5090 row without bounds at C=8 is `--spec-load none` on the current server.
- Gemma 4 31B's C=8 cell ran the earlier table 2:7,3:3,5:7. `spec-tune --load` measures depth 5 and
  7 at C=8 within 0.4 % of each other (737.6 and 734.7 tok/s, plain 467.1, at 1024 tokens of
  context).
- Nemotron 3.5 Lightning drafts with its published MTP head (a GGUF file of its own, attached as the
  drafter `mtp`) two deep on both GPUs. Its RTX 4090 table has no bound: depth 2 wins at every
  concurrency measured. Its second and third columns therefore run the same configuration and
  differ by run-to-run noise (up to 2.4 %), as do the RTX 5090 row's at C=1 and C=2, below its
  table's first bound (3 lanes).

### Stop and usage with speculation

Speculative backends preserve protocol output shapes, stop behavior, and usage accounting. If a stop
truncates a multi-token MTP, DFlash or DFlash2 round, the Engine commits the exact accepted target
prefix so that a following compatible turn can reuse it.

## 4. Endpoints and transport

| Method and path | Behavior |
|---|---|
| `GET /health` | Engine readiness |
| `GET /v1/models` | configured OpenAI model alias, effective `max_model_len`, and a llama.cpp-compatible `meta` object |
| `GET /props` | llama.cpp-style server properties (model alias, context, training context, vocabulary size, slots) |
| `GET /metrics` | Prometheus text exposition: llama.cpp's `llamacpp:*` series (prompt and generated tokens, decode rounds, busy slots, running and deferred requests) and `ninferno_reused_prompt_tokens_total`, `ninferno_requests_prefilling`, `ninferno_requests_materializing`, `ninferno_http_active_requests` |
| `GET /v1/models/{id}` | lookup of the configured alias, `max_model_len`, and `meta` object |
| `POST /v1/chat/completions` | OpenAI-style chat generation |
| `POST /v1/responses` | OpenAI Responses Core generation, state, typed Items, and SSE |
| `POST /v1/responses/input_tokens` | Responses prompt-token count without generation |
| `GET /v1/responses/{id}` | retrieve a locally stored terminal Response |
| `DELETE /v1/responses/{id}` | delete a locally stored Response |
| `GET /v1/responses/{id}/input_items` | list that Response's normalized input Items |
| `POST /v1/responses/{id}/cancel` | 400 `background_not_supported` (404 for an unknown id): there is no background execution |
| `POST /v1/responses/compact` | `compaction_not_supported` |
| `POST /v1/messages` | Anthropic-style message generation |
| `POST /v1/messages/count_tokens` | checkpoint-native expanded input-token count |

### Base URLs

Anthropic SDKs append `/v1/messages` to their base URL, so their base URL is
`http://127.0.0.1:8080`. OpenAI SDKs take `http://127.0.0.1:8080/v1`. A client given the OpenAI-style
base URL requests `/v1/v1/...`. Every path above except `/health` also answers under that doubled
prefix, with the same behavior, errors, and request IDs.

### Health, drain and failure

- `GET /health` returns HTTP 200 with `{"status":"ok"}` while the Engine can accept work. After an
  Engine-wide failure it returns HTTP 503 with `{"status":"unavailable"}`; while draining or stalled
  (below), 503 with `"draining"` or `"stalled"`. The body also carries the load gauges
  `active_requests`, `running_requests`, `waiting_requests` and `materializing_requests`. Temporary queue
  saturation does not make the Engine unavailable. The endpoint remains unauthenticated.
- On SIGTERM or SIGINT the server drains: `/health` and new requests get 503 while requests in
  flight finish, for at most `--drain-timeout-ms` (default 60000). A second signal stops at once.
  Either way the requests still running then end: a stream stops (its client sees the connection
  close), and a request without streaming is answered 503 `server_draining` with `Retry-After`
  (never its cut text as a finished answer). On Windows, closing the console window, logoff and
  shutdown stop at once.
- **Update check.** At start the server fetches the release manifest once, in the background, and
  logs a newer release as one warning line. `NINFERNO_UPDATE_CHECK=0` (or `off`, `no`, `false`)
  turns it off; `NINFERNO_UPDATE_URL` names another manifest. Nothing waits for it, and a failed
  check is logged at debug level only.
- With `--health-stall-ms` (default 60000; 0 off), `/health` is 503 while the Engine has work and
  makes no progress that long.
- **Engine-wide failure.** It is permanent for the process: the requests it held have already
  failed, so the server drains as after SIGTERM (bounded by `--drain-timeout-ms`) and exits with
  status 1. A supervisor that restarts on failure (systemd `Restart=on-failure`, a Docker restart
  policy, a Kubernetes container) then starts a fresh server; no liveness probe is needed for that.
  A failed CUDA call during generation (a device-side fault, an illegal address, a lost device) is
  such a failure, as in vLLM and SGLang. Every request in flight is answered HTTP 503
  `service_unavailable` with `Retry-After` (a stream that already started ends with an error
  event), and the process exits with status 1 after the drain.
- **Recoverable CUDA error.** A failed CUDA call the device survives (a failed allocation, a bad
  launch configuration) is not Engine-wide. The requests being generated are answered HTTP 503 with
  `Retry-After`, the Engine resets its sequence state and the context cache, and the queued
  requests go on. A second such error within five minutes is treated as Engine-wide, so a condition
  the reset did not cure ends in a restart.

### Request IDs

Every OpenAI-compatible response carries an `x-request-id` header, including streaming and
error responses: the client's own `X-Request-Id` when it sends one that is a plain token (letters,
digits, `-`, `_`, `.`, `:`; at most 128 characters), else one the server makes. Anthropic endpoints
use their separate `request-id` contract. The server's log names that id on each request's start
line and on every refused request (a 4xx before generation is logged too).

### Keep-alive and dead peers

- After five seconds without a protocol event, the OpenAI SSE endpoints emit the standard
  `: keep-alive` comment, and Anthropic Messages emits `event: ping` with `{"type":"ping"}`, as
  Anthropic does. Clients that time out on events rather than bytes stay live. The heartbeat is
  transport-only: it does not change generated text, semantic event ordering, usage, stored
  Responses, or request logs.
- On Linux, accepted connections also use TCP keepalive and a 15-second `TCP_USER_TIMEOUT`.
  Together with the heartbeat, a dead or unacknowledging peer is normally cancelled within about 20
  seconds, including while the request is waiting or prefilling.
- A peer whose TCP stack remains connected and acknowledges data cannot be distinguished from a
  reading application. Proxies must close their upstream NInferno connection when the downstream
  client disappears.

## 5. Models endpoint

`GET /v1/models` and `GET /v1/models/{id}` return the configured public OpenAI model alias
(default: the artifact's `metadata.name`, else its file name, overridable with `--model-id`). They also return the
effective `max_model_len` (the `--max-context` ceiling) and a `meta` object in the shape exposed by
`llama.cpp`. The `meta` facts describe the registered artifact behind the alias:

| Field | Meaning |
|---|---|
| `n_vocab` | tokenizer token domain |
| `n_ctx` | configured per-request context ceiling (equal to `max_model_len`) |
| `n_ctx_train` | model native/training context |
| `n_embd` | model embedding width |
| `n_params` | total logical weight elements across the registered artifact tensors |
| `size` | encoded weight payload bytes of the registered artifact |
| `ftype` | encoded tensor formats of the artifact, joined with `+` when mixed |

`GET /v1/models/{id}` returns the same object for the single configured alias and `404` for any
other id.

The list also carries an empty top-level `models` array. Codex decodes that field as its own
backend's model catalog on every start. With it present the refresh succeeds and Codex uses its
fallback metadata for the alias instead of logging a decode error.

## 6. OpenAI Chat Completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is speculative decoding?"}
    ],
    "max_tokens": 128
  }'
```

The request `model` may name anything: a server serves one model, so a client's own default
(`gpt-4o`, an alias the user configured) means it. Responses carry the public model ID: the
artifact's `metadata.name`, else its file name by default, or the explicit `--model-id` override.
Every endpoint also answers without its `/v1` prefix (`/chat/completions`, `/models`), for
llama.cpp-style base URLs.

### Supported request features

**Messages and content**

- `system`, `developer`, `user`, `assistant`, and `tool` history, plus legacy `function` history.
- String content and ordered text/refusal parts. Adjacent parts are preserved without inserted
  separators, and empty wire content remains an empty turn.
- User `image_url` parts, tool-result `image_url` parts used by compatible clients, and the User
  `video_url` extension, using HTTP(S) or data URIs. Image detail `auto`, `low` and `high` (or
  omitted) all take the fixed Vision preprocessing profile.
- Assistant `reasoning_content` and `reasoning` history aliases.
- Assistant tool-call history, tool-result messages, and legacy function-call history. Results
  returned out of order (by clients that run the calls concurrently) are rendered in the order of
  their calls when their `tool_call_id`s all name calls of that assistant turn.

**Output control and sampling**

- `max_completion_tokens` and the legacy `max_tokens` spelling. Zero performs prompt processing
  without generation; a negative value (`-1`, how llama.cpp-style clients say "no limit") is the
  server's `--default-max-tokens`, as when the field is absent.
- `temperature`, `top_p`, presence/frequency penalties, and integer `seed`; a negative seed (`-1`)
  is a fresh one, as when the field is absent (`--seed` when set). A seed repeats a
  sampled answer under `--activations exact`. Under `fast`, a prompt prefix taken from the cache, or
  batch neighbours, can change the last bits of the logits and so a later draw (Nemotron 3.5: 1 of
  48 repeats; none with `--no-prefix-reuse`).
- The compatible `top_k` (`0..64`) and `min_p` (`0..1`) sampler extensions.
- Up to sixteen stop strings, applied to the answer only (a stop never ends a thinking model's
  reasoning). Empty strings mean none and are dropped.
- The thinking switch as `enable_thinking`, `chat_template_kwargs.enable_thinking`, Ollama's `think`
  and `thinking: {"type": "enabled" | "disabled"}`; conflicting values are refused. A chat template
  that renders the same prompt either way (gpt-oss's harmony template, which always reasons) has no
  switch: the value is accepted and changes nothing; use `reasoning_effort` there.
- The boolean `ignore_eos` extension. `true` suppresses the checkpoint's own stop tokens so
  generation runs to the requested token budget. Caller-supplied stop strings and stop token ids
  still apply. Omitted or `false` keeps them. A constrained output (a forced or strict tool call, a
  `json_schema` response) still ends when its grammar completes.
- `n:1`, text-only `modalities`, and `response_format: {"type":"text"}`.

**Streaming and observation**

- Non-streaming responses and server-sent event streams.
- `stream_options.include_usage`, optionally shaped by `--usage-chunk-choice` for strict client
  parsers.
- llama.cpp-compatible terminal `timings`, plus opt-in `timings_per_token` and streaming
  `return_progress` observations ([below](#llamacpp-compatible-request-observations)).

**Tools**

- Function tools and free-form `custom` tools. A custom tool is served to the model as a
  single-string-input function under the caller's own tool name, so callers that dispatch by name
  keep working. A call to one is answered as `{"type": "custom", "custom": {"name", "input"}}` and
  read back from history in that shape. Its declared `format` (text, or a grammar given as
  `{"grammar": {"definition", "syntax"}}`) is prompt metadata only.
- `tool_choice` `auto`/`none`, and `required`, named-function, `custom`, or function-only
  `allowed_tools` selections. The forcing ones are enforced by a tool-call constraint on decoding
  (after any reasoning, a forced answer is the call or calls, with no text before them). `none`
  keeps the declarations in the prompt, as vLLM does by default, so a turn without calls (an
  agent's compaction or summary) reuses the conversation's cached prefix. No output is read back
  as a call then.
- `strict:true`, whose arguments the same constraint holds to the declared JSON Schema, and
  `parallel_tool_calls:false`, under which generation stops after the first call.
- A JSON `response_format` (`json_schema` or `json_object`) beside enabled tools, as Agent SDKs
  send. The answer is the calls or a text answer in that format, nothing else. When a call is
  required or named, the answer is calls only and the format has nothing to apply to. A regex,
  grammar or choice format beside enabled tools is rejected.

**Thinking**

- The top-level `reasoning_effort` field, where the `default` and `auto` aliases mean the
  template's own default (as if omitted).
- `enable_thinking` and `preserve_thinking`, either at top level or in `chat_template_kwargs`.

### Rejected options

Options whose observable behavior the Engine cannot provide are rejected when they request that
behavior:

- JSON constrained output outside the cases above;
- nonzero `logit_bias`;
- requested log probabilities;
- audio/file input or audio output;
- image detail levels other than `auto`, `low` and `high`;
- web search, moderation, low/high verbosity;
- stored Chat Completions;
- non-empty legacy `functions`.

Each capability rejection identifies the affected field and the guarantee NInferno cannot provide.
Known constrained-decoding aliases (`grammar`, `structured_outputs`, `guided_json`, `guided_regex`,
`guided_choice`, and `guided_grammar`) receive the same explicit rejection instead of being treated
as unknown hints.

### Accepted but neutral fields

- All-zero `logit_bias`, `logprobs:false`, `top_logprobs:0`, `verbosity:"medium"`, empty legacy tool
  controls, text-only `audio` configuration, and `prediction` are accepted without changing Engine
  execution.
- Metadata, user/safety identifiers, service-tier and prompt-cache hints are advisory.
- `repetition_penalty` is accepted only at its neutral value `1`. `mm_processor_kwargs` is accepted
  when empty or containing only null values. String-form image/video URLs are also accepted.
- Unknown top-level fields are accepted and have no effect. So that a misspelt option
  (`max_token`) does not pass unnoticed, the response carries
  `X-NInferno-Warning: unknown fields ignored: <names>` (Chat Completions and Anthropic Messages;
  the names are escaped to printable ASCII), and the request log records the same note.
- A string `name` on any message is accepted as an ignored, output-neutral compatibility hint:
  clients send it on user and assistant turns and mirror the function name onto tool results. It
  does not take part in tool identity, prompt rendering, or output, because the loaded chat
  templates cannot represent participant identity. Non-string values are malformed.

### Tool names and history arguments

- Tool and function names are 1 to 256 bytes of ASCII letters, digits, `_`, `-`, `.`, `:` and `/` on
  every endpoint, in declarations and in replayed calls alike. MCP and namespaced names such as
  `github.create_issue` or `fs:read` therefore work. A generated call is parsed by the same rule.
- A history tool call's `arguments` (in `tool_calls` or the legacy `function_call`) is a JSON string
  by the protocol. Some clients send the JSON object itself, which is taken as its serialized text.
  Arguments of any other type are malformed.
- Text that is not a JSON object (a call cut off or garbled by a model, possibly another backend's)
  is accepted in Chat, Responses and prompt files and reaches the template as `{"raw": "<text>"}`.
  Refusing it would fail every later request of the session.

### Errors

- Malformed protocol values return field-specific HTTP 400 errors.
- Invalid media sources, bytes, or decoded content use `invalid_media`.
- A media URL that is itself wrong (a host that does not exist, an HTTP 4xx answer, a redirect
  without a location or past the limit) is a 400 `invalid_media_url`. A remote server that cannot be
  reached or answers 5xx is a 502 `media_fetch_failed`. A fetch that runs out of time is a 504
  `media_fetch_timeout`.
- Failures in the normalized prompt contract use `invalid_prompt`.
- Typed capacity and availability failures retain their dedicated codes.
- Internal invariant failures are not relabeled as client input errors.

See also [error codes](#error-codes-summary).

### Reasoning output

- Reasoning is returned separately as `reasoning_content`; answer text remains in `content`.
- A reasoning close that the model writes right after a quote mark or backtick, and follows with
  anything but a line break (it is quoting the marker, in code or prose about the format), stays
  reasoning. A close followed by a line break, or one that ends the output, ends it.
- Under an output constraint (`response_format`, strict tools, `tool_choice` required or named, or
  `parallel_tool_calls: false`) the first close ends the reasoning, as the constraint's grammar opens
  there.

### Tool-call parameter conversion (Qwen)

This applies across Chat Completions, Responses, and Anthropic Messages.

- A direct top-level tool-parameter `type`, or an `anyOf`/`oneOf` composed entirely of explicit
  primitive types, guides conversion of Qwen's untyped parameter text. It does not decide whether
  structurally complete markup is a tool call.
- String-admitting values remain strings, including the empty string. An empty block for a declared
  non-string parameter is omitted.
- Admitted JSON values retain their JSON type. Case-insensitive boolean text is normalized to `true`
  or `false`.
- A nonempty schema mismatch remains a structured call: valid JSON retains its represented type and
  other text becomes a JSON string, so the tool consumer can report the validation error and
  continue the agent loop.
- Schemas without a supported explicit type retain untyped inference. Text that is JSON keeps its
  JSON type. Other text is a string with only the framing newlines removed, so indentation and a
  final newline remain.
- NInferno does not apply defaults, enforce required properties, perform recursive JSON Schema
  validation, or use constrained decoding for these conversions.

### Tool-call parsing details (Qwen)

- **Nested markers.** String parameters preserve function/tool-call markers and balanced nested
  `<parameter=...>...</parameter>` text as value bytes. The Qwen wire format has no delimiter
  escape. A standalone `</parameter>` that is not followed, after whitespace, by `<parameter=` or
  `</function>` is value text. A string value therefore cannot contain `</parameter>` followed by
  whitespace and then `<parameter=` or `</function>`, nor an unmatched nested parameter opener.
  Either causes the complete tool-call region to fall back to ordinary content.
- **Repeated parameter.** A parameter repeated within one call keeps its last value, as a repeated
  JSON key does.
- **Quoted opener.** When text quotes the tool-call opener before the real call, the region
  starting at that quote does not parse. If that first opener does not begin a complete call, the
  parser tries the last four later openers, earliest first. The first one that parses completely
  supplies the tool calls, and the text before it is returned as ordinary content. If none parses,
  the whole output falls back to content with the first opener's fallback reason.
- **Trailing text.** A complete first call followed by text is not a quote. The calls are kept and
  that text is returned as content after the text before them (vLLM keeps both too), counted in
  `trailing_content_bytes`. When the text after the calls opens another call, the whole output
  falls back as `trailing_content`.
- **Bare blocks and unclosed calls.** A Qwen 3.x model now and then writes its calls as bare
  `<function=NAME>...</function>` blocks without `<tool_call>`, or ends its turn with the stop token
  after a call's last `</parameter>`, before `</function>`. As vLLM does, bare blocks are calls when
  every block names a declared tool (counted in `unwrapped_calls`), and such a last call is closed
  (`closed_calls`). A call the output limit cut off is not closed, since more of its parameters may
  have followed. Neither applies to a call recovered from a thinking span the stop token left open.
- **Undeclared names.** See [Responses function tools](#function-tools) and `undeclared_calls` in
  [the request log](#tool-call-diagnostics).

### Chat template and message order

- `--chat-template FILE` selects a local Jinja template. By default the server uses the template
  stored in the artifact. See the [CLI guide](../cli.md#input-and-output) for an example.
- Messages enter the selected template in their input order. The maintained Qwen templates keep
  system/developer messages at their original positions.
- A stock template that refuses a system or developer message after the first ("System message must
  be at the beginning.") renders the conversation again, with each of them after the first as a
  user message. This also applies when several open the conversation (two system messages; a
  Responses `instructions` with a developer message). So does a template that would drop such a
  message after the conversation began without an error (one Qwen 3.6 template renders only the
  leading ones), so its text is never lost.
- Anthropic Messages accepts `role: "system"` entries anywhere in `messages` (proxies such as
  LiteLLM send them first or between user turns), after the top-level `system`. A run between an
  assistant's `tool_use` and its `tool_result` turn moves after that turn, as LiteLLM moves it.
- Control-token spellings quoted in message content, tool data or ordinary template kwargs are
  encoded as text. Media placeholders come from the template and bind to actual image/video inputs.
- `chat_template_kwargs` passes a JSON object to the template in Chat Completions, Responses and
  Anthropic Messages. Values duplicated in typed request fields must agree. Null standard options
  mean unspecified; other null values remain `none`. Messages, tools, generation mode and tokenizer
  special tokens cannot be overridden through kwargs.
- A completed assistant turn in the history with no text, no media and no tool calls (only
  reasoning, or nothing) is not rendered, in every API: a model imitates such turns
  and ends later turns without the call it was asked for. The final message is always rendered.

### JSON member order

Prompt-bearing JSON objects retain their received member order through request parsing and prompt
rendering, including tool schemas and historical tool inputs. Canonical model-origin tool arguments
retain that member order in aggregate and streaming responses, so an unmodified replay reconstructs
the same ordered tool call. NInferno does not canonicalize semantically equivalent JSON. If a client
reorders members, inserts defaults, or otherwise rewrites a tool object, the changed rendered input
does not match the model-held endpoint and can reuse only an earlier exact checkpoint.

### Thinking controls

- `--default-thinking-budget N` sets a positive default thinking-token cap for requests that think,
  on models whose chat template defines how to end thinking early (Qwen, Gemma 4;
  gpt-oss's template does not, and there the default is not applied, while a request's own budget
  is refused). Non-thinking requests receive no cap. It may coexist with `--no-thinking`, because
  requests can explicitly enable thinking. Anthropic `thinking:{"type":"enabled","budget_tokens":N}`
  overrides this default for that request. For example, add `--default-thinking-budget 512` to the
  startup command to cap model-origin thinking at 512 tokens for every thinking-enabled request.
  Where the prompt starts the thinking (Qwen), the cap counts from the first output token; where the
  model opens it itself (Gemma 4 writes `<|channel>thought`), it counts from the token after that
  opener, and an answer that begins without one is not capped.
- **At the cap.** The Engine first honors a natural `</think>`, stop condition, cancellation, or
  total output/context limit. If thinking remains open, it commits the template's early-close
  control (Qwen's canonical guidance and close marker; Gemma 4's channel close) to the same model sequence without sampling, streams the guidance as a
  reasoning delta, and continues normal content or tool-call generation. Inserted tokens count in
  completion usage and the request's `max_tokens`/`max_output_tokens` budget.
- **No room.** If the effective output capacity extends past the cap but cannot fit the complete
  tokenizer-derived control suffix plus one post-close model token, preparation is rejected with
  HTTP 400 `thinking_budget_capacity_insufficient` rather than partially inserting control. Where the
  model opens its thinking itself, that is only known once it has, so there the output limit ends the
  thinking instead of a refusal. The
  server does not promise that the model will emit nonempty content or a tool call after the marker.
- **Effort.** `reasoning_effort: "none"` requests disabled thinking. The selected template
  interprets the other standard values (`minimal`, `low`, `medium`, `high`, `xhigh`, `max`). A value
  the template does not take renders at the next stronger level it takes (Qwen 3.8 takes `low`,
  `medium` and `xhigh`: `minimal` runs as `low`, `high` as `xhigh`), never a weaker one. An effort
  passed in `chat_template_kwargs` is the template's own input and is not changed. Conflicting
  explicit `enable_thinking` and effort values return `conflicting_template_option`.
- **Preserve.** `preserve_thinking` controls reasoning retention according to the selected
  template. Request options override server defaults set with `--no-thinking` and
  `--preserve-thinking`. Unspecified thinking, effort and preservation options use the template's
  defaults.

### Streaming format

- A stream opens with an SSE comment (`: stream`), which clients skip.
- The assistant role travels in the first chunk's delta: with the first token, as vLLM, SGLang and
  llama.cpp send it, or with the first prompt-progress chunk when `return_progress` is set. A client
  that times the first chunk therefore measures prefill.
- Reasoning and content deltas are separate, then a finish-reason chunk and `[DONE]`.
- An error after the stream started is a `data:` chunk with the error object, then `[DONE]`.
- When `stream_options.include_usage` is true, a final empty `choices` chunk contains completed
  usage.
- Aggregate and streamed usage include cached prompt tokens and reasoning-token details. Choices
  carry `logprobs: null` when log probabilities were not requested. Aggregate assistant messages
  carry `refusal: null`, because refusal output is not supported.

### llama.cpp-compatible request observations

Every successful Chat Completions response includes a top-level `timings` object. This is a
llama.cpp-compatible response extension, not an OpenAI field. In a stream it is attached to the last
JSON chunk before `[DONE]`: the empty `choices` usage chunk when `stream_options.include_usage` is
true, otherwise the finish-reason chunk.

```json
{
  "timings": {
    "cache_n": 4096,
    "prompt_n": 4096,
    "prompt_ms": 83.0,
    "prompt_per_token_ms": 0.020263671875,
    "prompt_per_second": 49349.39759036145,
    "predicted_n": 129,
    "predicted_ms": 1140.0,
    "predicted_per_token_ms": 8.90625,
    "predicted_per_second": 112.28070175438596
  }
}
```

- `cache_n` is the exact Engine-proven reused prompt prefix and `prompt_n` is the remaining prompt
  suffix, so `cache_n + prompt_n` equals `usage.prompt_tokens`.
- Prompt time starts when admission commits that exact reuse choice and ends when the first output
  token is committed. Generation time starts at that first token and ends at the last committed
  output token. Generation speed therefore uses `max(predicted_n - 1, 0)` token intervals: the first
  token belongs to prompt latency and is not counted again as a decode interval.
- Zero-token, one-token, zero-duration, and exact-cache-hit cases report finite zero rates rather
  than `NaN` or infinity.
- Speculative requests additionally include terminal `draft_n` and `draft_n_accepted` when draft work
  occurred.

**`timings_per_token`.** Set top-level `timings_per_token: true` on a streaming request to attach
the latest cumulative timing snapshot to each visible reasoning or content chunk and, when
`return_progress` is also enabled, to each prompt-processing chunk.

- During prompt processing the snapshot is prompt-only: `predicted_n`, `predicted_ms`,
  `predicted_per_token_ms`, and `predicted_per_second` remain zero until the first output commit.
  `prompt_ms` is the elapsed wall time since committed admission. The prompt rates span the
  processed suffix (`processed - cache`) tokens rather than the complete prompt.
- This does not enable terminal timings, which are always present.
- A model commit that is temporarily hidden by UTF-8, stop-string, reasoning, or tool-call buffering
  still advances the cumulative token count. The next visible chunk observes that committed
  frontier.
- The option increases response serialization and transport volume and is off by default.

**`return_progress`.** Set top-level `return_progress: true` together with `stream: true` to receive
prompt-processing chunks:

```json
{
  "prompt_progress": {
    "total": 8192,
    "cache": 4096,
    "processed": 6144,
    "time_ms": 41
  },
  "timings": {
    "cache_n": 4096,
    "prompt_n": 4096,
    "prompt_ms": 41.5,
    "prompt_per_token_ms": 0.020263671875,
    "prompt_per_second": 49349.39759036145,
    "predicted_n": 0,
    "predicted_ms": 0.0,
    "predicted_per_token_ms": 0.0,
    "predicted_per_second": 0.0
  }
}
```

- The `timings` object appears only when `timings_per_token: true` is also set. It reports the same
  prompt-only live snapshot as above. `time_ms` is integral milliseconds while `timings.prompt_ms`
  keeps sub-millisecond precision. The prompt rates cover the 2,048 processed suffix tokens so far.
- The initial event has `processed == cache`. Later cumulative events are published only after the
  corresponding prefill unit commits, may be coalesced when the consumer is slower than prefill, and
  never move backwards.
- The final event has `processed == total` and precedes the first output delta. For an exact
  full-prefix hit, the initial event already has `cache == processed == total` and no synthetic
  prompt work is reported.
- `time_ms` is elapsed wall time since committed admission. Clients may calculate actual suffix
  progress as `(processed-cache)/(total-cache)` when the denominator is nonzero.

### Multimodal request

Start the server with `--vision` before sending media:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }],
    "max_tokens": 128
  }'
```

OpenAI image and video sources may be HTTP(S) URLs or base64 data URLs.

**Limits.**

- Text and media requests use one complete-prompt context contract. After chat-template rendering
  and media-token expansion, the result must fit Engine `--max-context`.
- The current Vision runtime also has a 32,768 merged-token envelope (131,072 raw patches). The
  effective Vision limit is `min(--max-context, 32768)`.
- There is no fixed image/video item-count limit. Item count is admitted through aggregate
  source-byte, decoded-pixel, raw-patch, Vision-token, and live-memory budgets.

**Media cache.**

- Cache misses run as independent decode, resize and BF16-pack tasks on a bounded host worker pool.
- Prepared payloads are keyed by SHA-256 of the acquired bytes plus modality. Repeated media in later
  requests reuses the exact immutable BF16 patch input. Concurrent identical misses use one
  single-flight build.
- `--media-cache-mib` bounds LRU-retained payloads. `--media-live-mib` bounds every cache-,
  request-, or runtime-referenced payload.
- Cache eviction does not invalidate a request reference. Live bytes are returned only when the final
  reference is released.
- A request-level preparation gate derived from the live limit prevents concurrent partial builds
  from deadlocking the memory account.

**Size errors.**

- An expanded prompt beyond `--max-context` returns HTTP 400 `context_length_exceeded`, including the
  prepared token count and configured context ceiling. Its message starts with
  `prompt is too long: the request exceeds the available context size`, the phrase agent harnesses
  (Claude Code, Pi and others) match to compact the conversation and retry.
- A media preprocessing resource rejection returns HTTP 400 `media_budget_exceeded`.
- HTTP 413 `request_too_large` is reserved for a raw request body that exceeds `--max-request-mib`
  before JSON parsing. It is not used for model-context or media resource errors.
- Request bodies are read uncompressed: a `Content-Encoding` such as `gzip` returns HTTP 415
  `unsupported_content_encoding`.

## 7. OpenAI prompt caching

Chat Completions and Responses translate OpenAI cache hints into optional shared-prefix write
candidates:

| Hint | Effect |
|---|---|
| `prompt_cache_options` omitted | a default implicit candidate at the latest representable content boundary |
| `mode:"implicit"` | the same automatic candidate, requested explicitly |
| `mode:"explicit"` | disables that implicit write for the request |
| `prompt_cache_breakpoint:{"mode":"explicit"}` on supported content | an explicit candidate |

- One request carries at most four distinct writes. An implicit target occupies one slot unless it
  coincides with an explicit target. The remaining slots contain the latest explicit boundaries.
- Earlier schema-valid historical breakpoints are accepted but are not new write candidates. Exact
  reads of already-published prefixes do not require the request to repeat a marker.
- These fields are optimization hints. A legal boundary that cannot be represented as an exact
  rendered-token frontier is ignored without changing prompt content.
- `prompt_cache_key` (at most 256 bytes) is not an Engine session key or prefix identity;
  `safety_identifier` may carry up to 128 characters.
- Valid TTL/retention values are accepted, but NInferno does not promise their wall-clock residency.
  Physical retention follows the resource scheduler.

### Automatic system/developer shared prefix

OpenAI-protocol requests publish an automatic shared-prefix candidate at the last content. Agent
clients (Hermes, OpenCode, Copilot, MCP hosts) rarely send `prompt_cache_breakpoint`, so a new
session that shares only the system/developer head with an earlier one would prefill that head
again.

With automatic writes on, the server also publishes a `DefaultAutomatic` candidate at the end of the
contiguous leading system/developer run (Chat Completions `messages`, Responses `instructions` and
leading Items). Sibling sessions reuse that head without a client marker. The candidate takes one of
the four marker slots: when a request also carries the most explicit breakpoints, the oldest
explicit one is dropped, as for the last-content candidate. `prompt_cache_options.mode: "explicit"`
never adds it. `--no-auto-system-shared-prefix` turns it off for both protocols.

## 8. OpenAI Responses Core

NInferno implements the typed-Item and semantic-event core of the OpenAI
[Responses API](https://developers.openai.com/api/reference/resources/responses/overview). All
supported model instances use this same adapter and Engine route. It is intentionally not
advertised as full parity with OpenAI-hosted tools, durable cloud storage, background jobs,
Conversations, or compaction.

### Create a Response

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "instructions": "Answer concisely.",
    "input": "What is speculative decoding?",
    "max_output_tokens": 128,
    "store": true
  }'
```

The same endpoint works with OpenAI SDKs by replacing their base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-secret")
response = client.responses.create(
    model="qwen3.8-27b",
    instructions="Answer concisely.",
    input="What is speculative decoding?",
    max_output_tokens=128,
)
print(response.output_text)  # SDK helper derived from response.output
```

`output_text` is an SDK convenience property. It is not emitted as a top-level wire field. The wire
response contains typed `output` Items.

### Create request fields

| Field | NInferno Responses Core contract |
|---|---|
| `model` | required non-empty string; any name means the served model |
| `input` | string or typed Item array; it may be omitted or empty only when `previous_response_id` already supplies a user query |
| `instructions` | optional string, inserted before the reconstructed conversation for this request only |
| `previous_response_id` | optional ID of a retained local Response |
| `max_output_tokens` | non-negative integer; omission executes with `--default-max-tokens` but remains `null` in the Response object |
| `stream` | boolean; `true` selects Responses SSE rather than a JSON body |
| `store` | boolean, default `true`; controls local retrieval and continuation state |
| `temperature` | finite number in `[0,2]` |
| `top_p` | finite number in `[0,1]` |
| `metadata` | at most 16 string pairs; keys at most 64 characters and values at most 512 |
| `client_metadata` | Codex client extension; an object or `null`, accepted as opaque tracing metadata with no generation effect |
| `reasoning.effort` | `none` requests disabled thinking; other standard effort values pass to the selected template |
| `reasoning.summary` | omitted, `null`, or any string; every string requests the same fixed protocol placeholder without changing model execution, and the original value is echoed in the response |
| `chat_template_kwargs` | template parameters as a JSON object; standard options merge with typed fields |
| `preserve_thinking` | alias for `chat_template_kwargs.preserve_thinking`; conflicting values are rejected |
| `text.format` | omitted or `{"type":"text"}` only |
| `tools` | direct function definitions or namespace groups containing function definitions; see [Function tools](#function-tools) |
| `tool_choice` | `auto`, `none`, or function-only `allowed_tools` with mode `auto`; a namespaced selection carries both `namespace` and `name` |
| `parallel_tool_calls` | `true` by default; `false` is accepted only when no effective tool is callable |
| `max_tool_calls` | non-negative integer accepted as a hosted-tool no-op; NInferno does not execute hosted tools |
| `truncation` | omitted or `disabled`; overlong input fails instead of silently dropping Items |
| `top_logprobs` | omitted or `0` |
| `service_tier` | omitted, `auto`, or `default`; the response reports `default` |
| `background` | omitted or `false` |
| `include` | omitted, empty, or `["reasoning.encrypted_content"]`; the supported value requests the local raw-reasoning mirror described below |
| `stream_options.include_obfuscation` | optional boolean; accepted as a transport hint, but this local server emits no padding |
| cache and client hints | `prompt_cache_key`, `prompt_cache_options`, `prompt_cache_retention`, and explicit breakpoints follow [OpenAI prompt caching](#7-openai-prompt-caching); `safety_identifier` and `user` are accepted as client hints |

Unknown top-level fields fail with `unknown_parameter`. Recognized but unsupported features fail
with a field-specific 400 error instead of being silently ignored. When a rejection is caused by a
specific value, the error names that value (ASCII-escaped and truncated), its size where relevant,
and its location in the request. The location appears both in the error `param` (for example
`tools[0].function.name` or `messages[2].tool_calls[0].function.name`) and in the message text.

Unsupported Create fields include Conversations, prompt templates, context management, hosted
moderation, Structured Outputs/JSON mode, `include` values other than `reasoning.encrypted_content`,
background execution, compaction, files/audio, and OpenAI-hosted and MCP tools. Except for the two
explicitly documented placeholders (`reasoning.summary` and the local raw-reasoning mirror), these
are compatibility boundaries rather than silently accepted approximations.

### Input Items

String `input` is normalized to one user `message` with an `input_text` part. Array input accepts:

| Item | Supported form |
|---|---|
| `message` | roles `user`, `assistant`, `system`, and `developer`; string content or typed content array |
| `input_text` | message content part containing string `text` |
| `output_text` | assistant-message replay part containing string `text` |
| `refusal` | assistant-message replay part; its text enters assistant history |
| `input_image` | user- or assistant-message part with HTTP(S) or data-URI `image_url`; detail omitted, `auto`, `low` or `high` (one fixed preprocessing profile); requires server `--vision` |
| `input_video` | NInferno extension with HTTP(S) or data-URI `video_url`; requires server `--vision` and a model that takes video (Qwen 3.5/3.6/3.8; Gemma 4 takes images only) |
| `reasoning` | raw replay Item with `reasoning_text` content; summary/encrypted metadata may accompany raw text. An Item without raw text (a session carried over from OpenAI: summary and another server's `encrypted_content`) is echoed but left out of the prompt, with a note in the request log |
| `function_call` | completed assistant call with optional `id` and namespace, plus required `call_id`, `name`, and JSON-object string `arguments` |
| `function_call_output` | completed result with required `call_id` and optional matching name/namespace assertion; `output` may be a string or a non-empty array of `input_text`/`input_image` parts |

`input_file` with `file_data` (a data URI or base64 of a PDF) or `file_url` is read as a PDF
document ([PDF documents](#pdf-documents)); `file_id` is not supported.

Not supported: `input_audio`, image and file `file_id`, image detail other than
`auto`/`low`/`high`, partial tool Items, and other Item/content types. HTTP media URLs stored in a
response chain are fetched again when that chain is continued. Use data URIs when the historical
media bytes must be immutable.

**History assembly.**

- Contiguous assistant-owned Items form one assistant history turn in the representable order
  `reasoning` -> assistant message content -> `function_call`. Multiple message Items append their
  content parts, multiple calls retain declaration order, and a reasoning-only turn is retained.
- A `reasoning` Item between calls (a model that reasons before each parallel call) joins the turn.
  Its text follows the earlier reasoning, separated by a blank line, ahead of all the calls.
- A user, system, developer, or `function_call_output` Item ends the group. An order that would
  require rearranging assistant content fails with `invalid_assistant_history`.
- Results are validated by `call_id` and reordered to call declaration order before prompt
  rendering. Unknown, duplicate, or unrepresentable partial result sets fail with
  `invalid_tool_history`.
- Canonical input Items retain client order. Input Item IDs are preserved when supplied and generated
  otherwise. Duplicate IDs fail.
- System and developer message Items retain their positions in the input array. Top-level
  `instructions` is represented as a leading developer turn for the current request. Target-specific
  role lowering occurs only in the Qwen family frontend.
- An `input_text`, `input_image`, or tool-result part may carry
  `prompt_cache_breakpoint:{"mode":"explicit"}`. Write selection follows
  [OpenAI prompt caching](#7-openai-prompt-caching). Boundaries affect reuse opportunities, not
  prompt identity or output semantics. String message status/phase metadata is accepted but has no
  Qwen prompt representation.

### Function tools

Responses function definitions may be declared directly rather than inside Chat Completions' nested
`function` object:

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get current weather",
  "parameters": {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"]
  },
  "strict": false
}
```

They may also be grouped in a Responses namespace:

```json
{
  "type": "namespace",
  "name": "mcp__weather",
  "description": "Weather service",
  "tools": [{"type": "function", "name": "get_current"}]
}
```

- NInferno gives each namespace/function pair a distinct internal Engine identity and restores the
  separate `namespace` and `name` fields in aggregate output, SSE events, and replayed Items. The
  same function name may therefore appear in different namespaces. Namespace members remain
  ordinary client-executed functions; this does not add a remote MCP executor.
- NInferno renders these definitions in the Qwen prompt and parses model output into separate
  `function_call` output Items. Each output has a protocol Item `id` (`fc_...`) and a distinct
  `call_id` (`call_...`). The client executes the function and sends a `function_call_output` Item
  in a later request.
- A call whose name is not in the current effective tool set is still published as a call, as vLLM
  and llama.cpp do. The client answers it as an unknown tool and the model recovers, where text
  would end the agent's turn and lose the parallel calls beside it. Declared names pick the call
  region first, so a quoted example of an undeclared call does not displace a declared call after
  it. Llama 3's JSON form, whose call opens like any JSON answer, still needs a declared name.
- `allowed_tools` with mode `auto` filters that set without changing declaration order.
  `tool_choice:"none"` disables structured tool output even when the history contains earlier calls.
- Function definitions are request configuration rather than conversation Items. They must be sent
  again on tool-result turns.

**Hosted, custom and deferred tools.**

- NInferno does not execute tools, so hosted tools and remote MCP tools are rejected, except a hosted
  `web_search` tool: Codex declares one in every request, so it is left out of the prompt (a
  `tool_choice` that forces it is still rejected).
- A custom free-form tool (Codex declares its own built-ins, such as `apply_patch`, this way) is
  served to the model as a single-string-input function under its own name. Its call is answered as
  a `custom_tool_call` Item whose `input` is that string, streamed as
  `response.custom_tool_call_input.delta`/`.done`. `custom_tool_call` and `custom_tool_call_output`
  history Items are read back. `tool_choice` `{"type": "custom", "name"}` forces it. Its declared
  `format` (text, or a grammar with `syntax` and `definition`) is prompt metadata only.
- Deferred loading, output schemas, and caller restrictions that exclude direct invocation are
  rejected because their semantics cannot be honored.
- Tools may also arrive as `additional_tools` input Items (`{"type": "additional_tools", "tools":
  [...]}`, as Codex sends them). Their tools join the request's `tools`, in input order, and the
  Item makes no message. A later Item may define a tool or namespace again. The latest definition
  replaces the earlier one where it stands, and a namespace keeps the description it was first
  declared with.

### Response object and usage

A terminal wire response has `object: "response"`, one of `completed`, `incomplete`, or `cancelled`
in `status`, and a typed `output` array. NInferno may emit:

- a `reasoning` item containing raw `reasoning_text` (with a placeholder summary if
  `reasoning.summary` is requested);
- an assistant `message` containing an `output_text` part;
- one or more `function_call` Items.

When `include:["reasoning.encrypted_content"]` is requested, reasoning Items also carry an
`encrypted_content` equal to their raw `reasoning_text`. This field is **not encrypted** and
provides no confidentiality.

Status:

- Ordinary model/string stops produce `completed`.
- Output-token or context-capacity exhaustion produces `incomplete` with
  `incomplete_details.reason: "max_output_tokens"`.
- Errors accepted after an SSE response has started produce `response.failed`. Validation and
  preparation errors remain normal HTTP error responses.
- `completed_at` is populated only for completed Responses.
- A reasoning-only incomplete result contains no invented empty assistant message.

Usage is checkpoint-native:

```json
{
  "input_tokens": 42,
  "input_tokens_details": {"cached_tokens": 17},
  "output_tokens": 12,
  "output_tokens_details": {"reasoning_tokens": 5},
  "total_tokens": 54
}
```

- `input_tokens` includes the chat template and expanded media tokens.
- `cached_tokens` is the exact checkpoint-proven prompt prefix reused by the Engine.
- `output_tokens` is the count of accepted generated token IDs, including a withheld stop token when
  applicable.
- `reasoning_tokens` is counted in the Qwen output decoder while accepted tokens are still in the
  reasoning channel. It is not estimated by re-tokenizing decoded text. A reasoning channel that
  carried no text counts 0, because its markers are not reasoning. (Gemma 4 opens and closes an
  empty thought after a tool result with thinking off.)

### Responses streaming

Set `stream:true` for semantic Server-Sent Events. Every frame uses both the SSE event name and a
matching JSON `type`, and every JSON event has a monotonically increasing `sequence_number`:

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":7,...}

```

The normal lifecycle is:

1. `response.created`, then `response.in_progress`;
2. `response.output_item.added` and `response.content_part.added`;
3. zero or more `response.reasoning_text.delta` or `response.output_text.delta` events;
4. matching `*.done`, `response.content_part.done`, and `response.output_item.done` events;
5. exactly one `response.completed`, `response.incomplete`, or `response.failed` terminal event.

- **Summary events.** For a reasoning Item requested with any string-valued `reasoning.summary`, its
  `response.output_item.added` and `.done` payloads carry the same placeholder summary. Immediately
  after the Item is added, the stream emits `response.reasoning_summary_part.added`,
  `response.reasoning_summary_text.delta`, `response.reasoning_summary_text.done`, and
  `response.reasoning_summary_part.done` with `summary_index:0`, then continues with the raw
  `reasoning_text` content lifecycle. Omitted or `null` summary requests emit none of these summary
  events and retain an empty Item `summary` array.
- **Encrypted-content mirror.** For `include:["reasoning.encrypted_content"]`, the in-progress
  `response.output_item.added` Item omits `encrypted_content`, because the complete reasoning text is
  not available yet. `response.output_item.done` and the terminal Response output contain the same
  complete raw mirror. A response with no reasoning Item emits no encrypted placeholder.
- **Function arguments.** They use `response.function_call_arguments.delta` and `.done`. IDs, output
  indices, and content indices remain stable, and concatenated deltas equal the terminal Item.
- **No `[DONE]`.** Responses SSE does not emit the Chat Completions `[DONE]` sentinel.
- **Tool markup.** With tools enabled, ordinary answer text still streams immediately. Only an
  ambiguous `<tool_call>` suffix or the structured tool region is held. Malformed tool markup is
  flushed back as ordinary text without losing bytes.

### Local response state and resources

`store` defaults to `true`. Stored Responses live only in this server process and are bounded by an
LRU store (`--response-store-max-records`, `--response-store-max-mib`). They are lost on restart and
are not OpenAI's durable cloud retention service.

- **`previous_response_id`** reconstructs the complete stored input/output Item history before the
  new input. The current `instructions` value is placed first but is not saved into the continuation
  context, matching the Responses rule that previous top-level instructions do not carry forward.
  The reconstructed prompt follows the ordinary Engine path, so compatible checkpoint reuse applies
  naturally.
- **`preserve_thinking`.** A stored Response also retains its resolved value. A child which omits the
  field inherits the parent value. An explicit different value creates a new semantic branch; prompt
  rendering and identity still determine reuse. Changing the boolean alone never invalidates an
  exact checkpoint already proved compatible by the model runtime.
- **Session key.** For Engine-local reuse, a stored root Response receives one bounded session key
  derived from its response ID, and every `previous_response_id` child inherits that key.
  `store:false` roots remain anonymous. A `store:false` child may read its inherited session
  checkpoint but does not replace the stored chain's latest endpoint. Response-store eviction or
  deletion removes the HTTP object, not an independently retained Engine checkpoint; the latter
  remains bounded by the Engine's own retention and pressure policy. No session key or cache marker
  is added to the HTTP schema.

Resource behavior:

| Endpoint | Contract |
|---|---|
| `GET /v1/responses/{id}` | returns the stored terminal object, or 404 `response_not_found`; stream recovery and non-empty `include` are rejected rather than ignored |
| `DELETE /v1/responses/{id}` | removes public retrieval and returns `response.deleted`; descendant contexts already retained by other Responses remain usable |
| `GET /v1/responses/{id}/input_items` | returns normalized Items supplied to that request; supports `after`, `limit` `1..100` (default `20`), and `order` `asc\|desc` (default `desc`); image URLs are redacted unless `include=message.input_image.image_url` |
| `POST /v1/responses/{id}/cancel` | explicitly fails because background execution is unsupported |
| `POST /v1/responses/compact` | explicitly fails with `compaction_not_supported` |

- `store:false` Responses cannot be retrieved or used as `previous_response_id`. LRU eviction and
  explicit deletion also make an ID unavailable.
- A request whose input and `previous_response_id` context alone exceed the configured store
  capacity fails with `response_store_capacity_exceeded` and status 400 before generation. One whose
  output then makes the Response larger than the store is answered unstored: the Response says
  `"store": false` (it cannot be retrieved or named as `previous_response_id`) and the server logs a
  warning, rather than discarding the generated answer.

### Responses input token count

`POST /v1/responses/input_tokens` uses the same prompt path as Create and does not run generation.
It accepts `model`, `input`, `instructions`, `previous_response_id`, reasoning, function tools and
tool choice, supported text/truncation values, and the `preserve_thinking` extension. Parent lookup,
call-ID normalization, template rendering, and media expansion are therefore identical to the
corresponding Create request.

```bash
curl http://127.0.0.1:8080/v1/responses/input_tokens \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","input":"Count this prompt."}'
```

```json
{"object":"response.input_tokens","input_tokens":11}
```

## 9. Anthropic Messages

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Explain prefix reuse in one sentence."}
    ]
  }'
```

The request `model` is any non-empty local proxy label. It is echoed in the response and does not
select the resident artifact.

### Messages and system text

The endpoint accepts top-level System text, ordered User/Assistant/System history, text and image
blocks, Thinking history, tool-use history, tool results, user-defined tools, aggregate responses,
and Anthropic SSE.

- Consecutive User or Assistant messages are joined without adding separators. Mid-conversation
  System messages retain their input position.
- A final text-only Assistant message is an Assistant prefill: generation continues its existing
  text instead of opening another turn. Assistant prefill cannot contain media, Thinking, or tool
  calls and cannot start with Thinking enabled.
- **Claude Code billing header.** Claude Code may place its attribution metadata in the first block
  of a top-level System array. If that block is a text block beginning exactly with
  `x-anthropic-billing-header:`, NInferno consumes the whole block before token counting, prompt
  preparation, and cache identity construction. The rule is positional: a string-form System value,
  a later array block, or an inline System message with the same text remains ordinary prompt
  content. A `cache_control` marker attached to the consumed block is consumed with it rather than
  moved to adjacent content.

### Limits and stops

- `max_tokens` is optional for local clients and otherwise uses `--default-max-tokens`. A positive
  value is the complete output budget. `max_tokens:0` is rejected, because NInferno does not
  expose a completed zero-output cache-prewarm lifecycle.
- `temperature`, `top_p`, `top_k`, and `stop_sequences` enter Engine execution. Chat `stop` and
  Messages `stop_sequences` take at most 16 strings; an empty one is ignored.
- A matched custom stop is returned as `stop_reason:"stop_sequence"` together with the actual
  `stop_sequence`. Context exhaustion returns `model_context_window_exceeded`.

### Thinking

- Thinking supports `disabled`, `adaptive`, and `enabled`. Enabled Thinking requires
  `budget_tokens >= 1024`, and that budget is passed to the Engine.
- Unlike the Anthropic API, a budget at or above `max_tokens` is accepted. The output limit ends
  thinking before the budget can, so the budget takes no effect. Clients such as Qwen Code send a
  fixed budget while shrinking `max_tokens` to the context left.
- Visible Thinking is returned with an opaque compatibility signature. SSE emits its
  `signature_delta` before closing the block. Request lowering reconstructs the local prompt from
  the visible `thinking` text and treats `signature` as non-semantic transport metadata, so retained
  history remains usable across serve restarts.
- `display:"omitted"` (Claude Code sends it in headless mode) returns Thinking blocks with empty
  `thinking` text and streams no `thinking_delta`. The reasoning travels base64url-encoded in the
  signature, and an echoed empty block with such a signature restores exactly that text, so the
  next prompt equals the visible-Thinking one. The signature is not encrypted: it hides the
  reasoning from display, not from the client.
- `preserve_thinking` remains a NInferno extension for closed-turn reasoning history.
  `output_config.effort` passes its protocol-validated value to the selected template.

### Tools

- User-defined, non-strict tools support `name`, `description`, object `input_schema`, and
  `input_examples`.
- `tool_choice` `auto`, `none`, `any`, and a named `tool` are executable. `any`, a named tool,
  `strict:true` arguments, and `disable_parallel_tool_use` attach a tool-call constraint that
  enforces them.
- Rejected, because their loader or executor is absent: deferred tools, tools that exclude direct
  model calls, Anthropic-provided/server tools, toolsets (MCP among them), and containers.
- `tool_result` preserves text/image order and marks `is_error:true` explicitly in the model prompt.
  For a visible Assistant tool-use turn, the next User turn must provide exactly one leading result
  for every declared ID. Valid results are matched by ID and normalized to call order. A history
  that begins with results remains valid as a truncated or imported conversation.

### Cache control

- Block-level ephemeral `cache_control` on tools and supported System/User/Assistant/tool-history
  blocks creates explicit shared-prefix candidates. At most four distinct block-level breakpoints
  are accepted.
- Request-level `cache_control` targets the last cacheable block. It merges with an explicit
  breakpoint at the same target and TTL, conflicts at the same target with a different TTL, and
  needs an available fifth slot when four different explicit targets already exist.
- TTL must be `5m` or `1h`. It is a protocol hint, not a wall-clock residency guarantee.
- NInferno maps representable boundaries to exact prompt frontiers and ignores a legal but
  unrepresentable advisory boundary without changing the prompt. Reuse still requires exact rendered
  identity and can read an existing owner without another `cache_control`.
- Aggregate usage reports verified reused tokens in `cache_read_input_tokens` and leaves cache
  creation unknown. Streaming emits `message_start` after Engine admission commits the prefix
  selection and before transfer/prefill output, so its uncached/cache-read split is already exact.
  Terminal cumulative usage matches the aggregate response.

### PDF documents

A PDF reaches the model as text: a `<document>` element with the title (the block's `title`, or the
OpenAI `filename`) and context, then one `<page number="N">` element per page with that page's
text layer. All three protocols take it: an Anthropic `document` block (base64 or URL source), an
OpenAI Chat `file` part (`file.file_data`) and a Responses `input_file`. The PDF counts against the
request's media byte budget; it needs no vision.

The text comes from pdfium, loaded when the first PDF arrives: `libpdfium.so` (`pdfium.dll` on
Windows) at the path in `NINFERNO_PDFIUM` (tried first), beside `ninferno-serve`, in the release's
`lib/`, or on the system library path. Without it a PDF is refused (`documents_not_supported`) and nothing else
changes. A page without a text layer (a scan, a figure) reaches a vision server as the page's
image (at most 150 dpi and about 2 megapixels, up to 100 such pages per document), in its place
among the pages; a text-only server marks it as unread and refuses a PDF with no text at all.
Refused as well: a password-protected or unreadable PDF (`invalid_document`) and one past 1000
pages (`document_too_large`).

### Rejected and ignored content

- Documents with a text source (`source.type` `text`) or text content blocks (`content`), in a
  message or a tool result, are read as text: a `<document>` element with the block's `title` and
  `context`. No citations are produced.
- PDF documents (`source.type` `base64` with `application/pdf`, or `url`) are read by their text
  layer ([PDF documents](#pdf-documents)).
- Rejected, with the missing capability identified: Files API sources, Search Results,
  Structured Outputs, server-tool results, container uploads, and other execution-dependent
  blocks.
- Do not block an otherwise executable request: metadata, service tier, inference geography,
  protocol-version/beta headers, cache TTL, and unknown advisory fields.

### Errors and streaming

- Every Messages response carries a `request-id` header. Error bodies also carry `request_id` and use
  Anthropic error categories.
- Local admission overload maps to HTTP 529 and queue/media timeouts to HTTP 504.
- Streaming owns the full Anthropic block lifecycle for Thinking, text, and tool use.

### Token counting

`POST /v1/messages/count_tokens` uses the artifact's tokenizer, chat template, and media expansion
without generation. It shares the same prompt normalization, tools, Thinking mode, Assistant
prefill, media processing, and cache-marker interpretation as Messages. Output-only sampling and
streaming fields do not affect the count.

```bash
curl http://127.0.0.1:8080/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Count this prompt."}]
  }'
```

## 10. Authentication, CORS and exposure

### API key

Pass `--api-key VALUE` to require the same value as an OpenAI bearer token or Anthropic `x-api-key`
header. Other users of the host can list a command line, so the key can instead come from a file
(`--api-key-file PATH`, a file holding the key alone on one line) or from the `NINFERNO_API_KEY` environment variable, which is
read when neither flag is given.

- `GET /health` and CORS preflight requests remain unauthenticated.
- An empty value is refused at startup (it would otherwise serve without authentication).
- The key is compared in constant time.

```bash
curl http://127.0.0.1:8080/v1/models \
  -H 'Authorization: Bearer local-secret'
```

### CORS

`--cors` adds permissive browser CORS headers.

- Its preflight answer allows whatever request headers the browser asks for (browser SDKs add their
  own, such as `x-stainless-*`), grants a public page's Private Network Access request, and may be
  cached for ten minutes.
- It is disabled by default. Then a request that carries an `Origin` header (a browser page's,
  cross-site or DNS-rebound) is refused with HTTP 403 `cross_origin_request`, except
  `GET /health`. Otherwise a page could run generations through a "simple request" (a `text/plain`
  or form body needs no preflight).
- curl and the OpenAI and Anthropic SDKs send no `Origin`.

## 11. Server options

The tables list executable defaults. Run `./build/apps/ninferno-serve --help` for the exact option
contract.

### Model source

| Option | Meaning | Default |
|---|---|---:|
| `--recipe NAME` | conversion recipe of a hub model or checkpoint | the official one |
| `--quant LABEL` | the GGUF quantization of a hub model | NVFP4, MXFP4, Q8_0, BF16, ... in that order |
| `--prepare-only` | fetch and convert, then exit | off |
| `--with-drafters` | also attach the drafters of a GGUF hub model (a separate download) | off |
| `--model-id ID` | override the public OpenAI model alias | the artifact's `metadata.name`, else its file name; a Hugging Face id when serving one |
| `--chat-template FILE` | local Jinja chat template | the artifact's |
| `--device N` | CUDA device index | `0` |

### Network and access

| Option | Meaning | Default |
|---|---|---:|
| `--host H` | listen address | `127.0.0.1` |
| `--port N` | listen port | `8080` |
| `--api-key KEY` | required bearer or `x-api-key` value | unset |
| `--api-key-file PATH` | the same key read from a file (one line) | unset; `NINFERNO_API_KEY` when neither is given |
| `--cors` | permissive browser CORS headers | off |
| `--max-request-mib N` | body-size limit before JSON parsing (a body nesting arrays and objects deeper than 128 levels is refused with HTTP 400) | `384` |
| `--max-request-messages N` | most messages in one request (Responses: `input` Items); more is HTTP 400 `too_many_messages` before the prompt is parsed or rendered | `16384` |
| `--drain-timeout-ms N` | longest wait for requests in flight after SIGTERM/SIGINT | `60000` |
| `--health-stall-ms N` | `/health` is 503 while the Engine has work and makes no progress this long; `0` off | `60000` |

### Memory, context and concurrency

| Option | Meaning | Default |
|---|---|---:|
| `--max-context N` | logical context ceiling of each sequence; a prompt whose bytes alone prove it longer (no token covers more bytes than the vocabulary's longest) is refused before it is tokenized to its end | auto: the model's own, or the longest multiple of 64 (at least 1024) that fits |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory | omitted means `auto` with an automatic context, or with an explicit one and several slots (at most `--max-concurrency` x `--max-context`, at least one context); `--max-context` with one slot |
| `--kv-auto-headroom-mib N` | MiB the automatic KV capacity leaves free for the CUDA driver's own allocations; needs an automatic capacity (refused with an explicit one); 0 disables the reserve, and then allocations the planner does not model may fail at startup or during execution | `32` |
| `--max-concurrency N` | maximum admitted requests; valid range `1..8` | `1` |
| `--max-pending-requests N` | additional requests allowed to wait for admission | `16` |
| `--pending-timeout-ms N` | maximum preparation-plus-admission wait | `600000` |
| `--prefill-chunk N` | text-prefill chunk | `1024` (halved down to 128 when needed; an explicit value is kept) |
| `--kv-lease-window N` | KV a request reserves past its prompt at admission, in output tokens (at least one speculative round); it extends the reservation as it decodes, releasing idle cached context when the pool is short. When a reservation still cannot grow, the request admitted last leaves its lane (recompute preemption, as in vLLM: the older requests keep running), its conversation cached like a cancelled one's, and queues again at the head to resume from its text once another request has finished, recomputing only what the cache lost; its stream does not break. Requests that cannot resume from their text (a media prompt, presence or frequency penalties, a thinking budget, an output constraint) wait instead, their KV kept and no request admitted meanwhile, while the others run and free pages; only when every running request waits does the one furthest into its answer end with its output limit. An agent client asking `max_tokens` 32000 no longer holds a whole context while it answers in a few hundred tokens. Where the pool does not bind it costs nothing measurable (RTX 4090, RTX 5090, A100: decode at 24K and C1/C4 within noise); `0` reserves the whole output budget | `4096` |
| `--concurrent-prefill` | admit waiting requests while others prefill: the prefilling requests take chunks in turn (a later request's first token comes sooner, an earlier long prompt's later); output unchanged | off |
| `--kv-dtype auto\|bf16\|int8\|fp8\|nvfp4\|k8v4` | KV-cache storage; see below | `auto` |
| `--sliding-kv-dtype bf16\|fp8` | the sliding-window layers' rings (Gemma 4, the dense decoders that have them): `fp8` keeps E4M3 codes with an FP32 scale per row, about half the StateImage bytes ([FP8 keys and values](#fp8-keys-and-values)); a model without sliding-window layers refuses `fp8` | `bf16` |
| `--state-dtype f32\|bf16\|f16` | the recurrent state of the Qwen 3.5/3.6/3.8 Gated DeltaNet layers: `bf16` and `f16` halve every state slot and cached state image (`f16` rounds eight times finer, within FP16's range); the recurrence still runs in FP32 and rounds only what it stores. Omitted, these families take `f16` under `--activations fast` (the startup's capacity line says "state f16") and `f32` under `--activations exact`. The rounding carries from call to call, so `--activations exact` refuses an explicit `bf16` or `f16`, as do the other families (Nemotron-H keeps its Mamba2 state in FP32). Measured on Qwen3.8 27B: a 16K stepwise decode with `f16` stays closest to FP32 prefill logits (KLD 0.0018, top-1 98.1 %; `bf16` 0.0052 / 97.0 %), and batched decode (C4, C8) is up to 5 % faster on consumer GPUs, flat on the A100 | `f16` (Qwen 3.5 fast), else `f32` |
| `--activations exact\|fast\|int8` | numeric mode; see below | `fast` |
| `--rope-yarn-factor F` | Qwen 3.5/3.6: rotate with YaRN over the checkpoint's own context and admit `F` (`1..64`) times it. Other families refuse any other value. | `1` |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--moe-offload off\|auto\|N` | keep the routed experts of MoE layers in pinned Host memory: the first `N` MoE layers, or `auto`, the fewest that let the model fit, none when it fits or has no MoE layers ([MoE expert offload](#moe-expert-offload)); a count on a model without routed experts is refused, and Nemotron-H refuses every mode | `off` |
| `--moe-offload-compute auto\|cpu\|gpu` | where decode computes offloaded experts: the CPU (fast mode) or the GPU over PCIe (byte-equal); `auto` is `cpu` under `--activations fast` when the family and format allow | `auto` |
| `--moe-offload-threads N` | CPU expert worker threads | one per physical core |
| `--context-cost-presets FILE` | optional runtime context-cost preset registry | generic + compiled defaults |

**`--kv-dtype`.** `auto` takes the model family's format:

| Family | `auto` picks | Accepted |
|---|---|---|
| Qwen 3.5/3.6/3.8 | `int8` | `bf16`, `int8`, `fp8`, `nvfp4`, `k8v4` |
| dense decoders (gpt-oss, Llama, Mistral, Granite, Muse Glimmer, ...) | `fp8` on GPUs that convert FP8 in hardware (sm_89 on: Ada, Hopper, Blackwell); `bf16` below | `bf16`, `fp8` |
| Gemma 4 | `fp8` for the full layers on sm_89 on, `bf16` below; sliding rings `bf16` | `bf16`, `fp8` |
| Nemotron-H | `bf16` | `bf16`, `fp8` |

`int8`, `nvfp4` and `k8v4` are Qwen 3.5/3.6/3.8 formats. Other families refuse the formats they do
not take at startup.

**`--activations`.**

- `exact`: BF16 activations on every quantized linear, so an answer does not depend on batch
  neighbours, prompt cuts, the prefix cache or speculation. It is slower, most at high concurrency
  and in NVFP4 prefill. Under `exact`, attention splits the KV cache by its row (the
  `--max-context`), so the guarantee holds within one server: another `--max-context` can change
  the last bits. A model with no BF16-activation route for a layer refuses `exact` at startup.
- `fast`: on wide calls NVFP4 activations in four bits and, for integer weights recorded `AllowA8`
  (GGUF Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL/IQ4_XS/Q4_K/Q5_K/Q6_K, u4; not the output head), int8
  activations. The answer can depend on the load.
- `int8` (Qwen; other families refuse it at startup): also gives int8 activations to the sites the
  artifact kept BF16 (heads and
  embeddings excepted), which may move quality.

### Context cache

| Option | Meaning | Default |
|---|---|---:|
| `--no-prefix-reuse` | disable compatible-prefix caching | prefix reuse on |
| `--device-state-slots N` | extra Device checkpoint StateImages beyond the active-lane guarantee | up to `2 * max-concurrency` |
| `--host-state-slots N` | pinned Host StateImage capacity | `max(8, 2 * private continuations)`, scaled with Host KV to what the host can pin when neither is given |
| `--host-kv-mib N` | shared pinned Host Main/Backend KV byte capacity in MiB | `8192`, scaled likewise |
| `--max-private-continuations N` | private continuation descriptor capacity | `max(2 * max-concurrency, max-concurrency + 8)` |
| `--max-shared-prefixes N` | Engine-wide shared stable-prefix descriptor capacity | `max(max-concurrency, 7)` |
| `--max-long-anchors-per-continuation N` | private long-anchor limit per continuation | `2` |
| `--no-auto-system-shared-prefix` | disable the automatic shared-prefix candidate at the leading system/developer frontier ([section 7](#automatic-systemdeveloper-shared-prefix)) | on |
| `--response-store-max-records N` | maximum locally retained Responses objects | `1024` |
| `--response-store-max-mib N` | total local Response envelope/Item/context budget | `256` |

### Speculation

| Option | Meaning | Default |
|---|---|---:|
| `--spec mtp\|dflash\|dflash2\|auto\|none` | speculative backend, or `auto`: the model's tuned profile (`off` is `none`) | `auto` |
| `--spec-select fixed\|thinking` | under `--spec auto`: one configuration for every request, or the profile's thinking/non-thinking default per request | `fixed` |
| `--spec-load LANES:DEPTH[:block\|drafts],...\|none` | bounds on speculation under decode load instead of the profile's `load` table, each optionally with the DFlash draft width of its launches | the profile's |
| `--draft-width block\|drafts` | the DFlash draft width of every launch (`block`: every column the drafter was trained on; `drafts`: the anchor and the round's drafts); `--spec auto` drops a width its speculation does not run | `block` at one row, `drafts` from two |
| `--spec-shared-floor F` | a request below its break-even floor keeps drafting while another request of its step drafts and it emits at least `F` tokens per round (`1` or less: it stops at its floor) | `1.4` |
| `--draft-tokens N` | MTP `1..15`, up to the model's own MTP depth (Qwen 3.5 `5`, the Gemma 4 assistant `15`); DFlash/DFlash2 `1..15` | with `--spec mtp\|dflash\|dflash2`: the model's profile depth for that backend, else MTP `3`, DFlash/DFlash2 `7` |
| `--drafter DRAFTER.ninferno` | drafter artifact | the model's |
| `--lm-head-draft` | optimized proposal head | off |
| `--ngram-draft-tokens N` | Copy rounds beside MTP: a request whose last tokens recur earlier in its context verifies up to `N` tokens that followed them instead of its MTP drafts. Output unchanged. `0` turns them off; families without copy rounds refuse a nonzero value. | auto (`15` where the family runs copy rounds) |
| `--ngram-min-match M` | Tokens that must recur before a copy round runs, at least 1. | `12` |

### Generation defaults

| Option | Meaning | Default |
|---|---|---:|
| `--default-max-tokens N` | output limit when omitted by a request | `8192` |
| `--default-thinking-budget N` | positive thinking cap inherited by thinking-enabled requests | unset |
| `--repetition-stop N` | end an output whose last N tokens (16..65536) repeat one period of at most 64 tokens four or more times: finish_reason `length` (Anthropic `max_tokens`, Responses `incomplete`), `repetition` in the request log | `0` (off) |
| `--repetition-thinking stop\|conclude` | `conclude`: a loop `--repetition-stop` finds while the thinking is still open closes the thinking (the control suffix a thinking budget injects, once) and the answer follows; `thinking_loop_concluded` in the request log. A loop in the answer, with no room left for the suffix, or on a template without a thinking-budget control (gpt-oss), still ends the output | `stop` |
| `--vision` | enable media input and load Vision GPU allocations | off |
| `--no-thinking` | disable thinking by default | thinking on |
| `--preserve-thinking` | preserve closed-turn assistant reasoning by default | off |
| `--usage-chunk-choice` | give the streamed usage chunk a zero-delta choice, for strict client parsers that reject the OpenAI-conformant empty `choices` array | off |
| `--temperature F` | process-level temperature override | unset |
| `--top-p F` | process-level top-p override | unset |
| `--top-k N` | process-level top-k override (`0..64`; zero selects the top-64 cap) | unset |
| `--min-p F` | process-level min-p override | unset |
| `--presence-penalty F` | process-level presence-penalty override | unset |
| `--frequency-penalty F` | process-level frequency-penalty override | unset |
| `--seed N` | fixed seed when a request omits one | fresh random seed per request |
| `--greedy` | force exact argmax for all requests | off |

### Media

| Option | Meaning | Default |
|---|---|---:|
| `--media-cache-mib N` | LRU-retained prepared BF16 media payloads; `0` disables retention | `1024` |
| `--media-live-mib N` | all live prepared BF16 media payloads | `2048` |
| `--media-preprocess-threads N` | bounded media preprocessing workers; `0` selects at most 16 from host concurrency | `0` |

### Logging

| Option | Meaning | Default |
|---|---|---:|
| `--log-level trace\|debug\|info\|warning\|error\|critical\|off` | pretty stderr verbosity (`warn` and `fatal` are taken as `warning` and `critical`) | `info` |
| `--log-stats-interval-ms N` | aggregate throughput report interval; `0` disables it | `5000` |
| `--request-log-jsonl FILE` | append full-precision server/request records; past 1 GiB the file moves to `FILE.1` and a new one starts | disabled |

## 12. Logging and metrics

### Operational log (stderr)

Serve writes human-readable operational records to stderr as
`YYYY-MM-DD HH:MM:SS.mmm  LEVEL  message`.

- Normal output covers material startup milestones, readiness, request lifecycle, fixed-interval
  throughput, and shutdown. `--log-level debug` exposes internal startup and resource-planning
  detail.
- A terminal may use one transient line during startup, but Serve throughput is always a persistent
  record. Redirected stderr contains no terminal control sequences.
- Pretty values use readable units and rounded rates. Use the independent request JSONL for complete
  fields and full precision.
- Operational records never contain prompts, generated text, request bodies, credentials, or
  arbitrary client error messages.
- If a tool marker is returned to text because its structure or tool identity cannot be represented,
  Serve emits one warning with only the failure classification, never the generated markup.

### Structured request log (JSONL)

`--request-log-jsonl FILE` enables the machine-readable measurement log.

- The server opens `FILE` in append mode and flushes every event, so successive model or MTP blocks
  may share one campaign file. The parent directory must already exist. Failure to open the file
  aborts startup. The path is also rejected if it resolves to the model artifact.
- Every line is one `ninferno_serve_request_log` schema-v24 JSON object. All events carry
  `timestamp_unix_ms` and a process-unique `server_instance_id`. Request IDs are monotonic only
  within that server instance.
- Successful request-start records include request-scoped acquisition, media-preprocessing
  wall/work, tokenizer, cache hit/miss/single-flight, and payload-size fields. They do not infer
  request behavior from process-global counter deltas.

| Event | Contents |
|---|---|
| `server_start` | artifact path, architecture, public name, actual formats and prefill signature; resolved Engine and context-cache capacities, thinking/non-thinking sampler defaults plus process overrides, thinking-history and thinking-budget defaults, Device arenas, the optional non-additive Vision layout inside the unified workspace, Host State/KV capacity and occupancy, KV sizing ledger, CUDA Graph allowance, CUDA/GPU environment, and redacted argv |
| `request_start` | protocol, resolved sampler and seed, requested reasoning effort, actual initial thinking mode and optional budget, Responses semantic-change flag, output budget, stream/message/tool shape, and `notes`: what the server accepted but did not use as sent (a reasoning Item without text left out, say) |
| `request_rejected` | parsed request shape, requested reasoning effort, media-item count, `phase: "prepare"`, and the exact HTTP status/type/code/parameter/message for a synchronous preparation rejection |
| `request_done` | finish reason, prompt/completion/cache/computed-prefill tokens, prefix reuse path, tool-call parse diagnostics, request-owned materialization cost/search diagnostics, thinking-budget application counters, unrounded request-stage seconds, per-request Engine Host exposure, and complete speculative-decoding counters |
| `request_error` | the resolved request configuration and the generation, cancellation, or pre-outcome transport terminal message |
| `throughput` | interval token/decode/context-cache pressure counter deltas, authoritative worker Host-work deltas, current scheduler/resource gauges, and decode-round batch statistics |

`requested_reasoning_effort` and `preserve_thinking` record the explicit options, or `null` when
unspecified. `enable_thinking` records whether the request thinks: its prompt starts the reasoning
(Qwen), or leaves the model to open it (Gemma 4 with thinking on, gpt-oss). A thinking request takes
the thinking sampler preset and, under `--spec-select thinking`, the thinking speculation.

**Event lifecycle.**

- OpenAI Responses, OpenAI Chat, and Anthropic generation requests receive a request ID when they
  enter synchronous preparation.
- Successful preparation produces `request_start`. A preparation failure produces `request_rejected`
  without a matching start.
- Each started generation transaction then has exactly one machine terminal: `request_done` when the
  Engine returns its outcome, or `request_error` when generation fails before an outcome exists.
- Later response rendering, Responses storage, or terminal transport failures are operational
  response events only and do not add a second JSONL terminal.
- Schema/model validation rejections before preparation and token-count-only calls are not
  measurement requests and do not receive request IDs.
- The file contains no generated response text and never records an API-key value; `argv` replaces
  that value with `<redacted>`.
- Operational stderr summaries are rounded and are not the aggregation source.

#### Tool-call diagnostics

`request_done.result.tool_call_parse` records:

- whether a complete marker was seen;
- the structured call count;
- empty non-string arguments omitted during normalization;
- schema-mismatched arguments preserved for consumer validation;
- parameters repeated within a call (each kept its last value);
- whether the calls were recovered from reasoning;
- a stable text-fallback reason.

Details:

- A turn whose model ends with its stop token inside a thinking span it never closed has empty
  content. When that reasoning ends in a complete call region that the strict parser accepts whole,
  the calls are published and `recovered_from_reasoning` is `true`. The reasoning
  text stays in the reasoning channel. A call quoted earlier in the reasoning, followed by more
  reasoning, is not taken. A turn that ended so with no call recovered reports
  `ended_inside_reasoning` `true`: its reply has reasoning and no content.
- When the output limit cuts the last call off, the complete calls before it are published, and
  `cut_calls_dropped` counts the unfinished call whose markup was dropped (the finish reason stays
  `length`).
- `undeclared_calls` counts the published calls whose name the request did not declare.
- Fallback reasons are `none`, `malformed_structure`, `duplicate_parameter`, `invalid_tool_name`,
  `undeclared_tool` (Llama 3's JSON form only), and `trailing_content`.
- Related counters: `trailing_content_bytes`, `unwrapped_calls`, `closed_calls`.
- These counters contain no tool arguments or generated text.

#### Timing and token fields

- `request_done.result.model_thinking_tokens` counts the model-origin tokens accepted while thinking
  was open, including the token that closes it, for every response that starts in thinking, with or
  without a budget. Inserted thinking control is not model-origin. Before schema 22 the count was
  recorded only under a thinking budget and was 0 otherwise.
- `request_done.timings_seconds` contains `prepare`, `ttft`, `vision`, `prefill`, `decode`, and
  `total` as full-precision JSON numbers. Its `speculative` object contains `backend`,
  `draft_window`, `rounds`, `drafted_tokens`, `accepted_tokens`, `fallback_steps`, and
  `accepted_per_position`. Rates can be derived downstream from raw token counts and seconds instead
  of rounded stderr strings.
- For `server_start.memory`, `workspace.capacity_bytes` is the only physical workspace allocation.
  When Vision is enabled, `vision_workspace` reports the aggregate prompt and maximum-item token
  bounds plus encode peak and handoff layout/usage within that same allocation. These bytes must not
  be added to `workspace.capacity_bytes`. The field is `null` when Vision is disabled.
- `request_done.engine_timing` separates FIFO `queue_wait_seconds`, blocking
  `device_wait_exposed_seconds`, and five mutually exclusive Host-active exposure phases under
  `host_exposed_seconds`: `engine_boundary`, `program_submit`, `program_post`,
  `engine_commit_output`, and `engine_maintenance`. `total` is exactly their sum and excludes Device
  wait. The nested `decode` object reports the request's decode-class Host exposure, Device wait,
  and round count. `units` reports its prefill/control unit counts. In a compact batch every
  participating request is delayed by the full round, so these values explain request latency but
  **must not be summed across concurrent requests**.

#### Throughput records

By default the server persistently reports aggregate activity every five seconds
(`--log-stats-interval-ms`).

- `prefill` counts prompt suffix tokens actually computed during the interval, excluding
  prefix-cache hits.
- `decode` counts tokens finally committed by decode rounds, excluding the first token produced by
  prefill. For MTP, DFlash and DFlash2 this is the accepted committed output, not draft or rejected
  tokens.
- Pretty `batch` and JSONL `average_size` are decode row-rounds divided by decode rounds during the
  same interval.
- The `running`, `prefilling`, `decode_ready`, `waiting`, `materializing`, `capture_pending`, and
  `terminal_pending` fields are the Engine scheduler snapshot at the end of the interval.
- The JSONL `context_cache` object reports selection, capture, transfer, COW, pressure spill,
  private/shared owner degradation and eviction, checkpoint drop, pressure search, budget
  exhaustion, maximal fallback, and historical-fork counters as interval deltas. `occupancy` and
  `last_selection` are end-of-interval gauges. Materialization predictions are request-owned and
  appear only on the corresponding `request_done` event.
- `pressure.searches` counts plans accepted into Program resource transactions, including a
  transaction that later ends in request-local abort. Committed victim counters likewise report the
  resulting stable cache changes.
- The JSONL `throughput.host_work` object is the aggregation authority: the Engine worker counts each
  wall-time segment once, independent of batch size. `elapsed_seconds` contains the same five
  mutually exclusive Host phases and their `total`. `device_wait_seconds` is separate.
  `work_class_seconds` splits Host and Device-wait time into decode, prefill, and control classes.
  `detail_subset_seconds` and `detail_invocations` expose admission, context-transaction, replica,
  and stats-publication slow paths. These detail values are already contained in a top-level Host
  phase and must not be added to `total`. Per-round, per-row-round, and per-invocation normalized
  values are `null` when their denominator is zero.
- Pretty throughput contains nonzero token rates and counts, the current running/prefill/decode-ready
  composition, nonzero waiting/materialization/terminal states, average decode batch, and Host-active
  time plus its fraction of the interval. Use JSONL for complete measurement analysis.
- Intervals with context materialization or retention activity are retained even when they contain
  no token execution. Only fully idle intervals are omitted. Downstream measurement should prefer
  the raw counters and seconds over rounded stderr rates.

### Prefix-reuse paths

The pretty completion record shows `cache N (P%, path)` using readable path labels. JSONL retains the
exact `prefix_cache_hit_tokens` and `prefix_reuse_path` fields. Machine paths are `root`,
`private_endpoint`, `private_turn_closure`, `private_response_replay`, `private_long_anchor`, and
`shared_stable_prefix`.

## 13. Execution behavior

### Batching and admission

- The server owns one resident Engine with a startup-fixed capacity of `1..8` active generation
  requests.
- At each decode boundary, every decode-ready request is compacted into one batch and processed by
  one model traversal and, when graphs are enabled, one exact-batch CUDA Graph replay.
- A request joins that batch only after its single-request prefill finishes. When it completes or is
  cancelled, the next boundary rebuilds the batch without an empty row.
- `--max-pending-requests` bounds the requests waiting behind the active set. The total generation
  request lifetime capacity is `max_concurrency + max_pending_requests`, including requests still in
  CPU/media preparation and completed model results whose response has not yet been released.
- A full capacity returns HTTP 429 with code `server_overloaded`.
- Token-count requests (Anthropic `count_tokens`, Responses `input_tokens`) run the same preparation
  without generating. A separate capacity of the same size bounds how many run at once. Beyond it
  they are rejected as overloaded, as generation requests are.
- The absolute `--pending-timeout-ms` deadline starts before preparation, covers media acquisition
  and Engine FIFO waiting, and returns HTTP 503 with code `request_queue_timeout` if admission does
  not occur in time.
- Both answers carry `Retry-After` (1 s for a full capacity, 5 s for a timed-out admission). There
  is no admission ETA or unbounded overflow queue.
- Admission reserves the full prompt-plus-effective-output page entitlement through request
  completion. A request remains queued until a legal resource plan can satisfy that entitlement.

### Numerical failure check

Every round checks each request's final hidden state (the prompt's last position after prefill, each
decode step's, and each verified column of a speculative round) for a NaN or an infinity. The check
rides on the round's own synchronization.

- A request whose state is not finite fails alone with HTTP 500 and code `numerical_failure`. The
  batch's other requests go on.
- The message names the model, its architecture, its weight formats and the GPU class. A failure
  that repeats on the same prompt points at a kernel route of those weight formats on that GPU
  class, or at a damaged artifact.
- For tests, `NINFERNO_DEBUG_NAN_FILE=<path>` makes the next check after that file appears remove it
  and write a NaN into the first row of the checked state.

### Input and output memory

Input memory is bounded by the outstanding-request count and the per-request `--max-request-mib`
limit. Media requests additionally share one preparation permit, so a waiting media request retains
the same cancellation and timeout deadline. Model output is bounded by the same finite request count
and each request's effective output-token limit. Output callbacks and network serialization run
outside the GPU executor and do not delay formation of the next batch.

### Automatic context selection

`--max-context` is each sequence's logical ceiling. Omitted, the Engine chooses it at startup from
the free memory measured once:

1. The model's own position capacity, when a plan with it (and the default cached Device states)
   fits the free GPU memory.
2. Else the longest whole number of 64-token KV pages (at least 1024 tokens) that fits, by
   bisection over the plans. A page is the KV's own grain: a context between two pages takes the
   next page's memory.
3. An explicit value above the model's own is refused.
4. When that leaves less than 8192 tokens (or the model's own, when smaller), the choice gives up
   cached Device states of the context cache for a longer context: down to one, or to none when the
   Host tier keeps states (checkpoints are then captured to and restored from Host memory).
5. When even that stays below 8192 tokens and `--prefill-chunk` is omitted, the same again with half
   the chunk, down to 128. A narrower chunk takes less workspace and, on a model with sliding-window
   layers (Gemma 4), a smaller StateImage per state (window + chunk - 1 rows per layer). The first
   chunk that reaches the floor is kept, else the one with the longest context. The startup log
   states the chunk.

Examples on a 16 GB card (RTX 5080):

| Model | Result |
|---|---|
| Gemma 4 26B-A4B NVFP4 (14.3 GiB of weights) | 12,288 tokens without a drafter; 8,192 at chunk 256 with its MTP assistant (0.28 GiB); both with the Host tier keeping the checkpoints and no cached Device state; 14,336 with one cached Device state when the Host tier is off |
| Gemma 4 31B UD-IQ3_XXS (11.0 GiB) | 27,648 tokens at chunk 1024 alone; 20,480 with its MTP drafter (0.51 GB) |
| Gemma 4 31B UD-IQ3_XXS, `--max-concurrency 4`, no drafter | 6,144 at chunk 128; the log names 13,312 / 19,456 / 27,648 for 3 / 2 / 1 slots |
| same, with the drafter | four slots do not fit; the refusal names 3 slots with 8,192, 2 with 11,264 and 1 with 20,480 |

- An explicit `--max-context` takes the widest chunk it fits with. When an explicit `--prefill-chunk`
  fits no concurrency, the refusal names the widest narrower chunk (each half down to 128) that fits
  and what it fits.
- With `--max-concurrency` above 1 every slot keeps its own StateImage. When the automatic context
  stays below the floor, the startup log names the lower concurrencies that fit longer. When nothing
  fits, the refusal names them (and, for an explicit `--max-context`, the longest context that
  fits).

### KV capacity

- `--kv-capacity` fixes the shared Main Text KV pool used by active requests and retained prefixes.
- `auto` plans every runtime byte (KV, cached StateImages, workspace, the CUDA Graph allowance; a
  drafter's weights are loaded before the free memory is measured). It leaves 32 MiB for the CUDA
  driver's own allocations (streams, events, kernel code loaded on first launch), measured at 9-20
  MiB while starting and nothing after it; the `--kv-auto-headroom-mib` default is the table value
  above. Omitted, `--kv-capacity` is `auto` with an automatic context.
- On a GPU Windows shares (WSL 2), `auto` also leaves Windows a 32nd of the card, at most 1 GiB, and
  an 8th, at most 4 GiB, on the card that drives the display. The desktop's memory grows while
  ninferno runs, and a full card makes Windows page the pool out, with the desktop frozen meanwhile
  (an RTX 5090 left with 0.55 GiB took 1.8 s to give another process 3 GiB; with 5.5 GiB left, 51
  ms).
- An explicit `--max-context` is each request's limit. With one slot the pool is exactly that many
  tokens. With several it is `auto`, bounded by `--max-concurrency` x `--max-context` above (every
  slot's full context) and one context below. When the full pool does not fit, the startup log says
  how much it holds. A request then waits while the pool is full (8 slots of 4,096 tokens in a
  4,096-token pool ran 7 requests at a time with 512-token outputs).
- Capacity resolves once at startup. `/v1/models` reports the context as `max_model_len`.

### FP8 keys and values

- **Dense decoders** (dense_decoder and moe_decoder artifacts: gpt-oss, Llama, Muse Glimmer, ...)
  keep their full layers' keys and values in FP8 rows by default on sm_89 and newer
  (`--kv-dtype auto`). `--kv-dtype bf16` restores BF16. Below sm_89 the software conversion makes
  FP8 slower (RTX 3090 gpt-oss at 88K: 136 -> 91 t/s), so `auto` stays BF16 there. Their sliding
  rings stay BF16 unless `--sliding-kv-dtype fp8`.
- gpt-oss-20b at about 88K context decodes 157 -> 185 t/s (RTX 4090), 159 -> 192 (RTX 5080), and 316
  -> 350 at 48K (RTX 5090), with perplexity on a book unchanged within noise (windows 256 and 2048).
  Short requests are unchanged. The prefill of a long prompt costs 2-5 % more (the attention widens
  the FP8 tiles in shared memory).
- **Gemma 4** keeps its full layers' keys and values in FP8 by default on sm_89 and newer
  (`--kv-dtype bf16` restores BF16). FP8 sliding rings are opt-in with `--sliding-kv-dtype fp8`;
  the rings grow with every slot and cached state (31B: 0.78 MiB per ring row in BF16, window
  1024 + chunk - 1 rows each).
- Measured with every layer in FP8 against BF16 (`ninferno-gemma4 score`, 32 chat windows of 2,047
  scored tokens, 65,504 in all): mean next-token NLL moves by -0.002 +- 0.018 and +0.007 +- 0.014
  (26B-A4B NVFP4, artifact / BF16-activation profile) and +0.009 +- 0.012 and +0.006 +- 0.007 (31B
  NVFP4). The +- is the standard error over the windows, of the same order as switching the model's
  own activation profile (+0.028 and -0.001). The top-1 token agrees with BF16 at 75-97 % of
  positions, so FP8 is neutral in NLL, not in bytes.

### MoE expert offload

`--moe-offload` runs a MoE model whose weights do not fit the GPU. The routed experts (88-91 % of the
parameters of a 20-35B MoE) of chosen layers stay in pinned Host memory. Attention, the router,
shared experts and KV stay on the GPU.

- `N` offloads the first `N` MoE layers.
- `auto` loads the experts of as many leading layers as the host can pin (every layer on a host with
  the memory) to Host memory, keeps the fewest leading layers there with which the plan at the
  automatic context's floor fits, and copies the rest to the GPU. The floor is 8192 tokens, or the
  model's own when smaller; an explicit `--max-context` is used instead.
- A decode step's kernels read the selected experts over PCIe in place (zero-copy, also inside the
  CUDA graph). A call whose tokens select at least as many experts as the layer has (prefill)
  streams the whole layer into one of two GPU staging slots on a copy stream, the next offloaded
  layer already in flight. The kernels are the same and read the same bytes, so the output is the
  same as without offload.
- Pinned memory counts against the host. The experts loaded to Host memory must fit what the host
  can pin (MemAvailable less max(4 GiB, MemTotal/8), and at most 3/5 of MemTotal pinned by the
  process in all), else the start is refused with the numbers. An automatic Host tier fits what is
  left after the offload.
- A model that does not fit even with every MoE layer offloaded is refused with the numbers. Without
  `--moe-offload`, weights that do not fit are refused with their size, the free memory and, for a
  MoE model, the bytes its experts take.
- The startup log's `moe offload |` line and the request log's `engine.moe_offload` state the
  placement.

`--moe-offload-compute auto|cpu|gpu` chooses where a decode step (up to 16 tokens per call) computes
the offloaded layers' experts.

- `gpu`: the GPU reads them over PCIe, byte-equal to no offload, bounded by the PCIe bandwidth.
- `cpu`: the CPU computes them from the same pinned Host bytes in their published format (MXFP4,
  NVFP4), reading only the selected experts at DRAM bandwidth. The GPU runs the router and top-k (the
  same selection), copies the routed tokens to the host inside the step (a host node, also inside
  the CUDA graph), and receives the experts' output. For Gemma 4 the CPU runs while the GPU computes
  the dense MLP.
- One pinned worker per physical core of the process affinity (`--moe-offload-threads N` to set it),
  AVX-512 or AVX2 when the CPU has it.
- The CPU's arithmetic has a fixed order (fixed 32-value blocks, fixed reduction tree, one owner per
  output), so its result does not depend on the thread count or instruction set. It is close to the
  GPU's, not byte-equal, so `cpu` is the fast mode and is refused with `--activations exact`.
- Prefill always streams whole layers to the GPU.
- `auto`, the default, is `cpu` under `--activations fast` when the model family has the CPU path
  (dense MoE families such as gpt-oss, Gemma 4) and the experts' format is one the CPU reads, else
  `gpu`. An explicit `cpu` that cannot hold is refused with the reason.
- The startup log's second `moe offload |` line and `engine.moe_offload.compute`, `cpu_threads`,
  `cpu_isa` and `compute_reason` state the choice.

### Checkpoints and resource pressure

Each reusable checkpoint contains KV and complete continuation state. At admission, capture, and
finish boundaries, resource pressure may keep it on Device, move its StateImage and/or KV replicas to
pinned Host memory, or evict it. The planner compares incoming-request work with the later recovery
cost imposed on retained checkpoints. Active requests retain their state and completion reservations,
and placement choices preserve model semantics. The full policy and invariants are defined in
[Resource scheduling and context cache](../maintainer/resource-scheduling-and-context-cache.md).

### Prefix reuse

Compatible prefixes are reused for both text and multimodal histories unless the server starts with
`--no-prefix-reuse`.

- A multimodal hit additionally requires matching token types, three-axis MRoPE positions,
  encoded-media digest, grid, and consumer spans. Media wholly inside a matched prefix skips Vision
  execution, while new suffix media is encoded normally.
- Reuse validation covers KV, recurrent state, hidden state, selected-backend state, and the exact
  prompt frontier.
- With stable `preserve_thinking=true`, the auxiliary checkpoint rolls to the message frontier
  immediately before the current response's deterministic generation prologue. A normalized
  response, compact-summary instruction, or replacement user suffix therefore replays the small
  generation prologue and only the changed suffix, while retaining the complete stable conversation
  prefix.
- Stable `false` places the turn-closure checkpoint before the first assistant opener in the open
  turn, so closing that turn can recompute its opener and omit its reasoning without discarding the
  preceding conversation.
- `preserve_thinking` selects the capture frontier for newly created checkpoints. Existing exact
  checkpoints remain reusable across a mode change. If the desired boundary is behind the selected
  reuse frontier and has no snapshot, the Engine keeps the valid hit and defers the new checkpoint.
  A later request that diverges before every retained checkpoint starts from root. The JSONL
  completion record exposes the restored checkpoint as `prefix_reuse_path`.
- Reasoning-effort changes participate in rendered-token identity and exact-prefix selection.
- An appended mid-conversation system message is an ordinary prompt suffix, so an unchanged prior
  history remains eligible for `private_endpoint`. If the client modifies, removes, or moves a
  historical system message, the token prefix genuinely differs and a miss/reset is correct.

### Finish reasons, tools and usage

- Output-limit and context-capacity finishes map to `length` / `max_tokens`. Ordinary model or
  string stops map to `stop` / `end_turn`.
- Function tools are rendered into the model prompt and generated calls are parsed into protocol
  responses. NInferno does not execute tools. It enforces client JSON Schema through constrained
  decoding only where stated above (forced or `strict` tool calls and JSON `response_format`).
- Prompt-token usage includes chat-template and expanded media tokens. Generated-token usage comes
  from accepted output token IDs, including a stop token whose decoded text may be withheld.

## Error codes summary

| Status | Code | Cause |
|---|---|---|
| 400 | `vision_disabled` | media or media token count without `--vision` |
| 400 | `invalid_media`, `invalid_media_url` | bad media bytes/content; bad media URL |
| 400 | `invalid_prompt` | failure in the normalized prompt contract |
| 400 | `context_length_exceeded` | expanded prompt beyond `--max-context` |
| 400 | `media_budget_exceeded` | media preprocessing resource rejection |
| 400 | `too_many_messages` | more than `--max-request-messages` |
| 400 | `thinking_budget_capacity_insufficient` | no room for the early-close suffix |
| 400 | `conflicting_template_option` | conflicting `enable_thinking` and effort |
| 400 | `unknown_parameter` | unknown top-level Responses field |
| 400 | `background_not_supported` | `POST /v1/responses/{id}/cancel` |
| 400 | (no code; `param` names the field) | a sampling value out of range: `temperature` [0,2] (Chat, Responses) or [0,1] (Messages), `top_p` and `min_p` [0,1], `top_k` [0,64], penalties [-2,2] |
| 401 | `invalid_api_key` | missing or wrong key with `--api-key` set (Anthropic shape on `/v1/messages`) |
| 400 | `invalid_assistant_history`, `invalid_tool_history` | unrepresentable Responses history |
| 400 | `response_store_capacity_exceeded` | the request's input and context are larger than the response store (an answer that outgrows it after generation is returned unstored) |
| 403 | `cross_origin_request` | request with `Origin` while `--cors` is off |
| 404 | `response_not_found` | unknown stored Response |
| 404 | `not_found` | a path no endpoint serves (JSON body naming the endpoints) |
| 413 | `request_too_large` | body above `--max-request-mib` |
| 415 | `unsupported_content_encoding` | compressed request body |
| 429 | `server_overloaded` | request capacity full (`Retry-After` 1 s) |
| 499 | `client_disconnected` | the client closed the connection before the response finished |
| 500 | `numerical_failure` | non-finite hidden state for one request |
| 502 / 504 | `media_fetch_failed` / `media_fetch_timeout` | remote media unreachable or 5xx / timed out |
| 503 | `request_queue_timeout` | admission deadline passed (`Retry-After` 5 s) |
| 503 | `service_unavailable` | CUDA failure; drain; `Retry-After` set |
| 529 / 504 | (Anthropic) | local admission overload / queue and media timeouts |
| 200 + error event | (any code above) | a streamed request that fails after its headers were sent (a queue timeout, an Engine failure): the event carries the code and message, the stream ends |
| any | `compaction_not_supported` | `POST /v1/responses/compact` |
