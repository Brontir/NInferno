# HTTP serving

`build/apps/ninferno-serve` loads one model and serves OpenAI- and Anthropic-compatible HTTP
endpoints from one resident NInferno Engine on one NVIDIA GPU.

- GPUs: Volta `sm_70` through Blackwell `sm_120`. NVFP4 weights run natively on `sm_120` (RTX 50); older GPUs run them on slower generic kernels (not yet measured), where a GGUF or FP8 checkpoint is the measured choice.
- Models: only original Hugging Face checkpoints. NInferno publishes no prebuilt model files. The
  server converts the checkpoint into a `.ninferno` file on first start and reuses it afterwards.
- Speeds per GPU and model: [performance matrix](performance/README.md).

This page covers everyday use. Every detail (request fields, compatibility rules, state, logs, error
codes) is in the [serving reference](reference/serving.md).

## Contents

- [Start the server](#start-the-server)
- [Important options](#important-options)
- [Endpoints](#endpoints)
- [Examples](#examples)
- [Speculative decoding](#speculative-decoding)
- [Memory, context and concurrency](#memory-context-and-concurrency)
- [Authentication](#authentication)
- [Health, shutdown and logs](#health-shutdown-and-logs)
- [More](#more)

## Start the server

One command starts a model from the Hub:

```bash
./build/apps/ninferno-serve Qwen/Qwen3-1.7B            # or org/name@revision
```

The model argument can be:

| Argument | What happens |
|---|---|
| hub id `org/name[@revision]` | downloaded into the Hugging Face cache, converted, and reused on later starts |
| checkpoint directory or GGUF file | converted once into `<dir>/.ninferno-cache/` |
| `.ninferno` file | served as is |

For a hub id:

- The cache is `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `$XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub`. The converted
  model goes to `<cache>/models--org--name/ninferno/<commit>/`, with the drafters the model
  declares. Nothing else on the machine is searched.
- `$HF_TOKEN` authorizes gated and private models. `HF_HUB_OFFLINE=1` uses the cache only.
- The served model id is the hub id, without an `@revision`.
- `--prepare-only` downloads and converts, then exits.
- `--plan` prints the memory plan these options give on this GPU now (the plan ladder, every rung's
  bytes, the rung that fits and what would get more), then exits; `--plan --json` prints it as
  JSON for a UI's fit bar ([reference](reference/serving.md#automatic-context-selection)).
- `--plan --no-load` plans without loading or downloading the weights and without a GPU. It reads
  only the model's header or config: for a hub id, the file listing and one GGUF header. For a GGUF
  repository it lists every quantization, says whether each fits the asked `--max-context` x
  `--max-concurrency`, and names the largest that fits. `--plan-device "RTX 4090"` plans for another
  card. It is an estimate; the start decides
  ([reference](reference/serving.md#the-plan-without-the-weights)).
- `--recipe NAME` picks another conversion recipe. `--quant LABEL` picks a GGUF quantization.
  Without `--quant`, a GGUF repository is fetched at the largest quantization that fits the asked
  context and streams on this GPU's card type, and the log says which (`quant | ...`):
  - "fits" means each stream holds the asked context of its own; with an automatic context, at
    least 32K each (or the model's own length), then 16K, then 8K, before the streams share one
    pool, and that is logged in plain words; with several streams and no `--max-context`, the
    start keeps the context it chose the file for as the automatic context's ceiling
    (`--max-context-cap`);
  - below 3.5 bits per weight only when nothing larger fits at all;
  - a quantization already downloaded that fits is kept, so other flags do not download another
    file;
  - when none fits, nothing is downloaded and the refusal says what fits instead;
  - when the hub cannot be reached, it takes the largest cached file that fits; with none it can
    plan, the fetch's own default applies;
  - the fetch's own default order (when no plan applies) now puts BF16 and F16 files last: the
    engine serves 8 bits or fewer, so a repository without Q8_0 gets a K-quant, not BF16.

The server checks before downloading any weight and refuses with a reason if the architecture is not
supported, the repository is gated or missing, the weights do not fit the GPU memory, or the disk is
too small.

A fuller example for a converted file:

```bash
./build/apps/ninferno-serve models/qwen3_8_27b_nvfp4.ninferno \
  --host 127.0.0.1 --port 8080 \
  --max-context 240000 --kv-capacity 240000 \
  --max-concurrency 2 --kv-dtype fp8 \
  --device-state-slots 2 --host-state-slots 8 --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --preserve-thinking
```

You need none of these flags to start. Context, KV capacity and the Host cache are chosen
automatically, and the startup log states each choice (`context | ...`, `capacity | ...`,
`context cache | ...`). Run `./build/apps/ninferno-serve --help` for the exact option list.

## Important options

| Option | Default | What it does |
|---|---|---|
| `--version` | | print the version, the GPU architecture of this build and whether the driver and GPUs here run it, then exit |
| `--host H` | `127.0.0.1` | listen address |
| `--port N` | `8080` | listen port |
| `--api-key KEY`, `--api-key-file PATH` | unset | require a bearer / `x-api-key` value |
| `--model-id ID` | `metadata.name`, else the file name | public model alias; it does not change execution |
| `--max-context N` | auto: the first rung of the plan ladder that fits (the model's own context at the top) | per-request context ceiling |
| `--max-context-cap N` | none; a GGUF repository fetched without `--quant` for several requests: the context each was planned to keep of its own | keeps the context automatic (the free memory picks the rung, a busy card steps down) but at most N; not with `--max-context` |
| `--kv-capacity N\|auto` | `auto`; with `--max-context` and one request, that context | shared KV pool size |
| `--max-concurrency N` | `1` | requests generated at once, `1..8` |
| `--max-pending-requests N` | `16` | requests allowed to wait for admission |
| `--pending-timeout-ms N` | `600000` (10 min) | longest wait before admission |
| `--prefill-chunk N` | `1024` | text prefill chunk, a multiple of 128 (lowered automatically if memory is short) |
| `--kv-dtype auto\|bf16\|int8\|fp8\|nvfp4\|k8v4` | `auto` | KV-cache storage; `auto` is chosen for the model on your GPU ([configuration choice](maintainer/configuration-choice.md); the start logs it on a `choice \|` line) |
| `--activations exact\|fast\|int8` | chosen (`fast`) | `exact` makes answers independent of load, but is slower; `int8` is for Qwen models (other families refuse it). Omitted, the configuration choice picks it (`fast` unless `exact` costs little) |
| `--spec mtp\|dflash\|dflash2\|auto\|none` | `auto` | speculative decoding backend |
| `--draft-tokens N` | profile's depth, else 3 (MTP) or 7 (DFlash) | draft depth, `1..15` |
| `--vision` | off | accept image input (Qwen3.5, 3.6, 3.8, Gemma 4 26B-A4B and 31B) and video input (Qwen); a GGUF needs its `mmproj` file beside it |
| `--default-max-tokens N` | `8192` | output limit when the request gives none |
| `--no-thinking` | the template's default | disable thinking by default |
| `--default-thinking-budget N` | unset | cap on thinking tokens, where the chat template can end thinking (Qwen, Gemma 4; not gpt-oss) |
| `--no-prefix-reuse` | reuse on | disable prefix caching |
| `--cors` | off | permissive browser CORS headers |
| `--request-log-jsonl FILE` | off | machine-readable request log |
| `--log-level LEVEL` | `info` | stderr verbosity (`trace` ... `off`) |
| `--device N` | `0` | CUDA device index |
| `--temperature`, `--top-p`, `--top-k`, `--min-p`, `--seed`, `--greedy` | unset | process-level sampling overrides ([presets](cli.md#sampling-presets)); `--greedy` forces argmax |

All other options (media, Host cache, response store, penalties, limits, decode pacing with
`--decode-steps`, `--decode-wait` and `--cuda-sync`) are in the
[option tables](reference/serving.md#11-server-options). The maintainer options (`--route-table`,
`--route-dir`, `--dense-tune`, `--moe-tune`, `--tune-cache`, `--drafter-registry`, `--debug-*`)
are in [maintainer options](maintainer/environment.md); no environment variable changes what the
server decides.

## Endpoints

| Path | API | Notes |
|---|---|---|
| `GET /health` | server | `200 {"status":"ok"}` with the load gauges; `503` when unavailable, draining or stalled; no authentication |
| `GET /v1/models`, `GET /v1/models/{id}` | OpenAI | model alias, `max_model_len`, llama.cpp-style `meta`; `meta.published_activations`: the activation widths (16, 8, 4, unknown) the checkpoint's publisher validated, as kept-matrix counts, when it says; `meta.decision_fingerprint` and `meta.decision_manifest`: the engine's [decision fingerprint](maintainer/deterministic-decisions.md#fingerprint); `meta.speculation`: the speculation it runs since startup and the fallback startup took |
| `GET /props` | llama.cpp | the model alias, context, vocabulary and slots llama.cpp's web UI reads |
| `GET /metrics` | Prometheus | llama.cpp's `llamacpp:*` token and request series, plus prompt tokens served from the cache and requests in prefill, and `ninferno_decision_fingerprint_info{fingerprint,manifest,schema} 1` |
| `POST /v1/chat/completions` | OpenAI Chat | streaming, tools, images, `timings` extension |
| `POST /v1/responses` | OpenAI Responses | typed Items, SSE events, local state |
| `POST /v1/responses/input_tokens` | OpenAI Responses | prompt token count, no generation |
| `POST /v1/responses/{id}/cancel`, `POST /v1/responses/compact` | OpenAI Responses | answered with an error: no background execution, no compaction |
| `GET`, `DELETE /v1/responses/{id}` | OpenAI Responses | read or delete a locally stored Response |
| `GET /v1/responses/{id}/input_items` | OpenAI Responses | list that Response's input Items |
| `POST /v1/messages` | Anthropic Messages | streaming, tools, thinking |
| `POST /v1/messages/count_tokens` | Anthropic Messages | prompt token count, no generation |

Base URLs:

- OpenAI SDKs: `http://127.0.0.1:8080/v1`.
- Anthropic SDKs: `http://127.0.0.1:8080` (they append `/v1/messages`).

The `model` field must be present in OpenAI requests, but any value is accepted: one server runs
one model, and the response names the served id. For Anthropic Messages it is only a label and is
echoed back. The examples below assume a server started with
`--model-id qwen3.8-27b`.

## Examples

OpenAI Chat Completions:

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

Streaming: add `"stream": true` to any request. Chat Completions sends server-sent events ending with
`[DONE]`. Add `"stream_options": {"include_usage": true}` for a final usage chunk.

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Say hello."}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

OpenAI Responses:

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

Anthropic Messages:

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

Image input needs a server started with `--vision`. See the
[multimodal example](reference/serving.md#multimodal-request).

Using the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-secret")
response = client.responses.create(
    model="qwen3.8-27b",
    instructions="Answer concisely.",
    input="What is speculative decoding?",
    max_output_tokens=128,
)
print(response.output_text)
```

Notes that often matter:

- Reasoning arrives separately: `reasoning_content` in Chat, `reasoning` Items in Responses,
  `thinking` blocks in Messages.
- Unknown top-level fields in Chat and Messages are ignored, and the response carries an
  `X-NInferno-Warning: unknown fields ignored: ...` header. Responses rejects them with
  `unknown_parameter`.
- NInferno never runs tools. It renders your tool definitions into the prompt and returns parsed
  calls. Your client runs them and sends the results back.
- Unsupported features (audio, logprobs, hosted tools, stored Chat Completions, and similar) are
  rejected with an error that names the field. See the
  [compatibility details](reference/serving.md#6-openai-chat-completions).

## Speculative decoding

- `--spec auto` is the default. It runs the speculation tuned for your model and GPU, which
  `ninferno-convert` attached to the model file. A GPU nobody tuned uses the portable `*` profile.
- Force a backend with `--spec mtp`, `--spec dflash` or `--spec dflash2`, and a depth with
  `--draft-tokens N`. Add `--lm-head-draft` for the optimized proposal head.
- Disable it with `--spec none`.
- Under concurrency the profile lowers the draft depth per decode step, because deep drafts stop
  paying off on a busy server. `--spec-load LANES:DEPTH,...` replaces those bounds (`none` removes
  them).
- Best settings per model and task: [CLI guide](cli.md#choosing-the-backend-and-draft-count).
- Internals, bounds, graphs and measured throughput under load:
  [reference](reference/serving.md#3-speculative-decoding).

## Memory, context and concurrency

- **Plan.** Without `--max-context`, the free GPU memory picks the first rung that fits of a
  declared plan ladder: the model's own context for every request at the top, then the second
  cached state of each request, the KV pool, the cached Device states (with the Host tier, on by
  default on Linux, WSL 2 and Windows; without it one per request stays down to 8,192 tokens), the
  context (steps of about 4.2 % down to 8,192 tokens at chunk 1024), then survival rungs with a
  warning and without the desktop reserve (8,192 tokens at chunk 1024, then narrower chunks, fewer
  cached states, shorter contexts). More free memory never selects less. The same model, options and card
  type always give the same ladder, so the same free memory gives the same plan. The startup log
  names the rung (`plan | rung k of N: ...`) and how much more free memory the rung above would
  need; `/v1/models` reports it as `meta.plan_level` and `meta.plan`, and `--plan` prints the whole
  ladder without starting. A prompt longer than the context returns HTTP 400
  `context_length_exceeded`.
- **Desktop reserve.** On Windows (WDDM) and WSL 2 the plan keeps a reserve for what Windows adds
  while the server runs: an 8th of the card, at most 4 GiB, when the card drives the display (a
  32nd, at most 1 GiB, headless); on a GPU sharing the host's memory (GB10, Jetson) an 8th, at
  least 4 GiB. It is given up only by a survival rung or a plan that fits nothing else, both with
  a warning, and by the options that keep none: the reserve belongs to an automatic KV pool, so an
  explicit `--kv-capacity` keeps none, and so does an explicit `--max-context` with
  `--max-concurrency 1` (the pool is then that one context). When the reserve
  alone limits the plan, the startup warns and names the option that gives it up, for example
  `--max-context 50,816 (or --kv-capacity 50,816) gives it up, and the desktop may stall`.
- **KV pool.** `--kv-capacity auto` takes the plan rung's pool, after weights, workspace and
  graphs. One shared pool serves all admitted requests. With an explicit `--max-context` and
  `--max-concurrency 1` the pool is that one context.
- **Concurrency.** `--max-concurrency` (1..8) sets how many requests decode together. Up to
  `--max-concurrency + --max-pending-requests` requests are accepted. Beyond that the server
  answers HTTP 429 `server_overloaded`. A request that waits longer than `--pending-timeout-ms`
  gets HTTP 503 `request_queue_timeout` (on `/v1/messages`: 529 and 504, as Anthropic does). Both
  carry `Retry-After`. A streamed request has its HTTP 200 and event-stream headers sent when it
  is admitted to the queue, so its queue timeout arrives as the dialect's error event
  (the same code and message) and the stream ends.
- **Speculation needs a profile.** `--spec auto` speculates with the profile measured for the
  model and GPU. A model without one for this GPU (most GGUF conversions of models nobody
  measured) decodes plainly, and the startup log says so; `--spec mtp` or `--spec dflash`
  speculates anyway at the default depth.
- **Speculation needs room.** With `--spec auto`, an explicit `--max-context`, more than one
  request and the automatic KV pool, the drafter's weights and state come out of the pool. When
  the pool then holds fewer full contexts than `--max-concurrency` and plain decode holds more, the server
  drafts with the model's own MTP head instead when it has one and that holds at least four fifths as
  many contexts (`speculation | fit fallback`), else it decodes without speculation and says so in
  the startup log (`speculation | ... decodes plainly`). Single-stream speed can then
  be less than half. Size the server for its load: a lower `--max-concurrency` or `--max-context`
  keeps the drafter, and an explicit `--spec` always keeps it.
- **Prefix cache.** Compatible prompt prefixes are reused automatically, for text and images. Reuse
  state lives in Device memory and in pinned Host memory (`--host-state-slots`, `--host-kv-mib`).
- **Does not fit?** Try a smaller quantization, a lower `--max-concurrency`, a shorter
  `--max-context`, or a smaller `--prefill-chunk`. The refusal message names options that fit.
- **Requests with images or video** need `--vision`. Without it they fail with HTTP 400
  `vision_disabled`.

Details and worked examples: [reference](reference/serving.md#13-execution-behavior).

## Authentication

Pass `--api-key VALUE`. Clients then send `Authorization: Bearer VALUE` (OpenAI) or
`x-api-key: VALUE` (Anthropic). Other users of the host can see command lines, so prefer
`--api-key-file PATH` (a file holding the key alone, on one line) or the `NINFERNO_API_KEY` environment variable.

```bash
curl http://127.0.0.1:8080/v1/models -H 'Authorization: Bearer local-secret'
```

- `GET /health` needs no key, and with `--cors` neither do CORS preflight requests; `/props` and
  `/metrics` do.
- An empty key is refused at startup.
- Browsers: without `--cors`, requests that carry an `Origin` header are refused with HTTP 403.
  curl and the official SDKs send none.

## Health, shutdown and logs

- `GET /health` returns `{"status":"ok"}` while the Engine accepts work, and 503 after an
  Engine-wide failure.
- SIGTERM or SIGINT starts a drain: new generation requests (every POST and DELETE) and `/health`
  get 503 while running requests finish; other GET requests are still answered, for at most `--drain-timeout-ms` (default 60000). A second signal stops at once.
- After an Engine-wide failure (for example a lost GPU) the server drains and exits with status 1.
  Run it under a supervisor that restarts on failure (systemd `Restart=on-failure`, a Docker
  restart policy).
- At the drain timeout, a request still running without streaming is answered 503
  `server_draining`; a stream stops.
- At start the server checks the release manifest once in the background and logs a newer release;
  `NINFERNO_UPDATE_CHECK=0` turns that off, `NINFERNO_UPDATE_URL` names another manifest.
- Every OpenAI response has an `x-request-id` header. Anthropic responses use `request-id`.
- Logs go to stderr, one line per event, with a throughput summary every 5 s
  (`--log-stats-interval-ms`). They never contain prompts or generated text.
- `--request-log-jsonl FILE` writes one JSON record per event for measurement. The format is in the
  [logging reference](reference/serving.md#12-logging-and-metrics).

## More

[Serving reference](reference/serving.md):

- [Model sources and first start](reference/serving.md#1-model-sources-and-first-start)
- [Startup configuration](reference/serving.md#2-startup-configuration)
- [Speculative decoding](reference/serving.md#3-speculative-decoding)
- [Endpoints and transport](reference/serving.md#4-endpoints-and-transport)
- [Chat Completions](reference/serving.md#6-openai-chat-completions)
- [Prompt caching](reference/serving.md#7-openai-prompt-caching)
- [Responses](reference/serving.md#8-openai-responses-core)
- [Anthropic Messages](reference/serving.md#9-anthropic-messages)
- [All options](reference/serving.md#11-server-options)
- [Logs and metrics](reference/serving.md#12-logging-and-metrics)
- [Execution behavior](reference/serving.md#13-execution-behavior)
- [Error codes](reference/serving.md#error-codes-summary)

Related: [CLI guide](cli.md), [performance](performance/README.md),
[weight conversion](weight-conversion.md).
