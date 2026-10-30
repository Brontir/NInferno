# NInferno CLI

`build/apps/ninferno` runs one request against one converted `.ninferno` model on one NVIDIA GPU
(Volta `sm_70` through Blackwell `sm_120`).

Before you start:

1. Build NInferno ([project README](../README.md)).
2. Convert a model from its original Hugging Face checkpoint. Use
   `ninferno-serve --prepare-only <org/name>` ([serving guide](serving.md)) or `ninferno-convert`
   ([weight conversion](weight-conversion.md)). NInferno publishes no prebuilt model files.

The examples use Qwen3.8-27B NVFP4 with FP8 KV storage.

## Quick start

Text prompt with MTP speculation:

```bash
./build/apps/ninferno models/qwen3_8_27b_nvfp4.ninferno \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 32768 --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

Image input from a messages file (run from the repository root; media paths are relative to it):

```bash
./build/apps/ninferno models/qwen3_8_27b_nvfp4.ninferno \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 --max-new 128 --kv-dtype fp8 --vision \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

Save the answer and the log separately:

```bash
./build/apps/ninferno models/qwen3_8_27b_nvfp4.ninferno \
  --prompt "Return one sentence." --max-context 4096 --max-new 64 --kv-dtype fp8 \
  > answer.txt 2> run.log
```

Run `./build/apps/ninferno --help` for the exact option contract.

## Options

Exactly one of `--prompt` and `--messages` is required. The model path is the first argument.

### Input and output

| Option | Default | Meaning |
|---|---|---|
| `--version` | | Print the version and exit. |
| `--prompt TEXT` | none | Text prompt. |
| `--messages FILE` | none | JSON messages file ([format](#structured-messages)). |
| `--max-new N` | `128` | Output-token limit. Counts every committed token, including inserted control tokens. |
| `--stop-token-id N` | none | Extra stop token ID. Repeatable. |
| `--stop TEXT` | none | Extra stop string on the answer channel. Repeatable. |
| `--reasoning-stop TEXT` | none | Extra stop string on the reasoning channel. Repeatable. |
| `--raw-output` | off | Show the frontend's raw output stream. |
| `--print-token-ids` | off | Include generated token IDs in the diagnostics. |
| `--log-level trace\|debug\|info\|warning\|error\|critical\|off` | `info` | Log detail. `debug` shows every startup phase. `warn` and `fatal` are taken as `warning` and `critical`. |

### Context and memory

> **Changed: which GPU runs on a host with several.** The default GPU and `--device N` used to
> follow CUDA's enumeration, `FASTEST_FIRST` unless `CUDA_DEVICE_ORDER` said otherwise: on a host
> with an RTX 3060 on PCI bus 01 and an RTX 4090 on bus 02, GPU 0 was the 4090. Now both count
> GPUs in PCI bus order, as `nvidia-smi` does, so GPU 0 there is the 3060. To pin a card whatever
> the order, name it: `--device GPU-<uuid>` (from `nvidia-smi -L`) or
> `--device pci:0000:02:00.0`. At startup, a host with more than one GPU logs which one runs
> (`device | 2 GPUs visible; using 0000:01:00.0 ...`). MIG instances are not named by `--device`:
> choose one with `CUDA_VISIBLE_DEVICES=MIG-<uuid>`, and the process runs on it by default.

| Option | Default | Meaning |
|---|---|---|
| `--device N\|pci:BUSID\|GPU-UUID` | `0` | The GPU: N counts GPUs in PCI bus order, as nvidia-smi does (0 first); `pci:0000:01:00.0` or `GPU-<uuid>` (`nvidia-smi -L`) names one board. |
| `--max-context N` | `2048` | Per-sequence logical context ceiling. |
| `--kv-capacity N\|auto` | `--max-context` (`2048`) | Shared Main Text KV pool size. `auto` takes the plan ladder's pool rung the free GPU memory fits. Must be at least `--max-context`. |
| `--prefill-chunk N` | `1024` | Text-prefill chunk. Positive multiple of 128. |
| `--plan --no-load` | off | Print the plan these options give without loading the weights (the plan ladder, the step that fits, the speculation it would run), then exit; no `--prompt` needed. `--json` prints it as JSON (`ninferno-dry-plan/1`), `--plan-device NAME` plans for a named board. An estimate: the start decides ([reference](reference/serving.md#the-plan-without-the-weights)). |
| `--kv-dtype auto\|bf16\|int8\|fp8\|nvfp4\|k8v4` | `auto` | KV storage. `auto`: the [configuration choice](maintainer/configuration-choice.md) for the model on this GPU, logged on a `choice \|` line (see [KV storage](#kv-storage-and-context)). |
| `--sliding-kv-dtype bf16\|fp8` | follows the KV cache | Storage of sliding-window KV: `fp8` under a narrower KV codec where the family's rings take it, else `bf16`. |
| `--state-dtype f32\|bf16\|f16` | one step finer than the KV cache | Recurrent-state storage: `f32` under a `bf16` KV cache and under `--activations exact`, else `f16` (Qwen 3.5). |
| `--activations exact\|fast\|int8` | chosen (`fast`) | Activation precision on quantized linears ([details](#activation-precision)); omitted, the configuration choice picks it. |
| `--rope-yarn-factor F` | `1` | Qwen 3.5/3.6: rotate with YaRN over the checkpoint's own context and admit `F` (`1..64`) times it. Other families refuse any other value. |
| `--vision` | off | Enable image input (Qwen, Gemma 4) and video input (Qwen), and load the Vision GPU allocations. A GGUF needs its `mmproj` file beside it. |
| `--no-cuda-graph` | graphs on | Disable CUDA Graph decode. |
| `--decode-steps auto\|N` | `auto` | Ordinary decode steps per round ([decode round pacing](#decode-round-pacing)). |
| `--decode-wait auto\|spin\|block` | `auto` | How the host waits for a round's end ([decode round pacing](#decode-round-pacing)). |
| `--cuda-sync spin\|blocking\|yield\|auto` | `spin` | CUDA device synchronization schedule ([CUDA synchronization](#cuda-synchronization)). |

Maintainers also have `--route-table FILE`, `--route-dir DIR`, `--drafter-registry FILE`,
`--debug-prefill-cuts P1,P2,...`, `--debug-poison BYTE`, `--debug-poison-skip SITE,...` and the
route pins below ([maintainer options](maintainer/environment.md)). No environment variable stands in for any of
them. The lab tuners (`--dense-tune`, `--moe-tune`, `--tune-cache`) time kernels on the machine,
so `ninferno` refuses them; they run in `ninferno-perplexity` (and the dense tuner in
`ninferno-dense` and `ninferno-gemma4`). They and `--cuda-sync` are process-wide: an Engine built while another of the process is
alive must ask for the same values, or it is refused.

#### Route pins (maintainers)

Every automatic choice of a kernel family can also be made by hand, so that an A/B run can check
that the automatic one is the faster. These are test options: a pin other than `auto` is taint in
the decision fingerprint, and the startup log prints a `routes | pinned` warning for each.

| Option | Values | Pins |
|---|---|---|
| `--prefill-route` | `auto`, `a16`, `w4a8`, `i8`, `mmq`, `w4a4` | The projections of calls of 17 tokens or more: BF16 activations (the register MMA, tiled or staged tiles), the int8 W4A8 tiles, the Q4/Q5/Q6 G64 int8 tiles, ggml's int8 matrix kernel, or NVFP4 activations on NVFP4 weights. |
| `--decode-route` | `auto`, `a16`, `r8`, `vec`, `mmq` | The projections of calls of 1..16 tokens (decode, speculative verify): BF16 activations, the int8 R8 kernel (1..8 tokens), ggml's dp4a vector kernel (1..8 tokens), or ggml's matrix kernel (5..16 tokens). |
| `--moe-path` | `auto`, `fused`, `generic` | The routed experts: the family's fused MoE kernels (Qwen 3.5/3.6 SparseMoe, Gemma 4 NVFP4 experts) or the generic MoE Op. |

`auto` is the release's selection, unchanged. A pinned family takes every call of its band it can
compute; a call it cannot (another format, a K or alignment its tiles do not take, a width past its
kernel, a policy without its activations) keeps the automatic route. A pin is refused at startup,
naming what can run instead, when this build and GPU hold no kernel of the family, when
`--activations` rules it out (`exact` takes no prefill or decode pin; `int8` no `w4a4`), when no
startup call of its band could take it, or (`--moe-path`) when no MoE layer of the model has the
choice. `ninferno`, `ninferno-serve` and the serve bench tools (`tools/bench/run_serve_corpus.py`,
`run_serve_concurrency.py`) take all three; `ninferno-perplexity` takes `--prefill-route` and
`--moe-path`. Details, and what each pin does not reach:
[route pins](maintainer/environment.md#route-pins).

### Speculative decoding

| Option | Default | Meaning |
|---|---|---|
| `--spec auto\|none\|mtp\|dflash\|dflash2` | `auto` | Backend (`off` is `none`). `auto` follows the speculation profile that came with the model for this GPU; when that does not fit the context it gives way as the server does (the profile's fit fallback, a shallower draft, the model's MTP head, plain decode), with a warning. `none` decodes plainly. |
| `--draft-tokens N` | unset | `1..15`, within the model's own ceiling ([Speculative decoding](#speculative-decoding)). Required with an explicit `--spec mtp\|dflash\|dflash2`, refused under `auto` (the server takes the profile's depth instead). |
| `--lm-head-draft` | off | Use the optimized proposal head. Needs an explicit `--spec mtp\|dflash\|dflash2`; Qwen models only. |
| `--ngram-draft-tokens N` | auto (`15` where the family runs copy rounds) | Copy rounds beside MTP: a request whose last tokens recur earlier in its context verifies up to `N` tokens that followed them instead of its MTP drafts. Output unchanged. `0` turns them off; families without copy rounds refuse a nonzero value. |
| `--ngram-min-match M` | `12` | Tokens that must recur before a copy round runs, at least 1. |
| `--acceptance-floor TOKENS_PER_ROUND` | profile's, or off without a profile | Watchdog floor, `0..1000`. |

### Thinking and templates

| Option | Default | Meaning |
|---|---|---|
| `--chat-template FILE` | artifact template | Local Jinja template. |
| `--no-thinking` | template default | Disable thinking. |
| `--reasoning-effort none\|minimal\|low\|medium\|high\|xhigh\|max` | template default | Effort passed to the template. `none` = `--no-thinking`; other values cannot be combined with it. |
| `--thinking-budget N` | unlimited | Positive cap on model-origin thinking tokens. Cannot be combined with `--no-thinking` or `--reasoning-effort none`; other efforts combine. |

### Sampling

Omitted flags take the registered preset of the loaded architecture and prompt mode
([presets](#sampling-presets)).

| Option | Default | Meaning |
|---|---|---|
| `--greedy` | off | Exact argmax decoding (temperature 0). |
| `--temperature F` | preset | Temperature, `0..2`. |
| `--top-p F` | preset | Nucleus threshold, `0..1`. |
| `--top-k N` | preset | `0..64`; `0` selects the top-64 cap. |
| `--min-p F` | preset | Min-p threshold, `0..1`. |
| `--presence-penalty F` | preset | Presence penalty, `-2..2`. |
| `--frequency-penalty F` | preset (`0`) | Frequency penalty, `-2..2`. |
| `--seed N` | `0` | Sampling seed, any unsigned 64-bit integer. |

## Output streams

- Answer content goes to stdout.
- Startup milestones, runtime errors, reasoning and the result report go to stderr, without
  timestamps. The report has timings, throughput, GPU memory, token IDs (if requested) and
  speculation statistics. So stdout can be redirected on its own.
- On a terminal, weight loading is one transient progress line, then a compact Engine-ready summary.
- Redirected stderr has persistent progress for long loads and no carriage returns or ANSI escapes.
- Option and local prompt/message input errors are direct command diagnostics.

## Structured messages

`--messages` takes a non-empty JSON message array, or an object with `messages` and an optional
`tools` array.

```json
[
  {"role": "system", "content": "Answer concisely."},
  {"role": "user", "content": [
    {"type": "image", "image": "examples/cli/media/visual_chart.png"},
    {"type": "text", "text": "Describe the chart."}
  ]}
]
```

Roles: `system`, `developer`, `user`, `assistant`, `tool`. The selected template formats them. The
maintained Qwen templates keep system/developer messages at their input positions.

Content is a string or an ordered array of parts:

| Content type | Source field | Accepted source |
|---|---|---|
| text | `text` | string |
| image / image_url | `image` or `image_url` | local path, HTTP(S) URL, or base64 data URI |
| video / video_url | `video` or `video_url` | local path, HTTP(S) URL, or base64 data URI |

`image_url` and `video_url` may be strings or objects with a string `url`. Assistant history may
include `reasoning_content` and `tool_calls`. A tool result uses role `tool` and `tool_call_id`.

Committed text, image, video, mixed-media, thinking, long-decode and long-context inputs are in
[`examples/cli/`](../examples/cli/).

## Thinking controls

- Omitted thinking and effort options use the template's defaults. The template interprets the
  effort.
- `--greedy` selects argmax decoding independently of thinking options.
- With `--thinking-budget N`, if the model has not emitted `</think>` after N accepted tokens in
  an open new-turn Qwen thinking block, Engine appends
  [Qwen's canonical early-close guidance](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md)
  and `</think>` to the same resident sequence without sampling. The guidance goes to the
  reasoning stream. Generation then continues from the updated context.
- A natural thinking close, stop condition, cancellation, or total output/context limit at the
  boundary takes priority and suppresses the insertion.
- `--max-new` must leave room for the full tokenizer-derived control suffix plus one post-close
  model token whenever the output capacity extends beyond the budget. An undersized request is
  rejected, not truncated.
- Inserted guidance appears on stderr as reasoning. `--print-token-ids` includes the inserted IDs.
  `--raw-output` keeps the raw control representation.

Example: at most 512 model-origin thinking tokens, with room for the suffix and the answer:

```bash
./build/apps/ninferno models/qwen3_8_27b_nvfp4.ninferno \
  --prompt "Explain speculative decoding, then give a concise conclusion." \
  --max-context 4096 --max-new 1024 --thinking-budget 512 \
  --kv-dtype fp8 --spec mtp --draft-tokens 3 --lm-head-draft
```

Override the template:

```bash
./build/apps/ninferno models/qwen3_8_27b_nvfp4.ninferno \
  --chat-template tools/chat_templates/qwen3_8.jinja --prompt "Hello"
```

## Activation precision

| Mode | Behavior |
|---|---|
| `exact` | BF16 activations on every quantized linear. The answer does not depend on how the prompt is cut or on speculation. On Qwen models the Gated DeltaNet layers also prefill token by token, as decode runs them; the extra time grows linearly with the prompt: about 1.3 to 2 ms per such layer per 4096 prompt tokens on an RTX 5090 (35B-A3B and 27B head counts), more on older cards. |
| `fast` (default) | Encodes activations on wide calls (prefill, batched verify): 4 bits for NVFP4 weights; 8 bits (int8, one scale per 32 values) for 4/5/6-bit integer formats that record `AllowA8` (GGUF Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, IQ4_XS, Q4_K, Q5_K, Q6_K and u4 with zero points). |
| `int8` | Eight bits: every quantized linear may take int8 activations (one scale per 32 values) where its route has one, and none takes four, whatever the artifact recorded; a head or embedding recorded BF16-only keeps BF16. This includes NVFP4 weights recorded for 4 bits with a calibrated input scale (a W4A4 checkpoint), which before ran their 4-bit route under int8 too. NVFP4 weights take int8 activations on prefill calls of 17 tokens or more on every GPU but Volta (V100 multiplies NVFP4 with BF16 activations on its tiled route), including those `fast` runs at 4 bits (a W4A16 checkpoint such as nvidia/Qwen3.6-27B-NVFP4: `exact` 16, `int8` 8, `fast` 4 bits on RTX 50); on RTX 50 (FP4 tensor cores) decode keeps BF16, and so does speculative verify on the layers with native NVFP4 kernels (generic-path NVFP4 shapes, such as shared experts and shapes without a native kernel, take int8 on every call of 17 tokens or more, a wide verify included), while GPUs without FP4 tensor cores use int8 there too. Qwen models only; other families refuse it at startup. |

`fast` also uses int8 activations (one scale per 32 values) for codebook experts at decode widths:
the IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and IQ3_S gate/up banks of the generic MoE layer, as in
Qwen3.5/3.6 MoE GGUF artifacts. This applies to calls of at most four tokens (a prompt of at most
four tokens included) on GPU classes whose MoE route lists it. `--activations exact` disables it.

## Startup memory profile

GPU residency is fixed when the Engine starts. The defaults give the smallest resident profile.

- No `--spec` (and no model profile): MTP/DFlash/DFlash2 weights, their state and the optimized
  proposal head are omitted.
- `--spec mtp`, `--spec dflash` and `--spec dflash2` (Qwen3.8-27B) load only the selected
  backend.
- A backend used with the full proposal head omits the optimized proposal head.
- Vision is off by default: its weights and Vision-specific workspace extent are omitted.
  `--vision` loads the weights, expands the one Program workspace for Vision encode/handoff, and
  enables image/video input.
- The one-request CLI uses root-only context mode. It reserves no extra Device checkpoint
  StateImage and captures no continuation that no later request could use.

The complete `.ninferno` inventory is still validated. This is not lazy loading: an Engine started
without Vision rejects media and cannot enable Vision later. DFlash/DFlash2 and Vision may be
enabled together. The speculative backends apply to text decode after multimodal prefill; they do
not speed up Vision encode.

## Speculative decoding

Speculation follows `--spec` (see the options table). One backend per Engine.

- MTP: up to the model's own depth (five for Qwen3.5, seven for Nemotron-H, 15 for the Gemma 4
  assistant).
- DFlash (Qwen, and dense and MoE decoders such as gpt-oss with a paired drafter) and DFlash2
  (Qwen3.8-27B): one to fifteen, or fewer where the drafter's block is narrower
  (gpt-oss's holds eight). Seven forms block length eight;
  fifteen gives the maximum block length sixteen.
- `--lm-head-draft` needs a selected backend.
- DFlash2 needs an artifact with the companion weights; without them, selecting it reports a
  missing DFlash2 component. `groupwise-int` and `nvfp4` artifacts use the same Engine route,
  including CUDA Graph, concurrent requests, sampling penalties and prefix reuse.
- Vision, MTP and DFlash follow the same rule: weights are required only when the component is
  enabled at startup.

```bash
# MTP, 35B-A3B
./build/apps/ninferno models/qwen3_6_35b_a3b.ninferno \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 --lm-head-draft

# DFlash, 35B-A3B
./build/apps/ninferno models/qwen3_6_35b_a3b.ninferno \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 --kv-dtype fp8 \
  --spec dflash --draft-tokens 7 --lm-head-draft
```

### Choosing the backend and draft count

A greedy C=1 sweep over 40 prompts measured every backend and draft count, all with
`--lm-head-draft`. The prompts: 8 each of math, code, open prose, and math and code with thinking.
Outputs at most 512 tokens, context 4,096. Fastest settings:

| Artifact | Best overall | Math | Code | Prose | Thinking |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | `--spec dflash --draft-tokens 5`: 2.04×, 8 % over MTP K=3 | DFlash K=5 | DFlash K=11 | MTP K=2 | DFlash K=5 |
| Qwen3.8-27B `nvfp4` with DFlash2 | `--spec dflash2 --draft-tokens 15`: 4.36×, 63 % over MTP K=3; K=11 within 1.3 % | DFlash2 K=13 | DFlash2 K=15 | DFlash2 K=7 | DFlash2 K=15 |
| Qwen3.8-27B `nvfp4` without DFlash2 | `--spec mtp --draft-tokens 5`: 3.13×, 17 % over MTP K=3 | MTP K=5 | MTP K=5 | MTP K=3 | MTP K=5 |

- Speed-ups are over plain greedy decode of the same prompts.
- If 35B-A3B must use MTP, K=4 is 4.8 % faster than K=3.
- Long reasoning accepts fewer drafts, so the best K gets smaller.
- Under concurrency the server bounds these depths ([serving guide](serving.md)).
- Full tables, with acceptance and tokens per round: [performance matrix](performance/README.md).
  The recorded serving results used MTP K=3 and DFlash or DFlash2 K=7; those runs keep their
  configurations.

## KV storage and context

- The Qwen3.6/3.8 models in this guide have a native context limit of 262,144 tokens. Other
  families use the model's own limit. What fits on a GPU depends on free memory, model, media
  workload, output budget and KV type.
- The artifact describes its model configuration and weight representations. `--kv-dtype` chooses
  runtime KV storage independently.
- `--kv-dtype auto` takes the [configuration choice](maintainer/configuration-choice.md) on a GPU
  generation with lab constants (Ada today), and the model family's choice elsewhere:
  - `int8` for Qwen 3.5/3.6/3.8;
  - `fp8` for the dense decoders (gpt-oss, Llama, Mistral, Granite, Muse Glimmer, ...) on GPUs that
    convert FP8 in hardware (`sm_89` and up: Ada, Hopper, Blackwell); `bf16` below that;
  - `fp8` for Gemma 4's full-attention layers on `sm_89` and up, `bf16` below that;
  - `bf16` for Nemotron-H.
- Gemma 4 takes `bf16` or `fp8`. With `fp8`, the full-attention layers' page pool holds E4M3 codes
  with an FP32 scale per position and KV head (any head size). Sliding-window rings stay BF16.
- Accuracy of `fp8` against BF16 keys and values, on chat answers: top-1 agreement 0.98-0.99, ΔNLL
  within 0.004 (Gemma 4 12B QAT, and 26B-A4B with BF16 activations). Under the default
  `--activations fast` the gap is larger on models whose prefill runs 4-bit activations (NVFP4,
  W4A4; Gemma 4 26B-A4B top-1 0.94), because that profile amplifies small cache changes. For
  maximum accuracy use `--activations exact`.
- The prepared prompt must fit `--max-context`. Generation stops at the remaining context capacity.
- `--kv-capacity N` sets the shared physical Main Text KV pool independently, rounded up to the
  64-token page size. The one-request CLI normally omits it so it follows `--max-context`; the
  distinction matters mainly for a concurrent Engine or server.
- `--kv-capacity auto` loads the weights, measures the remaining GPU memory once, and takes the
  first rung of the declared plan ladder that fits (the pool steps down from every request's full
  context by about 4.2 % a step; with an automatic context the context then steps the same way
  down to 8,192 tokens; a prefill chunk narrower than 1024 is a survival rung, served with a
  warning (the one-request CLI keeps no cached states); the rungs come
  from the model, the options and the planner's bytes for the whole enabled runtime layout:
  selected speculative backend, fixed sequence state, unified workspace and CUDA Graph allowance).
  Beside a rung it keeps 32 MiB for the CUDA driver's own allocations (streams, events, kernel code
  loaded on first launch) and 64 MiB for the kernel warm-up; under WDDM (native Windows and WSL 2),
  where Windows shares the card and pages it out to Host memory when full, a class reserve of a
  32nd of it, at most 1 GiB (an eighth, at most 4 GiB, on a card that also drives the display). On
  GPUs that share memory with the host (GB10, Jetson) the reserve is an eighth, at least 4 GiB, and
  the Host tiers are off. Only a survival rung or a plan that fits nothing else gives it up. The
  same model, options and card type give the same ladder
  ([plan levels](maintainer/deterministic-decisions.md#plan-levels)); the pool is not probed or
  resized at request time. `ninferno-serve --plan` prints the ladder.

### Workspace

At startup NInferno reserves model weights, persistent sequence state, one phase-reused Program
workspace, and a separate CUDA Graph driver allowance.

- With Vision, the workspace has a general execution prefix and a fixed item-output handoff region.
  Vision encode may reuse the full backing before producing output. Text/MTP/decode work stays in
  the general prefix while the handoff is live. Capacity is the maximum legal simultaneous extent,
  not the sum of Text, Vision scratch and Vision output.
- Text prefill uses `min(--prefill-chunk, --max-context)`. Vision keeps a 32,768-token aggregate
  prompt budget but plans Device execution for the registered 16,384-token maximum single item.
- Requests do no project-owned device allocation or growth.
- Context-cache capacity controls are absent from this one-request interface. The persistent
  Engine and server routes own cross-request reuse and optional Host backing.
- All allocations are released when the Engine is destroyed.

## Sampling presets

Used when a sampling flag is omitted. Frequency penalty is `0` in every preset. Task-specific
profiles (such as Qwen's precise-coding profile) use explicit overrides.

| Model | Prompt mode | Temperature | Top-p | Top-k | Min-p | Presence penalty |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.6-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.8-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.8-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | thinking | `1.0` | `0.95` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |

Other families take the checkpoint's `generation_config.json` (temperature, top-p, top-k), the same
in both prompt modes with no presence penalty, and transformers' defaults (temperature 1, top-p 1,
top-k 50) where it names none.

## CUDA synchronization

`--cuda-sync` selects the CUDA device synchronization schedule at startup, for the CLI, the HTTP
server, `ninferno-perplexity`, `ninferno spec-tune` and the `ninferno-gemma4` lab tool
(`EngineOptions::pacing.cuda_sync`).

| Value | Behavior |
|---|---|
| `spin` (default) | Lowest synchronization latency; uses CPU while waiting for the GPU. |
| `blocking` | The waiting thread sleeps. The decode cost depends on the host. |
| `yield` | Yields the CPU while waiting. |
| `auto` | CUDA's scheduling heuristic, not an automatic performance benchmark. |

```bash
./build/apps/ninferno models/qwen3_8_27b_nvfp4.ninferno --cuda-sync blocking --prompt "Hello"
```

The Engine-ready log reports the mode. An unrecognized value, or a failure to apply the schedule,
fails startup. The setting controls device scheduling (including stream
synchronization); it does not override individual CUDA event creation flags.

## Decode round pacing

A container's CPU quota (CFS bandwidth limit) is shared by every thread of the cgroup, including
builds and other programs. Once it is used up, every thread that needs CPU stops for the rest of the
100 ms period, whatever its `nice` priority. A decode that waits for each step before submitting
the next loses the whole stall every time.

Engines with CUDA Graphs pace their decode rounds (rows without constrained decoding):

- **Run-ahead.** A round runs one step, or 16 when the process's CPU bandwidth quota is below
  min(4, logical cores) CPUs, feeding each step's sampled tokens to the next on the GPU, so the GPU
  keeps working while the host is stopped. The quota is the tightest along the process's cgroup
  and its parents (cgroup `cpu.max`, or `cpu.cfs_quota_us` / `cpu.cfs_period_us` under cgroup v1,
  read once at startup; none on Windows). Four CPUs cover the threads the engine keeps busy while
  decoding, and a quota at or above the logical cores can never run out.
  `--decode-steps N` replaces that default. The count is never chosen from timings: it
  decides which requests share each step's batch. The startup log reports it on the
  `engine ready` line (`decode N step(s) per round`, then `CPU quota ... CPUs` when there is one).
  - Nothing measures the host. On a contended host the rule does not see (other processes use the
    CPU without a quota, or a quota of 4 CPUs or more that a busy cgroup still runs out of), or
    where 16 steps do not hide the stall, set `--decode-steps N`: about 100 ms divided by
    the step time, at most 32. Both thresholds await calibration on lab containers.
  - Tokens reach the stream in groups of that size. A request that stops mid-round has its
    remaining steps discarded.
  - Applies to Gemma 4 and dense-decoder models. Models with recurrent state (Qwen 3.5/3.6,
    Nemotron-H) and speculative engines (MTP, DFlash) run one step (one verify) per round.
- **Wait mode.** A quiet host spins on the end of each round. When rounds lag (wall time exceeds
  GPU time by more than 2 ms, twice within a second), until five seconds after the last lag, the
  host waits by spinning, or by blocking (a blocking-sync CUDA event) where that decodes faster.
  The wait follows timings, but it cannot change a round's output or the next round's batch.
  - It tries both for a second each (at least 8 rounds, at most 3 seconds). It spins first when the
    first lag begins, and tries the kept wait first afterwards.
  - It blocks for the next 30 seconds only if it ran at least 3 % more steps per second while
    blocking (a speculative round counts as one step, however many drafts are accepted).
  - It tries again at once if the kept wait's rate halves or grows by half (the load changed).
    The choice holds while the lag pauses and returns.
  - Blocking spends almost none of the quota and usually wins. Under a load heavy enough that the
    quota runs out early in every period, a blocked host can wake too late for any of it, and
    spinning does better.

A request that arrives during a run-ahead round starts after it, so its time to first token can grow
by up to one round. Greedy outputs do not depend on the step
count at concurrency 1. With several requests, they stay identical only where the kernels give the
same result at every batch size: a run-ahead round keeps its batch for all its steps, while one step
per round admits and retires requests between steps.

| Option | Values | Default | Meaning |
|---|---|---|---|
| `--decode-steps` | `auto`, or integer `1..32` | `auto`: 16 under a CPU quota below min(4, cores) CPUs, else 1 | `N` runs N steps in every round that can; `1` disables run-ahead. An engine that cannot run ahead says so once at startup and runs one step per round. |
| `--decode-wait` | `auto`, `spin`, `block` | `auto` | `auto` chooses as above while rounds lag. `spin` never blocks (the CUDA synchronization schedule alone). `block` blocks in every round. |

Both are `ninferno` and `ninferno-serve` options (`EngineOptions::pacing`); other values are
refused. `ninferno spec-tune` waits as a quiet host does (`--decode-wait spin`) unless its own
`--decode-wait` says otherwise, and records the wait in its settings (`--load --resume` takes only
cells measured with the same wait).

## Measuring speculation profiles (maintainers)

`ninferno spec-tune <model.ninferno>` measures which speculation configuration decodes fastest on
this GPU over the built-in corpus and writes a profile; `ninferno spec-reselect <profile.json>`
selects again over a saved tune's measurements with this build's rules, without measuring. Both
take `--key value` pairs and refuse unknown keys. The full rules are in `apps/cli/spec_tune.cpp`.

| Option | Default | Meaning |
|---|---|---|
| `--configs LIST` | every configuration the artifact supports | Comma-separated labels to measure (`mtp:3,dflash:5,...`). |
| `--budget-seconds S` | `900` | Stop starting new configurations after `S` seconds. |
| `--repeat R` | `1` | Corpus passes per configuration. |
| `--kv-dtype`, `--max-context N`, `--prefill-chunk N` | `bf16`, `4096`, `1024` | The engine settings measured with. |
| `--corpus FILE` | built in | Prompt corpus. |
| `--decode-wait auto\|spin\|block` | `spin` | How the tune's engines wait for their rounds ([decode round pacing](#decode-round-pacing)). |
| `--cuda-sync MODE`, `--route-table FILE`, `--route-dir DIR`, `--drafter-registry FILE` | `spin`, the shipped tables, the compiled-in registry | The CUDA synchronization schedule, the route tables the tune's engines run (a lab tunes against tables under test) and the drafter registry; recorded in the settings when given, the files by content (one that cannot be read is refused). |
| `--device N\|pci:BUSID\|GPU-UUID` | `0` | The GPU, as `ninferno` takes it: N counts GPUs in PCI bus order. |
| `--out PROFILE` | `<model>.ninferno.spec_profile.json` | Where the profile goes. Serving reads it only when named: `ninferno-serve ... --spec-profile PROFILE` (likewise `ninferno` and the Gemma 4 app); a named tune that does not serve (unreadable, or another artifact, GPU class or drafters) refuses start-up. |
| `--profile PROFILE` | `<model>.ninferno.spec_profile.json` | With `--load`: the tune whose choices the load bounds are measured for. |
| `--publish FILE --applies-to REPO,... --measured-on TEXT [--gpu-class CLASS] [--requires LIST]` | none | Also write a registry entry (`tools/spec/registry_add.py` adds it). |
| `--load C,...` | none | Measure the bound per concurrency instead (`2..8`, default set `2,4,8`), with `--load-depths`, `--temperature`, `--top-p`, `--thinking on\|off\|corpus`, `--tolerance`, `--min-gain`, `--max-tokens`, `--draft-width` and `--resume FILE`. |

`spec-reselect` needs `--publish`, `--gpu-class`, `--gpu`, `--applies-to` and `--measured-on`, and
takes `--engine MODEL_TYPE` and `--out PROFILE`.
