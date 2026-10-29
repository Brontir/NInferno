# User measurement kit

The lab measures a few cards, and the plan prices every card from its facts
(`docs/maintainer/gpu-facts.md`). The kit lets a user check those prices on a card we do not have. One
command measures what our stands measure and records it beside what the plan predicts for the same
start, in one file the user can send us. Nothing in the kit feeds a decision. A submission is
evidence that a maintainer reviews: the validator says which estimates it confirms or contradicts.
A contradiction is fixed in the model from facts, after it is reproduced, and never by fitting a
constant.

## The command (user side)

    python tools/kit/measure.py --bin DIR [--budget-minutes 30] [--models auto|ID,...] [--device N]
        [--passes 2] [--hard50] [--spec-tune SECONDS] [--out FILE] [--yes]

- **`--bin DIR`** holds `ninferno-serve`, plus `ninferno` when `--spec-tune` is given: a release or a build.
- **Requirements:** Python 3.9+ standard library only, on Windows or Linux. `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set, so `--device` is `nvidia-smi`'s index.
- **Downloads:**
  - The models are fetched first through `ninferno-serve --prepare-only` into the user's hub cache.
  - The kit states what it will fetch and asks first (`--yes` skips the question).
  - A model whose fetch fails is not measured.
  - Downloads are outside the budget.
- **Budget:** it counts GPU time only. A model's passes run whole or not at all, and what the budget skips is recorded.
- **Progress:** each step prints as `[step i/N, used of budget] what`.

## What it runs

1. **Card facts.**
   - From `nvidia-smi`: name, PCI bus id, driver and driver model (WDDM/TCC), VBIOS, memory, power limits (current, default, maximum, enforced), maximum and application clocks, PCIe link, pstate, the clock event reasons now.
   - Fields are probed through `--help-query-gpu`: `clocks_event_reasons.*` on drivers 535 and newer, `clocks_throttle_reasons.*` before. An N/A value is kept as unknown, never as a pass.
   - The plan's `device` block.
   - The OS, the CPU, the RAM and, on Windows, AC or battery.
2. **Model set.**
   - `tools/kit/models.json` (versioned; its hash goes in the result) lists `{id, repo, quant, context}` in priority order. The last entry is small, so every supported card yields a row.
   - Each entry is planned on this card with `--plan --no-load --json` and the exact flags of the start (`--max-context C --max-concurrency 1 --spec none`).
   - The first `--max-models` (2) whose plan fits are measured.
3. **Per model, plain** (`--spec none`): `--passes` starts (2, a restart each, so a pass never reuses another's cache). Each start:
   - Records its `choice |` line and `/v1/models` (decision fingerprint, speculation).
   - Records the compute processes before and after.
   - Holds a 5 s idle window under telemetry.
   - Sends a warm-up request.
   - Measures the chat template's overhead, then sizes each timed prompt on its own slice of text: up to two
     probes per slice, each led by its own nonce line so nothing of a probe is reused.
   - **Decode:** `--prompts` (3) requests at the plan's decode depth (`choice.chosen.depth`), each answering `depth/8` tokens (32..256). Decode = `(predicted_n - 1) / predicted_ms`, from the server's own `timings` (serving.md).
   - **Prefill:** `--prompts` requests at the choice's typical prompt (`choice.task.prompt`, the length `prefill_tok_s` is priced at). Prefill = `prompt_n / prompt_ms`.
   - Every prompt is a distinct slice of the long text `tools/bench/perf_matrix.py` uses. A request with a reused prefix (`cache_n > 0`), or a decode request whose answer ended before half its tokens, is kept but excluded.
4. **Telemetry:** `nvidia-smi --query-gpu=... -lms=500` for the whole start, columnar:
   - SM and memory clocks, temperature, power, utilization, memory used, pstate and the clock event reasons;
   - a mark per window (idle, warm-up, `decode i`, `prefill i`).
5. **Optional, recorded only (the validator does not judge them in this version):**
   - `--hard50` runs `tools/bench/hard_set.py` against the plain start (as the lab's `q_hard`). With 50 items, the binomial standard error is about 7 points: it can show a gross failure, never a quality difference.
   - `--spec-tune S` runs `ninferno spec-tune <artifact> --budget-seconds S --publish ... --measured-on "<card> user-kit"`, the run our stands publish registry entries from. The plan's `--spec auto` prediction goes beside it.

## The result file: `ninferno-user-run/1.minor`

One JSON object. An additive change bumps the minor version, and the validator accepts any minor of
major 1.

| Field | What |
|---|---|
| `schema` | `"ninferno-user-run/1.0"` |
| `kit` | `{version, models_version, models_sha256}` |
| `created` | UTC time |
| `ninferno` | `{version}`: the first line of `ninferno-serve --version` |
| `host` | `{os, machine, cpu, cpus, ram_bytes, on_battery}` |
| `card` | `{smi: the nvidia-smi facts, plan_device: the plan's device block}` |
| `budget` | `{asked_s, used_s, skipped}` |
| `runs[]` | one per model and mode |

A plain run holds the following fields.

| Field | What |
|---|---|
| `id`, `repo`, `quant`, `mode` | the entry, `mode` `plain` |
| `flags` | the plan's and the start's flags |
| `plan` | `chosen` (the plan's `choice.chosen`); `prompt` (`choice.task.prompt`); `document` (the plan JSON's `schema, ask, device, choice, estimate, pick, rule`, `model` without `source`, and `plan`'s scalar fields) |
| `passes[]` | `start_s`; `served` `{id, fingerprint, speculation, choice}`; `processes_before` and `processes_after` `{listed, others, informative}`; `requests[]` (each with `prompt_tokens, cache_n, prompt_n, prompt_ms, predicted_n, predicted_ms, excluded?`); `measured` `{decode, prefill: {median, min, max, n, prompt_tokens}}`; `telemetry` `{period_ms, t, columns, marks, reason_fields}`; `error` |
| `error` | why a run was not measured (fetch, budget) |

**Privacy.**
- Every string is scrubbed before writing: the home, the working directory and the hub cache become `<local>`, and other absolute paths become `<path>`.
- Keys that hold file paths are dropped.
- Under WDDM only process counts are kept, never PIDs or names.
- No host name, user name, GPU UUID or serial is collected.

## Submitting (never automatic)

The kit writes `FILE.json` and `FILE.md`. The `.md` file is a ready PR or issue body with:
- the validator's report on this machine;
- the `git`/`gh` commands for a PR adding the JSON under `docs/performance/user-runs/`;
- the issue alternative.

The kit never runs them: the user sends it.

## The validator (our side)

    python tools/kit/validate.py RUN.json [--json] [--tolerance 0.10]

It applies the rules below in order, then gives per run and per metric **accepted** or **rejected**
with the rule. Each accepted metric gets its predicted and measured value and the ratio, and a
verdict:
- **confirms**: within the tolerance (±10 %);
- **contradicts**: outside it;
- **not used**: the card is capped, or R3 could not be checked.

A verdict covers **this board** only: this quant, length and C=1. A card name's other boards (OEM,
laptop) are not covered. When the estimate was `speed_transferred` from another card, the verdict
tests the transfer rule, not this card's facts. Speculation and hard50 runs are recorded, not judged,
in this version.

### Rejection rules (stated, applied in order)

| Rule | Rejects | When |
|---|---|---|
| R0 schema | the file | a major other than `ninferno-user-run/1`, or no `card` or `runs` |
| R1 identity | the file | the plan's device name differs from `nvidia-smi`'s |
| R1 identity | the run | a start's `choice \|` line names KV or activations other than the plan's, or the passes started with different decision fingerprints |
| R2 other load | the pass | another compute process listed on the card (TCC/Linux only: under WDDM the list holds the desktop's processes and says nothing); the median utilization of the idle window above 5 %; memory used growing more than 256 MiB across it |
| R3 thermal and slowdown | the pass, for that metric | a sample in the metric's windows with hardware slowdown, hardware or software thermal slowdown, or power brake. A software power cap is recorded only: under load it is the card's stock behaviour, which the model must price. No reason fields, or no sample in a window, is *unknown*: the metric is kept, but no verdict |
| R4 caps | the file is marked `capped`: kept, never used to confirm | the enforced power limit below the default; application clocks below the default; the applications-clocks event reason active; on battery. Fields N/A on the card are listed as "R4 could not check" |
| R5 repeats | the metric | the passes' medians more than 5 % apart, or one pass's min..max more than 10 % of its median |
| R6 completeness | the metric | fewer than 2 usable passes (a pass with an error is not usable) |
| R7 length | the metric | the measured prompt tokens more than 5 % from the length the plan priced (decode: the depth; prefill: the task's prompt) |

These numbers (5 %, 256 MiB, 5 %, 10 %, 5 %, ±10 %) are set at the top of `validate.py`. A change
goes through this page first.

## Not yet (later PRs)

- Judging speculation: acceptance (tokens per round) against the registry pair, and round cost
  against the bytes model R(k). Only a SPEC `basis` of `similar`, `generic` or `preliminary` is a
  real prediction; `own` would be circular.
- A `registry_add.py` path for user entries (`measured_on ... user-kit`, refused without
  `--accept-user`).
- Detecting an overclock against the gpu-facts boost clock (record the clocks now; judge later).
- Multi-GPU selection beyond `--device`.
