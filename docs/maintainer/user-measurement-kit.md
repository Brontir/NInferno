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
- **Requirements:** Python 3.9+ standard library only, on Windows or Linux. `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set and `CUDA_VISIBLE_DEVICES` dropped, so `--device N` is the N-th card by PCI bus id. Every `nvidia-smi` call names that card by its bus id, never by nvidia-smi's own index, which can follow another order.
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
   - The OS, the CPU, the RAM, and AC or battery (Windows `GetSystemPowerStatus`, Linux `/sys/class/power_supply`; a host without a battery is on AC). An unknown power source gives no verdict.
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
5. **Optional:**
   - `--hard50` runs `tools/bench/hard_set.py` against the plain start (as the lab's `q_hard`). With 50 items, the binomial standard error is about 7 points: it can show a gross failure, never a quality difference. Its budget estimate is 50 answers of about 1100 tokens at the plan's decode. Qwen3.5-9B Q4_K_M on an RTX 4060 Ti took 18 minutes. The run records the question set's SHA-256.
   - `--spec-tune S` runs `ninferno spec-tune <artifact> --budget-seconds S --publish ... --measured-on "<card> user-kit"`, the run our stands publish registry entries from, under telemetry (one window, `spec-tune`, and the compute processes while it runs and after). Where the registry has lab pairs of the repository and quantization, it runs their configurations (`--configs`): the default menu drafts through the proposal head where the artifact has one (`+head`), which no lab pair measured, so neither the pairs' acceptance nor the plan's round costs could be held to it. Beside the tune the run records:
     - the `--spec auto` plan's `plan.speculation` (`basis`, `source`, `served`, `by_thinking`, `fallbacks`, `notes`, and `round_cost`: what a round of each configuration costs in the resolved profile, in plain decode steps; serving.md);
     - the drafter manifest beside the artifact (`<artifact>.speculation.json`): the engine family and each drafter's kind and status, its repo@revision only when the registry declares that repository (else `other`); no profiles, no paths;
     - the target's identity: the hub revision and the SHA-256 of its weight files, hashed from their bytes (about 10 s per 5 GB; `blob_matches` says whether the hub's blob name agrees), the key of the registry's pairs.

## The result file: `ninferno-user-run/1.minor`

One JSON object. An additive change bumps the minor version, and the validator accepts any minor of
major 1.

| Field | What |
|---|---|
| `schema` | `"ninferno-user-run/1.1"` (1.1 added the spec and hard50 runs' fields below and `pci.device_id`) |
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

A `hard50` run holds `hard50` (hard_set.py's summary and `wrong`, the ids answered wrong),
`set_sha256` and `served.choice`. A `spec` run holds `plan` `{chosen, error, speculation}`,
`configs` (the lab pairs' menu it ran; empty: spec-tune's default), `manifest`, `target`
`{revision, weights: [{sha256, bytes, blob_matches?}]}`, `profile` (spec-tune `--out`), `publish`,
`processes_before`, `processes_after` and `telemetry`.

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
tests the transfer rule, not this card's facts. Speculation and hard50 runs have their own rules
(below).

### Rejection rules (stated, applied in order)

| Rule | Rejects | When |
|---|---|---|
| R0 schema | the file | a major other than `ninferno-user-run/1`, or `card` not an object, or `runs` not a list of objects |
| R1 identity | the file | the plan's device name differs from `nvidia-smi`'s |
| R1 identity | the run | a start's `choice \|` line names KV or activations other than the plan's, or the passes started with different decision fingerprints. A start that reports no choice line or fingerprint is *unknown*: no verdict |
| R2 other load | the pass | another compute process listed on the card (TCC/Linux only: under WDDM the list holds the desktop's processes and says nothing); the median utilization of the idle window above 5 %; memory used growing more than 256 MiB across it |
| R3 thermal and slowdown | the pass, for that metric | a sample in the metric's windows with hardware slowdown, hardware or software thermal slowdown, or power brake. A software power cap is recorded only: under load it is the card's stock behaviour, which the model must price. No reason fields, or no sample in a window, is *unknown*: the metric is kept, but no verdict |
| R4 caps | the file is marked `capped`: kept, never used to confirm | the enforced power limit below the default; application clocks below the default; the applications-clocks event reason active; on battery. Fields N/A on the card are listed as "R4 could not check" |
| R5 repeats | the metric | the passes' medians more than 5 % apart, or one pass's min..max more than 10 % of its median |
| R6 completeness | the metric | fewer than 2 usable passes (a pass with an error is not usable) |
| R7 length | the metric | a contributing request's prompt tokens more than 5 % from the length the plan priced (decode: the depth; prefill: the task's prompt), or no token count |

These numbers (5 %, 256 MiB, 5 %, 10 %, 5 %, ±10 %) are set at the top of `validate.py`. A change
goes through this page first.

## Judging speculation (`spec` runs)

A spec-tune measures two things, and they are judged apart:
- **Acceptance**, the tokens a round per configuration and corpus section: a property of the target, its drafter and the greedy corpus, not of the card (the six lab cards of Qwen3.5-9B Q4_K_M agree within 3 % per section). It is held to the registry's lab pairs of the same target.
- **Round cost**, what a round costs in plain decode steps: R(k). The user's is computed from the tune's own measurements as `spec::profile_round_costs` computes it (per section a round's seconds over plain decode's seconds a token, the mean over the sections). It is held to the plan's `round_cost` for the same configuration.

What the plan's round cost is, from its `source`, is said in every verdict, and only the first is a test of the facts:
- **the facts' R(k)**: the source says the rounds were priced from the weights' bytes and this card's rates;
- **the facts raised to a lab card's measurement**: a configuration the source lists after "conservative where the lab card measured rounds dearer" (part measurement);
- **a lab measurement, not the facts**: any other source, quoted (this class's own pair, a generation's agreeing lab cards, a sibling quantization, a generation's ratio). The verdict then tests that a lab measurement carries to this board.

The round costs are the resolved profile's: a configuration of a drafter whose profile the plan did not resolve, or deeper than the portable depth, has no prediction.

| Rule | Rejects | When |
|---|---|---|
| S1 the tune | the run | no profile or publish (spec-tune failed); a configuration with a section without a round. A tune that left a drafter `untested` or skipped configurations is judged on what it measured, but is no registry candidate |
| S2 identity | no reference | the target's weights not recorded (no pair can be named); the lab pairs of the target (by the weights' SHA-256) of another engine key (`key.engine`); settings other than the pairs' (context, chunk, KV, greedy; the wait mode moves the clock, never a token); a separate drafter other than the pair's (repo@revision) |
| S3 load and slowdown | round cost | another compute process listed on the card during or after the tune (TCC/Linux); a slowdown, thermal or power-brake sample in the `spec-tune` window. Unknown reasons, a capped card (R4) or an unknown power source: kept, no verdict. Acceptance is token counts and survives all of these |
| S4 steady plain | round cost | the tune's plain decode across its sections more than 5 % apart (the card's speed moved during the tune) |
| S5 acceptance | the configuration | the tokens a round pooled over the sections ((accepted + rounds) / rounds) more than 10 % from the median of the lab cards' over the same sections and prompt counts (lab pairs of another menu, other sections or another drafter are left out, each said): **contradicts** the registry pairs' acceptance (a broken drafter, conversion or kernel on this card); within: **confirms**. The worst section is reported beside it. Fewer than 2 lab cards with the configuration (a `+head` one included: the lab measures no proposal head): no reference |
| S6 round cost | the configuration | the measured R(k) more than 10 % from the plan's `round_cost`: **contradicts** that prediction (of the kind said above); within: **confirms**. A configuration the plan does not price (deeper than the portable depth): no prediction |

A spec run is a **registry candidate** when the file and the run are accepted, every drafter was tested and no configuration was skipped or ran a section without a round, the target's identity and the manifest are recorded, no configuration's acceptance contradicts the lab pairs, and the settings are the pairs' (when there are pairs). Being a candidate says only that `registry_add.py --from-user-run` will take it.

## Judging hard50 (`hard50` runs)

`tools/kit/hard50_reference.json` holds the lab's hard50 runs of each model (plain decode, one per lab card, the campaign and date, and the ids each answered wrong) and the question set's SHA-256. The kit's run records the same: its count, its wrong ids, the set's SHA-256 and the start's `choice |` line (its KV and activations).

| Rule | Rejects | When |
|---|---|---|
| H1 the run | the run | not 50 questions, or a request that failed (`errors` above 0) |
| H2 the set | no verdict | the run's question set differs from the reference's, or was not recorded; or the reference has no run of this repository and quantization |
| H3 below the lab | the verdict | either: fewer correct than the lab's lowest card minus two binomial standard deviations at the lab's median (`2 * sqrt(50 p (1 - p))`, p the median over 50); or, with 3 lab cards or more, more questions that every lab card answered right answered wrong here than `2 m + 2`, m the most any lab card has against the others (greedy decoding: the cards differ by numerics on a fixed set, and Qwen3.5-9B's seven cards differ by at most one such question). **Below the lab**: a gross failure (a broken kernel, a wrong template, a corrupted download); otherwise **consistent with the lab**. Never "better": 50 items cannot show it |

The reference is refreshed by hand from the stands' `q_hard` runs after a change that moves answers, never to fit a submission.

## A user's tune into the registry (maintainer)

    python tools/spec/registry_add.py --from-user-run RUN.json --accept-user [--run ID] [--registry PATH]

It validates the file again (the same `validate.py`, against `--registry`) and refuses unless the spec run is a registry candidate, or without `--accept-user`: a submission is evidence, and an entry is made only by a maintainer who read the validator's report. It builds the pair as `--pair` does, from what the run recorded (the tune, the manifest, the target's revision and weights), and:
- marks it `user_kit` `{kit, schema, created, ninferno, card, measured_on}`, and gives it the user's `board` (the PCI device id) instead of a `lab_board`;
- puts it into `tools/spec/user_pairs.json` (`--user-pairs FILE`), not the registry: the engine embeds the registry, and its hash is in every manifest, so a user's pair there would make every manifest stale. A user's pair is recorded, never served. Served, it would count as a lab card wherever the selector counts cards (two agreeing cards, the facts' template card, a generation's ratio, a group's choice), so serving it waits for the selector to tell user pairs apart (serving its own board only, after the lab's);
- refuses a target the lab measured on the same chip in the registry (the lab's measurement stands); a second run of the same class and board replaces the first.

The tool changes the file in the working tree (and its `version`); the change reaches the main branch only through a maintainer's PR, never automatically.

## Not yet (later PRs)

- Detecting an overclock against the gpu-facts boost clock (record the clocks now; judge later).
- Multi-GPU selection beyond `--device`.
