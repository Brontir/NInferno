# User measurement kit

The lab measures a few cards; the plan prices every card from its facts (`docs/maintainer/gpu-facts.md`).
The kit lets a user check those prices on a card we do not have: one command measures what our stands
measure, beside what the plan predicts for the same start, and writes one file the user can send us.
Nothing in the kit feeds a decision directly. A submission is evidence that a maintainer reviews. It
can confirm or contradict an estimate or a registry entry, and the validator reports which.

## The command (user side)

    python tools/kit/measure.py --bin DIR [--budget-minutes 30] [--hard50] [--models auto|ID,...]
        [--device N] [--out FILE] [--yes]

- `--bin DIR` is the directory holding `ninferno-serve` and `ninferno` (a release or a build).
  Nothing else needs building. Python 3.9+ standard library only, on Windows or Linux.
- The kit downloads the models it measures through `ninferno-serve --prepare-only` (the user's hub
  cache). It states the bytes it will fetch first and asks (`--yes` skips the question).
- Progress goes to stdout as `[step i/N, mm:ss of budget] what`. A step whose stated cost would pass
  the budget is skipped and recorded as skipped. It is never cut halfway.

## What it runs (the stands' tools, unchanged)

1. **Card facts.**
   - `nvidia-smi --query-gpu`: name, PCI bus id, driver, VBIOS, memory total, the default and
     current power limit, the default and current application clocks, maximum clocks, PCIe link.
   - `ninferno-serve <model> --plan --no-load --json`: the `device` block and the GPU key the plan
     prices with.
   - The OS, the CPU and the RAM.
2. **Model set.** `tools/kit/models.json` (versioned) lists `{id, repo, quant, context}` in priority
   order. The kit plans each entry on this card (`--plan --no-load --json`, the same flags as the
   run) and keeps those whose plan fits, in order, while the budget allows.
3. **Per model, plain** (`--spec none`): the prediction and the measurement of one start.
   - Prediction: `--plan --no-load --json` gives `choice.chosen` (decode at its depth, prefill,
     KLD, storages) and `choice`'s decision fingerprint.
   - Measurement: `ninferno-serve <model> --max-concurrency 1 --spec none` with the same flags, then
     `tools/bench/perf_matrix.py --skip-chat --prompts 3 --chars <depth x 3.9>`: decode at the
     plan's depth and prefill of the same prompts, median and min..max, each after a warm-up.
   - This runs twice. The two passes are the repeat the validator compares.
4. **Per model, speculation** (models with a drafter):
   - `ninferno spec-tune <artifact> --budget-seconds B --publish FILE --applies-to REPO --measured-on
     "<card> user-kit"`. This is the run our stands publish registry entries from.
   - Its profile and its registry entry are kept.
   - The plan's prediction beside it is `--plan --no-load --json --spec auto`: the drafter and the
     depth the plan picks, and the decode it predicts with them.
5. **Quality, only with `--hard50`:** `tools/bench/hard_set.py` against the plain start (as the
   lab's `q_hard`), then against the plan's speculation start.
6. **Telemetry during every measured window:** `nvidia-smi --query-gpu=... -lms 500` (SM and
   memory clocks, temperature, power draw, utilization, `clocks_event_reasons.active`, memory used),
   plus `--query-compute-apps` before and after.

## The result file: `ninferno-user-run/1`

One JSON object. A field is added only with a schema version bump, and the validator refuses an
unknown major version.

| Field | What |
|---|---|
| `schema` | `"ninferno-user-run/1"` |
| `kit` | `{version, models_version}` |
| `created` | UTC time |
| `ninferno` | `{version, commit}` from `ninferno-serve --version` |
| `host` | `{os, cpu, ram_bytes}`; no user, host name or path |
| `card` | the `nvidia-smi` facts above, and `plan_device` (the plan's `device` block) |
| `budget` | `{asked_s, used_s, skipped: [...]}` |
| `runs[]` | one per model and mode, below |

A run:

| Field | What |
|---|---|
| `id`, `repo`, `quant`, `mode` | the entry; `mode` is `plain`, `spec` or `hard50` |
| `flags` | the server flags, the same for the plan and the start |
| `plan` | `choice.chosen` as the plan wrote it; `fingerprint` (the plan's decision fingerprint) |
| `served` | `{fingerprint, speculation}` from the server's `/v1/models` |
| `passes[]` | per pass, the `perf_matrix` rows (`prefill`, `decode`: value, spread, prompt_len) |
| `telemetry` | per pass: `{samples, sm_clock_mhz: [min, median, max], mem_clock_mhz, temp_c_max, power_w_median, util_idle_before, event_reasons: [...], other_processes: [...]}` |
| `spec` | the spec-tune profile summary and the registry entry it wrote |
| `hard50` | `hard_set.py`'s summary |
| `error` | the reason when the step failed (the file still records the other runs) |

## Submitting (never automatic)

The kit writes `<out>.json` and `<out>.md`. The `.md` file is a ready PR or issue body: the card, the
runs, the validator's own report on this machine, and the two ways to send it:
- a PR adding the JSON under `docs/performance/user-runs/`;
- an issue with the JSON attached.

It also shows the `git`/`gh` commands for either way. The kit never runs them: the user sends the
file.

## The validator (our side)

    python tools/kit/validate.py RUN.json [--registry FILE] [--tolerance-file FILE]

It reads a submission, applies the rules below in order, and prints one report: per run, per metric,
**accepted** or **rejected** with the rule.

For each accepted metric it gives the predicted and the measured value with the ratio, and the entry
it would **confirm** (within tolerance) or **contradict** (outside it). The entries are:
- the speed model's card entry (`spec/registry` GPU class) for plain decode and prefill;
- the drafter registry entry for the model and GPU class, for speculation (its drafter and depth
  against spec-tune's pick, its tokens per round against the measured ones).

### Rejection rules (stated, applied in order)

| Rule | Rejects | When |
|---|---|---|
| R0 schema | the file | unknown `schema` major, missing `card` or `runs` |
| R1 identity | the file | the plan's device name or memory disagrees with `nvidia-smi`, or `served.fingerprint` differs from `plan.fingerprint` (the start did not run what was planned) |
| R2 other load | the pass | another compute process on the card, or utilization above 5 % in the idle sample before the pass |
| R3 thermal / power events | the pass | any sample in the window whose `clocks_event_reasons` holds thermal slowdown (software or hardware), hardware slowdown or power brake |
| R4 clock caps | the file, marked `capped` (kept for the record, not used to confirm) | the current power limit below the default, application clocks below the default, or a locked SM clock |
| R5 repeats | the metric | the two passes' medians differ by more than 5 %, or one pass's min..max spans more than 10 % of its median |
| R6 completeness | the metric | fewer than 2 passes, or a pass with an error |

The tolerances (5 %, 10 %, 5 % utilization) are stated here and in `validate.py`, and a change goes
through this page. Confirm or contradict uses ±10 % for decode and prefill by default
(`--tolerance-file` for another).
