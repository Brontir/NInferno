# Deterministic decisions

Every runtime decision must be derived from the chip's specification, the artifact and the user's
explicit options. Nothing may be measured on the user's machine. Measurements happen in the lab and
reach the runtime only as facts or as constants.

This page states the rule and its target. The engine does not meet all of it yet. Known gaps:

- the attention split under `exact` follows the free-memory-sized context;
- a constrained row's measured mask time decides whether its masks are filled before launch and
  whether a DFlash row drafts, and the decode wait (spin or block) follows measured host lag; the
  wait cannot change bytes or batches;
- a token mask the device does not see released within its wall-clock wait bound ends the request;
- wall-clock bounds decide whether a request or the server goes on (outcome, not bytes): a queued
  request expires after `--pending-timeout-ms` (also during media preparation), a DFlash round
  whose drafts do not reach the host within 30 s fails, and a second transient CUDA error within
  5 minutes restarts the server;
- `tools/audit/decision_inputs.sh` reads each listed decision file's own text, not what it includes:
  the Qwen3.5 planning files still reach the Program's CUDA event timers through `program_impl.h`;
- MMQ prefill bytes depend on where a chunk is cut;
- free memory sets the context, KV and drafter room directly; the plan levels below do not exist
  yet;
- without NVML the chip is guessed from its SM count, and an unknown PCI ID still has
  speed estimates where it should have *no estimate*;
- under `fast`, the Qwen Gated DeltaNet prefill scan counts its 16-token chunks from each call's
  start, so after a call that starts off that grid (a turn's rewrite frontier, a private endpoint
  restore, forced tokens, the next Vision item) the bytes depend on the prefill chunk size, and a
  prefix restored from the cache or a KV-lease re-prefill of decoded tokens gives other bytes than
  the run it replaces; `exact` runs the scan token by token and is free of this;
- `fp32acc_full_rate` still stands in for per-segment instruction rates;
- decode/verify parity is unverified for 18 formats;
- the batch dependence of `fast` bytes at T ≤ 16 is not documented yet;
- no CI property checks invariance yet;
- the PCI-ID reference table has no per-SKU instruction rates yet (the L40's half-rate
  FP32-accumulate HMMA is still the generation row's `fp32acc_full_rate`);
- the speed labels and the kernel hash on constants do not exist yet;
- model constants (speculative acceptance) do not record their keys, corpus hash and tolerance yet;
- the admission planner's work units (1 an option, 200 an assessment or expansion, 10 000 / 50 000 a
  boundary, 1 us an option assumed) and the capture search's 10 us a target and 50 targets a
  decision are declared estimates sized for a worst case of about 200 us an assessment; they await calibration against the request log's
  `search_elapsed_ns` / `search_units` and `planning_elapsed_ns` / `planning_units` on lab hosts;
- run-ahead under a CPU quota is a declared rule not yet calibrated: a round runs 16 steps when the
  process's CPU bandwidth quota is below min(4, logical cores) CPUs, else 1 (both numbers await
  measurements on lab containers). Nothing measures the host, so a host stalled by other
  processes without a quota, or one whose stall 16 steps do not hide, is not detected; its
  operator sets `--decode-steps N` (1..32) explicitly;
- under `--spec auto`, a capacity refusal makes the server retry with snapped load bounds, the
  profile's fit fallback, shallower drafts, the model's MTP head or plain decode, so the free
  memory picks the speculation; the fingerprint only records it, as the taint
  `spec_fit_fallback=<steps>`;
- the fingerprint below exists as v1 (`ninferno-fingerprint/1`,
  `src/runtime/engine/decision_fingerprint.h`): it hashes the device facts (of the clocks, the
  reference facts the decisions read: source, reference and memory clock, bus width, share and
  `mma_slow`; the driver's clocks only where they fed a fact, an unknown id or a laptop row without
  the bottom of its range, and as observed otherwise, so an overclocked board fingerprints as a
  stock one and two laptops of one name from two OEMs alike), the host's
  logical CPU count, RAM class and CPU bandwidth quota, the build (the git commit is read at every
  build, not at configure), the requested options (the decode pacing and the maintainer options
  among them; no environment variable decides), the route table, dense- and MoE-tune files,
  context-cost presets, speculation profile and drafter registry by content, a separate drafter by
  its artifact id, the plan startup resolved without the free memory (KV and state storage,
  speculative windows, route breakpoints, decode steps per round) and the violations of this page
  present (taint). The RAM class is `bit_floor` of the whole GiB the host reports (64 for 64 to
  127 GiB; 0 below 1 GiB or unknown), so the few hundred MiB a kernel keeps do not move it. What
  startup sized from the free memory (context, chunk, KV tokens, cached states, Host tier) is
  observed and enters only the manifest, so `nf1` is the same across identical starts on one
  machine while `nm1` v1 still varies with the free memory until the plan levels exist. An input
  that cannot be read makes the fingerprint `unreadable` without stopping the start. It does not
  yet cover the reference facts no decision reads yet (TDP, VRAM, segment), plan levels, the
  constants and the kernel hash, nor the
  route each shape took: two runs with equal fingerprints read the same inputs, which is not yet
  proof of the same decisions.

The PCI-ID reference table is [`src/core/data/gpu_facts.json`](gpu-facts.md). The cost model, the
board distance and the speculation load model read its facts (`DeviceClass::facts`). For a listed
board that is the table's reference (boost) clock, a chip fact. An unknown PCI id, and a laptop row
without the bottom of its boost range, take the driver's clock (on the 2^(1/16) grid); the
speculation load model prices such a device but never trusts it (`GpuKey::facts_extrapolated`:
unsure, the conservative fallback).

Each gap is closed in its own change, which updates this list.

What the design guarantees once complete:

- No timing, tuning or calibration at startup.
- The same decisions on every machine with the same chip type, changing only with a release.
- Any decision can be reproduced on a CPU from the fingerprint the engine prints.

## Principles

1. **Copy facts, never decisions.** A reference table may say what a chip *is*: its clocks, memory
   bus, TDP, segment and nominal VRAM. It never says what a chip *should run*.
2. **The more a decision changes the output bytes, the fewer facts it may read.** Decisions fall
   into three tiers:
   - **B1, the byte class.** Activation precision, summation class, KV codec, decode/verify parity,
     placement and plan level. It reads only discrete device facts, the artifact and the options.
     `--activations exact` must read nothing else, so that its bytes are the same across a
     generation.
   - **B2, routes and depth under `fast`.** It may also read canonical continuous facts from the
     PCI-ID reference table. Its bytes are the same for the same chip type and the same request
     schedule.
   - **S, byte-neutral choices.** These may read any canonical fact.
3. **No timings, clocks or hidden state in decisions.** That rules out a cache left by an earlier
   run, an environment variable that changes a choice, and a budget in milliseconds. The monitor is
   write-only: timings, step latencies, throughput and NVML readings (clocks, temperature, power)
   may be logged and exported, never read back by a decision. Counts that follow from the request
   schedule (drafts accepted, rounds run, requests runnable) are not observations of the machine.
   A planner reports the points of its search to a caller that times them, and reads nothing back.
   The CTest `ninferno_decision_inputs_test` keeps clock and timing headers, clock reads and timers
   out of the text of the decision files it lists; what those files include is not checked (see
   the known gaps and [Runtime observations](#runtime-observations)).
4. **Constants are fitted on at least two lab cards.** A constant about the hardware and its
   kernels (a cost, an efficiency, a rate) is fitted on two cards of its generation and records
   the hash of the kernels it was fitted on. When those kernels change and nobody refits, the
   generation's speed estimates become "no estimate".
   A constant about a model, such as a drafter's acceptance per context class, records instead
   what it depends on:
   - the target and drafter artifact ids;
   - the whole B1 byte class it was measured under (activation precision, summation class, KV
     codec, decode/verify parity, placement and plan level);
   - the sampling settings and the draft depth;
   - the corpus, by hash;
   - under `fast`, the lane count, because the bytes at T ≤ 16 depend on the batch.

   Every card runs the same request schedule. The cards must come from two generations, or
   take different routes, so that their agreement means something. The constant records the
   relative tolerance its cards must agree within. When they disagree, the constant depends on
   the hardware: it is fitted per generation like a cost, and until then the forecasts that use
   it are "no estimate".
5. **Capabilities decide what is admissible; speeds decide what pays.**
6. **Decisions compare ratios, not absolute times.**
7. **The unknown is conservative and labelled.** This covers an unknown PCI ID, MIG, laptops, JIT
   PTX and a generation without a lab card.
8. **Every decision is reproducible on a CPU.** The rules may not read CUDA, clocks, the
   environment or files. They get their facts passed in.

## Layers of rules

Rules apply in this order:

1. **Capability gates:** what can run at all (tensor MMA, BF16, FP8, FP4, TMA).
2. **B1.**
3. **B2.**
4. **S.**
5. **Speed estimates.** These use a per generation x mechanism efficiency measured in the lab. A
   law relating SM count and clock to speed serves only as a prior, and only for a generation with
   no lab card.

## Facts

| Kind | Source | Examples |
|---|---|---|
| Discrete | driver | generation and instruction set, SM count, L2, shared memory per block and SM, registers, compiled-kernel occupancy, VRAM class, driver model, MIG, display attached |
| Discrete | PCI ID | segment (GeForce / pro / datacenter), exceptions (TU116/117: no fast tensor MMA; L40: half rate) |
| Discrete | OS, once at startup | the process's CPU bandwidth quota: the tightest along its cgroup and the cgroup's parents (v2 `cpu.max`, v1 `cpu.cfs_quota_us` / `cpu.cfs_period_us`), and the logical CPU count |
| Continuous | PCI-ID reference table only | reference clock, bus width x memory clock, TDP, nominal VRAM, source |

- An unknown PCI ID takes the driver's values rounded to a 2^(1/16) grid, and its facts are
  labelled *extrapolated*. Its speed estimate is *no estimate*.
- PCIe is the highest generation the link supports, never the current one.
- Free memory must be the only observation made while running. It selects one of at most four
  declared plan levels, and the chosen level goes into the fingerprint.

## Fingerprint

The fingerprint is a hash of the following, plus a digest of the decision manifest:

- the facts and their sources;
- the build;
- the options;
- the plan level;
- the constants;
- the kernel hash.

The engine prints it at startup (`fingerprint | nf1-… | manifest nm1-… | ninferno-fingerprint/1`;
the canonical text and the taint at debug level), exports it in `/metrics`
(`ninferno_decision_fingerprint_info`) and `/v1/models` (`meta.decision_fingerprint`,
`meta.decision_manifest`), writes it into the server-start record of the request log, and
`tools/bench/perf_matrix.py` writes it into every bench row. Two runs with the same fingerprint
must make the same decisions. The manifest, `nm1-…`, hashes the plan and the observed sizes: an
option change moves the fingerprint only, a plan change both, an observed size the manifest only.

## What will be promised

- Same fingerprint: same decisions.
- `--activations exact`: the same bytes on every card of a generation.
- `fast`: the same bytes for the same request schedule on the same chip type. At T ≤ 16 the bytes
  may depend on the batch's composition, and this is documented.
- Speed: a forecast with one of these labels:
  - *verified*: a lab card;
  - *covered*: within the lab's bounds for the generation;
  - *extrapolated*: laptops, MIG, the smallest chips;
  - *no estimate*: no lab card for the generation, an unknown ID, JIT, or a model constant whose
    lab cards disagree.

## Not allowed

- Any measurement on the user's machine: startup grids, tuning, calibration.
- A reference table that stores decisions.
- A law that turns SM count and clock into speed, used inside a decision. It may only serve as a
  prior for UI estimates.
- Corrections by enforced over default power limit.
- 5 % clock buckets: they move the discontinuity, they do not remove it.
- Memory type (HBM/GDDR) as an input to a decision.
- An MMQ gate on `!fp32acc_full_rate`: it only works on wrong rates.
- A decision tree per generation: it overfits.
- Batch-invariant `fast` by default: its cost is too high.
- A per-site profile (`--site-profile`). It is deferred until more than 10 % of covered cards leave
  their band.
- Promises of a confidence level that two lab cards per generation cannot support, such as "at
  least 90 %" or n/(n+1).

## Runtime observations

Where the engine observes the machine while it runs, and whether a decision reads it. Classes: (a)
logged or exported only; (b) moves only scheduling or latency, not the bytes of a fixed schedule
(S tier); (c) changes routes, bytes or depth.

| Observation | Where | Class |
|---|---|---|
| Context-cache planning search | `context_cache/materialization_planner.h`, `capture_planner.h` | counted work units, no clock; the ResourceManager times the reported points (a) |
| Concurrent-prefill interleave | `engine_core.h`, `interleave_cost.h` | priced by shape, no clock |
| Decode run-ahead | `round_pacing.h` | from the CPU-quota fact, no clock |
| Decode wait, spin or block | `round_pacing.h` | (b), measured host lag |
| Grammar compile timeout | `structured/constraint.cpp` | (b), fails a request, no bytes |
| Queue expiry (`--pending-timeout-ms`) | `runtime/engine/engine_core.h` (submit, admission head and backfill), `qwen3_5/frontend/media_cache.cpp` | (b), expires a waiting request, no bytes |
| DFlash draft arrival, 30 s | `qwen3_5/program/decode.cpp` (`fill_dflash_draft_masks`) | (b), fails a stuck round, no bytes |
| Second transient CUDA error within 5 minutes | `runtime/engine/engine_core.h` (`kTransientCudaErrorWindow`) | (b), restarts the server instead of going on |
| Mask fill order, constrained DFlash drafting | `structured/request_constraint.cpp` → `qwen3_5/program/decode.cpp` | (c), open |
| Device mask-release wait bound | `ops/kernel/sampling_device.cuh` | (c), open: ends a request |
| Load-time MoE and dense tuners | `ops/moe/moe_tune.cpp`, `ops/generic/generic_tune.cpp` | (c), opt-in by environment, off by default |
| Free memory after warm-up | `runtime/engine/model_instance.cpp` | (c), open (gaps above) |
| Acceptance watchdog | `spec/watchdog.h` | counts of the request's own drafts, not a timing |
| Planning, transfer, phase and request timings; CUDA event profiling; NVML in the UI | `resource_manager.h`, `runtime/contract/timing.h`, `engine_core.h`, `speculation/decoder.cu`, `serve/`, `ui/gpu_stats.cpp` | (a) |

## Review checklist

When a change adds or alters a decision, its PR states:

1. The decision's tier (B1, B2 or S).
2. The facts it reads, and that each is a device fact, the artifact, an option or a reference-table
   fact.
3. Which of the following it reads, and why; the answer must be none:
   - a timing;
   - a clock;
   - free memory, outside the plan level;
   - an environment variable;
   - a file written by an earlier run.
4. The lab cards that set its constants, and for a model constant its keys and tolerance.
5. A property test showing invariance where the tier requires it.

Route tables (`routes/*.json`, `routes/tuner/dense_tune.json`) and speculation profiles keyed by GPU
class remain only as a transition. They must be deterministic per chip type, and no new decision
keyed by card identity may be added. Once the lab experiments show what they are worth end to end,
they move out of the runtime into lab tools, replaced by rules over facts.
