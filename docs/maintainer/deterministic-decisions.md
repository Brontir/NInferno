# Deterministic decisions

Every runtime decision must be derived from the chip's specification, the artifact and the user's
explicit options. Nothing may be measured on the user's machine. Measurements happen in the lab and
reach the runtime only as facts or as constants.

This page states the rule and its target. The engine does not meet all of it yet. Known gaps:

- the attention split under `exact` follows the free-memory-sized context;
- the dense tuner's user cache, local spec-tune, environment knobs, `calibrate_floor` and budgets in
  milliseconds still steer choices;
- MMQ prefill bytes depend on where a chunk is cut;
- free memory sets the context, KV and drafter room directly; the plan levels below do not exist
  yet;
- the cost model reads driver clocks: they are 0 on MIG/WSL, overclocks leak into it, and without
  NVML the chip is guessed from its SM count;
- `fp32acc_full_rate` still stands in for per-segment instruction rates;
- decode/verify parity is unverified for 18 formats;
- the batch dependence of `fast` bytes at T ≤ 16 is not documented yet;
- no CI property checks invariance yet;
- the TU116/117 exception, the PCI-ID reference table, the speed labels and the kernel hash on
  constants do not exist yet;
- under `--spec auto`, a capacity refusal makes the server retry with snapped load bounds, the
  profile's fit fallback, shallower drafts, the model's MTP head or plain decode, so the free
  memory picks the speculation; the fingerprint only records it, as the taint
  `spec_fit_fallback=<steps>`;
- the fingerprint below exists as v1 (`ninferno-fingerprint/1`,
  `src/runtime/engine/decision_fingerprint.h`): it hashes the driver's device facts, the host's
  logical CPU count and RAM class, the build (the git commit is read at every
  build, not at configure), the requested options, the route table, dense- and MoE-tune files,
  context-cost presets, speculation profile and drafter registry by content, a separate drafter by
  its artifact id, the decision environment variables that are set, the plan startup resolved
  without the free memory (KV and state storage, speculative windows, route breakpoints) and the
  violations of this page present (taint). The RAM class is `bit_floor` of the whole GiB the host reports (64 for 64 to
  127 GiB; 0 below 1 GiB or unknown), so the few hundred MiB a kernel keeps do not move it. What
  startup sized from the free memory (context, chunk, KV tokens, cached states, Host tier) is
  observed and enters only the manifest, so `nf1` is the same across identical starts on one
  machine while `nm1` v1 still varies with the free memory until the plan levels exist. An input
  that cannot be read makes the fingerprint `unreadable` without stopping the start. It does not
  yet cover the PCI-ID reference facts, plan levels, the constants and the kernel hash, nor the
  route each shape took: two runs with equal fingerprints read the same inputs, which is not yet
  proof of the same decisions.

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
   run, an environment variable that changes a choice, and a budget in milliseconds.
4. **Constants are fitted on at least two lab cards of a generation.** Each constant records the
   hash of the kernels it was fitted on. When those kernels change and nobody refits, the
   generation's speed estimates become "no estimate".
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
  - *no estimate*: no lab card for the generation, an unknown ID, or JIT.

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
4. The lab cards that set its constants.
5. A property test showing invariance where the tier requires it.

Route tables (`routes/*.json`, `routes/tuner/dense_tune.json`) and speculation profiles keyed by GPU
class remain only as a transition. They must be deterministic per chip type, and no new decision
keyed by card identity may be added. Once the lab experiments show what they are worth end to end,
they move out of the runtime into lab tools, replaced by rules over facts.
