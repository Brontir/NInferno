# Deterministic decisions

Every runtime decision is derived from the chip's specification, the artifact and the user's
explicit options. Nothing is measured on the user's machine. Measurements happen in the lab and
reach the runtime only as facts or as constants.

What a user gets from this:

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
     `--activations exact` reads nothing else, so its bytes are the same across a generation.
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

## Facts

| Kind | Source | Examples |
|---|---|---|
| Discrete | driver | generation and instruction set, SM count, L2, shared memory per block and SM, registers, compiled-kernel occupancy, VRAM class, driver model, MIG, display attached |
| Discrete | PCI ID | segment (GeForce / pro / datacenter), exceptions (TU116/117: no fast tensor MMA; L40: half rate) |
| Continuous | PCI-ID reference table only | reference clock, bus width x memory clock, TDP, nominal VRAM, source |

- An unknown PCI ID takes the driver's values rounded to a 2^(1/16) grid and is labelled
  *extrapolated*.
- PCIe is the highest generation the link supports, never the current one.
- Free memory is the only observation made while running. It selects one of at most four declared
  plan levels, and the chosen level goes into the fingerprint.

## Fingerprint

The fingerprint is a hash of the following, plus a digest of the decision manifest:

- the facts and their sources;
- the build;
- the options;
- the plan level;
- the constants;
- the kernel hash.

The engine prints it at startup, exports it in `/metrics` and writes it into every bench row. Two
runs with the same fingerprint make the same decisions.

## What is promised

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
- Power-limit corrections in decisions.
- Memory type (HBM/GDDR) as an input to a decision.
- Promises of a confidence level that two lab cards per generation cannot support.

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

Route tables and speculation profiles keyed by GPU class remain only as a transition. They must be
deterministic per chip type, and no new decision keyed by card identity may be added.
