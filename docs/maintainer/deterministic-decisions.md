# Deterministic decisions

Every runtime decision must be derived from the chip's specification, the artifact and the user's
explicit options. Nothing may be measured on the user's machine. Measurements happen in the lab and
reach the runtime only as facts or as constants.

This page states the rule and its target. The engine does not meet all of it yet. Known gaps:

- under `exact`, the DFlash drafter's attention over its context (`context_softmax_attention`)
  still takes its split count from the free-memory-sized context, so the drafts and how many are
  accepted can follow free memory; under greedy decoding the emitted bytes do not (the target's
  verify decides them). The target's causal attention splits follow each token's position only;
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
- the plan ladder's rungs are declared, but under `--spec auto` each Engine build the server tries
  (the profile's choice, its fit fallback, shallower drafts, plain decode) picks its own rung from
  its own free reading, and the kernel warm-up may still move the plan to a lower rung when it took
  more than its declared allowance (recorded as `selected_rung` in the plan summary, and as
  `observed.rung_selected` in the fingerprint's manifest);
- the dry run (`ninferno-serve --plan`) uploads the weights to size the rungs: the planner reads
  the bound model, and no family plans from the artifact alone yet;
- `ninferno-gemma4`'s session fit (a maintainer tool) still bisects the context on the free memory;
- without NVML the chip is guessed from its SM count, and an unknown PCI ID still has
  speed estimates where it should have *no estimate*;
- under `fast`, the Qwen Gated DeltaNet prefill scan counts its 16-token chunks from each call's
  start, so after a call that starts off that grid (a turn's rewrite frontier, a private endpoint
  restore, forced tokens, the next Vision item) the bytes depend on the prefill chunk size, and a
  prefix restored from the cache or a KV-lease re-prefill of decoded tokens gives other bytes than
  the run it replaces; `exact` runs the scan token by token and is free of this;
- launch counts stand as constants, the same on every board but not yet fitted on two lab cards
  or hashed with their kernels: the BF16 GDN gating split bands (byte-neutral) and the Gemma 4
  decode attention's 64-key split floor on the RTX 5090 alone; the rope 256-thread bound (1020
  tokens), the RMSNorm prefetch bound (170 blocks) and the sparse-MoE prefill grid bound (5440
  CTAs) on the RTX 5090 and 5080, where an SM-count rule lost on the 5080;
- SM-count rules rest on few boards: the literals that were the RTX 5090's 170 SMs and decided a
  grid (the FP8 TMA split wave, the NVFP4 2048x5376 TMA route) are rules over the device's SM
  count (`src/ops/common/sm_wave_rules.h`), the NVFP4 crossing fitted on the RTX 5090 and checked
  on the RTX 5070 Ti only; the generic R8 rule's IQ3 edge at 96 SMs (`generic_gemm.cpp`,
  `rules_r8`) is a bound placed between the two measured boards (84 and 170);
- the Gemma 4 Q8 5376x16384 projection keeps a literal tuned on the RTX 5090 that no device fact
  explains: the 256-token tile at T a multiple of 1024, on every board
  (`q8/shapes/n5376_k16384.cu`). Against the 128-token tile at those widths it wins on 60, 108
  and 170 SMs (RTX 4070 Ti 2-6 %, A100 17-34 % at 2048..4096, RTX 5090 2.6-5.7 %) and loses on
  70 and 128 (RTX 5070 Ti 4-20 %, RTX 4090 9-30 %), also where both grids fill their waves alike.
  Whole waves, a wave-cost comparison, L2 per SM and DRAM bytes per SM clock do not order the
  five boards, and neither fixed tile is conservative (always 128 costs the A100 up to a third,
  always 256 at those widths the 4090 up to 30 %);
- choices measured on one board still carry to others by board distance (`core/measured_boards.h`:
  `measured_gate`, the small-T bands, the R8 decode tables, the MoE A8 routes, the MMQ prefill
  and spread gates): the nearest measured board, not a rule over facts;
- `fp32acc_full_rate` still stands in for per-segment instruction rates;
- decode/verify parity is unverified for 18 formats;
- the batch dependence of `fast` bytes at T ≤ 16, which the promise below allows, is not written
  down per route yet (which routes, which batch properties move the bytes);
- under `fast`, a prefill segment of fewer than 17 tokens takes the decode routes (T ≤ 16), whose
  bytes differ from those of the prefill routes: where a prefix-cache frontier, a restore base or
  forced tokens leave fewer than 17 prompt tokens to prefill, those tokens get other bytes than
  the miss that prefilled them in a wider call (`exact` is free of this);
- no CI property checks invariance yet;
- the PCI-ID reference table has no per-SKU instruction rates yet (the L40's half-rate
  FP32-accumulate HMMA is still the generation row's `fp32acc_full_rate`);
- the speed labels and the kernel hash on constants do not exist yet;
- model constants (speculative acceptance) record their keys, corpus hash and tolerance, but no
  decision reads them yet, and the shipped ones name no complete source identity yet;
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
  context-cost presets, speculation profile and drafter registry by content, the artifact and a
  separate drafter by their source identity (their artifact ids are observed), the plan startup resolved without the free memory (KV and state storage,
  speculative windows, route breakpoints, decode steps per round) and the violations of this page
  present (taint). The RAM class is `bit_floor` of the whole GiB the host reports (64 for 64 to
  127 GiB; 0 below 1 GiB or unknown), so the few hundred MiB a kernel keeps do not move it. What
  startup sized from the Host memory (the Host tier), the rung the selection picked before the
  kernel warm-up and the tier it admitted it at are observed and enter only the manifest; the
  plan ladder's rung and its sizes (context, chunk, KV tokens, cached states) are the plan, so two
  starts that serve the same rung have one `nf1`, and one `nm1` too when they also picked it
  before the warm-up at the same tier and sized the same Host tier. An input
  that cannot be read makes the fingerprint `unreadable` without stopping the start. It does not
  yet cover the reference facts no decision reads yet (TDP, VRAM, segment), the constants and the
  kernel hash, nor the
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
   A planner reports the points of its search to a caller that times them, and reads nothing back
   (the stopwatch, `planning_stopwatch.h`, is included only by the ResourceManager).
   The CTest `ninferno_decision_inputs_test` keeps clock and timing headers, clock reads and timers
   out of the text of the decision files it lists; what those files include is not checked (see
   the known gaps and [Runtime observations](#runtime-observations)).
4. **Constants are fitted on at least two lab cards.** A constant about the hardware and its
   kernels (a cost, an efficiency, a rate) is fitted on two cards of its generation and records
   the hash of the kernels it was fitted on. When those kernels change and nobody refits, the
   generation's speed estimates become "no estimate".
   A constant about a model, such as a drafter's acceptance per context class, records instead
   what it depends on:
   - the target and drafter by their source identity (below), never by artifact id;
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

   **The source identity** (`src/artifact/source_identity.h`) is how a constant and the decision
   fingerprint name an artifact. The artifact id is a random UUID drawn at each conversion, so two
   conversions of the same model differ in id and a constant keyed by it applies to one file only.
   The source identity is the SHA-256 of a record of everything that can change the converted
   bytes (schema `ninferno-source/1`, compact JSON with sorted keys):
   - the inputs, which the converter records in the provenance (`"inputs"`): every checkpoint the
     conversion opened, by role (sorted by role), with the weight files it read, each by the
     `sha256` of its content alone (sorted). No path, repository or file name enters the inputs,
     so the same files typed as a relative or an absolute path, in another directory or under
     another name give the same inputs (a GGUF's name still reaches the identity through its
     quant label, below), and no side file enters: what a config, tokenizer or template gives the
     artifact is in its bytes and structure, which the content digests and the structure hash
     cover, and a file beside the weights that the conversion does not read cannot move it. An
     LFS snapshot file takes its blob name, the sha256 the fetch verified at download; any other
     file is hashed when it is converted (a source-consuming conversion keeps the inputs beside
     its journal, written durably before its first punch, so a resumed run never hashes a file it
     punched, and refuses to resume a journal without them);
   - the quant: a GGUF's label, read from its file name (`...-UD-IQ3_XXS.gguf`), so renaming
     the file moves the identity only when the label changes; a name without a label gives an
     empty quant, another identity, the safe side;
   - the converter format version (`"converter"`, `ninferno-v3`) and the recipe (an official
     name, or a custom recipe file by its sha256 and function name);
   - the byte-changing choices the provenance records (drafter quantization, vocabulary, GGUF
     mapper, a proposal head's token ranking by its file's sha256, `--round-tiny-scales`,
     `--maintainer-quantize`, and the Python converter's
     `--override` by its file's sha256 and function name). No path enters;
   - the structure: the SHA-256 of the directory's components, bindings, uses and metadata (but
     the public instance name, `--name`: a label, not bytes);
   - the tensor index: every object's id, format, layout, shape, size and divisor count (every
     resource's encoding and size) and its **content digest**, the SHA-256 of its payload bytes.
     The converter computes the digests once every byte is written, reading the output back, and
     seals them into the directory (provenance `"content_sha256"`) before it publishes the
     artifact; the Python converter does the same. So the identity changes exactly when the bytes
     change: a quantizer or rounding fix that keeps every shape moves it, and a byte-identical
     reconversion keeps it. The converter's build commit does not enter (it would void every
     constant at every commit). Startup reads the stored digests; it never hashes the weights.
     Reading back costs one sequential read of the output, hashed on the conversion's threads
     (the conversion logs it as `content digests: ...`).

   A run stopped after the directory was sealed and before the files were published resumes:
   the partial entry is compared with its digests masked and sealed again. A relayout to a new
   artifact (`ninferno-convert --relayout --out`) seals the digests again; one in place drops
   them (it moves the bytes): that artifact has no identity until it is converted again, nor has
   one converted before the converter recorded its inputs and digests. While an in-place
   relayout's journal is pending the artifact cannot be read at all, so its stale digests are
   never taken. The fingerprint names such an artifact
   `none:<artifact id>`, so two of them still differ, and every constant keyed by an identity
   reads as unknown for it. `ninferno-serve` logs the identity when it prepares a model
   (`source identity <hex>`).
   An embedded component (an MTP head, a DFlash companion) is keyed by the target's identity and
   its name (`component:<name>`); a separate drafter by its own identity. A constant measured
   before its identity could be computed records the parts it can prove and names the missing
   ones (`"target": {"source_identity": null, "proven": {...}, "missing": [...]}`); it is never
   known. The artifact id stays in the fingerprint's observed section, as the name of the file.
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
- Free memory must be the only observation made while running. It selects one rung of a declared
  plan ladder ([Plan levels](#plan-levels)), and the chosen rung goes into the fingerprint.

## Plan levels

The startup memory plan (context, KV pool, cached Device states, prefill chunk) is one rung of a
declared ladder (`src/runtime/engine/plan_ladder.h`). The ladder comes from the artifact (the
model's own context), the options and the planner's bytes for each rung (the model's shapes, the
KV and state codecs, the device's discrete facts). The free Device memory after the weights is the
one observation: it picks the first rung that fits and sizes nothing itself. The same card class,
options and artifact give the same ladder; two starts whose free memory picks the same rung make
the same plan and have the same fingerprint.

### The ladder

N is the context ceiling (the model's own, under an automatic context's cap and an explicit
`--kv-capacity`), C the concurrency, g = 2^(-1/16) and F = min(N, 8192). The **floor quality** is
the poorest plan admitted with the class reserve kept: the context F (or the pinned one), chunk
1024 (the configured chunk when narrower or pinned), and a cached Device state per lane without a
Host tier, none with one. From the top:

1. **top:** N tokens for every lane, a pool of C whole contexts, two cached states per lane, the
   configured chunk;
2. **extra states:** the second cached state of each lane gives way, one at a time, down to one per
   lane, min(C, default) (an agent's turn closure goes before any KV);
3. **pool:** the pool steps down by g from C contexts to one;
4. **states,** with a Host tier only: the cached Device states step down to none (the Host tier
   keeps the StateImages, so a Device state is worth less than a step of context);
5. **context:** the context steps down by g from N to F, the pool one context;
6. **chunk:** a configured chunk wider than 1024 halves down to it, at F;
7. **survival,** below the floor quality and admitted without the class reserve: with a class
   reserve, first the floor quality itself (the same content for less memory); then the chunk
   halves down to 128, the remaining cached states go (down to 1 without a Host tier), and, with an
   automatic context, the context steps on by g below F to min(N, 1024), and, without speculation,
   one last rung holds one prompt token and one decode round. A survival rung keeps the axis it
   stepped in its label (`survival chunk`, `survival states`, `survival context`).

Every step lowers one axis and keeps the others, so the content (context, KV tokens, cached states,
chunk) never grows down the ladder. With the selection's first fit, more free memory never selects
less on any axis. The warm-up's check may then move a rung down by what the warm-up kept for good,
which differs per rung, so the served content is monotone only as far as that is; what holds at
every check is that a free size served is never followed by a larger one refused (the check falls
back to the last rung on its held bytes, as the selection does). That holds while the probe's
lasting graph memory does not grow with the probed rung's context; when it does, a refusal band of
a few MiB can appear at the very bottom of the ladder: a startup refusal, never an out-of-memory. `tests/test_plan_ladder.cpp`
sweeps the free memory in 2 MiB steps over pinned and automatic contexts, with and without a class
reserve and a Host tier, and checks both, also with a warm-up that kept more than its allowance. Nothing
poorer than the floor quality is admitted with the reserve kept, and nothing richer without it.
Master's narrower chunk at a longer context than the floor (the chunk's context grid) is gone: it
made a smaller plan serve more context than a larger one.

Without a Host tier a cached state per lane stays down to the floor quality, as master fitted the
context with one per lane; with one the Device states go first. `ninferno-serve` configures a Host
tier by default on Linux, WSL 2 and native Windows: max(8, 2 x the private conversations)
StateImages (18, 24 and 32 at `--max-concurrency` 1, 4 and 8) and 8 GiB of KV, scaled down to what
the host can pin. It has none on a GPU sharing the host's memory (GB10, Jetson), with
`--no-prefix-reuse` or `--host-state-slots 0`; the one-request CLI has no context cache. The ladder
follows what is configured, not what the host can pin at the start: the fit to the host's free
memory sizes the tier (observed) and never moves a rung, so the fingerprint does not follow host
memory pressure.

With `--max-context` (or an explicit `--kv-capacity`) the context is pinned: the ladder steps the
pool and the states, and its survival range the chunk and the states (Qwen 27B at
`--max-context 32768 --max-concurrency 4` on a 16 GB card with a display and 2.8 GiB free: chunk
1024, a state per lane without a Host tier as master, none with one).

Then every candidate no cheaper than a rung above it is dropped (`prune_plan_ladder`), so the bytes
fall strictly down the ladder and a step that saves nothing is no rung. The survival range is
pruned on its own: its first rung, the floor quality without the reserve, has the bytes of the rung
above and needs less beside them. A step keeps about 95.7 % of the
axis it moves (less the rounding to whole pages): a fixed grid of g is as deterministic as four levels and loses at most
one step to master's continuous fit, where halving lost up to 50 %. The steps are integer
arithmetic (`plan_scaled`), the same on every host.

An explicit option pins its axis in every rung: `--max-context` the context (an explicit
`--kv-capacity` too: it bounds one request), `--kv-capacity` the pool, `--device-state-slots` the
states, `--prefill-chunk` the chunk. With every axis pinned the ladder has one rung. Pinned plans
that fit no rung are a `CapacityError` (`ninferno-serve`'s `--spec auto` fallbacks recover by that
type) that names the last rung's need and the free memory in GiB and what fits instead: the longest
context at this concurrency and at each lower one, and the narrower chunk when `--prefill-chunk` was
the reason.

### The selection

A rung needs its Program reservation (KV, states, workspace and the planner's CUDA-graph
allowance), the Host tier's pinning (declared: 2 MiB per pinned GiB), the kernel warm-up's
allowance and a margin:

| Part | Value | Source |
|---|---|---|
| Weights | not in the need: the free memory is read after the target's and a separate drafter's weights are uploaded, and the plan reports their sum | the family load |
| Base margin | 32 MiB (`--kv-auto-headroom-mib`), 0 with an explicit `--kv-capacity` | option |
| Class reserve | under WDDM (native Windows, WSL 2) an 8th of the VRAM class, at most 4 GiB, on a card driving the display, a 32nd, at most 1 GiB, headless; an 8th, at least 4 GiB, on a GPU sharing the host's memory; else 0. The VRAM class is the total memory rounded down to 256 MiB | driver model, display, integrated: discrete facts |
| Warm-up allowance | 64 MiB (kernel code and the per-thread stack, measured at 6-40 MiB on an RTX 5080), 0 when nothing is automatic | declared |

- The class reserve covers the growth of what Windows already holds: the free memory is measured
  with the desktop's current use taken, so the reserve is not a guess of that use.
- The class reserve is given up in three cases only: a survival rung (below the floor quality), a
  `warm_up` admission (nothing else fits), and the options that keep none (below). Down to the
  floor quality it is kept, so a desktop on a small card gets the floor before the reserve is given
  up (master gave it up for a longer context and let Windows page the pool out). Once it is given
  up the plan does not drop to the bottom: the floor quality comes first (an 8 GB card with a
  display and 1.0-1.3 GiB free: 8,192 tokens at chunk 1024), then the chunk, the states and
  shorter contexts.
- The reserve applies to an automatic pool only. An explicit `--kv-capacity` keeps none, and so
  does an explicit `--max-context` at `--max-concurrency 1`, which makes the pool that one context
  (an explicit `--kv-capacity`). When the rung above the chosen one fits without the reserve, the
  startup warns and names the override: `the 1.00 GiB kept for the Windows desktop limits the
  context to <k>; --max-context <n> (or --kv-capacity <n>) gives it up, and the desktop
  may stall when Windows pages the card out` (at more lanes, `--max-context` with `--kv-capacity`).
- The selection is one pass with one need per rung, so more free memory never picks a lower rung,
  and never selects less.
  When no rung fits with the allowance, the last rung is taken if it fits on the warm-up's
  measurement alone (`reserve warm_up`): an asked plan the graph bound alone would refuse.
- The rungs are not set at a budget's edge: they come from the model's context and the options,
  never from the free memory, so the free memory always has the slack between the rung that fits
  and the one above it.

### The warm-up

The kernel warm-up runs at the chosen rung, measures what it took for good (kernel code, stack,
the probe's graphs, the Host tier's pinning) and checks the rung again on the free memory left, the
allowance taken. The rung stays when it still fits. Otherwise the plan moves to the first lower rung
that fits, recorded as `selected_rung` (the rung before the warm-up) beside `rung` in the plan
summary (`meta.plan`, the server-start record). The re-measurement never moves the plan above the
selection. When no rung from the chosen one down fits on the margins, the last rung is taken on its
held bytes (`warm_up`), as the selection would have taken it with that free memory.

The chosen rung keeps the tier it was admitted at. A `warm_up` rung was admitted on its held bytes
(no graph allowance, no warm-up allowance, no class reserve), so the plan holds the graphs from the
warm-up's first pass, as master did, and every check needs the held bytes beside the base margin:
the prefill probe's 6-40 MiB never refuse it before a graph is measured (Gemma 4 31B IQ3_XXS + MTP
at 4 x 4,096 on an RTX 5080; an explicit `--max-context` at C > 1 under WDDM with a display, where
not even the survival rungs fit on the planner's graph bound). `plan_graphs_held` and `plan_warm_up_check` are
the pure steps the Engine runs; `tests/test_plan_ladder.cpp` replays the selection and the first
check, and the shortfall that moves a rung down.

### What the user sees

- The startup log: `plan | rung k of N: <context> tokens x <pool> contexts (<KV tokens>), <states>
  cached states, chunk <c> for <C> lanes (step <axis>, reserve kept|soft|warm_up) | free <GiB>
  after <GiB> of weights | <GiB> more would serve rung k-1, <GiB> the top | planned in <ms> ms`
  (`rung k-1 fits but for the <GiB> class reserve` when the reserve alone keeps the rung above
  out).
- A warning when the rung is below what the options ask for: a pool below every lane's asked
  `--max-context`, a survival rung, the class reserve not kept or limiting the plan, or a rung the
  warm-up moved down.
  It names the shortfall in GiB and the options that get more (`--max-concurrency`, a shorter
  `--max-context`).
- `/v1/models` `meta.plan_level` and `meta.plan` (the rung, the ladder's length, its sizes, the free
  and weight bytes, the shortfalls and the advice); every request record has `plan_level`, and the
  server-start record the whole plan summary.
- `ninferno-serve --plan` (`--plan --json`) prints the ladder with every rung's bytes and need, the
  rung the free memory picks now and the advice, then exits without serving. The weights are
  uploaded to size the rungs (the planner reads the bound model) and released; no tuner, warm-up
  or Program runs, so it takes the load's time (seconds from the page cache). A real start runs the
  load-time route tuner before it reads the free memory, and the code of the kernels it launched
  stays loaded, so `--plan` can read a little more free memory than the start; the start may then
  land a rung lower, and its warm-up may move it down again, never up.

### The fingerprint

The plan section records `rung`, `rungs` and the rung's `max_context`, `prefill_chunk`, `kv_tokens`
and `cached_device_states`. The rung the selection picked before the warm-up (`rung_selected`) and
the tier it was admitted at (`rung_reserve`) are the warm-up's measurement, recorded in the observed
section: they move `nm1` only, so a warm-up move that ends on the same rung keeps `nf1`. The free
memory itself is not recorded: two starts whose free memory picks the same rung have the same `nf1`.

### Against master's continuous fit

`tests/test_plan_ladder.cpp` replays the plan-levels review's scenarios (cards of 8 to 80 GB,
Linux headless or with a desktop, WSL with a display; Qwen 9B, 27B, 35B-A3B and gpt-oss) and four
RTX 5070 Ti rows measured on 21.10, with the review's byte arithmetic, against an emulation of
master's fit, the later reviews' rows where the reserve is given up and a 35B-A3B on a 32 GB card
with a display. With a Host tier (the review's rows) the delivered context is within 3 % of
master's or longer (no cached Device states give it more), the KV within 6 % at
`--max-concurrency` 4 and 8; the one exception is an 8 GB card under WSL with a display and
1.39 GiB free, where the 1 GiB reserve does not fit beside the floor quality: the ladder serves
8,192 tokens at chunk 1024 without the reserve (a warning), where master served 50,816 with none
left for Windows.

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
  may depend on the batch's composition, and this will be documented per route.
- `fast` prefill: on a given chip type, a token's bytes do not depend on where its prompt was cut
  (the chunk size, a prefix-cache frontier, a restore base) as long as every call that holds it
  has 17 tokens or more, and outside the Gated DeltaNet gap listed above. So a prefix-cache hit
  gives the bytes of a miss only when the segment left to prefill after the cached prefix has 17
  tokens or more; a shorter segment takes the decode routes (the gap listed above)
  (`include/ninferno/ops/prefill_routes.h`). The prefill profile's routes may read the device's
  discrete facts for speed only where that changes no bytes: ggml's matrix kernel
  (`ggml_mmq.tiles`) splits K by the weight's shape alone and picks its slice by the SM count, so
  its prefill bytes are also the same across SM counts of one architecture.
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
| Context-cache planning search | `src/runtime/engine/context_cache/materialization_planner.h`, `src/runtime/engine/context_cache/capture_planner.h` | counted work units, no clock; the ResourceManager times the reported points (a) |
| Concurrent-prefill interleave | `src/runtime/engine/engine_core.h`, `src/runtime/engine/interleave_cost.h` | priced by shape, no clock |
| Decode run-ahead | `src/models/qwen3_5/program/round_pacing.h` | from the CPU-quota fact, no clock |
| Decode wait, spin or block | `src/models/qwen3_5/program/round_pacing.h` | (b), measured host lag |
| Grammar compile timeout | `src/structured/constraint.cpp` | (b), fails a request, no bytes |
| Queue expiry (`--pending-timeout-ms`) | `src/runtime/engine/engine_core.h` (submit, admission head and backfill), `src/models/qwen3_5/frontend/media_cache.cpp` | (b), expires a waiting request, no bytes |
| DFlash draft arrival, 30 s | `src/models/qwen3_5/program/decode.cpp` (`fill_dflash_draft_masks`) | (b), fails a stuck round, no bytes |
| Second transient CUDA error within 5 minutes | `src/runtime/engine/engine_core.h` (`kTransientCudaErrorWindow`) | (b), restarts the server instead of going on |
| Mask fill order, constrained DFlash drafting | `src/structured/request_constraint.cpp` → `src/models/qwen3_5/program/decode.cpp` | (c), open |
| Device mask-release wait bound | `src/ops/kernel/sampling_device.cuh` | (c), open: ends a request |
| Load-time MoE and dense tuners | `src/ops/moe/moe_tune.cpp`, `src/ops/generic/generic_tune.cpp` | (c), opt-in by option (`--dense-tune`, `--moe-tune`), off by default |
| Free memory after warm-up | `src/runtime/engine/model_instance.cpp` | (c), open (gaps above) |
| Acceptance watchdog | `src/spec/watchdog.h` | counts of the request's own drafts, not a timing |
| Planning, transfer, phase and request timings; CUDA event profiling; NVML in the UI | `src/runtime/engine/context_cache/resource_manager.h`, `src/runtime/contract/timing.h`, `src/runtime/engine/engine_core.h`, `src/models/speculation/decoder.cu`, `src/serve/`, `src/ui/gpu_stats.cpp` | (a) |

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
