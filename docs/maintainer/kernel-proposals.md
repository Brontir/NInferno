# Kernel proposals

Every new kernel, or new variant of a kernel, starts here: one entry per proposal, written before
the code. This is the register of #619 (decision K4): a kernel is built only where a counted law
explains the current kernel's time, and its gain is predicted from facts before any micro-bench.
The [gap map](gap-map.md) names the regimes; this page holds what is proposed for them and what
the measurements then said.

## The rule

1. **Regime.** Name it the way the gap map does: the op, the format, the T band, the tile, the
   boards. Cite the gap-map rows or committed measurements that show the gap.
2. **Law.** The counted law that explains the current kernel's time in that regime: which term
   binds (DRAM bytes, tensor-pipe cycles, issue, latency, launch), counted from the kernel-facts
   snapshots (`src/ops/generic/data/kernel_facts/`) and the chip facts
   (`src/core/data/gpu_facts.json`, `src/core/data/isa_facts.json`). A gap no counted law explains
   yet is no proposal yet: the entry records the law that is missing, and the variant waits for it.
3. **Prediction, frozen.** The variant's time per regime cell, from the same facts. Write it as a
   range with its sources, before the micro-bench or A/B, and do not change it after a run.
4. **Acceptance and refutation, frozen.** These are fixed before the A/B:
   - what counts as the predicted gain (for example, at least half of the lower edge);
   - what refutes it;
   - the no-regression bound outside the regime;
   - the byte rule (`exact` bit-identical, or the KLD bound).
5. **A/B.** The A/B runs ABBA on the boards named, with its baseline, workload, GPU and method
   stated. Report predicted against measured, cell by cell. A cell under half its
   predicted gain, or over the upper edge, is a finding: it names the term that was wrong, from ncu
   on that cell, and the law is fixed. A constant is never fitted to close the gap.
6. **Decision.** Record it on the entry: kept (with its price in the cost model), narrowed to the
   cells where it held, or removed. The entry stays, with its numbers, either way.

## Entry template

```
### <id>. <name> (#issue, #PR)
- Regime: <op, format, T band, tile, boards; gap-map rows>
- Weight: <its share of the benchmark mix, where the mix is known for it>
- Law: <the binding term of today's kernel and its count, with the facts it reads; its residual>
- Variant: <what changes in the kernel>
- Prediction (frozen <date>): <time change per cell and arch, with sources>
- Prototype: <the cheap first measurement, as a command>
- Cost: <what this variant costs beyond the cost every new route has (below)>
- Acceptance / refutation (frozen): <bounds>
- A/B: <boards, protocol, predicted vs measured, or "not run">
- Decision: <kept / narrowed / removed / open>
```

Every new route, whatever its family, costs (#619 section 4):
- a conformance point in `test_generic_linear` or the family's `test_*_a16`;
- a `test_route_classes` summation class, and `same_summation` / `route_is_token_invariant`
  entries where the arithmetic differs by tile;
- the selector goldens, regenerated on three stands per arch set;
- `--build-manifest` and `test_prepared_rules`;
- `decision_inputs` where it reads a device fact.

A new family also costs a `Family` enum entry, which every route table parses, and a
`route_resources` branch.

## Register

The entries below come from the candidates table of #619 (section 4). Only A has a frozen
prediction so far.

### A. Pre-summed activation groups for the min-term formats (#619 steps 5-6, #709)
- Regime: the register MMA on the min-term formats, T >= 17; Gemma heads and SwiGLU. The formats
  are those codec_traits marks `min` (Q4_K, Q5_K, Q4_1, Q5_1) or `half_min` (Q2_K).
- Law: the ones-MMA doubles an m1 tile's tensor work: 16.7 against 8.4 cycles per counted MMA on
  the RTX 5090 (#501).
- Variant: a plane of T x K/G FP32 sums (G the format's group: 32, and 16 for Q2_K), written once
  per call by an activation pass (as the W4A8 quantizer is). Every tile reads the plane in place of the ones-MMA, which keeps one summation
  order at every T and tile.
- Prediction (frozen in #709's body before any run):
  - prefill, narrow tiles at T 24-191: −5..−30 % by board;
  - heads and SwiGLU tiles at T 32-256: −10..−50 %;
  - the pre-pass costs +1.7-2.5 µs per call;
  - the speed profile and `exact` are unchanged.

  A different variant, all-ALU sums, measured −5..−12 % (heads) and −8..−18 % (SwiGLU) at
  T 17-256 in an earlier ABBA, and lost at T <= 16 (+4..23 %).
- Prototype: `ninferno_route_tune` ABBA on the 5090 and 3060, Gemma head and SwiGLU at T 17-64.
- Acceptance / refutation (frozen): #709's body, items 1-5.
- A/B (12.11, #709): ABBA of master against the branch, `ninferno_route_tune --prefill-only 1
  --rotate 1 --routes-out` at chunks 16, 34 and 48 (T 8-256), three passes (6 + 6 runs a cell),
  medians; the prefill profile's admitted cells only (RTX 5090: Q2_K store and add; RTX 3060:
  Q4_K add), shapes 4096x4096, 4096x12288 and 3840x15360. The timer ticks 1.024 µs, so the
  RTX 5090's cells were also timed per kernel with nsys and read with ncu. Against the frozen
  protocol: no Ada board ran, only the prefill profile was run (the speed profile reads no plane),
  and 3840x15360 (Gemma 4 12B's down projection) is not in the frozen shape list.
  - RTX 3060, Q4_K add: the fastest route −7..−20 % at T 17-256 against the predicted −10..−30 %;
    item 1 holds at T >= 17 in every cell. Its frozen "T = 8 within ±1 %" clause fails: T = 8
    moved −10..−15 %, because the prefill profile reads the plane below 17 tokens too. Item 2
    fails: the m2.n2 plane instance runs +3..+10 % over its twin in 11 of 24 cells.
  - RTX 5090, Q2_K store and add: the fastest route 0..−16 % at T 8-17 and 128-256, +8..−12 %
    at T 24-32 (mostly under half of the predicted −8 %), +4..+14 % at T 48 (over the 1.03 bound),
    0..+3 % at T 64. Items 1 and 2 fail: m1.n1 runs +3..+18 % over its twin at T 48-256 (+8 % at
    T 24 on the store), m1.n8 and m2.n1 +3..+9 %.
  - Finding, the term that was wrong: the law took the ones-MMA's tensor work as binding. On the
    RTX 5090 (ncu, Q2_K store 4096x4096, T 24-128) tensor-active is 14-41 % on master; the plane
    halves it, but long-scoreboard stalls per issue rise (m1.n1 1.0-1.7 to 2.3-2.4, m1.n8 2.1-2.4
    to 3.5-5.9): the plane's group loads lengthen the K loop's load chain, and the narrow tiles'
    kernel gets slower (nsys, Q2_K add 4096x12288: m1.n1 +6..+20 % at T 48-128). The pre-pass
    costs 0.8 µs at T 8 to 2.9 µs at T 128 (K 12288): it grows with T·K/G, below the predicted
    1.7-2.5 µs at small T and above it from T 96.
  - Items 4-5 (e2e TTFT, KLD) were not run: items 1-2 already refute the prediction.
- Decision (12.11): off by default, the code kept. `rmma_group_sums` answers false unless
  `set_group_sums_plane_enabled` turns it on: the plane's tests, and `ninferno_route_tune
  --group-sums-plane 1` for its next A/B (no environment variable decides it, principle 3). The
  prefill profile needs one answer per format, epilogue and device, so the plane cannot be
  narrowed by width; the RTX 3060's Q4_K add cell held per cell but failed item 2, and no fact
  separates it from the RTX 5090's refuted cells. The plane comes back with a law for the narrow
  tiles' load chain (step 6 prices it), and a new A/B.

### B. Blackwell MoE expert kernel at 4-8 rows (#619 step 11)
- Regime: the routed experts on GB20x at 4 and 8 rows. They run at 0.74 and 0.63 of the DRAM-bound
  speed, against Ada's 0.91 and 0.81 (DECISIONS 65). The 5070 Ti IQ3_XXS estimate is 7-13 % slow
  at 2-6 rows.
- Law: none yet. The gap map lists MoE under "No law yet". The loss may come from the expert grid
  or from the GB20x tensor rate rather than from the kernel.
- Prototype: `moe_bench` (`r8vs.sh`) and nsys on the 5090 and 5070 Ti at C = 4 and 8.
- Decision: law first, then decide.

### C. Turing int8 tile: dense `w4a8_kernel` and MoE `moe_w4a8` (#619 steps 7-8)
- Regime: int8 prefill on sm_75: 30-34 SM clocks per K, against 12 on Ada. MMQ Q4_K reaches 22 % of
  peak there, against 37 % on Ada.
- Law (partial): Turing's int8 MMA is m8n8k16 only, half the shape of sm_80's m16n8k32, so the facts
  predict about 2x the MMA issues. The measured 2.5-2.8x leaves 30-40 % unexplained.
- Prototype: ncu of `w4a8_kernel` at K 4096 on the 2080 Ti (stalls, issue, ldmatrix count).
- Decision: open. The study (step 7) writes a counted-instruction prediction before the kernel
  (step 8).

### D. Gemma 4 int8 KV plane for sm_75/80/86 (#619 steps 9-10)
- Regime: Gemma 4 attention below sm_89 (`kGemma4Fp8Hardware = arch >= 89`). The KV stays BF16
  there, which means twice the bytes and half the context.
- Law: long-context decode is bytes-bound on Ampere (1.0-1.16 of DRAM time). On Turing it is
  SM-bound, so the gain there is capacity, not speed.
- Variant: an int8 plane through the FP8 plane's "codes widened in registers" mechanism
  (`fp8_row.cuh`).
- Prototype: `gemma4_attention_bench` on the 3060 at 24K, then `mdecl.sh`.
- Decision: open. It is a B1 byte class, so the choice must be a fact of the arch.

### E. Narrow-T (17-64) register-MMA load chain: a deeper ring or split loads (#619 step 12)
- Regime: the register MMA at T 17-64. On four cards the winner is latency-bound at 2-4 warps per
  SMSP, with DRAM at 28-50 % and tensor at 25-63 %.
- Law: the bytes one SM keeps in flight against its DRAM latency (Little's law, #690). The
  per-architecture rate and bytes-in-flight facts that `tools/bench/warp_inflight` measures count
  it. #735 (open, draft) is to wire them into the price.
- Risk: the winning schedules sit at 245-255 registers, so a deeper ring spills.
- Prototype: `rmma_kernel` m2.n4.w4x1 with ring depth 3, `route_tune --rotate` on the 4090 and
  3060 at T 24 and 32.
- Decision: prototype after A; adopt only with the law.

### F. Gated gate/up twin at T = 3-4 (vec kernel)
- Regime: gated launches at C = 4. The 3060 still loses 3-4 % there; the 5060 Ti is at parity.
- Law: the vec kernel's bounded twin excludes gated launches (`kVecHasBoundedTwin`): they spill
  under 72 registers. A twin needs the gate and up rows on separate warps. The gain is small, on one
  class (SMs x clock / bandwidth ~0.17).
- Prototype: `tools/bench/run_521_gateup.sh` (#606, not run).
- Decision: run the kit first. Build only if the loss is above 3 % and #606's rule holds on the
  3090.

### M. MMQ column widths per arch and format (#619 step 13, DECISIONS 74)
- Regime: MMQ on short prompts. Ada and Turing are done (#597, #717). Q2_K and the IQ formats still
  keep the fewest tiles.
- Prototype: the forced-J grid on the 4090 and 2080 Ti.
- Decision: open. These are data rows.

### N. The natural merges' split loop and CTA count at one decode token (#865)
- Regime: the split merge after the KV-grouped decode attention, `causal_natural_merge_kernel`
  (INT8, FP8, NVFP4, K8V4) and `bf16_kv_merge_kernel` (BF16), at W = 1 on the Qwen3.5 geometry
  (16 query heads over 4 KV heads, D 256). The schedule's chunk is 256 columns there, so the grid
  is 16 x 1 x 1: 16 CTAs on every board, one merge per full-attention layer (8 a step on
  Qwen3.5-4B/9B). The #862 bench printed this grid on every card; the census (#853) puts the fixed
  part of an INT8 step at 32.6-46.5 µs on Ada and Blackwell and 66.9-74.2 µs on Ampere.
- Weight: every decode step at every context; the KV term it sits beside is ~115 µs a step at 8K
  on the RTX 5090.
- Law (counted, master 66c53fca): the merge reads splits x 16 heads x 1 KiB of FP32 partials,
  just written by the attention kernel and so in L2. Its time is not bytes:
  - each thread owns one output column and runs the split loop alone. The sm_120a SASS of the
    loop is an x8 body that issues its loads in two rounds of four (the second round waits for
    the first FFMA), then x4, x2 and x1 tails: R(S) = 2 floor(S/8) + [S mod 8 >= 4] +
    [S mod 4 >= 2] + [S odd] dependent L2 round trips for S splits;
  - before the loop, four more dependent round trips: the positions, then the maxima, then (after
    two barriers) the sums, and after the loop the gate;
  - the statistics' barriers and shuffles, ~700 SM clocks;
  - plus one launch, t0: 0.84-0.96 µs a node on the RTX 3090 (`core/pdl.cuh`), 1.5 µs fitted on
    Ada (`core/cost_constants.h`).

  So the old call is t0 + (R(S) + 4) x L2 + 700 clk, with S = the wave partition's 2 x SMs / 4
  splits (`plan_rules.h`, the same for BF16's targets), and L2 = the ISA table's lone-warp L2
  latency (`isa_facts.json` l2_latency_ns_w1: sm_75 110, sm_86 115, sm_89 94, sm_120a 123 ns).
  The 16 SMs in use each hold 8 warps x 4 loads x 128 B = 4 KiB in flight, under the 32 B/clk
  one SM can fill (`sm_rate_facts.json`, 31.7-32.0 B/clk on every architecture) times its L2
  latency: the loop is latency-bound.
- Variant (#865's PR):
  1. Every thread of the CTA loads the partials, 16-byte vectors, in stages of 16 KiB, two
     stages in flight in registers while the column owners sum the current one out of shared
     memory (`common/causal_merge_columns.cuh`). The first two stages go out before the
     statistics. A stage in flight (16 KiB) exceeds 32 B/clk x the L2 latency (6-10 KiB) on
     every card, so the loop runs at the SM's fill rate: S x DChunk x 4 B / 32 B/clk.
  2. The maxima, the sums and the gate are loaded beside the positions: one round trip, not four.
  3. A 256-column schedule whose grid leaves SMs idle (query heads x columns < SMs) launches 64
     columns a CTA (`common/causal_merge_shape.h`): 64 CTAs, a quarter of the bytes on the most
     loaded SM (on the RTX 3060's 28 SMs, three CTAs share an SM: 3/4). The NVFP4 and K8V4
     merges store whole rows through the inverse rotation and keep 256 columns: 16 CTAs, each
     at the fill-rate bound.

  New call: t0 + L2 + max(700 clk, L2 + bytes on the most loaded SM / 32 B/clk) + the last
  stage's FFMA chain (5 clk a split) + 40 clk of barriers a stage.
- Numerics: no change. A column's sum is the same FFMA chain in split order, zero-weight splits
  skipped, as before; the statistics are the same code. The chunk only changes which CTA owns a
  column. The output is bit-identical at every codec, width, batch and partition, including the
  invariant route profile's aligned splits. `test_causal_merge_order` pins this against master's
  kernels, at split counts 1-256.
- Prediction (frozen 13.11, before any run). Merge call time without t0, at the boards' boost
  clocks (`gpu_facts.json`), at both bench depths (7843 and 21961 keys run the same S):

  | Board | SMs | S | R(S) | old µs | new, 64 cols µs | new, 256 cols µs | intercept drop a step (8 layers), BF16 / FP8 / INT8 µs | NVFP4 / K8V4 µs |
  |---|---|---|---|---|---|---|---|---|
  | RTX 3090 | 82 | 41 | 11 | 2.1 | 0.7 | 1.1 | 12 (9-18) | 8 (6-12) |
  | RTX 3060 | 28 | 14 | 4 | 1.3 | 0.6 | 0.6 | 6 (4-9) | 6 (4-9) |
  | RTX 2080 Ti | 68 | 34 | 9 | 1.9 | 0.7 | 1.0 | 9 (7-14) | 7 (5-10) |
  | RTX 4070 Ti | 60 | 30 | 8 | 1.4 | 0.4 | 0.6 | 8 (6-11) | 6 (5-9) |
  | RTX 4090 | 128 | 64 | 16 | 2.2 | 0.5 | 1.1 | 13 (10-20) | 9 (6-13) |
  | RTX 5070 Ti | 70 | 35 | 10 | 2.0 | 0.5 | 0.8 | 12 (9-18) | 10 (8-15) |
  | RTX 5090 | 170 | 85 | 22 | 3.5 | 0.6 | 1.5 | 23 (17-35) | 16 (12-24) |

  The range is 0.75-1.5x the point: the L2 latency under load lies between the lone-warp figure
  and 1.5x it, and lengthens the old loop's 4 + R(S) round trips more than the new one's two. The
  read rate (the slope) does not change. With `--activations exact` the aligned partition runs
  ceil(keys / 512) splits (16 at 7843 keys, 43 at 21961), so there the old loop's cost grows with
  depth and part of the gain moves into the slope; that cell is not predicted here.
- Other widths: the same law at every W. The staged loop runs at every width (bits unchanged).
  The 64-column chunk applies while 16 x W x B < SMs (RTX 5090 W <= 10, 4090 <= 7, 3090 <= 5,
  5070 Ti and 2080 Ti <= 4, 4070 Ti <= 3, 3060 W = 1); wider grids keep their 256-column
  launches. The geometries whose schedule is already 64 columns (KV heads other than 4) keep
  their grid and get the staged loop. No W > 1 cell is predicted here.
- Prototype: `ninferno_kv_decode_attention_bench --codec <c> --context 7843,21961 --layers 8`
  on master and the branch; read "fixed part (intercept) ... us a step", and the merge's grid in
  the launch list.
- Cost: one more merge instance per (batch, mask, aligned) form of the 256-column schedules. It
  is no new route: the summation, the selector goldens and the build manifest are unchanged.
- Acceptance / refutation (frozen): kept where the intercept a step drops by at least half the
  lower edge on at least five of the seven boards, and the slope stays within ±2 %. Refuted on a
  board where it drops by less than half the lower edge: ncu of the merge there names the term
  (the per-SM fill rate for L2 hits, or the latency of the statistics), and the law is fixed. The
  byte rule: `exact`, bit-identical (`test_causal_merge_order`, and the softmax attention tests
  unchanged).
- A/B: not run.
- Decision: open.
