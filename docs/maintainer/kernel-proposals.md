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

### O. The natural merges' split loop and CTA count at one decode token (#865)
- Regime: the split merge after the KV-grouped decode attention, `causal_natural_merge_kernel`
  (INT8, FP8, NVFP4, K8V4) and `bf16_kv_merge_kernel` (BF16), at W = 1 on the Qwen3.5 geometry
  (16 query heads over 4 KV heads, D 256). The schedule's chunk is 256 columns there, so the grid
  is 16 x 1 x 1: 16 CTAs on every board, one merge per full-attention layer (8 a step on
  Qwen3.5-4B/9B). The #862 bench printed this grid on every card; the census (#853) puts the fixed
  part of an INT8 step at 32.6-46.5 µs on Ada and Blackwell and 66.9-74.2 µs on Ampere.
- Weight: every decode step at every context; the KV term it sits beside is ~115 µs a step at 8K
  on the RTX 5090.
- Law (counted, master 66c53fca): the merge reads splits x 16 heads x 1 KiB of FP32 partials. Its
  time is not bytes:
  - each thread owns one output column and runs the split loop alone. The SASS keeps four 4-byte
    loads a thread in flight and waits for them before the next four. sm_75/86/89 have an x4 body
    and an x1 tail: R(S) = floor(S/4) + (S mod 4) dependent round trips for S splits. sm_120a has
    an x8 body issued as two rounds of four, then x4, x2 and x1 tails: R(S) = 2 floor(S/8) +
    [S mod 8 >= 4] + [S mod 4 >= 2] + [S odd];
  - before the loop come four more dependent round trips: the positions, then the maxima, then
    (after two barriers) the sums, and after the loop the gate;
  - the statistics' barriers and shuffles take ~700 SM clocks;
  - plus one launch, t0: 0.84-0.96 µs a node on the RTX 3090 (`core/pdl.cuh`), 1.5 µs fitted on
    Ada (`core/cost_constants.h`).

  So the old call is t0 + (R(S) + 4) x L + 700 clk. S is the wave partition's 2 x SMs / 4 splits
  (`plan_rules.h`, the same for BF16's targets). L is the round trip to wherever the partials are:
  - **L2-hot** (the bench, and serving): the attention kernel has just written them, so L is the
    ISA table's lone-warp L2 latency (`isa_facts.json` l2_latency_ns_w1: sm_75 110, sm_86 115,
    sm_89 94, sm_120a 123 ns);
  - **DRAM-cold** (ncu): ncu's default `--cache-control all` flushes the caches before the
    profiled kernel. The partials then come from DRAM, and L is the lone-warp DRAM fetch time
    (`sm_rate_facts.json` dram_stream_latency_ns: sm_75 275, sm_86 303, sm_89 253, sm_120a
    374 ns; the RTX 4060 Ti's own probe 254 ns).

  Each of the 16 SMs in use holds 8 warps x 4 loads x 128 B = 4 KiB in flight. That is under what
  one SM can fill (`sm_rate_facts.json`: 31.7-32.0 B/clk on every architecture) times its latency,
  so the loop is latency-bound in both cases.
- Check against the one measurement made before this entry (#865's comment): ncu on the RTX 4060 Ti
  (34 SMs, S = 17, R = 5), INT8 at 21,961 keys, 4.86-5.06 µs a call at 21-22 % of DRAM.
  - The DRAM-cold law gives (5 + 4) x 254 ns + 700 clk = 2.6 µs. The L2-hot law (1.1 µs) does not
    apply to that cell.
  - The bytes match a cold read: 17 x 16 KiB of partials, plus the statistics and the gate, is
    290 KB in 5 µs, 20 % of 288 GB/s.
  - The remaining 2.3-2.5 µs is put down to ncu's own per-kernel floor: the gpu__time_duration of
    an empty 16 x 256 kernel under the same flags. Nobody has measured it yet.
  - It is a named residual, not a fitted constant. The ncu A/B measures it (below), and if it
    reads under 1.5 µs the cell is refuted.
- Variant (#865's PR):
  1. Every thread of the CTA loads the partials as 16-byte vectors, in stages of 16 KiB. Two
     stages are in flight in registers while the column owners sum the current one out of shared
     memory (`common/causal_merge_columns.cuh`). The first two stages go out before the
     statistics. One stage in flight (16 KiB) is more than 32 B/clk x the L2 latency (6-10 KiB)
     on every card, so the loop runs at the SM's fill rate: S x DChunk x 4 B / 32 B/clk.
  2. The maxima, the sums and the gate are loaded beside the positions: one round trip, not four.
  3. A 256-column schedule launches 64 columns a CTA only within one wave, and only where that
     lowers the bytes on the most loaded SM (`common/causal_merge_shape.h`):
     - n = query heads x columns is the CTA count at 256 columns. At 64 columns the most loaded
       SM streams ceil(4 n / SMs) x 64 columns a split, against ceil(n / SMs) x 256;
     - the rule narrows only while 4 n <= 3 SMs, so the narrow grid puts at most three CTAs on
       an SM and fits in one wave;
     - it is restricted to one wave on purpose. Past a wave the byte inequality still holds in
       places: at n = SMs + 1 it is ceil(4 + 4 / SMs) x 64 = 320 against 2 x 256 = 512. But each
       narrow CTA repeats the head's statistics (the m and l loads, the reductions and their
       barriers), and the registers cap the merge at 4 CTAs a SM (Cost, below). So past
       4 x SMs narrow CTAs a second wave runs that whole chain again. The bytes saved then no
       longer pay for a serial statistics phase that this law does not count, and the schedule
       keeps 256 columns there;
     - at W = 1 the merge runs 64 CTAs on every board from 22 SMs up. The most loaded SM streams
       ceil(64 / SMs) / 4 of the old bytes: a quarter from 64 SMs up (RTX 5090, 4090, 3090,
       5070 Ti, 2080 Ti), a half from 32 to 63 SMs (4070 Ti's 60 and 4060 Ti's 34, two CTAs an
       SM), and 3/4 from 22 to 31 (3060's 28, three CTAs an SM). The table's 64-column columns
       price these shares.

     The NVFP4 and K8V4 merges store whole rows through the inverse rotation, so they keep 256
     columns: 16 CTAs, each at the fill-rate bound.

  New call: t0 + L + max(700 clk, L + bytes on the most loaded SM / 32 B/clk) + the last stage's
  FFMA chain (5 clk a split) + 40 clk of barriers a stage. DRAM-cold, the call is also bounded
  below by L + all its bytes / the board's DRAM bandwidth (`gpu_facts.json`).
- Numerics: no change.
  - A column's sum is the same FFMA chain in split order, with zero-weight splits skipped, as
    before. The statistics are the same code.
  - The chunk only changes which CTA owns a column.
  - The output is bit-identical at every codec, width, batch and partition, including the
    invariant route profile's aligned splits. `test_causal_merge_order` pins this against
    master's kernels at split counts 1-256.
- Prediction (frozen 13.11, before any run). It was revised the same day after review, before
  any result came back, for three things: the per-arch R(S), the DRAM-cold case and the narrowing
  bound. The table gives the merge's call time without t0, at the boards' boost clocks
  (`gpu_facts.json`), at both bench depths (7843 and 21961 keys run the same S):

  | Board | S | R(S) | L2-hot old µs | new, 64 cols | new, 256 cols | bench intercept drop a step (8 layers), BF16 / FP8 / INT8 µs | NVFP4 / K8V4 µs | DRAM-cold (ncu) old µs | new, 64 cols | new, 256 cols |
  |---|---|---|---|---|---|---|---|---|---|---|
  | RTX 3090 | 41 | 11 | 2.1 | 0.7 | 1.1 | 12 (9-18) | 8 (6-12) | 5.0 | 1.0 | 1.5 |
  | RTX 3060 | 14 | 5 | 1.4 | 0.6 | 0.6 | 7 (5-10) | 7 (5-10) | 3.1 | 1.0 | 1.0 |
  | RTX 2080 Ti | 34 | 10 | 2.0 | 0.7 | 1.0 | 10 (8-16) | 8 (6-12) | 4.3 | 1.2 | 1.3 |
  | RTX 4070 Ti | 30 | 9 | 1.5 | 0.4 | 0.6 | 8 (6-13) | 7 (5-11) | 3.6 | 1.3 | 1.3 |
  | RTX 4090 | 64 | 16 | 2.2 | 0.5 | 1.1 | 13 (10-20) | 9 (6-13) | 5.3 | 1.3 | 1.4 |
  | RTX 5070 Ti | 35 | 10 | 2.0 | 0.5 | 0.8 | 12 (9-18) | 10 (8-15) | 5.5 | 1.0 | 1.3 |
  | RTX 5090 | 85 | 22 | 3.5 | 0.6 | 1.5 | 23 (17-35) | 16 (12-24) | 10.0 | 1.2 | 2.0 |
  | RTX 4060 Ti | 17 | 5 | 1.1 | 0.4 | 0.4 | 6 (4-8) | 6 (4-8) | 2.6 | 1.3 | 1.3 |

  - The bench drop's range is 0.75-1.5x the point. The L2 latency under load lies between the
    lone-warp figure and 1.5x that, and it lengthens the old loop's 4 + R(S) round trips more
    than the new one's two.
  - Under ncu, every cell adds the same unmeasured per-kernel floor to old and new. So the ncu
    prediction is the drop in duration a call: the DRAM-cold old column minus the new one.
  - ncu's default `--clock-control base` also locks the base clock, while the cold columns use
    boost clocks. Only the clock-counted terms (700 clk of statistics, the fill rate, the FFMA
    chain) stretch, by about boost / base: about +0.05 µs a call on the RTX 5090. That is inside
    the ncu range, so no column is restated.
  - The read rate (the slope) does not change.
  - With `--activations exact` the aligned partition runs ceil(keys / 512) splits (16 at 7843
    keys, 43 at 21961). There the old loop's cost grows with depth and part of the gain moves
    into the slope; that cell is not predicted.
- Other widths: the same law holds at every W.
  - The staged loop runs at every width, and the bits are unchanged.
  - The 64-column chunk applies while 64 x W x B <= 3 x SMs: RTX 5090 W <= 7, 4090 <= 6; 3090,
    5070 Ti and 2080 Ti <= 3; 4070 Ti <= 2; 3060 W = 1 only. Wider grids keep their 256-column
    launches.
  - The geometries whose schedule is already 64 columns (KV heads other than 4) keep their grid
    and get the staged loop.
  - No W > 1 cell is predicted here.
- Prototype:
  - `ninferno_kv_decode_attention_bench --codec <c> --context 7843,21961 --layers 8` on master
    and the branch. Read "fixed part (intercept) ... us a step", and the merge's grid in the
    launch list.
  - ncu: the merge's gpu__time_duration through the same bench with default flags, plus an empty
    16 x 256 kernel under the same flags (the floor).
- Cost: one more merge instance per (batch, mask, aligned) form of the 256-column schedules, and
  some residency:
  - the merge now uses 56-79 registers a thread, against master's 32-42 (42 on sm_120a);
  - so registers cap it at 4 CTAs a SM on sm_86/89/120a, where master ran 6 (thread-limited; 5 on
    sm_120a), and at 3 on sm_75, where master ran 4 (thread-limited);
  - the 17-18 KiB of static shared memory would allow 5 (3 on sm_75).

  It is not a new route: the summation, the selector goldens and the build manifest are unchanged.
- Acceptance / refutation (frozen, two-sided):
  - kept where the bench intercept a step drops by at least half the lower edge on at least five
    of the seven stand boards (RTX 3090, 3060, 2080 Ti, 4070 Ti, 4090, 5070 Ti, 5090), and the
    slope stays within ±2 %. The RTX 4060 Ti row is an ncu-only cell: it is there to check the
    law against the one measurement made before this entry, and it is not counted in the bench
    criterion. If the 4060 Ti bench runs, its cell is a finding under the next item like any
    other, but it counts toward neither the five nor the seven;
  - a board whose drop is under half the lower edge, or over 1.5x the upper edge, is a finding.
    ncu of the merge there names the term that was wrong (the per-SM fill rate for L2 hits, the
    latency of the statistics, or the round-trip count), and the law is fixed;
  - ncu cells: the duration drop a call must fall within 0.5-1.5x the DRAM-cold old minus new.
    Outside that range it is a finding, handled the same way. An empty-kernel floor under 1.5 µs
    on the RTX 4060 Ti refutes the DRAM-cold law's account of the 4.86-5.06 µs cell;
  - the byte rule: `exact`, bit-identical (`test_causal_merge_order`, and the softmax attention
    tests unchanged).
- A/B: not run.
- Decision: open.
