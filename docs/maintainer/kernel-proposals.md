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

Entries A-F and M come from the candidates table of #619 (section 4), G from #850's plan. A and G
have frozen predictions so far.

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

### G. R8 decode schedule of the wide-K Q6_K down projection (#850 candidate (a))
- Regime: the R8 decode kernel (`r8_kernel`, A8) at T = 1, store and add epilogues: Qwen3.5-9B's
  down projection, Q6_K 4096x12288. On the boards with native BF16 MMA (sm_80 and up) `rules_r8`'s
  K rule (K >= 8192) picks `w8r2`; on the RTX 2080 Ti its Turing Q6_K rule picks `w4r2`. No R8
  decode table has this problem. The #850 census (RTX 4090 ncu, master 3f20b638c): 48.7 us for
  41.3 MB (849 GB/s, 84 % of DRAM), 256 CTAs of 256 threads at 209 registers, 16 % occupancy; RTX
  5090 nsys: 29.3 us. Both sit 7-8 % above the line the Q4_K calls of the same step define.
- Weight: one call a layer, 32 a 9B step; 1-2.5 % of the C = 1 step (#850 plan, section 4).
- Law (#850 plan, sections 1.2-1.5): per call, t = t_node + t_ramp + L_first + B / BW_eff +
  min(R - 1, 3) b_round + t_partial + t_tail + t_imb. On the RTX 4090 it holds to 0.6 us on six
  of the step's eight calls with no fitted constant. At `w8r2` the registers (209 x 256 threads)
  leave one CTA a SM, so the 256 CTAs are two a SM on the 4090 (no dynamic balancing), 1.5 on the
  5090 (a second round on half the SMs) and 3.1 on the 3090 (a last round of 10 CTAs). The residual
  of this call, 3.6 us on the 4090 and 2.5 on the 5090 (8-9 % of the stream on both), is named
  t_imb: a hypothesis, not yet a fact. ncu `sm__cycles_active.max` against `.avg` on this call
  decides it (#850 PR-1). Bytes in flight do not bind (x8-50 margin, x2.3 on the 5090).
- Variant: no kernel. The compiled schedule `w8r1` on sm_80 and up: 120 registers, two CTAs a SM,
  512 CTAs of 80 KB. On Turing `w4r1`: 126 registers, four CTAs a SM, 1024 CTAs of 40 KB. Both keep
  the bytes in flight a SM, halve (quarter) the work quantum an SM can be left with, and put 16
  warps on a SM. Every R8 schedule sums a row in one order (lane l takes block groups l, l + 32,
  ...; generic_r8.cuh), so the change is bit-identical: `check_r8_schedules_agree` holds it at
  K = 12288 in `test_generic_linear`, and the sweep's `same=1` holds it on every row.
- Prediction (frozen 13.11, before any sweep; change per call, us, candidate minus the rules'
  schedule). The bands are the plan's, re-derived below; they are the acceptance basis:

  | card | today | candidate | band | this entry's re-derivation |
  |---|---|---|---|---|
  | RTX 4090 | 48.7 measured (`w8r2`) | `w8r1` | -2.0 .. -4.0 | -1.7 .. -3.5 |
  | RTX 5090 | 29.3 measured (`w8r2`) | `w8r1` | -0.8 .. -2.2 | -1.0 .. -2.3 |
  | RTX 3090 | 56.4 law (`w8r2`) | `w8r1` | -3 .. -7 | -3.0 .. -4.9 |
  | RTX 3060 | 128.7 law (`w8r2`) | `w8r1` | -1 .. -4 | -2.3 .. -3.5 |
  | RTX 2080 Ti | 78.2 law (`w4r2`) | `w4r1` | -1 .. -4 | -1.4 .. -2.8 |
  | RTX 4070 Ti | not priced by the plan (`w8r2`) | `w8r1` | -0.5 .. -2.0 | (this entry's) |
  | RTX 5070 Ti | not priced by the plan (`w8r2`) | `w8r1` | -0.5 .. -2.0 | (this entry's) |

  Where this entry differs from the plan:
  - A `w8r1` CTA has half a `w8r2` CTA's bytes at half the SM's share, so it lasts as long. What
    halves is the work an SM can be left with at the end, so t_imb halves; it need not vanish. The
    central estimate is therefore the band's lower edge: 4090 -1.7 (half of 3.6, less 0.08 us of
    ramp for the 512 CTAs), 5090 -1.0 (half of 2.5, less 0.2 us of ramp).
  - The plan's 3090 and 3060 laws count the last partial round in full, although its bytes are
    already in B / BW_eff. Only the round's time beyond its bytes' share is extra. 3090: 10 CTAs of
    161 KB at 10 x 54 GB/s take 3.0 us, of which 1.8 are in the stream, so +1.2. 3060: 4 CTAs, 3.0
    against 1.9, so +1.1. At `w8r1` the last round's CTAs pull more than the DRAM gives (3090: 20 x
    54 GB/s = 1.08 TB/s > 896 GB/s; 3060: 8 x 54 > 344), so its extra is 0, not the plan's 1.5
    and 1.4. With t_imb half to all: 3090 -3.0 .. -4.9, 3060 -2.3 .. -3.5. A 3090 beyond -5 would
    say t_imb is larger than the 4090's share carried over.
  - RTX 2080 Ti: the last round of `w4r2` (104 CTAs on 136 slots) already pulls more than the DRAM,
    so there is no partial term. Only t_imb moves: 2.8 to 1.4, so -1.4 .. -2.8.
  - RTX 4070 Ti (60 SMs, 504 GB/s, 482 effective: a 85.7 us stream) and RTX 5070 Ti (70 SMs,
    896 GB/s, 840 effective at the 5090's stream share: 49.2 us): `w8r2` gives 4.3 and 3.7 CTAs a
    SM. Their last rounds (16 and 46 CTAs) pull more than the DRAM, so there is no partial term.
    t_imb of a multi-round grid: between the 3060's 2 % of the stream (nine rounds) and the 4090's
    8 % (two), halved by `w8r1`. Hence -0.5 .. -2.0 on both. These two cards check the rule
    below more than they test a gain.
  - `w4r1` on sm_80 and up: a quarter of the grain, but twice `w8r1`'s activation prologue (3 blocks
    a thread against 1.5) and four times its activation reads from L2 (25 MB a call on the 4090).
    It is predicted within 1 us of `w8r1`; the summary prints it, and it is not the candidate.
  - Q4_K and Q5_K 4096x12288 (the same call in other mixes): the same direction where `w8r2` holds
    one CTA a SM. No snapshot has their `w8r2` registers (kernel_facts has no r8 rows yet), so no
    magnitude is frozen. An entry there lands on its sweep's margin and is reported as outside this
    prediction.
  - The rest of the step (the context sweep) is predicted unchanged: the rules' `w4r1` calls sit on
    the Q4_K line (plan section 0). A schedule 0.5 % faster there is a finding for another entry, not
    merged here.
  - Outside, noted: at K >= 8192 the K rule also gives SwiGLU calls `w8r2`, whose x,2 SwiGLU
    kernels spill (255 registers + 296-344 B; the plan's "never chosen for SwiGLU" holds for the
    9B, whose SwiGLU K is 4096, not for the goldens' 2560x17408 and 14336x8192 SwiGLU rows).
- Prototype: `tools/bench/run_850_r8_sweep.sh <build>/bench/ninferno_r8_schedule_bench` on each idle
  card: three sweeps (`--samples 15 --launches 60`) of the down projections and of the rest of the
  step. `tools/bench/r8_sweep_summary.py` prints every cell against this entry.
- Cost: no new route or summation class, and nothing beyond the table. The schedules exist and are
  in `test_generic_linear`'s every-schedule checks.
- Acceptance / refutation (frozen 13.11):
  1. Gain: on each card, the candidate's median minus the rules' on Q6_K 4096x12288 store is at or
     below half the band's lower edge. The edges in us: 4090 -1.0, 5090 -0.4, 3090 -1.5,
     3060 -0.5, 2080 Ti -0.5, 4070 Ti -0.25, 5070 Ti -0.25.
  2. Refutation: a card above its edge refutes the prediction there. A card beyond its band's upper
     edge is a finding. Either names the term from that card's ncu of the call (no t_imb: the law
     loses the term).
  3. Bytes: `exact`, `same=1` on every R8 row of every sweep.
  4. No regression outside: the merge writes an entry only where its schedule is faster in every
     sweep, so no merged problem gets slower. The 9B step does not get slower on any card: plain
     decode, C = 1, 22K, `--spec none`, six repeats, ABBA against master (#850 PR-2).
- Selection (decided before the sweeps): the change lands as R8 decode table entries, through
  `tools/bench/r8_decode_table.py --merge` of the down sweeps. They are per board, measured and
  bit-identical. The K rule itself moves to `w8r1` for Q4_K, Q5_K and Q6_K on boards with
  `mma_bf16_native` only if the sweeps show, on at least four sm_80+ boards and on none against,
  `w8r1` never slower than `w8r2` on any K >= 8192 K-quant row that is not a SwiGLU call (the
  4096x12288, 2560x9216, 5120x17408 and 28672x8192 rows of both sweeps), with no row slower by more
  than 0.5 % on any board. Why not the rule now: the committed tables show `w8r1` over `w8r2` at
  K >= 8192 on six boards, but those gains do not follow the grain law. The Ampere boards gain the
  most (Q6_K 5120x17408: -8 % RTX 3090, -10 % A10, -6 % A100). The coarsest grid, the RTX
  5090's 1.9 CTAs a SM, gains 2.8 %. And the 5090 table has no entry for Q4_K 5120x17408, so
  `w8r1` was not 0.5 % faster there. A residual the law does not explain is no rule.
- A/B: not run. TODO(#850 (a)): the sweeps on the RTX 3090, 3060, 2080 Ti, 4070 Ti, 4090, 5070 Ti
  and 5090; the merge of their down files; the 9B step ABBA; then this field and the decision.
- Decision: open.

### M. MMQ column widths per arch and format (#619 step 13, DECISIONS 74)
- Regime: MMQ on short prompts. Ada and Turing are done (#597, #717). Q2_K and the IQ formats still
  keep the fewest tiles.
- Prototype: the forced-J grid on the 4090 and 2080 Ti.
- Decision: open. These are data rows.
