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
   snapshots (`src/ops/generic/data/kernel_facts/`) and the chip facts (`core/data/gpu_facts.json`,
   `core/data/isa_facts.json`). A gap no law explains is no proposal yet. It goes under "No law yet"
   in the gap map.
3. **Prediction, frozen.** The variant's time per regime cell, from the same facts. Write it as a
   range with its sources, before the variant is built, and do not change it after a run.
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
- Law: <the binding term and its count, with the facts it reads>
- Prediction (frozen <date>): <time change per cell, with sources>
- Acceptance / refutation (frozen): <bounds>
- A/B: <boards, protocol, predicted vs measured, or "not run">
- Decision: <kept / narrowed / removed / open>
```

## Register

### P1. Pre-summed activation group plane for the min-term formats (#619 steps 5-6, #709)
- Regime: the register MMA on min-term formats (Q4_K, Q5_K, Q4_1 and Q5_1 rows), prefill, T 17-256.
  Today each tile adds the activation group sums with the MMA with ones.
- Law: the ones-MMA costs as many tensor-pipe cycles as the tile's own MMAs at MT = 1 (kernel-facts
  `mma` / `mma_ones`). Where the tensor pipe binds, the plane removes that term. A pre-pass per call
  writes the sums.
- Prediction (frozen in #709's body, before any run):
  - prefill, narrow tiles at T 24-191: −5..−30 % by board;
  - heads and SwiGLU tiles at T 32-256: −10..−50 %;
  - the pre-pass costs +1.7-2.5 µs per call;
  - the speed profile and `exact` are unchanged.
- Acceptance / refutation (frozen): #709's body, items 1-5.
- A/B: not run; it belongs to step 6 with the plane's price. The −5..−18 % in #619 is the earlier
  micro-bench of the kernel.
- Decision: open. 5a (#709) takes the plane only in the prefill profile's admitted cells.

### P2. Turing int8 tile (#619 steps 7-8)
- Regime: int8 prefill tiles on sm_75, at about a third of the int8 peak (#619).
- Law, prediction: not written yet. The study (step 7) writes them before the kernel (step 8).
- Decision: open.

### P3. Gemma 4 int8 KV plane for sm_75/80/86 (#619 steps 9-10)
- Regime: Gemma 4 attention below sm_89, where the KV stays BF16.
- Law, prediction: not written yet.
- Decision: open.

### P4. MoE expert-call law, plus R8 against the register MMA per format (#619 step 11)
- Regime: Blackwell MoE at 4-8 rows (0.74 and 0.63 of DRAM, #619), on the 5070 Ti, 3060 and 2080 Ti.
- Law: not counted yet. The gap map lists MoE under "No law yet": it needs the gating, the
  distinct experts a batch touches and the down format.
- Decision: open.

### P5. Ring-depth prototype for the narrow-T register MMA (#619 step 12)
- Regime: the register MMA at narrow T, on the 4090 and 3060.
- Law: the bytes one SM keeps in flight against its DRAM latency (Little's law, #690, #735). The
  per-architecture rate and bytes-in-flight facts measured by `tools/bench/warp_inflight` count it.
- Prediction: not written yet. It waits for #735's facts.
- Decision: open.
