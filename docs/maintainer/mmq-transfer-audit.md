# MMQ transfer audit between same-generation cards (#373)

`tools/bench/mmq_transfer_audit.py` (test: `tests/test_mmq_transfer_audit.py`, no GPU) asks, for each
pair of cards of one generation (GA10x, Ada, GB20x), whether the MMQ choice recorded on one card
carries to the other, split by workload regime. It selects nothing and fits nothing. Run
`python tools/bench/mmq_transfer_audit.py` for the full tables (`--json` for the data).

## Regimes (architecture facts only)

- decode, T <= 4: the vec / R8 row range;
- verify, T 5..16: below the 17 tokens the MMQ gate is defined for (`mmq_prefill_wins`);
- prefill, T >= 17, split by the card's ridge T* = int8 ops/s x bytes per value / (2 x DRAM bytes/s),
  with the generation's int8 rate (`core/generation.h` s8), the SM count and the reference boost
  clock (`core/data/gpu_facts.json`), for Q4_K (4.5 bits). A pair is memory-bound on both below the
  smaller T*, straddles between the two, compute-bound on both above. Ridges: GA10x 80-86 (3060,
  3090), Ada 172-227 (4060 Ti 172, 4070 Ti 179, 4090 184, L4 227), GB20x 110-132 (5070 Ti 110, 5060 Ti
  119, 5090 and 5080 132). Same-generation ridges are close, so the straddle band is narrow (524 of
  26 000 points).

## Layer 1: route tables, agreement (no regret possible)

A table keeps only the winning route per band, so the time of the route the other card chose is not
in the repository. The audit counts, over the shared problems (format, n, k, epilogue; 10-186 per
pair) and a grid of widths, how often "ggml_mmq or not" disagrees.

| regime | points | MMQ-or-not disagreements | points on MMQ (A+B) |
|---|---|---|---|
| decode T <= 4 | 3814 | 0 | 0 |
| verify T 5..16 | 11733 | 802 (6.8 %) | 1766 |
| prefill, memory-bound on both | 6948 | 254 (3.7 %) | 666 |
| prefill, straddling | 524 | 0 | 0 |
| prefill, compute-bound on both | 7287 | 0 | 0 |

- No table picks MMQ at T <= 4 or at T above the ridge: all of MMQ's table use is T 5 .. ~127, so
  every disagreement sits in the verify and the memory-bound-prefill bands.
- There the pair's disagreement rate follows the board distance, not the generation: Ada 4.6-10.6 %
  (distance 0.8-4.3), GA10x 3060/3090 10.0 % (3.64), GB20x 5090/5070 Ti 1.6 % (2.78), 5080/5070 Ti
  0 % (0.57), 5090/5080 0 % (2.21), but 5090/5060 Ti 20.5 % and 5070 Ti/5060 Ti 23.9 %: the 5060 Ti
  table never picks MMQ in the pairs' shared problems (MMQ points 0 on that side), the 5090 and the
  70-SM card do. The same split the gate has (36 SMs false, 170 true).
- The exact route disagrees far more (about 50-70 % of points), which is expected between
  different SM counts and is not an MMQ question.

## Layer 2: gate label regret on paired timings

The gate label (`kMmqPrefillPays`, nearest measured board) applied from card A to card B, scored on
B's paired prefill timings (`tools/bench/data/mmq_paired_prefill.json`: the mmq and w4a8 arms of
issue #373's two paired sets, 9K and 24K prompts, Q4_K_M / Q6_K / IQ4_XS, copied by hand). Regret is
time: 1 / (mmq/w4a8) - 1 where the label picks MMQ and loses, (mmq/w4a8) - 1 where it picks w4a8 and
loses. All prompts are far above every ridge, i.e. compute-bound prefill.

| generation | label of -> applied on | label | wrong / rows | max | mean |
|---|---|---|---|---|---|
| Ada (4060 Ti, 4070 Ti, 4090) | any pair, both directions | mmq | 0 / 4 each | 0 % | 0 % |
| GB20x | 5090 -> 5070 Ti | mmq | 3 / 4 | 7.0 % | 3.6 % |
| GB20x | 5070 Ti -> 5090 | w4a8 | 4 / 4 | 9.0 % | 5.1 % |

Ada transfers at the gate level in the compute-bound regime on every measured pair (distance up to
2.4 in the tables). GB20x does not: the label cannot be carried between 70 and 170 SMs in either
direction, up to 9 % of prefill time. This is the information the production rule already uses (the
5070 Ti takes the 5060 Ti's `false`, the 4070 Ti the L4's `true` at 0.797); the audit adds the
size of the loss when the donor is the wrong side.

## Missing data (plainly)

- Timings of the route the other card chose, for the small-T bands: not in the repository, so
  layer 1 has no regret. The dense-tune cache (`routes/tuner/dense_tune.json`) keeps winners only;
  `evidence_us` holds default against chosen time of one card. Needs `regret_collect.py` rows
  (`ninferno-selection-regret/1`) from both cards of a pair, same build and protocol.
- Paired mmq/w4a8 timings are all prefill at 9K-24K tokens: nothing for the memory-bound prefill
  window (T 17 .. ~170) or for verify widths, where layer 1 shows the disagreements are.
- GA10x has one timed card (3060): the 3060/3090 pair has no regret at all (no 3090 timings; the
  3090 is not a gate row). Ada has no L4 or 4080 timings; GB20x has no card between 70 and 170 SMs
  (the 5080 table has no MMQ entries in common with a measured prefill) and no 5060 Ti timings.
- The ridge uses one representative format (Q4_K) and the reference boost clock, not a measured
  clock; per-format ridges move by the bytes per value (IQ4_XS, Q6_K).
- Layer 2 is one prompt length per set and Q4_K family-heavy: the IQ4_XS miss on the 5070 Ti
  (1.3 %) is a format effect inside one card, not a transfer effect.
