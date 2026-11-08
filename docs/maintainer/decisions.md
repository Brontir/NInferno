# Decision log

The configuration choice (`src/product/choice`, `src/spec`, the route selector in `src/ops/generic`)
follows rules derived from chip facts; measurements check those rules rather than set them. Where a rule
needed a judgement call, or a measurement disagreed with the model, the reasoning is recorded here under
a stable number. Source comments cite these entries as `DECISIONS <n>`.

Entries 29-53 are summaries; 54 onwards are kept as written at the time, including the open questions.
Dates are 2026. Card names refer to the lab's own GPUs (see [docs/performance](../performance/README.md)).
The fit scripts these entries mention live in [`tools/fits`](../../tools/fits/). Measurement scripts named
here without a path (`mpref.sh`, `mbatch.sh`, `census.sh`, `sm_bench.py` and similar) are the lab's own
stand scripts and are not shipped; the rows they produced are in `tests/fixtures/choice/` and `tools/fits/data/`.

## 29. A moved branch whose original commit still exists: the same rule as a deleted one (25.10)

When the Hugging Face branch a model was converted from moves to a new commit, the server keeps serving
the converted weights and says so: a console notice, `--refetch` downloads the new commit,
`--accept-converted` remembers the choice. An unreachable hub is a warning, not an error, so the weights
served never depend on the network. The GUI never starts a download by itself: it marks the model as
updated and offers the download. A new commit is downloaded and converted in the background while the old
build keeps serving; the new one is picked up at the next start (a running model restarts once, after its
current work). The old build is deleted only after the new one has started. At most one automatic restart
per commit change: if the new build fails to start, the server returns to the old one and logs it.

## 45. GGUF models in the GUI pick a separate drafter like any other axis (25.10-26.10)

Without `--with-drafters` the GUI used to start a GGUF model with no separate drafter, so Gemma 4 GGUF ran
without its assistant (x1.96-2.44 decode in the drafter campaign) and Qwen GGUF without built-in MTP layers
ran without DFlash (x2.2-2.6, x3.7 with reasoning). Decision: a separate drafter is an ordinary axis of the
choice. It is attached, and downloaded, when the choice finds it worth it at the requested context, stream
count and memory; `--with-drafters` / `--no-drafters` pin the axis. Measured on Qwen3.5-9B MTP GGUF
(RTX 4070 Ti and 4090): DFlash is 24-34 % faster than the built-in MTP on the task mix and ~45 % faster
with reasoning, at the cost of a 1.28 GiB drafter; on 12 GB at 64K context the choice keeps MTP, at 32K it
takes DFlash.

## 48. An unmeasured quant of measured weights takes its sibling's profile (26.10)

A quant of a repository whose other quants have measured drafter pairs used to get the conservative group
profile, which often came out as plain decode. Measured on the RTX 4070 Ti (Qwen3.5-9B MTP GGUF): the
nearest sibling by bits per weight predicts a new quant within 0-4 % (Q4_K_S, Q5_K_M, Q6_K); across three
families the error is -7..+4 %, mostly on the low side. Rule: between "own pair" and "group" sits the sibling
pair: same repository and commit, same bit class (< 4 / >= 4 bits), nearest bits per weight, measured with
the same drafter version. A sibling counts only from the same commit, since another commit may hold other
weights.

## 49. Unmeasured is not plain: the group's gain at a fine-tune's acceptance (26.10)

A target with neither its own pair nor a sibling is priced by its group (same engine, drafter kind and bit
class, at least two cards). The old group rule vetoed speculation whenever the weakest pair did not pay off,
which left 102 of 165 leave-one-out cases on plain decode. New rule: every depth measured on at least two
cards is scored by the group's mean gain at a fine-tune's acceptance (0.535 of the measured); the best is
taken, and plain only if no depth reaches x1.05. Leave-one-out over the shipped registry (51 pairs): plain
cases 102 -> 0, mean gain x1.63 -> x2.19, 98 % of each pair's own best.

## 50. Speculation gains are weighted by our benchmark task mix (26.10-27.10)

The Qwen3.8-27B + DFlash2 pair on the RTX 4090 measured x3.30 without reasoning, but an eight-task mix gave
x2.25: prose gains far less than code and math (x1.86 against x4.6-5.5). Decision: each section of a
speculation measurement weighs by its share in the project's benchmark prompts (`examples/cli/messages`):
code 7, math 4, prose 8, reasoning 1 (`spec::section_weight`). The weights apply to the default choice, its
speed estimate and the guard threshold; whether a drafter is accepted at all still weighs every section
equally, since a drafter that only loses on prose is still worth it for reasoning.

## 53. The round cost of speculation from facts; only acceptance is measured (27.10)

Configuration speed = decode x E(k) / R(k). E(k), the tokens per round, is a property of the pair
(registry, `spec-tune`). R(k), the round's cost in plain steps, comes from facts (`src/spec/round_cost.h`):

    R(k) = [C_t(k+1) + n * C_d] / C_t(1),   C(r) = B(r) + kappa * F(r),   kappa = BW / (eta * int8 peak)

B is the bytes of weights the pass reads (an MoE reads experts per row), F = 2 x rows x active parameters,
n = k for MTP and 1 for DFlash (one block). Only eta, the share of the int8 peak small-T kernels reach, is
fitted (`tools/fits/round_cost/fit_eta.py`), and a generation's constant needs two cards of that
generation. Against measurement (R measured / R facts, k <= 7): Qwen dense within -5..+1 % on six cards,
Qwen MoE and Muse within 1 %, gpt-oss +12.5 %. It does not hold for Gemma 4 (+10..23 %), Nemotron MTP
(+18 %), Q8_0 (+23..25 %) and k > 7 (up to +60 %): there a lab measurement more than 5 % dearer than the
facts raises the estimate (only upward). Held-out checks: the RTX 4070 Ti with eta from the 4090 -3..+4 %,
the 4090 from the 4070 Ti +1..+9 %, the 5090 (outside the registry) from the 5070 Ti -8..+3 %.

## 54. Small-T eta (kSmallTEta 0.148): no fact derivation found yet, the value stays (28.10, revision)

The maintainer (28.10): rules come from architecture facts, and measurements only check them.
`kSmallTEta` (`src/spec/round_cost.h`) is the one fitted constant in the round cost: the share of
the int8 peak that small-T kernels reach. Two fact hypotheses were checked (`tools/fits/round_cost/eta_facts.py`,
the master 9c2d8c91 registry, Qwen rounds at k 4..7, each card's own eta):

| card | rounds | eta | 4 x eta = share of the dp4a peak (512 ops a SM-clock) | excess over the bytes, ms a round (9B DFlash, k 5 -> 7) |
|---|---|---|---|---|
| RTX 2080 Ti | 2 | 0.099 | 0.40 | 8.5 -> 7.7 |
| RTX 3060 | 6 | 0.197 | 0.79 | 7.4 -> 8.0 |
| RTX 4070 Ti | 48 | 0.149 | 0.60 | 3.9 -> 4.0 |
| RTX 4090 | 12 | 0.144 | 0.57 | 1.7 -> 1.8 |
| RTX 5070 Ti | 18 | 0.130 | 0.52 | (27B MTP 4.9 -> 4.9) |
| Quadro GV100 | 6 | 0.651 (Volta's peak is dp4a's) | 0.65 | 6.4 -> 6.8 |

1. **The small-T compute runs at the dp4a peak, at one share on every generation.** No: 0.40 to 0.79,
   and the order follows neither the peak nor the DRAM bytes per SM-clock. Caveats: the 2080 Ti has
   two rounds; Blackwell's dp4a rate (unified INT32/FP32 lanes) is not checked against the
   Programming Guide's table (at 1024 the 5070 Ti's share is 0.26). The two Ada cards agree (0.57
   and 0.60), which is why the Ada row stands.
2. **The excess over the bytes is a fixed cost a round** (excess = (R measured - R bytes) x the
   measured plain step). Partly. Of the 46 series (one pair and mode each), one is below its bytes (4070 Ti 4B AWQ MTP,
   -0.7 .. -1.0 ms: no relative growth). In 33 of the other 45 it moves by 10 % or
   less from the smallest k to the largest (median +2 %), it grows with the model (4B about 1.3 ms,
   9B about 4 ms on the 4070 Ti) and falls with the card (9B DFlash: 4070 Ti 3.9 ms, 4090 1.8 ms):
   not launch time (that would not follow the card), but a pass that k does not change, the DFlash
   drafter's block (16 rows at every k, q8 since #156). The other 12 series do depend on k
   (4070 Ti 4B AWQ DFlash +40 %, 9B NVFP4 DFlash +25 %, several 5070 Ti series +19..+30 %; the
   small MTP residuals swing widely in relative terms): a row-dependent part remains.
   kappa F today spreads both over the verify and drafter rows at one eta.

The missing facts: the drafter pass's cost (a 16-row block on its q8 MMA route) from the
generation's tensor rate and the route's tiles, apart from the verify, and the verify's own
per-row compute where it is not bound by bytes (NVFP4, AWQ). That is a study of the kernels, not
cleanup. Until then eta stays the Ada row fitted on two cards (principle 4); the value and the
behaviour do not change. Other generations keep no estimate (as in 53).

## 55. Tuners only in maintainer tools; the shipped dense-tuner measurements stay as data, with a plan to remove (28.10, cleanup)

**Tuners.** `--dense-tune`, `--moe-tune` and `--tune-cache` time kernels on the machine and keep
their choices per machine. Under principle 3 (no timing at startup) they belong to the lab. Until
now `ninferno` and `ninferno-serve` accepted them, and both tuners ran if a user passed `on`. Now
both refuse them (`product::is_tuner_flag`, `TunerFlags::Refused`). The tuner hosts are:
- `ninferno-perplexity`: both tuners. It is an Engine, so the MoE tuner runs at its model load and
  the dense one at the warm-up and the scored prefills.
- `ninferno-dense` and `ninferno-gemma4`: the dense tuner.

Nothing changes for a user. The defaults were already `off`, and the GUI never passed the flags.

**`routes/tuner/dense_tune.json`, the shipped tuner measurements.** It is read at every load, even
with `--dense-tune off` (`model_instance.cpp` `enable_dense_tune(true)`). Each entry is keyed by the
board's exact class id: 356 problems, and in 347 of them the tuned route differs from what the
selector picked when it was timed (`prefill_base`):

| class | problems | with an override | override points (T bands) |
|---|---|---|---|
| ad102:128sm (RTX 4090) | 10 | 8 | 40 |
| ad104:58sm | 45 | 45 | 462 |
| ad106:34sm (RTX 4060 Ti) | 102 | 99 | 1006 |
| ga100:108sm (A100) | 14 | 14 | 192 |
| gb206:36sm (RTX 5060 Ti) | 102 | 101 | 1191 |
| gv100:80sm (Quadro GV100) | 12 | 9 | 88 |
| tu102:68sm (RTX 2080 Ti) | 71 | 71 | 823 |

These are per-board decisions, which principle 1 ("copy facts, never decisions") rules out. They
also carry nothing over: a board of another SM count (the RTX 4070 Ti is ad104 with 60 SMs) gets
none of them.

No rule over facts that reproduces them exists yet. Dropping the file would change routes on these
seven classes, so this PR keeps it and documents it as measured data.

Plan to remove it:
1. For each class, compare its overrides with the selector's rules (T-band edges against SMs,
   DRAM bytes per SM-clock, tile waves). This is work for the selector maintainers in
   `src/ops/generic`.
2. Move every override that a rule explains into the selector, and regenerate the selector golden.
3. Drop those entries.
4. A/B-measure what is left on its card. Turn it into a rule or drop it.
5. Once the file is empty, remove the read path (`generic_tune.cpp` `load_cache_locked` under Off).

## 56. Blackwell's eta from two lab cards: gb20x 0.165 (28.10)

53 left Blackwell without a small-T eta: one lab card (RTX 5070 Ti) had registry pairs. The RTX 5090
(gb202, 170 SMs, 1792 GB/s) now has pairs of the same targets as the 5070 Ti (gb203, 70 SMs, 896
GB/s): Qwen3.5-9B GGUF and MTP GGUF Q4_K_M, Gemma 4 12B Q4_K_M, gpt-oss-20b MXFP4 (spec-tune at
325f95ff, the built-in corpus, the pairs' usual settings). The 5070 Ti got the 9B and Gemma 4 12B
Q4_K_M pairs the same way. Registry 173 -> 180.

**Hold-out first (principle: measurements verify).** `fit_eta.py` item 3 fits each card's eta on
the generation's other card only and predicts the held-out card's rounds (Qwen, k <= 7):

| held-out card | eta of the other card | R measured / R facts | rms |
|---|---|---|---|
| RTX 5070 Ti | 0.184 (5090) | +4.3 % | 8.4 % |
| RTX 5090 | 0.162 (5070 Ti) | +1.4 % | 4.4 % |
| (Ada, for scale: 4070 Ti / 4090) | 0.144 / 0.149 | +0.1 % / +0.2 % | 3.7 % / 2.0 % |

The plan's pick for each target the other card measured (its tokens a round, this card's R from
facts, the mix weights, the conservative raise) against this card's measured best (k <= 7):
9B and 9B MTP dflash:7 = dflash:7 on both cards; Gemma 4 12B mtp:5 = mtp:5 on the 5090, mtp:4
against mtp:5 on the 5070 Ti (-1.5 %); gpt-oss dflash:5 against dflash:7 on the 5090 (equal,
x1.22), dflash:3 against dflash:5 on the 5070 Ti (-4 %, its MoE rounds are dearer than their
facts on both cards and are raised from the other card's measurement).

**Decision.** The row `{"gb20x", 0.165, 2}` (both cards' rounds fitted together). No fact formula
for eta exists yet (54); the two cards' own fits differ (0.162 / 0.184, Ada's 0.144 / 0.149), so
the row is a two-card constant under AGENTS.md, not a derived one. What it buys: a Blackwell card
or target without its own pair is priced by the facts instead of taking the 5070 Ti's raw
measurement.

**hard50 (no flags, the branch against master; explicit depths for the measured best).** Served =
measured best everywhere, to within 1 %:

| card | model | served on this branch | tok/s | measured best | tok/s | predicted x / measured x of the served |
|---|---|---|---|---|---|---|
| RTX 5090 | 9B Q4_K_M | dflash:7 | 714 | dflash:7 | 713 | 2.65 / 2.71 |
| RTX 5090 | 9B MTP Q4_K_M | dflash:7 | 738 | dflash:7 | 738 | 2.66 / 2.76 |
| RTX 5090 | Gemma 4 12B Q4_K_M | mtp:5 | 564 | mtp:5 | 564 | 3.02 / 3.04 |
| RTX 5090 | gpt-oss-20b | dflash:5 | 754 | dflash:7 | 761 | 1.77 / 1.64 |
| RTX 5090 | Qwen3.8-27B UD-Q4_K_S (no 5090 pair: facts) | mtp:3 (non-thinking; mtp:5 thinking) | 226 | mtp:5 | 236 | 2.39 / 2.41 |
| RTX 5070 Ti | 9B Q4_K_M | dflash:7 | 431 | dflash:7 | 431 | 2.61 / 2.91 |
| RTX 5070 Ti | 9B MTP Q4_K_M | dflash:7 | 441 | dflash:7 | 441 | 2.61 / 2.94 |
| RTX 5070 Ti | Gemma 4 12B Q4_K_M | mtp:5 | 325 | mtp:5 | 325 | 2.96 / 3.13 |
| RTX 5070 Ti | gpt-oss-20b | dflash:7 / 5 by thinking | 429 | dflash:7 | 434 | 1.75 / 1.59 |

Predicted = the run's own tokens a round / R(facts, eta 0.165). The 27B on the 5090 is the real
hold-out (only the 5070 Ti measured it): R from facts is within -1..+2 % at every depth (mtp:2..5
x2.14/2.39/2.49/2.51 predicted, x2.10/2.41/2.45/2.52 measured). Its non-thinking pick (mtp:3,
-4 % against mtp:5 on hard50) comes from the corpus' acceptance, not R: hard50's math rounds
accept more than the corpus' prose (50). Master served the same (the 5070 Ti's profile).

**Mismatches (the missing facts, open).**
1. The 5070 Ti's 9B DFlash rounds of 6 and 8 verify rows cost 7-12 % less than the facts (its 9B
   alone fits eta 0.29); the 5090's agree (-4..+6 %). On both cards dflash:5 and dflash:7 rounds
   cost no more than dflash:3 (5070 Ti 1.67-1.73, 5090 1.70-1.76 plain steps): the verify of 4
   rows is dearer per row than of 6-8, a route fact (the T=4 verify kernel), not a byte fact. The
   plan's speedup on the 5070 Ti is predicted low (x1.89 against x2.16 for the 9B MTP pair held
   out; the test bounds it at 15 %), its pick is right.
2. gpt-oss rounds are 6-10 % dearer than their facts on both cards (MoE routing: 53's +12.5 %).
3. MTP at depth 3 is 7-9 % dearer than its facts on both cards (the 9B, Gemma 4 12B).

## 57. Turing's MTP rounds 12 % dearer than the R(k) the plan transfers to it (29.10, open, measured)

The user kit's first run on a card without its own pair of the target (#210; RTX 2080 Ti Rev. A,
tu102, build 5ab1c936; Qwen3.5-9B MTP GGUF Q4_K_M, spec-tune on the lab pairs' menu,
414 s, plain decode 92.8-93.0 tok/s across the sections). Acceptance agrees with the six lab
pairs at every configuration (within 0.7 %). The round cost does not, at shallow MTP depths:

| configuration | R measured here | R the plan uses | measured / plan |
|---|---|---|---|
| mtp:2 | 1.626 | 1.449 | x1.122 |
| mtp:3 | 1.867 | 1.661 | x1.124 |
| mtp:4 | 1.987 | 1.884 | x1.055 |
| mtp:5 | 2.201 | 2.077 | x1.060 |
| dflash:3 | 1.726 | 1.676 | x1.030 |
| dflash:5 | 1.743 | 1.718 | x1.015 |
| dflash:7 | 1.846 | 1.735 | x1.063 |
| dflash:11 | 2.687 | (not priced: deeper than the portable depth) | |

R in plain decode steps a round, `spec::profile_round_costs` of the tune and of the plan's resolved
profile (`plan.speculation.round_cost`).

**What the plan's R is.** Not the facts: since 53 / #161 the facts price only Ada (two lab cards
give its eta); Turing has one lab card and no eta, so `round_cost_estimate` is unknown there. With
no Turing pair of this target and no estimate, `profile_for_target` takes a lab pair's measurement
raw: the plan's R equals the RTX 4070 Ti's measured R to the third digit (mtp:2 1.449, mtp:3
1.661, mtp:4 1.884, dflash:7 1.735). The source says "registry pair ... (6 lab card(s))", nothing
of a transfer.

**What it shows.** An Ada card's rounds are the cheapest of the lab's (the 4090: mtp:2 1.450,
mtp:3 1.654); Ampere, Volta and Blackwell measure 1.53-1.59 and 1.80-1.86 at mtp:2/3, Turing
1.63 / 1.87. Off Ada the shallow MTP rounds cost relatively more; why is not measured yet (the
MTP head's launches and the T=3-4 verify route are candidates, compare 56's mismatches 1 and 3). The
plan's speedup on Turing is predicted high by up to 12 % at mtp:2/3; its pick (dflash:7) is
unaffected here, dflash rounds agree within 6.4 %. Most of the 12 % is the source being an Ada
card: against the other non-Ada lab cards (mtp:2 1.53-1.59) Turing is about 5 % dearer.

**For comparison, the facts path (Ada).** The same kit run on the RTX 4060 Ti 16GB (ad106,
build 11388484; no lab pair of ad106, so the plan prices the rounds from the facts, ada, eta 0.148;
591 s). Acceptance confirms the six lab pairs at all 8 configurations. R measured / R facts: mtp:2
x0.998, mtp:3 x0.995, mtp:4 x0.972, mtp:5 x0.961, dflash:3 x0.949, dflash:5 x0.911, dflash:7 x0.884
(1.559 against 1.764: contradicts at ±10 %); dflash:11 not priced. The MTP rounds hold the
facts; the DFlash rounds of 6 and 8 verify rows are 9-12 % cheaper than their facts, the same sign
and size as the 5070 Ti's in 56 (mismatch 1). Open with it: the facts overprice the T=6-8 verify
on cards with few SMs (34 here, 70 there), a route fact not yet in the bytes model.

**Open.** Nothing changes from one card: a Turing eta needs a second Turing card in the registry
(53). Until then two things are wrong in the transfer, not in Turing: the source pair is the first
current pair in the registry's order (`profile_for_target`'s fallback, `current.front()`), against
the rule that the registry's order never decides; and the source says "registry pair ... (6 lab
card(s))" where it should say another generation's raw measurement and name the card. The gap
stays recorded here. Rerun: `tools/kit/measure.py --spec-tune 900` on the RTX 2080 Ti (result
`ninferno-user-run/1.1`, validate.py verdicts S5/S6).

**Resolved in the transfer (29.10).** `profile_for_target` no longer lets the registry's order pick
the card: the records are ordered by GPU class id; two or more lab cards of the card's generation
give their median R per configuration; with none and no facts, the median R of every lab card that
measured the target, said in the source as "transferred: median of N lab cards of other
generations (no facts for turing)". The generation ratios (`generation_round_costs`, off in the
shipped registry since #161 (entry 53): they needed two cards of each generation) are gone. The same RTX 2080 Ti tune
against the new prediction (the six lab cards' median): mtp:2 x1.062, mtp:3 x1.021, mtp:4 x1.019,
mtp:5 x1.023, dflash:3 x1.011, dflash:5 x1.020, dflash:7 x1.061: every configuration within the
kit's ±10 %. Still open: Turing's own eta (a second Turing card), and the T=6-8 DFlash verify the
facts overprice on Ada (above).

## 58. Verify widths stay on the band rule: the register MMA priced as its best one-tile schedule is wrong for the schedule that runs (29.10, #209)

**Decision.** The selector keeps the band rule at verify widths. On the native-MMA boards measured
here (Ampere, Ada, Blackwell) the K-quant and legacy formats take the vector kernel at T <= 4 and
the register MMA from 5. This is the fast-activation route with no saved dense route: under
`--activations exact` `select_route` returns the register-MMA invariant route at every width before
the band is consulted, and in fast mode a saved `tuned_dense_route` replaces the vector kernel at
T = 2..4 where one exists (`generic_gemm.cpp`, `select_route`). Other boards keep their own rules: Volta's vector route extends past T = 4
for formats `volta_hmma_route` does not take (`vec_rule_max_tokens`), and Turing sends codebook
formats at T = 5..8 to the MMQ kernel (`mmq_verify_rule`). `verify_route_seconds` (bytes at 0.93 of DRAM and counted warp instructions at
0.6 of the issue rate, combined as sqrt(tb^2 + ti^2), plus 1.5 us a launch) was never used and is
removed; it and its instruction counts are in the history before that removal.

- Variants: pick the cheaper of the vector kernel and the register MMA by that price / the band rule.
- The price of the vector kernel holds (RTX 4070 Ti, Q6_K 3840x15360, T = 5: 117 us predicted,
  119 measured). The register MMA's price is its best one-tile schedule (w1x4 / w1x8 / w1x16), but
  heuristic_rmma's schedule is what runs: scored against it (T = 2..4, 889 calls where master takes
  the vector kernel, six boards) the priced pick moved 556 calls for a geomean time over master's of
  1.0001, with regressions to 1.55x the time (4070 Ti IQ1_M 4096x4096 T = 3: 13.0 -> 20.1 us on w1x8). The hard50 gains
  on the 5070 Ti / 5090 / 3060 (+5..9 % dflash:3 and mtp:3, Qwen3.5-9B) did not carry to other shapes.
- The missing fact: the running schedule's DRAM efficiency. Q6_K 3840x15360 T = 5 on the 4070 Ti
  streams 57 / 66 / 77 % of DRAM on w1x4 / w1x8 / w1x16 at 16 warps per SM each, 4.2 KB of loads in
  flight per warp in every schedule (13x the bytes in flight that
  DRAM bandwidth x latency / SMs asks): not latency. Study open: DRAM row locality
  (bytes per row activation from each schedule's K interleave) and wave quantization.
- The Gemma 4 12B MTP round's excess on the 4070 Ti (+1.0..1.3 ms of a 21.07 ms round) is this: its Q6_K down
  projection on w1x8 (149 us, 96 us of bytes; the vector kernel 119, w1x16 121). It waits for that fact
  (no dense-tuner stopgap: 55).

## 59. Vector kernel: loads in flight per warp is the fact, source-level hints do not hold it (29.10)

**Decision.** No change to the vector kernel now: neither hints nor a source-level load() /
convert() split hold loads in flight (kern/vec-prefetch keeps attempts 1-2, kern/vec-raw-split
attempt 3, for reference).

- The fact: RTX 5070 Ti, Q4_K 4096x4096, T = 2..8 vector 14.6 / 15.2 / 21.0 / 18.2 / 16.7 / 23.0 /
  26.1 us (repeats within 0.5 %); registers -> CTAs per SM -> rounds does not order them (T = 6, 5
  CTAs per SM and two rounds, is the fastest from T = 4 on; a rounds term fitted to 3204 calls gains nothing), the
  stalls are long-scoreboard (77-85 %), and the slice loop's SASS issues 2 / 1 / 2 / 2 / 5 / 2 / 3
  global loads before the first IDP: the instance the compiler batched most is the fast one.
- Attempt 1, the next step's slices fetched into registers before this step's products: geomean
  time over master's 1.043 (5070 Ti) and 1.096 (3060), instances 0.62..1.50.
- Attempt 2, the next step's codes prefetched to L2 (`prefetch.global.L2`, no registers): geomean time
  over master's 1.010 on the 5070 Ti, instances 0.68..1.44 (IQ2_S T = 4 22.9 -> 16.1 us, Q5_K T = 8 24.6 -> 35.3).
- Both re-roll the compiler's per-instance schedule rather than fix it.
- Attempt 3, the loads batched in source (kern/vec-raw-split, 29.10): every decoder split into `load()`
  (all of a slice's global loads, written together) and `convert()` (none), the kernel loading slice
  it + 1 before slice it's products. Output byte-identical. Geomean time over master's 0.99 on the
  RTX 5070 Ti (instances 0.67..1.52), 0.99 on the RTX 5090 (0.72..1.29), 1.01 on the RTX 4090
  (0.85..1.20) and 1.00 on the RTX 2080 Ti (0.80..1.24): the same re-roll. Measured with `ninferno_r8_schedule_bench --shapes
  tools/fits/data/vec/raw_split.shapes --tokens 2,3,4,5,6,7,8 --samples 7 --launches 30 --rmma 1`, the vector route's
  time at T = 2..8 on the 44 of the file's 46 shapes the vector kernel takes (each of the 18 ggml
  formats at 4096x4096 store and 12288x4096 SwiGLU, Q6_K add 3840x15360 / 5120x17408 / 4096x12288 /
  2048x6144 / 4096x4096, Q4_K add 3840x15360 / 5120x17408, Q5_K add 5120x17408): 308 instances,
  master and branch builds interleaved twice, the median of each instance's two runs.
  ptxas still places the loads: Q4_K's runtime-K slice loop issued 1..8 loads before its first
  IDP against master's 1..5, and the instances that gained loads ahead were the faster ones (T = 7,
  8 loads ahead against 2, Q4_K add 3840x15360 -28 %; T = 4, 1 against 2, +32 %). Source order
  does not hold them either. Untried: loads fixed in PTX / inline asm, or more warps per SM.
- A trap on the way: a raw slice that kept the 12 Q4_K / Q5_K scale bytes as a register array,
  indexed by the lane's sub-block at run time, put the whole slice in local memory (96-112 bytes of
  stack per thread on sm_120, Q4_K 53 % slower than master). The raw slice keeps only the bytes it
  uses, loaded by index from memory; check `cuobjdump -res-usage` STACK before timing a kernel
  change.

## 60. Decode speed from the bytes and the kernels of a step; a generation's one card prices only its own boards (28.10)

The dense decode estimate (src/product/choice, PR #184) had one fitted term, kDenseDecode's share of
the int8 peak. It is replaced by measured per-generation efficiencies of the kernels a step runs. A
step of C rows takes:

    t(C) = W / (eta_rows(C, class) * BW) + C * row_bytes / (eta_dram * BW) + N_gemv * t_gemv + N_small(C) * t_small

- W: the weights the step reads. The input table, the MTP layers and the own drafter are left out.
- N_gemv and N_small: the kernels the family's graph launches a step (nsys, a fact of the graph:
  `kDecodeLaunches`). They are the same on every card:
  - Qwen3.5-9B Q4_K_M: 145 + 167 (9.75 a layer);
  - Q8_0: 177 + 191 (11.5 a layer);
  - Gemma 4 12B: 213 + 249 (9.6 a layer).

  The output head, the gather, the final norm and the sampling run once a step. They do not scale
  with another size's layers.
- t_gemv and t_small (`kKernelCost`): the census's measured residual, a kernel's time beyond its
  bytes. Where the census explains it:
  - Ada's plain R8: 1.5 us launch plus the 0.29 us graph gap;
  - Blackwell: PDL hides the plain projections' launch (0.2 us median).
- eta_rows: the share of DRAM a weight class's routes reach, by generation:
  - K-quants: vec at 2-4 rows, rmma/MMQ at 5-8;
  - Q8: R8 at every width.

`fit_decode.py` derives every row from two fixtures: `decode_measured.json` and `decode_census.json`.

**IQ4 (IQ4_XS, IQ4_NL) is its own class**, measured on two Ada cards (RTX 4060 Ti, 4070 Ti, 29.10):
its graph launches 161 + 167 kernels a step (R8 IQ4 GEMVs), and its rows are 0.927 / 0.885 / 0.773 at
2 / 4 / 8 rows. At one row (kDramEta and Ada's kernel cost, not fitted on it) it comes within 1.6 %.

**A weight class without a census** (`WeightClass::Other`: other IQ*, Q4_0/1, Q2_K/Q3_K, FP4)
takes the K-quant rows, marked transferred. It does not win by worth while a measured class is in
the tier (even one that does not fit); then the rule's pick stands.

**The tie band between quantizations** is the model's measured error on the generation:
`kDecodePredictionError`, rms of ln(estimate/measured), Ada 2.8 %, recomputed by the test.
- The error is in-sample. Out of the fixture, Q5_K_M, UD-Q5_K_XL, Q6_K, IQ4_NL and UD-Q4_K_XL come
  within 2.1 %.
- Only Ada's row is read, because only Ada has a reference card.
- Inside the band the lower KLD wins: a speed difference smaller than the estimate's error is no
  argument against accuracy.

**One-card rows** (Turing 2080 Ti, Ampere 3060) stay:
- They price only their own boards and are marked transferred.
- No decision reads them: the choice prices on the reference card (Ada only).
- A second card of the generation in the census and the decode campaign removes the mark: any TU10x
  for Turing, a second GA10x (RTX 3070/3080/3090, A4000/A5000) for Ampere.
- The 3060 census (29.10) gives t_gemv 4.55 us and t_small 2.61 us, and 1.8 % rms on its 12 points
  (up to +18 % on Ada's rows).
- Volta takes Turing's rows, transferred (decode: its own rows since DECISIONS 72).

**The defaults this changes against master 8bbf2e72.** Ada; plan without load, 7 models x 3 cards
x {default, 4 streams, 128K}. The measurements are on the RTX 4060 Ti:
- speed: plain decode per stream at 1 / 4 streams;
- KLD: `--quick` against a fresh Q8_0 reference that gives Q4_K_M 0.0192, as kQuantKld has it.

| model, ask | master | this PR | why |
|---|---|---|---|
| Qwen3.5-9B, 128K | Q4_1 | Q4_K_M | Q4_1 is unmeasured (Other) |
| Qwen3.5-9B-MTP, 128K | Q4_0 mtp:3 | Q4_K_S mtp:4 | Q4_0 is unmeasured (Other) |
| Qwen3.5-9B / -MTP, 4 streams | UD-Q4_K_XL nvfp4 | Q4_K_S k8v4 / IQ4_NL k8v4 | worth with the measured KLDs and IQ4 rows |
| Qwen3.5-9B-MTP, default | UD-Q4_K_XL int8 mtp:3 (40,640) | Q5_K_S nvfp4 mtp:5 (34,176) | worth with the measured KLDs, plus the one-stream plan fix below |

The measured numbers behind those rows (KLD on the RTX 4060 Ti and 4070 Ti, each against its own
Q8_0 reference under the exact route: the same file gives the same KLD to six digits on both):

| quant | KLD (curve before) | 1 / 4 streams, tok/s |
|---|---|---|
| Q4_1 | 0.0242 | 50.5 / 45.9 |
| Q4_K_M | 0.0192 | 51.1 / 45.7 |
| Q4_0 | 0.0509 | 54.4 / 49.3 |
| Q4_K_S | 0.0225 (0.028) | 54.0 / 47.9 |
| IQ4_NL | 0.0287 | 53.2 / 47.9 |
| UD-Q4_K_XL | 0.0183 (0.0160) | 48.5 / 43.6 |
| Q5_K_S | 0.0110 (about 0.013) | 46.4 / 41.7 |

**The default row** goes from an estimated 98 to 113 tok/s.
- Measured on essays at one stream: UD-Q4_K_XL mtp:3 71 tok/s, Q5_K_S mtp:5 62. The worth still
  favours Q5_K_S: ln(71/62) = 0.14 < 25 x 0.0073 = 0.18.
- The one-stream plan fix: a one-stream pick by worth was scored at the target context with its
  drafter, then launched at the longest context that fits. On 8 GB that context drops the drafter
  (Q5_K_S plain, 46 tok/s). The launch now keeps the scored context where the drafter returns there.

**Qwen3.8-27B on the 4090** changes too:
- 128K: UD-Q4_K_XL to UD-Q5_K_S;
- 4 streams: UD-Q4_K_XL to UD-Q4_K_M.

Both follow from the measured KLD points moving the curve. They are not measured on 27B: the 4090
stand's disk has 4.7 GB free.

IQ4_XS on the 4060 Ti is a measured class now: plain 55.0 / 49.4 tok/s, KLD 0.0286, and dflash:7 on
essays 85.8 tok/s.

## 61. The decode K split of one-tile calls: the ring fact holds, the last round's cost is open (29.10, open, measured)

**Rule now (`rmma_k_split_fit`).** A one-tile store / add call of more CTAs than SMs (T <= 8) keeps
its K split where the CTA's longest warp walks more K steps than its two-step load ring holds
(steps / warps rounded up, at least three; a step is four decoder runs, `rmma_k_step`: 32 values
for BF16 / Q8, 64 for FP8 / NVFP4 / the 16-value groupwise runs, 128 for the 32-value decoders);
an eight-warp split that breaks it takes four warps where four keep it and the grid fills their
CTA slots (CTAs x 4 >= 16 x SMs), else it stands. The split's one-tile store / add kernels (one
warp along M) are capped at 128 registers a thread (`kRmmaMinBlocks`), so `w` K warps hold 16 / w
CTAs a SM as a fact (ptxas gave Q2_K w1x4 146-152 registers and Q5_K w1x4 129 on sm_120 before the
cap; capped, no rmma kernel gains stack or local memory on sm_75 / sm_89 / sm_120 — the cap first
reached the two-warp-M U4Zp w2x2 too, 159 registers to 128 and 16 bytes of stack on sm_75, and was
narrowed to one warp along M).

**Verified.** Q4_K K = 4096 on sixteen warps (two steps each): 57-70 % of DRAM peak on five lab
boards against 71-74 % for the best split. Q6_K 4096x2560 on eight warps (three steps for half the
warps) against four: RTX 4070 Ti 19.6 against 21.1 us, RTX 3060 32.4 against 32.9 (the shortest
warp's count would have taken four). Against it: the RTX 5070 Ti and RTX 2080 Ti ran four warps
faster on the same shape (10.1 against 11.3 us; 22.5 against 23.9), so at K = 2560 the rounded-up
count keeps master's eight warps without the boards agreeing on it (two each way).

**Open: what a grid's partly filled last round costs.** Pricing it by its empty slots (the fill
CTAs / (slots x rounds), the better fill winning) was wrong on two of four boards and right on
two, measured with `ninferno_r8_schedule_bench` against master 8e7db18c:
- wrong: RTX 4090 Q6_K 3840x15360 add, four warps in one half-filled round 65.3 us, eight warps
  in a full round 79.0 (x1.21); RTX 3060 Q4_K 4096x2560 / 4096x3072 store, eight warps 4-10 %
  slower although they fill more;
- right: RTX 4070 Ti Q4_K 4096x3072 / 5120x5120 store, eight warps 5-8 % faster; RTX 5070 Ti Q4_K
  5120x5120 store and Q5_K 5120x17408 add, eight warps 6-12 % faster; RTX 2080 Ti Q4_K / Q5_K
  5120x17408 add, eight warps 19 % faster (135.3 -> 109.5 us, 161.0 -> 130.8).
The light K-quants' four-warp start (`light_ggml_step`, measured on one board, the RTX 5080) is
itself what these rows question: eight warps won on four of five boards past two CTAs a SM.
The slot model predicted 17-25 % where the boards showed -10..+8 %: a round with fewer resident
warps is slow only below the warps that keep DRAM busy, which depends on the bytes each warp has
in flight (its ring) and the DRAM latency under load. That latency is not a fact we hold; until it
is, the tail stays out of the rule (no fitted band).

**K = 1024: the ring fact alone was wrong on the boards it moved (29.10, measured).** Against master
the rule above changed one shape on every golden board with more CTAs than SMs: 2048x1024 groupwise
Q4 / Q5 / Q6, FP8 and NVFP4 store / add (128 CTAs, 16 steps of 64 values), eight warps (two steps
each, the ring never refills) to four. Four against eight, every rmma one-tile route timed at
T = 1..8, three repeats (`ninferno_r8_schedule_bench`):
- RTX 5070 Ti (70 SMs, 1.8 CTAs a SM): four warps 20-37 % slower (Q4 x1.20, Q5 x1.22, Q6 x1.37);
- RTX 5090 (170, 0.75): 18-38 % slower; RTX 4070 Ti (60, 2.1): within 3 % (Q6 3.5 % slower);
- RTX 2080 Ti (68, 1.9): mixed, NVFP4 8 % slower, Q4 within 1 %, Q5 2 % and Q6 6 % faster;
- RTX 4060 Ti (34, 3.8): four warps 2-16 % faster (Q4 x0.98, Q5 x0.90, Q6 x0.93, NVFP4 x0.84).
With the 128-register cap a SM holds two eight-warp CTAs or four four-warp ones, so on a grid of
fewer than four CTAs a SM the split halves the warps a SM holds, and the loads in flight with them;
the refilled ring does not buy that back. The rule now takes fewer warps only where the grid fills
their slots (CTAs x warps >= 16 x SMs), which moves no golden board's pick (the 4060 Ti, at 3.8,
stays on eight and leaves its 2-16 % there). What a nearly full grid gains from the refilled ring
is the same open question as the tail above.

## 62. A prefill request's fixed cost is the prompt's split passes, not a per-card constant (29.10, open, measured)

**What the kits showed.** The user kits on master ac3a9859 (prompt 2048, chunk 1024) measured the
prefill under the product's estimate on every lab board but the GV100: Qwen3.5 9B Q4_K_M x0.79
(RTX 2080 Ti) .. x0.96 (RTX 5070 Ti), Gemma 4 12B Q4_K_M x0.72 .. x0.89, Gemma always 5-9 % below
Qwen on the same board.

**Why the estimate is high.** `fit_prefill.py` fits a fixed time `c0` per card next to the linears'
and the attention's rates and the product leaves `c0` out. The fitted `c0` is not a fact: 189 ms on
the RTX 2080 Ti and 144 ms on the RTX 3060, where the 2080 Ti kit's server log shows one rate from
1.7K to 8K tokens (2.44k .. 2.51k tok/s). With `c0` dropped from the fit the attention's share of
the BF16 peak comes out above 1 (Ada 1.09, gb20x 1.40): the fit form misses a term.

**The missing fact (server nsys, RTX 4070 Ti, the kit's flags; prompts of 962 / 1912 / 2862 tokens).**
The server's `prompt_ms` equals the request's GPU window within 2.5 ms, so the request is GPU-bound.
Against the prompt length it is a line: Qwen 9B 5880 tok/s plus 25 ms, Gemma 12B 3496 tok/s plus
16 ms; the same requests give Qwen 12.5k tok/s plus about 30 ms on the RTX 4090 and 7.1k plus
about 17 ms on the RTX 5070 Ti (Gemma's shortest request varies by 50 % between repeats there).
The intercept is whole-model passes over a few tokens each:
- Qwen 9B: the prefill splits at the generation prompt's execution boundaries
  (`inspect_prompt_layout`: the assistant turn's begin and its body's begin; the boundary after the
  reasoning open falls inside the `"

"` token with thinking off and is dropped), so after the
  prompt's chunks two more calls run: `<|im_start|>assistant
` (T = 3) and the empty reasoning block
  (T = 4). Each reads every weight
  (`vec_kernel` per linear: 20 ms of the 25), and each boundary copies the GDN states to the host
  (24 x 1 MiB, about 2 ms a boundary).
- Gemma 4 12B: after the prompt's chunks, a host pause of about 21 ms and one call of about 50
  tokens (25 ms, `mul_mat_q` on 60 CTAs: a weight read): about 45 ms a request in the timeline, 8 %
  of a 2048-token prefill on the 4070 Ti, which is the gap between Gemma and Qwen above. The line's
  16 ms intercept understates it: the 963-token request took 291 and 353 ms in its two repeats.
  The pause is the boundary's checkpoint: 161 device-to-host copies, 340.6 MB of the sliding layers'
  KV rings, at about 17 GB/s, the stream waiting on them (PCIe 3 on the RTX 2080 Ti's host halves
  that rate).
Kernels only (the difference of two prompts, warm-up and the split passes cancelling) the
product's linear and attention terms land within 1 % on the 4070 Ti for both models (Qwen 6142
measured against 6204, Gemma 3911 against 3908). For Gemma that total hides two errors that cancel:
its attention runs below its share (51.5 ms over 1900 tokens against 37.4 at
`kPrefillAttentionEta`, about 2 % of the request) and its linears about as much above theirs.

**Not the cause.** The prefill route: `ninferno_r8_schedule_bench --tokens 512,1024,2048` with the
W4A8 tiles and ggml's matrix kernel offered (bench/route-bench-w4a8) puts ggml's kernel 4-20 %
ahead of the best W4A8 tile on the RTX 5090 and the RTX 4090 for every 9B shape but one, where
`kMmqPrefillPays` sends them. That one picks the W4A8 tile on the 4090 at T = 1024 (Q4_K store
4096 x 4096: the W4A8 tile 171 us against ggml's 153), about 1 % of a prefill.

**Open.**
1. The product's prefill: a request's split passes as a term (passes x the weights over the DRAM
   rate, plus the boundary copies), the count read from the chat template's execution boundaries
   after the last message, and `fit_prefill.py` refit with that term in place of `c0`.
2. The engine: the split passes are there so that a restored checkpoint and a cold prefill sum the
   GDN recurrence alike. Each costs one decode step's weight read; on a short prompt two of them are
   a large share of the time to the first token. A pass that reads the weights once for the tail's
   few tokens whatever its boundaries, or the boundary states taken from the chunk that crosses them,
   would keep the restore exact without the extra reads.
3. Gemma's ring checkpoint stalls the stream for the whole copy. The capture takes
   `CaptureStatePlacement::HostSnapshot` (qwen3_5/program/transactions/capture.cpp,
   `capture_prefers_device`) because no Device state image is free: a run of distinct requests
   leaves the cached images holding earlier requests' checkpoints, and the policy counts only free
   images, not ones it could evict. The continuing request then writes the very replica being
   copied, so it waits for the 340 MB to reach the host. A Device fork into an evictable image
   (a copy at the DRAM rate, under 1 ms, released to the host in the background) or only the ring's
   filled rows for a prompt shorter than the window would take it off the request; which
   checkpoint to evict is the cache policy's call.

**Item 1 done (30.10).** `ModelFacts::prompt_tail_calls` is read at start from the artifact's own
frontend (`runtime::prompt_tail_calls`: the execution boundaries a one-message prompt's tail
carries, the same at two prompt lengths); the choice prices each as one decode step of the
candidate without its drafter, in the prefill rate and in a task's time, and its line says
`prefill (3 tail calls)`. The calls of the RTX 4070 Ti profiles (the embedding's grid gives each
call's tokens; 962 / 1912 / 2862-token requests):
- Qwen 9B: 896 | 59 | 3 | 4, 1024 + 832 | 49 | 3 | 4, 1024 + 1024 + 768 | 39 | 3 | 4.
- Gemma 12B (2863 tokens): 1024 + 1024 + 768 | 40 | 3 | 3 | 1.

The last two (Qwen) or three (Gemma) are the generation prompt's: Qwen 3.5 counts 2 with thinking
on or off (`<|im_start|>assistant\n` and the reasoning block; the boundary after `<think>\n` falls
at the end or inside a token), Gemma 4 counts 3 (its template renders an empty thought channel when
thinking is off, its default: `<|turn>model\n`, `<|channel>thought\n`, `<channel|>`). The call
before them is the cache's: the automatic shared prefix at the last message's end (the frontend's
EngineObserved candidate, OpenAI's default automatic boundary) is captured at the 64-token grid
point below it (`neutral_frontier.h`: a capture off the grid would change the bytes after it), so
the prompt's last chunk stops there and the 17 .. 80 tokens to the message's end are a call of
their own, a weight read like the tail's (24 ms for Qwen 9B, 26 ms for Gemma 12B with its ring
checkpoint's pause before it, item 3). The count adds it where a probe's opportunity is off the
grid: Qwen 3.5 3, Gemma 4 4. The UI does not link the engine, so it prices none: its estimate of a
request is short of the server's by those calls (10 .. 60 ms), and a near tie between candidates can
fall the other way there (a later step: the count written into the artifact at conversion, where
the UI reads its facts). Clean server sweeps (`prompt_ms` at 962 .. 7612 tokens, three repeats, the
kit's flags) against two weight reads over the DRAM rate:

| card | model | slope tok/s | intercept ms | 2 x weights / DRAM ms |
|---|---|---|---|---|
| RTX 2080 Ti | Qwen 9B Q4_K_M | 2583 | 24 | 18 |
| RTX 3060 | Qwen 9B Q4_K_M | 2110 | 46 | 32 |
| RTX 4070 Ti | Qwen 9B Q4_K_M | 5973 | 33 | 23 |
| RTX 4060 Ti | Qwen 27B Q4_K_XL | 1081 | 123 | 104 |
| RTX 5070 Ti | Qwen 27B Q4_K_XL | 2195 | 59 | 34 |
| RTX 5090 | Qwen 27B Q4_K_XL | 4906 | 35 | 17 |
| RTX 2080 Ti | Gemma 12B Q4_K_M | 1389 | 41 | 23 |
| RTX 3060 | Gemma 12B Q4_K_M | 1246 | 77 | 39 |
| RTX 4090 | Qwen 35B-A3B Q4_K_M | 20906 | 27 | (MoE: active share) |
| RTX 5090 | Qwen 35B-A3B Q4_K_M | 25362 | 16 | (MoE: active share) |
| RTX 5090 | gpt-oss 20B MXFP4 | 29882 | 7 | |

The two reads are the floor of every dense intercept; the rest (1.2 .. 2.1 x) is the small GEMMs'
and the matrix kernels' partial waves at T = 3 and 4 (item 2 removes both). The GV100's Qwen
intercept (113 ms against 13) is under a server profile.

`fit_prefill.py` now takes the tail as that fact instead of a fitted `c0` (its rows ran with `--no-prefix-reuse`: Qwen 2 calls, Gemma 3). The attention's share
becomes physical on Ada (0.75) and ga10x (0.71); not on gb20x (1.07), where the RTX 5090 runs 4 ..
14 % below the fit on the dense models and the RTX 5070 Ti 3 .. 8 % above it: one card size's
partial waves at chunk 1024 (170 SMs against 70) that one rate per generation cannot hold. On Turing
and Volta one attention share cannot hold Gemma and Qwen together (GV100 Gemma 12B 1476 tok/s at
1.9K tokens and 738 at 21K, Qwen 9B 2950 and 2864): Gemma's 256- and 512-wide heads on the pre-Ampere
attention path are the missing fact. The product's constants stay as they are until those two facts
are in; the refit is the diagnosis.

**The big boards' shortfall is not the tiles' waves (30.10, predicted against measured).** The
partial-wave reading above does not hold for the RTX 5090. After DECISIONS 63 the user kits measure the 2048-token prefill at x0.97 of the estimate on the RTX 5070 Ti
(Qwen 9B and Gemma 12B) but x0.85 / x0.81 on the RTX 5090 and x0.91 / x0.83 on the RTX 4090. The
first prediction was the W4A8 producer-warp tile's partial waves (one CTA a SM, ceil(N / tile rows) x
ceil(T / 128) tiles: the 5070 Ti's 70 SMs fill them to 0.91 .. 0.99, the 5090's 170 to 0.75 .. 0.90).
It does not hold, for three measured reasons:

- The 5090 does not run that tile. Under the route profile (`NINFERNO_BENCH_PREFILL_ROUTES=1`) its
  Q4_K A8 linears, like Ada's, sm_86 Ampere's and Turing's (not the A100's), take ggml's MMQ (`kMmqPrefillPays`); only the
  5070 Ti (and the 5060 Ti) prefill on the tiles.
- Where the tile runs it costs the same per wave: ~104k SM cycles a wave at K = 4096 on the 5070 Ti
  and the 5090 alike (`linear_route_sweep_bench --w4a8-schedules`); a 5090 wave is slower only in the
  bench's long loops, where the power limit holds the clock at 2.45 GHz. The kits' windows run above
  boost (their `sm_clock_mhz`: the kit's clocks, user-measurement-kit.md).
- Against MMQ the tiles lose on the big boards: every board on the tiles (`exp/mmq-prefill-off-all`,
  A/B/A at 312 .. 22K tokens, Qwen 9B and Gemma 12B Q4_K_M) is 3 .. 8 % slower on the 5090,
  8 .. 13 % on the 4090, 9 .. 11 % on the RTX 4070 Ti, 6 .. 8 % on the RTX 4060 Ti, 2 .. 5 % on
  the RTX 3060 and 7 .. 15 % on the RTX 2080 Ti. The gate stays as it is.

MMQ's side kernels are not the gap either. Under PDL (sm_90 on) a dependent kernel starts early and
waits, so nsys's kernel durations overstate it: their sum puts the stream-K fixup, the epilogue
(FP32 plane to BF16) and the Q8_1 quantization at 17 % of the 5090's Qwen 9B prefill, but the time
they add to the union of the kernels' intervals is 9.7 % (fixup 4.4, quantize 2.9, epilogue 2.4),
against 7.9 % on the 4090 (no PDL: the durations are exact). The main kernel takes 0.805 of the 4090's
time where the SMs times the boost clocks predict 0.723 (the same int8 rate a SM and clock), and 468 of its 480 Q4_K calls run stream-K with a fixup, against 140 on the 4090 (ggml's
rule: whole waves only at 90 % wave efficiency). Stream-K is not the loss: whole waves instead
(`exp/mmq-no-streamk`, A/B/A) are 11 .. 15 % slower on the 5090 (Qwen 9B; Gemma 12B 7 .. 12 %), 7 %
on the 4070 Ti (Qwen; Gemma 1.5 %) and even on the 4060 Ti and the 3060. What the 5090 loses
depends on the shape: one Q4_K call at 2048 tokens (route sweep bench, prefill profile) takes 0.71
.. 0.74 of the 4090's time on the large matrices (15360x3840, 3840x15360) but 0.87 .. 0.90 on the
small ones (4096x4096, 3840x3840, 8192x3840: 512 .. 1024 output tiles, three to six a SM), while
both boards' MMQ reaches 27 .. 31 % of the int8 tensor peak. The prefill estimate's rate per
generation does not see the shape: a Qwen 9B or Gemma 12B layer is mostly small matrices for 170
SMs. The nsys ratio above (0.805) mixed both and the profile's clocks, which were not recorded.

**Volta's own prefill rows (30.10, predicted against measured).** The Quadro GV100's kits measured
its 2048-token prefill at x1.28 (Qwen 9B) and x1.21 (Gemma 12B) of the estimate: Volta had no row
and took Turing's (`prefill_stand_in`: the same 1536 BF16 rate a SM), but Turing's int8 row is
ggml's MMQ on the int8 MMA, which Volta does not have. Volta's prefill linears dequantize each
chunk's weights to FP16 (`decode_rows_kernel`, 17 % of the step) and multiply on the HMMA tiles
(CUTLASS sm70: 0.70 of the m8n8k4 rate at the 1.28 GHz the board holds at its power limit). Its
census (server nsys, master 59933708, prompts of 962 .. 2862 tokens twice; kernel time without the
attention kernels and the tail calls) gives the linears 0.235 (Qwen) and 0.261 (Gemma) of
`peak(Int8)` at the board's 1627 MHz boost, and the attention 0.139 (Qwen's sm70_d256 kernels) and
0.097 (Gemma's m8n8k4 kernel). The rows: 0.248 and 0.100 (Gemma's: the attention is 8 % of its
prefill against Qwen's 2 %). Predicted at 2048 tokens: Qwen 2977 tok/s against 3002 measured
(x1.01), Gemma 1825 against 1807 (x0.99). One card: transferred.

The same census shows item 3 three times larger on a PCIe 3 host: each Gemma request waits twice
on about 670 MB of copies to the host (52 and 80 ms on the GV100, 6 % of a 2048-token prefill
each; twice the 340 MB the RTX 4070 Ti's profile copied, the KV storage the GV100's plan took not
yet broken down), against 19 ms on the RTX 4090 and 7 ms on the RTX 5090 (340 MB at 17 and 48
GB/s: the intercepts' excess over the tail calls on those boards).

**Item 3, the bytes (30.10, predicted against measured).** The pause is the copy, and half of it
was dead rows: a sliding ring holds window + chunk - 1 rows (Gemma 4 12B: 1024 + 1023), while a
call from a checkpoint's frontier reads keys from frontier - window + 1 on (the Gemma kernels stage
no key below a row's window and zero-fill what they skip). A checkpoint's Host transfers now move
only its last `window` positions' rows (`StateImageStore` records the frontier at freeze;
`StateImageDevicePool::copy_to_host` / `copy_from_host`), 1024 of 2047: half the bytes, so half the
pause. Predicted on the RTX 4090 (six distinct 1721-token prompts, the default pool, from the third
request on): 307 - 27 = 280 ms; measured 280-281 (master 306-309; with 16 cached Device states, no
Host copy at all, 250-255). RTX 4060 Ti: 907-916 against 936-940. What remains is the other half,
a Device fork's call: the policy's, not the bytes'.

## 63. OpenAI's automatic cache boundary is taken at the closed turn's end (30.10, measured)

**What the profiles showed.** Every chat request on the RTX 4070 Ti ran one whole-model call more
than the prompt's chunks and the generation prompt's tail (DECISIONS 62): the embedding's grid gives
Qwen 9B's calls as 896 | 59 | 3 | 4 for 962 tokens, 1024 + 832 | 49 | 3 | 4 for 1912, and
1024 + 1024 + 768 | 39 | 3 | 4 for 2862; Gemma 12B the same with 40 tokens. The 39 .. 59-token call
costs 24 ms (Qwen) and 26 ms (Gemma, its ring checkpoint's copy waiting before it), two decode
steps' time for a weight read and a few tokens; on the GV100 the same call ran on the FMA tile
(125 ms).

**Why.** `ninferno-serve` gives an OpenAI chat request a shared-prefix candidate after the last
message's last part (`openai_common.cpp`, its default automatic boundary). That point lies inside
the turn, before `<|im_end|>`, off every execution frontier, so the program moves the capture down
to the 64-token grid point below it (`neutral_frontier.h`: a capture elsewhere would change the
bytes after it), and the last chunk stops there; the tokens up to the turn's end are a call of
their own.

**The change.** A boundary after a message's last part (a message without tool calls) resolves to
the closed turn's end (`message_boundaries`: the layout's closed block, or a rendered prefix), as a
message boundary does. Before the generation prompt or an assistant turn the turn's end is where
the next turn opens, an execution frontier of the prompt (`prompt_layout.cpp`), so the capture
needs no split; for the last message it is the turn closure checkpoint's frontier, and the shared
candidate and the private capture become one group where the cache takes both
(`select_shared_captures`). Where the turn's end is no execution frontier (a tool result followed
by another tool turn, an assistant turn followed by a user one) the planner moves it down to the
grid as before, from a point at or above the old one. A
continuation of the conversation renders the turn closed the same way, so it matches the prefix as
before; a sibling prompt that differs only after the content and before the close could not have
followed the same messages.

**Measured** (`prompt_ms` of the server at 312 .. 7612 tokens, three repeats, the kit's flags; slope
and intercept over 962 .. 7612, the best of three at about 960 tokens):

| card | model | master: tok/s + ms | this: tok/s + ms | 960 tokens, ms |
|---|---|---|---|---|
| RTX 4090 | Qwen 9B Q4_K_M | 12282 + 29.1 | 12339 + 17.1 | 107.1 -> 92.9 |
| RTX 4090 | Gemma 12B Q4_K_M | 7458 + 91.9 | 7383 + 60.1 | 216.8 -> 177.0 |
| RTX 5070 Ti | Qwen 9B Q4_K_M | 7209 + 19.9 | 7182 + 10.9 | 153.2 -> 148.8 |
| RTX 5070 Ti | Gemma 12B Q4_K_M | 4400 + 34.3 | 4364 + 12.8 | 243.7 -> 244.3 |

The intercept falls by the one call on every row; the rate is the same within 1 %.

**Checked.** `ninferno_qwen3_5_frontend_test`: the boundary after the last part lands on the turn
closure's frontier, one between parts stays at its part's end. `ninferno_qwen3_5_prefix_real_test`
(Qwen3.5 9B MTP Q4_K_M, RTX 5070 Ti): every restore case byte-exact against its cold prefill. The prefill tail count (`prompt_tail.cpp`,
DECISIONS 62) reads where the program places the capture, so it drops the call without a change.


## 64. Served speculation = the tune's round cost × the requests' acceptance; acceptance is the model's and the text's, not the card's (30.10, measured)

**Question.** The plan's speculation gain is above the gain measured on the hard question set
(hard50, `--spec auto`, one request at a time) on every card. Is the round cost model wrong
(DECISIONS 53), or is the text different?

**Prediction.** A served round costs what the tune measured (same kernels, same depth), and plain
decode costs what the tune measured; only the tokens a round keeps depend on the text. So
gain on hard50 = tune plain ms/token × hard50 tokens per round / tune ms per round, with hard50's
tokens per round one number per model for every card (greedy decoding, same drafter).

**Measured** (hard50 with the #244 registry, 30.10; tokens per round from the server's per-request
`accepted` counts; the prediction uses the mean of the cards, 4.758 Qwen, 4.360 Gemma, except GV100
Gemma which served mtp:3 by the thinking flag):

| card | model, served | tokens/round: corpus math, hard50 | ms/round: tune, hard50 | gain: predicted, measured |
|---|---|---|---|---|
| RTX 5090 | Qwen 9B Q4_K_M, dflash:7+head | 5.51, 4.71 | 6.40, 6.27 | ×2.79, ×2.81 |
| RTX 4090 | Qwen 9B Q4_K_M, dflash:7+head | 5.41, 4.74 | 9.68, 9.67 | ×3.01, ×2.99 |
| RTX 5070 Ti | Qwen 9B Q4_K_M, dflash:7+head | 5.46, 4.75 | 10.77, 10.66 | ×2.97, ×2.97 |
| RTX 2080 Ti | Qwen 9B Q4_K_M, dflash:7+head | 5.39, 4.81 | 19.23, 19.18 | ×2.65, ×2.65 |
| GV100 | Qwen 9B Q4_K_M, dflash:7 | 5.34, 4.77 | 22.83, 20.89 | ×2.49, ×2.43 |
| RTX 5090 | Gemma 12B Q4_K_M, mtp:5 | 4.87, 4.34 | 7.37, 7.63 | ×3.12, ×3.06 |
| RTX 4090 | Gemma 12B Q4_K_M, mtp:5 | 4.97, 4.35 | 11.69, 11.85 | ×3.21, ×3.21 |
| RTX 5070 Ti | Gemma 12B Q4_K_M, mtp:5 | 4.97, 4.29 | 13.16, 13.35 | ×3.13, ×3.09 |
| RTX 2080 Ti | Gemma 12B Q4_K_M, mtp:5 | 4.97, 4.46 | 24.93, 27.39 | ×2.66, ×2.60 |
| GV100 | Gemma 12B Q4_K_M, mtp:3 | 3.60, 3.37 | 22.24, 23.83 | ×2.35, ×2.33 |

Every gain within 2.5 % of the prediction. The served round costs the tune's round within 2 % on the
RTX 5090, 4090 and 5070 Ti and the 2080 Ti's Qwen; the 2080 Ti's Gemma (+9.9 %) and the GV100's rows
(-8.5 %, +7.1 %) move with the plain decode of the same run (clock; the GV100's older tune), and
their round over their plain step agrees with the tune's within 5 % (+4.6, +2.4, +0.7 %): the
prediction takes that ratio. hard50's tokens per round agree across cards within ±1.1 % (Qwen) and
±2.2 % (Gemma at mtp:5). The gap to the tune's math gain is the text: hard50's math keeps fewer tokens per
round than the tune corpus's math section, 11–15 % for Qwen at dflash:7(+head), 10–14 % for Gemma at
mtp:5 (6 % at the GV100's mtp:3); the gap to the plan is the same against the thinking section the
plan prices (below).

**Decision.** The round cost model stands; nothing is fitted. Acceptance is a property of the model,
the drafter and the text, so one card's measurement of a text transfers to every card. The plan
keeps pricing the tune corpus by the benchmark mix (DECISIONS 50); a mix of harder text is a new
corpus section measured once per model, not a per-card factor. The GV100 Gemma row is the mix at
work: by the thinking flag the non-thinking requests get mtp:3, the best on the corpus mix (prose
weighs 8 of 19), while math alone would take mtp:5 (predicted ×2.78 on hard50).

**What the plan shows.** A server started without a flag takes thinking on (`enable_thinking`
defaults to true), so the plan prices the served configuration on the corpus's thinking section:
on the RTX 4090 with the #244 registry Qwen 9B serves dflash:7+head fixed and the plan says 580 tok/s
(x3.54 of 163.7), Gemma 12B serves mtp:5 and the plan says 395 (x3.41 of 115.9); hard50 measured x2.99
and x3.21, its requests asking with thinking off (a fixed server serves both flags the same
configuration). Same configuration, same round cost: the difference is the text alone. The
corpus's thinking section keeps 5.57 tokens a round for Qwen (dflash:7+head) and 4.64 for Gemma
(mtp:5) against hard50's 4.74 and 4.35, so hard50's measured gains price the plan's text at
x2.99 x 5.57 / 4.74 = x3.51 and x3.21 x 4.64 / 4.35 = x3.42, within 1 % of the plan's x3.54 and x3.41.
An earlier reading of this gap as the plan pricing a configuration the server does not run was
wrong: the plan's and the server's choice come from the same `served_speculation`, and a pricing
of the served configuration (`config_gain`) equalled the pick's speedup on every profile checked.

## 65. An MoE's decode from facts: concurrent rows route apart (30.10, measured)

The MoE decode estimate had its own fit (`kMoeDecode`: a share of the int8 peak and two terms,
`kMoeExactFactor` for the exact route), apart from the dense model's facts (DECISIONS 60). It is
replaced by the same step:

    t(C) = (W_rest + R (1 - (1 - k/E)^C)) / (eta * BW) + C * row_bytes / (eta_dram * BW) + launches

- W_rest: the weights outside the routed banks (not the input table): the attention, the shared
  expert, the router, the output head. R: the routed banks as the GGUF stores them
  (`ffn_*_exps`, `routed_expert_bytes` of the plan facts).
- Concurrent rows are separate requests and route independently: a step reads the experts any of
  them routes to, 1 - (1 - k/E)^C of the banks, each expert at the C k / E-th of the rows it was
  picked for (its read at that many rows' rate). `kMoeRoutedShare` (0.30, DECISIONS 53) was
  fitted on the speculation round, whose rows are consecutive tokens of one text and route
  together; it stays there (`spec/round_cost.cpp`).
- Launches: the census of the family's graph (nsys, RTX 5070 Ti, one row): Qwen3.6-35B-A3B 311
  weight-reading + 267 small kernels a step, gpt-oss-20b 97 + 199; Gemma 4 26B-A4B (RTX 4090)
  181 + 219, its dense MLP beside the experts in each of 30 layers; at Ada's and Blackwell's
  `kKernelCost`.

**How a step is measured.** Two traps:
- The fixture's short-chat rows of 2-8 streams are wave aggregates (their prompts, ramp and
  stragglers in), not a step: the dense check never used them.
- Concurrent requests with the same text route the same experts (at temperature 0 they write
  the same tokens): eight copies of one story ran Qwen3.6 Q4_K_M on the RTX 4090 at 128.9 tok/s a
  stream, eight different tasks at 108.1, and the copies' step would read the routed banks at
  1056 GB/s, above the board's 1008.

The step rows (`"step": true` in `speed_measured.json`) are the server's own 5-second throughput
windows whose mean batch is exactly C, eight different tasks (code, math, translation, prose),
3000 tokens each, the median (`mbatch.sh`); at 1-8 rows they agree with the difference of 320- and
64-token waves within 3 % where both batch fully (on the 16 GB cards short waves of eight do not).

**The routed banks' share of the DRAM rate.** The census of Ada's step (RTX 4090, no PDL: kernel
times are the kernels' own) puts the routed kernels (`moe_gemm`, `moe_r8`) at 0.68 / 0.83 / 0.81
of the DRAM rate at 1 / 4 / 8 rows once their launches are off: the experts the rows route to are
many small reads, one or two rows each, not the dense GEMVs kRowsEta measures. Their share is a
measured row of its own (`kMoeRoutedEta`), derived like kRowsEta: the step rows less every other
term of the step, the mean over the generation's cards and quantizations
(`tools/fits/speed_model/fit_moe_decode.py`):

| generation | 2 rows | 4 rows | 8 rows | cards |
|---|---|---|---|---|
| Ada | 0.928 | 0.911 | 0.806 (4090 only) | RTX 4090, 4060 Ti |
| Blackwell | 0.918 | 0.740 | 0.634 | RTX 5070 Ti, 5090 |

Ada's are its K-quant rows within 0.035; Blackwell's are lower from four rows on (0.74 against the
K-quants' 0.80, 0.63 against 0.73).

**Predicted / measured** (tok/s a stream at C rows, Qwen3.6-35B-A3B, master kit build):

| card | quant | C: predicted / measured |
|---|---|---|
| RTX 4090 | UD-Q4_K_M | 1: 244 / 254, 2: 203 / 205, 4: 156 / 158, 8: 101 / 107 |
| RTX 4090 | UD-Q3_K_S | 1: 259 / 263, 2: 223 / 219, 4: 180 / 175, 8: 125 / 119 |
| RTX 5070 Ti | UD-IQ3_XXS | 1: 271 / 270, 2: 235 / 218, 4: 168 / 153, 6: 135 / 119 |
| RTX 5070 Ti | UD-Q3_K_S | 1: 241 / 245, 2: 209 / 207, 4: 149 / 150, 7: 108 / 106, 8: 99 / 98 |
| RTX 5090 | UD-Q3_K_S | 1: 375 / 383, 2: 334 / 348, 4: 252 / 265, 8: 176 / 177 |
| RTX 4060 Ti | UD-Q3_K_S | 1: 96 / 96, 2: 80 / 81, 4: 61 / 63, 7: 44 / 45 |
| RTX 4090 | Gemma 4 26B-A4B UD-Q4_K_M (check only) | 1: 217 / 210, 2: 173 / 164, 4: 128 / 124, 8: 82 / 90 |

33 points with the four one-row points of 2026-10-23: rms 4.4 %, worst 13 %
(`ninferno_choice_test`). The Gemma rows check the table and do not make it (below). The old fit's check ran on the wave aggregates and missed by 68 % at
eight rows once its constants were gone.

**Open.**
- UD-IQ3_XXS on the RTX 5070 Ti, 7-13 % slow at 2-6 rows: its banks are IQ3_XXS throughout (a
  codebook decode a weight), UD-Q3_K_S's a mix. The census without PDL (a lab build with
  `pdl::supported()` false, so the kernels' times are their own) puts its routed kernels at 6.1 /
  6.9 / 6.9 weights a SM-clock at 1 / 4 / 8 rows: the same rate whatever the rows, compute-bound
  on the 70-SM board (0.45-0.51 of its DRAM rate), where the RTX 4090's 128 SMs at that rate reach
  their DRAM rate. A format's decode rate a SM-clock against the bytes is the candidate fact for
  the routed share (Blackwell's lower row with it); the next decision.
- Gemma 4 26B-A4B (UD-Q4_K_M, RTX 4090, KV and ring fp8 as the server chose): its routed share
  measured as above rises with the rows, 0.81 / 0.79 / 0.85 / 0.95 at 1 / 2 / 4 / 8, where Qwen3.6's
  falls (0.93 / 0.91 / 0.81 at 2 / 4 / 8): the estimate is 3-6 % fast at 1-4 rows and 9 % slow at
  eight. Its experts are 4 MB each (Qwen3.6's 1.3 MB), 30 MB a layer at one row, and a dense MLP
  runs beside them; which of these the step's census shows is the next measurement. Until then its
  rows are kept out of the table (`CHECK_ONLY` in the fit).
- gpt-oss-20b (MXFP4): decode and prefill take format routes of their own on every generation;
  the next decision. What the censuses show (nsys, master kit build):
  - Blackwell's prefill runs the experts on the block-scaled MMA (`moe_w4a8_kernel<W4a8Mxfp4Mx>`,
    `kind::mxf8f6f4`, 71 % of the RTX 5070 Ti's prefill), the weights entering as stored: 1.29 x
    (5090) and 1.57 x (5070 Ti) the MoE row's estimate. Ada converts them to int8
    (`W4a8Mxfp4`) and stays on its row (4090 x1.03, 4060 Ti x1.14).
  - The GV100's prefill runs the same int8 kernel (83 % of it) with no int8 MMA, on dp4a: 512
    operations a SM-clock against the table's 1536, a third, and it measured 0.34 of the estimate.
  - Its decode under this model (the server's windows, as above): one row within 1-5 % on the
    RTX 4090, 4060 Ti, 5070 Ti and 5090; at 4-8 rows 3-26 % slow, the estimate the slower. Its
    experts are ten times Qwen3.6's reads (about 13 MB a layer against 1.3 MB): the routed share
    measured on Qwen's small reads is low for them. The read's size is the candidate fact.
  - The GV100's decode reads its dense Q8 linears at 0.44 of the DRAM rate and the routed MXFP4
    banks at 0.28: about 3-4 weights a SM-clock unpacked on the CUDA cores, not bytes (79.7 tok/s,
    the estimate 185).

## 66. Prefill linears from a tile's cost (30.10, open, measured)

A prefill linear runs as a grid of tiles: a tile of the int8 kernels (the dense `w4a8_kernel`, the
MoE experts' `moe_w4a8_kernel`) is 64 tokens x 64 output rows, the whole K deep. The census (nsys of
one ~1.8K-token prompt after a warm-up, master kit build; the RTX 4090 without PDL, the RTX 5070 Ti
on a lab build with `pdl::supported()` false, so a kernel's time is its own) gives one law for every
such call:

    t(call) = waves * (c0 + K * c1),   waves = CTAs / SMs,   CTAs = ceil(T / 64) * N / 64

| kernel | generation | c0 | c1 (a K step) | calls it was read from |
|---|---|---|---|---|
| int8 tile (W4A8) | Ada | 1.33 us | 4.82 ns | Qwen3.6 experts K 2048 / 512, gpt-oss K 2880; dense K 2048 |
| int8 tile (W4A8) | Blackwell | 0.5 us | 4.74 ns | dense K 2048 / 2880 / 4096 |
| block-scaled MX (MXFP4 experts) | Blackwell | ~0.5 us | ~3.4 ns | gpt-oss experts K 2880 |
| 16-bit register MMA (A16 experts) | Ada | 0.93 us | 7.1 ns | Qwen3.6 UD-Q3_K_S experts K 2048 (IQ3_XXS) / 512 (IQ3_S), a 64 x 64 tile's share |
| 16-bit register MMA (A16 experts) | Blackwell | 0.32 us | 6.1 ns | the same |

On the RTX 4090 the law gives the dense calls of Qwen3.6-35B-A3B within 1-6 % (the Gated DeltaNet
in-projection 11.2 against 11.5 us a CTA, attention's qkv 370 against 375 us, the shared expert's
gate/up 45 against 47 us) and gpt-oss's experts from Qwen's (15.2 against 15.0 / 13.6 us a CTA).
In SM-clocks a K step of the int8 tile is one number across generations: 12.1 on the RTX 4090
(4.82 ns at 2520 MHz), 12.8 on the 4070 Ti and 12.5 on the RTX 3060 (Ampere: 7.05 ns at 1777 MHz;
both read from Qwen3.5 9B Q8_0's calls at K 4096 and 12288 once the grid runs token-major,
PR #249), about 11.6 on the RTX 5070 Ti: the tile's cost is a count of clocks, the card's clock
gives its time. Between the two K the line puts c0 slightly below zero on the Ampere and the
4070 Ti (the down projection at K 12288 has fewer CTAs a SM): c0 stays the 4090's read until a
third K is read. c1 is the same in nanoseconds on both generations at their different clocks: 1.7 int8 TOPS a SM,
a third of the tensor peak, in every call. The 16-bit register MMA's K step costs only 1.5x the int8 tile's at a quarter of
its tensor peak: both are bound by decoding the weights on the CUDA cores, not by the MMA (the
same bound the decode census found for IQ3_XXS's routed kernels, DECISIONS 65); the int8 tile's
step is the same for Q8_0, Q4_K and MXFP4 weights, the 16-bit one's is read on IQ3 weights only.

An MoE's expert grid counts jobs, not tokens: a job is an expert's run of up to 64 of the chunk's
slots (token x pick). The host launches the bound before the routing is known (`ops/moe`
`max_jobs`): ceil(slots / 64) + min(slots, experts) rows of tiles, every expert's partial job
counted; the rows past the device's job total exit at once. The census's grids are that bound
exactly: Qwen3.6-35B-A3B 356 / 384 / 484 rows at chunks of 795 / 1024 / 1819 tokens (256 experts,
8 picks), gpt-oss-20b 96 at 1024 (32, 4), Gemma 4 26B-A4B 342 at 1714 (128, 8). With K and the
jobs the law gives the RTX 4090's Qwen3.6 experts at 0.61 of gpt-oss's efficiency; measured 0.62.
Gemma 4 26B-A4B UD-Q4_K_M on the RTX 4090 (1714 tokens): its experts' gate/up (K 2816) 879 against
955 us a layer, the down (K 704) 557 against 584; its dense int8 calls 402 / 209 / 211 against
411 / 225 / 203 us (N 8192 / 4224 / 2816).

What this replaces: the per-generation prefill efficiencies (kPrefillEta's rows, kPrefillTilesEta,
the MoE rows and kMoeChunkShare) are one number a generation for every shape; a narrow model's
linears (Qwen3.6's 2048 hidden) are 11 % (Ada) to 31 % (Blackwell) slower than its row says, and
an MoE's expert route (int8, the 16-bit register MMA, MXFP4) changes the kernel, not a share.

**Checked whole** (every linear of the model from its geometry, the waves per call rounded up,
against the census's kernel time): Qwen3.6-35B-A3B UD-Q4_K_M on the RTX 4090, the experts 45.7
against 50.5 ms, the dense int8 calls 28.2 against 30.0 ms; gpt-oss-20b's experts 67.9 against
68.6 ms (RTX 4090) and 83.7 against 80.0 ms (RTX 5070 Ti, MX). Not yet priced: the activations'
quantization, attention, the recurrence, the MoE's routing kernels (about a quarter of the
prompt's kernel time) and the prompt's tail calls (DECISIONS 62).

**The prototype against the MoE campaign** (GEMMs by the tile law with these jobs, the routing's
gather and scatter at T k hidden 2 bytes twice a layer over the DRAM rate, attention as
before; not yet the recurrence, the small kernels and the prompt's tail calls): gpt-oss-20b -7 ..
+10 % on the RTX 4090 and 5070 Ti at 2K .. 21K tokens; Qwen3.6-35B-A3B +0 .. +31 % on the RTX 4090
(the missing terms, a fifth of a 2K prompt there), +24 .. +75 % on Blackwell before the jobs
bound (at chunk 1024 the bound's 384 rows hold 21 slots a job, the 4090's 2048-token chunk 30).

Done: the plan's linear shapes from the geometry it reads (`runtime::PlanLinear`: attention,
recurrent, FFN, shared expert, experts) reach the choice, and `prefill_linears_seconds` prices
an MoE's linears by these tiles (`kTileCost`, the jobs bound; Qwen3.6 UD-Q4_K_M on the RTX 4090:
79 against the census's 86.4 ms), not yet in the estimate. What the rest of a prompt is (the
RTX 4090's Qwen3.6 UD-Q4_K_M, one 1819-token prompt, the window from the 2-second pause: 103.3 ms
of a 106 ms request): the experts 45.1 ms (the law 45.9), the int8 dense calls 29.9 (28.2), the
routing's gather and scatter 4.3 (5.0); not yet priced: the Gated DeltaNet's recurrence,
convolution, preparation and gating 7.6, the prompt's tail calls about 7, the router, plan and
selection 2.8, the norms and the activations' quantization about 3, attention 2.4.

A dense model's Q8 linears by the same law (Qwen3.5 9B Q8_0, one 1819-token prompt in chunks of
1024 and 795, token-major grid since PR #249; its linears qkv 10240 x 4096 and o 4096 x 4096 in 8
layers, the Gated DeltaNet's 12288 x 4096 and 4096 x 4096 in 24, the FFN's 24576 x 4096 and
4096 x 12288 in 32; the int8 row in SM-clocks): predicted / measured kernel time RTX 3060 808 /
767 ms, 4060 Ti 467 / 440, 4070 Ti 257 / 246, 4090 125 / 127, 5070 Ti 217 / 212 (-2 .. +6 %,
three generations, from the SMs and the clock alone). The rest of that prompt: the two tail
calls (DECISIONS 62, `r8_kernel` at a decode step's reads: 16.7 ms on the 4090, two weights over
the DRAM rate 18.6), attention 2.6 ms, and the small kernels (the recurrence 3.5, the activations'
quantization 3.3, the convolution 1.4, the norms 1.7, the preparation 0.9: about 10 % of the
prompt on the 4090, 8 % on the RTX 3060). The small kernels read activations a few times a layer:
on the 4090 from its 72 MB L2 (the quantization 4.7 us a call against 12.5 at the DRAM rate),
on the RTX 3060 from DRAM (26 against 35); their bytes against the L2 is the next term.

The small kernels split by what bounds them (Qwen3.5 9B Q8_0, the 1819-token prompt, the census
outside the linears, the tail calls and attention; seven cards):

| card | Gated DeltaNet scan ms | x SMs x GHz | the rest ms | x DRAM TB/s |
|---|---|---|---|---|
| RTX 3060 | 23.4 | 1165 | 40.4 | 14.6 |
| RTX 4060 Ti | 13.0 | 1121 | 22.3 | 6.4 |
| RTX 4070 Ti | 9.3 | 1457 | 12.3 | 6.2 |
| RTX 4090 | 4.4 | 1419 | 7.1 | 7.2 |
| RTX 5070 Ti | 5.3 | 909 | 8.8 | 7.9 |
| RTX 5090 | 3.2 | 1309 | 4.7 | 8.4 |
| RTX 2080 Ti | 42.7 | 4483 | 34.2 | 21.1 |

The chunked scan is a count of SM-clocks (about 1.1-1.5 G on Ampere and Ada, 0.9-1.3 G on
Blackwell, three times that on Turing, whose scan has no tensor path): about 30 K SM-clocks a
token a recurrent layer. The rest (the activations' quantization, the norms, the convolution,
the gates) is bytes: about 7 GB on Ada and Blackwell, some 15 passes over a token's 8 KB hidden
row a layer; the RTX 3060's 3 MB L2 and Turing's kernels double it.

`prefill_small_seconds` prices them so (`kSmallKernelCost`: the scan's SM-clocks a token and state
element, 0.058 Ada / 0.048 Blackwell / 0.051 Ampere / 0.196 Turing; the rest's bytes a token, layer
and hidden element, 30 / 30 / 60 / 88), within -16 .. +14 % of the seven points it was read from.
**Checked on another model**, predicted before reading (Gemma 4 12B Q4_K_M, no recurrence, hidden
3840 in 48 layers, a 1714-token prompt; the census's norms, the q/k/v preparation and the
activations' quantization, MMQ's epilogue left with its linears): RTX 4090 9.4 against 12.9 ms,
4070 Ti 18.8 / 23.9, 4060 Ti 33.0 / 43.7, 5070 Ti 10.6 / 12.5 (its K-quant runs the int8 tiles),
RTX 3060 52.4 / 53.5, 2080 Ti 45.0 / 35.8: -27 .. +26 %. Ada's shortfall is MMQ's quantization
(q8_1 of every linear's input, 5.9 of the 4090's 12.9 ms against the Q8 tiles' 3.3 on Qwen): the
quantization's bytes follow the route's codec and the linears' input widths, the next refinement.
At 5-10 % of a prompt the term's error is 1-3 % of it.

**In the estimate** (`linear_prefill` through `tiled_chunk_seconds`): a chunk on the int8 route
whose linears have a tile row (a dense model's Q8_0, an MoE's banks with Q8_0 beside them) costs its
linears by the tiles plus the small kernels; attention and the tail calls as before; every other
route keeps its generation's share. On the choice test's Q8_0 points (Qwen3.5 9B on seven cards,
Gemma 4 12B on three, the fixture's models now carrying their linears): rms 7.6 % (worst 18 %),
against 8.9 % (worst 25 %) for the clock-count share of DECISIONS 67. What closed most of the RTX
5090's gap is the grid's rounds: the int8 tile holds two CTAs a SM (128 threads at 177-255
registers, 40 KB), so a grid runs in rounds of SMs x 2, each two tiles' time; the 5090's N 4096
calls at a 1024-token chunk (1024 CTAs) take 4 rounds of 340, 8 tiles' time, where 6.02 waves of
170 said 7 (`TileCost::resident`; the other cards move 0-3 %). The 5090 is now +6 .. +17 %, the
RTX 4090's Gemma +10 .. +18 %; its clock is not the cause (2546 MHz on average over the prompt
against its reference 2407). The MoE campaign's 16-GB quants (UD-IQ3_XXS on the
RTX 4060 Ti and 5070 Ti, master f8d4399d) keep the shares: their linears beside the experts are
Q6_K (`w4a8_ws_kernel` on Blackwell, 55-66 ms of a 230 ms prompt; the Q6_K tile and MMQ on Ada),
a tile without a clock count yet.
An MoE's whole prefill through the tiles (the choice test's `prefills_moe_by_tiles`, master
f8d4399d's rows at 2K / 7K / 22K tokens): Qwen3.6-35B-A3B UD-Q4_K_M on the RTX 4090 +7 / +7 / +0 %,
the RTX 5090 +13 / +8 / -1 %; Gemma 4 26B-A4B UD-Q4_K_M on the RTX 4090 +12 / +13 / +11 % with its
two tail calls (the census: two decode steps' kernels after the chunk, 13 ms). Not yet priced: the
router and the selection (1.7-2.8 ms), Gemma's heavier norms (its small kernels ran 21-27 % over the
term on Ada, DECISIONS 66 above).
Measured clocks (nsys GPU metrics over the prefill window): the RTX 4090 runs Qwen3.5 9B Q8_0's
prompt at 2422 MHz on average (2225-2535) against its reference 2520; the RTX 2080 Ti board at
1860 MHz (power-capped) against 1545: the "power-limited clock" reading of the 2080 Ti's Q8 gap is
refuted (its clock is above the reference); its int8 tile takes 30-34 SM-clocks a K step at that
clock, 2.5-2.8x Ada's, not the 2x of half the CUDA cores a SM.

**Gemma 4 12B on the RTX 5090** (added after, sm_bench.py prefill at 3K-14K, master f8d4399d):
Q8_0 +17 .. +26 %, Q4_K_M +12 .. +17 % (MMQ's share, not the tiles): the shared term is attention.
The census at 1714 tokens: Gemma's attention kernel 14.2 ms against the term's 7.7 on the 5090,
18.3 / 9.7 on the 4090, 29.1 / 20.0 on the 4070 Ti, 51.4 / 36.4 on the 4060 Ti, 23.4 / 18.3 on the
5070 Ti, 87.4 / 94.1 on the RTX 3060: kPrefillAttentionEta's Ada and Blackwell rows (0.74) are Qwen's
kernels', and Gemma's falls further below them the more SMs the card has (the RTX 3060's own row
holds): a grid's quantization, its sliding layers' 352 CTAs (44 query blocks x 8 KV heads) two a
SM, 2 rounds on the 4090 and on the 5090 (whose second holds 12), 6 on the 4060 Ti, about 190-214 K
SM-clocks a round on Ada and 156 K on Blackwell. Next: attention by its own tiles. Q8_0 points: rms
9.5 %, worst 26 %.

Next: a dense K-quant's prefill runs
ggml's MMQ (stream-K: on the RTX 4070 Ti one wave of 60 CTAs, or waves of 71.9 us at K 4096 for
grids of 216-384 CTAs, the same a wave within 1 %), the Gated DeltaNet's recurrence and the small
kernels, Volta's dp4a route; then the estimate, checked against data/prefill/rows.json and the
MoE campaign.

## 67. A dense model's Q8 prefill from the int8 tile's clock count (30.10, measured)

**What changed.** Since PR #249 the 64-token W4A8 tiles (the Q8 codec of `q8_prefill_a8` among
them) run token-major, so a call no longer streams a matrix larger than L2 from DRAM once per 64
tokens. Qwen3.5 9B Q8_0's prefill (sm_bench.py prefill, 3K-14K-token prompts, the choice test's
`prefill_measured.json` points) rose 1.65x on the RTX 4060 Ti, 1.46x on the 4070 Ti, 1.35x on the
4090 and 1.18x on the 5070 Ti, and the fitted `kPrefillTilesEta` rows (master 03afc418 points)
priced it 25-40 % slow.

**The fact instead of a fit.** A call bound by its tiles' own work takes, per 64 x 64 tile,
c0 + K c1 SM-clocks. The census (nsys, the token-major grid) reads c1 at 12.1 SM-clocks a K step on
the RTX 4090, 12.8 on the 4070 Ti, 12.5 on the RTX 3060 (Ampere) and 11.6 on the 5070 Ti
(Blackwell): the same count on every generation with an int8 MMA, a third of its tensor peak
(decoding the weights on the CUDA cores bounds it). The share of the int8 peak a dense model's Q8
linears reach is then 2 x 64 x 64 / ((c0 / K + c1) x s8) at a hidden width of 4096: Ada 0.308 (c0
3352, c1 12.15), Blackwell 0.336 (1226, 11.62), Ampere 0.300 (the RTX 3060's c1 12.5 with Ada's c0:
one card, so transferred). Turing's int8 tile is another kernel with no clock count read: its row
stays the 03afc418 fit (its points rose 1.11x and it lands within 4 % of them).

**Predicted / measured** (`ninferno_choice_test`, `prefill_measured.json`: sm_bench.py prefill on
master f8d4399d, 3K-14K-token prompts, estimate / measured - 1):

| card | Qwen3.5 9B Q8_0 | Gemma 4 12B Q8_0 |
|---|---|---|
| RTX 2080 Ti (Turing, its measured row) | -1 .. -4 % | |
| RTX 3060 (Ampere, transferred) | -7 .. -15 % (it ran 1.82 GHz against the table's 1.78) | |
| RTX 4060 Ti | -1 .. -9 % | 0 .. +5 % |
| RTX 4070 Ti | +2 .. -7 % | |
| RTX 4090 | +10 .. -1 % | +7 .. +15 % |
| RTX 5070 Ti | -1 .. -9 % | -3 .. +1 % |
| RTX 5090 | +14 .. +26 % | |

60 points, rms 8.9 %; the fitted rows they replace priced the same points 25-40 % slow on Ada. The
error falls with the prompt on most cards (attention's term, the same as Q4_K_M's).

**Open: the RTX 5090.** Its census (nsys, token-major) runs the tiles at the law's time at its
reference clock (2407 MHz: about 20 us a wave at K 4096, the law 20.3), yet a call of 1024 CTAs
(the N 4096 projections at a 1024-token chunk) is 6.02 waves of 170 SMs and takes 7: about 14 % of
the linears, which one share a generation cannot see. The tile law by linear with its waves
(DECISIONS 66, `prefill_linears_seconds` on the prefill branch) prices it; it replaces these rows
when the rest of a prompt (the small kernels) is priced too.

## 68. A prompt's attention from its CTAs (30.10, measured)

**What changed.** `kPrefillAttentionEta` priced attention as one share of the BF16 peak a
generation, read on Qwen's points; Gemma 4's kernel ran 1.3-1.9x that term on the cards with most
SMs (DECISIONS 66), and Qwen's own grid leaves SMs idle on a large card. `prefill_attention_tiled_seconds`
prices the INT8-KV prompt kernel (int8/prompt_fast.cuh: sm_80 on, head width 256, Qwen 3.5's full
layers at `--kv-dtype int8`, the default) from its grid, the way the kernel issues it:

- a CTA is 128 query rows of one head over every key before its last row; one CTA a SM (98 KiB of
  shared memory); the grid is ceil(chunk / 128) row blocks x the heads, issued longest first (the
  last row block of every head, then the one before) to the SM that frees first;
- a CTA takes `in_chunk` SM-clocks a key of its own chunk and `prefix` a key before it
  (`kAttentionTileCost`): Ada 304 / 194, Ampere 374 / 188, Blackwell 210 / 154;
- a prompt is its chunks in turn, each over the keys of the chunks before.

**Read** on the census of a 7K-token prompt (seven 1024-token chunks, a layer's median a chunk; the
RTX 5090 on a lab build without PDL): RTX 4060 Ti and 4090 (its 1.8K prompt) -4 .. +2 %, RTX 3060
-1 .. 0 %, RTX 5090 -1 .. +1 %. **Checked** where it was not read: a 21K prompt's 21 chunks on the
RTX 3060 -1 .. 0 %, 4060 Ti -1 .. +2 %, 5090 -1 .. +2 %; the RTX 5070 Ti's 7K prompt -10 .. +1 %
(its first chunk, where the kernel's half-height slots fill the last wave, is the -10 %).

What it showed on the way: issued in plain launch order the same constants missed the 70-SM and
34-SM cards' first chunk by +10 .. +30 %; the kernel issues longest first, and with that order one
pair of constants holds on both Ada cards. On a 170-SM card the grid of 128 CTAs (16 heads x 8
blocks at a 1024-token chunk) is one round with 42 SMs idle: the RTX 5090's attention is the
longest CTA's time, not the work over the SMs.

A call of 256 tokens or fewer (a short prompt, a short last chunk, a chunk pinned that narrow) takes
the grouped kernels (int8/plan.cpp, the keys split over the SMs), not this one: it keeps the share.
Not modelled at first: the kernel's narrow layouts (four-warp CTAs, or half-height leading slots
where they shorten the last wave), and the invariant route profile (every width on ParallelGrouped:
exact activations keep the share).

**The kernel's layout and a CTA's own cost** (21K-token prompts, 21 chunks, on five cards). The 21K
census showed the RTX 4070 Ti +12 .. +17 % past the first chunks: its 60 SMs take two half-height
leading slots (grid 9 x 16, int8_prompt_fast_layout), which the law dealt as full CTAs. The law now
runs the kernel's own layout choice (its makespan in hundredths of a CTA, half-height at the
kernel's figure 0.72) and deals those CTAs last. Then a constant a CTA (`cta`) besides its keys:
Blackwell's two cards agree on 20800 SM-clocks (the RTX 5070 Ti's first chunk, 128 CTAs over 70
SMs, was its -10 %), Ampere's one card cannot tell it from zero. Read on both cards of a generation
at once: Blackwell 195 / 152 / 20800 (5090 -1 .. +2 %, 5070 Ti -1 .. +2 %), Ampere 378 / 190 / 0
(3060 -0.4 .. +0.8 %), Ada 328 / 188 / 0 (4060 Ti -3 .. +9 %, 4070 Ti -8 .. +4 %). Open on Ada: the
two cards' first chunks want opposite CTA costs (the 4060 Ti none, the 4070 Ti 28500 SM-clocks;
each alone fits within 1 %); a clock ramp after the pause is refuted (the chunk's linears run their
steady time from the first chunk on).

In the estimate (Qwen3.5 9B's points): Q8_0 rms 7.6 -> 7.1 %, Q4_K_M 6.0 -> 6.4 % (MMQ's shares were
read with the old attention term beside them). Not yet: Gemma 4's kernel (sliding layers: 2 CTAs a SM,
a round of 352-520 CTAs about 160-177 K SM-clocks on Ada and Blackwell; full layers a persistent grid
on the RTX 5090), the BF16 and FP8 KV kernels, Turing and Volta's kernels (no row: the shares stand).

**Gemma 4's kernel by its CTAs** (`kGemmaAttentionCost`, ops/softmax_attention/gemma4): a CTA is a
row block of one KV head (32 packed rows: 16 tokens of the sliding layers' group 2, 2 tokens of the
12B full layers' group 16), the grid one block more than the rows need, issued longest first within
a head. The sliding layers' D=256 tile holds two CTAs a SM, each over at most the window's keys; the
full layers' D=512 tile one, over every key before its rows; on Blackwell (`attention_chain`) a full
layer whose grid fills its last wave under 90 % runs persistent, its work over the SMs. SM-clocks a
key of a CTA, sliding / full: Ada 186 / 250 (RTX 4060 Ti), Ampere 180 / 255 (RTX 3060), Blackwell
157 / 222 (RTX 5070 Ti, 5090). Read on 7K and 21K prompts' 1024-token chunks: sliding -8 .. +7 %,
full -13 .. +9 % a layer; a chunk's whole attention on the 4060 Ti -3 .. -1 %, the 5090 -9 .. +2 %.
A short last chunk whose last round holds one CTA a SM runs faster than dealt (the lone CTA,
kLoneTile's effect: +14 .. +80 % on that chunk, not modelled). A group-8 model's full layers (26B,
31B) take the wide tile from 1024 tokens: not read, their share stands.

In the estimate: Gemma 4 12B on the RTX 5090 Q8_0 +17 .. +26 % -> +12 .. +22 %, Q4_K_M +12 .. +17 %
-> +7 .. +14 %; Q8_0 points rms 9.2 -> 8.0 %. What is left on Gemma is not attention: its small
kernels (four norms a layer, GELU) ran 21-27 % over the term on Ada (DECISIONS 66).

**BF16 keys by their kernel's CTAs** (`kBf16AttentionTileCost`, bf16/tiled_mma.cuh at `--kv-dtype
bf16`): a CTA is 64 query rows of one head (98 KiB, one a SM), issued in plain launch order (a head's
row blocks first to last, then the next head) — the opposite of the INT8 kernel's longest first, and
with it the same deal holds. SM-clocks a key of a CTA, in its chunk / before it: RTX 5090 144 / 120
(-1 .. +2 %), RTX 4060 Ti 168 / 144 (-2 .. +1 %), RTX 3060 168 / 142 (0 %); checked where not read,
the RTX 5070 Ti's 7K prompt: 0 .. +1 %.

**FP8 keys by their kernel's CTAs and its merge** (`kFp8AttentionTileCost`, common/mxfp8_tiled_mma.cuh
at `--kv-dtype fp8`, a call wider than 80 tokens). The grid is 128-row tiles x heads x splits in
launch order (one CTA a SM, 99 KiB). The split count is the kernel's own (mxfp8_tiled_partition: the
fewest waves a split, 1 .. 8), capped at one split a 512 keys the call's last row sees; split s owns
the 64-key tiles [s T / a, (s + 1) T / a), a CTA those up to its last row. A CTA takes `cta` SM-clocks
(its query rows' load and Hadamard rotation, which a CTA left with no keys pays too: that reading fits
every card better than a free empty CTA) and `key` a key. Read on 7K and 21K prompts, each generation's
cards together: Ada 182 / 35000 (4060 Ti and 4070 Ti -3 .. +3 %), Blackwell 138 / 30000 (-6 .. +6 %:
the 5090 runs slower and the 5070 Ti faster than one pair gives; open), Ampere 284 / 75000 (3060 -2 ..
+3 %, one card). The merge after every call (one split too) reads each row's FP32 partial of every
active split, 256 x 4 + 8 bytes, whether that split held keys or not (a CTA with no keys stores an
empty one), and writes the BF16 row: 0.85 .. 1.04 of the DRAM rate where those bytes outgrow the
board's L2 (`kAttentionMergeDramShare` 0.9), 2.4 .. 4.7 of it where the L2 holds them
(`kAttentionMergeL2Share` 3.0: the RTX 5090 at 7K, the 4070 Ti's and 5070 Ti's first chunk). The
L2 is the board table's (gpu_facts.json `l2_mib`, MachineFacts, `CardFacts::l2_bytes`), the same for
a live device and a named board; a board whose row does not give it takes the DRAM rate. Kernel and
merge together on the 7K and 21K prompts' chunks: RTX 5090 -6 .. +2 %, 4060 Ti -3 .. +13 % (its first
chunk: 40 MiB over its 32 MiB L2, which still holds a part, 1.27x the DRAM rate), 3060 -2 .. +2 %,
5070 Ti +3 .. +7 %, 4070 Ti -2 .. +8 %. Open: the 5090's 21K chunks hold the partials in its L2 only
partly (1.6x, not 3x): the merge of a long prompt there is under-priced.

**Turing's row for Gemma 4's kernel** (31.10): the RTX 2080 Ti's census of Gemma 4 12B Q4_K_M at 7K
tokens. Its sliding layers' grid (65 x 8 CTAs of 128 threads at 230 registers, 20 KiB) runs two CTAs
a SM as on the other generations (one a SM misses the first chunk by -22 % and the rest by +5 %), its
full layers' (513 CTAs of 256 threads at 229 registers) one: 430 / 575 SM-clocks a key (sliding -9 ..
+4 %, the first chunk the -9 %; full -2 .. +5 %), about 2.3x Ada's (Turing's HMMA without the int8
and BF16 paths Ada's kernel takes). One card: transferred. The prefill estimate of Gemma 4 12B on the
RTX 2080 Ti had +16 .. +25 % from its attention's share (DECISIONS 69's open point).

**Volta's row for Gemma 4's kernel** (31.10): the Quadro GV100's census of Gemma 4 12B Q4_K_M at 7K
tokens (Volta's m8n8k4 route of the kernel, `kGemma4VoltaHmma`, same grids: sliding 65 x 8 CTAs of 128
threads at 255 registers and 44 KiB, two a SM by both; full 513 CTAs of 256 threads, 89 KiB, one).
615 / 805 SM-clocks a key (sliding -16 .. +3 %, the first chunk the -16 %; full -1 .. +1 %), 1.4x
Turing's. The first chunk, where a CTA holds 16 .. 1024 keys, runs slower than its keys say on Turing
(-9 %) and Volta (-12 % of the chunk): open, not a per-CTA constant (two levels of data, one
unknown each). One card: transferred.

## 69. A dense K-quant's prefill linears from ggml's matrix kernel's tiles (31.10, measured)

**What changed.** A dense model's K-quant linears (Q4_K_M ... Q6_K: `WeightClass::KQuant`) were priced
by a share of the int8 peak a generation. Where the product prefills them on ggml's matrix kernel
(ops/generic `mmq_prefill_wins`: the nearest measured board of the card's instruction set pays,
`measured::kMmqPrefillPays`, which now lives in core/measured_boards.h so the ops and the estimate
read one table) `prefill_linears_seconds` prices them by MMQ's tiles:

- a call runs 512 tokens at a time (`kMmqPrefillSlice`); a tile is 128 weight rows x 128 tokens over
  the whole K, one CTA a SM (256 threads at 254 registers); a slice of fewer tokens takes ggml's
  narrowest column tile with the fewest tiles (16, 32, 64, 128), its K step 0.43 + 0.57 J / 128 of
  a 128-column tile's (the RTX 4060 Ti's 64-column Q4_K tiles: 0.716);
- a gated FFN's gate and up run as two grids of half the rows (`LinearShape::gated`), each with its
  own rounds or stream-k;
- a slice's grid of ceil(N / 128) x ceil(t / 128) tiles runs a tile a CTA in rounds of SMs where it
  fills its last wave to 90 %, else stream-k: the SMs share the tiles' K steps evenly, and a fixup
  kernel adds the split tiles (ggml mmq.cuh);
- a tile takes c0 + K c1 SM-clocks, c1 by the weights' format: Q3_K's decode (its high-bit mask and
  6-bit scales) costs as much as Q6_K's, Q4_K the least. A hub row names no tensor's format, so a mix
  takes its label's format (Q4_K_M: Q4_K), its bits a weight past that format's moving it toward
  Q6_K's (the mixes raise ffn_down, attn_v and the like to Q5_K / Q6_K); Q2_K and the IQ formats are
  not read (the share);
- beside the tiles, every grid's quantization of its input to q8_1 (3.125 bytes an input element: the
  census runs one a grid, a gated pair's two each), the epilogue's FP32 -> 16-bit output (4 bytes read
  an output element, 2 written an element of the call's output: a SwiGLU pair writes one row of the
  two), and a stream-k call's fixup (a tile's FP32 partial a SM, written and read), all at the DRAM
  rate. The small kernels' passes, read on the int8 route, also count that route's one quantization an
  input, which MMQ does not run: not taken out (without MMQ's own quantization the fixture's points ran
  +2 .. +5 % faster than measured).

**Read** on Qwen3.5 9B's Q3_K_M, Q4_K_M (the MTP file: Q4_K / Q5_K / Q6_K by tensor), Q5_K_M and Q6_K
at 7K tokens (nsys; the calls by ggml type and grid against the GGUF's tensor types and the runtime's
fusions — a Gated DeltaNet layer's qkv and gate fuse when their formats match, a full layer's q, k
and v likewise; gate and up run apart), each generation's cards together; checked on Q4_K_S, which
they were not read on:

| generation | cards | c0 | Q3_K / Q4_K / Q5_K / Q6_K | read | Q4_K_S |
|---|---|---|---|---|---|
| Ada | RTX 4060 Ti, 4070 Ti | 2000 | 51.5 / 42.75 / 45.25 / 50.75 | -5 .. +3 % | -3 .. +8 % |
| Ampere | RTX 3060 | 4000 | 49.25 / 41.5 / 43.5 / 48.75 | -1 .. +1 % | 0 % |
| Turing | RTX 2080 Ti | 0 | 91.75 / 71.75 / 74.5 / 77.75 | -8 .. +4 % | -2 .. +2 % |
| Blackwell | RTX 5090 (stream-k throughout) | 3000 | 60.5 / 45.5 / 49.5 / 59 | -7 .. +7 % | -5 .. +7 % |

A first reading by bits a weight alone (c1 = a + b x bits, from Q4_K / Q5_K / Q6_K) missed Q3_K by
-32 %: its decode is not its bits'.
Checked where not read: Gemma 4 12B Q4_K_M on the RTX 2080 Ti, its calls by grid -4 .. -3 %, the
stream-k mixture +2 %. The bytes beside the tiles: RTX 3060, 2080 Ti and 5090 within 8 %; Ada's L2
holds part of them (0.65 .. 0.77 of the bytes' time). A card whose nearest measured board keeps the
tiles (the RTX 5070 Ti, as the 5060 Ti: `w4a8_ws` for Q4_K / Q6_K on Blackwell, the classic tiles
for the rest) keeps the share: those tiles are not read yet (the 5070 Ti's Q6_K `w4a8_ws` tile:
about 20000 + 24.1 K SM-clocks, 128 tokens x 64 rows, one CTA a SM). An MoE keeps its rows.

In the estimate (the fixture's prefill points): Q4_K_M rms 6.9 -> 6.7 %; Qwen3.5 9B's points within
-3 .. +5 % but the RTX 4090's +4 .. +12 % (no 4090 census of MMQ), Gemma 4 12B's -7 .. +7 % on six
cards. Open: Gemma 4 12B on the RTX 2080 Ti +19 .. +25 % (its MMQ priced right, its attention is the
share's: Turing's Gemma kernel runs one CTA a SM, DECISIONS 66); the stream-k fixup on the RTX 5090
runs at 1.7x its bytes' time; Q3_K mixes (below: Q3_K mixes on MMQ); the
bits a weight are the whole file's.

**Where MMQ does not pay: the W4A8 tiles** (`kW4a8KquantCost`, 31.10). A card whose nearest measured
board keeps the tiles (the RTX 5070 Ti, as the 5060 Ti) prefills a dense K-quant on the W4A8 tiles:
on Blackwell Q4_K and Q6_K the producer-warp 128 x 64 tile (`w4a8_ws_wins`), Q5_K the classic one. A
tile is 128 tokens x 64 weight rows (a gated pair's 32 gate and 32 up rows) over the whole K, one CTA
a SM, a call's grid ceil(N / 64) x ceil(t / 128) in rounds of SMs; the activations' quantization and
the epilogue are the small kernels' (the int8 route's own). Read on the RTX 5070 Ti's census of
Qwen3.5 9B's Q4_K_M, Q4_K_S, Q5_K_M and Q6_K at 7K tokens: c0 within a few K steps, Q4_K 23, Q5_K 28,
Q6_K 29.5 SM-clocks a K step, the K-4096 and K-12288 calls within 2 % — a 128 x 128 MMQ tile's on the
RTX 5090 per element (Q4_K 45.5 = 2 x 22.75). One card: transferred. In the estimate the 5070 Ti's
Q6_K and Q5_K_M rows went from +14 .. +16 % to 0 .. +4 %, its Q4_K_M from 0 .. +4 % to +2 .. +6 %;
every K-quant point of the fixture rms 4.5 -> 3.5 %, worst 16.4 -> 12.1 % (Qwen3.5 9B Q4_K_M on the
RTX 4090's 3K prompt).

**Gemma 4 12B's Q5_K_M and Q6_K** (31.10, DECISIONS 69's law, predicted before the measurement):
mpref.sh rows at 7K and 22K on the seven cards (master f8d4399d): 26 of 27 points within -3 .. +11 %
(rms with the outlier 6 %). Open: the RTX 2080 Ti's Q6_K at 7K runs +27 % slower than the estimate
(1150 prompt tokens a second; its Q5_K_M 1353, +11 %), where every other card's Q6_K runs 0.92 .. 0.96
of its Q5_K_M and the 2080 Ti's 0.85; 1.4 GiB free after 9.1 GiB of weights (its 22K row did not
fit). A census is queued.

**Q3_K mixes on MMQ** (31.10): a Q3_K mix stays `WeightClass::Other` (its decode keeps the Other
row), but its prefill linears take DECISIONS 69's law where MMQ pays: the label's format Q3_K
(3.4375 bits a weight) with its own K step read on the census's Q3_K_M calls (Ada 51.5, Ampere
49.25, Turing 91.75, Blackwell 60.5: Q6_K's within 3 % but on Turing, 18 % above it), toward Q6_K by
the file's bits as the other formats. Another Other label (a K-quant label whose file holds another
format) keeps the share. Predicted for
Qwen3.5 9B's Q3_K_M (Q3_K gate, up, q, k and attention gate; Q4_K the rest; 4.176 bits a weight) at 7K
and 22K on six cards: -9 .. -3 %, rms 6.6 % (slower than measured throughout: the mix's Q4_K
calls, cheaper than Q3_K's, priced on the label's one step toward Q6_K). Open: where MMQ does not pay (the RTX 5070 Ti) Q3_K runs
the classic W4A8 64 x 64 tile (two CTAs a SM, about 31.5 us a tile at K 4096) and the mix's Q4_K the
producer-warp one: one label, two tiles, so the share prices it (+22 .. +24 %) until a model's
linears carry their own formats. Another engine's repacked layouts (Q4_K_R4) stay out.

**Linears by their tensors' formats** (31.10): a GGUF's plan facts tag each linear with its role
(attention in and out, a recurrent mixer's in and out, FFN up and down: `gguf_role`, a recurrent
layer's attn_qkv and attn_gate its mixer's input) and split a role whose tensors differ in format
into one linear a format, its layers at the format's share of the role's elements
(`split_linears_by_format`: a Q4_K_M's FFN down half Q6_K). MMQ and the W4A8 tiles price each by its
format's K step; a linear of no named K format (a role without tensors, Q8_0's ssm_out in a Q6_K
file) keeps the label's. On the fixture (the GGUFs' role shares, a one-off lab script): Qwen3.5 9B's
Q3_K_M -9 .. -3 % -> -4 .. +3 % (rms 6.6 -> 2.5 %); the K-quants together (167 points) rms 4.9 % with
the two RTX 5070 Ti Q3_K_M points still the share's (+22, +24 %: Q3_K's 64 x 64 W4A8 tile not read).
Gemma 4 12B's sliding attention input was marked a gated pair in the fixture: fixed.
A table's rows each take their own file's formats (their GGUF tensor table's linears) and a row without
its own keeps the label's step, never the read file's (a repository's smallest file is the
shape); a split part's head names no shares. Where the file names formats, a linear of none (Q8_0,
16-bit) takes the label's format's own step, not its bits toward Q6_K (the Q6_K parts priced as
theirs already).

**Q3_K on the W4A8 tiles** (31.10): where MMQ does not pay (the RTX 5070 Ti) a named Q3_K linear
runs the classic W4A8 tile (W4a8Q3K, `W4a8Schedule<64, 64, 2, 2>`: 64 tokens x 64 rows, 128 threads
at 185 registers and 40 KiB, two CTAs a SM), a grid of ceil(N / 64) x ceil(t / 64) tiles in rounds
of 2 x SMs, a round 37.4 SM-clocks a K step (the census's three grids' rounds 62.2 .. 64.3 us at K
4096); its Q4_K and Q6_K parts keep the producer-warp tile. Qwen3.5 9B's Q3_K_M on the RTX 5070 Ti:
+24, +22 % (the share) -> +3, +2 %; the K-quants together rms 4.9 -> 4.2 %. A Q3_K label whose
linears are not named keeps the share there (two tiles, no split). Open: the dense tuner's shipped bands
(routes/tuner/dense_tune.json) move some gb206 (RTX 5060 Ti) Q3_K problems of a 27B's shapes (hidden
5120, FFN 17408) from the 64 x 64 tile to the classic 128 x 64 at 128 .. 2048 tokens; the estimate
prices them on the 64 x 64 (the census card, the 5070 Ti, has no Q3_K band).

**IQ4_XS on MMQ and the W4A8 tiles** (31.10): an IQ4_XS file (`WeightClass::IQ4`) prefills its
dense linears on MMQ where MMQ pays, its K step read on the census of Qwen3.5 9B's IQ4_XS at 7K
(the calls by grid, c0 the generation's): Ada 38.3, Ampere 36.6, Turing 59.6, Blackwell 42.9 SM-clocks
(each card's call groups within 2 %), below Q4_K's (the IQ4_XS block's lookup decode is cheaper
than Q4_K's scale unpacking); where it does not (the RTX 5070 Ti) the producer-warp 128 x 64 tile
as Q4_K, 20.2 a K step (the four grids' rounds 20.0 .. 20.4). Its Q5_K linears (attention values,
the GDN output) by their own format. Predicted against mpref.sh rows at 7K and 22K on six cards:
-4 .. +4 %, rms 2.5 %. IQ4_NL: not read (the share): an IQ4_NL label, and a file whose linears name
another ggml block format (an IQ4_XS file's iq4_nl FFN down, where the FFN is no multiple of 256:
the W4A8 64 x 64 tile there). Open: the dense tuner's shipped gb206 (RTX 5060 Ti) IQ4_XS bands move a
27B's shapes to the 64 x 64 tile at 128 .. 1024 tokens; the estimate prices the producer-warp one. A file whose linears name their
formats is priced by them whatever its class: unsloth's UD-Q4_K_XL (Q4_K, Q5_K, Q6_K, IQ4_XS and Q8_0
linears; the IQ4 class by its iq4_xs tensors, its label Q4_K's) on Qwen3.5 9B, 7K and 22K on seven
cards: -1 .. +9 % (the RTX 4090 the +9, +7), rms 3.6 %. A named format the laws do not read (IQ4_NL,
MXFP4, NVFP4) keeps the share. Open: a named file's Q8_0 and 16-bit linears (UD-Q4_K_XL's GDN output)
take the label's K step, where the runtime runs them on the int8 tile and the 16-bit GEMM; a split
UD file names no linears and its IQ4 class keeps the share.

## 70. Volta's dense prefill from its CUTLASS route (31.10, measured)

**What changed.** Volta has no int8 or FP16-accumulate mma.sync path the other generations' tiles take:
a dense prefill casts the activation to FP16, decodes each linear's weight rows to FP16 in chunks
(`kChunkWeightBytes`, 17408 rows at K 4096, a gated pair's gate and up together), multiplies each
chunk on CUTLASS's sm_70 GEMM (128 x 128 x 32 tiles of tokens x weight rows, two CTAs a SM) and stores
through its epilogue (generic_volta_cutlass.cu). Its prefill was a share of the peak (no tiles row,
no small-kernel row: `tiled_chunk_seconds` gave up). `volta_linears_seconds` prices the route:

- the GEMM: a chunk's tiles in rounds of SMs x 2, each two tiles' time, a tile's K step 52.4 SM-clocks
  (format-blind: one GEMM after the decode);
- the decode: a weight element's SM-clocks by format — Q8_0 0.69, Q4_K 0.99, Q5_K 1.11, Q6_K 0.94 (the
  K-quants' decode is compute, Q8_0's near its bytes), a mix by its label's format;
- the cast (4 bytes an input element) and the epilogue (6 bytes an output, a SwiGLU pair 10; a plain
  projection of the hidden width is stored by the GEMM) at Volta's DRAM rate.

and `kSmallKernelCost` takes a Volta row: the scan 0.381 SM-clocks a token and state element (the
chunk recurrence and its preparation), the rest 56 bytes a token, layer and hidden element.

**Read** on the Quadro GV100's census of Qwen3.5 9B's Q8_0, Q6_K and Q4_K_M (the MTP file) at 7K
tokens (nsys 2025.1: the system's 2026.3 records no kernel on Volta): the GEMM's calls -4 .. 0 %, the
decode -2 .. +1 %, the cast and epilogue at 0.75 of the DRAM rate (-21 .. -27 % on a tenth of the
time); the linears together Q8_0 -2.3 %, Q6_K +1.2 %. One card: transferred (DECISIONS 52: an
architecture's facts). Not read: Qwen's attention (its sm70 flash kernel) keeps its share; Gemma 4's is Volta's row of
DECISIONS 68.

**Where the route is not CUTLASS** (the review): CUTLASS is built for sm_70 alone, so an sm_72 board
(Jetson Xavier) keeps the share, as does a call under 17 tokens (the HMMA band); a tiled format's call (Q8_0, Q4_K, Q5_K) under 256 tokens on a board
below 770 GB/s (the Titan V) runs its own tile (`volta_copy_pays`), not read: the share prices it.
Open: the epilogue's bytes follow Qwen3.5's fused calls (its qkv and GDN input projections of 2 .. 4
segments, its gate and up as `linear_swiglu`); Gemma 4 and the generic dense pass store their q / k /
v and gate-up calls directly (a separate `gelu_tanh_mul` after the gate-up) and cast once a call, a few
per cent of their linears' time. A grid of at most SMs CTAs (a short call of few rows) is priced at two
tiles' time, not a lone tile's.


**The Quadro GV100 end to end** (31.10): mpref.sh rows (the Quadro GV100's kit build fd6e572c) of Qwen3.5 9B's
Q6_K, Q5_K_M, Q4_K_S and Gemma 4 12B's Q4_K_M, Q5_K_M, Q6_K at 7K and 22K against the estimate with
DECISIONS 70's linears and Gemma's Volta row: rms 6.4 %, -6 .. +13 %. Open: Qwen's Q4_K_S and Q5_K_M
at 7K (+13, +11 %) where its Q6_K is +4 %: the GV100 runs Qwen's Q6_K faster than both (3017 against
2737 and 2742 prompt tokens a second), the decode costs read on the Q4_K_M file give them the other
order.

**Dense decode steps by quantization** (31.10, verification): mbatch.sh steps at 1, 2, 4 and 8 rows of
Qwen3.5 9B's Q3_K_M, IQ4_XS, Q4_K_S, Q5_K_M, Q6_K and Gemma 4 12B's Q4_K_M, Q5_K_M, Q6_K on seven
cards, the non-MTP GGUFs (speed_measured.json, 200 points: Gemma's 4 and 8 rows missing for four card
and quantization pairs whose windows never held them) against the decode estimate as it stood (predicted before the
measurement): rms 6.0 %, one row 3.0 %; by card 3.2 .. 9.7 % (the RTX 2080 Ti the 9.7). Open: at 8 rows
the estimate is 6 % slower on average (Gemma 4 12B on the RTX 4060 Ti -16 .. -18 %); the RTX 2080 Ti's
Qwen3.5 9B Q6_K at 2 and 4 rows +25, +20 % (the estimate faster).

**Qwen3.5 4B's prefill** (31.10, verification): another geometry of the same family (hidden 2560, FFN
9216, the 9B's heads, KV and GDN, tied embeddings), its Q4_K_M by its linears' formats, mpref.sh rows at
7K and 22K on eight cards predicted before the measurement: -6 .. +9 % on the seven (the RTX 5070 Ti the
+9, +7), the Quadro GV100 +13, +1 %. Facts as plan_facts derives them from the GGUF's metadata (prefill_measured.json).

**Long prompts** (31.10, verification): mprefl.sh rows at 41K and 82K tokens (the median of three
slices, master f8d4399d; the Quadro GV100 fd6e572c) of Qwen3.5 9B's Q4_K_M and Q8_0 and Gemma 4 12B's
Q4_K_M on seven cards against the estimate as it stood (the attention laws of DECISIONS 68 read up to
22K): 42 points, rms 5.1 %; the consumer cards -6 .. +10 % (the RTX 5090's Q8_0 the +10), the GV100's
Qwen -5 .. -14 % (its attention the share's, DECISIONS 70), its Gemma +3, -1 %.

**Long-context decode** (31.10, verification): mdecl.sh rows (one stream, 513 tokens after 7K, 41K and
82K-token prompts less the first token's time, the median of three; master f8d4399d, the Quadro GV100
fd6e572c) of Qwen3.5 9B's Q4_K_M and Q8_0 (KV int8) and Gemma 4 12B's Q4_K_M (KV and rings FP8 on
sm_89 and up, BF16 below, as the servers' logs state on the RTX 5090, 3060 and 2080 Ti) on eight cards
against the decode estimate as it stood (decode_measured.json, rows marked long, the context the
prompt + 256): Ada, Blackwell and Ampere 54 points, rms 3.1 %; an extra KV byte costs them 1.0 .. 1.16
of its time at the full DRAM rate (the estimate's 1 / kDramEta, 1.06). Open: the RTX 4070 Ti at 82K,
+6, +10, +15 % (measured twice; its 7K and 41K within 5 %). Turing reads int8 KV at 1.70 of its time
at the full DRAM rate (Qwen, both files, all spans: 1.62 .. 1.81) and Gemma's BF16 at 2.16, where the
estimate takes 1.0 and 3.38 (kKvReadFactor's BF16 row, read on Qwen at 22K): 9 points -12 .. +9 %.
The GV100 is +6 .. +24 % at 7K already (no decode row measured on it beyond the HBM2 share,
kDramEtaOf) and its spans are not monotonic (Qwen 4.4 then 0.9): 9 points +6 .. +35 %. The campaign
rows of 28.10 price Gemma on the RTX 3060 and 2080 Ti with FP8 KV, which the server no longer picks
below sm_89 (open).

## 71. Turing's decode attention bound by its SMs (31.10, measured)

**Decision.** A decode attention kernel that reads its KV below the DRAM rate on a generation has a row
of kKvSmRate (choice_constants.h): the bytes it reads a SM-clock at the board's reference clock, for
the family's kernel and the codec. A row's KV then costs the longer of its DRAM time and SMs x clock x
that rate, in place of kKvReadFactor's codec factor (read on another family's kernel). Rows: Turing,
Qwen3.5's int8 grouped kernel 3.58 B a SM-clock, Gemma 4's BF16 full-layer kernel 2.67, each for
the geometry it was read on (a full layer's head width, query and KV heads: 256 / 16 / 4 on Qwen3.5
9B, 512 / 16 / 1 on Gemma 4 12B; the grouped kernels' rows and tiles follow the group): another
geometry keeps the codec's factor.

**Why.** The long-context decode rows (DECISIONS 70's verification, Long-context decode) put the RTX 2080 Ti's KV at 1.70 of
its DRAM time on int8 and 2.16 on Gemma's BF16 where the estimate took 1.0 and 3.38. The nsys census of
one stream's decode at 7K and 82K (mdcen.sh) reads the kernels directly: per call, the 75K tokens
between add 423 us for 159 MB on Qwen's int8_kv_grouped_mma (376 GB/s) and 545 us for 153 MB on
Gemma's 512-wide full layer (281 GB/s): 61 and 46 % of the card's 616 GB/s, at the 1770 MHz the board
holds (sampled). The same kernels on the RTX 3060 and 4070 Ti reach 92 .. 94 % of their DRAM: the SMs,
not the DRAM, bound Turing's (68 SMs of 64 FP32 lanes; the 3060's 128 lanes a SM read 6.6 B a SM-clock
there, still DRAM-bound). The rate a reference SM-clock carries to Turing boards of other SM counts and
buses whose held clock is the same share of their reference (the 2080 Ti's 1.15; a power-capped board
such as the T4 holds less: its KV priced cheap); one card: transferred. That the SMs and not one row's
CTAs bound it is read on one stream only.

**Check.** The 9 Turing long rows -12 .. +9 % before, -4 .. +2 % after; Turing, Ada, Blackwell and
Ampere's 63 long rows rms 3.0 %. Turing's 12 short-context rows keep their stated rms (the KV term is
small there). speed_measured.json now names Qwen3.5 9B's and Gemma 4 12B's head geometry (as
choice_model_of does): its one-stream rows rms 4.73 -> 4.71 % (prefill 6.18 -> 5.67 %, DECISIONS 68's
tiles now read there), its dense steps 6.03 -> 6.05 %. Open: kKvReadFactor's Turing
rows were read on Qwen at 22K against int8 taken at the DRAM rate (now 1.55 of it on a 2080 Ti): its
BF16 3.38 for Qwen is that much low; Qwen3.5's MoE family (qwen3_5_moe_text) keeps 1.0 (its kernel
not censused on Turing); Volta's kernels (the GV100's census queued).

**The RTX 4070 Ti at 82K** (31.10, the long-context decode's open rows): the server's own decode rate
(mdecl2.sh, timings.predicted_per_second) repeats them (Qwen Q4_K_M 64.9, Gemma 51.1), while the nsys
census puts its attention kernel at the estimate's KV time (+2.67 ms a step against 2.66). Its clocks
sampled through the run: power-capped at 284 W in the 82K prefill, then 2760 MHz in the decode's first
seconds and 210 .. 825 MHz under a thermal slowdown (reason 0x20) for the rest, the GPU at 57 .. 65 C
(the memory's heat, not read): the board's cooling, not a law.

**Steps at a long context** (31.10, verification of DECISIONS 71 and the rows' KV term): mbatchl.sh
steps at 1, 2, 4 and 8 rows, each row its own ~18K-token prompt and 6000 tokens (the server's 5-second
windows whose mean batch is C, the median; such windows pool the C-row phase with the ends of the later
phases, so the context, the rows' mean prompt + 3000, holds within about 3000 tokens; row counts with
fewer than 4 windows left out), Qwen3.5 9B's Q4_K_M and Q8_0 on seven cards and Gemma 4 12B's Q4_K_M on
the five of sm_89 and up, predicted before the measurement (decode_measured.json, rows marked long from
mbatchl): 72 points, rms 3.2 %. The nsys census of the same steps (mdcenb.sh, 3000 tokens, Qwen
Q4_K_M, the int8 grouped kernel per call against every row's KV bytes) reads the KV term directly: the
RTX 2080 Ti 294, 322, 307, 308 GB/s at 1, 2, 4, 8 rows (78 .. 86 % of the 376 DECISIONS 71 reads at
82K: the rate holds at every row count, its gap at 18K not a fixed per-call cost, open), the RTX 5090
1256 .. 1496 GB/s and the 5070 Ti 697 .. 760 (74 .. 88 % and 82 .. 90 % of 0.946 of their DRAM rate).
Open: the 8-row steps where the estimate is slow (the 2080 Ti's Qwen Q4_K_M -9 %, the 5070 Ti's -8 %,
the 4060 Ti's -6 %): the rest of the step, not the KV (the 2080 Ti's 8-row step less its attention
kernels 13.1 ms against the estimate's 16.8; its K-quant weights at 8 rows reach 0.74 of the DRAM rate
against kRowsEta's 0.628, read 28.10 on an older build: the campaign is being measured again on master
f68eca34); the RTX 3060's Qwen Q4_K_M at 8 rows +14 % (its 4 rows +1 %). Gemma 4 12B at 8 rows on the
12 GB RTX 3060 and 11 GB 2080 Ti plans an 8,192-token KV and refuses the ~18K prompts (HTTP 400); its
server then segfaulted at exit (being reproduced).

**Volta's decode attention** (31.10, DECISIONS 71's census on the Quadro GV100, mdcenv.sh with the
cuda129x nsys that records Volta's kernels): Qwen3.5 9B's int8 kernel (int8_kv_key_major_kt) adds 247 us
a call for the 159 MB the 75K tokens between 7K and 82K add, 643 GB/s, 97 % of the 660 its HBM2 with
ECC holds (kDramEtaOf): the estimate's DRAM time, no row. Gemma 4 12B's BF16 full layer adds 707 us for
153 MB, 216 GB/s (25 % of 870): kKvSmRate's Volta row, 1.66 B a reference SM-clock (it took Ada's 1.06
BF16 factor before, a third of the time): the GV100's Gemma long rows +23, +35 % at 41K, 82K -> +8,
+8 %. Open: the GV100's step without its attention, +16 .. +19 % at
7K (its decode census 12.0 ms a step against the estimate's 10.1; its launches priced at Turing's
kKernelCost): the speed-model campaign at master f68eca34 runs on it too, for its own kKernelCost and
kRowsEta rows.

**The decode campaign at master f68eca34** (31.10, the constants measured again): sm_run.sh on eight
cards (plain decode at 1 / 2 / 4 / 8 streams of Qwen3.5 9B's Q4_K_M and Q8_0 and Gemma 4 12B's Q4_K_M,
the 28.10 method; IQ4_XS and IQ4_NL on the RTX 4060 Ti and 4070 Ti) and census.sh's nsys of one step at
1 / 2 / 8 streams on six (the RTX 4070 Ti's census rows and the IQ4 ones stay those of 28.10) in place
of the 28.10 rows; fit_decode.py derives kDecodeLaunches, kKernelCost, kRowsEta and
kDecodePredictionError from them (it now reads each generation's DRAM share, kDramEtaOf). Since 28.10:
the RTX 2080 Ti 3.5 .. 9 % faster (Turing's K-quant 8-row share 0.628 -> 0.691), IQ4_NL at 8 rows +14
and +10 % on the 4060 Ti and 4070 Ti (Ada's IQ4 8-row share 0.773 -> 0.824), the rest within 3 %;
Gemma's rows on the RTX 3060 and 2080 Ti now name the KV the server picks there (BF16, not FP8). The
RTX 4070 Ti's first campaign run was 4 .. 12 % slow under its thermal slowdown (reason 0x20, as the
82K rows above): measured again on a cooled board (thchk.sh, a 120-second rest, the throttle reasons
sampled) and those rows taken. The 4090's 8-row Q4_K_M step adds 16 mul_mat_q_stream_k_fixup launches:
small kernels, not weight reads. The estimate against the new rows: decode_measured.json's 100 rms
2.43 % (from 3.06), Ada 0.0226 and Blackwell 0.0170 stated; the long-context steps 2.97 % (from 3.19);
the 200 dense steps 5.81 % (from 6.05), their worst the 2080 Ti's Q6_K at 2 rows +27 % and the 3060's
long-context Q4_K_M at 8 rows +16 % (open). The Quadro GV100's 12 points stay out of the fit and its
census out of decode_census.json: its one-stream repeats spread 9 .. 11 % and its Q8_0 runs one stream
slower than two (open: measured again before Volta gets rows).

**The big models' steps at a long context** (31.10, verification): mbatchl.sh steps (each row its own
~18K-token prompt and 6000 tokens; the context the rows' mean prompt + 3000, within about 3000 tokens;
row counts with fewer than 4 windows left out) of Qwen3.6 35B-A3B's UD-Q3_K_S and UD-Q4_K_M and Qwen3.8
27B's UD-Q4_K_XL at 1, 2, 4 and 8 rows on the RTX 5090 and 4090 (the 5090's 35B 1- and 2-row phases
under 4 windows), and the 35B's UD-Q3_K_S at 1 and 2 rows on the 5070 Ti (its pool held three; the
others refused there), each row under the KV the server picked (its capacity line: the 4090's 35B
UD-Q4_K_M K8V4 and 27B NVFP4 to hold their context, int8 elsewhere), predicted before the measurement
(speed_measured.json, rows marked long): 22 points, rms 4.8 %, -13 .. +7 %. Open: the 35B's UD-Q4_K_M
at 4 and 8 rows on the 5090, -11 and -13 % (the estimate slower; the 4090's -1, -5 %): Blackwell's
kMoeRoutedEta was read on UD-IQ3_XXS and UD-Q3_K_S only. The fixture's older rows of these two models
on the 4090: its 35B UD-Q4_K_M steps of 30.10 (mbatch.sh) ran under K8V4 too (the server's capacity
line), now so named; kMoeRoutedEta's Ada rows refitted on them (fit_moe_decode.py, which now leaves the
long rows out): 0.928 / 0.911 / 0.806 -> 0.926 / 0.909 / 0.803 at 2 / 4 / 8 rows.

## 72. Volta's decode weight reads bound by its SMs (31.10, measured)

**Decision.** A weight class whose decode routes the SMs bound below the DRAM rate has rows of
kWeightSmRate (choice_constants.h): the bytes its reads reach a SM-clock (the reference clock) at 1, 2, 4
and 8 rows. Its weights then take the longer of their time at the generation's DRAM share and at SMs x
clock x that rate, in place of kRowsEta. Rows: Volta's K-quants, 4.41 / 3.90 / 3.15 / 2.45 B a SM-clock;
a class without its own row on Volta (IQ4, the codebook formats) takes them, transferred. Volta's Q8_0
keeps kRowsEta (its own rows, 0.952 / 0.871 / 0.730 in kDramEta's units) and Volta its own kKernelCost
(a weight-reading kernel 6.37 us beyond its bytes, a small one 5.06 us). One card: transferred, an
estimate only (Volta has no kReferenceCards row).

**Why.** Volta has no int8 MMA: its K-quants decode on the SMs at every width (R8's dp4a at one row,
ggml's vector kernel at 2-4, the HMMA tiles from five). The Quadro GV100's nsys census of one row
(80 SMs, 1627 MHz reference) reads it: Qwen3.5 9B Q4_K_M's and Gemma 4 12B's weight-reading kernels
527 and 523 GB/s, 80 % of the 660 its HBM2 with ECC holds, its Q8_0's 607 (92 %); per kernel R8's
Q4_K 86 %, its Q6_K 69 % (more decode a byte). The RTX 2080 Ti's reads, with an int8 MMA, hold
83-85 % of its DRAM in every format alike. Priced by kernels the GV100's residual came out by format
(Q8_0 6.4 us a kernel, the K-quants 13.3), and the Turing rows Volta took put its K-quant steps at
+6 .. +48 % (8 rows: Qwen +32, Gemma +48), Q8_0 within -6 .. 0.

**Measurements.** The GV100 runs hot under sustained load: 82-83 C, its clock 1447 -> 1140-1300 MHz
(reason 0x20, a software thermal slowdown) and its campaign points 8-17 % slow at random. Its rows are
a cool board's (thchk.sh: sm_bench.py decode after a 120-second rest, the clocks sampled; two runs
repeat within 1 %). Volta's node-level trace (the cuda129x nsys; the system one records no kernels
there) lengthens a one-row step by 2.7-3.7 us a kernel against the untraced step (Qwen Q4_K_M, Q8_0,
Gemma: 3.70, 2.71, 3.32); the census rows are taken less their own run's delta (the census gives the
split, the untraced step the total), one row only (at 2-8 rows the traced step departs unevenly).
Volta's graph appends Qwen3.5's int8 KV in a kernel of its own a full-attention layer (8 more small
kernels at one row; the launches keep the family's count). fit_decode.py derives every Volta row
(SM_BOUND: the class's census rows give no launch cost, its rates come from the cool steps).

**Check.** The GV100's 12 short-context points rms 3.4 %, worst 5.8 % (the Turing rows: up to +48 %);
the other generations' 100 points unchanged (rms 2.4 %). Open: the long-context rows (a cool re-measure,
mdecl3.sh, queued); a rate for V100 boards transfers by SMs x reference clock; Volta's MoE: its rest
by its own linears' class (rest_weights), its routed banks at kMoeRoutedEta's Ada shares capped by the
file's class rate (the same SMs decode them; no census); the 8 extra small
kernels a Qwen step are not launches the model counts (about 40 us, 0.3 %).


## 73. The worth prices the benchmark's request mix (#507, 08.11, measured)

**Decision.** A configuration's worth is -[(20/26) ln T(short) + (6/26) ln T(long)] - 25 KLD
(choice.cpp request_seconds; choice_constants.h kMixShortRequests, kMixLongRequests, kLongPrompt).
Short is the request the worth priced alone before: a 2048-token prompt and a 512-token answer. Long
is a prompt of the asked context less the answer, up to the benchmark's 24576 tokens, and the same
answer; where the ask leaves no prompt longer than 2048 (a context of 2560 or less), the long request
is dropped and the short one stands. The asked context is the largest of the task's contexts (the
scorer prices every candidate once, for the contexts in turn). The plan's `choice.task` carries
`long_prompt` and the two weights; its `why` reads "N s on the request mix: 20 of 2048+512, 6 of
L+512 tokens"; the quant rows' `estimate.request_seconds` is the same mix for the asked context.

**Why.** On the RTX 4060 Ti with Qwen3.6-35B-A3B UD-IQ3_XXS, one stream, the choice kept BF16 KV
and an FP32 state, and the user who asked for 32K of context got the default that is 1.58 ms a step
slower there than INT8 KV and an F16 state (issue #507: KV 1.270 -> 2.611 ms and the state 0.242 ->
0.484 ms a step at 32K against 0.089 -> 0.184 ms and the same state at 2304 tokens; decode 97.8 against
84.7 tok/s at 32K, 110.6 against 106.7 at 2304). The speed model is right (on the RTX 4090 it runs
10-20 % over the measured difference at 2.3K and 22K); the request it priced is the gap: one
2048 + 512 request decodes at a depth of 2304 whatever the context asked, and there BF16 / FP32
wins by 0.0025 worth (INT8 / F16 wins by +0.018 at 8192 and +0.025 at a 32K request). The asked
context is the only statement the user makes about their requests.

**The weights.** 20 and 6 are the request counts of the project's published benchmark
(docs/performance/methodology.md, tools/bench/perf_matrix.py: twenty short chats answered with up to
512 tokens, six prompts of about 24K tokens), the mix DECISIONS 50 takes the weights of our
decisions from. A convention from the benchmark, not a fit. The mean is geometric, as the worth is
in ln T: a configuration 10 % faster on either request gains the same, whatever the request's
length.

**Not A, not B** (the study of 11 models x 7 cards x {8K, 32K, 131K} x C {1, 4}):
- A, the short request decoded at a depth of max(2304, asked / 4): not a request anyone sends (the
  prefill of 2048 and a decode at the depth of a prompt it never read). It changed one decision of the
  grid (the one C changes at 32K below), and only by the depth it was given.
- B, one request whose prompt is half the asked context (at least 2048): at 131K it prices a prompt of
  65536 for every chat. Qwen3.6-35B-A3B on the 4060 Ti at 131K dropped its DFlash drafter (decode at
  the plan's displayed depth, a quarter of the asked context, 244.6 -> 97.3 tok/s, a plain FP8 KV in its place); at C = 4 it took the larger
  quantization where the benchmark's chats do not need it (Qwen3.5-9B on the 4070 Ti at 8192: Q6_K ->
  Q8_0, decode 60.1 -> 49.6 tok/s, KLD 0.0104 -> 0.0042; on the 4060 Ti at 131K UD-Q5_K_XL -> Q8_0);
  and it moved Qwen3.5-4B / 9B at 131K from INT8 to FP8 KV, on the cost of a 65536-token prompt no
  chat sends.

**Scope.** The worth decides only where a generation has a reference card and the constants behind the
accuracy terms rest on two cards: the Ada reference card (RTX 4090) today. Every other board keeps the
family's defaults; the worth is still computed there for the estimate it shows (`request_seconds`
changes everywhere, the choice only on Ada).

**Changed defaults** (all 462 groups of the study, 26 decisions change; the rest are the same). The
4060 Ti 32K row of Qwen3.6 is the quant of the grid (UD-Q3_K_S), whose default is INT8 already;
the issue's UD-IQ3_XXS case is pinned in tests/test_choice.cpp
(prices_the_request_mix) on the Ada reference card's facts (the RTX 4090, whose decision every Ada board
takes; not on a 4060 Ti board): a 32K ask chooses KV int8 / state f16 / chunk 4096, a 2560 ask keeps
BF16 / FP32. A 4060 Ti's `--plan` for it gives the same (checked on 08.11).

| Model | Card | Context | C | Before | After |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | 4090 | 32768 | 1 | KV bf16, state f32, chunk 2048 | KV int8, state f16, chunk 4096 |
| Qwen3.6-35B-A3B | 4070 Ti | 32768 | 4 | KV int8, chunk 1024 | KV fp8, chunk 2048 (a near tie, below) |
| Qwen3.6-35B-A3B | 4060 Ti | 8192 / 32768 / 131072 | 1 | chunk 2048 | chunk 4096 |
| Qwen3.6-35B-A3B | 4060 Ti | 131072 | 4 | chunk 2048 | chunk 4096 |
| Qwen3.6-35B-A3B | 4070 Ti | 8192 | 1, 4 | chunk 2048 | chunk 4096 |
| Qwen3.6-35B-A3B | 4070 Ti | 32768 / 131072 | 1 | chunk 2048 | chunk 4096 |
| Qwen3.6-35B-A3B | 4090 | 8192 | 1, 4 | chunk 2048 | chunk 4096 |
| Qwen3.6-35B-A3B | 4090 | 32768 | 4 | chunk 2048 | chunk 4096 |
| Qwen3.6-35B-A3B | 4090 | 131072 | 1 | chunk 2048 | chunk 4096 |
| gemma-4-26B-A4B | 4060 Ti | 8192 / 32768 / 131072 | 1 | chunk 2048 | chunk 4096 |
| gemma-4-26B-A4B | 4070 Ti | 8192 / 32768 | 1 | chunk 2048 | chunk 4096 |
| gemma-4-26B-A4B | 4090 | 8192 / 32768 / 131072 | 1 | chunk 2048 | chunk 4096 |
| Nemotron-3.5-30B-A3B | 4090 | 131072 | 1 | chunk 2048 | chunk 4096 |
| gpt-oss-20b | 4060 Ti | 8192 | 1 | chunk 1024 | chunk 2048 |
| gpt-oss-20b | 4090 | 8192 | 1, 4 | chunk 1024 | chunk 2048 |

The chunk changes are the MoE chunk shares (kMoeChunkShare): a 24K prompt fills the wide chunk
that a 2048-token prompt only half fills, so the prefill of the long request picks it by speed (worth
+0.0001 .. +0.0055; a chunk has no accuracy term). Their decode at the plan's displayed depth and KLD do not
change. The KV and state changes: the 4090 at 32K, decode at the plan's displayed depth (a quarter of the asked context, 8192 here) 484.0 -> 502.2 tok/s for
KLD 0.0238 -> 0.0250 (the speed gain is worth more at lambda 25); the 4070 Ti at 32K with four
streams (UD-IQ1_M), 127.1 -> 126.8 tok/s and KLD 0.8898 -> 0.8903, is a near tie, checked on
08.11 with `--plan` (resolution 0.0226 in worth): the chunk 2048 wins by speed on the long prompt, but
INT8 KV with a 2048 chunk (and with 4096) does not hold 32768 tokens for four streams (`no_fit`, own
context 0: the wider chunk's workspace), so FP8's bytes buy the chunk. Against the pick, INT8 alone
(chunk 2048, does not fit) is +0.0165, the chunk 1024 with FP8 -0.0173 (`within_resolution`), K8V4
-0.0451; the old pick, INT8 with chunk 1024, is about -0.001 (the sum of the two single-axis gaps; not
scored on its own): inside the resolution, the point estimate decides. Dense models, the other families and every card without a reference card keep their
defaults (Qwen3-8B, Qwen3.5-4B / 9B, Qwen3.8-27B, gemma-4-31B and -12b, Muse-Glimmer: no change).

## 74. ggml's matrix kernel takes the column tile of least work where the generation row says so (#524, 08.11, measured)

**Decision.** Where the generation row has `mmq_work_tiles` (GA10x: RTX 3060, 3090; GB20x: RTX
5090) a prefill product on ggml's matrix kernel takes, among J = 16, 32, 64, 80, 96, 128, the width
whose tiles cost least, ceil(t / J) x (f + (1 - f) J / 128) of a 128-column tile's K steps, the
narrower on a tie; f is the row's `mmq_tile_fixed` (GA10x 0.20, GB20x 0.10). Every other generation
keeps J = 16, 32, 64, 128 and the fewest tiles. The kernel's `select_launch`, the estimate's
`mmq_linears_seconds` and cost_route's MMQ term (generic_cost.cpp, which priced continuous 8-column
steps: 144 tokens as two 128-column tiles) read the one table (core/mmq_columns.h). J = 112 is not
instantiated.

**Why.** A tile is 128 weight rows x J tokens over the whole K; the rows' decode is the same for every
J and the MMA, the y tile's loads and its share of the epilogue are in J, so a tile costs a fixed
part plus a part in J. A slice of 144 columns on two 128-column tiles pays for 256; two 80-column
tiles pay f + 0.625 (1 - f) of each. The part in J is no fixed fraction of a card: the int8 MMA alone
is 4096 of a GA10x Q4_K tile's 10600 SM-clocks a 256-value K step (1024 MMAs of 4096 MACs at 1024
MACs a clock), a generation with a higher MMA rate (ga100) has a smaller part in J and a larger
fixed share, so the share is measured per generation on its cards, not carried: the forced-J grids
(8 widths x 21 slice sizes, N = 16..512, Qwen3-8B Q4_K_M, `--no-prefix-reuse`, four requests a
point) give the regret of the model's picks against the best width of the set as a function of f:
RTX 3060 0.00 % at 0.25-0.30 (0.21 % at 0.15-0.20), RTX 3090 0.00 % at 0.15-0.20 (0.28 % at
0.25-0.30), RTX 5090 0.06 % at 0.10 (0.31 % at 0.05, 0.45 % at 0.15-0.20); 0.43 (the RTX 4060 Ti's
point, DECISIONS 69) costs 0.28 %, 0.70 %, 1.0 %. GA10x takes 0.20 (the two cards' common best
range), GB20x 0.10 (one card: the RTX 5070 Ti was busy). Ada (RTX 4090, 4060 Ti) and Turing
(RTX 2080 Ti) were not run over the widths (their stands were loaded by other builds): they keep the
fewest tiles, a row joins by its own grid.

**Measured** (A B A B, three rounds of four requests, the same binary with `old` and `new` tables):
RTX 5090 (f 0.10) 144..160-token prompts 25.1..25.3 -> 20.9..21.0 ms (-17 %), 136 -13 %, 176..192
-11 %, 264..288 -13 %, 320 -8 %, 96 -8 %; 200..224 -2 %, 256, 384, 512 unchanged; the 136..256 mix
25.3 -> 23.2 ms (-8.3 %). RTX 3090 (f 0.20, the final table): 144..160 59.8..59.9 -> 48.5..48.6 ms
(-18.8 %), 136 -15.8 %, 176..192 -12.5..-12.9 %, 264..288 -13.3 %, 320 -5.6 %, 96 -10.5 %, 448
-2.9 %; 128, 200..256, 384, 512 unchanged; the 136..256 mix 60.4 -> 55.1 ms (-8.7 %). RTX 3060
(against master, f 0.43 and without J = 112, whose picks differ from f 0.20 only at 289..320 and
448): 136..160 tokens 141.9..143.0 -> 114.4..115.6 ms (-19 %), 168..192 -14..-12.6 %, 264 / 288
-11 / -13.6 %, mix 145.1 -> 133.2 ms (-8.2 %, predicted -7.8 % before the measurement).

**Widths.** 80 and 96 carry the gain. J = 112 was 6 % slower than 128 for one tile of 97..112 columns
on the RTX 3060 (99.8 against 93.8 ms) while the model priced it cheaper, and the 5090's grid shows
it no better than 128 for 97..128 columns: not instantiated. 48 adds 0.1 %. The 80- and 96-column
kernels are compiled only for the architectures whose row takes them (`work_tiles_arch` in
ggml_mmq_internal.cuh, checked against the table by a static_assert; the skip is the kernel's own
unused-specialization exit in mmq.cuh), so Ada, Turing, Hopper and the others keep their code size.
A width costs about 0.2 MB and 1.1 s of compile a format and architecture it is built for.

**Numerics.** A product's sums depend on the column tile count (stream-k splits the K steps
differently), as they already did between a prefill chunk of 128 and 1024: on natural text the mean
|logprob| difference old vs new is 0.000-0.03, old build's chunks 128 vs 1024 0.003-0.026.

**Not done.** Ada, Turing, Volta and Thor rows (own grids); the A10; other formats than Q4_K
(the Q4_0 and IQ decodes give other fixed shares); the RTX 5070 Ti and 5060 Ti prefill Q4_K / Q6_K
on the W4A8 tiles, not this kernel, which steps the same way (RTX 5060 Ti 104 -> 107 ms from 144 to
256 tokens): its 64 x 64 against 128 x 64 tile steps need their own grid.
