# Disposition of `a8_default_route` and the sparse MoE frontier (#616 step 3)

Two answers of the MoE ops rest on a table of boards where something was measured, carried to other
boards by `board_distance`. The rule of this project is that a decision comes from the card's
published facts and the kernels' counted facts, and a measurement only verifies it. So each table
gets a disposition: **derived** from facts, or **declared** (a literal with its provenance, labelled
as such in every answer). This note records the disposition of each and why. No choice changes: a
derivation that would change one stays a proposal until it is predicted and then verified.

## The mechanism (code)

`measured::table_answer` (`core/measured_boards.h`) is the one lookup of a measured value table by
board: `Own` (the board is a table row, distance 0), `Carried` (a row within
`measured::kCarryDistance`, so every compared quantity is near), `Fallback` (no row is near: the
table's declared literal, no donor). It selects the donor with `nearest_known_measured`, exactly as
the lookups did before, so a disposition cannot disagree with the value.

- `moe::a8_route_answer(Board)` (`ops/moe/moe_internal.h`): the class A8 route with its disposition;
  `a8_default_route()` is the same lookup on the running device.
- `sparse_moe_q4_groupwise_frontier_answer(Board)` (`ops/sparse_moe/prefill`): the first T of the
  grouped prefill route for Q4/Q5 and Q4/Q6 groupwise banks, with its disposition.
- `tests/ops/test_table_dispositions.cpp` checks both against the production lookups on the selector
  boards and pins the disposition of every target SKU of `core/data/gpu_facts.json` in
  `tests/ops/table_dispositions/dispositions.txt` (host only; `--update` rewrites it).

Counts over the 105 target SKUs: A8 route Own 4 / Carried 8 / Fallback 93; Q4 frontier Own 4 /
Carried 13 / Fallback 88.

## `a8_default_route`

**What it decides.** For a gate/up bank of a published 2-bit codebook format (IQ2_XXS, IQ2_XS,
IQ2_S), which R8 schedule (w4r1, w4r2, w8r1, w2r1) takes each T = 1..8, and which register-MMA
schedule an IQ3_XXS down of width 512..1023 takes. Every other R8 bank takes w4r1 at every T.

**What can be derived.** One part is a rule from the kernel's bytes: two rows per warp (w4r2) read
each quantized activation once for both rows, so it pays only where a row's weights are the fewest
bytes per value and the activation reads are the largest share. That is why only the 2-bit formats
have a row and every other format takes w4r1. It is already fact-based, and it stays.

**What cannot be derived now.**
- The per-T edge (w4r1 to w4r2 at T = 3 or 4 on the RTX 5090 and 5080, w2r1 at T = 4..7 on the 5080)
  is a 2-9 % difference between schedules (a 2 us timer step at the small calls). The cost model
  prices no grouped stage (step 4 builds `price_moe_stage`), and the laws tried for R8 against the
  register MMA (Little's law, registers, bursts, TLB, instruction count) each ranked one card wrong.
  The best predictor has rms 9-13 % on the RTX 4090 and 11-22 % on the RTX 3060, larger than the
  difference to be decided. A rule with that resolution would only restate three boards.
- The down schedule (2 x 2 on the RTX 4090, 4 x 1 on the 5080) is a 1-2 % effect at Qwen3.6 decode.

**Disposition: declared.**
- Rows (RTX 4090, 5090, 5080): measured with `bench/ops/moe_bench.cu --a8 1 --plans sweep`, the
  numbers in `moe.cpp` beside the table. Disposition `Own` on those boards; `Carried` within distance
  1.0 (RTX PRO 6000 Blackwell 0.35 from the 5090, RTX 5070 Ti 0.57 and the laptop 5090 0.13 from the
  5080, RTX 4090 D, 6000 Ada and L40 at 0.17 / 0.43 / 0.58 from the 4090).
- Fallback literal (`kFallbackA8Route`): w4r1 on gate/up at every T, the register MMA on the down.
  Provenance: the load-time tuner's choices on eight cards (A100, RTX 2080 Ti, 3090, 4060 Ti, 5060
  Ti, V100, 4090, 5090) put an R8 schedule on gate/up at T = 1..4 for nearly every A8 layer, mostly
  w4r1, where the register MMA lost by x1.15..1.36 (V100 x3.5..4). On the one card where the schedule
  was measured against it (RTX 4060 Ti) every schedule is within 1..3 % of w4r1: the literal's cost,
  where measured, is at most that.
- 93 of 105 SKUs take the Fallback: every Ampere, Turing, Volta, Hopper, datacenter Blackwell, and
  Ada card outside distance 1.0 of the 4090. That is the table's real coverage; it is labelled, not
  hidden.

**What would move it.** Step 4 (a priced grouped stage with a replica against the sweeps on disk)
turns the Fallback from a literal into a ranked answer if its residual is below the 3-9 % to be
decided; step 5's Ampere sweep fills the Ampere row from measurement.

## The sparse MoE Q4/Q5 groupwise frontier

**What it decides.** The first T from which a Q4/Q5 or Q4/Q6 groupwise MoE layer leaves the per-token
small-T kernels for the grouped prefill kernels in the speed profile: 47 on RTX 5090 and 4090, 17 on
the RTX 3090. (Under the prefill route profile the frontier is 17 anyway; the invariant profile
takes the prefill kernels at every T.)

**What can be derived.** The frontier is a crossover of two kernels' costs: the small-T kernel's
per-token weight re-reads against the grouped kernel's setup, scaled by the card's bandwidth and
compute. Deriving it needs both kernels' priced terms; no cost model for the sparse MoE kernels
exists yet.

**What cannot be derived now.** The only candidate fact is the card's ridge T* (int8 ops per byte,
`docs/maintainer/mmq-transfer-audit.md`): GA10x 80-86, Ada 172-227, GB20x 110-132. It orders the 3090
below the other two but not the 4090 (184) against the 5090 (132), which share the value 47. The data
are three points of a two-valued step (Qwen3.6-35B DFlash7 C4 tok/s at frontier 47 / 25 / 17 / 13:
5090 843 / 812 / 815 / 802, 4090 636 / 536 / 520 / 528, 3090 292 / 389 / 403 / 411); any rule with two
parameters fits them exactly, which is a fit, not a derivation.

**Disposition: declared.** Rows RTX 5090, 4090 (`kSparseMoePrefillQ4Q5Min` = 47) and RTX 3090 (17),
measured ABBA as above; `Carried` within 1.0 (RTX PRO 6000 0.35, 4090 D 0.17, 6000 Ada 0.43, L40
0.58; 3090 Ti 0.14, 3080 0.70, 3080 Ti 0.07, A6000 0.32, A5000 0.64, A40 0.46, A10 0.83).
Fallback literal: 47, the Blackwell/Ada edge.

**Open risk, found by this disposition.** 88 of 105 SKUs take the 47 literal, among them Ampere cards
the table does not reach: the RTX 3060, 3060 Ti, 3070, 3070 Ti, A4000, A10G, 3050, A2 and the 3080
12 GB. The only Ampere measurement (3090) lost 28 % at C4 with
47 against 17. The 3060 is a lab card, so this is checkable: a C4 sweep at frontier 47 / 25 / 17 on
the 3060 would show whether the Ampere step carries to the whole generation. If it does, a
generation-keyed rule (a frontier of 17 for a board without Ada's tensor-core rate) would turn those
Ampere SKUs from Fallback to a measured-and-labelled answer. That is a behaviour change and needs the
GPU run; nothing here changes the default. It is recorded as the follow-up, not applied.

## Verified and not verified

Host test and the production-equality checks pass on the base build. Not verified: the dispositions'
meaning on a GPU (no kernel runs here), and the open Ampere frontier risk above (needs the 3060 sweep).
