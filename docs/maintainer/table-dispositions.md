# Disposition of `a8_default_route` and the sparse MoE frontier (#616 step 3)

Two answers of the MoE ops rest on a table of boards where something was measured, carried to other
boards by `board_distance`. The rule of this project is that a decision comes from the card's
published facts and the kernels' counted facts, and a measurement only verifies it. So each table
gets a disposition: **derived** from facts, or **declared** (a literal with its provenance, labelled
as such in every answer). This note records the disposition of each and why. No choice changes: a
derivation that would change one stays a proposal until it is predicted and then verified.

## The mechanism (code)

`measured::table_answer` (`core/measured_boards.h`) is the one lookup of a measured value table by
board. It answers in the words the measured gates use (`measured::Disposition`,
`core/measured_disposition.h`, shared with `GateAnswer`):

- `Measured`: the board's facts are a table row's (distance 0, every quantity known). It does not say
  that this very SKU was run: an RTX 5090 D has the 5090's SMs, L2 and bandwidth and takes the 5090's
  row, though only the 5090 was measured.
- `Carried`: a row within `measured::kCarryDistance`, every compared quantity known. The reach (1.0)
  is a threshold placed between measurements, not a chip fact: it is meant to cover the variants
  and near neighbours (0.14 to 0.57), and the RTX 4090's R8 decode entries fail on the RTX 4080
  SUPER at 1.2. Over the 105 rows below every answer is the same for any reach from 0.83 to 1.21
  (the A10 at 0.83 from the 3090 is the farthest `Carried` row, the RTX 4080 SUPER at 1.22 from the
  4090 the nearest row out of reach).
- `Incomplete`: no row with every quantity known is within the distance, but the nearest row by plain
  distance is, with its L2 or bandwidth unknown on either side. The distance could then understate
  the difference, so the table's literal answers (as a bounded gate takes the unmeasured default for
  an Incomplete donor), and the label says the facts could not tell.
- `None`: no row is within the distance; the table's declared literal, no donor.

It selects the donor with `nearest_known_measured`, exactly as the lookups did before, so a
disposition cannot disagree with the value. Every row of both tables knows its facts (each table
asserts `measured_boards_known`), so an unknown is the board's own: no row is complete for it, and
the answer is `Incomplete` or `None`. Only a table with an incomplete row could answer `Measured` or
`Carried` from a complete row while a nearer row is `Incomplete`.

- `moe::a8_route_answer(Board)` (`ops/moe/moe_internal.h`): the class A8 route with its disposition.
  `a8_default_route()` is that answer on the running device, and `class_a8_route_evidence()` reads
  the same answer's disposition, distance and donor, so the route and its evidence cannot differ.
- `sparse_moe_q4_groupwise_frontier_answer(Board)` (`ops/sparse_moe/prefill`): the first T of the
  grouped prefill route for Q4/Q5 and Q4/Q6 groupwise banks, with its disposition.
- `tests/ops/test_table_dispositions.cpp` checks both against the production lookups, ties the
  route's evidence to the answer, checks fourteen rows against answers written by hand from the
  measurement comments beside the tables (with the donor where the comment names one), checks a
  board at exactly the reach (`Carried`) and one with its bandwidth unknown (`Incomplete`), checks
  the counts below, and pins the disposition of every
  target SKU of `core/data/gpu_facts.json` (keyed by PCI id) in
  `tests/ops/table_dispositions/dispositions.txt` (host only; `--update` rewrites it from the code
  under test, so that file pins behaviour, it is not an independent expectation).

**What the counts below are over.** They are over the published-facts table: each SKU's board is
built from its `gpu_facts.json` row (`measured::table_board`), whose L2 is the row's `l2_mib`. Six
rows of the table (five SKUs) are `Incomplete` in one table or both, because the row has no
`l2_mib` and a measured row is near: A10G and GeForce RTX 3080 12GB (frontier), L40S and RTX PRO 6000
Blackwell Server Edition (both tables), and the two RTX 5080 Laptop GPU rows (A8 route). No `l2_mib`
is invented for them. A device reports its L2 through the driver, so on hardware those SKUs would be
`Carried` (or `None`) by their real distance, not `Incomplete`. The table is not a statement about
what a device of that SKU gets.

Counts over the 105 target SKUs (measured / carried / incomplete / none):
A8 route 4 / 8 / 4 / 89; Q4 frontier 4 / 13 / 4 / 84.

## `a8_default_route`

**What it decides.** For a gate/up bank of a published 2-bit codebook format (IQ2_XXS, IQ2_XS,
IQ2_S), which R8 schedule (w4r1, w4r2, w8r1, w2r1) takes each T = 1..8, and which register-MMA
schedule an IQ3_XXS down of width 512..1023 takes. Every other R8 bank takes w4r1 at every T.

**What can be derived.** One part is a rule from the kernel's bytes: two rows per warp (w4r2) read
each quantized activation once for both rows, so it pays only where a row's weights are the fewest
bytes per value and the activation reads are the largest share. That is why only the 2-bit formats
have a row and every other format takes w4r1. It is already fact-based, and it stays.

**What cannot be derived now.**
- The per-T edge (w4r1 to w4r2 for IQ2_XXS at T = 4 on the RTX 5090 and T = 3 on the 5080, for
  IQ2_XS at T = 8 and T = 2; w2r1 for IQ2_S at T = 4..7 on the 5080)
  is a 2-9 % difference between schedules (a 2 us timer step at the small calls). The cost model
  prices no grouped stage yet, so there is no predicted schedule to compare the measurements with; a
  rule fitted to three boards' sweeps would only restate them.
- The down schedule (2 x 2 on the RTX 4090, 4 x 1 on the 5080) is a 1-2 % effect at Qwen3.6 decode.

**Disposition: declared.**
- Rows (RTX 4090, 5090, 5080): measured with `bench/ops/moe_bench.cu --a8 1 --plans sweep`, the
  numbers in `moe.cpp` beside the table. `Measured` on those boards; `Carried` within distance 1.0
  (RTX PRO 6000 Blackwell 0.35 from the 5090, RTX 5070 Ti 0.57 and the laptop 5090 0.13 from the
  5080, RTX 4090 D, 6000 Ada and L40 at 0.17 / 0.43 / 0.58 from the 4090).
- Fallback literal (`kFallbackA8Route`): w4r1 on gate/up at every T, the register MMA on the down.
  Provenance: the load-time tuner's choices on eight cards (A100, RTX 2080 Ti, 3090, 4060 Ti, 5060
  Ti, V100, 4090, 5090) put an R8 schedule on gate/up at T = 1..4 for nearly every A8 layer, mostly
  w4r1, where the register MMA lost by x1.15..1.36 (V100 x3.5..4). On the one card where the schedule
  was measured against it (RTX 4060 Ti) every schedule is within 1..3 % of w4r1: the literal's cost,
  where measured, is at most that.
- The literal answers 93 of 105 rows (`None` 89, `Incomplete` 4): every Ampere, Turing and Volta card
  (no row of those instruction sets), every Ada card outside distance 1.0 of the 4090 (RTX 4080 and
  SUPER, the 4070 and 4060 families, L4, RTX 2000..4500 Ada, the Ada laptop parts; the L40S is
  `Incomplete`), and the consumer Blackwell cards outside reach of the 5090 and 5080: the RTX 5070,
  5060 Ti and 5060 (`None`), the RTX PRO 6000 Blackwell Server Edition and the two 5080 Laptop rows
  (`Incomplete`: no L2 in the facts table). Hopper and datacenter Blackwell (H100, H200, H800, B200)
  are not among the 105 target SKUs. That is the table's real coverage; it is labelled, not hidden.

**What would move it.** A priced grouped stage in the cost model (a replica against the sweeps on
disk) would turn the literal from a constant into a ranked answer if its residual is below the 3-9 %
to be decided, and a sweep on an Ampere card would fill the Ampere row from measurement.

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
Fallback literal: 47, the Blackwell/Ada edge. The literal answers 88 of 105 rows (`None` 84,
`Incomplete` 4: A10G, RTX 3080 12GB, L40S, RTX PRO 6000 Server).

**Open risk.** Ampere cards outside the 3090's reach take the 47 literal: the RTX 3060, 3060 Ti,
3070, 3070 Ti, A4000, 3050, A2 and the two RTX 3060 Laptop GPU rows (14 rows, with their LHR and 8GB
/ 6GB rows). So do the cards of the generations with no row at all: the 8 GA100 rows (A100, A30,
A800), the 23 Turing rows and the 7 Volta rows. (The A10G and the RTX 3080 12GB are `Incomplete` in
the facts table only because it has no L2 for them; a device reports its own L2, so they may well be
`Carried` from the 3090 at their real distance.) The only Ampere measurement (3090) lost 28 % at C4
with 47 against 17, so the step may carry to the whole generation. A C4 sweep at frontier 47 / 25 /
17 on an RTX 3060 would show whether it does; if it does, a generation-keyed rule (a frontier of 17
for a board without Ada's tensor-core rate) would turn those SKUs into a measured-and-labelled
answer. That is a behaviour change and needs the GPU run; nothing here changes the default.

## Verified and not verified

The host test and the production-equality checks run without a GPU. The production-equality checks
cover the sixteen boards the selector goldens are kept for (`generic::kFactsBoards`, each under
`ScopedDeviceClass`), not the 105 rows. The tie rule of the lookup (two rows at equal distance: the
earlier answers) is pinned by a local two-row case in the test, not by any row. Not verified: the
dispositions' meaning on a GPU (no kernel runs in the test), and the open Ampere frontier risk above
(it needs the sweep on an Ampere card).
