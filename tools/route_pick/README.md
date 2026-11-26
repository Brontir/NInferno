# route_pick: the analyzer's predicted route bands

`gen.py` writes `routes/pick/<chip>-<sms>sm.json` for every board class of `src/core/data/gpu_facts.json` whose chip
has no route table in `routes/` and whose board is past `kCarryDistance` from every table's board (nearer, the table's
answers carry and the engine reads no file: the RTX 4070 Ti SUPER, the RTX 4060 and the RTX 5070 today), and whose SM
count lies between the fewest and the most SMs of the same compute capability's table boards: the law interpolates
between measured boards and is not extrapolated past them. Five classes today: AD103 (76, 80 SMs) and GA104 (38, 46,
48); AD107, GA107 (10-22 SMs) and Turing (one table board, 68 SMs) get none. Each file holds the T 9-16 band (T 9-32 on sm_89, below) of every A16 generic-path problem with the store or
SwiGLU epilogue that the route tables of the same compute capability list (the release models' shapes), as `pick.py`
prices it from repo facts. The add epilogue (o_proj, down) is priced but not written: on the RTX 4060 Ti its bands lost
2.8 % against the borrowed table, the law's n2.w1x4 against the measured n4.w1x4 on the Q6_K down (#1051).
Nothing is measured on the card.

- `pick.py` is the route analyzer: analyzer core after round 11, with its constants frozen. It prices every route
  that has a kernel-facts row and takes the cheapest. An rmma schedule is priced as rounds of resident CTAs; ggml_mmq,
  r8 and ggml_vec as bus time plus the kernel's stall clocks outside the K loop. Its docstring lists every term and its
  inputs.
- `facts.py` reads the inputs: `gpu_facts.json`, `isa_facts.json` (isa_probe), `sm_rate_facts.json`, and the
  kernel facts `sm_<arch>.txt`.
- `analyzer.py` (with `analyzer_facts.py`) is the analyzer of T 1-8, rounds 15-20: the same inputs plus the launch,
  DRAM-latency and narrow-load facts, one price per route of ggml_vec, r8, ggml_mmq and every rmma schedule, and on
  sm_70 the Volta substitutes once their kernel facts (sections vhmma / vgemv / vgemvr of `sm_70.txt`) are committed;
  until then those routes have no price there. `rank()` lists the priced routes cheapest first; `gen.py` writes its
  lists (T 1-8, below), which the engine answers those widths by.
  `tests/test_route_analyzer.py` (ctest `ninferno_route_analyzer_test`) checks that every A16 release shape at
  T 1-8 has a priced route on sm_75, sm_86, sm_89 and sm_120a.
- `gen.py` writes the files. `gen.py --check` exits 1 when a file differs from what the facts give now; it runs as
  the `ninferno_route_pick_fresh_test` ctest. After a kernel-facts, gpu-facts or route-table change, run
  `python3 tools/route_pick/gen.py` and commit `routes/pick`.

## Where the engine reads them
`src/ops/route_table.cpp` reads `routes/pick/<class>.json` on the board it was priced for. Where the device's nearest
measured table is past `measured::kCarryDistance`, every table answer would be Transferred or Borrowed, and
`--route-policy evidence` would demote it to the ranker; there the predicted band answers first, as evidence kind
`predicted`. Within it (a tabled board, #1068) a band answers only in a gap of the table: a problem it holds no entry
of its own for, which would be Borrowed from another board's table or left to the ranker. A tabled board's file holds
only those gaps (the problems of its capability's other tables, the add epilogue included), and none is written where
there are none (the RTX 2080 Ti, the only sm_75 table). Such an answer is
never demoted and never claims a shape-owned problem. An explicit `--route-table` stands alone and reads no overlay,
and `--route-policy legacy` reads none either (the lookup answers as before the overlay).

## T 1-8: the analyzer inside the engine
Decode and verify calls (T 1-8) are priced by the engine itself: `src/ops/generic/route_analyzer.cpp` is `analyzer.py`
in C++, term for term, over the compiled-in facts (machine facts, arch limits, isa / sm_rate / launch facts, the build's
kernel facts). The select list's authority `analyzer` (route_authorities.cpp `analyzer_entry`, after the route probe,
the pins and Volta's fixed routes) asks `route_analyzer::device_routes` at the call: every route it prices for the
call, cheapest first, cut after the first register-MMA schedule, memoized per board, kernel snapshot and call. It
takes the first route its policy (exact activations refuse the int8 routes ggml_vec, r8, ggml_mmq), profile and
build admit, as evidence `predicted`, ahead of the route tables, the dense tuner and the rules. Any shape the
kernels take is priced, a model no table lists included; nothing is read from `routes/pick` at these widths. A
prefill call and a list with no admitted route fall through to the older authorities.

`route_analyzer::kHeldChips` lists the chips it does not answer yet (the switch's gate: no card below -1 % end to
end against master): AD102 (rmma priced 12-16 % below its time at T 2-4, #1211), AD106 / GA106 (w2x2 priced 8-19 %
above its time, w1x8 1-18 % below, the wave term of #1224; Qwen3.5-9B Q4_K_M at C 5-8 on the RTX 3060 -1.1 to
-2.1 %, on the RTX 4060 Ti -0.7 to -1.1 %) and TU102 / TU104 / TU106 (the activation-ring latency kernel sm_75
launches is priced 1.2-1.3x its time, #1183). Those price errors are of the analyzer before the issue bound joined
every rmma price (#1235). The end-to-end checks of 26.11 at master 27c115aac, Qwen3.5-9B Q4_K_M, analyzer against the
held rules:

| chip, card | C 1 | C 2 | C 4 | C 6 | C 8 |
|---|---|---|---|---|---|
| AD102, RTX 4090 | -2.2 % | -1.5 % | -2.9 % | -0.2 % | -0.1 % |
| AD106, RTX 4060 Ti | -1.1 % | -0.3 % | +0.5 % | -1.0 % | -0.7 % |

On the 4060 Ti the result is the same with the issue bound.

`ninferno-serve route-analyzer` prices cells of facts-table boards on a CPU (the lab's exams). `analyzer.py` stays
the lab's copy: `ninferno_route_analyzer_parity_test` holds the two equal (the same candidates, order and prices
within 1e-9) on one board of every chip, every A16 table shape and a new model's shapes.

## T 17-32 on sm_89 (#785)
On compute capability 8.9 (`NARROW_CC`) every class's file also holds the bands of T 17-32 (`NARROW_HI`), on tabled
boards too, for every problem it prices there, also where the table covers T 9-16 and the entry then holds the 17-32
bands alone. Past T 16 the bands stop at the first width `pick.py` cannot price. These bands answer a different set of calls than the T 9-16 ones: the int8 calls of T 17-32 outside the prefill
profile (`a8_wide_call`: a speculative verify round, a decode batch), which skip the table, and whose measured entries
hold A16 schedules only. For such a call (`route_table_entry`, `table_predicted_answer`) the band answers ahead of the
cost model, *whether or not the table holds an entry of its own*. It yields only to this board's own tuned W4A8 band of
the width (`dense_tune_prefill_band`, `auto_answer`). A band the call does not admit leaves it to the cost model, as
before. A16 calls of T 17-32 read these bands like the T 9-16 ones: in the table's gaps on a board within
`kCarryDistance`, ahead of the table past it (the speed profile; no predicted band is read at prefill from T 17).

Evidence (open grids, T 24 / 32, the cells the cost-model authority answers): pick.py's mean regret against the timed
routes is +2.0 % on the RTX 4070 Ti and +4.1 % on the 4090, where the cost model's is +27.1 / +19.9 %. On the 3090 it
is 5.1 % against 16.1 %, but the 3060 had no such cell to score, so cc 8.6 stays out. On the 5090 the pick is not
better (+5.7 % against +3.9 %), so cc 12.0 stays out. End to end on the RTX 4090 (ninferno-serve, MTP draft 3,
Qwen3.5-9B Q4_K_M): +4.8 % at 6 lanes (verify width 24), +0.8 % at 8 (width 32, noise level).

## Evidence (analyzer_r14, 19.11)
The test removed a card's own table, as for a card of the generation that nobody measured. Qwen3-8B prefill at chunk
16 and 12, where every linear call runs at T 16 or 12, compared the pick's bands with the borrowed path:

| card, model | pick against borrowed |
|---|---|
| RTX 3090, Q4_K_M | +5.5 % |
| RTX 3090, Q5_K_M | +0.3 / +0.4 % |
| RTX 4060 Ti, Q4_K_M (store and SwiGLU bands, a1e7bc095) | +0.25 / -0.46 % |
| RTX 4060 Ti, with add bands (8f339ae73) | -2.8 / -2.6 % |
| RTX 2080 Ti | -0.02 / -0.02 % (ties) |

Against a card's own measured table, the pick equals it on the 3090 and the 2080 Ti and loses 2.4-2.6 % on the 4060 Ti.
So measured tables stay: the pick serves the classes nobody measured, and a tabled board's gaps.

## Evidence for the gaps (#1068)
Each tabled card's table was left out and both answers of every cell scored on a full route_tune grid (T 9-16):

| card (generation) | borrowed / ranker: mean, p90 regret | predicted: mean, p90 |
|---|---|---|
| RTX 3090 (GA10x) | 8.46 %, 28.4 % | 5.67 %, 25.1 % |
| RTX 3060 (GA10x) | 8.95 %, 27.6 % | 5.52 %, 21.4 % |
| RTX 4060 Ti (Ada) | 5.10 %, 17.5 % | 4.70 %, 13.9 % |
| RTX 5090 (GB20x) | 13.46 %, 43.8 % | 8.81 %, 31.5 % |

End to end (Qwen3-8B prefill, gap-only against shipped): the 4060 Ti, 5070 Ti and 4090 within their A/A bands.

Left out (constants fitted without the card): about 2.5 % pooled regret on the open grids. A whole generation left
out of the fit costs at most 0.35 more.
