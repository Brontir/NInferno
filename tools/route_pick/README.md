# route_pick: the analyzer's predicted route bands

`gen.py` writes `routes/pick/<chip>-<sms>sm.json` for every board class of `src/core/data/gpu_facts.json` whose chip
has no route table in `routes/` and whose board is past `kCarryDistance` from every table's board (nearer, the table's
answers carry and the engine reads no file: the RTX 4070 Ti SUPER, the RTX 4060 and the RTX 5070 today), and whose SM
count lies between the fewest and the most SMs of the same compute capability's table boards: the law interpolates
between measured boards and is not extrapolated past them. Five classes today: AD103 (76, 80 SMs) and GA104 (38, 46,
48); AD107, GA107 (10-22 SMs) and Turing (one table board, 68 SMs) get none. Each file holds the T 9-16 band of every A16 generic-path problem with the store or
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
- `gen.py` writes the files. `gen.py --check` exits 1 when a file differs from what the facts give now; it runs as
  the `ninferno_route_pick_fresh_test` ctest. After a kernel-facts, gpu-facts or route-table change, run
  `python3 tools/route_pick/gen.py` and commit `routes/pick`.

## Where the engine reads them
`src/ops/route_table.cpp` reads `routes/pick/<class>.json` only when the device's nearest measured table is past
`measured::kCarryDistance`. There, every table answer would be Transferred or Borrowed, and `--route-policy evidence`
would demote it to the ranker. The predicted band answers first, as evidence kind `predicted`. Such an answer is
never demoted and never claims a shape-owned problem. An explicit `--route-table` stands alone and reads no overlay,
and `--route-policy legacy` reads none either (the lookup answers as before the overlay).

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
So measured tables stay, and the pick serves only the classes nobody measured.

Left out (constants fitted without the card): about 2.5 % pooled regret on the open grids. A whole generation left
out of the fit costs at most 0.35 more.
