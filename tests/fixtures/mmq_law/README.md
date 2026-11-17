# The counted MMQ column-tile law: frozen predictions (#613 PR-5, step 5b)

`predicted_sm_<arch>.csv` is written by `tools/dev/mmq_law.py` from the committed kernel-facts
snapshots only (the mmq section: one counted iteration per column tile J). Per ggml type it gives
the tile step's fixed part F and per-column part V (least squares over J = 16, 32, 64, 128), once
over a warp's instructions (issue) and once over its stall sum, the fixed share
f = F / (F + 128 V), and the per-width ratio T(J) / T(128) = f + (1 - f) J / 128.

These files were committed before any grid that scores them was read: the sealed `pr5grid` runs
(route_tune over k-quant A8 shapes, three passes, RTX 4070 Ti, 4090, 3060, 5070 Ti, master
7cdb22d0) were started on 2026-11-17 and stay unread until this commit is an ancestor of the
scoring branch. A file changed after that commit no longer counts as a prediction.

## The chain law (step 5c, #613 comments of 2026-11-17)

`waits_sm_<arch>.csv` is written by `tools/dev/mmq_waits.py` from the SASS of a master build
(c6a1b7d50) of `ninferno_mmq_launch_microbench`: per mul_mat_q kernel (ggml_prec 30), the counted
iteration's global loads, its scoreboard waits on global loads, on shared loads and on the rest,
its stall sum (equal to the snapshot's `stall_sum`) and its instructions.

`predicted_chain_<card>.csv` is written by `tools/dev/mmq_chain_law.py` from those counts, the
board's `gpu_facts.json` row and the arch's isa_probe facts, plus one per-card fact measured
before any of these predictions: the L2's bytes per SM clock per SM, the floor of a coalesced
32-bit `ld.cg` stream over an L2-resident buffer, one CTA of 32 warps on every SM, 8 loads in flight a warp, from a scratch microbench of the same loop isa_probe 18 now runs at 16 warps (RTX 4090 15.2, RTX 4060 Ti
17.2, RTX 3090 16.2, RTX 3060 17.75, RTX 5070 Ti 18.1; waits_sm_120a.csv from a sm_120a build of the same commit). It predicts the launch law's unit (one
256-K iteration on the `law` grid) as the load phase (the weight tiles over DRAM, every tile and
activation tile through the L2, the larger of the two, plus one DRAM latency) followed by the
warp's chain (the stall sum plus one shared latency per wait on a shared load). The cards' launch
law runs (`mmq_launch_law.py derive`, master c6a1b7d50) were made before this commit and are
read only after it.

## The production-launch prediction (sealed grids, #613 PR-5)

`predicted_launch_rtx4070ti.csv` and `predicted_launch_rtx5070ti.csv` are the selector's ggml MMQ
price with the counted unit on and off (`counted_us`, `fitted_us`; the A8 policy), from the price
tool of eecbcfd23 built host-only, for the k-quant A8 shapes of the sealed route_tune grids
(`fmt epilogue n k`) at T = 17, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024 and 2048. Only
Q4_K, Q5_K and Q6_K have a counted price (`mmq_unit::scored`); the other formats' `counted_us` is
the fitted price. These grids (route_tune of master, three passes, on the RTX 4070 Ti and the RTX
5070 Ti) ran before this file and are read only after it is committed; the RTX 3060 and RTX 4090
grids stay sealed for the end-to-end A/B.

## The Turing grid prediction (#607 forced-width grid, #613 PR-5)

`predicted_grid_rtx2080ti.csv` is the counted MMQ price of one Qwen3-8B Q4_K_M prefill's
mul_mat_q launches on the RTX 2080 Ti, for each prompt size N of `tools/bench/run_524_turing_grid.sh`
and each forced column tile J (`N,columns,J,ms`; columns = N - 7, the probe's k). Each launch is
`mmq_unit::launch_seconds` plus the kernel's graph node, as the selector's counted slice prices it
(generic_cost.cpp). The launches are `qwen3_8b_q4km_launches.txt`, from the GGUF header and the
dense loader's packing. `tools/dev/mmq_grid_price.cpp` printed it at 01436ff66, built host-only
like the price tool (the decisions tree's archives and the device-seam stubs). J 48 and 112 have
no counted unit (no snapshot row), so their cells are empty.

The grid prices only the MMQ launches, so a size's other work cancels: what is scored is each
width's difference to the size's J 128 cell, and the width the price picks of the six production
widths (16, 32, 64, 80, 96, 128). It was committed before the grid's next run on the card was read.
The run of 10.11 (#607, f046b590) was read before; the unit was not fitted to it.
