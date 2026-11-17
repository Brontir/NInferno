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
32-bit `ld.cg` stream over an L2-resident buffer, 32 warps on every SM (RTX 4090 15.2, RTX 4060 Ti
17.2, RTX 3090 16.2, RTX 3060 17.75, RTX 5070 Ti 18.1). It predicts the launch law's unit (one
256-K iteration on the `law` grid) as the load phase (the weight tiles over DRAM, every tile and
activation tile through the L2, the larger of the two, plus one DRAM latency) followed by the
warp's chain (the stall sum plus one shared latency per wait on a shared load). The cards' launch
law runs (`mmq_launch_law.py derive`, master c6a1b7d50) were made before this commit and are
read only after it.
