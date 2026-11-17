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
