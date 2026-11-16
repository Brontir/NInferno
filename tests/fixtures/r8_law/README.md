# R8 law predictions, frozen before scoring (#613 PR-10)

`predicted_<board>.csv` is `ninferno_r8_law_predict --board <board> --pci <id> --shapes shapes.txt` at
the commit that added it (snapshots of sm_70/75/86/89 with their r8 sections, the law of
`src/ops/generic/r8_law.h`). They were committed before any r8 grid was read against them; the
scorer (`tools/dev/r8_law_score.py`) refuses a prediction file with changes after its commit.
A law change makes new files in a new commit; these stay as the record of what was predicted.

Boards: rtx4090 0x2684, rtx4070ti 0x2782, rtx4060ti 0x2805, rtx3090 0x2204, rtx3060 0x2504,
rtx2080ti 0x1E07, v100 0x1DB5 (the boards whose r8 grids the lab ran, on architectures with an
SM rate row the law can read; sm_120a's transfer is refuted, so the law abstains there).
