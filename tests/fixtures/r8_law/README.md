# R8 law predictions, frozen before scoring (#613 PR-10)

`predicted_<board>.csv` is `ninferno_r8_law_predict --board <board> --pci <id> --shapes shapes.txt` at
the commit that added it (snapshots of sm_70/75/86/89 with their r8 sections, the law of
`src/ops/generic/r8_law.h`). They were committed before any r8 grid was read against them; the
scorer (`tools/dev/r8_law_score.py`) refuses a prediction file with changes after its commit.
A law change makes new files in a new commit; these stay as the record of what was predicted.

Boards: rtx4090 0x2684, rtx4070ti 0x2782, rtx4060ti 0x2805, rtx3090 0x2204, rtx3060 0x2504,
rtx2080ti 0x1E07, v100 0x1DB5 (the boards whose r8 grids the lab ran, on architectures with an
SM rate row the law can read; sm_120a's transfer is refuted, so the law abstains there).

## Law v2

`predicted_v2_<board>.csv` is the same command for law v2 (a round's chain adds the latency its waits
on global loads expose, from the snapshots' `glb_waits` columns), at the commit that added it. The
v1 files stay as they are. The GV100 stand is a Quadro GV100, so v2 names its board `gv100 0x1DBA`
(v1's `v100 0x1DB5` was the wrong board, and that score is void).

`predicted_v2_hold_<board>_t<NT>.csv` is `--shapes hold_shapes.txt --nt NT` for NT 1, 2, 4 and 8:
the holdout, 192 problems no law version was shaped on. Its grids were run before the law was
frozen and are read only after these files are committed (`r8_law_score.py --nt NT`). From 2
tokens on, the predictor reads the kernel the launch runs with a workspace: the pre-quantized
verify form where the codec has one (`kPre`). At 8 tokens there is no other form, so a codec
without it answers `none` (36 of the 192 problems; 24 on the GV100), as the bench cannot time it
either.

## Law v3

`predicted_v3_hold_<board>_t<NT>.csv` is the same command on PR 1180's law (#771): a warp's exposed DRAM
latency is the window L0 + (b_w - 512 B) / F of its counted bytes, not L0. Made at c25cfca34 (the build of
PR 1180's head; its later commits change no law input) before the holdout grids were read against it; the
v2 files stay as the record. test_r8_law holds the selector to the v3 T = 1 files.
