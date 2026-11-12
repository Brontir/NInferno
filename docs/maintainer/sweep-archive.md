# The route sweep archive

The route sweeps the cost-model work is scored on live on the lab machines, not in the repository.
This page lists them: one row per card and directory of `ninferno_route_tune` CSVs. It is the
"manifest of the sweep archive" of Roadmap 3 (#613, PR-0). The band-weighted regret that scores
them is `tools/bench/regret_bands.py` (#631, `--pick model` for the cost model's own pick).

Snapshot of 2026-11-12: 758 sweeps (`--routes-out`) and 502 selections (`--selected-out`),
0.29 GB, on 9 cards. 21 files end in a row a killed sweep cut short (left out,
`cut_last_row`). Only 39 of the 1260 readable files name their run and device (route_tune
wrote those columns from #518 on), and 38 sweeps carry a `model_us` with more than six significant
digits, the precision `regret_bands.py` needs to tell a tie. The older files can still be scored with
`--routes` and the board named by hand. A PR's A/B passes may be cleaned up with its builds, so the
counts can fall between snapshots.

Regenerate (the per-file manifest, with each file's SHA-256, stays with the archive):

```bash
python3 -B tools/bench/sweep_manifest.py scan LAB_DIR --card rtx4090 > rtx4090.json   # on each machine
python3 -B tools/bench/sweep_manifest.py merge *.json > sweep_archive.json
python3 -B tools/bench/sweep_manifest.py summary sweep_archive.json > docs/maintainer/sweep-archive.md   # then restore this header
```

Columns: `sweeps` / `selections` count the files; `rows` sums their rows; `T` is the widths'
range; `protocols` the timing protocols the files name (`unnamed` before route_tune wrote them);
`model_us digits` the most significant digits of a cost-model price in the directory's sweeps
(`none`: no price); `named run/device` how many files name both; `dates` the files' modification
dates. Directory names are the lab's own; a directory of copies (`ab709_out_*` and similar) holds
one A/B's passes.

## By card

### gv100

40 sweeps, 22 selections, 6.3 MB.

| directory | sweeps | selections | rows | formats | T | protocols | model_us digits | named run/device | dates |
|---|---|---|---|---|---|---|---|---|---|
| kshapes | 8 | 8 | 26448 | bf16 q4k q5k q6k q8 | 1..8192 | unnamed | none | 0/16 | 2026-11-04 |
| kshapes3 | 8 | 8 | 39636 | bf16 iq2s iq2xs iq3s iq3xxs iq4nl iq4xs q3k q4k q5k q6k q8 | 1..8192 | unnamed | none | 0/16 | 2026-11-04..2026-11-05 |
| rmtsv | 2 | 0 | 806 | q4k q6k | 2..8192 | unnamed | 6 | 0/2 | 2026-11-05 |
| rot | 6 | 0 | 5226 | q4k q6k q8 | 1..256 | unnamed | none | 0/6 | 2026-11-04 |
| rot_v1 | 6 | 0 | 5226 | q4k q6k q8 | 1..256 | unnamed | none | 0/6 | 2026-11-04 |
| rw | 6 | 6 | 24972 | q4k q6k | 1..8192 | unnamed | 6 | 0/12 | 2026-11-08 |
| vab | 4 | 0 | 2660 | q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |

### rtx2080ti

40 sweeps, 20 selections, 11.5 MB.

| directory | sweeps | selections | rows | formats | T | protocols | model_us digits | named run/device | dates |
|---|---|---|---|---|---|---|---|---|---|
| kitrunm/run.logs | 4 | 4 | 18038 | bf16 q4k q5k q6k | 1..8192 | unnamed | none | 0/8 | 2026-11-04 |
| kshapes | 6 | 6 | 28232 | bf16 q4k q5k q6k | 1..8192 | unnamed | none | 0/12 | 2026-11-04 |
| kshapes3 | 2 | 2 | 9019 | bf16 q4k q5k q6k | 1..8192 | unnamed | none | 0/4 | 2026-11-04 |
| ncuf | 1 | 0 | 1661 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncuf/a16d | 1 | 0 | 1717 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncur8 | 1 | 0 | 1852 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncur8/d | 1 | 0 | 1852 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| retune2 | 2 | 0 | 59320 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4xs q2k q3k q4k q5k q6k q8 | 1..8192 | unnamed | none | 0/2 | 2026-11-04 |
| rmts3c | 4 | 0 | 4232 | q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rot | 6 | 0 | 5472 | q4k q6k q8 | 1..256 | unnamed | none | 0/6 | 2026-11-04 |
| rro | 2 | 2 | 9019 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rro2 | 2 | 2 | 9019 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rt1 (1 cut) | 2 | 0 | 4829 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4xs q2k q3k q4k | 1..8192 | unnamed | 6 | 0/2 | 2026-11-06 |
| rtr2 | 2 | 0 | 8582 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/2 | 2026-11-05 |
| rtraf | 2 | 2 | 9019 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rw | 2 | 2 | 7766 | q4k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-08 |

### rtx3060

77 sweeps, 55 selections, 34.5 MB.

| directory | sweeps | selections | rows | formats | T | protocols | model_us digits | named run/device | dates |
|---|---|---|---|---|---|---|---|---|---|
| kitrun/run.logs (1 cut) | 4 | 3 | 15248 | bf16 q4k q5k q6k | 1..8192 | unnamed | none | 0/7 | 2026-11-04 |
| kitrunm/run.logs | 4 | 4 | 18194 | bf16 q4k q5k q6k | 1..8192 | unnamed | none | 0/8 | 2026-11-04 |
| mmqcmp | 1 | 0 | 1254 | iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k | 32..8192 | unnamed | none | 0/1 | 2026-11-05 |
| mmqk | 1 | 0 | 537 | q4k q6k | 17..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| mmqk2 | 1 | 0 | 1259 | q4k q6k | 17..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncuf | 1 | 0 | 1666 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncuf/a16d | 1 | 0 | 1666 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncur8 | 1 | 0 | 1828 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncur8/d | 1 | 0 | 1828 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| r301 | 8 | 8 | 110945 | bf16 iq3s iq4nl iq4xs q3k q40 q4k q50 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/16 | 2026-11-07 |
| rcost | 2 | 2 | 9097 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-04 |
| rcost2 | 2 | 2 | 9097 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-04 |
| retune2 | 2 | 0 | 101212 | bf16 iq3s iq4nl iq4xs q3k q40 q4k q50 q5k q6k q8 | 1..8192 | unnamed | none | 0/2 | 2026-11-04 |
| rgt2 | 2 | 2 | 8216 | q4k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rgt3 | 2 | 2 | 8216 | q4k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rgt4 | 2 | 2 | 8216 | q4k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rgt6 | 2 | 2 | 5222 | q4k q6k | 17..8192 | unnamed | 6 | 0/4 | 2026-11-06 |
| rmts | 2 | 0 | 908 | q4k q6k | 17..8192 | unnamed | 6 | 0/2 | 2026-11-05 |
| rmts3 | 4 | 0 | 4096 | q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rmts3c | 4 | 0 | 3928 | q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rr301 | 3 | 3 | 11538 | iq4xs q4k q5k q6k q8 | 1..8192 | unnamed | none | 0/6 | 2026-11-04 |
| rro | 2 | 2 | 9097 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rro2 | 2 | 2 | 9097 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rtr2 | 2 | 0 | 8660 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/2 | 2026-11-05 |
| rtraf | 2 | 2 | 9097 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rw (2 cut) | 13 | 13 | 68518 | bf16 iq3s iq4nl iq4xs q40 q4k q6k | 1..8192 | unnamed | 6 | 0/26 | 2026-11-07 |
| rw2 | 6 | 6 | 26138 | q4k q6k | 1..8192 | unnamed | 6 | 0/12 | 2026-11-08 |

### rtx3090

81 sweeps, 67 selections, 26.6 MB.

| directory | sweeps | selections | rows | formats | T | protocols | model_us digits | named run/device | dates |
|---|---|---|---|---|---|---|---|---|---|
| cov/run.logs | 38 | 38 | 194140 | bf16 iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q41 q4k q50 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/76 | 2026-11-06 |
| gaext | 4 | 0 | 17320 | bf16 q4k q5k q6k | 1..8192 | unnamed | none | 0/4 | 2026-11-05 |
| kitrunm/run.logs | 4 | 4 | 18194 | bf16 q4k q5k q6k | 1..8192 | unnamed | none | 0/8 | 2026-11-04 |
| logs/ab521_rt | 2 | 2 | 4320 | q4k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-08 |
| logs/rtq_86 | 4 | 4 | 2388 | q4k q5k q6k | 3..8192 | unnamed | 6 | 0/8 | 2026-11-08 |
| logs/vecab_86 | 4 | 4 | 8640 | q4k q6k | 1..8192 | unnamed | 6 | 0/8 | 2026-11-08 |
| logs/vecab_86_boundall | 4 | 4 | 8640 | q4k q6k | 1..8192 | unnamed | 6 | 0/8 | 2026-11-08 |
| logs/vecab_86_gatedbound (1 cut) | 4 | 3 | 8558 | q4k q6k | 1..8192 | unnamed | 6 | 0/7 | 2026-11-08 |
| mmqcmp | 1 | 0 | 1254 | iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k | 32..8192 | unnamed | none | 0/1 | 2026-11-05 |
| ncuclk | 1 | 0 | 378 | iq1s iq2xxs iq3xxs q4k | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncuf | 1 | 0 | 1666 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncuf/a16d | 1 | 0 | 1666 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncur8 | 1 | 0 | 1828 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncur8/d | 1 | 0 | 1828 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| rt1 (1 cut) | 3 | 0 | 10587 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4xs q2k q3k q4k q5k q8 | 1..8192 | unnamed | 6 | 0/3 | 2026-11-06 |
| rtraf | 2 | 2 | 9097 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rw | 6 | 6 | 26138 | q4k q6k | 1..8192 | unnamed | 6 | 0/12 | 2026-11-08 |

### rtx4060ti

69 sweeps, 43 selections, 21.6 MB.

| directory | sweeps | selections | rows | formats | T | protocols | model_us digits | named run/device | dates |
|---|---|---|---|---|---|---|---|---|---|
| a5_f24d0079 (1 cut) | 2 | 1 | 6212 | q4k q6k | 1..8192 | copies flushed | 17 | 3/3 | 2026-11-12 |
| clk | 1 | 0 | 612 | q4k q6k | 17..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| drlaw (2 cut) | 5 | 0 | 16879 | q4k q6k | 5..8192 | unnamed | 6 | 0/5 | 2026-11-07 |
| kitrunm/run.logs | 4 | 4 | 18194 | bf16 q4k q5k q6k | 1..8192 | unnamed | none | 0/8 | 2026-11-04 |
| kshapes | 14 | 14 | 73212 | bf16 iq2s iq2xs iq3s iq3xxs iq4nl iq4xs q3k q4k q5k q6k q8 | 1..8192 | unnamed | none | 0/28 | 2026-11-04 |
| logs | 1 | 1 | 809 | q4k q5k q6k | 2..8192 | unnamed | 6 | 0/2 | 2026-11-07 |
| mmqk2 | 1 | 0 | 1259 | q4k q6k | 17..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncuf | 1 | 0 | 1666 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncuf/a16d | 1 | 0 | 1666 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncur8 | 1 | 0 | 1828 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncur8/d | 1 | 0 | 1828 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| retune2 | 2 | 0 | 74316 | bf16 iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4xs q2k q3k q4k q5k q6k q8 | 1..8192 | unnamed | none | 0/2 | 2026-11-04 |
| rgt2 | 2 | 2 | 8216 | q4k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rgt3 | 2 | 2 | 8216 | q4k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rgt4 | 2 | 2 | 8216 | q4k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rgt6 | 2 | 2 | 5222 | q4k q6k | 17..8192 | unnamed | 6 | 0/4 | 2026-11-06 |
| rmts2 | 4 | 0 | 1884 | q4k q6k | 17..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rmts3c | 4 | 0 | 4012 | q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rro | 2 | 2 | 9097 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rro2 | 2 | 2 | 9097 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rro5 | 2 | 2 | 9097 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rt1 (1 cut) | 3 | 0 | 9221 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4xs q2k q3k q4k q8 | 1..8192 | unnamed | 6 | 0/3 | 2026-11-06 |
| rtraf | 2 | 2 | 9097 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rw | 6 | 6 | 26138 | q4k q6k | 1..8192 | unnamed | 6 | 0/12 | 2026-11-08 |
| rw/ncu4060 | 2 | 1 | 176 | q4k | 24..8192 | unnamed | 6 | 0/3 | 2026-11-08 |

### rtx4070ti

91 sweeps, 44 selections, 23.4 MB.

| directory | sweeps | selections | rows | formats | T | protocols | model_us digits | named run/device | dates |
|---|---|---|---|---|---|---|---|---|---|
| . | 3 | 1 | 2109 | q4k q5k q6k q8 | 1..8192 | unnamed | none | 0/4 | 2026-11-04 |
| clk | 1 | 0 | 612 | q4k q6k | 17..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| drlaw (1 cut) | 9 | 0 | 13545 | q4k q6k | 5..8192 | unnamed | 6 | 0/9 | 2026-11-07 |
| drlaw/rtx4060ti | 3 | 0 | 10261 | q4k q6k | 5..256 | unnamed | 6 | 0/3 | 2026-11-07 |
| kitrun/run.logs | 4 | 4 | 18194 | bf16 q4k q5k q6k | 1..8192 | unnamed | none | 0/8 | 2026-11-04 |
| kitrunm/run.logs | 4 | 4 | 18194 | bf16 q4k q5k q6k | 1..8192 | unnamed | none | 0/8 | 2026-11-04 |
| kqmid | 1 | 0 | 1104 | q4k q6k | 1..8192 | unnamed | none | 0/1 | 2026-11-04 |
| kshapes | 6 | 6 | 28544 | bf16 q4k q5k q6k | 1..8192 | unnamed | none | 0/12 | 2026-11-04 |
| logs | 2 | 0 | 1442 | q4k q5k q6k q8 | 2..256 | unnamed | 6 | 0/2 | 2026-11-07 |
| mmqcmp | 1 | 0 | 1254 | iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k | 32..8192 | unnamed | none | 0/1 | 2026-11-05 |
| mmqk2 | 1 | 0 | 1259 | q4k q6k | 17..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| mmqmid | 5 | 0 | 935 | q4k q6k | 12..1024 | unnamed | none | 0/5 | 2026-11-04 |
| ncu303 | 3 | 0 | 355 | q6k | 5..8192 | unnamed | 6 | 0/3 | 2026-11-07 |
| ncu303/keep | 3 | 0 | 355 | q6k | 5..8192 | unnamed | 6 | 0/3 | 2026-11-07 |
| ncuclk | 1 | 0 | 378 | iq1s iq2xxs iq3xxs q4k | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncur8 | 1 | 0 | 1828 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncur8/d | 1 | 0 | 1828 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| p448 | 2 | 1 | 894 | q4k | 1..8192 | unnamed | 6 | 0/3 | 2026-11-06 |
| prow | 1 | 1 | 444 | q4k | 1..8192 | unnamed | 6 | 0/2 | 2026-11-06 |
| q6kv | 1 | 0 | 342 | q6k | 1..8192 | unnamed | none | 0/1 | 2026-11-04 |
| rcost | 2 | 2 | 9097 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-04 |
| retune | 1 | 0 | 41191 | bf16 iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q4k q5k q6k q8 | 1..8192 | unnamed | none | 0/1 | 2026-11-04 |
| retune2 | 2 | 0 | 82382 | bf16 iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q4k q5k q6k q8 | 1..8192 | unnamed | none | 0/2 | 2026-11-04 |
| rgt | 2 | 2 | 7766 | q4k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rgt6 | 1 | 1 | 2611 | q4k q6k | 17..8192 | unnamed | 6 | 0/2 | 2026-11-06 |
| rmts3 | 4 | 0 | 4054 | q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rro | 2 | 2 | 9097 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rro2 | 2 | 2 | 9097 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rro4 | 2 | 2 | 9097 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rt1 | 4 | 0 | 13596 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q4k q5k q6k q8 | 1..8192 | unnamed | 6 | 0/4 | 2026-11-06 |
| rtraf | 2 | 2 | 9097 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rw | 6 | 6 | 26138 | q4k q6k | 1..8192 | unnamed | 6 | 0/12 | 2026-11-08 |
| sweepnew | 4 | 4 | 18194 | bf16 q4k q5k q6k | 1..8192 | unnamed | none | 0/8 | 2026-11-04 |
| sweepnew_v1 | 4 | 4 | 18194 | bf16 q4k q5k q6k | 1..8192 | unnamed | none | 0/8 | 2026-11-04 |

### rtx4090

123 sweeps, 98 selections, 50.2 MB.

| directory | sweeps | selections | rows | formats | T | protocols | model_us digits | named run/device | dates |
|---|---|---|---|---|---|---|---|---|---|
| . | 1 | 1 | 1674 | q4k q5k q6k q8 | 1..8192 | unnamed | none | 0/2 | 2026-11-04 |
| clk | 1 | 0 | 612 | q4k q6k | 17..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| kitrunm/run.logs | 4 | 4 | 18194 | bf16 q4k q5k q6k | 1..8192 | unnamed | none | 0/8 | 2026-11-04 |
| kshapes | 16 | 16 | 79006 | bf16 iq2s iq2xs iq3s iq3xxs iq4nl iq4xs q3k q4k q5k q6k q8 | 1..8192 | unnamed | none | 0/32 | 2026-11-04 |
| kshapes_m | 16 | 16 | 79006 | bf16 iq2s iq2xs iq3s iq3xxs iq4nl iq4xs q3k q4k q5k q6k q8 | 1..8192 | unnamed | none | 0/32 | 2026-11-04 |
| logs/ab521_rt | 2 | 2 | 4320 | q4k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-08 |
| logs/rtq_89 | 4 | 4 | 2388 | q4k q5k q6k | 3..8192 | unnamed | 6 | 0/8 | 2026-11-08 |
| logs/vecab_89 | 4 | 4 | 8640 | q4k q6k | 1..8192 | unnamed | 6 | 0/8 | 2026-11-08 |
| logs/vecab_89_boundall | 4 | 4 | 8640 | q4k q6k | 1..8192 | unnamed | 6 | 0/8 | 2026-11-08 |
| logs/vecab_89_gatedbound (1 cut) | 3 | 2 | 5023 | q4k q6k | 1..8192 | unnamed | 6 | 0/5 | 2026-11-08 |
| mmqk | 1 | 0 | 537 | q4k q6k | 17..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| mmqk2 | 1 | 0 | 1259 | q4k q6k | 17..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncuf | 1 | 0 | 1828 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncuf/a16d | 1 | 0 | 1666 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncur8 | 1 | 0 | 1828 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncur8/d | 1 | 0 | 1828 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| r301 | 6 | 6 | 139808 | bf16 iq2s iq2xs iq3s iq3xxs iq4nl iq4xs q3k q40 q4k q5k q6k q8 | 1..8192 | unnamed | 6 | 0/12 | 2026-11-07 |
| r301/killed (2 cut) | 1 | 1 | 5856 | bf16 iq3s iq4nl iq4xs q3k q40 | 1..8192 | unnamed | 6 | 0/2 | 2026-11-07 |
| rcost | 2 | 2 | 9097 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-04 |
| retune | 1 | 0 | 46404 | bf16 iq3s iq4nl iq4xs q3k q40 q4k q5k q6k q8 | 1..8192 | unnamed | none | 0/1 | 2026-11-04 |
| retune2 | 2 | 0 | 92808 | bf16 iq3s iq4nl iq4xs q3k q40 q4k q5k q6k q8 | 1..8192 | unnamed | none | 0/2 | 2026-11-04 |
| rgt | 2 | 2 | 7766 | q4k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rgt2 | 2 | 2 | 8272 | q4k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rgt3 | 2 | 2 | 8272 | q4k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rgt4 | 2 | 2 | 8272 | q4k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rgt6 | 2 | 2 | 5222 | q4k q6k | 17..8192 | unnamed | 6 | 0/4 | 2026-11-06 |
| rmts3 | 4 | 0 | 4075 | q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rot | 6 | 0 | 5472 | q4k q6k q8 | 1..256 | unnamed | none | 0/6 | 2026-11-04 |
| rr301 | 3 | 3 | 8049 | q4k q6k | 1..8192 | unnamed | none | 0/6 | 2026-11-04 |
| rro2 | 2 | 2 | 9097 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rro4 | 2 | 2 | 9097 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rro5 | 2 | 2 | 9097 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rt1 | 4 | 0 | 13596 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q4k q5k q6k q8 | 1..8192 | unnamed | 6 | 0/4 | 2026-11-06 |
| rtr2 | 2 | 0 | 8660 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/2 | 2026-11-05 |
| rtraf | 2 | 2 | 9097 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rw (2 cut) | 13 | 13 | 86713 | bf16 iq3s iq4nl iq4xs q3k q40 q4k q5k q6k q8 | 1..8192 | unnamed | 6 | 0/26 | 2026-11-07 |
| selt | 0 | 2 | 60 | q4k q6k | 17..32 | unnamed | none | 0/2 | 2026-11-05 |

### rtx5070ti

64 sweeps, 44 selections, 29.4 MB.

| directory | sweeps | selections | rows | formats | T | protocols | model_us digits | named run/device | dates |
|---|---|---|---|---|---|---|---|---|---|
| . | 1 | 1 | 1802 | q4k q5k q6k q8 | 1..8192 | unnamed | none | 0/2 | 2026-11-04 |
| clk | 1 | 0 | 636 | q4k q6k | 17..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| kitrun/run.logs | 4 | 4 | 19942 | bf16 q4k q5k q6k | 1..8192 | unnamed | none | 0/8 | 2026-11-04 |
| kitrunm/run.logs | 4 | 4 | 19942 | bf16 q4k q5k q6k | 1..8192 | unnamed | none | 0/8 | 2026-11-04 |
| kshapes | 14 | 14 | 80356 | bf16 iq2s iq2xs iq3s iq3xxs iq4nl iq4xs q3k q4k q5k q6k q8 | 1..8192 | unnamed | none | 0/28 | 2026-11-04 |
| mmqk2 | 1 | 0 | 1319 | q4k q6k | 17..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncuclk | 1 | 0 | 386 | iq1s iq2xxs iq3xxs q4k | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncur8 | 1 | 0 | 1866 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncur8/d | 1 | 0 | 1866 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| q6kv | 1 | 0 | 380 | q6k | 1..8192 | unnamed | none | 0/1 | 2026-11-04 |
| retune | 1 | 0 | 86748 | bf16 fp8 nvfp4 q4 q5 q6 q6k q8 | 1..8192 | unnamed | none | 0/1 | 2026-11-04 |
| retune2 | 2 | 0 | 173496 | bf16 fp8 nvfp4 q4 q5 q6 q6k q8 | 1..8192 | unnamed | none | 0/2 | 2026-11-04 |
| rgt | 2 | 2 | 8526 | q4k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rgt6 | 1 | 1 | 2731 | q4k q6k | 17..8192 | unnamed | 6 | 0/2 | 2026-11-06 |
| rmts | 2 | 0 | 954 | q4k q6k | 17..8192 | unnamed | 6 | 0/2 | 2026-11-05 |
| rmts3 | 4 | 0 | 4374 | q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rmts3c | 4 | 0 | 4248 | q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rro2 | 2 | 2 | 9971 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rro4 | 2 | 2 | 9971 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rro5 | 2 | 2 | 9971 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rtraf | 2 | 2 | 9971 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rw | 6 | 6 | 28418 | q4k q6k | 1..8192 | unnamed | 6 | 0/12 | 2026-11-08 |
| sw70 | 5 | 4 | 22417 | bf16 q4k q5k q6k | 1..8192 | unnamed | none | 0/9 | 2026-11-04 |

### rtx5090

173 sweeps, 109 selections, 84.8 MB.

| directory | sweeps | selections | rows | formats | T | protocols | model_us digits | named run/device | dates |
|---|---|---|---|---|---|---|---|---|---|
| . | 1 | 1 | 1802 | q4k q5k q6k q8 | 1..8192 | unnamed | none | 0/2 | 2026-11-04 |
| ab709_out_5090 | 36 | 0 | 11580 | q2k q4k | 8..384 | copies flushed | 17 | 36/36 | 2026-11-12 |
| clk | 1 | 0 | 636 | q4k q6k | 17..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| kitrunm/run.logs | 4 | 4 | 19942 | bf16 q4k q5k q6k | 1..8192 | unnamed | none | 0/8 | 2026-11-04 |
| kshapes | 16 | 16 | 86682 | bf16 iq2s iq2xs iq3s iq3xxs iq4nl iq4xs q3k q4k q5k q6k q8 | 1..8192 | unnamed | none | 0/32 | 2026-11-04 |
| kshapes_m | 16 | 16 | 86682 | bf16 iq2s iq2xs iq3s iq3xxs iq4nl iq4xs q3k q4k q5k q6k q8 | 1..8192 | unnamed | none | 0/32 | 2026-11-04 |
| kshapes_v1 | 16 | 16 | 86682 | bf16 iq2s iq2xs iq3s iq3xxs iq4nl iq4xs q3k q4k q5k q6k q8 | 1..8192 | unnamed | none | 0/32 | 2026-11-04 |
| mmqcmp | 1 | 0 | 1286 | iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k | 32..8192 | unnamed | none | 0/1 | 2026-11-05 |
| mmqk2 | 1 | 0 | 1319 | q4k q6k | 17..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncuclk | 1 | 0 | 386 | iq1s iq2xxs iq3xxs q4k | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncuf | 1 | 0 | 1704 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncuf/a16d | 1 | 0 | 1704 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncur8 | 1 | 0 | 1866 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| ncur8/d | 1 | 0 | 1866 | iq1m iq1s iq2s iq2xs iq2xxs iq3s iq3xxs iq4nl iq4xs q2k q3k q40 q41 q4k q50 q51 q5k q6k q8 | 1..8192 | unnamed | 6 | 0/1 | 2026-11-06 |
| r301 | 6 | 6 | 271414 | bf16 fp8 fp8r32 iq2s iq2xs iq3s iq3xxs iq4nl iq4xs nvfp4 q3k q4 q4k q5 q5k q6 q6k q8 | 1..8192 | unnamed | 6 | 0/12 | 2026-11-07 |
| r301/killed (2 cut) | 1 | 1 | 5067 | bf16 | 1..8192 | unnamed | 6 | 0/2 | 2026-11-07 |
| r301/skewed | 2 | 2 | 8526 | q4k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-07 |
| rcost | 2 | 2 | 9971 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-04 |
| retune | 1 | 0 | 99113 | bf16 fp8 fp8r32 nvfp4 q4 q4k q5 q6 q6k q8 | 1..8192 | unnamed | none | 0/1 | 2026-11-04 |
| retune2 | 2 | 0 | 198226 | bf16 fp8 fp8r32 nvfp4 q4 q4k q5 q6 q6k q8 | 1..8192 | unnamed | none | 0/2 | 2026-11-04 |
| rgt2 | 2 | 2 | 8976 | q4k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rgt3 | 2 | 2 | 8976 | q4k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rgt4 | 2 | 2 | 8976 | q4k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rgt6 | 2 | 2 | 5462 | q4k q6k | 17..8192 | unnamed | 6 | 0/4 | 2026-11-06 |
| rmts | 2 | 0 | 954 | q4k q6k | 17..8192 | unnamed | 6 | 0/2 | 2026-11-05 |
| rmts2 | 4 | 0 | 2028 | q4k q6k | 17..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rmts3 | 4 | 0 | 4332 | q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rmts3c (1 cut) | 1 | 0 | 971 | q4k q5k q6k | 1..8192 | unnamed | 6 | 0/1 | 2026-11-05 |
| rot | 6 | 0 | 5952 | q4k q6k q8 | 1..256 | unnamed | none | 0/6 | 2026-11-04 |
| rr301 | 3 | 3 | 8847 | q4k q6k | 1..8192 | unnamed | none | 0/6 | 2026-11-04 |
| rr301_memset | 3 | 3 | 8847 | q4k q6k | 1..8192 | unnamed | none | 0/6 | 2026-11-04 |
| rro2 | 2 | 2 | 9971 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rro4 | 2 | 2 | 9971 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rro5 | 2 | 2 | 9971 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rt502 | 2 | 2 | 8604 | q4k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-08 |
| rtr2 | 2 | 0 | 9534 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/2 | 2026-11-05 |
| rtraf | 2 | 2 | 9971 | bf16 q4k q5k q6k | 1..8192 | unnamed | 6 | 0/4 | 2026-11-05 |
| rw (2 cut) | 13 | 13 | 150768 | bf16 fp8 fp8r32 nvfp4 q4 q4k q5 q6 q6k q8 | 1..8192 | unnamed | 6 | 0/26 | 2026-11-07 |
| rw2 | 6 | 6 | 28418 | q4k q6k | 1..8192 | unnamed | 6 | 0/12 | 2026-11-08 |
| selt | 0 | 2 | 60 | q4k q6k | 17..32 | unnamed | none | 0/2 | 2026-11-05 |
