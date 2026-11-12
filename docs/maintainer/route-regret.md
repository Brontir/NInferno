# Route selection regret (#614 A5)

How far the selector's pick is from the fastest route measured, per board and band, on the nine
lab boards. Master f24d0079 (#768). The summaries are in `tools/bench/data/route_regret/<board>.json`:
the band tables of `tools/bench/regret_bands.py` for the production pick and for the cost model's
own pick, and the `selection_regret.py` report by board x regime x authority, with the per-case
rows left out.

## How it was measured

- `ninferno-serve route-tune` (the `ninferno_route_tune` tool) `--shapes tools/bench/data/route_regret/shapes.txt --samples 7
  --a8-candidates 1 --out /dev/null --routes-out routes_pN.csv --selected-out sel_pN.csv`, three
  passes (N = 1, 2, 3), on a release build of the commit. Each pass writes every candidate's
  median and the production pick. The ten shapes are the golden shapes of the two lab models'
  q4_K and q6_K matrices (store and SwiGLU), at the tool's default widths: 190 scored cases per
  board, T 1..512 (1-8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512). The 1024-8192 rows
  belong to the prefill profile and are left out of the scoring. A decode or verify
  sample (T <= 16) is one call on each of ceil(max(2 x L2, 128 MiB) / the shape's weight bytes)
  weight copies (`--rotate`, on by default), clamped to 1..64, so the count differs by shape; the
  q6_K 248320 x 4096 weight, larger than that span, runs from one copy. A wider sample is one call
  after the tuners' L2 read flush.
- No other job on the GPU during the passes (a queue holder, the stand lock, no compute process).
- Regret is the extra time of the pick over the best observed route, summed over a band's cases
  (time-weighted). "Best observed" is the best qualified candidate in the sweep, not a global
  optimum.
- Bands (T): decode 1-8, verify 9-16, narrow 17-32, wide 33-128, batch129 129+.
- Scoring, per board:
  - `regret_bands.py --routes routes_p1.csv routes_p2.csv routes_p3.csv --selected sel_p1.csv
    --board <board> --qualified <text> --pick selected|model --json ...` gives the band tables;
  - `regret_collect.py --selected sel_p1.csv sel_p2.csv sel_p3.csv --routes routes_p1.csv
    routes_p2.csv routes_p3.csv --board <board> --model qwen3.5-9b+gemma4-12b-dense
    --selector-revision f24d0079 --build-id a5-f24d0079-<board> --holdout board --qualified <text>
    --pci-id <id> > regret.json`, then `selection_regret.py regret.json > report.json`, gives the
    report by regime x authority. The ridge comes from the board's PCI id.
- The summary JSON is those two outputs with the per-case rows and the GPU UUIDs left out.

## Qualification and holdout

- The candidates are qualified by a full ctest of master on each build set, not at f24d0079
  itself (the summaries' `legacy_qualification`). For sm_70, sm_86 and sm_120a that run is
  752bae56, and 752bae56..f24d0079 is #768 alone, which changes kernel-facts data, no kernel
  source: the candidates measured are the code that was qualified. For sm_89 it is f24d0079.
  **For sm_75 (RTX 2080 Ti) it is 65e573fd, and the commits after it change kernel sources
  (ggml_mmq, ggml_vec): its candidates were not qualified at the measured revision.** Its regret
  is dominated by the cost-model error of #781 either way, but read its per-route numbers with
  that caveat until a sweep at a qualified revision replaces them.
- `holdout: board` is the tools' default label, and no `fit_boards` were declared, so `passed`
  does not establish a board holdout. Production answers here include route-table and dense-tune
  entries measured on the board itself, and the selector carries constants fitted on named lab
  boards. These are regrets of the selector as shipped, not out-of-sample results.

## Scope, and what it does not cover

Narrower than #614's PR-A5 plan:
- Only q4_K and q6_K, and only the AllowA8 decisions are scored. The A16-policy and other-format
  regret is not here.
- No shapes recorded from the lab models (`NINFERNO_ROUTE_RECORD`).
- T to 512 only: the plan asks for `--points` covering 17..2048.
- The commands above are now one script, `tools/bench/regret_job.sh` ([below](#the-regret-job)).
- The per-case rows, which `route_table_prune.py` (PR-F3) reads, are not committed. They stay in
  the sweep's CSVs on the stands (`docs/maintainer/sweep-archive.md`).

Each is a follow-up box on #614.

## Production pick (S)

| board | decode | verify | narrow | wide | batch129 |
|---|---:|---:|---:|---:|---:|
| RTX 2080 Ti | +19.6% | +1.0% | +179.7% | +297.8% | +343.2% |
| Quadro GV100 | +11.9% | +3.6% | +41.5% | +3.8% | +0.0% |
| RTX 3060 | +4.8% | +1.5% | +0.0% | +0.2% | +0.1% |
| RTX 3090 | +2.9% | +2.6% | +0.8% | +1.2% | +0.1% |
| RTX 4060 Ti | +1.0% | +2.6% | +2.3% | +1.1% | +0.4% |
| RTX 4070 Ti | +1.6% | +1.1% | +8.0% | +0.2% | +0.7% |
| RTX 4090 | +1.1% | +1.8% | +7.5% | +1.4% | +0.2% |
| RTX 5070 Ti | +3.9% | +2.1% | +9.2% | +2.6% | +1.9% |
| RTX 5090 | +3.1% | +2.2% | +2.0% | +0.4% | +1.2% |

## The cost model's own pick (M, T >= 17)

| board | narrow | wide | batch129 |
|---|---:|---:|---:|
| RTX 2080 Ti | +179.7% | +297.8% | +343.2% |
| Quadro GV100 | +67.5% | +221.4% | +853.4% |
| RTX 3060 | +0.0% | +0.2% | +0.1% |
| RTX 3090 | +0.8% | +1.2% | +0.1% |
| RTX 4060 Ti | +2.3% | +1.1% | +0.4% |
| RTX 4070 Ti | +8.0% | +0.2% | +0.7% |
| RTX 4090 | +7.5% | +1.4% | +0.2% |
| RTX 5070 Ti | +9.2% | +2.6% | +1.9% |
| RTX 5090 | +2.0% | +0.4% | +1.2% |

At T >= 17 the cost model is the authority for these A8 cases on every board but the GV100, so M
equals S there. On the GV100 the `volta-tiled` rule decides, and it is what keeps S far below M.

## By regime and authority

`median`, `p95` and `max` are per-case regrets of the production pick within each group.
`evidence` says where the authority's choice came from (`ranker`: the selector's own ranking;
`measured` / `borrowed`: a table entry measured on this board or another).

| board | regime | authority | cases | evidence | median | p95 | max |
|---|---|---|---:|---|---:|---:|---:|
| RTX 2080 Ti | decode T<=4 | dense-tune | 2 | measured 2 | +0.0% | +0.0% | +0.0% |
| RTX 2080 Ti | decode T<=4 | r8-rules | 9 | ranker 9 | +0.0% | +3.3% | +3.3% |
| RTX 2080 Ti | decode T<=4 | vec-rule | 29 | ranker 29 | +9.1% | +70.8% | +71.3% |
| RTX 2080 Ti | verify T5-16 | mmq-verify-rule | 20 | ranker 20 | +0.0% | +42.8% | +42.9% |
| RTX 2080 Ti | verify T5-16 | rmma-rule | 20 | ranker 20 | +0.0% | +17.5% | +18.4% |
| RTX 2080 Ti | verify T5-16 | route-table | 20 | measured 20 | +0.0% | +0.0% | +0.0% |
| RTX 2080 Ti | T>=17 below ridge | cost-model | 50 | ranker 50 | +121.4% | +352.9% | +389.8% |
| RTX 2080 Ti | T>=17 above ridge | cost-model | 40 | ranker 40 | +219.8% | +442.1% | +470.1% |
| Quadro GV100 | decode T<=4 | r8-rules | 10 | ranker 10 | +0.7% | +6.3% | +6.3% |
| Quadro GV100 | decode T<=4 | vec-rule | 30 | ranker 30 | +3.5% | +34.5% | +37.4% |
| Quadro GV100 | verify T5-16 | volta-hmma | 60 | ranker 60 | +0.2% | +20.2% | +40.8% |
| Quadro GV100 | T>=17 ridge unknown | volta-tiled | 90 | ranker 90 | +0.0% | +118.2% | +163.5% |
| RTX 3060 | decode T<=4 | r8-rules | 10 | ranker 10 | +0.0% | +8.7% | +8.7% |
| RTX 3060 | decode T<=4 | vec-rule | 30 | ranker 30 | +4.6% | +23.9% | +25.1% |
| RTX 3060 | verify T5-16 | rmma-rule | 36 | ranker 36 | +0.0% | +17.1% | +18.3% |
| RTX 3060 | verify T5-16 | route-table | 24 | measured 24 | +0.0% | +0.0% | +0.0% |
| RTX 3060 | T>=17 below ridge | cost-model | 40 | ranker 40 | +0.0% | +2.0% | +15.4% |
| RTX 3060 | T>=17 above ridge | cost-model | 50 | ranker 50 | +0.0% | +4.5% | +7.8% |
| RTX 3090 | decode T<=4 | r8-rules | 10 | ranker 10 | +0.0% | +8.4% | +8.4% |
| RTX 3090 | decode T<=4 | vec-rule | 30 | ranker 30 | +2.9% | +32.5% | +56.1% |
| RTX 3090 | verify T5-16 | rmma-rule | 36 | ranker 36 | +6.0% | +28.3% | +28.7% |
| RTX 3090 | verify T5-16 | route-table | 24 | borrowed 12, measured 12 | +0.0% | +19.6% | +20.6% |
| RTX 3090 | T>=17 below ridge | cost-model | 40 | ranker 40 | +0.3% | +29.6% | +42.9% |
| RTX 3090 | T>=17 above ridge | cost-model | 50 | ranker 50 | +0.0% | +15.2% | +22.0% |
| RTX 4060 Ti | decode T<=4 | r8-rules | 10 | ranker 10 | +0.0% | +1.0% | +1.0% |
| RTX 4060 Ti | decode T<=4 | vec-rule | 30 | ranker 30 | +3.1% | +12.1% | +13.8% |
| RTX 4060 Ti | verify T5-16 | rmma-rule | 30 | ranker 30 | +2.0% | +14.0% | +21.1% |
| RTX 4060 Ti | verify T5-16 | route-table | 30 | borrowed 12, measured 18 | +0.0% | +24.5% | +25.2% |
| RTX 4060 Ti | T>=17 below ridge | cost-model | 60 | ranker 60 | +0.0% | +10.1% | +26.3% |
| RTX 4060 Ti | T>=17 above ridge | cost-model | 30 | ranker 30 | +0.0% | +4.1% | +4.5% |
| RTX 4070 Ti | decode T<=4 | r8-rules | 10 | ranker 10 | +0.0% | +1.7% | +1.7% |
| RTX 4070 Ti | decode T<=4 | vec-rule | 30 | ranker 30 | +0.4% | +11.5% | +14.2% |
| RTX 4070 Ti | verify T5-16 | rmma-rule | 30 | ranker 30 | +4.7% | +26.7% | +28.2% |
| RTX 4070 Ti | verify T5-16 | route-table | 30 | measured 30 | +0.0% | +0.0% | +0.1% |
| RTX 4070 Ti | T>=17 below ridge | cost-model | 60 | ranker 60 | +0.0% | +47.5% | +60.4% |
| RTX 4070 Ti | T>=17 above ridge | cost-model | 30 | ranker 30 | +0.0% | +6.2% | +10.6% |
| RTX 4090 | decode T<=4 | r8-rules | 10 | ranker 10 | +0.0% | +1.6% | +1.6% |
| RTX 4090 | decode T<=4 | vec-rule | 30 | ranker 30 | +3.2% | +15.7% | +30.1% |
| RTX 4090 | verify T5-16 | rmma-rule | 30 | ranker 30 | +0.0% | +25.9% | +26.7% |
| RTX 4090 | verify T5-16 | route-table | 30 | borrowed 12, measured 18 | +0.0% | +31.4% | +32.6% |
| RTX 4090 | T>=17 below ridge | cost-model | 60 | ranker 60 | +0.0% | +34.8% | +68.0% |
| RTX 4090 | T>=17 above ridge | cost-model | 30 | ranker 30 | +0.0% | +11.0% | +14.7% |
| RTX 5070 Ti | decode T<=4 | r8-rules | 10 | ranker 10 | +2.6% | +9.3% | +9.3% |
| RTX 5070 Ti | decode T<=4 | vec-rule | 30 | ranker 30 | +10.3% | +58.7% | +76.1% |
| RTX 5070 Ti | verify T5-16 | rmma-rule | 30 | ranker 30 | +6.4% | +32.1% | +32.1% |
| RTX 5070 Ti | verify T5-16 | route-table | 30 | measured 30 | +0.0% | +0.0% | +0.0% |
| RTX 5070 Ti | T>=17 below ridge | cost-model | 50 | ranker 50 | +0.0% | +26.7% | +61.5% |
| RTX 5070 Ti | T>=17 above ridge | cost-model | 40 | ranker 40 | +0.0% | +5.5% | +8.5% |
| RTX 5090 | decode T<=4 | r8-rules | 10 | ranker 10 | +0.0% | +13.8% | +13.8% |
| RTX 5090 | decode T<=4 | vec-rule | 30 | ranker 30 | +10.4% | +53.1% | +97.4% |
| RTX 5090 | verify T5-16 | rmma-rule | 30 | ranker 30 | +0.0% | +26.9% | +36.3% |
| RTX 5090 | verify T5-16 | route-table | 30 | borrowed 6, measured 24 | +0.0% | +0.4% | +0.4% |
| RTX 5090 | T>=17 below ridge | cost-model | 60 | ranker 60 | +0.0% | +33.3% | +50.0% |
| RTX 5090 | T>=17 above ridge | cost-model | 30 | ranker 30 | +0.0% | +5.8% | +11.1% |

## Table exit metric (#616 step 9)

The report's `table_exit` gives one entry per board. It is the number to watch over releases for
whether the measured tables are still needed.
- `answered_by` and `share` count the board's cases by the kind of answer. The kind comes from
  the case's evidence:
  - `table`: measured on this board;
  - `carried`: borrowed, carried or transferred from another board;
  - `priced`: the cost model's ranking;
  - `rules`: any other rule or fallback of the selector;
  - `pin`;
  - `unknown`: a collection without the columns, or an evidence name the tool does not know.
- Of the board's measured overrides (`route-table`, `dense-tune`, `dense-tune-prefill-band`, the
  ones `route_table_prune.py` prunes), the report splits them three ways:
  - `caught_up`: the ranker's route is within the resolution of the override's (5 % by default,
    `--resolution` there, `exit_resolution` here);
  - `holds`: the ranker's route is still slower;
  - `unscored`: the case has no ranker time.
- `exit_share = caught_up / (caught_up + holds)`. It counts cases, not table entries.

As the cost model improves, `exit_share` rises toward 1. `route_table_prune.py` may then delete
an entry whose cases all caught up; it keeps one for its other reasons too (a claim, a borrower,
a prefill range, an uncovered segment). A board whose `table` share stays high while its `exit_share` is low still
rests on its own measurements. The committed reports above were collected before the metric
existed. They carry no per-case rows, so the first numbers come with the next regret run (#620
step 10).

## What it found

- **Turing (#781, #786):** the A8 routes were never priced (the gate read BF16-native MMA, not
  int8 MMA), and Turing's BF16 rate was a fitted 1536 against the spec's 512. With the fix, the
  host prediction of the 2080 Ti's A8 regret is at most +0.3 %. Open items: #789.
- **Volta (#787):** the register MMA is priced 5-11x low, and tiled and W4A8 are not priced. In
  the narrow band the `volta-tiled` rule's pick costs +41.5 % band-weighted, with worst cases
  2.2-2.6x slower than the dp4a W4A8 tile.
- **Ada and consumer Blackwell, narrow band (#785):** MMQ is priced about 11-16 % high there. On
  Ada the W4A8 32x128 tile is also priced low, so the pick flips (q4_K +30 % on the RTX 4070 Ti).
  On the RTX 5070 Ti that tile is priced high instead.
- **Borrowed route-table entries in the verify regime:** on the RTX 3090, 4060 Ti and 4090 the
  `route-table` authority's p95 is +20-31 %. Per case, all of it is in the 12 entries each board
  borrows from another board (worst +33 % on the 4090, +25 % on the 4060 Ti, +21 % on the 3090);
  the entries measured on the board itself are at most +0.1 %. The boards whose verify entries are
  all their own (RTX 3060, 4070 Ti, 5070 Ti, 2080 Ti) are at about 0. The RTX 5090 also borrows 6
  verify entries, and they lose at most +0.4 %. This is PR-F3's input: an entry that carries to
  another board without a law can lose there.
- **Decode T 2-4 (#788):** the vec rule's bound of 4 loses to the register MMA where a board's
  DRAM bytes per SM-clock are high (q6_K most). The counted SASS refutes an issue-rate crossover;
  an L1-sector law fits 6 of 8 boards for each format and waits for its facts.

## The route policy's moved decisions (#614 F1)

`--route-policy evidence` (`docs/maintainer/environment.md#route-policy`) demotes a route-table
answer that is `borrowed` or `transferred` below the ranker. Which cells of this sweep's grid move,
read on a CPU before any GPU run: `tools/bench/data/route_regret/policy_evidence_moves.json`,
written by `ninferno_route_policy_test --write <file>` (each board from its own build set: run it in
a base, a full (sm_120a) and an sm_75 tree; a run keeps the boards of the other sets). The grid is
this sweep's shapes at route_tune's points (T 1..512) under A16Only and AllowA8, and the prefill
chunk (T 1024) under the prefill profile, 400 cells a board. Each board is asked twice: under the
shipped tables (`own`), and under them without its own table (`without own`: its nearest table is
another board's, as on a card the tables do not hold; `--route-dir` to a copy of `routes/` without
that file runs it). The Quadro GV100 needs an sm_70 tree (CUDA 12.x) and is not in the file; in the
other build sets it has no route-table answer in the grid, so nothing of it moves. The RTX 2080 Ti
without its own table has no other sm_75 table, so the rules decide there under both policies.

| board | tables | build set | borrowed | transferred | moved | a8 verify T5-16 | a16 decode T<=4 | a16 verify T5-16 | a16 T>=17 | a16 prefill | demoted, same route | model ratio (median) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RTX 5090 | own | full | 26 | 0 | 16 | 4 | 4 | 4 | 3 | 1 | 10 | 0.81 |
| RTX 5090 | without own | full | 124 | 0 | 75 | 20 | 15 | 14 | 25 | 1 | 49 | 0.85 |
| RTX 5070 Ti | own | full | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - |
| RTX 5070 Ti | without own | full | 98 | 0 | 64 | 16 | 9 | 10 | 29 | 0 | 34 | 0.92 |
| RTX 4090 | own | base | 50 | 0 | 31 | 8 | 4 | 6 | 13 | 0 | 19 | 0.64 |
| RTX 4090 | without own | base | 0 | 128 | 61 | 15 | 8 | 13 | 25 | 0 | 67 | 0.70 |
| RTX 4070 Ti | own | base | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - |
| RTX 4070 Ti | without own | base | 50 | 0 | 32 | 8 | 4 | 6 | 14 | 0 | 18 | 0.93 |
| RTX 4060 Ti | own | base | 52 | 0 | 38 | 11 | 8 | 11 | 8 | 0 | 14 | 0.93 |
| RTX 4060 Ti | without own | base | 52 | 76 | 66 | 19 | 12 | 17 | 18 | 0 | 62 | 0.93 |
| RTX 3090 | own | base | 48 | 0 | 18 | 4 | 0 | 0 | 14 | 0 | 30 | 0.90 |
| RTX 3090 | without own | base | 0 | 100 | 36 | 8 | 0 | 4 | 24 | 0 | 64 | 0.94 |
| RTX 3060 | own | base | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - |
| RTX 3060 | without own | base | 0 | 50 | 27 | 6 | 4 | 4 | 13 | 0 | 23 | 0.96 |
| RTX 2080 Ti | own | sm75 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - |
| RTX 2080 Ti | without own | sm75 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - |

`borrowed` and `transferred` count the cells whose legacy answer is a route-table entry of that
kind. No AllowA8 cell outside verify moves: there the route table stands aside (the vector kernel,
R8, the MMQ verify rule and the W4A8 calls from 17 tokens). Every moved cell's evidence answer is
the register-MMA rule's (T <= 16, and the prefill chunk) or the cost model's (T >= 17). The RTX
5090's own-table moves are all the q6_K 248320 x 4096 output projection, whose entry it borrows from
the RTX 5070 Ti's table. Without their own tables the RTX 5090, 5070 Ti and 4070 Ti find a nearest
table within the carry distance (`carried`, kept) and move only on the entries it lacks
(`borrowed`); the RTX 4090, 3090, 3060 and part of the 4060 Ti's grid are `transferred`. The A16Only
moves are outside this sweep's scoring (AllowA8 only): the A/B measures them for the first time.

Each move carries two ratios of the evidence route's time over the legacy route's, stated before any
run: `model_ratio`, the cost model's own prices of the two routes (`priced_routes_query`; below 17
tokens a figure of the formula, which the selector does not rank by there), and `a5_ratio`, the
two routes' medians in this sweep's routes CSVs, which stay on the stands: filled by
`tools/bench/policy_moves_predict.py --moves <file> --board <board> --routes routes_p1.csv
routes_p2.csv routes_p3.csv --write` (it touches only the moves of the compute classes the CSVs
hold, so this sweep's AllowA8 CSVs fill the a8 cells and leave the a16 ones null). A rewrite by
`ninferno_route_policy_test --write` keeps a move's `a5_ratio` while its cell and both routes are
unchanged. A ratio below 1 predicts the evidence pick faster.

The measured prediction, committed before any A/B run (f24d0079's sweeps, three passes; the median
over the moved cells this sweep timed):

| board | own tables (cells timed) | without own table: median, worst |
|---|---|---|
| RTX 5090 | 0.996 (4 of 16) | 0.954, worst 1.693 |
| RTX 5070 Ti | no move | 1.061, worst 1.759 |
| RTX 4090 | 0.821 (8 of 31) | 0.897, worst 1.077 |
| RTX 4070 Ti | no move | 1.026, worst 1.093 |
| RTX 4060 Ti | 1.002 (11 of 38) | 1.005, worst 1.061 |
| RTX 3090 | 1.126 (4 of 18) | 1.032, worst 1.186 |
| RTX 3060 | no move | 0.981, worst 1.358 |

The evidence policy wins on the RTX 4090's borrowed verify entries and loses on the RTX 3090's:
there the borrowed entries are ggml's MMQ at T 12-16, and the register MMA the evidence walk hands
the call to runs 1.07-1.19x their time. At these widths the register-MMA rule decides, not the cost
model. The cost model orders the two the measured way but prices the gap wrong (the register MMA at
1.25-1.76x the MMQ against a measured 1.07-1.19x), and a ranker that is to stand in for borrowed
MMQ entries needs MMQ priced right over the verify and narrow widths, which is #785's question.
**F2 (evidence as the default) is blocked on #785, the narrow-band MMQ price:** a blanket default
would regress the RTX 3090 as the tables stand.

A new sweep is the same commands on a later master. The summaries are rewritten by board, and the
per-case CSVs stay on the stands (`docs/maintainer/sweep-archive.md`).

## The regret job

`tools/bench/regret_job.sh BUILD_DIR [OUT_DIR]` (#620 step 10) runs the sweep above on one board,
with the GPU idle. It takes about 25 minutes on a lab board.

**The build must be fresh from this checkout.** A build records no revision of its own, so the job
ties the build to the checkout in two ways:
- `CMAKE_HOME_DIRECTORY` in its `CMakeCache.txt` must be this checkout, else it refuses;
- it builds the job's three targets (`ninferno-serve`, `ninferno_linear_generic_test`,
  `ninferno_route_classes_test`) before measuring, which does nothing on a fresh build. A dry run
  (`ninja -n`) cannot tell freshness: CMake's glob check reruns on every invocation.

The report's `revision` is then the checkout's HEAD. `nvcc` is the version of the CUDA compiler
the cache names. When the cache names none, `nvcc` is `unknown`: the job never reads the `nvcc`
on PATH, which can be another toolkit than the build's (a 13.x on PATH beside a 12.8 sm_70 build).

1. **Qualify.** ctest in `BUILD_DIR` runs `ninferno_linear_generic_test` (with
   `NINFERNO_TEST_FORMATS` set to the formats of the grid) and `ninferno_route_classes_test`. Both
   must pass and neither may be skipped. Their names become the `--qualified` statement, so the
   candidates are qualified at the measured build itself.
2. **Sweep.** Three passes of `ninferno-serve route-tune --shapes
   tools/bench/data/route_regret/shapes.txt --samples 7 --a8-candidates 1`, plus `--route-policy`
   when `ROUTE_POLICY` is set.
3. **Score.** `regret_collect.py --pci-id` (the board label and card name come from the card's PCI id
   in `gpu_facts.json`), then `selection_regret.py` and `regret_bands.py --doc`.
4. **Report.** `regret_report.py build` writes `OUT_DIR/<stamp>-<rev12>.json` and a `.md` summary.
   `<stamp>` is the report's UTC date, `YYYYMMDDTHHMMSSZ`, so the file names sort by time. The
   maintainer commits the JSON as `tools/bench/data/route_regret/runs/<board>/<stamp>-<rev12>.json`;
   the per-case CSVs stay on the stand.

The job refuses to start if another compute process is on the card. That check is a snapshot at
the start only: the job takes no lock. On a stand it runs through the stand's own queue lock, which
keeps other GPU work off the card for the whole run. During the passes it samples the SM clock once
a second.

**The report, `ninferno-regret-run/1`.** `regret_report.py check FILE...` validates it.
- **Provenance:** `revision` (the checkout's HEAD, `-dirty` with uncommitted edits), `board`,
  `card`, `device` (route_tune's class id), `pci_device_id`, `date` (UTC), `nvcc` and
  `nvcc_source` (`CMAKE_CUDA_COMPILER`, or `none` with `nvcc` `unknown`), `driver`, `qualification`
  and `sweep` (grid file and its sha256, passes, samples, route policy).
- **`gpu`:** name, power limit, max SM clock, application graphics clock, persistence mode, and
  the SM clock sampled during the sweep (`min`, `max`, `samples`).
- **Scores:** `summary` (selection_regret's `by_board` row), `by_board_regime_authority` (this
  board's groups) and `bands` (regret_bands' `overall`, the production pick).
- **Table exit:** `table_exit` is the metric of #616 step 9. `overrides` holds the case ids behind
  its counts (`caught_up`, `holds`, `unscored`). Both are null when selection_regret.py predates
  the metric.
- **`delta`:** a comparison with the board's previous committed report, which is the newest
  `runs/<board>/*.json` at HEAD, else the A5 summary `<board>.json`.
  - `sweep_matches` says whether both runs swept the same grid (sha256), samples, passes and route
    policy: `true`, `false`, or `unknown` against an A5 summary, which does not record them. When
    it is not `true`, the moves are not the selector's alone, and the `.md` labels them not
    comparable.
  - The figures compared, all as slowdown factors:
    - per regime x authority group and for the board: the geomean and the p95;
    - per band: the time-weighted factor.
  - A factor that grew by more than `threshold` (5 %, the table exit's resolution) is listed under
    `regressions`, with both values and the case counts. One that shrank by as much is listed under
    `improvements`.
  - `new_groups` and `gone_groups` are the regime x authority groups found in only one of the two
    reports.
  - `new_holds` and `new_caught_up` are the override cases whose state changed. They are null
    against an A5 summary, which keeps no per-case rows.

The narrowest decode cases sit within a few ticks of the 1.024 us event timer. A move there below
the threshold is not read as a change.

**When it runs.**
- **After a merge:** `tools/audit/regret_due.py OLD..NEW` exits 0 when the range touches any of:
  - a decision-path file (`tools/audit/decision_paths.txt`, the golden gate's list);
  - the grid (`shapes.txt`) or the job that runs it (`regret_job.sh`: its passes, samples,
    qualification and route-tune flags);
  - a route kernel: a `.cu` or `.cuh` under `src/ops/generic`, or any file a `.cu` there includes,
    transitively (`src/ops/common/mma.cuh`, ...), at either end of the range.

  It also names the boards:
  - a route table (`routes/<class>.json`) moves only the boards of its own compute capability;
    a table of an architecture no lab board has moves the boards of its major, since a sibling
    minor may borrow it;
  - the grid and the job move every board;
  - any other file is read by `affected_arches.py`, each file on its own. An arch-guarded file
    names its arch's boards;
    every other answer, host code included (the selector runs on the host for every board), names
    every board.

  If only the scoring tools changed, the job is not due: re-score the archived CSVs instead. The
  script exits 1 when the job is not due and 2 on an error. `--json` prints the answer with a
  reason for each file.
- **Weekly:** on each lab stand, one run of master. A cron line, not installed:

  ```
  # m h dom mon dow: Sundays 03:00, a Release build of origin/master (the job's three targets) in build-regret
  0 3 * * 0  cd "$HOME/ninferno" && mkdir -p profiles/bench/regret && { git fetch -q origin && git checkout -q --detach origin/master && cmake -S . -B build-regret -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON && cmake --build build-regret -j8 --target ninferno-serve ninferno_linear_generic_test ninferno_route_classes_test && tools/bench/regret_job.sh build-regret; } >> profiles/bench/regret/cron.log 2>&1
  ```

  The job reads the previous report at HEAD, so after a run is committed, the next one compares
  against it.

  On a stand whose GPU work goes through a queue, put the same command into the queue at that
  hour instead.
