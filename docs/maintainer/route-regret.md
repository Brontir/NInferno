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
  belong to the prefill profile and are left out of the scoring. The weight copies rotate
  (`--rotate` is on by default: 57 consecutive calls over copies of 2x L2 or 128 MiB).
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

## Scope, and what it does not cover

Narrower than #614's PR-A5 plan:
- Only q4_K and q6_K, and only the AllowA8 decisions are scored. The A16-policy and other-format
  regret is not here.
- No shapes recorded from the lab models (`NINFERNO_ROUTE_RECORD`).
- T to 512 only: the plan asks for `--points` covering 17..2048.
- No `tools/bench/run_route_regret.sh` yet: the commands above are the procedure.
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

A new sweep is the same commands on a later master. The summaries are rewritten by board, and the
per-case CSVs stay on the stands (`docs/maintainer/sweep-archive.md`).
