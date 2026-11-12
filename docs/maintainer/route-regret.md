# Route selection regret (#614 A5)

How far the selector's pick is from the fastest route measured, per board and band, on the nine
lab boards. Master f24d0079 (#768). The summaries are in `tools/bench/data/route_regret/<board>.json`:
the band tables of `tools/bench/regret_bands.py` for the production pick and for the cost model's
own pick, and the `selection_regret.py` report by board x regime x authority, with the per-case
rows left out.

## How it was measured

- `ninferno_route_tune --shapes <the golden shapes> --samples 7 --a8-candidates 1`, three passes,
  each pass writing every candidate's median (`--routes-out`) and the production pick
  (`--selected-out`). 190 cases per board: q4_K and q6_K, store and SwiGLU, T 1..2048.
- No other job on the GPU during the passes (a queue holder, the stand lock, no compute process).
- Regret is the extra time of the pick over the best observed route, summed over a band's cases
  (time-weighted). "Best observed" is the best qualified candidate in the sweep, not a global
  optimum.
- Bands (T): decode 1-8, verify 9-16, narrow 17-32, wide 33-128, batch129 129+.

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

## What it found

- **Turing (#781, #786):** the A8 routes were never priced (the gate read BF16-native MMA, not
  int8 MMA), and Turing's BF16 rate was a fitted 1536 against the spec's 512. With the fix, the
  host prediction of the 2080 Ti's A8 regret is at most +0.3 %. Open items: #789.
- **Volta (#787):** the register MMA is priced 5-11x low, and tiled and W4A8 are not priced. The
  narrow band loses 2.2-2.6x to the dp4a W4A8 tile.
- **Ada and consumer Blackwell, narrow band (#785):** MMQ is priced about 13 % high and W4A8
  32x128 low, so the pick flips (q4_K +30 % on the RTX 4070 Ti).
- **Decode T 2-4 (#788):** the vec rule's bound of 4 loses to the register MMA where a board's
  DRAM bytes per SM-clock are high (q6_K most). The counted SASS refutes an issue-rate crossover;
  an L1-sector law fits 6-7 of 8 boards and waits for its facts.

A new sweep is the same commands on a later master. The summaries are rewritten by board, and the
per-case CSVs stay on the stands (`docs/maintainer/sweep-archive.md`).
