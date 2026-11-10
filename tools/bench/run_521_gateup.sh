#!/usr/bin/env bash
# #521: is the gated gate/up launch of ggml's vector kernel at T = 3 / 4 the loss that remains on
# the RTX 3060 / 3090 after #560, and what takes its place? Run it on one machine, GPU idle:
#
#   tools/bench/run_521_gateup.sh [path/to/ninferno-serve] [out-dir]
#
# It runs `route-tune` over tools/bench/data/521_gateup.shapes at T = 2,3,4 with every route timed
# (--routes-out) and the production pick (--selected-out), two passes, then prints per shape and T
# the candidates in microseconds, pass by pass, and the paired per-pass difference (summarize below).
#
# Reading it (candidate rule in docs/maintainer/decisions.md, entry 76: the gated pair takes the
# vector kernel at T <= 3 only where SMs x GHz / (GB/s) is under a threshold between 0.148 and
# 0.207; RTX 3060 0.138, RTX 3090 0.148, RTX 5060 Ti 0.207, RTX 4090 0.320, all from
# src/core/data/gpu_facts.json):
#   gated   = ggml_vec on the swiglu row (one launch over both halves, one activation quantisation)
#   twin2   = 2 x ggml_vec on the store row of the same n x k: an UPPER BOUND on two non-gated launches
#   rmma    = the fastest register-MMA schedule on the swiglu row (no quantisation)
#
# Stated limit of twin2. The tuner times one call of a route as `linear` runs it. A single ggml_vec
# call includes its own activation quantisation; production quantises once for both launches of a
# gated pair (ggml_vec_gated_pair, generic_ggml.cpp) and runs StoreF32 then GateProduct epilogues,
# while the tuner's store row writes BF16. The tuner cannot time that sequence, so 2 x single is
# too high by one quantisation (plus the small epilogue difference) and nothing here times the
# quantisation alone. Consequence for the reading:
#   - verdict "alt"    : min(twin2, rmma) beats gated in BOTH passes by more than the pass spread.
#                        Robust: twin2 is an overestimate, so the true alternative is no slower.
#   - verdict "gated*" : gated beats both in both passes by more than the spread. Robust against
#                        rmma; against twin2 it holds only if the margin exceeds one quantisation,
#                        which is not measured, so the twinned form is not excluded by this row.
#   - verdict "within spread": the sign of the difference is not the same in both passes or the
#                        difference is not larger than the pass-to-pass disagreement.
# The rule is supported if, on the 3060 and 3090, at T = 4 the verdict is "alt" and at T = 3 it is
# not "alt". It is refuted (leave the band) if the verdict at T = 4 is "gated*" with a margin that
# a quantisation cannot explain, or "within spread". Run it on a 4090 or a 5060 Ti as a control:
# gated is expected to be the least there at T <= 4. If a control or a card disagrees with the
# prediction, the model is wrong there; record it as a mismatch.
#
# A cell the tuner did not write aborts the run (exit 1): ggml_vec is not a tuner candidate, its
# row exists only where production picks it (--selected-out), so on a card or shape where
# production picks something else the gated or single cell is empty and no comparison exists.
set -euo pipefail

BIN=${1:-ninferno-serve}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 |
      tr -c 'A-Za-z0-9\n' '_' | sed 's/_*$//') || true
OUT=${2:-logs/521_gateup_${GPU:-gpu}_$(date +%m%d_%H%M)}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"

for pass in 1 2; do
  "$BIN" route-tune --shapes "$HERE/data/521_gateup.shapes" --points 2,3,4 --a8-candidates 1 \
    --out "$OUT/table_$pass.json" --routes-out "$OUT/routes_$pass.csv" \
    --selected-out "$OUT/selected_$pass.csv" 2>&1 | tee "$OUT/tune_$pass.log"
done

# CSV: format,compute,epilogue,n,k,T,route,median_us,...  Argument 1 is pass 1, argument 2 pass 2.
# Per pass and cell: the fastest row of each slot. Decision on the per-pass difference
# d = gated - min(twin2, rmma); spread = |d1 - d2|. Any missing cell: message on stderr, exit 1,
# no table.
summarize() {
  awk -F, 'FNR == 1 { pass++; next }
    {
      key = $4 "x" $5 " T=" $6
      v = $8 + 0
      if ($7 == "ggml_vec") {
        slot = ($3 == "swiglu") ? "gated" : "single"
      } else if ($7 ~ /^rmma/ && $3 == "swiglu") {
        slot = "rmma"
      } else next
      id = pass SUBSEP key SUBSEP slot
      if (!(id in best) || v < best[id]) best[id] = v
      keys[key] = 1
    }
    function absv(x) { return x < 0 ? -x : x }
    END {
      bad = 0
      for (k in keys) for (p = 1; p <= 2; p++) {
        split("gated single rmma", slots, " ")
        for (i = 1; i <= 3; i++)
          if (!((p SUBSEP k SUBSEP slots[i]) in best)) {
            printf "missing: %s pass %d slot %s\n", k, p, slots[i] > "/dev/stderr"
            bad = 1
          }
      }
      if (pass != 2 || bad) {
        print "aborted: not every cell has gated, single (ggml_vec) and rmma rows in both passes" > "/dev/stderr"
        exit 1
      }
      printf "%-16s | %8s %8s %8s | %8s %8s %8s | %7s %7s %7s | %s\n", "shape, T",
             "gated1", "2xsngl1", "rmma1", "gated2", "2xsngl2", "rmma2", "d1", "d2", "spread", "verdict"
      fflush()
      for (k in keys) {
        for (p = 1; p <= 2; p++) {
          g[p] = best[p SUBSEP k SUBSEP "gated"]
          t[p] = 2 * best[p SUBSEP k SUBSEP "single"]
          r[p] = best[p SUBSEP k SUBSEP "rmma"]
          a = (t[p] < r[p]) ? t[p] : r[p]
          d[p] = g[p] - a
        }
        spread = absv(d[1] - d[2])
        lo = (d[1] < d[2]) ? d[1] : d[2]
        hi = (d[1] > d[2]) ? d[1] : d[2]
        verdict = "within spread"
        if (lo > 0 && lo > spread) verdict = "alt"
        else if (hi < 0 && -hi > spread) verdict = "gated*"
        printf "%-16s | %8.1f %8.1f %8.1f | %8.1f %8.1f %8.1f | %7.1f %7.1f %7.1f | %s\n", k,
               g[1], t[1], r[1], g[2], t[2], r[2], d[1], d[2], spread, verdict
      }
    }
    ' "$@"
}
summarize "$OUT"/routes_1.csv "$OUT"/routes_2.csv | tee "$OUT/summary.txt"
echo "wrote $OUT (summary.txt, routes_*.csv, selected_*.csv)"
