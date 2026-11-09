#!/usr/bin/env bash
# #521: is the gated gate/up launch of ggml's vector kernel at T = 3 / 4 the loss that remains on
# the RTX 3060 / 3090 after #560, and what takes its place? Run it on one stand, GPU idle:
#
#   tools/bench/run_521_gateup.sh [path/to/ninferno-serve] [out-dir]
#
# It runs `route-tune` over tools/bench/data/521_gateup.shapes at T = 2,3,4 with every route timed
# (--routes-out) and the production pick (--selected-out), two passes, then prints per shape and T
# the three candidates in microseconds (summarize below).
#
# Reading it (rule candidate: the gated pair takes the vector kernel at T <= 3 only where
# SMs x GHz / (GB/s) is under ~0.17; 3060 0.138, 3090 0.166, 4090 0.352, 5060 Ti 0.206):
#   gated   = ggml_vec on the swiglu row (one launch over both halves)
#   twin2   = 2 x ggml_vec on the store row of the same n x k (two non-gated launches, the twinned
#             instance at T = 4; add ~1-2 us of launch for the second one)
#   rmma    = the fastest register-MMA schedule on the swiglu row
# The rule is supported if, on the 3060 and 3090, at T = 4 min(twin2, rmma) < gated by more than the
# pass spread, and at T = 3 gated <= min(twin2, rmma). It is refuted (leave the band) if gated is
# the least at T = 4 on either card, or the gain is under the spread. Run it on the 4090 or a 5060 Ti
# as a control: gated must be the least there at T <= 4.
set -euo pipefail

BIN=${1:-ninferno-serve}
OUT=${2:-logs/521_gateup_$(hostname)_$(date +%m%d_%H%M)}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"

for pass in 1 2; do
  "$BIN" route-tune --shapes "$HERE/data/521_gateup.shapes" --points 2,3,4 --a8-candidates 1 \
    --out "$OUT/table_$pass.json" --routes-out "$OUT/routes_$pass.csv" \
    --selected-out "$OUT/selected_$pass.csv" 2>&1 | tee "$OUT/tune_$pass.log"
done

# CSV: format,compute,epilogue,n,k,T,route,median_us,...  Best of the two passes per cell.
summarize() {
  awk -F, 'FNR > 1 && $1 != "format" {
      key = $4 "x" $5 " T=" $6
      v = $8 + 0
      if ($7 == "ggml_vec") {
        slot = ($3 == "swiglu") ? "gated" : "single"
      } else if ($7 ~ /^rmma/ && $3 == "swiglu") {
        slot = "rmma"
      } else next
      id = key SUBSEP slot
      if (!(id in best) || v < best[id]) best[id] = v
      keys[key] = 1
    }
    END {
      printf "%-18s %9s %9s %9s %9s\n", "shape (n x k, T)", "gated", "2xsingle", "rmma", "least"
      for (k in keys) {
        g = best[k SUBSEP "gated"]; t = 2 * best[k SUBSEP "single"]; r = best[k SUBSEP "rmma"]
        least = (g <= t && g <= r) ? "gated" : (t <= r ? "twin2" : "rmma")
        printf "%-18s %9.1f %9.1f %9.1f %9s\n", k, g, t, r, least
      }
    }' "$@" | sort
}
summarize "$OUT"/routes_1.csv "$OUT"/routes_2.csv | tee "$OUT/summary.txt"
echo "wrote $OUT (summary.txt, routes_*.csv, selected_*.csv)"
