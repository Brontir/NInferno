#!/usr/bin/env bash
# #521: is the gated gate/up launch of ggml's vector kernel at T = 2..4 the loss that remains on the
# RTX 3060 after #560, and what would take its place? Run it on one machine, GPU idle:
#
#   tools/bench/run_521_gateup.sh [path/to/ninferno-serve] [out-dir]
#   (out-dir default: profiles/bench/521_gateup_<card>_<MMDD_HHMM>, relative to the current directory)
#   tools/bench/run_521_gateup.sh --summarize routes_1.csv routes_2.csv   (no GPU: re-read 2 files)
#
# It runs `route-tune` over tools/bench/data/521_gateup.shapes at T = 2,3,4 with every route timed
# (--routes-out) and the production pick (--selected-out), two passes, then prints per shape and T
# the candidates in microseconds, pass by pass, and the paired per-pass difference (summarize
# below). The tuner also writes its prefill band widths (T = 32 and up) whatever --points says; the
# summary reads only the shapes of the shapes file at the points asked for and ignores every other
# row. With several cards, name the one to measure (CUDA_DEVICE_ORDER=PCI_BUS_ID
# CUDA_VISIBLE_DEVICES=<i>); the out-dir name carries the card that variable selects (queried
# with nvidia-smi -i, an index or a UUID; indices follow the PCI bus order only under
# CUDA_DEVICE_ORDER=PCI_BUS_ID), else the first card nvidia-smi lists, plus "_plusN" for N more.
#
# Slots (candidate rule and its ratio table: docs/maintainer/decisions.md, entry 76):
#   gated = ggml_vec on the swiglu row (one launch over both halves, one activation quantisation).
#           For gate and up of one format this is what production runs (one gated launch).
#   twin2 = 2 x ggml_vec on the store row of the same n x k: an ESTIMATE of the alternative "two
#           non-gated launches". Expected high by about one activation quantisation (each single
#           call carries its own; the alternative has one) and low by the FP32 plane write and read
#           of the real two-launch form (StoreF32, then GateProduct), which the tuner cannot time.
#           Net sign not known. Not production's sequence: StoreF32 then GateProduct is the
#           mixed-format path.
#   rmma  = the fastest register-MMA schedule on the swiglu row (no activation quantisation): a
#           direct measurement, not an estimate.
#   other = the fastest remaining swiglu route (r8, w4a8, ...), printed for information only.
# "alt" compares gated with min(twin2, rmma) only. The tuner times more candidates than these two
# (other column): a row where "other" beats both is not an answer to "what takes its place".
#
# Verdicts. d = gated - min(twin2, rmma) per pass (microseconds; > 0: gated is slower).
#   spread = max(|d1 - d2|, TICK). TICK = 1.024 us is the raw tick of the tuner's event timer as
#   observed on the consumer boards and GV100 and recorded in the source comment at the sample loop
#   (src/ops/generic/route_tune_tool.cpp: "the timer's tick, 1.024 us on the consumer boards and
#   GV100"); the source holds no constant for it. A decode-width sample is one event pair around
#   `copies` calls and its median is divided by `copies`, so a median moves in steps of at most
#   1.024 / copies (some medians sit off that grid): copies = 3 on the swiglu rows of the kit's
#   shapes on a card with L2 <= 72 MiB (step 0.341 us), 4 on a 5090 (0.256 us), 5 on the store rows
#   at a 128 MiB span on a card with L2 <= 64 MiB (the twin's 2 x single then moves in steps of
#   0.410 us; more copies on larger L2). One raw tick is the largest step of any single median
#   (copies >= 1), so it is used as the floor, chosen as that bound; it is not an estimate of the
#   noise. TICK can be overridden from the environment for a timer with another tick. A difference
#   of one tick or less in both passes is not beyond the floor and gets no verdict: the comparison
#   leaves a margin of 0.0005 us, half the last digit the route CSV prints for these medians, so
#   that 209.579 - 208.555 (one tick plus a rounding error) is not read as more than one tick.
#   Two passes are two samples of d, not a range with a confidence: |d1 - d2| of two samples is
#   often small by chance. With d truly 0 and noise large against the floor, "both passes beyond
#   their own difference" fires in about one cell in five. A verdict on a cell within a few
#   microseconds of zero needs more passes; drift between sessions is not captured at all.
#   alt(rmma) / alt(twin) : d > spread in both passes (lo > spread); the name is the alternative
#                           that is fastest in both passes (alt: the two passes disagree on which).
#                           alt(rmma) is measured; alt(twin) rests on the estimate above.
#   gated*                : d < -spread in both passes: gated is fastest of the two named
#                           alternatives. Against twin2 not decisive (estimate), against rmma it is.
#   within spread         : otherwise (signs differ between passes, or |d| <= spread).
#
# Reading it. The candidate rule of entry 76 (vector kernel for a gated pair at T <= 3 only, under a
# per-card SMs x GHz / (GB/s) threshold) predicts on a card under the threshold: the same verdict
# pattern on every shape, with "alt" appearing at T = 4 and not at T = 3. Earlier route-tune files
# of an unverified build (RTX 3060, rows labelled a8, no twin rows for these shapes) give, gated
# against the best rmma: 12288x4096 -4.7..-5.7 % (T = 2), -3.9..-4.2 % (3), -0.9..-1.8 % (4), gated
# faster throughout; 15360x3840 +0.5..+0.9 % (2), +2.9..+3.4 % (3), +4.4..+6.2 % (4), gated slower.
# (In microseconds those are gated minus rmma = -1.7 .. -11 for 12288x4096 and +1.0 .. +1.8 at T = 2
# for 15360x3840, +6 .. +7 at T = 3, +9 .. +13 at T = 4; against the 1.024 us floor above, the T = 2 sign
# of 15360x3840 is one to two ticks, so it is not a result.)
# So the sign depends on the shape and a per-card cut does not fit both. This kit prints the shapes
# side by side on a known build. If the verdict pattern differs between shapes on a card, the
# candidate needs a shape term, and none is adopted from this table; if a card disagrees with a
# prediction, the model is wrong there and the mismatch is recorded, not fitted. Controls: a 4090 or
# a 5060 Ti (gated expected to be the least there at T <= 4).
#
# Abort rules (exit 1, message on stderr, no table): the same file, or a copy of one run, given for
# both passes (every row of a pass carries the run id the tuner wrote, so each file must hold one
# run id and the two files different ones; a file written by an older build has no run column and
# gets only the same-file check, so a copy of it under another name goes unseen there), a
# routes file empty or without a header, a
# header lacking one of the columns read by name, a cell (shape x T x slot) of the expected set
# absent in a pass, an empty or zero median_us on a row, or more than one ggml_vec row in a cell.
# ggml_vec is not a tuner candidate; its row exists only where production picks it (--selected-out),
# so where production picks something else the gated or single cell is empty and nothing compares.
# Expected cells: every swiglu line of the shapes file x every point, with a store line of the same
# n x k for the single slot. Rows of other widths or other shapes are skipped, not counted.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SHAPES=${SHAPES:-$HERE/data/521_gateup.shapes}
POINTS=${POINTS:-2,3,4}
TICK=${TICK:-1.024}   # us; see Verdicts above (source fact, not read off measurements)

# Arguments: ROUTES_1 ROUTES_2 (the two passes). Columns are read by header name, not by position:
# format, epilogue, n, k, T, route, median_us. Per pass and cell the fastest row of each
# slot. Output order: the shapes file's order, then the points' order.
summarize() {
  awk -F, -v shapes="$SHAPES" -v pts="$POINTS" -v tick="$TICK" '
    function fail(msg) { print "aborted: " msg > "/dev/stderr"; bad = 1 }
    function absv(x) { return x < 0 ? -x : x }
    function minv(a, b) { return a < b ? a : b }
    BEGIN {
      np = split(pts, pt, ",")
      for (i = 1; i <= np; i++) wantT[pt[i]] = 1
      ns = 0
      while ((rc = (getline line < shapes)) > 0) {
        sub(/\r$/, "", line)
        if (line ~ /^[ \t]*(#|$)/) continue
        if (split(line, f, " ") != 5) { fail("bad line in " shapes ": " line); continue }
        sk = f[1] SUBSEP f[4] SUBSEP f[5]
        if (f[3] == "swiglu" && !(sk in isg)) {
          ns++; order[ns] = sk; label[ns] = f[1] " " f[4] "x" f[5]
        }
        if (f[3] == "swiglu") isg[sk] = 1
        if (f[3] == "store") hasst[sk] = 1
      }
      close(shapes)
      if (rc < 0) fail("cannot read shapes file " shapes)
      else if (ns == 0) fail("no swiglu line in " shapes)
      split("format epilogue n k T route median_us", need, " ")
    }
    { sub(/\r$/, "") }
    FNR == 1 {
      pass++
      if (pass == 2 && FILENAME == file1) fail("the same file given for both passes: " FILENAME)
      file1 = FILENAME
      split("", col)
      for (i = 1; i <= NF; i++) col[$i] = i
      hasrun[pass] = ("run" in col)
      for (i = 1; i <= 7; i++) if (!(need[i] in col)) {
        fail("header of " FILENAME " has no column " need[i]); hdrbad[pass] = 1
      }
      next
    }
    pass > 2 || hdrbad[pass] { next }
    hasrun[pass] {
      rid = $col["run"]
      if (rid == "") norun[pass]++
      else if (!((pass SUBSEP rid) in runseen)) { runseen[pass SUBSEP rid] = 1; runs[pass]++
                                                  runid[pass] = rid }
    }
    {
      t = $col["T"]
      if (!(t in wantT)) { skipped++; next }
      sk = $col["format"] SUBSEP $col["n"] SUBSEP $col["k"]
      if (!(sk in isg)) { skipped++; next }
      e = $col["epilogue"]; r = $col["route"]
      if (r == "ggml_vec") {
        if (e == "swiglu") slot = "gated"
        else if (e == "store" && (sk in hasst)) slot = "single"
        else { skipped++; next }
      } else if (e == "swiglu") slot = (r ~ /^rmma/) ? "rmma" : "other"
      else { skipped++; next }
      v = $col["median_us"]
      if (v !~ /^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/ || v + 0 <= 0) { empty++; next }
      id = pass SUBSEP sk SUBSEP t SUBSEP slot
      if (slot == "gated" || slot == "single") if (++cnt[id] > 1)
        fail("more than one ggml_vec row for " $col["n"] "x" $col["k"] " T=" t " pass " pass \
             " (" slot ")")
      if (!(id in best) || v + 0 < best[id]) { best[id] = v + 0; bname[id] = r }
    }
    END {
      if (pass != 2) fail("expected two routes files, each with a header line, got " pass \
                          " (an empty file has no header)")
      if (pass == 2 && hasrun[1] != hasrun[2])
        fail("only one of the two files has a run column: the passes come from different builds")
      for (p = 1; p <= 2; p++) if (hasrun[p]) {
        if (norun[p]) fail("pass " p ": " norun[p] " row(s) with an empty run id")
        if (runs[p] > 1) fail("pass " p " holds " runs[p] " run ids: concatenated files, not one run")
      }
      if (pass == 2 && hasrun[1] && hasrun[2] && runs[1] == 1 && runs[2] == 1 && runid[1] == runid[2])
        fail("both passes carry the run id " runid[1] ": one measurement given twice")
      if (empty) fail(empty " row(s) with an empty, zero or non-numeric median_us (treated as missing)")
      expect = 0; seen = 0
      if (pass == 2 && ns > 0)
      for (i = 1; i <= ns; i++) for (j = 1; j <= np; j++) for (p = 1; p <= 2; p++) {
        sk = order[i]
        nsl = split((sk in hasst) ? "gated rmma single" : "gated rmma", sl, " ")
        for (s = 1; s <= nsl; s++) {
          expect++
          if ((p SUBSEP sk SUBSEP pt[j] SUBSEP sl[s]) in best) seen++
          else printf "missing: %s T=%s pass %d slot %s\n", label[i], pt[j], p, sl[s] \
                      > "/dev/stderr"
        }
      }
      if (seen != expect) fail("saw " seen " of " expect " expected cells (x 2 passes)")
      if (bad) exit 1
      printf "%-20s | %8s %8s %8s | %8s %8s %8s | %7s %7s %7s | %-14s | %s\n", "shape, T",
             "gated1", "2xsngl1", "rmma1", "gated2", "2xsngl2", "rmma2", "d1", "d2", "spread",
             "verdict",
             "fastest other (us, route)"
      for (i = 1; i <= ns; i++) for (j = 1; j <= np; j++) {
        sk = order[i]; k = sk SUBSEP pt[j]
        for (p = 1; p <= 2; p++) {
          g[p] = best[p SUBSEP k SUBSEP "gated"]
          rm[p] = best[p SUBSEP k SUBSEP "rmma"]
          tw[p] = (sk in hasst) ? 2 * best[p SUBSEP k SUBSEP "single"] : -1
          a[p] = (tw[p] >= 0) ? minv(tw[p], rm[p]) : rm[p]
          d[p] = g[p] - a[p]
        }
        spread = absv(d[1] - d[2]); if (spread < tick) spread = tick
        lo = minv(d[1], d[2]); hi = (d[1] > d[2]) ? d[1] : d[2]
        verdict = "within spread"
        # eps: half the last printed digit of a median, so one tick plus a rounding error is
        # not read as beyond the floor (see the header).
        eps = 0.0005
        if (lo - spread > eps) {
          verdict = "alt(rmma)"
          if (tw[1] >= 0) {
            t1 = (tw[1] < rm[1]); t2 = (tw[2] < rm[2])
            verdict = (t1 && t2) ? "alt(twin)" : (!t1 && !t2) ? "alt(rmma)" : "alt"
          }
        } else if (hi < 0 && -hi - spread > eps) verdict = "gated*"
        other = "-"
        for (p = 1; p <= 2; p++) if ((p SUBSEP k SUBSEP "other") in best) {
          o = best[p SUBSEP k SUBSEP "other"]
          if (other == "-" || o < ov) {
            ov = o; other = sprintf("%.1f %s", o, bname[p SUBSEP k SUBSEP "other"])
          }
        }
        tws1 = (tw[1] >= 0) ? sprintf("%8.1f", tw[1]) : sprintf("%8s", "-")
        tws2 = (tw[2] >= 0) ? sprintf("%8.1f", tw[2]) : sprintf("%8s", "-")
        printf "%-20s | %8.1f %s %8.1f | %8.1f %s %8.1f | %7.1f %7.1f %7.1f | %-14s | %s\n",
               label[i] " T=" pt[j], g[1], tws1, rm[1], g[2], tws2, rm[2], d[1], d[2], spread,
               verdict, other
      }
      printf "# %d shape(s) x %d point(s), all %d cells present in both passes; %d row(s) of " \
             "other widths, shapes or epilogues skipped\n", ns, np, expect / 2, skipped
    }
  ' "$@"
}

if [ "${1:-}" = "--summarize" ]; then
  [ $# -eq 3 ] || { echo "usage: $0 --summarize routes_1.csv routes_2.csv" >&2; exit 2; }
  if [ "$2" -ef "$3" ]; then echo "aborted: the same file given for both passes: $2" >&2; exit 1; fi
  summarize "$2" "$3"
  exit
fi

BIN=${1:-ninferno-serve}
GPU=gpu
# The cards route-tune measures: the ones CUDA_VISIBLE_DEVICES selects (nvidia-smi takes an index or
# a UUID), else every card.
card_names() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    local id one
    local -a ids
    IFS=, read -r -a ids <<< "$CUDA_VISIBLE_DEVICES"
    for id in "${ids[@]}"; do
      one=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$id" 2>/dev/null | head -n 1) ||
        return 1
      [ -n "$one" ] || return 1
      echo "$one"
    done
  else
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
  fi
}
if names=$(card_names); then
  first=$(printf '%s\n' "$names" | head -n 1 | tr -c 'A-Za-z0-9\n' '_' | tr -s '_' |
          sed 's/^_//; s/_$//')
  [ -n "$first" ] && GPU=$first
  more=$(printf '%s\n' "$names" | grep -c . || true)
  [ "$more" -gt 1 ] && GPU=${GPU}_plus$((more - 1))
fi
OUT=${2:-profiles/bench/521_gateup_${GPU}_$(date +%m%d_%H%M)}
mkdir -p "$OUT"

for pass in 1 2; do
  "$BIN" route-tune --shapes "$SHAPES" --points "$POINTS" --a8-candidates 1 \
    --out "$OUT/table_$pass.json" --routes-out "$OUT/routes_$pass.csv" \
    --selected-out "$OUT/selected_$pass.csv" 2>&1 | tee "$OUT/tune_$pass.log"
done

summarize "$OUT"/routes_1.csv "$OUT"/routes_2.csv 2>&1 | tee "$OUT/summary.txt"
echo "wrote $OUT (summary.txt, routes_*.csv, selected_*.csv)"
