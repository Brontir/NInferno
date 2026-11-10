#!/usr/bin/env bash
# PR #598 (Refs #501): route-tune regret of master against the PR branch on ONE machine, and the
# selector goldens.  Usage (on the machine):
#
#   tools/bench/run_598_regret.sh regret  TAG ARCH      # sm_86 board: 86; sm_120a boards: 120a
#   tools/bench/run_598_regret.sh goldens TAG ARCH      # regenerate selector goldens in the PR tree
#
# Environment: PR_SRC (default: the checkout holding this script), MASTER_SRC (default
# $PR_SRC/../wt_598_master, a detached worktree of origin/master made on first use), PASSES (3),
# MODES ("a16 a8": a16 = A16Only, the exact mode's linears; a8 = --a8-candidates 1, the default
# Fast policy), SAMPLES (route_tune default 15), OUT (regret_598_TAG_ARCH), JOBS (8).
# Build dirs are build_TAG_ARCH inside each tree (the name kt3.sh gives them); an existing one is
# reused and only the targets below are (re)built.  Nothing here fits a constant.
#
# WHAT THE PR CHANGES.  generic_cost.cpp multiplies the register MMA's tensor term by
# (Sets*MT + 1)/(Sets*MT) (rmma_min_term.h) for the formats with a min term: q4k q5k q2k q41 q51
# (read from rmma_has_min_term below), except where the tile adds the activation sums in FP32
# (NT >= 4 token tiles and rmma_add_sums(): speed profile, bf16 MMA; Volta always).
# Per schedule (generic_schedules.h, X(i, warps_m, warps_k, MT, NT, cluster)), Sets = 1 for
# store/add, 2 for swiglu:
#     NT 1 or 2, MT 1 (idx 0 1 2 3 5 6 16 17 18 19)   2.00  (swiglu 1.50)
#     NT 1 or 2, MT 2 (idx 4 7)                       1.50  (swiglu 1.25)
#     NT >= 4          (idx 8..15)                    1.00  (unchanged)
# Only the mma term of `cost` is scaled; dram, L2, chain and reduce terms are as before, and the
# other families (tiled, staged, ggml, R8, W4A8) are untouched.
#
# PREDICTED MOVES (the "moved" table this script prints checks them; theory first, measurement
# only verifies).  The factor hits exactly the narrow tiles (8 or 16 tokens), which compete with
# the wide ones at T >= 17 only (T <= 16 is one column of them and does not use this price).  It
# changes a pick only where the mma term is a visible part of the additive sum, near the ridge
# T* ~ peak_sm / bw_sm * bpw / 2 (q4k bpw 0.5625; datasheet-derived, approximate):
#     RTX 3060  GA106, 28 SMs, 360 GB/s,  T* ~ 20  -> T = 17..32 move, mid and large n
#     5070 Ti   GB203, 70 SMs, 896 GB/s,  T* ~ 27  -> T = 17..48 move
#     5090      GB202, 170 SMs, 1792 GB/s, T* ~ 33 -> T = 24..48 move, large n only
# Direction: m1.n1 / m1.n2 (x2.0) lose to m2.n1 / m2.n2 (x1.5), to the NT >= 4 tiles (x1.0, 32+
# tokens per CTA column) and to tiled / staged; swiglu (x1.5 / x1.25) moves least.  The smallest
# n (256x2048, 1024x4096) gives fewer CTAs than SMs on the 5090 and 5070 Ti: chain latency, not
# the mma term, sets the price, so no move there; on the 28-SM 3060 the same n fills the wave and
# can move.  No move expected at T >= 64 (narrow tiles already lose on the L2 term), where an
# NT >= 4 tile wins, for formats without a min term, and on sm_70 (adds_sums).  Regret should fall
# where master picked a narrow tile that measures slower than a 32-token tile; the PR comment warns
# it can rise where the sum already over-prices (4060 Ti q4k 4096x3840 T=24: m2.n2 171 us priced vs
# 50.7 measured), so a move onto tiled.64x128 there is the failure to look for.  PR regret above
# master's in a TOTAL row blocks the merge.
#
# SELECTOR GOLDENS (after the regret numbers are accepted; each needs a GPU of its arch, because
# the 86 and 120a boards are keyed by the facts digest of the compiled kernels, and the release
# toolkit is nvcc 13.4.92, tests/ops/linear/selector_golden/facts_index.txt):
#     an sm_86 board (86) -> base/*  (a10, rtx3090 by digest; rtx4060ti, rtx4090, l4, ... from the same build)
#     an sm_120a board (120a) -> full/*  (rtx5090, rtx5080, rtx5070ti, rtx5060ti, unknown-sm120-48sm, rest)
#     an RTX 2080 Ti (75) -> sm75/*  (Turing has tensor MMA: rows may move)
#     a Volta board (70, CUDA 12.8 toolkit) -> sm70/*  (no tensor MMA: expect NO diff)
# Per machine:   tools/bench/run_598_regret.sh goldens TAG ARCH
# which is: cmake --build build_TAG_ARCH --target ninferno_selector_golden_test -j8 ;
#           build_TAG_ARCH/.../ninferno_selector_golden_test --update ;
#           git status --short tests/ops/linear/selector_golden ;
#           build_TAG_ARCH/.../ninferno_selector_golden_test   (must pass without --update)
# then commit the changed golden files and facts_index.txt on the PR branch.

set -euo pipefail

MODE_CMD=${1:?usage: $0 regret|goldens TAG ARCH}
TAG=${2:?TAG}
ARCH=${3:?ARCH}
PR_SRC=${PR_SRC:-$(cd "$(dirname "$0")/../.." && pwd)}
MASTER_SRC=${MASTER_SRC:-$PR_SRC/../wt_598_master}
PASSES=${PASSES:-3}
MODES=${MODES:-"a16 a8"}
JOBS=${JOBS:-8}
OUT=${OUT:-regret_598_${TAG}_${ARCH}}
SAMPLES_ARG=()
if [ -n "${SAMPLES:-}" ]; then SAMPLES_ARG=(--samples "$SAMPLES"); fi
TARGETS=(ninferno_route_tune ninferno_linear_generic_test ninferno_route_classes_test)

build_dir() { echo "$1/build_${TAG}_${ARCH}"; }

build() { # SRC [extra targets...]
    local src=$1 b
    shift
    b=$(build_dir "$src")
    if [ ! -f "$b/CMakeCache.txt" ]; then
        cmake -S "$src" -B "$b" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
              -DBUILD_TESTING=ON -DNINFERNO_BUILD_BENCHMARKS=ON
    fi
    cmake --build "$b" --target "${TARGETS[@]}" "$@" -j"$JOBS"
}

find_bin() { find "$(build_dir "$1")" -type f -name "$2" -perm -u+x | head -n1; }

if [ "$MODE_CMD" = goldens ]; then
    build "$PR_SRC" ninferno_selector_golden_test
    BIN=$(find_bin "$PR_SRC" ninferno_selector_golden_test)
    "$BIN" --update
    (cd "$PR_SRC" && git status --short tests/ops/linear/selector_golden &&
     git diff --stat -- tests/ops/linear/selector_golden)
    "$BIN"
    exit 0
fi
[ "$MODE_CMD" = regret ] || { echo "unknown command $MODE_CMD" >&2; exit 2; }

# --- the two trees and their builds -------------------------------------------------------------
if [ ! -d "$MASTER_SRC" ]; then
    git -C "$PR_SRC" fetch origin master
    git -C "$PR_SRC" worktree add --detach "$MASTER_SRC" origin/master
fi
build "$MASTER_SRC"
build "$PR_SRC"
mkdir -p "$OUT"

# --- shapes derived from the code ---------------------------------------------------------------
# Formats: rmma_has_min_term in rmma_min_term.h (GGML_Q4_K -> q4k, GGML_Q4_1 -> q41).
FORMATS=$(sed -n '/rmma_has_min_term/,/^}/p' "$PR_SRC/src/ops/generic/rmma_min_term.h" |
          grep -o 'GGML_Q[0-9_A-Z]*' | sort -u | sed 's/GGML_//; s/_//g' | tr 'A-Z' 'a-z' | tr '\n' ' ')
echo "min-term formats: $FORMATS"
# Widths: the narrow schedules (NT < 4: the factor applies) cover at most 8*NT tokens, so the
# first width past them is T = 17; NT >= 4 tiles start at 8*4 = 32 tokens (FP32 sums, factor 1).
POINTS=$(python3 -I - "$PR_SRC/src/ops/generic/generic_schedules.h" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
block = text[text.index("NINFERNO_GENERIC_RMMA_SCHEDULES"):text.index("// W4A4")]
nts = [int(m.group(1)) for m in re.finditer(r"X\(\d+,\s*\d+,\s*\d+,\s*\d+,\s*(\d+),\s*\d+\)", block)]
narrow = max(n for n in nts if n < 4) * 8
wide = min(n for n in nts if n >= 4) * 8
pts = sorted({narrow + 1, wide} | {t for t in (24, 48, 64, 96, 128, 256) if t > narrow})
print(",".join(map(str, pts)))
PY
)
echo "route-tune points (T >= 17): $POINTS"
# (n, k): the selector golden's shapes (test_selector_golden.cpp kShapes) plus the #501 shape.
SHAPES="$OUT/shapes.txt"
python3 -I - "$FORMATS" > "$SHAPES" <<'PY'
import sys
shapes = [(256, 2048), (1024, 4096), (2048, 1024), (4096, 4096), (6144, 5120), (17408, 5120),
          (5120, 17408), (28672, 8192), (4096, 3840)]  # K a multiple of 256 (K-quant superblock)
for f in sys.argv[1].split():
    for e in ("store", "add", "swiglu"):
        for n, k in shapes:
            if e == "add" and n > 8192:
                continue
            print(f"{f} a16 {e} {n} {k}")
PY
echo "$(wc -l < "$SHAPES") problems"

# --- correctness qualification (the collector marks a candidate ok only with this) ---------------
for src in "$MASTER_SRC" "$PR_SRC"; do
    (cd "$(build_dir "$src")" &&
     ctest -R 'ninferno_linear_generic_test|ninferno_route_classes_test' --output-on-failure)
done
QUALIFIED="ninferno_linear_generic_test, ninferno_route_classes_test passed on this build and board"

# --- whole runs interleaved: master p1, PR p1, master p2, PR p2 ... -------------------------------
for mode in $MODES; do
    extra=()
    if [ "$mode" = a8 ]; then extra=(--a8-candidates 1); fi
    for p in $(seq 1 "$PASSES"); do
        for who in master pr; do
            src=$MASTER_SRC
            if [ "$who" = pr ]; then src=$PR_SRC; fi
            mkdir -p "$OUT/$who/$mode"
            "$(find_bin "$src" ninferno_route_tune)" --shapes "$SHAPES" --out /dev/null \
                --points "$POINTS" ${SAMPLES_ARG[@]+"${SAMPLES_ARG[@]}"} ${extra[@]+"${extra[@]}"} \
                --routes-out "$OUT/$who/$mode/p$p.csv" --selected-out "$OUT/$who/$mode/sel_p$p.csv"
        done
    done
    for who in master pr; do
        src=$MASTER_SRC
        if [ "$who" = pr ]; then src=$PR_SRC; fi
        d="$OUT/$who/$mode"
        python3 "$PR_SRC/tools/bench/regret_collect.py" \
            --selected $(for p in $(seq 1 "$PASSES"); do echo "$d/sel_p$p.csv"; done) \
            --routes   $(for p in $(seq 1 "$PASSES"); do echo "$d/p$p.csv"; done) \
            --board "$TAG" --model synthetic-grid-598 \
            --selector-revision "$(git -C "$src" rev-parse --short HEAD)" \
            --build-id "$who-${TAG}-${ARCH}" --holdout model --fit-boards "" --fit-models "" \
            --qualified "$QUALIFIED" > "$d/selection.json"
        python3 "$PR_SRC/tools/bench/selection_regret.py" "$d/selection.json" > "$d/report.json" || true
    done
done

# --- master vs PR ------------------------------------------------------------------------------
python3 -I - "$PR_SRC/tools/bench" "$OUT" $MODES <<'PY'
import json, statistics, sys
from collections import defaultdict
sys.path.insert(0, sys.argv[1])
from selection_regret import score_case, summarize
out, modes = sys.argv[2], sys.argv[3:]


def band(t):
    return "T17-32" if t <= 32 else "T33-64" if t <= 64 else "T65+"


for mode in modes:
    docs = {w: json.load(open(f"{out}/{w}/{mode}/selection.json")) for w in ("master", "pr")}
    rows = {w: {c["id"]: score_case(c) for c in d["cases"]} for w, d in docs.items()}
    # The kernels are the same code; only the pick differs.  Each route's time is the median of
    # both builds' medians, and both picks are scored against that one table (pooled), beside
    # each build's own report.
    cls = defaultdict(lambda: {"master": [], "pr": []})
    moved = []
    for cid in sorted(rows["master"]):
        m, p = rows["master"][cid], rows["pr"].get(cid)
        if p is None or m["status"] != "complete" or p["status"] != "complete":
            continue
        pooled = {r: statistics.median([m["median_ms"][r], p["median_ms"][r]])
                  for r in m["median_ms"] if r in p["median_ms"]}
        if m["selected"] not in pooled or p["selected"] not in pooled:
            continue
        best = min(pooled.values())
        reg = {"master": pooled[m["selected"]] / best - 1, "pr": pooled[p["selected"]] / best - 1}
        fmt, _, epi, n, k, t = cid.split(".")
        key = f"{fmt}/{epi}/{band(int(t[1:]))}"
        for w in reg:
            cls[key][w].append({"regret": reg[w], "status": "complete", "passes": 1})
            cls["TOTAL"][w].append({"regret": reg[w], "status": "complete", "passes": 1})
        if m["selected"] != p["selected"]:
            moved.append((cid, m["selected"], p["selected"], reg["master"], reg["pr"]))
    print(f"\n== {out}  mode {mode}: regret = selected / pooled best - 1 ==")
    print(f"{'class':26} {'cases':>5} | {'master geo':>10} {'p95':>6} {'max':>6} | "
          f"{'PR geo':>10} {'p95':>6} {'max':>6}")
    for key in sorted(cls, key=lambda k: (k == "TOTAL", k)):
        s = {w: summarize(cls[key][w]) for w in ("master", "pr")}
        print(f"{key:26} {s['master']['scored_cases']:5d} | "
              f"{s['master']['geomean_slowdown']:10.4f} {s['master']['p95_regret']:6.3f} "
              f"{s['master']['max_regret']:6.3f} | {s['pr']['geomean_slowdown']:10.4f} "
              f"{s['pr']['p95_regret']:6.3f} {s['pr']['max_regret']:6.3f}")
    for w in ("master", "pr"):
        r = json.load(open(f"{out}/{w}/{mode}/report.json"))["summary"]
        print(f"own report {w:6}: geomean {r['geomean_slowdown']:.4f} p95 {r['p95_regret']:.3f} "
              f"max {r['max_regret']:.3f} ({r['scored_cases']}/{r['cases']} scored)")
    print(f"moved picks: {len(moved)} of {len(rows['master'])} cases "
          f"(better {sum(1 for x in moved if x[4] < x[3] - 0.005)}, "
          f"worse {sum(1 for x in moved if x[4] > x[3] + 0.005)})")
    for cid, a, b, ra, rb in moved:
        print(f"  {cid:46} {a:>18} -> {b:<18} regret {ra:6.3f} -> {rb:6.3f}")
PY
