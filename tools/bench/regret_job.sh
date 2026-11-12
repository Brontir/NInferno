#!/usr/bin/env bash
# The regret job (#620 step 10): one board's route selection regret at one revision, as one
# ninferno-regret-run/1 report with its delta against the board's previous committed report.
# Run it on a stand, GPU idle, with a Ninja build freshly made from this checkout (the build
# records no revision: the job refuses a build configured from another source directory or with
# work left to do for its three targets):
#
#   tools/bench/regret_job.sh BUILD_DIR [OUT_DIR]
#   (OUT_DIR default: profiles/bench/regret/<board>_<YYYYMMDDTHHMMSSZ>, relative to the current directory)
#
# Steps (docs/maintainer/route-regret.md#the-regret-job; about 25 min on a lab board):
#   1. qualify: ctest in BUILD_DIR runs ninferno_linear_generic_test (NINFERNO_TEST_FORMATS = the
#      formats of the shapes file) and ninferno_route_classes_test; both must pass, none skipped.
#      Their names are the --qualified statement of the collection.
#   2. sweep: PASSES (3) runs of `ninferno-serve route-tune --shapes SHAPES --samples SAMPLES
#      --a8-candidates 1 [--route-policy ROUTE_POLICY]`, each writing --routes-out and --selected-out.
#   3. score: regret_collect.py, selection_regret.py and regret_bands.py --doc.
#   4. report: regret_report.py build, with the previous committed report of the board (--previous
#      auto: the newest tools/bench/data/route_regret/runs/<board>/*.json at HEAD, else the A5
#      summary <board>.json) and THRESHOLD (0.05). It writes OUT_DIR/<YYYYMMDDTHHMMSSZ>-<rev12>.json
#      and .md; committing it to tools/bench/data/route_regret/runs/<board>/ is a maintainer's step.
#   The card's clocks and persistence mode are recorded, and its SM clock is sampled once a second
#   during the sweep (min and max in the report).
# Environment: SHAPES, PASSES, SAMPLES, ROUTE_POLICY, THRESHOLD, PREVIOUS (a file, auto, none),
# BOARD (default: from the card's PCI id and gpu_facts.json), MODEL (the label of the shapes' source).
# One card: with several, select it (CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<i>).
# The job refuses to start when a compute process is on the card, but that is a snapshot at the
# start: it takes no lock, so a stand runs it through its own queue lock, which holds other GPU
# work off for its 25 minutes.
# Exit: 0 report written (its delta may list regressions: read it), 1 a step failed, 2 usage.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
SHAPES=${SHAPES:-$ROOT/tools/bench/data/route_regret/shapes.txt}
PASSES=${PASSES:-3}
SAMPLES=${SAMPLES:-7}
THRESHOLD=${THRESHOLD:-0.05}
PREVIOUS=${PREVIOUS:-auto}
MODEL=${MODEL:-qwen3.5-9b+gemma4-12b-dense}
PY=${PYTHON:-python3}

die() { echo "regret_job: $*" >&2; exit 1; }
[ $# -ge 1 ] && [ $# -le 2 ] || { sed -n '2,9p' "$0" >&2; exit 2; }
BUILD=$(cd "$1" && pwd) || exit 2
[ -f "$BUILD/CMakeCache.txt" ] || { echo "regret_job: $BUILD is not a build tree" >&2; exit 2; }
case $PASSES in ''|*[!0-9]*|0) echo "regret_job: PASSES must be a positive integer" >&2; exit 2;; esac
case $SAMPLES in ''|*[!0-9]*|0) echo "regret_job: SAMPLES must be a positive integer" >&2; exit 2;; esac

SERVE=$(find "$BUILD" -maxdepth 3 -type f -name ninferno-serve -perm -u+x | head -n 1)
[ -n "$SERVE" ] || die "no ninferno-serve under $BUILD"

# The card: exactly one visible, idle.
GPU_SEL=()
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && GPU_SEL=(-i "$CUDA_VISIBLE_DEVICES")
case ${CUDA_VISIBLE_DEVICES:-} in *,*) false;; esac ||
  die "CUDA_VISIBLE_DEVICES names more than one card"
query() { nvidia-smi ${GPU_SEL[@]+"${GPU_SEL[@]}"} --query-gpu="$1" --format=csv,noheader; }
cards=$(query name | grep -c . || true)
[ "$cards" = 1 ] || die "$cards cards visible: select one (CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<i>)"
busy=$(nvidia-smi ${GPU_SEL[@]+"${GPU_SEL[@]}"} --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
[ "$busy" = 0 ] || die "$busy compute process(es) on the card: the sweep needs it idle"
# pci.device_id is 0xDDDDVVVV (device, vendor)
PCI=$(query pci.device_id | sed -E 's/^0x([0-9A-Fa-f]{4}).*/0x\1/' | tr 'A-F' 'a-f')
DRIVER=$(query driver_version)
GPU_FIELDS=$(nvidia-smi ${GPU_SEL[@]+"${GPU_SEL[@]}"} --format=csv,noheader,nounits \
  --query-gpu=name,power.limit,clocks.max.sm,clocks.applications.graphics,persistence_mode)
board_card=$("$PY" "$HERE/regret_report.py" board "$PCI") || die "PCI id $PCI is not in gpu_facts.json"
BOARD=${BOARD:-${board_card%%$'\t'*}}
CARD=${board_card#*$'\t'}

# The revision: the checkout's HEAD, "-dirty" with uncommitted edits. The build records no
# revision of its own, so it is tied to the checkout instead: its cache must name this checkout as
# its source, and ninja must have nothing left to build for the job's targets (a build of other
# sources, or a stale one, is refused).
REV=$(git -C "$ROOT" rev-parse HEAD)
git -C "$ROOT" diff --quiet HEAD -- || REV=$REV-dirty
cache() { sed -n "s/^$1:[A-Z]*=//p" "$BUILD/CMakeCache.txt"; }
SOURCE=$(cache CMAKE_HOME_DIRECTORY)
[ -n "$SOURCE" ] && [ "$(cd "$SOURCE" 2>/dev/null && pwd -P)" = "$(cd "$ROOT" && pwd -P)" ] ||
  die "$BUILD was configured from '${SOURCE:-?}', not from this checkout ($ROOT)"
TARGETS=(ninferno-serve ninferno_linear_generic_test ninferno_route_classes_test)
[ -f "$BUILD/build.ninja" ] || die "$BUILD is not a Ninja build: its freshness cannot be checked"
NINJA=$(cache CMAKE_MAKE_PROGRAM)   # the ninja the build was configured with, not PATH's
[ -n "$NINJA" ] && [ -x "$NINJA" ] || die "the build's CMAKE_MAKE_PROGRAM '${NINJA:-?}' is not runnable"
"$NINJA" -C "$BUILD" -n "${TARGETS[@]}" 2>&1 | grep -q "no work to do" ||
  die "$BUILD is not up to date with this checkout: cmake --build $BUILD --target ${TARGETS[*]}"
# nvcc: the build's own CUDA compiler (CMakeCache.txt), never the one on PATH.
NVCC_BIN=$(cache CMAKE_CUDA_COMPILER)
if [ -n "$NVCC_BIN" ]; then
  NVCC_SOURCE=CMAKE_CUDA_COMPILER
  NVCC=$({ "$NVCC_BIN" --version 2>/dev/null || true; } |
         sed -n 's/.*release \([0-9.]*\), V\([0-9.]*\).*/\2/p')
  [ -n "$NVCC" ] || die "the build's CUDA compiler $NVCC_BIN did not report its version"
else
  NVCC_SOURCE=none NVCC=unknown
fi
DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
STAMP=${DATE//[-:]/}   # 20261112T165648Z: names sort by time
OUT=${2:-profiles/bench/regret/${BOARD}_$STAMP}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
echo "regret_job: $CARD ($PCI) board $BOARD, $REV, nvcc $NVCC, driver $DRIVER -> $OUT"

# 1. qualify
FORMATS=$(awk '!/^[ \t]*(#|$)/ { print $1 }' "$SHAPES" | sort -u | paste -sd, -)
[ -n "$FORMATS" ] || die "no shapes in $SHAPES"
TESTS='^(ninferno_linear_generic_test|ninferno_route_classes_test)$'
if ! NINFERNO_TEST_FORMATS=$FORMATS ctest --test-dir "$BUILD" -R "$TESTS" --no-tests=error \
     --output-on-failure > "$OUT/qualify.log" 2>&1; then
  die "qualification failed ($OUT/qualify.log)"
fi
grep -q "100% tests passed, 0 tests failed out of 2" "$OUT/qualify.log" &&
  ! grep -q "did not run" "$OUT/qualify.log" ||
  die "qualification did not run both tests to a pass ($OUT/qualify.log)"
QUALIFIED="ninferno_linear_generic_test (NINFERNO_TEST_FORMATS=$FORMATS) and ninferno_route_classes_test passed on this build and board, $DATE"

# 2. sweep
POLICY_ARGS=()
[ -n "${ROUTE_POLICY:-}" ] && POLICY_ARGS=(--route-policy "$ROUTE_POLICY")
ROUTES=() SELECTED=()
# The SM clock once a second over the passes, for the report's min and max.
nvidia-smi ${GPU_SEL[@]+"${GPU_SEL[@]}"} --query-gpu=clocks.sm --format=csv,noheader,nounits \
  -lms 1000 > "$OUT/sm_clock.log" 2>/dev/null &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT
for p in $(seq 1 "$PASSES"); do
  echo "regret_job: pass $p of $PASSES"
  "$SERVE" route-tune --shapes "$SHAPES" --samples "$SAMPLES" --a8-candidates 1 \
    ${POLICY_ARGS[@]+"${POLICY_ARGS[@]}"} \
    --out /dev/null --routes-out "$OUT/routes_p$p.csv" --selected-out "$OUT/sel_p$p.csv" \
    > "$OUT/tune_p$p.log" 2>&1 || die "route-tune pass $p failed ($OUT/tune_p$p.log)"
  ROUTES+=("$OUT/routes_p$p.csv") SELECTED+=("$OUT/sel_p$p.csv")
done
kill "$SAMPLER" 2>/dev/null || true
wait "$SAMPLER" 2>/dev/null || true
trap - EXIT   # the sampler is gone: its PID must not be killed again at exit

# 3. score
"$PY" "$HERE/regret_collect.py" --selected "${SELECTED[@]}" --routes "${ROUTES[@]}" \
  --board "$BOARD" --model "$MODEL" --selector-revision "$REV" \
  --build-id "regret-${REV:0:12}-$BOARD" --holdout board --qualified "$QUALIFIED" --pci-id "$PCI" \
  > "$OUT/regret.json" || die "regret_collect.py failed"
rc=0
"$PY" "$HERE/selection_regret.py" "$OUT/regret.json" > "$OUT/selection_report.json" || rc=$?
[ "$rc" -le 1 ] || die "selection_regret.py failed"   # 1: some case not scored, still a report
"$PY" "$HERE/regret_bands.py" --doc "$OUT/regret.json" --json "$OUT/bands.json" ||
  die "regret_bands.py failed"

# 4. report
NAME=$STAMP-${REV:0:12}
"$PY" "$HERE/regret_report.py" build --selection-report "$OUT/selection_report.json" \
  --bands "$OUT/bands.json" --board "$BOARD" --card "$CARD" --revision "$REV" --date "$DATE" \
  --nvcc "$NVCC" --nvcc-source "$NVCC_SOURCE" --driver "$DRIVER" --gpu "$GPU_FIELDS" \
  --sm-clock-log "$OUT/sm_clock.log" --pci-id "$PCI" \
  --qualification "$QUALIFIED" --shapes "$SHAPES" --passes "$PASSES" --samples "$SAMPLES" \
  --route-policy "${ROUTE_POLICY:-}" --previous "$PREVIOUS" --repo "$ROOT" \
  --threshold "$THRESHOLD" --out "$OUT/$NAME.json" --md "$OUT/$NAME.md" ||
  die "regret_report.py failed"
cat "$OUT/$NAME.md"
echo "regret_job: to commit: tools/bench/data/route_regret/runs/$BOARD/$NAME.json (from $OUT)"
