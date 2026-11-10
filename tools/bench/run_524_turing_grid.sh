#!/usr/bin/env bash
# #524, DECISIONS 74: the forced-width grid and the A/B that qualify the RTX 2080 Ti (Turing, sm_75)
# for the column tile of least work. Run it on an idle RTX 2080 Ti, nothing else on the GPU.
#
# BENCH-ONLY HOOK. The widths are forced by tools/bench/524_grid_hook.patch, a patch of
# src/ops/generic/ggml_mmq/ggml_mmq_internal.cuh that this script applies to the tree for the build
# and reverts straight after. It is never merged into the kernel and is not an engine option. The
# script checks that the patch applies at the current tree (and that the tree is clean) BEFORE it
# builds; after an edit of that header the patch has to be regenerated, the script says so.
#
# PREDICTION, to verify (an assumption, not a derivation): f = 0.20 for Turing, the best range
# 0.15-0.25; the grid decides.
#   - Basis: f 0.20 is the value MEASURED on GA10x (RTX 3060 0.15-0.30, RTX 3090 0.15-0.20) and on Ada
#     (0.15-0.20), assumed equal for Turing because two facts match: the int8 tensor rate per SM
#     (generation.h s8 2048 on both rows; 215 TOPS at 68 SMs x 1.545 GHz) and the K-quant decode on
#     the same 64 int32 lanes. Those two facts do not fix f: Blackwell (GB20x) has the same s8 2048
#     and runs the same Q4_K / Q6_K tile configuration, and measured 0.10 (best 0.05-0.15). So the
#     transfer is the weakest step; the grid scans f 0.05-0.43 and decides, and the A/B ranges below
#     are the bracket of the GA10x and Ada results, not a derivation.
#   - Shared memory: 64 KiB on sm_75. By core/mmq_columns.h (tile_shared_bytes, pinned in
#     tests/test_choice.cpp) a Q4_K / Q6_K tile is 38 912 B of weight rows + 144 B x J of y (rounded
#     up to 1 KiB) + 4 B x J of ids: 42 048 B (J 16), 44 160 (32), 46 272 (48), 48 384 (64),
#     51 520 (80), 53 632 (96), 55 744 (112), 57 856 B (J 128). All eight widths fit 64 KiB. Two
#     blocks fit sm_86's 100 KiB per SM up to J 64 (48 384 B), not from J 80 (51 520 B); Turing holds
#     one block per SM at every width. So J 16, 32, 48 and 64 (J 64 is a width the table picks)
#     lose the second block's latency hiding here, which can raise f for the narrow tiles.
#   - Wave quantization on 68 SMs: a 4096-row product has 32 row tiles, so T = 129..256 on two
#     tiles of 80 / 96 / 128 columns is 64 CTAs = 94 % of one wave (stream-k keeps one block per
#     tile above 90 %); three 64-column tiles is 96 CTAs = 1.41 waves. GA10x's 82 SMs run 64 CTAs
#     at 78 % (stream-k splits K). So the saved padding may convert into time more cleanly here than
#     on the RTX 3090.
#   - Q2_K (not in the grid: the table's formats are Q4_K / Q6_K): its 128-column tile is 70 144 B
#     > 64 KiB, widest_tile (core/mmq_columns.h, #596) = 64; Q4_K and Q6_K fit all eight widths, so
#     this grid has no width cut. Q2_K keeps the fewest tiles.
#   Predicted A/B if f is 0.20 (the GA10x shape): 144..160 tokens -14..-19 %, 176..192 -9..-13 %,
#   208..256 within 0.5 %.
#
# MEASUREMENT (reconstructed from DECISIONS 74 and PR #597; the grids of those PRs ran with
# temporary hooks that were not committed, tools/bench/524_grid_hook.patch is that hook again):
# Qwen3-8B Q4_K_M, ninferno-serve --no-prefix-reuse --max-concurrency 1, 8 forced widths
# (16 32 48 64 80 96 112 128) x 21 prompt sizes (16..512 tokens, one chunk), 4 requests a point,
# thinking off, metric = the server's prefill time (timings.prompt_ms of a 1-token chat completion,
# median, ms) with the prompt size checked per request (timings.prompt_n). The regret of the model's
# picks (core/mmq_columns.h rule at each f, widths 16 32 64 80 96 128) against the best of all eight
# widths, mean over the sizes, is printed for f 0.05..0.43 and for the fewest tiles. The A/B is the
# same binary with NINFERNO_MMQ_MODE=old (fewest tiles) and =new (work table at F_AB), old new old
# new old new, 4 requests a size per visit; its table shows the spread of the visits.
# One result file per visit (grid_J<w>.jsonl, ab_<round>_<mode>.jsonl): a directory that already
# holds result files is refused so that no earlier run mixes in; use a new OUT for a new run.
#
# Usage (from a clean checkout of the repository, idle RTX 2080 Ti):
#   MODEL=/path/Qwen3-8B-Q4_K_M.gguf tools/bench/run_524_turing_grid.sh [all|build|grid|ab]
# Env: MODEL (required), OUT (results dir, default profiles/524_turing), BUILD (default build_524t),
# PORT (18524), F_AB (A/B f, default 0.20; set it to the grid's best), JOBS (8), FORCE=1 (skip the
# busy-GPU check). `grid` and `ab` alone need BUILD to hold the hooked binary.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PHASE=${1:-all}
MODEL=${MODEL:?set MODEL to the Qwen3-8B Q4_K_M model}
OUT=${OUT:-profiles/524_turing}
BUILD=${BUILD:-build_524t}
PORT=${PORT:-18524}
F_AB=${F_AB:-0.20}
JOBS=${JOBS:-8}
PATCH=tools/bench/524_grid_hook.patch
PY=${PY:-python3}
BIN="$BUILD/apps/ninferno-serve"
HOOKED=0
SERVER_PID=

case "$PHASE" in build | grid | ab | all) ;; *) echo "usage: $0 [all|build|grid|ab]"; exit 2 ;; esac

cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $name sm_${cc/./}"
[ "$cc" = "7.5" ] || { echo "not a Turing card (compute capability $cc)"; exit 1; }
if [ "${FORCE:-0}" != 1 ] && [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "the GPU is busy (nvidia-smi compute apps); stop them or FORCE=1"; exit 1
fi

# no stale rows: the phases that write result files refuse a directory that holds earlier ones
refuse_old() { # glob
  if compgen -G "$OUT/$1" > /dev/null; then
    echo "$OUT already holds result files ($1): stale rows would change the answer; use a new OUT"; exit 1
  fi
}
case "$PHASE" in
  grid) refuse_old 'grid_J*.jsonl' ;;
  ab) refuse_old 'ab_*.jsonl' ;;
  all) refuse_old '*.jsonl' ;;
esac
mkdir -p "$OUT"

hook_off() {
  [ "$HOOKED" = 1 ] || return 0
  if git apply -R "$PATCH"; then HOOKED=0; else echo "could not revert the hook: run 'git apply -R $PATCH' by hand"; fi
}
trap 'hook_off; [ -z "${SERVER_PID:-}" ] || kill -- "-$SERVER_PID" 2> /dev/null || true' EXIT

# The patch must apply to this tree, and the tree must be clean, before the (long) build starts.
check_hook() {
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "the working tree is not clean (the hook is applied and reverted with git apply):"
    git status --short --untracked-files=no; exit 1
  fi
  if git apply -R --check "$PATCH" 2> /dev/null; then
    echo "the hook is already applied (an interrupted run?); revert it: git apply -R $PATCH"; exit 1
  fi
  if ! git apply --check "$PATCH"; then
    echo "$PATCH does not apply at $(git rev-parse --short HEAD): the header it patches,"
    echo "src/ops/generic/ggml_mmq/ggml_mmq_internal.cuh, changed since the patch was made."
    echo "Regenerate the patch (same content, current lines); do not build without it."
    exit 1
  fi
}

# `grid` / `ab` alone: the binary must be the hooked one, otherwise the variables are ignored and
# all eight "widths" would measure the same kernel launch.
need_hooked_binary() {
  [ -x "$BIN" ] || { echo "$BIN not found: run the build phase first (BUILD=$BUILD)"; exit 1; }
  grep -qa NINFERNO_MMQ_MODE "$BIN" || {
    echo "$BIN is not the hooked build (no NINFERNO_MMQ_MODE in it): the forced widths would be ignored"; exit 1; }
}

build() {
  check_hook
  git apply "$PATCH"; HOOKED=1
  cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 > "$OUT/cmake.log" 2>&1 ||
    { tail -20 "$OUT/cmake.log"; exit 1; }
  cmake --build "$BUILD" --target ninferno-serve -j "$JOBS" > "$OUT/build.log" 2>&1 ||
    { tail -30 "$OUT/build.log"; exit 1; }
  hook_off
  git diff --quiet || { echo "the tree is not clean after unhooking"; exit 1; }
  need_hooked_binary
}

# start_server ENV...   runs the server with the given environment, waits for /v1/models
start_server() {
  env "$@" setsid "$BIN" "$MODEL" --port "$PORT" --no-prefix-reuse \
    --max-concurrency 1 --max-context 2048 > "$OUT/server.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && return 0
    kill -0 "$SERVER_PID" 2> /dev/null || { echo "server died"; tail -20 "$OUT/server.log"; exit 1; }
    sleep 2
  done
  echo "server did not come up"; tail -20 "$OUT/server.log"; exit 1
}
stop_server() { kill -- "-$SERVER_PID" 2> /dev/null || true; wait "$SERVER_PID" 2> /dev/null || true; SERVER_PID=; sleep 3; }

# measure LABEL FILE [options]: one visit; a failure shows the server log's tail
measure() {
  local label=$1 file=$2
  shift 2
  $PY tools/bench/mmq_width_grid.py measure --port "$PORT" --label "$label" --out "$file" "$@" || {
    echo "measurement $label failed (see above); server log tail:"; tail -20 "$OUT/server.log"; exit 1; }
}

grid() {
  need_hooked_binary
  for J in 16 32 48 64 80 96 112 128; do
    start_server NINFERNO_MMQ_MODE=new NINFERNO_MMQ_J=$J
    measure "J$J" "$OUT/grid_J$J.jsonl"
    stop_server
  done
  $PY tools/bench/mmq_width_grid.py grid "$OUT"/grid_J*.jsonl | tee "$OUT/grid.txt"
}

ab() {
  need_hooked_binary
  for round in 1 2 3; do
    for mode in old new; do
      start_server NINFERNO_MMQ_MODE=$mode NINFERNO_MMQ_F="$F_AB"
      measure "$mode" "$OUT/ab_${round}_$mode.jsonl" \
        --sizes 96,128,136,144,160,176,192,208,224,240,256,264,288,320,384,448,512
      stop_server
    done
  done
  $PY tools/bench/mmq_width_grid.py ab "$OUT"/ab_*.jsonl | tee "$OUT/ab.txt"
}

case "$PHASE" in
  build) build ;;
  grid) grid ;;
  ab) ab ;;
  all) build; grid; ab ;;
esac
echo "results in $OUT (grid.txt, ab.txt, *.jsonl); commit nothing from the hooked build"
