#!/usr/bin/env bash
# #524, DECISIONS 74: the forced-width grid and the A/B that qualify the RTX 2080 Ti (Turing, sm_75)
# for the column tile of least work. Run it on an idle RTX 2080 Ti, nothing else on the GPU. Another
# sm_75 board (T4, RTX 2080, ...) is accepted: its name is printed and kept in OUT/gpu.txt, and the
# notes below that use the RTX 2080 Ti's 68 SMs then do not apply; the result is that board's.
#
# BENCH-ONLY HOOK. The widths are forced by tools/bench/524_grid_hook.patch, a patch of
# src/ops/generic/ggml_mmq/ggml_mmq_internal.cuh (the width choice) and ggml_mmq.cu (the stream-k fixup
# buffer is sized for J 128, so a forced wide tile on a narrow call cannot write past it) that this
# script applies to the tree for the build and reverts straight after. It is never merged into the kernel and is not an engine option. The
# script checks that the patch applies at the current tree (and that the tree is clean) BEFORE it
# builds; after an edit of those files the patch has to be regenerated, the script says so.
#
# PREDICTION, to verify (an assumption, not a derivation): f = 0.20 for Turing, best range 0.15-0.25;
# the grid decides. Nothing in the kit is tuned to the data: every predicted value is printed beside the
# measured one, and what would REFUTE f = 0.20 is fixed here, before any data exists
# (tools/bench/mmq_width_grid.py, CRITERIA, prints the same text at the top of the grid output):
#   R5 grid: f MEASURED by a least-squares fit of the work law T = C_N + a tiles (1 - J/128)
#      + b tiles J/128 over all cells (f = a / b, one constant per size, relative residuals), with a
#      95 % bootstrap interval over the width visits (fixed seed), against the predicted RANGE
#      0.15..0.25: the whole interval outside refutes, the whole interval inside stands, an interval
#      across an edge is undecided. When the law does not fit (lack-of-fit ratio above 3: the residual,
#      after one fitted offset per server run, against the cells' own request noise), R5 gives no verdict
#      on f.
#   R2 grid: more than 25 % of the sizes have a width picked at f 0.20 that is more than 2 % slower
#      than the best of the six production widths (J 48 / J 112 are measured, their gains are shown
#      apart: they win some sizes by padding whatever f is, and the table cannot pick them).
#   R3 grid: the regret at f 0.20 is not below the regret of the fewest tiles.
#   R4 ab: the mix change lies more than 2 pp outside the predicted mix range.
#   R1 grid (a report, not a verdict): the regret at f 0.20 against the least over the f buckets;
#      neighbouring buckets' tables differ at one to three sizes, so it cannot separate 0.10 from 0.30.
#   Any one of R2..R5 met refutes f = 0.20 for this card; the thresholds, the range, the interval
#   level and the lack-of-fit bound are a priori, not fitted.
#   - Basis: f 0.20 is a value measured on other cards, assumed equal for Turing because two facts match:
#     the int8 tensor rate per SM (generation.h s8 2048 on both rows; 215 TOPS at 68 SMs x 1.545 GHz)
#     and the K-quant decode on the same 64 int32 lanes. The measured cards split two against two:
#     RTX 3090 and RTX 4070 Ti best at 0.15-0.20, RTX 3060 and RTX 4090 best at 0.25-0.30, the buckets
#     around 0.20 differing by 0.07-0.28 pp of mean regret (DECISIONS 74). The two facts do not fix f
#     either: Blackwell (GB20x) has the same s8 2048 and the same Q4_K / Q6_K tile configuration and was
#     best at 0.10 alone (0.06 % against 0.31 % at 0.05 and 0.45 % at 0.15-0.20). So the transfer is
#     the weakest step; the grid fits f to all cells and prints it with its interval beside 0.20.
#   - Shared memory: 64 KiB on sm_75. By core/mmq_columns.h (tile_shared_bytes; the J 128 values are
#     pinned in tests/test_choice.cpp) a Q4_K / Q6_K tile is 38 912 B of weight rows + 144 B x J of y
#     (rounded up to 1 KiB) + 4 B x J of ids: 42 048 B (J 16), 44 160 (32), 46 272 (48), 48 384 (64),
#     51 520 (80), 53 632 (96), 55 744 (112), 57 856 B (J 128). All eight widths fit 64 KiB. Two
#     blocks fit sm_86's 100 KiB per SM up to J 64 (48 384 B), not from J 80 (51 520 B); Turing holds
#     one block per SM at every width (shared memory only; registers per thread are not checked). So
#     J 16, 32, 48 and 64 (J 64 is a width the table picks) lose the second block's latency hiding
#     here, which can raise f for the narrow tiles.
#   - Wave quantization on 68 SMs: a 4096-row product has 32 row tiles, so two column tiles are 64 CTAs
#     = 94 % of one wave, for 2 x 128 and for 2 x 80 / 96 alike: at 129..192 columns the saving is
#     padding, not waves. GA10x's 82 SMs run 64 CTAs at 78 % (stream-k splits K).
#   - Q2_K (not in the grid: the table's formats are Q4_K / Q6_K): its 128-column tile is 70 144 B
#     > 64 KiB, widest_tile (core/mmq_columns.h, #596) = 64; Q4_K and Q6_K fit all eight widths, so
#     this grid has no width cut. Q2_K keeps the fewest tiles.
#   Predicted A/B if f is 0.20 (the bracket of the measured cards, in COLUMNS, not a derivation):
#   144..160 columns -14..-19 %, 176..192 -9..-13 %, 208..256 within 0.5 % (the group edges 168 / 200
#   are midpoints between the sample sizes). The A/B's new arm does not move the host cost model
#   (generic_cost.cpp, choice.cpp): route choice stays on the production table.
#
# SIZES ARE PROMPT TOKENS, THE KERNEL SEES COLUMNS = TOKENS - k. The prefill is split at execution
# boundaries (prompt_layout.cpp), so an N-token prompt runs a body call of N - k columns plus small
# tail calls. k is the env K (default 7, an ASSUMPTION that is not measured); the `probe` phase prints
# the interval of k that the time step at 128 columns implies (and `all` applies it, see Usage).
# Every pick, tile count and prediction uses N - k and the output header says so. Sizes sit at least
# 4 columns from a tile edge of the widths 64 80 96 128 at k = 7 (136 and 264 were moved to 142 and 270).
# `prompt_ms` holds a width-independent part (tail calls, the first token), which dilutes the
# percentages. The older grids' N (DECISIONS 74) were columns.
#
# MEASUREMENT (reconstructed from DECISIONS 74 and PR #597; the grids of those PRs ran with
# temporary hooks that were not committed, tools/bench/524_grid_hook.patch is that hook again):
# Qwen3-8B Q4_K_M, ninferno-serve --no-prefix-reuse --max-concurrency 1 --prefill-chunk 1024 (above the
# largest size; confirm in the server log that no size was split), 8 forced widths
# (16 32 48 64 80 96 112 128) x 21 prompt sizes (32..512 tokens) x 2 visits, 4 requests a point, thinking off,
# metric = the server's prefill time (timings.prompt_ms of a 1-token chat completion, median, ms)
# with the prompt size checked per request (timings.prompt_n). A size that cannot be built is skipped
# and reported (the readers then need it skipped at every width). f is fitted to all cells with its
# interval (R5); the regret of the model's picks (core/mmq_columns.h rule at each f, widths 16 32 64 80
# 96 128) against the best of those six widths, mean over the sizes, is printed per f bucket, at f 0.20
# and for the fewest tiles, with the predicted width per size beside the measured best. Every width
# is measured twice, one server run each, in counterbalanced order (16..128, then 128..16), so that a
# drift in time (clocks, temperature, server start) does not line up with J; the fit's interval is a
# bootstrap over these 16 width runs. The A/B is the same binary with NINFERNO_MMQ_MODE=old (fewest
# tiles) and =new (work table at F_AB), counterbalanced by round (old new, new old, old new), 4
# requests a size per visit; its table shows the spread of the visits and the predicted range beside
# each change. One result file per visit (grid_J<w>_v<1|2>.jsonl, ab_<round>_<mode>.jsonl): a
# directory that already holds result files is refused so that no earlier run mixes in; use a new OUT
# for a new run.
#
# Usage (from a clean checkout of the repository, idle RTX 2080 Ti):
#   MODEL=/path/Qwen3-8B-Q4_K_M.gguf tools/bench/run_524_turing_grid.sh [all|build|probe|grid|ab]
# `all` = build, probe, grid, ab; after the probe it takes K from the probe when the probe gives one
# value, and stops when K lies outside the probed interval or the probe is not clear. The first GPU is
# used (set CUDA_VISIBLE_DEVICES to one index or UUID for another: the GPU checks query that device,
# and CUDA_DEVICE_ORDER=PCI_BUS_ID makes CUDA's index the one nvidia-smi shows).
# Env: MODEL (required), OUT (results dir, default profiles/524_turing), BUILD (default build-524t),
# PORT (18524; it must be free: a server that already answers is refused), K (tokens - columns,
# default 7, assumed), F_AB (A/B f, default 0.20; set it to the grid's best), JOBS (8), FORCE=1 (skip
# the busy-GPU check), PY (the Python command, default python3). `probe`, `grid` and `ab` alone need
# BUILD to hold the hooked binary.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PHASE=${1:-all}
MODEL=${MODEL:?set MODEL to the Qwen3-8B Q4_K_M model}
OUT=${OUT:-profiles/524_turing}
BUILD=${BUILD:-build-524t}
PORT=${PORT:-18524}
K=${K:-7}
[[ "$K" =~ ^[0-9]+$ ]] || { echo "K must be a whole number of tokens (got $K)"; exit 2; }
PREFILL_CHUNK=1024
F_AB=${F_AB:-0.20}
JOBS=${JOBS:-8}
PATCH=tools/bench/524_grid_hook.patch
PY=${PY:-python3}
BIN="$BUILD/apps/ninferno-serve"
HOOKED=0
SERVER_PID=
SLOG=

case "$PHASE" in build | probe | grid | ab | all) ;; *) echo "usage: $0 [all|build|probe|grid|ab]"; exit 2 ;; esac

# The server takes visible device 0: the first entry of CUDA_VISIBLE_DEVICES (an index or a UUID), or
# device 0. With PCI_BUS_ID order CUDA's index is nvidia-smi's, so the checks below query that board.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
DEV=${CUDA_VISIBLE_DEVICES:-0}
DEV=${DEV%%,*}
[ -n "$DEV" ] || { echo "CUDA_VISIBLE_DEVICES is set but empty: no device is visible"; exit 1; }
cc=$(nvidia-smi -i "$DEV" --query-gpu=compute_cap --format=csv,noheader)
name=$(nvidia-smi -i "$DEV" --query-gpu=name --format=csv,noheader)
echo "GPU $DEV: $name sm_${cc/./}"
[ "$cc" = "7.5" ] || { echo "not a Turing card (compute capability $cc)"; exit 1; }
if [ "${FORCE:-0}" != 1 ] && [ -n "$(nvidia-smi -i "$DEV" --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "the GPU is busy (nvidia-smi compute apps); stop them or FORCE=1"; exit 1
fi

# no stale rows: the phases that write result files refuse a directory that holds earlier ones
refuse_old() { # glob
  if compgen -G "$OUT/$1" > /dev/null; then
    echo "$OUT already holds result files ($1): stale rows would change the answer; use a new OUT"; exit 1
  fi
}
case "$PHASE" in
  probe) refuse_old 'probe.jsonl' ;;
  grid) refuse_old 'grid_J*.jsonl' ;;
  ab) refuse_old 'ab_*.jsonl' ;;
  all) refuse_old '*.jsonl' ;;
esac
mkdir -p "$OUT"
echo "$name (sm_${cc/./}, device $DEV)" > "$OUT/gpu.txt"

hook_off() {
  [ "$HOOKED" = 1 ] || return 0
  if git apply -R "$PATCH"; then HOOKED=0; else echo "could not revert the hook: run 'git apply -R $PATCH' by hand"; fi
}
trap 'hook_off; [ -z "${SERVER_PID:-}" ] || kill -- "-$SERVER_PID" 2> /dev/null || true' EXIT

# The patch must apply to this tree, and the tree must be clean, before the (long) build starts.
check_hook() {
  if git apply -R --check "$PATCH" 2> /dev/null; then
    echo "the hook is already applied (an interrupted run?); revert it: git apply -R $PATCH"; exit 1
  fi
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "the working tree is not clean (the hook is applied and reverted with git apply):"
    git status --short --untracked-files=no; exit 1
  fi
  if ! git apply --check "$PATCH"; then
    echo "$PATCH does not apply at $(git rev-parse --short HEAD): a file it patches"
    echo "(src/ops/generic/ggml_mmq/ggml_mmq_internal.cuh, ggml_mmq.cu) changed since the patch was made."
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

# the port must be free: a server that already answers (an earlier run killed with SIGKILL leaves one
# behind) would be measured instead of the one started here
refuse_busy_port() {
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
    echo "port $PORT already answers: a server is running there; stop it or set PORT"; exit 1
  fi
}

# start_server LABEL ENV...   runs the server with the given environment (log: $OUT/server_LABEL.log),
# waits for /v1/models and requires that the process started here is alive when it answers
start_server() {
  local label=$1
  shift
  refuse_busy_port
  env "$@" setsid "$BIN" "$MODEL" --port "$PORT" --no-prefix-reuse --prefill-chunk "$PREFILL_CHUNK" \
    --max-concurrency 1 --max-context 2048 > "$OUT/server_$label.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; then
      kill -0 "$SERVER_PID" 2> /dev/null && return 0
      echo "something answers on port $PORT but the server started here is gone"; tail -20 "$OUT/server_$label.log"; exit 1
    fi
    kill -0 "$SERVER_PID" 2> /dev/null || { echo "server died"; tail -20 "$OUT/server_$label.log"; exit 1; }
    sleep 2
  done
  echo "server did not come up"; tail -20 "$OUT/server_$label.log"; exit 1
}
# stop the server's process group: TERM, then KILL after 30 s
stop_server() {
  [ -n "${SERVER_PID:-}" ] || return 0
  kill -- "-$SERVER_PID" 2> /dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$SERVER_PID" 2> /dev/null || break
    sleep 1
  done
  kill -0 "$SERVER_PID" 2> /dev/null && { kill -KILL -- "-$SERVER_PID" 2> /dev/null || true; }
  wait "$SERVER_PID" 2> /dev/null || true
  SERVER_PID=
  sleep 3
}

# measure LABEL FILE [options]: one visit; a failure shows the server log's tail
measure() {
  local label=$1 file=$2
  shift 2
  "$PY" tools/bench/mmq_width_grid.py measure --port "$PORT" --label "$label" --out "$file" --k "$K" "$@" || {
    echo "measurement $label failed (see above); server log tail:"; tail -20 "$OUT/server_$SLOG.log"; exit 1; }
}

# the interval of k: width forced to 128, consecutive sizes around the 128-column edge
probe() {
  need_hooked_binary
  SLOG=probe
  start_server probe NINFERNO_MMQ_MODE=new NINFERNO_MMQ_J=128
  measure probe "$OUT/probe.jsonl" --sizes 128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146
  stop_server
  "$PY" tools/bench/mmq_width_grid.py kprobe "$OUT/probe.jsonl" --width 128 | tee "$OUT/probe.txt"
  echo "set K to a value inside the interval (the default $K is assumed) for grid and ab"
}

# `all`: the grid and the A/B run with the probed k. The last line of probe.txt is
# "KPROBE k_lo=<a> k_hi=<b> trusted=<0|1>" (mmq_width_grid.py kprobe). One value: K takes it. An
# interval: K must lie in it. A probe that is not clear, or a K outside: stop before any grid row.
apply_probe() {
  local line lo hi trusted
  line=$(grep '^KPROBE ' "$OUT/probe.txt" || true)
  line=${line##*$'\n'}
  [ -n "$line" ] || { echo "$OUT/probe.txt has no KPROBE line"; exit 1; }
  read -r _ lo hi trusted <<< "$line"
  lo=${lo#k_lo=} hi=${hi#k_hi=} trusted=${trusted#trusted=}
  # k is a whole number of tokens, as K must be
  [[ "$lo" =~ ^[0-9]+$ && "$hi" =~ ^[0-9]+$ && "$trusted" =~ ^[01]$ ]] ||
    { echo "malformed probe line: $line"; exit 1; }
  if [ "$trusted" != 1 ]; then
    echo "the k probe is not clear (see $OUT/probe.txt): set K by hand and run the grid and ab phases"; exit 1
  fi
  if [ "$lo" = "$hi" ]; then
    [ "$K" = "$lo" ] || echo "K $K -> $lo: the probe measured k = $lo"
    K=$lo
  elif [ "$K" -lt "$lo" ] || [ "$K" -gt "$hi" ]; then
    echo "K $K lies outside the probed interval [$lo, $hi]: set K inside it and run the grid and ab phases"; exit 1
  else
    echo "K $K lies inside the probed interval [$lo, $hi]"
  fi
}

# two visits of every width, counterbalanced: 16..128, then 128..16 (one server run per width visit)
grid() {
  need_hooked_binary
  local v J
  for v in 1 2; do
    for J in $([ "$v" = 1 ] && echo 16 32 48 64 80 96 112 128 || echo 128 112 96 80 64 48 32 16); do
      SLOG=J${J}_v$v
      start_server "$SLOG" NINFERNO_MMQ_MODE=new NINFERNO_MMQ_J=$J
      measure "J$J" "$OUT/grid_J${J}_v$v.jsonl"
      stop_server
    done
  done
  "$PY" tools/bench/mmq_width_grid.py grid "$OUT"/grid_J*_v*.jsonl | tee "$OUT/grid.txt"
}

ab() {
  need_hooked_binary
  local round mode
  for round in 1 2 3; do
    # counterbalanced by round: old new, new old, old new (a drift does not line up with the mode)
    for mode in $([ "$round" = 2 ] && echo new old || echo old new); do
      SLOG=${mode}_$round
      start_server "$SLOG" NINFERNO_MMQ_MODE=$mode NINFERNO_MMQ_F="$F_AB"
      measure "$mode" "$OUT/ab_${round}_$mode.jsonl" --sizes ab --f "$F_AB"
      stop_server
    done
  done
  "$PY" tools/bench/mmq_width_grid.py ab "$OUT"/ab_*.jsonl | tee "$OUT/ab.txt"
}

case "$PHASE" in
  build) build ;;
  probe) probe ;;
  grid) grid ;;
  ab) ab ;;
  all) build; probe; apply_probe; grid; ab ;;
esac
echo "results in $OUT (probe.txt, grid.txt, ab.txt, *.jsonl); commit nothing from the hooked build"
