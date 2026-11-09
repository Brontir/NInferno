#!/usr/bin/env bash
# #524, DECISIONS 74: the forced-width grid and the A/B that qualify the RTX 2080 Ti (Turing, sm_75)
# for the column tile of least work. Run it ON the stand (l2080), nothing else on the GPU.
#
# PREDICTION (from architecture facts, written before any measurement; to verify):
#   f = 0.20 for Turing, the best range 0.15-0.25; the grid decides.
#   - The part in J is the int8 MMA plus the y tile's loads. Turing's int8 tensor rate is GA10x's
#     per SM (2048 ops = 1024 MACs a clock: generation.h s8 2048 on both rows; 215 TOPS at 68 SMs
#     x 1.545 GHz), and the K-quant decode runs on the same 64 int32 lanes, so the split of a
#     tile's step between a fixed part (the 128 weight rows' decode) and the part in J is GA10x's:
#     f 0.20 (RTX 3060 0.15-0.30, RTX 3090 0.15-0.20), Ada's 0.15-0.20 as well.
#   - Against GA10x two facts push f up, not down: no cp.async on sm_75 (the weight tile's loads are
#     synchronous and exposed per K step, a fixed cost); 64 KiB of shared memory: a Q4_K / Q6_K tile
#     is 43 008 B of rows + 144 B x J of y + ids = 45.4 KB (J 16) .. 62.0 KB (J 128), one block per
#     SM from J 64 up on every card, but J 16 / 32 (two blocks on GA10x's 100 KiB) are one block
#     here, so narrow tiles lose the second block's latency hiding. That only affects 16 / 32,
#     which the work table rarely picks above 48 tokens.
#   - Wave quantization on 68 SMs: a 4096-row product has 32 row tiles, so T = 129..256 on two
#     tiles of 80 / 96 / 128 columns is 64 CTAs = 94 % of one wave (stream-k keeps one block per
#     tile above 90 %); three 64-column tiles is 96 CTAs = 1.41 waves. GA10x's 82 SMs run 64 CTAs
#     at 78 % (stream-k splits K). So the saved padding converts into time more cleanly here than
#     on the RTX 3090, which supports 80 / 96 paying at 136..192 as they do on the 3060 / 3090.
#   - Q2_K (not in the grid: the table's formats are Q4_K / Q6_K): its 128-column tile is 70 144 B
#     > 64 KiB, widest_tile (core/mmq_columns.h, #596) = 64; Q4_K and Q6_K fit all eight widths, so
#     this grid has no width cut. Q2_K keeps the fewest tiles.
#   Predicted A/B on the final f (the GA10x shape, 0.20): 144..160 tokens -14..-19 %, 176..192
#   -9..-13 %, 208..256 within 0.5 %, the 144..256 mix -4..-8 %.
#
# MEASUREMENT (reconstructed from DECISIONS 74 and PR #597; the grids of those PRs ran with
# temporary hooks that were not committed, tools/bench/524_grid_hook.patch is that hook again on
# current master): Qwen3-8B Q4_K_M, ninferno-serve --no-prefix-reuse --max-concurrency 1, 8 forced
# widths (16 32 48 64 80 96 112 128) x 21 prompt sizes (16..512 tokens, one chunk), 4 requests a
# point, metric = time to the first token of a 1-token chat completion (median, ms). The regret of
# the model's picks (core/mmq_columns.h rule at each f, widths 16 32 64 80 96 128) against the best
# of all eight widths, mean over the sizes, is printed for f 0.05..0.43 and for the fewest tiles.
# The A/B is the same binary with NINFERNO_MMQ_MODE=old (fewest tiles) and =new (work table at
# F_AB), old new old new old new, 4 requests a size per visit.
#
# Usage (from the checkout of perf/524-turing-grid on the stand):
#   MODEL=/path/Qwen3-8B-Q4_K_M.gguf tools/bench/run_524_turing_grid.sh [all|build|grid|ab]
# Env: MODEL (required), OUT (results dir, default profiles/524_turing), BUILD (default
# build_524t), PORT (18524), F_AB (A/B f, default 0.20; set it to the grid's best), JOBS (8),
# FORCE=1 (skip the busy-GPU check).
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
mkdir -p "$OUT"

cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $name sm_${cc/./}"
[ "$cc" = "7.5" ] || { echo "not a Turing card (compute capability $cc)"; exit 1; }
if [ "${FORCE:-0}" != 1 ] && [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "the GPU is busy (nvidia-smi compute apps); stop them or FORCE=1"; exit 1
fi

hook_on() { git apply --check "$PATCH" 2> /dev/null && git apply "$PATCH"; }
hook_off() { git apply -R --check "$PATCH" 2> /dev/null && git apply -R "$PATCH" || true; }
trap 'hook_off; [ -z "${SERVER_PID:-}" ] || kill -- "-$SERVER_PID" 2> /dev/null || true' EXIT

build() {
  hook_on
  cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 > "$OUT/cmake.log"
  cmake --build "$BUILD" --target ninferno-serve -j "$JOBS" > "$OUT/build.log" 2>&1
  hook_off
  git diff --quiet || { echo "the tree is not clean after unhooking"; exit 1; }
}

# start_server ENV...   runs the server with the given environment, waits for /v1/models
start_server() {
  env "$@" setsid "$BUILD/ninferno-serve" "$MODEL" --port "$PORT" --no-prefix-reuse \
    --max-concurrency 1 --max-context 2048 > "$OUT/server.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && return 0
    kill -0 "$SERVER_PID" 2> /dev/null || { echo "server died"; tail -20 "$OUT/server.log"; exit 1; }
    sleep 2
  done
  echo "server did not come up"; exit 1
}
stop_server() { kill -- "-$SERVER_PID" 2> /dev/null || true; wait "$SERVER_PID" 2> /dev/null || true; SERVER_PID=; sleep 3; }

grid() {
  for J in 16 32 48 64 80 96 112 128; do
    start_server NINFERNO_MMQ_MODE=new NINFERNO_MMQ_J=$J
    $PY tools/bench/mmq_width_grid.py measure --port "$PORT" --label "J$J" --out "$OUT/grid.jsonl"
    stop_server
  done
  $PY tools/bench/mmq_width_grid.py grid "$OUT/grid.jsonl" | tee "$OUT/grid.txt"
}

ab() {
  for round in 1 2 3; do
    for mode in old new; do
      start_server NINFERNO_MMQ_MODE=$mode NINFERNO_MMQ_F="$F_AB"
      $PY tools/bench/mmq_width_grid.py measure --port "$PORT" --label "$mode" --out "$OUT/ab.jsonl" \
        --sizes 96,128,136,144,160,176,192,208,224,240,256,264,288,320,384,448,512
      stop_server
    done
  done
  $PY tools/bench/mmq_width_grid.py ab "$OUT/ab.jsonl" | tee "$OUT/ab.txt"
}

case "$PHASE" in
  build) build ;;
  grid) grid ;;
  ab) ab ;;
  all) build; grid; ab ;;
  *) echo "usage: $0 [all|build|grid|ab]"; exit 2 ;;
esac
echo "results in $OUT (grid.txt, ab.txt, *.jsonl); commit nothing from the hooked build"
