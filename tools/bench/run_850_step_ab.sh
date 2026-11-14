#!/usr/bin/env bash
# #850 (a), docs/maintainer/kernel-proposals.md entry P, acceptance item 4: the whole Qwen3.5-9B
# decode step, A = master against B = this checkout's HEAD, on one idle card. Plain decode, C = 1,
# a fixed ~22K-token prompt, `--spec none`, greedy, ROUNDS rounds a build in ABBA order.
#
#   tools/bench/run_850_step_ab.sh [OUT]
#   tools/bench/run_850_step_ab.sh --summarize OUT          (no GPU: the summary again)
#
# Run it from the PR's checkout (B is its HEAD, committed: a dirty tree is refused). OUT defaults to
# ${TMPDIR:-/tmp}/r8_850_step_<card>_<MMDD_HHMM>, never the repository. With several cards, pick one
# (CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<i>). A compute process on the card stops the
# script, at the start and before every round (ALLOW_BUSY=1 runs anyway; the figures are then not
# the protocol's).
#
# Steps:
#   1. sources: `git fetch origin master`, then a detached worktree of the merge base of BASE
#      (origin/master) and HEAD, and of HEAD, under BUILD_ROOT/a/src and BUILD_ROOT/b/src (the
#      same layout, so ccache shares B's unchanged objects with A's: CCACHE_BASEDIR=BUILD_ROOT).
#   2. builds: cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=<ARCH>
#      -DBUILD_TESTING=OFF -DNINFERNO_CUDA_LINEINFO=OFF with ccache launchers, target ninferno-serve
#      only, -j JOBS. ARCH is the card's (nvidia-smi compute_cap: 8.9 -> 89, 12.0 -> 120a).
#   3. the model: `ninferno-serve MODEL --quant QUANT --spec none --prepare-only` with each build
#      (fetch and convert once into the Hugging Face cache, no GPU); the rounds then run offline.
#   4. rounds A B B A A B B A ...: each starts that build's server on 127.0.0.1:PORT
#      (--spec none --max-concurrency 1 --max-context MAX_CONTEXT --no-prefix-reuse --greedy),
#      sends one warm-up and REQUESTS times the fixed prompt (tools/bench/step_ab.py: the first
#      CHARS characters of docs/performance's long24k prompt set, ~22.2K tokens; ignore_eos, so
#      every request decodes OUT_TOKENS tokens), reads the server's decode tok/s
#      (timings.predicted_per_second) and stops the server. nvidia-smi samples the SM and memory
#      clocks, temperature and power twice a second through each round (OUT/clocks_<round>.csv).
#   5. OUT/summary.md: per-round decode tok/s, each arm's median and spread, B/A, the noise (half
#      the larger of the A-A and B-B spreads) and the verdict against "no slower": B/A >= 1 - noise.
#      Also whether the greedy outputs of both builds are identical (they should be: the entries
#      are bit-identical schedules).
#   With KEEP_BUILDS=0 (the default) the two build trees and worktrees are removed after a complete
#   run; a failed run keeps them, and a rerun with the same BUILD_ROOT reuses them.
#
# Environment (defaults): MODEL (unsloth/Qwen3.5-9B-GGUF), QUANT (Q4_K_M; its down projections are
# Q6_K and Q4_K; QUANT=Q5_K_M runs the Q5_K down entries), ROUNDS (6 a build: proposal P's frozen
# six repeats, refused otherwise), REQUESTS (2 a round), OUT_TOKENS (512), CHARS (95940: step_ab.py
# refuses a prompt outside 22,184 tokens ±3 %, the frozen workload), MAX_CONTEXT (32768), PORT
# (18850), BASE (origin/master: A is its merge base with HEAD, the master the branch was built on),
# NO_FETCH (0), JOBS (16), ARCH (the card's), BUILD_ROOT (OUT/trees), MIN_FREE_GB (20),
# STARTUP_TIMEOUT (900 s a server start), KEEP_BUILDS (0), ALLOW_BUSY (0), NO_CCACHE (0), PYTHON.
#
# Cost on one stand: two serve-only trees, about 2.5-3 GB each at sm_86/sm_89 and 4-5 GB at sm_120a
# (MIN_FREE_GB 20 also covers the model: the GGUF, 5.7 GB, and its conversion, when not cached),
# 15-30 min cold for A at -j16 and a few minutes for B with ccache; then 12 rounds of 1.5-3 min
# (server start, the warm-up, two 22K prefills and two 512-token decodes).
# Exit: 0 summary written (read its verdict), 1 a step failed, 2 usage, 3 card busy or disk short.
# No GPU UUID or IP is written: the server logs are scrubbed of both.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
py=${PYTHON:-python3}
helper="$here/step_ab.py"
MODEL=${MODEL:-unsloth/Qwen3.5-9B-GGUF}
QUANT=${QUANT:-Q4_K_M}
ROUNDS=${ROUNDS:-6}
REQUESTS=${REQUESTS:-2}
OUT_TOKENS=${OUT_TOKENS:-512}
CHARS=${CHARS:-95940}
MAX_CONTEXT=${MAX_CONTEXT:-32768}
PORT=${PORT:-18850}
BASE=${BASE:-origin/master}
JOBS=${JOBS:-16}
MIN_FREE_GB=${MIN_FREE_GB:-20}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-900}
KEEP_BUILDS=${KEEP_BUILDS:-0}

die() { echo "run_850_step_ab: $*" >&2; exit 1; }
busy_exit() { echo "run_850_step_ab: $*" >&2; exit 3; }

summarize() {
    local out=$1 rounds
    [[ -f "$out/rounds.jsonl" && -f "$out/meta.txt" ]] || die "$out has no rounds.jsonl and meta.txt"
    rounds=$(sed -n 's/^rounds per build //p' "$out/meta.txt" | head -1)
    "$py" "$helper" summarize "$out" ${rounds:+--rounds "$rounds"}
}

if [[ ${1:-} == --summarize ]]; then
    [[ $# -eq 2 ]] || { echo "usage: $0 --summarize OUT" >&2; exit 2; }
    summarize "$2"
    exit 0
fi
[[ $# -le 1 ]] || { sed -n '6,7p' "$0" >&2; exit 2; }
for v in ROUNDS REQUESTS OUT_TOKENS CHARS MAX_CONTEXT PORT JOBS MIN_FREE_GB STARTUP_TIMEOUT; do
    case ${!v} in '' | *[!0-9]* | 0) echo "$v must be a positive integer" >&2; exit 2 ;; esac
done
((ROUNDS == 6)) || { echo "ROUNDS is frozen at 6 a build (kernel proposal P, six repeats)" >&2; exit 2; }
command -v nvidia-smi >/dev/null || die "no nvidia-smi"
command -v curl >/dev/null || die "no curl"
command -v setsid >/dev/null || die "no setsid"
command -v cmake >/dev/null || die "no cmake"

# The card: exactly one visible, idle (as tools/bench/regret_job.sh selects it).
GPU_SEL=()
[[ -n ${CUDA_VISIBLE_DEVICES:-} ]] && GPU_SEL=(-i "$CUDA_VISIBLE_DEVICES")
[[ ${CUDA_VISIBLE_DEVICES:-} != *,* ]] || die "CUDA_VISIBLE_DEVICES names more than one card"
if [[ ${CUDA_VISIBLE_DEVICES:-} == [0-9]* && ${CUDA_DEVICE_ORDER:-} != PCI_BUS_ID ]]; then
    die "a numeric CUDA_VISIBLE_DEVICES needs CUDA_DEVICE_ORDER=PCI_BUS_ID (nvidia-smi's order)"
fi
smi() { nvidia-smi ${GPU_SEL[@]+"${GPU_SEL[@]}"} "$@"; }
query() { smi --query-gpu="$1" --format=csv,noheader; }
cards=$(query name | grep -c . || true)
[[ $cards == 1 ]] ||
    die "$cards cards visible: select one (CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<i>)"
compute_apps() { smi --query-compute-apps=pid,process_name --format=csv,noheader | grep . || true; }
require_idle() {
    local busy
    busy=$(compute_apps)
    if [[ -n $busy && ${ALLOW_BUSY:-0} != 1 ]]; then
        busy_exit "compute processes on the card $1 (ALLOW_BUSY=1 to run anyway):"$'\n'"$busy"
    fi
}
require_idle "at the start"
GPU_NAME=$(query name)
CC=$(query compute_cap | tr -d ' ')
if [[ -z ${ARCH:-} ]]; then
    ARCH=${CC/./}
    case $ARCH in 120 | 121) ARCH=${ARCH}a ;; esac
fi

# The revisions: B is HEAD, committed; A is the merge base of BASE (after a fetch) and HEAD.
[[ -z $(git -C "$root" status --porcelain --untracked-files=no) ]] ||
    die "the checkout has uncommitted changes: B is its HEAD, commit them first"
[[ ${NO_FETCH:-0} == 1 ]] || git -C "$root" fetch -q origin master || die "git fetch origin master failed"
git -C "$root" rev-parse -q --verify "$BASE^{commit}" >/dev/null || die "no commit $BASE"
SHA_B=$(git -C "$root" rev-parse HEAD)
# A is the master B was built on (their merge base), not today's master: a master that moved on
# since would put its own changes into B/A.
SHA_A=$(git -C "$root" merge-base "$BASE" "$SHA_B") || die "HEAD shares no history with $BASE"
BRANCH_B=$(git -C "$root" rev-parse --abbrev-ref HEAD)
[[ $SHA_A != "$SHA_B" ]] || die "HEAD is already in $BASE ($SHA_A): nothing to compare"

card_tag=$(tr -c 'A-Za-z0-9\n' _ <<<"${GPU_NAME#NVIDIA GeForce }" | sed 's/__*/_/g; s/_$//')
OUT=${1:-${TMPDIR:-/tmp}/r8_850_step_${card_tag}_$(date +%m%d_%H%M)}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
case $OUT/ in "$root"/*) die "OUT $OUT is inside the repository: pick a directory outside it" ;; esac
[[ ! -e "$OUT/rounds.jsonl" ]] || die "$OUT already holds rounds: pick another OUT (BUILD_ROOT reuses builds)"
BUILD_ROOT=${BUILD_ROOT:-$OUT/trees}
mkdir -p "$BUILD_ROOT"
BUILD_ROOT=$(cd "$BUILD_ROOT" && pwd)
free=$(df -BG --output=avail "$BUILD_ROOT" | tail -1 | tr -dc 0-9)
((free >= MIN_FREE_GB)) || busy_exit "only $free GB free under $BUILD_ROOT, $MIN_FREE_GB needed (MIN_FREE_GB)"

LAUNCH=()
if [[ ${NO_CCACHE:-0} != 1 ]]; then
    command -v ccache >/dev/null || die "no ccache (NO_CCACHE=1 builds both trees cold)"
    LAUNCH=(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
    export CCACHE_BASEDIR=$BUILD_ROOT CCACHE_NOHASHDIR=1
fi
GEN=()
command -v ninja >/dev/null && GEN=(-G Ninja)
FLAGS=(-DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=$ARCH" -DBUILD_TESTING=OFF
    -DNINFERNO_CUDA_LINEINFO=OFF)
SERVE_FLAGS=(--host 127.0.0.1 --port "$PORT" --spec none --max-concurrency 1
    --max-context "$MAX_CONTEXT" --no-prefix-reuse --greedy)

# Writes no GPU UUID and no IP address.
scrub() {
    sed -i -E 's/GPU-[0-9A-Fa-f]{8}(-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}/GPU-<uuid>/g;
        s/\b([0-9]{1,3}\.){3}[0-9]{1,3}\b/<ip>/g' "$@" 2>/dev/null || true
}

{
    echo "kit tools/bench/run_850_step_ab.sh (#850 (a), kernel proposal P, acceptance item 4)"
    echo "date $(date -Is)"
    echo "gpu $GPU_NAME, compute capability $CC, driver $(query driver_version)"
    echo "gpu clocks (sm, mem, max sm, max mem), power limit, persistence:" \
        "$(query clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,power.limit,persistence_mode)"
    echo "A merge-base($BASE, HEAD) $SHA_A ($BASE at $(git -C "$root" rev-parse "$BASE^{commit}"))"
    echo "B $BRANCH_B $SHA_B"
    echo "model $MODEL --quant $QUANT"
    echo "server ${SERVE_FLAGS[*]}"
    echo "rounds per build $ROUNDS"
    echo "requests per round $REQUESTS, warm-up 1, output tokens $OUT_TOKENS (ignore_eos), prompt chars $CHARS"
    echo "build cmake ${GEN[*]} ${FLAGS[*]} ${LAUNCH[*]}, target ninferno-serve, -j $JOBS"
    echo "cmake $(cmake --version | head -1)"
    [[ -n ${LAUNCH[*]} ]] && echo "ccache $(ccache --version | head -1)"
} >"$OUT/meta.txt"
cat "$OUT/meta.txt"

SERVER_PID= SAMPLER_PID=
stop_sampler() {
    [[ -n $SAMPLER_PID ]] || return 0
    kill "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
    SAMPLER_PID=
}
# TERM to the server's process group, KILL after 30 s.
stop_server() {
    [[ -n $SERVER_PID ]] || return 0
    kill -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    kill -0 "$SERVER_PID" 2>/dev/null && { kill -KILL -- "-$SERVER_PID" 2>/dev/null || true; }
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
    sleep 3
}
trap 'stop_sampler; stop_server' EXIT

# 1-2. sources and builds
tree() {
    local arm=$1 sha=$2 dir=$BUILD_ROOT/$1
    mkdir -p "$dir"
    if [[ -e "$dir/src" ]]; then
        [[ $(git -C "$dir/src" rev-parse HEAD 2>/dev/null) == "$sha" ]] ||
            die "$dir/src is not a worktree at $sha: remove it (git worktree remove $dir/src)"
    else
        git -C "$root" worktree add -q --detach "$dir/src" "$sha" || die "git worktree add $dir/src failed"
    fi
    echo "build $arm ($sha) in $dir/build"
    cmake -S "$dir/src" -B "$dir/build" ${GEN[@]+"${GEN[@]}"} "${FLAGS[@]}" \
        ${LAUNCH[@]+"${LAUNCH[@]}"} >"$OUT/cmake_$arm.log" 2>&1 ||
        { tail -20 "$OUT/cmake_$arm.log"; die "configure $arm failed ($OUT/cmake_$arm.log)"; }
    cmake --build "$dir/build" --target ninferno-serve -j "$JOBS" >"$OUT/build_$arm.log" 2>&1 ||
        { grep -m8 -E "error|FAILED" "$OUT/build_$arm.log" || tail -20 "$OUT/build_$arm.log"
          die "build $arm failed ($OUT/build_$arm.log)"; }
    [[ -x "$dir/build/apps/ninferno-serve" ]] || die "no $dir/build/apps/ninferno-serve"
    local nvcc version
    nvcc=$(sed -n 's/^CMAKE_CUDA_COMPILER:[A-Z]*=//p' "$dir/build/CMakeCache.txt")
    version=$("$nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\),.*/\1/p' || true)
    echo "build $arm nvcc ${version:-unknown}, $(du -sh "$dir/build" | cut -f1)" | tee -a "$OUT/meta.txt"
}
started=$(date +%s)
tree a "$SHA_A"
tree b "$SHA_B"
echo "builds took $(( ($(date +%s) - started) / 60 )) min" | tee -a "$OUT/meta.txt"
serve_of() { echo "$BUILD_ROOT/${1,,}/build/apps/ninferno-serve"; }

# 3. the model, once into the cache (no GPU)
for arm in B A; do
    log="$OUT/prepare_$arm.log"
    "$(serve_of $arm)" "$MODEL" --quant "$QUANT" --spec none --prepare-only >"$log" 2>&1 ||
        { tail -20 "$log"; die "prepare with build $arm failed ($log)"; }
done
scrub "$OUT"/prepare_*.log
"$py" "$helper" prompt --chars "$CHARS" >"$OUT/prompt.txt" || die "the prompt could not be made"

# 4. rounds
start_server() {
    local arm=$1 r=$2 log
    log="$OUT/server_${r}_$arm.log"
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        die "port $PORT already answers: a server is running there; stop it or set PORT"
    fi
    HF_HUB_OFFLINE=1 setsid "$(serve_of "$arm")" "$MODEL" --quant "$QUANT" "${SERVE_FLAGS[@]}" \
        >"$log" 2>&1 &
    SERVER_PID=$!
    local deadline=$(($(date +%s) + STARTUP_TIMEOUT))
    while (($(date +%s) < deadline)); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            kill -0 "$SERVER_PID" 2>/dev/null && return 0
            die "something answers on port $PORT but the server started here is gone"
        fi
        kill -0 "$SERVER_PID" 2>/dev/null ||
            { tail -20 "$log"; die "the server of round $r ($arm) died ($log)"; }
        sleep 2
    done
    tail -20 "$log"
    die "the server of round $r ($arm) did not come up in $STARTUP_TIMEOUT s"
}
order=()
for ((i = 0; i < 2 * ROUNDS; i++)); do
    case $((i % 4)) in 0 | 3) order+=(A) ;; *) order+=(B) ;; esac
done
echo "order ${order[*]}" | tee -a "$OUT/meta.txt"
for ((i = 0; i < ${#order[@]}; i++)); do
    r=$((i + 1)) arm=${order[$i]}
    require_idle "before round $r"
    echo "round $r: $arm"
    start_server "$arm" "$r"
    # nvidia-smi itself in the background (not the smi function's subshell), so $! is what to kill
    nvidia-smi ${GPU_SEL[@]+"${GPU_SEL[@]}"} --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw \
        --format=csv,noheader,nounits -lms 500 >"$OUT/clocks_$r.csv" 2>/dev/null &
    SAMPLER_PID=$!
    "$py" "$helper" request --port "$PORT" --arm "$arm" --round "$r" --prompt "$OUT/prompt.txt" \
        --out "$OUT/rounds.jsonl" --requests "$REQUESTS" --out-tokens "$OUT_TOKENS" ||
        { tail -20 "$OUT/server_${r}_$arm.log"; die "round $r ($arm) failed"; }
    others=$(compute_apps | grep -c . || true)
    if ((others > 1)) && [[ ${ALLOW_BUSY:-0} != 1 ]]; then
        busy_exit "another compute process joined the card during round $r:"$'\n'"$(compute_apps)"
    fi
    stop_sampler
    stop_server
    scrub "$OUT/server_${r}_$arm.log"
done
trap - EXIT

# 5. the summary
echo
summarize "$OUT"
scrub "$OUT"/*.log "$OUT/meta.txt"
if [[ $KEEP_BUILDS == 0 ]]; then
    for arm in a b; do
        rm -rf "$BUILD_ROOT/$arm/build"
        git -C "$root" worktree remove "$BUILD_ROOT/$arm/src" ||
            echo "left $BUILD_ROOT/$arm/src: git worktree remove it by hand" >&2
    done
    rmdir "$BUILD_ROOT"/a "$BUILD_ROOT"/b "$BUILD_ROOT" 2>/dev/null || true
fi
echo "summary: $OUT/summary.md"
