#!/bin/bash
# kernel_facts_archs.sh [ARCH...] : check the kernel-facts snapshots of the release architectures no
# stand builds (default: 80 90 100) against a build of each, on a build machine's CPU.
#
# ninferno_kernel_facts_test compares a build's register-MMA kernels with the snapshot of that
# build's own architecture (src/ops/generic/data/kernel_facts/sm_<arch>.txt), and the stands build
# sm_70, sm_75, sm_86, sm_89 and sm_120a. Without this job nothing checks sm_80.txt, sm_90.txt and
# sm_100.txt before tools/release/package.sh does at release time (#758, #762). Per architecture:
#   - configures a test tree with the release flags (Release, NINFERNO_CUDA_LINEINFO=OFF,
#     NINFERNO_CUDA_EMBED_PTX=OFF; ccache launchers where ccache is installed);
#   - builds ninferno_selector_golden_test, the binary the check reads (1.3 GB a tree; 20 to 30
#     minutes cold at -j8 on a 16-core machine, less with ccache), at the lowest CPU and I/O
#     priority;
#   - runs ninferno_kernel_facts_test (cuobjdump on the binary, the SASS pass included: about ten
#     minutes), with no GPU visible. Its skip (exit 77: this nvcc is not the snapshot's) is a
#     failure here: the job exists to compare, so it refuses to start on a toolkit other than the
#     snapshots' nvcc.
# Source tree: KFA_SOURCE=DIR checks that tree as it is; otherwise the job checks out
# origin/KFA_BRANCH (default master; any PR branch, e.g. one touching src/ops/**/*.cu*) detached and
# cleaned in $KFA_ROOT/wt_kfa, a worktree of KFA_REPO.
# Exit (the largest over the architectures): 0 every architecture's kernels equal its snapshot
#       1 a snapshot differs from its build (the check names the kernels and the regenerate command)
#       2 configure or build failed
#       3 usage/setup (no snapshot, wrong nvcc, disk, fetch, worktree)
# Env: KFA_ROOT (work dir, default: current directory) KFA_REPO (an existing checkout, default
#      $KFA_ROOT/wt_tst) KFA_BRANCH (master) KFA_SOURCE (a source tree instead of a checkout)
#      CUDACXX (nvcc, default /usr/local/cuda-13.4/bin/nvcc) KFA_JOBS (8) KFA_MIN_FREE_GB (10)
#      KFA_CLEAN=1 (remove each build tree after its check; default: kept for the next run)
set -Eeuo pipefail
trap 'rc=$?; echo "kernel_facts_archs.sh: internal error (exit $rc) at line $LINENO" >&2; exit 3' ERR
ARCHS=("$@"); [ ${#ARCHS[@]} -gt 0 ] || ARCHS=(80 90 100)
ROOT=$(realpath "${KFA_ROOT:-$PWD}"); JOBS=${KFA_JOBS:-8}; NVCC=${CUDACXX:-/usr/local/cuda-13.4/bin/nvcc}
L=$ROOT/logs; mkdir -p "$L"
exec 9> "$ROOT/kfa.lock"
flock -n 9 || { echo "kernel_facts_archs.sh: another run is active, waiting"; flock 9; }
LOW="nice -n 19 ionice -c3"
T0=$(date +%s); tick() { echo "  [$(($(date +%s) - T0))s] $*"; }

# 1. the source tree
if [ -n "${KFA_SOURCE:-}" ]; then
  SRC=$(realpath "$KFA_SOURCE")
  [ -f "$SRC/CMakeLists.txt" ] || { echo "KFA_SOURCE=$KFA_SOURCE: no source tree"; exit 3; }
else
  REPO=${KFA_REPO:-$ROOT/wt_tst}; BR=${KFA_BRANCH:-master}; SRC=$ROOT/wt_kfa
  [ -d "$REPO" ] || { echo "no git checkout at $REPO (set KFA_REPO or KFA_SOURCE)"; exit 3; }
  [ -d "$SRC" ] || git -C "$REPO" worktree add -q --detach "$SRC" || { echo "worktree add $SRC failed"; exit 3; }
  [ "$(realpath "$(git -C "$SRC" rev-parse --show-toplevel)")" = "$SRC" ] \
    || { echo "$SRC is not its own git worktree root; refusing checkout -f / clean"; exit 3; }
  # a standalone clone at $SRC is also a worktree root: it must be a worktree of $REPO (same common
  # git dir, as tools/dev/qc.sh checks)
  gcd() { ( cd "$1" && realpath "$(git rev-parse --git-common-dir)" ); }
  [ "$(gcd "$SRC")" = "$(gcd "$REPO")" ] \
    || { echo "$SRC is not a worktree of $REPO (other git dir); refusing checkout -f / clean"; exit 3; }
  git -C "$SRC" fetch -q origin "+refs/heads/$BR:refs/remotes/origin/$BR" || { echo "fetch of $BR failed"; exit 3; }
  git -C "$SRC" checkout -q -f --detach "origin/$BR"
  git -C "$SRC" clean -fdxq
fi
echo "== kernel facts of ${ARCHS[*]} at $(git -C "$SRC" log --oneline -1 2> /dev/null | cut -c1-100 || echo "$SRC")"
FACTS=$SRC/src/ops/generic/data/kernel_facts

# 2. the toolkit: the one every snapshot records (a check under another nvcc only skips)
[ -x "$NVCC" ] || { echo "no nvcc at $NVCC (set CUDACXX)"; exit 3; }
VERSION=$("$NVCC" --version | sed -n 's/.*, V\([0-9.]*\)$/\1/p')
for a in "${ARCHS[@]}"; do
  [ -f "$FACTS/sm_$a.txt" ] || { echo "no snapshot $FACTS/sm_$a.txt"; exit 3; }
  want=$(sed -n 's/^nvcc //p' "$FACTS/sm_$a.txt")
  [ "$want" = "$VERSION" ] || { echo "sm_$a.txt was made with nvcc $want, $NVCC is $VERSION: the check would only skip"; exit 3; }
done
LAUNCH=()
if command -v ccache > /dev/null; then
  LAUNCH=(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
  export CCACHE_BASEDIR=$SRC CCACHE_NOHASHDIR=1
fi

# 3. per architecture: configure, build, check
declare -A RESULT
WORST=0
worst() { [ "$1" -le "$WORST" ] || WORST=$1; }
for a in "${ARCHS[@]}"; do
  B=$ROOT/build_kfa_$a
  FREE=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
  [ "$FREE" -ge "${KFA_MIN_FREE_GB:-10}" ] || { echo "only $FREE GB free on $ROOT; clean up first"; exit 3; }
  # A tree configured from another source or compiler is configured again from scratch.
  if [ -f "$B/CMakeCache.txt" ] && { ! grep -qx "CMAKE_HOME_DIRECTORY:INTERNAL=$SRC" "$B/CMakeCache.txt" \
    || ! grep -qxE "CMAKE_CUDA_COMPILER:[A-Z]+=$NVCC" "$B/CMakeCache.txt"; }; then
    rm -rf "$B"
  fi
  if ! cmake -S "$SRC" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$a" \
    -DCMAKE_CUDA_COMPILER="$NVCC" -DNINFERNO_CUDA_LINEINFO=OFF -DNINFERNO_CUDA_EMBED_PTX=OFF \
    -DBUILD_TESTING=ON "${LAUNCH[@]}" > "$L/kfa_cmake_$a.log" 2>&1; then
    tail -15 "$L/kfa_cmake_$a.log"; RESULT[$a]="configure failed ($L/kfa_cmake_$a.log)"; worst 2; continue
  fi
  tick "sm_$a configured"
  if ! (cd "$B" && $LOW ninja -j"$JOBS" ninferno_selector_golden_test > "$L/kfa_ninja_$a.log" 2>&1); then
    grep -m8 -E "error|FAILED" "$L/kfa_ninja_$a.log" || tail -8 "$L/kfa_ninja_$a.log"
    RESULT[$a]="build failed ($L/kfa_ninja_$a.log)"; worst 2; continue
  fi
  tick "sm_$a built ($(du -sh "$B" | cut -f1))"
  # ctest reports a skipped test as passed: the check's own line says what it compared.
  (cd "$B" && CUDA_VISIBLE_DEVICES= $LOW ctest -R '^ninferno_kernel_facts_test$' -V > "$L/kfa_check_$a.log" 2>&1) && rc=0 || rc=$?
  line=$(grep -m1 -oE "kernel_facts: [0-9]+ kernels of sm_$a equal .*" "$L/kfa_check_$a.log" || true)
  if [ $rc = 0 ] && [ -n "$line" ] && [[ $line != *"(resources only)"* ]]; then
    RESULT[$a]="ok: ${line#kernel_facts: }"
  elif [ $rc = 0 ] && [ -n "$line" ]; then
    RESULT[$a]="resources only: sm_$a.txt has no loop counts (schema 1), regenerate it"; worst 1
  elif grep -q "kernel_facts: SKIPPED" "$L/kfa_check_$a.log"; then
    RESULT[$a]="skipped by the check ($L/kfa_check_$a.log)"; worst 3
  elif grep -q "differ from" "$L/kfa_check_$a.log"; then
    grep -E "^[0-9]+: .*(rmma|build:|missing in|not in the snapshot|kernel_facts:)" "$L/kfa_check_$a.log" | head -24 || true
    RESULT[$a]="DIFFERS from sm_$a.txt ($L/kfa_check_$a.log)"; worst 1
  else
    tail -15 "$L/kfa_check_$a.log"; RESULT[$a]="check failed (ctest rc $rc, $L/kfa_check_$a.log)"; worst 1
  fi
  tick "sm_$a checked"
  [ "${KFA_CLEAN:-0}" != 1 ] || rm -rf "$B"
done

echo "== kernel facts, nvcc $VERSION, $(($(date +%s) - T0)) s"
for a in "${ARCHS[@]}"; do printf '  sm_%-5s %s\n' "$a" "${RESULT[$a]}"; done
exit $WORST
