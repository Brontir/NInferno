#!/bin/bash
# kernel_facts_families.sh REPO [ARCH...] : regenerate the mmq and w4a8 sections of the kernel-facts
# snapshots (#785) on the release toolchain, on a build machine's CPU, and check them.
#
# The sections were first read from nvcc 13.2.86 builds (their family_nvcc line); a release build
# compares them in full only under the nvcc they record. This job rewrites them with the
# snapshots' own nvcc (the `nvcc` line of each file, 13.4.92), leaving every other byte of each
# file as it was: the register-MMA rows and the file's header lines. Per architecture (default:
# 75 80 86 89 90 100 120a):
#   - configures a test tree from REPO with the release flags (Release, NINFERNO_CUDA_LINEINFO=OFF,
#     NINFERNO_CUDA_EMBED_PTX=OFF; ccache launchers where ccache is installed), as
#     tools/dev/kernel_facts_archs.sh does;
#   - builds ninferno_selector_golden_test alone, the binary ninferno_kernel_facts_test reads (about
#     1.3 GB a tree; 20 to 30 minutes cold at -j8 on a 16-core machine), at the lowest CPU and I/O
#     priority. Not an object-only build: the W4A8 tiles are relocatable device code (ninferno_ops),
#     whose shared memory and stack the device link decides, so their rows must be read off the
#     linked binary, and `check` compares the register-MMA rows of the same binary too;
#   - kernel_facts.py extract --families mmq,w4a8 --update sm_<arch>.txt with this nvcc and its own
#     cuobjdump and cu++filt (a few minutes);
#   - kernel_facts.py check on the updated file: every family compared in full (the SASS pass, about
#     ten minutes), no GPU;
#   - verifies that the file up to its first `family` line is byte for byte the committed one
#     (git show HEAD:<file>) and that both sections now record this nvcc.
# REPO must be a clean checkout (no uncommitted change under src/ops/generic/data/kernel_facts): the
# job writes the snapshots there, and `git diff` then shows only section lines. Commit them, run
# tools/kernel_facts/mmq_units.py and refresh the tables of tools/kernel_facts/mmq_units.md.
# Exit (the largest over the architectures): 0 every section regenerated and every check equal
#       1 a check differs, the extract audit failed, or a byte outside the sections moved
#       2 configure or build failed
#       3 usage/setup (no snapshot, wrong nvcc, disk, dirty checkout)
# Env: KFF_ROOT (build dir, default $HOME/kff) CUDACXX (nvcc, default /usr/local/cuda-13.4/bin/nvcc)
#      KFF_JOBS (8) KFF_MIN_FREE_GB (10) KFF_CLEAN=1 (remove each build tree after its check)
set -Eeuo pipefail
trap 'rc=$?; echo "kernel_facts_families.sh: internal error (exit $rc) at line $LINENO" >&2; exit 3' ERR
[ $# -ge 1 ] || { echo "usage: kernel_facts_families.sh REPO [ARCH...]"; exit 3; }
REPO=$(realpath "$1"); shift
ARCHS=("$@"); [ ${#ARCHS[@]} -gt 0 ] || ARCHS=(75 80 86 89 90 100 120a)
ROOT=$(realpath -m "${KFF_ROOT:-$HOME/kff}"); mkdir -p "$ROOT"
JOBS=${KFF_JOBS:-8}; NVCC=${CUDACXX:-/usr/local/cuda-13.4/bin/nvcc}
L=$ROOT/logs; mkdir -p "$L"
exec 9> "$ROOT/kff.lock"
flock -n 9 || { echo "kernel_facts_families.sh: another run is active, waiting"; flock 9; }
LOW="nice -n 19 ionice -c3"
T0=$(date +%s); tick() { echo "  [$(($(date +%s) - T0))s] $*"; }

# 1. the checkout
[ -f "$REPO/CMakeLists.txt" ] && [ -f "$REPO/tools/kernel_facts/kernel_facts.py" ] \
  || { echo "$REPO: no NInferno source tree"; exit 3; }
git -C "$REPO" rev-parse --is-inside-work-tree > /dev/null 2>&1 || { echo "$REPO: not a git checkout"; exit 3; }
FACTS_REL=src/ops/generic/data/kernel_facts
FACTS=$REPO/$FACTS_REL
case "$ROOT/" in "$REPO"/*) echo "KFF_ROOT=$ROOT is inside $REPO; build outside the checkout"; exit 3 ;; esac
[ -z "$(git -C "$REPO" status --porcelain -- "$FACTS_REL")" ] \
  || { echo "$REPO has uncommitted changes in $FACTS_REL; start from a clean checkout"; exit 3; }
echo "== mmq and w4a8 sections of ${ARCHS[*]} at $(git -C "$REPO" log --oneline -1 | cut -c1-100)"

# 2. the toolkit: the nvcc every snapshot records, and its own cuobjdump and cu++filt
[ -x "$NVCC" ] || { echo "no nvcc at $NVCC (set CUDACXX)"; exit 3; }
CUDA_BIN=$(dirname "$(realpath "$NVCC")")
for t in cuobjdump cu++filt; do
  [ -x "$CUDA_BIN/$t" ] || { echo "no $t beside $NVCC ($CUDA_BIN)"; exit 3; }
done
VERSION=$("$NVCC" --version | sed -n 's/.*, V\([0-9.]*\)$/\1/p')
for a in "${ARCHS[@]}"; do
  f=$FACTS/sm_$a.txt
  [ -f "$f" ] || { echo "no snapshot $f"; exit 3; }
  want=$(grep -m1 '^nvcc ' "$f" | cut -d' ' -f2)
  [ "$want" = "$VERSION" ] || { echo "sm_$a.txt records nvcc $want, $NVCC is $VERSION"; exit 3; }
done
LAUNCH=()
if command -v ccache > /dev/null; then
  LAUNCH=(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
  export CCACHE_BASEDIR=$REPO CCACHE_NOHASHDIR=1
fi
KF=(python3 -B "$REPO/tools/kernel_facts/kernel_facts.py")
# The part of a snapshot before its first section: the header lines and the register-MMA rows.
top() { sed '/^family /,$d'; }

# 3. per architecture: configure, build, extract, check, verify
declare -A RESULT
WORST=0
worst() { [ "$1" -le "$WORST" ] || WORST=$1; }
for a in "${ARCHS[@]}"; do
  B=$ROOT/build_kff_$a
  f=$FACTS/sm_$a.txt
  FREE=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
  [ "$FREE" -ge "${KFF_MIN_FREE_GB:-10}" ] || { echo "only $FREE GB free on $ROOT; clean up first"; exit 3; }
  # A tree configured from another source or compiler is configured again from scratch.
  if [ -f "$B/CMakeCache.txt" ] && { ! grep -qx "CMAKE_HOME_DIRECTORY:INTERNAL=$REPO" "$B/CMakeCache.txt" \
    || ! grep -qxE "CMAKE_CUDA_COMPILER:[A-Z]+=$NVCC" "$B/CMakeCache.txt"; }; then
    rm -rf "$B"
  fi
  if ! cmake -S "$REPO" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$a" \
    -DCMAKE_CUDA_COMPILER="$NVCC" -DNINFERNO_CUDA_LINEINFO=OFF -DNINFERNO_CUDA_EMBED_PTX=OFF \
    -DBUILD_TESTING=ON "${LAUNCH[@]}" > "$L/kff_cmake_$a.log" 2>&1; then
    tail -15 "$L/kff_cmake_$a.log"; RESULT[$a]="configure failed ($L/kff_cmake_$a.log)"; worst 2; continue
  fi
  tick "sm_$a configured"
  if ! (cd "$B" && $LOW ninja -j"$JOBS" ninferno_selector_golden_test > "$L/kff_ninja_$a.log" 2>&1); then
    grep -m8 -E "error|FAILED" "$L/kff_ninja_$a.log" || tail -8 "$L/kff_ninja_$a.log"
    RESULT[$a]="build failed ($L/kff_ninja_$a.log)"; worst 2; continue
  fi
  BIN=$B/tests/ninferno_selector_golden_test
  [ -x "$BIN" ] || { RESULT[$a]="no $BIN after the build"; worst 2; continue; }
  tick "sm_$a built ($(du -sh "$B" | cut -f1))"
  # The audit refuses counts it cannot settle (exit 1): nothing is kept then.
  cp "$f" "$L/sm_$a.before.txt"
  if ! CUDA_VISIBLE_DEVICES= $LOW "${KF[@]}" extract --binary "$BIN" --arch "$a" --nvcc "$VERSION" \
    --cuda-bin "$CUDA_BIN" --families mmq,w4a8 --update "$f" > "$L/kff_extract_$a.log" 2>&1; then
    cp "$L/sm_$a.before.txt" "$f"
    tail -8 "$L/kff_extract_$a.log"; RESULT[$a]="extract failed ($L/kff_extract_$a.log)"; worst 1; continue
  fi
  tick "sm_$a sections written"
  # Nothing before the first section may move: the header lines and the register-MMA rows.
  if ! cmp -s <(git -C "$REPO" show "HEAD:$FACTS_REL/sm_$a.txt" | top) <(top < "$f"); then
    cp "$L/sm_$a.before.txt" "$f"
    RESULT[$a]="the part before the sections changed: restored, nothing kept"; worst 1; continue
  fi
  sections=$(grep -c '^family ' "$f" || true)
  nvccs=$(grep '^family_nvcc ' "$f" | cut -d' ' -f2 | sort -u | tr '\n' ' ')
  if [ "$sections" != 2 ] || [ "$nvccs" != "$VERSION " ]; then
    RESULT[$a]="expected 2 sections at nvcc $VERSION, found $sections at ${nvccs:-none}"; worst 1; continue
  fi
  (CUDA_VISIBLE_DEVICES= $LOW "${KF[@]}" check --binary "$BIN" --expected "$f" --nvcc "$VERSION" \
    --cuda-bin "$CUDA_BIN" > "$L/kff_check_$a.log" 2>&1) && rc=0 || rc=$?
  rmma=$(grep -m1 -oE "^kernel_facts: [0-9]+ kernels of sm_$a equal" "$L/kff_check_$a.log" | grep -oE "[0-9]+ kernels" || true)
  mmq=$(grep -m1 -oE "^kernel_facts: [0-9]+ mmq kernels of sm_$a equal" "$L/kff_check_$a.log" | grep -oE "[0-9]+" | head -1 || true)
  w4a8=$(grep -m1 -oE "^kernel_facts: [0-9]+ w4a8 kernels of sm_$a equal" "$L/kff_check_$a.log" | grep -oE "[0-9]+" | head -1 || true)
  if [ $rc = 0 ] && [ -n "$rmma" ] && [ -n "$mmq" ] && [ -n "$w4a8" ]; then
    RESULT[$a]="ok: rmma $rmma, mmq $mmq, w4a8 $w4a8 equal (nvcc $VERSION), header and rmma rows byte-identical"
  else
    grep -E "differ from|missing in|not in the snapshot|SKIPPED" "$L/kff_check_$a.log" | head -12 || true
    RESULT[$a]="check failed (rc $rc, $L/kff_check_$a.log)"; worst 1
  fi
  tick "sm_$a checked"
  [ "${KFF_CLEAN:-0}" != 1 ] || rm -rf "$B"
done

echo "== mmq and w4a8 sections, nvcc $VERSION, $(($(date +%s) - T0)) s"
for a in "${ARCHS[@]}"; do printf '  sm_%-5s %s\n' "$a" "${RESULT[$a]:-not run}"; done
# What moved in the checkout: section lines only (each file's part before its first section was
# compared with HEAD above).
git -C "$REPO" diff --stat -- "$FACTS_REL" || true
exit $WORST
