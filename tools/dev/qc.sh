#!/bin/bash
# qc.sh BRANCH [TARGET...] : quick compile check of a Brontir/NInferno branch on a stand's CPU.
#
# Compiles only what the branch touches, in one persistent build dir and one worktree, at the lowest
# CPU and I/O priority so a ctest running on the stand is not starved. Never creates a second build.
#   - checks out origin/BRANCH (detached) in $QC_ROOT/wt_qc;
#   - without TARGETs: maps `git diff origin/master...origin/BRANCH` to ninja targets:
#       * the object of every changed .cpp/.cu (compile only, no link),
#       * the objects of the sources that include a changed header (transitively, by file name),
#       * the test executables those sources belong to or that the diff adds to a tests cmake file
#         (linked only when the library objects they need are few: QC_MAX_CU CUDA units, see below);
#   - builds with `nice -n 19 ionice -c3 ninja -k 0 -j4`;
#   - runs the built test executables through ctest with CUDA_VISIBLE_DEVICES empty (no GPU use);
#     GPU tests then fail or skip (77) and are reported as such;
#   - prints compile errors as file:line and a result table.
# Env: QC_ROOT (/root/claude) QC_ARCH (89) QC_JOBS (4) QC_MAX_OBJ (60) QC_MAX_CU (40) QC_MAX_UNITS (150, CUDA+C++ per target)
#      QC_TEST_TIMEOUT (300 s per test) QC_MAX_GB (40, abort when the build dir is larger) QC_NO_TESTS=1 (build only)
# Exit: 0 all built and tests passed/skipped, 1 compile/link errors, 2 test failures, 3 usage/setup.
BR=$1; [ -n "$BR" ] || { echo "usage: qc.sh BRANCH [TARGET...]" >&2; exit 3; }
shift
ROOT=${QC_ROOT:-/root/claude}; ARCH=${QC_ARCH:-89}; JOBS=${QC_JOBS:-4}
W=$ROOT/wt_qc; B=$ROOT/build_qc_$ARCH; L=$ROOT/logs; mkdir -p "$L"
exec 9>"$ROOT/qc.lock"; flock -n 9 || { echo "qc.sh: another qc.sh run is active, waiting"; flock 9; }
. "$ROOT/lab.sh"; cuda_env
export CCACHE_DIR=/root/.ccache CCACHE_BASEDIR=$W CCACHE_NOHASHDIR=1
LOW="nice -n 19 ionice -c3"
T0=$(date +%s); tick() { echo "  [$(( $(date +%s) - T0 ))s] $*"; }

# 1. fetch + checkout
[ -d "$W" ] || git -C "$ROOT/wt_tst" worktree add -q --detach "$W" origin/master || exit 3
git -C "$W" fetch -q origin master "+refs/heads/$BR:refs/remotes/origin/$BR" || { echo "fetch of $BR failed"; exit 3; }
git -C "$W" checkout -q -f --detach "origin/$BR" || exit 3
echo "== qc $BR at $(git -C "$W" log --oneline -1 | cut -c1-100)"
tick fetched
GB=$(du -s --block-size=1G "$B" 2>/dev/null | cut -f1); GB=${GB:-0}
[ "$GB" -le "${QC_MAX_GB:-40}" ] || { echo "build_qc is $GB GB (> ${QC_MAX_GB:-40}); rm -rf $B to restart"; exit 3; }

# 2. configure once (same options as kt3.sh); later branch switches re-run cmake through ninja
if [ ! -f "$B/build.ninja" ]; then
  cmake -S "$W" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$ARCH \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DBUILD_TESTING=ON \
    > "$L/qc_cmake.log" 2>&1 || { tail -15 "$L/qc_cmake.log"; echo "CONFIGURE FAILED"; exit 1; }
  tick configured
fi
( cd "$B" && $LOW ninja build.ninja > "$L/qc_regen.log" 2>&1 ) || { tail -15 "$L/qc_regen.log"; echo "CMAKE REGENERATE FAILED (cmake error on this branch)"; exit 1; }

# 3. targets
PLAN=$(mktemp); trap 'rm -f "$PLAN" "$PLAN.py"' EXIT
cat > "$PLAN.py" <<'PY'
import sys, os, re, json, subprocess, collections
src, bld, br = sys.argv[1:4]
MAXOBJ = int(os.environ.get("QC_MAX_OBJ", "60")); MAXCU = int(os.environ.get("QC_MAX_CU", "40")); MAXU = int(os.environ.get("QC_MAX_UNITS", "150"))
def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout
def note(*a): print(*a, file=sys.stderr)
rng = f"origin/master...origin/{br}"
changed = sh("git", "-C", src, "diff", "--name-only", "--diff-filter=d", rng).split()
# object per source from the build graph
comp = json.loads(sh("ninja", "-C", bld, "-t", "compdb") or "[]")
obj_of = collections.defaultdict(list)
for e in comp:
    out = e.get("output", "")
    if not out.endswith((".o", ".obj")): continue
    f = e["file"]; f = f if os.path.isabs(f) else os.path.join(e["directory"], f)
    obj_of[os.path.realpath(f)].append(out)
exe_of = {}  # test name -> exe target path
for line in sh("ninja", "-C", bld, "-t", "targets", "all").splitlines():
    m = re.match(r"^(\S+): \w+_EXECUTABLE_LINKER", line)
    if m: exe_of[os.path.basename(m.group(1))] = m.group(1)
def owner(obj):  # the target a test object belongs to
    m = re.search(r"CMakeFiles/([^/]+)\.dir/", obj); return m.group(1) if m else None
# include index by file name (over-approximate, by basename)
exts = (".cpp", ".cu", ".h", ".hpp", ".cuh", ".inl", ".cc")
includers = collections.defaultdict(set)
for d in ("src", "include", "tests", "apps", "bench"):
    for root, _, fs in os.walk(os.path.join(src, d)):
        for fn in fs:
            if not fn.endswith(exts): continue
            p = os.path.join(root, fn)
            try: txt = open(p, errors="ignore").read()
            except OSError: continue
            for inc in re.findall(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', txt, re.M):
                includers[os.path.basename(inc)].add(os.path.relpath(p, src))
direct = [c for c in changed if c.endswith((".cpp", ".cu", ".cc"))]
hdrs = [c for c in changed if c.endswith((".h", ".hpp", ".cuh", ".inl"))]
seen = set(hdrs); queue = list(hdrs); via = set()
while queue:
    h = queue.pop()
    for p in includers.get(os.path.basename(h), ()):
        if p.endswith((".cpp", ".cu", ".cc")): via.add(p)
        elif p not in seen: seen.add(p); queue.append(p)
via -= set(direct)
cdirs = {os.path.dirname(c) for c in changed}
def prio(p):  # same directory as a changed file first, then the other tests, then the rest
    return (0 if os.path.dirname(p) in cdirs else 1 if p.startswith("tests/") else 2, p)
def objs(paths, order=sorted):
    r = []
    for p in order(paths):
        r += obj_of.get(os.path.realpath(os.path.join(src, p)), [])
    return r
o_direct, o_via = objs(direct), objs(via, lambda ps: sorted(ps, key=prio))
if len(o_direct) + len(o_via) > MAXOBJ:
    note(f"note: {len(o_direct)+len(o_via)} objects affected through headers; keeping {MAXOBJ} (QC_MAX_OBJ)")
    o_via = o_via[:max(0, MAXOBJ - len(o_direct))]
allobjs = o_direct + o_via
# test executables: owners of the objects, plus names added in tests cmake files
tests = {owner(o) for o in allobjs if o.startswith("tests/") or "/tests/" in o}
tests.discard(None)
cm = sh("git", "-C", src, "diff", "-U0", rng, "--", "*.cmake", "CMakeLists.txt")
for l in cm.splitlines():
    if l.startswith("+") and not l.startswith("+++"):
        m = re.search(r"(?:ninferno_add_test|add_executable|add_test)\(\s*(?:NAME\s+)?([A-Za-z0-9_]+)", l)
        if m: tests.add(m.group(1))
exes = sorted(exe_of[t] for t in tests if t in exe_of)
keep = []
deps = {}  # object -> its recorded header dependencies (ninja's own depfile log)
cur = None
for line in sh("ninja", "-C", bld, "-t", "deps").splitlines():
    if line and not line[0].isspace() and ": #deps" in line: cur = line.split(": #deps")[0]; deps[cur] = []
    elif line.strip() and cur: deps[cur].append(line.strip())
def mt(p):
    try: return os.stat(os.path.join(bld, p)).st_mtime
    except OSError: return None
def stale_units(exe):  # compile commands of the exe's closure whose object is missing or older than a dependency
    cu = cx = 0
    for cmd in sh("ninja", "-C", bld, "-t", "commands", exe).splitlines():
        m = re.search(r" -o (\S+\.o)", cmd)
        if not m or " -c " not in cmd: continue
        o = m.group(1); t = mt(o)
        if t is None or any((mt(d) or 0) > t for d in deps.get(o, ())):
            if "nvcc" in cmd: cu += 1
            else: cx += 1
    return cu, cx
for e in exes:
    cu, cx = stale_units(e)
    if cu > MAXCU:
        note(f"note: {e}: link skipped, {cu} CUDA + {cx} C++ units stale (QC_MAX_CU={MAXCU}); its objects are still compiled")
    else:
        note(f"test target {e}: {cu} CUDA + {cx} C++ stale units to build"); keep.append(e)
print("changed:", len(changed), "direct src:", len(direct), "headers:", len(hdrs), "via-header src:", len(via), file=sys.stderr)
keepobj = []
for o in allobjs:
    cu, cx = stale_units(o)
    if cu > MAXCU or cu + cx > MAXU:
        note(f"note: {o}: skipped, its dependencies need {cu} CUDA + {cx} C++ units (QC_MAX_CU={MAXCU}, QC_MAX_UNITS={MAXU})")
    else: keepobj.append(o)
for t in keep: print("EXE", t)
for o in sorted(set(keepobj)): print("OBJ", o)
PY
if [ $# -gt 0 ]; then
  for t in "$@"; do echo "OBJ $t"; done > "$PLAN"
else
  python3 -I "$PLAN.py" "$W" "$B" "$BR" > "$PLAN" 2> "$L/qc_plan.err"; cat "$L/qc_plan.err"
  if ! grep -q . "$PLAN" && [ "$BR" = master ]; then   # nothing changed on master: smoke targets
    for t in ninferno_spec_test ninferno_spec_round_cost_test ninferno_choice_test; do echo "EXE $t"; done > "$PLAN"
    echo "master has no diff against itself: smoke targets"
  fi
fi
TARGETS=$(awk '{print $2}' "$PLAN" | sort -u | tr '\n' ' ')
EXES=$(awk '$1=="EXE"{n=split($2,a,"/"); print a[n]}' "$PLAN" | tr '\n' ' ')
NT=$(echo $TARGETS | wc -w)
[ "$NT" -gt 0 ] || { echo "nothing to compile (no .cpp/.cu/.h or test change)"; exit 0; }
tick "$NT ninja targets"

# 4. build (lowest priority, keep going to collect every error)
TAG=$(echo "$BR" | tr / _); LOG=$L/qc_ninja_$TAG.log
( cd "$B" && $LOW ninja -k 0 -j"$JOBS" $TARGETS > "$LOG" 2>&1 ); RC=$?
tick "build rc=$RC"
echo "$(grep -cE '^\[[0-9]+/[0-9]+\] (Building|Linking)' "$LOG") steps run"
RES=0
if [ $RC -ne 0 ]; then
  RES=1
  echo "--- errors (full log $LOG)"
  grep -E "(error|undefined reference|multiple definition)" "$LOG" | grep -vE "^FAILED|^ninja:" \
    | sed -E "s#$W/##g; s#^\.\./(\.\./)*##" | awk '!s[$0]++' | cut -c1-230 | head -40
  echo "--- failed units"; grep -E "^FAILED:" "$LOG" | sed -E 's/^FAILED: (\[[^]]*\] )?//' | cut -c1-150 | head -20
fi

# 5. host-only tests: through ctest, GPU hidden so the stand's running ctest is undisturbed
if [ -z "$QC_NO_TESTS" ] && [ -n "$EXES" ]; then
  RX=$(for e in $EXES; do [ -x "$(find "$B" -name "$e" -type f -perm -u+x 2>/dev/null | head -1)" ] && echo -n "^$e\$|"; done)
  RX=${RX%|}
  if [ -n "$RX" ]; then
    echo "--- tests (CUDA_VISIBLE_DEVICES empty, timeout ${QC_TEST_TIMEOUT:-300} s)"
    ( cd "$B" && CUDA_VISIBLE_DEVICES= $LOW ctest -R "$RX" --timeout ${QC_TEST_TIMEOUT:-300} --output-on-failure > "$L/qc_ctest_$TAG.log" 2>&1 )
    grep -E "Test +#[0-9]+:" "$L/qc_ctest_$TAG.log" | sed -E 's/^ *[0-9]+\/[0-9]+ //' | cut -c1-120
    grep -E "tests passed|tests failed" "$L/qc_ctest_$TAG.log"
    # a failure whose output says there is no CUDA device only needs a GPU: not counted
    awk '
      function flush() { if (name != "") { if (bad && gpu) print "  needs-GPU (not run): " name; else if (bad) { print "  REAL FAILURE: " name; real++ } } name = ""; bad = gpu = 0 }
      /^ *[0-9]+\/[0-9]+ Test +#/ { flush(); name = $0; sub(/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: /, "", name); sub(/ \.+.*/, "", name); bad = ($0 ~ /\*\*\*|Failed|Exception|Timeout/) }
      /cudaErrorNoDevice|no CUDA-capable device|CUDA_VISIBLE_DEVICES/ { gpu = 1 }
      END { flush(); exit real ? 1 : 0 }' "$L/qc_ctest_$TAG.log"
    if [ $? -ne 0 ]; then
      [ $RES -eq 0 ] && RES=2
      echo "  failing output (log $L/qc_ctest_$TAG.log):"
      grep -vE "^(Test project|Constructing|\s*Start|\s*[0-9]+/[0-9]+ Test|$)" "$L/qc_ctest_$TAG.log" | grep -iE "error|fail|assert|expect|what\(\)" | grep -v "cudaErrorNoDevice" | sed -E "s#$W/##g" | cut -c1-200 | head -12
    fi
  else
    echo "--- no test executable was built, nothing to run"
  fi
fi
echo "== qc $BR: $([ $RES -eq 0 ] && echo OK || echo "FAIL($RES)") in $(( $(date +%s) - T0 )) s; build_qc $(du -s --block-size=1G "$B" | cut -f1) GB"
exit $RES
