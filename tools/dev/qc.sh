#!/bin/bash
# qc.sh BRANCH [TARGET...] : quick compile check of a Brontir/NInferno branch on a stand's CPU.
#
# Compiles only what the branch touches, in one persistent build dir and one worktree, at the lowest
# CPU and I/O priority so a ctest running on the stand is not starved. Never creates a second build.
#   - checks out origin/BRANCH (detached, cleaned) in $QC_ROOT/wt_qc;
#   - without TARGETs: maps `git diff origin/master...origin/BRANCH` (deleted and renamed-away files
#     included) to ninja targets:
#       * the object of every changed .cpp/.cu (compile only, no link),
#       * the objects of the sources that include a changed header (transitively, by file name),
#       * the test executables those sources belong to or that the diff adds to a tests cmake file;
#   - builds with `nice -n 19 ionice -c3 ninja -k 0 -j4`;
#   - runs the test executables this run linked (or found current) through ctest, by the ctest test
#     name that ctest itself reports for the executable, with CUDA_VISIBLE_DEVICES empty (no GPU use);
#   - prints compile errors as file:line, a result table, and what was NOT verified.
# Exit: 0 everything the diff touches was compiled and its tests passed (nothing skipped or capped)
#       1 compile/link/cmake/plan error (also: a real diff that yields an empty plan)
#       2 a test failed
#       3 usage/setup (fetch, disk, internal error)
#       4 partial: no error, but something was capped, skipped, not linked, needs a GPU or has no test
# Env: QC_ROOT (/root/claude) QC_ARCH (89) QC_JOBS (4) QC_MAX_OBJ (60) QC_MAX_CU (40)
#      QC_MAX_UNITS (150, CUDA+C++ per target) QC_TEST_TIMEOUT (300 s per test)
#      QC_MAX_GB (40, abort when the build dir is larger) QC_MIN_FREE_GB (15) QC_NO_TESTS=1 (build only)
set -Eeuo pipefail
trap 'rc=$?; echo "qc.sh: internal error (exit $rc) at line $LINENO" >&2; exit 3' ERR
BR=${1:-}; [ -n "$BR" ] || { echo "usage: qc.sh BRANCH [TARGET...]" >&2; exit 3; }
shift
ROOT=${QC_ROOT:-/root/claude}; ARCH=${QC_ARCH:-89}; JOBS=${QC_JOBS:-4}
W=$ROOT/wt_qc; B=$ROOT/build_qc_$ARCH; L=$ROOT/logs; mkdir -p "$L"
exec 9>"$ROOT/qc.lock"; flock -n 9 || { echo "qc.sh: another qc.sh run is active, waiting"; flock 9; }
set +u; . "$ROOT/lab.sh"; cuda_env; set -u
export CCACHE_DIR=/root/.ccache CCACHE_BASEDIR=$W CCACHE_NOHASHDIR=1
LOW="nice -n 19 ionice -c3"
T0=$(date +%s); tick() { echo "  [$(( $(date +%s) - T0 ))s] $*"; }
TAG=$(echo "$BR" | tr / _)
PLAN=$(mktemp); trap 'rm -f "$PLAN" "$PLAN.py"' EXIT
PARTIAL=()   # everything the run did not verify

# 0. disk
FREE=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
[ "$FREE" -ge "${QC_MIN_FREE_GB:-15}" ] || { echo "only $FREE GB free on $ROOT (< ${QC_MIN_FREE_GB:-15}); clean up first"; exit 3; }

# 1. fetch + checkout (the build dir lives outside the worktree, so a clean is safe)
[ -d "$W" ] || git -C "$ROOT/wt_tst" worktree add -q --detach "$W" origin/master
git -C "$W" fetch -q origin master "+refs/heads/$BR:refs/remotes/origin/$BR" || { echo "fetch of $BR failed"; exit 3; }
git -C "$W" checkout -q -f --detach "origin/$BR"
git -C "$W" clean -fdxq
echo "== qc $BR at $(git -C "$W" log --oneline -1 | cut -c1-100)"
tick fetched
GB=$(du -s --block-size=1G "$B" 2>/dev/null | cut -f1 || true); GB=${GB:-0}
[ "$GB" -le "${QC_MAX_GB:-40}" ] || { echo "build_qc is $GB GB (> ${QC_MAX_GB:-40}); rm -rf $B to restart"; exit 3; }

# 2. configure once (same options as kt3.sh); later branch switches re-run cmake through ninja
if [ ! -f "$B/build.ninja" ]; then
  cmake -S "$W" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$ARCH \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DBUILD_TESTING=ON \
    > "$L/qc_cmake.log" 2>&1 || { tail -15 "$L/qc_cmake.log"; echo "CONFIGURE FAILED"; exit 1; }
  tick configured
fi
( cd "$B" && $LOW ninja build.ninja > "$L/qc_regen.log" 2>&1 ) || { tail -15 "$L/qc_regen.log"; echo "CMAKE REGENERATE FAILED (cmake error on this branch)"; exit 1; }

# 3. plan. Lines: EXE<TAB>target<TAB>ctest names (comma) | OBJ<TAB>target | SKIP<TAB>text | NOOP<TAB>text
cat > "$PLAN.py" <<'PY'
import sys, os, re, json, subprocess, collections
src, bld, br = sys.argv[1:4]; explicit = sys.argv[4:]
MAXOBJ = int(os.environ.get("QC_MAX_OBJ", "60")); MAXCU = int(os.environ.get("QC_MAX_CU", "40"))
MAXU = int(os.environ.get("QC_MAX_UNITS", "150"))
def sh(*a):  # a failing git/ninja/ctest call fails the plan, it never reads as "empty"
    return subprocess.run(a, capture_output=True, text=True, check=True).stdout
def note(*a): print(*a, file=sys.stderr)
def out(*a): print(*a, sep="\t")
comp = json.loads(sh("ninja", "-C", bld, "-t", "compdb"))
obj_of = collections.defaultdict(list)
for e in comp:
    o = e.get("output", "")
    if not o.endswith((".o", ".obj")): continue
    f = e["file"]; f = f if os.path.isabs(f) else os.path.join(e["directory"], f)
    obj_of[os.path.realpath(f)].append(o)
exe_of = {}  # executable file name -> ninja target path
for line in sh("ninja", "-C", bld, "-t", "targets", "all").splitlines():
    m = re.match(r"^(\S+): \w+_EXECUTABLE_LINKER", line)
    if m: exe_of[os.path.basename(m.group(1))] = m.group(1)
# ctest test names per executable, as ctest itself registers them (command or any argument)
tests_of = collections.defaultdict(list)
for t in json.loads(sh("ctest", "--test-dir", bld, "--show-only=json-v1")).get("tests", []):
    for a in t.get("command", []):
        b = os.path.basename(str(a))
        if b in exe_of and t["name"] not in tests_of[b]: tests_of[b].append(t["name"])
def owner(obj):
    m = re.search(r"CMakeFiles/([^/]+)\.dir/", obj); return m.group(1) if m else None
deps, cur = {}, None   # object -> recorded header dependencies (ninja's depfile log)
for line in sh("ninja", "-C", bld, "-t", "deps").splitlines():
    if line and not line[0].isspace() and ": #deps" in line: cur = line.split(": #deps")[0]; deps[cur] = []
    elif line.strip() and cur: deps[cur].append(line.strip())
def mt(p):
    try: return os.stat(os.path.join(bld, p)).st_mtime
    except OSError: return None
def stale_units(target):  # compile commands in the target's closure whose object is missing or older than a dependency
    cu = cx = 0
    for cmd in sh("ninja", "-C", bld, "-t", "commands", target).splitlines():
        m = re.search(r" -o (\S+\.o)", cmd)
        if not m or " -c " not in cmd: continue
        o = m.group(1); t = mt(o)
        if t is None or any((mt(d) or 0) > t for d in deps.get(o, ())):
            if "nvcc" in cmd: cu += 1
            else: cx += 1
    return cu, cx
def emit_exe(path):
    out("EXE", path, ",".join(tests_of.get(os.path.basename(path), [])))
objs, exes = [], []
if explicit:  # the caller names the targets: an executable by file name or path, anything else as ninja target
    for t in explicit:
        if t in exe_of: exes.append(exe_of[t])
        elif os.path.basename(t) in exe_of and t == exe_of[os.path.basename(t)]: exes.append(t)
        else: objs.append(t)
else:
    rng = f"origin/master...origin/{br}"
    changed = sh("git", "-C", src, "diff", "--no-renames", "--name-only", rng).split()  # deleted and renamed-away included
    if not changed: out("NOOP", "no difference from master"); sys.exit(0)
    srcx, hdrx = (".cpp", ".cu", ".cc"), (".h", ".hpp", ".cuh", ".inl")
    relevant = [c for c in changed if c.endswith(srcx + hdrx + (".cmake", ".in")) or os.path.basename(c) == "CMakeLists.txt"]
    if not relevant:
        out("NOOP", f"{len(changed)} changed file(s), none compilable (docs/scripts/data)"); sys.exit(0)
    includers = collections.defaultdict(set)
    for d in ("src", "include", "tests", "apps", "bench"):
        for root, _, fs in os.walk(os.path.join(src, d)):
            for fn in fs:
                if not fn.endswith(srcx + hdrx): continue
                p = os.path.join(root, fn)
                try: txt = open(p, errors="ignore").read()
                except OSError: continue
                for inc in re.findall(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', txt, re.M):
                    includers[os.path.basename(inc)].add(os.path.relpath(p, src))
    direct = [c for c in changed if c.endswith(srcx) and os.path.exists(os.path.join(src, c))]
    hdrs = [c for c in changed if c.endswith(hdrx)]
    seen, queue, via = set(hdrs), list(hdrs), set()
    while queue:
        h = queue.pop()
        for p in includers.get(os.path.basename(h), ()):
            if p.endswith(srcx): via.add(p)
            elif p not in seen: seen.add(p); queue.append(p)
    via -= set(direct)
    cdirs = {os.path.dirname(c) for c in changed}
    prio = lambda p: (0 if os.path.dirname(p) in cdirs else 1 if p.startswith("tests/") else 2, p)
    def objs_of(paths, what):
        r = []
        for p in paths:
            o = obj_of.get(os.path.realpath(os.path.join(src, p)))
            if o: r += o
            else: out("SKIP", f"{what} {p} has no object in the configured build graph")
        return r
    o_direct = objs_of(sorted(direct), "changed source")
    o_via = objs_of(sorted(via, key=prio), "header-dependent source")
    room = max(0, MAXOBJ - len(o_direct))
    if len(o_via) > room:
        out("SKIP", f"QC_MAX_OBJ={MAXOBJ} cap: {len(o_via) - room} of {len(o_via)} objects that include a changed header not compiled")
        o_via = o_via[:room]
    allobjs = o_direct + o_via
    tests = {owner(o) for o in allobjs if o.startswith("tests/") or "/tests/" in o}
    tests.discard(None)
    cm = sh("git", "-C", src, "diff", "--no-renames", "-U0", rng, "--", "*.cmake", "CMakeLists.txt")
    for l in cm.splitlines():
        if l.startswith("+") and not l.startswith("+++"):
            m = re.search(r"(?:ninferno_add_test|add_executable|add_test)\(\s*(?:NAME\s+)?([A-Za-z0-9_]+)", l)
            if m: tests.add(m.group(1))
    exes = sorted(exe_of[t] for t in tests if t in exe_of)
    objs = allobjs
keep = []
for e in exes:
    cu, cx = stale_units(e)
    if cu > MAXCU or cu + cx > MAXU:
        out("SKIP", f"link of {e} skipped, {cu} CUDA + {cx} C++ stale units behind it (QC_MAX_CU={MAXCU}, QC_MAX_UNITS={MAXU}); not tested")
        note(f"note: {e}: link skipped ({cu} CUDA + {cx} C++ stale)")
    else:
        note(f"test target {e}: {cu} CUDA + {cx} C++ stale units"); keep.append(e)
keepobj = []
for o in objs:
    if explicit: keepobj.append(o); continue
    cu, cx = stale_units(o)
    if cu > MAXCU or cu + cx > MAXU:
        out("SKIP", f"object {o} skipped, its dependencies need {cu} CUDA + {cx} C++ units (QC_MAX_CU={MAXCU}, QC_MAX_UNITS={MAXU})")
    else: keepobj.append(o)
for e in keep: emit_exe(e)
for o in sorted(set(keepobj)): out("OBJ", o)
if not keep and not keepobj:
    note("empty plan for a diff that touches the build"); sys.exit(1)
PY
ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ] && [ "$BR" = master ]; then ARGS=(ninferno_spec_test ninferno_spec_round_cost_test ninferno_choice_test); echo "master has no diff against itself: smoke targets"; fi
python3 -I "$PLAN.py" "$W" "$B" "$BR" "${ARGS[@]}" > "$PLAN" 2> "$L/qc_plan_$TAG.err" \
  || { cat "$L/qc_plan_$TAG.err"; echo "PLAN FAILED (exit 1): the plan could not be made or came out empty for a diff that touches the build"; exit 1; }
cat "$L/qc_plan_$TAG.err"
while IFS=$'\t' read -r kind a b; do if [ "$kind" = SKIP ]; then PARTIAL+=("$a"); fi; done < "$PLAN"
if grep -q '^NOOP' "$PLAN"; then echo "nothing to compile: $(cut -f2 "$PLAN" | head -1)"; exit 0; fi
TARGETS=$(awk -F'\t' '$1=="EXE"||$1=="OBJ"{print $2}' "$PLAN" | sort -u | tr '\n' ' ')
NT=$(echo "$TARGETS" | wc -w)
tick "$NT ninja targets"

# 4. build (lowest priority, keep going to collect every error)
LOG=$L/qc_ninja_$TAG.log
BSTART=$(date +%s)
RC=0; ( cd "$B" && $LOW ninja -k 0 -j"$JOBS" $TARGETS > "$LOG" 2>&1 ) || RC=$?
if [ $RC -ge 128 ]; then   # ninja itself aborted (a dyndep assertion in ninja 1.13 seen once): one retry
  echo "  ninja aborted (rc=$RC), retrying once"; tail -2 "$LOG" | cut -c1-200
  ( cd "$B" && ninja -t recompact > /dev/null 2>&1 ) || true
  RC=0; ( cd "$B" && $LOW ninja -k 0 -j"$JOBS" $TARGETS > "$LOG" 2>&1 ) || RC=$?
fi
tick "build rc=$RC"
echo "$(grep -cE '^\[[0-9]+/[0-9]+\] (Building|Linking)' "$LOG" || true) steps run"
RES=0
if [ $RC -ne 0 ]; then
  RES=1
  echo "--- errors (full log $LOG)"
  { grep -E "(error|undefined reference|multiple definition)" "$LOG" | grep -vE "^FAILED|^ninja:" \
    | sed -E "s#$W/##g; s#^\.\./(\.\./)*##" | awk '!s[$0]++' | cut -c1-230 | head -40; } || true
  echo "--- failed units"; { grep -E "^FAILED:" "$LOG" | sed -E 's/^FAILED: (\[[^]]*\] )?//' | cut -c1-150 | head -20; } || true
fi

# 5. tests: only executables ninja reports current (linked by this run or already up to date for this
#    checkout), by the ctest name ctest registers; the GPU is hidden so the stand's ctest is undisturbed
EXPECT=()
while IFS=$'\t' read -r kind path names; do
  [ "$kind" = EXE ] || continue
  if [ -n "${QC_NO_TESTS:-}" ]; then continue; fi
  if [ -z "$names" ]; then PARTIAL+=("executable $(basename "$path") built but no ctest test runs it"); continue; fi
  # current = the whole build succeeded, or this very run relinked the file (never a leftover of another branch)
  MT=$(stat -c %Y "$B/$path" 2>/dev/null || echo 0)
  if [ $RC -eq 0 ] || [ "$MT" -ge "$BSTART" ]; then
    IFS=, read -ra ns <<< "$names"; EXPECT+=("${ns[@]}")
  else
    PARTIAL+=("executable $(basename "$path") was not relinked by this run (link failed or skipped); not tested")
  fi
done < "$PLAN"
[ -z "${QC_NO_TESTS:-}" ] || PARTIAL+=("tests not run (QC_NO_TESTS)")
if [ ${#EXPECT[@]} -gt 0 ]; then
  RX=$(printf '%s|' "${EXPECT[@]}"); RX="^(${RX%|})\$"
  echo "--- tests: ${#EXPECT[@]} (CUDA_VISIBLE_DEVICES empty, timeout ${QC_TEST_TIMEOUT:-300} s)"
  CLOG=$L/qc_ctest_$TAG.log; CRC=0
  ( cd "$B" && CUDA_VISIBLE_DEVICES= $LOW ctest -j1 -R "$RX" --timeout "${QC_TEST_TIMEOUT:-300}" --output-on-failure > "$CLOG" 2>&1 ) || CRC=$?
  { grep -E "Test +#[0-9]+:" "$CLOG" | sed -E 's/^ *[0-9]+\/[0-9]+ //' | cut -c1-120; } || true
  { grep -E "tests passed|tests failed" "$CLOG"; } || true
  # classify: SKIP (exit 77), GPU (CUDA device missing and no assertion failed), FAIL (anything else bad)
  CLASS=$(awk '
    function flush() { if (name == "") return
      if (st == "skip") print "SKIP " name; else if (st == "bad") print ((nodev && !asrt) ? "GPU " : "FAIL ") name; name = "" }
    /^ *[0-9]+\/[0-9]+ Test +#/ { flush(); name = $0; sub(/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: /, "", name); sub(/ \.+.*/, "", name); nodev = asrt = 0
      st = ($0 ~ /\*\*\*Skipped/) ? "skip" : ($0 ~ /\*\*\*(Failed|Exception|Not Run|Timeout)/) ? "bad" : "ok"; if ($0 ~ /\*\*\*Timeout/) asrt = 1 }
    /cudaErrorNoDevice|no CUDA-capable device/ { nodev = 1 }
    /FAIL:|Assertion|assert|EXPECT_|CHECK failed/ { asrt = 1 }
    END { flush() }' "$CLOG")
  NFAIL=$(echo "$CLASS" | grep -c '^FAIL ' || true)
  GPUN=$(echo "$CLASS" | sed -n 's/^GPU //p' | tr '\n' ' '); SKIPN=$(echo "$CLASS" | sed -n 's/^SKIP //p' | tr '\n' ' ')
  [ -z "$GPUN" ] || PARTIAL+=("tests that need a GPU, not run: $GPUN")
  [ -z "$SKIPN" ] || PARTIAL+=("tests that skipped themselves (77): $SKIPN")
  SEEN=$(grep -oE "Test +#[0-9]+: [A-Za-z0-9_.+-]+" "$CLOG" | sed -E 's/.*: //' | sort -u || true)
  for n in "${EXPECT[@]}"; do grep -qx "$n" <<< "$SEEN" || { PARTIAL+=("ctest did not run $n"); }; done
  if [ "$NFAIL" -gt 0 ]; then
    [ $RES -ne 0 ] || RES=2
    echo "$CLASS" | sed -n 's/^FAIL /  FAILED TEST: /p'
    echo "  failing output (log $CLOG):"
    { grep -vE "^(Test project|Constructing|\s*Start|\s*[0-9]+/[0-9]+ Test|$)" "$CLOG" | grep -iE "error|fail|assert|expect|what\(\)" | sed -E "s#$W/##g" | cut -c1-200 | head -12; } || true
  elif [ "$CRC" -ne 0 ] && [ -z "$GPUN" ] && [ -z "$SKIPN" ]; then
    [ $RES -ne 0 ] || RES=2
    echo "  ctest exited $CRC without a classified failure (log $CLOG)"; tail -5 "$CLOG"
  fi
fi

# 6. verdict: exit 0 only when nothing was capped, skipped or left untested
if [ $RES -eq 0 ] && [ ${#PARTIAL[@]} -gt 0 ]; then RES=4; fi
if [ ${#PARTIAL[@]} -gt 0 ]; then
  echo "--- NOT VERIFIED (${#PARTIAL[@]}):"; for p in "${PARTIAL[@]}"; do echo "  - ${p:0:300}"; done
fi
VERDICT=OK; [ $RES -eq 0 ] || VERDICT="FAIL($RES)"; [ $RES -ne 4 ] || VERDICT="PARTIAL(4)"
echo "== qc $BR: $VERDICT in $(( $(date +%s) - T0 )) s; build_qc $(du -s --block-size=1G "$B" | cut -f1) GB"
exit $RES
