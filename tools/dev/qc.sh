#!/bin/bash
# qc.sh BRANCH [TARGET...] : quick compile check of a Brontir/NInferno branch on a build machine's CPU.
#
# Compiles only what the branch touches, in one persistent build dir and one worktree, at the lowest
# CPU and I/O priority so a ctest running on the machine is not starved. Never creates a second build.
#   - checks out origin/BRANCH (detached, cleaned) in $QC_ROOT/wt_qc;
#   - without TARGETs: maps `git diff origin/master...origin/BRANCH` (deleted and renamed-away files
#     included) to ninja targets:
#       * the object of every changed .c/.cpp/.cu (compile only, no link),
#       * the objects of the sources that include a changed header (transitively, by file name; also
#         third_party),
#       * the objects that embed a changed data file (ninferno_embed_text and friends, found in
#         build.ninja),
#       * the ctest tests of the test executables those objects belong to, the tests whose command
#         names a changed file (python tests), and the tests the diff adds to a cmake file; every
#         executable such a test runs is built too (apps, helpers), so no binary of another branch runs;
#   - builds with `nice -n 19 ionice -c3 ninja -k 0 -j4`;
#   - runs those tests through ctest -V with CUDA_VISIBLE_DEVICES empty (no GPU use), and scans passing
#     tests' output for skip / no-device markers;
#   - prints compile errors as file:line, a result table, and what was NOT verified.
# Exit: 0 everything the diff touches was compiled and its tests passed (nothing skipped or capped)
#       1 compile/link/cmake/plan error (also: a real diff that yields an empty plan with nothing skipped)
#       2 a test failed or crashed
#       3 usage/setup (fetch, disk, wrong worktree or build dir, internal error)
#       4 partial: no error, but something was capped, skipped, compiled but not linked or tested,
#         needs a GPU, printed a skip marker, or has no test
# Exit 0 with "nothing to compile" only when every changed file is documentation (docs/, *.md, .github/).
# Env: QC_ROOT (work dir, default: current directory) QC_REPO (an existing checkout of the repo, default $QC_ROOT/wt_tst)
#      QC_ENV_SH (optional script defining cuda_env, default $QC_ROOT/lab.sh if present) QC_ARCH (89) QC_JOBS (4)
#      QC_MAX_OBJ (60) QC_MAX_CU (40)
#      QC_MAX_UNITS (150, CUDA+C++ per target) QC_TEST_TIMEOUT (300 s per test)
#      QC_MAX_GB (40, abort when the build dir is larger) QC_MIN_FREE_GB (15) QC_NO_TESTS=1 (build only)
set -Eeuo pipefail
trap 'rc=$?; echo "qc.sh: internal error (exit $rc) at line $LINENO" >&2; exit 3' ERR
BR=${1:-}; [ -n "$BR" ] || { echo "usage: qc.sh BRANCH [TARGET...]" >&2; exit 3; }
shift
ROOT=$(realpath "${QC_ROOT:-$PWD}"); REPO=${QC_REPO:-$ROOT/wt_tst}; ARCH=${QC_ARCH:-89}; JOBS=${QC_JOBS:-4}
W=$ROOT/wt_qc; B=$ROOT/build_qc_$ARCH; L=$ROOT/logs; mkdir -p "$L"
exec 9>"$ROOT/qc.lock"; flock -n 9 || { echo "qc.sh: another qc.sh run is active, waiting"; flock 9; }
ENV_SH=${QC_ENV_SH:-$ROOT/lab.sh}
set +u; if [ -f "$ENV_SH" ]; then . "$ENV_SH"; cuda_env; fi; set -u
export CCACHE_DIR=${CCACHE_DIR:-$HOME/.ccache} CCACHE_BASEDIR=$W CCACHE_NOHASHDIR=1
LOW="nice -n 19 ionice -c3"
T0=$(date +%s); export QC_T0=$T0; tick() { echo "  [$(( $(date +%s) - T0 ))s] $*"; }
TAG=$(echo "$BR" | tr / _)
PLAN=$(mktemp); trap 'rm -f "$PLAN" "$PLAN.py"' EXIT
PARTIAL=()   # everything the run did not verify

# 0. disk
FREE=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
[ "$FREE" -ge "${QC_MIN_FREE_GB:-15}" ] || { echo "only $FREE GB free on $ROOT (< ${QC_MIN_FREE_GB:-15}); clean up first"; exit 3; }

# 1. fetch + checkout (the build dir lives outside the worktree, so a clean is safe)
[ -d "$W" ] || git -C "$REPO" worktree add -q --detach "$W" origin/master
[ "$(realpath "$(git -C "$W" rev-parse --show-toplevel)")" = "$(realpath "$W")" ] || { echo "$W is not its own git worktree root; refusing checkout -f / clean"; exit 3; }
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
    > "$L/qc_cmake.log" 2>&1 9>&- || { tail -15 "$L/qc_cmake.log"; echo "CONFIGURE FAILED"; exit 1; }
  tick configured
fi
HOMEDIR=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$B/CMakeCache.txt")
[ "$(realpath "$HOMEDIR")" = "$(realpath "$W")" ] || { echo "$B was configured for '$HOMEDIR', not $W; remove it (rm -rf $B) and rerun"; exit 3; }
( cd "$B" && $LOW ninja build.ninja > "$L/qc_regen.log" 2>&1 9>&- ) || { tail -15 "$L/qc_regen.log"; echo "CMAKE REGENERATE FAILED (cmake error on this branch)"; exit 1; }

# 3. plan. Lines: TEST<TAB>ctest name<TAB>exe targets (comma) | OBJ<TAB>target | SKIP<TAB>text | NOOP<TAB>text
cat > "$PLAN.py" <<'PY'
import sys, os, re, json, subprocess, collections
src, bld, br = sys.argv[1:4]; explicit = sys.argv[4:]
MAXOBJ = int(os.environ.get("QC_MAX_OBJ", "60")); MAXCU = int(os.environ.get("QC_MAX_CU", "40"))
MAXU = int(os.environ.get("QC_MAX_UNITS", "150"))
def sh(*a):  # a failing git/ninja/ctest call fails the plan, it never reads as "empty"
    return subprocess.run(a, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout
def note(*a): print(*a, file=sys.stderr)
nskip = 0
def out(*a):
    global nskip
    if a and a[0] == "SKIP": nskip += 1
    print(*a, sep="\t")
srcx, hdrx = (".cpp", ".cu", ".cc", ".c"), (".h", ".hpp", ".cuh", ".inl", ".inc")
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
# every ctest test: its executables (basenames in the command) and its arguments
cmd_of = {}
for t in json.loads(sh("ctest", "--test-dir", bld, "--show-only=json-v1")).get("tests", []):
    cmd_of[t["name"]] = [str(a) for a in t.get("command", [])]
# ctest leaves the command out for a test whose executable is not built yet (a test the branch
# adds): the generated CTestTestfile.cmake still names it
ADD_TEST = re.compile(r'^add_test\(\[=\[(.+?)\]=\] "([^"]+)"(.*)\)\s*$', re.M)
for root, _, fs in os.walk(bld):
    if "CTestTestfile.cmake" not in fs: continue
    try: txt = open(os.path.join(root, "CTestTestfile.cmake"), encoding="utf-8", errors="replace").read()
    except OSError: continue
    for name, exe, rest in ADD_TEST.findall(txt):
        if name in cmd_of and not any(os.path.basename(a) == os.path.basename(exe) for a in cmd_of[name]):
            cmd_of[name] = [exe] + re.findall(r'"([^"]*)"', rest)
exes_of_test = {n: sorted({os.path.basename(a) for a in c if os.path.basename(a) in exe_of}) for n, c in cmd_of.items()}
tests_of_exe = collections.defaultdict(list)
for n, es in exes_of_test.items():
    for e in es: tests_of_exe[e].append(n)
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
objs, tests = [], set()
def add_exe_tests(e):  # the ctest tests that run executable e; none is reported
    ts = tests_of_exe.get(e, [])
    if ts: tests.update(ts)
    else: out("SKIP", f"executable {e} has no ctest test")
if explicit:  # the caller names the targets: a ctest test or executable by name, anything else as ninja target
    for t in explicit:
        if t in cmd_of: tests.add(t)
        elif t in exe_of: add_exe_tests(t)
        else: objs.append(t); out("SKIP", f"explicit target {t} is not a test: built only, nothing run")
else:
    rng = f"origin/master...origin/{br}"
    changed = [c for c in sh("git", "-C", src, "diff", "--no-renames", "--name-only", "-z", rng).split("\0") if c]  # deleted and renamed-away included
    if not changed: out("NOOP", "no difference from master"); sys.exit(0)
    docs = lambda f: f.startswith(("docs/", ".github/")) or f.endswith(".md") or f in ("LICENSE", "NOTICE", ".gitignore", ".clang-format")
    if all(docs(c) for c in changed):
        out("NOOP", f"{len(changed)} changed file(s), all documentation"); sys.exit(0)
    includers = collections.defaultdict(set)
    for d in ("src", "include", "tests", "apps", "bench", "third_party"):
        for root, _, fs in os.walk(os.path.join(src, d)):
            for fn in fs:
                if not fn.endswith(srcx + hdrx + tuple(h + ".in" for h in hdrx)): continue
                p = os.path.join(root, fn)
                try: txt = open(p, errors="ignore").read()
                except OSError: continue
                for inc in re.findall(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', txt, re.M):
                    includers[os.path.basename(inc)].add(os.path.relpath(p, src))
    direct = [c for c in changed if c.endswith(srcx) and os.path.exists(os.path.join(src, c))]
    for c in changed:
        if c.endswith(srcx) and not os.path.exists(os.path.join(src, c)):
            out("SKIP", f"deleted source {c}: the targets that used it are not relinked, so a missing symbol would go unseen")
    hdrx_in = tuple(h + ".in" for h in hdrx)
    hdrs = [c for c in changed if c.endswith(hdrx)] + [c[:-3] for c in changed if c.endswith(hdrx_in)]
    extra_objs = []
    cmakefile = lambda f: f.endswith(".cmake") or os.path.basename(f) == "CMakeLists.txt"
    others = [c for c in changed if not c.endswith(srcx + hdrx + hdrx_in) and not docs(c) and not cmakefile(c)]
    ninja_lines = [l for l in open(os.path.join(bld, "build.ninja"), errors="ignore") if l.startswith("build ")] if others else []
    T0E = float(os.environ.get("QC_T0", "0"))
    def regenerated_headers_objs():  # objects including a header the regeneration just rewrote (configure-time embedding)
        r = set()
        for o, ds in ((o, ds) for o, ds in deps.items() if o.endswith(".o")):
            for d in ds:
                p = d if os.path.isabs(d) else os.path.join(bld, d)
                if p.startswith(bld + "/") and (mt(p) or 0) >= T0E - 2: r.add(o); break
        return sorted(r)
    gen_objs = regenerated_headers_objs() if others else []
    for f in others:   # data/script files: the python tests that name them, the objects that embed them
        via_test = via_embed = False
        for n, c in cmd_of.items():
            if any(a == f or a.endswith("/" + f) for a in c):
                tests.add(n); via_test = True
        rel, ab = os.path.relpath(os.path.join(src, f), bld), os.path.join(src, f)
        for l in ninja_lines:
            head, _, rest = l.partition(": ")
            if "build.ninja" in head.split() and ((" " + rel + " ") in (" " + rest + " ") or (" " + ab + " ") in (" " + rest + " ")) and gen_objs:
                extra_objs += gen_objs; via_embed = True   # a configure dependency: embedded at configure time
            if (" " + rel + " ") in (" " + rest + " ") or (" " + ab + " ") in (" " + rest + " ") or rest.endswith(" " + rel) or rest.endswith(" " + ab):
                for o in head[len("build "):].split():
                    if o == "|": break
                    g = os.path.realpath(os.path.join(bld, o))
                    if g in obj_of: extra_objs += obj_of[g]; via_embed = True
                    elif o.endswith(hdrx): hdrs.append(o); via_embed = True
        if not (via_test or via_embed): out("SKIP", f"changed file {f} maps to no object and no test (data/script input); nothing checks it")
        elif not via_test: out("SKIP", f"data file {f}: the object that embeds it is compiled; tests that read it at run time are not mapped")
    via = set()
    for h0 in hdrs:   # each changed header separately: one that reaches no source is reported
        seen, queue, mine = {h0}, [h0], set()
        while queue:
            h = queue.pop()
            names_h = {os.path.basename(h)}
            if h.endswith(hdrx_in): names_h.add(os.path.basename(h)[:-3])
            for p in set().union(*(includers.get(x, set()) for x in names_h)):
                if p.endswith(srcx): mine.add(p)
                elif p not in seen: seen.add(p); queue.append(p)
        if not mine: out("SKIP", f"changed header {h0} is included by no compiled source; nothing checks it")
        via |= mine
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
    o_direct = objs_of(sorted(direct), "changed source") + extra_objs
    o_via = objs_of(sorted(via, key=prio), "header-dependent source")
    room = max(0, MAXOBJ - len(o_direct))
    if len(o_via) > room:
        out("SKIP", f"QC_MAX_OBJ={MAXOBJ} cap: {len(o_via) - room} of {len(o_via)} objects that include a changed header not compiled")
        o_via = o_via[:room]
    objs = o_direct + o_via
    for o in objs:   # tests of the test executables the objects belong to
        if o.startswith("tests/") or "/tests/" in o:
            ow = owner(o)
            if ow in exe_of: add_exe_tests(ow)
    # cmake changes: every changed line, added or removed, must sit in a test/executable registration (or be a
    # comment); a registration of a literal name must name a ctest test or an executable that exists; anything
    # else (apps, libraries, options, an unparsable statement) is NOT VERIFIED
    names = set(cmd_of) | set(exe_of)
    REG = {"ninferno_add_test", "add_test", "add_executable", "set_tests_properties"}
    is_reg = lambda cmd: cmd in REG or re.search(r"add_\w*test", cmd) is not None
    IDENT = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(")
    def parse_cmake(text):  # statements as (first line, last line, command, args, unclosed); strings, [[..]] and comments skipped
        res, n, i, line = [], len(text), 0, 1
        def bracket(i):  # end index of a bracket argument/comment starting at i, or -1
            m = re.match(r"\[(=*)\[", text[i:i + 40])
            if not m: return -1
            e = text.find("]" + m.group(1) + "]", i + len(m.group(0)))
            return n if e < 0 else e + len(m.group(1)) + 2
        while i < n:
            c = text[i]
            if c == "\n": line += 1; i += 1; continue
            if c == "#":
                j = bracket(i + 1)
                if j < 0: j = text.find("\n", i); j = n if j < 0 else j
                line += text.count("\n", i, j); i = j; continue
            m = IDENT.match(text, i)
            if not m: i += 1; continue
            start, cmd, i = line, m.group(1).lower(), m.end()
            depth, args, cur, have = 1, [], "", False
            while i < n and depth > 0:
                c = text[i]
                if c == "\n": line += 1
                if c in " \t\r\n":
                    if have: args.append(cur); cur, have = "", False
                    i += 1
                elif c == "#":
                    j = bracket(i + 1)
                    if j < 0: j = text.find("\n", i); j = n if j < 0 else j
                    line += text.count("\n", i, j); i = j
                elif c == '"':
                    k = i + 1
                    while k < n and text[k] != '"': k += 2 if text[k] == "\\" else 1
                    cur += text[i + 1:k]; have = True; line += text.count("\n", i, k); i = k + 1
                elif c == "[" and bracket(i) > 0:
                    j = bracket(i); cur += text[i:j]; have = True; line += text.count("\n", i, j); i = j
                elif c == "\\": cur += text[i:i + 2]; have = True; line += text.count("\n", i, i + 2); i += 2   # an escape: \( \) \# are text
                elif c == "(": depth += 1; i += 1
                elif c == ")":
                    depth -= 1; i += 1
                    if depth == 0 and have: args.append(cur); cur, have = "", False
                else: cur += c; have = True; i += 1
            res.append((start, line, cmd, args, depth > 0 or line - start > 300))
        return res
    def reg_name(st):  # (literal registered name or None, is a variable-built name)
        cmd, a = st[2], st[3]
        if cmd == "set_tests_properties": return None, True
        k = 1 if cmd == "add_test" and len(a) > 1 and a[0].upper() == "NAME" else 0
        if len(a) <= k: return None, True
        v = a[k]
        return (v, False) if re.fullmatch(r"[A-Za-z0-9_.+-]+", v) else (None, True)
    def pass_statement(st, side):  # True when the changed line in statement st is accounted for
        if st[4] or not is_reg(st[2]): return False
        lit, var = reg_name(st)
        if st[2] == "set_tests_properties":
            ts = [t for t in (st[3][:st[3].index("PROPERTIES")] if "PROPERTIES" in st[3] else st[3])]
            if all(t in cmd_of for t in ts): tests.update(ts); return True
            return False
        if lit is not None:
            if lit in cmd_of: tests.add(lit); return True
            if lit in exe_of: add_exe_tests(lit); return True
            if side == "old": return True   # a registration that is gone: nothing to run
            out("SKIP", f"cmake registers '{lit}' but ctest lists no test or executable of that name; not checked")
            return False
        # a variable-built name (moe_random${share}): accounted for only through the executable its
        # COMMAND runs, whose every test then runs; other arguments naming a test or an exe prove nothing
        a = st[3]
        cmd_at = next((i for i, x in enumerate(a) if x.upper() == "COMMAND"), None)
        exe = a[cmd_at + 1] if cmd_at is not None and cmd_at + 1 < len(a) else None
        if exe is not None and exe in exe_of:
            add_exe_tests(exe); return True
        return False
    # every statement a line sits in (two statements may share one line)
    at = lambda sts, ln: [x for x in sts if x[0] <= ln <= x[1]]
    cm = sh("git", "-C", src, "diff", "--no-renames", "--no-ext-diff", "--no-textconv", "--no-color", "-U0", rng, "--", "*.cmake", "*CMakeLists.txt")
    base = sh("git", "-C", src, "merge-base", "origin/master", f"origin/{br}").strip()   # a removed line's numbers are the merge-base's
    bad_lines, new_cache, old_cache = [], {}, {}
    def sts_of(path, side):   # (statements, source lines), or None when the file cannot be read
        cache = new_cache if side == "new" else old_cache
        if path not in cache:
            try:
                text = open(os.path.join(src, path), encoding="utf-8", errors="strict").read() if side == "new" else sh("git", "-C", src, "show", f"{base}:{path}")
                sts, lines = parse_cmake(text), text.split("\n")
                # the bodies of function/macro/foreach blocks: a change there reaches every caller
                bodies, stack = [], []
                for st in sts:
                    if st[2] in ("function", "macro", "foreach"): stack.append(st[0])
                    elif st[2] in ("endfunction", "endmacro", "endforeach") and stack: bodies.append((stack.pop(), st[1]))
                bodies += [(b, len(lines)) for b in stack]   # an unclosed block runs to the end
                cache[path] = (sts, lines, bodies)
            except (OSError, UnicodeDecodeError, subprocess.CalledProcessError): cache[path] = None
        return cache[path]
    def line_ok(path, side, ln):   # True when the changed line at ln is accounted for
        parsed = sts_of(path, side)
        if parsed is None: return False   # unreadable: not checked
        sts, lines, bodies = parsed
        if any(b <= ln <= e for b, e in bodies): return False   # inside a function/macro/foreach body
        here = at(sts, ln)
        if not here:   # outside every statement: only a blank or comment line passes
            t = lines[ln - 1].strip() if 0 < ln <= len(lines) else ""
            return not t or t.startswith("#")
        return all(pass_statement(st, side) for st in here)
    old_file = new_file = None; nln = oln = 0; o_left = n_left = 0; unknown = False
    for l in cm.splitlines():
        if o_left == 0 and n_left == 0 and not l.startswith(("@@", "+", "-", " ")):
            continue   # "diff --git", "index", mode lines between files
        if o_left == 0 and n_left == 0 and l.startswith("--- "):
            old_file = l[6:] if l.startswith("--- a/") else None
            if old_file is None and l != "--- /dev/null": unknown = True
        elif o_left == 0 and n_left == 0 and l.startswith("+++ "):
            new_file = l[6:] if l.startswith("+++ b/") else None
            if new_file is None and l != "+++ /dev/null": unknown = True
        elif l.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))?", l)
            oln, nln = int(m.group(1)), int(m.group(3))
            o_left = int(m.group(2)) if m.group(2) is not None else 1
            n_left = int(m.group(4)) if m.group(4) is not None else 1
        elif l.startswith("+") and n_left > 0:
            if new_file is None or not line_ok(new_file, "new", nln): bad_lines.append(f"{new_file}:{nln}")
            nln += 1; n_left -= 1
        elif l.startswith("-") and o_left > 0:
            if old_file is None or not line_ok(old_file, "old", oln): bad_lines.append(f"{old_file}:{oln}(removed)")
            oln += 1; o_left -= 1
    if unknown: bad_lines.append("a diff header in a form this check does not read")
    seen_cm = {l[6:] for l in cm.splitlines() if l.startswith(("--- a/", "+++ b/"))}
    for c in changed:   # every changed cmake file must show up in the diff this check read
        if cmakefile(c) and c not in seen_cm: bad_lines.append(f"{c}(not in the cmake diff)")
    if bad_lines:
        uniq = sorted(set(bad_lines))
        out("SKIP", f"cmake change not mapped to a checked test/target ({len(uniq)} place(s), e.g. {', '.join(uniq[:3])}); configuring succeeded, but the targets it affects are not built")
# guard every test's executables (and the test objects/targets) against a large stale library build
for n in sorted(tests):   # a test that reads the build tree's objects runs over whatever build_qc last built
    c = cmd_of.get(n, [])
    if any(a.startswith("--build") for a in c) or any(a.endswith((".o", ".a", ".so")) for a in c):
        out("SKIP", f"test {n} reads the build tree's objects, which hold whatever build_qc last built; not verified for this branch")
ok_tests = {}
for n in sorted(tests):
    es = [exe_of[e] for e in exes_of_test.get(n, [])]
    bad = None
    for e in es:
        cu, cx = stale_units(e)
        if cu > MAXCU or cu + cx > MAXU: bad = (e, cu, cx); break
    if bad: out("SKIP", f"test {n} not run: link of {bad[0]} skipped, {bad[1]} CUDA + {bad[2]} C++ stale units behind it (QC_MAX_CU={MAXCU}, QC_MAX_UNITS={MAXU})")
    else: ok_tests[n] = es
keepobj = []
for o in objs:
    if explicit: keepobj.append(o); continue
    cu, cx = stale_units(o)
    if cu > MAXCU or cu + cx > MAXU:
        out("SKIP", f"object {o} skipped, its dependencies need {cu} CUDA + {cx} C++ units (QC_MAX_CU={MAXCU}, QC_MAX_UNITS={MAXU})")
    else: keepobj.append(o)
if not explicit:   # objects that no linked test executable contains were compiled only
    linked = {os.path.basename(e) for es in ok_tests.values() for e in es}
    lone = collections.defaultdict(list)
    for o in keepobj:
        if owner(o) not in linked: lone[owner(o) or "?"].append(o)
    for ow, os_ in sorted(lone.items()):
        out("SKIP", f"compiled, not linked, no test: {len(os_)} object(s) of target {ow} (e.g. {os.path.basename(os_[0])})")
for n, es in sorted(ok_tests.items()): out("TEST", n, ",".join(es))
for o in sorted(set(keepobj)): out("OBJ", o)
if not ok_tests and not keepobj:
    if explicit: note("empty plan for the targets given"); sys.exit(1)
    if not nskip: note("empty plan for a diff that touches the build"); sys.exit(1)
    note("everything the diff touches was skipped (see NOT VERIFIED)")
PY
ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ] && [ "$BR" = master ]; then ARGS=(ninferno_spec_test ninferno_spec_round_cost_test ninferno_choice_test); echo "master has no diff against itself: smoke targets"; fi
python3 -I "$PLAN.py" "$W" "$B" "$BR" "${ARGS[@]}" > "$PLAN" 2> "$L/qc_plan_$TAG.err" 9>&- \
  || { cat "$L/qc_plan_$TAG.err"; echo "PLAN FAILED (exit 1): the plan could not be made"; exit 1; }
cat "$L/qc_plan_$TAG.err"
while IFS=$'\t' read -r kind a b; do if [ "$kind" = SKIP ]; then PARTIAL+=("$a"); fi; done < "$PLAN"
if grep -q '^NOOP' "$PLAN"; then echo "nothing to compile: $(cut -f2 "$PLAN" | head -1)"; exit 0; fi
# a test needs its executables built from this checkout, so they are targets too
TARGETS=$(awk -F'\t' '$1=="OBJ"{print $2} $1=="TEST"{n=split($3,a,","); for(i=1;i<=n;i++) if(a[i]!="") print a[i]}' "$PLAN" | sort -u | tr '\n' ' ')
NT=$(echo "$TARGETS" | wc -w)
tick "$NT ninja targets"

# 4. build (lowest priority, keep going to collect every error)
LOG=$L/qc_ninja_$TAG.log
RC=0; BSTART=$(date +%s)
if [ "$NT" -gt 0 ]; then
  ( cd "$B" && $LOW ninja -k 0 -j"$JOBS" $TARGETS > "$LOG" 2>&1 9>&- ) || RC=$?
  if [ $RC -ge 128 ]; then   # ninja itself aborted (a dyndep assertion in ninja 1.13 seen): one retry
    echo "  ninja aborted (rc=$RC), retrying once"; tail -2 "$LOG" | cut -c1-200
    BUSY=0   # we hold the lock; compact the logs only when no other ninja works in this build dir
    for p in $(pgrep -x ninja || true); do if [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$B" ]; then BUSY=1; fi; done
    [ $BUSY -ne 0 ] || ( cd "$B" && ninja -t recompact > /dev/null 2>&1 9>&- ) || true
    RC=0; ( cd "$B" && $LOW ninja -k 0 -j"$JOBS" $TARGETS > "$LOG" 2>&1 9>&- ) || RC=$?
  fi
  tick "build rc=$RC"
  echo "$(grep -cE '^\[[0-9]+/[0-9]+\] (Building|Linking)' "$LOG" || true) steps run"
fi
RES=0
if [ $RC -ne 0 ]; then
  RES=1
  echo "--- errors (full log $LOG)"
  { grep -E "(error|undefined reference|multiple definition)" "$LOG" | grep -vE "^FAILED|^ninja:" \
    | sed -E "s#$W/##g; s#^\.\./(\.\./)*##" | awk '!s[$0]++' | cut -c1-230 | head -40; } || true
  echo "--- failed units"; { grep -E "^FAILED:" "$LOG" | sed -E 's/^FAILED: (\[[^]]*\] )?//' | cut -c1-150 | head -20; } || true
fi

# 5. tests: only those whose executables this run relinked (or the whole build succeeded), through ctest
#    with the GPU hidden so a running ctest is undisturbed
EXPECT=()
while IFS=$'\t' read -r kind name exes; do
  [ "$kind" = TEST ] || continue
  if [ -n "${QC_NO_TESTS:-}" ]; then continue; fi
  good=1
  IFS=, read -ra es <<< "$exes"
  for e in "${es[@]}"; do
    [ -n "$e" ] || continue
    MT=$(stat -c %Y "$B/$e" 2>/dev/null || echo 0)
    if [ $RC -ne 0 ] && [ "$MT" -lt "$BSTART" ]; then good=0; PARTIAL+=("test $name not run: $(basename "$e") was not relinked by this run (build failed)"); break; fi
  done
  if [ $good -eq 1 ]; then EXPECT+=("$name"); fi
done < "$PLAN"
[ -z "${QC_NO_TESTS:-}" ] || PARTIAL+=("tests not run (QC_NO_TESTS)")
if [ ${#EXPECT[@]} -gt 0 ]; then
  RX=$(printf '%s\n' "${EXPECT[@]}" | sort -u | sed -E 's#[]\\[(){}.*+?^$|]#\\&#g' | paste -sd'|'); RX="^($RX)\$"
  echo "--- tests: ${#EXPECT[@]} (CUDA_VISIBLE_DEVICES empty, timeout ${QC_TEST_TIMEOUT:-300} s)"
  CLOG=$L/qc_ctest_$TAG.log; CRC=0
  ( cd "$B" && CUDA_VISIBLE_DEVICES= $LOW ctest -V -j1 -R "$RX" --timeout "${QC_TEST_TIMEOUT:-300}" > "$CLOG" 2>&1 9>&- ) || CRC=$?
  { grep -E "Test +#[0-9]+:" "$CLOG" | sed -E 's/^ *[0-9]+\/[0-9]+ //' | cut -c1-120; } || true
  { grep -E "tests passed|tests failed" "$CLOG"; } || true
  # classes: SKIP (exit 77), GPU (ordinary nonzero exit, no-device message, no assertion), PSKIP (passed but
  # printed a skip / no-device marker), FAIL (anything else, crashes included)
  CLASS=$(awk -v mark='skip|no cuda|cuda device|cudaerrornodevice|(need|require)s? (a )?gpu|no gpu' '
    /^[0-9]+: / { n = $1; sub(/:$/, "", n); l = tolower($0)
      if (l ~ mark) flag[n] = 1
      if (l ~ /cudaerrornodevice|no cuda-capable device/) nodev[n] = 1
      if (l ~ /fail:|assertion|assert|expect_|check failed/) asrt[n] = 1
      next }
    /Could not find executable|Unable to find executable/ { nf = 1 }   # ctest: the test cannot start, no binary
    /Test +#[0-9]+: / { line = $0; match(line, /#[0-9]+/); n = substr(line, RSTART + 1, RLENGTH - 1)
      name = line; sub(/^.*Test +#[0-9]+: /, "", name); sub(/ \.\.\..*/, "", name)
      if (line ~ /\*\*\*Skipped/) print "SKIP " name
      else if (line ~ /\*\*\*Failed/) print ((nodev[n] && !asrt[n]) ? "GPU " : "FAIL ") name
      else if (line ~ /\*\*\*Not Run/ && line !~ /Disabled/ && nf) print "UNBUILT " name   # its executable was never built here
      else if (line ~ /\*\*\*(Exception|Timeout|Not Run)|Subprocess aborted/) print ((line ~ /Disabled/) ? "SKIP " : "FAIL ") name
      else if (line ~ /Passed/) { if (flag[n]) print "PSKIP " name }
      else print "FAIL " name
      nf = 0 }' "$CLOG")
  NFAIL=$(grep -c '^FAIL ' <<< "$CLASS" || true)
  GPUN=$(sed -n 's/^GPU //p' <<< "$CLASS" | tr '\n' ' '); SKIPN=$(sed -n 's/^SKIP //p' <<< "$CLASS" | tr '\n' ' ')
  PSKIPN=$(sed -n 's/^PSKIP //p' <<< "$CLASS" | tr '\n' ' ')
  UNBUILTN=$(sed -n 's/^UNBUILT //p' <<< "$CLASS" | tr '\n' ' ')
  [ -z "$UNBUILTN" ] || PARTIAL+=("tests whose executable this run did not build (a test qc did not map, or one behind the object cap): $UNBUILTN")
  [ -z "$GPUN" ] || PARTIAL+=("tests that stopped for lack of a GPU, not run: $GPUN")
  [ -z "$SKIPN" ] || PARTIAL+=("tests that skipped themselves (77): $SKIPN")
  [ -z "$PSKIPN" ] || PARTIAL+=("tests that passed but printed a skip / no-device marker (check the log $CLOG): $PSKIPN")
  SEEN=$(grep -E "Test +#[0-9]+: " "$CLOG" | sed -E 's/^.*Test +#[0-9]+: //; s/ \.\.\..*$//' | sort -u || true)
  while IFS= read -r n; do grep -qFx -- "$n" <<< "$SEEN" || PARTIAL+=("ctest did not run $n"); done < <(printf '%s\n' "${EXPECT[@]}" | sort -u)
  if [ "$NFAIL" -gt 0 ]; then
    [ $RES -ne 0 ] || RES=2
    sed -n 's/^FAIL /  FAILED TEST: /p' <<< "$CLASS"
    echo "  failing output (log $CLOG):"
    { grep -E "^[0-9]+: " "$CLOG" | grep -iE "error|fail|assert|expect|what\(\)|abort" | sed -E "s#$W/##g" | cut -c1-200 | head -12; } || true
  elif [ "$CRC" -ne 0 ] && [ -z "$GPUN" ] && [ -z "$SKIPN" ] && [ -z "$UNBUILTN" ]; then
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
