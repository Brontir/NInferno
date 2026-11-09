#!/usr/bin/env python3
"""Mechanical review lint for a PR diff (issue #611, step 3). Needs no build and no GPU.

A text heuristic, not a compiler: it works on bare names and never resolves overloads, namespaces or
macros. Where it is not precise it warns; it fails only where a finding is near certain.

  removed-def      (FAILS, exit 1) a C/C++ function or method definition removed from .h/.cpp/.cu/.cuh
                   whose bare name is still called as `name(` in the tree at the head. Not reported when
                   any declaration or definition of that name remains (overloads, moves, a return type on
                   the line above), when the only uses are member initializers or comments, or when the
                   name is passed to an UPPER_CASE macro (it may be generated). Operators, constructors,
                   destructors, names under 3 characters and `main` are skipped.
  removed-include  (WARNS) a removed quoted #include whose header defines, at column 0, a type the file
                   still uses, when no remaining include reaches the header. Forward declarations count
                   as a declaration; a file with an include that cannot be resolved is skipped.
  dup-block        (WARNS) a run of >= 4 consecutive added non-blank lines that repeats in the same file
                   (other added lines or old content); only copies of 10+ lines, or a copy that is the
                   whole added run, are reported, on the later copy.

A warning is printed as a GitHub `::warning` annotation and does not change the exit code.

Escapes: `// diff-lint: allow` on the reported line (for the removal rules also on the removed line); or,
for a deliberate removal, a line `diff-lint: allow removed-def NAME` (or `removed-include HEADER`) in a
commit message of the PR.

Usage: diff_lint.py [--base REF] [--head REV] [--exact]   (default: merge base with origin/master, HEAD)
Exit 1 on a removed-def finding.
"""
import argparse
import os
import re
import subprocess
import sys

ALLOW = "diff-lint: allow"
CODE_EXT = {".h", ".hpp", ".cpp", ".cc", ".c", ".cu", ".cuh", ".py", ".sh", ".cmake"}
CXX_EXT = {".h", ".hpp", ".cpp", ".cc", ".c", ".cu", ".cuh"}
DEF_EXT = {".h", ".cpp", ".cu", ".cuh"}
SKIP_DIRS = ("third_party/", "routes/", "build/")
MIN_RUN = 4
LONG_COPY = 10
WARN_RULES = {"dup-block", "removed-include"}  # not precise enough to fail a check: they warn, exit 0
KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "else", "new", "delete", "throw", "case",
            "goto", "sizeof", "co_return", "co_await", "co_yield", "using", "typedef", "operator", "do",
            "defined", "alignof", "decltype", "static_assert", "namespace", "template", "typename"}


def ext(path):
    if os.path.basename(path) == "CMakeLists.txt":
        return ".cmake"
    return os.path.splitext(path)[1]


def lintable(path, exts):
    return ext(path) in exts and not path.startswith(SKIP_DIRS) and "/third_party/" not in path


# ---------------------------------------------------------------- trees

class GitTree:
    """The repository at one revision."""

    def __init__(self, rev, cwd=None):
        self.rev, self.cwd, self._cache, self._names = rev, cwd, {}, None

    def _git(self, *args):
        r = subprocess.run(["git", *args], cwd=self.cwd, capture_output=True, check=False)
        return r.returncode, r.stdout.decode("utf-8", "replace")

    def names(self):
        if self._names is None:
            self._names = self._git("ls-tree", "-r", "--name-only", self.rev)[1].splitlines()
        return self._names

    def read(self, path):
        if path not in self._cache:
            rc, out = self._git("show", f"{self.rev}:{path}")
            self._cache[path] = out if rc == 0 else None
        return self._cache[path]

    def grep(self, pattern, exts):
        """(path, line number, line) for every ERE match in files of the given extensions."""
        _, out = self._git("grep", "-n", "-I", "-E", "-e", pattern, self.rev, "--")
        hits = []
        for row in out.splitlines():
            parts = row.split(":", 3)  # rev:path:line:text
            if len(parts) == 4 and lintable(parts[1], exts):
                hits.append((parts[1], int(parts[2]), parts[3]))
        return hits


class DictTree:
    """In-memory tree for the unit tests: {path: text}."""

    def __init__(self, files):
        self.files = files

    def names(self):
        return list(self.files)

    def read(self, path):
        return self.files.get(path)

    def grep(self, pattern, exts):
        rx = re.compile(pattern)
        return [(p, i, line) for p, t in self.files.items() if lintable(p, exts)
                for i, line in enumerate(t.splitlines(), 1) if rx.search(line)]


# ---------------------------------------------------------------- diff parsing

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_diff(text):
    """{new path: {"added": [(line, text)], "removed": [[(old line, text)] per hunk]}}."""
    files, cur = {}, None
    old = new = 0
    for row in text.splitlines():
        if row.startswith("diff "):
            cur = None
        elif row.startswith("+++ "):
            if row.endswith("/dev/null"):
                cur = None
            else:
                path = row[6:] if row.startswith("+++ b/") else row[4:]
                cur = files.setdefault(path, {"added": [], "removed": []})
        elif row.startswith("--- "):
            continue
        elif HUNK.match(row):
            m = HUNK.match(row)
            old, new = int(m.group(1)), int(m.group(3))
            if cur is not None:
                cur["removed"].append([])
        elif cur is not None and row.startswith("+"):
            cur["added"].append((new, row[1:]))
            new += 1
        elif cur is not None and row.startswith("-"):
            cur["removed"][-1].append((old, row[1:]))
            old += 1
    return files


def parse_deleted(text):
    """Removed lines of deleted files: {old path: [(line, text)]}."""
    out, path, pending, old = {}, None, None, 0
    for row in text.splitlines():
        if row.startswith("diff "):
            path = pending = None
        elif row.startswith("--- "):
            pending = None if row.endswith("/dev/null") else row[6:]
        elif row.startswith("+++ "):
            path = pending if row.endswith("/dev/null") else None
        elif HUNK.match(row):
            old = int(HUNK.match(row).group(1))
        elif path and row.startswith("-"):
            out.setdefault(path, []).append((old, row[1:]))
            old += 1
    return out


# ---------------------------------------------------------------- rule: duplicated added block

def is_comment(line):
    s = line.strip()
    if s.startswith(("#include", "#define", "#if", "#pragma", "#endif", "#else")):
        return False
    return s.startswith(("//", "/*", "*", "#"))


def informative(window):
    code = [w for w in window if not is_comment(w)]
    # Closing braces and `return x;` tails do not count: three lines must carry real content.
    return sum(len(re.sub(r"[^A-Za-z0-9_]", "", w)) >= 10 for w in code) >= 3


def dup_blocks(path, added, content):
    """Findings for an added run that repeats inside the file."""
    if content is None:
        return []
    lines = content.splitlines()
    norm = [line.strip() for line in lines]
    added_at = {n for n, _ in added}
    windows = {}
    for i in range(len(norm) - MIN_RUN + 1):
        w = tuple(norm[i:i + MIN_RUN])
        if all(w):
            windows.setdefault(w, []).append(i)
    flagged = []
    for i in range(len(norm) - MIN_RUN + 1):
        if not all((i + k + 1) in added_at for k in range(MIN_RUN)):
            continue
        w = tuple(norm[i:i + MIN_RUN])
        if not all(w) or not informative(w):
            continue
        others = [j for j in windows.get(w, []) if abs(j - i) >= MIN_RUN]
        # Two added copies: report the later one only. An old copy: report the added one.
        if not others or (all((j + 1) in added_at for j in others) and min(others) > i):
            continue
        if any(ALLOW in lines[i + k] for k in range(MIN_RUN)):
            continue
        flagged.append((i + 1, min(others) + 1))
    # Merge overlapping windows into regions (1-based first line, last line, other copy's first line).
    regions = []
    for start, other in flagged:
        if regions and start <= regions[-1][1]:
            regions[-1][1] = start + MIN_RUN - 1
        else:
            regions.append([start, start + MIN_RUN - 1, other])
    out = []
    for start, end, other in regions:
        # Test setup and error handling repeat in short stretches all the time. Report a copy only when
        # it is long, or when it is the whole added run (a block pasted as it stands).
        lo, hi = start, end
        while lo - 1 in added_at and norm[lo - 2]:
            lo -= 1
        while hi + 1 in added_at and norm[hi]:
            hi += 1
        if end - start + 1 < LONG_COPY and (lo, hi) != (start, end):
            continue
        out.append((path, start, "dup-block",
                    f"{end - start + 1} added lines repeat the block at line {other} of this file "
                    f"(pasted twice?)"))
    return out


# ---------------------------------------------------------------- rule: removed definition still called

DEF_RX = re.compile(
    r"(?:^|\n)[ \t]*(?:template\s*<[^\n]*>\s*)?"
    r"((?:[\w:<>,*&~\[\]]+[ \t\n]+|[\w>*&\]][*&]+[ \t]*)+)"       # return type words, at least one
    r"((?:[A-Za-z_]\w*::)*~?[A-Za-z_]\w*)[ \t]*\(([^;{}()]*(?:\([^;{}()]*\)[^;{}()]*)*)\)"
    r"[ \t\n]*(?:const|noexcept|override|final|&&|&|\s)*(?::[^{;]*)?\{")


def removed_definitions(removed_lines):
    """[(unqualified name, old line)] for definitions in the removed lines of one hunk."""
    if not removed_lines:
        return []
    text = "\n".join(t for _, t in removed_lines)
    texts = text.split("\n")
    starts = [0]
    for t in texts:
        starts.append(starts[-1] + len(t) + 1)
    out = []
    for m in DEF_RX.finditer(text):
        words = re.findall(r"[\w:<>~]+", m.group(1))
        if not words or any(w in KEYWORDS or not re.search(r"\w", w) for w in words):
            continue
        name = m.group(2).split("::")[-1]
        if name in KEYWORDS or len(name) < 3 or name == "main" or name.isupper() or name.startswith("~"):
            continue
        idx = max(i for i, s in enumerate(starts[:-1]) if s <= m.start(2))
        if ALLOW in texts[idx]:
            continue
        out.append((name, removed_lines[idx][0]))
    return out


DECL_TYPE_END = re.compile(r"[\w>*&\]]\s*$")


def classify(tree, name, hit):
    """'comment', 'allowed', 'decl' (a declaration or definition), 'init' (member init) or 'call'."""
    path, n, line = hit
    stripped = line.strip()
    if stripped.startswith(("//", "*", "/*")):
        return "comment"
    if ALLOW in line:
        return "allowed"
    nm = re.escape(name)
    m = re.search(rf"\b{nm}[ \t]*\(", line)
    if not m:  # a mention without a call: `name` as a macro argument
        return "macro" if re.search(rf"[(,][ \t]*{nm}\b", line) and re.search(r"\b[A-Z][A-Z0-9_]{2,}[ \t]*\(", line) \
            else "other"
    before = line[:m.start()]
    # a type word before the name on this line: `double Class::name(` / `static int name(`
    tm = re.search(r"(?:^|[^\w.>:])(\w[\w:<>*&\]]*)[ \t*&]+(?:\w+::)*$", before)
    if tm and tm.group(1) not in KEYWORDS:
        return "decl"
    # the return type on the line above: `double\nname(int k) const;`
    if re.fullmatch(r"[ \t]*(?:\w+::)*", before) and n > 1:
        prev = (tree.read(path) or "").splitlines()
        prev = prev[n - 2] if n - 2 < len(prev) else ""
        if prev.strip() and DECL_TYPE_END.search(prev) and not prev.strip().startswith(("#", "//", "return", "else")) \
                and prev.split()[-1] not in KEYWORDS:
            return "decl"
    # a member initializer `: name(x)` or `, name(x)` at paren depth 0 of an unfinished statement
    if not line.rstrip().endswith(";") and re.search(r"(?:^|[:,])[ \t]*$|\)[ \t]*(?:noexcept[ \t]*)?:[ \t]*$", before):
        if before.count("(") <= before.count(")"):
            return "init"
    return "call"


def removed_calls(tree, removed_by_path, allow=frozenset()):
    """Findings for removed definitions whose bare name is still called."""
    out, seen, mentions = [], set(), {}
    for path, hunks in removed_by_path:
        if not lintable(path, DEF_EXT):
            continue
        for hunk in hunks:
            for name, old_line in removed_definitions(hunk):
                if (path, name) in seen or ("removed-def", name) in allow:
                    continue
                seen.add((path, name))
                if name not in mentions:  # one grep per name serves the call sites and the "still defined" test
                    mentions[name] = [(h, classify(tree, name, h))
                                      for h in tree.grep(rf"\b{re.escape(name)}\b", CXX_EXT)]
                kinds = {k for _, k in mentions[name]}
                uses = [h[:2] for h, k in mentions[name] if k == "call"]
                if not uses or "decl" in kinds or "macro" in kinds:
                    continue
                sites = ", ".join(f"{p}:{n}" for p, n in uses[:3]) + (" ..." if len(uses) > 3 else "")
                p0, n0 = uses[0]
                out.append((p0, n0, "removed-def",
                            f"`{name}(` was defined at {path}:{old_line}, the definition is removed, "
                            f"and it is still called at {len(uses)} site(s): {sites}"))
    return out


# ---------------------------------------------------------------- rule: removed include still needed

INC_RX = re.compile(r'^\s*#\s*include\s+"([^"]+)"')
BODY = r"\s*(?:final\s*)?(?::[^;{]*)?\{"
# A type the file or a reachable header declares anywhere (a forward declaration counts).
ANY_TYPE_RX = re.compile(r"\b(?:struct|class|enum(?:\s+class)?|union)\s+(\w+)(?:" + BODY + r"|\s*;)"
                         r"|\busing\s+(\w+)\s*=")
# A type the removed header owns: defined at column 0 only, so nested member aliases and member types
# (`using value_type = ...` inside a struct) are not mistaken for names a user of the header needs.
OWN_TYPE_RX = re.compile(r"^(?:struct|class|enum(?:\s+class)?|union)\s+(\w+)" + BODY + r"|^using\s+(\w+)\s*=",
                         re.M)


def strip_noise(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    return re.sub(r'"(?:\\.|[^"\\\n])*"', '""', text)


def types_of(text, rx=ANY_TYPE_RX):
    return {g for m in rx.finditer(strip_noise(text)) for g in m.groups() if g}


def resolve(names, including, inc):
    cand = os.path.normpath(os.path.join(os.path.dirname(including), inc)).replace("\\", "/")
    if cand in names:
        return cand
    if inc in names:
        return inc
    hits = [n for n in names if n.endswith("/" + inc)]
    return hits[0] if len(hits) == 1 else None


def closure(tree, names, path, memo):
    """(files reachable from path through quoted includes, whether every include was resolved)."""
    seen, stack, complete = set(), [path], True
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        if p not in memo:
            incs = [INC_RX.match(line) for line in (tree.read(p) or "").splitlines()]
            found = [(m.group(1), resolve(names, p, m.group(1))) for m in incs if m]
            memo[p] = ([r for _, r in found if r], all(r for _, r in found))
        stack.extend(memo[p][0])
        complete = complete and memo[p][1]
    return seen, complete


def removed_includes(tree, files, allow=frozenset()):
    """Findings for a removed quoted include that the file still needs (a type, nothing else)."""
    out, names, memo, tcache = [], set(tree.names()), {}, {}

    def types(path):  # each file's types are collected once
        if path not in tcache:
            tcache[path] = types_of(tree.read(path) or "")
        return tcache[path]

    for path, d in files.items():
        if not lintable(path, CXX_EXT):
            continue
        text = tree.read(path)
        gone = [(n, m.group(1)) for hunk in d["removed"] for n, t in hunk
                for m in [INC_RX.match(t)] if m and ALLOW not in t and ("removed-include", m.group(1)) not in allow]
        if text is None or not gone:
            continue
        reach, complete = closure(tree, names, path, memo)
        if not complete:  # an include we cannot follow may bring the type: no precise answer
            continue
        own = types(path)
        body_lines = strip_noise(text).splitlines()
        raw_lines = text.splitlines()
        for old_line, inc in gone:
            header = resolve(names, path, inc)
            htext = tree.read(header) if header else None
            if htext is None or header in reach:
                continue
            syms = {s for s in types_of(htext, OWN_TYPE_RX) if len(s) >= 4} - own
            for other in reach - {path}:
                syms -= types(other)
            for sym in sorted(syms):
                rx = re.compile(rf"\b{re.escape(sym)}\b")
                hit = next((n for n, line in enumerate(body_lines, 1)
                            if rx.search(line) and ALLOW not in raw_lines[n - 1]), None)
                if hit:
                    out.append((path, hit, "removed-include",
                                f'#include "{inc}" (was line {old_line}) is removed, but `{sym}` declared '
                                f"in {header} is still used and no remaining include reaches it"))
                    break
    return out


# ---------------------------------------------------------------- driver

def lint(diff_text, tree, allow=frozenset()):
    files = parse_diff(diff_text)
    findings = []
    for path, d in files.items():
        if lintable(path, CODE_EXT) and d["added"]:
            findings += dup_blocks(path, d["added"], tree.read(path))
    removed = [(p, d["removed"]) for p, d in files.items()]
    removed += [(p, [rows]) for p, rows in parse_deleted(diff_text).items()]
    findings += removed_calls(tree, removed, allow)
    findings += removed_includes(tree, files, allow)
    return sorted(set(findings))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="origin/master")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--exact", action="store_true", help="diff BASE..HEAD instead of from the merge base")
    a = ap.parse_args(argv)

    def git(*x):
        return subprocess.run(["git", *x], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", check=True).stdout

    base = a.base if a.exact else git("merge-base", a.base, a.head).strip()
    allow = set(re.findall(r"diff-lint: allow (removed-def|removed-include) (\S+)",
                           git("log", "--format=%B", f"{base}..{a.head}")))
    found = lint(git("diff", "-U0", "--no-color", "-M", base, a.head), GitTree(a.head), allow)
    errors = [f for f in found if f[2] not in WARN_RULES]
    for path, line, rule, why in found:
        if rule in WARN_RULES:  # a GitHub annotation; it does not fail the check
            print(f"::warning file={path},line={line},title=diff-lint {rule}::{why}")
        else:
            print(f"{path}:{line}: [{rule}] {why}")
    if errors:
        print(f"diff-lint: {len(errors)} error(s). Fix them, or mark an intended case with `// {ALLOW}`.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
