#!/usr/bin/env python3
"""Mechanical review lint for a PR diff (issue #611, step 3). Needs no build and no GPU.

A text heuristic, not a compiler: it works on bare names and never resolves overloads, namespaces or
macros.
Comments and string/char/raw-string literals are blanked (mask_cxx) before any regex looks for
definitions, calls, includes or types; only the repeated-block rule and the escape markers read raw text.
Where it is not precise it warns; it fails only where a finding is near certain.

  removed-def      (WARNS) a C/C++ function or method definition removed from .h/.cpp/.cu/.cuh
                   whose bare name is still called as `name(` in the tree at the head. On the history
                   of master about half of its findings were false (a same-named member, lambda or
                   class), so it warns and never fails a check. Not reported when
                   any declaration or definition of that name remains (overloads, moves, a return type on
                   the line above), when the only uses are member initializers or comments, or when the
                   name is passed to an UPPER_CASE macro (it may be generated). Operators, constructors,
                   destructors, names of a class or struct, names under 3 characters and `main` are
                   skipped; `std::name(` is not a call, and neither is `x.name(` of a removed free function.
  bad-literal      (FAILS, exit 1) an added line in a .c/.cc/.cpp/.cu/.h/.cuh/.hpp/.inl/.inc file that ends
                   inside an ordinary string or char literal (a real newline where `\n` was meant), or in a
                   .py file inside a single-quoted string. Comments, escapes, '"', C++ raw strings (tracked
                   across lines in the whole file), `#error`/`#warning` text, `#if 0` blocks and Python
                   triple quotes are understood. CMake, shell, JSON and YAML files are not checked.
  removed-include  (WARNS) a removed quoted #include whose header defines, at column 0, a type the file
                   still uses, when no remaining include reaches the header. Forward declarations count
                   as a declaration; a file with an include that cannot be resolved is skipped, except
                   for an unresolved `.inc` (generated tables), which is ignored.
  dup-block        (WARNS) a run of >= 4 consecutive added non-blank lines that repeats in the same file
                   (other added lines or old content); only copies of 10+ lines, or a copy that is the
                   whole added run, are reported, on the later copy.

A warning is printed as a GitHub `::warning` annotation and does not change the exit code.

Escapes: `// diff-lint: allow` on the reported line (for the removal rules also on the removed line); or,
for a deliberate removal, a line `diff-lint: allow removed-def NAME` (or `removed-include HEADER`) in a
commit message of the PR.

Usage: diff_lint.py [--base REF] [--head REV] [--exact]   (default: merge base with origin/master, HEAD)
       diff_lint.py --scan-tree [--head REV]   (bad-literal over every line of the tree, not a diff)
Exit 1 on a bad-literal finding.
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
LIT_CXX_EXT = {".c", ".cc", ".cpp", ".cu", ".h", ".cuh", ".hpp", ".inl", ".inc"}
SKIP_DIRS = ("third_party/", "routes/", "build/")
MIN_RUN = 4
LONG_COPY = 10
WARN_RULES = {"dup-block", "removed-include", "removed-def"}  # not precise enough to fail a check: they warn, exit 0
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
        # -z: every name verbatim and NUL-ended. Without it git C-quotes a name with a non-ASCII or
        # special character ("\303\274nit.cpp"), whose extension then reads `.cpp"`.
        if self._names is None:
            out = self._git("ls-tree", "-r", "-z", "--name-only", self.rev)[1]
            self._names = [n for n in out.split("\0") if n]
        return self._names

    def read(self, path):
        if path not in self._cache:
            rc, out = self._git("show", f"{self.rev}:{path}")
            self._cache[path] = out if rc == 0 else None
        return self._cache[path]

    def grep(self, pattern, exts):
        """(path, line number, line) for every ERE match in files of the given extensions."""
        # -z: `rev:path` NUL line NUL text, the path verbatim (not C-quoted, a `:` in it kept).
        _, out = self._git("grep", "-z", "-n", "-I", "-E", "-e", pattern, self.rev, "--")
        hits, head = [], self.rev + ":"
        for m in GREP_ROW.finditer(out):
            path = m.group(1)[len(head):] if m.group(1).startswith(head) else m.group(1)
            if lintable(path, exts):
                hits.append((path, int(m.group(2)), m.group(3).rstrip("\r")))
        return hits


# One row of `git grep -z -n`: the name, the line number, the line.
GREP_ROW = re.compile(r"([^\0]*)\0(\d+)\0([^\n]*)\n?")


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
C_ESCAPES = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12, "r": 13, '"': 34, "\\": 92}


def diff_path(field, prefix):
    """The path of a `--- a/x` / `+++ b/x` field: git C-quotes a name with a special or non-ASCII
    character ("b/\\303\\274nit.cpp") and ends a name holding a blank with a tab."""
    field = field.rstrip("\t")
    if field.startswith('"') and field.endswith('"') and len(field) >= 2:
        raw, out, i = field[1:-1], bytearray(), 0
        while i < len(raw):
            c = raw[i]
            if c == "\\" and i + 1 < len(raw):
                octal = re.match(r"[0-7]{1,3}", raw[i + 1:])
                if octal:  # git writes three digits; fewer are read as far as they go
                    out.append(int(octal.group(0), 8) & 0xFF)
                    i += 1 + len(octal.group(0))
                    continue
                out.append(C_ESCAPES.get(raw[i + 1], ord(raw[i + 1])))
                i += 2
                continue
            out += c.encode("utf-8")
            i += 1
        field = out.decode("utf-8", errors="replace")
    return field[len(prefix):] if field.startswith(prefix) else field


def parse_diff(text):
    """{new path: {"added": [(line, text)], "removed": [[(old line, text)] per hunk], "old": old path}}.

    The hunk header's line counts say how many rows still belong to the hunk, so an added line that
    reads `++ x` or a removed one that reads `-- x` is not taken for a file header."""
    files, cur, old_path = {}, None, None
    old = new = left_old = left_new = 0
    for row in text.splitlines():
        if left_old > 0 or left_new > 0:
            if row.startswith("+") and left_new > 0:
                if cur is not None:
                    cur["added"].append((new, row[1:]))
                new, left_new = new + 1, left_new - 1
                continue
            if row.startswith("-") and left_old > 0:
                if cur is not None:
                    cur["removed"][-1].append((old, row[1:]))
                old, left_old = old + 1, left_old - 1
                continue
            if row.startswith(("\\", " ")):  # "\ No newline at end of file", or a context row
                if row.startswith(" "):
                    old, new, left_old, left_new = old + 1, new + 1, left_old - 1, left_new - 1
                continue
        if row.startswith("diff "):
            cur = old_path = None
        elif row.startswith("+++ "):
            if row[4:] == "/dev/null":  # the whole field: a file may be named dev/null
                cur = None
            else:
                path = diff_path(row[4:], "b/")
                cur = files.setdefault(path, {"added": [], "removed": [], "old": old_path or path})
        elif row.startswith("--- "):
            old_path = None if row[4:] == "/dev/null" else diff_path(row[4:], "a/")
        elif HUNK.match(row):
            m = HUNK.match(row)
            old, new = int(m.group(1)), int(m.group(3))
            left_old = 1 if m.group(2) is None else int(m.group(2))
            left_new = 1 if m.group(4) is None else int(m.group(4))
            if cur is not None:
                cur["removed"].append([])
    return files


def parse_deleted(text):
    """Removed lines of deleted files: {old path: [(line, text)]}."""
    out, path, pending, old, left = {}, None, None, 0, 0
    for row in text.splitlines():
        if left > 0 and row.startswith("-"):
            if path:
                out.setdefault(path, []).append((old, row[1:]))
            old, left = old + 1, left - 1
        elif row.startswith("diff "):
            path = pending = None
        elif row.startswith("--- "):
            pending = None if row[4:] == "/dev/null" else diff_path(row[4:], "a/")
        elif row.startswith("+++ "):
            path = pending if row[4:] == "/dev/null" else None
        elif HUNK.match(row):
            m = HUNK.match(row)
            old, left = int(m.group(1)), 1 if m.group(2) is None else int(m.group(2))
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


def old_file_lines(base, path):
    """(raw lines, masked lines) of `path` at the base, or None when there is no base tree or file.

    A hunk of `-U0` shows only the removed lines. Masked on their own, the tail of a block comment
    or the middle of a raw string looks like code; masked inside the whole old file it does not."""
    if base is None:
        return None
    cache = base.__dict__.setdefault("_old", {})
    if path not in cache:
        text = base.read(path)
        cache[path] = None if text is None else (text.split("\n"), mask_cxx(text).split("\n"))
    return cache[path]


def mask_removed(removed_lines, old):
    """The removed lines of one hunk with comments and literals blanked: in the context of the whole
    old file `old` when it is known and agrees with the hunk, else on their own."""
    if old is not None:
        raw, masked = old
        if all(0 < n <= len(raw) and raw[n - 1].rstrip("\r") == t.rstrip("\r") for n, t in removed_lines):
            return [masked[n - 1].rstrip("\r").ljust(len(t)) for n, t in removed_lines]
    return mask_cxx("\n".join(t for _, t in removed_lines)).split("\n")


def removed_definitions(removed_lines, old=None):
    """[(unqualified name, old line, is a member)] for definitions in the removed lines of one hunk."""
    if not removed_lines:
        return []
    texts = [t for _, t in removed_lines]
    text = "\n".join(mask_removed(removed_lines, old))  # a commented-out or quoted signature is not a definition
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
        # a member: qualified (`Class::name`) or indented (defined inside a class or a nested scope)
        head = max(i for i, s in enumerate(starts[:-1]) if s <= m.start(1))
        out.append((name, removed_lines[idx][0], "::" in m.group(2) or texts[head][:1] in (" ", "\t")))
    return out


DECL_TYPE_END = re.compile(r"[\w>*&\]]\s*$")


def masked_lines(tree, path):
    """The lines of `path` at the head with comments and literals blanked (computed once per file)."""
    cache = tree.__dict__.setdefault("_masked", {})
    if path not in cache:
        cache[path] = mask_cxx(tree.read(path) or "").split(chr(10))
    return cache[path]


def classify(tree, name, hit, member=False):
    """'comment', 'allowed', 'decl' (a declaration or definition), 'init' (member init) or 'call'.

    `member`: the removed definition is a member function, so `x.name(` and `x->name(` still call it;
    for a free function they call something else."""
    path, n, raw = hit
    masked = masked_lines(tree, path)
    line = masked[n - 1] if n - 1 < len(masked) else mask_cxx(raw)  # comments and literals blanked
    if not line.strip() or raw.strip().startswith(("//", "*", "/*")):
        return "comment"
    if ALLOW in raw:
        return "allowed"
    nm = re.escape(name)
    # `x.name(`, `x->name(`, `std::name(` and `::name(` call something else than a free function `name`
    ms = [m for m in re.finditer(rf"\b{nm}[ \t]*\(", line)
          if not re.search(r"(?:\bstd::|(?<![\w>]):: ?)[ \t]*$", line[:m.start()])
          and (member or not re.search(r"(?:\.|->)[ \t]*$", line[:m.start()]))]
    if not ms:  # a mention without a call: `name` as a macro argument, or a variable named like it
        if re.search(rf"[(,][ \t]*{nm}\b", line) and re.search(r"\b[A-Z][A-Z0-9_]{2,}[ \t]*\(", line):
            return "macro"
        # a variable, lambda or parameter of that name: `auto name = [`, `Type name{`, `Type name,`
        vm = re.search(rf"(?:^|[^\w.>:])(\w[\w:<>*&\]]*|\w[\w:]*<[^;=]*>)[ \t*&]+{nm}[ \t]*(?:=(?!=)|\{{|;|,|\))", line)
        if vm and vm.group(1) not in KEYWORDS:
            return "decl"
        return "other"
    m = ms[0]
    before = line[:m.start()]
    # a type word before the name on this line: `double Class::name(` / `static int name(`
    tm = re.search(r"(?:^|[^\w.>:])(\w[\w:<>*&\]]*)[ \t*&]+(?:\w+::)*$", before)
    if tm and tm.group(1) not in KEYWORDS:
        return "decl"
    # the return type on the line above: `double\nname(int k) const;`
    if re.fullmatch(r"[ \t]*(?:\w+::)*", before) and n > 1:
        prev = masked[n - 2] if n - 2 < len(masked) else ""
        if prev.strip() and DECL_TYPE_END.search(prev) and not prev.strip().startswith(("#", "//", "return", "else")) \
                and prev.split()[-1] not in KEYWORDS:
            return "decl"
    # a member initializer `: name(x)` or `, name(x)` at paren depth 0 of an unfinished statement; at the
    # start of a line only when the line above is itself part of an initializer list (ends in `:` or `,`),
    # not a statement that begins a multi-line call
    if not line.rstrip().endswith(";") and re.search(r"(?:(?<!:):|,)[ \t]*$|\)[ \t]*(?:noexcept[ \t]*)?:[ \t]*$", before):
        if before.count("(") <= before.count(")"):
            return "init"
    if not line.rstrip().endswith(";") and not before.strip() and n > 1:
        prev = next((p.strip() for p in reversed(masked[max(0, n - 4):n - 1]) if p.strip()), "")
        if prev.endswith((",", ":")) and not prev.startswith("#"):
            return "init"
    return "call"


def removed_calls(tree, removed_by_path, allow=frozenset(), base=None):
    """Findings for removed definitions whose bare name is still called."""
    out, seen, mentions = [], set(), {}
    for path, hunks, old_path in removed_by_path:
        if not lintable(path, DEF_EXT) or not any(hunks):
            continue
        old = old_file_lines(base, old_path)
        for hunk in hunks:
            for name, old_line, member in removed_definitions(hunk, old):
                if (path, name) in seen or ("removed-def", name) in allow:
                    continue
                seen.add((path, name))
                if (name, member) not in mentions:  # one grep serves the call sites and the "still defined" test
                    mentions[name, member] = [(h, classify(tree, name, h, member))
                                              for h in tree.grep(rf"\b{re.escape(name)}\b", CXX_EXT)]
                kinds = {k for _, k in mentions[name, member]}
                uses = [h[:2] for h, k in mentions[name, member] if k == "call"]
                if not uses or "decl" in kinds or "macro" in kinds:
                    continue
                if tree.grep(rf"\b(class|struct) +{re.escape(name)}\b", CXX_EXT):  # git grep -E: no `(?:`, no `\t`
                    continue  # the name of a class: what was removed is a constructor, `name(` builds an object
                sites = ", ".join(f"{p}:{n}" for p, n in uses[:3]) + (" ..." if len(uses) > 3 else "")
                p0, n0 = uses[0]
                out.append((p0, n0, "removed-def",
                            f"`{name}(` was defined at {path}:{old_line}, the definition is removed, "
                            f"and it is still called at {len(uses)} site(s): {sites}"))
    return out


# ---------------------------------------------------------------- rule: removed include still needed

INC_RX = re.compile(r'^\s*#\s*include\s+"([^"]+)"')
ANY_INC_RX = re.compile(r'[ \t]*#[ \t]*include[ \t]*[<"]([^>"]+)[>"]')  # quoted or angle
BODY = r"\s*(?:final\s*)?(?::[^;{]*)?\{"
# A type the file or a reachable header declares anywhere (a forward declaration counts).
ANY_TYPE_RX = re.compile(r"\b(?:struct|class|enum(?:\s+class)?|union)\s+(\w+)(?:" + BODY + r"|\s*;)"
                         r"|\busing\s+(\w+)\s*=")
# A type the removed header owns: defined at column 0 only, so nested member aliases and member types
# (`using value_type = ...` inside a struct) are not mistaken for names a user of the header needs.
OWN_TYPE_RX = re.compile(r"^(?:struct|class|enum(?:\s+class)?|union)\s+(\w+)" + BODY + r"|^using\s+(\w+)\s*=",
                         re.M)


RAW_PREFIX = re.compile(r"(?:u8|u|U|L)?R$")
INCLUDE_HEAD = re.compile(r"[ \t]*#[ \t]*include[ \t]*$")
DIAG_HEAD = re.compile(r"[ \t]*#[ \t]*(?:error|warning)\b")  # the text of #error / #warning is prose, not tokens


def _mask_cxx_physical(text):
    """`text` with comments and the insides of string, char and raw-string literals blanked to spaces.

    Same length and same newlines, so offsets and line numbers still match the original. The quotes stay.
    A `//` comment that ends with a backslash goes on in the next line (line splicing), a literal that is
    still open at a newline ends there, and the path in `#include "x.h"` is kept as it stands."""
    out, i, n = list(text), 0, len(text)

    def blank(a, b):
        for k in range(a, min(b, n)):
            if out[k] not in "\r\n":
                out[k] = " "

    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = i
            while j < n:
                e = text.find("\n", j)
                e = n if e < 0 else e
                if e < n and text[j:e].rstrip("\r").endswith("\\"):
                    j = e + 1
                else:
                    j = e
                    break
            blank(i, j)
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            blank(i, j)
            i = j
        elif c == '"' or c == "'":
            ls = text.rfind("\n", 0, i) + 1
            head = text[ls:i]
            if DIAG_HEAD.match(head):
                e = text.find("\n", i)
                i = n if e < 0 else e
                continue
            if c == '"' and INCLUDE_HEAD.match("".join(out[ls:i])):  # a comment before it does not hide the path
                j = text.find('"', i + 1)
                i = n if j < 0 else j + 1
                continue
            if c == "'" and re.search(r"(?<![\w.])[0-9][0-9A-Za-z_.]*$", head):  # a digit separator: 1'000'000
                i += 1
                continue
            m = RAW_PREFIX.search(head) if c == '"' else None
            d = re.match(r"[^\s()\\]{0,16}\(", text[i + 1:i + 19]) if m else None
            if d and not (m.start() > 0 and (head[m.start() - 1].isalnum() or head[m.start() - 1] == "_")):
                close = ")" + d.group(0)[:-1] + '"'
                j = text.find(close, i + d.end())
                j = n if j < 0 else j
                blank(i + 1, j)
                i = min(n, j + len(close))
                continue
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            blank(i + 1, j)
            i = j + 1 if j < n and text[j] == c else j
        else:
            i += 1
    return "".join(out)


SPLICE_RX = re.compile(r"\\\r?\n")


def mask_cxx(text):
    """`_mask_cxx_physical` over the text as the preprocessor reads it: each backslash-newline pair is
    spliced out first, so a comment opener or closer split by one (`/`, backslash, newline, `*`) is still
    a comment. The result keeps the length and every newline of `text`; a spliced-out backslash is
    blanked when the characters on both sides of it are blanked."""
    if not SPLICE_RX.search(text):
        return _mask_cxx_physical(text)
    kept, pos, i, n = [], [], 0, len(text)
    while i < n:
        m = SPLICE_RX.match(text, i)
        if m:
            i = m.end()
            continue
        kept.append(text[i])
        pos.append(i)
        i += 1
    masked = _mask_cxx_physical("".join(kept))
    out = list(text)
    gone = [masked[k] != ch for k, ch in enumerate(kept)]
    for k, g in enumerate(gone):
        if g:
            out[pos[k]] = " "
    for k in range(1, len(kept)):  # the splices between kept[k - 1] and kept[k]
        if gone[k - 1] and gone[k]:
            for j in range(pos[k - 1] + 1, pos[k]):
                if out[j] not in "\r\n":
                    out[j] = " "
    return "".join(out)


def types_of(text, rx=ANY_TYPE_RX):
    return {g for m in rx.finditer(mask_cxx(text)) for g in m.groups() if g}


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
            incs = [INC_RX.match(line) for line in mask_cxx(tree.read(p) or "").splitlines()]
            found = [(m.group(1), resolve(names, p, m.group(1))) for m in incs if m]
            # an unresolved `.inc` is a generated table: it is ignored, any other unresolved include is unknown
            memo[p] = ([r for _, r in found if r], all(r or inc.endswith(".inc") for inc, r in found))
        stack.extend(memo[p][0])
        complete = complete and memo[p][1]
    return seen, complete


def removed_includes(tree, files, allow=frozenset(), base=None):
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
        old = old_file_lines(base, d.get("old") or path)
        added_incs = {m.group(1) for _, t in d["added"] for m in [ANY_INC_RX.match(t)] if m}
        gone = [(n, m.group(1)) for hunk in d["removed"]
                for (n, t), c in zip(hunk, mask_removed(hunk, old))
                for m in [INC_RX.match(c)] if m and ALLOW not in t and not any(a == m.group(1) or m.group(1).endswith("/" + a) for a in added_incs)  # swapped for an angle include
                and ("removed-include", m.group(1)) not in allow]
        if text is None or not gone:
            continue
        reach, complete = closure(tree, names, path, memo)
        if not complete:  # an include we cannot follow may bring the type: no precise answer
            continue
        body_lines = mask_cxx(text).splitlines()
        # the file's own types and functions: a name it defines itself is not needed from the header
        own = types(path) | {m.group(2).split("::")[-1] for m in DEF_RX.finditer("\n".join(body_lines))}
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


# ---------------------------------------------------------------- rule: line ends inside a string literal

DIRECTIVE_RX = re.compile(r"[ \t]*#[ \t]*(\w+)(.*)")
# `__has_include(<name>)` / `__has_include_next(<name>)` in a condition: group 1 ends before `<`
HAS_INCLUDE_ANGLE = re.compile(r"(__has_include(?:_next)?\s*\(\s*)<[^>]*>")


COND_LITERAL = re.compile(
    r"(0[xX][0-9A-Fa-f']+|0[bB][01']+|[0-9][0-9']*)(?:[uU](?:ll|LL|[lL])?|(?:ll|LL|[lL])[uU]?)?")


def top_level_split(expr, op):
    """`expr` cut at each `op` (`&&` or `||`) outside parentheses; None where the parentheses do not
    balance."""
    parts, depth, start, i = [], 0, 0, 0
    while i < len(expr):
        c = expr[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                return None
        elif depth == 0 and expr.startswith(op, i):
            parts.append(expr[start:i])
            i += len(op)
            start = i
            continue
        i += 1
    return parts + [expr[start:]] if depth == 0 else None


def strip_parens(expr):
    """`expr` without the parentheses that enclose all of it (`(a) && (b)` keeps its own)."""
    expr = expr.strip()
    while expr.startswith("(") and expr.endswith(")"):
        depth = 0
        for i, c in enumerate(expr):
            depth += c == "("
            depth -= c == ")"
            if depth == 0 and i < len(expr) - 1:
                return expr  # the first parenthesis closes before the end
        expr = expr[1:-1].strip()
    return expr


def cond_literal(rest, cxx=True):
    """The value of a condition the preprocessor decides without a macro: one integer literal (any
    base, digit separators and suffixes allowed, inside any number of parentheses, a trailing
    comment ignored), or `&&` and `||` over such terms whose literal operands settle it the way
    short-circuit evaluation does (`X && 0` and `0 && X` are 0, `X || 1` and `1 || X` are 1, in any
    order and grouping; #712). Else None: an operand that is not a literal and does not settle it,
    or any other operator. `true` and `false` are literals in C++ only (C before C23 reads `true` as
    an undefined name, 0)."""
    rest = re.sub(r"/\*.*?\*/", " ", rest)  # a comment is whitespace to the preprocessor
    rest = re.sub(r"[ \t]*(?://.*|/\*.*)$", "", rest).strip()
    return cond_value(rest, cxx)


def has_char_constant(rest):
    """True when a quote in `rest` opens a character constant: one that does not continue a number
    (a digit separator: 1'000, 0x1'F), so also a prefixed one (u8'x', L'x')."""
    for m in re.finditer("'", rest):
        token = re.search(r"\w*$", rest[:m.start()]).group(0)
        if not token[:1].isdigit():
            return True
    return False


def cond_value(rest, cxx):
    """cond_literal's value of a condition with its comments removed."""
    rest = strip_parens(rest)
    # `?:` and the comma bind looser than `||`, and a character constant can hold `&&` or `||`
    # (`'||1'`): not decided here.
    if "?" in rest or "," in rest or has_char_constant(rest):
        return None
    for op, settles in (("||", 1), ("&&", 0)):
        parts = top_level_split(rest, op)
        if parts is None:
            return None
        if len(parts) > 1:
            values = [cond_value(part, cxx) for part in parts]
            if any(v is not None and int(v != 0) == settles for v in values):
                return settles
            return 1 - settles if all(v is not None for v in values) else None
    if cxx and rest in ("false", "true"):  # C++ keywords, valid in #if
        return int(rest == "true")
    m = COND_LITERAL.fullmatch(rest)
    if not m:
        return None
    digits = m.group(1).replace("'", "")
    if digits.startswith(("0x", "0X")):
        return int(digits[2:], 16) if len(digits) > 2 else None
    if digits.startswith(("0b", "0B")):
        return int(digits[2:], 2) if len(digits) > 2 else None
    if len(digits) > 1 and digits.startswith("0"):
        return int(digits, 8) if re.fullmatch(r"[0-7]+", digits) else None
    return int(digits)


def spliced_lines(text):
    """[(number of the last physical line, logical line)]: each line ending in a backslash is joined
    with the next one (the backslash dropped), as the preprocessor does before it reads anything."""
    out, buf = [], ""
    rows = text.split("\n")
    for num, row in enumerate(rows, 1):
        row = row.rstrip("\r")
        if row.endswith("\\") and num < len(rows):
            buf += row[:-1]
        else:
            out.append((num, buf + row))
            buf = ""
    return out


def cxx_open_lines(text, cxx=True):
    """1-based numbers of lines that end inside an ordinary string or char literal (C/C++).

    Comments (a `//` comment ending in a backslash goes on in the next line) and raw strings R"d(...)d"
    are tracked across lines; a backslash at the end of the line
    continues the literal and is not reported. After a report the scan restarts in code on the next line."""
    bad, block, raw, carry = [], False, None, None  # raw: the closing `)delim"` while inside a raw string
    if0 = 0  # nesting depth inside a `#if 0` group
    chains = []  # per open conditional group: whether a branch with a literal-true condition was taken
    # The open block comment began a line (only whitespace and closed comments before its `/*`).
    # The whole comment is one space to the preprocessor, so where it closes, the rest of its last
    # line is the start of that line: `/* note` + `*/ #if 0` is a directive.
    block_blank = False
    last_bad = -2  # index of the logical line of the last report
    # Backslash-newline pairs are joined first, as translation phase 2 does, so a comment opener, a
    # closer, a directive or a condition split by one is read whole; a finding is reported at the
    # last physical line of its logical line.
    for idx, (num, line) in enumerate(spliced_lines(text)):
        i, n, quote, carry = 0, len(line), carry, None
        if block and block_blank and quote is None and raw is None:
            j = line.find("*/")
            if j >= 0:  # blanked in place, so every column keeps its meaning
                line, block = " " * (j + 2) + line[j + 2:], False
        if quote is None and raw is None and not block:
            # A comment is whitespace to the preprocessor: `#/**/if 0` and `/* note */ #if 0` are
            # directives. The mask keeps every column, so a position found in it is one in the line.
            masked = re.sub(r"/\*.*?\*/", lambda m: " " * len(m.group(0)), line)
            dm = DIRECTIVE_RX.match(masked)
            kind, rest = (dm.group(1), dm.group(2).strip()) if dm else ("", "")
            value = cond_literal(rest, cxx)
            zero, one = value == 0, value is not None and value != 0
            if if0:  # the text of a `#if 0` group is not code: prose with an apostrophe is fine there
                if kind in ("if", "ifdef", "ifndef"):
                    if0 += 1
                elif kind == "endif":
                    if0 -= 1
                    if if0 == 0 and chains:
                        chains.pop()
                elif if0 == 1 and (kind == "else" or (kind == "elif" and not zero)):
                    if not (chains and chains[-1]):  # no branch of this group was taken yet
                        if0 = 0  # `#elif 0` keeps the group dead
                        if chains:
                            chains[-1] = kind == "else" or one
                continue
            if kind == "if" and zero:
                chains.append(False)
                if0 = 1
                continue
            if kind in ("if", "ifdef", "ifndef"):
                chains.append(kind == "if" and one)
            elif kind == "endif" and chains:
                chains.pop()
            elif kind in ("elif", "else") and chains:
                # A branch whose condition is a literal true was taken: the alternatives after it are
                # dead. After a condition this cannot evaluate the next branch stays code (checked).
                if chains[-1] or (kind == "elif" and zero):
                    if0 = 1
                    continue
                chains[-1] = kind == "else" or one
            if kind in ("error", "warning"):  # the text is a message
                continue
            if kind in ("include", "include_next", "import") and rest.startswith("<"):
                # A header name <...> is one token: an apostrophe in it opens no literal.
                close = masked.find(">", masked.find("<"))
                i = close + 1 if close >= 0 else n
            if kind in ("if", "elif"):
                # `__has_include(<don't.h>)`: the header name is one token here too. Blanked in
                # place, so every column keeps its meaning.
                line = HAS_INCLUDE_ANGLE.sub(
                    lambda m: m.group(1) + " " * (len(m.group(0)) - len(m.group(1))), line)
        while i < n:
            c = line[i]
            if raw is not None:
                j = line.find(raw, i)
                if j < 0:
                    i = n
                else:
                    i, raw = j + len(raw), None
            elif block:
                j = line.find("*/", i)
                if j < 0:
                    i = n
                else:
                    i, block = j + 2, False
            elif quote:
                if c == "\\":
                    if i + 1 == n:  # line continuation: the literal goes on in the next line
                        carry, quote = quote, None
                        break
                    i += 2
                    continue
                if c == quote:
                    quote = None
                i += 1
            elif line.startswith("//", i):
                i = n
            elif line.startswith("/*", i):
                block_blank = re.sub(r"/\*.*?\*/", " ", line[:i]).strip() == ""
                block, i = True, i + 2
            elif c == '"':
                m = RAW_PREFIX.search(line[:i])
                d = re.match(r"[^\s()\\]{0,16}\(", line[i + 1:]) if m else None
                if d and not (m.start() > 0 and (line[m.start() - 1].isalnum() or line[m.start() - 1] == "_")):
                    raw = ")" + d.group(0)[:-1] + '"'
                    i += 1 + d.end()
                else:
                    quote, i = '"', i + 1
            elif c == "'":
                tok = re.search(r"[0-9A-Za-z_.]*$", line[:i]).group(0)
                # A C++14 digit separator inside a preprocessing number, which starts with a digit
                # or with a dot and a digit: 1'000'000, .1'2
                if re.match(r"\.?[0-9]", tok):
                    i += 1
                else:
                    quote, i = "'", i + 1
            else:
                i += 1
        if quote and last_bad != idx - 1:  # the tail of a broken literal is not a second finding
            bad.append(num)
            last_bad = idx
    return bad


def py_open_lines(text, comments=None):
    """1-based numbers of lines that end inside a single-quoted (not triple-quoted) Python string.

    `comments`, when given, collects the numbers of the lines that end in a `#` comment: a backslash at
    the end of one is part of the comment, not a line continuation."""
    bad, triple, carry = [], None, None
    for num, line in enumerate(text.split("\n"), 1):
        line = line.rstrip("\r")
        i, n, quote, carry = 0, len(line), carry, None
        while i < n:
            c = line[i]
            if triple:
                if c == "\\":
                    i += 2
                elif line.startswith(triple, i):
                    i, triple = i + 3, None
                else:
                    i += 1
            elif quote:
                if c == "\\":
                    if i + 1 == n:
                        carry, quote = quote, None
                        break
                    i += 2
                    continue
                if c == quote:
                    quote = None
                i += 1
            elif c == "#":
                if comments is not None:
                    comments.add(num)
                break
            elif c in "\"'":
                if line.startswith(c * 3, i):
                    triple, i = c * 3, i + 3
                else:
                    quote, i = c, i + 1
            else:
                i += 1
        if quote and not (bad and bad[-1] == num - 1):
            bad.append(num)
    return bad


def bad_literals(path, content, only=None):
    """Findings for lines (restricted to the line numbers in `only`) that end inside a string literal."""
    if content is None:
        return []
    e = ext(path)
    comments = set()  # Python lines ending in a `#` comment: a trailing backslash does not continue them
    if e in LIT_CXX_EXT:
        lines = cxx_open_lines(content, e != ".c")
    elif e == ".py":
        lines = py_open_lines(content, comments)
    else:
        return []
    src = content.split("\n")
    out = []
    for last in lines:
        # the finding belongs to its whole logical line: the physical lines joined by backslashes
        first = last
        while first > 1 and first - 1 not in comments and src[first - 2].rstrip("\r").endswith("\\"):
            first -= 1
        span = range(first, last + 1)
        mine = [n for n in span if only is None or n in only]
        if mine and not any(ALLOW in src[n - 1] for n in span):
            # reported at the last line of the span, or at the first added one when that is unchanged
            out.append((path, last if last in mine else mine[0], "bad-literal",
                        "the line ends inside an unterminated string or char literal (a real newline where "
                        "`\\n` was meant?)"))
    return out


# ---------------------------------------------------------------- driver

def lint(diff_text, tree, allow=frozenset(), base=None):
    """`base`: the tree before the change, for masking removed lines inside the whole old file."""
    files = parse_diff(diff_text)
    findings = []
    # A file with added lines that the head tree cannot read would lose its findings silently
    # (a path the parser did not decode, a submodule): that is a failure of the lint, not a pass.
    for path, d in files.items():
        if d["added"] and lintable(path, CODE_EXT | LIT_CXX_EXT | {".py"}) and tree.read(path) is None:
            first = d["added"][0][0]
            findings.append((path, first, "unread-file", "the head tree has no such file; nothing "
                             "in it was checked"))
    for path, d in files.items():
        if lintable(path, CODE_EXT) and d["added"]:
            findings += dup_blocks(path, d["added"], tree.read(path))
    for path, d in files.items():
        if lintable(path, LIT_CXX_EXT | {".py"}) and d["added"]:
            findings += bad_literals(path, tree.read(path), {n for n, _ in d["added"]})
    removed = [(p, d["removed"], d.get("old") or p) for p, d in files.items()]
    removed += [(p, [rows], p) for p, rows in parse_deleted(diff_text).items()]
    findings += removed_calls(tree, removed, allow, base)
    findings += removed_includes(tree, files, allow, base)
    return sorted(set(findings))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="origin/master")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--exact", action="store_true", help="diff BASE..HEAD instead of from the merge base")
    ap.add_argument("--scan-tree", action="store_true", help="bad-literal over every line of HEAD's tree")
    a = ap.parse_args(argv)

    def git(*x):
        return subprocess.run(["git", *x], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", check=True).stdout

    if a.scan_tree:
        tree = GitTree(a.head)
        found = [f for p in tree.names() if lintable(p, LIT_CXX_EXT | {".py"}) for f in bad_literals(p, tree.read(p))]
        for path, line, rule, why in found:
            print(f"{path}:{line}: [{rule}] {why}")
        return 1 if found else 0

    base = a.base if a.exact else git("merge-base", a.base, a.head).strip()
    allow = set(re.findall(r"diff-lint: allow (removed-def|removed-include) (\S+)",
                           git("log", "--format=%B", f"{base}..{a.head}")))
    found = lint(git("diff", "-U0", "--no-color", "-M", "--src-prefix=a/", "--dst-prefix=b/", base, a.head), GitTree(a.head), allow, GitTree(base))
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
