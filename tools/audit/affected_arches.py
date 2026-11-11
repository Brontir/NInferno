#!/usr/bin/env python3
"""Maps a diff to the GPU arches whose behaviour it can affect, and to the ctest selection to run.

A change that cannot be pinned to one arch answers "all": routes/**, src/core/data/**, src/ops/generic/data/**,
build files, golden files, any file that uses an arch macro (NINFERNO_BUILD_ARCH, WITHOUT_NVFP4,
__CUDA_ARCH__), any device source, any file a .cu file includes (transitively, whatever its
extension), Python tools, test fixtures and anything no rule matches. Only a short list of host
product directories (chat, connect, media, text, structured, convert, docs, ...) answers "host": any
one machine, since no device is read; a host file that a C++ unit built only under an arch guard is,
or includes, answers "all". A source listed in CMake only under one arch's guard answers
that arch. A diff is judged on both sides (merge base and head, read from git, not from the
checkout), so removing an arch macro still answers "all". The answer always names the full ctest:
label filters are not suggested (the labels of the suite are not read here). With --ninja the dirty
objects of a build tree are joined with the git diff and the uncommitted edits, because a dry run
cannot see run-time files and stops at the manifest step.
"""
import argparse
import json
import os
import posixpath
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

ARCHES = ("sm_70", "sm_75", "sm_86", "sm_89", "sm_120a")
# Machines that test each arch are site-specific and are not kept in the repository. They are read
# from the JSON file named by $NINFERNO_STANDS_FILE, else from tools/audit/stands.local.json (not
# tracked). Format, see tools/audit/stands.example.json:
#   {"stands": {"sm_89": ["stand-a"], ...}, "any_one": "stand-a"}
# Without a file every arch lists no machine and host-only changes name none.
STANDS_ENV = "NINFERNO_STANDS_FILE"
STANDS_LOCAL = Path(__file__).resolve().parent / "stands.local.json"


def load_stands():
    """Returns ({arch: [machine, ...]}, machine for host-only changes or None)."""
    name = os.environ.get(STANDS_ENV)
    path = Path(name) if name else STANDS_LOCAL
    stands = {a: [] for a in ARCHES}
    if not path.is_file():
        if name:
            raise SystemExit(f"{STANDS_ENV}={name}: no such file")
        return stands, None
    # The machines are advice only: a malformed file is reported and read as none, so it never
    # changes which arches the diff affects or the exit code.
    def unusable(why):
        print(f"warning: {path}: {why}; no machines listed", file=sys.stderr)
        return {a: [] for a in ARCHES}, None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return unusable(f"not readable JSON ({e})")
    table = data.get("stands", {}) if isinstance(data, dict) else None
    if not isinstance(table, dict):
        return unusable('"stands" is not an object')
    for arch, names in table.items():
        if arch not in stands:
            print(f"warning: {path}: unknown arch {arch}, skipped", file=sys.stderr)
            continue
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            return unusable(f"{arch} is not a list of names")
        stands[arch] = list(names)
    any_one = data.get("any_one")
    return stands, any_one if isinstance(any_one, str) else None


# Paths whose change can move any arch's behaviour.
ALL_PATTERNS = [
    r"^routes/",
    r"^src/core/data/",
    # every build embeds all kernel-facts snapshots; a substituted board reads its own arch's file
    r"^src/ops/generic/data/",
    r"(^|/)CMakeLists\.txt$",
    r"\.cmake$",
    r"(^|/)CMakePresets\.json$",
    # golden files: one directory holds the files of several arches, and the arch in a file name
    # says whose board it describes, not which build set reads it
    r"^tests/.*_golden/",
    r"^third_party/",
]
# Arch hints in a file name, matched on whole tokens: a word must not continue a longer word
# ("conv1000" is not "v100") and a number must not continue a longer number.
def _hint(words, numbers):
    return r"(?:(?<![a-z])(?:%s)|(?<![0-9])(?:%s))(?![0-9])" % (words, numbers)


ARCH_NAME_PATTERNS = [
    ("sm_70", _hint(r"volta|sm_?70|gv100|v100", r"sm_?70")),
    ("sm_75", _hint(r"turing|sm_?75|tu10[0-9]", r"2080|sm_?75")),
    ("sm_86", _hint(r"sm_?86|ga10[2-9]", r"3060|3090")),
    ("sm_89", _hint(r"sm_?89|ad10[0-9]", r"4060|4070|4090")),
    ("sm_120a", _hint(r"sm_?120|gb20[0-9]|blackwell", r"5070|5090")),
]
# Arches outside the five lab arches cannot be tested on one machine: run everything.
# Any sm_<number> that is not one of the lab arches (sm_121, sm_130, sm_100a, sm_90a ...) is one.
OTHER_ARCH_NAME = (r"(?<![a-z0-9])(?:sm_?(?!(?:70|75|86|89|120a)(?![0-9a-z]))[0-9]{2,}"
                   r"|ga100|gh100|hopper|a100|h100)(?![0-9])")
# Host-only: no read of the card's compute capability, generation, nvfp4 or mma_slow facts (checked
# by grep). Not listed, so they answer "all": src/ui (catalog.cpp, app.cpp pick fp4 by cc), src/serve
# (request_log.cpp reads the capability), apps/cli, apps/serve, apps/ui; src/artifact and
# tests/artifact (src/artifact/materializer.cpp includes cuda_runtime.h and its test needs a device);
# tools/ (kernel_facts.py runs per arch in ctest) and tests/fixtures/ (issue #611 maps both to all).
HOST_PREFIXES = (
    "src/chat/", "src/connect/", "src/media/", "src/text/", "src/structured/", "src/convert/",
    "apps/connect/", "apps/convert/", "tests/chat/", "tests/convert/", "tests/convert_native/",
    "tests/release/", "docs/", ".github/",
)
# Inside a host directory only these file types are host code or text. Anything else (a .h.in
# template configured into a header, a .def, a script run by add_custom_command, data) answers "all".
# This list is a CHOICE of the maintainers, not derived from a fact of the build: the code types are
# host everywhere in the host directories; the text types (.md .txt .json .py .yml .yaml) only outside
# src/ and apps/, where a data file can be consumed by a device source at build or run time (a .txt
# generator input, an embedded .json). Under docs/, .github/ and tests/ they are read by tools and
# host tests only. A text file under src/ answers "all".
HOST_CODE_EXT = (".cpp", ".cc", ".h", ".hpp", ".inl")
HOST_TEXT_EXT = (".md", ".txt", ".json", ".py", ".yml", ".yaml")
HOST_TEXT_EXCLUDED_PREFIXES = ("src/", "apps/")
# Arch-keyed allowlists of the SASS audits are data of device code.
ALL_PATTERNS_EXTRA = [r"(^|/)pdl_[a-z_]*\.txt$", r"^tools/pdl_"]
HOST_FILE = re.compile(
    r"^tests/test_(connect|pdf|media|host|load_refusals|kit|registry|admission)[^/]*$")
HOST_EXT = (".md",)
ARCH_MACROS = re.compile(
    r"NINFERNO_BUILD_ARCH\w*|WITHOUT_NVFP4|WITH_NVFP4|__CUDA_ARCH__|NINFERNO_(?:VOLTA|HMMA)\w*|"
    r"NINFERNO_ARCH_FAMILY_SPECIFIC|NINFERNO_FULL_CUDA_ARCHS")
DEVICE_EXT = (".cu", ".cuh")
SOURCE_EXT = (".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".inl")
# Translation units: only these can be built on one arch's tree alone. A header listed under a guard
# is still read by every unguarded file that includes it.
TU_EXT = (".cu", ".cpp", ".cc")
HOST_TU_EXT = (".cpp", ".cc")  # a C++ translation unit; .cu ones are device sources anyway
# CMake commands that name files without building them: a name there lists no source. Any other
# statement (a source-list call, or the bare lines of a list) is read as a listing.
NOT_SOURCE_COMMAND = re.compile(
    r"(message|install|configure_file|string|option|file|add_test|set_tests_properties|set_property"
    r"|get_filename_component|add_custom_command|add_custom_target)$", re.I)
# A host file that names the device-facts API reads the card (or the arch it was built for) at run
# time, whatever its includes say: it answers "all". Checked on the changed file itself.
DEVICE_API = re.compile(
    r"cuda_runtime|cudaDeviceGetAttribute|cudaGetDeviceProperties|device_class|DeviceFeatures|"
    r"gpu_facts|kernel_facts")
# Text files that hold code without being compiled by themselves: scanned for arch macros.
TEXT_CODE_EXT = SOURCE_EXT + (".inc", ".def", ".ipp", ".hh", ".hxx", ".in")
# `#include MACRO`: the file that is included is chosen by a macro (often an arch-derived one).
COMPUTED_INCLUDE = re.compile(r"^\s*#\s*include\s+[A-Za-z_]", re.M)
INCLUDE = re.compile(r'^\s*(?:/\*.*?\*/\s*)*#\s*include\s*([<"])([^>"]+)[>"]', re.M)
# A backslash at the end of a line joins it to the next (translation phase 2).
LINE_SPLICE = re.compile(r"\\[ \t]*\r?\n")


def guard_arches(cond):
    """Arches a guard narrows to: set() if it names none, None if it mentions the arch in any form
    but a positive NINFERNO_BUILD_ARCH STREQUAL "<n>" (NOT, OR, GREATER, LESS, VERSION_, lists)."""
    if "ARCH" not in cond:
        return set()
    table = {"70": "sm_70", "75": "sm_75", "86": "sm_86", "89": "sm_89", "120": "sm_120a"}
    m = re.search(r'NINFERNO_BUILD_ARCH\s+STREQUAL\s+"?(\d+)"?', cond)
    if m and not re.search(r"\b(NOT|OR|GREATER|LESS|VERSION_\w+)\b", cond):
        return {table[m.group(1)]} if m.group(1) in table else None
    return None


class Verdict:
    def __init__(self):
        self.arches = set()
        self.all = False
        self.host = False
        self.reasons = {}

    def add_all(self, path, why):
        self.all = True
        self.reasons[path] = "all: " + why

    def add_arch(self, path, arch, why):
        self.arches.add(arch)
        self.reasons[path] = f"{arch}: {why}"

    def add_host(self, path, why):
        self.host = True
        self.reasons[path] = "host: " + why

    def result(self):
        arches = list(ARCHES) if self.all else sorted(self.arches, key=ARCHES.index)
        table, any_one = load_stands()
        stands = {a: table[a] for a in arches}
        if not arches and self.host:
            stands = {"any": [any_one] if any_one else []}
        return {
            "arches": arches,
            "all": self.all,
            "host": self.host or self.all,
            "stands": stands,
            "ctest": self.ctest(),
            "reasons": self.reasons,
        }

    def ctest(self):
        """The test selection. It is always the whole suite: which tests carry which label is not
        read here, and a label filter named without that would silently skip tests."""
        if self.all:
            return "full ctest on every arch's stand"
        if not (self.arches or self.host):
            return "no test needed"
        return "full ctest on the listed stands" if self.arches else "full ctest on one stand"


# Ignored build trees (.gitignore: build/, build-*/, cmake-build-*/) hold generated CMake files that
# list the same sources unguarded; they never take part in discovery. In a working tree a directory is
# a build tree when it sits directly under the root with such a name (the path is taken relative to
# the root, so a root that lives below a directory called "build-x" is not affected) or holds a
# CMakeCache.txt. In a git ref only tracked files exist and nothing is excluded.
BUILD_DIR = re.compile(r"^(?:build|build-.*|cmake-build-.*)$")
CMAKE_NAME = re.compile(r"(?:^|/)CMakeLists\.txt$|\.cmake$")


def git_run(root, *args):
    cp = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if cp.returncode:
        raise SystemExit("git %s failed: %s" % (" ".join(args), cp.stderr.decode(errors="replace").strip()))
    return cp.stdout


def decode_text(data):
    """Source bytes as text; UTF-16 with a byte order mark is read as UTF-16."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8", errors="replace")


# Listing entry of a source under an arch condition guard_arches() cannot narrow (OR, a range, NOT,
# a bracket comment in the file): not "unguarded", and not one arch either.
UNKNOWN_GUARD = object()


class Repo:
    """Header graph of a source tree: which files a .cu file includes, transitively.

    With ref=None the tree is the working directory; with a git ref every file name and content is
    read from that commit, whatever is checked out. Nodes are all files of the tree, whatever
    their extension (an .inc or .def a header includes is part of the graph)."""

    def __init__(self, root, files=None, ref=None):
        self.root = Path(root)
        self.ref = ref
        self._text = {}
        self._all = None
        if ref is not None:
            self._all = [n for n in git_run(root, "ls-tree", "-r", "--name-only", "-z", ref)
                         .decode("utf-8", errors="replace").split("\0") if n]
        # Every file of the tree is a node: a .cu file may include a header anywhere (docs/,
        # tools/, a generated file), and a node only widens the answer.
        self.files = files if files is not None else list(self.all_files())
        self.fileset = set(self.files)
        self.by_suffix = {}
        for f in self.files:
            parts = PurePosixPath(f).parts
            for i in range(len(parts)):
                self.by_suffix.setdefault("/".join(parts[i:]), []).append(f)
        self.device_headers = None
        if ref is not None:
            self._preload([f for f in self._all
                           if f.endswith(TEXT_CODE_EXT)
                           or CMAKE_NAME.search(f)])

    def all_files(self):
        """Every file of the tree (root-relative, forward slashes): the ref's tree, or the working
        directory without build trees and .git."""
        if self._all is None:
            out = []
            for dirpath, dirs, names in os.walk(self.root):
                here = Path(dirpath)
                dirs[:] = [d for d in dirs if d != ".git"
                           and not (here == self.root and BUILD_DIR.match(d))
                           and not (here / d / "CMakeCache.txt").is_file()]
                out.extend((here / n).relative_to(self.root).as_posix() for n in names)
            self._all = out
        return self._all

    def _preload(self, names):
        """Blobs of the ref through one `git cat-file --batch`."""
        if not names:
            return
        req = "".join(f"{self.ref}:{n}\n" for n in names).encode()
        cp = subprocess.run(["git", "-C", str(self.root), "cat-file", "--batch"], input=req,
                            capture_output=True)
        if cp.returncode:
            raise SystemExit("git cat-file failed: " + cp.stderr.decode(errors="replace").strip())
        data, pos = cp.stdout, 0
        for n in names:
            eol = data.index(b"\n", pos)
            head = data[pos:eol].split()
            pos = eol + 1
            if len(head) == 3:                  # "<sha> <type> <size>": a body of that size follows
                size = int(head[2])
                if head[1] == b"blob":
                    self._text[n] = decode_text(data[pos:pos + size])
                pos += size + 1

    def read(self, path):
        if self.ref is not None:
            if path not in self._text:
                cp = subprocess.run(["git", "-C", str(self.root), "cat-file", "blob",
                                     f"{self.ref}:{path}"], capture_output=True)
                self._text[path] = decode_text(cp.stdout) if cp.returncode == 0 else ""
            return self._text[path]
        try:
            return decode_text((self.root / path).read_bytes())
        except OSError:
            return ""

    def cmake_files(self):
        return [n for n in self.all_files() if CMAKE_NAME.search(n)]

    def resolve(self, includer, name, quoted=False):
        """Every tracked file an `#include "name"` or `<name>` in `includer` can mean. A quoted
        name is looked up in the includer's own directory first and only then in the include
        directories, so a hit in the includer's directory is the only answer. Otherwise (angle
        include, or no local hit): every file whose path ends with the name."""
        parts = []
        for part in (PurePosixPath(includer).parent / name).parts:
            if part == "..":
                if parts:
                    parts.pop()
            elif part != ".":
                parts.append(part)
        local = "/".join(parts)
        if quoted and local in self.fileset:
            return [local]
        tail = [q for q in PurePosixPath(name).parts if q not in ("..", ".")]
        # An angle include does not search the includer's directory first, and an include
        # directory may hold a same-named file: every candidate is an edge (only widens).
        found = list(self.by_suffix.get("/".join(tail), []))
        if local in self.fileset and local not in found:
            found.insert(0, local)
        return found

    def cmake_listing(self):
        """basename -> one entry per listing: the set of arches of its guards, None if unguarded."""
        if getattr(self, "_listing", None) is None:
            listing = {}
            for cm in self.cmake_files():
                stack = []
                text = self.read(cm) or ""
                # `#[` opens a bracket comment only in some places (not inside a line comment or a
                # quoted argument); a file holding one is not parsed for guards at all, so a
                # misread bracket can only widen: every listing in it is unguarded.
                opaque = "#[" in text
                for stmt in cmake_statements(text):
                    m = re.match(r"\s*(if|elseif|else|endif)\s*\((.*)\)\s*$", stmt, re.I | re.S)
                    if m:
                        kw = m.group(1).lower()
                        if kw == "if":
                            stack.append([m.group(2), "ARCH" in m.group(2)])
                        elif kw in ("elseif", "else") and stack:
                            # an else / elseif branch is where an earlier arch condition did not
                            # hold: if any condition of the chain names the arch, the branch is
                            # arch-dependent too, and its guard cannot be narrowed (UNKNOWN). Once a
                            # chain names the arch, every later elseif / else of it is UNKNOWN, even
                            # an elseif that alone could be narrowed (this only widens). "ARCH" is
                            # matched as a substring of the condition, as guard_arches() does, so
                            # any variable with ARCH in its name counts (again only widens).
                            cond = m.group(2) if kw == "elseif" else ""
                            stack[-1][1] = stack[-1][1] or "ARCH" in cond
                            stack[-1][0] = "NOT ARCH_CHAIN" if stack[-1][1] else cond
                        elif kw == "endif" and stack:
                            stack.pop()
                        continue
                    # A name in message() or install() lists nothing (it could otherwise narrow a
                    # globbed unit to one guard's arch).
                    cmd = re.match(r"\s*(\w+)\s*\(", stmt)
                    if cmd and NOT_SOURCE_COMMAND.match(cmd.group(1)):
                        continue
                    guards = [guard_arches(g) for g, _ in stack]
                    if opaque or any(g is None for g in guards):
                        arches = UNKNOWN_GUARD     # guarded by something this cannot narrow
                    else:
                        arches = set().union(*guards) or None   # None: no arch guard at all
                    for f in re.findall(r"[\w./${}-]+\.(?:cuh|cu|cpp|cc|hpp|h)\b", stmt):
                        listing.setdefault(PurePosixPath(f).name, []).append(arches)
            self._listing = listing
        return self._listing

    def built_on(self, path):
        """The one arch whose tree alone builds a listed source; "all" if unguarded or on several
        trees; None if no cmake file lists it (a header). Listings are keyed by file name, so when
        another file of the tree has the same name the listing may be that file's: "all"."""
        name = PurePosixPath(path).name
        entries = self.cmake_listing().get(name)
        if not entries:
            return None
        if sum(1 for f in self.files if PurePosixPath(f).name == name) > 1:
            return "all"
        if any(e is None or e is UNKNOWN_GUARD for e in entries):
            return "all"
        union = set().union(*entries)
        return next(iter(union)) if len(union) == 1 else "all"

    def arch_guarded(self, path):
        """True when some cmake listing of the source is under an arch guard (one this can narrow
        or not): some trees may not build it. Listings are keyed by file name, so a name shared with
        an unguarded unit elsewhere keeps the guarded status (only widens)."""
        entries = self.cmake_listing().get(PurePosixPath(path).name)
        return bool(entries) and any(e is not None for e in entries)

    def guarded_host_includer(self, path):
        """The first C++ translation unit that is `path` or includes it and is built only under an
        arch guard, else None: a host stand may not build that unit at all."""
        for f in sorted(self.files):
            if f.endswith(HOST_TU_EXT) and self.arch_guarded(f) and (
                    f == path or path in self.include_closure(f)):
                return f
        return None

    def include_closure(self, path):
        """Every file `path` includes, transitively (not `path` itself unless it includes itself)."""
        memo = self.__dict__.setdefault("_closure", {})
        if path not in memo:
            seen = set()
            stack = [path]
            while stack:
                f = stack.pop()
                for quoted, name in include_names(self.read(f)):
                    for h in self.resolve(f, name, quoted):
                        if h not in seen:
                            seen.add(h)
                            stack.append(h)
            memo[path] = seen
        return memo[path]

    def arch_user_in_closure(self, path):
        """The first file of the include closure of `path` that uses an arch macro or a computed
        include, else None."""
        for h in sorted(self.include_closure(path)):
            if h.endswith(TEXT_CODE_EXT) and uses_arch(self.read(h)):
                return h
        return None

    def headers_reached_from_cu(self):
        if self.device_headers is None:
            seen = set()
            stack = [f for f in self.files if f.endswith(".cu")]
            while stack:
                f = stack.pop()
                for quoted, name in include_names(self.read(f)):
                    for h in self.resolve(f, name, quoted):
                        if h not in seen:
                            seen.add(h)
                            stack.append(h)
            self.device_headers = seen
        return self.device_headers


def cmake_code(text):
    """(code, shape) of a CMake file, equal length: `code` is the text without comments, `shape`
    is the same with the inside of every quoted argument and bracket argument blanked (line breaks
    kept), so a `#` or a parenthesis inside a string is neither a comment nor a bracket. A line
    comment runs to the line end, a bracket comment (#[[ ... ]], #[=[ ... ]=]) to its closer; a
    quoted argument may span lines and holds `\\"` escapes."""
    code, shape = [], []
    i, n = 0, len(text)

    def keep(chunk, blank):
        code.append(chunk)
        shape.append(re.sub(r"[^\n]", " ", chunk) if blank else chunk)

    while i < n:
        c = text[i]
        if c == "#":
            m = re.compile(r"#\[(=*)\[").match(text, i)
            if m:
                end = text.find("]" + m.group(1) + "]", m.end())
                end = n if end < 0 else end + len(m.group(1)) + 2
            else:
                end = text.find("\n", i)
                end = n if end < 0 else end
            # a comment leaves its line breaks, so line structure stays that of the file
            gap = "\n" * text.count("\n", i, end)
            code.append(gap)
            shape.append(gap)
            i = end
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            keep(text[i:j], True)
            i = j
        else:
            m = re.compile(r"\[(=*)\[").match(text, i) if c == "[" else None
            if m:
                end = text.find("]" + m.group(1) + "]", m.end())
                end = n if end < 0 else end + len(m.group(1)) + 2
                keep(text[i:end], True)
                i = end
            else:
                keep(c, False)
                i += 1
    return "".join(code), "".join(shape)


def cmake_statements(text):
    """The commands of a CMake file as one string each, comments removed, a call that spans lines
    joined (so `if(A` / `OR B)` is one condition). A `#` or a parenthesis inside a quoted or
    bracket argument is part of the argument."""
    out, buf, depth = [], [], 0
    code_text, shape_text = cmake_code(text)
    for code, shape in zip(code_text.split("\n"), shape_text.split("\n")):
        if not buf and not code.strip():
            continue
        buf.append(code)
        depth += shape.count("(") - shape.count(")")
        if depth <= 0:
            out.append(" ".join(buf))
            buf, depth = [], 0
    if buf:
        out.append(" ".join(buf))
    return out


def without_comments(text):
    """The text with each block comment replaced by its line breaks and a blank (the preprocessor
    reads a comment as whitespace): `# /* c */ include "x.h"` is an include. A comment that ends
    in the middle of a line may let a following `#` start a line here though it does not in C:
    that only adds include edges, which only widens the answer."""
    return re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n") + " ", text, flags=re.S)


def include_names(text):
    """The names of the `#include "..."` and `#include <...>` lines of a source: those of the raw
    text and those of the text without comments (a `/*` inside a string literal would hide an
    include from the second alone), so stripping comments only adds edges. The spliced text and
    the unspliced one are both read (MSVC does not splice backslash + blanks + newline, GCC
    does), so splicing only adds edges too."""
    names, seen = [], set()
    for text in (spliced(raw := text), raw):
        for scan in (text, without_comments(text)):
            for q, n in INCLUDE.findall(scan):
                if (q == '"', n) not in seen:
                    seen.add((q == '"', n))
                    names.append((q == '"', n))
    return names


def spliced(text):
    """The text with each backslash-newline pair removed, as the preprocessor joins such lines
    before it reads a directive (`#include \\` + `"x.h"` is one include). Nothing here reports a
    line number, so the joined lines are not counted again."""
    return LINE_SPLICE.sub("", text)


def uses_arch(text):
    text = spliced(text)
    return bool(ARCH_MACROS.search(text) or COMPUTED_INCLUDE.search(text)
                or COMPUTED_INCLUDE.search(without_comments(text)))


def is_host_path(p):
    """True when the path is host code or text by its directory and type alone."""
    if p.startswith(HOST_PREFIXES):
        if p.endswith(HOST_CODE_EXT):
            return True
        if p.endswith(HOST_TEXT_EXT) and not p.startswith(HOST_TEXT_EXCLUDED_PREFIXES):
            return True
    return bool(HOST_FILE.match(p)) or p.endswith(HOST_EXT)


def host_reads_device(p, repo, read, base):
    """Why a host source file is not host after all (None if it is): it names the device-facts API,
    or a file of its include closure uses an arch macro or a computed include. Both sides of a diff
    are judged. Only widens the answer."""
    for side_read, side in ((read, repo), (base.read if base is not None else None, base)):
        if side_read is None:
            continue
        if DEVICE_API.search(side_read(p) or ""):
            return "reads the device-facts API (cuda_runtime, device_class, gpu_facts, ...)"
        hit = side.arch_user_in_closure(p)
        if hit:
            return f"includes {hit}, which uses an arch macro or a computed include"
        unit = side.guarded_host_includer(p)
        if unit:
            return (f"{unit} is built only under an arch guard"
                    + ("" if unit == p else " and includes it"))
    return None


def classify(paths, repo, read=None, base=None, extra_all=None, missing=()):
    """read(path) gives the text of a changed file; default: the repo's tree.

    `base` is the Repo of the other side of a diff: a change that REMOVES an arch macro or the last
    .cu include of a header is judged on both sides, so removals still get GPU coverage.
    `extra_all` is a reason that makes the answer "all" whatever the paths say."""
    read = read or repo.read
    v = Verdict()
    if extra_all:
        v.add_all("(ninja -n)", extra_all)
    device = set(repo.headers_reached_from_cu())
    if base is not None:
        device |= base.headers_reached_from_cu()
    for p in sorted(set(paths)):
        if p in missing:
            v.add_all(p, "named file does not exist (a deletion): its old content is not read")
            continue
        if p.startswith("../") or p.startswith("/") or re.match(r"[A-Za-z]:", p):
            v.add_all(p, "outside the source tree")
            continue
        if any(re.search(pat, p) for pat in ALL_PATTERNS):
            v.add_all(p, "routes, data, build or golden file")
            continue
        if p.endswith(TEXT_CODE_EXT) and (uses_arch(read(p) or "")
                                          or (base is not None and uses_arch(base.read(p) or ""))):
            v.add_all(p, "uses an arch macro or a computed include (before or after the change)")
            continue
        name = PurePosixPath(p).name.lower()
        if re.search(OTHER_ARCH_NAME, name):
            v.add_all(p, "arch outside the lab set")
            continue
        if any(re.search(pat, p) for pat in ALL_PATTERNS_EXTRA):
            v.add_all(p, "allowlist of a device audit")
            continue
        if p in device:
            v.add_all(p, "file included by a .cu file")
            continue
        named = [a for a, pat in ARCH_NAME_PATTERNS if re.search(pat, name)]
        if len(named) == 1 and p.endswith(TU_EXT) and repo.built_on(p) == named[0]:
            v.add_arch(p, named[0], "built on that arch's tree alone (cmake guard)")
            continue
        if p.endswith(DEVICE_EXT):
            v.add_all(p, "device source")
            continue
        if is_host_path(p):
            if p.endswith(SOURCE_EXT):
                why = host_reads_device(p, repo, read, base)
                if why:
                    v.add_all(p, why)
                    continue
            v.add_host(p, "host-only code or docs")
            continue
        v.add_all(p, "no rule: conservative")
    return v.result()


def normalise_path(path, root):
    """A user-given path as a root-relative path with forward slashes. Backslashes separate
    directories, `./` and `..` are folded, an absolute path is made relative to the root. A path
    that leaves the root is an error."""
    s = path.replace(chr(92), "/")
    rootp = Path(root).resolve()
    if s.startswith("/") or re.match(r"[A-Za-z]:/", s):
        # Case-insensitive on a Windows drive path (the file system is); a drive path on another
        # system cannot lie below the root.
        if re.match(r"[A-Za-z]:/", s) and os.name != "nt":
            raise SystemExit(f"{path}: not inside the repository root {rootp}")
        full = Path(s).resolve().as_posix()
        base = rootp.as_posix().rstrip("/")
        if os.name == "nt":
            full_cmp, base_cmp = full.lower(), base.lower()
        else:
            full_cmp, base_cmp = full, base
        if not full_cmp.startswith(base_cmp + "/"):
            raise SystemExit(f"{path}: not inside the repository root {rootp}")
        s = full[len(base) + 1:]
    rel = posixpath.normpath(s)
    if rel in (".", "") or rel == ".." or rel.startswith("../"):
        raise SystemExit(f"{path}: not a path inside the repository root {rootp}")
    return rel


def _names(out):
    return [n for n in out.decode("utf-8", errors="replace").split("\0") if n]


def git_diff_paths(base, head, root):
    """Paths changed from the merge base of base and head to head. Renames are not followed: the old
    and the new path both appear (a moved file is a removal and an addition)."""
    cmd = ["git", "-C", str(root), "diff", "--name-only", "--no-renames", "-z"]
    cp = subprocess.run(cmd + [f"{base}...{head}"], capture_output=True)
    if cp.returncode:
        cp = subprocess.run(cmd + [base, head], capture_output=True)
    if cp.returncode:
        raise SystemExit(f"git diff failed: {cp.stderr.decode(errors='replace').strip()}")
    return _names(cp.stdout)


def worktree_paths(root):
    """Uncommitted edits: tracked files that differ from HEAD (old and new path of a move) and
    untracked files that are not ignored."""
    changed = _names(git_run(root, "diff", "--name-only", "--no-renames", "-z", "HEAD"))
    untracked = _names(git_run(root, "ls-files", "--others", "--exclude-standard", "-z"))
    return changed + untracked


def merge_base_repo(root, base, head):
    """The Repo of the merge base (what `base...head` compares against), read from git."""
    mb = subprocess.run(["git", "-C", str(root), "merge-base", base, head],
                        capture_output=True, text=True)
    return Repo(root, ref=mb.stdout.strip() if mb.returncode == 0 and mb.stdout.strip() else base)


BUILD_LINE = re.compile(r"\bBuilding \w+ object\b")
# `.o` (Unix), `.obj` (Ninja/MSVC) or `.res` (resource compiler); the object path may hold spaces and
# may be quoted.
OBJECT_LINE = re.compile(r"\bBuilding \w+ object\s+(.+?)\.(?:o|obj|res)[\"']?\s*$")
# The steps a dry run stops at: it cannot know what CMake would then regenerate.
MANIFEST_STEP = re.compile(r"Re-running CMake|Re-checking globbed directories")
NO_WORK = re.compile(r"ninja: no work to do", re.I)


def sources_from_ninja(text, repo):
    """Sources of the objects a `ninja -n` listing would rebuild; the build tree mirrors the source.

    Reads both spellings of a dry run: `a/CMakeFiles/t.dir/x.cu.o` and the native Windows
    `a\\CMakeFiles\\t.dir\\x.cu.obj` (CRLF line ends, quoted paths). A line that says "Building ...
    object" but cannot be parsed is an error, never a silent "nothing to test"."""
    out = set()
    unparsed = []
    for line in text.splitlines():
        line = line.strip()
        if not BUILD_LINE.search(line):
            continue
        m = OBJECT_LINE.search(line)
        if not m:
            unparsed.append(line)
            continue
        obj = m.group(1).strip().strip("\"'").replace(chr(92), "/")
        m2 = re.match(r"(.*?)/?CMakeFiles/[^/]+\.dir/(.*)$", obj)
        cand = obj
        if m2:
            cand = f"{m2.group(1)}/{m2.group(2)}" if m2.group(1) else m2.group(2)
        cand = posixpath.normpath(cand)
        if cand not in repo.fileset and cand.endswith(".cuda"):
            cand = cand[:-len(".cuda")] + ".cu"
        if cand in repo.fileset:
            out.add(cand)
        else:
            hits = repo.by_suffix.get(PurePosixPath(cand).name, [])
            out.update(hits if hits else [cand])
    if unparsed:
        raise SystemExit("ninja -n: %d build line(s) not understood, refusing to answer "
                         "'no test needed'; first: %s" % (len(unparsed), unparsed[0]))
    return sorted(out)


def ninja_listing(text, repo):
    """(sources, reason): the dirty sources of a dry run and, when the listing cannot be trusted to
    be complete, the reason that makes the answer "all". A dry run that stops at the manifest step
    ("Re-running CMake", "Re-checking globbed directories") lists no object; one with neither a build
    line nor "no work to do" says nothing."""
    sources = sources_from_ninja(text, repo)
    if MANIFEST_STEP.search(text):
        return sources, ("ninja -n stops at the manifest step (CMake would re-run or the globbed "
                         "directories are re-checked): the dirty objects are not listed")
    if not sources and not NO_WORK.search(text):
        return sources, "ninja -n lists no build line and does not say 'no work to do'"
    return sources, None


def run_ninja(build_dir):
    cp = subprocess.run(["ninja", "-C", str(build_dir), "-n"], capture_output=True, text=True)
    if cp.returncode:
        raise SystemExit(f"ninja -n failed: {cp.stderr.strip() or cp.stdout.strip()}")
    return cp.stdout


def render(res):
    if res["all"]:
        lines = ["arches: all (" + ", ".join(res["arches"]) + ")"]
    elif res["arches"]:
        lines = ["arches: " + ", ".join(res["arches"])]
    else:
        lines = ["arches: any one" if res["host"] else "arches: none"]
    for a, st in res["stands"].items():
        lines.append(f"  {a}: {', '.join(st) or '(none configured)'}")
    if res["host"]:
        lines.append("host: yes")
    lines.append("ctest: " + res["ctest"])
    for p, why in res["reasons"].items():
        lines.append(f"  {p}  ({why})")
    return "\n".join(lines)


EPILOG = """\
Modes (the three are exclusive):
  default        the diff of --base...--head, read from git at those refs. It sees commits only: edits
                 that are not committed are not in it.
  --files        the paths you name. Backslashes, ./ and .. are folded; an absolute path must lie in
                 the root. Nothing is read from a diff, so name moves and deletions yourself.
  --ninja DIR    the objects `ninja -n` would rebuild in DIR, joined with the diff --base...--head and
                 the uncommitted edits (a dry run cannot see files read at run time and stops at the
                 manifest step; when it does, the answer is "all").
Answers: "all" (every arch's stand), a list of arches, "any one" (host only: one machine, no device
read) or "none" (nothing changed). The ctest line is always the full suite.
Needs no GPU. Exit 0 on success, a message and non-zero on a usage or git failure.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Maps a diff to the GPU arches whose tests it must run on.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/master",
                    help="base ref of the diff (default origin/master); the merge base is used")
    ap.add_argument("--head", default="HEAD", help="head ref of the diff (default HEAD)")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]),
                    help="repository root (default: the repository this script is in)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--files", nargs="+", metavar="PATH",
                      help="classify these paths instead of a diff")
    mode.add_argument("--ninja", metavar="BUILD_DIR",
                      help="classify the dirty objects of a ninja build tree, the diff and the "
                           "uncommitted edits together")
    ap.add_argument("--json", action="store_true", help="print the answer as JSON")
    args = ap.parse_args(argv)
    base_repo = None
    extra_all = None
    missing = ()
    if args.files:
        repo = Repo(args.root)
        paths = [normalise_path(f, args.root) for f in args.files]
        # a code file that is not there was deleted: nothing here can say what it contained
        missing = {p for p in paths if p.endswith(TEXT_CODE_EXT) and not (Path(args.root) / p).is_file()}
    elif args.ninja:
        repo = Repo(args.root)
        sources, extra_all = ninja_listing(run_ninja(args.ninja), repo)
        if (git_run(args.root, "rev-parse", "--verify", args.head + "^{commit}").strip()
                != git_run(args.root, "rev-parse", "HEAD").strip()):
            # the objects and the uncommitted edits are those of the checkout, the diff is --head's
            extra_all = extra_all or "--head is not the checked-out commit: the working tree is not that ref"
        paths = (sources + git_diff_paths(args.base, args.head, args.root)
                 + worktree_paths(args.root))
        base_repo = merge_base_repo(args.root, args.base, args.head)
    else:
        paths = git_diff_paths(args.base, args.head, args.root)
        # classify what the head ref contains, not what is checked out; the base side is the merge
        # base (what `base...head` compares against), read the same way
        repo = Repo(args.root, ref=args.head)
        base_repo = merge_base_repo(args.root, args.base, args.head)
    res = classify(paths, repo, base=base_repo, extra_all=extra_all, missing=missing)
    print(json.dumps(res, indent=1) if args.json else render(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
