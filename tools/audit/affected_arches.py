#!/usr/bin/env python3
"""Maps a diff to the GPU arches whose behaviour it can affect, and to the ctest selection to run.

A change that cannot be pinned to one arch answers "all": routes/**, src/core/data/**, build files,
selector goldens, any file that uses an arch macro (NINFERNO_BUILD_ARCH, WITHOUT_NVFP4,
__CUDA_ARCH__), any device source, and any header a .cu file includes, transitively. Otherwise a
conservative path map applies: per-arch files (volta_*, *_sm75*, a chip's name) map to their own
arch; host-only product, chat, serve, UI and convert code maps to "any one arch" plus the host
label. A path nothing matches answers "all". A diff is judged on both sides (merge base and head,
read from git, not from the checkout), so removing an arch macro still answers "all". Label-filtered
ctest commands are named only when every registered test has a LABELS property; until then the answer
is the full ctest and says why. A `ninja -n` listing of the native Windows build (`.obj`, backslashes,
CRLF) reads like the Unix one, and a listing with build lines it cannot read is an error, never
"no test needed".

Usage:
  affected_arches.py [--base REF] [--head REF] [--json]    diff base...head (default origin/master..HEAD)
  affected_arches.py --files a b c                          explicit paths
  affected_arches.py --ninja BUILD_DIR                      the dirty targets of a build tree (ninja -n)
Needs no GPU. Exit 0 on success, a message and non-zero on a usage or git failure.
"""
import argparse
import json
import os
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
    r"(^|/)CMakeLists\.txt$",
    r"\.cmake$",
    r"(^|/)CMakePresets\.json$",
    r"^tests/ops/linear/selector_golden/",
    r"^third_party/",
]
# File-name hints, matched on the base name (the first match with exactly one arch wins).
ARCH_NAME_PATTERNS = [
    ("sm_70", r"volta|sm_?70|gv100|v100"),
    ("sm_75", r"turing|sm_?75|tu10[0-9]|2080"),
    ("sm_86", r"sm_?86|ga10[2-9]|3060|3090"),
    ("sm_89", r"sm_?89|ad10[0-9]|4060|4070|4090"),
    ("sm_120a", r"sm_?120|gb20[0-9]|5070|5090|blackwell"),
]
# Arches outside the five lab arches cannot be tested on one machine: run everything.
OTHER_ARCH_NAME = r"sm_?(80|90|100|103)|ga100|gh100|hopper|a100|h100"
# Host-only: no read of the card's compute capability, generation, nvfp4 or mma_slow facts (checked
# by grep). src/ui (catalog.cpp, app.cpp pick fp4 by cc), src/serve (request_log.cpp reads the
# capability), apps/cli, apps/serve and apps/ui are not listed: they answer "all".
HOST_PREFIXES = (
    "src/chat/", "src/connect/", "src/media/", "src/text/", "src/structured/", "src/convert/",
    "src/artifact/", "apps/connect/", "apps/convert/", "tests/chat/", "tests/convert/",
    "tests/convert_native/", "tests/artifact/", "tests/release/", "tests/fixtures/", "tools/",
    "docs/", ".github/",
)
# Arch-keyed allowlists of the SASS audits are data of device code.
ALL_PATTERNS_EXTRA = [r"(^|/)pdl_[a-z_]*\.txt$", r"^tools/pdl_"]
HOST_FILE = re.compile(
    r"^tests/test_(connect|pdf|media|host|load_refusals|kit|registry|admission)[^/]*$")
HOST_EXT = (".md", ".yml", ".yaml")
ARCH_MACROS = re.compile(
    r"NINFERNO_BUILD_ARCH\w*|WITHOUT_NVFP4|WITH_NVFP4|__CUDA_ARCH__|NINFERNO_(?:VOLTA|HMMA)\w*|"
    r"NINFERNO_ARCH_FAMILY_SPECIFIC|NINFERNO_FULL_CUDA_ARCHS")
DEVICE_EXT = (".cu", ".cuh")
SOURCE_EXT = (".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".inl")
DATA_EXT = (".json", ".txt", ".csv")
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.M)


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

    def result(self, labels=(0, 0)):
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
            "ctest": self.ctest(labels),
            "reasons": self.reasons,
        }

    def ctest(self, labels=(0, 0)):
        """The test selection. Label-filtered commands are named only when every registered test
        carries a LABELS property (issue #611 step 1, labels arrive with #625); until then `-L`
        would select a fraction of the suite and silently skip the rest, so the answer is the full
        ctest, with the reason."""
        total, labelled = labels
        if self.all:
            return "full ctest on every arch's stand"
        if not (self.arches or self.host):
            return "no test needed"
        where = "on the listed stands" if self.arches else "on one stand"
        if total == 0 or labelled < total:
            return (f"full ctest {where} (only {labelled} of {total} registered tests carry a "
                    "LABELS property, so a label filter would skip tests)")
        if self.arches:
            return "ctest -L 'host|gpu|golden' on the listed stands"
        return "ctest -L host on one stand"


# Ignored build trees (.gitignore: build/, build-*/, cmake-build-*/) hold generated CMake files that
# list the same sources unguarded; they never take part in discovery.
BUILD_DIR = re.compile(r"^(?:build|build-.*|cmake-build-.*|\.git)$")
CMAKE_NAME = re.compile(r"(?:^|/)CMakeLists\.txt$|\.cmake$")
TOP_DIRS = ("src", "apps", "tests", "third_party", "include")


def git_run(root, *args):
    cp = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if cp.returncode:
        raise SystemExit("git %s failed: %s" % (" ".join(args), cp.stderr.decode(errors="replace").strip()))
    return cp.stdout


class Repo:
    """Header graph of a source tree: which headers a .cu file includes, transitively.

    With ref=None the tree is the working directory; with a git ref every file name and content is
    read from that commit, whatever is checked out."""

    def __init__(self, root, files=None, ref=None):
        self.root = Path(root)
        self.ref = ref
        self._text = {}
        self._tree = None
        if ref is not None:
            self._tree = [n for n in git_run(root, "ls-tree", "-r", "--name-only", "-z", ref)
                          .decode("utf-8", errors="replace").split("\0") if n
                          and not any(BUILD_DIR.match(q) for q in PurePosixPath(n).parts)]
        self.files = files if files is not None else self._walk()
        self.fileset = set(self.files)
        self.by_suffix = {}
        for f in self.files:
            parts = PurePosixPath(f).parts
            for i in range(len(parts)):
                self.by_suffix.setdefault("/".join(parts[i:]), []).append(f)
        self.device_headers = None
        if ref is not None:
            self._preload([f for f in self._tree if f.endswith(SOURCE_EXT) or CMAKE_NAME.search(f)])

    def _walk(self):
        if self._tree is not None:
            return [n for n in self._tree if n.split("/")[0] in TOP_DIRS and n.endswith(SOURCE_EXT)]
        out = []
        for top in TOP_DIRS:
            base = self.root / top
            if not base.is_dir():
                continue
            for p in base.rglob("*"):
                if p.is_file() and p.suffix in SOURCE_EXT:
                    out.append(p.relative_to(self.root).as_posix())
        return out

    def _preload(self, names):
        """All blobs of the ref through one `git cat-file --batch`."""
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
            if len(head) == 3 and head[1] == b"blob":
                size = int(head[2])
                self._text[n] = data[pos:pos + size].decode("utf-8", errors="replace")
                pos += size + 1

    def read(self, path):
        if self.ref is not None:
            return self._text.get(path, "")
        try:
            return (self.root / path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def cmake_files(self):
        if self._tree is not None:
            return [n for n in self._tree if CMAKE_NAME.search(n)]
        out = []
        for cm in self.root.rglob("*"):
            rel = cm.relative_to(self.root).parts
            if any(BUILD_DIR.match(q) for q in rel[:-1]) or not (cm.name == "CMakeLists.txt"
                                                                  or cm.suffix == ".cmake"):
                continue
            if cm.is_file():
                out.append("/".join(rel))
        return out

    def test_labels(self):
        """(registered tests, tests with a LABELS property) read from the CMake files of the tree."""
        names, labelled = set(), set()
        for cm in self.cmake_files():
            text = re.sub(r"(?m)#.*$", "", self.read(cm) if self.ref is not None else self._disk(cm))
            names.update(re.findall(r"add_test\s*\(\s*NAME\s+(\S+)", text, re.I))
            for m in re.finditer(r"set_tests_properties\s*\(([^)]*?)PROPERTIES([^)]*)\)", text, re.I | re.S):
                if "LABELS" in m.group(2):
                    labelled.update(m.group(1).split())
        return len(names), len(names & labelled)

    def _disk(self, path):
        try:
            return (self.root / path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def resolve(self, includer, name):
        parts = []
        for part in (PurePosixPath(includer).parent / name).parts:
            if part == "..":
                if parts:
                    parts.pop()
            elif part != ".":
                parts.append(part)
        local = "/".join(parts)
        if local in self.fileset:
            return local
        tail = [q for q in PurePosixPath(name).parts if q not in ("..", ".")]
        hits = self.by_suffix.get("/".join(tail), [])
        return hits[0] if hits else None

    def cmake_listing(self):
        """basename -> one entry per listing: the set of arches of its guards, None if unguarded."""
        if getattr(self, "_listing", None) is None:
            listing = {}
            for cm in self.cmake_files():
                stack = []
                for line in (self.read(cm) if self.ref is not None else self._disk(cm)).splitlines():
                    code = line.split("#")[0]
                    m = re.match(r"\s*(if|elseif|else|endif)\s*\((.*)", code, re.I)
                    if m:
                        kw = m.group(1).lower()
                        if kw == "if":
                            stack.append(m.group(2))
                        elif kw in ("elseif", "else") and stack:
                            stack[-1] = m.group(2)
                        elif kw == "endif" and stack:
                            stack.pop()
                        continue
                    guards = [guard_arches(g) for g in stack]
                    arches = None if any(g is None for g in guards) else set().union(*guards)
                    for f in re.findall(r"[\w./${}-]+\.(?:cuh|cu|cpp|h)\b", code):
                        listing.setdefault(PurePosixPath(f).name, []).append(arches or None)
            self._listing = listing
        return self._listing

    def built_on(self, path):
        """The one arch whose tree alone builds a listed source; "all" if unguarded or on several
        trees; None if no cmake file lists it (a header)."""
        entries = self.cmake_listing().get(PurePosixPath(path).name)
        if not entries:
            return None
        if any(e is None for e in entries):
            return "all"
        union = set().union(*entries)
        return next(iter(union)) if len(union) == 1 else "all"

    def headers_reached_from_cu(self):
        if self.device_headers is None:
            seen = set()
            stack = [f for f in self.files if f.endswith(".cu")]
            while stack:
                f = stack.pop()
                for name in INCLUDE.findall(self.read(f)):
                    h = self.resolve(f, name)
                    if h and h not in seen:
                        seen.add(h)
                        stack.append(h)
            self.device_headers = seen
        return self.device_headers


def classify(paths, repo, read=None, base=None):
    """read(path) gives the text of a changed file; default: the repo's tree.

    `base` is the Repo of the other side of a diff: a change that REMOVES an arch macro or the last
    .cu include of a header is judged on both sides, so removals still get GPU coverage."""
    read = read or repo.read
    v = Verdict()
    device = set(repo.headers_reached_from_cu())
    if base is not None:
        device |= base.headers_reached_from_cu()
    for p in sorted(set(paths)):
        if any(re.search(pat, p) for pat in ALL_PATTERNS):
            v.add_all(p, "routes, data, build or golden file")
            continue
        if p.endswith(SOURCE_EXT) and (ARCH_MACROS.search(read(p) or "")
                                       or (base is not None and ARCH_MACROS.search(base.read(p) or ""))):
            v.add_all(p, "uses an arch macro (before or after the change)")
            continue
        name = PurePosixPath(p).name.lower()
        if re.search(OTHER_ARCH_NAME, name):
            v.add_all(p, "arch outside the lab set")
            continue
        if any(re.search(pat, p) for pat in ALL_PATTERNS_EXTRA):
            v.add_all(p, "allowlist of a device audit")
            continue
        named = [a for a, pat in ARCH_NAME_PATTERNS if re.search(pat, name)]
        if len(named) == 1 and p.endswith(DATA_EXT):
            v.add_arch(p, named[0], "per-arch data file name")
            continue
        if len(named) == 1 and p.endswith(SOURCE_EXT) and repo.built_on(p) == named[0]:
            v.add_arch(p, named[0], "built on that arch's tree alone (cmake guard)")
            continue
        if p in device:
            v.add_all(p, "header included by a .cu file")
            continue
        if p.endswith(DEVICE_EXT):
            v.add_all(p, "device source")
            continue
        if p.startswith(HOST_PREFIXES) or HOST_FILE.match(p) or p.endswith(HOST_EXT):
            v.add_host(p, "host-only code or docs")
            continue
        v.add_all(p, "no rule: conservative")
    return v.result(repo.test_labels())


def git_diff_paths(base, head, root):
    cp = subprocess.run(["git", "-C", str(root), "diff", "--name-only", f"{base}...{head}"],
                        capture_output=True, text=True)
    if cp.returncode:
        cp = subprocess.run(["git", "-C", str(root), "diff", "--name-only", base, head],
                            capture_output=True, text=True)
    if cp.returncode:
        raise SystemExit(f"git diff failed: {cp.stderr.strip()}")
    return [line for line in cp.stdout.splitlines() if line]


BUILD_LINE = re.compile(r"\bBuilding \w+ object\b")
# `.o` (Unix) or `.obj` (Ninja/MSVC); the object path may hold spaces and may be quoted.
OBJECT_LINE = re.compile(r"\bBuilding \w+ object\s+(.+?)\.(?:o|obj)[\"']?\s*$")


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


def ninja_dirty_sources(build_dir, repo):
    cp = subprocess.run(["ninja", "-C", str(build_dir), "-n"], capture_output=True, text=True)
    if cp.returncode:
        raise SystemExit(f"ninja -n failed: {cp.stderr.strip() or cp.stdout.strip()}")
    return sources_from_ninja(cp.stdout, repo)


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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Maps a diff to the affected GPU arches.")
    ap.add_argument("--base", default="origin/master")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--files", nargs="+")
    ap.add_argument("--ninja", metavar="BUILD_DIR")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    base_repo = None
    if args.ninja or args.files:
        repo = Repo(args.root)
        paths = ninja_dirty_sources(args.ninja, repo) if args.ninja else args.files
    else:
        paths = git_diff_paths(args.base, args.head, args.root)
        # classify what the head ref contains, not what is checked out; the base side is the merge
        # base (what `base...head` compares against), read the same way
        repo = Repo(args.root, ref=args.head)
        mb = subprocess.run(["git", "-C", str(args.root), "merge-base", args.base, args.head],
                            capture_output=True, text=True)
        base_repo = Repo(args.root, ref=mb.stdout.strip() if mb.returncode == 0 and mb.stdout.strip()
                         else args.base)
    res = classify(paths, repo, base=base_repo)
    print(json.dumps(res, indent=1) if args.json else render(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
