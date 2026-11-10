#!/usr/bin/env python3
"""Maps a diff to the GPU arches whose behaviour it can affect, and to the ctest selection to run.

A change that cannot be pinned to one arch answers "all": routes/**, src/core/data/**, build files,
selector goldens, any file that uses an arch macro (NINFERNO_BUILD_ARCH, WITHOUT_NVFP4,
__CUDA_ARCH__), any device source, and any header a .cu file includes, transitively. Otherwise a
conservative path map applies: per-arch files (volta_*, *_sm75*, a chip's name) map to their own
arch; host-only product, chat, serve, UI and convert code maps to "any one arch" plus the host
label. A path nothing matches answers "all".

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
    data = json.loads(path.read_text(encoding="utf-8"))
    for arch, names in data.get("stands", {}).items():
        if arch not in stands:
            raise SystemExit(f"{path}: unknown arch {arch}")
        stands[arch] = list(names)
    return stands, data.get("any_one")


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
        """Labels of the test registrations (issue #611 step 1): host everywhere, gpu on the arches."""
        if self.all:
            return "full ctest on every arch's stand"
        if self.arches:
            return "ctest -L 'host|gpu|golden' on the listed stands"
        if self.host:
            return "ctest -L host on one stand"
        return "no test needed"


class Repo:
    """Header graph of a source tree: which headers a .cu file includes, transitively."""

    def __init__(self, root, files=None):
        self.root = Path(root)
        self.files = files if files is not None else self._walk()
        self.fileset = set(self.files)
        self.by_suffix = {}
        for f in self.files:
            parts = PurePosixPath(f).parts
            for i in range(len(parts)):
                self.by_suffix.setdefault("/".join(parts[i:]), []).append(f)
        self.device_headers = None

    def _walk(self):
        out = []
        for top in ("src", "apps", "tests", "third_party", "include"):
            base = self.root / top
            if not base.is_dir():
                continue
            for p in base.rglob("*"):
                if p.is_file() and p.suffix in SOURCE_EXT:
                    out.append(p.relative_to(self.root).as_posix())
        return out

    def read(self, path):
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
            for cm in self.root.rglob("*"):
                if not (cm.name == "CMakeLists.txt" or cm.suffix == ".cmake") or not cm.is_file():
                    continue
                if any(part in ("build", ".git") for part in cm.parts):
                    continue
                stack = []
                for line in cm.read_text(encoding="utf-8", errors="replace").splitlines():
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


def classify(paths, repo, read=None):
    """read(path) gives the text of a changed file; default: the repo's working tree."""
    read = read or repo.read
    v = Verdict()
    device = repo.headers_reached_from_cu()
    for p in sorted(set(paths)):
        if any(re.search(pat, p) for pat in ALL_PATTERNS):
            v.add_all(p, "routes, data, build or golden file")
            continue
        if p.endswith(SOURCE_EXT) and ARCH_MACROS.search(read(p) or ""):
            v.add_all(p, "uses an arch macro")
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
    return v.result()


def git_diff_paths(base, head, root):
    cp = subprocess.run(["git", "-C", str(root), "diff", "--name-only", f"{base}...{head}"],
                        capture_output=True, text=True)
    if cp.returncode:
        cp = subprocess.run(["git", "-C", str(root), "diff", "--name-only", base, head],
                            capture_output=True, text=True)
    if cp.returncode:
        raise SystemExit(f"git diff failed: {cp.stderr.strip()}")
    return [line for line in cp.stdout.splitlines() if line]


def sources_from_ninja(text, repo):
    """Sources of the objects a `ninja -n` listing would rebuild; the build tree mirrors the source."""
    out = set()
    for line in text.splitlines():
        m = re.search(r"Building \w+ object (\S+?)\.o\b", line)
        if not m:
            continue
        m2 = re.match(r"(.*?)/?CMakeFiles/[^/]+\.dir/(.*)$", m.group(1))
        cand = m.group(1)
        if m2:
            cand = f"{m2.group(1)}/{m2.group(2)}" if m2.group(1) else m2.group(2)
        if cand in repo.fileset:
            out.add(cand)
        else:
            hits = repo.by_suffix.get(PurePosixPath(cand).name, [])
            out.update(hits if hits else [cand])
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
    repo = Repo(args.root)
    if args.ninja:
        paths = ninja_dirty_sources(args.ninja, repo)
    elif args.files:
        paths = args.files
    else:
        paths = git_diff_paths(args.base, args.head, args.root)
    res = classify(paths, repo)
    print(json.dumps(res, indent=1) if args.json else render(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
