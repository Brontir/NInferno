#!/usr/bin/env python3
"""Source scan of the selection rules (docs/maintainer/deterministic-decisions.md, principle 8).

The files under src/ops that choose a route, a plan, a band or a tune (their names say route,
select, plan, tune, band, heuristic, dispatch, policy or rules, and the generic path's selector
generic_gemm.cpp) may not read the environment, a clock, a timer, free memory or a file. Each
existing exception is listed in selection_static_allow.txt with its count; the scan fails on a call
the list does not cover and on an entry whose count is now lower (tighten it).

It also checks that every NINFERNO_* variable named in src/, apps/ or include/ is one the selector
property suite sets (kEnvVars, tests/ops/linear/test_selector_properties.cpp), so a new knob is
tried against the choices.

Usage: selection_static_check.py [repository root]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SELECTION_NAME = re.compile(r"route|select|plan|tune|band|heuristic|dispatch|policy|rules")
EXTRA_FILES = {"src/ops/generic/generic_gemm.cpp"}
SUFFIXES = {".cpp", ".h", ".cu", ".cuh"}

# Name -> pattern of what a rule may not call.
BANNED = {
    "getenv": re.compile(r"\bgetenv\s*\("),
    "sys::env": re.compile(r"\bsys::env\s*\("),
    "std::chrono": re.compile(r"\bstd::chrono\b|\b(steady|system|high_resolution)_clock\b"),
    "cudaEvent": re.compile(r"\bcudaEvent[A-Za-z]*\b"),
    "cudaMemGetInfo": re.compile(r"\bcudaMemGetInfo\b"),
    "fopen": re.compile(r"\bfopen\s*\("),
    "ifstream": re.compile(r"\bifstream\b"),
    "TuneTimer": re.compile(r"\bTuneTimer\b"),
    "tune_mode": re.compile(r"\btune_mode\s*\("),
    "read_cache_file": re.compile(r"\bread_cache_file\s*\("),
}

ENV_LITERAL = re.compile(r'"(NINFERNO_[A-Z0-9_]+)"')


def strip_comments(text: str) -> str:
    """Drops // and /* */ comments, keeping string literals and line numbers."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("\n" * text.count("\n", i, j))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def selection_files(root: Path) -> list[Path]:
    files = []
    for path in sorted((root / "src" / "ops").rglob("*")):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if "/shapes/" in rel:  # generated per-shape kernels, no rules
            continue
        if rel in EXTRA_FILES or SELECTION_NAME.search(path.name):
            files.append(path)
    return files


def read_allow(path: Path) -> dict[tuple[str, str], int]:
    allow: dict[tuple[str, str], int] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3 or parts[1] not in BANNED or not parts[2].isdigit():
            raise SystemExit(f"{path}:{number}: expected '<file> <call> <count>': {raw}")
        allow[(parts[0], parts[1])] = int(parts[2])
    return allow


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2])
    allow = read_allow(root / "tools" / "audit" / "selection_static_allow.txt")
    failures: list[str] = []
    seen: dict[tuple[str, str], int] = {}
    where: dict[tuple[str, str], list[str]] = {}
    files = selection_files(root)
    for path in files:
        rel = path.relative_to(root).as_posix()
        code = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        for number, line in enumerate(code.splitlines(), 1):
            for name, pattern in BANNED.items():
                hits = len(pattern.findall(line))
                if hits:
                    key = (rel, name)
                    seen[key] = seen.get(key, 0) + hits
                    where.setdefault(key, []).append(f"{rel}:{number}: {line.strip()}")
    for key, count in sorted(seen.items()):
        allowed = allow.get(key, 0)
        if count > allowed:
            failures.append(
                f"{key[0]} calls {key[1]} {count}x, {allowed} allowed:\n  "
                + "\n  ".join(where[key])
            )
    for key, allowed in sorted(allow.items()):
        if seen.get(key, 0) < allowed:
            failures.append(
                f"selection_static_allow.txt allows {key[0]} {key[1]} {allowed}x, the source has "
                f"{seen.get(key, 0)}: lower the entry (and update the pillar's gap list)"
            )

    # Every NINFERNO_* variable is one the property suite sets.
    suite = (root / "tests" / "ops" / "linear" / "test_selector_properties.cpp").read_text(
        encoding="utf-8"
    )
    tried = set(re.findall(r'\{"(NINFERNO_[A-Z0-9_]+)",', suite))
    named: dict[str, str] = {}
    for top in ("src", "apps", "include"):
        for path in sorted((root / top).rglob("*")):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in ENV_LITERAL.finditer(text):
                named.setdefault(match.group(1), path.relative_to(root).as_posix())
    for name, first in sorted(named.items()):
        if name not in tried:
            failures.append(
                f"{name} ({first}) is not among the variables test_selector_properties.cpp sets "
                "(kEnvVars): add it with a non-default value"
            )

    for failure in failures:
        print(f"FAIL {failure}")
    print(
        f"selection static check: {len(files)} selection files, {len(seen)} listed exceptions in "
        f"use, {len(named)} NINFERNO_* variables: {'FAIL' if failures else 'PASS'}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
