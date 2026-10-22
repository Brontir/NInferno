#!/usr/bin/env python3
"""Source scan of the selection rules (docs/maintainer/deterministic-decisions.md, principle 8).

The files that hold decision rules (an explicit list of globs, selection_files.txt: the generic
selector, tuners, route tables, per-op plans, dispatchers and policies) may not read the
environment, a clock, a timer, free memory or a file. Each existing exception is listed in
selection_static_allow.txt with its count; the scan fails on a call the list does not cover and on
an entry whose count is now lower (tighten it). An environment read whose name is not a string
literal is flagged on its own (env-nonliteral): a built-up name escapes the variable check below.

It also checks that every NINFERNO_* variable named in src/, apps/ or include/ is one the selector
property suite sets (kEnvVars, tests/ops/linear/test_selector_properties.cpp), so a new knob is
tried against the choices.

Usage: selection_static_check.py [repository root]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SUFFIXES = {".cpp", ".h", ".cu", ".cuh"}

# Name -> pattern of what a rule may not call.
BANNED = {
    "getenv": re.compile(r"\bgetenv\s*\("),
    "sys::env": re.compile(r"\bsys::env\s*\("),
    # An environment read whose name argument is not a string literal (a built-up name).
    "env-nonliteral": re.compile(
        r"\b(?:getenv|secure_getenv|sys::env)\s*\(\s*(?!\")|\b_dupenv_s\s*\([^,]*,[^,]*,\s*(?!\")"
    ),
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


RAW_PREFIXES = ("R", "u8R", "uR", "UR", "LR")


def _token_start(text: str, i: int) -> int:
    """Where the identifier or number that ends just before `i` starts (`'` and `.` included, so a
    number with digit separators is one token)."""
    j = i
    while j > 0 and (text[j - 1].isalnum() or text[j - 1] in "_'."):
        j -= 1
    return j


def strip_comments(text: str) -> str:
    """Drops // and /* */ comments, keeping string literals (raw ones too) and line numbers."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        prefix = text[_token_start(text, i) : i] if c in "\"'" else ""
        if c == '"' and prefix.endswith("R") and prefix in RAW_PREFIXES:
            # R"delim( ... )delim": nothing inside is a comment or an escape.
            open_paren = text.find("(", i)
            delim = text[i + 1 : open_paren] if open_paren >= 0 else ""
            close = text.find(")" + delim + '"', open_paren + 1) if open_paren >= 0 else -1
            j = n if close < 0 else close + len(delim) + 2
            out.append(text[i:j])
            i = j
        elif c == "'" and prefix[:1].isdigit():
            # A digit separator (1'000'000), not a character literal.
            out.append(c)
            i += 1
        elif c == '"' or c == "'":
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


def self_test() -> None:
    """Planted text: comments go; literals (raw ones too), digit separators and line breaks stay."""
    cases = [
        ("a // getenv(\n", "a \n"),
        ("int x = 1'000'000; // getenv(x)\n", "int x = 1'000'000; \n"),
        ('auto s = R"x(// kept )" still)x"; getenv(y)', 'auto s = R"x(// kept )" still)x"; getenv(y)'),
        ('auto s = u8R"(/* kept */)"; /* gone\n */ b', 'auto s = u8R"(/* kept */)"; \n b'),
        ("char c = '/'; // gone", "char c = '/'; "),
        ('auto t = L"//x"; z', 'auto t = L"//x"; z'),
    ]
    for given, want in cases:
        got = strip_comments(given)
        if got != want:
            raise SystemExit(f"strip_comments self-test: {given!r} -> {got!r}, want {want!r}")


def selection_files(root: Path) -> tuple[list[Path], list[str]]:
    """The files selection_files.txt names, and the globs that match none."""
    files: set[Path] = set()
    empty: list[str] = []
    listing = root / "tools" / "audit" / "selection_files.txt"
    for raw in listing.read_text(encoding="utf-8").splitlines():
        pattern = raw.split("#", 1)[0].strip()
        if not pattern:
            continue
        matched = [p for p in root.glob(pattern) if p.is_file() and p.suffix in SUFFIXES]
        if not matched:
            empty.append(pattern)
        files.update(matched)
    return sorted(files), empty


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
    self_test()
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2])
    allow = read_allow(root / "tools" / "audit" / "selection_static_allow.txt")
    failures: list[str] = []
    seen: dict[tuple[str, str], int] = {}
    where: dict[tuple[str, str], list[str]] = {}
    files, empty = selection_files(root)
    for pattern in empty:
        failures.append(f"selection_files.txt: {pattern} matches no file (renamed? fix the list)")
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
