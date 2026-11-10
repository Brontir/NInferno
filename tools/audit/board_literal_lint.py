#!/usr/bin/env python3
"""Fail on a hand-written board identity: a `Board{12, 0, 170, ...}` literal.

A board is named by its PCI id and read from the registry (core/board_registry.h, generated from
core/data/gpu_facts.json): `board_registry::board(0x2B85)`. A `Board{...}` whose first initializer
is a number is a second spelling of a board's compute capability, SMs, L2 and bandwidth, the class
of bug behind #502, #507, #556 and #594 (roadmap #612, step 9).

Not flagged: `Board{cc / 10, ...}` style conversions from fields of another record (an expression,
not a number), and unrelated structs of the same name (`struct Board {`).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("src", "tests", "tools")
SUFFIXES = {".h", ".hpp", ".cpp", ".cc", ".cu", ".cuh"}

# path -> reason. Keep it short; every entry is a reviewed exception.
ALLOW = {
    "src/core/measured_boards.h":
        "pins: static_asserts that each registry row equals the literal it replaced",
    "tests/test_device.cpp":
        "hypothetical boards (cc 13.0, synthetic SM counts) probing board_distance",
    "tests/test_gate_atlas.cpp":
        "hypothetical boards with a quantity zeroed or altered, probing the lookup rules",
}

# `Board{` / `Board name{` / `Board name = {` followed by a number. `\w` before: `FactsBoard{` etc.
PATTERN = re.compile(r"(?<![\w])Board\s*(?:\w+\s*)?=?\s*\{\s*[-+]?\.?\d")


def strip(line, in_block):
    """The code of a line: comments removed. Returns (code, still_in_block_comment)."""
    out, i = [], 0
    while i < len(line):
        if in_block:
            j = line.find("*/", i)
            if j < 0:
                return "".join(out), True
            i, in_block = j + 2, False
        elif line.startswith("/*", i):
            in_block = True
            i += 2
        elif line.startswith("//", i):
            break
        else:
            out.append(line[i])
            i += 1
    return "".join(out), in_block


def violations(text):
    """[(line number, line)] of the board literals in `text`."""
    found, in_block = [], False
    for n, line in enumerate(text.splitlines(), 1):
        code, in_block = strip(line, in_block)
        if PATTERN.search(code):
            found.append((n, line.strip()))
    return found


def scan(root, allow=ALLOW):
    bad = []
    for d in SCAN_DIRS:
        for p in sorted((root / d).rglob("*")):
            if p.suffix not in SUFFIXES or not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            if rel in allow:
                continue
            for n, line in violations(p.read_text(encoding="utf-8", errors="replace")):
                bad.append((rel, n, line))
    return bad


def main():
    bad = scan(ROOT)
    for rel, n, line in bad:
        print(f"{rel}:{n}: board literal outside the registry: {line}")
    if bad:
        print("Name the board by PCI id: board_registry::board(0x....) (core/board_registry.h).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
