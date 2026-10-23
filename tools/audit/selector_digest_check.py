#!/usr/bin/env python3
"""Self-consistency of the selector digests (tests/ops/linear/selector_digest/<set>.txt).

For every route set (full, base, sm70, sm75), the digest lines of each golden board must be the
FNV-1a-64 of its golden file (selector_golden/<set>/<board>.txt), whole and per format, as
test_selector_properties writes them; every digest file must name the same boards. The property
test checks the digests against the selector itself, but only for the route set its build
compiles; this check covers the sets no CPU job builds (sm70, sm75: their stands regenerate them)
against their golden files.

Usage: selector_digest_check.py [repository root]
"""

from __future__ import annotations

import sys
from pathlib import Path

SETS = ("full", "base", "sm70", "sm75")


def fnv1a(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for byte in data:
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def golden_lines(board: str, data: bytes) -> list[str]:
    """The digest lines of one golden file: whole text, then each format in file order."""
    newline = b"\n"  # outside the f-strings: Python 3.11 allows no backslash inside them
    lines = [f"{board} * {fnv1a(data):016x} {data.count(newline)}"]
    parts: dict[str, bytes] = {}
    for line in data.split(b"\n")[1:]:
        if line:
            fmt = line.split(b" ", 1)[0].decode()
            parts[fmt] = parts.get(fmt, b"") + line + b"\n"
    for fmt, text in parts.items():
        lines.append(f"{board} {fmt} {fnv1a(text):016x} {text.count(newline)}")
    return lines


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2])
    base = root / "tests" / "ops" / "linear"
    failures: list[str] = []
    board_sets: dict[str, set[str]] = {}
    checked = 0
    for route_set in SETS:
        digest = base / "selector_digest" / f"{route_set}.txt"
        if not digest.is_file():
            failures.append(f"{digest}: missing")
            continue
        by_board: dict[str, list[str]] = {}
        for line in digest.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                by_board.setdefault(line.split(" ", 1)[0], []).append(line)
        board_sets[route_set] = set(by_board)
        for golden in sorted((base / "selector_golden" / route_set).glob("*.txt")):
            want = golden_lines(golden.stem, golden.read_bytes())
            have = by_board.get(golden.stem, [])
            # The digest orders formats as the selector lists them; the golden file's order is
            # the same, so compare as sets of lines.
            if sorted(want) != sorted(have):
                failures.append(
                    f"{route_set}/{golden.stem}: the digest lines are not the golden file's "
                    f"hashes (regenerate both with --update in a {route_set} tree)"
                )
            checked += 1
    names = set().union(*board_sets.values()) if board_sets else set()
    for route_set, boards in board_sets.items():
        if boards != names:
            failures.append(
                f"{route_set}.txt lacks boards {sorted(names - boards)} the other sets have"
            )
    for failure in failures:
        print(f"FAIL {failure}")
    print(
        f"selector digest check: {checked} golden files over {len(board_sets)} route sets, "
        f"{len(names)} boards: {'FAIL' if failures else 'PASS'}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
