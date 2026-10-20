#!/usr/bin/env python3
"""Fail when a relative Markdown link names a file that does not exist, or an #anchor that names
no heading of the target file (GitHub slugs). Links in code blocks and code spans are not links."""
from functools import lru_cache
import re
import sys
from pathlib import Path
from urllib.parse import unquote

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
FENCE = re.compile(r"^ {0,3}(```|~~~)")
LINK = re.compile(r"\]\(([^)\s]*)\)")


def slug(heading: str) -> str:
    s = heading.strip().lower()
    s = re.sub(r"`", "", s)
    s = re.sub(r"[^\w\- 一-鿿]", "", s)
    return s.replace(" ", "-")


def prose(path: Path, spans: bool = False) -> list[str]:
    """The file's lines outside fenced code blocks; code spans removed unless `spans`."""
    out, fence = [], None
    for line in path.read_text(errors="replace", encoding="utf-8").splitlines():
        m = FENCE.match(line)
        if m and (fence is None or m.group(1) == fence):
            fence = None if fence else m.group(1)
            continue
        if fence is None:
            out.append(line if spans else re.sub(r"`[^`]*`", "", line))
    return out


@lru_cache(maxsize=None)
def anchors(path: Path) -> frozenset:
    out, seen = set(), {}
    for line in prose(path, spans=True):
        m = re.match(r"#{1,6}\s+(.*)", line)
        if m:
            s = slug(m.group(1))
            n = seen.get(s, 0)
            out.add(s if n == 0 else f"{s}-{n}")
            seen[s] = n + 1
    return frozenset(out)


bad = 0
for md in sorted(root.rglob("*.md")):
    if any(p in md.parts for p in ("third_party", "corpora", ".git", "build")):
        continue
    for line in prose(md):
        for m in LINK.finditer(line):
            link = m.group(1)
            if not link or re.match(r"[a-z][a-z0-9+.-]*:", link):  # http:, mailto:, ...
                continue
            file, _, anchor = link.partition("#")
            target = (md.parent / unquote(file)).resolve() if file else md
            if not target.exists():
                problem = "no such file"
            elif anchor and target.suffix == ".md" and unquote(anchor) not in anchors(target):
                problem = "no such heading"
            else:
                continue
            bad += 1
            print(f"{md.relative_to(root)} -> {link}: {problem}")
print(f"broken links: {bad}")
sys.exit(1 if bad else 0)
