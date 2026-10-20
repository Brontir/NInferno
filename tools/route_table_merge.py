#!/usr/bin/env python3
"""Merge route tables of one GPU class.

    tools/route_table_merge.py --into routes/C.json FILE [FILE ...] [--replace]

Adds every problem (`entries`, keyed by format, compute, epilogue, n, k) and every MoE layer
(`moe`, keyed by format, down format, experts, top-k, hidden, inter) of each FILE that the table
lacks, so tables written by separate `ninferno_route_tune` runs (one per model's recorded problems)
and by `moe_tune_to_table.py` become one. A key the table already has stays unless --replace, which
takes the later file's. Every file must be `ninferno-route-table/1` of the table's class (a chip
table takes its SM variants' files). The table is created from the first FILE when missing.
"""
import argparse
import json
import sys
from pathlib import Path


def entry_key(e):
    return (e["format"], e["compute"], e.get("epilogue", "store"), e["n"], e["k"])


def moe_key(e):
    return (e["format"], e.get("down_format", e["format"]), e["experts"], e["top_k"], e["hidden"], e["inter"])


def load(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("schema") != "ninferno-route-table/1":
        sys.exit(f"{path}: not ninferno-route-table/1")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--into", required=True, type=Path)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--replace", action="store_true")
    a = ap.parse_args()

    files = list(a.files)
    table = load(a.into) if a.into.exists() else load(files.pop(0))
    cls = table["gpu"]["class"]
    counts = {"entries": [0, 0], "moe": [0, 0]}
    for f in files:
        d = load(f)
        other = d["gpu"]["class"]
        if other != cls and (":" in cls or other.split(":")[0] != cls):
            sys.exit(f"{f}: class {other}, the table's is {cls}")
        for part, key in (("entries", entry_key), ("moe", moe_key)):
            rows = table.setdefault(part, [])
            index = {key(e): i for i, e in enumerate(rows)}
            for e in d.get(part, []):
                k = key(e)
                if k not in index:
                    index[k] = len(rows)
                    rows.append(e)
                    counts[part][0] += 1
                elif a.replace:
                    rows[index[k]] = e
                    counts[part][1] += 1
    if not table.get("moe"):
        table.pop("moe", None)
    a.into.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{a.into}: {counts['entries'][0]} problems added, {counts['entries'][1]} replaced; "
          f"{counts['moe'][0]} moe layers added, {counts['moe'][1]} replaced")


if __name__ == "__main__":
    main()
