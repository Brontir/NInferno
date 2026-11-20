#!/usr/bin/env python3
"""Re-time a route table's bands over a token range from its own board's route_tune grid (#1073).

    tools/route_table_retime.py --table routes/C.json --routes routes.csv --tokens 9-11 [--margin 0.03]

For every generic entry of the table (format, compute, epilogue, n, k) that the grid timed at a token
count inside --tokens, the band's route is replaced by the grid's fastest route at that count where the
table's own route is slower than it by more than --margin (or was not timed). Bands outside the range
are kept; adjacent bands of one route are joined. The grid must be the table's board's: route_tune
--routes-out on the same card, at the same master. Prints one line a moved entry; writes the table in
place.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

INF = 16777216


def route_at(bands, t):
    for lo, hi, r in bands:
        if lo <= t <= hi:
            return r
    return None


def rebuild(bands, moved):
    """bands with the per-token routes of `moved` ({t: route}) spliced in, equal neighbours joined."""
    points = sorted(moved)
    out = []
    for lo, hi, r in bands:
        cuts = [t for t in points if lo <= t <= hi]
        start = lo
        for t in cuts:
            if start < t:
                out.append([start, t - 1, r])
            out.append([t, t, moved[t]])
            start = t + 1
        if start <= hi:
            out.append([start, hi, r])
    joined = []
    for b in out:
        if joined and joined[-1][2] == b[2] and joined[-1][1] + 1 == b[0]:
            joined[-1][1] = b[1]
        else:
            joined.append(list(b))
    return joined


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--routes", required=True, type=Path)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--margin", type=float, default=0.03)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.tokens.split("-"))
    grid = defaultdict(dict)
    for r in csv.DictReader(a.routes.open(encoding="utf-8")):
        t = int(r["T"])
        if lo <= t <= hi:
            # route_tune --a8-candidates 1 labels an a16 shape's rows a8 (the A8 routes are among
            # its candidates); the tables hold such a problem as a16.
            compute = "a16" if r["compute"] == "a8" else r["compute"]
            key = (r["format"], compute, r["epilogue"], int(r["n"]), int(r["k"]))
            grid[(key, t)][r["route"]] = float(r["median_us"])
    doc = json.loads(a.table.read_text(encoding="utf-8"))
    moved_entries = 0
    for e in doc.get("entries", []):
        if "bands" not in e:
            continue
        key = (e["format"], e["compute"], e.get("epilogue", "store"), e["n"], e["k"])
        moved = {}
        for t in range(lo, hi + 1):
            times = grid.get((key, t))
            if not times:
                continue
            best, best_us = min(times.items(), key=lambda kv: kv[1])
            have = route_at(e["bands"], t)
            have_us = times.get(have)
            if have != best and (have_us is None or have_us > best_us * (1 + a.margin)):
                moved[t] = best
                print("%s %s %s %dx%d T%d: %s %s -> %s %.1f us" % (*key[:3], key[3], key[4], t, have,
                      "untimed" if have_us is None else "%.1f us" % have_us, best, best_us))
        if moved:
            e["bands"] = rebuild(e["bands"], moved)
            moved_entries += 1
    a.table.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("%d entries moved in %s" % (moved_entries, a.table), file=sys.stderr)


if __name__ == "__main__":
    main()
