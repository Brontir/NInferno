#!/usr/bin/env python3
"""Re-time a route table's bands over a token range from its own board's route_tune grid (#1073).

    tools/route_table_retime.py --table routes/C.json --routes routes.csv --tokens 9-11 [--margin 0.03]

For every generic entry of the table (format, compute, epilogue, n, k) that the grid timed inside
--tokens, where the table's own route is slower than the grid's fastest by more than --margin at some
count there, the range takes one route: the one whose worst regret over the range is least (the
table's own route on a tie), unless it is slower than the table's own route by more than --margin at
some count, where the entry is kept (and named on stderr). The candidates are the routes a table's a16 bands hold: A16 schedules and
ggml_mmq, as the dense tuner writes them; the W4A8 and R8 routes (route_tune --a8-candidates 1 times
them too) are never tabled (route_tune_tool.cpp, route_authorities.cpp). A GGUF format's a8 rows are
its a16 entry's; FP8's a8 entry keeps its own key. An entry whose table route the grid
did not time, or where no route was timed at every count, is kept (and named on stderr). Bands outside
the range are kept; adjacent bands of one route are joined. A moved count's evidence_us row timed the
old route, so it is dropped. The grid must be the table's board's: route_tune --routes-out on the same
card, at the same master. Prints one line a moved count; writes the table in place.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

INF = 16777216
# Routes a table's a16 bands do not hold: W4A8 and R8 are timed by --a8-candidates 1 but never tabled
# (src/ops/generic/route_tune_tool.cpp; the W4A8 heuristic and the R8 rules answer them).
NOT_TABLE_ROUTES = ("w4a8", "r8.")


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
        if lo <= t <= hi and not r["route"].startswith(NOT_TABLE_ROUTES):
            # route_tune --a8-candidates 1 labels a GGUF a16 shape's rows a8 (the A8 routes are
            # among its candidates); the tables hold such a problem as a16. FP8's a8 is its own
            # key (route_tune_tool.cpp: a8 is FP8's only), kept.
            compute = "a16" if r["compute"] == "a8" and r["format"] != "fp8" else r["compute"]
            key = (r["format"], compute, r["epilogue"], int(r["n"]), int(r["k"]))
            grid[(key, t)][r["route"]] = float(r["median_us"])
    doc = json.loads(a.table.read_text(encoding="utf-8"))
    moved_entries = 0
    for e in doc.get("entries", []):
        if "bands" not in e:
            continue
        key = (e["format"], e["compute"], e.get("epilogue", "store"), e["n"], e["k"])
        timed = [t for t in range(lo, hi + 1) if grid.get((key, t))]
        if not timed:
            continue
        # Counts no band of the entry covers are not the table's answer there: left alone.
        haves = {t: r for t in timed if (r := route_at(e["bands"], t)) is not None}
        if not haves:
            continue
        timed = sorted(haves)
        untimed = sorted({h for t, h in haves.items() if h not in grid[(key, t)]})
        if untimed:
            # The grid did not time the table's route: it may be absent for a reason (not offered
            # on the board, refused by the shape), so the entry is left for a person to look at.
            print("%s %s %s %dx%d: table route %s not timed, kept" % (*key[:3], key[3], key[4],
                  ",".join(untimed)), file=sys.stderr)
            continue

        def regret(route, t):
            times = grid[(key, t)]
            return times[route] / min(times.values()) - 1

        if all(regret(h, t) <= a.margin for t, h in haves.items()):
            continue
        # One route over the whole range (the one whose worst regret there is least), so a single
        # noisy grid cannot checker the bands (T9 A, T10 B, T11 A).
        common = set.intersection(*(set(grid[(key, t)]) for t in timed))
        if not common:
            print("%s %s %s %dx%d: no route timed at every count, kept" % (*key[:3], key[3], key[4]),
                  file=sys.stderr)
            continue
        own = set(haves.values())
        best = min(sorted(common), key=lambda r: (max(regret(r, t) for t in timed), r not in own))
        moved = {t: best for t, h in haves.items() if h != best}
        # A count where the range's route is slower than the table's own by more than the margin:
        # the move would trade it for the others (GV100, #1125: w1x16 best at T 9-13 spread over
        # n2.w1x4's 14-16, 3-6 % slower there). The entry is kept for a person to look at.
        slower = [t for t in sorted(moved)
                  if grid[(key, t)][best] > grid[(key, t)][haves[t]] * (1 + a.margin)]
        if slower:
            print("%s %s %s %dx%d: %s would slow T%s past the margin, kept"
                  % (*key[:3], key[3], key[4], best, ",".join(map(str, slower))), file=sys.stderr)
            continue
        for t in sorted(moved):
            times = grid[(key, t)]
            print("%s %s %s %dx%d T%d: %s %.1f us -> %s %.1f us" % (*key[:3], key[3], key[4], t,
                  haves[t], times[haves[t]], best, times[best]))
        if moved:
            e["bands"] = rebuild(e["bands"], moved)
            if "evidence_us" in e:
                # [T, default us, the table route's us]: a moved count's row timed the old route.
                e["evidence_us"] = [row for row in e["evidence_us"] if int(row[0]) not in moved]
            moved_entries += 1
    a.table.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("%d entries moved in %s" % (moved_entries, a.table), file=sys.stderr)


if __name__ == "__main__":
    main()
