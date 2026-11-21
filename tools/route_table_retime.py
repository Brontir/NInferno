#!/usr/bin/env python3
"""Re-time a route table's bands over a token range from its own board's route_tune grid (#1073).

    tools/route_table_retime.py --table routes/C.json --routes routes.csv --tokens 9-11 [--margin 0.03]
        [--evidence ROUTES SELECTED]

For every generic entry of the table (format, compute, epilogue, n, k) that the grid timed inside
--tokens, where the table's own route is slower than the grid's fastest by more than --margin at
some count there, the range takes one route: the one whose worst regret over the range is least (the
table's own route on a tie), unless it is slower than the table's own route by more than --margin at
some count or does not lower the worst regret of the counts it moves by more than --margin; then
the entry is kept (and named on stderr). The candidates are the routes a table's
a16 bands hold: A16 schedules and ggml_mmq, as the dense tuner writes them; the W4A8 and R8 routes
(route_tune --a8-candidates 1 times them too) are never tabled (route_tune_tool.cpp,
route_authorities.cpp). A GGUF format's a8 rows are its a16 entry's; FP8's a8 entry keeps its own
key. An entry whose table route the grid did not time, or where no route was timed at every count,
is kept (and named on stderr). Bands outside the range are kept; adjacent bands of one route are
joined. A moved count's evidence_us row timed the old route, so it is dropped; with --evidence
ROUTES SELECTED (a second grid of the same card, its route_tune --routes-out and --selected-out) it
is written again from that grid as [T, ranker us, shipped us], the ranker being --selected-out's
`ranker` column (route_tune's default answer), and an entry that grid cannot give a row for is kept
(named on stderr), as is one where that grid does not repeat the move (the range's route slower than
the old one, or than that grid's default answer, by more than --margin at a moved count). A count a
band covers but the grid did not time keeps its old route. The grid must be the table's board's: route_tune --routes-out on the same card,
at the same master. Prints one line a moved count; writes the table in place.
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


def row_key(r):
    # route_tune --a8-candidates 1 labels a GGUF a16 shape's rows a8 (the A8 routes are among its
    # candidates); the tables hold such a problem as a16. FP8's a8 is its own key
    # (route_tune_tool.cpp: a8 is FP8's only), kept.
    compute = "a16" if r["compute"] == "a8" and r["format"] != "fp8" else r["compute"]
    return (r["format"], compute, r["epilogue"], int(r["n"]), int(r["k"]))


def load_grid(path, lo, hi, every_route=False):
    grid = defaultdict(dict)
    for r in csv.DictReader(path.open(encoding="utf-8")):
        t = int(r["T"])
        if lo <= t <= hi and (every_route or not r["route"].startswith(NOT_TABLE_ROUTES)):
            grid[(row_key(r), t)][r["route"]] = float(r["median_us"])
    return grid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--routes", required=True, type=Path)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--evidence", nargs=2, type=Path, metavar=("ROUTES", "SELECTED"))
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.tokens.split("-"))
    grid = load_grid(a.routes, lo, hi)
    # The evidence grid: every route it timed (the ranker may answer a route no table holds).
    ev_grid, ranker = {}, {}
    if a.evidence:
        ev_grid = load_grid(a.evidence[0], lo, hi, every_route=True)
        for r in csv.DictReader(a.evidence[1].open(encoding="utf-8")):
            if lo <= int(r["T"]) <= hi:
                ranker[(row_key(r), int(r["T"]))] = r["ranker"]
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
        # The move must lower the worst regret of the counts it moves by more than the margin: a
        # least-worst route no better than the bands as they are (T 17-20 A the fastest, T 21-23 B
        # 4 % behind and A 7 %) would trade one count's loss for another's. Counts it leaves alone
        # are the same on both sides and take no part.
        if not moved:
            continue  # the table's own route is the range's: nothing to move
        old_worst = max(regret(haves[t], t) for t in moved)
        new_worst = max(regret(best, t) for t in moved)
        if new_worst >= old_worst - a.margin:
            print("%s %s %s %dx%d: %s would not lower the worst regret (%.1f %% -> %.1f %%), kept"
                  % (*key[:3], key[3], key[4], best, 100 * old_worst, 100 * new_worst),
                  file=sys.stderr)
            continue
        new_rows = {}
        if moved and a.evidence:
            for t in sorted(moved):
                times = ev_grid.get((key, t), {})
                rk = ranker.get((key, t))
                if rk in times and best in times:
                    new_rows[t] = [t, round(times[rk], 2), round(times[best], 2)]
            # A count the second grid cannot confirm (no ranker, new or old route row there) is a
            # gap: the move would stand on the first grid alone.
            gaps = [t for t in sorted(moved)
                    if t not in new_rows or haves[t] not in ev_grid.get((key, t), {})]
            if gaps:
                print("%s %s %s %dx%d: the evidence grid has no ranker / %s / old route row at T%s,"
                      " kept" % (*key[:3], key[3], key[4], best, ",".join(map(str, gaps))),
                      file=sys.stderr)
                continue
            # A move the second grid reverses (the range's route slower than the old one there
            # past the margin) is noise in the first grid: the entry is kept as it was.
            # ...or puts it behind the call's default answer (the ranker) past the margin: a table
            # entry slower than what the call runs without it overrides nothing worth keeping.
            reversed_at = [t for t in sorted(moved)
                           if ev_grid[(key, t)][best] > ev_grid[(key, t)][haves[t]] * (1 + a.margin)
                           or ev_grid[(key, t)][best]
                           > ev_grid[(key, t)][ranker[(key, t)]] * (1 + a.margin)]
            if reversed_at:
                print("%s %s %s %dx%d: the evidence grid does not repeat %s at T%s, kept"
                      % (*key[:3], key[3], key[4], best, ",".join(map(str, reversed_at))),
                      file=sys.stderr)
                continue
        for t in sorted(moved):
            times = grid[(key, t)]
            print("%s %s %s %dx%d T%d: %s %.1f us -> %s %.1f us" % (*key[:3], key[3], key[4], t,
                  haves[t], times[haves[t]], best, times[best]))
        if moved:
            e["bands"] = rebuild(e["bands"], moved)
            if "evidence_us" in e or new_rows:
                # [T, default us, the table route's us]: a moved count's row timed the old route; the
                # evidence grid's row takes its place.
                rows = {int(row[0]): row for row in e.get("evidence_us", []) if int(row[0]) not in moved}
                rows.update(new_rows)
                e["evidence_us"] = [rows[t] for t in sorted(rows)]
            moved_entries += 1
    a.table.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("%d entries moved in %s" % (moved_entries, a.table), file=sys.stderr)


if __name__ == "__main__":
    main()
