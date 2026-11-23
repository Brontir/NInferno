#!/usr/bin/env python3
"""Re-time a route table's bands over a token range from its own board's route_tune grid (#1073).

    tools/route_table_retime.py --table routes/C.json --routes routes.csv --tokens 9-11 [--margin 0.03]
        [--evidence ROUTES SELECTED [--repeat [--fill-rule SELECTED]]]

With --fill-rule (the first grid's --selected-out), a count inside --tokens that no band of an entry
covers, where the register MMA's shape rule answered the call (authority rmma-rule), counts as that
rule route's: the same tests move it, and a moved count becomes a band of the entry (#1160: ga102's
K-quant 5120x17408 add entries had bands from T 9 only, and the rule's w1x4 lost T 4-8 by 1.2x).

For every generic entry of the table (format, compute, epilogue, n, k) that the grid timed inside
--tokens, where the table's own route is slower than the grid's fastest by more than --margin at
some count there, the range takes one route: the one whose worst regret over the range is least (the
table's own route on a tie), unless it is slower than the table's own route by more than --margin at
some count or does not lower the worst regret of the counts it moves by more than --margin; then the
entry is kept (and named on stderr). The candidates are the routes a table's a16 bands hold: A16
schedules and ggml_mmq, as the dense tuner writes them; the W4A8 and R8 routes (route_tune
--a8-candidates 1 times them too) are never tabled (route_tune_tool.cpp, route_authorities.cpp). A
GGUF format's a8 rows are its a16 entry's; FP8's a8 entry keeps its own key. An entry whose table
route the grid did not time, or where no route was timed at every count, is kept (and named on
stderr). Bands outside the range are kept; adjacent bands of one route are joined. A moved count's
evidence_us row timed the old route, so it is dropped; with --evidence ROUTES SELECTED (a second
grid of the same card, its route_tune --routes-out and --selected-out) it is written again from that
grid as [T, default us, shipped us], the default being --selected-out's `ranker` column (route_tune's
default answer) where it names a table route; where it names W4A8 or R8, an A8 route the table stands
aside for (route_authorities.cpp: a call that may take it never reads the measured entry, #1154),
the default is that grid's fastest table route, what an A16 call could run. An entry that grid cannot give a row for is kept (named on stderr), as is one
where that grid does not repeat the move (the range's route slower than the old one, or than that
grid's default answer, by more than --margin at a moved count). A count the grid did not time
between two it did takes the range's route too where its band's route is one the grid beat (#1149: a
band joined from T 16 and T 24 across n2's 16-token edge, T 17-23 untimed), unless a round edge of
the old or the new route lies between those two; a band route the grid never timed in the range is
left alone. With --repeat (needs --evidence) a move stands only where the second grid repeats it; an
entry tuned on another board class (tuned_on) moves but gets no rows of this board. The grid must be
the table's board's: route_tune --routes-out on the same card, at the same master. Prints one line a
moved count; writes the table in place.
"""
import argparse
import csv
import re
import json
import sys
from collections import defaultdict
from pathlib import Path

INF = 16777216
# Routes a table's a16 bands do not hold: W4A8 and R8 are timed by --a8-candidates 1 but never tabled
# (src/ops/generic/route_tune_tool.cpp; the W4A8 heuristic and the R8 rules answer them).
NOT_TABLE_ROUTES = ("w4a8", "r8.")


def board_keys(gpu):
    """A table's own board keys, as tests/test_route_table_evidence.py reads them: its class, and
    class:NNsm from the registry row of its PCI id (src/core/data/gpu_facts.json) or its own
    board facts."""
    keys = {gpu.get("class")}
    sms = None
    if "pci_id" in gpu:
        facts = Path(__file__).resolve().parent.parent / "src/core/data/gpu_facts.json"
        rows = json.loads(facts.read_text(encoding="utf-8"))["gpus"]
        sms = {int(r["id"], 16): r["sms"] for r in rows}.get(int(str(gpu["pci_id"]), 16))
    sms = sms or (gpu.get("board") or {}).get("sms")
    if sms and ":" not in str(gpu.get("class")):
        keys.add("%s:%dsm" % (gpu["class"], sms))
    return keys


# The first count of the W4A8 band (route_authorities.cpp kPrefillBandMinTokens).
A8_WIDE_MIN_T = 17


def a8_wide(times, t):
    """A count where the call may take W4A8 (route_tune times it only where the format and K admit
    it): under an A8 policy route_table_entry reads only the predicted band there (a8_wide_call),
    and under A16 only ggml_mmq (int8 activations) is refused. A measured entry there answers A16
    calls, so its candidates are the A16 routes (#1149 review)."""
    return t >= A8_WIDE_MIN_T and any(r.startswith("w4a8") for r in times)


def a16_routes(times, t):
    """The routes of one count an A16 call of the table can run."""
    wide = a8_wide(times, t)
    return {r: us for r, us in times.items()
            if not r.startswith(NOT_TABLE_ROUTES) and not (wide and r == "ggml_mmq")}


def default_route(times, ranker, t):
    """The default an a16 band is held to at one count: the ranker where it names a route an A16
    call can run there; where it names an A8 route the table never competes with (W4A8, R8, or
    ggml_mmq at a W4A8 count), the fastest route an A16 call can run there. No ranker, or one
    that names a table route the grid did not time, gives None: the caller counts a gap."""
    if ranker is None:
        return None
    table = a16_routes(times, t)
    if ranker in table:
        return ranker
    if ranker.startswith(NOT_TABLE_ROUTES) or (ranker == "ggml_mmq" and a8_wide(times, t)):
        return min(table, key=table.get) if table else None
    return None


def round_width(route):
    """The token count a route's round count steps at (ceil(T / width)): eight columns a tile for the
    register MMA's nN (rmma.m1.nN.*); the token tile its name carries for the staged
    (staged.ROWSxTOKENS), tiled, W4A8, W4A4, I8 and NVFP4 tiles (tiled.MxN, w4a8.MxN, w4a4.MxN, i8.MxN,
    nvfp4t.cC.MxN: M tokens, generic_routes.cpp route_name's block_m; generic_routes.cpp
    route_name); eight, the narrowest column step, for any other route (ggml_mmq's column tiles
    are multiples of eight)."""
    m = re.search(r"(?:^|\.)n(\d+)(?:\.|$)", route)
    if m:
        return 8 * int(m.group(1))
    m = re.match(r"staged\.\d+x(\d+)", route)
    if m:
        return int(m.group(1))
    m = re.match(r"(?:tiled|w4a8|w4a8ws|w4a4|w4a4ws|i8|nvfp4t\.c\d+)\.(\d+)x\d+", route)
    if m:
        return int(m.group(1))
    return 8


def crosses_an_edge(route, lo, hi):
    """Whether a round edge of `route` (a multiple of its width) lies strictly between two timed
    counts lo < hi: the untimed counts between them are of two round counts."""
    w = round_width(route)
    return (lo // w + 1) * w < hi


# --fill-rule: the rule authorities whose answer a band may replace where the entry has none. The
# register MMA's shape rule only: its route is one a band holds, and a band there is read before
# it. A count the vec rule answers stays its own: the call takes ggml_vec there whatever the
# entry's bands say (RTX 3090 grid, #1160: Q4_0 4096x2688 at T 2-3 ships vec-rule beside a band),
# so a band written there would never be read.
FILL_AUTHORITIES = {"rmma-rule"}


def route_at(bands, t):
    for lo, hi, r in bands:
        if lo <= t <= hi:
            return r
    return None


def rebuild(bands, moved):
    """bands with the per-token routes of `moved` ({t: route}) spliced in, equal neighbours joined.
    A moved count no band covers (--fill-rule) becomes a band of its own."""
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
    out += [[t, t, moved[t]] for t in points if route_at(bands, t) is None]
    out.sort(key=lambda b: b[0])
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
    every = defaultdict(dict)
    for r in csv.DictReader(path.open(encoding="utf-8")):
        t = int(r["T"])
        if lo <= t <= hi:
            every[(row_key(r), t)][r["route"]] = float(r["median_us"])
    if every_route:
        return every
    # The candidates: what an A16 call can run at the count (a16_routes).
    grid = defaultdict(dict)
    for (key, t), times in every.items():
        if routes := a16_routes(times, t):
            grid[(key, t)] = routes
    return grid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--routes", required=True, type=Path)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--evidence", nargs=2, type=Path, metavar=("ROUTES", "SELECTED"))
    ap.add_argument("--fill-rule", type=Path, metavar="SELECTED",
                    help="the first grid's route_tune --selected-out: a count no band covers, where "
                         "the register MMA's rule answered (rmma-rule), is the rule route's to lose")
    ap.add_argument("--repeat", action="store_true",
                    help="with --evidence: keep a move only where the second grid repeats it: at "
                         "every moved or filled count it timed, the range's route is within --margin "
                         "of that grid's fastest table route or faster than the old route by more "
                         "than --margin")
    a = ap.parse_args()
    if a.repeat and not a.evidence:
        ap.error("--repeat needs --evidence (the second grid it checks the moves on)")
    if a.fill_rule and not a.repeat:
        ap.error("--fill-rule needs --evidence and --repeat (a filled count stands on two grids)")
    lo, hi = (int(x) for x in a.tokens.split("-"))
    grid = load_grid(a.routes, lo, hi)
    # --fill-rule: what the first grid's call shipped where no band covers the count, when one of
    # FILL_AUTHORITIES answered it with a route a band can hold.
    shipped = {}
    if a.fill_rule:
        for r in csv.DictReader(a.fill_rule.open(encoding="utf-8")):
            if lo <= int(r["T"]) <= hi and r["authority"] in FILL_AUTHORITIES:
                shipped[(row_key(r), int(r["T"]))] = r["selected"]
    grid_all = load_grid(a.routes, lo, hi, every_route=True)
    # The evidence grid: every route it timed (the ranker may answer a route no table holds).
    ev_grid, ranker = {}, {}
    if a.evidence:
        ev_grid = load_grid(a.evidence[0], lo, hi, every_route=True)
        for r in csv.DictReader(a.evidence[1].open(encoding="utf-8")):
            if lo <= int(r["T"]) <= hi:
                ranker[(row_key(r), int(r["T"]))] = r["ranker"]
    doc = json.loads(a.table.read_text(encoding="utf-8"))
    board = doc.get("gpu", {}).get("class")
    own_keys = board_keys(doc.get("gpu", {}))
    moved_entries = 0
    for e in doc.get("entries", []):
        if "bands" not in e:
            continue
        key = (e["format"], e["compute"], e.get("epilogue", "store"), e["n"], e["k"])
        timed = [t for t in range(lo, hi + 1) if grid.get((key, t))]
        if not timed:
            continue
        # Counts no band of the entry covers are not the table's answer there: left alone, unless
        # --fill-rule names the rule route the call shipped there (then that route is the count's
        # old route, and a move gives the entry a band there).
        haves = {t: r for t in timed if (r := route_at(e["bands"], t)) is not None}
        for t in timed:
            s = shipped.get((key, t))
            if t not in haves and s is not None and s in grid[(key, t)]:
                haves[t] = s
        if not haves:
            continue
        timed = sorted(haves)
        # A band route an A16 call cannot run where the call may take W4A8 (ggml_mmq at an a8_wide
        # count) is a dead entry: no policy reads it (a16_routes), so it is always replaced, timed
        # or not. Dead is read off the grid (it timed W4A8): a grid run without --a8-candidates 1
        # times no W4A8 and finds none.
        dead = {t for t, h in haves.items()
                if h not in grid[(key, t)] and h == "ggml_mmq" and a8_wide(grid_all[(key, t)], t)}
        untimed = sorted({h for t, h in haves.items() if h not in grid[(key, t)] and t not in dead})
        if untimed:
            # The grid did not time the table's route: it may be absent for a reason (not offered
            # on the board, refused by the shape), so the entry is left for a person to look at.
            print("%s %s %s %dx%d: table route %s not timed, kept" % (*key[:3], key[3], key[4],
                  ",".join(untimed)), file=sys.stderr)
            continue

        def regret(route, t):
            times = grid[(key, t)]
            if route not in times:
                return float("inf")  # a dead entry (above)
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
        slower = [t for t in sorted(moved) if t not in dead
                  and grid[(key, t)][best] > grid[(key, t)][haves[t]] * (1 + a.margin)]
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
        # Dead counts always move; the live ones must earn it (a dead count's infinite regret would
        # wave every other move of the range through).
        live = [t for t in moved if t not in dead]
        old_worst = max((regret(haves[t], t) for t in live), default=float("inf"))
        new_worst = max((regret(best, t) for t in live), default=0.0)
        if live and new_worst >= old_worst - a.margin:
            print("%s %s %s %dx%d: %s would not lower the worst regret (%.1f %% -> %.1f %%), kept"
                  % (*key[:3], key[3], key[4], best, 100 * old_worst, 100 * new_worst),
                  file=sys.stderr)
            continue
        # Untimed counts between the timed ones: a band is one route over its counts, so where its
        # route is one the grid beat at a timed count, the range's route takes it there too (else
        # T 17 n4, T 19 n1, T 20 n4 from a grid timed at 17/18/20/22/23). A route the grid never
        # timed in the range is a band it did not see: left alone.
        # A gap that holds a round edge of the old or the new route is not filled: its counts are
        # of two round counts, and the timed ones on either side speak for one each (#1149's own
        # mistake: T 17-32, n4's one pass, filled from T 16 and T 33 across its 32 edge).
        beaten = {haves[t] for t in moved}
        filled = {}
        for u in range(timed[0] + 1, timed[-1]):
            h = route_at(e["bands"], u)
            if u not in haves and h is not None and h != best and h in beaten:
                below = max(t for t in timed if t < u)
                above = min(t for t in timed if t > u)
                if not (crosses_an_edge(h, below, above) or crosses_an_edge(best, below, above)):
                    filled[u] = best
        new_rows = {}
        if moved and a.evidence:
            # A filled count's row where the evidence grid timed it (a second grid may time the
            # counts the first left out); only a timed moved count's row is required.
            for t in sorted({**moved, **filled}):
                times = ev_grid.get((key, t), {})
                rk = default_route(times, ranker.get((key, t)), t)
                if rk in times and best in times:
                    new_rows[t] = [t, round(times[rk], 2), round(times[best], 2)]
            # A count the second grid cannot confirm (no ranker, a ranker it did not time, new or
            # old route row there) is a gap: the move would stand on the first grid alone. A dead
            # count's old route runs nowhere (#1157 review): only its new and default rows count.
            gaps = [t for t in sorted(moved)
                    if t not in new_rows
                    or (t not in dead and haves[t] not in ev_grid.get((key, t), {}))]
            if gaps:
                print("%s %s %s %dx%d: the evidence grid has no ranker / %s / old route row at T%s,"
                      " kept" % (*key[:3], key[3], key[4], best, ",".join(map(str, gaps))),
                      file=sys.stderr)
                continue
            # A move the second grid reverses (the range's route slower than the old one there
            # past the margin) is noise in the first grid: the entry is kept as it was.
            # ...or puts it behind the call's default answer past the margin (default_route: an
            # A8 ranker is no rival, the table stands aside where it may run): a table entry slower
            # than what the call runs without it overrides nothing worth keeping.
            # A filled count the second grid timed is held to the same two tests: the move writes
            # the range's route there too (#1157 review). Its old route is the band's; a dead one
            # (ggml_mmq where W4A8 may run) or one the second grid did not time is no rival. A
            # moved count is dead by the first grid (`dead`, above); a filled one, which the first
            # grid did not time, by the second.
            def reversed_by_evidence(t, old):
                times = ev_grid.get((key, t), {})
                if best not in times:
                    return False
                if old in times and times[best] > times[old] * (1 + a.margin):
                    return True
                rk = default_route(times, ranker.get((key, t)), t)
                return rk in times and times[best] > times[rk] * (1 + a.margin)

            def filled_old(u):
                h = route_at(e["bands"], u)
                return None if h == "ggml_mmq" and a8_wide(ev_grid.get((key, u), {}), u) else h

            reversed_at = sorted(
                [t for t in moved if reversed_by_evidence(t, None if t in dead else haves[t])]
                + [u for u in filled if reversed_by_evidence(u, filled_old(u))])
            if reversed_at:
                print("%s %s %s %dx%d: the evidence grid does not repeat %s at T%s, kept"
                      % (*key[:3], key[3], key[4], best, ",".join(map(str, reversed_at))),
                      file=sys.stderr)
                continue
            # An entry whose evidence is borrowed (tuned_on another board's class) keeps that
            # borrow: this board's rows beside it would break the borrow's copy of the donor
            # (test_route_table_evidence, rule c). The move stands; its rows are not written.
            # A borrow names another table's board key; free text (a model, a tool) is no key.
            tuned = str(e.get("tuned_on", ""))
            if tuned and tuned not in own_keys and re.fullmatch(r"[a-z0-9-]+(:\d+sm)?", tuned):
                new_rows = {}
            if a.repeat:
                # An A/A grid that does not repeat a count's move (a near-tie, noise in the first
                # grid) keeps the entry as it was (#1142's discipline). A move repeats where the
                # range's route is within the margin of the second grid's fastest, or still beats
                # the old route by more than the margin there (a large gain to a route the second
                # grid puts a little behind another).
                def repeats(t, old):
                    times = a16_routes(ev_grid.get((key, t), {}), t)
                    if best not in times:
                        return True  # not timed there: the first grid's word stands
                    if times[best] <= min(times.values()) * (1 + a.margin):
                        return True
                    return old in times and times[best] * (1 + a.margin) < times[old]

                olds = {**{t: haves[t] for t in moved},
                        **{u: route_at(e["bands"], u) for u in filled}}
                unrepeated = [t for t in sorted(olds) if not repeats(t, olds[t])]
                if unrepeated:
                    print("%s %s %s %dx%d: the second grid does not repeat %s at T%s, kept"
                          % (*key[:3], key[3], key[4], best, ",".join(map(str, unrepeated))),
                          file=sys.stderr)
                    continue
        for t in sorted(moved):
            times = grid_all[(key, t)]
            old_us = ("%.1f us" % times[haves[t]]) if haves[t] in times else "untimed"
            print("%s %s %s %dx%d T%d: %s %s%s -> %s %.1f us" % (*key[:3], key[3], key[4], t,
                  haves[t], old_us, " (dead: no policy reads it)" if t in dead else "",
                  best, times[best]))
        for u in sorted(filled):
            print("%s %s %s %dx%d T%d: %s -> %s (untimed, between timed counts)"
                  % (*key[:3], key[3], key[4], u, route_at(e["bands"], u), best))
        if moved:
            e["bands"] = rebuild(e["bands"], {**moved, **filled})
            if "evidence_us" in e or new_rows:
                # [T, default us, the table route's us]: a moved count's row timed the old route; the
                # evidence grid's row takes its place.
                rows = {int(row[0]): row for row in e.get("evidence_us", [])
                        if int(row[0]) not in moved and int(row[0]) not in filled}
                rows.update(new_rows)
                e["evidence_us"] = [rows[t] for t in sorted(rows)]
            moved_entries += 1
    a.table.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("%d entries moved in %s" % (moved_entries, a.table), file=sys.stderr)


if __name__ == "__main__":
    main()
