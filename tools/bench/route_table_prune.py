#!/usr/bin/env python3
"""Exit tooling for measured route overrides (#614, PR-F3): prune what the ranker has caught up with.

A route table entry (routes/<class>.json) or a dense tuner problem (routes/tuner/dense_tune.json)
is a measured override of the ranker (the cost model where it prices, else the rules). When the
regret campaign shows the ranker's own pick within its resolution of the override on that board,
at every width the override covers, the override earns nothing and is deleted, as a diff a
reviewer can read.

Input: the selection_regret.py report (ninferno-selection-regret-report/1) of one board, whose
cases carry `authority`, `evidence`, `t`, `ranker` (the route the ranker alone picks) and
`ranker_vs_selected` (its median time over the selected route's, minus one; it must be a finite
number). Only cases whose authority is `route-table` or `dense-tune` with `measured` evidence speak
for an entry. The campaign runs the speed profile, so a case measures a prefill override only if it
says so (`profile: "prefill"`, which no run writes yet): an entry with `prefill`, `prefill_bands`
or `prefill_tuned_on` (a dense problem with a prefill route) is kept until prefill-profile cases
exist for it, whatever the speed cases say.

An entry is pruned only when every segment of every one of its `bands` has at least one case in
its T range and every case there has `ranker_vs_selected` <= the resolution. A band is cut into
segments at the regime edges (decode T<=4, verify 5..16) and at the board's ridge T* (the report's
own, `by_board_regime_authority`; without one a wide band is a single segment), so a band that
spans a regime needs a case in each part. A case without a ranker time, or an incomplete one, keeps
the entry; so does an entry no case names.

The resolution is the ranker's slowdown against the override, 5 % by default: ranker time at most
1.05 x the override's. route_as_table.py's --min-gain 5 counts the other way (the override saves
5 % of the default's time, 1 - 1/1.05 = 4.76 % of it saved when the default is 5 % slower), so the
two thresholds describe the same size of effect to within a quarter of a percent, not the same
number.

    route_table_prune.py report.json --table routes/ad102.json [--dense-tune routes/tuner/dense_tune.json
        --class ad102:128sm] [--resolution 0.05] [--apply]

The report's board, the table's `gpu.class` and --class must name one class (equal, or one the chip
of the other: `ad102` and `ad102:128sm`), or the tool refuses. Default is a dry run: the plan is
printed with the board and nothing is written. --apply rewrites the files, which must be in the
canonical form (json.dumps(indent=1) plus a newline) so that the diff is only the deleted entries;
otherwise it refuses.

Also the evidence check of the tables (evidence_gaps): every measured entry (one with `tuned_on`)
should record `evidence_us`, rows [T, default us, chosen us], with a gain above the resolution.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import selection_regret as regret

RESOLUTION = 0.05  # the ranker's slowdown within which an override earns nothing
MIN_GAIN = 0.05  # route_as_table.py --min-gain: the least a measured override must save
TABLE_AUTHORITIES = {"route-table", "dense-tune"}
DECODE_MAX, VERIFY_MAX = 4, 16
PREFILL_PROFILE = "prefill"
INF = float("inf")


def entry_key(entry):
    return (entry["format"], entry["compute"], entry.get("epilogue", "store"), int(entry["n"]),
            int(entry["k"]))


def case_key(case):
    """(format, compute, epilogue, n, k, T) of a case id `<format>.<compute>.<epilogue>.n<N>.k<K>.T<T>`."""
    found = re.fullmatch(r"(.+)\.(a16|a8|a4)\.([a-z_0-9]+)\.n(\d+)\.k(\d+)\.T(\d+)", case["id"])
    if not found:
        return None
    fmt, compute, epilogue, n, k, t = found.groups()
    return (fmt, compute, epilogue, int(n), int(k)), int(t)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def ridge_of(report):
    """The board's ridge T* the report carries (by_board_regime_authority), or None."""
    for group in report.get("by_board_regime_authority", []):
        if finite(group.get("ridge_t")):
            return group["ridge_t"]
    return None


def segments(lo, hi, ridge):
    """The band [lo, hi] cut at the regime edges (4|5, 16|17) and at the ridge: [(a, b)]."""
    cuts = [DECODE_MAX + 1, VERIFY_MAX + 1]
    if ridge and ridge > VERIFY_MAX + 1:
        cuts.append(math.ceil(ridge))
    out, start = [], lo
    for cut in sorted(cuts):
        if start < cut <= hi:
            out.append((start, cut - 1))
            start = cut
    out.append((start, hi))
    return out


def evidence_by_entry(report):
    """{(authority, entry key): {"speed": [(T, slowdown or None)], "prefill": [...]}}: the speaking
    cases of the report. A slowdown that is not a finite number, or of a case that is not complete,
    is None (it keeps the entry)."""
    out = {}
    for case in report["cases"]:
        if case.get("authority") not in TABLE_AUTHORITIES or case.get("evidence") != "measured":
            continue
        parsed = case_key(case)
        if parsed is None or case.get("t") is None:
            continue
        key, t = parsed
        slowdown = case.get("ranker_vs_selected")
        if case.get("status") != "complete" or not finite(slowdown):
            slowdown = None
        profile = "prefill" if case.get("profile") == PREFILL_PROFILE else "speed"
        out.setdefault((case["authority"], key), {"speed": [], "prefill": []})[profile].append(
            (t, slowdown))
    return out


def judge(cases, label, resolution):
    """Reasons a set of (T, slowdown) cases does not clear `label`'s range; empty: cleared."""
    if not cases:
        return [f"{label}: no case"]
    if any(s is None for _, s in cases):
        return [f"{label}: a case without a ranker time"]
    worst = max(s for _, s in cases)
    if worst > resolution:
        return [f"{label}: ranker {100 * worst:.1f} % slower somewhere"]
    return []


def verdict(evidence, ranges, has_prefill, ridge, resolution):
    """(prune, reasons) of an entry: `ranges` its speed bands [(lo, hi)], `has_prefill` whether it
    holds a prefill override, `evidence` its cases (speed and prefill)."""
    reasons = []
    if not ranges and not has_prefill:
        reasons.append("no speed band")
    for lo, hi in ranges:
        for a, b in segments(lo, hi, ridge):
            inside = [(t, s) for t, s in evidence["speed"] if a <= t <= b]
            reasons += judge(inside, f"T {a}..{'inf' if b == INF else b}", resolution)
    if has_prefill:
        reasons += judge(evidence["prefill"], "prefill override",
                         resolution) if evidence["prefill"] else [
            "prefill override: no prefill-profile case (the campaign runs the speed profile)"]
    return not reasons, reasons


def plan_table(report, table, resolution):
    """[(entry index, key, prune, reasons)] over the table's measured entries."""
    seen = evidence_by_entry(report)
    ridge = ridge_of(report)
    out = []
    for i, entry in enumerate(table["entries"]):
        if not ("tuned_on" in entry or "prefill_tuned_on" in entry):
            continue  # not a measured override of this board
        key = entry_key(entry)
        evidence = seen.get(("route-table", key))
        if evidence is None:
            out.append((i, key, False, ["no case"]))
            continue
        ranges = [(lo, hi) for lo, hi, _ in entry.get("bands", [])]
        has_prefill = bool(entry.get("prefill") or entry.get("prefill_bands")
                           or "prefill_tuned_on" in entry)
        prune, reasons = verdict(evidence, ranges, has_prefill, ridge, resolution)
        out.append((i, key, prune, reasons))
    return out


def dense_key(text):
    """(class id, key) of `ad102:128sm|q40|a16|store|n262144|k2816|<schedule set>...`."""
    parts = text.split("|")
    n = next(p for p in parts if re.fullmatch(r"n\d+", p))
    k = next(p for p in parts if re.fullmatch(r"k\d+", p))
    return parts[0], (parts[1], parts[2], parts[3], int(n[1:]), int(k[1:]))


def plan_dense(report, dense, device_class, resolution):
    """[(problem key text, key, prune, reasons)] over the class's problems with a route set. The
    key drops the schedule-set and variant suffixes: two problems of the class that parse to one
    key cannot be told apart by the report's cases, and are refused."""
    seen = evidence_by_entry(report)
    ridge = ridge_of(report)
    parsed, seen_keys = [], {}
    for text, problem in dense["problems"].items():
        cls, key = dense_key(text)
        if cls != device_class:
            continue
        if key in seen_keys:
            raise ValueError(f"dense keys collide after parsing ({label(key)}): {seen_keys[key]} "
                             f"and {text}")
        seen_keys[key] = text
        parsed.append((text, key, problem))
    out = []
    for text, key, problem in parsed:
        times = sorted(int(t) for t, route in problem.get("routes", {}).items() if route is not None)
        has_prefill = any(r is not None for r in problem.get("prefill", {}).values())
        if not times and not has_prefill:
            continue
        evidence = seen.get(("dense-tune", key))
        if evidence is None:
            out.append((text, key, False, ["no case"]))
            continue
        # the segments (regime edges, ridge) that hold a width the tuner routed need a case
        prune, reasons = verdict(evidence, _segment_ranges(times, ridge), has_prefill, ridge,
                                 resolution)
        out.append((text, key, prune, reasons))
    return out


def _segment_ranges(times, ridge):
    """The segments (regime edges and ridge) that hold a routed width: the ranges to cover."""
    out = []
    for t in times:
        for a, b in segments(1, INF, ridge):
            if a <= t <= b and (a, b) not in out:
                out.append((a, b))
    return out


def classes_agree(a, b):
    """The same class id, or one the chip of the other (`ad102` and `ad102:128sm`)."""
    return a == b or a.split(":")[0] == b.split(":")[0] and (":" not in a or ":" not in b)


def canonical(doc):
    return json.dumps(doc, indent=1) + "\n"


def evidence_gaps(table, resolution=RESOLUTION):
    """Measured entries (`tuned_on`) of a route table that do not record evidence_us with a gain
    above the resolution: {entry key: "missing" | "weak"}. evidence_us rows are [T, default us,
    chosen us]; `evidence_decode_us` (the 5090's decode evidence) is read alike."""
    gaps = {}
    for entry in table["entries"]:
        if not ("tuned_on" in entry or "prefill_tuned_on" in entry):
            continue
        rows = (entry.get("evidence_us") or []) + (entry.get("evidence_decode_us") or [])
        if not rows:
            gaps[entry_key(entry)] = "missing"
        elif not any(default > 0 and 1 - chosen / default >= resolution
                     for _, default, chosen in rows):
            gaps[entry_key(entry)] = "weak"
    return gaps


def label(key):
    fmt, compute, epilogue, n, k = key
    return f"{fmt}.{compute}.{epilogue}.n{n}.k{k}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=Path, help="selection_regret.py report of one board")
    parser.add_argument("--table", type=Path, help="routes/<class>.json of that board")
    parser.add_argument("--dense-tune", type=Path)
    parser.add_argument("--class", dest="device_class",
                        help="the device class id of the dense tuner's keys (ad102:128sm)")
    parser.add_argument("--resolution", type=float, default=RESOLUTION,
                        help="ranker slowdown within which an override earns nothing (0.05)")
    parser.add_argument("--apply", action="store_true", help="write the pruned files (default: dry run)")
    args = parser.parse_args(argv)
    try:
        if not (args.table or args.dense_tune):
            raise ValueError("nothing to prune: give --table and/or --dense-tune")
        if args.dense_tune and not args.device_class:
            raise ValueError("--dense-tune needs --class")
        if not 0 <= args.resolution < 1:
            raise ValueError("--resolution must be in [0, 1)")
        report = json.loads(args.report.read_text(encoding="utf-8"),
                            object_pairs_hook=regret.unique_object)
        if report.get("schema") != "ninferno-selection-regret-report/1":
            raise ValueError("expected a ninferno-selection-regret-report/1 (selection_regret.py output)")
        boards = {c["board"] for c in report["cases"]}
        if len(boards) != 1:
            raise ValueError("the report must be one board's, it has " + ", ".join(sorted(boards)))
        board = boards.pop()
        table_class = None
        if args.table:
            table_class = json.loads(args.table.read_text(encoding="utf-8")).get("gpu", {}).get("class")
            if not table_class:
                raise ValueError(f"{args.table}: no gpu.class")
            if not classes_agree(board, table_class):
                raise ValueError(f"the report is of board {board}, the table of class {table_class}")
        if args.device_class:
            for other, what in ((board, "the report's board"), (table_class, "the table's gpu.class")):
                if other and not classes_agree(args.device_class, other):
                    raise ValueError(f"--class {args.device_class} differs from {what} {other}")
        print(f"board {board}" + (f", table class {table_class}" if table_class else "")
              + (f", dense class {args.device_class}" if args.dense_tune else ""))
        writes = []
        for path, kind in ((args.table, "table"), (args.dense_tune, "dense")):
            if path is None:
                continue
            with open(path, encoding="utf-8", newline="") as f:
                text = f.read()
            doc = json.loads(text)
            if kind == "table":
                plan = plan_table(report, doc, args.resolution)
                drop = {i for i, _, prune, _ in plan if prune}
                doc["entries"] = [e for i, e in enumerate(doc["entries"]) if i not in drop]
                shown = [(label(key), prune, why) for _, key, prune, why in plan]
            else:
                plan = plan_dense(report, doc, args.device_class, args.resolution)
                for text_key, _, prune, _ in plan:
                    if prune:
                        del doc["problems"][text_key]
                shown = [(label(key), prune, why) for _, key, prune, why in plan]
            print(f"{path}: {sum(p for _, p, _ in shown)} of {len(shown)} measured entries "
                  f"within {100 * args.resolution:g} %")
            for name, prune, why in shown:
                print(f"  {'prune' if prune else 'keep '} {name}" + ("" if prune else f"  ({'; '.join(why)})"))
            writes.append((path, text, doc, any(p for _, p, _ in shown)))
        if args.apply:
            for path, text, doc, changed in writes:
                if not changed:
                    continue
                json_text = json.loads(text)
                if canonical(json_text) != text:
                    raise ValueError(f"{path} is not in the canonical form (json indent=1 and a "
                                     "newline); normalise it first so the diff is the entries only")
            for path, _, doc, changed in writes:
                if changed:
                    with open(path, "w", encoding="utf-8", newline="") as f:
                        f.write(canonical(doc))
                    print(f"wrote {path}")
        else:
            print("dry run: nothing written (--apply writes)")
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        print(f"route_table_prune: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
