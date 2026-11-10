#!/usr/bin/env python3
"""Exit tooling for measured route overrides (#614, PR-F3): prune what the ranker has caught up with.

A route table entry (routes/<class>.json) or a dense tuner problem (routes/tuner/dense_tune.json)
is a measured override of the ranker (the cost model where it prices, else the rules). When the
regret campaign shows the ranker's own pick within its resolution of the override on that board and
regime, the override earns nothing and is deleted, as a diff a reviewer can read.

Input: the selection_regret.py report (ninferno-selection-regret-report/1) of one board, whose
cases carry `authority`, `evidence`, `t`, `ranker` (the route the ranker alone picks) and
`ranker_vs_selected` (its median time over the selected route's, minus one). Only cases whose
authority is `route-table` or `dense-tune` with `measured` evidence speak for an entry. An entry
is pruned only when, in every regime it has routes in (decode T<=4, verify 5..16, wide T>=17), at
least one case speaks for it and every such case has `ranker_vs_selected` <= the resolution. A case
without a ranker time, or an incomplete one, keeps the entry; so does an entry no case names.

    route_table_prune.py report.json --table routes/ad102.json [--dense-tune routes/tuner/dense_tune.json
        --class ad102:128sm] [--resolution 0.05] [--apply]

Default is a dry run: the plan is printed and nothing is written. --apply rewrites the files, which
must be in the canonical form (json.dumps(indent=1) plus a newline) so that the diff is only the
deleted entries; otherwise it refuses.

Also the evidence check of the tables (evidence_gaps): every measured entry (one with `tuned_on`)
should record `evidence_us`, rows [T, default us, chosen us], with a gain above the resolution.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import selection_regret as regret

RESOLUTION = 0.05  # route_as_table.py --min-gain: the least a measured override must gain
TABLE_AUTHORITIES = {"route-table", "dense-tune"}
DECODE_MAX, VERIFY_MAX = 4, 16
REGIMES = ("decode T<=4", "verify T5-16", "wide T>=17")


def regime_of(t):
    if t <= DECODE_MAX:
        return REGIMES[0]
    return REGIMES[1] if t <= VERIFY_MAX else REGIMES[2]


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


def regimes_of_entry(entry):
    """The regimes an entry has routes in: bands below 17 are decode/verify, a band from 17, a
    `prefill` route or `prefill_bands` the wide calls."""
    out = set()
    for lo, hi, _ in entry.get("bands", []):
        for regime in REGIMES[:2]:
            low, high = (1, DECODE_MAX) if regime == REGIMES[0] else (DECODE_MAX + 1, VERIFY_MAX)
            if lo <= high and hi >= low:
                out.add(regime)
        if hi > VERIFY_MAX:
            out.add(REGIMES[2])
    if entry.get("prefill") or entry.get("prefill_bands"):
        out.add(REGIMES[2])
    return out


def evidence_by_entry(report, resolution):
    """{entry key: {regime: [ranker slowdown or None per speaking case]}} of the report's cases."""
    out = {}
    for case in report["cases"]:
        if case.get("authority") not in TABLE_AUTHORITIES or case.get("evidence") != "measured":
            continue
        parsed = case_key(case)
        if parsed is None or case.get("t") is None:
            continue
        key, t = parsed
        slowdown = case.get("ranker_vs_selected") if case.get("status") == "complete" else None
        out.setdefault((case["authority"], key), {}).setdefault(regime_of(t), []).append(slowdown)
    return out


def verdict(regimes, wanted, resolution):
    """(prune, reasons): every wanted regime has speaking cases, all within the resolution."""
    reasons = []
    for regime in sorted(wanted, key=REGIMES.index):
        seen = regimes.get(regime)
        if not seen:
            reasons.append(f"{regime}: no case")
        elif any(s is None for s in seen):
            reasons.append(f"{regime}: a case without a ranker time")
        elif max(seen) > resolution:
            reasons.append(f"{regime}: ranker {100 * max(seen):.1f} % slower somewhere")
    return (not reasons and bool(wanted)), reasons


def plan_table(report, table, resolution):
    """[(entry index, key, prune, reasons)] over the table's measured entries."""
    seen = evidence_by_entry(report, resolution)
    out = []
    for i, entry in enumerate(table["entries"]):
        if not ("tuned_on" in entry or "prefill_tuned_on" in entry):
            continue  # not a measured override of this board
        key = entry_key(entry)
        regimes = seen.get(("route-table", key))
        if regimes is None:
            out.append((i, key, False, ["no case"]))
            continue
        prune, reasons = verdict(regimes, regimes_of_entry(entry), resolution)
        out.append((i, key, prune, reasons))
    return out


def dense_regimes(problem):
    out = set()
    for t, route in problem.get("routes", {}).items():
        if route is not None:
            out.add(regime_of(int(t)))
    if any(r is not None for r in problem.get("prefill", {}).values()):
        out.add(REGIMES[2])
    return out


def dense_key(text):
    """(class id, key) of `ad102:128sm|q40|a16|store|n262144|k2816|<schedule set>...`."""
    parts = text.split("|")
    n = next(p for p in parts if re.fullmatch(r"n\d+", p))
    k = next(p for p in parts if re.fullmatch(r"k\d+", p))
    return parts[0], (parts[1], parts[2], parts[3], int(n[1:]), int(k[1:]))


def plan_dense(report, dense, device_class, resolution):
    """[(problem key text, key, prune, reasons)] over the class's problems with a route set."""
    seen = evidence_by_entry(report, resolution)
    out = []
    for text, problem in dense["problems"].items():
        cls, key = dense_key(text)
        wanted = dense_regimes(problem)
        if cls != device_class or not wanted:
            continue
        regimes = seen.get(("dense-tune", key))
        if regimes is None:
            out.append((text, key, False, ["no case"]))
            continue
        prune, reasons = verdict(regimes, wanted, resolution)
        out.append((text, key, prune, reasons))
    return out


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
