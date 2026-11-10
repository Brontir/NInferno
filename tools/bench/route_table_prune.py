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
for an entry; a case's width is its `t`. The campaign runs the speed profile, so a case measures a
prefill override only if it says so (`profile: "prefill"`, which no run writes yet). An entry is
kept, whatever the speed cases say, when
  - it holds a prefill override (`prefill`, `prefill_bands`, `prefill_tuned_on`; a dense problem
    with a prefill route) and no prefill-profile case covers each of its prefill ranges (every
    `prefill_bands` range, and 17.. where a `prefill` route stands), cut as below;
  - it `claim`s a problem (a measured claim the ranker does not make);
  - another table of the routes directory holds the same key (after pruning, this board would
    borrow that table's override, not the ranker's pick: route_table.cpp fill_from lends a missing
    key from the next nearest table), or, unless --allow-borrowers, a table lacking the key exists
    (it may borrow this one's: the nearest holder lends, and distances across generations are not
    mirrored here). Dense problems are keyed by exact class id and lend nothing.
An entry is pruned only when every segment of every range it has routes in (`bands` and
`r8_bands`) has at least one case in its T range and every case there has `ranker_vs_selected` <=
the resolution. A range is cut into segments at the regime edges (decode T<=4, verify 5..16) and at
the board's ridge T* (the report's own, `by_board_regime_authority`), so a range that spans a
regime needs a case in each part. A range that reaches 17 or more with no ridge in the report
cannot be split, and keeps the entry. A case without a ranker time, or an incomplete one, keeps the
entry; so does an entry no case names.

The resolution is the ranker's slowdown against the override, 5 % by default: ranker time at most
1.05 x the override's. route_as_table.py's --min-gain 5 counts the other way (the override saves
5 % of the default's time: 1 - 1/1.05 = 4.76 % saved when the default is 5 % slower), so the two
thresholds describe the same size of effect to within a quarter of a percent, not the same number.

    route_table_prune.py report.json --table routes/ad102.json [--routes-dir routes]
        [--dense-tune routes/tuner/dense_tune.json --class ad102:128sm] [--resolution 0.05]
        [--allow-borrowers] [--apply]

The report's board must be the table's class: equal ids, or a full `chip:Nsm` id for a table of the
bare chip whose `gpu.board.sms` is N (the RTX 5080's `gb203` is not the 70-SM `gb203:70sm`). --class
must be a full class id (the dense keys carry it whole: a bare chip matches none) equal to the
report's board. Default is a dry run: the plan is printed with the board and nothing is written.
--apply rewrites the files (both or neither: temp files, then renames), which must be in the
canonical form (json.dumps(indent=1) plus a newline) so that the diff is only the deleted entries;
otherwise it refuses.

Also the evidence check of the tables (evidence_gaps): every measured entry (one with `tuned_on`)
should record `evidence_us`, rows [T, default us, chosen us], with a gain above the resolution.
"""
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import selection_regret as regret

RESOLUTION = 0.05  # the ranker's slowdown within which an override earns nothing
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
    """The range [lo, hi] cut at the regime edges (4|5, 16|17) and at the ridge: [(a, b)]."""
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
    cases of the report, at their `t`. A slowdown that is not a finite number, or of a case that is
    not complete, is None (it keeps the entry)."""
    out = {}
    for case in report["cases"]:
        if case.get("authority") not in TABLE_AUTHORITIES or case.get("evidence") != "measured":
            continue
        parsed = case_key(case)
        if parsed is None:
            continue
        key, from_id = parsed
        t = case["t"] if type(case.get("t")) is int else from_id
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


def cover(cases, ranges, ridge, resolution):
    """Reasons the cases of one profile do not clear every segment of `ranges`."""
    reasons = []
    for lo, hi in ranges:
        if ridge is None and hi > VERIFY_MAX:
            reasons.append(f"T {max(lo, VERIFY_MAX + 1)}..: no ridge in the report, a wide range "
                           "cannot be split")
        for a, b in segments(lo, hi, ridge):
            inside = [(t, s) for t, s in cases if a <= t <= b]
            reasons += judge(inside, f"T {a}..{b}", resolution)
    return reasons


def verdict(evidence, ranges, prefill_ranges, ridge, resolution):
    """(prune, reasons) of an entry: `ranges` its speed ranges [(lo, hi)] (bands and r8_bands),
    `prefill_ranges` those of its prefill override, `evidence` its cases (speed and prefill)."""
    reasons = []
    if not ranges and not prefill_ranges:
        reasons.append("no band")
    reasons += cover(evidence["speed"], ranges, ridge, resolution)
    if prefill_ranges:
        if evidence["prefill"]:
            reasons += ["prefill " + why for why in
                        cover(evidence["prefill"], prefill_ranges, ridge, resolution)]
        else:
            reasons.append("prefill override: no prefill-profile case (the campaign runs the "
                           "speed profile)")
    return not reasons, reasons


def entry_ranges(entry):
    """(speed ranges, prefill ranges) of a table entry: its bands and r8_bands, and the ranges of
    its prefill override (each prefill band; 17.. where a `prefill` route stands)."""
    speed = [(lo, hi) for lo, hi, *_ in entry.get("bands", []) + entry.get("r8_bands", [])]
    prefill = [(lo, hi) for lo, hi, *_ in entry.get("prefill_bands", [])]
    if entry.get("prefill") or "prefill_tuned_on" in entry:
        if not prefill or entry.get("prefill"):
            prefill.append((VERIFY_MAX + 1, INF))
    return speed, prefill


def lending(others, key):
    """(holders, lackers) among the other tables {name: doc} of the routes directory for `key`:
    those that hold the entry (the board would borrow it after pruning) and those that lack it (they
    may borrow it from this table)."""
    holders, lackers = [], []
    for name, doc in sorted(others.items()):
        held = {entry_key(e) for e in doc.get("entries", [])}
        (holders if key in held else lackers).append(name)
    return holders, lackers


def plan_table(report, table, resolution, others=None, allow_borrowers=False):
    """[(entry index, key, prune, reasons)] over the table's measured entries. `others`: the other
    tables of the routes directory {name: doc}, for the lending rules."""
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
        speed, prefill = entry_ranges(entry)
        _, reasons = verdict(evidence, speed, prefill, ridge, resolution)
        if entry.get("claim"):
            reasons.append("the entry claims its problem (a measured claim the ranker does not make)")
        if others:
            holders, lackers = lending(others, key)
            if holders:
                reasons.append("another table holds this key (" + ", ".join(holders) +
                               "): the board would borrow its override, not the ranker's pick")
            elif lackers and not allow_borrowers:
                reasons.append("tables lacking this key may borrow it (" + ", ".join(lackers) +
                               "): --allow-borrowers to prune anyway")
        out.append((i, key, not reasons, reasons))
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
        prefill_times = sorted(int(t) for t, r in problem.get("prefill", {}).items()
                               if r is not None)
        if not times and not prefill_times:
            continue
        evidence = seen.get(("dense-tune", key))
        if evidence is None:
            out.append((text, key, False, ["no case"]))
            continue
        # the segments (regime edges, ridge) that hold a width the tuner routed need a case
        prefill_ranges = _segment_ranges(prefill_times, ridge) if prefill_times else []
        prune, reasons = verdict(evidence, _segment_ranges(times, ridge), prefill_ranges, ridge,
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


def chip_of(class_id):
    return class_id.split(":")[0]


def class_sms(class_id):
    found = re.search(r":(\d+)sm$", class_id)
    return int(found[1]) if found else None


def board_matches_table(board, table):
    """The report's board names the table's board: equal class ids, or a full `chip:Nsm` id for a
    table of the bare chip whose gpu.board.sms is N. A bare chip names no SM count, so it never
    matches a table of a full class id, nor a different board of the chip."""
    table_class = table.get("gpu", {}).get("class", "")
    if board == table_class:
        return True
    sms = class_sms(board)
    return (":" not in table_class and sms is not None and chip_of(board) == table_class and
            sms == table.get("gpu", {}).get("board", {}).get("sms"))


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


def write_both(writes):
    """Rewrite the files of `writes` [(path, doc)] both or neither: each to a temp file beside it
    first, then renamed over it; if a rename fails, the files already renamed are put back."""
    temps, backups, done = [], {}, []
    try:
        for path, doc in writes:
            temp = path.with_name(path.name + ".prune.tmp")
            with open(temp, "w", encoding="utf-8", newline="") as f:
                f.write(canonical(doc))
            temps.append((path, temp))
            with open(path, "rb") as f:
                backups[path] = f.read()
        for path, temp in temps:
            os.replace(temp, path)
            done.append(path)
    except BaseException:
        for path in done:
            with open(path, "wb") as f:
                f.write(backups[path])
        raise
    finally:
        for _, temp in temps:
            if temp.exists():
                temp.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=Path, help="selection_regret.py report of one board")
    parser.add_argument("--table", type=Path, help="routes/<class>.json of that board")
    parser.add_argument("--routes-dir", type=Path,
                        help="the routes directory whose other tables lend and borrow (default: "
                             "the table's own)")
    parser.add_argument("--allow-borrowers", action="store_true",
                        help="prune although tables lacking the key may borrow it (a table "
                             "holding it still refuses)")
    parser.add_argument("--dense-tune", type=Path)
    parser.add_argument("--class", dest="device_class",
                        help="the full device class id of the dense tuner's keys (ad102:128sm)")
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
        table_doc = None
        if args.table:
            table_doc = json.loads(args.table.read_text(encoding="utf-8"))
            table_class = table_doc.get("gpu", {}).get("class")
            if not table_class:
                raise ValueError(f"{args.table}: no gpu.class")
            if not board_matches_table(board, table_doc):
                raise ValueError(f"the report is of board {board}, which is not the table's board "
                                 f"(class {table_class}, {table_doc['gpu'].get('board', {}).get('sms')} SMs)")
        if args.device_class:
            if class_sms(args.device_class) is None:
                raise ValueError(f"--class {args.device_class} is not a full class id (chip:Nsm): "
                                 "the dense keys carry it whole and a bare chip matches none")
            if args.device_class != board:
                raise ValueError(f"--class {args.device_class} differs from the report's board {board}")
        print(f"board {board}" + (f", table class {table_doc['gpu']['class']}" if table_doc else "")
              + (f", dense class {args.device_class}" if args.dense_tune else ""))
        others = None
        if args.table:
            routes_dir = args.routes_dir or args.table.parent
            others = {}
            for path in sorted(routes_dir.glob("*.json")):
                if path.resolve() == args.table.resolve():
                    continue
                doc = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(doc, dict) and "entries" in doc:
                    others[path.stem] = doc
        replaced = []
        for path, kind in ((args.table, "table"), (args.dense_tune, "dense")):
            if path is None:
                continue
            with open(path, encoding="utf-8", newline="") as f:
                text = f.read()
            doc = json.loads(text)
            if kind == "table":
                plan = plan_table(report, doc, args.resolution, others, args.allow_borrowers)
                drop = {i for i, _, prune, _ in plan if prune}
                doc["entries"] = [e for i, e in enumerate(doc["entries"]) if i not in drop]
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
            if any(p for _, p, _ in shown):
                if canonical(json.loads(text)) != text and args.apply:
                    raise ValueError(f"{path} is not in the canonical form (json indent=1 and a "
                                     "newline); normalise it first so the diff is the entries only")
                replaced.append((path, doc))
        if args.apply:
            write_both(replaced)
            for path, _ in replaced:
                print(f"wrote {path}")
        else:
            print("dry run: nothing written (--apply writes)")
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        print(f"route_table_prune: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
