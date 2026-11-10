#!/usr/bin/env python3
"""Band-weighted selection regret (#613 PR-0). See README.md.

Reads recorded route choices and reports the extra time they cost, per T band, grouped by board,
format and route family. Offline and descriptive: it selects nothing, fits nothing, runs no GPU.

Inputs (repeatable, mixable):
  --doc FILE.json     a ninferno-selection-regret/1 document (what regret_collect.py writes);
  --routes P1.csv ... --selected SEL.csv --board NAME
                      route_tune --routes-out passes of one board and its --selected-out file. These
                      are accepted without the run/device columns regret_collect.py demands, because
                      the existing sweep archive predates them (see data/sweep_archive.json).
  --pick model        with --routes: score the pick of the model price (lowest model_us) instead of
                      the recorded production selection.
"""
import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regret_collect as collect  # noqa: E402
import selection_regret as regret  # noqa: E402

# Upper edge (inclusive) of each band and what it is in the product. The decode regime is T <= 8
# (R8 and the vector kernel), 9..16 is a speculative verify, 17..32 the narrow batched widths where
# the cost model misranks most, 33..128 wide verify/batched decode, and 129+ prefill chunks.
BANDS = (("decode", 1, 8), ("verify", 9, 16), ("narrow", 17, 32), ("wide", 33, 128),
         ("prefill", 129, None))
BAND_NAMES = tuple(b[0] for b in BANDS)


def band_of(t):
    for name, lo, hi in BANDS:
        if t >= lo and (hi is None or t <= hi):
            return name
    raise ValueError(f"T={t}: below the first band")


def family_of(route):
    """The route family: the part of the id before its first dot (rmma, ggml_mmq, r8, w4a8 ...)."""
    return route.split(".", 1)[0]


def tokens_of(case):
    for text, pattern in ((case["id"], r"\.T(\d+)$"), (case.get("workload", ""), r"\bT=(\d+)")):
        found = re.search(pattern, text)
        if found:
            return int(found.group(1))
    raise ValueError(f"{case['id']}: no T in id or workload")


def format_of(case):
    return case["id"].split(".", 1)[0]


def rows_from_document(document):
    """Complete cases of a regret document as rows; other cases are only counted."""
    report = regret.evaluate(document)
    by_id = {c["id"]: c for c in document["cases"]}
    rows, skipped = [], 0
    for case in report["cases"]:
        if case["status"] != "complete":
            skipped += 1
            continue
        src = by_id[case["id"]]
        medians = case["median_ms"]
        rows.append({
            "id": case["id"], "board": case["board"], "format": format_of(case),
            "t": tokens_of(src), "selected": case["selected"], "best": case["best_observed"],
            "selected_ms": medians[case["selected"]], "best_ms": medians[case["best_observed"]]})
    return rows, skipped


def rows_from_csv(route_paths, selected_path, board, pick="selected"):
    """Rows from legacy route_tune CSVs: per case, medians over passes of the routes of its class."""
    if pick not in ("selected", "model"):
        raise ValueError("pick must be selected or model")
    timed, model = {}, {}  # (key, route) -> [us per pass]
    for path in route_paths:
        columns = collect.KEY + ("route", "median_us") + (("model_us",) if pick == "model" else ())
        seen = set()
        for row in collect.read_rows(path, columns):
            at = (collect.key_of(row), row["route"].strip())
            if at in seen:
                raise ValueError(f"{path}: {collect.case_id(at[0])} {at[1]} timed twice in one pass")
            seen.add(at)
            timed.setdefault(at, []).append(float(row["median_us"]))
            if pick == "model":
                model.setdefault(at, []).append(float(row["model_us"]))
    passes = len(route_paths)
    selected = None
    if pick == "selected":
        if not selected_path:
            raise ValueError("--selected is needed to score the recorded selection")
        selected = {}
        for row in collect.read_rows(selected_path, collect.KEY + ("selected",)):
            selected[collect.key_of(row)] = row["selected"].strip()
    routes = {}
    for key, route in timed:
        routes.setdefault(key, []).append(route)
    rows, skipped = [], 0
    for key in sorted(routes, key=lambda k: (k[:5], int(k[5]))):
        # A route missing from a pass has no comparable median: it is not a candidate.
        cands = {r: statistics.median(timed[(key, r)]) for r in routes[key]
                 if len(timed[(key, r)]) == passes
                 and not (key[1] == "a16" and collect.route_class(r) != "a16")}
        if pick == "model":
            priced = {r: statistics.median(model[(key, r)]) for r in cands
                      if model[(key, r)][0] > 0}
            if not priced:
                skipped += 1
                continue
            chosen = min(priced, key=lambda r: (priced[r], r))
        else:
            chosen = selected.get(key)
        if chosen not in cands:
            skipped += 1
            continue
        best = min(cands, key=lambda r: (cands[r], r))
        rows.append({"id": collect.case_id(key), "board": board, "format": key[0],
                     "t": int(key[5]), "selected": chosen, "best": best,
                     "selected_ms": cands[chosen] / 1000.0, "best_ms": cands[best] / 1000.0})
    return rows, skipped


def stats(rows):
    """Time-weighted regret: extra time of the picks over the best, as a share of the best time."""
    sel = math.fsum(r["selected_ms"] for r in rows)
    best = math.fsum(r["best_ms"] for r in rows)
    ratios = [r["selected_ms"] / r["best_ms"] for r in rows]
    return {
        "cases": len(rows), "selected_ms": sel, "best_ms": best,
        "weighted_regret": sel / best - 1,
        "geomean_regret": math.exp(math.fsum(map(math.log, ratios)) / len(ratios)) - 1,
        "worst_regret": max(ratios) - 1,
        "cases_over_5pct": sum(1 for x in ratios if x > 1.05),
    }


def by_band(rows):
    return {b: stats([r for r in rows if band_of(r["t"]) == b])
            for b in BAND_NAMES if any(band_of(r["t"]) == b for r in rows)}


def report(rows, skipped=0, band_weights=None):
    """Per band overall, then per board / format / route family, each by band.

    `band_weights` (band -> weight) adds one workload-weighted number per group: the weighted mean of
    its band regrets. The default report has none, since no workload mix is frozen yet.
    """
    for r in rows:
        r["family"] = family_of(r["selected"])
    out = {"schema": "ninferno-regret-bands/1", "bands": {n: [lo, hi] for n, lo, hi in BANDS},
           "metric": "sum(selected time) / sum(best observed time) - 1 per group and band",
           "cases": len(rows), "cases_skipped": skipped, "overall": by_band(rows)}
    for dim in ("board", "format", "family"):
        out["by_" + dim] = {k: by_band([r for r in rows if r[dim] == k])
                            for k in sorted({r[dim] for r in rows})}
    if band_weights:
        unknown = set(band_weights) - set(BAND_NAMES)
        if unknown or any(w < 0 for w in band_weights.values()) or not any(band_weights.values()):
            raise ValueError(f"band weights must name {BAND_NAMES} and not all be zero")
        out["band_weights"] = band_weights
        groups = [out["overall"]] + [g for k in out if k.startswith("by_") for g in out[k].values()]
        for group in groups:
            w = {b: band_weights.get(b, 0) for b in group}
            total = sum(w.values())
            group["weighted"] = (sum(group[b]["weighted_regret"] * w[b] for b in w) / total
                                 if total else None)
    return out


def pct(x):
    return "-" if x is None else f"{100 * x:+.1f}%"


def markdown(rep):
    lines = ["# Band-weighted selection regret", "",
             f"{rep['cases']} cases ({rep['cases_skipped']} skipped). Each cell is the extra time "
             "of the picks over the best observed route, summed over the band's cases (time-weighted).",
             "", "Bands (T): " + ", ".join(f"{n} {lo}-{hi if hi else '+'}"
                                           for n, (lo, hi) in rep["bands"].items()), ""]
    weighted = "band_weights" in rep

    def table(title, groups):
        head = ["| " + title + " | " + " | ".join(f"{b} (n)" for b in BAND_NAMES)
                + (" | weighted |" if weighted else " |"),
                "|---|" + "---:|" * (len(BAND_NAMES) + weighted)]
        for name, bands in groups.items():
            cells = [f"{pct(bands[b]['weighted_regret'])} ({bands[b]['cases']})" if b in bands
                     else "" for b in BAND_NAMES]
            if weighted:
                cells.append(pct(bands.get("weighted")))
            head.append(f"| {name} | " + " | ".join(cells) + " |")
        return head + [""]

    lines += table("all", {"all": rep["overall"]})
    for dim in ("board", "format", "family"):
        lines += table(dim, rep["by_" + dim])
    return "\n".join(lines)


def archive_summary(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != "ninferno-sweep-archive/1":
        raise ValueError("unsupported manifest schema")
    out = {}
    for stand, dirs in document["stands"].items():
        out[stand] = {k: sum(d[k] for d in dirs)
                      for k in ("route_sweeps", "selected", "regret_docs", "stamped", "bytes")}
        out[stand]["dirs"] = len(dirs)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--doc", type=Path, action="append", default=[])
    p.add_argument("--routes", type=Path, nargs="+", action="append", default=[],
                   help="route_tune passes of one board (one group per --routes)")
    p.add_argument("--selected", type=Path, action="append", default=[],
                   help="selection CSV for the matching --routes group")
    p.add_argument("--board", action="append", default=[], help="board name for the matching --routes group")
    p.add_argument("--pick", choices=("selected", "model"), default="selected")
    p.add_argument("--band-weights", help='JSON, e.g. {"decode":0.4,"verify":0.2}')
    p.add_argument("--json", type=Path, help="write the JSON report here")
    p.add_argument("--md", type=Path, help="write the markdown report here")
    p.add_argument("--archive", type=Path, help="print the totals of a sweep_archive.json and exit")
    args = p.parse_args(argv)
    try:
        if args.archive:
            print(json.dumps(archive_summary(args.archive), indent=1, sort_keys=True))
            return 0
        rows, skipped = [], 0
        for doc in args.doc:
            r, s = rows_from_document(json.loads(doc.read_bytes(), object_pairs_hook=regret.unique_object))
            rows += r
            skipped += s
        if len(args.board) != len(args.routes):
            raise ValueError("one --board per --routes group")
        for i, passes in enumerate(args.routes):
            sel = args.selected[i] if i < len(args.selected) else None
            r, s = rows_from_csv(passes, sel, args.board[i], args.pick)
            rows += r
            skipped += s
        if not rows:
            raise ValueError("no scored cases")
        weights = json.loads(args.band_weights) if args.band_weights else None
        rep = report(rows, skipped, weights)
    except (OSError, UnicodeError, ValueError, KeyError, OverflowError) as error:
        print(f"regret_bands: {error}", file=sys.stderr)
        return 2
    text = json.dumps(rep, indent=2, sort_keys=True, allow_nan=False)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    if args.md:
        args.md.write_text(markdown(rep) + "\n", encoding="utf-8")
    if not args.json and not args.md:
        print(markdown(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
