#!/usr/bin/env python3
"""Regret lines of the fusion sites (#617 step H): measured fused vs decomposed time at the branch
the site decided, per site and call width.

One `ninferno-site-regret/1` JSON document holds the rows of one board and one run:

  {"schema": "ninferno-site-regret/1", "board": "...", "run": "...", "protocol": "graph_reps32_min3",
   "rows": [{"site": "<id of tools/audit/fusion_sites.json>", "case": "32+2", "T": 128,
             "decided": "fused",
             "candidates": [{"branch": "fused", "median_us": 4.1},
                            {"branch": "decomposed", "median_us": 5.0}]}, ...]}

`regret` of a row is decided / best - 1 over its candidates (0: the decided branch is the fastest).
A site id the registry does not have, a decided branch that is not a candidate, a candidate time that
is not finite and positive, or one case twice, is refused: nothing is dropped silently. The
manifest side of the same sites is ops/common/fusion_site_manifest.h (its line says the branch a
board takes at a width; this document says what that branch cost against the other).

  site_regret.py from-rope-crossover CSV --board B --run R [--bound 256]   > doc.json
      the `--crossover` CSV of ninferno_rmsnorm_rope_bench (geometry,T,fused_us,split_us,...): the
      qk_norm_rope site. The decided branch is the dispatch's: fused for T <= --bound (256 today),
      decomposed above it.
  site_regret.py check doc.json
      validates a document against the registry and prints per-site regret statistics.

Offline: never selects, fits or runs a GPU.
"""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

SCHEMA = "ninferno-site-regret/1"
ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "tools" / "audit" / "fusion_sites.json"


def registry_ids(path=REGISTRY):
    with open(path, encoding="utf-8") as f:
        return {s["id"] for s in json.load(f)["sites"]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(doc, ids):
    require(isinstance(doc, dict) and doc.get("schema") == SCHEMA, f"schema must be {SCHEMA}")
    for key in ("board", "run", "protocol"):
        require(isinstance(doc.get(key), str) and doc[key].strip(), f"{key}: expected nonempty string")
    rows = doc.get("rows")
    require(isinstance(rows, list) and rows, "rows: expected a nonempty list")
    seen = set()
    for i, row in enumerate(rows):
        where = f"rows[{i}]"
        require(row.get("site") in ids, f"{where}: site {row.get('site')!r} is not in the registry")
        key = (row["site"], row.get("case"), row.get("T"))
        require(key not in seen, f"{where}: {key} twice")
        seen.add(key)
        require(isinstance(row.get("case"), str) and row["case"], f"{where}: case")
        require(isinstance(row.get("T"), int) and row["T"] > 0, f"{where}: T")
        cands = row.get("candidates")
        require(isinstance(cands, list) and len(cands) >= 2, f"{where}: at least two candidates")
        branches = [c.get("branch") for c in cands]
        require(len(set(branches)) == len(branches), f"{where}: duplicate candidate")
        for c in cands:
            us = c.get("median_us")
            require(type(us) in (int, float) and math.isfinite(us) and us > 0,
                    f"{where}: {c.get('branch')}: median_us must be finite and positive")
        require(row.get("decided") in branches, f"{where}: decided branch is not a candidate")


def regret_of(row):
    times = {c["branch"]: c["median_us"] for c in row["candidates"]}
    return times[row["decided"]] / min(times.values()) - 1.0


def summarize(doc):
    by_site = {}
    for row in doc["rows"]:
        by_site.setdefault(row["site"], []).append(regret_of(row))
    return {
        site: {"rows": len(v), "median_regret": statistics.median(v), "max_regret": max(v),
               "geomean_slowdown": math.exp(math.fsum(math.log1p(x) for x in v) / len(v))}
        for site, v in sorted(by_site.items())
    }


def from_rope_crossover(csv_path, board, run, bound):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip() and not ln.startswith("#")]
    for r in csv.DictReader(lines):
        t = int(r["T"])
        rows.append({
            "site": "qk_norm_rope", "case": r["geometry"], "T": t,
            "decided": "fused" if t <= bound else "decomposed",
            "candidates": [{"branch": "fused", "median_us": float(r["fused_us"])},
                           {"branch": "decomposed", "median_us": float(r["split_us"])}]})
    return {"schema": SCHEMA, "board": board, "run": run, "protocol": "graph_reps32_min3",
            "rows": rows}


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("from-rope-crossover")
    f.add_argument("csv")
    f.add_argument("--board", required=True)
    f.add_argument("--run", required=True)
    f.add_argument("--bound", type=int, default=256)
    c = sub.add_parser("check")
    c.add_argument("doc")
    args = ap.parse_args(argv)
    try:
        ids = registry_ids()
        if args.cmd == "from-rope-crossover":
            doc = from_rope_crossover(args.csv, args.board, args.run, args.bound)
            validate(doc, ids)
            json.dump(doc, sys.stdout, indent=1)
            print()
            return 0
        with open(args.doc, encoding="utf-8") as fh:
            doc = json.load(fh)
        validate(doc, ids)
        print(json.dumps(summarize(doc), indent=1))
        return 0
    except (ValueError, KeyError, OSError) as error:
        print(f"site_regret: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
