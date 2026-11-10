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
A site id the registry does not have, a site with no `regret_branches` in the registry, a candidate
branch outside them, a decided branch that is not a candidate, a candidate time that is not finite
and positive, a non-object row or candidate, a non-integer T (a bool is not one), or one case twice,
is refused: nothing is dropped silently. The manifest side of the same sites is
ops/common/fusion_site_manifest.h (its line says the branch a board takes at a width; this document
says what that branch cost against the other).

  site_regret.py from-rope-crossover CSV --board B --run R [--bound 256]   > doc.json
      the `--crossover` CSV of ninferno_rmsnorm_rope_bench (geometry,T,fused_us,split_us,...): the
      qk_norm_rope site. The CSV's header lines carry what the run measured with: `graph_reps=N`
      (the protocol) and `dispatch_bound <geometry>=<tokens>` (the width the dispatch fuses up to on
      that board). The decided branch is fused for T <= that bound, decomposed above it; --bound
      overrides it for every geometry. A CSV without either header field is refused.
  site_regret.py check doc.json
      validates a document against the registry and prints per-site regret statistics.

Offline: never selects, fits or runs a GPU.
"""
import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

SCHEMA = "ninferno-site-regret/1"
ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "tools" / "audit" / "fusion_sites.json"


def registry_branches(path=REGISTRY):
    """{site id: set of regret branch names, or None when the site declares none}."""
    with open(path, encoding="utf-8") as f:
        return {s["id"]: set(s["regret_branches"]) if "regret_branches" in s else None
                for s in json.load(f)["sites"]}


def registry_ids(path=REGISTRY):
    return set(registry_branches(path))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(doc, ids, branches=None):
    """`ids`: the registry's site ids; `branches`: {id: allowed branch names or None} (the registry's
    `regret_branches`); without it only the ids are checked."""
    require(isinstance(doc, dict) and doc.get("schema") == SCHEMA, f"schema must be {SCHEMA}")
    for key in ("board", "run", "protocol"):
        require(isinstance(doc.get(key), str) and doc[key].strip(), f"{key}: expected nonempty string")
    rows = doc.get("rows")
    require(isinstance(rows, list) and rows, "rows: expected a nonempty list")
    seen = set()
    for i, row in enumerate(rows):
        where = f"rows[{i}]"
        require(isinstance(row, dict), f"{where}: expected an object")
        site = row.get("site")
        require(isinstance(site, str) and site in ids, f"{where}: site {site!r} is not in the registry")
        require(isinstance(row.get("case"), str) and row["case"], f"{where}: case")
        t = row.get("T")
        require(type(t) is int and t > 0, f"{where}: T must be a positive integer")
        key = (site, row["case"], t)
        require(key not in seen, f"{where}: {key} twice")
        seen.add(key)
        cands = row.get("candidates")
        require(isinstance(cands, list) and len(cands) >= 2, f"{where}: at least two candidates")
        branch_names = []
        for j, c in enumerate(cands):
            require(isinstance(c, dict), f"{where}.candidates[{j}]: expected an object")
            name = c.get("branch")
            require(isinstance(name, str) and name, f"{where}.candidates[{j}]: branch")
            us = c.get("median_us")
            require(type(us) in (int, float) and math.isfinite(us) and us > 0,
                    f"{where}: {name}: median_us must be finite and positive")
            branch_names.append(name)
        require(len(set(branch_names)) == len(branch_names), f"{where}: duplicate candidate")
        require(row.get("decided") in branch_names, f"{where}: decided branch is not a candidate")
        if branches is not None:
            allowed = branches.get(site)
            require(allowed is not None, f"{where}: site {site} declares no regret_branches")
            require(set(branch_names) <= allowed,
                    f"{where}: branches {sorted(set(branch_names) - allowed)} are not in "
                    f"{site}'s regret_branches {sorted(allowed)}")


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


def read_header(lines):
    """(graph_reps, {geometry: dispatch bound}) from the CSV's `#` lines."""
    graph_reps, bounds = None, {}
    for ln in lines:
        if not ln.startswith("#"):
            continue
        m = re.search(r"\bgraph_reps=(\d+)", ln)
        if m:
            graph_reps = int(m.group(1))
        m = re.search(r"\bdispatch_bound (\d+\+\d+)=(\d+)", ln)
        if m:
            bounds[m.group(1)] = int(m.group(2))
    return graph_reps, bounds


def from_rope_crossover(csv_path, board, run, bound=None):
    with open(csv_path, newline="", encoding="utf-8") as f:
        raw = [ln for ln in f if ln.strip()]
    graph_reps, bounds = read_header(raw)
    if graph_reps is None:
        raise ValueError(f"{csv_path}: no graph_reps= in the header (not a --crossover CSV)")
    rows = []
    for r in csv.DictReader(ln for ln in raw if not ln.startswith("#")):
        t = int(r["T"])
        limit = bound if bound is not None else bounds.get(r["geometry"])
        if limit is None:
            raise ValueError(f"{csv_path}: no dispatch_bound for {r['geometry']} and no --bound")
        rows.append({
            "site": "qk_norm_rope", "case": r["geometry"], "T": t,
            "decided": "fused" if t <= limit else "decomposed",
            "candidates": [{"branch": "fused", "median_us": float(r["fused_us"])},
                           {"branch": "decomposed", "median_us": float(r["split_us"])}]})
    return {"schema": SCHEMA, "board": board, "run": run,
            "protocol": f"graph_reps{graph_reps}_min3", "rows": rows}


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("from-rope-crossover")
    f.add_argument("csv")
    f.add_argument("--board", required=True)
    f.add_argument("--run", required=True)
    f.add_argument("--bound", type=int, default=None)
    c = sub.add_parser("check")
    c.add_argument("doc")
    args = ap.parse_args(argv)
    try:
        branches = registry_branches()
        ids = set(branches)
        if args.cmd == "from-rope-crossover":
            doc = from_rope_crossover(args.csv, args.board, args.run, args.bound)
            validate(doc, ids, branches)
            json.dump(doc, sys.stdout, indent=1)
            print()
            return 0
        with open(args.doc, encoding="utf-8") as fh:
            doc = json.load(fh)
        validate(doc, ids, branches)
        print(json.dumps(summarize(doc), indent=1))
        return 0
    except (ValueError, KeyError, OSError) as error:
        print(f"site_regret: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
