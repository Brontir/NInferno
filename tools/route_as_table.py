#!/usr/bin/env python3
"""route_as_table.py: route_as entries for a GPU class from ninferno_linear_route_sweep_bench runs.

    tools/route_as_table.py --gpu-class gb203 --gpu-name "NVIDIA GeForce RTX 5080" \
        --pass DIR1 --pass DIR2 [--min-gain 5] [--min-us 2.1] [--merge routes/gb203.json] \
        [--exclude-format fp8] --out routes/gb203.json

Each --pass is a directory of the bench's CSVs (one per shape), all measured on one card with
--random-weight --no-class-table. For every shape and swept T the tool keeps the candidates
"route as T'" whose output equals the dispatcher's bit for bit (rel_l2 == 0 with random weights
and wide activations: the same K order, so the same numerics; ops/common/shape_routes.h) and
that beat the dispatcher by at least --min-gain percent and --min-us microseconds in EVERY pass
(the 5080's event timer resolves ~2 us); those within --tie percent of the fastest are equals
(several T' often name one compiled route). Consecutive swept T are grouped greedily into bands
[first, last, T'] with the smallest T' every T of the band accepts.

Calls below --min-t (default 2) are left to the table: a single-token decode step runs inside a
captured graph with the next kernels overlapping it, where a cold-L2 single-call timing does not
predict it (on the RTX 5080 the T = 1 overrides the bench ranked 5-10 % faster cost Gemma 4 26B's
decode 0.5 %; without them decode was within 0.1 % of the table-free run).

The table consulted at T is the shape's A16 or its A4/A8 one, as the shape itself decides (its
uses_a4/uses_a8 threshold is left alone: that switch changes numerics); the tool finds which by
the forced-class rows of --quantized both (the forced row whose output equals the dispatcher's).

--merge keeps every entry of an existing class file that this run does not produce (for example
the generic path's "bands" entries from ninferno_route_tune) and replaces route_as entries of the
same problem. The output is ninferno-route-table/1; keys other than the loaders' are evidence.
"""
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

FORMATS = {"Q4": "q4", "Q5": "q5", "Q6": "q6", "Q8": "q8", "BF16": "bf16", "NVFP4": "nvfp4",
           "FP8": "fp8"}


def fmt_name(q):
    q = q.upper()
    for key, name in FORMATS.items():
        if q.startswith(key):
            return name
    return q.lower()


def load_pass(directory):
    """{(format, n, k): {t: {"d": us, "class": a16|a4|a8, "cand": {as: (us, exact)}}}}"""
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        if not rows:
            continue
        key = (fmt_name(rows[0]["qtype"]), int(rows[0]["n"]), int(rows[0]["k"]))
        policy = rows[0]["policy"].lower()
        per_t = out.setdefault(key, {})
        for r in rows:
            t = int(r["t"])
            e = per_t.setdefault(t, {"d": None, "class": "a16", "cand": {}, "forced": {}})
            status = r["status"]
            med = float(r["median_us"]) if r["median_us"] not in ("", "nan") else math.nan
            err = float(r["rel_l2"]) if r["rel_l2"] not in ("", "nan") else math.nan
            if status == "dispatcher":
                e["d"] = med
            elif r["quantized"] != "-1":
                e["forced"][int(r["quantized"])] = err
            elif status in ("ok", "mismatch") and not math.isnan(med):
                e["cand"][int(r["route_as"])] = (med, err == 0.0)
        quantized_class = {"a4": "a4", "a8": "a8"}.get(policy)
        for t, e in per_t.items():
            # The forced row that reproduces the dispatcher's output names the table it asked.
            if quantized_class and e["forced"].get(1) == 0.0 and e["forced"].get(0) != 0.0:
                e["class"] = quantized_class
            elif quantized_class and e["forced"].get(0) == 0.0:
                e["class"] = "a16"
            elif quantized_class:
                e["class"] = None  # undecided: skip this T
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-class", required=True)
    ap.add_argument("--gpu-name", default="")
    ap.add_argument("--pass", dest="passes", action="append", required=True)
    ap.add_argument("--min-gain", type=float, default=5.0)
    ap.add_argument("--min-us", type=float, default=2.1)
    ap.add_argument("--tie", type=float, default=3.0)
    ap.add_argument("--min-t", type=int, default=2)
    ap.add_argument("--exclude-format", action="append", default=[])
    ap.add_argument("--merge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report")
    args = ap.parse_args()

    passes = [load_pass(d) for d in args.passes]
    keys = set(passes[0])
    for p in passes[1:]:
        keys &= set(p)
    entries = []
    report = []
    for key in sorted(keys):
        fmt, n, k = key
        if fmt in args.exclude_format:
            continue
        # Per swept T: every acceptable candidate (bit-exact, and at least --min-gain % and
        # --min-us faster than the table's own choice in every pass) whose summed time is within
        # --tie % of the best one: several T' often name the same compiled route.
        accept = {}  # (class, t) -> {as: (mean table us, mean override us)}
        ts = sorted(set.intersection(*[set(p[key]) for p in passes]))
        for t in ts:
            if t < args.min_t:
                continue
            es = [p[key][t] for p in passes]
            cls = es[0]["class"]
            if cls is None or any(e["class"] != cls for e in es) or any(e["d"] is None for e in es):
                continue
            ok = {}
            for cand in set.intersection(*[set(e["cand"]) for e in es]):
                good = True
                for e in es:
                    med, exact = e["cand"][cand]
                    if (not exact or 100.0 * (1.0 - med / e["d"]) < args.min_gain
                            or e["d"] - med < args.min_us):
                        good = False
                        break
                if good:
                    ok[cand] = (sum(e["d"] for e in es) / len(es),
                                sum(e["cand"][cand][0] for e in es) / len(es))
            if ok:
                best = min(v[1] for v in ok.values())
                accept[(cls, t)] = {c: v for c, v in ok.items()
                                    if v[1] <= best * (1.0 + args.tie / 100.0)}
        by_class = defaultdict(list)
        for (cls, t), cands in accept.items():
            by_class[cls].append((t, cands))
        for cls, items in sorted(by_class.items()):
            items.sort()
            # Greedy bands over consecutive swept T: keep the candidates every T of the band
            # accepts; close the band when none is left, taking the smallest such T'.
            bands, evidence = [], []
            run, common = [], None
            def close():
                if run:
                    as_t = min(common)
                    bands.append([run[0][0], run[-1][0], as_t])
                    for t, cands in run:
                        d, b = cands[as_t]
                        evidence.append([t, round(d, 2), round(b, 2)])
                        report.append((fmt, cls, n, k, t, as_t, d, b, 100.0 * (1 - b / d)))
            for t, cands in items:
                contiguous = run and ts.index(t) - 1 == ts.index(run[-1][0])
                inter = (common & set(cands)) if contiguous else set()
                if contiguous and inter:
                    run.append((t, cands))
                    common = inter
                else:
                    close()
                    run, common = [(t, cands)], set(cands)
            close()
            entries.append({"format": fmt, "compute": cls, "epilogue": "store", "n": n, "k": k,
                            "route_as": bands,
                            "evidence_us": evidence})

    root = {"schema": "ninferno-route-table/1",
            "gpu": {"class": args.gpu_class, "name": args.gpu_name},
            "entries": []}
    produced = {(e["format"], e["compute"], e["n"], e["k"]) for e in entries}
    if args.merge and os.path.exists(args.merge):
        old = json.load(open(args.merge, encoding="utf-8"))
        for e in old.get("entries", []):
            key = (e.get("format"), e.get("compute"), e.get("n"), e.get("k"))
            if "route_as" in e and key in produced:
                rest = {k2: v for k2, v in e.items() if k2 not in ("route_as", "evidence_us")}
                if "bands" in rest:
                    root["entries"].append(rest)
                continue
            root["entries"].append(e)
    root["entries"].extend(entries)
    root["note"] = ("route_as: for a call of t_min..t_max tokens the shape-owned table is asked "
                    "at T' (ops/common/shape_routes.h); measured with "
                    "ninferno_linear_route_sweep_bench --random-weight --no-class-table, every "
                    "override bit-identical to the table's own choice; evidence_us rows are "
                    "[T, table us, override us], mean over %d passes; min gain %.1f%%, %.1f us." %
                    (len(passes), args.min_gain, args.min_us))
    with open(args.out, "w", newline="\n", encoding="utf-8") as f:
        f.write(json.dumps(root, indent=1) + "\n")
    if args.report:
        with open(args.report, "w", newline="\n", encoding="utf-8") as f:
            f.write("| format | table | N | K | T | as T' | table us | override us | gain % |\n")
            f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for r in report:
                f.write("| %s | %s | %d | %d | %d | %d | %.1f | %.1f | %.1f |\n" % r)
    print("%d entries, %d (shape, T) points" % (len(entries), len(report)))


if __name__ == "__main__":
    main()
