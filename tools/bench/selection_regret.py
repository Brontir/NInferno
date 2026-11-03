#!/usr/bin/env python3
"""Offline selection regret from qualified, matched lab measurements. See README.md.

Consumes recorded production selections; never selects routes, fits constants or runs a GPU.
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def name(value, where):
    require(isinstance(value, str) and bool(value.strip()), f"{where}: expected nonempty string")
    return value


def strings(value, where):
    require(isinstance(value, list), f"{where}: expected list")
    result = [name(v, where) for v in value]
    require(len(result) == len(set(result)), f"{where}: duplicate identity")
    return set(result)


def positive(value, where):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            f"{where}: expected finite positive milliseconds")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def summarize(rows):
    complete = [r for r in rows if r["status"] == "complete"]
    values = sorted(r["regret"] for r in complete)
    return {
        "cases": len(rows),
        "statuses": dict(sorted(Counter(r["status"] for r in rows).items())),
        "scored_cases": len(complete),
        "geomean_slowdown": math.exp(math.fsum(math.log1p(x) for x in values) / len(values))
        if values else None,
        "median_regret": statistics.median(values) if values else None,
        "p95_regret": values[math.ceil(0.95 * len(values)) - 1] if values else None,
        "max_regret": values[-1] if values else None,
    }


def score_case(case):
    require(isinstance(case, dict), "case: expected object")
    result = {k: name(case.get(k), k) for k in
              ("id", "board", "model", "workload", "numerical_contract", "selected")}
    candidates = case.get("candidates")
    require(isinstance(candidates, list) and candidates, f"{result['id']}: no candidates")
    seen, timings, excluded, unavailable = set(), {}, {}, {}
    for row in candidates:
        require(isinstance(row, dict), f"{result['id']}: candidate must be an object")
        key = name(row.get("id"), "candidate.id")
        require(key not in seen, f"{result['id']}: duplicate candidate {key}")
        seen.add(key)
        status = row.get("status")
        require(status in ("ok", "ineligible", "oracle_failed", "unmeasured"),
                f"{key}: invalid status")
        if status != "ok":
            reason = name(row.get("reason"), f"{key}.reason")
            require("samples_ms" not in row, f"{key}: non-ok candidate must not supply timings")
            excluded[key] = {"status": status, "reason": reason}
            if status == "unmeasured":
                unavailable[key] = reason
            continue
        samples = row.get("samples_ms")
        require(isinstance(samples, dict) and samples, f"{key}: no measurements")
        timings[key] = {name(p, "pass"): positive(t, f"{key}/{p}")
                        for p, t in samples.items()}
    selected = result["selected"]
    require(selected in seen, f"{result['id']}: selected candidate is absent from inventory")
    result["candidate_count"] = len(seen)
    result["qualified_measured_count"] = len(timings)
    result["excluded"] = dict(sorted(excluded.items()))
    if selected not in timings:
        result["status"] = "incomplete" if selected in unavailable else "invalid_selection"
        return result
    # A missing alternative must not disappear from the denominator and make the case look good.
    if unavailable:
        result["status"] = "incomplete"
        return result
    if len(timings) < 2:
        result["status"] = "no_comparison"
        return result
    passes = set(timings[selected])
    require(all(set(t) == passes for t in timings.values()),
            f"{result['id']}: candidates must have the same measurement passes")
    medians = {k: statistics.median(t[p] for p in sorted(passes)) for k, t in timings.items()}
    for key, median in medians.items():
        positive(median, f"{key}: median")
    best = min(medians, key=lambda k: (medians[k], k))
    ratio = medians[selected] / medians[best]
    require(math.isfinite(ratio), f"{result['id']}: timing ratio overflow")
    # Compare each pass with one fixed best-by-median candidate, not a hindsight winner per pass.
    paired = [timings[selected][p] / timings[best][p] - 1 for p in sorted(passes)]
    require(all(math.isfinite(x) for x in paired), f"{result['id']}: paired ratio overflow")
    result.update(status="complete", best_observed=best, passes=len(passes),
                  median_ms=dict(sorted(medians.items())), regret=ratio - 1,
                  paired_regret_range=[min(paired), max(paired)])
    return result


def evaluate(document, max_regret=None):
    require(isinstance(document, dict), "expected JSON object")
    require(document.get("schema") == "ninferno-selection-regret/1", "unsupported schema")
    provenance = {k: name(document.get(k), k) for k in ("selector_revision", "build_id")}
    holdout = document.get("holdout")
    require(holdout in ("board", "model", "both"), "holdout must be board, model or both")
    fit_boards = strings(document.get("fit_boards"), "fit_boards")
    fit_models = strings(document.get("fit_models"), "fit_models")
    cases = document.get("cases")
    require(isinstance(cases, list) and cases, "no evaluation cases")
    if max_regret is not None:
        require(type(max_regret) in (int, float) and math.isfinite(max_regret) and max_regret >= 0,
                "max_regret must be finite and nonnegative")
    rows, seen = [], set()
    for case in cases:
        row = score_case(case)
        key = (row["board"], row["model"], row["id"])
        require(key not in seen, f"duplicate case identity: {key}")
        seen.add(key)
        require(holdout not in ("board", "both") or row["board"] not in fit_boards,
                f"{row['id']}: held-out board is in fit_boards")
        require(holdout not in ("model", "both") or row["model"] not in fit_models,
                f"{row['id']}: held-out model is in fit_models")
        rows.append(row)
    rows.sort(key=lambda r: (r["board"], r["model"], r["id"]))
    # Compare slowdown factors: subtracting one first can turn 1.1 into a regret just above 0.1.
    violations = [r["id"] for r in rows if r["status"] != "complete" or
                  (max_regret is not None and 1 + r["regret"] > 1 + max_regret)]
    return {
        "schema": "ninferno-selection-regret-report/1", **provenance,
        "holdout": holdout, "fit_boards": sorted(fit_boards), "fit_models": sorted(fit_models),
        "scope": "best observed qualified candidate in the supplied inventory; not a global optimum",
        "max_regret": max_regret, "passed": not violations,
        "summary": summarize(rows),
        "by_board": {k: summarize([r for r in rows if r["board"] == k])
                     for k in sorted({r["board"] for r in rows})},
        "by_model": {k: summarize([r for r in rows if r["model"] == k])
                     for k in sorted({r["model"] for r in rows})},
        "cases": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--max-regret", type=float, help="maximum slowdown minus one, e.g. 0.05")
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        report = evaluate(json.loads(raw, object_pairs_hook=unique_object), args.max_regret)
        report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    except (OSError, UnicodeError, ValueError, OverflowError) as error:
        print(f"selection_regret: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
