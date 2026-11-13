#!/usr/bin/env python3
"""Offline selection regret from qualified, matched lab measurements. See README.md.

Consumes recorded production selections; never selects routes, fits constants or runs a GPU.
"""
import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import mmq_transfer_audit as audit  # the ridge, as #605 computes it

UNKNOWN = "unknown"
# Regimes of a case by its width T, from architecture facts as #605 states them: decode T<=4 (the
# vector/R8 row range), verify 5..16, and from 17 the card's ridge T* splits the wide calls (below:
# the weight bytes dominate; above: the tensor rate does).
REGIMES = ("decode T<=4", "verify T5-16", "T>=17 below ridge", "T>=17 above ridge",
           "T>=17 ridge unknown", "unknown width")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def name(value, where):
    require(isinstance(value, str) and bool(value.strip()), f"{where}: expected nonempty string")
    # An identity with outer whitespace would compare unequal to the same name (a holdout leak).
    require(value == value.strip(), f"{where}: leading or trailing whitespace")
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


def width_of(case):
    """The case's T: its `t` field, else the id's `.T<n>` / `-t<n>` suffix, else None."""
    t = case.get("t")
    found = re.search(r"[.\-][Tt](\d+)$", str(case.get("id", "")))
    suffix = int(found[1]) if found else None
    if t is not None:
        require(type(t) is int and t >= 1, f"{case.get('id')}: t must be a positive integer")
        require(suffix is None or suffix == t,
                f"{case.get('id')}: t is {t} but the id ends in T{suffix}")
        return t
    return suffix


def regime_of(t, ridge):
    if t is None:
        return REGIMES[5]
    if t <= audit.DECODE_MAX:
        return REGIMES[0]
    if t <= audit.VERIFY_MAX:
        return REGIMES[1]
    if not ridge:
        return REGIMES[4]
    return REGIMES[2] if t < ridge else REGIMES[3]


def ridges_of(document, generation_header):
    """{board: ridge T*} for the boards the document gives `board_facts`, the way #605 computes it
    (int8 tensor rate of the generation x SMs x boost clock, over DRAM rate, for the 4.5-bit
    Q4_K): ops/s x bytes per value / (2 x DRAM bytes/s). A board with none has no ridge, nor has
    one without tensor cores (`mma_slow`, a GTX 16): the generation's tensor rate is not its."""
    require(isinstance(document, dict), "expected JSON object")
    # Absent is "no facts"; a value that is not an object (null, false, a list) is a bad document.
    facts = document.get("board_facts", {})
    require(isinstance(facts, dict), "board_facts: expected object")
    if not facts:
        return {}
    rows = audit.generations(generation_header)
    out = {}
    for board, f in facts.items():
        where = f"board_facts.{board}"
        require(isinstance(f, dict), f"{where}: expected object")
        for key in ("cc", "sms", "dram_gbps", "boost_mhz"):
            require(key in f, f"{where}: missing {key}")
            require(type(f[key]) in (int, float) and math.isfinite(f[key]) and f[key] > 0,
                    f"{where}.{key}: expected a finite positive number")
        require(type(f["cc"]) is int, f"{where}.cc: expected an integer")
        slow = f.get("mma_slow", False)
        require(type(slow) is bool, f"{where}.mma_slow: expected true or false")
        if slow:
            continue  # no tensor cores: the tensor ridge is not this board's, said as unknown
        try:
            generation = audit.generation_of(rows, f["cc"])
        except ValueError:
            continue  # a compute capability the generation table lacks: no ridge, said as such
        ridge = audit.ridge_t(generation, f["sms"], f["boost_mhz"], f["dram_gbps"])
        if ridge:
            out[board] = ridge
    return out


def summarize(rows):
    complete = [r for r in rows if r["status"] == "complete"]
    values = sorted(r["regret"] for r in complete)
    return {
        "cases": len(rows),
        "statuses": dict(sorted(Counter(r["status"] for r in rows).items())),
        "scored_cases": len(complete),
        "single_pass_cases": sum(1 for r in complete if r["passes"] == 1),
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
    # Which authority answered and on what evidence (#614); a case without them reads "unknown".
    for k in ("authority", "evidence"):
        result[k] = name(case[k], k) if k in case else UNKNOWN
    result["t"] = width_of(case)
    # The route the ranker alone picks, where the run recorded it: its time against the selected's
    # is what the exit tooling (route_table_prune.py) reads.
    if "ranker" in case:
        result["ranker"] = name(case["ranker"], "ranker")
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
    # Matched passes are part of the input's validity, whatever status the case ends with.
    pass_sets = {frozenset(t) for t in timings.values()}
    require(len(pass_sets) <= 1, f"{result['id']}: candidates must have the same measurement passes")
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
    medians = {k: statistics.median(t[p] for p in sorted(passes)) for k, t in timings.items()}
    for key, median in medians.items():
        positive(median, f"{key}: median")
    best = min(medians, key=lambda k: (medians[k], k))
    ratio = medians[selected] / medians[best]
    require(math.isfinite(ratio), f"{result['id']}: timing ratio overflow")
    # Compare each pass with one fixed best-by-median candidate, not a hindsight winner per pass.
    paired = [timings[selected][p] / timings[best][p] - 1 for p in sorted(passes)]
    require(all(math.isfinite(x) for x in paired), f"{result['id']}: paired ratio overflow")
    if result.get("ranker") in medians:
        result["ranker_vs_selected"] = medians[result["ranker"]] / medians[selected] - 1
    result.update(status="complete", best_observed=best, passes=len(passes),
                  median_ms=dict(sorted(medians.items())), regret=ratio - 1,
                  paired_regret_range=[min(paired), max(paired)])
    return result


def by_regime_authority(rows, ridges):
    """The report's groups: board x regime x authority, each summarised, with the evidence kinds
    its cases rested on."""
    groups = {}
    for r in rows:
        regime = regime_of(r["t"], ridges.get(r["board"]))
        groups.setdefault((r["board"], regime, r["authority"]), []).append(r)
    order = {name: i for i, name in enumerate(REGIMES)}
    return [{"board": b, "regime": g, "authority": a,
             "ridge_t": ridges.get(b),
             "evidence": dict(sorted(Counter(r["evidence"] for r in members).items())),
             **summarize(members)}
            for (b, g, a), members in sorted(groups.items(),
                                             key=lambda kv: (kv[0][0], order[kv[0][1]], kv[0][2]))]


# Which kind of answer a case's route was (#616 step 9), by the evidence it rested on:
#   table   - an override measured on this board (route table, dense tuner, r8 decode table, ...)
#   carried - an answer measured elsewhere (a borrowed table entry, a carried or transferred one)
#   priced  - the cost model's ranking
#   rules   - a rule or fallback of the selector (`ranker` evidence, any other authority)
#   pin     - a pinned route; unknown - a row without authority/evidence (an older collection)
ANSWER_KINDS = ("table", "carried", "priced", "rules", "pin", UNKNOWN)
# The override authorities route_table_prune.py prunes: a measured answer of these replaces the
# ranker's, so its ranker_vs_selected says whether the override still earns its place.
OVERRIDE_AUTHORITIES = ("route-table", "dense-tune", "dense-tune-prefill-band")
# The ranker's slowdown against an override within which the override earns nothing (5 %; the
# default of route_table_prune.py's --resolution, which reads this one).
EXIT_RESOLUTION = 0.05


def answer_kind(row):
    evidence = row["evidence"]
    if evidence == "measured":
        return "table"
    if evidence in ("borrowed", "carried", "transferred"):
        return "carried"
    if evidence == "pin":
        return "pin"
    if evidence == "ranker":
        return "priced" if row["authority"] == "cost-model" else "rules"
    return UNKNOWN


def table_exit(rows, resolution=EXIT_RESOLUTION):
    """The table exit metric, per board (#616 step 9): how the board's calls were answered
    (ANSWER_KINDS: case counts and shares), and of its measured overrides (OVERRIDE_AUTHORITIES)
    how many the ranker has caught up with: the ranker's route within `resolution` of the
    override's (`caught_up`: route_table_prune.py may delete them), still slower (`holds`), or no
    ranker time in the case (`unscored`: the run did not time the ranker's route, or the case is
    incomplete). `exit_share` = caught_up / (caught_up + holds), None with no scored override."""
    out = {}
    for board in sorted({r["board"] for r in rows}):
        members = [r for r in rows if r["board"] == board]
        kinds = Counter(answer_kind(r) for r in members)
        overrides = [r for r in members
                     if r["authority"] in OVERRIDE_AUTHORITIES and r["evidence"] == "measured"]
        scored = [r for r in overrides if r["status"] == "complete" and "ranker_vs_selected" in r]
        caught = sum(1 for r in scored if r["ranker_vs_selected"] <= resolution)
        out[board] = {
            "cases": len(members),
            "answered_by": {k: kinds[k] for k in ANSWER_KINDS if kinds[k]},
            "share": {k: kinds[k] / len(members) for k in ANSWER_KINDS if kinds[k]},
            "overrides": len(overrides),
            "caught_up": caught,
            "holds": len(scored) - caught,
            "unscored": len(overrides) - len(scored),
            "exit_share": caught / len(scored) if scored else None,
            "resolution": resolution,
        }
    return out


def evaluate(document, max_regret=None, ridges=None, exit_resolution=EXIT_RESOLUTION):
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
        require(row["id"] not in seen, f"duplicate case identity: {row['id']}")
        seen.add(row["id"])
        require(holdout not in ("board", "both") or row["board"] not in fit_boards,
                f"{row['id']}: held-out board is in fit_boards")
        require(holdout not in ("model", "both") or row["model"] not in fit_models,
                f"{row['id']}: held-out model is in fit_models")
        rows.append(row)
    rows.sort(key=lambda r: (r["board"], r["model"], r["id"]))
    # Compare slowdown factors: subtracting one first can turn 1.1 into a regret just above 0.1.
    violations = [r["id"] for r in rows if r["status"] != "complete" or
                  (max_regret is not None and 1 + r["regret"] > 1 + max_regret)]
    # The measured device (route_tune's class id; regret_collect drops the card's UUID), passed on:
    # route_table_prune.py matches it to the table it prunes.
    if "device" in document:
        provenance["device"] = name(document["device"], "device")
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
        "by_board_regime_authority": by_regime_authority(rows, ridges or {}),
        "table_exit": table_exit(rows, exit_resolution),
        "cases": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--max-regret", type=float, help="maximum slowdown minus one, e.g. 0.05")
    parser.add_argument("--generation-header", type=Path,
                        default=Path(__file__).resolve().parents[2] / "src/core/generation.h",
                        help="core/generation.h, for the int8 rate of the ridge")
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        document = json.loads(raw, object_pairs_hook=unique_object)
        report = evaluate(document, args.max_regret, ridges_of(document, args.generation_header))
        report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    except (OSError, UnicodeError, ValueError, OverflowError, RecursionError) as error:
        print(f"selection_regret: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
