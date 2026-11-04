#!/usr/bin/env python3
"""Lab collector for selection_regret.py (#301): route sweeps and the production selection.

Combines ninferno_route_tune's CSVs of one board into one ninferno-selection-regret/1 document:
  --selected  each pass's --selected-out CSV: per (problem, T), the route the production path runs;
  --routes    one --routes-out CSV per measurement pass (interleave whole runs, same build and board).
Every (problem, T) of the selected CSV becomes a case whose candidates are the routes timed for it.
A route timed in some passes and not others is `unmeasured`; nothing timed is dropped silently.

route_tune does not check results. A candidate is `ok` only with --qualified, which states that this
build's route correctness tests (ninferno_linear_generic_test, ninferno_route_classes_test) passed
on this board; the statement is copied into the document. Without it every candidate is
`unmeasured` and the evaluator reports the cases incomplete.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

KEY = ("format", "compute", "epilogue", "n", "k", "T")

# Route families by the activations they compute with: a case compares only routes of its compute
# class. route_tune also times ggml's matrix kernel under an a16 key (the default policy runs it at
# verify widths), which computes int8 activations.
INT8_FAMILIES = ("ggml_mmq", "ggml_vec", "r8.", "w4a8", "i8.")
FP4_FAMILIES = ("w4a4",)


def route_class(route):
    if route.startswith(INT8_FAMILIES):
        return "a8"
    if route.startswith(FP4_FAMILIES):
        return "a4"
    return "a16"


def read_rows(path, columns):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in columns if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        return list(reader)


def key_of(row):
    return tuple(row[c].strip() for c in KEY)


def case_id(key):
    fmt, compute, epilogue, n, k, t = key
    return f"{fmt}.{compute}.{epilogue}.n{n}.k{k}.T{t}"


# The policy route_tune selects with for a compute class (its problems' policy), and the route
# profile it runs: the speed profile (prefill_routes inactive), so a T >= 17 case is a batched
# decode call, not a prefill chunk. The default Fast mode runs a published model's non-NVFP4
# projections AllowA8 (the vector kernel, R8, W4A8 ahead of the a16 routes): an a16 case is the
# A16Only path, the exact mode's linears.
POLICY = {"a16": "A16Only", "a8": "AllowA8", "a4": "AllowA4"}


def read_selections(paths):
    selected = None
    for path in paths:
        run = {}
        for row in read_rows(path, KEY + ("selected",)):
            key = key_of(row)
            route = row["selected"].strip()
            if key in run and run[key] != route:
                raise ValueError(f"{path}: {case_id(key)} selected twice ({run[key]}, {route})")
            run[key] = route
        if selected is not None and run != selected:
            raise ValueError(f"{path}: the selection differs from the first pass's")
        selected = run
    return selected


def collect(selected_csvs, pass_csvs, board, model, revision, build_id, holdout, fit_boards,
            fit_models, qualified):
    if not pass_csvs:
        raise ValueError("at least one --routes pass")
    if isinstance(selected_csvs, (str, Path)):
        selected_csvs = [selected_csvs]
    selected = read_selections(selected_csvs)
    passes = {}
    for i, path in enumerate(pass_csvs, 1):
        timed = {}
        for row in read_rows(path, KEY + ("route", "median_us")):
            us = float(row["median_us"])
            at = (key_of(row), row["route"].strip())
            # route_tune times a (route, T) once per run: a repeat is two runs in one file.
            if at in timed:
                raise ValueError(f"{path}: {case_id(at[0])} {at[1]} timed twice in one pass")
            timed[at] = us
        passes[f"p{i}"] = timed
    cases = []
    for key in sorted(selected, key=lambda k: (k[:5], int(k[5]))):
        routes = sorted({r for timed in passes.values() for (k, r) in timed if k == key})
        candidates = []
        for route in routes:
            samples = {p: timed[(key, route)] / 1000.0 for p, timed in passes.items()
                       if (key, route) in timed}
            if key[1] == "a16" and route_class(route) != "a16":
                candidates.append({"id": route, "status": "ineligible",
                                   "reason": f"{route_class(route)} activations, not this case's a16"})
            elif len(samples) != len(passes):
                missing = sorted(set(passes) - set(samples))
                candidates.append({"id": route, "status": "unmeasured",
                                   "reason": "not timed in pass " + ", ".join(missing)})
            elif not qualified:
                candidates.append({"id": route, "status": "unmeasured",
                                   "reason": "no correctness qualification stated (--qualified)"})
            else:
                candidates.append({"id": route, "status": "ok", "samples_ms": samples})
        if selected[key] not in routes:
            candidates.append({"id": selected[key], "status": "unmeasured",
                               "reason": "the selected route was not timed"})
        fmt, compute, epilogue, n, k, t = key
        cases.append({
            "id": case_id(key), "board": board, "model": model,
            "workload": f"route_tune {fmt} {compute} {epilogue} n={n} k={k} T={t}, policy "
                        f"{POLICY.get(compute, compute)}, speed route profile, L2 flushed, median "
                        "of the run's samples",
            "numerical_contract": f"{compute} activations under {POLICY.get(compute, compute)}",
            "selected": selected[key], "candidates": candidates,
        })
    if not cases:
        raise ValueError("no selections")
    document = {
        "schema": "ninferno-selection-regret/1", "selector_revision": revision,
        "build_id": build_id, "holdout": holdout, "fit_boards": sorted(set(fit_boards)),
        "fit_models": sorted(set(fit_models)), "cases": cases,
    }
    if qualified:
        document["qualification"] = qualified
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selected", type=Path, nargs="+", required=True,
                        help="each pass's --selected-out; they must agree")
    parser.add_argument("--routes", type=Path, nargs="+", required=True)
    parser.add_argument("--board", required=True)
    parser.add_argument("--model", required=True, help="the checkpoint the shapes come from")
    parser.add_argument("--selector-revision", required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--holdout", choices=("board", "model", "both"), required=True)
    parser.add_argument("--fit-boards", default="", help="comma-separated")
    parser.add_argument("--fit-models", default="", help="comma-separated")
    parser.add_argument("--qualified", default="",
                        help="the correctness tests that passed on this build and board")
    args = parser.parse_args(argv)
    split = lambda s: [x for x in (v.strip() for v in s.split(",")) if x]
    try:
        document = collect(args.selected, args.routes, args.board, args.model,
                           args.selector_revision, args.build_id, args.holdout,
                           split(args.fit_boards), split(args.fit_models), args.qualified)
    except (OSError, ValueError, KeyError, AttributeError, TypeError) as error:  # a short row
        print(f"regret_collect: {error}", file=sys.stderr)
        return 2
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
