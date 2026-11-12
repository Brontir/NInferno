#!/usr/bin/env python3
"""The measured prediction of the route policy's A/B (#614 F1), before the run.

Reads tools/bench/data/route_regret/policy_evidence_moves.json (ninferno_route_policy_test --write)
and the A5 sweep's route_tune --routes-out CSVs of one board (docs/maintainer/route-regret.md;
the CSVs stay on the stand, docs/maintainer/sweep-archive.md). For every moved cell of that board
(speed profile; the prefill chunk's rows carry no profile column and are left out) it takes the
median over the passes of each route's median_us at the cell, and gives a5_ratio = t(evidence
route) / t(legacy route): the ratio the A/B's B/A should come out at if the sweep's timings hold.
A cell whose two routes are not both timed in every pass gets a5_ratio null. The A5 sweep ran
AllowA8 only (--a8-candidates 1, compute a8), so an A16Only cell is matched only by an A16 sweep's
rows (compute a16) given in --routes; a move of a compute class the CSVs hold no row of, and a
prefill-profile move, is left untouched, so the two classes' runs do not reset each other.

  policy_moves_predict.py --moves MOVES.json --board rtx4090 [--tables own|without_own]
                          --routes routes_p1.csv routes_p2.csv routes_p3.csv [--write]

Prints one line per moved cell and a summary; --write stores a5_ratio in MOVES.json (a later
ninferno_route_policy_test --write resets it to null: run this again after it). Offline: it times
nothing and runs no GPU.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


def cell_key(fmt, compute, epilogue, n, k, t):
    return (fmt, compute, epilogue, int(n), int(k), int(t))


def read_passes(paths):
    """Per pass: (format, compute, epilogue, n, k, T) -> route -> median_us."""
    passes = []
    for path in paths:
        times = {}
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    us = float(row["median_us"])
                except (KeyError, ValueError):
                    continue  # a killed sweep's last row, or a route that did not run
                key = cell_key(row["format"].strip(), row["compute"].strip(),
                               row["epilogue"].strip(), row["n"], row["k"], row["T"])
                times.setdefault(key, {})[row["route"].strip()] = us
        passes.append(times)
    return passes


def median_time(passes, key, route):
    values = [p.get(key, {}).get(route) for p in passes]
    if not values or any(v is None for v in values):
        return None
    return statistics.median(values)


def compute_classes(passes):
    """The compute classes (a16, a8, ...) the passes hold rows of."""
    return {key[1] for times in passes for key in times}


def predict(moves_doc, board, tables, passes):
    """Fills a5_ratio of the board's moved cells the passes speak for; returns (move, ratio) pairs.

    A move whose compute class no pass holds (an a8 cell under an A16 sweep's CSVs, or the other
    way round) and a prefill-profile move are left as they are, their a5_ratio untouched: a run of
    one class's CSVs never resets the other class's ratios.
    """
    classes = compute_classes(passes)
    out = []
    for entry in moves_doc["boards"]:
        if entry["board"] != board or entry.get("tables", "own") != tables:
            continue
        for move in entry["moves"]:
            if move["profile"] != "speed" or move["policy"] not in classes:
                continue
            key = cell_key(move["format"], move["policy"], move["epilogue"], move["n"], move["k"],
                           move["T"])
            old = median_time(passes, key, move["legacy"]["route"])
            new = median_time(passes, key, move["evidence"]["route"])
            ratio = new / old if old and new else None
            move["a5_ratio"] = ratio
            out.append((move, ratio))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--moves", required=True, type=Path)
    ap.add_argument("--board", required=True)
    ap.add_argument("--tables", default="own", choices=("own", "without_own"))
    ap.add_argument("--routes", required=True, nargs="+", type=Path)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    doc = json.loads(args.moves.read_text(encoding="utf-8"))
    passes = read_passes(args.routes)
    pairs = predict(doc, args.board, args.tables, passes)
    if not pairs:
        print(f"{args.board} {args.tables}: no moved speed cell of the CSVs' compute classes "
              f"({', '.join(sorted(compute_classes(passes))) or 'none'}) in {args.moves}",
              file=sys.stderr)
        return 2
    known = [r for _, r in pairs if r is not None]
    for move, ratio in pairs:
        text = "-" if ratio is None else f"{ratio:.3f}"
        print(f"{move['format']} {move['policy']} {move['epilogue']} {move['n']}x{move['k']} "
              f"T={move['T']}: {move['legacy']['route']} -> {move['evidence']['route']} "
              f"a5_ratio {text} model_ratio {move.get('model_ratio')}")
    if known:
        print(f"{args.board} {args.tables}: {len(known)} of {len(pairs)} cells timed; median "
              f"a5_ratio {statistics.median(known):.3f}, worst {max(known):.3f}")
    else:
        print(f"{args.board} {args.tables}: none of {len(pairs)} cells timed in these passes")
    if args.write:
        args.moves.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
