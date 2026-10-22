#!/usr/bin/env python3
"""Lab gate: automatic speculation must not be slower than plain decoding.

    python3 tools/spec_gate.py [--bench docs/performance/data/bench.jsonl] [--threshold 0.97]
                               [--since-commit REV | --since-date ISO] [--losses] [--json]

Every decode row measured with the automatic speculation decision (no `--spec` in its flags) is
paired with its plain twin (`--spec none`): the same card, model, recipe, KV format, activation
precision, sampling, prompt set, prompt length, concurrency, metric and every other flag. Prefill
rows are left out (speculation does not decide the prefill), and so are rows that name a drafter
explicitly. When a key was measured more than once, the newest row of each side is used.

A cell FAILS when all of these hold:
  * the decision speculates (the auto row's served `spec` is not `none`);
  * auto < threshold x plain;
  * the spreads do not overlap (a row without a spread counts as the single value);
  * the auto row is newer than the decision code: its `commit` descends from the last commit that
    touched DECISION_FILES, or, for rows without a commit, its date is later than that commit's.
Losing cells that miss one of these are counted as plain-served, overlapping or stale.

There is no exemption list and nothing is written back. Exit code 1 when a cell fails, 2 on a
usage or git error.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import sys
from pathlib import Path

# The files whose change can change what the automatic speculation decision serves. A row measured
# before the last commit touching one of them says nothing about today's decision (it is stale).
DECISION_FILES = (
    "src/spec",
    "src/product/speculation",
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = REPO / "docs" / "performance" / "data" / "bench.jsonl"
DECODE_METRICS = ("decode", "decode_aggregate")
# Flags that choose the speculation; everything else in `flags` is part of the pairing key.
SPEC_FLAGS = ("--spec", "--draft-tokens", "--spec-load")
# Row fields that are measurements or describe the speculation, not the measured setting.
NOT_KEY = {"spec", "draft_tokens", "flags", "value", "spread", "date", "repeats", "commit"}


def parse_flags(flags: str | None) -> tuple[str | None, tuple[str, ...]]:
    """Returns (the --spec value or None, the other flags in a canonical order)."""
    words = shlex.split(flags or "")
    spec, rest, i = None, [], 0
    while i < len(words):
        word, value = words[i], None
        if "=" in word and word.startswith("--"):
            word, value = word.split("=", 1)
        elif i + 1 < len(words) and not words[i + 1].startswith("--"):
            value = words[i + 1]
            i += 1
        i += 1
        if word == "--spec":
            spec = value or ""
        elif word not in SPEC_FLAGS:
            rest.append(word if value is None else f"{word}={value}")
    return spec, tuple(sorted(rest))


def side_of(row: dict) -> str | None:
    spec, _ = parse_flags(row.get("flags"))
    if spec is None:
        return "auto"
    return "plain" if spec == "none" else None  # an explicit drafter is neither


def pair_key(row: dict) -> str:
    key = {k: v for k, v in row.items() if k not in NOT_KEY}
    key["flags"] = parse_flags(row.get("flags"))[1]
    return json.dumps(key, sort_keys=True, default=str)


def interval(row: dict) -> tuple[float, float]:
    spread = row.get("spread")
    if isinstance(spread, (list, tuple)) and len(spread) == 2 and None not in spread:
        return float(min(spread)), float(max(spread))
    return float(row["value"]), float(row["value"])


def parse_date(text: str) -> dt.datetime:
    d = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def cell_name(row: dict) -> str:
    quant = row.get("recipe", "").split("--quant", 1)[-1].strip() or row.get("recipe", "")
    model = str(row.get("model", "")).split("/")[-1]
    extra = [w for w in parse_flags(row.get("flags"))[1]
             if not w.startswith(("--max-context", "--max-concurrency"))]
    gpu = row.get("machine", {}).get("gpu_name", "?").replace("NVIDIA ", "").replace("GeForce ", "")
    return " ".join([gpu, model, quant, f"{row.get('prompt_set')}", f"C{row.get('concurrency')}",
                     row.get("metric", ""), *extra])


def gate(rows, since_date: dt.datetime, threshold: float = 0.97, newer=None) -> dict:
    """Pairs and judges the rows. `newer(row)` may answer from the row's commit (True/False) or
    return None to fall back to the date."""
    stats = {"rows": 0, "prefill_excluded": 0, "explicit_spec": 0, "superseded": 0}
    sides: dict[str, dict[str, dict]] = {"auto": {}, "plain": {}}
    for row in rows:
        metric = row.get("metric")
        if metric == "prefill":
            stats["prefill_excluded"] += 1
            continue
        if metric not in DECODE_METRICS:
            continue
        stats["rows"] += 1
        side = side_of(row)
        if side is None:
            stats["explicit_spec"] += 1
            continue
        key, have = pair_key(row), sides[side]
        if key in have:
            stats["superseded"] += 1
            if parse_date(have[key]["date"]) >= parse_date(row["date"]):
                continue
        have[key] = row

    cells, counts = [], {"paired": 0, "unpaired_auto": 0, "unpaired_plain": 0, "pass": 0,
                         "losses": 0, "plain_served": 0, "overlap": 0, "stale": 0, "fail": 0}
    for key, auto in sides["auto"].items():
        plain = sides["plain"].get(key)
        if plain is None:
            counts["unpaired_auto"] += 1
            continue
        counts["paired"] += 1
        ratio = auto["value"] / plain["value"] if plain["value"] else float("inf")
        if ratio >= threshold:
            counts["pass"] += 1
            continue
        counts["losses"] += 1
        (alo, ahi), (plo, phi) = interval(auto), interval(plain)
        fresh = newer(auto) if newer else None
        if fresh is None:
            fresh = parse_date(auto["date"]) > since_date
        if auto.get("spec") in ("none", "", None):
            verdict = "plain_served"
        elif ahi >= plo and phi >= alo:
            verdict = "overlap"
        elif not fresh:
            verdict = "stale"
        else:
            verdict = "fail"
        counts[verdict] += 1
        cells.append({"cell": cell_name(auto), "served": auto.get("spec"),
                      "draft_tokens": auto.get("draft_tokens"), "auto": auto["value"],
                      "plain": plain["value"], "ratio": round(ratio, 3),
                      "auto_spread": [alo, ahi], "plain_spread": [plo, phi],
                      "auto_date": auto["date"], "plain_date": plain["date"], "verdict": verdict})
    counts["unpaired_plain"] = sum(1 for k in sides["plain"] if k not in sides["auto"])
    cells.sort(key=lambda c: c["ratio"])
    return {"stats": stats, "counts": counts, "cells": cells}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def render(result: dict, since: str, show_all: bool) -> str:
    c, s = result["counts"], result["stats"]
    shown = [x for x in result["cells"] if show_all or x["verdict"] == "fail"]
    lines = [f"decision code: {since}",
             f"decode rows {s['rows']} (prefill excluded {s['prefill_excluded']}, explicit spec "
             f"{s['explicit_spec']}, superseded {s['superseded']}); paired {c['paired']}, unpaired "
             f"auto {c['unpaired_auto']}, unpaired plain {c['unpaired_plain']}",
             f"pass {c['pass']}; losses {c['losses']}: plain-served {c['plain_served']}, overlap "
             f"{c['overlap']}, stale {c['stale']}, FAIL {c['fail']}"]
    if shown:
        lines += ["", "| cell | served | auto | plain | ratio | auto date | verdict |",
                  "|---|---|---|---|---|---|---|"]
        for x in shown:
            served = f"{x['served']}:{x['draft_tokens']}" if x["draft_tokens"] else x["served"]
            lines.append(f"| {x['cell']} | {served} | {x['auto']:g} | {x['plain']:g} | "
                         f"{x['ratio']:.3f} | {x['auto_date'][:16]} | {x['verdict']} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--threshold", type=float, default=0.97)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--since-commit", help="the decision commit (default: the last commit "
                       "touching DECISION_FILES)")
    group.add_argument("--since-date", help="ISO date instead of a commit (no git needed)")
    ap.add_argument("--losses", action="store_true", help="list every losing cell, not only fails")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    newer, ancestry = None, {}
    try:
        if a.since_date:
            since_date, since = parse_date(a.since_date), a.since_date
        else:
            rev = a.since_commit or git("log", "-1", "--format=%H", "--", *DECISION_FILES)
            if not rev:
                raise subprocess.CalledProcessError(1, "git log", stderr="no decision commit")
            full, when = git("log", "-1", "--format=%H %cI", rev).split()
            since_date, since = parse_date(when), f"{full[:12]} {when}"

            def newer(row):
                commit = row.get("commit")
                if not commit:
                    return None
                if commit not in ancestry:
                    r = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor",
                                        full, commit], capture_output=True)
                    ancestry[commit] = {0: True, 1: False}.get(r.returncode)
                return ancestry[commit]
    except (subprocess.CalledProcessError, OSError, ValueError) as e:
        print(f"spec_gate: cannot find the decision commit: {getattr(e, 'stderr', '') or e}",
              file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in a.bench.read_text(encoding="utf-8").splitlines() if l.strip()]
    result = gate(rows, since_date, a.threshold, newer)
    result["since"] = since
    result["threshold"] = a.threshold
    if a.json:
        print(json.dumps(result, indent=1))
    else:
        print(render(result, since, a.losses))
    return 1 if result["counts"]["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
