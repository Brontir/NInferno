#!/usr/bin/env python3
"""Lab gate: automatic speculation must not be slower than plain decoding.

    python3 tools/spec_gate.py [--bench docs/performance/data/bench.jsonl] [--threshold 0.97]
                               [--since-commit REV | --since-date ISO] [--losses] [--json]

Every decode row measured with the automatic speculation decision (no `--spec`, or `--spec auto`)
is paired with its plain twin (`--spec none`): the same engine, card (GPU name, SM and memory in
whole GB), model, recipe, KV
format, activation precision, sampling, prompt set, prompt length, output tokens, unit,
concurrency, metric and
every other flag. Left out: prefill rows (speculation does not decide the prefill), rows that name
a drafter, and forced rows (`--draft-tokens` or `--spec-load` without a named drafter). The twin
must come from the same build and environment (the same commit when both rows name one, else the
same driver, CUDA and power limit); an auto row whose plain rows all differ is listed as "unmatched
build/env", not judged. Of the matching plain rows the newest is the twin. When an auto key
was measured more than once, the newest row of each side is used.

A cell FAILS when all of these hold:
  * the decision speculates (the auto row's served `spec` is not `none`);
  * auto < threshold x plain;
  * the spreads do not overlap (a row without a spread counts as the single value);
  * the auto row is newer than the decision code: its `commit` (or a `version` that is a commit
    id) descends from the last commit that touched DECISION_FILES, or, for rows without a known
    commit (none, or one this clone lacks: counted), its date is later than that commit's;
  * and so is its plain twin (else the cell counts as "stale plain": rerun the plain row).
Losing cells that miss one of these are counted as plain-served, overlapping, stale or stale
plain. Auto rows without a plain twin, or without one of their build/env, are listed on stderr.

There is no exemption list and nothing is written back. Exit code 1 when a cell fails or nothing
pairs; 2 on a missing or malformed bench file, a shallow clone or another git error.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import sys
from pathlib import Path

# The files whose change can change what the automatic speculation decision serves (the drafter,
# the depth, the load table, the fallbacks). A row measured before the last commit touching one of
# them says nothing about today's decision (it is stale).
DECISION_FILES = (
    "src/spec",                                              # profiles, load tables, cost, watchdog
    "src/product/speculation",                               # the auto request -> drafter, depth
    "src/product/speculative_options.h",
    "src/models/speculation",                                # next_draft_depth (draft_target.h)
    "src/serve/generation_service.cpp",                      # auto fit / shallower / pool fallbacks
    "src/models/qwen3_5/program/planning/startup.cpp",       # the speculative plan rung
    "src/models/qwen3_5/program/planning/request_plan.cpp",  # the per-request speculation plan
    "src/models/qwen3_5/program/planning/speculation_load.h",
    "src/models/qwen3_5/program/planning/sequence_bytes.cpp",  # the speculative rung's bytes
    "src/runtime/engine/plan_dry_run.cpp",                     # the plan the start serves from
    "src/models/qwen3_5/program/prefill.cpp",                # the request's depth at admission
    "src/models/qwen3_5/program/decode.cpp",                 # watchdog trips and step-downs
    "src/models/gemma4/program_family.cpp",                  # Gemma 4 draft depth
    "src/models/gemma4/session.h",                           # Gemma 4 depth adaptation
    "src/models/dense/session.h",                            # dense draft_tokens / min_draft
    "src/models/dense/program_family.cpp",                   # dense drafter window refusals
    "src/models/nemotron_h/program_family.cpp",              # speculative memory from the window
    "src/serve/serve_options.cpp",                           # no --spec -> --spec auto
    "src/runtime/engine/model_instance.cpp",                 # capacity refusal of a drafter's plan
    "src/runtime/engine/plan_ladder.cpp",                    # the memory plan rungs it picks from
    "src/runtime/engine/plan_ladder.h",
    "src/runtime/engine/kv_capacity.h",                      # the contexts a rung holds
    "src/models/qwen3_5/program/graphs.cpp",                 # the shortage auto answers shallower
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = REPO / "docs" / "performance" / "data" / "bench.jsonl"
DECODE_METRICS = ("decode", "decode_aggregate")
DEFAULT_OUT_TOKENS = 512  # tools/bench/perf_matrix.py --out-tokens, for rows that predate the field
# Flags that choose the speculation; everything else in `flags` is part of the pairing key.
SPEC_FLAGS = ("--spec", "--draft-tokens", "--spec-load")
# The row fields that make the measured cell (an allow-list: anything else a row carries, such as
# the measurement, the speculation served, the build, the decision fingerprint and manifest (they
# hash the resolved speculation, so an auto row and its plain twin always differ), dates, run ids
# or host names, is not part of the key). Besides these, the key holds the card (GPU name, SM and
# memory in whole GB, not the driver, CUDA or power limit), the engine (missing = "ninferno"), the
# activation precision (field, else flag, else "fast"), the output tokens per request (missing =
# perf_matrix's default) and the non-speculation flags.
KEY_FIELDS = ("model", "recipe", "kv", "sampling", "metric", "prompt_set", "prompt_len",
              "concurrency", "unit", "context", "stride", "corpus")


def parse_flags(flags: str | None) -> tuple[str | None, tuple[str, ...]]:
    """Returns (the --spec value or None, the other non-speculation flags in a canonical order)."""
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


def side_of(row: dict) -> str:
    """auto, plain, explicit (a named drafter) or forced (a depth or load without a drafter)."""
    spec, _ = parse_flags(row.get("flags"))
    if spec == "none":
        return "plain"
    if spec not in (None, "auto"):
        return "explicit"
    words = shlex.split(row.get("flags") or "")
    if any(w.split("=", 1)[0] in ("--draft-tokens", "--spec-load") for w in words):
        return "forced"
    return "auto"


def flag_value(row: dict, name: str) -> str | None:
    for word in parse_flags(row.get("flags"))[1]:
        if word.startswith(name + "="):
            return word.split("=", 1)[1]
    return None


def is_commit(text) -> bool:
    """A git commit id (7 to 40 hex digits), not a release version such as 0.1.0."""
    return isinstance(text, str) and 7 <= len(text) <= 40 and all(
        c in "0123456789abcdef" for c in text.lower())


def row_commit(row: dict) -> str | None:
    commit = row.get("commit") or row.get("version")
    return commit if is_commit(commit) else None


def pair_key(row: dict) -> str:
    key = {k: row.get(k) for k in KEY_FIELDS}
    machine = row.get("machine") or {}
    vram = machine.get("vram_gb")  # one name can come in two memory sizes (4060 Ti 8 / 16 GB)
    key["card"] = [machine.get("gpu_name"), str(machine.get("sm")),
                   None if vram is None else round(float(vram))]
    key["engine"] = row.get("engine") or "ninferno"
    key["out_tokens"] = row.get("out_tokens") or DEFAULT_OUT_TOKENS
    key["activations"] = row.get("activations") or flag_value(row, "--activations") or "fast"
    # --activations is folded into key["activations"] above (an explicit `--activations fast` is
    # the default; a field and a flag naming the same mode are the same cell).
    key["flags"] = tuple(w for w in parse_flags(row.get("flags"))[1]
                         if w.split("=", 1)[0] != "--activations")
    return json.dumps(key, sort_keys=True, default=str)


def build_env(row: dict) -> dict:
    machine = row.get("machine") or {}
    return {"commit": row_commit(row), "driver": machine.get("driver"),
            "cuda": machine.get("cuda"), "power_limit_w": machine.get("power_limit_w")}


def same_build_env(a: dict, b: dict) -> bool:
    """Twins come from one build and environment: the same commit when both rows name one (a short
    id matches its full id), else the same driver, CUDA version and power limit."""
    ca, cb = row_commit(a), row_commit(b)
    if ca and cb:
        ca, cb = ca.lower(), cb.lower()
        return ca.startswith(cb) or cb.startswith(ca)
    ea, eb = build_env(a), build_env(b)
    return all(ea[k] == eb[k] for k in ("driver", "cuda", "power_limit_w"))


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
    stats = {"rows": 0, "prefill_excluded": 0, "explicit_spec": 0, "forced_spec": 0,
             "superseded": 0}
    autos: dict[str, dict] = {}
    plains: dict[str, list[dict]] = {}  # every plain row: the twin is chosen by build/env first
    for row in rows:
        metric = row.get("metric")
        if metric == "prefill":
            stats["prefill_excluded"] += 1
            continue
        if metric not in DECODE_METRICS:
            continue
        stats["rows"] += 1
        side = side_of(row)
        if side in ("explicit", "forced"):
            stats[f"{side}_spec"] += 1
            continue
        key = pair_key(row)
        if side == "plain":
            plains.setdefault(key, []).append(row)
            continue
        if key in autos:
            stats["superseded"] += 1
            if parse_date(autos[key]["date"]) >= parse_date(row["date"]):
                continue
        autos[key] = row

    stats["commit_unknown"] = 0
    unknown_seen = set()

    def is_fresh(row: dict) -> bool:
        """Newer than the decision code: by the row's commit when git knows it, else by date."""
        fresh = newer(row) if newer else None
        if fresh is None:
            if row_commit(row) and id(row) not in unknown_seen:  # a commit this clone lacks
                unknown_seen.add(id(row))
                stats["commit_unknown"] += 1
            fresh = parse_date(row["date"]) > since_date
        return fresh

    cells, unpaired, unmatched = [], [], []
    counts = {"paired": 0, "unpaired_auto": 0, "unmatched_build_env": 0, "unpaired_plain": 0,
              "pass": 0, "losses": 0, "plain_served": 0, "overlap": 0, "stale": 0,
              "stale_plain": 0, "fail": 0}
    for key, auto in autos.items():
        candidates = plains.get(key)
        if not candidates:
            counts["unpaired_auto"] += 1
            unpaired.append({"cell": cell_name(auto), "served": auto.get("spec"),
                             "auto": auto["value"], "auto_date": auto["date"]})
            continue
        same = [p for p in candidates if same_build_env(auto, p)]
        if not same:
            counts["unmatched_build_env"] += 1
            unmatched.append({"cell": cell_name(auto), "served": auto.get("spec"),
                              "auto": auto["value"], "auto_date": auto["date"],
                              "auto_build_env": build_env(auto),
                              "plain_build_env": [build_env(p) for p in candidates]})
            continue
        if len(candidates) > 1:
            stats["superseded"] += len(candidates) - 1
        plain = max(same, key=lambda p: parse_date(p["date"]))
        counts["paired"] += 1
        ratio = auto["value"] / plain["value"] if plain["value"] else float("inf")
        if ratio >= threshold:
            counts["pass"] += 1
            continue
        counts["losses"] += 1
        (alo, ahi), (plo, phi) = interval(auto), interval(plain)
        if auto.get("spec") in ("none", "", None):
            verdict = "plain_served"
        elif ahi >= plo and phi >= alo:
            verdict = "overlap"
        elif not is_fresh(auto):
            verdict = "stale"
        elif not is_fresh(plain):
            verdict = "stale_plain"
        else:
            verdict = "fail"
        counts[verdict] += 1
        cells.append({"cell": cell_name(auto), "served": auto.get("spec"),
                      "draft_tokens": auto.get("draft_tokens"), "auto": auto["value"],
                      "plain": plain["value"], "ratio": round(ratio, 3),
                      "auto_spread": [alo, ahi], "plain_spread": [plo, phi],
                      "auto_date": auto["date"], "plain_date": plain["date"],
                      "auto_commit": row_commit(auto), "verdict": verdict})
    counts["unpaired_plain"] = sum(1 for k in plains if k not in autos)
    cells.sort(key=lambda c: c["ratio"])
    unpaired.sort(key=lambda c: c["cell"])
    unmatched.sort(key=lambda c: c["cell"])
    return {"stats": stats, "counts": counts, "cells": cells, "unpaired_auto": unpaired,
            "unmatched_build_env": unmatched}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def decision_commit(since_commit: str | None) -> tuple[str, str]:
    """(full id, committer date) of the decision commit; RuntimeError when git cannot say."""
    if git("rev-parse", "--is-shallow-repository") == "true":
        raise RuntimeError("a shallow clone: the decision commit and the rows' ancestry are "
                           "unknown (git fetch --unshallow, or pass --since-date)")
    rev = since_commit or git("log", "-1", "--format=%H", "--", *DECISION_FILES)
    if not rev:
        raise RuntimeError("no commit touches DECISION_FILES")
    full, when, *parents = git("log", "-1", "--format=%H %cI %P", rev).split()
    if not parents:
        raise RuntimeError(f"the decision commit {full[:12]} is the root commit (a truncated "
                           "history?); pass --since-commit or --since-date")
    return full, when


def render(result: dict, since: str, show_all: bool) -> str:
    c, s = result["counts"], result["stats"]
    shown = [x for x in result["cells"] if show_all or x["verdict"] == "fail"]
    lines = [f"decision code: {since}",
             f"decode rows {s['rows']} (prefill excluded {s['prefill_excluded']}, explicit spec "
             f"{s['explicit_spec']}, forced spec {s['forced_spec']}, superseded "
             f"{s['superseded']}); paired {c['paired']}, unpaired auto {c['unpaired_auto']}, "
             f"unmatched build/env {c['unmatched_build_env']}, unpaired plain {c['unpaired_plain']}",
             f"pass {c['pass']}; losses {c['losses']}: plain-served {c['plain_served']}, overlap "
             f"{c['overlap']}, stale {c['stale']}, stale plain {c['stale_plain']}, FAIL "
             f"{c['fail']}"]
    if s.get("commit_unknown"):
        lines.append(f"{s['commit_unknown']} judged rows name a commit this clone lacks: "
                     "judged by date")
    if shown:
        lines += ["", "| cell | served | auto | plain | ratio | auto date | verdict |",
                  "|---|---|---|---|---|---|---|"]
        for x in shown:
            served = f"{x['served']}:{x['draft_tokens']}" if x["draft_tokens"] else x["served"]
            lines.append(f"| {x['cell']} | {served} | {x['auto']:g} | {x['plain']:g} | "
                         f"{x['ratio']:.3f} | {x['auto_date'][:16]} | {x['verdict']} |")
    return "\n".join(lines)


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not all(isinstance(r, dict) for r in rows):
        raise ValueError("a line is not a JSON object")
    return rows


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

    try:
        rows = read_rows(a.bench)
    except (OSError, ValueError) as e:
        print(f"spec_gate: cannot read {a.bench}: {e}", file=sys.stderr)
        return 2

    newer, ancestry = None, {}
    try:
        if a.since_date:
            since_date, since = parse_date(a.since_date), a.since_date
        else:
            full, when = decision_commit(a.since_commit)
            since_date, since = parse_date(when), f"{full[:12]} {when}"

            def newer(row):
                commit = row_commit(row)
                if not commit:
                    return None
                if commit not in ancestry:
                    r = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor",
                                        full, commit], capture_output=True)
                    ancestry[commit] = {0: True, 1: False}.get(r.returncode)  # else: unknown
                return ancestry[commit]
    except (subprocess.CalledProcessError, OSError, ValueError, RuntimeError) as e:
        print(f"spec_gate: cannot find the decision commit: {getattr(e, 'stderr', '') or e}",
              file=sys.stderr)
        return 2

    try:
        result = gate(rows, since_date, a.threshold, newer)
    except (KeyError, TypeError, ValueError) as e:
        print(f"spec_gate: malformed row in {a.bench}: {e!r}", file=sys.stderr)
        return 2
    result["since"] = since
    result["threshold"] = a.threshold
    if a.json:
        print(json.dumps(result, indent=1))
    else:
        print(render(result, since, a.losses))
    if result["unpaired_auto"]:
        print(f"spec_gate: WARNING: {len(result['unpaired_auto'])} auto rows have no plain twin "
              "and are not judged:", file=sys.stderr)
        for x in result["unpaired_auto"]:
            print(f"  {x['cell']} ({x['served']}, {x['auto']:g}, {x['auto_date'][:16]})",
                  file=sys.stderr)
    if result["unmatched_build_env"]:
        print(f"spec_gate: WARNING: {len(result['unmatched_build_env'])} auto rows have plain "
              "twins only from another build or environment and are not judged:", file=sys.stderr)
        for x in result["unmatched_build_env"]:
            print(f"  {x['cell']} (auto {x['auto_build_env']}; plain {x['plain_build_env']})",
                  file=sys.stderr)
    if result["counts"]["paired"] == 0:
        print("spec_gate: FAIL: no auto row pairs with a plain row", file=sys.stderr)
        return 1
    return 1 if result["counts"]["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
