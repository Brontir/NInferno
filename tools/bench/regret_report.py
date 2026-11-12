#!/usr/bin/env python3
"""The report of one regret run on one board (#620 step 10): schema `ninferno-regret-run/1`.

  regret_report.py build --selection-report report.json [--bands bands.json] --board B --card C
      --revision SHA --date ISO --nvcc V --nvcc-source CMAKE_CUDA_COMPILER|none --driver V
      [--gpu CSV --sm-clock-log FILE] [--pci-id 0x2782] [--qualification TEXT]
      --shapes FILE --passes N --samples N [--route-policy P]
      [--previous FILE|auto|none] [--repo DIR] [--threshold 0.05] --out run.json [--md run.md]
  regret_report.py check run.json ...     the schema checker: exit 0 valid, 2 not
  regret_report.py board 0x2782           the board label and card name of a PCI id, tab-separated

`build` takes what the job's steps wrote (tools/bench/regret_job.sh: selection_regret.py's report
and regret_bands.py's) and keeps, for this board: the summary, by_board_regime_authority,
table_exit (#616 step 9; null from a selection_regret.py before it), the per-case state of the
measured overrides table_exit counts, and the band table. Its `delta` compares them with the
previous committed report of the board: by regime x authority, by band and for the board, a
slowdown factor that grew by more than --threshold is a regression and one that shrank by as much
an improvement; with the override states, the cases that newly hold or newly caught up.
`delta.sweep_matches` says whether both runs swept the same grid, samples, passes and policy
(true / false, "unknown" against an A5 summary); when it is not true the moves are not the
selector's alone, and the summary says so.
`--previous auto` is the newest tools/bench/data/route_regret/runs/<board>/*.json at the repo's
HEAD, else the A5 summary tools/bench/data/route_regret/<board>.json (read from git, so an
uncommitted file is not "committed"). The A5 summaries carry no per-case rows: against one, the
new holds and caught-ups are null. See docs/maintainer/route-regret.md#the-regret-job.
"""
import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regret_collect as collect  # noqa: E402
import selection_regret as regret  # noqa: E402

SCHEMA = "ninferno-regret-run/1"
ROOT = Path(__file__).resolve().parents[2]
DATA = "tools/bench/data/route_regret"
# A slowdown factor that moves by more than this between two runs is reported. 5 %: the resolution
# the table exit metric reads (selection_regret.EXIT_RESOLUTION, #616 step 9). Below it lie the
# timer's tick on the narrowest decode cases (1.024 us, route-regret.md) and run-to-run drift.
THRESHOLD = 0.05
# The figures compared, each read as a slowdown factor (1 + regret): per regime x authority group
# and for the board, the geometric mean and the p95 of the per-case factors; per band, the
# time-weighted factor regret_bands.py reports.
GROUP_METRICS = ("geomean_slowdown", "p95")
BAND_METRIC = "weighted"
SUMMARY_KEYS = ("cases", "scored_cases", "geomean_slowdown", "median_regret", "p95_regret",
                "max_regret", "statuses")
TABLE_EXIT_KEYS = ("cases", "answered_by", "overrides", "caught_up", "holds", "unscored",
                   "exit_share", "resolution")
EXIT_DELTA_KEYS = ("overrides", "caught_up", "holds", "exit_share")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
REVISION = re.compile(r"[0-9a-f]{7,40}(-dirty)?\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
NVCC_VERSION = re.compile(r"\d+(\.\d+)+\Z")
# Where `nvcc` comes from: the build's CMakeCache.txt (CMAKE_CUDA_COMPILER, asked for its version),
# or "none" when the cache names no CUDA compiler, and then nvcc is "unknown": never the nvcc on
# PATH, which need not be the one the build used.
NVCC_SOURCES = ("CMAKE_CUDA_COMPILER", "none")
# What a delta needs equal between two runs for the selector to be all that changed.
SWEEP_KEYS = ("shapes_sha256", "samples", "passes", "route_policy")
GPU_KEYS = ("name", "power_limit_w", "max_sm_mhz", "applications_graphics_mhz", "persistence_mode",
            "sm_mhz_during_sweep")


def reading(text):
    """One nvidia-smi value (`nounits`): a number, else the text it printed ("[N/A]", "Enabled")."""
    text = text.strip()
    try:
        value = float(text)
    except ValueError:
        return text
    return int(value) if value.is_integer() else value


def gpu_record(fields, clock_log=None):
    """The card as nvidia-smi read it. fields: name, power.limit, clocks.max.sm,
    clocks.applications.graphics, persistence_mode (one csv,noheader,nounits line); clock_log: the
    SM clock (MHz) sampled once a second during the sweep, one value a line, whose min, max and
    count are kept (None with no numeric sample)."""
    values = [v.strip() for v in fields.split(",")]
    if len(values) != 5:
        raise ValueError(f"--gpu: expected 5 comma-separated fields, got {fields!r}")
    out = {"name": values[0], **{k: reading(v) for k, v in zip(GPU_KEYS[1:5], values[1:])}}
    samples = []
    for line in (clock_log or "").splitlines():
        value = reading(line)
        if type(value) in (int, float):
            samples.append(value)
    out["sm_mhz_during_sweep"] = ({"min": min(samples), "max": max(samples),
                                   "samples": len(samples)} if samples else None)
    return out


# A committed label kept for a registry name that carries the board's memory size: the lab's RTX
# 4060 Ti is the 16 GB board (0x2805), which its A5 summary, the selector goldens and the
# performance pages name rtx4060ti; the 8 GB board is the same chip (AD106, 34 SMs, the same
# clocks), so a run of either compares with that baseline.
ESTABLISHED_LABELS = {"rtx4060ti16gb": "rtx4060ti"}


def board_of(name):
    """(label, card) of a gpu_facts.json name: "GeForce RTX 4070 Ti" -> ("rtx4070ti", "RTX 4070
    Ti"), "Quadro GV100" -> ("gv100", "Quadro GV100"): the names of the committed reports."""
    card = re.sub(r"^(NVIDIA\s+)?(GeForce\s+)?", "", name.strip())
    label = re.sub(r"[^a-z0-9]", "", re.sub(r"^quadro\s+", "", card.lower()))
    label = ESTABLISHED_LABELS.get(label, label)
    if not label:
        raise ValueError(f"no board label in the name {name!r}")
    return label, card


def factor(value):
    """A regret as a slowdown factor; a figure already a factor (geomean_slowdown) as it is."""
    return None if value is None else 1.0 + value


def group_figures(group):
    return {"geomean_slowdown": group.get("geomean_slowdown"),
            "p95": factor(group.get("p95_regret"))}


def override_states(report, board, table_exit):
    """{resolution, caught_up, holds, unscored}: the case ids of the board's measured overrides by
    the rule of selection_regret.table_exit, or None when the report has no table_exit."""
    if not table_exit:
        return None
    resolution = table_exit["resolution"]
    out = {"resolution": resolution, "caught_up": [], "holds": [], "unscored": []}
    for case in report.get("cases", []):
        if (case["board"] != board or case["authority"] not in regret.OVERRIDE_AUTHORITIES
                or case["evidence"] != "measured"):
            continue
        if case["status"] != "complete" or "ranker_vs_selected" not in case:
            out["unscored"].append(case["id"])
        elif case["ranker_vs_selected"] <= resolution:
            out["caught_up"].append(case["id"])
        else:
            out["holds"].append(case["id"])
    for key in ("caught_up", "holds", "unscored"):
        out[key].sort()
    return out


def previous_view(document, board):
    """What a previous report says about the board: (revision, summary, groups, bands, table_exit,
    overrides, sweep). A ninferno-regret-run/1 report, or an A5 summary ({regret_report,
    selected_bands}, the files of #790), whose sweep is not recorded in these terms (None)."""
    if not isinstance(document, dict):
        raise ValueError("previous report: expected a JSON object")
    if document.get("schema") == SCHEMA:
        if document.get("board") != board:
            raise ValueError(f"previous report is of board {document.get('board')}, not {board}")
        return (document["revision"], document["summary"], document["by_board_regime_authority"],
                document.get("bands"), document.get("table_exit"), document.get("overrides"),
                document["sweep"])
    report = document.get("regret_report")
    if isinstance(report, dict) and report.get("schema") == "ninferno-selection-regret-report/1":
        if board not in report.get("by_board", {}):
            raise ValueError(f"previous report has no board {board}")
        groups = [g for g in report["by_board_regime_authority"] if g["board"] == board]
        bands = (document.get("selected_bands") or {}).get("overall")
        exit_ = (report.get("table_exit") or {}).get(board)
        revision = document.get("source_sha") or report["selector_revision"]
        return revision, report["by_board"][board], groups, bands, exit_, None, None
    raise ValueError("previous report: neither ninferno-regret-run/1 nor an A5 summary")


def change(previous, current, threshold):
    """'regression', 'improvement' or None for two slowdown factors."""
    if previous is None or current is None:
        return None
    if current / previous - 1 > threshold:
        return "regression"
    if previous / current - 1 > threshold:
        return "improvement"
    return None


def sweep_matches(previous, current):
    """True when both runs swept the same grid (its sha256) with the same samples, passes and route
    policy; False when not; "unknown" when the previous report does not record them (an A5
    summary). A delta across different sweeps is not a comparison of the selector alone."""
    if previous is None:
        return "unknown"
    return all(previous.get(k) == current.get(k) for k in SWEEP_KEYS)


def compare(current, previous, threshold):
    """The delta of `current` (the new report, without delta) against previous_view()'s tuple."""
    p_rev, p_summary, p_groups, p_bands, p_exit, p_overrides, p_sweep = previous
    moved = {"regression": [], "improvement": []}

    def note(entry, before, after):
        kind = change(before, after, threshold)
        if kind:
            moved[kind].append({**entry, "previous": before, "current": after,
                                "change": after / before - 1})

    # The figures leave out a case either run could not score: two runs that did not score every
    # case may sum different case sets, so their board and group figures are not compared.
    scored_all = lambda s: s.get("scored_cases") is None or s["scored_cases"] == s.get("cases")
    scored_match = scored_all(p_summary) and scored_all(current["summary"])
    for metric in GROUP_METRICS if scored_match else ():
        note({"scope": "board", "metric": metric},
             group_figures(p_summary)[metric], group_figures(current["summary"])[metric])
    key = lambda g: (g["regime"], g["authority"])
    before = {key(g): g for g in p_groups}
    after = {key(g): g for g in current["by_board_regime_authority"]}
    for k in sorted(before.keys() & after.keys()) if scored_match else ():
        for metric in GROUP_METRICS:
            note({"scope": "regime_authority", "regime": k[0], "authority": k[1], "metric": metric,
                  "cases": [before[k]["cases"], after[k]["cases"]]},
                 group_figures(before[k])[metric], group_figures(after[k])[metric])
    for band in sorted((p_bands or {}).keys() & (current.get("bands") or {}).keys()):
        note({"scope": "band", "band": band, "metric": BAND_METRIC},
             factor(p_bands[band].get("weighted_regret")),
             factor(current["bands"][band].get("weighted_regret")))
    side = lambda keys, groups: [{"regime": k[0], "authority": k[1], "cases": groups[k]["cases"]}
                                 for k in sorted(keys)]
    overrides = current.get("overrides")
    states = overrides is not None and p_overrides is not None
    exit_now = current.get("table_exit")
    return {
        "threshold": threshold,
        "sweep_matches": sweep_matches(p_sweep, current["sweep"]),
        "all_cases_scored": scored_match,
        "regressions": moved["regression"],
        "improvements": moved["improvement"],
        "new_groups": side(after.keys() - before.keys(), after),
        "gone_groups": side(before.keys() - after.keys(), before),
        "new_holds": sorted(set(overrides["holds"]) - set(p_overrides["holds"])) if states else None,
        "new_caught_up": (sorted(set(overrides["caught_up"]) - set(p_overrides["caught_up"]))
                          if states else None),
        "table_exit": ({k: [p_exit.get(k), exit_now.get(k)] for k in EXIT_DELTA_KEYS}
                       if p_exit and exit_now else None),
    }


def git_text(repo, ref, path):
    return subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{path}"], check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout


def pick_previous(names, board):
    """The previous report among the committed file names: the newest run of the board (the file
    names start with the date, so the last in order), else its A5 summary, else None."""
    runs = sorted(n for n in names if re.fullmatch(rf"{DATA}/runs/{re.escape(board)}/[^/]+\.json", n))
    if runs:
        return runs[-1]
    summary = f"{DATA}/{board}.json"
    return summary if summary in names else None


def committed_previous(repo, board, ref="HEAD"):
    """(path, document) of the previous committed report of the board at `ref`, or (None, None)."""
    names = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", ref, "--", DATA],
                           check=True, capture_output=True, text=True).stdout.split("\n")
    path = pick_previous([n for n in names if n], board)
    if path is None:
        return None, None
    return path, json.loads(git_text(repo, ref, path), object_pairs_hook=regret.unique_object)


def build(selection_report, board, meta, bands=None, previous=None, previous_path=None,
          threshold=THRESHOLD):
    """The ninferno-regret-run/1 report. `meta`: card, revision, date, nvcc, nvcc_source, driver,
    qualification, sweep (a dict of SWEEP_KEYS), and optionally gpu (gpu_record()),
    pci_device_id."""
    report = selection_report
    if report.get("schema") != "ninferno-selection-regret-report/1":
        raise ValueError("--selection-report: not a ninferno-selection-regret-report/1")
    if board not in report.get("by_board", {}):
        raise ValueError(f"--selection-report has no board {board} "
                         f"(it has {', '.join(sorted(report.get('by_board', {}))) or 'none'})")
    if len(report["by_board"]) != 1:
        raise ValueError("--selection-report holds more than one board: one run is one board")
    if not 0 <= threshold < math.inf:
        raise ValueError("--threshold must be finite and nonnegative")
    exit_ = (report.get("table_exit") or {}).get(board)
    out = {
        "schema": SCHEMA, "board": board, "device": report.get("device"),
        **{k: meta[k] for k in ("card", "revision", "date", "nvcc", "nvcc_source", "driver",
                                "qualification", "sweep")},
        "selector_revision": report["selector_revision"], "build_id": report["build_id"],
        "holdout": report["holdout"], "passed": report["passed"],
        "summary": report["by_board"][board],
        "by_board_regime_authority": [g for g in report["by_board_regime_authority"]
                                      if g["board"] == board],
        "table_exit": exit_,
        "overrides": override_states(report, board, exit_),
        "bands": (bands or {}).get("overall"),
        "selection_report_sha256": hashlib.sha256(
            json.dumps(report, sort_keys=True).encode()).hexdigest(),
    }
    for k in ("gpu", "pci_device_id"):
        if meta.get(k):
            out[k] = meta[k]
    if previous is None:
        out["delta"] = {"previous": None, "threshold": threshold}
    else:
        view = previous_view(previous, board)
        out["delta"] = {"previous": {"path": previous_path, "revision": view[0]},
                        **compare(out, view, threshold)}
    problems = check(out)
    if problems:
        raise ValueError("the report fails its own schema: " + "; ".join(problems))
    return out


def check(doc):
    """The schema of ninferno-regret-run/1: a list of what is wrong, empty when valid."""
    bad = []

    def need(cond, what):
        if not cond:
            bad.append(what)
        return cond

    def text(key, pattern=None):
        v = doc.get(key)
        if need(isinstance(v, str) and v.strip() == v and v, f"{key}: expected a nonempty string"):
            if pattern is not None:
                need(pattern.match(v), f"{key}: {v!r} is not of the form {pattern.pattern}")

    def number(v):
        return type(v) in (int, float) and math.isfinite(v)

    def ids(v, where):
        need(isinstance(v, list) and all(isinstance(x, str) and x for x in v)
             and len(set(v)) == len(v), f"{where}: expected a list of distinct case ids")

    if not need(isinstance(doc, dict), "expected a JSON object"):
        return bad
    if not need(doc.get("schema") == SCHEMA, f"schema: expected {SCHEMA}"):
        return bad
    for key in ("board", "card", "device", "nvcc", "driver", "qualification", "selector_revision",
                "build_id", "holdout"):
        text(key)
    text("revision", REVISION)
    text("date", DATE)
    source = doc.get("nvcc_source")
    if need(source in NVCC_SOURCES, f"nvcc_source: expected one of {NVCC_SOURCES}"):
        if source == "none":
            need(doc.get("nvcc") == "unknown",
                 "nvcc: must be unknown when the build's cache names no CUDA compiler")
        else:
            need(isinstance(doc.get("nvcc"), str) and NVCC_VERSION.match(doc["nvcc"]),
                 "nvcc: expected the version of the cache's CUDA compiler")
    need(type(doc.get("passed")) is bool, "passed: expected true or false")
    sweep = doc.get("sweep")
    if need(isinstance(sweep, dict), "sweep: expected an object"):
        for key in ("passes", "samples"):
            need(type(sweep.get(key)) is int and sweep[key] > 0,
                 f"sweep.{key}: expected a positive integer")
        need(isinstance(sweep.get("shapes_sha256"), str) and SHA256.match(sweep["shapes_sha256"]),
             "sweep.shapes_sha256: expected the grid's sha256")
        need(isinstance(sweep.get("shapes"), str) and sweep["shapes"], "sweep.shapes: expected a path")
        need(isinstance(sweep.get("route_policy"), str) and sweep["route_policy"],
             "sweep.route_policy: expected a nonempty string")
    gpu = doc.get("gpu")
    if gpu is not None and need(isinstance(gpu, dict) and set(gpu) == set(GPU_KEYS),
                                f"gpu: expected an object of {', '.join(GPU_KEYS)}"):
        clocks = gpu["sm_mhz_during_sweep"]
        need(clocks is None or (isinstance(clocks, dict)
                                and all(number(clocks.get(k)) for k in ("min", "max"))
                                and type(clocks.get("samples")) is int and clocks["samples"] > 0
                                and clocks["min"] <= clocks["max"]),
             "gpu.sm_mhz_during_sweep: expected null or {min <= max, samples > 0}")
    summary = doc.get("summary")
    if need(isinstance(summary, dict), "summary: expected an object"):
        for key in SUMMARY_KEYS:
            need(key in summary, f"summary: missing {key}")
        for key in ("cases", "scored_cases"):
            need(type(summary.get(key)) is int and summary[key] >= 0,
                 f"summary.{key}: expected a nonnegative integer")
        for key in ("geomean_slowdown", "median_regret", "p95_regret", "max_regret"):
            v = summary.get(key)
            need(v is None or number(v), f"summary.{key}: expected a number or null")
        need(isinstance(summary.get("statuses"), dict), "summary.statuses: expected an object")
    groups = doc.get("by_board_regime_authority")
    if need(isinstance(groups, list), "by_board_regime_authority: expected a list"):
        seen = set()
        for i, g in enumerate(groups):
            where = f"by_board_regime_authority[{i}]"
            if not need(isinstance(g, dict), f"{where}: expected an object"):
                continue
            need(g.get("board") == doc.get("board"), f"{where}: board is not the report's")
            need(g.get("regime") in regret.REGIMES, f"{where}: unknown regime {g.get('regime')!r}")
            need(isinstance(g.get("authority"), str) and g.get("authority"),
                 f"{where}: expected an authority")
            need(type(g.get("cases")) is int and g.get("cases") > 0, f"{where}: expected cases")
            k = (g.get("regime"), g.get("authority"))
            need(k not in seen, f"{where}: {k} listed twice")
            seen.add(k)
    exit_ = doc.get("table_exit")
    if exit_ is not None and need(isinstance(exit_, dict), "table_exit: expected an object or null"):
        for key in TABLE_EXIT_KEYS:
            need(key in exit_, f"table_exit: missing {key}")
    overrides = doc.get("overrides")
    need("overrides" in doc, "overrides: missing (null without table_exit)")
    if overrides is not None and need(isinstance(overrides, dict) and exit_ is not None,
                                      "overrides: an object, and only with table_exit"):
        for key in ("caught_up", "holds", "unscored"):
            ids(overrides.get(key), f"overrides.{key}")
            if isinstance(exit_, dict) and isinstance(overrides.get(key), list):
                need(len(overrides[key]) == exit_.get(key),
                     f"overrides.{key}: {len(overrides[key])} ids, table_exit counts "
                     f"{exit_.get(key)}")
    bands = doc.get("bands")
    if bands is not None and need(isinstance(bands, dict), "bands: expected an object or null"):
        for name, row in bands.items():
            need(isinstance(row, dict) and "weighted_regret" in row,
                 f"bands.{name}: missing weighted_regret")
    delta = doc.get("delta")
    if not need(isinstance(delta, dict), "delta: expected an object"):
        return bad
    need(number(delta.get("threshold")) and delta.get("threshold") >= 0,
         "delta.threshold: expected a nonnegative number")
    prev = delta.get("previous")
    if prev is None:
        need(set(delta) == {"previous", "threshold"},
             "delta: with no previous report it holds only previous and threshold")
        return bad
    need(isinstance(prev, dict) and isinstance(prev.get("revision"), str) and "path" in prev,
         "delta.previous: expected {path, revision}")
    for key in ("regressions", "improvements"):
        entries = delta.get(key)
        if not need(isinstance(entries, list), f"delta.{key}: expected a list"):
            continue
        for i, e in enumerate(entries):
            where = f"delta.{key}[{i}]"
            if not need(isinstance(e, dict), f"{where}: expected an object"):
                continue
            scope = e.get("scope")
            fields = {"board": (), "regime_authority": ("regime", "authority", "cases"),
                      "band": ("band",)}.get(scope)
            if not need(fields is not None, f"{where}: unknown scope {scope!r}"):
                continue
            for f in fields:
                need(f in e, f"{where}: missing {f}")
            need(e.get("metric") in GROUP_METRICS + (BAND_METRIC,), f"{where}: unknown metric")
            if need(all(number(e.get(f)) for f in ("previous", "current", "change")),
                    f"{where}: previous, current and change must be numbers"):
                grew = e["current"] > e["previous"]
                need(grew == (key == "regressions"), f"{where}: the change has the wrong sign")
    need(delta.get("sweep_matches") in (True, False, "unknown") and "sweep_matches" in delta,
         "delta.sweep_matches: expected true, false or unknown")
    need(delta.get("all_cases_scored", True) in (True, False),
         "delta.all_cases_scored: expected true or false")
    for key in ("new_groups", "gone_groups"):
        entries = delta.get(key)
        if not need(isinstance(entries, list), f"delta.{key}: expected a list"):
            continue
        for i, g in enumerate(entries):
            need(isinstance(g, dict) and g.get("regime") in regret.REGIMES
                 and isinstance(g.get("authority"), str) and g["authority"]
                 and type(g.get("cases")) is int and g["cases"] > 0,
                 f"delta.{key}[{i}]: expected {{regime, authority, cases}}")
    for key in ("new_holds", "new_caught_up"):
        if key not in delta:
            bad.append(f"delta.{key}: missing")
        elif delta[key] is not None:
            ids(delta[key], f"delta.{key}")
            need(overrides is not None, f"delta.{key}: listed without override states")
    if need("table_exit" in delta, "delta.table_exit: missing") and delta["table_exit"] is not None:
        moved = delta["table_exit"]
        need(isinstance(moved, dict) and set(moved) == set(EXIT_DELTA_KEYS)
             and all(isinstance(v, list) and len(v) == 2
                     and all(x is None or number(x) for x in v) for v in moved.values()),
             f"delta.table_exit: expected null or {{{', '.join(EXIT_DELTA_KEYS)}}}: "
             "[previous, current]")
    return bad


def pct(x):
    return "-" if x is None else f"{x * 100:+.1f}%"


def comparability(delta):
    """Why the delta's moves are not the selector's alone, or "" when the sweeps match."""
    why = {True: "", False: "the sweep differs (grid, samples, passes or route policy)",
           "unknown": "the previous report does not record its sweep"}[delta["sweep_matches"]]
    if delta.get("all_cases_scored") is False:
        why = "; ".join(w for w in (why, "a run left cases unscored: board and group figures not "
                                         "compared") if w)
    return why


def markdown(doc):
    """A short summary for a PR or an issue comment."""
    s, d = doc["summary"], doc["delta"]
    lines = [f"**{doc['card']}** ({doc['device']}), {doc['revision'][:12]}, {doc['date']}, "
             f"nvcc {doc['nvcc']}, driver {doc['driver']}",
             "",
             f"- cases {s['scored_cases']}/{s['cases']} scored; geomean slowdown "
             f"{pct(s['geomean_slowdown'] - 1 if s['geomean_slowdown'] else None)}, "
             f"p95 {pct(s['p95_regret'])}, max {pct(s['max_regret'])}"]
    exit_ = doc.get("table_exit")
    if exit_:
        share = exit_["exit_share"]
        lines.append(f"- table exit: {exit_['caught_up']} caught up, {exit_['holds']} hold, "
                     f"{exit_['unscored']} unscored of {exit_['overrides']} measured overrides"
                     + ("" if share is None else f" (exit share {share:.0%})"))
    else:
        lines.append("- table exit: not in this selection_regret.py's report")
    gpu = doc.get("gpu")
    if gpu:
        clocks = gpu["sm_mhz_during_sweep"]
        lines.append(f"- card: {gpu['name']}, max SM {gpu['max_sm_mhz']} MHz, application clock "
                     f"{gpu['applications_graphics_mhz']} MHz, persistence "
                     f"{gpu['persistence_mode']}, power limit {gpu['power_limit_w']} W; SM clock "
                     "during the sweep " + ("not sampled" if clocks is None else
                                            f"{clocks['min']}-{clocks['max']} MHz "
                                            f"({clocks['samples']} samples)"))
    if doc.get("bands"):
        lines.append("- bands (time-weighted): " + ", ".join(
            f"{b} {pct(r['weighted_regret'])}" for b, r in doc["bands"].items()))
    if d["previous"] is None:
        lines.append("- no previous report of this board: no delta")
        return "\n".join(lines)
    caveat = comparability(d)
    lines.append(f"- against {d['previous']['path']} ({d['previous']['revision'][:12]}), "
                 f"threshold {d['threshold']:.0%}: {len(d['regressions'])} regressions, "
                 f"{len(d['improvements'])} improvements"
                 + (f" ({caveat})" if caveat else " (same sweep)"))
    for key in ("new_holds", "new_caught_up"):
        if d[key] is not None:
            lines.append(f"- {key.replace('_', ' ')}: " + (", ".join(d[key]) or "none"))
    for g in d["new_groups"]:
        lines.append(f"- new group: {g['regime']} / {g['authority']} ({g['cases']} cases)")
    for g in d["gone_groups"]:
        lines.append(f"- gone group: {g['regime']} / {g['authority']} ({g['cases']} cases)")
    if d["regressions"] or d["improvements"]:
        if caveat:
            lines += ["", f"Not comparable as a selector change: {caveat}."]
        lines += ["", "| change | where | metric | previous | current |", "|---|---|---|---:|---:|"]
        for kind in ("regressions", "improvements"):
            for e in d[kind]:
                where = {"board": "board", "band": e.get("band"),
                         "regime_authority": f"{e.get('regime')} / {e.get('authority')}"}[e["scope"]]
                lines.append(f"| {pct(e['change'])} | {where} | {e['metric']} | "
                             f"{e['previous']:.3f} | {e['current']:.3f} |")
    return "\n".join(lines)


def load(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=regret.unique_object)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="write one run's report")
    b.add_argument("--selection-report", type=Path, required=True)
    b.add_argument("--bands", type=Path, help="regret_bands.py --json of the same document")
    b.add_argument("--board", required=True)
    for key in ("card", "revision", "date", "nvcc", "driver"):
        b.add_argument(f"--{key}", required=True)
    b.add_argument("--nvcc-source", choices=NVCC_SOURCES, required=True,
                   help="where --nvcc was read: the cache's CUDA compiler, or none (nvcc unknown)")
    b.add_argument("--qualification", default="",
                   help="the statement regret_collect.py --qualified was given")
    b.add_argument("--gpu", default="",
                   help="nvidia-smi --query-gpu=name,power.limit,clocks.max.sm,"
                        "clocks.applications.graphics,persistence_mode --format=csv,noheader,nounits")
    b.add_argument("--sm-clock-log", type=Path,
                   help="clocks.sm sampled during the sweep, one MHz value a line")
    b.add_argument("--pci-id", default="")
    b.add_argument("--shapes", type=Path, required=True)
    b.add_argument("--passes", type=int, required=True)
    b.add_argument("--samples", type=int, required=True)
    b.add_argument("--route-policy", default="")
    b.add_argument("--previous", default="auto", help="a report file, auto or none")
    b.add_argument("--repo", type=Path, default=ROOT, help="where --previous auto looks (git)")
    b.add_argument("--threshold", type=float, default=THRESHOLD)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--md", type=Path)
    c = sub.add_parser("check", help="validate reports against the schema")
    c.add_argument("reports", type=Path, nargs="+")
    g = sub.add_parser("board", help="the board label and card name of a PCI id")
    g.add_argument("pci_id")
    g.add_argument("--gpu-facts", type=Path, default=ROOT / "src/core/data/gpu_facts.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "board":
            label, card = board_of(collect.board_facts(args.gpu_facts, args.pci_id)["name"])
            print(f"{label}\t{card}")
            return 0
        if args.command == "check":
            failed = False
            for path in args.reports:
                problems = check(load(path))
                for p in problems:
                    print(f"{path}: {p}", file=sys.stderr)
                failed |= bool(problems)
            return 2 if failed else 0
        sweep = {"passes": args.passes, "samples": args.samples,
                 "route_policy": args.route_policy or "the build's default"}
        try:  # a path in the repository, not the stand's checkout directory
            sweep["shapes"] = args.shapes.resolve().relative_to(args.repo.resolve()).as_posix()
        except ValueError:
            sweep["shapes"] = args.shapes.name
        sweep["shapes_sha256"] = hashlib.sha256(args.shapes.read_bytes()).hexdigest()
        clock_log = (args.sm_clock_log.read_text(encoding="utf-8")
                     if args.sm_clock_log else None)
        meta = {"card": args.card, "revision": args.revision, "date": args.date,
                "nvcc": args.nvcc, "nvcc_source": args.nvcc_source, "driver": args.driver,
                "qualification": args.qualification.strip() or "none stated", "sweep": sweep,
                "gpu": gpu_record(args.gpu, clock_log) if args.gpu else None,
                "pci_device_id": args.pci_id}
        if args.previous == "auto":
            previous_path, previous = committed_previous(args.repo, args.board)
        elif args.previous == "none":
            previous_path, previous = None, None
        else:
            previous_path, previous = args.previous, load(args.previous)
        doc = build(load(args.selection_report), args.board, meta,
                    load(args.bands) if args.bands else None, previous, previous_path,
                    args.threshold)
        args.out.write_text(json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n",
                            encoding="utf-8")
        if args.md:
            args.md.write_text(markdown(doc) + "\n", encoding="utf-8")
        d = doc["delta"]
        print(f"regret_report: wrote {args.out}"
              + ("" if d["previous"] is None else
                 f"; against {d['previous']['path']}: {len(d['regressions'])} regressions, "
                 f"{len(d['improvements'])} improvements"
                 + (f" ({comparability(d)})" if comparability(d) else "")))
    except (OSError, UnicodeError, ValueError, KeyError, TypeError,
            subprocess.CalledProcessError) as error:
        print(f"regret_report: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
