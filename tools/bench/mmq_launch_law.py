#!/usr/bin/env python3
"""The launch law of ggml's matrix kernel (core/generation.h MmqLaunchLaw; DECISIONS 78; #785)
from the microbench tools/bench/mmq_launch_microbench.cu: run it on one card, derive that card's
law, and fit a generation's law row on two or more cards of it, each card predicted from the
others (the holdout).

    mmq_launch_law.py run    --exe BUILD/bench/ninferno_mmq_launch_microbench --out CARD.json
                             [--idle-unchecked] [-- microbench options]
    mmq_launch_law.py derive --input CARD.json [--format q4_K]
    mmq_launch_law.py fit    --input A.json --input B.json [--format q4_K] [--tolerance 0.10]
                             [--out LAW.json]
    mmq_launch_law.py check  --input CARD.json [--law kTuringMmqLaw] [--format q4_K]
                             [--tolerance 0.10]

`run` starts the microbench (the GPU otherwise idle: nothing else may run on it) with nvidia-smi
listing the compute processes and the clocks every 200 ms through the run, as
tools/bench/isa_probe and warp_inflight do, and writes the microbench's JSON with the notes
(isa_probe's: compute processes before, during and after the run) and each configuration's SM
clock span. A run with another compute process on its GPU, or whose process list could not be read
(unless --idle-unchecked, which the file then says), is written to CARD.json.refused and is no
input of `fit`.

`derive` reads one card (its medians: the median over the samples of each pass, then over the
passes), for each format and width J:
  unit        the slope of a call's time over its longest CTA's units on the `law` grid (a tile a
              CTA, floor(S / 2) x 2 tiles, one wave), the unit DECISIONS 78 measured on Turing;
              unit_clocks = unit x the board's reference clock (the clock the law is stated at);
  fixup       the fixup kernel alone (`fixup_alone`: both stream-k grids at every width, so its
              bytes differ about 8x and do not depend on K) fitted as L + bytes / rate:
              launch_seconds = L, and the rate is the fixup's own (its partials and output tiles
              are a few MB, much of it in the L2), printed against the DRAM rate the row prices
              the bytes at. A negative L gives no row;
  stream-k    the slope on each stream-k grid (`streamk`: floor(S / 2) tiles on S CTAs;
              `streamk_wide`: floor(3 S / 2) tiles on S CTAs, the narrow band's kind of launch) of
              the time less the fixup's bytes at that rate, against the unit; stream_k_unit = the
              mean over both grids and the widths, each grid's mean printed apart, so that one
              stream_k_unit can be seen to hold or not;
  checks      the unit on `wave1` (every weight tile read once), `full` (four waves) and
              `halfwave` against the law grid's, the launch's fixed part (the law grid's
              intercept), the in-situ fixup (the streamk intercept less the halfwave one, beyond
              the bytes: biased, a check only), the empty kernel's node cost on a stream and in a
              graph, the SM clock the card ran at against its reference clock (a configuration
              whose clock moved over 3 % is flagged), and every configuration whose kernel fits
              more than one CTA on a SM (the law counts one).
The law row is the one format's (q4_K by default, as kTuringMmqLaw: the row has no format axis);
the other formats' units are printed beside it.

`fit` takes the cards' mean for each value of the row, then predicts each card from the others'
mean (with two cards: A from B and B from A): its unit and stream-k unit (each grid apart) by J at
its own reference clock, its launch_seconds on its own, and every call it measured end to end
(units x unit, x stream_k_unit where it splits, the fixup's bytes at the others' fixup rate plus
launch_seconds, and the others' fixed part of a call, which the row does not hold). A prediction
off by more than --tolerance (default 0.10, the decision bound of #785 PR-B, fixed before any
data) is flagged, and the row is then no generation's law: the cards disagree. Cards whose clocks
stand at shares of their reference clocks more than 3 % apart are named: the transfer assumes the
same share. All cards must be of one generation (the rows of core/generation.h), each a run that
`run` accepted.

`check` compares one card with a law row of core/generation.h (kTuringMmqLaw: the RTX 2080 Ti
check of the tool against DECISIONS 78).

Nothing is fitted beyond these means and slopes; the per-card values stay in the output so that a
later form (a DRAM term, a format term) can be read from the same runs.
"""

import argparse
import datetime
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SOURCE = os.path.join(HERE, "mmq_launch_microbench.cu")
GENERATION_H = os.path.join(ROOT, "src", "core", "generation.h")
SCHEMA = "ninferno-mmq-launch-microbench/1"
LAW_SCHEMA = "ninferno-mmq-launch-law/1"
# core/mmq_columns.h kTilesWork: the widths MmqLaunchLaw::unit_clocks holds, in its order.
WIDTHS = (16, 32, 64, 80, 96, 128)
GRIDS = ("law", "wave1", "full", "streamk", "streamk_wide", "halfwave")
# The stream-k grids (S / 2 and 3 S / 2 tiles on S CTAs, both with the fixup), and the grids whose
# calls the holdout predicts end to end.
STREAM_K_GRIDS = ("streamk", "streamk_wide")
CALL_GRIDS = ("law", "full", "streamk", "streamk_wide")
# A configuration whose SM clock moved more than this over its samples is flagged, and so are two
# cards whose clocks stand at shares of their reference clocks this far apart.
CLOCK_SPREAD = 0.03
NODE_KINDS = ("stream", "stream_pdl", "graph", "graph_pdl")
DEFAULT_TOLERANCE = 0.10
# The generations of core/generation.h that take the work table (mmq_work_tiles), by their row's
# compute capability range, and the name their law row takes (test_mmq_launch_law.py holds the
# ranges to generation.h).
GENERATIONS = {
    "turing": ((7, 5), (7, 9), "kTuringMmqLaw"),
    "ga10x": ((8, 1), (8, 8), "kGa10xMmqLaw"),
    "ada": ((8, 9), (8, 9), "kAdaMmqLaw"),
    "gb20x": ((12, 0), (12, 9), "kGb20xMmqLaw"),
}


def generation_of(cc):
    """The generation row a compute capability "M.m" lies in, or None (a row without the work
    table: the law does not apply there)."""
    major, minor = (int(x) for x in str(cc).split("."))
    for name, (lo, hi, _) in GENERATIONS.items():
        if lo <= (major, minor) <= hi:
            return name
    return None


# --- reading a card ---

def time_s(cfg):
    """A configuration's time a call in seconds: the median of each pass's samples, then the
    median over the passes."""
    passes = [statistics.median(p) for p in cfg.get("samples_us") or [] if p]
    if not passes:
        return None
    return statistics.median(passes) * 1e-6


def pass_spread(cfg):
    """The largest relative difference between two passes' medians (0 for one pass)."""
    passes = [statistics.median(p) for p in cfg.get("samples_us") or [] if p]
    if len(passes) < 2:
        return 0.0
    return (max(passes) - min(passes)) / statistics.median(passes)


def linfit(xs, ys):
    """Least squares y = a + b x: (b, a, rms of the relative residual)."""
    n = len(xs)
    if n < 2 or len(set(xs)) < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    rel = [((a + b * x) - y) / y for x, y in zip(xs, ys) if y > 0]
    return b, a, math.sqrt(sum(r * r for r in rel) / len(rel)) if rel else 0.0


def load_card(path):
    with open(path) as f:
        card = json.load(f)
    if card.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a microbench document ({card.get('schema')!r})")
    if card.get("refused"):
        raise ValueError(f"{path}: a refused run ({card['refused']})")
    return card


def series(card, fmt, grid, j):
    """The measured configurations of one format, grid and width, by K."""
    out = [c for c in card["configs"]
           if c["format"] == fmt and c["grid"] == grid and c["j"] == j and not c.get("skipped")]
    return sorted(out, key=lambda c: c["k"])


def fit_grid(card, fmt, grid, j, fixup_rate=None):
    """(seconds a unit, the intercept in seconds, relative rms) of one grid's calls over their
    longest CTA's units; with fixup_rate (bytes/s), each call's fixup bytes at that rate taken off
    first (a K changes the partial tiles, so the bytes may differ by K)."""
    cfgs = series(card, fmt, grid, j)
    xs, ys = [], []
    for c in cfgs:
        t = time_s(c)
        if t is None:
            continue
        if fixup_rate and c.get("fixup"):
            t -= c["fixup_bytes"] / fixup_rate
        xs.append(c["units"])
        ys.append(t)
    return linfit(xs, ys)


def fixup_fit(card, fmt):
    """The fixup kernel alone (`fixup_alone`, both stream-k grids, every width): its time against
    its bytes, which do not depend on K and differ about 8x over the widths. (seconds a byte, the
    fixed time L, relative rms), or None. The partials and the output tiles it reads are a few MB,
    so they come from the L2 as much as from DRAM: the rate is the fixup's own, not the bus's."""
    xs, ys = [], []
    for a in card.get("fixup_alone", []):
        t = time_s(a) if a["format"] == fmt else None
        if t is not None:
            xs.append(a["fixup_bytes"])
            ys.append(t)
    return linfit(xs, ys)


def clock_of(cfg):
    """The middle of a configuration's SM clock span (MHz), or None (not recorded)."""
    span = cfg.get("sm_mhz")
    return (span[0] + span[1]) / 2.0 if span else None


def derive(card, fmt="q4_K", clock_spread=CLOCK_SPREAD):
    """One card's law and checks (see the module text)."""
    info = card["card"]
    ref, dram = info["reference_clock_hz"], info["dram_bytes_per_s"]
    if not (ref > 0 and dram > 0):
        raise ValueError(f"{info['name']}: the board has no roofline facts (reference clock, DRAM "
                         "rate): the law cannot be stated for it")
    formats = sorted({c["format"] for c in card["configs"]})
    if fmt not in formats:
        raise ValueError(f"{info['name']}: no {fmt} configurations (has {', '.join(formats)})")
    per_format, fixups = {}, {}
    for f in formats:
        fx = fixup_fit(card, f)
        rate = 1.0 / fx[0] if fx and fx[0] > 0 else None
        fixups[f] = {"seconds_per_byte": fx[0] if fx else None, "launch_s": fx[1] if fx else None,
                     "rms": fx[2] if fx else None, "rate": rate,
                     "rate_ratio": rate / dram if rate else None}
        rows = {}
        for j in WIDTHS:
            law = fit_grid(card, f, "law", j)
            if law is None:
                continue
            row = {"unit_s": law[0], "unit_clocks": law[0] * ref, "intercept_s": law[1],
                   "law_rms": law[2]}
            for g, key in (("wave1", "wave1_ratio"), ("full", "full_ratio"),
                           ("halfwave", "halfwave_ratio")):
                other = fit_grid(card, f, g, j)
                if other:
                    row[key] = other[0] / law[0]
            for g in STREAM_K_GRIDS:
                sk = fit_grid(card, f, g, j, rate or dram)
                if sk:
                    row[f"{g}_unit_s"] = sk[0]
                    row[f"{g}_ratio"] = sk[0] / law[0]
            # The in-situ difference the first version read launch_seconds from: kept as a check
            # only (it also holds the partial writes, the launch the half-idle halfwave grid hides
            # and the extrapolation over two ranges of units).
            sk, half = fit_grid(card, f, "streamk", j, dram), fit_grid(card, f, "halfwave", j)
            if sk and half:
                row["insitu_fixup_beyond_bytes_s"] = sk[1] - half[1]
            rows[j] = row
        per_format[f] = rows
    rows = per_format[fmt]
    missing = [j for j in WIDTHS if j not in rows]
    by_grid = {g: [rows[j][f"{g}_ratio"] for j in WIDTHS if f"{g}_ratio" in rows.get(j, {})]
               for g in STREAM_K_GRIDS}
    ratios = [r for g in STREAM_K_GRIDS for r in by_grid[g]]
    law = {"unit_clocks": [rows[j]["unit_clocks"] if j in rows else None for j in WIDTHS],
           "stream_k_unit": statistics.mean(ratios) if ratios else None,
           "launch_seconds": fixups[fmt]["launch_s"],
           # Not in MmqLaunchLaw (the row prices the fixup's bytes at the board's DRAM rate): the
           # fixup's measured rate against it, and a call's fixed part, for the holdout's calls.
           "fixup_rate_ratio": fixups[fmt]["rate_ratio"],
           "call_fixed_s": [rows[j]["intercept_s"] if j in rows else None for j in WIDTHS]}
    stream_k_by_grid = {g: statistics.mean(v) if v else None for g, v in by_grid.items()}
    calls = []
    for c in card["configs"]:
        t = time_s(c) if c["format"] == fmt and c["grid"] in CALL_GRIDS else None
        if t is not None:
            calls.append({k: c[k] for k in ("grid", "j", "k", "units", "split", "fixup",
                                             "fixup_bytes")} | {"seconds": t})
    nodes = {n["kind"]: (statistics.median(n["samples_us"]) * 1e-6 if n.get("samples_us")
                         else None) for n in card.get("launch_node", [])}
    measured = [c for c in card["configs"] if not c.get("skipped")]
    resident = sorted({c.get("resident_ctas_per_sm", 0) for c in measured if c["format"] == fmt})
    # partition_of and the law count one CTA a SM: a kernel that can hold more is flagged on every
    # launch of it (the scheduler may still place one a SM where the grid has at most S CTAs).
    resident_flags = [f"{c['format']} J {c['j']} {c['grid']} K {c['k']}: "
                      f"{c['resident_ctas_per_sm']} CTAs a SM fit"
                      for c in measured if c.get("resident_ctas_per_sm", 0) > 1]
    clocks = [x for x in (clock_of(c) for c in measured if c["format"] == fmt) if x]
    clock_ratio = statistics.median(clocks) / (ref / 1e6) if clocks else None
    clock_flags = [f"{c['format']} J {c['j']} {c['grid']} K {c['k']}: SM clock "
                   f"{c['sm_mhz'][0]}..{c['sm_mhz'][1]} MHz"
                   for c in measured
                   if c.get("sm_mhz") and c["sm_mhz"][1] > (1 + clock_spread) * c["sm_mhz"][0]]
    spread = max((pass_spread(c) for c in measured), default=0.0)
    return {"card": info["name"], "cc": info["cc"], "generation": generation_of(info["cc"]),
            "sms": info["sms"], "reference_clock_hz": ref, "dram_bytes_per_s": dram,
            "format": fmt, "law": law, "missing_widths": missing, "per_format": per_format,
            "stream_k_by_grid": stream_k_by_grid, "fixup": fixups, "calls": calls,
            "launch_node_s": nodes, "resident_ctas_per_sm": resident,
            "resident_flags": resident_flags, "clock_ratio": clock_ratio,
            "clock_flags": clock_flags, "largest_pass_spread": spread}


# --- a generation's row ---

def mean_law(laws):
    """The mean of each value of the row (and of the holdout's extras) over the cards' laws."""
    def mean_of(values):
        values = [v for v in values if v is not None]
        return statistics.mean(values) if values else None
    out = {"unit_clocks": [mean_of([law["unit_clocks"][i] for law in laws])
                           for i in range(len(WIDTHS))]}
    for key in ("stream_k_unit", "launch_seconds", "fixup_rate_ratio"):
        out[key] = mean_of([law.get(key) for law in laws])
    out["call_fixed_s"] = [mean_of([(law.get("call_fixed_s") or [None] * len(WIDTHS))[i]
                                    for law in laws]) for i in range(len(WIDTHS))]
    return out


def call_seconds(law, held, call):
    """A call's time by the law on the held card: its longest CTA's units at the unit (times
    stream_k_unit where it splits), the fixup's bytes at the fixup rate (the board's DRAM rate x
    the law's fixup_rate_ratio, 1 where the law has none) and launch_seconds where it has one, and
    the call's fixed part (the law's call_fixed_s; the held card's own where the law has none, as a
    row of core/generation.h has not). None where the law cannot price it."""
    i = WIDTHS.index(call["j"])
    if law["unit_clocks"][i] is None or law["stream_k_unit"] is None:
        return None
    unit = law["unit_clocks"][i] / held["reference_clock_hz"]
    t = call["units"] * unit * (law["stream_k_unit"] if call["split"] else 1.0)
    if call["fixup"]:
        if law["launch_seconds"] is None:
            return None
        rate = held["dram_bytes_per_s"] * (law.get("fixup_rate_ratio") or 1.0)
        t += call["fixup_bytes"] / rate + law["launch_seconds"]
    fixed = (law.get("call_fixed_s") or [None] * len(WIDTHS))[i]
    if fixed is None:
        fixed = held["law"]["call_fixed_s"][i]
    return None if fixed is None else t + fixed


def predict(law, held):
    """A held-out card predicted from a law: (what, J, predicted, measured, relative error) for
    each width's unit and stream-k unit (each stream-k grid apart), the fixup's launch_seconds on
    its own, and every measured call end to end (the law's whole price of it)."""
    rows = held["per_format"][held["format"]]
    out = []
    for i, j in enumerate(WIDTHS):
        if j not in rows or law["unit_clocks"][i] is None:
            continue
        r = rows[j]
        unit = law["unit_clocks"][i] / held["reference_clock_hz"]
        out.append(("unit", j, unit, r["unit_s"]))
        for g in STREAM_K_GRIDS:
            if f"{g}_unit_s" in r and law["stream_k_unit"] is not None:
                out.append((f"{g} unit", j, unit * law["stream_k_unit"], r[f"{g}_unit_s"]))
    measured_l = held["law"]["launch_seconds"]
    if law["launch_seconds"] is not None and measured_l:
        out.append(("launch_seconds", None, law["launch_seconds"], measured_l))
    for c in held["calls"]:
        p = call_seconds(law, held, c)
        if p is not None:
            out.append((f"call {c['grid']} K {c['k']}", c["j"], p, c["seconds"]))
    return [(what, j, p, m, p / m - 1.0) for what, j, p, m in out]


def kind_of(what):
    """The holdout's kinds: a call's kind is its grid."""
    return " ".join(what.split()[:2]) if what.startswith("call ") else what


def rows_of(rows, tolerance):
    return [{"what": w, "j": j, "predicted_s": p, "measured_s": m, "error": e,
             "outside": abs(e) > tolerance} for w, j, p, m, e in rows]


def fit(derived, tolerance=DEFAULT_TOLERANCE):
    """A generation's row from two or more cards' derivations, and each card predicted from the
    others."""
    if len(derived) < 2:
        raise ValueError("a generation's row takes two or more cards (AGENTS.md: a constant fitted "
                         "on two or more cards of a generation)")
    gens = {d["generation"] for d in derived}
    if len(gens) != 1 or None in gens:
        raise ValueError("the cards are not of one work-table generation: " +
                         ", ".join(f"{d['card']} ({d['cc']}: {d['generation']})" for d in derived))
    formats = {d["format"] for d in derived}
    if len(formats) != 1:
        raise ValueError(f"the derivations are of different formats: {sorted(formats)}")
    law = mean_law([d["law"] for d in derived])
    holdout = []
    for i, held in enumerate(derived):
        others = [d for k, d in enumerate(derived) if k != i]
        holdout.append({"card": held["card"], "from": [d["card"] for d in others],
                        "rows": rows_of(predict(mean_law([d["law"] for d in others]), held),
                                        tolerance)})
    worst = {}
    for h in holdout:
        for r in h["rows"]:
            k = kind_of(r["what"])
            worst[k] = max(worst.get(k, 0.0), abs(r["error"]))
    # The transfer between cards states the unit at the reference clock: it holds where the cards
    # run at the same share of it.
    ratios = [d["clock_ratio"] for d in derived if d["clock_ratio"]]
    clock_note = None
    if len(ratios) == len(derived) and max(ratios) > (1 + CLOCK_SPREAD) * min(ratios):
        clock_note = ("the cards ran at different shares of their reference clocks (" +
                      ", ".join(f"{d['card']} {d['clock_ratio']:.3f}" for d in derived) +
                      "): a unit transferred at the reference clock carries that difference")
    generation = gens.pop()
    return {"schema": LAW_SCHEMA, "generation": generation,
            "name": GENERATIONS[generation][2], "format": derived[0]["format"],
            "cards": [d["card"] for d in derived], "law": law, "tolerance": tolerance,
            "holdout": holdout, "worst_error": worst,
            "within_tolerance": all(v <= tolerance for v in worst.values()),
            "missing_widths": sorted({j for d in derived for j in d["missing_widths"]}),
            "clock_ratios": {d["card"]: d["clock_ratio"] for d in derived},
            "clock_note": clock_note, "derived": derived}


def cpp_row(name, law):
    """The law as core/generation.h writes it; None where a value is missing or launch_seconds is
    negative (a fixup that costs less than nothing is no row)."""
    if any(v is None for v in law["unit_clocks"]) or law["stream_k_unit"] is None or \
            law["launch_seconds"] is None or law["launch_seconds"] < 0:
        return None
    units = ", ".join(str(int(round(v))) for v in law["unit_clocks"])
    return (f"inline constexpr MmqLaunchLaw {name}{{{{{units}}}, "
            f"{law['stream_k_unit']:.2f}, {law['launch_seconds']:.2g}}};")


def row_problem(law, missing):
    if any(v is None for v in law["unit_clocks"]):
        return f"incomplete (missing widths {missing})"
    if law["launch_seconds"] is None:
        return "incomplete (no fixup-alone timings)"
    if law["launch_seconds"] < 0:
        return f"refused: launch_seconds {law['launch_seconds'] * 1e6:.2f} us is negative"
    return "incomplete (no stream-k grid)"


# --- a row of core/generation.h ---

LAW_ROW = re.compile(r"inline constexpr MmqLaunchLaw (k\w+MmqLaw)\{\{([^}]*)\},\s*([^,]+),\s*"
                     r"([^}]+)\};")


def generation_laws(path=GENERATION_H):
    """The MmqLaunchLaw rows of core/generation.h: name -> law."""
    with open(path) as f:
        text = f.read()
    laws = {}
    for m in LAW_ROW.finditer(text):
        laws[m.group(1)] = {"unit_clocks": [float(v) for v in m.group(2).split(",")],
                            "stream_k_unit": float(m.group(3)),
                            "launch_seconds": float(m.group(4))}
    return laws


def check(derived, law, tolerance=DEFAULT_TOLERANCE):
    rows = rows_of(predict(law, derived), tolerance)
    return {"card": derived["card"], "rows": rows,
            "within_tolerance": not any(r["outside"] for r in rows)}


# --- printing ---

def us(v):
    return "-" if v is None else f"{v * 1e6:.2f}"


def ratio(v):
    return "-" if v is None else f"{v:.3f}"


def print_derived(d, out=None):
    def p(*a):
        print(*a, file=out or sys.stdout)
    p(f"{d['card']} (sm {d['cc']}, {d['generation']}, {d['sms']} SMs, reference clock "
      f"{d['reference_clock_hz'] / 1e6:.0f} MHz, DRAM {d['dram_bytes_per_s'] / 1e9:.0f} GB/s), "
      f"format {d['format']}")
    p("  J  unit us  unit clk  streamk  streamk_wide  wave1   full  halfwave  intercept us  "
      "in-situ fixup us")
    rows = d["per_format"][d["format"]]
    for j in WIDTHS:
        r = rows.get(j)
        if r is None:
            p(f"{j:3d}  (not measured)")
            continue
        p(f"{j:3d} {r['unit_s'] * 1e6:7.3f} {r['unit_clocks']:9.0f} "
          f"{ratio(r.get('streamk_ratio')):>8} {ratio(r.get('streamk_wide_ratio')):>13} "
          f"{ratio(r.get('wave1_ratio')):>6} {ratio(r.get('full_ratio')):>6} "
          f"{ratio(r.get('halfwave_ratio')):>9} {us(r['intercept_s']):>13} "
          f"{us(r.get('insitu_fixup_beyond_bytes_s')):>17}")
    sk = d["stream_k_by_grid"]
    p(f"  stream_k_unit: streamk {ratio(sk.get('streamk'))}, streamk_wide "
      f"{ratio(sk.get('streamk_wide'))}, the row's (both) {ratio(d['law']['stream_k_unit'])}")
    for f, frows in sorted(d["per_format"].items()):
        if f == d["format"]:
            continue
        p(f"  {f} unit clk: " + " ".join(f"{frows[j]['unit_clocks']:.0f}" if j in frows else "-"
                                        for j in WIDTHS))
    fx = d["fixup"][d["format"]]
    if fx["launch_s"] is not None:
        rate = (f"{fx['rate'] / 1e9:.0f} GB/s, {fx['rate_ratio']:.2f} x the DRAM rate"
                if fx["rate"] else "no positive rate")
        p(f"  fixup alone over its bytes: L {us(fx['launch_s'])} us, {rate} "
          f"(rms {fx['rms'] * 100:.1f} %); the row prices the bytes at the DRAM rate")
    p("  empty-kernel node, us: " + " ".join(f"{k} {us(v)}" for k, v in d["launch_node_s"].items()))
    p(f"  SM clock / reference clock: {ratio(d['clock_ratio'])}"
      + ("" if d["clock_ratio"] else " (clocks not recorded)"))
    for flag in d["clock_flags"]:
        p(f"  CLOCK: {flag} moved more than {CLOCK_SPREAD * 100:.0f} %")
    p(f"  CTAs a SM: {d['resident_ctas_per_sm']}; largest pass-to-pass spread "
      f"{d['largest_pass_spread'] * 100:.1f} %")
    for flag in d["resident_flags"]:
        p(f"  RESIDENT: {flag} (the law counts one)")
    row = cpp_row("kThisCardMmqLaw", d["law"])
    p("  this card's row: " + (row or row_problem(d["law"], d["missing_widths"])))


def print_rows(rows, out=None):
    """The holdout's rows: every unit and launch row; of the calls, a summary per grid and the
    calls outside the tolerance."""
    def p(*a):
        print(*a, file=out or sys.stdout)
    calls = {}
    for r in rows:
        if r["what"].startswith("call "):
            calls.setdefault(kind_of(r["what"]), []).append(r)
            if not r["outside"]:
                continue
        j = "" if r["j"] is None else f"J {r['j']:3d}"
        p(f"    {r['what']:24s} {j:6s}: predicted {us(r['predicted_s']):>9} us, "
          f"measured {us(r['measured_s']):>9} us, {r['error'] * 100:+6.1f} %"
          f"{'  OUTSIDE' if r['outside'] else ''}")
    for kind, rs in calls.items():
        worst = max(rs, key=lambda r: abs(r["error"]))
        p(f"    {kind:24s}: {len(rs)} calls end to end, worst {worst['error'] * 100:+.1f} % "
          f"(J {worst['j']}), {sum(r['outside'] for r in rs)} outside")


# --- run ---

def _isa_probe():
    """tools/bench/isa_probe/isa_probe.py: its idle check and nvidia-smi notes."""
    path = os.path.join(HERE, "isa_probe", "isa_probe.py")
    spec = importlib.util.spec_from_file_location("isa_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CLOCK_QUERY = "--query-gpu=timestamp,pci.bus_id,pstate,clocks.sm,clocks.mem"
CLOCK_FIELDS = 5


def smi_time(text):
    """nvidia-smi's timestamp (2026/11/12 14:03:21.123, its own local time) as epoch seconds read
    in this process's time zone; smi_offset() corrects for a zone that differs (WSL)."""
    return datetime.datetime.strptime(text.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()


def smi_offset(now, stamp):
    """What to add to smi_time(stamp) to get the epoch, from one stamp read at `now`: a whole
    number of quarter hours (time zones are), so the query's own latency does not enter."""
    return round((now - smi_time(stamp)) / 900.0) * 900.0


def read_smi_offset(ip):
    rows, _ = ip._smi(["--query-gpu=timestamp"])
    now = time.time()
    try:
        return smi_offset(now, rows[0][0]) if rows else 0.0
    except ValueError:
        return 0.0


def attach_clocks(doc, samples, bus, offset=0.0):
    """Each configuration's SM and memory clock span (MHz) from nvidia-smi's samples of its GPU
    inside its passes' windows, and the span over all of them. `samples` are rows of CLOCK_QUERY;
    `offset` is smi_offset's."""
    mine = []
    for s in samples:
        if len(s) != CLOCK_FIELDS or (bus is not None and _bus(s[1]) != bus):
            continue
        try:
            mine.append((smi_time(s[0]) + offset, s[2], int(s[3]), int(s[4])))
        except ValueError:
            continue
    timed = doc["configs"] + doc.get("fixup_alone", [])
    windows = [(b, e) for c in timed for b, e in zip(c.get("t_begin", []), c.get("t_end", []))]
    for c in timed:
        inside = [m for m in mine
                  if any(b <= m[0] <= e for b, e in zip(c.get("t_begin", []), c.get("t_end", [])))]
        if inside:
            c["sm_mhz"] = [min(m[2] for m in inside), max(m[2] for m in inside)]
            c["memory_mhz"] = [min(m[3] for m in inside), max(m[3] for m in inside)]
    measured = [m for m in mine if any(b <= m[0] <= e for b, e in windows)]
    doc["clocks"] = {"samples": len(measured), "offset_s": offset,
                     "sm_mhz": ([min(m[2] for m in measured), max(m[2] for m in measured)]
                                if measured else None),
                     "memory_mhz": ([min(m[3] for m in measured), max(m[3] for m in measured)]
                                    if measured else None),
                     "pstates": sorted({m[1] for m in measured})}
    return doc


def _bus(text):
    parts = text.strip().lower().split(":")
    if len(parts) != 3:
        return text.strip().lower()
    return f"{int(parts[0], 16):x}:{parts[1]}:{parts[2]}"


def cmd_run(a):
    ip = _isa_probe()
    out = a.out
    extra = [x for x in a.args if x != "--"]
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "card.json")
        offset = read_smi_offset(ip)
        before = ip._compute_apps()
        clocks = ip.sampler(CLOCK_QUERY, 200)
        apps = ip.sampler("--query-compute-apps=timestamp,gpu_bus_id,pid", 200)
        bench = subprocess.Popen([a.exe, "--out", raw] + extra)
        code = bench.wait()
        samples = clocks.rows(CLOCK_FIELDS) if clocks else []
        during = apps.rows(3) if apps else None
        after = ip._compute_apps()
        if code != 0 or not os.path.exists(raw):
            print(f"mmq_launch_law: the microbench exited {code}; no run", file=sys.stderr)
            return code or 3
        with open(raw) as f:
            doc = json.load(f)
    bus = _bus(doc["card"]["pci_bus_id"]) if doc["card"].get("pci_bus_id") else None
    notes = [ip._apps_note("before", before, bus), ip._apps_note("after", after, bus),
             ip._during_note(during, bus, str(bench.pid))]
    if a.idle_unchecked and any(
            m and (m.group(3).startswith("not read") or
                   (m.group(1).startswith("any GPU") and m.group(3).startswith("0")))
            for m in (ip.PROCESS_NOTE.fullmatch(n[5:]) for n in notes)):
        notes.append("note " + ip.WAIVED)
    doc["notes"] = notes
    attach_clocks(doc, samples, bus, offset)
    refusal = ip.idle_refusal([n[5:] for n in notes])
    path = out
    if refusal:
        doc["refused"] = refusal
        path += ".refused"
    with open(path, "w", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    for n in notes:
        print(n, file=sys.stderr)
    if refusal:
        print(f"mmq_launch_law: {refusal}; wrote {path}, which is no input of fit",
              file=sys.stderr)
        return 3
    print(f"mmq_launch_law: wrote {path}")
    print_derived(derive(doc, a.format))
    return 0


def cmd_derive(a):
    d = derive(load_card(a.input), a.format)
    print_derived(d)
    if a.out:
        with open(a.out, "w", newline="\n") as f:
            json.dump(d, f, indent=1)
            f.write("\n")
    return 0


def cmd_fit(a):
    derived = [derive(load_card(p), a.format) for p in a.input]
    for d in derived:
        print_derived(d)
    r = fit(derived, a.tolerance)
    print(f"\n{r['generation']} law from {', '.join(r['cards'])} ({r['format']}):")
    row = cpp_row(r["name"], r["law"])
    print("  " + (row or row_problem(r["law"], r["missing_widths"])))
    print("  clock / reference: " + ", ".join(f"{c} {ratio(v)}" for c, v in
                                              r["clock_ratios"].items()))
    if r["clock_note"]:
        print(f"  CLOCK: {r['clock_note']}")
    for h in r["holdout"]:
        print(f"  holdout: {h['card']} predicted from {', '.join(h['from'])}")
        print_rows(h["rows"])
    print("  worst |error|: " + ", ".join(f"{k} {v * 100:.1f} %" for k, v in r["worst_error"].items()))
    print(f"  {'within' if r['within_tolerance'] else 'NOT within'} the tolerance "
          f"{a.tolerance * 100:.0f} %" + ("" if r["within_tolerance"] else
                                           ": the cards disagree, the row is no generation's law"))
    if a.out:
        r["cpp"] = row
        with open(a.out, "w", newline="\n") as f:
            json.dump(r, f, indent=1)
            f.write("\n")
    return 0 if r["within_tolerance"] else 4


def cmd_check(a):
    laws = generation_laws(a.generation_h)
    if a.law not in laws:
        print(f"mmq_launch_law: no row {a.law} in {a.generation_h} (has {', '.join(laws)})",
              file=sys.stderr)
        return 2
    d = derive(load_card(a.input), a.format)
    print_derived(d)
    r = check(d, laws[a.law], a.tolerance)
    print(f"\n{d['card']} against {a.law}:")
    print_rows(r["rows"])
    print(f"  {'within' if r['within_tolerance'] else 'NOT within'} the tolerance "
          f"{a.tolerance * 100:.0f} %")
    return 0 if r["within_tolerance"] else 4


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the microbench on this card (the GPU otherwise idle)")
    r.add_argument("--exe", required=True, help="the built ninferno_mmq_launch_microbench")
    r.add_argument("--out", required=True)
    r.add_argument("--format", default="q4_K")
    r.add_argument("--idle-unchecked", action="store_true",
                   help="accept a run whose process list could not be read (said in the file)")
    r.add_argument("args", nargs=argparse.REMAINDER, help="-- microbench options")
    d = sub.add_parser("derive", help="one card's law and checks")
    d.add_argument("--input", required=True)
    d.add_argument("--format", default="q4_K")
    d.add_argument("--out")
    f = sub.add_parser("fit", help="a generation's row on two or more cards, with the holdout")
    f.add_argument("--input", action="append", required=True)
    f.add_argument("--format", default="q4_K")
    f.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    f.add_argument("--out")
    c = sub.add_parser("check", help="one card against a law row of core/generation.h")
    c.add_argument("--input", required=True)
    c.add_argument("--law", default="kTuringMmqLaw")
    c.add_argument("--format", default="q4_K")
    c.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    c.add_argument("--generation-h", default=GENERATION_H)
    a = ap.parse_args(argv)
    try:
        return {"run": cmd_run, "derive": cmd_derive, "fit": cmd_fit, "check": cmd_check}[a.cmd](a)
    except ValueError as e:
        print(f"mmq_launch_law: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
