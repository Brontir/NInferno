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
  stream-k    the slope on the `streamk` grid (floor(S / 2) tiles on S CTAs) of the time less the
              fixup's bytes at the DRAM rate; stream_k_unit = its ratio to the unit, the mean
              over the widths;
  fixup       the intercept of that fit less the intercept of the `halfwave` grid (the same shape
              as a tile a CTA): what a fixup launch adds beyond its bytes, whatever K is;
              launch_seconds = the mean over the widths (the law's form: one number);
  checks      the unit on `wave1` (every weight tile read once) and `full` (four waves) against
              the law grid's, the launch's fixed part (the law grid's intercept), the fixup kernel
              alone, and the empty kernel's node cost on a stream and in a graph.
The law row is the one format's (q4_K by default, as kTuringMmqLaw: the row has no format axis);
the other formats' units are printed beside it.

`fit` takes the cards' mean for each value of the row, then predicts each card from the others'
mean (with two cards: A from B and B from A): its unit and stream-k unit by J at its own reference
clock, and its fixup (its bytes at its DRAM rate plus the others' launch_seconds). A prediction
off by more than --tolerance (default 0.10, the decision bound of #785 PR-B, fixed before any
data) is flagged, and the row is then no generation's law: the cards disagree. All cards must be
of one generation (the rows of core/generation.h), each a run that `run` accepted.

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
GRIDS = ("law", "wave1", "full", "streamk", "halfwave")
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


def fit_grid(card, fmt, grid, j, less_fixup=False):
    """(seconds a unit, the intercept in seconds, relative rms) of one grid's calls over their
    longest CTA's units; with less_fixup, the fixup's bytes at the DRAM rate taken off first."""
    cfgs = series(card, fmt, grid, j)
    dram = card["card"]["dram_bytes_per_s"]
    xs, ys = [], []
    for c in cfgs:
        t = time_s(c)
        if t is None:
            continue
        if less_fixup and c.get("fixup"):
            t -= c["fixup_bytes"] / dram
        xs.append(c["units"])
        ys.append(t)
    return linfit(xs, ys)


def derive(card, fmt="q4_K"):
    """One card's law and checks (see the module text)."""
    info = card["card"]
    ref, dram = info["reference_clock_hz"], info["dram_bytes_per_s"]
    if not (ref > 0 and dram > 0):
        raise ValueError(f"{info['name']}: the board has no roofline facts (reference clock, DRAM "
                         "rate): the law cannot be stated for it")
    formats = sorted({c["format"] for c in card["configs"]})
    if fmt not in formats:
        raise ValueError(f"{info['name']}: no {fmt} configurations (has {', '.join(formats)})")
    per_format = {}
    for f in formats:
        rows = {}
        for j in WIDTHS:
            law, sk, half = (fit_grid(card, f, "law", j), fit_grid(card, f, "streamk", j, True),
                             fit_grid(card, f, "halfwave", j))
            if law is None:
                continue
            w1, full = fit_grid(card, f, "wave1", j), fit_grid(card, f, "full", j)
            row = {"unit_s": law[0], "unit_clocks": law[0] * ref, "intercept_s": law[1],
                   "law_rms": law[2]}
            if w1:
                row["wave1_ratio"] = w1[0] / law[0]
            if full:
                row["full_ratio"] = full[0] / law[0]
            if sk:
                row["stream_k_unit_s"] = sk[0]
                row["stream_k_ratio"] = sk[0] / law[0]
            if sk and half:
                row["fixup_launch_s"] = sk[1] - half[1]
                row["halfwave_ratio"] = half[0] / law[0]
                streamk = series(card, f, "streamk", j)
                row["fixup_bytes"] = streamk[-1]["fixup_bytes"] if streamk else 0.0
            rows[j] = row
        per_format[f] = rows
    rows = per_format[fmt]
    missing = [j for j in WIDTHS if j not in rows]
    ratios = [rows[j]["stream_k_ratio"] for j in WIDTHS if "stream_k_ratio" in rows.get(j, {})]
    launches = [rows[j]["fixup_launch_s"] for j in WIDTHS if "fixup_launch_s" in rows.get(j, {})]
    law = {"unit_clocks": [rows[j]["unit_clocks"] if j in rows else None for j in WIDTHS],
           "stream_k_unit": statistics.mean(ratios) if ratios else None,
           "launch_seconds": statistics.mean(launches) if launches else None}
    alone = {}
    for a in card.get("fixup_alone", []):
        if a["format"] == fmt and a.get("samples_us"):
            t = time_s(a)
            alone[a["j"]] = {"seconds": t, "beyond_bytes_s": t - a["fixup_bytes"] / dram}
    nodes = {n["kind"]: (statistics.median(n["samples_us"]) * 1e-6 if n.get("samples_us")
                         else None) for n in card.get("launch_node", [])}
    resident = sorted({c.get("resident_ctas_per_sm", 0) for c in card["configs"]
                       if c["format"] == fmt and not c.get("skipped")})
    spread = max((pass_spread(c) for c in card["configs"] if not c.get("skipped")), default=0.0)
    return {"card": info["name"], "cc": info["cc"], "generation": generation_of(info["cc"]),
            "sms": info["sms"], "reference_clock_hz": ref, "dram_bytes_per_s": dram,
            "format": fmt, "law": law, "missing_widths": missing, "per_format": per_format,
            "fixup_alone": alone, "launch_node_s": nodes, "resident_ctas_per_sm": resident,
            "largest_pass_spread": spread}


# --- a generation's row ---

def mean_law(laws):
    """The mean of each value of the row over the cards' laws."""
    def mean_of(values):
        values = [v for v in values if v is not None]
        return statistics.mean(values) if values else None
    return {"unit_clocks": [mean_of([law["unit_clocks"][i] for law in laws])
                            for i in range(len(WIDTHS))],
            "stream_k_unit": mean_of([law["stream_k_unit"] for law in laws]),
            "launch_seconds": mean_of([law["launch_seconds"] for law in laws])}


def predict(law, held):
    """Each value of a held-out card's derivation predicted from a law: (what, J, predicted,
    measured, relative error)."""
    rows = held["per_format"][held["format"]]
    out = []
    for i, j in enumerate(WIDTHS):
        if j not in rows or law["unit_clocks"][i] is None:
            continue
        r = rows[j]
        unit = law["unit_clocks"][i] / held["reference_clock_hz"]
        out.append(("unit", j, unit, r["unit_s"]))
        if "stream_k_unit_s" in r and law["stream_k_unit"] is not None:
            out.append(("stream-k unit", j, unit * law["stream_k_unit"], r["stream_k_unit_s"]))
        if "fixup_launch_s" in r and law["launch_seconds"] is not None:
            by_bytes = r["fixup_bytes"] / held["dram_bytes_per_s"]
            out.append(("fixup", j, by_bytes + law["launch_seconds"],
                        by_bytes + r["fixup_launch_s"]))
    return [(what, j, p, m, p / m - 1.0) for what, j, p, m in out]


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
        rows = predict(mean_law([d["law"] for d in others]), held)
        holdout.append({"card": held["card"], "from": [d["card"] for d in others],
                        "rows": [{"what": w, "j": j, "predicted_s": p, "measured_s": m,
                                  "error": e, "outside": abs(e) > tolerance}
                                 for w, j, p, m, e in rows]})
    worst = {}
    for h in holdout:
        for r in h["rows"]:
            worst[r["what"]] = max(worst.get(r["what"], 0.0), abs(r["error"]))
    generation = gens.pop()
    return {"schema": LAW_SCHEMA, "generation": generation,
            "name": GENERATIONS[generation][2], "format": derived[0]["format"],
            "cards": [d["card"] for d in derived], "law": law, "tolerance": tolerance,
            "holdout": holdout, "worst_error": worst,
            "within_tolerance": all(v <= tolerance for v in worst.values()),
            "missing_widths": sorted({j for d in derived for j in d["missing_widths"]}),
            "derived": derived}


def cpp_row(name, law):
    """The law as core/generation.h writes it."""
    if any(v is None for v in law["unit_clocks"]) or law["stream_k_unit"] is None or \
            law["launch_seconds"] is None:
        return None
    units = ", ".join(str(int(round(v))) for v in law["unit_clocks"])
    return (f"inline constexpr MmqLaunchLaw {name}{{{{{units}}}, "
            f"{law['stream_k_unit']:.2f}, {law['launch_seconds']:.2g}}};")


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
    rows = predict(law, derived)
    return {"card": derived["card"], "rows": [{"what": w, "j": j, "predicted_s": p,
                                               "measured_s": m, "error": e,
                                               "outside": abs(e) > tolerance}
                                              for w, j, p, m, e in rows],
            "within_tolerance": all(abs(e) <= tolerance for *_, e in rows)}


# --- printing ---

def us(v):
    return "-" if v is None else f"{v * 1e6:.2f}"


def print_derived(d, out=None):
    def p(*a):
        print(*a, file=out or sys.stdout)
    p(f"{d['card']} (sm {d['cc']}, {d['generation']}, {d['sms']} SMs, reference clock "
      f"{d['reference_clock_hz'] / 1e6:.0f} MHz, DRAM {d['dram_bytes_per_s'] / 1e9:.0f} GB/s), "
      f"format {d['format']}")
    p("  J  unit us  unit clk  stream-k  fixup-launch us  wave1  full  halfwave  intercept us")
    rows = d["per_format"][d["format"]]
    for j in WIDTHS:
        r = rows.get(j)
        if r is None:
            p(f"{j:3d}  (not measured)")
            continue
        p(f"{j:3d} {r['unit_s'] * 1e6:7.3f} {r['unit_clocks']:9.0f} "
          f"{r.get('stream_k_ratio', float('nan')):9.3f} {us(r.get('fixup_launch_s')):>16} "
          f"{r.get('wave1_ratio', float('nan')):6.3f} {r.get('full_ratio', float('nan')):5.3f} "
          f"{r.get('halfwave_ratio', float('nan')):9.3f} {us(r['intercept_s']):>13}")
    for f, frows in sorted(d["per_format"].items()):
        if f == d["format"]:
            continue
        p(f"  {f} unit clk: " + " ".join(f"{frows[j]['unit_clocks']:.0f}" if j in frows else "-"
                                        for j in WIDTHS))
    if d["fixup_alone"]:
        p("  fixup kernel alone, us (beyond its bytes): " +
          " ".join(f"J{j} {us(v['seconds'])} ({us(v['beyond_bytes_s'])})"
                   for j, v in sorted(d["fixup_alone"].items())))
    p("  empty-kernel node, us: " + " ".join(f"{k} {us(v)}" for k, v in d["launch_node_s"].items()))
    p(f"  CTAs a SM: {d['resident_ctas_per_sm']}; largest pass-to-pass spread "
      f"{d['largest_pass_spread'] * 100:.1f} %")
    row = cpp_row("kThisCardMmqLaw", d["law"])
    p("  this card's row: " + (row or f"incomplete (missing widths {d['missing_widths']})"))


def print_rows(rows, out=None):
    for r in rows:
        print(f"    {r['what']:14s} J {r['j']:3d}: predicted {us(r['predicted_s']):>9} us, "
              f"measured {us(r['measured_s']):>9} us, {r['error'] * 100:+6.1f} %"
              f"{'  OUTSIDE' if r['outside'] else ''}", file=out or sys.stdout)


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
    print("  " + (row or f"incomplete (missing widths {r['missing_widths']})"))
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
