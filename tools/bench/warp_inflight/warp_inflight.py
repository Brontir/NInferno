#!/usr/bin/env python3
"""Build and run warp_inflight on this card, or read saved runs, and give the per-SM DRAM rate as a
function of resident warps with the bytes a warp keeps in flight (b_w) and the saturated per-SM
rate (P) read from it: the facts the decode price's per-SM bound min(P, W x b_w / L0) needs
(#690, #735).

    warp_inflight.py run  --arch 89 [--nvcc nvcc] [--out warp_inflight_89.txt] [--idle-unchecked]
                          [--clocks-waived]
    warp_inflight.py rows --input warp_inflight_89.txt [--input ...] [--isa-table ...] [--out F]
    warp_inflight.py show --input warp_inflight_89.txt [--isa-table ...]

`run` compiles tools/bench/warp_inflight/warp_inflight.cu for sm_<arch> into a temporary directory
(nvcc -O3 -std=c++17 -arch=sm_<arch>), counts the 16-byte loads of each variant's kernel in the
build's SASS (cuobjdump), runs it (5 to 15 s, the GPU otherwise idle) and saves its output, headed
by the source's sha256, the build's -arch and those load counts, and followed by notes of the
compute processes on the probe's GPU before, during and after the run (isa_probe's notes, read by
its own code), of its warm-up (the probe streams on every SM until nvidia-smi has shown one memory
clock for 1 s in a performance P-state, at most 10 s, else the run is refused), and of the
clocks nvidia-smi saw inside the probe's measured span (its `span_ns` line), which must be the
settled one. A run with another process on the GPU, or whose memory clock or P-state moved in the
span, is no run; nor is one below P0 or under the rated memory clock, unless `--clocks-waived` (a
steady clock only, said in the run). A refused run's output goes to <out>.refused. `rows` turns
runs into JSON rows with their provenance (source file, card, probe version, nvcc, driver,
runtime, source hash) and the derived b_w and P; `show` prints the summary. No fact is fitted
here: b_w = rate(W) x L0 / W is Little's law on the measured rate and the run's own L0 (its free,
k = 1, chain1, W = 1 fetch), beside the ISA table's L0 where that row is of the same card; P is
the largest rate.
"""

import argparse
import datetime
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SOURCE = os.path.join(HERE, "warp_inflight.cu")
DEFAULT_ISA_TABLE = os.path.join(ROOT, "src", "core", "data", "isa_facts.json")
SCHEMA = 1


def _isa_probe():
    """tools/bench/isa_probe/isa_probe.py: its source hash, idle check and nvidia-smi notes."""
    path = os.path.join(ROOT, "tools", "bench", "isa_probe", "isa_probe.py")
    spec = importlib.util.spec_from_file_location("isa_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IP = _isa_probe()

WARP_LOAD = 512  # one warp-wide 16-byte load: 32 lanes x 16 B
# The probe's variants (warp_inflight.cu kVariants): name -> (loads a fetch, ring). A chain keeps
# F loads a lane in flight, a ring 2F; the SASS of a chain's loop holds F 16-byte loads, a ring's
# 3F (the first fetch before the loop, two in it).
VARIANTS = {"chain1": (1, False), "chain2": (2, False), "chain4": (4, False),
            "ring1": (1, True), "ring2": (2, True), "ring4": (4, True), "ring8": (8, True)}
# The runs of an earlier probe still parse, as what that probe measured: its source's sha256 (the
# warp_inflight.cu of that version, in the history) and its variants. Probe 3 (#739) had no ring8,
# which probe 4 (#757) added; its other kernels are the same.
EARLIER_PROBES = {
    "warp_inflight 3": {
        "source_sha256": "c903f40008fb45fa109c837b07bd78d9a5dbb565c9906afa1cbcd9b4711d93e8",
        "variants": ("chain1", "chain2", "chain4", "ring1", "ring2", "ring4")},
}
W_GRID = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
K_GRID = (1, 2, 4, 8)
MODES = ("free", "placed")
COLUMNS = ("mode", "k", "variant", "inflight", "w", "gbps", "gbps_lo", "gbps_hi", "ns_fetch",
           "sm_mhz", "mib")
HEADER = ("probe", "card", "cc", "sms", "l2_bytes", "mem_clock_khz", "max_warps_per_sm",
          "smem_per_sm", "driver", "runtime", "nvcc", "span_ns")
RUN_HEADER = ("source_sha256", "arch", "sass_ldg128")
# Run header lines from a probe version on: sass_inflight, the most 16-byte loads a lane each
# kernel's SASS keeps in flight (peak_in_flight), which must be the variant's count.
RUN_HEADER_SINCE = {"sass_inflight": 4}
COUNTS = ("cc", "sms", "l2_bytes", "max_warps_per_sm", "smem_per_sm", "driver", "runtime")
PROBE_LINE = re.compile(r'printf\("probe (warp_inflight \d+)\\n"\)')
PROBE_VALUE = re.compile(r"warp_inflight (\d+)")
NO_ROW = re.compile(r"(free|placed) (\w+) w(\d+): \d+ of \d+ CTAs fit an SM \(registers\), "
                    r"no row")
REFUSED_LINE = "refused"
REFUSED = ".refused"
# b_w's linear region: the rows from W = 1 up whose b_w stays within this share of b_w at W = 1
# (a reading of the measured table, not a fitted constant: it decides only the summary's
# `linear_to_w`, never a value).
LINEAR_SHARE = 0.9
# The clocks nvidia-smi saw inside the measured span (`run` samples every 100 ms): at least this
# many samples, else the span's clocks are not shown.
CLOCK_PERIOD_MS = 100
MIN_CLOCK_SAMPLES = 3
CLOCK_HEAD = "clocks over the measured span"
CLOCK_NOTE = re.compile(CLOCK_HEAD + r" \((\d+) samples, nvidia-smi every \d+ ms\): memory "
                        r"(\d+)\.\.(\d+) MHz, SM (\d+)\.\.(\d+) MHz, P-state (\S+)")
CLOCK_WAIVED = "the clock check was waived (--clocks-waived)"
# The row whose fetch is the run's own L0: one warp alone on one SM, one load a lane in flight.
OWN_L0 = ("free", 1, "chain1", 1)


def smi_time_ns(text):
    """An nvidia-smi timestamp (`2026/11/11 14:05:33.123`, the machine's local time) in ns since
    the epoch, the probe's span_ns clock; None when it does not read as one."""
    try:
        t = datetime.datetime.strptime(text.strip(), "%Y/%m/%d %H:%M:%S.%f")
    except ValueError:
        return None
    return int(t.timestamp() * 1e9)


def clock_note(samples, span, bus):
    """The note line of the clocks nvidia-smi saw on the probe's GPU inside the measured span;
    `samples` are rows (timestamp, bus id, P-state, SM MHz, memory MHz), `span` the probe's
    (begin, end) in ns since the epoch."""
    inside = []
    for s in samples:
        if len(s) != 5 or bus is None or IP._bus(s[1]) != bus:
            continue
        t = smi_time_ns(s[0])
        if t is not None and span[0] <= t <= span[1] and s[3].isdigit() and s[4].isdigit():
            inside.append(s)
    if not inside:
        return f"note {CLOCK_HEAD}: no nvidia-smi sample of the probe's GPU"
    mem, sm = sorted(int(s[4]) for s in inside), sorted(int(s[3]) for s in inside)
    states = "/".join(sorted({s[2] for s in inside}))
    return (f"note {CLOCK_HEAD} ({len(inside)} samples, nvidia-smi every {CLOCK_PERIOD_MS} ms): "
            f"memory {mem[0]}..{mem[-1]} MHz, SM {sm[0]}..{sm[-1]} MHz, P-state {states}")


# The warm-up streams until the memory clock has held one value this long (and that value is
# the highest since the warm-up began), within this bound; the measured span begins after it.
SETTLE_HOLD_NS = 1_000_000_000
# A clock below the warm-up's peak counts once it has held this long: a board that touches its
# top clock and then holds a lower one (a power limit) settles there; a step of a ramp does not
# hold this long.
SETTLE_LOWER_HOLD_NS = 3_000_000_000
SETTLE_TIMEOUT_NS = 10_000_000_000
SETTLE_HEAD = "warm-up: the memory clock held"
SETTLE_NOTE = re.compile(SETTLE_HEAD + r" (\d+) MHz for ([\d.]+) s before the measured span "
                         r"\(settled ([\d.]+) s into the warm-up\)")


def memory_clocks(samples, bus, since_ns):
    """[(ns since the epoch, memory MHz, P-state)] of the probe's GPU in nvidia-smi rows
    (timestamp, bus id, P-state, SM MHz, memory MHz) from `since_ns` on, in order."""
    out = []
    for s in samples:
        if len(s) != 5 or bus is None or IP._bus(s[1]) != bus or not s[4].isdigit():
            continue
        t = smi_time_ns(s[0])
        if t is not None and t >= since_ns:
            out.append((t, int(s[4]), s[2]))
    return out


# The performance P-states (NVIDIA: P0 the highest, P2 where GeForce runs CUDA); P5 and up are
# the idle and power-saving states, whose memory clock is no run's.
PERFORMANCE_PSTATES = ("P0", "P1", "P2")


def settled(samples, bus, since_ns, hold_ns=SETTLE_HOLD_NS):
    """(memory MHz, ns it has held) once the samples from `since_ns` on end in one memory clock
    held for at least `hold_ns` (first to last sample of that value, all in a performance
    P-state: the hold begins where both began) when it is the highest clock seen since
    `since_ns`, or for SETTLE_LOWER_HOLD_NS when it is lower; None until then (a clock still
    idle, ramping or dropping has not settled)."""
    mem = memory_clocks(samples, bus, since_ns)
    if not mem:
        return None
    t_last, value, state = mem[-1]
    if state not in PERFORMANCE_PSTATES:
        return None
    need = hold_ns if value == max(v for _, v, _ in mem) else max(hold_ns, SETTLE_LOWER_HOLD_NS)
    # The same clock read in an idle P-state before the GPU went to a performance one is no part
    # of the hold.
    i = len(mem) - 1
    while i > 0 and mem[i - 1][1] == value and mem[i - 1][2] in PERFORMANCE_PSTATES:
        i -= 1
    held = t_last - mem[i][0]
    return (value, held) if held >= need else None


def settle_refusal(notes, span_note):
    """Why the run shows no memory clock settled before its measured span (or one other than the
    span's own), or None."""
    mine = [SETTLE_NOTE.fullmatch(n) for n in notes if n.startswith(SETTLE_HEAD)]
    if len(mine) != 1 or not mine[0]:
        return "no note of a memory clock settled before the measured span (warm-up)"
    m = CLOCK_NOTE.fullmatch(span_note or "")
    if m and m.group(2) != mine[0].group(1):
        return (f"the warm-up settled at {mine[0].group(1)} MHz, the measured span ran at "
                f"{m.group(2)} MHz")
    return None


ENDED_IN_WARM_UP = "the probe ended before its warm-up settled"


def warm_until_settled(probe_lines, clock_rows, send, now_ns, sleep, alive,
                       timeout_ns=SETTLE_TIMEOUT_NS):
    """Hold the probe in its warm-up until the memory clock has settled: once the probe has
    printed `note warm-up: begins` (and its GPU's bus id), poll the nvidia-smi rows
    (`clock_rows()`) and `send("go")` when settled() holds, or `send("stop")` when it has not
    within `timeout_ns`. Returns (the settle note, None) or (None, why there is none)."""
    begun, bus = None, None
    while alive():
        lines = list(probe_lines())
        if begun is None and "note warm-up: begins" in lines:
            begun = now_ns()
            for line in lines:
                m = re.fullmatch(r"note device 0 is the GPU at PCI (\S+)", line)
                if m:
                    bus = IP._bus(m.group(1))
        if begun is not None:
            s = settled(clock_rows(), bus, begun)
            if s:
                send("go")
                return (f"note {SETTLE_HEAD} {s[0]} MHz for {s[1] / 1e9:.1f} s before the "
                        f"measured span (settled {(now_ns() - begun) / 1e9:.1f} s into the "
                        "warm-up)"), None
            if now_ns() - begun > timeout_ns:
                send("stop")
                mem = [v for _, v, _ in memory_clocks(clock_rows(), bus, begun)]
                seen = f"it read {min(mem)}..{max(mem)} MHz" if mem else "no nvidia-smi sample"
                return None, (f"the memory clock did not settle (its peak held "
                              f"{SETTLE_HOLD_NS / 1e9:g} s, or a lower clock "
                              f"{SETTLE_LOWER_HOLD_NS / 1e9:g} s) within {timeout_ns / 1e9:g} s "
                              f"of warm-up ({seen}); nothing measured")
        sleep(0.05)
    return None, ENDED_IN_WARM_UP


Lines = IP.Lines  # a stream read on a thread of its own (the probe's stdout, nvidia-smi's)


def clock_refusal(notes, rated_khz):
    """(why the run's clocks over its measured span do not show one steady memory clock at P0
    and the rated memory clock, or None; whether `--clocks-waived` may accept it). A clock or
    P-state that moved, or too few samples, is never waived; a steady P-state other than P0 or a
    steady memory clock under the rated one is, and the waiver must then be noted, and only
    then."""
    mine = [n for n in notes if n.startswith(CLOCK_HEAD)]
    waived = any(n.startswith(CLOCK_WAIVED) for n in notes)
    if len(mine) != 1:
        return f"{len(mine)} notes of the clocks over the measured span, not one", False
    m = CLOCK_NOTE.fullmatch(mine[0])
    if not m:
        return f"the clocks over the measured span were not read: {mine[0]!r}", False
    n, mem_lo, mem_hi = int(m.group(1)), int(m.group(2)), int(m.group(3))
    states = m.group(6)
    if n < MIN_CLOCK_SAMPLES:
        return f"{n} clock samples in the measured span, fewer than {MIN_CLOCK_SAMPLES}", False
    if mem_lo != mem_hi:
        return f"the memory clock moved in the measured span ({mem_lo}..{mem_hi} MHz)", False
    if "/" in states:
        return f"the P-state moved in the measured span ({states})", False
    soft = []
    if states != "P0":
        soft.append(f"P-state {states}, not P0")
    if rated_khz > 0 and mem_lo * 1000 < rated_khz:
        soft.append(f"memory clock {mem_lo} MHz, under the rated {rated_khz / 1000:g} MHz")
    if soft and not waived:
        return ("; ".join(soft) + " in the measured span; `run --clocks-waived` accepts a steady "
                "clock and says so in the run"), True
    if waived and not soft:
        return ("the clock check is noted as waived, yet the span ran at P0 and the rated "
                "clock"), False
    return None, False


def inflight_of(variant):
    f, ring = VARIANTS[variant]
    return 2 * f if ring else f


def sass_loads_of(variant):
    f, ring = VARIANTS[variant]
    return 3 * f if ring else f


def checked_in_probe():
    """The `probe` header value the checked-in warp_inflight.cu prints."""
    with open(SOURCE) as f:
        m = PROBE_LINE.search(f.read())
    if not m:
        raise ValueError("warp_inflight.cu prints no probe version")
    return m.group(1)


def source_sha256():
    return IP.source_sha256(SOURCE)


def expected_points(head, no_row, variants=tuple(VARIANTS)):
    """Every (mode, k, variant, w) the probe measures on this card: the free rows up to 32 warps,
    the placed rows up to the SM's resident warps (two CTAs above 32), of `variants` (the run's
    probe's), less the (mode, variant, w) the run noted the registers do not fit."""
    sms, max_w = int(head["sms"]), int(head["max_warps_per_sm"])
    points = set()
    for mode in MODES:
        for k in K_GRID:
            if k > sms:
                continue
            for v in variants:
                for w in W_GRID:
                    if w > max_w or (mode == "free" and w > 32) or (mode, v, w) in no_row:
                        continue
                    if mode == "placed" and w > 32 and w % 2:
                        continue
                    points.add((mode, k, v, w))
    return points


def probe_variants(probe):
    """The variants a run of `probe` (its header value) measured: the checked-in probe's, or an
    earlier probe's (EARLIER_PROBES); ValueError for any other."""
    if probe == checked_in_probe():
        return tuple(VARIANTS)
    if probe in EARLIER_PROBES:
        return EARLIER_PROBES[probe]["variants"]
    raise ValueError(f"{probe} is neither the checked-in probe ({checked_in_probe()}) nor an "
                     f"earlier one this parser reads ({', '.join(EARLIER_PROBES)})")


def _positive(text, what):
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{what} is no number: {text!r}") from None
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{what} is not a positive finite number: {text!r}")
    return value


def parse_run(text):
    """The header fields and the rows of one warp_inflight output (as `run` saves it); ValueError
    when a line is odd or unknown, the columns are not the parser's, a row is malformed, out of
    the grid, given twice or missing, the probe version and source hash are not the checked-in
    probe's or a recorded earlier one's (EARLIER_PROBES: its rows are its own variants'),
    the build's -arch is not the card's, the SASS load counts are not the variants', the
    self-check failed, the run did not end, or the process notes do not show the GPU idle."""
    head, rows, notes, ended, checked, failures = {}, {}, [], False, False, []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        word = parts[0]
        if word == "row":
            if len(parts) != 1 + len(COLUMNS):
                raise ValueError(f"odd row line: {line!r}")
            r = dict(zip(COLUMNS, parts[1:]))
            if r["mode"] not in MODES:
                raise ValueError(f"unknown mode: {line!r}")
            if r["variant"] not in VARIANTS:
                raise ValueError(f"unknown variant: {line!r}")
            if not (r["k"].isdigit() and r["w"].isdigit() and r["inflight"].isdigit()):
                raise ValueError(f"odd row line: {line!r}")
            key = (r["mode"], int(r["k"]), r["variant"], int(r["w"]))
            if key in rows:
                raise ValueError(f"row given twice: {key}")
            if int(r["inflight"]) != inflight_of(r["variant"]):
                raise ValueError(f"{r['variant']} keeps {inflight_of(r['variant'])} loads a lane "
                                 f"in flight, not {r['inflight']}: {line!r}")
            row = {"mode": key[0], "k": key[1], "variant": key[2],
                   "inflight_loads": int(r["inflight"]),
                   "warp_bytes_in_flight": int(r["inflight"]) * WARP_LOAD, "w": key[3]}
            for c in COLUMNS[5:]:
                row[c] = _positive(r[c], f"{c} of {key}")
            if not row["gbps_lo"] <= row["gbps"] <= row["gbps_hi"]:
                raise ValueError(f"gbps is outside its SMs' range: {line!r}")
            rows[key] = row
        elif word == "columns":
            if tuple(parts[1:]) != COLUMNS:
                raise ValueError(f"the probe's columns are not the parser's: {line!r}")
            head["columns"] = True
        elif word == "note":
            notes.append(" ".join(parts[1:]))
        elif word == "selfcheck":
            checked = True
            if parts[1:2] != ["ok"]:
                failures.append(" ".join(parts[2:]))
        elif word == "end":
            ended = True
        elif word == REFUSED_LINE:
            raise ValueError(f"`run` refused this run: {' '.join(parts[1:])}")
        elif word in HEADER or word in RUN_HEADER or word in RUN_HEADER_SINCE:
            if word in head:
                raise ValueError(f"{word} line given twice")
            head[word] = " ".join(parts[1:])
        else:
            raise ValueError(f"unknown line: {line!r}")
    if not ended or not checked:
        raise ValueError(f"incomplete run: ended={ended}, selfcheck={checked}")
    if failures:
        raise ValueError("the probe's self-check failed: " + "; ".join(failures))
    for k in HEADER + RUN_HEADER + ("columns",):
        if not head.get(k):
            raise ValueError(f"no {k} line")
    for k in COUNTS:
        if not (head[k].isdigit() and int(head[k]) > 0):
            raise ValueError(f"{k} is no positive count: {head[k]!r}")
    if not head["mem_clock_khz"].isdigit():
        raise ValueError(f"mem_clock_khz is no count: {head['mem_clock_khz']!r}")
    if not PROBE_VALUE.fullmatch(head["probe"]):
        raise ValueError(f"odd probe line: {head['probe']!r}")
    # A run is of the checked-in probe (its version and its source's hash) or of an earlier one
    # (that version's recorded hash), and holds that probe's variants.
    variants = probe_variants(head["probe"])
    want_sha = (source_sha256() if head["probe"] == checked_in_probe()
                else EARLIER_PROBES[head["probe"]]["source_sha256"])
    if head["source_sha256"] != want_sha:
        raise ValueError(f"source_sha256 {head['source_sha256']} is not that of {head['probe']}'s "
                         f"warp_inflight.cu ({want_sha}): a changed probe bumps its version")
    own = "sm_" + IP.arch_key(int(head["cc"]), "")
    if head["arch"] != own:
        raise ValueError(f"built for {head['arch']}, not {own} (cc {head['cc']})")
    want = " ".join(f"{v}:{sass_loads_of(v)}" for v in variants)
    if head["sass_ldg128"] != want:
        raise ValueError(f"the build's 16-byte loads per kernel are {head['sass_ldg128']!r}, not "
                         f"the variants' {want!r}")
    version = int(PROBE_VALUE.fullmatch(head["probe"]).group(1))
    for k, since in RUN_HEADER_SINCE.items():
        if version >= since and not head.get(k):
            raise ValueError(f"no {k} line (owed from probe {since})")
        if version < since and k in head:
            raise ValueError(f"a {k} line in a run of {head['probe']} (it began with probe "
                             f"{since})")
    if "sass_inflight" in head:
        want = " ".join(f"{v}:{inflight_of(v)}" for v in variants)
        if head["sass_inflight"] != want:
            raise ValueError(f"the build's SASS keeps {head['sass_inflight']!r} 16-byte loads a "
                             f"lane in flight, not the variants' {want!r}")
    max_w = int(head["max_warps_per_sm"])
    no_row = set()
    for n in notes:
        m = NO_ROW.fullmatch(n)
        if m:
            if m.group(2) not in variants:
                raise ValueError(f"odd note: {n!r}")
            no_row.add((m.group(1), m.group(2), int(m.group(3))))
    want_points = expected_points(head, no_row, variants)
    extra = sorted(set(rows) - want_points)
    if extra:
        raise ValueError(f"rows outside the grid (max {max_w} warps, {head['sms']} SMs): {extra}")
    missing = sorted(want_points - set(rows))
    if missing:
        raise ValueError(f"rows missing: {missing}")
    span = head["span_ns"].split()
    if len(span) != 2 or not all(s.isdigit() for s in span) or int(span[1]) <= int(span[0]):
        raise ValueError(f"odd span_ns line: {head['span_ns']!r}")
    refusal = IP.idle_refusal(notes)
    if refusal:
        raise ValueError(refusal)
    refusal, _ = clock_refusal(notes, int(head["mem_clock_khz"]))
    if refusal:
        raise ValueError(refusal)
    # From probe 3 the measured span begins only once the warm-up's memory clock settled.
    refusal = settle_refusal(notes, next((n for n in notes if n.startswith(CLOCK_HEAD)), None))
    if refusal:
        raise ValueError(refusal)
    head["notes"] = notes
    return head, [rows[k] for k in sorted(rows, key=lambda k: (MODES.index(k[0]), k[1],
                                                              list(VARIANTS).index(k[2]), k[3]))]


def isa_l0(cc, table_path):
    """(L0 in ns, its source, the card it was measured on) from the ISA facts table: the
    architecture's dram_tlbwarm_latency_ns_w1, the L0 of #735's law; (None, why, None) without
    one."""
    try:
        with open(table_path) as f:
            table = json.load(f)
    except OSError as e:
        return None, f"no ISA table ({type(e).__name__})", None
    key = IP.arch_key(cc, "")
    row = table.get("arches", {}).get(key)
    fact = (row or {}).get("facts", {}).get("dram_tlbwarm_latency_ns_w1")
    if not fact:
        return None, f"the ISA table has no sm_{key} row", None
    return fact["value"], (f"isa_facts sm_{key} ({row['card']}, {row['source']}) "
                           "dram_tlbwarm_latency_ns_w1"), row["card"]


def own_l0(rows):
    """The run's own L0: the fetch time of one warp alone on one SM with one load a lane in
    flight (free, k = 1, chain1, W = 1), in ns. That row's b_w is therefore 512 B by
    construction, no measurement: b_w is read from the linear region at W > 1 (and from the other
    variants and the placed rows), where the rate is measured against this L0."""
    return next(r["ns_fetch"] for r in rows if (r["mode"], r["k"], r["variant"], r["w"]) == OWN_L0)


def derive(rows, l0_ns, table_l0_ns=None):
    """Each row's b_w (bytes a warp keeps in flight, by Little's law on the row's rate and the
    run's own L0) and, where given (the ISA row is of the run's card), b_w at the table's L0; and
    the summary: per mode and variant at k = 1, b_w at W = 1 and the largest W up to which every
    row keeps LINEAR_SHARE of it; per mode at k = 1, P (the largest rate) in GB/s and in bytes per
    SM clock at that row's clock."""
    for r in rows:
        r["b_w_bytes"] = round(r["gbps"] * l0_ns / r["w"], 1) if l0_ns else None
        r["b_w_table_bytes"] = round(r["gbps"] * table_l0_ns / r["w"], 1) if table_l0_ns else None
    summary = {"variants": {}, "p_gbps": {}, "p_bytes_per_clk": {}}
    for mode in MODES:
        one = [r for r in rows if r["mode"] == mode and r["k"] == 1]
        if not one:
            continue
        top = max(one, key=lambda r: r["gbps"])
        summary["p_gbps"][mode] = top["gbps"]
        summary["p_bytes_per_clk"][mode] = round(top["gbps"] * 1e3 / top["sm_mhz"], 2)
        per = summary["variants"].setdefault(mode, {})
        for v in VARIANTS:
            vr = sorted((r for r in one if r["variant"] == v), key=lambda r: r["w"])
            if not vr or vr[0]["w"] != 1 or not l0_ns:
                continue
            b1, linear, at = vr[0]["b_w_bytes"], 1, vr[0]["b_w_bytes"]
            for r in vr:
                if r["b_w_bytes"] < LINEAR_SHARE * b1:
                    break
                linear, at = r["w"], r["b_w_bytes"]
            # b_w_at_linear_w: the reading at the top of the linear region (W > 1 where it has
            # more than one row); free chain1's b_w_w1 is 512 B by construction (own_l0).
            per[v] = {"warp_bytes_in_flight": vr[0]["warp_bytes_in_flight"], "b_w_w1": b1,
                      "linear_to_w": linear, "b_w_at_linear_w": at}
    return summary


NOTE = ("Per-SM DRAM rate against resident warps W, written by `tools/bench/warp_inflight/"
        "warp_inflight.py rows` from `warp_inflight.py run` outputs (one per card): every lane "
        "streams 16-byte ld.global.cs loads from a range of its own, the warp keeping a known "
        "number of them in flight (chainF: F a lane; ringF: the decode kernels' two-slot ring, "
        "up to 2F); `free` is k CTAs of W <= 32 warps on k distinct SMs with the L1-largest "
        "carve-out, `placed` 1 or 2 CTAs on each of k chosen SMs held there by shared memory "
        "(shared-largest carve-out). gbps is bytes per ns of one SM (median over the k SMs, "
        "median of three launches); b_w_bytes = gbps x L0 / W with L0 the run's own (its free, "
        "k = 1, chain1, W = 1 fetch time, so that row's b_w is 512 B by construction and no "
        "measurement: b_w is read from the linear region at W > 1, the summary's "
        "b_w_at_linear_w), b_w_table_bytes the same with the ISA table's "
        "dram_tlbwarm_latency_ns_w1 where that row was measured on the run's card (by card name: "
        "the ISA rows carry no PCI id), else null; the summary's P is the largest rate at k = 1. "
        "Free rows at k > 1 sit on SMs the CTA scheduler chose (they may share a TPC or a GPC): "
        "the modes are compared at k = 1. Provenance Measured for every run, with its source "
        "file, card, probe version, nvcc, driver, runtime and the sha256 of the warp_inflight.cu "
        "it was built from; every run's clocks were steady over its measured span (the notes), "
        "clocks_waived true where that steady clock was below P0 or the rated memory clock.")


def run_of(head, rows, source, isa_table):
    own = own_l0(rows)
    table, why, table_card = isa_l0(int(head["cc"]), isa_table)
    matches = table is not None and table_card == head["card"]
    summary = derive(rows, own, table if matches else None)
    return {
        "provenance": "Measured", "source": source, "card": head["card"],
        "cc": int(head["cc"]), "sms": int(head["sms"]), "l2_bytes": int(head["l2_bytes"]),
        "max_warps_per_sm": int(head["max_warps_per_sm"]),
        "smem_per_sm": int(head["smem_per_sm"]), "mem_clock_khz": int(head["mem_clock_khz"]),
        "tool": {"probe": head["probe"], "nvcc": head["nvcc"], "driver": head["driver"],
                 "runtime": head["runtime"], "arch": head["arch"],
                 "source_sha256": head["source_sha256"], "sass_ldg128": head["sass_ldg128"],
                 "sass_inflight": head.get("sass_inflight")},
        "notes": head["notes"],
        "clocks_waived": any(n.startswith(CLOCK_WAIVED) for n in head["notes"]),
        "l0": {"own_ns": own, "own_source": "this run's free, k = 1, chain1, W = 1 ns_fetch (so "
                                            "that row's b_w is 512 B by construction)",
               "table_ns": table, "table_source": why, "table_card": table_card,
               "table_is_this_card": matches},
        "rows": rows,
        "summary": summary,
    }


def table_of(paths, isa_table):
    runs = []
    for p in paths:
        with open(p) as f:
            text = f.read()
        try:
            head, rows = parse_run(text)
        except ValueError as e:
            raise ValueError(f"{os.path.basename(p)}: {e}") from None
        runs.append(run_of(head, rows, os.path.basename(p), isa_table))
    return {"schema": SCHEMA, "note": NOTE, "runs": runs}


def _kernels(sass):
    """{variant: its kernel's SASS body} of `cuobjdump -sass` text (sass_counts' parser)."""
    sc = IP._sass_counts()
    tag = {f"streamILi{f}ELb{int(ring)}E": v for v, (f, ring) in VARIANTS.items()}
    out = {}
    for name, body in sc.iter_functions(sass.splitlines(),
                                        lambda n: any(t in n for t in tag)):
        variant = next(v for t, v in tag.items() if t in name)
        if variant in out:
            raise ValueError(f"two kernels of {variant} in the build's SASS")
        out[variant] = body
    missing = [v for v in VARIANTS if v not in out]
    if missing:
        raise ValueError(f"no kernel of {missing} in the build's SASS")
    return out


def _is_ldg128(opcode):
    return IP._sass_counts().base_opcode(opcode) == "LDG" and ".128" in opcode


def sass_loads(sass):
    """{variant: 16-byte global loads in its kernel} of `cuobjdump -sass` text, read with
    sass_counts' parser: LDG instructions whose opcode carries .128."""
    return {v: sum(1 for row in body if _is_ldg128(row[2])) for v, body in _kernels(sass).items()}


SASS_REGISTER = re.compile(r"\bR(\d+)(\.64)?\b")


def peak_in_flight(body):
    """The most 16-byte loads the kernel's SASS keeps in flight a lane, read in program order: a
    load stays in flight until an instruction reads or overwrites a register it writes (its
    quad), which waits for it. The loop's body is read once, after the code before it, which is
    where the probe's fetches meet; a guarded load counts as issued."""
    sc = IP._sass_counts()
    pending, peak = [], 0
    for _, _, opcode, operands, _ in body:
        written = sc.written_registers(opcode, operands)
        fields = operands.split(",")
        reads = set()
        # The registers read: every one named after the destination (its first operand, where
        # the instruction writes one), a .64 address as its pair.
        for f in fields[1:] if written else fields:
            for m in SASS_REGISTER.finditer(f):
                n = int(m.group(1))
                reads.update({f"R{n}", f"R{n + 1}"} if m.group(2) else {f"R{n}"})
        pending = [q for q in pending if not (q & reads or q & written)]
        if _is_ldg128(opcode):
            pending.append(written)
            peak = max(peak, len(pending))
    return peak


def sass_inflight(sass):
    """{variant: the most 16-byte loads a lane its kernel's SASS keeps in flight} (peak_in_flight):
    a chain's F, a ring's 2F, when ptxas issues every load of a fetch before the data it waits
    on; fewer when it recycles a fetch's registers (ring8 without its clock read)."""
    return {v: peak_in_flight(body) for v, body in _kernels(sass).items()}


def cmd_run(a):
    head = [f"source_sha256 {source_sha256()}", f"arch sm_{a.arch}"]
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "warp_inflight.exe" if os.name == "nt" else "warp_inflight")
        cmd = [a.nvcc, "-O3", "-std=c++17", f"-arch=sm_{a.arch}", "-o", exe, SOURCE]
        print("+", " ".join(cmd), file=sys.stderr)
        subprocess.run(cmd, check=True)
        cuobjdump = a.cuobjdump
        if cuobjdump is None:
            import shutil
            nvcc = shutil.which(a.nvcc) or a.nvcc
            near = os.path.join(os.path.dirname(nvcc),
                                "cuobjdump" + (".exe" if os.name == "nt" else ""))
            cuobjdump = near if os.path.exists(near) else "cuobjdump"
        sass = subprocess.run([cuobjdump, "-sass", exe], stdout=subprocess.PIPE, text=True,
                              check=True).stdout
        try:
            loads, peaks = sass_loads(sass), sass_inflight(sass)
        except ValueError as e:
            print(f"warp_inflight: {e}; no run", file=sys.stderr)
            return 3
        head.append("sass_ldg128 " + " ".join(f"{v}:{loads[v]}" for v in VARIANTS))
        head.append("sass_inflight " + " ".join(f"{v}:{peaks[v]}" for v in VARIANTS))
        before = IP._compute_apps()

        # Both nvidia-smi streams drained as they come (IP.Sampler, #741).
        clocks = IP.sampler("--query-gpu=timestamp,pci.bus_id,pstate,clocks.sm,clocks.mem",
                            CLOCK_PERIOD_MS)
        clock_lines = clocks.lines if clocks else None
        apps = IP.sampler("--query-compute-apps=timestamp,gpu_bus_id,pid", 200)
        # The probe streams in its warm-up until told the memory clock has settled.
        probe = subprocess.Popen([exe, "--wait-go"], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, text=True)
        probe_lines = Lines(probe.stdout)

        def send(word):
            try:
                probe.stdin.write(word + "\n")
                probe.stdin.flush()
            except OSError:
                pass

        settle_note, settle_why = warm_until_settled(
            lambda: probe_lines.lines, lambda: clock_lines.rows(5) if clock_lines else [], send,
            time.time_ns, time.sleep, lambda: probe.poll() is None)
        probe.wait()
        probe_lines.thread.join()
        stdout = "".join(line + "\n" for line in probe_lines.lines)
        samples = clocks.rows(5) if clocks else []
        during = apps.rows(3) if apps else None
        after = IP._compute_apps()
    m = re.search(r"^note device 0 is the GPU at PCI (\S+)$", stdout, re.M)
    bus = IP._bus(m.group(1)) if m else None
    tail = [IP._apps_note("before", before, bus), IP._apps_note("after", after, bus),
            IP._during_note(during, bus, str(probe.pid))]
    if settle_note:
        tail.append(settle_note)
    # The clocks inside the probe's measured span only: the start (an idle card's P8, the
    # warm-up) and the end are not the rows' clocks.
    m = re.search(r"^span_ns (\d+) (\d+)$", stdout, re.M)
    if m:
        tail.append(clock_note(samples, (int(m.group(1)), int(m.group(2))), bus))
    else:
        tail.append(f"note {CLOCK_HEAD}: the probe printed no span")
    m = re.search(r"^mem_clock_khz (\d+)$", stdout, re.M)
    rated = int(m.group(1)) if m else 0
    if a.clocks_waived and clock_refusal([t[5:] for t in tail], rated)[1]:
        tail.append("note " + CLOCK_WAIVED)
    if a.idle_unchecked and any(
            n and (n.group(3).startswith("not read") or
                   (n.group(1).startswith("any GPU") and n.group(3).startswith("0")))
            for n in (IP.PROCESS_NOTE.fullmatch(t[5:]) for t in tail)):
        tail.append("note " + IP.WAIVED)
    body = stdout if stdout.endswith("\n") or not stdout else stdout + "\n"
    out = "\n".join(head) + "\n" + body + "\n".join(tail) + "\n"
    refusal = None
    if settle_why == ENDED_IN_WARM_UP and probe.returncode != 0:
        # The probe's own error (above, in its output) is the reason, not the settle.
        refusal = f"the probe exited {probe.returncode} in its warm-up"
    elif settle_why:
        refusal = settle_why
    elif probe.returncode != 0:
        refusal = f"the probe exited {probe.returncode}"
    else:
        try:
            parse_run(out)
        except ValueError as e:
            refusal = str(e)
    path = a.out or f"warp_inflight_{a.arch}.txt"
    if refusal:
        path += REFUSED
        out += f"{REFUSED_LINE} {' '.join(refusal.split())}\n"
    with open(path, "w", newline="\n") as f:
        f.write(out)
    sys.stdout.write(out)
    if refusal:
        print(f"warp_inflight: {refusal}; wrote {path}, which is no row", file=sys.stderr)
        return probe.returncode or 3
    print(f"warp_inflight: wrote {path}")
    return 0


def cmd_rows(a):
    try:
        table = table_of(a.input, a.isa_table)
    except (OSError, ValueError) as e:
        print(f"warp_inflight: {e}", file=sys.stderr)
        return 1
    text = json.dumps(table, indent=2, allow_nan=False) + "\n"
    if a.out:
        with open(a.out, "w", newline="\n") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


def cmd_show(a):
    try:
        table = table_of(a.input, a.isa_table)
    except (OSError, ValueError) as e:
        print(f"warp_inflight: {e}", file=sys.stderr)
        return 1
    for run in table["runs"]:
        s = run["summary"]
        l0 = run["l0"]
        print(f"{run['card']} ({run['tool']['arch']}, {run['source']}); own L0 {l0['own_ns']} ns "
              f"({l0['own_source']}); table L0 {l0['table_ns']} ns from {l0['table_source']}"
              f"{'' if l0['table_is_this_card'] else ' (another card: no table b_w)'}"
              f"{'; clocks waived' if run['clocks_waived'] else ''}")
        print("  free chain1 b_w at W=1 is 512 B by construction (its fetch is the own L0); read "
              "b_w from the linear region at W > 1")
        for mode, p in s["p_gbps"].items():
            print(f"  {mode}: P {p:.2f} GB/s an SM ({s['p_bytes_per_clk'][mode]} B/clk)")
            for v, d in s["variants"].get(mode, {}).items():
                print(f"    {v:7s} in flight {d['warp_bytes_in_flight']:5d} B a warp: b_w at W=1 "
                      f"{d['b_w_w1']:7.1f} B, linear to W={d['linear_to_w']} (b_w there "
                      f"{d['b_w_at_linear_w']:.1f} B)")
        for r in run["rows"]:
            if r["k"] == 1:
                print(f"  {r['mode']:6s} {r['variant']:7s} W={r['w']:2d} {r['gbps']:8.3f} GB/s "
                      f"ns_fetch {r['ns_fetch']:7.1f} b_w {r['b_w_bytes']} (table L0: "
                      f"{r['b_w_table_bytes']})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arch", required=True, help="sm_<arch> to compile for (70, 75, 86, 89, 120a)")
    r.add_argument("--nvcc", default="nvcc")
    r.add_argument("--cuobjdump", help="default: the one beside --nvcc, else from PATH")
    r.add_argument("--out", help="the run's output; a refused run's goes to <out>" + REFUSED)
    r.add_argument("--idle-unchecked", action="store_true",
                   help="accept a run whose compute-process list nvidia-smi could not give")
    r.add_argument("--clocks-waived", action="store_true",
                   help="accept a steady P-state other than P0 or a steady memory clock under the "
                   "rated one over the measured span (GeForce runs CUDA at P2), noted in the run")
    r.set_defaults(fn=cmd_run)
    for name, fn in (("rows", cmd_rows), ("show", cmd_show)):
        s = sub.add_parser(name)
        s.add_argument("--input", action="append", required=True)
        s.add_argument("--isa-table", default=DEFAULT_ISA_TABLE)
        if name == "rows":
            s.add_argument("--out")
        s.set_defaults(fn=fn)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
